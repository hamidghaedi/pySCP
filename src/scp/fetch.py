"""The AnnData data-access contract — the replacement for Seurat's ``FetchData``.

Every plotting function funnels through this module.  Getting it right is the
difference between a port that reproduces SCP's figures and one that quietly
plots the wrong column.

Two rules carry most of the weight:

1. **Resolution order is features first, then obs, then embeddings.**  SCP does
   *not* use Seurat's own ``FetchData`` order (keyed vars -> meta.data ->
   features) for the plotting layer; it builds ``cbind(dat_gene, dat_meta,
   dat_embedding)`` by hand, so on a name collision the *gene* wins and only a
   warning is emitted.  See ``SCP-plot.R:2123-2178``.

2. **Categorical level order is order-of-first-appearance, never sorted.**
   R's ``factor(x, levels = unique(x))`` bypasses R's own sorting.  pandas
   sorts by default, so ``pd.Categorical(s)`` will silently reshuffle every
   palette relative to the R output.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import pandas as pd

try:  # anndata is a hard runtime dep but keep import errors legible
    from anndata import AnnData
except ImportError:  # pragma: no cover
    AnnData = object  # type: ignore[assignment,misc]

import scipy.sparse as sp

__all__ = [
    "as_ordered_categorical",
    "as_level_labels",
    "reduction_key",
    "embedding_coord_map",
    "default_reduction",
    "resolve_matrix",
    "fetch_data",
    "fetch_var",
    "infer_data_type",
    "calculate_coexp",
    "DEFAULT_REDUCTION_ORDER",
    "REDUCTION_KEY",
]

# --------------------------------------------------------------------------- #
# categorical handling
# --------------------------------------------------------------------------- #
def as_ordered_categorical(s: pd.Series, *, show_na: bool = False, na_label: str = "NA") -> pd.Series:
    """Coerce to a Categorical with SCP's level order.

    An existing Categorical is trusted (scanpy objects carry meaningful
    category order, and ``adata.uns['<key>_colors']`` is aligned to it).
    Anything else gets categories in order of first appearance.

    ``show_na=True`` promotes missing values to a real ``"NA"`` level appended
    last, so the palette assigns it a color — SCP's ``show_na`` switch.
    """
    if not isinstance(s, pd.Series):
        s = pd.Series(list(s))
    if isinstance(s.dtype, pd.CategoricalDtype):
        cat = s
    else:
        cats = pd.unique(s.dropna())
        cat = pd.Series(pd.Categorical(s, categories=list(cats)), index=s.index, name=s.name)
    if show_na and cat.isna().any():
        cat = cat.cat.add_categories([na_label]).fillna(na_label)
    return cat


def as_level_labels(s: pd.Series, *, show_na: bool = False,
                    na_label: str = "NA") -> tuple[pd.Series, list[str]]:
    """Per-cell string labels and the level order, for any column dtype.

    :func:`as_ordered_categorical`, then every category rendered with
    ``str`` -- so an integer, boolean, float or mixed column gives labels
    (``"1"``, ``"True"``, ``"0.5"``) that compare equal to its levels, the
    way the stat plots render them.  Missing values stay ``NaN`` (or become
    ``na_label`` with ``show_na``).  Returns ``(labels, levels)``; labels
    is an object Series on the input's index.  Two categories that render
    to the same string (``1`` and ``"1"``) raise, since no label could tell
    them apart.
    """
    cat = as_ordered_categorical(s, show_na=show_na, na_label=na_label)
    levels = [str(c) for c in cat.cat.categories]
    if len(set(levels)) != len(levels):
        dup = sorted({lv for lv in levels if levels.count(lv) > 1})
        raise ValueError(f"{s.name!r} has distinct values that print the same: {dup}")
    codes = np.asarray(cat.cat.codes)
    lab = np.array(levels + [np.nan], dtype=object)[np.where(codes < 0, len(levels), codes)]
    return pd.Series(lab, index=cat.index, name=s.name), levels


# --------------------------------------------------------------------------- #
# reductions
# --------------------------------------------------------------------------- #
#: Display key prefix per well-known obsm key. Seurat's ``@key`` has no AnnData
#: counterpart, so axis labels have to come from a table plus a fallback.
REDUCTION_KEY: dict[str, str] = {
    "X_umap": "UMAP_",
    "X_tsne": "tSNE_",
    "X_pca": "PC_",
    "X_diffmap": "DM_",
    "X_draw_graph_fr": "FR_",
    "X_draw_graph_fa": "FA_",
    "X_phate": "PHATE_",
    "X_pacmap": "PaCMAP_",
    "X_mde": "MDE_",
    "X_scvi": "scVI_",
    "X_harmony": "Harmony_",
}

#: The order ``DefaultReduction`` searches. Mirrors ``SCP-workflow.R:1034``,
#: rewritten for scanpy's obsm naming.
DEFAULT_REDUCTION_ORDER: tuple[str, ...] = (
    "umap",
    "tsne",
    "dm",
    "diffmap",
    "phate",
    "pacmap",
    "trimap",
    "largevis",
    "fr",
    "draw_graph_fr",
    "pca",
    "svd",
    "ica",
    "nmf",
    "mds",
    "glmpca",
)


def reduction_key(obsm_key: str) -> str:
    """Axis-label prefix for an obsm key, e.g. ``X_umap`` -> ``"UMAP_"``."""
    if obsm_key in REDUCTION_KEY:
        return REDUCTION_KEY[obsm_key]
    base = re.sub(r"^X_", "", obsm_key)
    return base.replace("_", "") + "_"


def embedding_coord_map(adata: AnnData) -> dict[str, tuple[str, int]]:
    """``{"UMAP_1": ("X_umap", 0), "X_umap:0": ("X_umap", 0), ...}``.

    Both grammars are accepted: the R-style 1-based ``<KEY><n>`` form that SCP
    users will type, and an unambiguous ``<obsm_key>:<i>`` 0-based form that is
    the recommended internal representation.  On a duplicate display name the
    first-registered obsm key wins, matching R's ``%in%`` lookup.
    """
    out: dict[str, tuple[str, int]] = {}
    for k, v in adata.obsm.items():
        arr = np.asarray(v)
        if arr.ndim != 2:
            continue
        key = reduction_key(k)
        for i in range(arr.shape[1]):
            out.setdefault(f"{key}{i + 1}", (k, i))
            out[f"{k}:{i}"] = (k, i)
    return out


def default_reduction(
    adata: AnnData,
    pattern: str | Sequence[str] | None = None,
    min_dim: int = 2,
    *,
    uns_key: str = "default_reduction",
) -> str:
    """Pick an embedding, or resolve a named one: ``SCP::DefaultReduction``.

    Port of ``DefaultReduction`` (``SCP-workflow.R:1034``) by Hao Zhang,
    https://github.com/zhanghao-njmu/SCP.  Every plot resolves its
    ``reduction`` through here, as SCP does (``DefaultReduction(srt, pattern =
    reduction)``, ``SCP-plot.R:1394``), so ``"umap"``, ``"UMAP"`` and
    ``"X_umap"`` all name ``adata.obsm["X_umap"]``.  The steps, over the
    ``obsm`` entries with at least ``min_dim`` columns:

    1. a single qualifying reduction is returned *without* consulting the
       pattern;
    2. the patterns (``pattern``, else ``adata.uns[uns_key]``, else
       :data:`DEFAULT_REDUCTION_ORDER`), each also with ``f"{min_dim}D"``
       appended, are tried as exact keys in order -- bare and ``X_``-prefixed;
    3. otherwise every key containing any pattern, case-insensitively, and
       failing that every key within R's ``agrep(max.distance = 0.1)`` of one
       (an approximate substring match allowing ``ceil(0.1 * len(pattern))``
       edits); failing that every candidate;
    4. several hits are re-ordered by the default patterns and ``"2D"`` /
       ``"3D"`` and the one with the **fewest columns** wins (so a 2-D UMAP
       beats a 3-D one, and any 2-column embedding beats a 50-column PCA).

    So ``pattern="UMAP", min_dim=3`` finds ``UMAP3D``, and patterns are
    matched case-sensitively only in step 2: with keys ``PCA`` and ``pcaX``,
    ``"pca"`` resolves to ``pcaX`` (fewer columns), as in R.  Pinned case by
    case against SCP in ``tests/fixtures/parity/dim/default_reduction.csv``.

    One divergence: an explicit ``pattern`` that names nothing -- no exact
    key, no grep or agrep hit -- **warns** before falling back as R does
    (R silently plots some other reduction).
    """
    cands = [
        k for k, v in adata.obsm.items()
        if np.asarray(v).ndim == 2 and np.asarray(v).shape[1] >= min_dim
    ]
    explicit = [pattern] if isinstance(pattern, str) else (
        list(pattern) if pattern is not None else [])
    if not cands:
        raise ValueError(f"No embedding in adata.obsm with at least {min_dim} dimensions.")

    def _fallback(found: str) -> str:
        if explicit and not any(_pattern_hits(p, found) for p in explicit):
            warnings.warn(
                f"reduction {explicit[0] if len(explicit) == 1 else explicit!r} matches no "
                f"embedding in adata.obsm ({list(adata.obsm)}); using {found!r}, as "
                "SCP's DefaultReduction does.",
                stacklevel=3,
            )
        return found

    if len(cands) == 1:
        return _fallback(cands[0])

    if explicit:
        pats: list[str] = list(explicit)
    elif uns_key in getattr(adata, "uns", {}):
        pats = [str(adata.uns[uns_key])]
    else:
        pats = list(DEFAULT_REDUCTION_ORDER)
    pats = pats + [f"{p}{min_dim}D" for p in pats]

    for p in pats:  # exact match first, in pattern order
        for k in (p, f"X_{p}"):
            if k in cands:
                return k
    # then grep(ignore.case = TRUE), then agrep(max.distance = 0.1); R keeps
    # duplicates in pattern order, which only matters through the tie-break
    hits = [k for p in pats for k in cands if p.lower() in k.lower()]
    if not hits:
        hits = [k for p in pats for k in cands if _agrep(p, k)]
    hits = hits or list(cands)
    if len(hits) > 1:
        ordered = [k for p in (*DEFAULT_REDUCTION_ORDER, "2D", "3D") for k in hits
                   if p.lower() in k.lower()]
        hits = ordered or hits  # R would return character(0) here; keep the hits
    return _fallback(min(hits, key=lambda k: np.asarray(adata.obsm[k]).shape[1]))


def _pattern_hits(pattern: str, key: str) -> bool:
    """Whether ``key`` is one a pattern names, exactly or as R's grep / agrep would."""
    return (key in (pattern, f"X_{pattern}") or pattern.lower() in key.lower()
            or _agrep(pattern, key))


def _agrep(pattern: str, x: str, max_distance: float = 0.1) -> bool:
    """R's ``agrep(pattern, x, max.distance, ignore.case = TRUE)`` for one string.

    An approximate *substring* match: ``pattern`` may sit anywhere in ``x``
    with at most ``ceil(max_distance * len(pattern))`` insertions, deletions
    or substitutions (Sellers' algorithm, the semantics of TRE's matcher).
    """
    p, s = pattern.lower(), x.lower()
    k = math.ceil(max_distance * len(p))
    prev = list(range(len(p) + 1))  # column for the empty prefix of s
    best = prev[-1]
    for ch in s:
        cur = [0]  # a match may start anywhere in s
        for i, pc in enumerate(p, start=1):
            cur.append(min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (pc != ch)))
        best = min(best, cur[-1])
        prev = cur
    return best <= k


# --------------------------------------------------------------------------- #
# expression matrices
# --------------------------------------------------------------------------- #
DataType = Literal["raw_counts", "log_normalized_counts", "raw_normalized_counts", "unknown"]


def infer_data_type(x) -> DataType:
    """Port of ``check_DataType`` (``SCP-workflow.R:14``).

    The heatmaps genuinely depend on this: ``lib_normalize`` is skipped and
    library sizes are reset to 1 when a matrix labelled "counts" turns out to
    be non-integer.
    """
    arr = x.data if sp.issparse(x) else np.asarray(x)
    arr = np.asarray(arr, dtype=float).ravel()
    if arr.size == 0:
        return "unknown"
    if not np.all(np.isfinite(arr)):
        return "unknown"
    if np.any(arr < 0):
        return "unknown"
    if np.allclose(arr, np.round(arr)):
        return "raw_counts"
    mx = float(arr.max())
    if np.isfinite(np.expm1(mx)):
        return "log_normalized_counts"
    return "raw_normalized_counts"


def resolve_matrix(adata: AnnData, layer: str | None = None, use_raw: bool = False):
    """Single choke point for "which matrix does ``slot=`` mean?".

    Mapping from the R API:

    ==========================  ===============================
    Seurat                      this package
    ==========================  ===============================
    ``slot="data"``             ``layer=None``  (-> ``adata.X``)
    ``slot="counts"``           ``layer="counts"``
    ``slot="scale.data"``       ``layer="scaled"``
    ``assay="spliced"``         ``layer="spliced"``
    ``assay=`` (multimodal)     a separate AnnData / MuData modality
    ==========================  ===============================
    """
    if use_raw and layer is not None:
        raise ValueError("`use_raw=True` and `layer=` are mutually exclusive.")
    if use_raw:
        if adata.raw is None:
            raise ValueError("`use_raw=True` but adata.raw is None.")
        return adata.raw.X, pd.Index(adata.raw.var_names)
    if layer is None:
        return adata.X, pd.Index(adata.var_names)
    if layer not in adata.layers:
        if layer == "counts" and adata.raw is not None:
            warnings.warn(
                "layer='counts' not found; falling back to adata.raw.X. "
                "Pass layer=None to use adata.X explicitly.",
                stacklevel=2,
            )
            return adata.raw.X, pd.Index(adata.raw.var_names)
        raise KeyError(f"layer {layer!r} not in adata.layers ({list(adata.layers)}).")
    return adata.layers[layer], pd.Index(adata.var_names)


def _densify(x) -> np.ndarray:
    return np.asarray(x.todense()) if sp.issparse(x) else np.asarray(x)


def calculate_coexp(
    adata: AnnData,
    features: Sequence[str],
    *,
    layer: str | None = None,
    use_raw: bool = False,
) -> pd.Series:
    """Per-cell co-expression of ``features``: their geometric mean.

    Port of the ``calculate_coexp`` block SCP repeats inline in
    ``FeatureDimPlot`` (``SCP-plot.R:2143``), ``FeatureDimPlot3D`` (3152),
    ``FeatureStatPlot`` (3859) and ``FeatureCorPlot`` (5298) — Hao Zhang,
    https://github.com/zhanghao-njmu/SCP.  The rule depends on what the matrix
    holds, as judged by :func:`infer_data_type` over the whole matrix:

    * raw or raw-normalised counts: ``exp(mean(log(x)))``;
    * log-normalised: ``log1p(exp(mean(log(expm1(x)))))``, i.e. the geometric
      mean taken on the linear scale and logged back;
    * anything else raises, as in R.

    A single zero among the features makes the cell's value zero; that is
    what a geometric mean means, and R does the same.

    Divergence: the ``FeatureCorPlot`` copy reads ``status`` without ever
    assigning it, so it errors in R whenever ``calculate_coexp=TRUE``.  Every
    caller here goes through this one function, which defines it.

    Only gene names are accepted; obs columns and embeddings are not part of
    the product, and the callers drop them with R's warning before calling.
    """
    x, var_names = resolve_matrix(adata, layer=layer, use_raw=use_raw)
    features = list(features)
    missing = [f for f in features if f not in var_names]
    if missing:
        raise KeyError(f"calculate_coexp needs gene features; not in var_names: {missing}")
    if not features:
        raise ValueError("calculate_coexp needs at least one gene feature.")
    status = infer_data_type(x)
    sub = _densify(x[:, var_names.get_indexer(features)]).astype(float)
    with np.errstate(divide="ignore"):
        if status in ("raw_counts", "raw_normalized_counts"):
            values = np.exp(np.log(sub).mean(axis=1))
        elif status == "log_normalized_counts":
            values = np.log1p(np.exp(np.log(np.expm1(sub)).mean(axis=1)))
        else:
            raise ValueError("Can not determine the data type of the expression matrix.")
    return pd.Series(values, index=adata.obs_names, name="CoExp")


# --------------------------------------------------------------------------- #
# the resolver
# --------------------------------------------------------------------------- #
def fetch_data(
    adata: AnnData,
    vars: str | Iterable[str],
    *,
    layer: str | None = None,
    use_raw: bool = False,
    cells: Sequence[str] | None = None,
    on_missing: Literal["warn", "raise", "ignore"] = "warn",
) -> pd.DataFrame:
    """Resolve names to a cells x variables frame, SCP-style.

    Resolution order per name, first hit wins:

    1. ``adata.var_names``   -> expression from ``layer`` / ``X`` / ``raw``
    2. ``adata.obs.columns`` -> the obs column verbatim
    3. embedding coordinates -> ``"UMAP_1"`` (1-based) or ``"X_umap:0"`` (0-based)
    4. missing               -> warn and drop (SCP's default), or raise

    Columns come back in resolution-block order (genes, then obs, then
    coordinates), matching R's ``cbind``, *not* in the order the caller asked
    for.  Unknown cell names are silently dropped and the surviving rows keep
    **object order**, not request order — both faithful to SCP.
    """
    names = [vars] if isinstance(vars, str) else list(dict.fromkeys(vars))

    var_index = adata.raw.var_names if use_raw else adata.var_names
    var_set = set(map(str, var_index))
    obs_set = set(map(str, adata.obs.columns))
    coords = embedding_coord_map(adata)

    genes, metas, embs, missing = [], [], [], []
    for nm in map(str, names):
        if nm in var_set:
            genes.append(nm)
        elif nm in obs_set:
            metas.append(nm)
        elif nm in coords:
            embs.append(nm)
        else:
            missing.append(nm)

    collisions = [g for g in genes if g in obs_set]
    if collisions:
        warnings.warn(
            f"{collisions} appear in both var_names and obs columns; the gene is used. "
            "Rename the obs column to disambiguate.",
            stacklevel=2,
        )
    if missing:
        msg = f"variables not found and dropped: {missing}"
        if on_missing == "raise":
            raise KeyError(msg)
        if on_missing == "warn":
            warnings.warn(msg, stacklevel=2)
    if not (genes or metas or embs):
        raise ValueError("There are no valid variables present.")

    index = pd.Index(adata.obs_names, name=None)
    blocks: list[pd.DataFrame] = []

    if genes:
        X, vindex = resolve_matrix(adata, layer=layer, use_raw=use_raw)
        pos = pd.Index(map(str, vindex)).get_indexer(genes)
        blocks.append(pd.DataFrame(_densify(X[:, pos]), index=index, columns=genes))
    if metas:
        blocks.append(adata.obs[metas].set_axis(index))
    if embs:
        data = {}
        for nm in embs:
            obsm_key, col = coords[nm]
            data[nm] = np.asarray(adata.obsm[obsm_key])[:, col]
        blocks.append(pd.DataFrame(data, index=index))

    df = pd.concat(blocks, axis=1)
    if cells is not None:
        df = df.loc[df.index.intersection(pd.Index(cells))]
    return df


def fetch_var(adata: AnnData, keys: str | Iterable[str]) -> pd.DataFrame:
    """Per-feature annotation — the ``@meta.features`` equivalent.

    Kept separate from :func:`fetch_data` on purpose: mixing per-gene and
    per-cell frames in one return value is how SCP's heatmap code became hard
    to follow.
    """
    ks = [keys] if isinstance(keys, str) else list(keys)
    missing = [k for k in ks if k not in adata.var.columns]
    if missing:
        raise KeyError(f"not in adata.var: {missing}")
    return adata.var[ks]
