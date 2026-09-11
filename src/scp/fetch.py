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
    "reduction_key",
    "embedding_coord_map",
    "default_reduction",
    "resolve_matrix",
    "fetch_data",
    "fetch_var",
    "infer_data_type",
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
    """Pick an embedding when the user did not name one.

    Reproduces the priority of ``SCP::DefaultReduction``, including the two
    behaviours that surprise people: a single qualifying reduction is returned
    *without* consulting the priority list, and ties are broken by **fewest
    columns** (so a 2D UMAP beats a 3D one, and a 2-column anything beats a
    50-column PCA).
    """
    cands = [
        k for k, v in adata.obsm.items()
        if np.asarray(v).ndim == 2 and np.asarray(v).shape[1] >= min_dim
    ]
    if not cands:
        raise ValueError(f"No embedding in adata.obsm with at least {min_dim} dimensions.")
    if len(cands) == 1:
        return cands[0]

    if isinstance(pattern, str):
        pats: list[str] = [pattern]
    elif pattern is not None:
        pats = list(pattern)
    elif uns_key in getattr(adata, "uns", {}):
        pats = [str(adata.uns[uns_key])]
    else:
        pats = list(DEFAULT_REDUCTION_ORDER)
    pats = pats + [f"{p}{min_dim}D" for p in pats]

    for p in pats:  # exact match first, in pattern order
        for k in (p, f"X_{p}"):
            if k in cands:
                return k
    hits: list[str] = []
    for p in pats:  # then case-insensitive substring
        for k in cands:
            if p.lower() in k.lower() and k not in hits:
                hits.append(k)
    hits = hits or cands
    return min(hits, key=lambda k: np.asarray(adata.obsm[k]).shape[1])


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
