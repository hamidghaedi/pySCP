"""Enrichment text tracks for the heatmaps -- a port of ``heatmap_enrichment``.

``GroupHeatmap``, ``FeatureHeatmap`` and ``DynamicHeatmap`` can annotate every
feature cluster with three word boxes (``SCP-plot.R`` 7297-7540):

* ``anno_terms`` -- the cluster's top enriched terms;
* ``anno_keys`` -- keywords drawn from those terms' descriptions;
* ``anno_features`` -- the member genes that recur across the enriched terms.

SCP computes the enrichment itself, through ``RunEnrichment`` and a
Bioconductor stack (clusterProfiler, GO.db, AnnotationHub).  None of that
belongs in a plotting package, so here the enrichment is an **input**: pass a
table matching :data:`scp.io.ENRICHMENT_COLUMNS` (one row per term and
feature cluster, ``Groups`` holding the cluster name ``C1``, ``C2``, ...), or a
callable ``enrichment_fn(features, groups) -> DataFrame``.  :func:`ora` builds
such a callable from a gene-set table; it is a line-for-line port of
``clusterProfiler::enricher``'s hypergeometric test, so with the same gene sets
it reproduces ``RunEnrichment``'s p-values.

Deliberate divergence -- ``anno_keys`` for GO databases.  R calls
``simplifyEnrichment::keyword_enrichment_from_GO``, a word-enrichment test over
the whole GO term-name corpus, which needs the GO DAG.  This port applies the
method R uses for every *other* database to GO as well: lower-case the term
descriptions, split on spaces, score each word by the summed
``-log10(p.adjust)`` of the terms containing it, count those terms, drop
bracketed tokens and :data:`WORDS_EXCLUDED`, and keep the ``top_word`` best
scores (ties broken alphabetically, as dplyr's C-locale ``group_by`` does).
It is deterministic and needs no database; for non-GO tables it matches R
exactly.

Upstream: Hao Zhang, SCP (GPL-3), https://github.com/zhanghao-njmu/SCP.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from ..colors import blendcolors
from ..palettes import PALETTES

__all__ = [
    "WORDS_EXCLUDED",
    "EnrichmentFn",
    "BoxKind",
    "ora",
    "p_adjust_bh",
    "resolve_enrichment",
    "select_enrichment",
    "term_boxes",
    "keyword_boxes",
    "feature_boxes",
    "enrichment_boxes",
    "capitalize",
    "matched_colors",
]

#: ``SCP::words_excluded`` -- the stop-word list ``anno_keys`` filters out.
WORDS_EXCLUDED: tuple[str, ...] = (
    "the", "is", "and", "or", "a", "in", "on", "under", "between", "of", "through",
    "via", "along", "that", "for", "with", "within", "without", "cell", "cellular",
    "dna", "rna", "protein", "peptide", "amino", "acid", "development", "involved",
    "organization", "system", "regulation", "positive", "negative", "response",
    "process", "processing", "small", "large", "change",
)

#: ``enrichment_fn(features, groups) -> DataFrame`` with ``ENRICHMENT_COLUMNS``.
EnrichmentFn = Callable[[Sequence[str], pd.Series], pd.DataFrame]

BoxKind = Literal["terms", "keys", "features"]


# --------------------------------------------------------------------------- #
# the backend: over-representation analysis
# --------------------------------------------------------------------------- #
def p_adjust_bh(p: np.ndarray) -> np.ndarray:
    """``p.adjust(p, "BH")``, including R's tie and ordering handling."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(-p, kind="stable")  # order(p, decreasing = TRUE)
    i = np.arange(n, 0, -1)
    adj = np.minimum.accumulate(n / i * p[o])
    out = np.empty(n)
    out[o] = np.minimum(1.0, adj)
    return out


def ora(
    gene_sets: pd.DataFrame | Mapping[str, Iterable[str]],
    term_names: pd.DataFrame | Mapping[str, str] | None = None,
    *,
    min_gs_size: int = 10,
    max_gs_size: int = 500,
    database: str = "custom",
) -> EnrichmentFn:
    """An ``enrichment_fn`` running over-representation analysis per cluster.

    A port of ``clusterProfiler::enricher`` as ``RunEnrichment`` calls it
    (``universe = NULL``, BH adjustment, no p-value cut-off):

    * the universe is every gene in ``gene_sets``;
    * ``n`` is the number of a cluster's genes found in any set, ``M`` a set's
      size within the universe, ``k`` the cluster genes it contains;
    * only sets with at least one hit and ``min_gs_size <= M <= max_gs_size``
      are tested, with ``p = P[X >= k]``, ``X ~ Hypergeom(N, M, n)``;
    * rows are sorted by p-value; ``geneID`` lists the hits in input order.

    ``gene_sets`` is a ``TERM2GENE`` frame (term, gene) or a mapping
    ``term -> genes``; ``term_names`` a ``TERM2NAME`` frame or mapping, the term
    id standing in for missing names.  ``qvalue`` is not computed.
    """
    if isinstance(gene_sets, pd.DataFrame):
        t2g = gene_sets.iloc[:, :2].dropna().astype(str)
        t2g.columns = ["Term", "gene"]
    else:
        t2g = pd.DataFrame(
            [(str(t), str(g)) for t, gs in gene_sets.items() for g in gs], columns=["Term", "gene"]
        )
    t2g = t2g.drop_duplicates()
    if term_names is None:
        names: dict[str, str] = {}
    elif isinstance(term_names, pd.DataFrame):
        tn = term_names.iloc[:, :2].astype(str)
        names = dict(zip(tn.iloc[:, 0], tn.iloc[:, 1], strict=True))
    else:
        names = {str(k): str(v) for k, v in term_names.items()}
    term_genes = t2g.groupby("Term", sort=True)["gene"].apply(list).to_dict()
    gene_terms = t2g.groupby("gene", sort=False)["Term"].apply(list).to_dict()
    universe = set(t2g["gene"])
    N = len(universe)

    def run(features: Sequence[str], groups: pd.Series) -> pd.DataFrame:
        from scipy.stats import hypergeom

        grp = pd.Series(np.asarray(groups), index=list(features))
        levels = (list(groups.cat.categories) if isinstance(groups.dtype, pd.CategoricalDtype)
                  else list(pd.unique(grp)))
        frames = []
        for g in levels:
            genes = list(dict.fromkeys(str(f) for f in grp.index[grp.to_numpy() == g]))
            mapped = [x for x in genes if x in gene_terms]
            n = len(mapped)
            if n == 0:
                continue
            hits: dict[str, list[str]] = {}
            for x in mapped:
                for t in gene_terms[x]:
                    hits.setdefault(t, []).append(x)
            rows = []
            for t in sorted(hits):
                M = len(term_genes[t])
                if not (min_gs_size <= M <= max_gs_size):
                    continue
                k = len(hits[t])
                rows.append((t, names.get(t, t), f"{k}/{n}", f"{M}/{N}",
                             float(hypergeom.sf(k - 1, N, M, n)), "/".join(hits[t]), k))
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=["ID", "Description", "GeneRatio", "BgRatio",
                                             "pvalue", "geneID", "Count"])
            df["p.adjust"] = p_adjust_bh(df["pvalue"].to_numpy())
            df = df.iloc[np.argsort(df["pvalue"].to_numpy(), kind="stable")]
            df["Groups"] = str(g)
            df["Database"] = database
            frames.append(df)
        cols = ["ID", "Description", "GeneRatio", "BgRatio", "pvalue", "p.adjust",
                "geneID", "Count", "Groups", "Database"]
        if not frames:
            return pd.DataFrame(columns=cols)
        return pd.concat(frames, ignore_index=True)[cols]

    return run


def resolve_enrichment(
    features: Sequence[str],
    groups: pd.Series,
    *,
    enrichment: pd.DataFrame | None = None,
    enrichment_fn: EnrichmentFn | None = None,
) -> pd.DataFrame:
    """The enrichment table for the given feature clusters.

    A precomputed ``enrichment`` wins; otherwise ``enrichment_fn`` is called
    with the features and their cluster labels (``C1``...).  With neither, the
    error says how to supply one -- this package does not download databases.
    """
    if enrichment is None:
        if enrichment_fn is None:
            raise ValueError(
                "anno_terms / anno_keys / anno_features need an enrichment result: pass "
                "`enrichment=` (a DataFrame with scp.io.ENRICHMENT_COLUMNS, `Groups` holding "
                "the feature cluster names C1, C2, ...) or `enrichment_fn=` -- for example "
                "scp.heatmap.enrichment.ora(gene_sets)."
            )
        enrichment = enrichment_fn(list(features), groups)
    df = pd.DataFrame(enrichment).copy()
    need = ["ID", "Description", "geneID", "Groups"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"enrichment table is missing {missing}; see scp.io.ENRICHMENT_COLUMNS")
    if "Database" not in df.columns:
        df["Database"] = "custom"
    df["Groups"] = df["Groups"].astype(str)
    df["Database"] = df["Database"].astype(str)
    return df


def select_enrichment(
    df: pd.DataFrame,
    *,
    db: str | Sequence[str] | None = None,
    padjust_cutoff: float | None = 0.05,
    pvalue_cutoff: float | None = None,
) -> tuple[pd.DataFrame, str]:
    """Filter and sort as ``heatmap_enrichment`` does; returns ``(df, metric)``.

    ``metric`` is ``p.adjust`` unless ``padjust_cutoff`` is None, in which case
    ``pvalue``; rows with ``metric < cutoff`` are kept and sorted by it (stable).
    ``db=None`` keeps every database in the table -- SCP defaults to
    ``"GO_BP"`` because it fetched that database itself.
    """
    metric = "pvalue" if padjust_cutoff is None else "p.adjust"
    cutoff = pvalue_cutoff if padjust_cutoff is None else padjust_cutoff
    out = df
    if db is not None:
        dbs = [db] if isinstance(db, str) else list(db)
        out = out[out["Database"].isin(dbs)]
    if cutoff is not None:
        out = out[pd.to_numeric(out[metric]) < cutoff]
    out = out.iloc[np.argsort(pd.to_numeric(out[metric]).to_numpy(), kind="stable")]
    return out, metric


# --------------------------------------------------------------------------- #
# the word boxes
# --------------------------------------------------------------------------- #
def capitalize(x: Iterable[str]) -> list[str]:
    """SCP's ``capitalize``: upper-case the first letter only when the first word
    (split on whitespace or ``-``) is entirely lower case -- so ``mRNA ...`` and
    ``DNA repair`` are left alone."""
    import re

    out = []
    for s in x:
        s = str(s)
        first = re.split(r"\s|-", s)[0]
        out.append(s[:1].upper() + s[1:] if first == first.lower() else s)
    return out


def matched_colors(values: np.ndarray, palette: str = "Spectral", n: int = 100) -> list[str]:
    """``palette_scp(x, type = "continuous", matched = TRUE)``, bin for bin.

    R cuts ``x`` into ``n`` equal-width, right-closed bins over its range
    (``cut(..., include.lowest = TRUE)``) and indexes a ``colorRampPalette``
    of ``n`` colors by bin; a constant ``x`` takes the palette's first color.
    """
    x = np.asarray(values, dtype=float)
    pal = list(PALETTES[palette].colors)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return ["grey80"] * len(x)
    if np.unique(finite).size == 1:
        return [pal[0]] * len(x)
    breaks = np.linspace(finite.min(), finite.max(), n + 1)
    idx = np.clip(np.searchsorted(breaks, x, side="left"), 1, n) - 1
    ramp = pal[:n] if n <= len(pal) else _r_color_ramp(pal, n)
    return [ramp[i] for i in idx]


def _r_color_ramp(colors: Sequence[str], n: int) -> list[str]:
    """``colorRampPalette(colors)(n)`` to the bit.

    :func:`scp.palettes.color_ramp` *rounds* each channel, which lands one
    8-bit step above R on most ramp positions -- invisible in a heatmap body,
    but it shows when the word colors are compared to R's.  ``colorRamp``
    interpolates channels in [0, 1] and scales by 255, and
    ``rgb(..., maxColorValue = 255)`` then coerces to integer, i.e. truncates.
    """
    stops = np.array([[int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)] for c in colors],
                     dtype=float) / 255.0
    xp = np.linspace(0.0, 1.0, len(stops))
    x = np.linspace(0.0, 1.0, n)
    chan = np.empty((n, 3))
    for row, v in enumerate(x):
        # R's approx1(), operation for operation, so exact integers survive
        i, j = 0, len(xp) - 1
        while i < j - 1:
            ij = (i + j) // 2
            if v < xp[ij]:
                j = ij
            else:
                i = ij
        if v == xp[j]:
            chan[row] = stops[j]
        elif v == xp[i]:
            chan[row] = stops[i]
        else:
            chan[row] = stops[i] + (stops[j] - stops[i]) * ((v - xp[i]) / (xp[j] - xp[i]))
    ints = np.trunc(np.clip(chan, 0.0, 1.0) * 255.0).astype(int)
    return ["#{:02X}{:02X}{:02X}".format(*row) for row in ints]


def _darken(cols: list[str]) -> list[str]:
    # blendcolors(c(x, "black")): the box text is the palette color darkened
    return [str(blendcolors([c, "#000000"])) for c in cols]


def _rescale(x: np.ndarray, to: tuple[float, float]) -> np.ndarray:
    """``scales::rescale``; a zero-range input maps to ``mean(to)``."""
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi == lo:
        return np.full(x.shape, (to[0] + to[1]) / 2.0)
    return (x - lo) / (hi - lo) * (to[1] - to[0]) + to[0]


def _words(df: pd.DataFrame, metric: str, column: str,
           tokens: Callable[[str], list[str]]) -> pd.DataFrame:
    rows = []
    for text, p in zip(df[column].astype(str), df[metric].astype(float), strict=True):
        s = -np.log10(p)
        for w in tokens(text):
            rows.append((w, s))
    if not rows:
        return pd.DataFrame(columns=["keyword", "score", "count"])
    t = pd.DataFrame(rows, columns=["keyword", "s"])
    agg = t.groupby("keyword", sort=False)["s"].agg(score="sum", count="size").reset_index()
    # dplyr's group_by sorts groups in the C locale -- the tie-break below
    return agg.sort_values("keyword", key=lambda k: k.map(lambda v: v.encode()), kind="stable")


def _desc_words(s: str) -> list[str]:
    return s.lower().split(" ")


def _genes(s: str) -> list[str]:
    return s.split("/")


def _top(words: pd.DataFrame, top: int, fontsize: tuple[float, float],
         rng: np.random.Generator) -> pd.DataFrame:
    if words.empty:
        return words
    order = np.argsort(-words["score"].to_numpy(dtype=float), kind="stable")[:top]
    out = words.iloc[order].reset_index(drop=True)
    out["col"] = _darken(matched_colors(out["score"].to_numpy()))
    out["fontsize"] = _rescale(out["count"].to_numpy(), fontsize)
    # R draws a random 0/90 angle with probability 60/40; ComplexHeatmap's
    # textbox ignores it, so it is kept only as data
    out["angle"] = 90 * rng.choice([0, 1], size=len(out), p=[0.6, 0.4])
    return out


def term_boxes(
    df: pd.DataFrame, metric: str, *, top_term: int = 5, show_termid: bool = False,
    fontsize: float = 8,
) -> pd.DataFrame:
    """``anno_terms``: the ``top_term`` best terms, colored by ``-log10(metric)``."""
    head = df.head(top_term)
    if show_termid:
        terms = [f"{i} {d}" for i, d in zip(head["ID"], head["Description"], strict=True)]
    else:
        terms = capitalize(head["Description"])
    out = pd.DataFrame({"keyword": terms})
    out["col"] = _darken(matched_colors(-np.log10(head[metric].to_numpy(dtype=float))))
    out["fontsize"] = float(fontsize)
    return out


def keyword_boxes(
    df: pd.DataFrame, metric: str, *, top_word: int = 20,
    fontsize: tuple[float, float] = (6, 10), words_excluded: Sequence[str] | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """``anno_keys``: description keywords -- see the module note on GO."""
    import re

    rng = rng if rng is not None else np.random.default_rng(11)
    excluded = {w.lower() for w in (WORDS_EXCLUDED if words_excluded is None else words_excluded)}
    w = _words(df, metric, "Description", _desc_words)
    if not w.empty:
        keep = [
            not re.search(r"\[.*\]", k) and len(k) >= 1 and k.lower() not in excluded
            for k in w["keyword"]
        ]
        w = w[keep]
    return _top(w, top_word, fontsize, rng)


def feature_boxes(
    df: pd.DataFrame, metric: str, *, top_word: int = 20,
    fontsize: tuple[float, float] = (6, 10), rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """``anno_features``: genes scored by the enriched terms that contain them."""
    rng = rng if rng is not None else np.random.default_rng(11)
    return _top(_words(df, metric, "geneID", _genes), top_word, fontsize, rng)


def enrichment_boxes(
    df: pd.DataFrame,
    metric: str,
    *,
    kinds: Sequence[BoxKind],
    top_term: int = 5,
    show_termid: bool = False,
    top_word: int = 20,
    terms_fontsize: float = 8,
    keys_fontsize: tuple[float, float] = (6, 10),
    features_fontsize: tuple[float, float] = (6, 10),
    words_excluded: Sequence[str] | None = None,
    seed: int = 11,
) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
    """``{database: {kind: {group: words}}}`` for every database in ``df``.

    Each ``words`` frame has ``keyword``, ``col`` and ``fontsize`` (plus
    ``score``/``count``/``angle`` for keys and features) -- exactly the data
    frames SCP hands to ``anno_textbox``.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
    for db in pd.unique(df["Database"]):
        sub_db = df[df["Database"] == db]
        per_kind: dict[str, dict[str, pd.DataFrame]] = {}
        for kind in kinds:
            boxes: dict[str, pd.DataFrame] = {}
            for g in pd.unique(sub_db["Groups"]):
                sub = sub_db[sub_db["Groups"] == g]
                if kind == "terms":
                    b = term_boxes(sub, metric, top_term=top_term, show_termid=show_termid,
                                   fontsize=terms_fontsize)
                elif kind == "keys":
                    b = keyword_boxes(sub, metric, top_word=top_word, fontsize=keys_fontsize,
                                      words_excluded=words_excluded, rng=rng)
                else:
                    b = feature_boxes(sub, metric, top_word=top_word,
                                      fontsize=features_fontsize, rng=rng)
                if len(b):
                    boxes[str(g)] = b
            if boxes:
                per_kind[kind] = boxes
        if per_kind:
            out[str(db)] = per_kind
    return out
