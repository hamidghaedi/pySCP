"""Pure transforms behind the ``network``, ``enrichmap`` and ``wordcloud`` plots.

Everything here builds tables; nothing draws.  Keeping the graph construction,
the community partition and the word tables separate from the matplotlib code
is what lets them be compared number-for-number with what SCP's
``EnrichmentPlot`` / ``GSEAPlot`` compute (``tests/test_enrich_parity.py``).

Ported from SCP (Hao Zhang, https://github.com/zhanghao-njmu/SCP, GPL-3),
``R/SCP-plot.R`` lines 13060-13392 (EnrichmentPlot) and 14080-14530
(GSEAPlot), plus ``simplifyEnrichment``'s ``keyword_enrichment``.
"""

from __future__ import annotations

import math
import random
import re
import string
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..colors import blendcolors, luminance_text_color
from ..palettes import discrete_palette

__all__ = [
    "WORDS_EXCLUDED",
    "capitalize",
    "r_strwrap",
    "network_tables",
    "enrichmap_tables",
    "graph_layout",
    "cluster_membership",
    "keyword_enrichment",
    "term_document_counts",
    "wordcloud_table",
    "NetworkTables",
    "EnrichmapTables",
]

#: ``SCP::words_excluded`` (SCP ``data/words_excluded.rda``), verbatim.
WORDS_EXCLUDED: tuple[str, ...] = (
    "the", "is", "and", "or", "a", "in", "on", "under", "between", "of", "through",
    "via", "along", "that", "for", "with", "within", "without", "cell", "cellular",
    "dna", "rna", "protein", "peptide", "amino", "acid", "development", "involved",
    "organization", "system", "regulation", "positive", "negative", "response",
    "process", "processing", "small", "large", "change",
)

#: ``tm::stopwords("en")`` with apostrophes removed (``i'm`` -> ``im``), the
#: stoplist simplifyEnrichment applied when it built its GO term-document
#: matrix.  Used only by the GO keyword substitute.
_TM_STOPWORDS: frozenset[str] = frozenset(
    w.replace("'", "") for w in (
        "i me my myself we our ours ourselves you your yours yourself yourselves he "
        "him his himself she her hers herself it its itself they them their theirs "
        "themselves what which who whom this that these those am is are was were be "
        "been being have has had having do does did doing would should could ought "
        "i'm you're he's she's it's we're they're i've you've we've they've i'd "
        "you'd he'd she'd we'd they'd i'll you'll he'll she'll we'll they'll isn't "
        "aren't wasn't weren't hasn't haven't hadn't doesn't don't didn't won't "
        "wouldn't shan't shouldn't can't cannot couldn't mustn't let's that's who's "
        "what's here's there's when's where's why's how's a an the and but if or "
        "because as until while of at by for with about against between into through "
        "during before after above below to from up down in out on off over under "
        "again further then once here there when where why how all any both each few "
        "more most other some such no nor not only own same so than too very"
    ).split()
)

GO_DATABASES = ("GO", "GO_BP", "GO_CC", "GO_MF")

_BRACKETED = re.compile(r"\[.*\]")


# --------------------------------------------------------------------------- #
# string helpers (SCP utils.R)
# --------------------------------------------------------------------------- #
def capitalize(x: str) -> str:
    """SCP's ``capitalize()`` (``utils.R:852``): upper-case the first letter only
    when the first word (split on whitespace or ``-``) is entirely lower case, so
    ``"mRNA splicing"`` and ``"T cell activation"`` are left alone."""
    s = str(x)
    first = re.split(r"\s|-", s)[0] if s else ""
    if s and first == first.lower():
        return s[:1].upper() + s[1:]
    return s


def r_strwrap(text: str, width: int) -> str:
    """SCP's ``str_wrap()`` (``utils.R:872``), i.e. base R ``strwrap`` joined
    with newlines.

    Not :func:`textwrap.fill`: ``strwrap`` keeps every line strictly shorter
    than ``width``, collapses runs of whitespace (including existing newlines)
    and never breaks inside a word.  The vertex names of the network graph are
    these wrapped strings, so the difference is not cosmetic.
    """
    words = str(text).split()
    if not words:
        return ""
    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) < width:
            cur = f"{cur} {w}"
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return "\n".join(lines)


def _split_genes(x) -> list[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    return [g for g in str(x).split("/") if g != ""]


def _neglog10(p) -> np.ndarray:
    return -np.log10(np.asarray(p, dtype=float))


# --------------------------------------------------------------------------- #
# igraph plumbing
# --------------------------------------------------------------------------- #
def _igraph():
    try:
        import igraph
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "plot_type='network' / 'enrichmap' need python-igraph: "
            "`pip install scp-plot[graph]`"
        ) from e
    return igraph


@contextmanager
def _seeded_igraph(seed: int) -> Iterator[None]:
    """Point igraph's RNG at a private, seeded ``random.Random`` for the call.

    python-igraph draws from the ``random`` module by default; R's igraph draws
    from R's RNG after ``set.seed(seed)``.  The two streams differ, so seeded
    layouts are reproducible on each side but not across them.
    """
    ig = _igraph()
    ig.set_random_number_generator(random.Random(seed))
    try:
        yield
    finally:
        ig.set_random_number_generator(random)


def _make_graph(n: int, edges: Sequence[tuple[int, int]], weights: Sequence[float] | None):
    ig = _igraph()
    g = ig.Graph(n=n, edges=list(edges), directed=False)
    if weights is not None:
        g.es["weight"] = list(map(float, weights))
    return g


def graph_layout(
    n: int,
    edges: Sequence[tuple[int, int]],
    weights: Sequence[float] | None,
    layout: str = "fr",
) -> np.ndarray:
    """``layout_with_<layout>`` / ``layout_in_circle`` / ``layout_as_tree`` /
    ``layout_on_grid``, with R igraph's defaults.

    R's igraph reads the ``weight`` edge attribute when ``weights = NULL``; the
    Python bindings do not, so it is passed explicitly.  ``layout_with_fr``'s
    start temperature defaults to ``sqrt(vcount)`` in R but ``sqrt(vcount)/10``
    in Python; R's value is passed.  Call inside :func:`_seeded_igraph`.
    """
    g = _make_graph(n, edges, weights)
    w = "weight" if weights is not None and len(edges) else None
    if n == 0:
        return np.zeros((0, 2))
    if layout == "circle":
        lay = g.layout_circle()
    elif layout == "tree":
        lay = g.layout_reingold_tilford()
    elif layout == "grid":
        lay = g.layout_grid()
    elif layout == "fr":
        lay = g.layout_fruchterman_reingold(weights=w, niter=500, start_temp=math.sqrt(n))
    elif layout == "kk":
        lay = g.layout_kamada_kawai(weights=w)
    elif layout == "drl":
        lay = g.layout_drl(weights=w)
    elif layout == "lgl":
        lay = g.layout_lgl()
    elif layout == "mds":
        lay = g.layout_mds()
    elif layout == "graphopt":
        lay = g.layout_graphopt()
    elif layout == "dh":
        lay = g.layout_davidson_harel()
    else:
        raise ValueError(
            f"layout {layout!r} is not available; choose from fr, kk, drl, lgl, mds, "
            "graphopt, dh, circle, tree, grid (python-igraph has no layout_with_gem)"
        )
    return np.asarray(lay.coords, dtype=float).reshape(n, 2)


def _dendrogram_membership(n: int, merges: Sequence, qs: Sequence[float] | None) -> list[int]:
    """Cut a merge dendrogram where R igraph cuts it.

    R's ``cluster_fast_greedy`` / ``cluster_walktrap`` take the C core's
    membership at the *first* modularity maximum, numbered by
    ``igraph_community_to_membership``.  python-igraph's ``as_clustering()``
    takes the last maximum and renumbers communities by first appearance, which
    gives the same partition under different names -- and the names are the
    ``C1, C2, ...`` labels that pick each cluster's colour.
    """
    from igraph import community_to_membership

    merges = [tuple(m) for m in merges]
    if not merges:
        return list(range(n))
    steps = int(np.argmax(np.asarray(qs, dtype=float))) if qs else len(merges)
    return list(community_to_membership(merges, n, steps))


def cluster_membership(
    n: int,
    edges: Sequence[tuple[int, int]],
    weights: Sequence[float] | None,
    method: str = "fast_greedy",
) -> list[int]:
    """``cluster_<method>(graph)$membership``, 1-based as in R.

    The partition is the comparable quantity; the R and Python results agree
    exactly for the deterministic methods (``fast_greedy``, ``walktrap``,
    ``edge_betweenness``, ``leading_eigen``).  ``louvain``, ``leiden``,
    ``infomap``, ``spinglass`` and ``label_prop`` are randomised and only agree
    in distribution.
    """
    ig = _igraph()
    g = _make_graph(n, edges, weights)
    w = "weight" if weights is not None and len(edges) else None
    if n == 0:
        return []
    base = ig.GraphBase
    if method == "fast_greedy":
        merges, qs = base.community_fastgreedy(g, w)
        memb = _dendrogram_membership(n, merges, qs)
    elif method == "walktrap":
        merges, qs = base.community_walktrap(g, w, 4)
        memb = _dendrogram_membership(n, merges, qs)
    elif method == "edge_betweenness":
        merges, qs = base.community_edge_betweenness(g, False, w)
        memb = _dendrogram_membership(n, merges, qs)
    elif method == "leading_eigen":
        memb = list(g.community_leading_eigenvector(weights=w).membership)
    elif method == "louvain":
        memb = list(g.community_multilevel(weights=w).membership)
    elif method == "leiden":
        # R's cluster_leiden defaults: objective "CPM", resolution 1, beta 0.01,
        # two iterations -- the same as python-igraph's.
        memb = list(g.community_leiden(objective_function="CPM", weights=w).membership)
    elif method == "infomap":
        memb = list(g.community_infomap(edge_weights=w).membership)
    elif method == "spinglass":
        memb = list(g.community_spinglass(weights=w).membership)
    elif method == "label_prop":
        memb = list(g.community_label_propagation(weights=w).membership)
    else:
        raise ValueError(
            f"enrichmap_cluster {method!r} is not available; choose from fast_greedy, "
            "walktrap, edge_betweenness, leading_eigen, louvain, leiden, infomap, "
            "spinglass, label_prop (R's cluster_fluid_communities needs a community "
            "count EnrichmentPlot never passes, so it fails in R too)"
        )
    return [int(m) + 1 for m in memb]


# --------------------------------------------------------------------------- #
# network: the term-gene bipartite graph
# --------------------------------------------------------------------------- #
@dataclass
class NetworkTables:
    """What SCP's network branch builds before it draws."""

    nodes: pd.DataFrame  # name, class, metric, color, label_color, label
    edges: pd.DataFrame  # from, to, weight, color, plus integer source / target
    term_levels: list[str]
    term_colors: dict[str, str] = field(default_factory=dict)


def network_tables(
    df: pd.DataFrame,
    *,
    metric: str = "p.adjust",
    gene_col: str = "geneID",
    palette: str = "Spectral",
    palcolor: Sequence[str] | None = None,
    blendmode: str = "blend",
    descriptions: Sequence[str] | None = None,
) -> NetworkTables:
    """Term-gene graph of one panel (``SCP-plot.R:13071-13126``).

    ``df`` is the panel's rows in plotting order (already cut to ``top_term``).
    ``descriptions`` overrides the vertex names of the terms; by default they
    are ``str_wrap(capitalize(Description))``.  ``GSEAPlot`` passes its own,
    suffixed with the NES / p-value line.

    Vertices are the terms in row order, then each gene at its first
    appearance; edges are (term, gene) in row-then-gene order.  Term colours
    come from the discrete palette over the term levels; each gene is coloured
    by ``blendcolors`` of the terms it belongs to, in row order; label colours
    flip on the summed-RGB > 510 rule.
    """
    if descriptions is None:
        descriptions = [r_strwrap(capitalize(d), 50) for d in df["Description"]]
    desc = [str(d) for d in descriptions]
    term_levels = list(dict.fromkeys(desc))
    genes_per_term = [_split_genes(g) for g in df[gene_col]]
    term_metric = _neglog10(df[metric].to_numpy())

    unnest = [(d, g) for d, gs in zip(desc, genes_per_term, strict=True) for g in gs]
    genes = list(dict.fromkeys(g for _, g in unnest))

    colors = discrete_palette(term_levels, palette=palette, palcolor=palcolor)
    member: dict[str, list[str]] = {}
    for d, g in unnest:
        member.setdefault(g, []).append(colors[d])
    gene_colors = {g: blendcolors(member[g], mode=blendmode) for g in genes}  # type: ignore[arg-type]
    all_colors = {**colors, **gene_colors}

    names = desc + genes
    nodes = pd.DataFrame({
        "name": names,
        "class": ["term"] * len(desc) + ["gene"] * len(genes),
        "metric": np.concatenate([term_metric, np.zeros(len(genes))]),
    })
    nodes["color"] = [all_colors[n] for n in names]
    nodes["label_color"] = [luminance_text_color(c) for c in nodes["color"]]
    level_no = {t: i + 1 for i, t in enumerate(term_levels)}
    nodes["label"] = [level_no.get(n) if k == "term" else None
                      for n, k in zip(names, nodes["class"], strict=True)]
    index = {n: i for i, n in enumerate(names)}
    edges = pd.DataFrame({
        "from": [d for d, _ in unnest],
        "to": [g for _, g in unnest],
        "weight": 1.0,
    })
    edges["color"] = [colors[d] for d in edges["from"]]
    edges["source"] = [index[d] for d in edges["from"]]
    edges["target"] = [index[g] for g in edges["to"]]
    return NetworkTables(nodes, edges, term_levels, dict(colors))


# --------------------------------------------------------------------------- #
# enrichmap: the term-term overlap graph
# --------------------------------------------------------------------------- #
@dataclass
class EnrichmapTables:
    """What SCP's enrichmap branch builds before it draws."""

    nodes: pd.DataFrame       # ID, Description, Count, metric, clusters, keyword1, keyword2
    edges: pd.DataFrame       # from, to, weight, source, target
    cluster_levels: list[str]
    keyword1: dict[str, str]  # cluster -> term keywords
    keyword2: dict[str, str]  # cluster -> gene keywords
    legend_title: str
    legend_labels: dict[str, str]
    mark_labels: dict[str, str]


def overlap_edges(gene_sets: Sequence[Sequence[str]]) -> list[tuple[int, int, int]]:
    """``t(combn(ID, 2))`` weighted by ``length(intersect(...))``, zero rows dropped.

    The weight is the number of shared genes -- a raw count, not a Jaccard or
    overlap coefficient.  That is SCP's similarity measure.
    """
    sets = [set(s) for s in gene_sets]
    out = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            w = len(sets[i] & sets[j])
            if w > 0:
                out.append((i, j, w))
    return out


def _top_keywords(
    rows: pd.DataFrame, cluster_levels: Sequence[str], nlabel: int, sep: str
) -> dict[str, str]:
    """``arrange(desc(score)) |> slice_head(n) |> paste(collapse = sep)`` per cluster.

    ``rows`` holds one row per distinct (keyword, cluster) and is already in
    dplyr's group order (keyword in C-locale order).  The sort on score is
    stable, so ties keep that order -- as dplyr's radix sort does.
    """
    out: dict[str, str] = {}
    ranked = rows.sort_values("score", ascending=False, kind="mergesort")
    for c in cluster_levels:
        sub = ranked[ranked["clusters"] == c].head(nlabel)
        if len(sub):
            out[c] = sep.join(sub["keyword"].astype(str))
    return out


def _keyword_scores(nodes: pd.DataFrame, tokens: list[list[str]], p: np.ndarray) -> pd.DataFrame:
    """Unnest ``tokens`` and score each (keyword, cluster) by ``sum(-log10(p))``."""
    rec = [(t, c, s) for toks, c, s in zip(tokens, nodes["clusters"], _neglog10(p), strict=True)
           for t in toks]
    tab = pd.DataFrame(rec, columns=["keyword", "clusters", "s"])
    if tab.empty:
        return pd.DataFrame(columns=["keyword", "clusters", "score", "count"])
    agg = tab.groupby(["keyword", "clusters"], sort=False).agg(
        score=("s", "sum"), count=("s", "size")).reset_index()
    # dplyr's group_by order: keyword (C locale), then the cluster factor
    agg["_c"] = agg["clusters"].map({c: i for i, c in enumerate(sorted(set(agg["clusters"]),
                                                                     key=_cluster_key))})
    agg = agg.sort_values(["keyword", "_c"], kind="mergesort").drop(columns="_c")
    return agg.reset_index(drop=True)


def _cluster_key(c: str) -> int:
    return int(str(c)[1:])


def enrichmap_tables(
    df: pd.DataFrame,
    *,
    metric: str = "p.adjust",
    gene_col: str = "geneID",
    cluster: str = "fast_greedy",
    label: str = "term",
    nlabel: int = 4,
    show_keyword: bool = False,
    character_width: int = 50,
    words_excluded: Sequence[str] = WORDS_EXCLUDED,
    descriptions: Sequence[str] | None = None,
    counts: Sequence[int] | None = None,
) -> EnrichmapTables:
    """Term-term graph, communities and cluster labels of one panel
    (``SCP-plot.R:13161-13283``).

    Vertices are the terms (by ``ID``) in row order; an edge joins every pair
    sharing at least one gene, weighted by the shared-gene count.  Communities
    are ``cluster_<cluster>`` on that weighted graph, named ``C<membership>``.

    Each cluster gets two labels: ``keyword1`` from the terms (the top
    ``nlabel`` descriptions by ``-log10(p)``, newline-joined; or, with
    ``show_keyword``, the top words of those descriptions scored by
    ``sum(-log10(p))``) and ``keyword2`` from the genes (top ``nlabel`` genes by
    the same score, space-joined).  ``label="term"`` writes ``keyword1`` on the
    marks and ``keyword2`` in the legend; ``"feature"`` swaps them.
    """
    ids = [str(i) for i in df["ID"]]
    if descriptions is None:
        descriptions = [capitalize(d) for d in df["Description"]]
    desc = [str(d) for d in descriptions]
    gene_sets = [_split_genes(g) for g in df[gene_col]]
    p = df[metric].to_numpy(dtype=float)
    met = _neglog10(p)

    trip = overlap_edges(gene_sets)
    pairs = [(i, j) for i, j, _ in trip]
    weights = [float(w) for _, _, w in trip]
    memb = cluster_membership(len(ids), pairs, weights, cluster)
    clusters = [f"C{m}" for m in memb]
    cluster_levels = [f"C{m}" for m in sorted(set(memb))]

    nodes = pd.DataFrame({
        "ID": ids,
        "Description": desc,
        "Count": (list(counts) if counts is not None else
                  df["Count"].to_numpy() if "Count" in df else [len(s) for s in gene_sets]),
        "metric": met,
        "clusters": clusters,
    })
    excluded = {w.lower() for w in words_excluded}

    if show_keyword:
        toks = [re.split(r"\s|\n", d.lower()) for d in desc]
        kw = _keyword_scores(nodes, toks, p)
        kw["keyword"] = [capitalize(k) for k in kw["keyword"]]
        kw = kw[~kw["keyword"].str.contains(_BRACKETED)]
        kw = kw[kw["keyword"].str.len() >= 1]
        kw = kw[~kw["keyword"].str.lower().isin(excluded)]
        kw = kw.drop_duplicates()
        keyword1 = _top_keywords(kw, cluster_levels, nlabel, " ")
        keyword1 = {c: r_strwrap(v, character_width) for c, v in keyword1.items()}
    else:
        shown = ([r_strwrap(d, character_width) for d in desc] if label == "term" else desc)
        keyword1 = {}
        order = np.argsort(-met, kind="stable")
        for c in cluster_levels:
            picked = list(dict.fromkeys(shown[i] for i in order if clusters[i] == c))[:nlabel]
            keyword1[c] = "\n".join(picked)

    kw2 = _keyword_scores(nodes, gene_sets, p).drop_duplicates()
    keyword2 = _top_keywords(kw2, cluster_levels, nlabel, " ")
    keyword2 = {c: r_strwrap(v, character_width) for c, v in keyword2.items()}

    nodes["keyword1"] = [keyword1.get(c) for c in clusters]
    nodes["keyword2"] = [keyword2.get(c) for c in clusters]

    edges = pd.DataFrame({
        "from": [ids[i] for i, _ in pairs],
        "to": [ids[j] for _, j in pairs],
        "weight": weights,
        "source": [i for i, _ in pairs],
        "target": [j for _, j in pairs],
    })

    def lab(d: dict[str, str]) -> dict[str, str]:
        return {c: f"{c}:\n{d[c]}" if c in d else f"{c}:\nNA" for c in cluster_levels}

    if label == "term":
        title, legend_labels, mark = "Feature", lab(keyword2), keyword1
    else:
        title, legend_labels, mark = "Term", lab(keyword1), keyword2
    return EnrichmapTables(nodes, edges, cluster_levels, keyword1, keyword2,
                           title, legend_labels, dict(mark))


# --------------------------------------------------------------------------- #
# wordcloud
# --------------------------------------------------------------------------- #
def _p_adjust_holm(p: np.ndarray) -> np.ndarray:
    """``p.adjust(p)`` -- R's default method is Holm, not BH."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(p, kind="stable")
    adj = np.minimum(1.0, np.maximum.accumulate((n - np.arange(n)) * p[o]))
    out = np.empty(n)
    out[o] = adj
    return out


def _tokenise_term_name(name: str) -> list[str]:
    """The term-document tokeniser of the GO keyword substitute.

    Lower-case, delete ASCII punctuation (so ``"T-helper"`` becomes
    ``"thelper"``, as tm's ``removePunctuation`` does), delete digits, split on
    whitespace, drop one-character tokens and tm's English stoplist.  This
    mirrors the vocabulary of simplifyEnrichment's shipped GO matrix (no
    digits, no punctuation, no stopwords) -- see ``tools/parity/enrich_parity.R``
    for the R implementation the fixtures were built with.
    """
    s = str(name).lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"[0-9]", "", s)
    return [t for t in s.split() if len(t) >= 2 and t not in _TM_STOPWORDS]


def term_document_counts(names: Mapping[str, str]) -> dict[str, dict[str, int]]:
    """keyword -> {term ID: occurrences}, the substitute's term-document matrix."""
    tdm: dict[str, dict[str, int]] = {}
    for tid, nm in names.items():
        for t in _tokenise_term_name(nm):
            row = tdm.setdefault(t, {})
            row[tid] = row.get(tid, 0) + 1
    return tdm


def keyword_enrichment(
    term_id: Sequence[str],
    tdm: Mapping[str, Mapping[str, int]],
    columns: Sequence[str],
    *,
    min_bg: int = 5,
    min_term: int = 2,
) -> pd.DataFrame:
    """``simplifyEnrichment:::keyword_enrichment``, transcribed.

    ``tdm`` maps keyword -> {term ID: occurrences}; ``columns`` are all the
    term IDs of the matrix (its columns, including terms with no keyword).
    For each keyword whose total frequency is at least ``min_bg``: a one-sided
    (``"greater"``) Fisher test of "term contains the keyword" against "term
    is in ``term_id``", over the columns.  Keywords hit by fewer than
    ``min_term`` selected terms are dropped; ``padj`` is ``p.adjust``'s
    default, which is Holm.  Sorted by ``padj`` then ``p``, ties in keyword
    (C-locale) order.
    """
    from scipy.stats import fisher_exact

    cols = set(map(str, columns))
    sel = set(map(str, term_id)) & cols
    n = len(cols)
    rec = []
    for kw in sorted(tdm):
        row = tdm[kw]
        if sum(row.values()) < min_bg:
            continue
        hits = {t for t, v in row.items() if v and t in cols}
        s11 = len(hits & sel)
        if s11 < min_term:
            continue
        s21 = len(hits) - s11
        s12 = len(sel) - s11
        s22 = n - s11 - s12 - s21
        p = fisher_exact([[s11, s12], [s21, s22]], alternative="greater").pvalue
        rec.append((kw, s11, s11 + s21, float(p)))
    out = pd.DataFrame(rec, columns=["keyword", "n_term", "n_bg", "p"])
    out["padj"] = _p_adjust_holm(out["p"].to_numpy())
    return out.sort_values(["padj", "p"], kind="mergesort").reset_index(drop=True)


def _frequency_words(df: pd.DataFrame, metric: str, excluded: set[str] | None,
                     tokens: list[list[str]]) -> pd.DataFrame:
    """The ``unnest -> group_by(keyword) -> reframe(score, count)`` path."""
    s = _neglog10(df[metric].to_numpy())
    rec = [(t, si) for toks, si in zip(tokens, s, strict=True) for t in toks]
    tab = pd.DataFrame(rec, columns=["keyword", "s"])
    if tab.empty:
        return pd.DataFrame(columns=["keyword", "score", "count"])
    agg = (tab.groupby("keyword", sort=False)
           .agg(score=("s", "sum"), count=("s", "size")).reset_index())
    agg = agg.sort_values("keyword", kind="mergesort")  # dplyr's C-locale group order
    if excluded is not None:
        agg = agg[~agg["keyword"].str.contains(_BRACKETED)]
        agg = agg[agg["keyword"].str.len() >= 1]
        agg = agg[~agg["keyword"].str.lower().isin(excluded)]
    return agg.reset_index(drop=True)


def wordcloud_table(
    df: pd.DataFrame,
    *,
    metric: str = "p.adjust",
    word_type: str = "term",
    gene_col: str = "geneID",
    top_word: int = 100,
    words_excluded: Sequence[str] = WORDS_EXCLUDED,
    background: Mapping[str, str] | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Words, scores, counts and angles of one wordcloud panel
    (``SCP-plot.R:13308-13392`` / ``14455-14520``).

    ``word_type="feature"``: genes, scored by ``sum(-log10(p))`` over the
    terms containing them, ``count`` = number of such terms; no stoplist.

    ``word_type="term"`` on a non-GO database: the descriptions are lower-cased
    and split on single spaces, scored the same way, then the bracketed /
    empty / ``words_excluded`` tokens are dropped.  This is SCP's path exactly.

    ``word_type="term"`` on a GO database: SCP calls
    ``simplifyEnrichment::keyword_enrichment_from_GO``, a one-sided Fisher
    test of each keyword against simplifyEnrichment's term-document matrix of
    all ~44,000 GO term names, with ``score = -log10(padj)`` (Holm) and
    ``count = n_term``.  That matrix ships only with the R package, so this
    path is a **documented substitute**:

    * with ``background`` -- a ``{GO ID: term name}`` map of the whole
      ontology, e.g. from ``goatools``' OBO parser -- the identical test
      (:func:`keyword_enrichment`, verified against simplifyEnrichment's own
      function on the same matrix) runs against a matrix built from it with
      the tokeniser of :func:`_tokenise_term_name`.  This approaches SCP's
      numbers; it matches them only as far as the GO release and tm's
      tokenisation agree;
    * without one, the frequency path above is used, which is what SCP does
      for every non-GO database.  The words are the same kind of words but
      the scores are ``sum(-log10(p))`` and the counts are occurrences, not a
      keyword-enrichment ``-log10(padj)`` and ``n_term``.  Testing against
      only the terms in the table instead of all of GO has no power at the
      sizes involved (every ``padj`` comes out 1), so that is not offered as
      a default.

    ``angle`` is 90 with probability 0.4, drawn before the ``top_word`` cut,
    as in SCP.  R's and numpy's generators differ, so the angles do too.
    """
    rng = rng if rng is not None else np.random.default_rng(11)
    excluded = {w.lower() for w in words_excluded}
    if word_type == "feature":
        toks = [_split_genes(g) for g in df[gene_col]]
        out = _frequency_words(df, metric, None, toks)
    else:
        dbs = set(map(str, df["Database"])) if "Database" in df else set()
        if background is not None and dbs and dbs <= set(GO_DATABASES):
            bg = {str(k): str(v) for k, v in background.items()}
            tdm = term_document_counts(bg)
            k = keyword_enrichment(df["ID"].astype(str).tolist(), tdm, list(bg))
            out = pd.DataFrame({"keyword": k["keyword"], "score": _neglog10(k["padj"]),
                                "count": k["n_term"].astype(int)})
            out = out[~out["keyword"].str.contains(_BRACKETED)]
            out = out[out["keyword"].str.len() >= 1]
            out = out[~out["keyword"].str.lower().isin(excluded)].drop_duplicates()
        else:
            toks = [str(d).lower().split(" ") for d in df["Description"]]
            out = _frequency_words(df, metric, excluded, toks)
    out = out.reset_index(drop=True)
    out["angle"] = 90 * rng.choice([0, 1], size=len(out), replace=True, p=[0.6, 0.4])
    order = np.argsort(-out["score"].to_numpy(dtype=float), kind="stable")[:top_word]
    return out.iloc[order].reset_index(drop=True)
