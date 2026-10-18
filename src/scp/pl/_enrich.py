"""Enrichment and GSEA plots.

These read **tables**, not AnnData internals, which is what makes them
backend-agnostic.  The contract is fixed in ``scp.io``; ``gseapy``,
``decoupler`` and ``GOATOOLS`` output all map onto it with renaming.

The graph and word tables behind ``network`` / ``enrichmap`` / ``wordcloud``
live in :mod:`scp.pl._enrich_graph` and the drawing in
:mod:`scp.pl._enrich_draw`.

Ported from SCP (Hao Zhang, https://github.com/zhanghao-njmu/SCP, GPL-3),
``R/SCP-plot.R``.  Spec: ``docs/porting_briefs/trajectory_enrichment.md`` §8-§9.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..layout import LegendColumn, PanelGrid
from ..palettes import color_ramp, continuous_palette, discrete_palette
from ..theme import theme_scp, theme_scp_rc
from . import _enrich_draw as draw
from . import _enrich_graph as eg

__all__ = ["enrichment_plot", "gsea_plot", "gsea_scores", "adjust_layout"]

EnrichPlotType = Literal["bar", "dot", "lollipop", "network", "enrichmap", "wordcloud", "comparison"]
GSEAPlotType = Literal["line", "bar", "network", "enrichmap", "wordcloud", "comparison"]

_GRAPH_TYPES = ("network", "enrichmap")
_PANEL_DEFAULT = {"network": (4.5, 4.5), "enrichmap": (4.5, 4.5), "wordcloud": (3.6, 3.6)}


# --------------------------------------------------------------------------- #
# shared preprocessing
# --------------------------------------------------------------------------- #
def _metric(pvalue_cutoff: float | None, padjust_cutoff: float | None) -> tuple[str, float]:
    """SCP's rule: ``padjust_cutoff`` wins; ``pvalue_cutoff`` only applies
    when ``padjust_cutoff is None``."""
    if pvalue_cutoff is None and padjust_cutoff is None:
        raise ValueError("one of 'pvalue_cutoff' or 'padjust_cutoff' must be given")
    if padjust_cutoff is None:
        return "pvalue", float(pvalue_cutoff)  # type: ignore[arg-type]
    return "p.adjust", float(padjust_cutoff)


def _levels(s: pd.Series) -> list[str]:
    """First-appearance levels, R's ``factor(x, levels = unique(x))``."""
    return [str(x) for x in dict.fromkeys(s.astype(str))]


def _id_list(id_use) -> list[str]:
    if id_use is None:
        return []
    if isinstance(id_use, Mapping):
        return [str(i) for v in id_use.values() for i in v]
    if isinstance(id_use, str):
        return [id_use]
    return [str(i) for i in id_use]


def _filter_ids(df: pd.DataFrame, id_use) -> pd.DataFrame:
    if id_use is None or (not isinstance(id_use, Mapping) and len(_id_list(id_use)) == 0):
        return df
    if isinstance(id_use, Mapping):
        missing = [k for k in id_use if str(k) not in set(df["Groups"].astype(str))]
        if missing:
            raise ValueError(f"names in 'id_use' are not groups: {missing}")
        parts = [df[df["ID"].astype(str).isin([str(i) for i in ids])
                    & (df["Groups"].astype(str) == str(g))] for g, ids in id_use.items()]
        return pd.concat(parts) if parts else df.iloc[:0]
    return df[df["ID"].astype(str).isin(_id_list(id_use))]


def _prepare_enrichment(
    enrichment: pd.DataFrame,
    *,
    plot_type: str,
    split_by: Sequence[str],
    group_use,
    id_use,
    pvalue_cutoff,
    padjust_cutoff,
    top_term: int | None,
):
    """``SCP-plot.R:12750-12835``: cutoffs, ordering and the panel split.

    Returns ``(enrichment, panels, metric, metric_value, top_term)`` where
    ``enrichment`` is the table after the group/ID filters (needed by
    ``comparison``, which also shows non-significant cells) and ``panels`` is a
    list of ``(key, (database, group), df)`` in R's ``split()`` order -- the
    first ``split_by`` factor varies fastest -- each ``df`` already cut to
    ``top_term`` rows per (Database x Group) and sorted by the metric.
    """
    metric, value = _metric(pvalue_cutoff, padjust_cutoff)
    missing = [c for c in ("Description", metric) if c not in enrichment.columns]
    if missing:
        raise ValueError(
            f"enrichment table is missing {missing}; expected the columns in "
            "scp.io.ENRICHMENT_COLUMNS (scp.io.enrichment_from_gseapy builds them)"
        )
    bad = [s for s in split_by if s not in ("Database", "Groups")]
    if bad:
        raise ValueError("'split_by' must be 'Database', 'Groups', or both")
    split_by = list(split_by)
    if plot_type in _GRAPH_TYPES and len(split_by) == 1:
        split_by = ["Database", "Groups"]
    df = enrichment.copy()
    if "ID" not in df:
        df["ID"] = df["Description"]
    if "Groups" not in df:
        df["Groups"] = "all"
    if "Database" not in df:
        df["Database"] = "custom"
    df["Groups"] = df["Groups"].astype(str)
    df["Database"] = df["Database"].astype(str)
    glev, dlev = _levels(df["Groups"]), _levels(df["Database"])
    if group_use is not None:
        df = df[df["Groups"].isin([str(g) for g in ([group_use] if isinstance(group_use, str)
                                                    else group_use)])]
    if id_use is not None and len(_id_list(id_use)):
        top_term = None  # R: id_use forces topTerm <- Inf
        df = _filter_ids(df, id_use)
    if top_term is None and not (id_use is not None and len(_id_list(id_use))):
        top_term = 100 if plot_type == "enrichmap" else 6

    ids = set(_id_list(id_use))
    sig = df[(df[metric].astype(float) < value) | df["ID"].astype(str).isin(ids)]
    sig = sig.iloc[np.argsort(sig[metric].to_numpy(dtype=float), kind="stable")]
    if sig.empty:
        raise ValueError(
            f"no terms survive the cutoffs ({metric} < {value}); nothing to plot"
        )
    levels = {"Database": dlev, "Groups": glev}
    # split(df, ~a + b): interaction levels with the first factor fastest
    combos: list[dict[str, str]] = [{}]
    for f in split_by:
        combos = [{**c, f: lv} for lv in levels[f] for c in combos]
    panels = []
    for c in combos:
        mask = np.ones(len(sig), dtype=bool)
        for f, lv in c.items():
            mask &= (sig[f] == lv).to_numpy()
        sub = sig[mask]
        if sub.empty:
            continue
        # split(df, list(Database, Groups)) |> head(topTerm) |> rbind
        parts = []
        for g in glev:
            for d in dlev:
                part = sub[(sub["Database"] == d) & (sub["Groups"] == g)]
                if len(part):
                    # the wordcloud branch never applies topTerm in R
                    cut = None if plot_type == "wordcloud" else top_term
                    parts.append(part if cut is None else part.head(cut))
        sub = pd.concat(parts)
        key = " | ".join(c[f] for f in ("Database", "Groups") if f in c)
        panels.append((key, (c.get("Database"), c.get("Groups")), sub))
    return df, panels, metric, value, top_term


def _ratio(x) -> float:
    """``"k/n"`` -> ``k / n``, as R's ``strsplit(x, "/")``; a number passes."""
    s = str(x)
    if "/" not in s:
        return float(s)
    a, b = s.split("/")
    return float(a) / float(b)


def _wrap_desc(descriptions, width: int) -> list[str]:
    return [eg.r_strwrap(eg.capitalize(d), width) for d in descriptions]


def _rescale_norm(v: np.ndarray) -> Normalize:
    """The continuous scale's range.  ``scales::rescale`` maps a zero-width
    range to the midpoint of the output, so a panel whose values are all equal
    is drawn in the middle colour, not the first one."""
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    if not hi > lo:
        return Normalize(lo - 0.5, lo + 0.5)
    return Normalize(lo, hi)


def _color_levels(enrichment: pd.DataFrame, color_by: str) -> list[str]:
    """``levels(df[[color_by]])``: first-appearance order over the whole input
    table, set before ``group_use`` / ``db`` filter it (``SCP-plot.R:12773-12779``)."""
    if color_by in enrichment:
        return _levels(enrichment[color_by])
    default = {"Database": "custom", "Groups": "all"}
    if color_by in default:
        return [default[color_by]]
    raise ValueError(f"color_by={color_by!r} is not a column of the enrichment table")


def enrichment_panel_frame(
    df: pd.DataFrame,
    *,
    plot_type: str,
    metric: str,
    character_width: int = 50,
) -> tuple[pd.DataFrame, list[str]]:
    """The data of one ``bar`` / ``dot`` / ``lollipop`` panel
    (``SCP-plot.R:12900-13017``).

    ``df`` is one panel of :func:`_prepare_enrichment`: significant terms in
    stable metric order, cut to ``top_term`` per (Database x Group).  Adds
    ``metric = -log10(<metric>)``, the capitalised and ``strwrap``-ped
    ``Description``, the parsed ``GeneRatio`` / ``FoldEnrichment`` where the
    type uses them, and ``_y``, the 0-based position on the term axis (0 at
    the bottom).  Returns the rows sorted by ``_y`` and the axis levels.

    The term order is R's factor levels:

    * ``bar``: ``unique(rev(Description))`` -- the most significant on top,
      and of two tied terms the one listed first on top;
    * ``dot``: rows ordered by ``GeneRatio`` decreasing (stable), then
      ``unique(rev(...))`` -- the highest ratio on top;
    * ``lollipop``: ``unique(Description[order(FoldEnrichment)])`` -- the
      highest fold enrichment on top.
    """
    d = df.reset_index(drop=True).copy()
    need = {"dot": ("GeneRatio", "Count"), "lollipop": ("GeneRatio", "BgRatio")}
    missing = [c for c in need.get(plot_type, ()) if c not in d]
    if missing:
        raise ValueError(f"plot_type={plot_type!r} needs the column(s) {missing}")
    d["metric"] = -np.log10(d[metric].to_numpy(dtype=float))
    d["Description"] = _wrap_desc(d["Description"], character_width)
    if plot_type in ("dot", "lollipop"):
        d["GeneRatio"] = [_ratio(x) for x in d["GeneRatio"]]
    if plot_type == "dot":
        d = d.iloc[np.argsort(-d["GeneRatio"].to_numpy(dtype=float), kind="stable")]
        levels = list(dict.fromkeys(reversed(d["Description"].tolist())))
    elif plot_type == "lollipop":
        d["BgRatio"] = [_ratio(x) for x in d["BgRatio"]]
        d["FoldEnrichment"] = d["GeneRatio"] / d["BgRatio"]
        o = d.iloc[np.argsort(d["FoldEnrichment"].to_numpy(dtype=float), kind="stable")]
        levels = list(dict.fromkeys(o["Description"]))
    else:
        levels = list(dict.fromkeys(reversed(d["Description"].tolist())))
    pos = {lv: i for i, lv in enumerate(levels)}
    d["_y"] = [pos[x] for x in d["Description"]]
    d = d.iloc[np.argsort(d["_y"].to_numpy(), kind="stable")].reset_index(drop=True)
    return d, levels


def _comparison_levels(sub: pd.DataFrame, ids: Sequence[str]) -> list[str]:
    """``factor(Description, levels = unique(rev(terms[ids])))``: the first id
    ends up at the top of the axis."""
    first: dict[str, str] = {}
    for i, d in zip(sub["ID"].astype(str), sub["Description"], strict=True):
        first.setdefault(i, d)
    return list(dict.fromkeys(reversed([first[i] for i in ids if i in first])))


def enrichment_comparison_frame(
    enrichment: pd.DataFrame,
    panels,
    *,
    metric: str,
    metric_value: float,
    character_width: int = 50,
    compare_only_sig: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """The data of ``EnrichmentPlot(plot_type = "comparison")``
    (``SCP-plot.R:12838-12868``): every row, significant or not, of the terms
    that made some panel's top list, so a term can be compared across groups."""
    ids: list[str] = []
    for _, _, df in panels:
        ids = list(dict.fromkeys([*ids, *df["ID"].astype(str)]))
    sub = enrichment[enrichment["ID"].astype(str).isin(ids)].copy()
    if "GeneRatio" in sub:
        sub["GeneRatio"] = [_ratio(x) for x in sub["GeneRatio"]]
    if "BgRatio" in sub:
        sub["BgRatio"] = [_ratio(x) for x in sub["BgRatio"]]
        sub["EnrichmentScore"] = sub["GeneRatio"] / sub["BgRatio"]
    sub["Description"] = _wrap_desc(sub["Description"], character_width)
    levels = _comparison_levels(sub, ids)
    if compare_only_sig:
        sub = sub[sub[metric].astype(float) < metric_value]
    return sub.reset_index(drop=True), levels


def _draw_comparison_panels(
    frame: pd.DataFrame,
    levels: list[str],
    *,
    size_col: str,
    fill_values: np.ndarray,
    cmap,
    norm: Normalize,
    fill_label: str,
    edge: Sequence[str],
    stroke: float,
    extra_legend: tuple[str, list] | None,
    panel_size: tuple[float, float],
    size_label: str,
    nan_color: str = "#CCCCCC",
) -> Figure:
    """``facet_grid(Database ~ ., scales = "free")``: one panel per database."""
    th = theme_scp()
    dbs = _levels(frame["Database"]) if len(frame) else ["all"]
    pg = PanelGrid(len(dbs), panel_size=panel_size, ncol=1, keys=dbs, theme=th,
                   legend=LegendColumn(position="right"))
    fill = np.array([nan_color if not np.isfinite(v) else cmap(norm(v)) for v in fill_values],
                    dtype=object)
    for d in dbs:
        ax = pg.axes[d]
        m = (frame["Database"].astype(str) == d).to_numpy()
        sub = frame[m]
        y_levels = [lv for lv in levels if lv in set(sub["Description"])]
        x_levels = [g for g in dict.fromkeys(frame["Groups"].astype(str))
                    if g in set(sub["Groups"].astype(str))]
        draw.draw_comparison(ax, sub, size_col=size_col, size_max=6.0,
                             fill=list(fill[m]), edge=[e for e, k in zip(edge, m, strict=True) if k],
                             stroke=stroke, y_levels=y_levels, x_levels=x_levels)
        ax.set_title(d, fontsize=th.size("strip.text"))
    pg.legend.add(size_label, draw.size_area_handles(frame[size_col].to_numpy(dtype=float), 6.0))
    if extra_legend is not None:
        pg.legend.add(*extra_legend)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cb = pg.colorbar(sm)
    cb.set_label(fill_label, fontsize=th.size("legend.title"))
    cb.outline.set_edgecolor("black")
    return pg.finish()


def _handle_signature(h) -> tuple:
    """What a legend key shows, so two panels' keys can be compared."""
    label = str(h.get_label())
    if isinstance(h, Line2D):
        return ("line", label, str(h.get_marker()), round(float(h.get_markersize()), 6),
                to_hex(h.get_markerfacecolor(), keep_alpha=True),
                to_hex(h.get_color(), keep_alpha=True), round(float(h.get_linewidth()), 6))
    if isinstance(h, Patch):
        return ("patch", label, to_hex(h.get_facecolor(), keep_alpha=True),
                to_hex(h.get_edgecolor(), keep_alpha=True))
    if isinstance(h, draw.GlyphKey):
        return ("glyph", label, h.glyph, round(h.size_pt, 6), h.color)
    if isinstance(h, draw.NumberedKey):
        return ("numbered", label, h.number, h.color)
    return ("other", id(h))


def _colorbar_signature(sm: ScalarMappable, label: str) -> tuple:
    norm = sm.norm
    ramp = np.asarray(sm.get_cmap()(np.linspace(0, 1, 16))).round(6)
    return (label, float(norm.vmin), float(norm.vmax), ramp.tobytes())  # type: ignore[arg-type]


class _LegendSink:
    """One panel's view of a :class:`_PanelLegends`, for the drawing helpers."""

    def __init__(self, owner: _PanelLegends, key: str) -> None:
        self.owner, self.key = owner, key

    def add(self, title: str, handles: Sequence) -> None:
        self.owner.legend(self.key, title, handles)


class _PanelLegends:
    """The guides of a multi-panel enrichment / GSEA figure, laid out as SCP's
    are.

    Every panel is its own ggplot in SCP, combined with
    ``patchwork::wrap_plots(plotlist = plist)`` -- with no
    ``guides = "collect"`` (``SCP-plot.R:13430``, ``14535``) -- so each panel
    keeps its own legends and colourbars, beside it, titled as in R (no panel
    suffix: the legend sits next to the panel it describes).  The one
    departure: a guide that is *identical* in every panel (the same title,
    keys and scale) is drawn once, in the figure's legend column, as patchwork
    would when collecting -- repeating it N times adds width and no
    information.  Within a panel the guides keep their registration order
    (SCP puts the size legend first, ``order = 1``).
    """

    def __init__(self, pg: PanelGrid) -> None:
        self.pg = pg
        # (kind, title) -> [(panel key, payload)], in first-registration order
        self.items: dict[tuple[str, str], list[tuple[str, object]]] = {}

    def sink(self, key: str) -> _LegendSink:
        return _LegendSink(self, key)

    def legend(self, key: str, title: str, handles: Sequence) -> None:
        self.items.setdefault(("legend", title), []).append((key, list(handles)))

    def colorbar(self, key: str, mappable: ScalarMappable, label: str) -> None:
        self.items.setdefault(("colorbar", label), []).append((key, mappable))

    @staticmethod
    def _signature(kind: str, title: str, payload) -> tuple:
        if kind == "colorbar":
            return _colorbar_signature(payload, title)
        return tuple(_handle_signature(h) for h in payload)

    def _colorbar(self, mappable: ScalarMappable, label: str, ax=None) -> None:
        cb = self.pg.colorbar(mappable, ax=ax)
        cb.set_label(label, fontsize=self.pg.theme.size("legend.title"))
        cb.outline.set_edgecolor("black")

    def finish(self) -> None:
        n = len(self.pg.keys)
        shared = set()
        for (kind, title), entries in self.items.items():
            if len(entries) == n and len({self._signature(kind, title, p)
                                          for _, p in entries}) == 1:
                shared.add((kind, title))
                if kind == "colorbar":
                    self._colorbar(entries[0][1], title)  # type: ignore[arg-type]
                else:
                    self.pg.legend.add(title, entries[0][1])  # type: ignore[arg-type]
        per_key: dict[str, list[tuple[str, str, object]]] = {}
        for (kind, title), entries in self.items.items():
            if (kind, title) in shared:
                continue
            for key, payload in entries:
                per_key.setdefault(key, []).append((kind, title, payload))
        for key in self.pg.keys:
            ax = self.pg.axes[key]
            for kind, title, payload in per_key.get(key, []):
                if kind == "colorbar":
                    self._colorbar(payload, title, ax=ax)  # type: ignore[arg-type]
                else:
                    self.pg.legend_beside(ax, title, payload)  # type: ignore[arg-type]
        self.items.clear()


def _layout_for(n: int, edges, weights, layout: str) -> np.ndarray:
    return eg.graph_layout(n, edges, weights, layout)


# --------------------------------------------------------------------------- #
# enrichment_plot
# --------------------------------------------------------------------------- #
def enrichment_plot(
    enrichment: pd.DataFrame,
    *,
    plot_type: EnrichPlotType = "bar",
    split_by: Sequence[str] = ("Database", "Groups"),
    color_by: str = "Database",
    group_use: Sequence[str] | None = None,
    id_use: Sequence[str] | dict[str, Sequence[str]] | None = None,
    pvalue_cutoff: float | None = None,
    padjust_cutoff: float | None = 0.05,
    top_term: int | None = None,
    compare_only_sig: bool = False,
    character_width: int = 50,
    palette: str = "Spectral",
    palcolor: Sequence[str] | None = None,
    # network / enrichmap
    network_layout: str = "fr",
    network_labelsize: float = 5.0,
    network_blendmode: str = "blend",
    network_layoutadjust: bool = True,
    network_adjscale: float = 60.0,
    network_adjiter: int = 100,
    enrichmap_layout: str = "fr",
    enrichmap_cluster: str = "fast_greedy",
    enrichmap_label: Literal["term", "feature"] = "term",
    enrichmap_labelsize: float = 5.0,
    enrichmap_nlabel: int = 4,
    enrichmap_show_keyword: bool = False,
    enrichmap_mark: Literal["ellipse", "hull"] = "ellipse",
    enrichmap_expand: tuple[float, float] = (0.5, 0.5),
    # wordcloud
    top_word: int = 100,
    word_type: Literal["term", "feature"] = "term",
    word_size: tuple[float, float] = (2.0, 8.0),
    words_excluded: Sequence[str] | None = None,
    keyword_background: Mapping[str, str] | None = None,
    panel_size: tuple[float, float] | None = None,
    nrow: int | None = None,
    ncol: int | None = None,
    seed: int = 11,
) -> Figure:
    """Over-representation results, seven ways. Milestone 14.

    Port of ``EnrichmentPlot`` (SCP, Hao Zhang,
    https://github.com/zhanghao-njmu/SCP, ``R/SCP-plot.R:12727``).

    Required columns (see ``scp.io.ENRICHMENT_COLUMNS``): ``ID``,
    ``Description``, ``GeneRatio``, ``BgRatio``, ``pvalue``, ``p.adjust``,
    ``geneID`` (``/``-joined), ``Count``, ``Groups``, ``Database``.

    The metric is ``p.adjust`` unless ``padjust_cutoff is None``, in which
    case it is ``pvalue`` and ``pvalue_cutoff`` applies; terms are kept when
    ``metric < cutoff`` (strict, as in R) and sorted by it.  Panels follow R's
    ``split()`` over ``split_by`` in first-appearance level order, and
    ``top_term`` (default 6, or 100 for ``enrichmap``) is applied per
    (Database x Group).  ``id_use`` selects terms by ID and lifts ``top_term``.

    Term labels are SCP's ``capitalize()`` then ``str_wrap(character_width)``
    (base R ``strwrap``: every line shorter than the width, never broken at a
    hyphen), and every label that wraps is set in italic.  Each panel is its
    own ggplot in R, so each has its own colour and size scale; a panel whose
    metric is constant is drawn in the palette's middle colour, as
    ``scales::rescale`` maps a zero-width range to 0.5.  Sizes follow
    ``scale_size(range = c(3, 6))``: diameter ``3 + 3 * sqrt(rescaled)`` mm.
    SCP combines the panels with ``wrap_plots`` without collecting guides, so
    each panel's legends and colourbar sit beside it; a guide identical in
    every panel is drawn once, in the legend column.

    ``bar``
        ``-log10(metric)``, filled by ``color_by`` (``palette_scp`` over that
        column's first-appearance levels in the whole input), with the
        ``Count`` at each bar end and the value axis fixed at
        ``[0, 1.3 * max]``.  The most significant term is on top; of tied
        terms, the one listed first.
    ``dot``
        x = ``GeneRatio``, rows ordered by it (highest on top), fill
        ``-log10(metric)``, size ``Count``, axis ``[0, 1.3 * max]``.
    ``lollipop``
        x = ``FoldEnrichment`` = ``GeneRatio / BgRatio``, rows ordered by it,
        a black stem under a stem in the fill colour, size ``GeneRatio``,
        axis ``[0, 1.2 * max]``.

    ``network``
        The term-gene bipartite graph: terms are numbered nodes coloured from
        ``palette``; each gene is a label filled with the ``network_blendmode``
        blend of its terms' colours.  Laid out with igraph's
        ``network_layout`` (``"fr"`` by default; python-igraph is required,
        ``pip install scp-plot[graph]``) then label-decollided by
        :func:`adjust_layout`.
    ``enrichmap``
        The term-term graph: an edge for every pair of terms sharing genes,
        weighted by the shared-gene count; communities by igraph
        ``cluster_<enrichmap_cluster>`` (fast-greedy modularity by default);
        each community outlined (``enrichmap_mark``) and labelled with its top
        ``enrichmap_nlabel`` terms, while the legend lists its top genes
        (``enrichmap_label="feature"`` swaps the two).
    ``wordcloud``
        Words from term descriptions (``word_type="term"``) or genes
        (``"feature"``), sized by count and coloured by summed
        ``-log10(metric)``.  For GO databases SCP instead scores keywords by
        ``simplifyEnrichment::keyword_enrichment_from_GO``, whose GO
        term-document matrix ships only with the R package.  **Documented
        divergence**: without ``keyword_background`` the GO panels use the
        frequency scoring SCP applies to every other database; with a
        ``{GO ID: term name}`` map of the ontology the same Fisher/Holm test is
        run against it.  See :func:`scp.pl._enrich_graph.wordcloud_table`.
    ``comparison``
        A Groups x term dot plot, one panel per database, of every term in
        some panel's top list -- including the cells where it is not
        significant, which are filled grey ("Non-sig").

    What does not match R: igraph layouts are seeded from Python's RNG, not
    R's, so node *positions* differ (the graph, weights and communities do
    not); ggforce places mark labels in free space, here they are pushed
    radially outward; the ``hull`` mark is a convex hull, not ``concaveman``'s
    concave one; and word placement is a square spiral, not ggwordcloud's
    pixel-mask packer.  R's argument ``enrlichmap_nlabel`` (sic) is
    ``enrichmap_nlabel`` here.
    """
    if plot_type not in ("bar", "dot", "lollipop", "network", "enrichmap", "wordcloud",
                         "comparison"):
        raise ValueError(f"unknown plot_type {plot_type!r}")
    full, panels, metric, value, top = _prepare_enrichment(
        enrichment, plot_type=plot_type, split_by=split_by, group_use=group_use,
        id_use=id_use, pvalue_cutoff=pvalue_cutoff, padjust_cutoff=padjust_cutoff,
        top_term=top_term)
    excluded = tuple(words_excluded) if words_excluded is not None else eg.WORDS_EXCLUDED
    psize = panel_size or _PANEL_DEFAULT.get(plot_type, (3.2, 2.6))
    th = theme_scp(aspect_ratio=1) if plot_type in ("network", "enrichmap", "wordcloud") \
        else theme_scp()

    if plot_type == "comparison":
        frame, levels = enrichment_comparison_frame(
            full, panels, metric=metric, metric_value=value,
            character_width=character_width, compare_only_sig=compare_only_sig)
        ramp = continuous_palette(palette=palette, palcolor=palcolor, n=100, reverse=True)
        cmap = LinearSegmentedColormap.from_list("cmp", ramp, N=256)
        vals = frame[metric].to_numpy(dtype=float)
        hi = min(value, 1.0)
        fill_vals = np.where((vals >= 0) & (vals <= hi), vals, np.nan)
        extra = None if compare_only_sig else ("Non-sig", [Line2D(
            [], [], marker="o", linestyle="none", markerfacecolor="#CCCCCC",
            markeredgecolor="black", markersize=3 * draw.DOT, label="")])
        return _draw_comparison_panels(
            frame, levels, size_col="GeneRatio", fill_values=fill_vals, cmap=cmap,
            norm=Normalize(0, hi), fill_label=metric, edge=["black"] * len(frame), stroke=0.5,
            extra_legend=extra, panel_size=panel_size or (3.6, 3.6), size_label="GeneRatio")

    keys = [k for k, _, _ in panels]
    if plot_type in ("network", "enrichmap", "wordcloud"):
        pg = PanelGrid(len(panels), panel_size=psize, nrow=nrow, ncol=ncol, keys=keys, theme=th,
                       legend=LegendColumn(position="right"))
        _draw_graph_family(
            pg, panels, plot_type=plot_type, metric=metric, gene_col="geneID",
            palette=palette, palcolor=palcolor, character_width=character_width,
            network_layout=network_layout, network_labelsize=network_labelsize,
            network_blendmode=network_blendmode, network_layoutadjust=network_layoutadjust,
            network_adjscale=network_adjscale, network_adjiter=network_adjiter,
            enrichmap_layout=enrichmap_layout, enrichmap_cluster=enrichmap_cluster,
            enrichmap_label=enrichmap_label, enrichmap_labelsize=enrichmap_labelsize,
            enrichmap_nlabel=enrichmap_nlabel, enrichmap_show_keyword=enrichmap_show_keyword,
            enrichmap_mark=enrichmap_mark, enrichmap_expand=enrichmap_expand,
            top_word=top_word, word_type=word_type, word_size=word_size,
            words_excluded=excluded, keyword_background=keyword_background,
            panel_size=psize, seed=seed)
        return pg.finish()

    ramp = continuous_palette(palette=palette, palcolor=palcolor, n=100)
    cmap = LinearSegmentedColormap.from_list("enr", ramp, N=256)
    fs = th.size("axis.text") * 0.75
    if plot_type == "bar":
        bar_colors = discrete_palette(_color_levels(enrichment, color_by), palette=palette,
                                      palcolor=palcolor)
    frames = [enrichment_panel_frame(g, plot_type=plot_type, metric=metric,
                                     character_width=character_width) for _, _, g in panels]
    # panel_size is the panel; PanelGrid measures the term labels (and the
    # colourbar) and grows the figure around them
    pg = PanelGrid(len(panels), panel_size=psize, nrow=nrow, ncol=ncol, keys=keys, theme=th,
                   legend=LegendColumn(position="right"))
    legs = _PanelLegends(pg)
    for (key, _dg, _g), (frame, levels) in zip(panels, frames, strict=True):
        ax = pg.axes[key]
        ax.set_title(str(key), fontsize=th.size("strip.text"))
        if plot_type == "bar":
            cols = [bar_colors.get(str(v), "#CCCCCC") for v in frame[color_by]]
            draw.draw_enrichment_bar(ax, frame, levels, colors=cols, metric=metric, fontsize=fs)
            continue
        vals = frame["metric"].to_numpy(dtype=float)
        norm = _rescale_norm(vals)
        fill = [to_hex(c) for c in cmap(norm(vals))]
        size_col = "Count" if plot_type == "dot" else "GeneRatio"
        sv = frame[size_col].to_numpy(dtype=float)
        sizes = draw._area_size(sv, float(np.nanmin(sv)), float(np.nanmax(sv)), (3.0, 6.0))
        draw.draw_enrichment_points(ax, frame, levels, plot_type=plot_type, fill=fill,
                                    sizes=sizes, fontsize=fs)
        # every panel is its own ggplot in R, so each has its own scales;
        # the size guide comes first (order = 1)
        legs.legend(key, size_col, draw.size_range_handles(sv))
        legs.colorbar(key, ScalarMappable(cmap=cmap, norm=norm), f"-log10({metric})")
    legs.finish()
    return pg.finish()


def _panel_mm(pg: PanelGrid, key: str) -> tuple[float, float]:
    fig = pg.fig
    pos = pg.axes[key].get_position()
    w, h = fig.get_size_inches()
    return pos.width * w * 25.4, pos.height * h * 25.4


def _draw_graph_family(
    pg: PanelGrid,
    panels,
    *,
    plot_type: str,
    metric: str,
    gene_col: str,
    palette: str,
    palcolor,
    character_width: int,
    network_layout: str,
    network_labelsize: float,
    network_blendmode: str,
    network_layoutadjust: bool,
    network_adjscale: float,
    network_adjiter: int,
    enrichmap_layout: str,
    enrichmap_cluster: str,
    enrichmap_label: str,
    enrichmap_labelsize: float,
    enrichmap_nlabel: int,
    enrichmap_show_keyword: bool,
    enrichmap_mark: str,
    enrichmap_expand: tuple[float, float],
    top_word: int,
    word_type: str,
    word_size: tuple[float, float],
    words_excluded: Sequence[str],
    keyword_background: Mapping[str, str] | None,
    panel_size: tuple[float, float],
    seed: int,
    descriptions: Mapping[str, Sequence[str]] | None = None,
    counts: Mapping[str, Sequence[int]] | None = None,
) -> None:
    """Draw ``network`` / ``enrichmap`` / ``wordcloud`` panels, shared by
    :func:`enrichment_plot` and :func:`gsea_plot` exactly as R shares the code."""
    th = pg.theme
    legs = _PanelLegends(pg)
    if plot_type in _GRAPH_TYPES:
        eg._igraph()
    if plot_type == "wordcloud":
        rng = np.random.default_rng(seed)
        ramp = continuous_palette(palette=palette, palcolor=palcolor, n=100)
        tables = []
        for _key, _dg, df in panels:
            parts = []
            groups = list(dict.fromkeys(zip(df["Database"].astype(str), df["Groups"].astype(str),
                                            strict=True)))
            for d, g in (groups if word_type == "term" else [(None, None)]):
                sub = df if d is None else df[(df["Database"].astype(str) == d)
                                              & (df["Groups"].astype(str) == g)]
                bg = keyword_background
                parts.append(eg.wordcloud_table(
                    sub, metric=metric, word_type=word_type, gene_col=gene_col,
                    top_word=top_word, words_excluded=words_excluded, background=bg, rng=rng))
            tables.append(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())
        # PanelGrid panels are their final size from the start (colourbars sit
        # outside them), so the words are packed into the panel as drawn
        for (key, _dg, _df), tab in zip(panels, tables, strict=True):
            ax = pg.axes[key]
            ax.set_title(str(key), fontsize=th.size("strip.text"))
            if tab.empty:
                continue
            sc = tab["score"].to_numpy(dtype=float)
            sm = ScalarMappable(cmap=LinearSegmentedColormap.from_list("wc", ramp, N=256),
                                norm=Normalize(float(sc.min()), float(sc.max())))
            cnt = tab["count"].to_numpy(dtype=float)
            br = sorted(set(int(math.ceil(v)) for v in np.linspace(cnt.min(), cnt.max(), 3)))
            sizes = draw._area_size(br, float(cnt.min()), float(cnt.max()), word_size)
            # guides(size = guide_legend(override.aes = list(label = "G"), order = 1))
            legs.legend(key, "Count", [draw.GlyphKey("G", float(s) * draw.PT, str(b))
                                       for b, s in zip(br, sizes, strict=True)])
            legs.colorbar(key, sm, "Score")
        for (key, _dg, _df), tab in zip(panels, tables, strict=True):
            draw.draw_wordcloud(pg.axes[key], tab, ramp, word_size=word_size)
        legs.finish()
        return

    with eg._seeded_igraph(seed):
        for key, _dg, df in panels:
            ax = pg.axes[key]
            ax.set_title(str(key), fontsize=th.size("strip.text"))
            if plot_type == "network":
                desc = (descriptions[key] if descriptions is not None
                        else _wrap_desc(df["Description"], character_width))
                nt = eg.network_tables(df, metric=metric, gene_col=gene_col, palette=palette,
                                       palcolor=palcolor, blendmode=network_blendmode,
                                       descriptions=desc)
                edges = list(zip(nt.edges["source"], nt.edges["target"], strict=True))
                lay = _layout_for(len(nt.nodes), edges, nt.edges["weight"].tolist(),
                                  network_layout)
                if network_layoutadjust and len(lay):
                    width = np.where(nt.nodes["class"] == "term", 8,
                                     nt.nodes["name"].str.len()).astype(float)
                    lay = adjust_layout(lay, width, edges, height=2.0, scale=network_adjscale,
                                        iterations=network_adjiter, seed=seed)
                draw.draw_network(ax, nt, lay, labelsize=network_labelsize, legend=legs.sink(key),
                                  legend_title="Term")
            else:
                edesc = descriptions[key] if descriptions is not None else None
                et = eg.enrichmap_tables(
                    df, metric=metric, gene_col=gene_col, cluster=enrichmap_cluster,
                    label=enrichmap_label, nlabel=enrichmap_nlabel,
                    show_keyword=enrichmap_show_keyword, character_width=character_width,
                    words_excluded=words_excluded, descriptions=edesc,
                    counts=counts[key] if counts is not None else None)
                edges = list(zip(et.edges["source"], et.edges["target"], strict=True))
                lay = _layout_for(len(et.nodes), edges, et.edges["weight"].tolist(),
                                  enrichmap_layout)
                colors = discrete_palette(et.cluster_levels, palette=palette, palcolor=palcolor)
                draw.draw_enrichmap(ax, et, lay, colors, mark=enrichmap_mark,
                                    labelsize=enrichmap_labelsize, expand=enrichmap_expand,
                                    panel_mm=_panel_mm(pg, key), legend=legs.sink(key))
    legs.finish()


# --------------------------------------------------------------------------- #
# GSEA
# --------------------------------------------------------------------------- #
def _key_parts(key: str) -> tuple[str, str]:
    """``"<Group>-<Database>"`` -> ``(group, database)``."""
    g, _, d = str(key).rpartition("-")
    return (g, d) if g else (str(key), "custom")


def _per_key_mapping(results) -> Mapping | None:
    """The per-key results of SCP's shape, or of a bare
    ``{key: {"result": df, "geneList": ..., ...}}`` mapping; else ``None``."""
    if isinstance(results, pd.DataFrame) or not isinstance(results, Mapping):
        return None
    if "enrichment" in results:
        return dict(results.get("results", {}))
    if results and "table" not in results and all(
            isinstance(v, Mapping) and "result" in v for v in results.values()):
        return results
    return None


def _gsea_sources(results) -> dict[str, Mapping]:
    """Per key, whatever of ``geneList`` / ``geneSets`` / ``exponent`` was
    passed: the ``gseaResult`` slots ``gsInfo`` reads (``SCP-plot.R:14545``)."""
    per = _per_key_mapping(results) or {}
    return {str(k): v for k, v in per.items()
            if isinstance(v, Mapping) and "geneList" in v and "geneSets" in v}


def _gsea_tables(results) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    """Normalise the accepted inputs to (flat table, per-key results, curves).

    Accepted: a flat GSEA DataFrame; ``{"table": df, "curves": {...}}``;
    SCP's own shape, ``{"enrichment": df, "results": {"<Group>-<Database>":
    df | {"result": df, "geneList": ..., "geneSets": ..., "exponent": ...}}}``;
    or the bare ``{"<Group>-<Database>": {"result": df, ...}}`` mapping.
    Without per-key results they are rebuilt from the flat table, ordered as
    RunGSEA orders them (groups fastest within each database).
    """
    curves: dict = {}
    per: dict[str, pd.DataFrame] = {}
    keyed = _per_key_mapping(results)
    if isinstance(results, pd.DataFrame):
        table = results.copy()
    elif keyed is not None:
        for k, v in keyed.items():
            r = pd.DataFrame(v["result"] if isinstance(v, Mapping) else v).copy()
            g, d = _key_parts(str(k))
            if "Groups" not in r:
                r["Groups"] = g
            if "Database" not in r:
                r["Database"] = d
            per[str(k)] = r
        if "enrichment" in results:
            table = pd.DataFrame(results["enrichment"]).copy()
        else:
            table = pd.concat(list(per.values()), ignore_index=True) if per else pd.DataFrame()
        curves = results.get("curves", {}) if "enrichment" in results else {}
    else:
        table = pd.DataFrame(results.get("table", [])).copy()
        curves = results.get("curves", {})
    if table.empty and not per:
        raise ValueError("no GSEA results to plot")
    if not per and len(table):
        t = table.copy()
        if "Groups" not in t:
            t["Groups"] = "all"
        if "Database" not in t:
            t["Database"] = "custom"
        for d in _levels(t["Database"]):
            for g in _levels(t["Groups"]):
                sub = t[(t["Groups"].astype(str) == g) & (t["Database"].astype(str) == d)]
                if len(sub):
                    per[f"{g}-{d}"] = sub.reset_index(drop=True)
    return table, per, curves


def gsea_select_ids(
    res: pd.DataFrame,
    *,
    metric: str,
    metric_value: float,
    top_term: int | None,
    direction: str,
    id_use=None,
    group: str | None = None,
) -> list[str]:
    """``GSEAPlot``'s term selection (``SCP-plot.R:14084-14108``).

    Significant sets (``metric < cutoff``) ordered by the metric; the top
    ``top_term`` with NES > 0 and with NES < 0; ``"both"`` takes
    ``ceiling(top_term / 2)`` of each.  ``top_term=None`` (the wordcloud) keeps
    them all.  ``id_use`` overrides: a list is used as is, a dict is looked up
    by the result's group and intersected with its IDs.
    """
    if id_use is not None and len(_id_list(id_use)):
        if isinstance(id_use, Mapping):
            want = {str(i) for i in id_use.get(group, [])}
            return [i for i in res["ID"].astype(str) if i in want]
        return _id_list(id_use)
    f = res[res[metric].astype(float) < metric_value]
    f = f.iloc[np.argsort(f[metric].to_numpy(dtype=float), kind="stable")]
    up = f[f["NES"] > 0]["ID"].astype(str).tolist()
    down = f[f["NES"] < 0]["ID"].astype(str).tolist()
    if top_term is None:
        pick = {"pos": up, "neg": down, "both": up + down}[direction]
    else:
        up, down = up[:top_term], down[:top_term]
        half = math.ceil(top_term / 2)
        pick = {"pos": up[:top_term], "neg": down[:top_term],
                "both": (up[:half] + down[:half])[:top_term]}[direction]
    return list(dict.fromkeys(pick))


def _p_sig(p: float) -> str:
    if p > 0.05:
        return "ns  "
    if p > 0.01:
        return "*   "
    if p > 0.001:
        return "**  "
    if p > 0.0001:
        return "*** "
    return "****"


def r_format_sci(values: Sequence[float], digits: int = 3) -> list[str]:
    """``format(x, digits = 3, scientific = TRUE)`` on a vector: one shared
    mantissa width -- the fewest significant digits (at most ``digits``) that
    represent every element -- and a common field width."""
    vals = [float(v) for v in values]
    need = 1
    for v in vals:
        if not np.isfinite(v) or v == 0:
            continue
        mant = f"{v:.{digits - 1}e}".split("e")[0].replace("-", "").replace(".", "")
        need = max(need, len(mant.rstrip("0")) or 1)
    out = [f"{v:.{need - 1}e}" for v in vals]
    width = max((len(o) for o in out), default=0)
    return [o.rjust(width) for o in out]


def _r_num(x: float) -> str:
    """``as.character(round(x, 3))``: up to 15 significant digits, no trailing zeros.

    Rounded with R's algorithm (``src/nmath/fround.c``, R >= 4.0): take the
    nearer of the two decimal candidates in double arithmetic, ties to the even
    one.  Python's ``round`` rounds the exact binary value instead, so
    ``round(0.0005, 3)`` is 0.001 in Python but 0 in R."""
    v = float(x)
    if not math.isfinite(v):
        return format(v, ".15g")
    sign = -1.0 if v < 0 else 1.0
    v = abs(v)
    x10 = v * 1e3
    i10 = math.floor(x10)
    lo, hi = i10 / 1e3, math.ceil(x10) / 1e3
    du, dd = hi - v, v - lo
    out = hi if (du < dd or (du == dd and i10 % 2 == 1)) else lo
    return "0" if out == 0 else format(sign * out, ".15g")


def gsea_network_descriptions(df: pd.DataFrame, metric: str, character_width: int = 50) -> list[str]:
    """Term vertex names of the GSEA network: the wrapped description plus a
    ``(NES=..., p.adjust=..., stars)`` line (``SCP-plot.R:14113-14122``)."""
    p = df[metric].to_numpy(dtype=float)
    fmt = r_format_sci(p, 3)
    return [f"{d}\n(NES={_r_num(n)}, {metric}={f}, {_p_sig(pv)})"
            for d, n, f, pv in zip(_wrap_desc(df["Description"], character_width), df["NES"],
                                   fmt, p, strict=True)]


def gsea_comparison_frame(
    table: pd.DataFrame,
    *,
    metric: str,
    metric_value: float,
    top_term: int,
    direction: str,
    group_use: Sequence[str] | None = None,
    id_use=None,
    character_width: int = 50,
    compare_only_sig: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """The data of ``GSEAPlot(plot_type = "comparison")``
    (``SCP-plot.R:13668-13706``)."""
    t = table.copy()
    t["Groups"] = t["Groups"].astype(str)
    groups = ([str(g) for g in group_use] if group_use is not None else _levels(t["Groups"]))
    t = t[t["Groups"].isin(groups)]
    if id_use is not None and len(_id_list(id_use)):
        t = _filter_ids(t, id_use)
        ids = _id_list(id_use)
    else:
        ids = []
        for g in groups:
            sel = gsea_select_ids(t[t["Groups"] == g], metric=metric, metric_value=metric_value,
                                  top_term=top_term, direction=direction)
            ids = list(dict.fromkeys([*ids, *sel]))
    sub = t[t["ID"].astype(str).isin(ids)].copy()
    sub["Description"] = _wrap_desc(sub["Description"], character_width)
    levels = _comparison_levels(sub, ids)
    sub["Significant"] = sub[metric].astype(float) < metric_value
    if compare_only_sig:
        sub = sub[sub["Significant"]]
    if direction == "pos":
        sub = sub[sub["NES"] > 0]
    elif direction == "neg":
        sub = sub[sub["NES"] < 0]
    return sub.reset_index(drop=True), levels


def gsea_plot(
    results: dict[str, dict] | pd.DataFrame,
    *,
    plot_type: GSEAPlotType = "line",
    direction: Literal["pos", "neg", "both"] = "pos",
    group_use: Sequence[str] | None = None,
    id_use: Sequence[str] | dict[str, Sequence[str]] | None = None,
    pvalue_cutoff: float | None = None,
    padjust_cutoff: float | None = 0.05,
    top_term: int | None = None,
    compare_only_sig: bool = False,
    n_coregene: int = 10,
    sample_coregene: bool = False,
    features_label: Sequence[str] | None = None,
    line_width: float = 1.5,
    line_color: str | Sequence[str] = "#6BB82D",
    character_width: int = 50,
    palette: str = "Spectral",
    palcolor: Sequence[str] | None = None,
    # network / enrichmap
    network_layout: str = "fr",
    network_labelsize: float = 5.0,
    network_blendmode: str = "blend",
    network_layoutadjust: bool = True,
    network_adjscale: float = 60.0,
    network_adjiter: int = 100,
    enrichmap_layout: str = "fr",
    enrichmap_cluster: str = "fast_greedy",
    enrichmap_label: Literal["term", "feature"] = "term",
    enrichmap_labelsize: float = 5.0,
    enrichmap_nlabel: int = 4,
    enrichmap_show_keyword: bool = False,
    enrichmap_mark: Literal["ellipse", "hull"] = "ellipse",
    enrichmap_expand: tuple[float, float] = (0.5, 0.5),
    # wordcloud
    top_word: int = 100,
    word_type: Literal["term", "feature"] = "term",
    word_size: tuple[float, float] = (2.0, 8.0),
    words_excluded: Sequence[str] | None = None,
    keyword_background: Mapping[str, str] | None = None,
    panel_size: tuple[float, float] | None = None,
    nrow: int | None = None,
    ncol: int | None = None,
    seed: int = 11,
) -> Figure:
    """The classic three-panel GSEA figure, plus five alternatives. Milestone 15.

    Port of ``GSEAPlot`` (SCP, Hao Zhang, https://github.com/zhanghao-njmu/SCP,
    ``R/SCP-plot.R:13563``).

    Every type except ``comparison`` is drawn once per result key
    (``"<Group>-<Database>"``, after ``group_use``), as R loops
    ``names(res)``, and the panels are laid out by :class:`~scp.layout.PanelGrid`.
    Gene sets are selected per key by SCP's rule (``SCP-plot.R:13743-13756``):
    ``metric < cutoff`` (strict; the metric is ``p.adjust``, or ``pvalue`` when
    ``padjust_cutoff is None``), ordered by the metric with ties in table
    order, the ``top_term`` best with NES > 0 (``"pos"``) or NES < 0
    (``"neg"``); ``"both"`` takes ``ceiling(top_term / 2)`` of each, up first.
    ``id_use`` overrides the selection (a dict is looked up by group).

    ``line`` -- one block per key, three sub-panels at relative heights
    ``1.5 : 0.5 : 1``, with every selected set overlaid:

    1. the running enrichment score over red/blue half-plane shading.  With a
       single set: a dashed crosshair to the peak
       (``which.max(abs(runningScore))``), a triangle there (up, red when
       NES > 0; down, blue otherwise), the first ``n_coregene`` core genes
       (or ``features_label``) marked and labelled, and the description as
       title with the ``(NES=..., p.adjust=..., stars)`` line as an italic
       subtitle.  With several sets the lines are coloured from ``palette``
       (unless ``line_color`` has one colour per set) and listed in a "Term"
       legend;
    2. one band of hit ticks per set, the first set on top; for a single set
       the ticks are black over a heat strip of the ranking metric (clipped to
       its 2-98% quantiles, 99 bins per sign on SCP's two ramps);
    3. the ranked metric as grey30 stems, labelled "Positively" /
       "Negatively correlated", with a dashed line and label at the zero
       cross ``median(x[y == y[which.min(abs(y))]])``.

    The rank axis carries ggplot's default breaks (``scales::extended_breaks``,
    about five), and the core-gene labels are kept inside the panel, as
    ggrepel keeps them.  The key is written down the right of its block, with
    the block's "Term" legend beside it.  ``line`` needs the
    running scores: pass ``geneList`` (a ranked :class:`pandas.Series`),
    ``geneSets`` (``{ID: genes}``) and ``exponent`` per key, or precomputed
    :func:`gsea_scores` frames as ``curves`` (see *Input* below).

    ``bar`` -- one panel per key: NES bars in ``order(NES)`` (most negative at
    the bottom), filled Pos/Neg from ``palette`` (``palette_scp(c("Neg",
    "Pos"))``, so ``palcolor`` is read as (Pos, Neg)) with alpha
    ``-log10(metric)`` on ggplot's 0.1-1 range, the description written
    inside the panel at zero on the side away from the bar, and x limits
    ``+-max|NES|``.  A "Direction" legend appears for ``direction="both"``.

    ``comparison``
        Groups x gene-set dots: size ``setSize``, fill NES on a symmetric
        scale, outline black when significant and grey when not; one panel per
        database.
    ``network`` / ``enrichmap`` / ``wordcloud``
        :func:`enrichment_plot`'s, run per result key (``"<Group>-<Database>"``)
        on the core-enrichment genes; the network's term names carry an
        ``(NES=..., p.adjust=..., stars)`` line and the enrichmap counts the
        core genes.  Term selection is SCP's: the ``top_term`` most significant
        sets per direction (``direction="both"`` takes half of each); the
        wordcloud uses every significant set of the direction.  ``top_term``
        defaults to 6, or 100 for ``enrichmap``.  The same divergences as
        :func:`enrichment_plot` apply.

    *Input* -- any of:

    * a flat GSEA table (``scp.io.GSEA_COLUMNS``; ``Groups`` / ``Database``
      default to ``"all"`` / ``"custom"``), split into keys as RunGSEA does;
    * SCP's ``{"enrichment": df, "results": {key: res}}``, where ``res`` is
      the key's result table or a dict ``{"result": df, "geneList": Series,
      "geneSets": {ID: genes}, "exponent": 1.0}`` holding the ``gseaResult``
      slots ``gsInfo`` reads;
    * the bare per-key mapping ``{key: {"result": df, "geneList": ...,
      "geneSets": ..., "exponent": ...}}``;
    * ``{"table": df, "curves": curves}`` with ``curves`` either
      ``{ID: gsea_scores(...)}`` or ``{key: {ID: gsea_scores(...)}}``.

    ``gseapy.prerank`` output maps on with renaming (``Term`` ->
    ``ID``/``Description``, ``FDR q-val`` -> ``p.adjust``, ``Lead_genes`` ->
    ``core_enrichment``).

    Deliberate divergences: R's label "Negtively correlated" is spelt
    correctly; R's three ``line`` sub-plots train their x scales separately
    (they disagree by up to one rank), here they share one; core-gene labels
    are nudged as ggrepel's ``nudge_x`` / ``nudge_y`` place them but not
    repelled; and a sign with a single ranked value, where R's ``cut()`` stops
    on non-unique breaks, gets the first colour of its ramp.
    """
    table, per, curves = _gsea_tables(results)
    if top_term is None:
        top_term = 100 if plot_type == "enrichmap" else 6

    if plot_type == "comparison":
        metric, value = _metric(pvalue_cutoff, padjust_cutoff)
        frame, levels = gsea_comparison_frame(
            table, metric=metric, metric_value=value, top_term=top_term, direction=direction,
            group_use=group_use, id_use=id_use, character_width=character_width,
            compare_only_sig=compare_only_sig)
        if frame.empty:
            raise ValueError("no gene sets survive the cutoffs")
        if "Database" not in frame:
            frame["Database"] = "custom"
        nes = frame["NES"].to_numpy(dtype=float)
        lim = float(np.max(np.abs(nes)))
        cmap = LinearSegmentedColormap.from_list(
            "nes", continuous_palette(palette=palette, palcolor=palcolor, n=100), N=256)
        edge = ["black" if s else "#E5E5E5" for s in frame["Significant"]]
        title = f"Significant\n({metric}<{value:g})"
        extra = None if compare_only_sig else (title, [
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                   markeredgecolor=c, markeredgewidth=1.2, markersize=3 * draw.DOT, label=lab)
            for c, lab in (("black", "TRUE"), ("#E5E5E5", "FALSE"))])
        return _draw_comparison_panels(
            frame, levels, size_col="setSize", fill_values=nes, cmap=cmap,
            norm=Normalize(-lim, lim), fill_label="NES", edge=edge, stroke=0.8,
            extra_legend=extra, panel_size=panel_size or (3.6, 3.6), size_label="setSize")

    metric, value = _metric(pvalue_cutoff, padjust_cutoff)
    selected = _gsea_selection(
        per, group_use=group_use, metric=metric, metric_value=value,
        top_term=None if plot_type == "wordcloud" else top_term, direction=direction,
        id_use=id_use)
    if not selected:
        raise ValueError("no gene sets survive the cutoffs")
    keys = [k for k, _, _ in selected]

    if plot_type in ("network", "enrichmap", "wordcloud"):
        panels = []
        descriptions: dict[str, list[str]] = {}
        counts: dict[str, list[int]] = {}
        for k, res, ids in selected:
            df = res.set_index(res["ID"].astype(str)).loc[ids].reset_index(drop=True)
            panels.append((k, (None, None), df))
            if plot_type == "network":
                descriptions[k] = gsea_network_descriptions(df, metric, character_width)
            elif plot_type == "enrichmap":
                descriptions[k] = _wrap_desc(df["Description"], character_width)
                counts[k] = [len(eg._split_genes(g)) for g in df["core_enrichment"]]
        th = theme_scp(aspect_ratio=1)
        psize = panel_size or _PANEL_DEFAULT[plot_type]
        pg = PanelGrid(len(panels), panel_size=psize, nrow=nrow, ncol=ncol, keys=keys, theme=th,
                       legend=LegendColumn(position="right"))
        excluded = tuple(words_excluded) if words_excluded is not None else eg.WORDS_EXCLUDED
        _draw_graph_family(
            pg, panels, plot_type=plot_type, metric=metric, gene_col="core_enrichment",
            palette=palette, palcolor=palcolor, character_width=character_width,
            network_layout=network_layout, network_labelsize=network_labelsize,
            network_blendmode=network_blendmode, network_layoutadjust=network_layoutadjust,
            network_adjscale=network_adjscale, network_adjiter=network_adjiter,
            enrichmap_layout=enrichmap_layout, enrichmap_cluster=enrichmap_cluster,
            enrichmap_label=enrichmap_label, enrichmap_labelsize=enrichmap_labelsize,
            enrichmap_nlabel=enrichmap_nlabel, enrichmap_show_keyword=enrichmap_show_keyword,
            enrichmap_mark=enrichmap_mark, enrichmap_expand=enrichmap_expand,
            top_word=top_word, word_type=word_type, word_size=word_size,
            words_excluded=excluded, keyword_background=keyword_background,
            panel_size=psize, seed=seed,
            descriptions=descriptions if plot_type != "wordcloud" else None,
            counts=counts if plot_type == "enrichmap" else None)
        return pg.finish()

    th = theme_scp()
    fs = th.size("axis.text") * 0.8
    if plot_type == "bar":
        frames = [gsea_bar_frame(res, ids, metric=metric, character_width=character_width)
                  for _, res, ids in selected]
        if panel_size is None:
            # the descriptions sit inside the panel, each on the half away from
            # its bar, so a half-panel must hold the widest of them
            label_in = draw.text_width_in([d for f in frames for d in f["Description"]], fs)
            rows = max(len(f) for f in frames)
            panel_size = (max(4.0, 2 * label_in + 1.0), max(3.0, 0.45 * rows))
        pg = PanelGrid(len(selected), panel_size=panel_size, nrow=nrow, ncol=ncol, keys=keys,
                       theme=th, legend=LegendColumn(position="right"))
        pal = list(reversed(palcolor)) if isinstance(palcolor, Sequence) and \
            not isinstance(palcolor, str) else palcolor
        colors = discrete_palette(["Neg", "Pos"], palette=palette, palcolor=pal)
        legs = _PanelLegends(pg)
        for (k, _res, _ids), frame in zip(selected, frames, strict=True):
            ax = pg.axes[k]
            draw.draw_gsea_bar(ax, frame, colors=colors, fontsize=fs)
            group, db = _key_parts(k)
            _facet_strips(ax, group, db, th)
            if direction == "both":
                legs.legend(k, "Direction", [
                    Patch(facecolor=colors[d], edgecolor="black", label=d) for d in ("Pos", "Neg")])
            legs.legend(k, f"-log10({metric})", _alpha_handles(frame["alpha_raw"]))
        legs.finish()
        return pg.finish()

    # plot_type == "line"
    sources = _gsea_sources(results)
    rng = np.random.default_rng(seed)
    pg = PanelGrid(len(selected), panel_size=panel_size or (4.5, 4.5), nrow=nrow, ncol=ncol,
                   keys=keys, theme=th, legend=LegendColumn(position="right"))
    legs = _PanelLegends(pg)
    for k, res, ids in selected:
        rows = res.set_index(res["ID"].astype(str), drop=False).loc[ids].reset_index(drop=True)
        labels = gsea_network_descriptions(rows, metric, character_width)
        sets = []
        gene_list: np.ndarray | None = None
        for tid, lab in zip(ids, labels, strict=True):
            cur = _gsea_curve(k, tid, sources, curves)
            x = cur["x"].to_numpy(dtype=float)
            sets.append({"id": tid, "label": lab, "x": x,
                         "running_score": cur["running_score"].to_numpy(dtype=float),
                         "hits": x[cur["position"].to_numpy() == 1], "frame": cur})
            if gene_list is None and "metric" in cur:
                gene_list = cur["metric"].to_numpy(dtype=float)
        if gene_list is None:
            raise ValueError(f"no ranked metric for {k!r}: pass its 'geneList', or curves "
                             "with a 'metric' column (scp.pl.gsea_scores emits one)")
        lc = [line_color] if isinstance(line_color, str) else list(line_color)
        if len(lc) == len(sets):
            color_use = lc
        else:
            pal_lines = discrete_palette(list(dict.fromkeys(labels)), palette=palette,
                                         palcolor=palcolor)
            color_use = [pal_lines[lab] for lab in labels]
        single = None
        strip = None
        if len(sets) == 1:
            s0 = sets[0]
            i = int(np.argmax(np.abs(s0["running_score"])))
            row = rows.iloc[0]
            single = {"peak": (float(s0["x"][i]), float(s0["running_score"][i])),
                      "nes": float(row["NES"]), "title": str(row["Description"]),
                      "subtitle": labels[0].rsplit("\n", 1)[-1],
                      "genes": _core_gene_rows(s0["frame"], row, n_coregene=n_coregene,
                                               sample_coregene=sample_coregene,
                                               features_label=features_label, rng=rng)}
            strip = gsea_heat_strip(gene_list)
        cross = (gsea_zero_cross(gene_list)
                 if np.nanmax(gene_list) > 0 and np.nanmin(gene_list) < 0 else None)
        n_rank = len(gene_list)
        lo = 0.0 if (single is not None or np.nanmax(gene_list) > 0) else 1.0
        hi = float(n_rank + 1) if strip else float(n_rank)
        pad = 0.01 * (hi - lo)
        axes = _split_block(pg, k)
        draw.draw_gsea_line(axes, sets, colors=color_use, gene_list=gene_list,
                            line_width=line_width, single=single, heat_strip=strip,
                            zero_cross=cross, xlim=(lo - pad, hi + pad), x_limits=(lo, hi),
                            fontsize=fs)
        # R centres the key on the whole block; on the top panel's centre it
        # cannot reach into the zero-gap stack and push the sub-panels apart
        axes[0].text(1.02, 0.5, k, transform=axes[0].transAxes, rotation=-90, ha="left",
                     va="center", fontsize=th.size("strip.text"))
        pg.axes[k] = axes[0]
        if len(set(color_use)) > 1:
            legs.legend(k, "Term", [
                Line2D([], [], color=c, linewidth=line_width * draw.LWD, label=lab)
                for lab, c in dict(zip(labels, color_use, strict=True)).items()])
    legs.finish()
    return pg.finish()


def _gsea_selection(
    per: Mapping[str, pd.DataFrame],
    *,
    group_use,
    metric: str,
    metric_value: float,
    top_term: int | None,
    direction: str,
    id_use,
) -> list[tuple[str, pd.DataFrame, list[str]]]:
    """``for (nm in names(res))`` after ``group_use``: each key's result and
    its selected gene-set IDs; keys selecting nothing are dropped, as R's
    ``plist[[nm]] <- NULL; next``."""
    gu = None
    if group_use is not None:
        gu = {str(g) for g in ([group_use] if isinstance(group_use, str) else group_use)}
    out = []
    for k, r in per.items():
        res = r.copy()
        res["Groups"] = res["Groups"].astype(str)
        group = str(res["Groups"].iloc[0]) if len(res) else _key_parts(k)[0]
        if gu is not None and group not in gu and _key_parts(k)[0] not in gu:
            continue
        ids = gsea_select_ids(res, metric=metric, metric_value=metric_value, top_term=top_term,
                              direction=direction, id_use=id_use, group=group)
        if ids:
            out.append((k, res, ids))
    return out


def gsea_bar_frame(
    res: pd.DataFrame,
    ids: Sequence[str],
    *,
    metric: str,
    character_width: int = 50,
) -> pd.DataFrame:
    """The data of one ``GSEAPlot(plot_type = "bar")`` panel
    (``SCP-plot.R:14041-14051``): the selected sets in ``order(NES)``
    (stable), wrapped descriptions with ``_y`` = their level (0 at the
    bottom), ``Direction`` Pos/Neg, ``alpha_raw = -log10(metric)`` and
    ``alpha``, ggplot's ``scale_alpha`` (range 0.1-1) over the panel."""
    sub = res.set_index(res["ID"].astype(str), drop=False).loc[list(ids)]
    sub = sub.iloc[np.argsort(sub["NES"].to_numpy(dtype=float), kind="stable")]
    sub = sub.reset_index(drop=True)
    sub["Description"] = _wrap_desc(sub["Description"], character_width)
    levels = list(dict.fromkeys(sub["Description"]))
    pos = {lv: i for i, lv in enumerate(levels)}
    sub["_y"] = [pos[d] for d in sub["Description"]]
    sub["Direction"] = np.where(sub["NES"].to_numpy(dtype=float) > 0, "Pos", "Neg")
    raw = -np.log10(sub[metric].to_numpy(dtype=float))
    sub["alpha_raw"] = raw
    sub["alpha"] = _alpha_scale(raw, raw)
    return sub


def _alpha_scale(v, domain) -> np.ndarray:
    """``scale_alpha(range = c(0.1, 1))``; a zero-width domain maps to 0.55."""
    v = np.asarray(v, dtype=float)
    lo, hi = float(np.nanmin(domain)), float(np.nanmax(domain))
    if not hi > lo:
        return np.full(v.shape, 0.55)
    return np.clip(0.1 + 0.9 * (v - lo) / (hi - lo), 0.0, 1.0)


def _alpha_handles(raw) -> list[Patch]:
    raw = np.asarray(raw, dtype=float)
    br = draw._size_breaks(float(np.nanmin(raw)), float(np.nanmax(raw)))
    return [Patch(facecolor="#595959", edgecolor="none", alpha=float(a), label=draw._fmt(b))
            for b, a in zip(br, _alpha_scale(br, raw), strict=True)]


def _facet_strips(ax, group: str, db: str, th) -> None:
    """``facet_grid(Database ~ Groups)`` on one panel: the group on top, the
    database down the right."""
    ax.set_title(group, fontsize=th.size("strip.text"))
    ax.text(1.02, 0.5, db, transform=ax.transAxes, rotation=-90, ha="left", va="center",
            fontsize=th.size("strip.text"))


def _split_block(pg: PanelGrid, key: str):
    """Replace a PanelGrid cell with GSEAPlot's 1.5 : 0.5 : 1 stack."""
    old = pg.axes[key]
    spec = old.get_subplotspec()
    assert spec is not None
    old.remove()
    sub = spec.subgridspec(3, 1, height_ratios=[1.5, 0.5, 1.0], hspace=0.0)
    with mpl.rc_context(theme_scp_rc(pg.theme)):
        return tuple(pg.fig.add_subplot(sub[i]) for i in range(3))


def _gsea_curve(key: str, tid: str, sources: Mapping[str, Mapping], curves: Mapping) -> pd.DataFrame:
    """The running score of set ``tid`` in result ``key``, as
    :func:`gsea_scores` lays it out (``x``, ``running_score``, ``position``,
    ``gene``, ``metric``): computed from the key's ``geneList`` / ``geneSets``
    when given, else looked up in ``curves`` by key then by ID."""
    src = sources.get(key)
    if src is not None:
        gl = src["geneList"]
        gl = gl if isinstance(gl, pd.Series) else pd.Series(gl)
        sets = src["geneSets"]
        if tid not in sets:
            raise ValueError(f"gene set {tid!r} is not in the geneSets of {key!r}")
        return gsea_scores(gl, sets[tid], float(src.get("exponent", 1.0)))
    cur = None
    if isinstance(curves.get(key), Mapping):
        cur = curves[key].get(tid)
    if cur is None and isinstance(curves.get(tid), pd.DataFrame):
        cur = curves[tid]
    if cur is None:
        raise ValueError(
            f"no running-score curve for {tid!r} in {key!r}; pass the key's geneList and "
            "geneSets, or {'table': df, 'curves': {id: scp.pl.gsea_scores(...)}}")
    cur = pd.DataFrame(cur).copy()
    # gsea_scores emits the port's own column names; accept R's gsInfo ones too
    if "running_score" not in cur:
        cur["running_score"] = cur["runningScore"]
    if "metric" not in cur and "geneList" in cur:
        cur["metric"] = cur["geneList"]
    if "x" not in cur:
        cur["x"] = np.arange(1, len(cur) + 1)
    if "gene" not in cur:
        cur["gene"] = cur["x"].astype(str)
    return cur


def _core_gene_rows(cur: pd.DataFrame, row: pd.Series, *, n_coregene: int, sample_coregene: bool,
                    features_label, rng) -> pd.DataFrame | None:
    """The labelled hits of a single-set line plot (``SCP-plot.R:13860-13893``):
    ``features_label``, else the first ``n_coregene`` core-enrichment genes in
    ranked-list order (a random draw with ``sample_coregene``)."""
    if not ((n_coregene is not None and n_coregene > 1) or features_label):
        return None
    if features_label:
        labels = [str(g) for g in features_label]
    else:
        core = eg._split_genes(row.get("core_enrichment"))
        n = min(int(n_coregene), len(core))
        if sample_coregene:
            labels = [str(g) for g in rng.choice(core, n, replace=False)]
        else:
            cs = set(core)
            labels = [g for g in cur["gene"].astype(str) if g in cs][:n]
    hit = (cur["position"].to_numpy() == 1) & cur["gene"].astype(str).isin(labels).to_numpy()
    df = cur[hit]
    drop = [g for g in labels if g not in set(df["gene"].astype(str))]
    if drop:
        warnings.warn(f"Gene {','.join(drop)} is not in the geneset of the "
                      f"{row['Description']}", stacklevel=3)
    return df


def gsea_zero_cross(values) -> float:
    """The rank GSEAPlot marks as the zero cross (``SCP-plot.R:13932-13934``):
    ``median(x[y == y[which.min(abs(y))]])`` over the ranked metric ``y``,
    ``x`` 1-based."""
    y = np.asarray(values, dtype=float)
    min_y = y[int(np.argmin(np.abs(y)))]
    return float(np.median(np.flatnonzero(y == min_y) + 1))


def _cut_ramp(v: np.ndarray, ramp: Sequence[str]) -> list[str]:
    """``ramp[cut(v, seq(min, max, len = 100), include.lowest = TRUE)]``:
    99 right-closed bins, so the ramp's 100th colour is never used."""
    lo, hi = float(np.min(v)), float(np.max(v))
    if not hi > lo:  # R's cut() stops on non-unique breaks here
        return [ramp[0]] * len(v)
    breaks = lo + np.arange(100) * ((hi - lo) / 99)
    breaks[-1] = hi
    idx = np.clip(np.searchsorted(breaks, v, side="left"), 1, 99)
    return [ramp[i - 1] for i in idx]


def gsea_heat_strip(values) -> list[tuple[float, float, str]]:
    """The ranking-metric strip of a single-set line plot
    (``SCP-plot.R:13895-13930``) as ``(xmin, xmax, colour)`` rectangles.

    The metric is clipped to its 2% / 98% quantiles (R's type-7, numpy's
    default); positives are binned onto ``#F5DCDC -> #C40003``, negatives onto
    ``#1D008F -> #DDDCF5``, zeros stay white.  As in R, each colour becomes
    one rectangle from its first rank, as wide as its count.
    """
    y = np.asarray(values, dtype=float).copy()
    q_hi, q_lo = np.quantile(y, 0.98), np.quantile(y, 0.02)
    y[y > q_hi] = q_hi
    y[y < q_lo] = q_lo
    col = np.array(["white"] * len(y), dtype=object)
    pos, neg = np.flatnonzero(y > 0), np.flatnonzero(y < 0)
    if len(pos):
        col[pos] = _cut_ramp(y[pos], color_ramp(["#F5DCDC", "#C40003"], 100))
    if len(neg):
        col[neg] = _cut_ramp(y[neg], color_ramp(["#1D008F", "#DDDCF5"], 100))
    first: dict[str, int] = {}
    count: dict[str, int] = {}
    for i, c in enumerate(col, start=1):
        first.setdefault(c, i)
        count[c] = count.get(c, 0) + 1
    return [(float(first[c]), float(first[c] + count[c]), str(c)) for c in first]


def gsea_scores(gene_list: pd.Series, gene_set: Sequence[str], exponent: float = 1.0) -> pd.DataFrame:
    """The weighted KS running score (Subramanian et al.).

    Port of ``gseaScores`` (SCP, Hao Zhang, https://github.com/zhanghao-njmu/SCP,
    ``R/SCP-plot.R:14570``); verified against it to 1e-12.

    ``gene_list`` must be sorted **descending** by the ranking metric::

        hits  = gene_list.index.isin(gene_set)
        Phit  = np.where(hits, np.abs(gene_list) ** exponent, 0)
        Phit  = np.cumsum(Phit / Phit.sum())
        Pmiss = np.cumsum(np.where(hits, 0, 1 / (N - Nh)))
        running = Phit - Pmiss

    If you are already using ``gseapy``, take its per-term ``RES`` array
    instead of recomputing — this exists so the plot can be driven by a
    ranking the user computed themselves.
    """
    gl = gene_list.astype(float)
    n = len(gl)
    hits = np.isin(gl.index.to_numpy(), np.asarray(list(gene_set)))
    nh = int(hits.sum())
    if nh == 0 or n == nh:
        raise ValueError("gene set does not overlap the ranked list (or covers all of it).")
    phit = np.where(hits, np.abs(gl.to_numpy()) ** exponent, 0.0)
    phit = np.cumsum(phit / phit.sum())
    pmiss = np.cumsum(np.where(hits, 0.0, 1.0 / (n - nh)))
    return pd.DataFrame(
        {
            "x": np.arange(1, n + 1),
            "running_score": phit - pmiss,
            "position": hits.astype(int),
            "gene": gl.index.to_numpy(),
            "metric": gl.to_numpy(),
        }
    )


def adjust_layout(
    layout: np.ndarray,
    widths: np.ndarray,
    edges: Sequence[tuple[int, int]],
    *,
    height: float = 2.0,
    scale: float = 60.0,
    iterations: int = 100,
    seed: int = 11,
) -> np.ndarray:
    """Separate overlapping *label boxes* after a force-directed layout. Milestone 14.

    Transcription of SCP's ``adjustlayout`` (Hao Zhang,
    https://github.com/zhanghao-njmu/SCP, ``R/SCP-plot.R:13441``).

    A spring layout treats nodes as points, but gene nodes in the enrichment
    network are rendered as text whose width scales with the symbol length.
    ``widths`` are the label widths (``nchar(name)``, 8 for terms) and boxes
    are ``widths / 2`` either side of the node and ``height`` tall.

    0. each axis is rescaled to span ``scale`` (no centring);
    1. vertices are visited in decreasing degree (ties by index); each
       not-yet-adjusted neighbour closer than ``r + nr`` is pushed radially
       outward by the deficit, and pushed at most once -- hubs stay put;
    2. ``iterations`` rounds: from a fresh distance matrix take each vertex's
       nearest neighbour(s); visiting vertices in random order, a neighbour
       whose box overlaps (``|dx| < r + nr`` and ``|dy| < height``) is moved
       by the overlap in x *or* y, chosen by a coin flip.

    **R bug fixed.** R computes the step-2 displacement as
    ``dx <- r + nr - (nx - x)`` and moves the neighbour to ``nx - dx``, which
    puts it at ``2 * nx - x - (r + nr)``: a neighbour to the *right* of ``x`` is
    reflected but still overlaps, so only left/below neighbours are ever
    separated.  Step 1 uses ``x - nx``; here step 2 does too, which places the
    neighbour flush against the box (``x - (r + nr)``) as intended.  A
    zero-distance pair in step 1 (a 0/0 in R that turns the layout into NaN) is
    pushed along +x instead.

    The coin flips and visiting order use numpy's generator seeded by
    ``seed``; R's stream differs, so results match R in kind, not in value.
    """
    pos = np.array(layout, dtype=float).copy()
    n = len(pos)
    if n == 0:
        return pos
    w = np.asarray(widths, dtype=float) / 2.0
    for k in range(2):
        span = float(np.ptp(pos[:, k]))
        if span > 0:
            pos[:, k] = pos[:, k] / span * scale
    nbrs: list[set[int]] = [set() for _ in range(n)]
    degree = np.zeros(n, dtype=int)
    for a, b in edges:
        a, b = int(a), int(b)
        degree[a] += 1
        degree[b] += 1
        nbrs[a].add(b)
        nbrs[b].add(a)
    rng = np.random.default_rng(seed)

    adjusted: set[int] = set()
    for vi in np.argsort(-degree, kind="stable"):
        v = int(vi)
        adjusted.add(v)
        todo = sorted(nbrs[v] - adjusted)
        x, y = pos[v]
        r = w[v]
        for nb in todo:
            nx, ny = pos[nb]
            ndist = math.hypot(nx - x, ny - y)
            expect = r + w[nb]
            if ndist < expect:
                if ndist == 0:
                    pos[nb, 0] = nx + expect
                else:
                    k = (expect - ndist) / ndist
                    pos[nb, 0] = nx - (x - nx) * k
                    pos[nb, 1] = ny - (y - ny) * k
                adjusted.add(nb)

    wl = w.tolist()
    for _ in range(iterations):
        ddx = pos[:, None, 0] - pos[None, :, 0]
        ddy = pos[:, None, 1] - pos[None, :, 1]
        d = np.sqrt(ddx * ddx + ddy * ddy)  # symmetric, bit for bit
        # each vertex's nearest neighbour(s) at a positive distance, all at
        # once: an O(n^2) array pass instead of n Python-level scans
        closest = np.where(d > 0, d, np.inf).min(axis=1)
        vs, nbs = np.nonzero((d == closest[:, None]) & np.isfinite(closest)[:, None])
        bounds = np.searchsorted(vs, np.arange(n + 1)).tolist()
        nbl = nbs.tolist()
        # the sweep is sequential (each move is seen by the next vertex), so
        # it runs on Python floats: the same IEEE arithmetic, without numpy's
        # per-scalar overhead
        px, py = pos[:, 0].tolist(), pos[:, 1].tolist()
        for vi in rng.permutation(n).tolist():
            x, y = px[vi], py[vi]
            r = wl[vi]
            for nb in nbl[bounds[vi]:bounds[vi + 1]]:
                nx, ny = px[nb], py[nb]
                rr = r + wl[nb]
                if abs(nx - x) < rr and abs(ny - y) < height:
                    dx = rr - (x - nx)
                    dy = height - (y - ny)
                    if rng.integers(2) == 1:
                        dx = 0.0
                    else:
                        dy = 0.0
                    px[nb] = nx - dx
                    py[nb] = ny - dy
        pos = np.column_stack([px, py])
    return pos
