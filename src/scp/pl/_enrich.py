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
import textwrap
from collections.abc import Mapping, Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ..layout import LegendColumn, PanelGrid
from ..palettes import continuous_palette, discrete_palette
from ..theme import theme_scp
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
    a, b = str(x).split("/")
    return float(a) / float(b)


def _wrap_desc(descriptions, width: int) -> list[str]:
    return [eg.r_strwrap(eg.capitalize(d), width) for d in descriptions]


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
    cb = pg.fig.colorbar(sm, ax=list(pg.axes.values()), fraction=0.03, pad=0.02, shrink=0.6)
    cb.set_label(fill_label, fontsize=th.size("legend.title"))
    cb.outline.set_edgecolor("black")
    return pg.finish()


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
    pg = PanelGrid(len(panels), panel_size=psize, nrow=nrow, ncol=ncol, keys=keys, theme=th,
                   legend=LegendColumn(position="right"))

    if plot_type in ("network", "enrichmap", "wordcloud"):
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
    for key, _dg, g in panels:
        ax = pg.axes[key]
        g = g.assign(_score=-np.log10(g[metric].astype(float).clip(lower=1e-300)))
        g = g.sort_values("_score", kind="mergesort")
        # terms are long; wrap them or the axis eats the panel
        labels = [textwrap.fill(str(d), character_width) for d in g["Description"]]
        y = np.arange(len(g))
        counts = (g["Count"].to_numpy(dtype=float) if "Count" in g
                  else np.full(len(g), 1.0))
        norm = Normalize(float(g["_score"].min()), float(g["_score"].max()))
        cvec = cmap(norm(g["_score"].to_numpy()))

        if plot_type == "bar":
            ax.barh(y, g["_score"], color=cvec, edgecolor="black", linewidth=0.4)
        elif plot_type == "dot":
            ax.scatter(g["_score"], y, s=20 + 80 * counts / max(counts.max(), 1),
                       color=cvec, edgecolor="black", linewidth=0.4, zorder=3)
        elif plot_type == "lollipop":
            ax.hlines(y, 0, g["_score"], color="#B0B0B0", linewidth=1, zorder=1)
            ax.scatter(g["_score"], y, s=20 + 80 * counts / max(counts.max(), 1),
                       color=cvec, edgecolor="black", linewidth=0.4, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel(f"-log10({metric})")
        ax.set_title(str(key), fontsize=th.size("plot.title"))
        ax.set_ylim(-0.7, len(g) - 0.3)

    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    cb = pg.fig.colorbar(sm, ax=list(pg.axes.values()), fraction=0.02, pad=0.02)
    cb.set_label(f"-log10({metric}), scaled per panel", fontsize=th.size("legend.title"))
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
    multi = len(panels) > 1
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
        # colourbars first: they take space from the axes, and the words must be
        # packed into the panel as it will finally be
        for (key, _dg, _df), tab in zip(panels, tables, strict=True):
            ax = pg.axes[key]
            ax.set_title(str(key), fontsize=th.size("strip.text"))
            if tab.empty:
                continue
            sc = tab["score"].to_numpy(dtype=float)
            sm = ScalarMappable(cmap=LinearSegmentedColormap.from_list("wc", ramp, N=256),
                                norm=Normalize(float(sc.min()), float(sc.max())))
            cb = pg.fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.03, shrink=0.7)
            cb.set_label("Score", fontsize=th.size("legend.title"))
            cb.outline.set_edgecolor("black")
            cnt = tab["count"].to_numpy(dtype=float)
            br = sorted(set(int(math.ceil(v)) for v in np.linspace(cnt.min(), cnt.max(), 3)))
            sizes = draw._area_size(br, float(cnt.min()), float(cnt.max()), word_size)
            pg.legend.add(f"Count ({key})" if multi else "Count", [
                Line2D([], [], marker="$G$", linestyle="none", color="black",
                       markersize=float(s) * draw.PT, label=str(b))
                for b, s in zip(br, sizes, strict=True)])
        pg.fig.set_layout_engine("constrained")
        pg.fig.canvas.draw()
        pg.fig.set_layout_engine("none")
        for (key, _dg, _df), tab in zip(panels, tables, strict=True):
            draw.draw_wordcloud(pg.axes[key], tab, ramp, word_size=word_size)
        return

    with eg._seeded_igraph(seed):
        for key, _dg, df in panels:
            ax = pg.axes[key]
            ax.set_title(str(key), fontsize=th.size("strip.text"))
            suffix = f" ({key})" if multi else ""
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
                draw.draw_network(ax, nt, lay, labelsize=network_labelsize, legend=pg.legend,
                                  legend_title=f"Term{suffix}")
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
                                    panel_mm=_panel_mm(pg, key), legend=pg.legend,
                                    legend_suffix=suffix)


# --------------------------------------------------------------------------- #
# GSEA
# --------------------------------------------------------------------------- #
def _gsea_tables(results) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    """Normalise the accepted inputs to (flat table, per-key results, curves).

    Accepted: a flat GSEA DataFrame; ``{"table": df, "curves": {...}}``; or
    SCP's own shape, ``{"enrichment": df, "results": {"<Group>-<Database>":
    df | {"result": df, ...}}}``.  Without per-key results they are rebuilt
    from the flat table, ordered as RunGSEA orders them (groups fastest within
    each database).
    """
    curves: dict = {}
    per: dict[str, pd.DataFrame] = {}
    if isinstance(results, pd.DataFrame):
        table = results.copy()
    elif "enrichment" in results:
        table = pd.DataFrame(results["enrichment"]).copy()
        for k, v in dict(results.get("results", {})).items():
            per[str(k)] = pd.DataFrame(v["result"] if isinstance(v, Mapping) else v).copy()
        curves = results.get("curves", {})
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

    ``plot_type="line"`` stacks three sub-panels at relative heights
    ``1.5 : 0.5 : 1``:

    1. the running enrichment score, over red/blue half-plane shading, with a
       peak marker (up-triangle when NES > 0, down when NES < 0) and optional
       core-gene labels;
    2. the hit ticks — one band per gene set, or, for a single set, a
       ranking-metric heat strip below the ticks;
    3. the ranked metric as stems to zero, with a dashed line at the
       zero-crossing rank.

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

    Input is a flat GSEA table (``scp.io.GSEA_COLUMNS``), ``{"table": df,
    "curves": {id: gsea_scores(...)}}`` for ``line``, or SCP's
    ``{"enrichment": df, "results": {key: df}}``.  ``gseapy.prerank`` output
    maps on with renaming (``Term`` -> ``ID``/``Description``, ``FDR q-val``
    -> ``p.adjust``, ``Lead_genes`` -> ``core_enrichment``).
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

    if plot_type in ("network", "enrichmap", "wordcloud"):
        metric, value = _metric(pvalue_cutoff, padjust_cutoff)
        keys = list(per)
        if group_use is not None:
            gu = {str(g) for g in ([group_use] if isinstance(group_use, str) else group_use)}
            keys = [k for k in keys if k.rsplit("-", 1)[0] in gu]
        panels = []
        descriptions: dict[str, list[str]] = {}
        counts: dict[str, list[int]] = {}
        for k in keys:
            res = per[k].copy()
            res["Groups"] = res["Groups"].astype(str) if "Groups" in res else k.rsplit("-", 1)[0]
            if "Database" not in res:
                res["Database"] = k.rsplit("-", 1)[-1]
            ids = gsea_select_ids(res, metric=metric, metric_value=value,
                                  top_term=None if plot_type == "wordcloud" else top_term,
                                  direction=direction, id_use=id_use,
                                  group=str(res["Groups"].iloc[0]) if len(res) else None)
            if not ids:
                continue
            df = res.set_index(res["ID"].astype(str)).loc[ids].reset_index(drop=True)
            panels.append((k, (None, None), df))
            if plot_type == "network":
                descriptions[k] = gsea_network_descriptions(df, metric, character_width)
            elif plot_type == "enrichmap":
                descriptions[k] = _wrap_desc(df["Description"], character_width)
                counts[k] = [len(eg._split_genes(g)) for g in df["core_enrichment"]]
        if not panels:
            raise ValueError("no gene sets survive the cutoffs")
        th = theme_scp(aspect_ratio=1)
        psize = panel_size or _PANEL_DEFAULT[plot_type]
        pg = PanelGrid(len(panels), panel_size=psize, nrow=nrow, ncol=ncol,
                       keys=[k for k, _, _ in panels], theme=th,
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

    if table.empty:
        raise ValueError("no GSEA results to plot")
    if padjust_cutoff is not None and "p.adjust" in table:
        table = table[table["p.adjust"] <= padjust_cutoff]
    if direction == "pos":
        table = table[table["NES"] > 0]
    elif direction == "neg":
        table = table[table["NES"] < 0]
    if table.empty:
        raise ValueError("no gene sets survive the cutoffs")
    table = table.reindex(table["NES"].abs().sort_values(ascending=False).index).head(top_term)
    psize = panel_size or (3.2, 3.0)

    if plot_type == "bar":
        ax_fig = plt.figure(figsize=(psize[0] * 1.4, psize[1]))
        ax = ax_fig.add_subplot()
        y = np.arange(len(table))
        ax.barh(y, table["NES"], color=np.where(table["NES"] > 0, "#B2182B", "#2166AC"),
                edgecolor="black", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels([textwrap.fill(str(d), 50) for d in table["Description"]],
                           fontsize=6.5)
        ax.set_xlabel("NES")
        ax_fig.set_layout_engine("constrained")
        return ax_fig

    # plot_type == "line": the canonical three-panel GSEA figure
    n = len(table)
    fig = plt.figure(figsize=(psize[0], psize[1] * n))
    gs = fig.add_gridspec(n * 3, 1,
                          height_ratios=[6, 1.2, 2.5] * n, hspace=0.08)
    colors = ([line_color] * n if isinstance(line_color, str) else list(line_color))
    for i, (_, row) in enumerate(table.iterrows()):
        cur = curves.get(str(row["ID"]))
        if cur is None:
            raise ValueError(
                f"no running-score curve for {row['ID']!r}; pass "
                "{'table': df, 'curves': {id: scp.pl.gsea_scores(...)}}"
            )
        x = cur["x"].to_numpy() if "x" in cur else np.arange(len(cur))
        # gsea_scores emits the port's own column names; accept R's too so a
        # frame lifted straight out of clusterProfiler still plots.
        rs_col = "running_score" if "running_score" in cur else "runningScore"
        rs = cur[rs_col].to_numpy()
        hits = np.flatnonzero(cur["position"].to_numpy() == 1) if "position" in cur else []

        top = fig.add_subplot(gs[i * 3])
        top.plot(x, rs, color=colors[i % len(colors)], linewidth=line_width)
        top.axhline(0, color="grey", linewidth=0.6, linestyle="--")
        top.set_xticks([])
        top.set_ylabel("Running\nenrichment", fontsize=7)
        top.set_title(textwrap.fill(str(row["Description"]), 46), fontsize=8)
        top.text(0.98, 0.95, f"NES = {row['NES']:.2f}\np.adj = {row['p.adjust']:.2g}",
                 transform=top.transAxes, ha="right", va="top", fontsize=6.5)

        mid = fig.add_subplot(gs[i * 3 + 1])
        mid.vlines(x[hits], 0, 1, color="black", linewidth=0.4)
        mid.set_xticks([])
        mid.set_yticks([])
        mid.set_xlim(x.min(), x.max())

        bot = fig.add_subplot(gs[i * 3 + 2])
        metric_col = "metric" if "metric" in cur else "geneList"
        if metric_col in cur:
            gl = cur[metric_col].to_numpy()
            bot.fill_between(x, 0, gl, color="#B0B0B0")
            bot.axhline(0, color="grey", linewidth=0.6)
        bot.set_xlabel("Rank in ordered gene list", fontsize=7)
        bot.set_ylabel("Ranked\nmetric", fontsize=7)
        bot.set_xlim(x.min(), x.max())
    fig.set_layout_engine("constrained")
    return fig


def gsea_scores(gene_list: pd.Series, gene_set: Sequence[str], exponent: float = 1.0) -> pd.DataFrame:
    """The weighted KS running score (Subramanian et al.).

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

    for _ in range(iterations):
        d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
        nearest = []
        for v in range(n):
            col = d[:, v]
            pos_d = col[col > 0]
            nearest.append(np.flatnonzero(col == pos_d.min()) if len(pos_d) else np.array([], int))
        for vi in rng.permutation(n):
            v = int(vi)
            x, y = pos[v]
            r = w[v]
            for nb in dict.fromkeys(int(i) for i in nearest[v]):
                nx, ny = pos[nb]
                rr = r + w[nb]
                if abs(nx - x) < rr and abs(ny - y) < height:
                    dx = rr - (x - nx)
                    dy = height - (y - ny)
                    if rng.integers(2) == 1:
                        dx = 0.0
                    else:
                        dy = 0.0
                    pos[nb, 0] = nx - dx
                    pos[nb, 1] = ny - dy
    return pos
