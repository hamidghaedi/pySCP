"""Statistical / distribution plots.

Two engines do the work; the ``*_plot`` functions are thin AnnData wrappers.
Keep that separation — it is the reason the R code is testable at all.

* :func:`expression_stat_plot` — continuous values per cell (violin/box/bar/dot/col)
* :func:`cell_stat_plot` / :func:`stat_plot` — composition of categorical columns

Spec: ``docs/porting_briefs/stat_plots.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from ..fetch import as_ordered_categorical, fetch_data
from ..layout import LegendColumn, PanelGrid
from ..palettes import discrete_palette
from ..theme import theme_scp

__all__ = [
    "feature_stat_plot",
    "expression_stat_plot",
    "cell_stat_plot",
    "stat_plot",
    "feature_cor_plot",
    "cell_density_plot",
    "volcano_plot",
]

ExprPlotType = Literal["violin", "box", "bar", "dot", "col"]
StatPlotType = Literal[
    "bar", "rose", "ring", "pie", "trend", "area", "dot", "sankey", "chord", "venn", "upset"
]


def feature_stat_plot(
    adata,
    stat_by: str | Sequence[str],
    *,
    group_by: str | Sequence[str] | None = None,
    split_by: str | None = None,
    bg_by: str | None = None,
    plot_by: Literal["group", "feature"] = "group",
    fill_by: Literal["group", "feature", "expression"] = "group",
    plot_type: ExprPlotType = "violin",
    layer: str | None = None,
    use_raw: bool = False,
    cells: Sequence[str] | None = None,
    # color
    palette: str = "Paired",
    palcolor: Sequence[str] | dict[str, str] | None = None,
    alpha: float = 1.0,
    bg_palette: str = "Paired",
    bg_palcolor: Sequence[str] | None = None,
    bg_alpha: float = 0.2,
    # overlays
    add_box: bool = False,
    add_point: bool = False,
    add_trend: bool = False,
    add_stat: Literal["none", "mean", "median"] = "none",
    add_line: float | None = None,
    jitter_width: float = 0.4,
    jitter_height: float = 0.1,
    # axes
    stack: bool = False,
    flip: bool = False,
    sort: bool | Literal["increasing", "decreasing"] = False,
    same_y_lims: bool = False,
    y_min: float | str | None = None,
    y_max: float | str | None = None,
    y_trans: str = "identity",
    y_nbreaks: int = 5,
    keep_empty: bool = False,
    individual: bool = False,
    # statistics
    comparisons: bool | Sequence[tuple[str, str]] | None = None,
    ref_group: str | None = None,
    pairwise_method: str = "wilcox",
    multiplegroup_comparisons: bool = False,
    multiple_method: str = "kruskal",
    sig_label: Literal["p.signif", "p.format"] = "p.signif",
    sig_labelsize: float = 3.5,
    calculate_coexp: bool = False,
    # layout
    panel_size: tuple[float, float] = (2.6, 2.2),
    aspect_ratio: float | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    xlab: str | None = None,
    ylab: str = "Expression level",
    legend_position: str = "right",
    nrow: int | None = None,
    ncol: int | None = None,
    byrow: bool = True,
    force: bool = False,
    seed: int = 11,
) -> Figure:
    """Per-cell distributions of features or numeric ``obs`` columns.

    R original: ``FeatureStatPlot`` -> ``ExpressionStatPlot``.  Milestone 4.

    Two mechanics carry the visual identity and must not be approximated:

    * ``bg_by`` stripes — rectangles spanning the full panel height behind each
      group, with the outermost stripes bleeding past the panel edge so there
      is no white gutter.  In matplotlib: ``ax.axvspan(..., zorder=0)``.
      ``bg_by`` must be a *coarser* partition than ``group_by``; validate it.
    * ``stack=True`` — the stacked-violin figure.  This is a shared-x subplot
      grid with per-row y-limits, exactly two y ticks per row, one shared y
      label and one figure legend.  R achieves it by gtable surgery; do it
      natively with a GridSpec rather than compositing finished panels.
    """
    for unsupported, name in (
        (stack, "stack"), (individual, "individual"), (calculate_coexp, "calculate_coexp"),
        (comparisons, "comparisons"), (ref_group, "ref_group"),
        (multiplegroup_comparisons, "multiplegroup_comparisons"), (add_trend, "add_trend"),
    ):
        if unsupported:
            raise NotImplementedError(
                f"`{name}` is specified in docs/porting_briefs/stat_plots.md §1 but not yet "
                "implemented; see docs/07_milestones.md."
            )
    if plot_by == "feature":
        raise NotImplementedError(
            "`plot_by='feature'` is the reshape described in "
            "docs/porting_briefs/stat_plots.md §1.2 and is not yet implemented."
        )

    feats = [stat_by] if isinstance(stat_by, str) else list(stat_by)
    groups = [group_by] if isinstance(group_by, str) else (list(group_by) if group_by else [None])

    values = fetch_data(adata, feats, layer=layer, use_raw=use_raw)
    feats = [f for f in feats if f in values.columns]
    if not feats:
        raise ValueError("none of `stat_by` could be resolved")

    dat = pd.DataFrame(index=pd.Index(adata.obs_names))
    for f in feats:
        dat[f] = pd.to_numeric(values[f], errors="coerce").to_numpy()
    declared_levels: dict[str, list] = {}
    for col in [g for g in groups if g] + [c for c in (split_by, bg_by) if c]:
        cat = as_ordered_categorical(adata.obs[col])
        declared_levels[col] = list(cat.cat.categories)
        dat[col] = cat.to_numpy()
    if cells is not None:
        dat = dat.loc[dat.index.intersection(pd.Index(cells))]

    def _limit(spec, pool):
        """``y_min``/``y_max`` accept a number or a ``"qNN"`` quantile string."""
        if isinstance(spec, str) and spec.startswith("q"):
            return float(np.quantile(pool, float(spec[1:]) / 100.0))
        return float(spec)

    shared: tuple[float, float] | None = None
    if same_y_lims:
        pool = dat[feats].to_numpy().ravel()
        pool = pool[np.isfinite(pool)]
        shared = (_limit(y_min, pool) if y_min is not None else float(pool.min()),
                  _limit(y_max, pool) if y_max is not None else float(pool.max()))

    th = theme_scp(aspect_ratio=aspect_ratio, legend_position=legend_position)
    keys = [f"{g}:{f}" for g in groups for f in feats]
    legend = LegendColumn(position=legend_position)
    pg = PanelGrid(len(keys), panel_size=panel_size, nrow=nrow, ncol=ncol, byrow=byrow,
                   keys=keys, theme=th, legend=legend)
    rng = np.random.default_rng(seed)

    seen_legend: set[str] = set()
    for g in groups:
        for f in feats:
            ax = pg.axes[f"{g}:{f}"]
            cols = [c for c in (g, split_by, bg_by) if c]
            sub = dat[cols + [f]].copy()
            sub = sub[np.isfinite(sub[f].to_numpy())]

            if g is None:
                sub["__group__"] = ""
                levels: list = [""]
            else:
                sub["__group__"] = sub[g]
                levels = [lv for lv in declared_levels[g]
                          if keep_empty or (sub["__group__"] == lv).any()]
            if sort and g is not None:
                med = sub.groupby("__group__", observed=True)[f].median()
                levels = [lv for lv in med.sort_values(
                    ascending=(sort == "increasing")).index if lv in levels]

            splits = [None] if split_by is None else declared_levels[split_by]
            n_split = len(splits)

            if fill_by == "feature":
                fill_levels: list = list(feats)
                fill_key = "Features"
            elif split_by is not None:
                fill_levels, fill_key = list(splits), split_by
            else:
                fill_levels, fill_key = list(levels), (g or "")
            colors = discrete_palette([str(x) for x in fill_levels],
                                      palette=palette, palcolor=palcolor)

            # bind the loop variables as defaults: this closure outlives the
            # iteration that created it.
            def _color(level, split, _f=f, _colors=colors):
                if fill_by == "feature":
                    return _colors[str(_f)]
                return _colors[str(split)] if split_by is not None else _colors[str(level)]

            # Stripes span the full panel height behind each group and are drawn
            # first, so every other layer sits on top of them.  They are NOT
            # opt-in: with no bg_by, R still stripes by group_by, alternating
            # transparent/grey85 rather than using bg_palette (SCP-plot.R:3991).
            if g is not None:
                if bg_by is not None:
                    # Each group must fall entirely inside one bg level.
                    spread = sub.groupby("__group__", observed=True)[bg_by].nunique()
                    if (spread > 1).any():
                        raise ValueError("`group_by` must be a part of `bg_by`")
                    mapping = sub.groupby("__group__", observed=True)[bg_by].first()
                    bg_colors = discrete_palette([str(b) for b in declared_levels[bg_by]],
                                                 palette=bg_palette, palcolor=bg_palcolor)
                    stripe_alpha = bg_alpha
                else:
                    mapping = pd.Series({lv: lv for lv in levels})
                    # grey85 == #D9D9D9; one geom_rect serves both branches, so
                    # bg_alpha applies here too and the banding stays subtle.
                    alt = [("transparent", "#D9D9D9")[i % 2] for i in range(len(levels))]
                    bg_colors = {str(lv): alt[i] for i, lv in enumerate(levels)}
                    stripe_alpha = bg_alpha
                for i, lv in enumerate(levels):
                    b = mapping.get(lv)
                    if b is None:
                        continue
                    col = bg_colors[str(b)]
                    if col == "transparent":
                        continue
                    ax.axvspan(i - 0.5, i + 0.5, color=col, alpha=stripe_alpha,
                               zorder=0, linewidth=0)

            width = 0.8 / max(n_split, 1)
            col_cursor = 0
            for i, lv in enumerate(levels):
                for si, sp in enumerate(splits):
                    off = 0.0 if n_split == 1 else (si - (n_split - 1) / 2) * width
                    m = ((sub["__group__"] == lv) if g is not None
                         else pd.Series(True, index=sub.index))
                    if split_by is not None:
                        m = m & (sub[split_by] == sp)
                    v = sub.loc[m, f].to_numpy()
                    if v.size == 0:
                        continue
                    c = _color(lv, sp)
                    pos = i + off

                    if plot_type == "violin":
                        # scale="width": every violin gets the same width, so a
                        # rare group stays as legible as a common one.
                        parts = ax.violinplot([v], positions=[pos], widths=width * 0.9,
                                              showextrema=False, showmedians=False)
                        for body in parts["bodies"]:
                            body.set(facecolor=c, alpha=alpha, edgecolor="black", linewidth=0.5)
                    elif plot_type == "box":
                        bp = ax.boxplot([v], positions=[pos], widths=width * 0.8,
                                        patch_artist=True, showfliers=False,
                                        medianprops=dict(color="black"))
                        for box in bp["boxes"]:
                            box.set(facecolor=c, alpha=alpha, edgecolor="black")
                        ax.scatter([pos], [np.median(v)], s=15, facecolor="white",
                                   edgecolor="black", zorder=4)
                    elif plot_type == "bar":
                        ax.bar(pos, v.mean(), width=width * 0.8, color=c, alpha=alpha,
                               edgecolor="black", linewidth=0.6)
                        ax.errorbar(pos, v.mean(), yerr=v.std(ddof=1) if v.size > 1 else 0.0,
                                    color="black", capsize=3, linewidth=0.8)
                    elif plot_type == "dot":
                        # 14 equal-width bins across the panel range; each bin
                        # collapses to its midpoint, sized by how many land in it.
                        allv = sub[f].to_numpy()
                        edges = np.linspace(allv.min(), allv.max(), 15)
                        mids = (edges[:-1] + edges[1:]) / 2
                        idx = np.clip(np.digitize(v, edges[1:-1]), 0, 13)
                        u, counts = np.unique(idx, return_counts=True)
                        ax.scatter([pos] * len(u), mids[u], s=6 + 36 * counts / counts.max(),
                                   facecolor=c, edgecolor="black", linewidth=0.4, alpha=alpha)
                    elif plot_type == "col":
                        ax.bar(np.arange(col_cursor, col_cursor + v.size), v, width=1.0,
                               color=c, alpha=alpha, linewidth=0)
                        col_cursor += v.size
                        continue

                    if add_point:
                        jx = pos + rng.uniform(-jitter_width, jitter_width, v.size) * width
                        ax.scatter(jx, v, s=2, color="#4D4D4D", linewidths=0, zorder=3)
                    if add_box and plot_type != "box":
                        ax.boxplot([v], positions=[pos], widths=width * 0.1,
                                   patch_artist=True, showfliers=False,
                                   boxprops=dict(facecolor="white", edgecolor="black"),
                                   medianprops=dict(color="black"), zorder=4)
                    if add_stat != "none":
                        y = v.mean() if add_stat == "mean" else float(np.median(v))
                        ax.scatter([pos], [y], marker="v", s=25, color="black", zorder=5)

            if plot_type == "col":
                acc = 0
                for lv in levels[:-1]:
                    acc += int((sub["__group__"] == lv).sum())
                    ax.axvline(acc, linestyle="--", color="black", linewidth=0.6)
                ax.set_xticks([])  # one column per cell: per-cell ticks are noise
            else:
                ax.set_xticks(range(len(levels)))
                ax.set_xticklabels([str(lv) for lv in levels], rotation=45, ha="right")
                ax.set_xlim(-0.5, len(levels) - 0.5)

            if add_line is not None:
                ax.axhline(add_line, color="red", linewidth=1.0)
            if plot_type == "bar":
                ax.axhline(0, linestyle="--", color="black", linewidth=0.6)

            lims = shared
            if lims is None:
                pool = sub[f].to_numpy()
                lims = (_limit(y_min, pool) if y_min is not None else float(pool.min()),
                        _limit(y_max, pool) if y_max is not None else float(pool.max()))
            if plot_type != "col" and lims[1] > lims[0]:
                ax.set_ylim(*lims)
            if y_trans == "log2":
                ax.set_yscale("log", base=2)
            ax.yaxis.set_major_locator(MaxNLocator(y_nbreaks))

            ax.set_xlabel(xlab if xlab is not None else (g or ""))
            ax.set_ylabel(ylab)
            ax.set_title(f"{subtitle}\n{title or f}" if subtitle else (title or f),
                         fontsize=th.size("plot.title"))

            if fill_key not in seen_legend and legend_position != "none":
                seen_legend.add(fill_key)
                legend.add(fill_key, [
                    Patch(facecolor=colors[str(x)], edgecolor="black", label=str(x))
                    for x in fill_levels
                ])

    return pg.finish()


def expression_stat_plot(data: pd.DataFrame, *args, **kwargs) -> Figure:
    """DataFrame engine behind :func:`feature_stat_plot`. Milestone 4."""
    raise NotImplementedError("Milestone 4. Spec: docs/porting_briefs/stat_plots.md §1.")


def cell_stat_plot(
    adata,
    stat_by: str | Sequence[str],
    *,
    group_by: str | None = None,
    split_by: str | None = None,
    bg_by: str | None = None,
    plot_type: StatPlotType = "bar",
    stat_type: Literal["percent", "count"] = "percent",
    position: Literal["stack", "dodge"] = "stack",
    palette: str = "Paired",
    palcolor: Sequence[str] | dict[str, str] | None = None,
    alpha: float = 1.0,
    na_color: str = "#BEBEBE",
    na_stat: bool = True,
    stat_level: str | Sequence[str] | dict[str, Sequence[str]] | None = None,
    label: bool = False,
    label_size: float = 3.5,
    flip: bool = False,
    keep_empty: bool = False,
    individual: bool = False,
    cells: Sequence[str] | None = None,
    panel_size: tuple[float, float] = (2.6, 2.2),
    aspect_ratio: float | None = None,
    legend_position: str = "right",
    nrow: int | None = None,
    ncol: int | None = None,
    byrow: bool = True,
    force: bool = False,
    seed: int = 11,
) -> Figure:
    """Composition of one categorical ``obs`` column across groups.

    R original: ``CellStatPlot`` -> ``StatPlot``.  Milestone 5.

    The aggregation core is a complete cross-tab, not a groupby-count::

        tab = pd.crosstab(obs[stat_by], obs[group_by], dropna=not na_stat)
        long = tab.melt(ignore_index=False)
        long["value"] = long.Freq / long.groupby(group_by).Freq.transform("sum")

    Empty cells are *retained* through the aggregation and dropped only at the
    end, which is what makes ``keep_empty`` meaningful.

    ``stat_by`` with two or more columns switches to the set-relationship
    types (``sankey``, ``chord``, ``venn``, ``upset``) and forces
    ``stat_type="count"``.
    """
    raise NotImplementedError("Milestone 5. Spec: docs/porting_briefs/stat_plots.md §2.")


def stat_plot(data: pd.DataFrame, *args, **kwargs) -> Figure:
    """DataFrame engine behind :func:`cell_stat_plot`. Milestone 5."""
    raise NotImplementedError("Milestone 5. Spec: docs/porting_briefs/stat_plots.md §2.")


def feature_cor_plot(
    adata,
    features: Sequence[str],
    *,
    group_by: str | None = None,
    split_by: str | None = None,
    layer: str | None = None,
    cells: Sequence[str] | None = None,
    cor_method: Literal["pearson", "spearman"] = "pearson",
    add_equation: bool = False,
    add_r2: bool = True,
    add_pvalue: bool = True,
    add_smooth: bool = True,
    palette: str = "Paired",
    palcolor: Sequence[str] | None = None,
    cor_palette: str = "RdBu",
    cor_range: tuple[float, float] = (-1.0, 1.0),
    pt_size: float | None = None,
    raster: bool | None = None,
    panel_size: tuple[float, float] = (1.4, 1.4),
    force: bool = False,
    seed: int = 11,
) -> Figure:
    """Pairwise scatter-plot matrix over features. Milestone 6.

    Lower triangle = solid correlation blocks, diagonal = per-group violins,
    upper triangle = scatter with an optional ``lm`` fit and r2/p annotation.
    Spearman is Pearson on row-ranked values, matching ``proxyC::simil``.
    """
    raise NotImplementedError("Milestone 6. Spec: docs/porting_briefs/stat_plots.md §3.")


def cell_density_plot(
    adata,
    features: str | Sequence[str],
    *,
    group_by: str | None = None,
    split_by: str | None = None,
    layer: str | None = None,
    cells: Sequence[str] | None = None,
    x_order: Literal["value", "rank"] = "value",
    decreasing: bool | None = None,
    flip: bool = False,
    reverse: bool = False,
    palette: str = "Paired",
    palcolor: Sequence[str] | None = None,
    same_y_lims: bool = False,
    panel_size: tuple[float, float] = (2.6, 2.2),
    force: bool = False,
) -> Figure:
    """Ridgeline plot of a feature per group. Milestone 6.

    R uses ``ggridges::geom_density_ridges`` (Gaussian KDE, ``bw.nrd0``
    bandwidth).  ``x_order="rank"`` replaces values by their rank, which is
    what makes pseudotime ridges spread evenly.  ``decreasing`` orders groups
    by their median of the feature; ``None`` keeps factor order but reversed,
    so level 1 lands on top.
    """
    raise NotImplementedError("Milestone 6. Spec: docs/porting_briefs/stat_plots.md §4.")


def volcano_plot(
    de: pd.DataFrame,
    *,
    group_col: str = "group1",
    gene_col: str = "gene",
    lfc_col: str = "avg_log2FC",
    padj_col: str = "p_val_adj",
    pct1_col: str = "pct.1",
    pct2_col: str = "pct.2",
    de_threshold: str = "avg_log2FC > 0 and p_val_adj < 0.05",
    x_metric: Literal["diff_pct", "avg_log2FC"] = "diff_pct",
    palette: str = "RdBu",
    palcolor: Sequence[str] | None = None,
    nlabel: int = 5,
    features_label: Sequence[str] | None = None,
    pt_size: float = 1.0,
    panel_size: tuple[float, float] = (2.6, 2.6),
    nrow: int | None = None,
    ncol: int | None = None,
) -> Figure:
    """Volcano plot from a differential-expression table.

    Takes a DataFrame, not an AnnData — use
    :func:`scp.io.de_from_rank_genes_groups` to build it from
    ``adata.uns['rank_genes_groups']``.  Milestone 6.

    This is **not** a standard volcano: y is signed.  ``y = -log10(padj)``, and
    genes whose *other* metric is negative get ``y`` negated, so
    down-regulated genes hang below zero and the axis is relabelled with
    ``abs``.  Points are colored by whichever metric is not on x, through a
    zero-anchored diverging ramp.
    """
    raise NotImplementedError("Milestone 6. Spec: docs/porting_briefs/stat_plots.md §5.")
