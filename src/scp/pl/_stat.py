"""Statistical / distribution plots.

Two engines do the work; the ``*_plot`` functions are thin AnnData wrappers.
Keep that separation — it is the reason the R code is testable at all.

* :func:`expression_stat_plot` — continuous values per cell (violin/box/bar/dot/col)
* :func:`cell_stat_plot` / :func:`stat_plot` — composition of categorical columns

Spec: ``docs/porting_briefs/stat_plots.md``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
from scipy.stats import gaussian_kde, linregress

from ..fetch import as_ordered_categorical, fetch_data
from ..layout import LegendColumn, PanelGrid
from ..palettes import continuous_palette, discrete_palette
from ..theme import halo, theme_scp

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
    if isinstance(stat_by, str):
        stat_cols = [stat_by]
    else:
        stat_cols = list(stat_by)

    # R takes unique(c(stat.by, group.by, split.by, bg.by)); the same column may
    # legitimately serve two roles, and selecting it twice makes a frame with
    # duplicate labels that cannot be reindexed.
    cols = list(dict.fromkeys(c for c in stat_cols + [group_by, split_by, bg_by] if c))
    data = adata.obs.loc[:, cols].copy()
    if cells is not None:
        data = data.loc[data.index.intersection(pd.Index(cells))]
    return stat_plot(
        data, stat_by=stat_cols, group_by=group_by, split_by=split_by, bg_by=bg_by,
        plot_type=plot_type, stat_type=stat_type, position=position, palette=palette,
        palcolor=palcolor, alpha=alpha, na_color=na_color, na_stat=na_stat,
        stat_level=stat_level, label=label, label_size=label_size, flip=flip,
        keep_empty=keep_empty, individual=individual, panel_size=panel_size,
        aspect_ratio=aspect_ratio, legend_position=legend_position, nrow=nrow,
        ncol=ncol, byrow=byrow, force=force, seed=seed,
    )


def stat_plot(
    data: pd.DataFrame,
    *,
    stat_by: Sequence[str],
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
    stat_level=None,
    label: bool = False,
    label_size: float = 3.5,
    flip: bool = False,
    keep_empty: bool = False,
    individual: bool = False,
    panel_size: tuple[float, float] = (2.6, 2.2),
    aspect_ratio: float | None = None,
    legend_position: str = "right",
    nrow: int | None = None,
    ncol: int | None = None,
    byrow: bool = True,
    force: bool = False,
    seed: int = 11,
) -> Figure:
    """DataFrame engine behind :func:`cell_stat_plot`.

    R original: ``StatPlot``.  The aggregation core is a **complete** cross-tab
    of ``stat_by x group_by`` within each split level, keeping empty cells, and
    (for ``stat_type="percent"``) dividing by the group total.  Get that right
    before worrying about any geometry: every ``plot_type`` is a different view
    of the same table.

    Multi-column ``stat_by`` is only meaningful for the set-like types
    (``venn``/``upset``), which binarise each column with ``stat_level``.
    """
    stat_cols = list(stat_by)
    if len(stat_cols) >= 2 and plot_type not in ("sankey", "chord", "venn", "upset"):
        raise ValueError(
            "several `stat_by` columns are only meaningful for plot_type in "
            "{'sankey', 'chord', 'venn', 'upset'}"
        )
    if plot_type in ("sankey", "chord"):
        raise NotImplementedError(
            f"plot_type={plot_type!r} is specified in docs/porting_briefs/stat_plots.md "
            "§2.3 but not yet implemented; see docs/07_milestones.md."
        )
    if individual:
        raise NotImplementedError(
            "`individual` is specified in docs/porting_briefs/stat_plots.md §2.2 "
            "but not yet implemented."
        )
    if plot_type in ("rose", "ring", "pie"):
        aspect_ratio = 1.0

    if plot_type in ("venn", "upset"):
        return _set_plot(data, stat_cols, plot_type, stat_level=stat_level, palette=palette,
                         palcolor=palcolor, alpha=alpha, panel_size=panel_size,
                         legend_position=legend_position, label_size=label_size)

    stat_col = stat_cols[0]
    stat_cat = as_ordered_categorical(data[stat_col], show_na=na_stat)
    stat_levels = [str(x) for x in stat_cat.cat.categories]
    data = data.copy()
    data[stat_col] = np.asarray(stat_cat).astype(str)

    if group_by is None:
        group_by = "__all__"
        data[group_by] = ""
        group_levels = [""]
    else:
        gcat = as_ordered_categorical(data[group_by])
        group_levels = [str(x) for x in gcat.cat.categories]
        data[group_by] = np.asarray(gcat).astype(str)

    if split_by is None:
        split_levels: list[str | None] = [None]
    else:
        scat = as_ordered_categorical(data[split_by])
        split_levels = [str(x) for x in scat.cat.categories]
        data[split_by] = np.asarray(scat).astype(str)

    colors = discrete_palette(stat_levels, palette=palette, palcolor=palcolor)
    if "NA" in stat_levels:
        colors["NA"] = na_color

    th = theme_scp(aspect_ratio=aspect_ratio, legend_position=legend_position)
    keys = [str(s) for s in split_levels]
    legend = LegendColumn(position=legend_position)
    pg = PanelGrid(len(keys), panel_size=panel_size, nrow=nrow, ncol=ncol, byrow=byrow,
                   keys=keys, theme=th, legend=legend,
                   polar=plot_type in ("rose", "ring", "pie"))

    for sp in split_levels:
        ax = pg.axes[str(sp)]
        block = data if sp is None else data[data[split_by] == sp]

        # The aggregation core: a COMPLETE cross-tab, empty cells retained, then
        # normalised within each group. Every plot_type below is a view of this.
        tab = pd.crosstab(block[stat_col], block[group_by], dropna=False)
        tab = tab.reindex(index=stat_levels, columns=group_levels, fill_value=0)
        if stat_type == "percent":
            totals = tab.sum(axis=0).replace(0, np.nan)
            values = tab / totals
        else:
            values = tab.astype(float)

        levels_x = list(group_levels)
        if not keep_empty:
            nonempty = [g for g in levels_x if tab[g].sum() > 0]
            levels_x = nonempty or levels_x
        if flip and plot_type not in ("pie", "rose"):
            levels_x = levels_x[::-1]

        _draw_stat_panel(ax, values[levels_x], stat_levels, levels_x, colors,
                         plot_type=plot_type, stat_type=stat_type, position=position,
                         alpha=alpha, label=label, label_size=label_size, flip=flip,
                         bg_by=bg_by, block=block, group_by=group_by, theme=th)

        if sp is not None:
            ax.set_title(str(sp), fontsize=th.size("plot.title"))

    handles = [Patch(facecolor=colors[lv], edgecolor="black", label=lv) for lv in stat_levels]
    if legend_position != "none":
        legend.add(stat_col, handles)
    return pg.finish()


def _draw_stat_panel(ax, values, stat_levels, levels_x, colors, *, plot_type, stat_type,
                     position, alpha, label, label_size, flip, bg_by, block, group_by, theme):
    """Draw one panel of the aggregated table."""
    n_x = len(levels_x)
    x = np.arange(n_x)

    if plot_type in ("bar", "area", "trend"):
        if position == "stack":
            bottom = np.zeros(n_x)
            # ggplot's position_stack fills from the top down, so the FIRST
            # level ends up at the top of the bar. Accumulating in declared
            # order instead silently flips the whole figure upside down.
            for lv in reversed(stat_levels):
                v = np.nan_to_num(values.loc[lv].to_numpy())
                if plot_type == "bar":
                    ax.bar(x, v, bottom=bottom, width=0.8, color=colors[lv], alpha=alpha,
                           edgecolor="black", linewidth=0.5)
                elif plot_type == "area":
                    ax.fill_between(x, bottom, bottom + v, color=colors[lv], alpha=alpha,
                                    edgecolor="black", linewidth=0.5)
                else:  # trend: a ribbon joining adjacent bars, plus a narrow bar
                    ax.fill_between(x, bottom, bottom + v, color=colors[lv],
                                    alpha=alpha / 2, edgecolor="#7F7F7F", linewidth=0.5)
                    ax.bar(x, v, bottom=bottom, width=0.6, color=colors[lv], alpha=alpha,
                           edgecolor="black", linewidth=0.5)
                if label:
                    for xi, (b, vv) in enumerate(zip(bottom, v, strict=True)):
                        if vv <= 0:
                            continue
                        txt = f"{vv * 100:.1f}%" if stat_type == "percent" else f"{vv:g}"
                        t = ax.text(xi, b + vv / 2, txt, ha="center", va="center",
                                    fontsize=label_size * 2, color="black")
                        halo(t, foreground="white", radius=0.1)
                bottom = bottom + v
        else:
            w = 0.8 / max(len(stat_levels), 1)
            for i, lv in enumerate(stat_levels):
                v = np.nan_to_num(values.loc[lv].to_numpy())
                off = (i - (len(stat_levels) - 1) / 2) * w
                ax.bar(x + off, v, width=w * 0.95, color=colors[lv], alpha=alpha,
                       edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(levels_x, rotation=45, ha="right")
        if stat_type == "percent" and position == "stack":
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    elif plot_type == "dot":
        # y is the CATEGORY, not the value: a dot matrix sized by the statistic
        vmax = np.nanmax(values.to_numpy()) or 1.0
        for yi, lv in enumerate(stat_levels):
            v = np.nan_to_num(values.loc[lv].to_numpy())
            ax.scatter(x, np.full(n_x, yi), s=12 ** 2 * v / vmax, facecolor=colors[lv],
                       edgecolor="black", linewidth=0.5, alpha=alpha)
        ax.set_xticks(x)
        ax.set_xticklabels(levels_x, rotation=45, ha="right")
        ax.set_yticks(range(len(stat_levels)))
        ax.set_yticklabels(stat_levels)
        ax.set_ylim(-0.6, len(stat_levels) - 0.4)

    elif plot_type == "rose":
        # Nightingale rose: group levels around the circle, radius = value
        width = 2 * np.pi / max(n_x, 1)
        theta = x * width
        bottom = np.zeros(n_x)
        for lv in stat_levels:
            v = np.nan_to_num(values.loc[lv].to_numpy())
            ax.bar(theta, v, width=width * 0.95, bottom=bottom, color=colors[lv],
                   alpha=alpha, edgecolor="black", linewidth=0.5)
            bottom = bottom + v
        ax.set_xticks(theta)
        ax.set_xticklabels(levels_x, fontsize=7)
        ax.set_yticklabels([])

    elif plot_type in ("ring", "pie"):
        # theta carries the value; ring reserves an inner hole, pie does not.
        # R prepends a dummy "   " level to the group factor to make the hole.
        totals = values.sum(axis=1)
        frac = (totals / totals.sum()).to_numpy() if totals.sum() else np.zeros(len(totals))
        start = np.pi / 2 if flip else 0.0
        inner = 0.45 if plot_type == "ring" else 0.0
        edges = start + 2 * np.pi * np.concatenate([[0], np.cumsum(frac)])
        for i, lv in enumerate(stat_levels):
            if frac[i] <= 0:
                continue
            ax.bar((edges[i] + edges[i + 1]) / 2, 1 - inner, width=edges[i + 1] - edges[i],
                   bottom=inner, color=colors[lv], alpha=alpha, edgecolor="black",
                   linewidth=0.5)
            if label:
                t = ax.text((edges[i] + edges[i + 1]) / 2, inner + (1 - inner) / 2,
                            f"{frac[i] * 100:.1f}%" if stat_type == "percent"
                            else f"{totals.iloc[i]:g}",
                            ha="center", va="center", fontsize=label_size * 2)
                halo(t, foreground="white", radius=0.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, 1)

    # Background stripes exist only when position != "stack" (R sets bg_layer
    # to NULL under stack, because the bars already tile the panel).
    if bg_by is not None and position != "stack" and plot_type in ("bar", "area", "trend", "dot"):
        bgcat = as_ordered_categorical(block[bg_by])
        bg_levels = [str(c) for c in bgcat.cat.categories]
        mapping = block.groupby(group_by, observed=True)[bg_by].first()
        spread = block.groupby(group_by, observed=True)[bg_by].nunique()
        if (spread > 1).any():
            raise ValueError("`group_by` must be a part of `bg_by`")
        bg_colors = discrete_palette(bg_levels, palette="Paired")
        for i, lv in enumerate(levels_x):
            b = mapping.get(lv)
            if b is not None:
                ax.axvspan(i - 0.5, i + 0.5, color=bg_colors[str(b)], alpha=0.2,
                           zorder=0, linewidth=0)


def _set_plot(data, stat_cols, plot_type, *, stat_level, palette, palcolor, alpha,
              panel_size, legend_position, label_size):
    """``venn`` and ``upset`` — the set-membership views.

    ``stat_level`` binarises each ``stat_by`` column into set membership; it has
    no effect on any other plot type.  Defaults to each column's first level,
    with a message, exactly as R does.
    """
    if stat_level is None:
        stat_level = {c: [str(as_ordered_categorical(data[c]).cat.categories[0])]
                      for c in stat_cols}
        warnings.warn(
            "`stat_level` not given; using the first level of each column: "
            + ", ".join(f"{k}={v[0]}" for k, v in stat_level.items()),
            stacklevel=3,
        )
    elif isinstance(stat_level, str):
        stat_level = {c: [stat_level] for c in stat_cols}
    elif not isinstance(stat_level, dict):
        stat_level = {c: list(stat_level) for c in stat_cols}
    else:
        stat_level = {k: ([v] if isinstance(v, str) else list(v)) for k, v in stat_level.items()}

    sets = {c: set(data.index[data[c].astype(str).isin([str(x) for x in stat_level[c]])])
            for c in stat_cols}
    colors = discrete_palette(list(sets), palette=palette, palcolor=palcolor)

    if plot_type == "venn":
        if len(sets) > 3:
            raise NotImplementedError(
                "venn beyond 3 sets needs a custom renderer; "
                "docs/porting_briefs/stat_plots.md §2.3."
            )
        try:
            from matplotlib_venn import venn2, venn3
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("venn plots need `pip install matplotlib-venn`") from e
        fig, ax = plt.subplots(figsize=panel_size)
        names = list(sets)
        draw = venn2 if len(sets) == 2 else venn3
        v = draw([sets[n] for n in names], set_labels=names, ax=ax)
        for i, n in enumerate(names):
            patch = v.get_patch_by_id("A" if i == 0 else ("B" if i == 1 else "C"))
            if patch is not None:
                patch.set_color(colors[n])
                patch.set_alpha(alpha * 0.6)
        return fig

    # upset: intersection sizes, largest first, with a membership matrix beneath
    members = pd.DataFrame({c: data.index.isin(list(sets[c])) for c in stat_cols},
                           index=data.index)
    members = members[members.any(axis=1)]
    combos = members.apply(lambda r: tuple(c for c in stat_cols if r[c]), axis=1)
    counts = combos.value_counts().head(20)

    fig = plt.figure(figsize=(max(panel_size[0], 0.5 * len(counts) + 1), panel_size[1] * 1.7))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], hspace=0.06)
    bar_ax, mat_ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    xs = np.arange(len(counts))
    bar_ax.bar(xs, counts.to_numpy(), width=0.7, color="#3C3C3C")
    for xi, c in zip(xs, counts.to_numpy(), strict=True):
        bar_ax.text(xi, c, str(c), ha="center", va="bottom", fontsize=label_size * 2)
    bar_ax.set_xticks([])
    bar_ax.set_ylabel("Intersection size")
    bar_ax.set_xlim(-0.6, len(counts) - 0.4)
    for yi, c in enumerate(stat_cols):
        for xi, combo in enumerate(counts.index):
            on = c in combo
            mat_ax.scatter(xi, yi, s=60, color=colors[c] if on else "#DDDDDD",
                           zorder=3 if on else 2)
        hits = [xi for xi, combo in enumerate(counts.index) if c in combo]
        if hits:
            mat_ax.plot([min(hits), max(hits)], [yi, yi], color="#DDDDDD", linewidth=1, zorder=1)
    mat_ax.set_yticks(range(len(stat_cols)))
    mat_ax.set_yticklabels(stat_cols)
    mat_ax.set_xticks([])
    mat_ax.set_xlim(-0.6, len(counts) - 0.4)
    mat_ax.set_ylim(-0.6, len(stat_cols) - 0.4)
    for side in ("top", "right", "bottom", "left"):
        mat_ax.spines[side].set_visible(False)
    return fig


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
    feats = list(features)
    if len(feats) < 2:
        raise ValueError("`features` needs at least two entries")
    if len(feats) > 10 and not force:
        raise ValueError(
            f"{len(feats)} features would make {len(feats) ** 2} panels. "
            "Pass force=True to proceed anyway."
        )

    frame = fetch_data(adata, feats, layer=layer)
    feats = [f for f in feats if f in frame.columns]
    dat = frame[feats].apply(pd.to_numeric, errors="coerce")
    if cells is not None:
        dat = dat.loc[dat.index.intersection(pd.Index(cells))]
    dat = dat[np.isfinite(dat.to_numpy()).all(axis=1)]

    if group_by is None:
        gvals = pd.Series(["all"] * len(dat), index=dat.index)
        glevels = ["all"]
    else:
        gcat = as_ordered_categorical(adata.obs[group_by]).loc[dat.index]
        gvals = pd.Series(np.asarray(gcat).astype(str), index=dat.index)
        glevels = [str(c) for c in gcat.cat.categories]
    colors = discrete_palette(glevels, palette=palette, palcolor=palcolor)

    # Spearman is Pearson on row-ranked values, which is what proxyC::simil does
    mat = dat.to_numpy(dtype=float)
    ranked = np.apply_along_axis(lambda c: pd.Series(c).rank().to_numpy(), 0, mat)
    corr = np.corrcoef((ranked if cor_method == "spearman" else mat), rowvar=False)

    n = len(feats)
    cor_ramp = continuous_palette(palette=cor_palette, n=200)
    cor_cmap = LinearSegmentedColormap.from_list("cor", cor_ramp, N=256)
    cor_norm = Normalize(*cor_range)

    fig, axes = plt.subplots(n, n, figsize=(panel_size[0] * n, panel_size[1] * n),
                             squeeze=False)
    size = pt_size if pt_size is not None else max(2000 / max(len(dat), 1), 1.0)
    for i, f1 in enumerate(feats):          # column
        for j, f2 in enumerate(feats):      # row
            ax = axes[j][i]
            if i == j:
                # diagonal: the distribution of that feature per group
                for k, lv in enumerate(glevels):
                    v = dat.loc[gvals == lv, f1].to_numpy()
                    if v.size < 2:
                        continue
                    parts = ax.violinplot([v], positions=[k], widths=0.85,
                                          showextrema=False)
                    for body in parts["bodies"]:
                        body.set(facecolor=colors[lv], alpha=0.9, edgecolor="black",
                                 linewidth=0.4)
                ax.set_xticks(range(len(glevels)))
                ax.set_xticklabels([])
            elif i < j:
                # upper: scatter coloured by group, with an lm fit and its stats
                ax.scatter(dat[f1], dat[f2], s=size,
                           c=[colors[g] for g in gvals], linewidths=0, alpha=0.9,
                           rasterized=len(dat) > 20000)
                if add_smooth and len(dat) > 2:
                    slope, intercept, r, pv, _ = linregress(dat[f1], dat[f2])
                    xs = np.linspace(dat[f1].min(), dat[f1].max(), 50)
                    ax.plot(xs, intercept + slope * xs, color="red", linewidth=1,
                            alpha=0.7)
                    lines = []
                    if add_equation:
                        sign = "-" if slope < 0 else "+"
                        lines.append(f"y = {intercept:.2g} {sign} {abs(slope):.2g}x")
                    if add_r2:
                        lines.append(f"$r^2$ = {r ** 2:.2g}")
                    if add_pvalue:
                        lines.append(f"p = {pv:.2g}")
                    if lines:
                        ax.text(0.04, 0.96, "\n".join(lines), transform=ax.transAxes,
                                ha="left", va="top", fontsize=6)
            else:
                # lower: a solid block whose colour encodes the correlation
                r = corr[i, j]
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       facecolor=cor_cmap(cor_norm(r)), edgecolor="none"))
                ax.text(0.5, 0.5, f"{f1}\n{f2}\nCor: {r:.3f}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=6.5, fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])

            # axis decoration only on the outer edges, as R does
            if i != 0 or i == j:
                ax.set_yticklabels([])
            if j != n - 1:
                ax.set_xticklabels([])
            if i == 0:
                ax.set_ylabel(f2, fontsize=7.5)
            if j == n - 1:
                ax.set_xlabel(f1, fontsize=7.5)
            ax.tick_params(labelsize=6)

    handles = [Patch(facecolor=colors[lv], edgecolor="black", label=lv) for lv in glevels]
    if group_by is not None:
        fig.legend(handles=handles, title=group_by, loc="upper left",
                   bbox_to_anchor=(1.0, 0.98), frameon=False, fontsize=7,
                   title_fontsize=7.5)
    cb = fig.colorbar(ScalarMappable(norm=cor_norm, cmap=cor_cmap),
                      ax=list(axes.ravel()), fraction=0.02, pad=0.02)
    cb.set_label(f"{cor_method} correlation", fontsize=7.5)
    fig.set_layout_engine("constrained")
    return fig


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
    y_min: float | None = None,
    y_max: float | None = None,
    y_nbreaks: int = 4,
    keep_empty: bool = False,
    panel_size: tuple[float, float] = (2.6, 2.2),
    aspect_ratio: float | None = None,
    legend_position: str = "right",
    nrow: int | None = None,
    ncol: int | None = None,
    byrow: bool = True,
    force: bool = False,
) -> Figure:
    """Ridgeline plot of a feature per group. Milestone 6.

    R uses ``ggridges::geom_density_ridges`` (Gaussian KDE, ``bw.nrd0``
    bandwidth).  ``x_order="rank"`` replaces values by their rank, which is
    what makes pseudotime ridges spread evenly.  ``decreasing`` orders groups
    by their median of the feature; ``None`` keeps factor order but reversed,
    so level 1 lands on top.
    """
    feats = [features] if isinstance(features, str) else list(features)
    if len(feats) > 50 and not force:
        raise ValueError(
            f"{len(feats)} features will not be readable. Pass force=True to proceed."
        )
    frame = fetch_data(adata, feats, layer=layer)
    feats = [f for f in feats if f in frame.columns]
    if not feats:
        raise ValueError("none of `features` could be resolved")

    if group_by is None:
        gcat = pd.Series(["all"] * adata.n_obs, index=adata.obs_names).astype("category")
        glevels = ["all"]
    else:
        cat = as_ordered_categorical(adata.obs[group_by])
        gcat = pd.Series(np.asarray(cat).astype(str), index=adata.obs_names)
        glevels = [str(c) for c in cat.cat.categories]
    colors = discrete_palette(glevels, palette=palette, palcolor=palcolor)

    split_levels: list[str | None] = [None]
    if split_by is not None:
        scat = as_ordered_categorical(adata.obs[split_by])
        split_levels = [str(c) for c in scat.cat.categories]

    th = theme_scp(aspect_ratio=aspect_ratio)
    keys = [f"{f}:{s}" for f in feats for s in split_levels]
    legend = LegendColumn(position=legend_position)
    pg = PanelGrid(len(keys), panel_size=panel_size, nrow=nrow, ncol=ncol, byrow=byrow,
                   keys=keys, theme=th, legend=legend)

    for f in feats:
        vals = pd.to_numeric(frame[f], errors="coerce")
        for sp in split_levels:
            ax = pg.axes[f"{f}:{sp}"]
            m = np.isfinite(vals.to_numpy())
            if sp is not None:
                m = m & (np.asarray(as_ordered_categorical(adata.obs[split_by])).astype(str) == sp)
            v = vals[m]
            g = gcat[m]
            if x_order == "rank":
                v = v.rank()  # rank mode is what spreads pseudotime ridges evenly

            # Default order is the factor order REVERSED, so level 1 ends up on
            # top of the ridge stack; `decreasing` sorts by median instead.
            if decreasing is None:
                order = list(reversed(glevels))
            else:
                med = v.groupby(g).median()
                order = list(med.sort_values(ascending=not decreasing).index)
            if flip:
                order = list(glevels)

            lo = y_min if y_min is not None else float(v.min())
            hi = y_max if y_max is not None else float(v.max())
            grid = np.linspace(lo, hi, 256)
            for k, lv in enumerate(order):
                sub = v[g == lv].to_numpy()
                if sub.size < 2 or np.allclose(sub, sub[0]):
                    continue
                kde = gaussian_kde(sub)  # scipy's default bandwidth, like bw.nrd0
                dens = kde(grid)
                dens = dens / dens.max() * 0.95   # ridges normalise each row
                ax.fill_between(grid, k, k + dens, facecolor=colors[lv], alpha=0.9,
                                edgecolor="black", linewidth=0.5, zorder=len(order) - k)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(order, fontsize=7)
            ax.set_ylim(-0.2, len(order) + 0.6)
            ax.set_xlim(lo, hi)
            ax.set_xlabel(f)
            ax.set_ylabel(group_by or "")
            if sp is not None:
                ax.set_title(str(sp), fontsize=th.size("plot.title"))

    if group_by is not None and legend_position != "none":
        legend.add(group_by, [Patch(facecolor=colors[lv], edgecolor="black", label=lv)
                              for lv in glevels])
    return pg.finish()


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
    df = de.copy()
    if pct1_col in df.columns and pct2_col in df.columns:
        df["diff_pct"] = df[pct1_col] - df[pct2_col]
    elif x_metric == "diff_pct":
        raise ValueError(f"`diff_pct` needs {pct1_col!r} and {pct2_col!r} in the table")
    if x_metric == "diff_pct" and not np.isfinite(df["diff_pct"]).any():
        # scanpy omits the detection fractions unless asked, and an all-NaN x
        # would otherwise render as an empty panel with no indication why.
        raise ValueError(
            f"{pct1_col!r}/{pct2_col!r} are all missing, so `diff_pct` cannot be computed. "
            "scanpy only fills them when called as "
            "sc.tl.rank_genes_groups(..., pts=True); alternatively pass "
            "x_metric='avg_log2FC'."
        )

    # DE_threshold is an expression evaluated against the frame, as in R
    df["DE"] = False
    df.loc[df.eval(de_threshold), "DE"] = True

    # x clipping: the 1%/99% quantiles, symmetrised when they straddle zero, and
    # outliers clamped onto the bound rather than dropped.
    xv = df[lfc_col].to_numpy(dtype=float)
    finite = xv[np.isfinite(xv)]
    hi = float(np.quantile(finite, 0.99)) or float(np.max(finite))
    lo = float(np.quantile(finite, 0.01)) or float(np.min(finite))
    if hi <= 0:
        hi = float(np.max(finite))
    if lo >= 0:
        lo = float(np.min(finite))
    if hi > 0 and lo < 0:
        v = min(abs(hi), abs(lo))
        hi, lo = v, -v
    df["border"] = (df[lfc_col] < lo) | (df[lfc_col] > hi)
    clipped_lfc = df[lfc_col].clip(lo, hi)

    y = -np.log10(df[padj_col].to_numpy(dtype=float))
    y = np.where(np.isfinite(y), y, np.nanmax(y[np.isfinite(y)]) if np.isfinite(y).any() else 0.0)

    # The signed-y trick: this is not a standard volcano. Down-regulated genes
    # hang BELOW zero and the axis is relabelled with abs(), so the reader sees
    # magnitude in both directions.
    if x_metric == "diff_pct":
        x = df["diff_pct"].to_numpy(dtype=float)
        y = np.where(df[lfc_col].to_numpy() < 0, -y, y)
        sort_key = np.abs(df[lfc_col].to_numpy(dtype=float))
        color_vals = df[lfc_col].to_numpy(dtype=float)
        color_name = lfc_col
    else:
        x = clipped_lfc.to_numpy(dtype=float)
        y = np.where(df["diff_pct"].to_numpy() < 0, -y, y)
        sort_key = np.abs(df["diff_pct"].to_numpy(dtype=float))
        color_vals = df["diff_pct"].to_numpy(dtype=float)
        color_name = "diff_pct"
    df["x"], df["y"] = x, y
    df["distance"] = x ** 2 + y ** 2
    df = df.iloc[np.argsort(sort_key, kind="stable")]  # extremes drawn last, on top

    groups = ([str(g) for g in as_ordered_categorical(df[group_col]).cat.categories]
              if group_col in df.columns else ["all"])
    th = theme_scp(aspect_ratio=None)
    legend = LegendColumn(position="right")
    pg = PanelGrid(len(groups), panel_size=panel_size, nrow=nrow, ncol=ncol,
                   keys=groups, theme=th, legend=legend)

    # zero-anchored diverging ramp: the colour is the OTHER metric
    ramp = continuous_palette(palette=palette, palcolor=palcolor, n=100)
    cmap = LinearSegmentedColormap.from_list("volcano", ramp, N=256)
    span = np.nanmax(np.abs(color_vals)) or 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-span, vmax=span)
    rng = np.random.default_rng(11)

    for g in groups:
        ax = pg.axes[g]
        sub = df if group_col not in df.columns else df[df[group_col].astype(str) == g]
        gx, gy = sub["x"].to_numpy(dtype=float), sub["y"].to_numpy(dtype=float)
        border = sub["border"].to_numpy()
        is_de = sub["DE"].to_numpy()
        # clamped points fan out so the pile-up on the clip boundary is visible
        jx = gx + np.where(border, rng.uniform(-0.2, 0.2, len(gx)), 0.0)
        jy = gy + np.where(border, rng.uniform(-0.2, 0.2, len(gy)), 0.0)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.scatter(jx[~is_de], jy[~is_de], s=pt_size * 6, c="#BFBFBF",
                   linewidths=0, zorder=1)
        ax.scatter(jx[is_de], jy[is_de], s=(pt_size + 0.5) * 14, c="black",
                   linewidths=0, zorder=2)  # halo
        ax.scatter(jx[is_de], jy[is_de], s=pt_size * 8,
                   c=sub[color_name].to_numpy()[is_de], cmap=cmap, norm=norm,
                   linewidths=0, zorder=3)

        # top nlabel by distance, computed SEPARATELY for each half
        if features_label is not None:
            picks = sub[sub[gene_col].astype(str).isin([str(f) for f in features_label])]
        else:
            upper = sub[sub["y"] >= 0].nlargest(nlabel, "distance")
            lower = sub[sub["y"] < 0].nlargest(nlabel, "distance")
            picks = pd.concat([upper, lower])
        for _, row in picks.iterrows():
            t = ax.annotate(str(row[gene_col]), (row["x"], row["y"]),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color="black",
                            arrowprops=None)
            halo(t, foreground="white", radius=0.1)

        ax.set_xlabel(x_metric)
        ax.set_ylabel("-log10(p-adjust)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{abs(v):g}"))
        if group_col in df.columns:
            ax.set_title(g, fontsize=th.size("plot.title"))

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = pg.fig.colorbar(sm, ax=list(pg.axes.values()), fraction=0.02, pad=0.02)
    cb.set_label(color_name, fontsize=th.size("legend.title"))
    return pg.finish()
