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
from typing import Literal, cast

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, PercentFormatter
from matplotlib.transforms import Bbox
from scipy.stats import gaussian_kde, linregress

from ..fetch import as_ordered_categorical, fetch_data
from ..fetch import calculate_coexp as calculate_coexp_fn
from ..layout import LegendColumn, PanelGrid
from ..palettes import continuous_palette, discrete_palette
from ..theme import apply_theme, halo, theme_scp
from ._stat_feature import ExprPlotType, _Ax, expression_stat_plot

__all__ = [
    "feature_stat_plot",
    "expression_stat_plot",
    "cell_stat_plot",
    "stat_plot",
    "feature_cor_plot",
    "cell_density_plot",
    "volcano_plot",
]

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

    R original: ``FeatureStatPlot`` (``SCP-plot.R:3524``), the Seurat wrapper
    over ``ExpressionStatPlot`` (3723) — Hao Zhang,
    https://github.com/zhanghao-njmu/SCP.  Milestone 4.

    This is the AnnData half: it resolves ``stat_by`` through
    :func:`scp.fetch.fetch_data` (genes, then obs, then embeddings; ``layer`` /
    ``use_raw`` pick the matrix), takes the grouping columns from ``obs`` with
    their declared level order, and hands one frame to
    :func:`expression_stat_plot`, which does everything else — every parameter
    not described here is documented there, including the significance layer
    and the deliberate divergences from R.

    ``group_by=None`` puts all cells in one group with ``"All groups"`` as the
    x title, as R does.

    ``calculate_coexp=True`` with two or more *gene* features adds a ``CoExp``
    panel: their per-cell geometric mean, from
    :func:`scp.fetch.calculate_coexp` (taken on the linear scale for
    log-normalised data).  Non-gene features are left out of the product with
    R's warning, and still get their own panels.

    The two mechanics that carry the visual identity:

    * ``bg_by`` stripes — rectangles spanning the full panel height behind each
      group, the outermost bleeding past the panel edge.  ``bg_by`` must be a
      *coarser* partition than ``group_by``.  With no ``bg_by``, R still stripes
      by ``group_by`` in alternating transparent / grey85.
    * ``stack=True`` — the stacked-violin figure: one shared-x GridSpec column
      per ``group_by``, one row per feature with its own y limits and exactly
      two y ticks, one y label and one legend.  R builds it by gtable surgery.
    """
    feats = [stat_by] if isinstance(stat_by, str) else list(stat_by)
    groups = [group_by] if isinstance(group_by, str) else (list(group_by) if group_by else [])

    coexp = None
    if calculate_coexp:
        var_index = adata.raw.var_names if use_raw else adata.var_names
        genes = [f for f in feats if f in var_index]
        if len(genes) > 1:
            others = [f for f in feats if f not in genes]
            if others:
                warnings.warn(
                    f"{','.join(others)} is not used when calculating co-expression",
                    stacklevel=2,
                )
            coexp = calculate_coexp_fn(adata, genes, layer=layer, use_raw=use_raw)

    values = fetch_data(adata, feats, layer=layer, use_raw=use_raw)
    feats = [f for f in feats if f in values.columns]
    if not feats:
        raise ValueError("none of `stat_by` could be resolved")
    frame = pd.DataFrame(index=pd.Index(adata.obs_names))
    for f in feats:
        frame[f] = pd.to_numeric(values[f], errors="coerce").to_numpy()
    if coexp is not None:
        frame["CoExp"] = coexp.to_numpy()
        feats.append("CoExp")
    for col in dict.fromkeys([*groups, *(c for c in (split_by, bg_by) if c)]):
        if col not in adata.obs.columns:
            raise KeyError(f"{col} is not in adata.obs.")
        # keep the Categorical: its categories are the declared level order
        frame[col] = as_ordered_categorical(adata.obs[col]).set_axis(frame.index)

    return expression_stat_plot(
        frame, feats, group_by=groups or None, split_by=split_by, bg_by=bg_by,
        plot_by=plot_by, fill_by=fill_by, plot_type=plot_type, cells=cells,
        palette=palette, palcolor=palcolor, alpha=alpha, bg_palette=bg_palette,
        bg_palcolor=bg_palcolor, bg_alpha=bg_alpha, add_box=add_box, add_point=add_point,
        add_trend=add_trend, add_stat=add_stat, add_line=add_line,
        jitter_width=jitter_width, jitter_height=jitter_height, stack=stack, flip=flip,
        sort=sort, same_y_lims=same_y_lims, y_min=y_min, y_max=y_max, y_trans=y_trans,
        y_nbreaks=y_nbreaks, keep_empty=keep_empty, individual=individual,
        comparisons=comparisons, ref_group=ref_group, pairwise_method=pairwise_method,
        multiplegroup_comparisons=multiplegroup_comparisons, multiple_method=multiple_method,
        sig_label=sig_label, sig_labelsize=sig_labelsize, panel_size=panel_size,
        aspect_ratio=aspect_ratio, title=title, subtitle=subtitle,
        xlab=xlab if (xlab is not None or groups) else "All groups", ylab=ylab,
        legend_position=legend_position, nrow=nrow, ncol=ncol, byrow=byrow, force=force,
        seed=seed,
    )


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

    R original: ``CellStatPlot`` (``SCP-plot.R`` L4463) -> ``StatPlot``
    (L4564), from SCP by Hao Zhang (GPL-3),
    https://github.com/zhanghao-njmu/SCP.  Milestone 5.  All eleven
    ``plot_type``\\ s are implemented; ``sankey``, ``chord``, ``venn`` and
    ``upset`` are drawn by ``_stat_flow.py``.

    The aggregation core is a complete cross-tab, not a groupby-count::

        tab = pd.crosstab(obs[stat_by], obs[group_by], dropna=not na_stat)
        long = tab.melt(ignore_index=False)
        long["value"] = long.Freq / long.groupby(group_by).Freq.transform("sum")

    Empty cells are *retained* through the aggregation and dropped only at the
    end, which is what makes ``keep_empty`` meaningful.

    ``stat_by`` with two or more columns switches to the set-relationship
    types (``sankey``, ``chord``, ``venn``, ``upset``) and forces
    ``stat_type="count"``.  ``individual=True`` gives every group level its
    own panel (see :func:`stat_plot`).
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

    R original: ``StatPlot`` (``SCP-plot.R`` L4564), from SCP by Hao Zhang
    (GPL-3), https://github.com/zhanghao-njmu/SCP.  The aggregation core is a
    **complete** cross-tab of ``stat_by x group_by`` within each split level,
    keeping empty cells, and (for ``stat_type="percent"``) dividing by the
    group total.  Get that right before worrying about any geometry: every
    ``plot_type`` is a different view of the same table.

    Empty groups follow R's filter, which drops rows whose value is ``NA``:
    without ``keep_empty`` a group with no cells loses its slot in percent mode
    (0/0) but keeps it, at zero, in count mode.

    Multi-column ``stat_by`` is only meaningful for the flow and set types:
    ``sankey`` (2+ columns, one node stack per column), ``chord`` (exactly 2),
    ``venn`` (1-7 sets) and ``upset``; the last two binarise each column with
    ``stat_level``. For those four ``group_by`` is ignored with a warning and
    the statistic is always a count, as in R -- R also warns about the count,
    but on the *default* ``stat_type``, so every call would warn; that one is
    not repeated here. The renderers live in ``_stat_flow.py``; see
    :func:`scp.pl._stat_flow.flow_plot` for the divergences from SCP and
    :func:`scp.pl._stat_flow.upset_layout` for how the upset bars are ordered.
    Like the other three, ``upset`` draws one block per ``split_by`` level.

    ``flip=True`` is ``coord_flip`` for ``bar``, ``area``, ``trend`` and
    ``dot``: the group order is reversed (first level on top), the panel and
    its ``aspect_ratio`` turn, and every layer is transposed (SCP-plot.R:4771,
    4955).  ``pie`` and ``rose`` start their polar angle at ``pi/2`` instead,
    as in R.  ``ring`` does the same here; R adds ``coord_flip`` to it, which
    replaces ``coord_polar`` and turns the ring into a flat bar with its dummy
    hole level showing — not reproduced.

    ``individual=True`` draws one panel per (``split_by`` level, ``group_by``
    level) pair, each holding that single group, in R's order: group levels
    vary fastest within each split level (SCP-plot.R:4710).

    ``label=True`` labels every type, as R's ``geom_text_repel(force = 0)``
    does: mid-segment when stacked, on the bar top when dodged, on the dot,
    mid-ring for rose; the text is R's ``round(value * 100, 1)`` percent or
    the count.  R also prints zero values, piled at a segment boundary; those
    are left out.
    """
    stat_cols = list(stat_by)
    flow_types = ("sankey", "chord", "venn", "upset")
    if len(stat_cols) >= 2 and plot_type not in flow_types:
        raise ValueError(
            "several `stat_by` columns are only meaningful for plot_type in "
            "{'sankey', 'chord', 'venn', 'upset'}"
        )
    if plot_type == "chord" and len(stat_cols) != 2:
        raise ValueError("`stat_by` must name exactly 2 columns when plot_type is 'chord'")
    if plot_type == "venn" and len(stat_cols) > 7:
        raise ValueError("`stat_by` can name at most 7 columns when plot_type is 'venn'")
    if plot_type in ("rose", "ring", "pie"):
        aspect_ratio = 1.0

    if plot_type in flow_types:
        if group_by is not None:
            warnings.warn("`group_by` is not used for sankey, chord, venn or upset plots",
                          stacklevel=2)
        if plot_type == "upset":
            from ._stat_flow import upset_plot

            return upset_plot(data, stat_cols, split_by=split_by, stat_level=stat_level,
                              label_size=label_size, panel_size=panel_size, nrow=nrow,
                              ncol=ncol, byrow=byrow)
        from ._stat_flow import flow_plot

        return flow_plot(data, stat_cols, plot_type, split_by=split_by, stat_level=stat_level,
                         palette=palette, palcolor=palcolor, alpha=alpha, na_color=na_color,
                         label_size=label_size, panel_size=panel_size,
                         legend_position=legend_position, nrow=nrow, ncol=ncol, byrow=byrow)

    stat_col = stat_cols[0]
    # Levels come from the columns as given: the same column may serve as
    # stat_by and split_by (or group_by), and reading it back after it was
    # turned into strings would re-derive its levels by first appearance.
    original = data
    stat_cat = as_ordered_categorical(original[stat_col], show_na=na_stat)
    stat_levels = [str(x) for x in stat_cat.cat.categories]
    data = data.copy()
    data[stat_col] = np.asarray(stat_cat).astype(str)

    if group_by is None:
        group_by = "__all__"
        data[group_by] = ""
        group_levels = [""]
    else:
        gcat = as_ordered_categorical(original[group_by])
        group_levels = [str(x) for x in gcat.cat.categories]
        if group_by != stat_col:
            data[group_by] = np.asarray(gcat).astype(str)

    if split_by is None:
        split_levels: list[str | None] = [None]
    else:
        scat = as_ordered_categorical(original[split_by])
        split_levels = [str(x) for x in scat.cat.categories]
        if split_by != stat_col:
            data[split_by] = np.asarray(scat).astype(str)

    colors = discrete_palette(stat_levels, palette=palette, palcolor=palcolor)
    if "NA" in stat_levels:
        colors["NA"] = na_color

    cartesian_flip = flip and plot_type in ("bar", "area", "trend", "dot")
    if cartesian_flip:
        # coord_flip: R inverts aspect.ratio; the panel turns with it
        aspect_ratio = 1 / aspect_ratio if aspect_ratio else aspect_ratio
        panel_size = (panel_size[1], panel_size[0])
    th = theme_scp(aspect_ratio=aspect_ratio, legend_position=legend_position)
    # One panel per split level, or -- individual=TRUE -- per (split, group)
    # pair with the group level varying fastest, named "group:element:split"
    # as R names them.
    if individual:
        g_name = "" if group_by == "__all__" else group_by
        panels = [(f"{g_name}:{g}:{'' if sp is None else sp}", sp, [g])
                  for sp in split_levels for g in group_levels]
    else:
        panels = [(str(sp), sp, list(group_levels)) for sp in split_levels]
    keys = [k for k, _, _ in panels]
    legend = LegendColumn(position=legend_position)
    pg = PanelGrid(len(keys), panel_size=panel_size, nrow=nrow, ncol=ncol, byrow=byrow,
                   keys=keys, theme=th, legend=legend,
                   polar=plot_type in ("rose", "ring", "pie"))

    for key, sp, elements in panels:
        ax = pg.axes[key]
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

        levels_x = list(elements)
        if not keep_empty:
            # R drops a group only when its rows were filtered out, i.e. when
            # its value is NA: percent of an empty group (0/0). In count mode
            # an empty group still has rows (Freq = 0) and keeps its slot.
            kept = [g for g in levels_x if stat_type == "count" or tab[g].sum() > 0]
            # an individual panel whose one group is empty is left blank, as
            # R leaves it; a shared panel never loses every slot
            levels_x = kept if individual else (kept or levels_x)
        if flip and plot_type not in ("pie", "rose"):
            levels_x = levels_x[::-1]

        if not levels_x:
            ax.set_xticks([])
            if sp is not None:
                ax.set_title(str(sp), fontsize=th.size("plot.title"))
            continue

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


_CARTESIAN = ("bar", "area", "trend", "dot")


def _draw_stat_panel(ax, values, stat_levels, levels_x, colors, *, plot_type, stat_type,
                     position, alpha, label, label_size, flip, bg_by, block, group_by, theme):
    """Draw one panel of the aggregated table.

    The cartesian types go through :class:`~scp.pl._stat_feature._Ax`, so
    ``flip`` (R's ``coord_flip``) transposes every layer; the caller has
    already reversed the group order, as R reverses the factor.  The marks
    come from :func:`_draw_stat_geoms`, which :func:`draw_stat_inset` shares;
    this adds what only a full panel has -- ticks, value limits, stripes.
    """
    n_x = len(levels_x)
    x = np.arange(n_x)
    A = _Ax(ax, flip and plot_type in _CARTESIAN)
    _draw_stat_geoms(A, values, stat_levels, colors, plot_type=plot_type, stat_type=stat_type,
                     position=position, alpha=alpha, label=label, label_size=label_size,
                     flip=flip)

    if plot_type in ("bar", "area", "trend"):
        A.set_pos_ticks(x, levels_x)
        if stat_type == "percent" and position == "stack":
            A.set_value_lim(0, 1)
            A.value_axis.set_major_formatter(PercentFormatter(xmax=1))
    elif plot_type == "dot":
        A.set_pos_ticks(x, levels_x)
        A.value_axis.set_ticks(range(len(stat_levels)))
        A.value_axis.set_ticklabels(stat_levels)
        A.set_value_lim(-0.6, len(stat_levels) - 0.4)
    elif plot_type == "rose":
        ax.set_xticks(x * 2 * np.pi / max(n_x, 1))
        ax.set_xticklabels(levels_x, fontsize=7)
        ax.set_yticklabels([])
    elif plot_type in ("ring", "pie"):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, 1)

    # Background stripes exist only when position != "stack" (R sets bg_layer
    # to NULL under stack, because the bars already tile the panel).
    if bg_by is not None and position != "stack" and plot_type in _CARTESIAN:
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
                A.pos_span(i - 0.5, i + 0.5, color=bg_colors[str(b)], alpha=0.2,
                           zorder=0, linewidth=0)


def _draw_stat_geoms(A: _Ax, values: pd.DataFrame, stat_levels, colors, *, plot_type,
                     stat_type, position, alpha, label, label_size, flip) -> None:
    """The marks of one ``plot_type``: ``values`` is stat level x group.

    Positions are ``0 .. n_groups - 1`` and sizes are physical (points), so
    the same call draws a full panel or a ``cell_dim_plot`` inset; limits and
    ticks are the caller's.
    """
    ax = A.ax
    n_x = values.shape[1]
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
                    A.bar(x, v, 0.8, bottom=bottom, color=colors[lv], alpha=alpha,
                          edgecolor="black", linewidth=0.5)
                elif plot_type == "area":
                    A.fill_between(x, bottom, bottom + v, color=colors[lv], alpha=alpha,
                                   edgecolor="black", linewidth=0.5)
                else:  # trend: a ribbon joining adjacent bars, plus a narrow bar
                    A.fill_between(x, bottom, bottom + v, color=colors[lv],
                                   alpha=alpha / 2, edgecolor="#7F7F7F", linewidth=0.5)
                    A.bar(x, v, 0.6, bottom=bottom, color=colors[lv], alpha=alpha,
                          edgecolor="black", linewidth=0.5)
                bottom = bottom + v
        else:
            w = 0.8 / max(len(stat_levels), 1)
            for i, lv in enumerate(stat_levels):
                v = np.nan_to_num(values.loc[lv].to_numpy())
                off = (i - (len(stat_levels) - 1) / 2) * w
                A.bar(x + off, v, w * 0.95, color=colors[lv], alpha=alpha,
                      edgecolor="black", linewidth=0.5)

    elif plot_type == "dot":
        # y is the CATEGORY, not the value: a dot matrix sized by the statistic
        vmax = np.nanmax(values.to_numpy()) or 1.0
        for yi, lv in enumerate(stat_levels):
            v = np.nan_to_num(values.loc[lv].to_numpy())
            A.scatter(x, np.full(n_x, yi), s=12 ** 2 * v / vmax, facecolor=colors[lv],
                      edgecolor="black", linewidth=0.5, alpha=alpha)

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
                            r_stat_label(frac[i] if stat_type == "percent"
                                         else totals.iloc[i], stat_type),
                            ha="center", va="center", fontsize=label_size * 2)
                halo(t, foreground="white", radius=0.1)
    if label and plot_type not in ("ring", "pie"):
        _draw_stat_labels(A, values, stat_levels, plot_type=plot_type, stat_type=stat_type,
                          position=position, label_size=label_size)


def r_stat_label(value: float, stat_type: str) -> str:
    """StatPlot's label text: ``paste0(round(value * 100, 1), "%")``, or the count."""
    if stat_type == "percent":
        return f"{round(float(value) * 100, 1):.15g}%"
    return f"{float(value):.15g}"


def _draw_stat_labels(A: _Ax, values: pd.DataFrame, stat_levels, *, plot_type, stat_type,
                      position, label_size) -> None:
    """``geom_text_repel(force = 0)``: one label per drawn value, where R puts it.

    Stacked labels sit mid-segment (``position_stack(vjust = 0.5)``); dodged
    ones on top of their bar, spread over 0.9 of the slot, which is where
    ``position_dodge2`` puts a zero-width text (not quite over its 0.8-wide
    bar); ``dot`` labels sit on the dot; ``rose`` labels mid-ring.  R also
    prints the zeros, at the segment boundary; those are left out.
    """
    n_x = values.shape[1]
    x = np.arange(n_x)

    def put(p, v, value, polar=False):
        if value <= 0:
            return
        txt = r_stat_label(value, stat_type)
        kw = dict(ha="center", va="center", fontsize=label_size * 2, color="black")
        t = A.ax.text(p, v, txt, **kw) if polar else A.text(p, v, txt, **kw)
        halo(t, foreground="white", radius=0.1)

    if plot_type in ("bar", "area", "trend") and position == "stack":
        bottom = np.zeros(n_x)
        for lv in reversed(stat_levels):
            v = np.nan_to_num(values.loc[lv].to_numpy())
            for xi in range(n_x):
                put(xi, bottom[xi] + v[xi] / 2, v[xi])
            bottom = bottom + v
    elif plot_type in ("bar", "area", "trend"):
        k = max(len(stat_levels), 1)
        for i, lv in enumerate(stat_levels):
            v = np.nan_to_num(values.loc[lv].to_numpy())
            for xi in range(n_x):
                put(xi + (i - (k - 1) / 2) * 0.9 / k, v[xi], v[xi])
    elif plot_type == "dot":
        for yi, lv in enumerate(stat_levels):
            v = np.nan_to_num(values.loc[lv].to_numpy())
            for xi in range(n_x):
                put(xi, yi, v[xi])
    elif plot_type == "rose":
        theta = x * 2 * np.pi / max(n_x, 1)
        bottom = np.zeros(n_x)
        for lv in stat_levels:
            v = np.nan_to_num(values.loc[lv].to_numpy())
            for xi in range(n_x):
                put(theta[xi], bottom[xi] + v[xi] / 2, v[xi], polar=True)
            bottom = bottom + v


# The inset types CellDimPlot can paste. Its StatPlot(individual = TRUE) call
# returns one subplot per group only for these; the other four are keyed by
# split level, so R finds no subplot to place (SCP-plot.R:1729).
INSET_TYPES = ("pie", "ring", "bar", "rose", "trend", "area", "dot")


def draw_stat_inset(ax, values: pd.Series, colors, *, plot_type: str, stat_type: str,
                    position: str, alpha: float, label: bool, label_size: float) -> None:
    """One group's ``StatPlot(individual = TRUE)`` panel, as ``CellDimPlot`` pastes it.

    ``values`` is the group's column of the cross-tab, indexed by stat level.
    The marks are :func:`_draw_stat_geoms`' -- the ones :func:`stat_plot`
    draws -- and the axes take what ``+ theme_void()`` leaves of R's panel:
    no axes, and the scale ranges ggplot trains on a single group.  Measured
    on SCP 0.5.6 (``tools/parity/dim_overlays.R``):

    * ``bar`` / ``trend``, stacked: ``scale_x_discrete(expand = c(0, 0))``
      makes the bar (width 0.8, or trend's 0.6) fill the box, and the value
      axis runs from 0 to the group's total.  Trend's ribbon sits exactly
      under its bar; ``stat_align`` pads its ends by 0.1 % of the range,
      which is the whole of R's x range.
    * dodged: one discrete slot plus ggplot's 0.6 expansion, and
      ``limits = c(0, max * 1.1)``.
    * ``dot``: the same slot, and the stat levels at ``1 .. k`` expanded by
      0.6.
    * ``rose``: ``coord_polar`` scales the radius to 0.4 of the box, so the
      stacked total reaches 80 % of its width.
    * ``area`` draws no area: ``geom_area`` needs two x positions to make a
      polygon, and an inset holds one group, so ggplot builds no rows.  Its
      labels still print, centred, as R's text layer does.
    """
    levels = [str(lv) for lv in values.index]
    v = np.nan_to_num(values.to_numpy(dtype=float))
    ax.set_axis_off()
    frame = pd.DataFrame({"g": v}, index=levels)
    A = _Ax(ax, False)
    if plot_type == "area":
        if label:
            _draw_stat_labels(A, frame, levels, plot_type=plot_type, stat_type=stat_type,
                              position=position, label_size=label_size)
    else:
        _draw_stat_geoms(A, frame, levels, colors, plot_type=plot_type, stat_type=stat_type,
                         position=position, alpha=alpha, label=label, label_size=label_size,
                         flip=False)
    if plot_type == "rose":
        ax.set_ylim(0, (v.sum() or 1.0) / 0.8)
    elif plot_type == "dot":
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, len(levels) - 0.4)
    elif position == "stack":
        half = {"bar": 0.4, "trend": 0.3 * 1.002}.get(plot_type, 0.5)
        ax.set_xlim(-half, half)
        ax.set_ylim(0, v.sum() or 1.0)
    else:
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(0, (v.max() or 1.0) * 1.1)


def feature_cor_plot(
    adata,
    features: Sequence[str],
    *,
    group_by: str | None = None,
    split_by: str | None = None,
    layer: str | None = None,
    cells: Sequence[str] | None = None,
    cor_method: Literal["pearson", "spearman"] = "pearson",
    adjust: float = 1.0,
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
    calculate_coexp: bool = False,
    panel_size: tuple[float, float] = (1.4, 1.4),
    nrow: int | None = None,
    ncol: int | None = None,
    byrow: bool = True,
    force: bool = False,
    seed: int = 11,
) -> Figure:
    """Pairwise scatter-plot matrix over features. Milestone 6.

    R original: ``FeatureCorPlot`` (``SCP-plot.R:5232``) — Hao Zhang,
    https://github.com/zhanghao-njmu/SCP.

    The grid puts feature ``f1`` on the column and ``f2`` on the row, as R
    does.  Below the diagonal (row > column, ``f1`` before ``f2``) is the
    scatter, coloured by group, with an optional ``lm`` fit line; above it
    (row < column) is a solid block whose colour and text give the
    correlation; the diagonal holds one violin per group (``geom_violin(scale
    = "width", trim = TRUE)``, bandwidth ``bw.nrd0 * adjust`` as ggplot's
    ``stat_ydensity`` computes it).  Spearman is Pearson on per-feature ranks,
    matching ``proxyC::simil``.

    The equation / r² / p text of the scatter panels follows ``add_equation``,
    ``add_r2`` and ``add_pvalue`` alone; ``add_smooth`` only draws the fit
    line, as in R (SCP-plot.R:5434-5439).

    ``split_by`` draws one matrix per split level, titled with the level, and
    computes the correlations, fits and violins within that split
    (SCP-plot.R:5371).  The matrices are panels of a :class:`PanelGrid`
    (``nrow`` / ``ncol`` / ``byrow``); each is ``n * panel_size`` across.
    Split levels without cells are skipped (R would attempt an empty matrix).

    ``calculate_coexp=True`` appends a ``CoExp`` feature, the per-cell
    geometric mean of the gene features (any number of them, as in R), from
    :func:`scp.fetch.calculate_coexp`; obs features are left out of the
    product with R's warning.  R bug fixed: ``FeatureCorPlot``'s copy of this
    block reads a ``status`` variable it never assigns, so in R the option
    always errors; the shared function computes the data type itself.

    ``pt_size`` is in ggplot millimetres; ``None`` is R's
    ``min(3000 / n_cells, 0.5)``.
    """
    from ._stat_feature import _PT, _nrd0

    feats = list(features)
    coexp = None
    if calculate_coexp:
        genes = [f for f in feats if f in adata.var_names]
        if genes:
            others = [f for f in feats if f not in genes]
            if others:
                warnings.warn(
                    f"{','.join(others)} is not used when calculating co-expression",
                    stacklevel=2,
                )
            coexp = calculate_coexp_fn(adata, genes, layer=layer)
    n_feats = len(feats) + (coexp is not None)
    if n_feats < 2:
        raise ValueError("`features` needs at least two entries")
    if n_feats > 10 and not force:
        raise ValueError(
            f"{n_feats} features would make {n_feats ** 2} panels. "
            "Pass force=True to proceed anyway."
        )

    frame = fetch_data(adata, feats, layer=layer)
    feats = [f for f in feats if f in frame.columns]
    if coexp is not None:
        frame = frame.assign(CoExp=coexp.to_numpy())
        feats.append("CoExp")
    dat = frame[feats].apply(pd.to_numeric, errors="coerce")
    if cells is not None:
        dat = dat.loc[dat.index.intersection(pd.Index(cells))]
    dat = dat[np.isfinite(dat.to_numpy()).all(axis=1)]

    def _levels(col: str | None) -> tuple[pd.Series, list[str]]:
        if col is None:
            return pd.Series([""] * len(dat), index=dat.index), [""]
        cat = as_ordered_categorical(adata.obs[col]).loc[dat.index]
        return (pd.Series(np.asarray(cat).astype(str), index=dat.index),
                [str(c) for c in cat.cat.categories])

    gvals, glevels = _levels(group_by)
    svals, slevels = _levels(split_by)
    slevels = [s for s in slevels if (svals == s).any()]
    colors = discrete_palette(glevels, palette=palette, palcolor=palcolor)

    n = len(feats)
    cor_ramp = continuous_palette(palette=cor_palette, n=200)
    cor_cmap = LinearSegmentedColormap.from_list("cor", cor_ramp, N=256)
    cor_norm = Normalize(*cor_range)
    size_mm = pt_size if pt_size is not None else min(3000 / max(len(dat), 1), 0.5)
    raster = raster if raster is not None else len(dat) * n * (n - 1) / 2 > 1e5

    th = theme_scp()
    legend = LegendColumn(position="right")
    keys = [s if split_by is not None else "all" for s in slevels]
    pg = PanelGrid(len(keys), panel_size=(panel_size[0] * n, panel_size[1] * n),
                   nrow=nrow, ncol=ncol, byrow=byrow, keys=keys, theme=th, legend=legend)
    fig = pg.fig
    grid_axes: list = []
    for key, s in zip(keys, slevels, strict=True):
        holder = pg.axes[key]
        spec = holder.get_subplotspec()
        assert spec is not None
        sub = spec.subgridspec(n, n, wspace=0.08, hspace=0.08)
        holder.remove()
        rows = (svals == s).to_numpy()
        d = dat.loc[rows]
        g = gvals.loc[rows]
        _splom(fig, sub, d, g, glevels, colors, feats, cor_method=cor_method, adjust=adjust,
               add_equation=add_equation, add_r2=add_r2, add_pvalue=add_pvalue,
               add_smooth=add_smooth, cor_cmap=cor_cmap, cor_norm=cor_norm,
               s=(size_mm * _PT) ** 2, raster=raster, nrd0=_nrd0, axes_out=grid_axes)
        mine = grid_axes[-n * n:]
        if split_by is not None:
            _matrix_title(mine[:n], s)
        # R binds the correlation legend to each matrix: beside its own panel
        cb = pg.colorbar(ScalarMappable(norm=cor_norm, cmap=cor_cmap), ax=mine)
        cb.set_label("Correlation", fontsize=th.size("legend.title"))
        cb.set_ticks(np.linspace(*cor_range, 3).tolist())  # n.breaks = 3: -1, 0, 1 by default
        cb.ax.set_ylim(*cor_range)
        cb.outline.set_edgecolor("black")

    if group_by is not None and len(glevels) > 1:
        legend.add(group_by, [Patch(facecolor=colors[lv], edgecolor="black", label=lv)
                              for lv in glevels])
    return pg.finish()


def _matrix_title(top_row: list, text: str) -> None:
    """R's bold split title over one matrix, centred on its top row.

    A blank title on the first axes reserves the strip in the layout; the
    text itself spans the row, which no single axes' title can.
    """
    top_row[0].set_title(" ", fontsize=13, fontweight="bold")

    def row_box(renderer):
        return Bbox.union([a.get_window_extent(renderer) for a in top_row])

    ann = top_row[0].annotate(text, xy=(0.5, 1.0), xycoords=row_box, xytext=(0, 5),
                              textcoords="offset points", ha="center", va="bottom",
                              fontsize=13, fontweight="bold", annotation_clip=False)
    ann.set_in_layout(False)


def _splom(fig, sub, dat, gvals, glevels, colors, feats, *, cor_method, adjust, add_equation,
           add_r2, add_pvalue, add_smooth, cor_cmap, cor_norm, s, raster, nrd0,
           axes_out) -> None:
    """One ``FeatureCorPlot`` matrix, drawn into the cells of ``sub``."""
    n = len(feats)
    mat = dat.to_numpy(dtype=float)
    if cor_method == "spearman":
        mat_c = np.apply_along_axis(lambda c: pd.Series(c).rank().to_numpy(), 0, mat)
    else:
        mat_c = mat
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(mat_c, rowvar=False) if len(dat) > 1 else np.full((n, n), np.nan)
    present = [lv for lv in glevels if (gvals == lv).any()]
    cell_colors = [colors[v] for v in gvals]

    th = theme_scp()
    axes = [[apply_theme(fig.add_subplot(sub[j, i]), th) for i in range(n)]
            for j in range(n)]  # row-major, as the tests and R's rbind of rows
    for j, f2 in enumerate(feats):          # row
        for i, f1 in enumerate(feats):      # column
            ax = axes[j][i]
            axes_out.append(ax)
            if i == j:
                # diagonal: geom_violin(scale = "width", trim = TRUE), bw.nrd0
                for k, lv in enumerate(present):
                    v = dat.loc[(gvals == lv).to_numpy(), f1].to_numpy(dtype=float)
                    if v.size < 2:
                        continue  # stat_ydensity drops groups of fewer than two
                    if np.ptp(v) == 0:
                        ax.plot([k - 0.45, k + 0.45], [v[0], v[0]], color="black",
                                linewidth=0.4)
                        continue
                    parts = ax.violinplot([v], positions=[k], widths=0.9, showextrema=False,
                                          bw_method=nrd0(v) * adjust / float(np.std(v, ddof=1)),
                                          points=512)
                    for body in cast(list, parts["bodies"]):
                        body.set(facecolor=colors[lv], alpha=1.0, edgecolor="black",
                                 linewidth=0.4)
                ax.set_xticks(range(len(present)))
                ax.set_xticklabels([])
                ax.set_xlim(-0.6, len(present) - 0.4)
            elif i < j:
                # f1 before f2: scatter coloured by group, fit line, fit stats
                ax.scatter(dat[f1], dat[f2], s=s, c=cell_colors, linewidths=0,
                           rasterized=raster)
                x, y = dat[f1].to_numpy(dtype=float), dat[f2].to_numpy(dtype=float)
                fit = linregress(x, y) if len(dat) > 2 and np.ptp(x) > 0 else None
                if add_smooth and fit is not None:
                    xs = np.linspace(x.min(), x.max(), 80)
                    ax.plot(xs, fit.intercept + fit.slope * xs, color="red", linewidth=1)
                lines = []
                if fit is not None:
                    if add_equation:
                        sign = "-" if fit.slope < 0 else "+"
                        lines.append(f"y = {fit.intercept:.2g} {sign} {abs(fit.slope):.2g}x")
                    if add_r2:
                        lines.append(f"$r^2$ = {fit.rvalue ** 2:.2g}")
                    if add_pvalue:
                        lines.append(f"p = {fit.pvalue:.2g}")
                if lines:
                    t = ax.text(0.04, 0.96, "\n".join(lines), transform=ax.transAxes,
                                ha="left", va="top", fontsize=6)
                    halo(t, foreground="white", radius=0.1)
            else:
                # f1 after f2: a solid block whose colour encodes the correlation
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
            if j != n - 1 or i == j:
                ax.set_xticklabels([])
            if i == 0:
                ax.set_ylabel(f2, fontsize=7.5)
            if j == n - 1:
                ax.set_xlabel(f1, fontsize=7.5)
            ax.tick_params(labelsize=6)


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
    cb = pg.colorbar(sm)
    cb.set_label(color_name, fontsize=th.size("legend.title"))
    return pg.finish()
