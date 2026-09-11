"""Statistical / distribution plots.

Two engines do the work; the ``*_plot`` functions are thin AnnData wrappers.
Keep that separation — it is the reason the R code is testable at all.

* :func:`expression_stat_plot` — continuous values per cell (violin/box/bar/dot/col)
* :func:`cell_stat_plot` / :func:`stat_plot` — composition of categorical columns

Spec: ``docs/porting_briefs/stat_plots.md``.
"""

from __future__ import annotations

from typing import Literal, Sequence

import pandas as pd
from matplotlib.figure import Figure

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
    raise NotImplementedError("Milestone 4. Spec: docs/porting_briefs/stat_plots.md §1.")


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
