"""Dimensional-reduction plots.

``cell_dim_plot`` is the **reference pattern** every other plot function
follows: resolve -> build a per-panel frame -> allocate a PanelGrid -> draw
into each Axes -> register legend handles.  Read it before porting anything
else.

The optional layers (density, marks, graph edges, hexagons, stat pies,
lineages, PAGA, velocity) live in :mod:`scp.pl._dim_layers`, with their pure
numeric cores in :mod:`scp.pl._dim_geom`, so the panel loop here stays
readable.  Do not change a signature without updating
``docs/04_api_mapping.md`` — the mapping table is the contract.

Ported from SCP by Hao Zhang, https://github.com/zhanghao-njmu/SCP (GPL-3).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..colors import adjcolors, blendcolors
from ..fetch import (
    as_ordered_categorical,
    default_reduction,
    fetch_data,
    reduction_key,
)
from ..fetch import (
    calculate_coexp as _coexp,
)
from ..layout import LegendColumn, PanelGrid
from ..palettes import NA_COLOR_DEFAULT, continuous_palette, discrete_palette
from ..theme import Theme, halo, theme_scp
from . import _dim_layers as layers

__all__ = ["cell_dim_plot", "feature_dim_plot", "cell_dim_plot_3d", "feature_dim_plot_3d"]

BlendMode = Literal["blend", "average", "screen", "multiply"]
_GREY90 = "#E5E5E5"
_GREY40 = "#666666"


def _auto_pt_size(n: int) -> float:
    """SCP: ``min(3000 / n_cells, 0.5)`` in ggplot mm; scaled to matplotlib pt^2."""
    mm = min(3000.0 / max(n, 1), 0.5)
    return max((mm * 2.845) ** 2, 0.5)  # mm -> pt, then area


def _mm_area(mm: float) -> float:
    """A ggplot point ``size`` (mm) as a matplotlib marker area (pt^2)."""
    return (mm * 2.845) ** 2


def _check_highlight(adata, cells_highlight) -> list[str] | bool | None:
    """R's up-front ``cells.highlight`` validation: none found errors, some missing warns."""
    if cells_highlight is None or cells_highlight is True:
        return cells_highlight
    if cells_highlight is False:
        return None
    req = [str(c) for c in cells_highlight]
    names = pd.Index(adata.obs_names)
    found = names.intersection(pd.Index(req))
    if len(found) == 0:
        raise ValueError("No cells in `cells_highlight` found in adata.")
    if len(found) < len(set(req)):
        warnings.warn("Some cells in `cells_highlight` not found in adata.", stacklevel=3)
    return [c for c in req if c in set(found)]


def _resolve_graph(adata, graph):
    if graph is None:
        return None
    if isinstance(graph, str):
        if graph not in adata.obsp:
            raise KeyError(f"graph {graph!r} not in adata.obsp ({list(adata.obsp)})")
        return adata.obsp[graph]
    return graph


def _positions(adata, index: pd.Index) -> np.ndarray:
    """Row positions in ``adata`` of the cells in ``index``."""
    return pd.Index(adata.obs_names).get_indexer(index)


def _read_paga_arg(adata, paga, type_: str):
    """``paga=True`` reads ``adata.uns['paga']``; a dict is used as given (SCP's ``srt@misc$paga``)."""
    src = adata.uns["paga"] if paga is True else paga
    if not isinstance(src, Mapping):
        raise TypeError("`paga` must be True (use adata.uns['paga']) or a PAGA dict")

    def _dense(m):
        return None if m is None else np.asarray(m.todense() if hasattr(m, "todense") else m,
                                                 dtype=float)

    return (_dense(src[type_]), str(src["groups"]), _dense(src.get("transitions_confidence")))


def _draw_paga(ax: Axes, adata, frame: pd.DataFrame, *, paga, palette, palcolor, type_: str,
               node_size: float, edge_threshold: float, edge_size, edge_color: str,
               edge_alpha: float, transition_threshold: float, transition_size,
               transition_color: str, transition_alpha: float, show_transition: bool):
    """The ``PAGAPlot`` layer over a panel: :func:`scp.pl.paga_plot`'s nodes and drawing.

    The node table is ``paga_plot``'s (``_traj._paga_nodes``: each group's
    per-axis median over the plotted cells; SCP ignores
    ``uns['paga']['pos']``), and :func:`scp.pl.graph_plot` draws it, so the
    layer and the standalone plot share one implementation.  Nodes take the
    cell palette over every level, so a node matches its cells.  Lengths
    arrive in ggplot mm and are converted once here; the rescale onto them is
    linear, so converting the range converts every width.  The new artists
    are then lifted to the PAGA layer's z-order.  Returns ``(groups_key,
    levels, colors)`` for the legend.
    """
    from ._traj import _paga_nodes, graph_plot

    conn, groups_key, trans = _read_paga_arg(adata, paga, type_)
    cat = as_ordered_categorical(adata.obs[groups_key])
    levels = [str(x) for x in cat.cat.categories]
    if conn.shape[0] != len(levels):
        raise ValueError(f"nlevels in {groups_key!r} is not identical with the group in paga")
    vals = np.asarray(cat).astype(str)[_positions(adata, frame.index)]
    node, idx = _paga_nodes(vals, frame["x"].to_numpy(dtype=float),
                            frame["y"].to_numpy(dtype=float), levels, groups_key,
                            categories=levels)
    present = [str(x) for x in node.index]
    colors = discrete_palette(levels, palette=palette, palcolor=palcolor)
    E = conn[np.ix_(idx, idx)]
    T = None
    if show_transition:
        if trans is None:
            warnings.warn("transitions_confidence need to be calculated first.", stacklevel=3)
        else:
            T = trans[np.ix_(idx, idx)]
    tree = type_ == "connectivities_tree"

    before = set(ax.get_children())
    graph_plot(node, E, transition=T, node_group=groups_key, node_palette=palette,
               node_palcolor=palcolor, node_size=float(node_size),
               edge_threshold=0.0 if tree else edge_threshold,
               use_triangular="both" if tree else "upper", edge_color=edge_color,
               edge_size=(layers.mm_lw(edge_size[0]), layers.mm_lw(edge_size[1])),
               edge_alpha=edge_alpha, transition_threshold=transition_threshold,
               transition_size=(layers.mm_lw(transition_size[0]),
                                layers.mm_lw(transition_size[1])),
               transition_color=transition_color, transition_alpha=transition_alpha,
               ax=ax, xlab=ax.get_xlabel(), ylab=ax.get_ylabel(), title="")
    for a in ax.get_children():
        if a not in before:
            a.set_zorder(layers.Z_PAGA + 0.1 * a.get_zorder())
    return groups_key, present, colors


def _repel(ax: Axes, xy: np.ndarray, texts: Sequence[str], fontsize: float, force: float,
           point_size: float) -> np.ndarray:
    """Label offsets in points, pushing labels off each other and off their anchors.

    A small stand-in for ggrepel: overlapping label boxes are pushed apart
    along the line between their centres, and a label covering any anchor is
    pushed away from it, for up to 300 rounds.  ``force`` scales each step as
    ``label_repulsion / 20`` (SCP's default is 20).  The result is in points,
    so it survives the final layout pass.
    """
    fig = ax.figure
    anchors = ax.transData.transform(xy)
    px_per_pt = fig.dpi / 72.0
    w = np.array([0.62 * fontsize * max(len(t), 1) for t in texts]) * px_per_pt
    h = np.full(len(texts), 1.2 * fontsize) * px_per_pt
    pos = anchors.copy()
    pad = 2 * px_per_pt
    rad = point_size * 2.845 / 2 * px_per_pt + pad
    step = 0.5 * max(force, 0.0) / 20.0
    if step == 0:
        return np.zeros_like(xy)
    rng = np.random.default_rng(0)
    for _ in range(300):
        moved = 0.0
        for i in range(len(pos)):
            push = np.zeros(2)
            for j in range(len(pos)):
                if i == j:
                    continue
                d = pos[i] - pos[j]
                ox = (w[i] + w[j]) / 2 + pad - abs(d[0])
                oy = (h[i] + h[j]) / 2 + pad - abs(d[1])
                if ox > 0 and oy > 0:
                    if not d.any():
                        d = rng.normal(size=2)
                    push += d / (np.hypot(*d) or 1.0) * min(ox, oy)
            for j in range(len(anchors)):
                d = pos[i] - anchors[j]
                if abs(d[0]) < w[i] / 2 + rad and abs(d[1]) < h[i] / 2 + rad:
                    if not d.any():
                        d = np.array([0.0, 1.0])
                    push += d / (np.hypot(*d) or 1.0) * rad * 0.5
            pos[i] += step * push
            moved = max(moved, float(np.hypot(*push)))
        if moved < 0.5:
            break
    return (pos - anchors) / px_per_pt


def _draw_labels(ax: Axes, xy: np.ndarray, texts: Sequence[str], *, size: float, fg, bg: str,
                 bg_r: float, repel: bool, repulsion: float, point_size: float,
                 point_color: str, segment_color: str) -> None:
    """``geom_text_repel`` as SCP calls it: bold halo text; with repel, a dot and a leader."""
    fontsize = size * 2.5
    fgs = [fg] * len(texts) if isinstance(fg, str) else list(fg)
    if repel:
        ax.scatter(xy[:, 0], xy[:, 1], s=_mm_area(point_size), c=point_color, linewidths=0,
                   zorder=layers.Z_LABEL - 0.1)
        off = _repel(ax, xy, texts, fontsize, repulsion, point_size)
    else:
        off = np.zeros_like(xy)
    for (x, y), txt, (dx, dy), col in zip(xy, texts, off, fgs, strict=True):
        moved = bool(repel and (abs(dx) > 0.5 or abs(dy) > 0.5))
        t = ax.annotate(
            txt, (x, y), xytext=(dx, dy), textcoords="offset points", ha="center",
            va="center", fontsize=fontsize, color=col, fontweight="bold",
            zorder=layers.Z_LABEL, annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color=segment_color, linewidth=0.6,
                            shrinkA=0, shrinkB=0) if moved else None,
        )
        halo(t, foreground=bg, radius=bg_r)


def _limits_of(frame: pd.DataFrame) -> tuple[float, float, float, float]:
    x, y = frame["x"].to_numpy(dtype=float), frame["y"].to_numpy(dtype=float)
    return (float(np.nanmin(x)), float(np.nanmax(x)), float(np.nanmin(y)), float(np.nanmax(y)))


def cell_dim_plot(
    adata,
    group_by: str | Sequence[str],
    *,
    reduction: str | None = None,
    dims: tuple[int, int] = (1, 2),
    split_by: str | None = None,
    cells: Sequence[str] | None = None,
    # aesthetics
    palette: str = "Paired",
    palcolor: Sequence[str] | dict[str, str] | None = None,
    bg_color: str = NA_COLOR_DEFAULT,
    pt_size: float | None = None,
    pt_alpha: float = 1.0,
    show_na: bool = False,
    show_stat: bool | None = None,
    raster: bool | None = None,
    raster_dpi: tuple[int, int] = (512, 512),
    hex: bool = False,
    hex_linewidth: float = 0.5,
    hex_count: bool = True,
    hex_bins: int = 50,
    hex_binwidth: float | tuple[float, float] | None = None,
    # labels
    label: bool = False,
    label_size: float = 4.0,
    label_fg: str = "white",
    label_bg: str = "black",
    label_bg_r: float = 0.1,
    label_insitu: bool = False,
    label_repel: bool = False,
    label_repulsion: float = 20.0,
    label_point_size: float = 1.0,
    label_point_color: str = "black",
    label_segment_color: str = "black",
    # highlight
    cells_highlight: Sequence[str] | bool | None = None,
    cols_highlight: str = "black",
    sizes_highlight: float = 1.0,
    alpha_highlight: float = 1.0,
    stroke_highlight: float = 0.5,
    # overlays (see docs/porting_briefs/dim_plots.md §1.6)
    add_density: bool = False,
    density_color: str = NA_COLOR_DEFAULT,
    density_filled: bool = False,
    density_filled_palette: str = "Greys",
    density_filled_palcolor: Sequence[str] | None = None,
    add_mark: bool = False,
    mark_type: Literal["hull", "ellipse", "rect", "circle"] = "hull",
    mark_expand: float = 3.0,
    mark_alpha: float = 0.1,
    mark_linetype: int | str = 1,
    graph: str | None = None,
    edge_size: tuple[float, float] = (0.05, 0.5),
    edge_alpha: float = 0.1,
    edge_color: str = _GREY40,
    lineages: Sequence[str] | None = None,
    lineages_trim: tuple[float, float] = (0.01, 0.99),
    lineages_span: float = 0.75,
    lineages_palette: str = "Dark2",
    lineages_palcolor: Sequence[str] | None = None,
    lineages_arrow: float | None = 0.1,
    lineages_linewidth: float = 1.0,
    lineages_line_bg: str = "white",
    lineages_line_bg_stroke: float = 0.5,
    lineages_whiskers: bool = False,
    lineages_whiskers_linewidth: float = 0.5,
    lineages_whiskers_alpha: float = 0.5,
    paga: bool | dict | None = None,
    paga_type: Literal["connectivities", "connectivities_tree"] = "connectivities",
    paga_node_size: float = 4.0,
    paga_edge_threshold: float = 0.01,
    paga_edge_size: tuple[float, float] = (0.2, 1.0),
    paga_edge_color: str = _GREY40,
    paga_edge_alpha: float = 0.5,
    paga_transition_threshold: float = 0.01,
    paga_transition_size: tuple[float, float] = (0.2, 1.0),
    paga_transition_color: str = "black",
    paga_transition_alpha: float = 1.0,
    paga_show_transition: bool = False,
    velocity: str | None = None,
    velocity_plot_type: Literal["raw", "grid", "stream"] = "raw",
    velocity_n_neighbors: int | None = None,
    velocity_density: float = 1.0,
    velocity_smooth: float = 0.5,
    velocity_scale: float = 1.0,
    velocity_min_mass: float = 1.0,
    velocity_cutoff_perc: float = 5.0,
    velocity_arrow_color: str = "black",
    velocity_arrow_angle: float = 20.0,
    streamline_width: tuple[float, float] = (0.0, 0.8),
    streamline_alpha: float = 1.0,
    streamline_color: str | None = None,
    streamline_palette: str = "RdYlBu",
    streamline_palcolor: Sequence[str] | None = None,
    streamline_bg_color: str = "white",
    streamline_bg_stroke: float = 0.5,
    stat_by: str | None = None,
    stat_type: Literal["percent", "count"] = "percent",
    stat_plot_type: Literal["pie", "ring", "bar"] = "pie",
    stat_plot_position: Literal["stack", "dodge"] = "stack",
    stat_plot_size: float = 0.15,
    stat_plot_palette: str = "Set1",
    stat_palcolor: Sequence[str] | dict[str, str] | None = None,
    stat_plot_alpha: float = 1.0,
    stat_plot_label: bool = False,
    stat_plot_label_size: float = 3.0,
    # theme / combination
    theme: Theme | None = None,
    aspect_ratio: float | None = 1.0,
    panel_size: tuple[float, float] = (2.6, 2.6),
    title: str | None = None,
    subtitle: str | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    legend_position: Literal["none", "left", "right", "bottom", "top"] = "right",
    nrow: int | None = None,
    ncol: int | None = None,
    byrow: bool = True,
    force: bool = False,
    seed: int = 11,
    ax: Axes | None = None,
) -> Figure:
    """Color cells on a 2-D embedding by one or more categorical ``obs`` columns.

    Port of ``CellDimPlot`` (``SCP-plot.R:1328``) by Hao Zhang,
    https://github.com/zhanghao-njmu/SCP.  Behaviours that are easy to lose
    and must be preserved:

    * every split panel contains **all** cells — those outside the current
      split are drawn in ``bg_color``, not dropped (contrast with
      :func:`feature_dim_plot`, which genuinely partitions);
    * axis limits come from the *global* embedding range, so panels are
      directly comparable;
    * draw order is background cells first, then a **seeded shuffle** of the
      rest, so no group systematically overplots another;
    * labels sit at the per-group *marginal* median of x and y (median of x and
      median of y computed independently — not a medoid, not a density peak);
    * with ``label_insitu=False`` the plot shows ``1, 2, 3 ...`` and the legend
      reads ``"1: Ductal"``.

    Layers stack in SCP's order: ``add_mark``, ``graph``, ``add_density``,
    points (or ``hex``), highlight, ``stat_by`` insets, ``lineages``,
    ``paga``, ``velocity``, labels.  Sizes and line widths are ggplot's
    millimetres (``mark_expand``, ``edge_size``, ``lineages_linewidth`` ...);
    ``lineages_arrow`` is the arrowhead length in inches, or ``None``.

    Deliberate divergences, each a fix or a matplotlib-native replacement:

    * ``graph`` reads ``adata.obsp[graph]`` and works on its sparse triplets;
      R densifies the whole adjacency (O(n^2) memory).
    * ``hex`` takes precedence over ``raster``: rasterisation here is only a
      rendering flag, where in R it is a different geom that silently
      disables ``hex``.
    * ``paga`` with ``cells=``: R's ``PAGAPlot`` intersects *group* names with
      *cell* names and so drops every node; here nodes sit at the medians of
      the plotted cells.
    * ``velocity`` reads ``adata.obsm[f"{velocity}_{basis}"]`` (scvelo's
      naming, e.g. ``velocity="velocity"`` -> ``velocity_umap``).  Grid and
      stream modes smooth it with :func:`scp.pl.compute_velocity_on_grid`,
      the numpy port of SCP's function that :func:`scp.pl.velocity_plot`
      uses too, so no scvelo is needed and the grid is R's; the lineage and
      PAGA layers likewise share :func:`scp.pl.lineage_plot`'s fit and
      :func:`scp.pl.paga_plot`'s nodes and drawing.  The streamlines are
      matplotlib's ``streamplot``, so ``metR``'s
      ``L``/``min.L``/``res``/``n`` have no counterpart and are not exposed,
      and the line width is constant rather than growing along the line.
    * ``label_repel`` uses a small built-in repulsion pass, not ggrepel.
    * ``add_mark="hull"`` ports ``concaveman``; it could not be checked
      against R because the parity environment lacks that package (see
      ``docs/07_milestones.md``).
    * ``stat_plot_type`` supports the inset types that read at that size:
      ``pie``, ``ring`` and ``bar``.
    """
    rng = np.random.default_rng(seed)
    groups = [group_by] if isinstance(group_by, str) else list(group_by)
    if stat_plot_type not in ("pie", "ring", "bar"):
        raise NotImplementedError(
            f"stat_plot_type={stat_plot_type!r} insets are not ported "
            "(docs/porting_briefs/dim_plots.md §1.6); use 'pie', 'ring' or 'bar'."
        )
    if (paga or velocity) and split_by is not None:
        raise ValueError(
            f"{'paga' if paga else 'velocity'} can only plot on the non-split data"
        )
    cells_highlight = _check_highlight(adata, cells_highlight)
    graph_mat = _resolve_graph(adata, graph)

    reduction = reduction or default_reduction(adata)
    key = reduction_key(reduction)
    emb = np.asarray(adata.obsm[reduction])
    xi, yi = dims[0] - 1, dims[1] - 1

    obs = adata.obs
    frame = pd.DataFrame(
        {"x": emb[:, xi], "y": emb[:, yi]}, index=pd.Index(adata.obs_names)
    )
    # Capture the level order HERE, while the Categorical still exists.
    # Storing `.to_numpy()` flattens it to an object array, and re-deriving the
    # levels from those values later yields first-appearance-in-data order
    # rather than the column's declared order -- which silently shifts every
    # palette index and legend position.
    group_levels: dict[str, list[str]] = {}
    for g in groups:
        cat = as_ordered_categorical(obs[g], show_na=show_na)
        group_levels[g] = [str(lv) for lv in cat.cat.categories]
        frame[g] = cat.to_numpy()
    split_levels: list[str | None]
    if split_by is None:
        split_levels = [None]
        frame["__split__"] = ""
    else:
        sp_cat = as_ordered_categorical(obs[split_by])
        frame["__split__"] = sp_cat.to_numpy()
        split_levels = list(sp_cat.cat.categories)

    # Global limits, computed before any `cells` subsetting -- panels stay aligned.
    xlim = (float(np.nanmin(emb[:, xi])), float(np.nanmax(emb[:, xi])))
    ylim = (float(np.nanmin(emb[:, yi])), float(np.nanmax(emb[:, yi])))

    if cells is not None:
        frame = frame.loc[frame.index.intersection(pd.Index(cells))]
    rows_all = _positions(adata, frame.index)

    n_cells = len(frame)
    size = pt_size if pt_size is not None else _auto_pt_size(n_cells)
    if raster is None:
        raster = n_cells > 100_000
    if show_stat is None:
        show_stat = not (theme.blank if theme else False)

    for g in groups:
        nlev = len(pd.unique(frame[g].dropna()))
        if nlev > 100 and not force:
            raise ValueError(
                f"`{g}` has {nlev} levels, which will produce an unreadable legend. "
                "Pass force=True to proceed anyway."
            )

    # Layers that do not depend on the panel are computed once.
    curves = None
    if lineages:
        curves = layers.lineage_curves(adata, list(lineages), emb[:, [xi, yi]], rows_all,
                                 trim=lineages_trim, span=lineages_span)
    stat_levels: list[str] = []
    stat_colors: dict[str, str] = {}
    stat_values = None
    if stat_by is not None:
        # NA_stat = TRUE: missing values are a real level, coloured R's "grey"
        has_na = bool(obs[stat_by].isna().any())
        st = as_ordered_categorical(obs[stat_by], show_na=True)
        stat_levels = [str(x) for x in st.cat.categories]
        real = stat_levels[:-1] if has_na else stat_levels
        stat_colors = discrete_palette(real, palette=stat_plot_palette, palcolor=stat_palcolor)
        if has_na:
            stat_colors["NA"] = "#BEBEBE"
        stat_values = pd.Series(np.asarray(st).astype(str), index=adata.obs_names).loc[frame.index]

    th = theme or theme_scp(aspect_ratio=aspect_ratio, legend_position=legend_position)
    keys = [f"{s or ''}:{g}" for g in groups for s in split_levels]
    legend = LegendColumn(position=legend_position)
    pg = PanelGrid(
        len(keys), panel_size=panel_size, nrow=nrow, ncol=ncol, byrow=byrow,
        keys=keys, theme=th, legend=legend,
    )

    seen_legend: set[str] = set()
    extra_legends: dict[str, list] = {}
    for g in groups:
        levels = group_levels[g]
        colors = discrete_palette(levels, palette=palette, palcolor=palcolor)

        for s in split_levels:
            panel = pg.axes[f"{s or ''}:{g}"]
            panel.set_xlim(*xlim)
            panel.set_ylim(*ylim)
            panel.set_xlabel(xlab if xlab is not None else f"{key}{dims[0]}")
            panel.set_ylabel(ylab if ylab is not None else f"{key}{dims[1]}")
            dat = frame.copy()
            if s is not None:
                dat.loc[dat["__split__"] != s, g] = np.nan  # mask, do not drop

            is_bg = dat[g].isna().to_numpy()
            order = np.concatenate([np.flatnonzero(is_bg), rng.permutation(np.flatnonzero(~is_bg))])
            dat = dat.iloc[order]
            present = [lv for lv in levels if (dat[g] == lv).any()]

            if add_mark:
                layers.draw_marks(panel, dat[~dat[g].isna()], g, present, colors, kind=mark_type,
                                  expand=mark_expand, alpha=mark_alpha, linetype=mark_linetype)
            if graph_mat is not None:
                layers.draw_graph(panel, graph_mat, _positions(adata, dat.index),
                                  dat["x"].to_numpy(dtype=float), dat["y"].to_numpy(dtype=float),
                                  edge_size=edge_size, alpha=edge_alpha, color=edge_color)
            if add_density:
                layers.draw_density(panel, dat["x"].to_numpy(dtype=float),
                                    dat["y"].to_numpy(dtype=float), (*xlim, *ylim),
                                    filled=density_filled, color=density_color,
                                    palette=density_filled_palette,
                                    palcolor=density_filled_palcolor)

            if hex:
                layers.draw_hex_groups(panel, dat, g, present, colors, bg_color, xlim=xlim,
                                       ylim=ylim, bins=hex_bins, binwidth=hex_binwidth,
                                       linewidth=hex_linewidth, count_alpha=hex_count)
            else:
                cvec = np.where(
                    dat[g].isna().to_numpy(),
                    bg_color,
                    [colors.get(str(v), bg_color) for v in dat[g]],
                )
                panel.scatter(
                    dat["x"], dat["y"], s=size, c=cvec, alpha=pt_alpha,
                    linewidths=0, rasterized=raster, zorder=layers.Z_POINTS,
                )

            if cells_highlight is not None and not hex:
                hl = (
                    dat.index[~dat[g].isna()]
                    if cells_highlight is True
                    else dat.index.intersection(pd.Index(cells_highlight))
                )
                sub = dat.loc[hl]
                if len(sub):
                    panel.scatter(sub["x"], sub["y"], s=_mm_area(sizes_highlight + stroke_highlight),
                                  c=cols_highlight, alpha=alpha_highlight, linewidths=0,
                                  rasterized=raster, zorder=layers.Z_HIGHLIGHT)
                    panel.scatter(sub["x"], sub["y"], s=_mm_area(sizes_highlight),
                                  c=[colors.get(str(v), bg_color) for v in sub[g]],
                                  alpha=alpha_highlight, linewidths=0, rasterized=raster,
                                  zorder=layers.Z_HIGHLIGHT + 0.01)

            centers = (dat.dropna(subset=[g]).groupby(g, observed=True)[["x", "y"]].median()
                       if present else pd.DataFrame(columns=["x", "y"]))
            centers.index = centers.index.astype(str)
            centers = centers.reindex([lv for lv in present if lv in centers.index])

            if stat_by is not None and stat_values is not None:
                in_split = dat.index[~dat[g].isna()]
                table = layers.stat_table(dat.loc[in_split, g].astype(str),
                                          stat_values.loc[in_split], present, stat_levels,
                                          stat_type=stat_type)
                used = [lv for lv in stat_levels if (stat_values.loc[in_split] == lv).any()]
                table = table.loc[used]
                x0, x1, y0, y1 = _limits_of(dat)
                layers.draw_stat_insets(panel, centers, table, stat_colors,
                                        plot_type=stat_plot_type, position=stat_plot_position,
                                        size=stat_plot_size, x_range=x1 - x0, y_range=y1 - y0,
                                        alpha=stat_plot_alpha, label=stat_plot_label,
                                        label_size=stat_plot_label_size, stat_type=stat_type)
                if "stat" not in extra_legends:
                    extra_legends["stat"] = [stat_by, [
                        Patch(facecolor=stat_colors[lv], edgecolor="black", label=lv)
                        for lv in used]]
            if curves is not None and lineages:
                handles = layers.draw_lineages(
                    panel, curves, list(lineages), palette=lineages_palette,
                    palcolor=lineages_palcolor, linewidth=lineages_linewidth,
                    line_bg=lineages_line_bg, line_bg_stroke=lineages_line_bg_stroke,
                    arrow=lineages_arrow, whiskers=lineages_whiskers,
                    whiskers_linewidth=lineages_whiskers_linewidth,
                    whiskers_alpha=lineages_whiskers_alpha)
                extra_legends.setdefault("lineages", ["Lineages", handles])
            if paga:
                pkey, plev, pcol = _draw_paga(
                    panel, adata, dat, paga=paga, palette=palette, palcolor=palcolor,
                    type_=paga_type, node_size=paga_node_size,
                    edge_threshold=paga_edge_threshold, edge_size=paga_edge_size,
                    edge_color=paga_edge_color, edge_alpha=paga_edge_alpha,
                    transition_threshold=paga_transition_threshold,
                    transition_size=paga_transition_size,
                    transition_color=paga_transition_color,
                    transition_alpha=paga_transition_alpha,
                    show_transition=paga_show_transition)
                # its legend would only repeat the cells' own when the groups agree
                if pkey != g and "paga" not in extra_legends:
                    extra_legends["paga"] = [pkey, [
                        Line2D([], [], marker="o", linestyle="none", markersize=6,
                               markerfacecolor=pcol[lv], markeredgecolor="black", label=lv)
                        for lv in plev]]
            if velocity:
                V = layers.velocity_embedding(adata, reduction, velocity)[
                    np.ix_(_positions(adata, dat.index), [xi, yi])]
                cell_cols = [colors.get(str(v), bg_color) for v in dat[g]]
                _f, mappable = layers.draw_velocity(
                    panel, dat[["x", "y"]].to_numpy(dtype=float), V,
                    plot_type=velocity_plot_type, cell_colors=cell_cols,
                    arrow_color=velocity_arrow_color, arrow_angle=velocity_arrow_angle,
                    density=velocity_density, smooth=velocity_smooth,
                    n_neighbors=velocity_n_neighbors or math.ceil(adata.n_obs / 50),
                    scale=velocity_scale, min_mass=velocity_min_mass,
                    cutoff_perc=velocity_cutoff_perc, streamline_width=streamline_width,
                    streamline_alpha=streamline_alpha, streamline_color=streamline_color,
                    streamline_palette=streamline_palette,
                    streamline_palcolor=streamline_palcolor,
                    streamline_bg_color=streamline_bg_color,
                    streamline_bg_stroke=streamline_bg_stroke, rng=rng)
                if mappable is not None and legend_position != "none":
                    cb = pg.colorbar(mappable, ax=panel)
                    cb.outline.set_edgecolor("black")
                    cb.set_label("Velocity", fontsize=th.size("legend.title"))

            if label and len(centers):
                txt = (list(centers.index) if label_insitu
                       else [str(i) for i in range(1, len(centers) + 1)])
                _draw_labels(panel, centers[["x", "y"]].to_numpy(dtype=float), txt,
                             size=label_size, fg=label_fg, bg=label_bg, bg_r=label_bg_r,
                             repel=label_repel, repulsion=label_repulsion,
                             point_size=label_point_size, point_color=label_point_color,
                             segment_color=label_segment_color)

            # overlays may autoscale; the global limits are the contract
            panel.set_xlim(*xlim)
            panel.set_ylim(*ylim)
            # title and subtitle are separate slots -- ggplot stacks them, and
            # so must we, or show_stat silently eats a user-supplied title.
            if title:
                panel.set_title(title, fontsize=th.size("plot.title"), loc="center")
            if show_stat or subtitle is not None:
                n_shown = int((~dat[g].isna()).sum())
                sub_t = subtitle if subtitle is not None else f"{s or ''} nCells:{n_shown}"
                panel.set_title(sub_t, fontsize=th.size("plot.subtitle"), loc="left")

            if g not in seen_legend and legend_position != "none":
                seen_legend.add(g)
                counts = dat[g].value_counts()
                handles = []
                for i, lv in enumerate(present, start=1):
                    lbl = str(lv)
                    if show_stat:
                        lbl = f"{lbl}({int(counts.get(lv, 0))})"
                    if label and not label_insitu:
                        lbl = f"{i}: {lbl}"
                    handles.append(
                        Line2D([], [], marker="o", linestyle="none", markersize=6,
                               markerfacecolor=colors[str(lv)], markeredgecolor="none", label=lbl)
                    )
                legend.add(g, handles)

    if legend_position != "none":
        for title_, handles in extra_legends.values():
            legend.add(title_, handles)
    if ax is not None:
        warnings.warn("`ax=` is ignored when more than one panel is produced.", stacklevel=2)
    return pg.finish()


def feature_dim_plot(
    adata,
    features: str | Sequence[str] | dict[str, Sequence[str]],
    *,
    reduction: str | None = None,
    dims: tuple[int, int] = (1, 2),
    split_by: str | None = None,
    cells: Sequence[str] | None = None,
    layer: str | None = None,
    use_raw: bool = False,
    # scaling -- see porting brief §2.4
    palette: str = "Spectral",
    palcolor: Sequence[str] | None = None,
    bg_cutoff: float | None = 0.0,
    bg_color: str = NA_COLOR_DEFAULT,
    keep_scale: Literal["feature", "all"] | None = "feature",
    lower_quantile: float = 0.0,
    upper_quantile: float = 0.99,
    lower_cutoff: float | None = None,
    upper_cutoff: float | None = None,
    # multi-feature blending
    compare_features: bool = False,
    color_blend_mode: BlendMode = "blend",
    calculate_coexp: bool = False,
    # shared with cell_dim_plot
    pt_size: float | None = None,
    pt_alpha: float = 1.0,
    raster: bool | None = None,
    raster_dpi: tuple[int, int] = (512, 512),
    hex: bool = False,
    hex_linewidth: float = 0.5,
    hex_color: str = _GREY90,
    hex_bins: int = 50,
    hex_binwidth: float | tuple[float, float] | None = None,
    label: bool = False,
    label_size: float = 4.0,
    label_fg: str = "white",
    label_bg: str = "black",
    label_bg_r: float = 0.1,
    label_insitu: bool = False,
    label_repel: bool = False,
    label_repulsion: float = 20.0,
    label_point_size: float = 1.0,
    label_point_color: str = "black",
    label_segment_color: str = "black",
    cells_highlight: Sequence[str] | bool | None = None,
    cols_highlight: str = "black",
    sizes_highlight: float = 1.0,
    alpha_highlight: float = 1.0,
    stroke_highlight: float = 0.5,
    add_density: bool = False,
    density_color: str = NA_COLOR_DEFAULT,
    density_filled: bool = False,
    density_filled_palette: str = "Greys",
    density_filled_palcolor: Sequence[str] | None = None,
    graph: str | None = None,
    edge_size: tuple[float, float] = (0.05, 0.5),
    edge_alpha: float = 0.1,
    edge_color: str = _GREY40,
    lineages: Sequence[str] | None = None,
    lineages_trim: tuple[float, float] = (0.01, 0.99),
    lineages_span: float = 0.75,
    lineages_palette: str = "Dark2",
    lineages_palcolor: Sequence[str] | None = None,
    lineages_arrow: float | None = 0.1,
    lineages_linewidth: float = 1.0,
    lineages_line_bg: str = "white",
    lineages_line_bg_stroke: float = 0.5,
    lineages_whiskers: bool = False,
    lineages_whiskers_linewidth: float = 0.5,
    lineages_whiskers_alpha: float = 0.5,
    theme: Theme | None = None,
    aspect_ratio: float | None = 1.0,
    panel_size: tuple[float, float] = (2.6, 2.6),
    show_stat: bool | None = None,
    title: str | None = None,
    subtitle: str | Sequence[str] | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    legend_position: Literal["none", "left", "right", "bottom", "top"] = "right",
    nrow: int | None = None,
    ncol: int | None = None,
    byrow: bool = True,
    force: bool = False,
    seed: int = 11,
) -> Figure:
    """Color cells on a 2-D embedding by continuous feature values.

    Port of ``FeatureDimPlot`` (``SCP-plot.R:2042``) by Hao Zhang,
    https://github.com/zhanghao-njmu/SCP.  Implementation notes live in
    ``docs/porting_briefs/dim_plots.md`` §2.  The three things that define how
    these figures look:

    * **draw order** is ascending value with NaN first, so the highest
      expressing cells land on top;
    * ``bg_cutoff=0`` NaN-masks every value ``<= 0`` *before* plotting, so
      zero-expression cells render in ``bg_color``.  Signed scores need
      ``bg_cutoff=None``;
    * the color limits are **winsorized** at ``upper_quantile=0.99`` — values
      above are saturated, never hidden — and ``keep_scale`` controls whether
      that range is pooled per feature, across everything, or per panel.

    ``compare_features=True`` with more than one feature switches to the
    per-cell color-blending path (``scp.colors.blendcolors``), which produces
    one colorbar per feature rather than a single scale.

    ``calculate_coexp=True`` appends a ``CoExp`` panel: the geometric mean of
    the *gene* features (:func:`scp.fetch.calculate_coexp`); obs and
    embedding features are dropped from it with R's warning.  Labels mark
    the median position of the cells in the feature's 95th-99th percentile
    band.  ``add_density``, ``graph``, ``lineages`` and ``hex`` behave as in
    :func:`cell_dim_plot`; the hexagons are the per-hexagon **mean** value
    (``stat_summary_hex``) over flat background hexagons, and the blend path
    has none, as in R.

    Deliberate divergences: ``keep_scale="all"`` keeps the ``is.finite``
    filter R's upper quantile drops (an R bug); ``hex`` takes precedence over
    ``raster`` (see :func:`cell_dim_plot`).
    """
    rng = np.random.default_rng(seed)
    cells_highlight = _check_highlight(adata, cells_highlight)
    graph_mat = _resolve_graph(adata, graph)

    # A named mapping flattens to a flat list whose names become subtitles.
    subtitles: dict[str, str] = {}
    if isinstance(features, dict):
        flat: list[str] = []
        for group, members in features.items():
            members = [members] if isinstance(members, str) else list(members)
            flat.extend(members)
            subtitles.update({m: group for m in members})
        feats = flat
    else:
        feats = [features] if isinstance(features, str) else list(features)

    # R guards at >50 features here, despite the roxygen text saying 100.
    if len(feats) > 50 and not force:
        raise ValueError(
            f"{len(feats)} features will produce an unreadable figure. "
            "Pass force=True to proceed anyway."
        )

    reduction = reduction or default_reduction(adata)
    key = reduction_key(reduction)
    emb = np.asarray(adata.obsm[reduction])
    xi, yi = dims[0] - 1, dims[1] - 1

    values = fetch_data(adata, feats, layer=layer, use_raw=use_raw)
    resolved = [f for f in feats if f in values.columns]
    if not resolved:
        raise ValueError("none of `features` could be resolved")

    frame = pd.DataFrame(
        {"x": emb[:, xi], "y": emb[:, yi]}, index=pd.Index(adata.obs_names)
    )
    for f in resolved:
        frame[f] = pd.to_numeric(values[f], errors="coerce").to_numpy()

    if calculate_coexp:
        var_names = set(map(str, adata.raw.var_names if use_raw else adata.var_names))
        genes = [f for f in resolved if f in var_names]
        others = [f for f in resolved if f not in var_names]
        if genes:
            if others:
                warnings.warn(f"{','.join(others)} is not used when calculating co-expression",
                              stacklevel=2)
            frame["CoExp"] = _coexp(adata, genes, layer=layer, use_raw=use_raw).to_numpy()
            resolved = [*resolved, "CoExp"]
        else:
            warnings.warn("calculate_coexp needs gene features; none were given, so no "
                          "CoExp panel is drawn.", stacklevel=2)

    if split_by is None:
        split_levels: list[str | None] = [None]
        frame["__split__"] = ""
    else:
        sp_cat = as_ordered_categorical(adata.obs[split_by])
        frame["__split__"] = sp_cat.to_numpy()
        split_levels = list(sp_cat.cat.categories)

    # Global limits, from every cell, so split panels stay comparable.
    xlim = (float(np.nanmin(emb[:, xi])), float(np.nanmax(emb[:, xi])))
    ylim = (float(np.nanmin(emb[:, yi])), float(np.nanmax(emb[:, yi])))

    # bg_cutoff is applied globally and destructively BEFORE plotting: every
    # value <= it becomes NaN and renders in bg_color.  The comparison is <=,
    # so exact zeros are background.  Signed scores need bg_cutoff=None.
    if bg_cutoff is not None:
        for f in resolved:
            frame.loc[frame[f] <= bg_cutoff, f] = np.nan
    # keep_scale pools over R's dat_exp: every cell, before `cells` subsets.
    exp_all = frame[resolved].copy()

    if cells is not None:
        frame = frame.loc[frame.index.intersection(pd.Index(cells))]

    n_cells = len(frame)
    size = pt_size if pt_size is not None else _auto_pt_size(n_cells)
    if raster is None:
        raster = n_cells > 100_000
    if show_stat is None:
        show_stat = not (theme.blank if theme else False)

    curves = None
    if lineages:
        # FeatureDimPlot calls LineagePlot without `cells`: curves use every cell
        curves = layers.lineage_curves(adata, list(lineages), emb[:, [xi, yi]],
                                 np.arange(adata.n_obs), trim=lineages_trim, span=lineages_span)

    blend_path = bool(compare_features) and len(resolved) > 1

    th = theme or theme_scp(aspect_ratio=aspect_ratio, legend_position=legend_position)
    panel_feats = ["|".join(resolved)] if blend_path else resolved
    keys = [f"{s or ''}:{f}" for s in split_levels for f in panel_feats]
    legend = LegendColumn(position=legend_position)
    pg = PanelGrid(
        len(keys), panel_size=panel_size,
        nrow=nrow if nrow is not None else (None if ncol else len(split_levels)),
        ncol=ncol if ncol is not None else (None if nrow else len(panel_feats)),
        byrow=byrow, keys=keys, theme=th, legend=legend,
    )

    def _limits(f: str, panel: pd.DataFrame) -> tuple[float, float]:
        """Winsorized colour limits; `keep_scale` chooses the pooling scope."""
        if keep_scale == "all":
            pool = exp_all.to_numpy().ravel()
        elif keep_scale == "feature":
            pool = exp_all[f].to_numpy()
        else:
            pool = panel[f].to_numpy()
        pool = pool[np.isfinite(pool)]
        if pool.size == 0:
            return (0.0, 1.0)
        lo = lower_cutoff if lower_cutoff is not None else float(np.quantile(pool, lower_quantile))
        hi = upper_cutoff if upper_cutoff is not None else float(np.quantile(pool, upper_quantile))
        return (lo, hi + 0.001)  # +0.001 so the range can never be degenerate

    def _label_xy(d: pd.DataFrame, col: str) -> np.ndarray | None:
        """Median position of the cells in the 95th-99th percentile band of ``col``."""
        v = d[col].to_numpy(dtype=float)
        fin = np.isfinite(v)
        if not fin.any():
            return None
        q95, q99 = np.quantile(v[fin], [0.95, 0.99])
        m = fin & (v >= q95) & (v <= q99)
        if not m.any():
            return None
        return np.array([np.median(d["x"].to_numpy()[m]), np.median(d["y"].to_numpy()[m])])

    ramp = continuous_palette(palette=palette if not blend_path else "Spectral",
                              palcolor=None if blend_path else palcolor, n=100)
    cmap = LinearSegmentedColormap.from_list("scp", ramp, N=256).with_extremes(bad=bg_color)

    def _labels(target: Axes, xy: np.ndarray, txt: Sequence[str], fg) -> None:
        _draw_labels(target, xy, txt, size=label_size, fg=fg, bg=label_bg, bg_r=label_bg_r,
                     repel=label_repel, repulsion=label_repulsion,
                     point_size=label_point_size, point_color=label_point_color,
                     segment_color=label_segment_color)

    seen_cbar: set[str] = set()
    lineage_handles = None
    for s in split_levels:
        # Unlike cell_dim_plot this genuinely partitions -- a split panel holds
        # only its own cells, with no grey "other split" points.
        part = frame if s is None else frame[frame["__split__"] == s]
        for f in panel_feats:
            ax_p = pg.axes[f"{s or ''}:{f}"]
            ax_p.set_xlim(*xlim)
            ax_p.set_ylim(*ylim)
            ax_p.set_xlabel(xlab if xlab is not None else f"{key}{dims[0]}")
            ax_p.set_ylabel(ylab if ylab is not None else f"{key}{dims[1]}")

            def _under(d: pd.DataFrame, _ax=ax_p) -> None:
                """Graph edges and density: drawn under the points, over this panel's cells."""
                if graph_mat is not None:
                    layers.draw_graph(_ax, graph_mat, _positions(adata, d.index),
                                      d["x"].to_numpy(dtype=float), d["y"].to_numpy(dtype=float),
                                      edge_size=edge_size, alpha=edge_alpha, color=edge_color)
                if add_density:
                    layers.draw_density(_ax, d["x"].to_numpy(dtype=float),
                                        d["y"].to_numpy(dtype=float), (*xlim, *ylim),
                                        filled=density_filled, color=density_color,
                                        palette=density_filled_palette,
                                        palcolor=density_filled_palcolor)

            if blend_path:
                hues = discrete_palette(resolved, palette="Set1", palcolor=palcolor)
                per_feature: list[list] = []
                for feat in resolved:
                    lo, hi = _limits(feat, part)
                    stops = [adjcolors(hues[feat], 0.1), hues[feat]]
                    sub_cmap = LinearSegmentedColormap.from_list(f"b{feat}", stops, N=256)
                    norm = Normalize(vmin=lo, vmax=hi, clip=True)
                    vals = part[feat].to_numpy()
                    per_feature.append(
                        [None if not np.isfinite(v) else to_hex(sub_cmap(norm(v))) for v in vals]
                    )
                blended = [
                    blendcolors([c[i] for c in per_feature], mode=color_blend_mode)
                    for i in range(len(part))
                ]
                # Sort brightest-first so the most co-expressing points land on top;
                # cells with no signal on any feature sort to the bottom.
                lum = np.array([
                    np.nan if c is None else float(sum(to_rgb(c))) for c in blended
                ])
                order = np.concatenate([
                    np.flatnonzero(np.isnan(lum)),
                    np.flatnonzero(~np.isnan(lum))[np.argsort(-lum[~np.isnan(lum)], kind="stable")],
                ])
                cvec = [bg_color if blended[i] is None else blended[i] for i in order]
                d = part.iloc[order]
                _under(d)
                ax_p.scatter(d["x"], d["y"], s=size, c=cvec, alpha=pt_alpha,
                             linewidths=0, rasterized=raster, zorder=layers.Z_POINTS)
                if cells_highlight is not None:
                    has = np.array([c != bg_color for c in cvec])
                    hl = (d.index[has] if cells_highlight is True
                          else d.index.intersection(pd.Index(cells_highlight)))
                    pos = d.index.get_indexer(hl)
                    if len(pos):
                        ax_p.scatter(d["x"].iloc[pos], d["y"].iloc[pos],
                                     s=_mm_area(sizes_highlight + stroke_highlight),
                                     c=cols_highlight, alpha=alpha_highlight, linewidths=0,
                                     rasterized=raster, zorder=layers.Z_HIGHLIGHT)
                        ax_p.scatter(d["x"].iloc[pos], d["y"].iloc[pos],
                                     s=_mm_area(sizes_highlight), c=[cvec[i] for i in pos],
                                     alpha=alpha_highlight, linewidths=0, rasterized=raster,
                                     zorder=layers.Z_HIGHLIGHT + 0.01)
                for feat in resolved:
                    if feat in seen_cbar or legend_position == "none":
                        continue
                    seen_cbar.add(feat)
                    lo, hi = _limits(feat, part)
                    stops = [adjcolors(hues[feat], 0.1), hues[feat]]
                    sm = ScalarMappable(norm=Normalize(lo, hi),
                                        cmap=LinearSegmentedColormap.from_list(f"c{feat}", stops))
                    cb = pg.colorbar(sm)
                    cb.outline.set_edgecolor("black")
                    cb.set_label(feat, fontsize=th.size("legend.title"))
                if label:
                    pts, names = [], []
                    for feat in resolved:
                        xy = _label_xy(d, feat)
                        if xy is not None:
                            pts.append(xy)
                            names.append(feat)
                    if pts:
                        txt = names if label_insitu else [str(i) for i in range(1, len(names) + 1)]
                        # in situ: white text; numbered: the text takes the feature hue
                        fg = label_fg if label_insitu else [hues[n] for n in names]
                        _labels(ax_p, np.vstack(pts), txt, fg)
                        if not label_insitu and legend_position != "none" and s == split_levels[0]:
                            legend.add("Label", [
                                Line2D([], [], marker="o", linestyle="none", markersize=6,
                                       markerfacecolor=hues[n], markeredgecolor="none",
                                       label=f"{i}: {n}")
                                for i, n in enumerate(names, start=1)])
            else:
                lo, hi = _limits(f, part)
                vals = part[f].to_numpy(dtype=float)
                # NaN first, then ascending, so the highest expressers draw last.
                order = np.concatenate([
                    np.flatnonzero(np.isnan(vals)),
                    np.flatnonzero(~np.isnan(vals))[np.argsort(vals[~np.isnan(vals)], kind="stable")],
                ])
                d = part.iloc[order].copy()
                # +-Inf are pulled in to the finite extremes before scaling
                fv = d[f].to_numpy(dtype=float)
                fin = np.isfinite(fv)
                if fin.any() and (~fin & ~np.isnan(fv)).any():
                    fv = np.where(fv == np.inf, fv[fin].max(), np.where(fv == -np.inf, fv[fin].min(), fv))
                v = np.clip(fv, lo, hi)  # winsorize, never drop
                d["__value__"] = v
                _under(d)
                norm = Normalize(lo, hi)
                if hex:
                    layers.draw_hex_values(ax_p, d, "__value__", cmap=cmap, norm=norm,
                                           bg_color=bg_color, edge_color=hex_color, xlim=xlim,
                                           ylim=ylim, bins=hex_bins, binwidth=hex_binwidth,
                                           linewidth=hex_linewidth)
                else:
                    # Two layers, as SCP does: background cells first so they can
                    # never overplot a cell that actually carries signal.
                    bg = np.isnan(v)
                    if bg.any():
                        ax_p.scatter(d["x"][bg], d["y"][bg], s=size, c=bg_color,
                                     alpha=pt_alpha, linewidths=0, rasterized=raster,
                                     zorder=layers.Z_POINTS)
                    if (~bg).any():
                        ax_p.scatter(d["x"][~bg], d["y"][~bg], s=size, c=v[~bg],
                                     cmap=cmap, norm=norm, alpha=pt_alpha,
                                     linewidths=0, rasterized=raster, zorder=layers.Z_POINTS)
                if cells_highlight is not None and not hex:
                    hl = (d.index[np.isfinite(v)] if cells_highlight is True
                          else d.index.intersection(pd.Index(cells_highlight)))
                    pos = d.index.get_indexer(hl)
                    if len(pos):
                        ax_p.scatter(d["x"].iloc[pos], d["y"].iloc[pos],
                                     s=_mm_area(sizes_highlight + stroke_highlight),
                                     c=cols_highlight, alpha=alpha_highlight, linewidths=0,
                                     rasterized=raster, zorder=layers.Z_HIGHLIGHT)
                        ax_p.scatter(d["x"].iloc[pos], d["y"].iloc[pos],
                                     s=_mm_area(sizes_highlight), c=v[pos], cmap=cmap, norm=norm,
                                     plotnonfinite=True, alpha=alpha_highlight, linewidths=0,
                                     rasterized=raster, zorder=layers.Z_HIGHLIGHT + 0.01)
                cbar_key = f if keep_scale == "feature" else ("__all__" if keep_scale == "all" else f"{s}:{f}")
                if cbar_key not in seen_cbar and legend_position != "none":
                    seen_cbar.add(cbar_key)
                    sm = ScalarMappable(norm=Normalize(lo, hi), cmap=cmap)
                    # one scale for every panel is a figure-level guide; any
                    # other scope is drawn beside the panel that introduces it
                    cb = pg.colorbar(sm) if keep_scale == "all" else pg.colorbar(sm, ax=ax_p)
                    cb.outline.set_edgecolor("black")
                if label:
                    xy = _label_xy(d, "__value__")
                    if xy is not None:
                        _labels(ax_p, xy[None, :], [f], label_fg)

            if curves is not None and lineages:
                lineage_handles = layers.draw_lineages(
                    ax_p, curves, list(lineages), palette=lineages_palette,
                    palcolor=lineages_palcolor, linewidth=lineages_linewidth,
                    line_bg=lineages_line_bg, line_bg_stroke=lineages_line_bg_stroke,
                    arrow=lineages_arrow, whiskers=lineages_whiskers,
                    whiskers_linewidth=lineages_whiskers_linewidth,
                    whiskers_alpha=lineages_whiskers_alpha)

            ax_p.set_xlim(*xlim)
            ax_p.set_ylim(*ylim)
            strip = "|".join(resolved) if blend_path else f
            head = title if title else strip
            if show_stat:
                # R reports the POSITIVE cells and their share, not the panel
                # total: cells at or below bg_cutoff are background and excluded.
                cols = resolved if blend_path else [f]
                n_pos = int(part[cols].notna().any(axis=1).sum())
                pct = (100.0 * n_pos / len(part)) if len(part) else 0.0
                stat = f"nPos:{n_pos}, {pct:.1f}%"
                if s is not None:
                    stat = f"{s} | {stat}"
                if not blend_path and f in subtitles:
                    stat = f"{subtitles[f]} | {stat}"
                # Stacked in one title: ggplot puts the subtitle above the strip,
                # and two matplotlib titles at different loc share a line.
                ax_p.set_title(f"{stat}\n{head}", fontsize=th.size("plot.subtitle"))
            else:
                ax_p.set_title(head, fontsize=th.size("plot.title"))

    if lineage_handles and legend_position != "none":
        legend.add("Lineages", lineage_handles)
    _ = rng  # seed is accepted for signature parity; this path has no shuffle
    return pg.finish()


def _plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("3-D plots need `pip install scp-plot[interactive]`") from e
    return go


def _embedding_3d(adata, reduction: str | None, dims):
    reduction = reduction or default_reduction(adata)
    emb = np.asarray(adata.obsm[reduction], dtype=float)
    if emb.shape[1] < 3:
        raise ValueError(f"{reduction!r} has {emb.shape[1]} dimensions; 3 are needed")
    return reduction, emb, tuple(d - 1 for d in dims)


def _scene_3d(emb: np.ndarray, cols, dims, key: str, axis_labs) -> dict:
    """R's scene: each axis titled and pinned to the data's range, a cube aspect."""
    labs = (list(axis_labs) if axis_labs is not None and len(axis_labs) == 3
            else [f"{key}{d}" for d in dims])

    def _axis(c: int, lab: str) -> dict:
        return dict(title=lab, range=[float(np.nanmin(emb[:, c])), float(np.nanmax(emb[:, c]))])

    return dict(xaxis=_axis(cols[0], labs[0]), yaxis=_axis(cols[1], labs[1]),
                zaxis=_axis(cols[2], labs[2]), aspectratio=dict(x=1, y=1, z=1))


def _highlight_3d(adata, cells_highlight, default: np.ndarray) -> np.ndarray | None:
    """Row positions of the highlighted cells, in R's order.

    ``True`` means ``default`` (positions, object order); a list is validated
    as R does -- none found raises, some missing warns -- and keeps its own
    order, each cell once (R's ``intersect``).
    """
    if cells_highlight is None or cells_highlight is False:
        return None
    if cells_highlight is True:
        return default
    req = _check_highlight(adata, cells_highlight)
    assert isinstance(req, list)
    return _positions(adata, pd.Index(list(dict.fromkeys(req))))


def _highlight_trace(go, emb, cols, pos: np.ndarray, *, size: float, color: str, symbol: str):
    return go.Scatter3d(
        x=emb[pos, cols[0]], y=emb[pos, cols[1]], z=emb[pos, cols[2]], mode="markers",
        name="highlight", showlegend=True,
        marker=dict(size=size, color=to_hex(color).upper(), symbol=symbol),
    )


def lineage_curve_3d(emb: np.ndarray, pseudotime: np.ndarray, span: float) -> pd.DataFrame:
    """``CellDimPlot3D``'s lineage line: the 3-D analogue of ``LineagePlot``'s curve.

    Every cell on the lineage (no trim, unlike the 2-D curve) is ordered by
    pseudotime and each of the three axes is LOESS-smoothed against it
    independently (``degree = 2``, ``span``; ``SCP-plot.R:2981-2987``).  Fitted
    points outside the range of *all* cells on any axis are dropped, then
    missing and duplicate rows (``SCP-plot.R:2988-2991``).  Returns ``x``,
    ``y``, ``z`` in pseudotime order.
    """
    pt = np.asarray(pseudotime, dtype=float)
    idx = np.flatnonzero(np.isfinite(pt))
    idx = idx[np.argsort(pt[idx], kind="stable")]
    if len(idx) < 3:
        return pd.DataFrame(columns=["x", "y", "z"])
    fit = layers.smooth_on_pseudotime(pt[idx], emb[idx], span,
                                      have_loess=layers.loess_available())
    df = pd.DataFrame(fit, columns=["x", "y", "z"])
    for c, k in enumerate("xyz"):
        lo, hi = np.nanmin(emb[:, c]), np.nanmax(emb[:, c])
        df = df[(df[k] <= hi) & (df[k] >= lo)]
    return df.dropna().drop_duplicates().reset_index(drop=True)


def cell_dim_plot_3d(
    adata,
    group_by: str,
    *,
    reduction: str | None = None,
    dims: tuple[int, int, int] = (1, 2, 3),
    axis_labs: Sequence[str] | None = None,
    palette: str = "Paired",
    palcolor: Sequence[str] | dict[str, str] | None = None,
    bg_color: str = NA_COLOR_DEFAULT,
    pt_size: float = 1.5,
    cells_highlight: Sequence[str] | bool | None = None,
    cols_highlight: str = "black",
    shape_highlight: str = "circle-open",
    sizes_highlight: float = 2.0,
    lineages: Sequence[str] | None = None,
    lineages_palette: str = "Dark2",
    span: float = 0.75,
    show_na: bool = True,
    width: int = 700,
    height: int = 600,
    force: bool = False,
):
    """plotly ``Scatter3d`` version of :func:`cell_dim_plot`.

    Port of ``CellDimPlot3D`` (``SCP-plot.R:2843``) by Hao Zhang,
    https://github.com/zhanghao-njmu/SCP.  One trace per level, so plotly's
    own legend does the grouping and levels can be toggled.  Level order is
    the column's declared order, as everywhere else.

    As in R, cells with a missing ``group_by`` are always drawn, as a last
    ``"NA"`` trace coloured ``bg_color`` (``palette_scp(..., NA_color =
    bg_color, NA_keep = TRUE)``, ``SCP-plot.R:2919-2926``); the palette is
    spread over the real levels only.  ``show_na=False`` is a Python-only
    switch that drops them instead.  The title is R's ``"Total (nCells:N)"``
    over every cell, each scene axis is pinned to the data's range and the
    scene is a cube (``SCP-plot.R:3016-3031``); ``axis_labs`` (three
    strings) replaces the axis titles.

    * ``cells_highlight`` adds one ``"highlight"`` trace after the levels:
      markers of ``sizes_highlight`` in ``cols_highlight``, plotly symbol
      ``shape_highlight`` (an open circle by default, so the cell's own
      colour shows through).  ``True`` highlights every cell with a group;
      a list keeps its own order and is validated as in :func:`cell_dim_plot`.
    * ``lineages`` (``obs`` pseudotime columns) adds one ``mode="lines"``
      trace per lineage, 6 px wide in ``lineages_palette``: each axis
      LOESS-smoothed against the pseudotime over every cell on the lineage --
      no ``trim``, unlike the 2-D curve -- see :func:`lineage_curve_3d`.
      Without ``skmisc`` a rolling mean stands in, with a warning.
    * more than 100 levels in ``group_by`` raises unless ``force=True`` (R
      asks interactively).

    ``save`` is dropped, as everywhere: the returned figure has ``write_html``.
    """
    go = _plotly()
    reduction, emb, cols = _embedding_3d(adata, reduction, dims)
    key = reduction_key(reduction)
    if group_by not in adata.obs.columns:
        raise KeyError(f"{group_by!r} is not in adata.obs")
    lineages = [] if lineages is None else [str(x) for x in lineages]
    missing = [lin for lin in lineages if lin not in adata.obs.columns]
    if missing:
        raise KeyError(f"lineages not in adata.obs: {missing}")

    cat = as_ordered_categorical(adata.obs[group_by])
    levels = [str(x) for x in cat.cat.categories]
    if len(levels) > 100 and not force:
        raise ValueError(
            f"`{group_by}` has {len(levels)} levels, which will produce an unreadable legend. "
            "Pass force=True to proceed anyway."
        )
    colors = discrete_palette(levels, palette=palette, palcolor=palcolor)
    na = np.asarray(cat.isna())
    values = np.asarray(cat).astype(str)
    if show_na and na.any():
        values[na] = "NA"
        levels = [*levels, "NA"]
        colors["NA"] = bg_color
    hl = _highlight_3d(adata, cells_highlight, np.flatnonzero(~na))

    fig = go.Figure()
    i, j, k = cols
    for lv in levels:
        m = (values == lv) & (~na if lv != "NA" else na)
        fig.add_trace(go.Scatter3d(
            x=emb[m, i], y=emb[m, j], z=emb[m, k], mode="markers", name=lv,
            marker=dict(size=pt_size, color=colors[lv]),
        ))
    if hl is not None:
        fig.add_trace(_highlight_trace(go, emb, cols, hl, size=sizes_highlight,
                                       color=cols_highlight, symbol=shape_highlight))
    if lineages:
        lcol = discrete_palette(lineages, palette=lineages_palette)
        e3 = emb[:, [i, j, k]]
        for lin in lineages:
            pt = pd.to_numeric(adata.obs[lin], errors="coerce").to_numpy(dtype=float)
            cv = lineage_curve_3d(e3, pt, span)
            fig.add_trace(go.Scatter3d(
                x=cv["x"].to_numpy(), y=cv["y"].to_numpy(), z=cv["z"].to_numpy(),
                mode="lines", name=lin, showlegend=True,
                line=dict(width=6, color=lcol[lin]),
            ))

    fig.update_layout(
        width=width, height=height, legend_title_text=group_by,
        title=dict(text=f"Total (nCells:{adata.n_obs})", font=dict(size=16, color="black"),
                   y=0.95),
        font=dict(size=12, color="black"), showlegend=True,
        legend=dict(itemsizing="constant", y=0.5, x=1, xanchor="left"),
        scene=_scene_3d(emb, cols, dims, key, axis_labs),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def _masked(v: np.ndarray, keep: np.ndarray) -> list:
    """``ifelse(keep, v, NA)`` as a plotly-ready list (``None`` is a gap)."""
    return [float(x) if k else None for x, k in zip(v, keep, strict=True)]


def feature_dim_plot_3d(
    adata,
    features: str | Sequence[str],
    *,
    reduction: str | None = None,
    dims: tuple[int, int, int] = (1, 2, 3),
    axis_labs: Sequence[str] | None = None,
    split_by: str | None = None,
    layer: str | None = None,
    calculate_coexp: bool = False,
    palette: str = "Spectral",
    palcolor: Sequence[str] | None = None,
    bg_cutoff: float | None = None,
    bg_color: str = NA_COLOR_DEFAULT,
    pt_size: float = 1.5,
    cells_highlight: Sequence[str] | bool | None = None,
    cols_highlight: str = "black",
    shape_highlight: str = "circle-open",
    sizes_highlight: float = 2.0,
    width: int = 700,
    height: int = 600,
    force: bool = False,
):
    """plotly ``Scatter3d`` version of :func:`feature_dim_plot`.

    Port of ``FeatureDimPlot3D`` (``SCP-plot.R:3066``) by Hao Zhang,
    https://github.com/zhanghao-njmu/SCP.

    **Deliberate divergence from R.**  ``FeatureDimPlot3D`` never calls
    ``palette_scp``: it falls back to plotly's default scale, so a 3-D feature
    plot does not match the 2-D Spectral colouring of the same data.  That is a
    defect rather than a decision, so this uses the same ramp as
    :func:`feature_dim_plot` and the two agree.

    **No masking by default.**  ``FeatureDimPlot3D`` has no ``bg_cutoff``: it
    passes every value to ``marker.color`` (``SCP-plot.R:3239-3242``), so a
    sparse gene keeps its zeros and a signed score its negatives.
    ``bg_cutoff`` (default ``None``) is a Python-only opt-in that NaN-masks
    values ``<= bg_cutoff``, the 2-D behaviour.

    ``calculate_coexp=True`` appends a ``CoExp`` trace, the geometric mean of
    the gene features, exactly as in :func:`feature_dim_plot`.  Several
    features become an ``updatemenus`` dropdown.  R keeps one ``All_cells``
    trace and swaps its marker; here each feature is its own trace and the
    dropdown toggles visibility -- the same values per feature, and each
    trace keeps its own colourbar.

    As in R: the title is ``"All_cells (nCells:N)"``, each scene axis is
    pinned to the data's range in a cube, ``axis_labs`` replaces the axis
    titles, and more than 50 features raise unless ``force=True`` (R asks).

    * ``cells_highlight`` adds a ``"highlight"`` trace after the features
      (``sizes_highlight``, ``cols_highlight``, plotly symbol
      ``shape_highlight``); ``True`` highlights every cell.
    * ``split_by`` adds a second dropdown, ``All.cells`` then one button per
      level, that keeps only that level's cells in every trace (the others
      become gaps) and retitles the plot ``"<level> (nCells:n)"``.

    **R bug, fixed.**  With ``split.by`` set, R's ``All.cells`` button reads
    the coordinate columns ``<key>1All.cells`` ... (``SCP-plot.R:3272-3279``),
    which only exist when ``split.by`` is left at its default (the all-cell
    columns are created as ``...All_cells``, ``:3208``), so pressing it
    empties the plot.  Here it restores every cell.  The split
    buttons also leave trace names and visibility alone, since renaming every
    feature trace to the split level would scramble the feature legend.
    """
    go = _plotly()
    feats = [features] if isinstance(features, str) else list(dict.fromkeys(features))
    reduction, emb, cols = _embedding_3d(adata, reduction, dims)
    key = reduction_key(reduction)

    values = fetch_data(adata, feats, layer=layer)
    feats = [f for f in feats if f in values.columns]
    if not feats:
        raise ValueError("none of `features` could be resolved")
    if calculate_coexp:
        genes = [f for f in feats if f in set(map(str, adata.var_names))]
        if genes:
            others = [f for f in feats if f not in genes]
            if others:
                warnings.warn(f"{','.join(others)} is not used when calculating co-expression",
                              stacklevel=2)
            values = values.copy()
            values["CoExp"] = _coexp(adata, genes, layer=layer).to_numpy()
            feats = [*feats, "CoExp"]
    if len(feats) > 50 and not force:
        raise ValueError(f"{len(feats)} features to plot (more than 50). "
                         "Pass force=True to proceed anyway.")
    ramp = continuous_palette(palette=palette, palcolor=palcolor, n=100)
    scale = [[n / (len(ramp) - 1), c] for n, c in enumerate(ramp)]
    hl = _highlight_3d(adata, cells_highlight, np.arange(adata.n_obs))

    fig = go.Figure()
    i, j, k = cols
    for n, f in enumerate(feats):
        v = pd.to_numeric(values[f], errors="coerce").to_numpy(dtype=float)
        if bg_cutoff is not None:
            v = np.where(v <= bg_cutoff, np.nan, v)
        fig.add_trace(go.Scatter3d(
            x=emb[:, i], y=emb[:, j], z=emb[:, k], mode="markers", name=f,
            visible=(n == 0),
            marker=dict(size=pt_size, color=v, colorscale=scale,
                        colorbar=dict(title=f), showscale=True),
        ))
    if hl is not None:
        fig.add_trace(_highlight_trace(go, emb, cols, hl, size=sizes_highlight,
                                       color=cols_highlight, symbol=shape_highlight))

    menus = []
    if split_by is not None:
        sp = as_ordered_categorical(adata.obs[split_by])
        spv = np.asarray(sp).astype(object)
        buttons: list[dict] = []
        for lv in ["All.cells", *[str(x) for x in sp.cat.categories]]:
            if lv == "All.cells":
                keep = np.ones(adata.n_obs, dtype=bool)
                if buttons:
                    continue  # a level named "All.cells" is skipped, as in R
            else:
                keep = np.array([str(x) == lv for x in spv]) & np.asarray(sp.notna())
            xyz = [[_masked(emb[:, c], keep)] * len(feats) for c in cols]
            if hl is not None:
                for d, c in enumerate(cols):
                    xyz[d].append(_masked(emb[hl, c], keep[hl]))
            buttons.append(dict(
                label=lv, method="update",
                args=[{"x": xyz[0], "y": xyz[1], "z": xyz[2]},
                      {"title": dict(text=f"{lv} (nCells:{int(keep.sum())})",
                                     font=dict(size=16, color="black"), y=0.95)}],
            ))
        menus.append(dict(buttons=buttons, y=0.67, direction="down", showactive=True))
    if len(feats) > 1:
        tail = [True] if hl is not None else []
        menus.append(dict(
            buttons=[dict(label=f, method="update",
                          args=[{"visible": [m == n for m in range(len(feats))] + tail}])
                     for n, f in enumerate(feats)],
            direction="down", showactive=True, y=0.33,
        ))
    fig.update_layout(
        width=width, height=height,
        title=dict(text=f"All_cells (nCells:{adata.n_obs})", font=dict(size=16, color="black"),
                   y=0.95),
        showlegend=True,
        legend=dict(itemsizing="constant", y=-0.2, x=0.5, xanchor="center"),
        scene=_scene_3d(emb, cols, dims, key, axis_labs),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    if menus:
        fig.update_layout(updatemenus=menus)
    return fig
