"""Trajectory, graph and velocity plots.

Ports of ``LineagePlot``, ``PAGAPlot``, ``GraphPlot``, ``segementsDf``,
``VelocityPlot``, ``compute_velocity_on_grid``, ``DynamicPlot`` and
``ProjectionPlot`` from SCP (Hao Zhang, NJMU, GPL-3;
https://github.com/zhanghao-njmu/SCP), ``R/SCP-plot.R``.

Most of this family is cheaper in Python than in R, because scanpy and scvelo
already persist what SCP had to reconstruct: ``adata.uns['paga']`` has the same
keys SCP reads, and ``adata.obsm['velocity_<basis>']`` is the velocity field.

The single-panel functions here (``graph_plot``, ``paga_plot``,
``lineage_plot``, ``velocity_plot``) are layers first: given ``ax=`` they draw
only their own marks into it -- no cells underneath -- so they can sit on top
of a ``cell_dim_plot`` panel, as SCP's ``return_layer = TRUE`` does.  Pass a
:class:`~scp.layout.LegendColumn` as ``legend=`` to have them register their
legend there.  Without ``ax=`` they allocate one themed panel through
:class:`~scp.layout.PanelGrid` and draw their legend beside it.

Spec: ``docs/porting_briefs/trajectory_enrichment.md``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ..fetch import (
    as_level_labels,
    as_ordered_categorical,
    default_reduction,
    fetch_data,
    reduction_key,
)
from ..io import read_paga
from ..layout import LegendColumn, PanelGrid
from ..palettes import discrete_palette, level_colors, uns_palcolor
from ..theme import halo, theme_scp
from . import _dim_layers as layers
from ._dim_geom import compute_velocity_on_grid, expand_range

__all__ = [
    "graph_plot",
    "paga_plot",
    "lineage_plot",
    "velocity_plot",
    "dynamic_plot",
    "projection_plot",
    "shorten_segments",
    "compute_velocity_on_grid",
]

#: ggplot's ``.pt``: one millimetre of ``size`` / ``linewidth`` in points.
_PT = 72.27 / 25.4

LegendPosition = Literal["none", "left", "right", "bottom", "top"]

#: the colour SCP gives cells whose group is missing (R's grey80)
_NA_GREY = "#CCCCCC"


def _panel(ax: Axes | None, legend: LegendColumn | None, legend_position: LegendPosition,
           panel_size: tuple[float, float] = (3.5, 3.5)):
    """The axes to draw into, the legend to register with, and the grid to finish.

    Given ``ax``, the caller owns the figure and its legend (``legend`` may be
    ``None``, in which case nothing is registered).  Without it, one square
    themed panel is allocated through :class:`PanelGrid`, which owns figure
    creation -- the R side's ``aspect.ratio = 1`` + ``theme_scp``.
    """
    if ax is not None:
        return ax, legend, None
    th = theme_scp(aspect_ratio=1, legend_position=legend_position)
    pg = PanelGrid(1, panel_size=panel_size, theme=th, legend=LegendColumn(position=legend_position))
    return pg.axes["0"], pg.legend, pg


def _finish(pg: PanelGrid | None) -> None:
    if pg is not None:
        pg.finish()


def _labels(ax: Axes, kwargs: dict, *, own: bool, xlab: str, ylab: str,
            title: str | None = None) -> None:
    """Axis titles and plot title: R's ``lab_layer``.

    ``xlab`` / ``ylab`` / ``title`` in ``kwargs`` always win (an empty one
    suppresses the default and leaves the axes as it is).  The defaults
    apply to a panel this function allocated; on a borrowed ``ax`` the axis
    titles are filled only where it has none yet, and the plot title never
    -- R's overlay layers (``return_layer = TRUE``) carry no title, and the
    host panel's own title or ``nCells`` subtitle must not be overprinted.
    """
    for k, default, get, set_ in (("xlab", xlab, ax.get_xlabel, ax.set_xlabel),
                                  ("ylab", ylab, ax.get_ylabel, ax.set_ylabel),
                                  ("title", title, ax.get_title, ax.set_title)):
        if k in kwargs:
            if kwargs[k]:
                set_(kwargs[k])
        elif default and (own or (k != "title" and not get())):
            set_(default)


def shorten_segments(
    x: np.ndarray, y: np.ndarray, xend: np.ndarray, yend: np.ndarray,
    shorten_start: float = 0.0, shorten_end: float = 0.0, offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of ``segementsDf`` (``SCP-plot.R:6666``) — pure geometry, ported verbatim.

    Ported from SCP by Hao Zhang, https://github.com/zhanghao-njmu/SCP
    (GPL-3).

    Shortening pulls segment ends back so arrowheads do not sit under node
    circles.  ``offset`` displaces the whole segment perpendicular to itself,
    which is how the forward and reverse transitions between the same pair of
    nodes avoid overlapping.
    """
    dx, dy = xend - x, yend - y
    dist = np.hypot(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        px, py = np.where(dist > 0, dx / dist, 0.0), np.where(dist > 0, dy / dist, 0.0)
    x = x + px * shorten_start
    y = y + py * shorten_start
    xend = xend - px * shorten_end
    yend = yend - py * shorten_end
    return x - py * offset, y + px * offset, xend - py * offset, yend + px * offset


def _area_pal(v: np.ndarray, lo: float, hi: float, rng: tuple[float, float] = (1.0, 6.0)):
    """``scale_size_continuous``: ``scales::area_pal(range)`` over limits ``lo..hi``."""
    r = np.full(v.shape, 0.5) if hi == lo else (v - lo) / (hi - lo)
    return rng[0] + np.sqrt(r) * (rng[1] - rng[0])


def _node_diameters(node: pd.DataFrame, node_size) -> tuple[np.ndarray, np.ndarray, str | None]:
    """``GraphPlot``'s node and outline sizes, in ggplot mm (a diameter).

    * a number (or a string that names no column) is used as is through
      ``scale_size_identity``; a non-numeric constant becomes 5, as in R.  The
      black outline disc is ``1.2 *`` that.
    * a numeric column maps through ``scale_size_continuous``, i.e.
      ``area_pal(range = c(1, 6))``.  Both point layers share the one scale,
      so it is trained on ``node_size`` **and** ``node_size * 1.2`` together
      (``SCP-plot.R:6551-6552``), and the outline is not a fixed 1.2x.
    * a non-numeric column maps through ``scale_size_discrete`` (ordinal,
      ``range = c(2, 6)`` spaced on area).

    Returns ``(fill, outline, size_title)``; the title is the column name when
    sizes are mapped, else ``None``.
    """
    n = len(node)
    if isinstance(node_size, str) and node_size in node.columns:
        col = node[node_size]
        if pd.api.types.is_numeric_dtype(col):
            v = col.to_numpy(dtype=float)
            pool = np.concatenate([v, 1.2 * v])
            lo, hi = float(np.nanmin(pool)), float(np.nanmax(pool))
            return _area_pal(v, lo, hi), _area_pal(1.2 * v, lo, hi), node_size
        cat = as_ordered_categorical(col)
        k = len(cat.cat.categories)
        steps = np.sqrt(np.linspace(2.0 ** 2, 6.0 ** 2, max(k, 1)))
        d = steps[np.asarray(cat.cat.codes)]
        return d, d * 1.2, node_size
    size = float(node_size) if isinstance(node_size, (int, float)) else 5.0
    return np.full(n, size), np.full(n, 1.2 * size), None


def graph_plot(
    node: pd.DataFrame,
    edge: np.ndarray,
    *,
    transition: np.ndarray | None = None,
    node_coord: tuple[str, str] = ("x", "y"),
    node_group: str | None = None,
    node_palette: str = "Paired",
    node_palcolor: Sequence[str] | dict[str, str] | None = None,
    node_size: float | str = 4.0,
    edge_threshold: float = 0.01,
    use_triangular: Literal["upper", "lower", "both"] = "upper",
    edge_line: Literal["straight", "curved"] = "straight",
    edge_line_curvature: float = 0.3,
    edge_color: str = "#666666",
    edge_size: tuple[float, float] = (0.2, 1.0),
    edge_alpha: float = 0.5,
    edge_shorten: float = 0.0,
    edge_offset: float = 0.0,
    transition_threshold: float = 0.01,
    transition_size: tuple[float, float] = (0.2, 1.0),
    transition_color: str = "black",
    transition_alpha: float = 1.0,
    transition_shorten: float = 0.05,
    transition_offset: float = 0.0,
    label: bool = False,
    label_insitu: bool = False,
    legend_position: LegendPosition = "right",
    legend: LegendColumn | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """The generic node/edge engine behind :func:`paga_plot`.

    Port of ``GraphPlot`` (``SCP-plot.R:6287``) by Hao Zhang,
    https://github.com/zhanghao-njmu/SCP.

    Anything expressible as (node table, edge matrix, optional directed
    transition matrix) routes through here.

    * **net transitions.** Only the net directed flow between each pair is
      drawn, as one arrow in the dominant direction::

          t1, t2 = T.copy(), T.copy()
          t1[np.tril_indices_from(t1)] = 0
          t2[np.triu_indices_from(t2)] = 0
          net = t1.T - t2          # net[i,j] = T[j,i] - T[i,j] for i > j
          # negative entries flip (from, to) and take abs

    * **edges go dashed when transitions are present** and solid otherwise, so
      connectivity and directed flow read differently.
    * **widths.** Edge weights map linearly onto ``edge_size`` and net flows,
      on their own scale, onto ``transition_size`` (R's two
      ``scale_linewidth_continuous`` separated by ``new_scale``).
    * **sizes.** Nodes are drawn twice -- a black disc, then the coloured one --
      which gives them their outline; see :func:`_node_diameters` for how a
      constant or a column becomes a diameter (ggplot mm, drawn as
      ``(mm * .pt) ** 2`` pt^2).
    * **shortening.** ``edge_shorten``, ``edge_offset``, ``transition_shorten``
      and ``transition_offset`` are fractions of
      ``global_size = sqrt(max(x)^2 + max(y)^2)`` over the **signed** node
      coordinates, exactly as R computes it (``SCP-plot.R:6389``).  That is
      not a norm -- with mostly negative coordinates it is small -- but it is
      what SCP draws, so it is kept.
    * **legend.** Node colours are registered as a legend titled by
      ``node_group``; with ``label=True`` and ``label_insitu=False`` the
      entries read ``"1: name"``, matching the numbers drawn on the nodes.

    Not ported: ``edge_line="curved"`` is accepted and drawn straight;
    ``node_alpha``, the ``*_highlight`` options and ``label_repel``.  R bug
    noted, not reproduced: the curved-transition branch tests
    ``edge_highlight`` where it means ``transition_highlight``.
    """
    xk, yk = node_coord
    X = node[xk].to_numpy(dtype=float)
    Y = node[yk].to_numpy(dtype=float)
    names = [str(i) for i in node.index]
    E = np.asarray(edge, dtype=float).copy()
    if E.shape[0] != E.shape[1] or E.shape[0] != len(node):
        raise ValueError(f"edge must be square and match node: {E.shape} vs {len(node)}")

    ax, legend, pg = _panel(ax, legend, legend_position)
    # global_size converts the fractional shorten/offset into data units; R
    # takes it on the signed coordinates (SCP-plot.R:6389), not on |x|, |y|
    global_size = float(np.sqrt(np.nanmax(X) ** 2 + np.nanmax(Y) ** 2))

    E[E <= edge_threshold] = np.nan
    if use_triangular == "upper":
        E[np.tril_indices_from(E)] = np.nan
    elif use_triangular == "lower":
        E[np.triu_indices_from(E)] = np.nan

    fi, ti = np.where(np.isfinite(E))
    if len(fi):
        widths = layers.geom.rescale(E[fi, ti], (float(edge_size[0]), float(edge_size[1])))
        x0, y0, x1, y1 = shorten_segments(
            X[fi], Y[fi], X[ti], Y[ti],
            global_size * edge_shorten, global_size * edge_shorten,
            global_size * edge_offset)
        # dashed when transitions are drawn, so connectivity and directed flow
        # read differently
        style = "--" if transition is not None else "-"
        segs = [[(a, b), (c, d)] for a, b, c, d in zip(x0, y0, x1, y1, strict=True)]
        ax.add_collection(LineCollection(segs, linewidths=widths, colors=edge_color,
                                         alpha=edge_alpha, linestyles=style, zorder=1))

    if transition is not None:
        T = np.asarray(transition, dtype=float)
        t1, t2 = T.copy(), T.copy()
        t1[np.tril_indices_from(t1)] = 0
        t2[np.triu_indices_from(t2)] = 0
        # only the NET flow between each pair is drawn, as one arrow in the
        # dominant direction; drawing every directed edge looks quite different
        net = t1.T - t2
        net[np.abs(net) <= transition_threshold] = np.nan
        ai, bi = np.where(np.isfinite(net))
        twidth = (layers.geom.rescale(np.abs(net[ai, bi]),
                                      (float(transition_size[0]), float(transition_size[1])))
                  if len(ai) else np.zeros(0))
        for a, b, wd in zip(ai, bi, twidth, strict=True):
            src, dst = (a, b) if net[a, b] > 0 else (b, a)
            x0, y0, x1, y1 = shorten_segments(
                np.array([X[src]]), np.array([Y[src]]),
                np.array([X[dst]]), np.array([Y[dst]]),
                global_size * transition_shorten, global_size * transition_shorten,
                global_size * transition_offset)
            ax.annotate("", xy=(x1[0], y1[0]), xytext=(x0[0], y0[0]),
                        arrowprops=dict(arrowstyle="-|>", color=transition_color,
                                        alpha=transition_alpha, linewidth=float(wd),
                                        shrinkA=0, shrinkB=0),
                        zorder=2)

    title_ = node_group or "node_name"
    if node_group is not None:
        cat = as_ordered_categorical(node[node_group])
        levels = [str(x) for x in cat.cat.categories]
        colors = discrete_palette(levels, palette=node_palette, palcolor=node_palcolor)
        vals = np.asarray(cat).astype(str)
        cvec = [colors[str(v)] for v in vals]
    else:
        levels = names
        colors = discrete_palette(names, palette=node_palette, palcolor=node_palcolor)
        vals = np.asarray(names)
        cvec = [colors[n] for n in names]

    d_fill, d_outline, _size_title = _node_diameters(node, node_size)
    # each node is drawn twice: a black disc, then the coloured one on top
    ax.scatter(X, Y, s=(_PT * d_outline) ** 2, c="black", zorder=3, linewidths=0)
    ax.scatter(X, Y, s=(_PT * d_fill) ** 2, c=cvec, zorder=4, linewidths=0)

    if label:
        for i in range(len(names)):
            # R labels with the group's level index, or its name in situ
            txt = vals[i] if label_insitu else str(levels.index(vals[i]) + 1)
            t = ax.text(X[i], Y[i], txt, ha="center", va="center", fontsize=3.5 * _PT,
                        color="white", fontweight="bold", zorder=5)
            halo(t, foreground="black", radius=0.1)

    if legend is not None and legend_position != "none":
        present = [lv for lv in levels if lv in set(vals)]
        handles = []
        for lv in present:
            lbl = f"{levels.index(lv) + 1}: {lv}" if label and not label_insitu else lv
            handles.append(Line2D([], [], marker="o", linestyle="none", markersize=4 * _PT,
                                  markerfacecolor=colors[lv], markeredgecolor="none",
                                  label=lbl))
        legend.add(title_, handles)

    _labels(ax, kwargs, own=pg is not None, xlab=xk, ylab=yk)
    _finish(pg)
    return ax


def _paga_nodes(groups: np.ndarray, x: np.ndarray, y: np.ndarray, levels: Sequence[str],
                groups_key: str, *, categories: Sequence[str] | None = None,
                ) -> tuple[pd.DataFrame, list[int]]:
    """``PAGAPlot``'s node table: one row per present level, at its cells' medians.

    SCP deliberately ignores ``uns['paga']['pos']`` and puts each node at the
    per-group **median** of the chosen embedding (x and y independently),
    which is what lets the same graph be drawn over UMAP, PCA or diffusion
    coordinates.  ``groups``, ``x`` and ``y`` are per cell, already restricted
    to the cells that count.  The table carries ``x``, ``y``, ``GroupSize``
    and ``groups_key`` (a Categorical over ``categories``, by default the
    present levels); ``idx`` gives each row's position in ``levels``, i.e. in
    the PAGA matrices.  Shared by :func:`paga_plot` and
    ``cell_dim_plot(paga=...)``.
    """
    groups = np.asarray(groups).astype(str)
    rows, sizes, present = [], [], []
    for lv in levels:
        m = groups == lv
        if not m.any():
            continue
        rows.append((float(np.median(x[m])), float(np.median(y[m]))))
        sizes.append(int(m.sum()))
        present.append(lv)
    node = pd.DataFrame(rows, columns=["x", "y"], index=pd.Index(present))
    node["GroupSize"] = sizes
    node[groups_key] = pd.Categorical(present, categories=list(categories or present))
    return node, [list(levels).index(lv) for lv in present]


def paga_plot(
    adata,
    *,
    reduction: str | None = None,
    dims: tuple[int, int] = (1, 2),
    type: Literal["connectivities", "connectivities_tree"] = "connectivities",
    show_transition: bool = False,
    node_palette: str = "Paired",
    node_size: float | str = 4.0,
    edge_threshold: float = 0.01,
    cells: Sequence[str] | None = None,
    use_uns_colors: bool = True,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """PAGA graph laid over an embedding.

    Port of ``PAGAPlot`` (``SCP-plot.R:6121``) by Hao Zhang,
    https://github.com/zhanghao-njmu/SCP; drawing goes through
    :func:`graph_plot`, and every other ``graph_plot`` keyword
    (``transition_size``, ``edge_shorten``, ``label``, ``legend`` ...) passes
    through ``**kwargs``.

    Reads ``adata.uns['paga']`` natively — ``connectivities``,
    ``connectivities_tree``, ``groups`` (scanpy) and ``transitions_confidence``
    (scvelo).  Rows and columns are indexed by
    ``adata.obs[groups].cat.categories``.

    SCP deliberately **ignores** ``uns['paga']['pos']`` and puts each node at
    the per-group *median* of the chosen embedding, which is what lets the
    same graph be drawn over UMAP, PCA or diffusion coordinates.  The group's
    cell count is available as ``node_size="GroupSize"``; the default is R's
    constant ``node_size = 4``.

    ``type="connectivities_tree"`` reads both triangles and disables the
    threshold, because the tree matrix is not symmetric-complete.
    ``show_transition=True`` needs scvelo's velocity-based PAGA
    (``scv.tl.paga``), which adds ``transitions_confidence``; without it the
    plot warns that it is missing and draws no transitions, as R does
    (``SCP-plot.R:6181``) and as ``cell_dim_plot(paga=...)`` does.

    ``reduction`` is resolved as SCP resolves it (``"umap"``, ``"UMAP"`` and
    ``"X_umap"`` all work; see :func:`scp.fetch.default_reduction`).  Nodes
    take ``adata.uns[f"{groups}_colors"]`` when scanpy stored one and neither
    ``node_palette`` nor ``node_palcolor`` was given (``use_uns_colors=False``
    restores R's Paired colours).  Drawn into a borrowed ``ax``, no
    ``"PAGA"`` title is set and existing axis titles are kept, as R's layer
    adds none.

    **R bug, not reproduced.** ``PAGAPlot(cells = ...)`` intersects the
    *group* names with the cell names (``SCP-plot.R:6188``), which drops every
    node, and never subsets the transitions.  Here ``cells`` restricts the
    cells the medians and group sizes are taken over, which is the evident
    intent.
    """
    conn, groups_key, categories = read_paga(adata, type=type)
    obs_cat = as_ordered_categorical(adata.obs[groups_key])
    levels = [str(x) for x in obs_cat.cat.categories]
    if conn.shape[0] != len(levels):
        raise ValueError(
            f"paga matrix is {conn.shape[0]}x{conn.shape[1]} but "
            f"{groups_key!r} has {len(levels)} levels"
        )

    reduction = default_reduction(adata, pattern=reduction)
    key = reduction_key(reduction)
    emb = np.asarray(adata.obsm[reduction])
    xi, yi = dims[0] - 1, dims[1] - 1

    vals = np.asarray(obs_cat).astype(str)
    keep = np.ones(adata.n_obs, dtype=bool)
    if cells is not None:
        keep = np.isin(np.asarray(adata.obs_names), np.asarray(cells))
    node, idx = _paga_nodes(vals[keep], emb[keep, xi], emb[keep, yi], levels, groups_key)

    E = np.asarray(conn.todense() if hasattr(conn, "todense") else conn, dtype=float)
    E = E[np.ix_(idx, idx)]
    transition = None
    if show_transition:
        if "transitions_confidence" not in adata.uns["paga"]:
            warnings.warn(
                "transitions_confidence need to be calculated first: show_transition needs "
                "velocity-based PAGA (scvelo's scv.tl.paga), which adds "
                "adata.uns['paga']['transitions_confidence']. Drawing no transitions.",
                stacklevel=2,
            )
        else:
            tr, _k, _c = read_paga(adata, type="transitions_confidence")
            transition = np.asarray(tr.todense() if hasattr(tr, "todense") else tr,
                                    dtype=float)[np.ix_(idx, idx)]

    # the tree matrix is not symmetric-complete, so read both triangles and
    # drop the threshold
    triangular: Literal["upper", "both"] = "both" if type == "connectivities_tree" else "upper"
    thresh = 0.0 if type == "connectivities_tree" else edge_threshold

    if (use_uns_colors and node_palette == "Paired" and "node_palcolor" not in kwargs
            and uns_palcolor(adata, groups_key, levels)):
        kwargs["node_palcolor"] = level_colors(adata, groups_key, levels, palette=node_palette,
                                               palcolor=None, default_palette="Paired")
    # a borrowed ax keeps its own title (R's PAGA layer carries none) and,
    # where it has them, its own axis titles
    if ax is None:
        kwargs.setdefault("title", "PAGA")
    if ax is None or not ax.get_xlabel():
        kwargs.setdefault("xlab", f"{key}{dims[0]}")
    if ax is None or not ax.get_ylabel():
        kwargs.setdefault("ylab", f"{key}{dims[1]}")
    return graph_plot(node, E, transition=transition, node_group=groups_key,
                      node_palette=node_palette, node_size=node_size,
                      edge_threshold=thresh, use_triangular=triangular, ax=ax, **kwargs)


def lineage_plot(
    adata,
    lineages: Sequence[str],
    *,
    reduction: str | None = None,
    dims: tuple[int, int] = (1, 2),
    cells: Sequence[str] | None = None,
    trim: tuple[float, float] = (0.01, 0.99),
    span: float = 0.75,
    palette: str = "Dark2",
    palcolor: Sequence[str] | dict[str, str] | None = None,
    lineages_arrow: float | None = 0.1,
    linewidth: float = 1.0,
    line_bg: str = "white",
    line_bg_stroke: float = 0.5,
    whiskers: bool = False,
    whiskers_linewidth: float = 0.5,
    whiskers_alpha: float = 0.5,
    legend_position: LegendPosition = "right",
    legend: LegendColumn | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """Principal curves over an embedding, one per pseudotime column.

    Port of ``LineagePlot`` (``SCP-plot.R:5918``) by Hao Zhang,
    https://github.com/zhanghao-njmu/SCP.

    For each lineage, the cells whose pseudotime lies **strictly** inside the
    ``trim`` quantiles are ordered by pseudotime and each embedding axis is
    LOESS-smoothed against it **independently** (``degree=2``, ``span``) --
    ``skmisc.loess``, the routine R's ``loess`` wraps; ``statsmodels`` lowess is
    degree 1 and would not match.  The fit and the drawing are shared with
    ``cell_dim_plot(lineages=...)`` and ``feature_dim_plot(lineages=...)``
    (``scp.pl._dim_layers.lineage_curves`` / ``draw_lineages``), so all three
    draw identical curves.

    Drawn, bottom to top: optional ``whiskers`` (a segment from each fitted
    point back to its cell), a ``line_bg`` casing ``linewidth +
    line_bg_stroke`` wide, then the coloured path.  Both paths end in an
    arrowhead of ``lineages_arrow`` inches (R's ``arrow(length = unit(0.1,
    "inches"))``; ``None`` draws none).  Lengths are ggplot millimetres.  The
    lineage colours are registered as a ``Lineages`` legend.

    Only the curves are drawn -- no cells -- so this is a layer: pass ``ax=``
    to lay it over an existing panel (and ``legend=`` to collect its legend).

    ``lineages`` are ``obs`` columns of pseudotime with NaN off-lineage,
    whatever produced them (scFates, Palantir, CellRank, pyslingshot).
    """
    reduction = default_reduction(adata, pattern=reduction)
    key = reduction_key(reduction)
    lineages = list(lineages)
    emb = np.asarray(adata.obsm[reduction], dtype=float)[:, [dims[0] - 1, dims[1] - 1]]
    rows = np.arange(adata.n_obs)
    if cells is not None:
        rows = np.flatnonzero(np.isin(np.asarray(adata.obs_names), np.asarray(cells)))
    curves = layers.lineage_curves(adata, lineages, emb, rows, trim=trim, span=span)

    ax, legend, pg = _panel(ax, legend, legend_position)
    handles = layers.draw_lineages(
        ax, curves, lineages, palette=palette, palcolor=palcolor, linewidth=linewidth,
        line_bg=line_bg, line_bg_stroke=line_bg_stroke, arrow=lineages_arrow,
        whiskers=whiskers, whiskers_linewidth=whiskers_linewidth,
        whiskers_alpha=whiskers_alpha)
    if legend is not None and legend_position != "none":
        legend.add("Lineages", handles)
    if pg is not None:
        ax.autoscale_view()
    _labels(ax, kwargs, own=pg is not None, xlab=f"{key}{dims[0]}", ylab=f"{key}{dims[1]}")
    _finish(pg)
    return ax


def velocity_plot(
    adata,
    *,
    basis: str = "umap",
    vkey: str = "velocity",
    dims: tuple[int, int] = (1, 2),
    cells: Sequence[str] | None = None,
    plot_type: Literal["raw", "grid", "stream"] = "raw",
    group_by: str | None = None,
    group_palette: str = "Paired",
    group_palcolor: Sequence[str] | dict[str, str] | None = None,
    use_uns_colors: bool = True,
    n_neighbors: int | None = None,
    density: float = 1.0,
    smooth: float = 0.5,
    scale: float = 1.0,
    min_mass: float = 1.0,
    cutoff_perc: float = 5.0,
    arrow_angle: float = 20.0,
    arrow_color: str = "black",
    streamline_width: tuple[float, float] = (0.0, 0.8),
    streamline_alpha: float = 1.0,
    streamline_color: str | None = None,
    streamline_palette: str = "RdYlBu",
    streamline_palcolor: Sequence[str] | None = None,
    streamline_bg_color: str = "white",
    streamline_bg_stroke: float = 0.5,
    legend_position: LegendPosition = "right",
    legend: LegendColumn | None = None,
    seed: int = 11,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """RNA velocity as per-cell arrows, a grid field, or streamlines.

    Port of ``VelocityPlot`` (``SCP-plot.R:6745``) by Hao Zhang,
    https://github.com/zhanghao-njmu/SCP.

    ``basis`` is resolved as SCP resolves a reduction (``"umap"``, ``"UMAP"``
    and ``"X_umap"`` all name ``obsm["X_umap"]``; see
    :func:`scp.fetch.default_reduction`), then ``adata.obsm[f"{vkey}_{basis}"]``
    is read -- scvelo's own convention.  SCP names the Seurat reduction
    ``"<mode>_<REDUCTION>"`` instead; that naming is not ported.

    Only the velocity is drawn -- no cells -- so this is a layer: pass ``ax=``
    to lay it over an existing panel; there it sets no ``"Cell velocity"``
    title (R's layer carries none) and keeps the panel's axis titles.

    * ``raw``: one arrow per cell from its position to ``position + scale *
      velocity``, R's ``geom_segment`` with a closed ``arrow_angle`` head of
      ``length / global_size`` npc -- the arrows ``cell_dim_plot(velocity=...)``
      draws, from the same layer.  As in R, a head comes out longer than its
      shaft on an embedding centred near zero, so large velocities read as
      triangles.  With ``group_by`` the **arrows** take the group colours
      (``group_palette``, or scanpy's ``uns[f"{group_by}_colors"]`` unless
      ``use_uns_colors=False``) and the groups are registered as a legend.
      ``0 < density < 1`` draws a random ``ceil(density * n)`` of them (seeded
      by ``seed``; numpy's generator, so not R's draw).
    * ``grid``: the field from :func:`compute_velocity_on_grid` (R's port, in
      numpy -- no scvelo needed), as the same arrows.
    * ``stream``: streamlines through the same grid in stream mode, drawn by
      the overlay layer ``cell_dim_plot(velocity=...)`` uses: a
      ``streamline_bg_color`` casing, the line (``streamline_color``, or
      coloured by speed through ``streamline_palette`` with a ``Velocity``
      colourbar), then ``arrow_color`` heads.  metR's ``streamline_L``,
      ``_minL``, ``_res`` and ``_n`` have no ``streamplot`` counterpart.

    ``n_neighbors=None`` means R's ``ceil(adata.n_obs / 50)``, counted over the
    whole object even when ``cells`` subsets it, as in R.

    R bug, not reproduced: ``VelocityPlot`` adds its ``lab_layer`` twice
    (``SCP-plot.R:6941-6942``), which is harmless.
    """
    basis_key = default_reduction(adata, pattern=basis)
    V_all = layers.velocity_embedding(adata, basis_key, vkey)
    if plot_type not in ("raw", "grid", "stream"):
        raise ValueError(f"plot_type must be 'raw', 'grid' or 'stream', not {plot_type!r}")
    xi, yi = dims[0] - 1, dims[1] - 1
    X = np.asarray(adata.obsm[basis_key], dtype=float)[:, [xi, yi]]
    V = V_all[:, [xi, yi]]
    rows = np.arange(adata.n_obs)
    if cells is not None:
        rows = np.flatnonzero(np.isin(np.asarray(adata.obs_names), np.asarray(cells)))
    X, V = X[rows], V[rows]
    k = n_neighbors if n_neighbors is not None else math.ceil(adata.n_obs / 50)
    rng = np.random.default_rng(seed)

    ax, legend, pg = _panel(ax, legend, legend_position)

    if plot_type in ("raw", "grid"):
        # the same arrows as cell_dim_plot(velocity=...): R's geom_segment with
        # a closed head of length / global_size npc
        cell_colors = None
        if plot_type == "raw" and group_by is not None:
            lab, levels = as_level_labels(adata.obs[group_by])
            colors = level_colors(adata, group_by, levels, palette=group_palette,
                                  palcolor=group_palcolor, default_palette="Paired",
                                  use_uns_colors=use_uns_colors)
            vals = lab.to_numpy()[rows]
            cell_colors = [colors[v] if isinstance(v, str) else "#CCCCCC" for v in vals]
        field, _m = layers.draw_velocity(
            ax, X, V, plot_type=plot_type, cell_colors=cell_colors, arrow_color=arrow_color,
            arrow_angle=arrow_angle, density=density, smooth=smooth, n_neighbors=k,
            scale=scale, min_mass=min_mass, cutoff_perc=cutoff_perc,
            streamline_width=streamline_width, streamline_alpha=streamline_alpha,
            streamline_color=streamline_color, streamline_palette=streamline_palette,
            streamline_palcolor=streamline_palcolor, streamline_bg_color=streamline_bg_color,
            streamline_bg_stroke=streamline_bg_stroke, rng=rng)
        if (group_by is not None and cell_colors is not None and legend is not None
                and legend_position != "none"):
            shown = {v for v in vals[field["row"].to_numpy()] if isinstance(v, str)}
            legend.add(group_by, [
                Line2D([], [], color=colors[lv], linewidth=2 * _PT, label=lv)
                for lv in levels if lv in shown])
        # an artist does not autoscale; the data it spans does
        ends = np.r_[field[["x", "y"]].to_numpy(),
                     field[["x", "y"]].to_numpy() + field[["u", "v"]].to_numpy()]
        ax.update_datalim(ends[np.isfinite(ends).all(axis=1)])
    else:
        # the same layer, and the same grid, as cell_dim_plot(velocity=...)
        _field, mappable = layers.draw_velocity(
            ax, X, V, plot_type="stream", cell_colors=None, arrow_color=arrow_color,
            arrow_angle=arrow_angle, density=density, smooth=smooth, n_neighbors=k,
            scale=1.0, min_mass=min_mass, cutoff_perc=cutoff_perc,
            streamline_width=streamline_width, streamline_alpha=streamline_alpha,
            streamline_color=streamline_color, streamline_palette=streamline_palette,
            streamline_palcolor=streamline_palcolor, streamline_bg_color=streamline_bg_color,
            streamline_bg_stroke=streamline_bg_stroke, rng=rng)
        if mappable is not None and legend_position != "none" and ax.figure is not None:
            # a borrowed ax belongs to the caller's figure; only our own grid
            # can place the bar outside the panel
            cb = (pg.colorbar(mappable, ax=ax) if pg is not None
                  else ax.figure.colorbar(mappable, ax=[ax], fraction=0.046, pad=0.03))
            cb.outline.set_edgecolor("black")
            cb.set_label("Velocity")

    ax.autoscale_view()
    key = reduction_key(basis_key)
    _labels(ax, kwargs, own=pg is not None, xlab=f"{key}{dims[0]}", ylab=f"{key}{dims[1]}",
            title="Cell velocity")
    _finish(pg)
    return ax


def _libsize(adata, layer: str | None, libsize) -> np.ndarray:
    """``colSums(counts)`` per cell, or the caller's ``libsize`` (array or obs column)."""
    if libsize is not None:
        if isinstance(libsize, str):
            return pd.to_numeric(adata.obs[libsize], errors="coerce").to_numpy(dtype=float)
        ls = np.asarray(libsize, dtype=float).ravel()
        if ls.size != adata.n_obs:
            raise ValueError(f"libsize has {ls.size} values for {adata.n_obs} cells")
        return ls
    # R always sums the assay's counts slot, whatever slot is plotted
    if "counts" in adata.layers:
        M = adata.layers["counts"]
    elif layer is not None and layer in adata.layers:
        M = adata.layers[layer]
    else:
        M = adata.X
    return np.asarray(M.sum(axis=1), dtype=float).ravel()


def _exp_transform(m: np.ndarray, exp_method: str) -> np.ndarray:
    """``DynamicPlot``'s per-lineage ``exp_method`` on a cells x features block."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if exp_method == "raw":
            out = m.copy()
        elif exp_method == "log1p":
            out = np.log1p(m)
        elif exp_method == "zscore":
            out = (m - np.nanmean(m, axis=0)) / np.nanstd(m, axis=0, ddof=1)
        elif exp_method == "fc":
            out = m / np.nanmean(m, axis=0)
        elif exp_method == "log2fc":
            out = np.log2(m / np.nanmean(m, axis=0))
        else:
            raise ValueError(f"exp_method must be log1p, raw, zscore, fc or log2fc, not {exp_method!r}")
    inf = np.isinf(out)
    if inf.any():
        finite = out[np.isfinite(out)]
        out[inf] = (np.max(np.abs(finite)) if finite.size else 0.0) * np.sign(out[inf])
    return out


def dynamic_plot(
    adata,
    lineages: Sequence[str],
    features: Sequence[str],
    *,
    group_by: str | None = None,
    cells: Sequence[str] | None = None,
    layer: str | None = "counts",
    exp_method: str = "log1p",
    lib_normalize: bool | None = None,
    libsize: np.ndarray | str | None = None,
    compare_lineages: bool = True,
    compare_features: bool = False,
    add_line: bool = True,
    add_interval: bool = True,
    line_size: float = 1.0,
    line_palette: str = "Dark2",
    line_palcolor: Sequence[str] | dict[str, str] | None = None,
    add_point: bool = True,
    pt_size: float = 1.0,
    point_palette: str = "Paired",
    point_palcolor: Sequence[str] | dict[str, str] | None = None,
    add_rug: bool = True,
    x_order: Literal["value", "rank"] = "value",
    legend_position: LegendPosition = "right",
    seed: int = 11,
    use_uns_colors: bool = True,
    **kwargs,
) -> Figure:
    """Expression along pseudotime: cells, fitted trends and confidence ribbons.

    Port of ``DynamicPlot`` (``SCP-plot.R:12102``) by Hao Zhang,
    https://github.com/zhanghao-njmu/SCP.  Milestone 9.

    What is reproduced from R, and checked against it:

    * **values.** With ``layer="counts"`` (R's ``slot = "counts"``) the gene
      columns are library-size normalised first, ``counts / libsize *
      median(libsize)`` with ``libsize = colSums(counts)`` over every gene, then
      transformed by ``exp_method`` within each lineage.  ``obs`` features are
      never normalised.  ``lib_normalize=None`` means ``layer == "counts"``;
      ``libsize`` overrides the column sums (an array, or an ``obs`` column --
      useful once the object has been subset to a few genes).  The y label is
      R's ``exp_name``, e.g. ``log1p(normalized counts)``.
    * **panels.** ``compare_lineages=True`` (the default) draws one panel per
      feature with every lineage overlaid; ``compare_features=True`` one panel
      per lineage with every feature overlaid; both, one panel.  Panel keys
      follow R's list names (``"Lineage1_Lineage2.Nnat"``).
    * **points** sit at ``x_assign``, each cell's mean pseudotime over the
      plotted lineages, coloured by ``group_by`` through ``point_palette``
      (black without it), with a matching rug.  Lines and ribbons sit at each
      lineage's own pseudotime, coloured by ``line_palette``.
      ``use_uns_colors=True`` (Python-only): with ``point_palette`` /
      ``point_palcolor`` at their defaults the points take scanpy's
      ``adata.uns[f"{group_by}_colors"]`` (:func:`scp.palettes.level_colors`);
      ``False`` gives R's palette colours exactly.

    Deliberate divergence: when the non-integer guard fires (the counts are
    not counts), R sets ``libsize`` to 1 but still multiplies by the median
    column sum, rescaling by an arbitrary constant; here the guard makes the
    normalisation a no-op, as ``group_heatmap`` does.

    .. warning::

       **The fitted curve does not match R and is not meant to.** ``mgcv``
       selects its smoothing parameter by GCV/REML over a penalised spline
       basis; ``pygam`` grid-searches a different basis and does not
       reproduce that criterion, so two fits of the same data differ by more
       than floating-point noise.  The trend here is fitted to the transformed
       values; R fits counts with a library-size offset.  What is stable
       across both is the *shape*: the trend direction, the location of the
       peak, and the ordering of features by peak time.  Without ``pygam`` the
       trend falls back to a rolling mean with no interval.

       Where ``scFates`` has already fitted and stored trends, reading them is
       strictly better than refitting and sidesteps the problem entirely.
    """
    lins = list(lineages)
    missing = [lin for lin in lins if lin not in adata.obs.columns]
    if missing:
        raise KeyError(f"lineages not in adata.obs: {missing}")
    line_palette = kwargs.get("palette", line_palette)  # the scaffold's name for it
    if group_by is not None and group_by not in adata.obs.columns:
        raise KeyError(f"{group_by!r} is not in adata.obs")
    exp = fetch_data(adata, list(features), layer=layer)
    # genes first, then obs columns: R's c(gene, meta), which is also
    # fetch_data's block order
    wanted = set(features)
    feats = [str(c) for c in exp.columns if c in wanted]
    if not feats:
        raise ValueError("none of `features` could be resolved")
    var_names = set(map(str, adata.var_names))
    is_gene = np.array([f in var_names for f in feats])
    Y = exp[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    T = np.column_stack([pd.to_numeric(adata.obs[lin], errors="coerce").to_numpy(dtype=float)
                         for lin in lins])
    keep_cell = np.ones(adata.n_obs, dtype=bool)
    if cells is not None:
        keep_cell = np.isin(np.asarray(adata.obs_names), np.asarray(cells))

    if lib_normalize is None:
        lib_normalize = layer == "counts"
    if lib_normalize and is_gene.any():
        ls = _libsize(adata, layer, libsize)
        if libsize is None and np.any(np.asarray(ls) % 1 != 0):
            warnings.warn("The values in the 'counts' layer are non-integer. "
                          "Set the library size to 1.", stacklevel=2)
            ls = np.ones_like(ls)
        on_any = np.isfinite(T).any(axis=1)
        if np.nanmin(Y[on_any][:, is_gene]) >= 0:
            Y = Y.copy()
            Y[:, is_gene] = Y[:, is_gene] / ls[:, None] * np.median(ls)
    data_nm = " ".join(s for s in ("normalized" if lib_normalize else "",
                                   layer if layer is not None else "data") if s)
    exp_name = f"{exp_method}({data_nm})"

    # per lineage: its cells and the transformed block, as R's raw_matrix_list
    on = {lin: np.flatnonzero(np.isfinite(T[:, i])) for i, lin in enumerate(lins)}
    vals = {lin: _exp_transform(Y[on[lin]], exp_method) for lin in lins}

    if group_by is not None:
        gcat = as_ordered_categorical(adata.obs[group_by])
        glevels = [str(x) for x in gcat.cat.categories]
        gcolors = level_colors(adata, group_by, glevels, palette=point_palette,
                               palcolor=point_palcolor, default_palette="Paired",
                               use_uns_colors=use_uns_colors)
        gvals = np.asarray(gcat).astype(object)
    lcolors = discrete_palette([str(x) for x in lins], palette=line_palette, palcolor=line_palcolor)
    fcolors = discrete_palette([str(x) for x in feats], palette=line_palette, palcolor=line_palcolor)

    lineages_guide = compare_lineages and len(lins) > 1
    fill_by_feature = compare_features and not lineages_guide
    lin_groups = [lins] if compare_lineages else [[lin] for lin in lins]
    feat_groups = [feats] if compare_features else [[f] for f in feats]
    panels = [(lg, fg) for lg in lin_groups for fg in feat_groups]
    keys = [f"{'_'.join(lg)}.{'_'.join(fg)}" for lg, fg in panels]
    th = theme_scp(legend_position=legend_position)
    pg = PanelGrid(len(keys), panel_size=kwargs.get("panel_size", (2.8, 2.0)),
                   nrow=kwargs.get("nrow"), ncol=kwargs.get("ncol"), byrow=kwargs.get("byrow", True),
                   keys=keys, theme=th, legend=LegendColumn(position=legend_position))
    rng = np.random.default_rng(seed)
    s_pt = (pt_size * _PT) ** 2
    # x_assign: each cell's mean pseudotime over ALL the lineages passed (R
    # takes rowMeans over `lineages`, whatever the panel shows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # cells on no lineage
        x_assign = np.nanmean(T, axis=1)
    present_groups: set[str] = set()

    for key, (lg, fg) in zip(keys, panels, strict=True):
        ax = pg.axes[key]
        # one row per (cell, x, value) over the panel's lineages and features:
        # a cell on two lineages with the same value is drawn once, as R's
        # unique(); points are skipped when features are compared, rugs are not
        pts = pd.DataFrame([
            (c, x_assign[c], vals[lin][n, feats.index(f)])
            for lin in lg for f in fg for n, c in enumerate(on[lin]) if keep_cell[c]
        ], columns=["cell", "x", "y"]).drop_duplicates()
        pts = pts.iloc[rng.permutation(len(pts))]
        px = pts["x"].to_numpy(dtype=float)
        if x_order == "rank":
            px = pd.Series(px).rank().to_numpy()
        cidx = pts["cell"].to_numpy(dtype=int)
        if group_by is not None:
            g = gvals[cidx]
            pcol = [gcolors[str(v)] if not pd.isna(v) else "#CCCCCC" for v in g]
            present_groups.update(str(v) for v in g if not pd.isna(v))
        else:
            pcol = ["black"] * len(px)
        if add_point and not compare_features:
            ax.scatter(px, pts["y"].to_numpy(dtype=float), s=s_pt, c=pcol, alpha=0.8,
                       linewidths=0, zorder=1)
        if add_rug and len(px):
            # geom_rug: ticks 5% of the panel tall, in axes units along y
            ax.add_collection(LineCollection(
                [[(x, 0.0), (x, 0.05)] for x in px], colors=pcol, linewidths=0.5,
                transform=ax.get_xaxis_transform(), zorder=1.5), autolim=False)

        for lin in lg:
            for f in fg:
                j = feats.index(f)
                sel = keep_cell[on[lin]]
                t = T[on[lin], lins.index(lin)][sel]
                yy = vals[lin][sel, j]
                ok = np.isfinite(t) & np.isfinite(yy)
                t, yy = t[ok], yy[ok]
                if len(t) < 2:
                    continue
                order = np.argsort(t, kind="stable")
                t, yy = t[order], yy[order]
                tx = pd.Series(t).rank().to_numpy() if x_order == "rank" else t
                # lines take the feature colour when features are compared;
                # ribbons follow R's fill_by, which stays "Lineages" whenever
                # several lineages share the panel
                col = fcolors[str(f)] if compare_features else lcolors[str(lin)]
                fill = fcolors[str(f)] if fill_by_feature else lcolors[str(lin)]
                fit, lo, hi = _fit_trend(t, yy)
                if add_interval and lo is not None:
                    ax.fill_between(tx, lo, hi, facecolor=fill, edgecolor="#E5E5E5",
                                    alpha=0.4, zorder=2)
                if add_line:
                    ax.plot(tx, fit, color=col, linewidth=layers.mm_lw(line_size), alpha=0.8,
                            zorder=3)

        # scale_x_continuous(expand = 0), scale_y_continuous(expand = c(0.1, 0.05))
        ax.set_xmargin(0)
        ax.set_ymargin(0)
        ax.autoscale_view()
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0 - 0.1 * (y1 - y0), y1 + 0.05 * (y1 - y0))
        ax.set_xlabel("Pseudotime(rank)" if x_order == "rank" else "Pseudotime")
        ax.set_ylabel(exp_name)
        if not compare_features:
            ax.set_title(fg[0], fontsize=th.size("plot.title"))
        if not compare_lineages:
            ax.text(1.02, 0.5, lg[0], transform=ax.transAxes, rotation=-90,
                    ha="left", va="center", fontsize=th.size("axis.title"))

    if legend_position != "none":
        if group_by is not None and add_point and not compare_features:
            pg.legend.add(group_by, [
                Line2D([], [], marker="o", linestyle="none", markersize=6,
                       markerfacecolor=gcolors[lv], markeredgecolor="none", label=lv)
                for lv in glevels if lv in present_groups])
        if lineages_guide and not compare_features and add_line:
            pg.legend.add("Lineages", [Line2D([], [], color=lcolors[str(lin)], linewidth=2,
                                              label=str(lin)) for lin in lins])
        if compare_features and len(feats) > 1:
            pg.legend.add("Features", [Line2D([], [], color=fcolors[str(f)], linewidth=2,
                                              label=str(f)) for f in feats])
    return pg.finish()


def _fit_trend(t: np.ndarray, y: np.ndarray):
    """A smooth trend plus an interval, via pygam when available.

    See the warning on :func:`dynamic_plot`: this does not reproduce mgcv's
    smoothing-parameter selection and is not meant to.
    """
    try:
        from pygam import LinearGAM, s
    except ImportError:
        warnings.warn(
            "pygam is not installed, so dynamic trends fall back to a rolling "
            "mean with no confidence interval (`pip install scp-plot[gam]`).",
            stacklevel=3,
        )
        w = max(len(t) // 20, 3)
        fit = pd.Series(y).rolling(w, center=True, min_periods=1).mean().to_numpy()
        return fit, None, None
    gam = LinearGAM(s(0)).fit(t, y)
    fit = gam.predict(t)
    ci = gam.confidence_intervals(t, width=0.95)
    return fit, ci[:, 0], ci[:, 1]


def projection_plot(
    adata_query,
    adata_ref,
    *,
    query_group: str | None = None,
    ref_group: str | None = None,
    query_reduction: str = "X_umap",
    ref_reduction: str = "X_umap",
    pt_size: float = 0.8,
    stroke_highlight: float = 0.5,
    use_uns_colors: bool = True,
    legend_position: LegendPosition = "right",
    legend: LegendColumn | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """Query cells mapped onto a reference embedding.

    Port of ``ProjectionPlot`` (``SCP-plot.R:12511``) by Hao Zhang,
    https://github.com/zhanghao-njmu/SCP.  Milestone 13.

    The reference cells are drawn first, coloured by ``ref_group`` (Paired,
    or ``#D9D9D9`` without a group); each query cell is then drawn twice, a
    black disc ``pt_size + stroke_highlight`` wide and its ``query_group``
    colour (Set1) on top, so query cells read as outlined discs over the
    reference cloud -- R's ``p3``.  ``palette`` / ``query_palette`` in
    ``kwargs`` override the two palettes; ``xlab``, ``ylab`` and ``title``
    label the axes.  Without ``ax``, one square themed panel
    (``panel_size`` in ``kwargs``, 3.5 in by default) is allocated through
    :class:`~scp.layout.PanelGrid`, like every other figure here.

    **R bug, deliberately not reproduced.**  ``ProjectionPlot`` computes the
    union of the reference and query ranges (``SCP-plot.R:12522-12531``) but
    applies it only to the throwaway query plot ``p2`` (``:12546``).  The
    composite ``p3`` is built on ``p1``, the reference ``CellDimPlot``, whose
    ``scale_x/y_continuous(limits = range(ref))`` (``:1636-1637``) censors
    every query cell outside the reference range to NA, and ggplot then drops
    them without a warning (111 of 362 query cells in the parity fixture).
    Here both axes are set to that union, so every query cell is drawn --
    what R evidently intends -- with ggplot's 5 % expansion on each side, as
    every SCP panel has.

    Legends, as R composes them beside the panel: ``"Ref: <ref_group>"``
    (the reference levels present) and ``"Query: <query_group>"`` (keys with
    a black ring, R's ``shape = 21`` override), stacked here where R places
    them side by side.  Into a borrowed ``ax`` they are registered with
    ``legend=`` if one is given.  Either group takes scanpy's
    ``uns[f"{group}_colors"]`` when stored and the palette was left alone
    (``use_uns_colors=False`` restores R's colours).  Both reductions are
    resolved as SCP resolves one (:func:`scp.fetch.default_reduction`).
    """
    ax, legend, pg = _panel(ax, legend, legend_position,
                            panel_size=kwargs.get("panel_size", (3.5, 3.5)))
    ref_reduction = default_reduction(adata_ref, pattern=ref_reduction)
    query_reduction = default_reduction(adata_query, pattern=query_reduction)
    Xr = np.asarray(adata_ref.obsm[ref_reduction], dtype=float)
    Xq = np.asarray(adata_query.obsm[query_reduction], dtype=float)

    # Shared limits over BOTH objects, or the projection reads as a shift that
    # is really just two different axis ranges.
    allx = np.concatenate([Xr[:, 0], Xq[:, 0]])
    ally = np.concatenate([Xr[:, 1], Xq[:, 1]])

    def _group(a, col, default_palette, palette):
        lab, levels = as_level_labels(a.obs[col])
        colors = level_colors(a, col, levels, palette=palette, palcolor=None,
                              default_palette=default_palette, use_uns_colors=use_uns_colors)
        vals = lab.to_numpy()
        cvec = [colors[v] if isinstance(v, str) else _NA_GREY for v in vals]
        present = [lv for lv in levels if (vals == lv).any()]
        return cvec, present, colors

    handles: list[tuple[str, list]] = []
    cref: list[str] | str = "#D9D9D9"
    if ref_group is not None:
        cref, present, colors = _group(adata_ref, ref_group, "Paired",
                                       kwargs.get("palette", "Paired"))
        handles.append((f"Ref: {ref_group}", [
            Line2D([], [], marker="o", linestyle="none", markersize=4 * _PT,
                   markerfacecolor=colors[lv], markeredgecolor="none", label=lv)
            for lv in present]))
    ax.scatter(Xr[:, 0], Xr[:, 1], s=pt_size * 6, c=cref, linewidths=0, zorder=1)

    cq: list[str] | str = "black"
    if query_group is not None:
        cq, qpresent, qcolors = _group(adata_query, query_group, "Set1",
                                       kwargs.get("query_palette", "Set1"))
        # R overrides the query key to shape 21: a black ring round the fill
        handles.append((f"Query: {query_group}", [
            Line2D([], [], marker="o", linestyle="none", markersize=4 * _PT,
                   markerfacecolor=qcolors[lv], markeredgecolor="black", label=lv)
            for lv in qpresent]))
    # halo then fill, so query cells read on top of a dense reference
    ax.scatter(Xq[:, 0], Xq[:, 1], s=(pt_size + stroke_highlight) ** 2 * 10,
               c="black", linewidths=0, zorder=2)
    ax.scatter(Xq[:, 0], Xq[:, 1], s=pt_size ** 2 * 10, c=cq, linewidths=0, zorder=3)

    if legend is not None and legend_position != "none":
        for title_, hs in handles:
            legend.add(title_, hs)

    ax.set_xlim(*expand_range(float(allx.min()), float(allx.max())))
    ax.set_ylim(*expand_range(float(ally.min()), float(ally.max())))
    ax.set_xlabel(kwargs.get("xlab", "Dim 1"))
    ax.set_ylabel(kwargs.get("ylab", "Dim 2"))
    if kwargs.get("title"):
        ax.set_title(kwargs["title"])
    _finish(pg)
    return ax
