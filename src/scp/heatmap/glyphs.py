"""Small plots drawn inside heatmap cells and annotation lanes.

SCP builds several heatmap decorations by rendering a complete ggplot per cell
and squeezing its grob into the cell (``anno_customize``, ``anno_block``,
``layer_fun``): a pie per column for a categorical ``cell_annotation`` and a
violin for a numeric one (``GroupHeatmap``), a violin per body cell for
``add_violin``, and a density ridgeline or fitted-curve strip spanning each
lineage for ``separate_annotation`` (``DynamicHeatmap``).  Here each is drawn
straight into the lane's axes, from the same numbers ggplot would compute:

* :func:`pie_wedges` -- ``geom_col(position = "stack") + coord_polar("y")``:
  the first level ends at 12 o'clock, the stack runs clockwise from it, and
  the pie's diameter is 80% of the cell's shorter side;
* :func:`violin_profile` -- ``geom_violin(scale = "width", trim = TRUE)``: a
  Gaussian density with ``bw.nrd0`` over the data range, scaled to a maximum
  half-width of 1; drawn 0.9 wide on a discrete axis that spans 1.2, inside a
  y range expanded by 5% (``coord_cartesian``);
* :func:`ridgelines` -- ``ggridges::geom_density_ridges``: one joint bandwidth
  (the mean of the groups' ``bw.nrd0``), densities over the panel's range
  +/- 3 bandwidths, scaled so the tallest ridge reaches 1.8 slots;
* :func:`draw_curves` -- ``DynamicPlot(compare_features = TRUE)``'s fitted
  lines and intervals, from fitted values the caller supplies.

``stats::density`` bins the data before its FFT convolution; the exact
Gaussian sum is used here, which agrees to about 1e-3 of the peak.

Upstream: Hao Zhang, SCP (GPL-3), https://github.com/zhanghao-njmu/SCP.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

__all__ = [
    "bw_nrd0",
    "density",
    "violin_profile",
    "violin_range",
    "curve_ranges",
    "pie_wedges",
    "ridgelines",
    "draw_pie",
    "draw_violin",
    "draw_ridgelines",
    "draw_curves",
    "GGPLOT_LINEWIDTH_PT",
]

#: ggplot's ``linewidth = 1`` in points (``.pt`` lwd units of 1/96 inch)
GGPLOT_LINEWIDTH_PT = 72.27 / 25.4 * 0.75


def bw_nrd0(x: np.ndarray) -> float:
    """``stats::bw.nrd0``: ``0.9 * min(sd, IQR / 1.34) * n^-1/5``, with R's fallbacks."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise ValueError("need at least 2 data points")
    hi = float(np.std(x, ddof=1))
    lo = min(hi, float(np.subtract(*np.quantile(x, [0.75, 0.25]))) / 1.34)
    if not lo:
        lo = hi or abs(float(x[0])) or 1.0
    return 0.9 * lo * x.size ** -0.2


def density(x: np.ndarray, bw: float, lo: float, hi: float, n: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """A Gaussian kernel density of ``x`` at ``n`` points from ``lo`` to ``hi``."""
    x = np.asarray(x, dtype=float)
    grid = np.linspace(lo, hi, n)
    z = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * z * z).sum(axis=1) / (x.size * bw * np.sqrt(2 * np.pi))
    return grid, dens


def violin_profile(values: np.ndarray, n: int = 512, adjust: float = 1.0):
    """``stat_ydensity(scale = "width", trim = TRUE)``: ``(y, width)``, max width 1.

    ``None`` for fewer than two finite values or no spread, which ggplot drops.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2 or np.ptp(v) == 0:
        return None
    y, d = density(v, bw_nrd0(v) * adjust, float(v.min()), float(v.max()), n)
    return y, d / d.max()


def pie_wedges(fractions: Sequence[float], *, start_deg: float = 90.0) -> list[tuple[float, float]]:
    """``(theta1, theta2)`` in matplotlib degrees for each level, in level order.

    ``position_stack`` puts the first level on top of the stack and
    ``coord_polar(theta = "y", start = 0)`` runs the stack clockwise from 12
    o'clock, so going *counter*-clockwise from the start the levels come in
    order.  ``start_deg`` is 90 (12 o'clock); a flipped heatmap's
    ``start = pi / 2`` is 0 (3 o'clock).
    """
    f = np.asarray(fractions, dtype=float)
    total = f.sum()
    if not total > 0:
        return []
    ends = start_deg + 360.0 * np.cumsum(f) / total
    starts = np.concatenate([[start_deg], ends[:-1]])
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def ridgelines(groups: Mapping[str, np.ndarray], positions: Mapping[str, float], *,
               scale: float = 1.8, n: int = 512) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float]:
    """``geom_density_ridges`` geometry: ``{group: (x, ymax)}`` and the top of the panel.

    ``positions`` gives each group's baseline (1, 2, ... on the discrete axis);
    groups with fewer than three values keep their slot but draw nothing.
    """
    xs = {g: np.asarray(v, dtype=float)[np.isfinite(np.asarray(v, dtype=float))]
          for g, v in groups.items()}
    bws = [bw_nrd0(v) for v in xs.values() if v.size > 1]
    allx = np.concatenate([v for v in xs.values()]) if xs else np.array([])
    if not bws or allx.size == 0:
        return {}, max(positions.values(), default=1.0)
    bw = float(np.mean(bws))
    lo, hi = float(allx.min()) - 3 * bw, float(allx.max()) + 3 * bw
    dens = {g: density(v, bw, lo, hi, n) for g, v in xs.items() if v.size >= 3}
    if not dens:
        return {}, max(positions.values(), default=1.0)
    hmax = max(float(d.max()) for _, d in dens.values())
    ys = [positions[g] for g in dens]
    nlev = len(set(ys))
    iscale = 1.0 if nlev < 2 else (max(ys) - min(ys)) / ((nlev - 1) * hmax)
    if nlev < 2:
        hmax_scale = 1.0 / hmax
    out = {}
    for g, (x, d) in dens.items():
        h = d * (iscale if nlev >= 2 else hmax_scale)
        out[g] = (x, positions[g] + scale * h)
    top = max(max(positions.values()), max(float(v[1].max()) for v in out.values()))
    return out, top


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def _cell_box_in(ax, x0: float, x1: float, y0: float, y1: float):
    """A data-space box's centre (data) and its size in inches."""
    fig = ax.get_figure()
    pos = ax.get_position()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    w_in = abs(x1 - x0) / abs(xlim[1] - xlim[0]) * pos.width * fig.get_figwidth()
    h_in = abs(y1 - y0) / abs(ylim[1] - ylim[0]) * pos.height * fig.get_figheight()
    return ((x0 + x1) / 2, (y0 + y1) / 2), w_in, h_in


def draw_pie(ax, box: tuple[float, float, float, float], fractions: Sequence[float],
             colors: Sequence[str], *, flip: bool = False, linewidth: float | None = None):
    """One pie centred in ``box = (x0, x1, y0, y1)`` (data units), a true circle."""
    from matplotlib.patches import Wedge
    from matplotlib.transforms import ScaledTranslation

    (cx, cy), w_in, h_in = _cell_box_in(ax, *box)
    r = 0.4 * min(w_in, h_in)
    fig = ax.get_figure()
    trans = fig.dpi_scale_trans + ScaledTranslation(cx, cy, ax.transData)
    lw = 0.5 * GGPLOT_LINEWIDTH_PT if linewidth is None else linewidth
    for (t1, t2), col in zip(pie_wedges(fractions, start_deg=0.0 if flip else 90.0), colors,
                             strict=False):
        if t2 <= t1:
            continue
        ax.add_patch(Wedge((0, 0), r, t1, t2, transform=trans, facecolor=col,
                           edgecolor="black", linewidth=lw, zorder=3, clip_on=False))


def violin_range(y_range: tuple[float, float]) -> tuple[float, float]:
    """``coord_cartesian(ylim = y_range)``: the shared range expanded by 5% a side."""
    lo, hi = float(y_range[0]), float(y_range[1])
    span = (hi - lo) or 1.0
    return lo - 0.05 * span, hi + 0.05 * span


def draw_violin(ax, box: tuple[float, float, float, float], values: np.ndarray,
                y_range: tuple[float, float], color, *, flip: bool = False):
    """One violin in ``box`` (data units), drawn as ggplot draws it in that box.

    ``y_range`` is the shared value range (``same.y.lims``); it is expanded by
    5% on each side, and the violin is 0.9 / 1.2 of the box wide at its widest.
    With ``flip`` the value axis runs along x (``coord_flip``).
    """
    from matplotlib.patches import Polygon

    prof = violin_profile(values)
    if prof is None:
        return
    y, w = prof
    x0, x1, y0, y1 = box
    lo, hi = violin_range(y_range)
    t = (y - lo) / (hi - lo)               # 0..1 along the value axis
    half = 0.45 / 1.2 * w                  # 0..0.375 across it
    if not flip:
        # values grow upward on screen, whichever way the axes' y runs
        vy = y1 - (y1 - y0) * t if ax.yaxis_inverted() else y0 + (y1 - y0) * t
        cx, cw = (x0 + x1) / 2, (x1 - x0)
        px = np.concatenate([cx - half * cw, (cx + half * cw)[::-1]])
        py = np.concatenate([vy, vy[::-1]])
    else:
        vx = x0 + (x1 - x0) * t
        cy, ch = (y0 + y1) / 2, abs(y1 - y0)
        py = np.concatenate([cy - half * ch, (cy + half * ch)[::-1]])
        px = np.concatenate([vx, vx[::-1]])
    ax.add_patch(Polygon(np.column_stack([px, py]), closed=True, facecolor=color,
                         edgecolor="#333333", linewidth=0.5 * GGPLOT_LINEWIDTH_PT, zorder=5))


def draw_ridgelines(ax, ridges: Mapping[str, tuple[np.ndarray, np.ndarray]],
                    positions: Mapping[str, float], colors: Mapping[str, str], *,
                    x_range: tuple[float, float], y_top: float, n_cells: int,
                    flip: bool = False):
    """Ridges into a lane whose long axis spans ``n_cells`` heatmap cells.

    ``x_range`` (possibly reversed) is the rank range mapped onto the lane;
    ridge points outside it are dropped, as the scale's limits censor them.
    """
    from matplotlib.patches import Polygon

    a, b = x_range
    order = sorted(ridges, key=lambda g: -positions[g])  # upper ridges first
    for g in order:
        x, ymax = ridges[g]
        keep = (x >= min(a, b)) & (x <= max(a, b))
        if keep.sum() < 2:
            continue
        x, ymax = x[keep], ymax[keep]
        u = (x - a) / (b - a) * n_cells          # along the lane
        base = positions[g]
        v0 = (base - 1) / (y_top - 1) if y_top > 1 else 0.0
        v1 = (ymax - 1) / (y_top - 1) if y_top > 1 else np.zeros_like(ymax)
        if not flip:
            pts = np.column_stack([np.concatenate([u, u[::-1]]),
                                   np.concatenate([v1, np.full(len(u), v0)])])
            line = (u, v1)
        else:
            pts = np.column_stack([np.concatenate([v1, np.full(len(u), v0)]),
                                   np.concatenate([u, u[::-1]])])
            line = (v1, u)
        z = 3 + (y_top - base)
        ax.add_patch(Polygon(pts, closed=True, facecolor=colors.get(g, "#B3B3B3"),
                             edgecolor="none", zorder=z))
        ax.plot(*line, color="black", linewidth=0.5 * GGPLOT_LINEWIDTH_PT, zorder=z + 0.1)


def curve_ranges(curves) -> tuple[tuple[float, float], tuple[float, float]]:
    """The panel ranges of ``DynamicPlot``: ``expansion(0)`` on x, ``(0.1, 0.05)`` on y."""
    xs = np.concatenate([np.asarray(c[0], dtype=float) for c in curves])
    ys = np.concatenate([np.concatenate([c[1], c[2], c[3]]) for c in curves]).astype(float)
    ys = ys[np.isfinite(ys)]
    if not ys.size:
        return (float(xs.min()), float(xs.max())), (np.nan, np.nan)
    y_lo, y_hi = float(ys.min()), float(ys.max())
    span = y_hi - y_lo
    return (float(xs.min()), float(xs.max())), (y_lo - 0.1 * span, y_hi + 0.05 * span)


def draw_curves(ax, curves: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]],
                *, n_cells: int, reverse: bool = False, flip: bool = False,
                line_size: float = 1.0):
    """Fitted lines and intervals into a lane spanning ``n_cells`` cells.

    ``curves`` holds ``(x, y, lower, upper, colour)`` per feature, ``x`` the
    rank 1..n.  The scales are ``DynamicPlot``'s: x unexpanded (rank 1 and n
    on the lane's ends), y expanded by 10% below and 5% above
    (:func:`curve_ranges`).
    """
    (x_lo, x_hi), (y_lo, y_hi) = curve_ranges(curves)
    if not np.isfinite(y_lo):
        return
    for x, y, lwr, upr, col in curves:
        o = np.argsort(x, kind="stable")
        x, y, lwr, upr = x[o], y[o], lwr[o], upr[o]
        u = (x - x_lo) / (x_hi - x_lo)
        if reverse:
            u = 1 - u
        u = u * n_cells
        v, vl, vu = ((arr - y_lo) / (y_hi - y_lo) for arr in (y, lwr, upr))
        if not flip:
            ax.fill_between(u, vl, vu, facecolor=col, alpha=0.4, edgecolor="#E5E5E5",
                            linewidth=0.5 * GGPLOT_LINEWIDTH_PT, zorder=3)
            ax.plot(u, v, color=col, alpha=0.8, linewidth=line_size * GGPLOT_LINEWIDTH_PT,
                    zorder=4)
        else:
            ax.fill_betweenx(u, vl, vu, facecolor=col, alpha=0.4, edgecolor="#E5E5E5",
                             linewidth=0.5 * GGPLOT_LINEWIDTH_PT, zorder=3)
            ax.plot(v, u, color=col, alpha=0.8, linewidth=line_size * GGPLOT_LINEWIDTH_PT,
                    zorder=4)
