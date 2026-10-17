"""Overlay layers for :func:`cell_dim_plot` and :func:`feature_dim_plot`.

Each ``draw_*`` function draws one of SCP's optional layers into an existing
``Axes`` and returns what the caller needs for the legend; none of them creates
a figure or calls ``ax.legend()``.  The numbers come from
:mod:`scp.pl._dim_geom`; this module only turns them into artists.

Two layers are geometry in *physical* units in R -- ``ggforce`` marks are grown
by ``expand = 3 mm``, and velocity arrowheads are sized in ``npc`` -- so they
are implemented as artists that finish their geometry at draw time, from the
axes' actual transform.  That is the matplotlib analogue of grid's
``makeContent`` and keeps them correct whatever size the panel ends up.

Upstream: ``CellDimPlot`` / ``FeatureDimPlot`` (``SCP-plot.R:1328`` and
``:2042``), ``LineagePlot`` (``:5918``), ``VelocityPlot`` (``:6745``) by Hao
Zhang, https://github.com/zhanghao-njmu/SCP.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform

from ..palettes import continuous_palette, discrete_palette
from . import _dim_geom as geom

__all__ = [
    "mm_lw",
    "linestyle",
    "draw_density",
    "draw_graph",
    "draw_marks",
    "draw_hex_groups",
    "draw_hex_values",
    "fit_lineages",
    "draw_lineages",
    "draw_velocity",
    "draw_stat_insets",
]

# z-order of every layer, bottom to top, following SCP's layer order (§1.3):
# marks, graph edges, density, points, highlight, stat pies, lineages, paga,
# velocity, labels.
Z_MARK, Z_GRAPH, Z_DENSITY = 0.5, 0.6, 0.7
Z_POINTS, Z_HIGHLIGHT = 1.0, 1.1
Z_STAT, Z_LINEAGE, Z_PAGA, Z_VELOCITY, Z_LABEL = 1.5, 3.0, 3.5, 4.0, 7.0


def mm_lw(mm: float) -> float:
    """ggplot ``linewidth`` (mm) -> matplotlib points.

    ggplot multiplies by ``.pt`` (72.27 / 25.4) into R ``lwd``, and one ``lwd``
    is 1/96 inch, i.e. 0.75 pt.
    """
    return float(mm) * 72.27 / 25.4 * 0.75


_LTY = {0: "None", 1: "-", 2: "--", 3: ":", 4: "-.", 5: (0, (8, 3)), 6: (0, (2, 2, 6, 2))}


def linestyle(lty):
    """R ``lty`` (integer or name) -> a matplotlib linestyle."""
    if isinstance(lty, int):
        return _LTY.get(lty, "-")
    names = {"blank": "None", "solid": "-", "dashed": "--", "dotted": ":",
             "dotdash": "-.", "longdash": _LTY[5], "twodash": _LTY[6]}
    return names.get(str(lty), lty)


# --------------------------------------------------------------------------- #
# density
# --------------------------------------------------------------------------- #
def draw_density(ax: Axes, x: np.ndarray, y: np.ndarray, lims: Sequence[float], *,
                 filled: bool, color: str, palette: str, palcolor) -> None:
    """``geom_density_2d`` (contours) or ``stat_density_2d(geom="raster")``.

    The KDE runs over **every cell in the panel**, background included, on a
    100 x 100 grid spanning the scale limits -- which SCP pins to the global
    embedding range.  Contours sit at ``pretty(range(density), 10)``; the
    filled variant is a raster through ``palette`` (``"Greys"``).
    """
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return
    gx, gy, z = geom.kde2d(x[ok], y[ok], lims=lims, n=100)
    if filled:
        ramp = continuous_palette(palette=palette, palcolor=palcolor, n=100)
        cmap = LinearSegmentedColormap.from_list("density", ramp, N=256)
        dx, dy = gx[1] - gx[0], gy[1] - gy[0]
        ax.imshow(z.T, origin="lower", cmap=cmap, aspect="auto", interpolation="nearest",
                  extent=(gx[0] - dx / 2, gx[-1] + dx / 2, gy[0] - dy / 2, gy[-1] + dy / 2),
                  zorder=Z_DENSITY)
    else:
        levels = geom.density_breaks(z)
        inside = levels[(levels > np.nanmin(z)) & (levels < np.nanmax(z))]
        if inside.size:
            ax.contour(gx, gy, z.T, levels=inside, colors=color, linewidths=mm_lw(0.5),
                       zorder=Z_DENSITY)


# --------------------------------------------------------------------------- #
# graph edges
# --------------------------------------------------------------------------- #
def draw_graph(ax: Axes, graph, order: np.ndarray, x: np.ndarray, y: np.ndarray, *,
               edge_size: tuple[float, float], alpha: float, color: str) -> int:
    """Edges of the cell graph, from COO triplets into one ``LineCollection``.

    ``order`` gives each drawn cell's row in ``graph``, in drawing order; ``x``
    and ``y`` are in the same order.  Weights map linearly onto ``edge_size``
    (mm), as ``scale_linewidth_continuous(range = edge_size)`` does.  Returns
    the number of edges kept (diagonal entries included, as in R).
    """
    r, c, w = geom.graph_segments(graph, order)
    if w.size == 0:
        return 0
    widths = np.array([mm_lw(v) for v in geom.rescale(w, edge_size)])
    segs = np.stack([np.column_stack([x[r], y[r]]), np.column_stack([x[c], y[c]])], axis=1)
    keep = r != c  # self-loops are zero-length; they count for the scale only
    ax.add_collection(LineCollection(segs[keep], linewidths=widths[keep], colors=color,
                                     alpha=alpha, capstyle="butt", zorder=Z_GRAPH))
    return int(w.size)


# --------------------------------------------------------------------------- #
# ggforce marks
# --------------------------------------------------------------------------- #
class _MarkPatch(Patch):
    """One ``geom_mark_*`` enclosure, finished at draw time in millimetres.

    ggforce works in the panel's physical space: it converts the points to mm,
    rounds them to 0.01 mm, encloses them, and grows the result by ``expand``
    with corners rounded to ``radius``.  Doing the same against the live
    ``transData`` keeps the 3 mm margin 3 mm whatever the final panel size.
    """

    def __init__(self, xy: np.ndarray, kind: str, expand: float, radius: float, **kw):
        super().__init__(**kw)
        self._xy = np.asarray(xy, dtype=float)
        self._kind = kind
        self._expand = float(expand)
        self._radius = float(radius)
        self._path = Path(np.zeros((1, 2)))
        self._cache: tuple[bytes, np.ndarray] | None = None
        self.set_transform(IdentityTransform())

    def get_path(self):
        return self._path

    def get_patch_transform(self):
        return IdentityTransform()

    def shape_mm(self, px_to_mm: float) -> np.ndarray:
        """The enclosure polygon in mm, for the current axes transform."""
        ax = self.axes
        assert ax is not None
        mm = np.round(ax.transData.transform(self._xy) * px_to_mm, 2)
        key = mm.tobytes() + self._kind.encode()
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        kind = self._kind
        if kind == "rect":
            lo, hi = mm.min(axis=0), mm.max(axis=0)
            core = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
            poly = geom.offset_shape(core, self._expand, self._radius, convex=True)
        elif kind == "circle":
            x0, y0, r = geom.enclose_circle(mm[:, 0], mm[:, 1])
            t = 2 * np.pi * np.arange(100) / 100
            core = np.column_stack([x0 + r * np.cos(t), y0 + r * np.sin(t)])
            poly = geom.offset_shape(core, self._expand, self._radius, convex=True)
        elif kind == "ellipse":
            x0, y0, a, b, ang = geom.enclose_ellipse(mm[:, 0], mm[:, 1], tol=0.01)
            t = 2 * np.pi * np.arange(100) / 100
            xt, yt = np.cos(t) * a, np.sin(t) * b
            core = np.column_stack([x0 + xt * np.cos(ang) - yt * np.sin(ang),
                                    y0 + xt * np.sin(ang) + yt * np.cos(ang)])
            poly = geom.offset_shape(core, self._expand, self._radius, convex=True)
        else:  # hull
            core = geom.concave_hull(mm, concavity=2.0)
            poly = geom.offset_shape(core, self._expand, self._radius,
                                     convex=len(core) < 3)
        self._cache = (key, poly)
        return poly

    def draw(self, renderer):
        if self.axes is None or self.figure is None:
            return
        px_to_mm = 25.4 / self.figure.dpi
        poly = self.shape_mm(px_to_mm) / px_to_mm
        self._path = Path(np.vstack([poly, poly[:1]]), closed=True)
        super().draw(renderer)


def draw_marks(ax: Axes, frame: pd.DataFrame, group: str, levels: Sequence[str],
               colors: Mapping[str, str], *, kind: str, expand: float, alpha: float,
               linetype) -> list[_MarkPatch]:
    """``geom_mark_hull`` / ``_ellipse`` / ``_rect`` / ``_circle`` per present level.

    Defaults are ggforce's as SCP calls them: ``expand = 3 mm`` (SCP's
    ``mark_expand``); ``radius = 2.5 mm`` for hull and rect, ``radius =
    expand`` for ellipse and circle; fill at ``alpha``, outline opaque and
    0.5 mm wide.
    """
    radius = expand if kind in ("ellipse", "circle") else 2.5
    out = []
    for lv in levels:
        sub = frame[frame[group] == lv]
        xy = sub[["x", "y"]].to_numpy(dtype=float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) == 0:
            continue
        col = colors[str(lv)]
        p = _MarkPatch(xy, kind, expand, radius, facecolor=to_rgba(col, alpha),
                       edgecolor=col, linewidth=mm_lw(0.5), linestyle=linestyle(linetype),
                       zorder=Z_MARK)
        ax.add_patch(p)
        out.append(p)
    return out


# --------------------------------------------------------------------------- #
# hexagons
# --------------------------------------------------------------------------- #
def _hex_binwidth(xlim, ylim, bins: int, binwidth) -> tuple[float, float]:
    if binwidth is not None:
        bw = np.atleast_1d(np.asarray(binwidth, dtype=float))
        return (float(bw[0]), float(bw[-1]))
    return ((xlim[1] - xlim[0]) / bins, (ylim[1] - ylim[0]) / bins)


def _in_limits(df: pd.DataFrame, xlim, ylim) -> np.ndarray:
    """ggplot censors a hexagon whose *centre* falls outside the pinned scale limits."""
    x, y = df["x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)
    return (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])


def _hex_polys(df: pd.DataFrame, bw: tuple[float, float]) -> np.ndarray:
    off = geom.hexagon_offsets(bw)
    centres = df[["x", "y"]].to_numpy(dtype=float)
    return centres[:, None, :] + off[None, :, :]


def draw_hex_groups(ax: Axes, frame: pd.DataFrame, group: str, levels: Sequence[str],
                    colors: Mapping[str, str], bg_color: str, *, xlim, ylim, bins: int,
                    binwidth, linewidth: float, count_alpha: bool) -> pd.DataFrame:
    """``CellDimPlot(hex = TRUE)``: hexagons binned **within** each group.

    ggplot's ``geom_hex(aes(fill = group.by, colour = group.by))`` bins each
    group separately, so hexagons of different groups overlap.  Groups draw in
    level order with the background (NA) group **last**, because ggplot gives
    the NA group the highest id -- faithful, though it means grey hexagons of
    other-split cells sit on top.  ``count_alpha`` maps the count onto alpha
    through ggplot's default ``range = c(0.1, 1)``, pooled across groups.
    Hexagons centred outside the global limits are dropped, as ggplot's
    ``scale_x_continuous(limits = )`` censors them (after the alpha scale
    has seen their counts).  Returns the binned table (``group``, ``x``,
    ``y``, ``count``), censored rows included.
    """
    bw = _hex_binwidth(xlim, ylim, bins, binwidth)
    tables = []
    for lv in [*levels, None]:
        sub = frame[frame[group].isna()] if lv is None else frame[frame[group] == lv]
        if sub.empty:
            continue
        hb = geom.hex_bin(sub["x"].to_numpy(), sub["y"].to_numpy(), bw)
        hb["group"] = lv
        tables.append(hb)
    if not tables:
        return pd.DataFrame(columns=["x", "y", "value", "group"])
    allhex = pd.concat(tables, ignore_index=True)
    counts = allhex["value"].to_numpy()
    alphas = geom.rescale(counts, (0.1, 1.0)) if count_alpha else np.ones(len(counts))
    start = 0
    for t in tables:
        n = len(t)
        col = bg_color if t["group"].iloc[0] is None else colors[str(t["group"].iloc[0])]
        a = alphas[start:start + n]
        start += n
        keep = _in_limits(t, xlim, ylim)
        faces = [to_rgba(col, float(v)) for v in a[keep]]
        ax.add_collection(PolyCollection(list(_hex_polys(t[keep], bw)), facecolors=faces,
                                         edgecolors=col, linewidths=mm_lw(linewidth),
                                         zorder=Z_POINTS))
    return allhex.rename(columns={"value": "count"})


def draw_hex_values(ax: Axes, frame: pd.DataFrame, value: str, *, cmap, norm, bg_color: str,
                    edge_color: str, xlim, ylim, bins: int, binwidth,
                    linewidth: float) -> pd.DataFrame:
    """``FeatureDimPlot(hex = TRUE)``: background hexagons, then the mean value per hexagon.

    Two layers, as R: ``geom_hex`` of the background cells in flat
    ``bg_color``, then ``stat_summary_hex`` (``fun = mean``) of the rest
    through the panel's colour scale.  Both outline in ``edge_color``
    (``hex.color``, grey90), and both drop hexagons centred outside the
    limits, as ggplot censors them.  Returns the value hexagons drawn.
    """
    bw = _hex_binwidth(xlim, ylim, bins, binwidth)
    vals = frame[value].to_numpy(dtype=float)
    bg = ~np.isfinite(vals)
    if bg.any():
        hb = geom.hex_bin(frame["x"].to_numpy()[bg], frame["y"].to_numpy()[bg], bw)
        hb = hb[_in_limits(hb, xlim, ylim)]
        ax.add_collection(PolyCollection(list(_hex_polys(hb, bw)), facecolors=bg_color,
                                         edgecolors=edge_color, linewidths=mm_lw(linewidth),
                                         zorder=Z_POINTS))
    if (~bg).any():
        hv = geom.hex_bin(frame["x"].to_numpy()[~bg], frame["y"].to_numpy()[~bg], bw,
                          z=vals[~bg], fun="mean")
        hv = hv[_in_limits(hv, xlim, ylim)]
        pc = PolyCollection(list(_hex_polys(hv, bw)), edgecolors=edge_color,
                            linewidths=mm_lw(linewidth), cmap=cmap, norm=norm, zorder=Z_POINTS)
        pc.set_array(hv["value"].to_numpy())
        ax.add_collection(pc)
        return hv
    return pd.DataFrame(columns=["x", "y", "value"])


# --------------------------------------------------------------------------- #
# lineages
# --------------------------------------------------------------------------- #
def _loess(t: np.ndarray, v: np.ndarray, span: float) -> np.ndarray:
    from skmisc.loess import loess

    fit = loess(t, v, span=span, degree=2)
    fit.fit()
    return np.asarray(fit.outputs.fitted_values)


def fit_lineages(emb: np.ndarray, pseudotime: Mapping[str, np.ndarray], *,
                 trim: tuple[float, float], span: float) -> dict[str, pd.DataFrame]:
    """``LineagePlot``'s principal curves: per-axis degree-2 LOESS on pseudotime.

    For each lineage, cells with pseudotime strictly inside its
    ``trim`` quantiles are ordered by pseudotime and each embedding axis is
    smoothed against it separately (``loess(axis ~ lineage, span, degree =
    2)``); the fitted values, in pseudotime order, are the curve.  Returns per
    lineage a frame of ``x``, ``y`` (fitted), ``raw_x``, ``raw_y`` and
    ``index`` (row in ``emb``).
    """
    try:
        import skmisc.loess  # noqa: F401
        have_loess = True
    except ImportError:
        have_loess = False
        warnings.warn(
            "skmisc is not installed, so lineage curves fall back to a rolling mean "
            "rather than the degree-2 LOESS SCP uses (`pip install scp-plot[stats]`).",
            stacklevel=3,
        )
    out: dict[str, pd.DataFrame] = {}
    for name, pt in pseudotime.items():
        pt = np.asarray(pt, dtype=float)
        fin = np.isfinite(pt)
        if fin.sum() < 3:
            continue
        lo, hi = np.quantile(pt[fin], trim[0]), np.quantile(pt[fin], trim[1])
        idx = np.flatnonzero(fin & (pt > lo) & (pt < hi))  # strict, as R
        idx = idx[np.argsort(pt[idx], kind="stable")]
        if len(idx) < 3:
            continue
        t = pt[idx]
        raw = emb[idx]
        if have_loess:
            fx, fy = _loess(t, raw[:, 0], span), _loess(t, raw[:, 1], span)
        else:
            w = max(len(t) // 20, 3)
            fx = pd.Series(raw[:, 0]).rolling(w, center=True, min_periods=1).mean().to_numpy()
            fy = pd.Series(raw[:, 1]).rolling(w, center=True, min_periods=1).mean().to_numpy()
        df = pd.DataFrame({"x": fx, "y": fy, "raw_x": raw[:, 0], "raw_y": raw[:, 1],
                           "index": idx})
        out[name] = df.dropna().drop_duplicates()
    return out


def _arrowhead(ax: Axes, xy0, xy1, *, color, lw, length_in: float, zorder: float) -> None:
    """grid's default open ``arrow(angle = 30)`` of ``length_in`` inches at ``xy1``."""
    hl = length_in * 72.0  # points
    hw = 2 * hl * math.tan(math.radians(30))
    style = f"->,head_length={hl / 10:.4f},head_width={hw / 20:.4f}"
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle=style, mutation_scale=10, color=color,
                                 linewidth=lw, shrinkA=0, shrinkB=0, zorder=zorder,
                                 capstyle="round", joinstyle="miter"))


def draw_lineages(ax: Axes, curves: Mapping[str, pd.DataFrame], lineages: Sequence[str], *,
                  palette: str, palcolor, linewidth: float, line_bg: str, line_bg_stroke: float,
                  arrow: float | None, whiskers: bool, whiskers_linewidth: float,
                  whiskers_alpha: float) -> list[Line2D]:
    """Draw the curves: optional whiskers, a ``line_bg`` casing, then the coloured path.

    The casing is ``linewidth + line_bg_stroke`` wide (R adds, it does not
    double).  ``arrow`` is the head length in inches (``arrow(length =
    unit(0.1, "inches"))``); ``None`` draws none.  Returns legend handles,
    one per lineage, titled ``Lineages`` by the caller.
    """
    colors = discrete_palette([str(x) for x in lineages], palette=palette, palcolor=palcolor)
    handles = []
    for lin in lineages:
        df = curves.get(lin)
        col = colors[str(lin)]
        handles.append(Line2D([], [], color=col, linewidth=mm_lw(linewidth) + 1, label=str(lin)))
        if df is None or len(df) < 2:
            continue
        xs, ys = df["x"].to_numpy(), df["y"].to_numpy()
        if whiskers:
            segs = np.stack([np.column_stack([xs, ys]),
                             df[["raw_x", "raw_y"]].to_numpy()], axis=1)
            ax.add_collection(LineCollection(list(segs), colors=col, alpha=whiskers_alpha,
                                             linewidths=mm_lw(whiskers_linewidth),
                                             zorder=Z_LINEAGE - 0.05))
        bg_lw = mm_lw(linewidth + line_bg_stroke)
        ax.plot(xs, ys, color=line_bg, linewidth=bg_lw, solid_capstyle="butt", zorder=Z_LINEAGE)
        ax.plot(xs, ys, color=col, linewidth=mm_lw(linewidth), solid_capstyle="butt",
                zorder=Z_LINEAGE + 0.02)
        if arrow:
            tail = (xs[-2], ys[-2])
            head = (xs[-1], ys[-1])
            _arrowhead(ax, tail, head, color=line_bg, lw=bg_lw, length_in=arrow,
                       zorder=Z_LINEAGE + 0.01)
            _arrowhead(ax, tail, head, color=col, lw=mm_lw(linewidth), length_in=arrow,
                       zorder=Z_LINEAGE + 0.03)
    return handles


# --------------------------------------------------------------------------- #
# velocity
# --------------------------------------------------------------------------- #
class _ArrowField(Artist):
    """Segments with closed heads whose length is a fraction of the panel.

    ``VelocityPlot`` gives every arrow ``arrow(length = unit(length / global_size,
    "npc"), type = "closed", angle = 20)``, so long arrows get long heads.  The
    heads are built at draw time in display space, from the live axes size.
    """

    def __init__(self, x0, y0, x1, y1, head_npc, *, colors, linewidth: float, angle: float,
                 zorder: float):
        super().__init__()
        self._seg = np.column_stack([x0, y0, x1, y1]).astype(float)
        self._head = np.asarray(head_npc, dtype=float)
        self._colors = colors
        self._lw = linewidth
        self._angle = math.radians(angle)
        self.set_zorder(zorder)

    def draw(self, renderer):
        ax = self.axes
        if ax is None or not self.get_visible():
            return
        p0 = ax.transData.transform(self._seg[:, :2])
        p1 = ax.transData.transform(self._seg[:, 2:])
        size = min(ax.bbox.width, ax.bbox.height)
        d = p1 - p0
        ln = np.hypot(d[:, 0], d[:, 1])
        ok = ln > 0
        u = np.zeros_like(d)
        u[ok] = d[ok] / ln[ok, None]
        hl = self._head * size
        back = p1 - u * (hl * math.cos(self._angle))[:, None]
        perp = np.column_stack([-u[:, 1], u[:, 0]]) * (hl * math.sin(self._angle))[:, None]
        heads = np.stack([p1, back + perp, back - perp], axis=1)[ok]
        lc = LineCollection(np.stack([p0, p1], axis=1), colors=self._colors,
                            linewidths=self._lw, transform=IdentityTransform(),
                            capstyle="round")
        cols = self._colors
        if not isinstance(cols, str) and len(np.atleast_1d(cols)) == len(ok):
            cols = np.asarray(cols, dtype=object)[ok].tolist()
        pc = PolyCollection(heads, facecolors=cols, edgecolors=cols, linewidths=self._lw,
                            transform=IdentityTransform())
        for art in (lc, pc):
            art.set_figure(self.figure)
            art.set_clip_box(ax.bbox)
            art.draw(renderer)


def _velocity_grid(X, V, *, density, smooth, n_neighbors, min_mass, cutoff_perc,
                   adjust_for_stream):
    """scvelo's ``compute_velocity_on_grid``, called with SCP's parameters.

    Called directly rather than through ``scp.pl.compute_velocity_on_grid``:
    scvelo already maps ``min_mass -> 10 ** (min_mass - 6)`` in stream mode,
    and grid mode must not autoscale, because ``VelocityPlot`` multiplies by
    ``scale`` itself.
    """
    try:
        from scvelo.plotting.velocity_embedding_grid import compute_velocity_on_grid
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "velocity_plot_type='grid' or 'stream' needs `pip install scp-plot[velocity]`"
        ) from e
    return compute_velocity_on_grid(X_emb=X, V_emb=V, density=density, smooth=smooth,
                                    n_neighbors=n_neighbors, min_mass=min_mass,
                                    autoscale=False, adjust_for_stream=adjust_for_stream,
                                    cutoff_perc=cutoff_perc)


def draw_velocity(ax: Axes, X: np.ndarray, V: np.ndarray, *, plot_type: str,
                  cell_colors: Sequence[str] | None, arrow_color: str, arrow_angle: float,
                  density: float, smooth: float, n_neighbors: int, scale: float,
                  min_mass: float, cutoff_perc: float, streamline_width: tuple[float, float],
                  streamline_alpha: float, streamline_color: str | None,
                  streamline_palette: str, streamline_palcolor, streamline_bg_color: str,
                  streamline_bg_stroke: float, rng: np.random.Generator,
                  grid_fn=None):
    """``VelocityPlot``'s layer: raw per-cell arrows, a grid of arrows, or streamlines.

    Returns ``(field, mappable)``: the arrow table that was drawn (``x``,
    ``y``, ``u``, ``v``) and, for streamlines coloured by speed, a
    ``ScalarMappable`` for the ``Velocity`` colourbar (else ``None``).
    ``grid_fn`` replaces the scvelo grid computation (tests use it).
    """
    from matplotlib.cm import ScalarMappable

    grid_fn = grid_fn or _velocity_grid
    if plot_type == "raw":
        keep = np.arange(len(X))
        if density is not None and 0 < density < 1:
            keep = np.sort(rng.choice(len(X), size=math.ceil(density * len(X)), replace=False))
        x, y = X[keep, 0], X[keep, 1]
        u, v = V[keep, 0] * scale, V[keep, 1] * scale
        cols = (np.asarray(cell_colors, dtype=object)[keep].tolist()
                if cell_colors is not None else arrow_color)
    elif plot_type == "grid":
        gx, gv = grid_fn(X, V, density=density, smooth=smooth, n_neighbors=n_neighbors,
                         min_mass=min_mass, cutoff_perc=cutoff_perc, adjust_for_stream=False)
        gx, gv = np.asarray(gx), np.asarray(gv) * scale
        x, y, u, v = gx[:, 0], gx[:, 1], gv[:, 0], gv[:, 1]
        cols = arrow_color
    elif plot_type == "stream":
        gx, gv = grid_fn(X, V, density=density, smooth=smooth, n_neighbors=n_neighbors,
                         min_mass=min_mass, cutoff_perc=cutoff_perc, adjust_for_stream=True)
        gx, gv = np.asarray(gx), np.asarray(gv, dtype=float)
        xs, ys = gx[0], gx[1]
        uu = np.ma.masked_invalid(gv[0])
        vv = np.ma.masked_invalid(gv[1])
        vv = np.ma.array(vv, mask=np.ma.getmaskarray(uu) | np.ma.getmaskarray(vv))
        speed = np.sqrt(uu**2 + vv**2)
        lw = mm_lw(max(streamline_width))
        dens = max(float(density), 0.1)
        # streamplot seeds deterministically, so the casing pass traces exactly
        # the lines the coloured pass draws on top of it
        bg = ax.streamplot(xs, ys, uu, vv, color=streamline_bg_color,
                           linewidth=mm_lw(max(streamline_width) + streamline_bg_stroke),
                           arrowstyle="-", zorder=Z_VELOCITY, density=dens)
        mappable = None
        if streamline_color is not None:
            sl = ax.streamplot(xs, ys, uu, vv, color=streamline_color, linewidth=lw,
                               arrowstyle="-|>", zorder=Z_VELOCITY + 0.01, density=dens)
        else:
            ramp = continuous_palette(palette=streamline_palette, palcolor=streamline_palcolor)
            cmap = LinearSegmentedColormap.from_list("stream", ramp, N=256)
            sp_ = np.ma.filled(speed.astype(float), np.nan)
            norm = Normalize(float(np.nanmin(sp_)), float(np.nanmax(sp_)))
            sl = ax.streamplot(xs, ys, uu, vv, color=sp_, cmap=cmap, norm=norm, linewidth=lw,
                               arrowstyle="-|>", zorder=Z_VELOCITY + 0.01, density=dens)
            mappable = ScalarMappable(norm=norm, cmap=cmap)
        # the third R layer draws only arrowheads, in arrow_color
        sl.arrows.set_color(arrow_color)
        for art in (bg.lines, sl.lines):
            art.set_alpha(streamline_alpha)
        field = pd.DataFrame({"x": np.repeat(xs[None, :], len(ys), 0).ravel(),
                              "y": np.repeat(ys[:, None], len(xs), 1).ravel(),
                              "u": np.ma.filled(uu, np.nan).ravel(),
                              "v": np.ma.filled(vv, np.nan).ravel()})
        return field, mappable
    else:
        raise ValueError(f"velocity_plot_type must be 'raw', 'grid' or 'stream', not {plot_type!r}")

    length = np.hypot(u, v)
    global_size = math.sqrt(np.nanmax(x) ** 2 + np.nanmax(y) ** 2) or 1.0
    head = length / global_size
    ax.add_artist(_ArrowField(x, y, x + u, y + v, head, colors=cols,
                              linewidth=mm_lw(0.5), angle=arrow_angle, zorder=Z_VELOCITY))
    return pd.DataFrame({"x": x, "y": y, "u": u, "v": v, "head_npc": head}), None


# --------------------------------------------------------------------------- #
# stat_by insets
# --------------------------------------------------------------------------- #
def stat_table(groups: pd.Series, stat: pd.Series, group_levels: Sequence[str],
               stat_levels: Sequence[str], *, stat_type: str) -> pd.DataFrame:
    """``StatPlot``'s cross-tab for one split: stat level x group, % within group."""
    tab = pd.crosstab(stat, groups, dropna=False)
    tab = tab.reindex(index=list(stat_levels), columns=list(group_levels), fill_value=0)
    if stat_type == "percent":
        tot = tab.sum(axis=0).replace(0, np.nan)
        return tab / tot
    return tab.astype(float)


def draw_stat_insets(ax: Axes, centres: pd.DataFrame, table: pd.DataFrame,
                     colors: Mapping[str, str], *, plot_type: str, position: str, size: float,
                     x_range: float, y_range: float, alpha: float, label: bool,
                     label_size: float, stat_type: str) -> list[Axes]:
    """One mini-plot per group, centred on its median position.

    The inset spans ``x_range * size`` by ``y_range * size`` data units, as
    ``annotation_custom(xmin = x - x_range * size / 2, ...)`` does.  Pies
    occupy 80 % of the inset, like ``coord_polar``; a ring keeps an inner hole
    of 1 / 1.8 of the radius, from R's dummy ``"   "`` level.
    """
    insets = []
    levels = list(table.index)
    for grp, row in centres.iterrows():
        if grp not in table.columns:
            continue
        vals = np.nan_to_num(table[grp].to_numpy(dtype=float))
        w, h = x_range * size, y_range * size
        iax = ax.inset_axes((row["x"] - w / 2, row["y"] - h / 2, w, h), transform=ax.transData,
                            zorder=Z_STAT)
        iax.set_axis_off()
        cols = [colors[lv] for lv in levels]
        if plot_type in ("pie", "ring"):
            if vals.sum() <= 0:
                continue
            wedge = dict(edgecolor="black", linewidth=mm_lw(0.5), alpha=alpha)
            if plot_type == "ring":
                wedge["width"] = 1 - 1 / 1.8
            total = float(vals.sum())

            def _fmt(p: float, _t: float = total) -> str:
                return f"{p:.1f}%" if stat_type == "percent" else f"{p * _t / 100:.0f}"

            # matplotlib starts at 12 o'clock and runs anticlockwise, which puts
            # the first level exactly where coord_polar's stacked bar ends it
            iax.pie(vals, colors=cols, startangle=90, counterclock=True, wedgeprops=wedge,
                    autopct=_fmt if label else None, textprops={"fontsize": label_size * 2})
        elif plot_type == "bar":
            if position == "stack":
                bottom = 0.0
                for lv, v in reversed(list(zip(levels, vals, strict=True))):
                    iax.bar(0, v, bottom=bottom, width=0.8, color=colors[lv], alpha=alpha,
                            edgecolor="black", linewidth=mm_lw(0.5))
                    bottom += v
                iax.set_xlim(-0.5, 0.5)
                iax.set_ylim(0, max(bottom, 1e-12))
            else:
                k = len(levels)
                wd = 0.9 / max(k, 1)
                for i, (lv, v) in enumerate(zip(levels, vals, strict=True)):
                    iax.bar(-0.45 + wd * (i + 0.5), v, width=wd, color=colors[lv],
                            alpha=alpha, edgecolor="black", linewidth=mm_lw(0.5))
                iax.set_xlim(-0.5, 0.5)
                iax.set_ylim(0, max(float(vals.max()) * 1.1, 1e-12))
        insets.append(iax)
    return insets
