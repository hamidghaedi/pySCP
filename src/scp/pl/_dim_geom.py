"""Pure transforms behind the dimensional-reduction overlays.

Everything here is numeric and side-effect free, so each function can be pinned
against R with a fixture.  The drawing lives in :mod:`scp.pl._dim_layers`.

What is ported, and from where (all upstream code by Hao Zhang and the authors
of the R packages SCP builds on; SCP itself is
https://github.com/zhanghao-njmu/SCP):

* :func:`r_pretty` -- R's ``pretty()`` (``src/appl/pretty.c``), which
  ``ggplot2::contour_breaks`` uses to choose the density contour levels;
* :func:`kde2d` / :func:`bandwidth_nrd` -- ``MASS::kde2d`` and
  ``MASS::bandwidth.nrd``, as called by ``ggplot2::stat_density_2d``;
* :func:`hex_bin` -- ``hexbin::hexbin`` + ``hexbin::hcell2xy`` as driven by
  ``ggplot2``'s ``hexBinSummarise``;
* :func:`enclose_ellipse` / :func:`enclose_circle` / :func:`concave_hull` /
  :func:`offset_shape` -- the geometry ``ggforce::geom_mark_*`` computes at
  draw time (Khachiyan's ellipse, Welzl's circle, ``concaveman``, and the
  ``polyclip`` offset that gives marks their ``expand``/``radius``);
* :func:`graph_segments` -- ``CellDimPlot``'s graph-edge melt
  (``SCP-plot.R:1586``), without densifying the adjacency;
* :func:`compute_velocity_on_grid` -- SCP's own ``compute_velocity_on_grid``
  (``SCP-plot.R:6963``), which both ``VelocityPlot`` and ``CellDimPlot``'s
  velocity layer call, so ``velocity_plot`` and ``cell_dim_plot(velocity=)``
  share it here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "r_pretty",
    "bandwidth_nrd",
    "kde2d",
    "density_breaks",
    "hex_bin",
    "hexagon_offsets",
    "enclose_ellipse",
    "enclose_circle",
    "concave_hull",
    "offset_shape",
    "graph_segments",
    "rescale",
    "expand_range",
    "compute_velocity_on_grid",
]

_DBL_EPS = float(np.finfo(float).eps)


# --------------------------------------------------------------------------- #
# R's pretty()
# --------------------------------------------------------------------------- #
def r_pretty(lo: float, hi: float, n: int = 10) -> np.ndarray:
    """Port of ``pretty.default(c(lo, hi), n)`` with R's default biases.

    ``min.n = n %/% 3``, ``shrink.sml = 0.75``, ``high.u.bias = 1.5``,
    ``u5.bias = 0.5 + 1.5 * high.u.bias`` and ``eps.correct = 0`` -- the
    defaults ``ggplot2::contour_breaks`` gets when it calls ``pretty(z, 10)``.
    """
    min_n = n // 3
    shrink = 0.75
    h = 1.5
    h5 = 0.5 + 1.5 * h
    rounding_eps = 1e-10
    dx = hi - lo
    if dx == 0 and hi == 0:
        cell = 1.0
        i_small = True
    else:
        cell = max(abs(lo), abs(hi))
        u = 1 + (1 / (1 + h) if h5 >= 1.5 * h + 0.5 else 1.5 / (1 + h5))
        u *= max(1, n) * _DBL_EPS
        i_small = dx < cell * u * 3
    if i_small:
        if cell > 10:
            cell = 9 + cell / 10
        cell *= shrink
        if min_n > 1:
            cell /= min_n
    else:
        cell = dx
        if n > 1:
            cell /= n
    cell = max(cell, 20 * np.finfo(float).tiny)
    base = 10.0 ** math.floor(math.log10(cell))
    unit = base
    if (ns := 2 * base) - cell < h * (cell - unit):
        unit = ns
        if (ns := 5 * base) - cell < h5 * (cell - unit):
            unit = ns
            if (ns := 10 * base) - cell < h * (cell - unit):
                unit = ns
    nlo = math.floor(lo / unit + rounding_eps)
    nup = math.ceil(hi / unit - rounding_eps)
    while nlo * unit > lo + rounding_eps * unit:
        nlo -= 1
    while nup * unit < hi - rounding_eps * unit:
        nup += 1
    k = int(0.5 + nup - nlo)
    if k < min_n:
        k = min_n - k
        if nlo >= 0:
            nup += k // 2
            nlo -= k // 2 + k % 2
        else:
            nlo -= k // 2
            nup += k // 2 + k % 2
        ndiv = min_n
    else:
        ndiv = k
    out = np.linspace(nlo * unit, nup * unit, ndiv + 1)
    if ndiv:
        delta = (nup * unit - nlo * unit) / ndiv
        out[np.abs(out) < 1e-14 * delta] = 0.0
    return out


# --------------------------------------------------------------------------- #
# 2-D kernel density, as stat_density_2d computes it
# --------------------------------------------------------------------------- #
def bandwidth_nrd(x: np.ndarray) -> float:
    """``MASS::bandwidth.nrd``: Scott-style rule of thumb, times four."""
    x = np.asarray(x, dtype=float)
    q1, q3 = np.quantile(x, [0.25, 0.75])
    h = (q3 - q1) / 1.34
    return float(4 * 1.06 * min(np.sqrt(np.var(x, ddof=1)), h) * len(x) ** (-1 / 5))


def _bw_nrd0(x: np.ndarray) -> float:
    """``stats::bw.nrd0`` -- ggplot's fallback when the IQR rule gives zero."""
    x = np.asarray(x, dtype=float)
    hi = np.std(x, ddof=1)
    q1, q3 = np.quantile(x, [0.25, 0.75])
    lo = min(hi, (q3 - q1) / 1.34)
    if not lo:
        lo = hi or abs(x[0]) or 1.0
    return float(0.9 * lo * len(x) ** (-0.2))


def kde2d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lims: Sequence[float],
    n: int = 100,
    h: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``MASS::kde2d`` with ggplot2's bandwidth choice (``precompute_2d_bw``).

    Returns ``(gx, gy, z)`` with ``z[i, j]`` the density at ``(gx[i], gy[j])``
    -- R's orientation, x along rows.  ``lims`` is ``(xmin, xmax, ymin,
    ymax)``; ggplot passes the *scale* limits, which SCP pins to the global
    embedding range, so the grid is the same in every split panel.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if h is None:
        hx, hy = bandwidth_nrd(x), bandwidth_nrd(y)
        if hx == 0 and len(x) > 1:
            hx = _bw_nrd0(x) * 4
        if hy == 0 and len(y) > 1:
            hy = _bw_nrd0(y) * 4
    else:
        hx, hy = float(h[0]), float(h[1])
    if not (hx > 0 and hy > 0):
        raise ValueError("the density bandwidth must be positive; too few distinct cells")
    gx = np.linspace(lims[0], lims[1], n)
    gy = np.linspace(lims[2], lims[3], n)
    hx, hy = hx / 4, hy / 4
    ax = (gx[:, None] - x[None, :]) / hx
    ay = (gy[:, None] - y[None, :]) / hy
    phi_x = np.exp(-0.5 * ax**2) / np.sqrt(2 * np.pi)
    phi_y = np.exp(-0.5 * ay**2) / np.sqrt(2 * np.pi)
    z = phi_x @ phi_y.T / (len(x) * hx * hy)
    return gx, gy, z


def density_breaks(z: np.ndarray) -> np.ndarray:
    """Contour levels: ``pretty(range(z), 10)``, ggplot's default with no bins."""
    finite = z[np.isfinite(z)]
    return r_pretty(float(finite.min()), float(finite.max()), 10)


# --------------------------------------------------------------------------- #
# hexagonal binning
# --------------------------------------------------------------------------- #
def _round_any(x: float, accuracy: float, f) -> float:
    return float(f(x / accuracy) * accuracy)


def hex_bin(
    x: np.ndarray,
    y: np.ndarray,
    binwidth: tuple[float, float],
    z: np.ndarray | None = None,
    fun: str = "count",
) -> pd.DataFrame:
    """One group's hexagons, exactly as ``ggplot2::hexBinSummarise`` builds them.

    ``binwidth`` is ``(diff(xlim) / bins, diff(ylim) / bins)``.  The bounds are
    snapped outward to multiples of the binwidth (``hex_bounds``), then
    ``hexbin``'s Fortran cell assignment and ``hcell2xy`` centres are applied.
    Returns columns ``x``, ``y``, ``value`` (count, or the mean of ``z``),
    ordered by cell id as R returns them.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bw_x, bw_y = float(binwidth[0]), float(binwidth[1])
    xbnds = (_round_any(x.min(), bw_x, math.floor) - 1e-6,
             _round_any(x.max(), bw_x, math.ceil) + 1e-6)
    ybnds = (_round_any(y.min(), bw_y, math.floor) - 1e-6,
             _round_any(y.max(), bw_y, math.ceil) + 1e-6)
    xbins = (xbnds[1] - xbnds[0]) / bw_x
    ybins = (ybnds[1] - ybnds[0]) / bw_y
    shape = ybins / xbins

    jmax = math.floor(xbins + 1.5001)
    c1 = 2 * math.floor((xbins * shape) / math.sqrt(3) + 1.5001)
    imax = math.trunc((jmax * c1 - 1) / jmax + 1)
    _ = imax  # bounds the lattice in hexbin; cell ids never exceed it here

    size = xbins
    xr = xbnds[1] - xbnds[0]
    yr = ybnds[1] - ybnds[0]
    k1 = size / xr
    k2 = size * shape / (yr * math.sqrt(3.0))
    iinc = 2 * jmax
    lat = jmax + 1
    sx = k1 * (x - xbnds[0])
    sy = k2 * (y - ybnds[0])
    j1 = np.trunc(sx + 0.5).astype(np.int64)
    i1 = np.trunc(sy + 0.5).astype(np.int64)
    dist1 = (sx - j1) ** 2 + 3.0 * (sy - i1) ** 2
    j2 = np.trunc(sx).astype(np.int64)
    i2 = np.trunc(sy).astype(np.int64)
    dist2 = (sx - j2 - 0.5) ** 2 + 3.0 * (sy - i2 - 0.5) ** 2
    near = i1 * iinc + j1 + 1
    far = i2 * iinc + j2 + lat
    cell = np.where(dist1 < 0.25, near,
                    np.where(dist1 > 1.0 / 3.0, far, np.where(dist1 <= dist2, near, far)))

    ids, inverse = np.unique(cell, return_inverse=True)
    if fun == "count" or z is None:
        value = np.bincount(inverse).astype(float)
    elif fun == "mean":
        zz = np.asarray(z, dtype=float)
        value = np.bincount(inverse, weights=zz) / np.bincount(inverse)
    else:
        raise ValueError(f"unknown hex summary {fun!r}")

    c3 = (xbnds[1] - xbnds[0]) / xbins
    c4 = ((ybnds[1] - ybnds[0]) * math.sqrt(3)) / (2 * shape * xbins)
    c0 = ids - 1
    row = c0 // jmax
    col = c0 % jmax
    cy = c4 * row + ybnds[0]
    cx = c3 * np.where(row % 2 == 0, col, col + 0.5) + xbnds[0]
    return pd.DataFrame({"x": cx, "y": cy, "value": value, "cell": ids})


def hexagon_offsets(binwidth: tuple[float, float]) -> np.ndarray:
    """The six vertex offsets ``geom_hex`` draws around each centre.

    ``hexbin::hexcoords(dx = width / 2, dy = height / sqrt(3) / 2)``: pointy
    side up, so a row of hexagons tiles at ``height * sqrt(3) / 2``.
    """
    dx = binwidth[0] / 2
    dy = binwidth[1] / math.sqrt(3) / 2
    return np.array([[dx, dy], [0, 2 * dy], [-dx, dy], [-dx, -dy], [0, -2 * dy], [dx, -dy]])


# --------------------------------------------------------------------------- #
# ggforce marks
# --------------------------------------------------------------------------- #
def enclose_ellipse(x: np.ndarray, y: np.ndarray, tol: float = 0.01
                    ) -> tuple[float, float, float, float, float]:
    """Minimum-area enclosing ellipse by Khachiyan's algorithm.

    Port of ``ggforce``'s ``enclose_ellip_points`` (C++), which stops when the
    weight update falls below ``tol`` (0.01 in ``geom_mark_ellipse``), so the
    result is a slightly-loose ellipse rather than the exact optimum.  Returns
    ``(x0, y0, a, b, angle)`` with ``a`` the major semi-axis and ``angle`` its
    direction, the parametrisation ``geom_mark_ellipse`` draws from.

    Divergence: centre and semi-axes agree with ggforce 0.5.0 to 1e-7 mm on
    ``pancreas_sub``, but ggforce's angle behaves like ``acos`` of the major
    axis' x-component -- it drops the sign of the y-component, so whenever
    the eigenvector points downward the ellipse it draws is mirrored and no
    longer encloses the points (4 of the 5 ``CellType`` groups; normalised
    radius 2.0 against 1.04).  This returns the enclosing orientation.
    """
    pts = np.unique(np.column_stack([x, y]).astype(float), axis=0)
    n = len(pts)
    if n == 1:
        return float(pts[0, 0]), float(pts[0, 1]), 0.0, 0.0, 0.0
    if n == 2:
        c = pts.mean(axis=0)
        d = pts[1] - pts[0]
        return (float(c[0]), float(c[1]), float(np.hypot(*d) / 2), 0.0,
                float(np.arctan2(d[1], d[0])))
    d = 2
    q = np.vstack([pts.T, np.ones(n)])
    u = np.full(n, 1.0 / n)
    err = 1.0 + tol
    while err > tol:
        xm = (q * u) @ q.T
        try:
            m = np.einsum("ij,ji->i", q.T @ np.linalg.inv(xm), q)
        except np.linalg.LinAlgError:  # collinear points
            break
        j = int(np.argmax(m))
        mx = m[j]
        step = (mx - d - 1) / ((d + 1) * (mx - 1))
        new_u = (1 - step) * u
        new_u[j] += step
        err = float(np.linalg.norm(new_u - u))
        u = new_u
    center = pts.T @ u
    cov = (pts.T * u) @ pts - np.outer(center, center)
    try:
        a_mat = np.linalg.inv(cov) / d
    except np.linalg.LinAlgError:
        a_mat = np.linalg.pinv(cov) / d
    uu, s, _vt = np.linalg.svd(a_mat)
    with np.errstate(divide="ignore"):
        # the smaller singular value of A is the major axis
        a, b = 1 / np.sqrt(s[1]), 1 / np.sqrt(s[0])
    angle = float(np.arctan2(uu[1, 1], uu[0, 1]))
    return float(center[0]), float(center[1]), float(a), float(b), angle


def enclose_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Smallest enclosing circle (Welzl), as ``ggforce``'s ``enclose_points``.

    The optimum is unique, so any correct algorithm agrees; this is the
    iterative move-to-front form.  Returns ``(x0, y0, r)``.
    """
    pts = np.unique(np.column_stack([x, y]).astype(float), axis=0)
    rng = np.random.default_rng(0)
    pts = pts[rng.permutation(len(pts))]

    def _two(a, b):
        c = (a + b) / 2
        return c, float(np.hypot(*(a - c)))

    def _three(a, b, c):
        ax_, ay_ = a
        bx, by = b
        cx, cy = c
        dd = 2 * (ax_ * (by - cy) + bx * (cy - ay_) + cx * (ay_ - by))
        if abs(dd) < 1e-14:
            pair = max(((a, b), (a, c), (b, c)), key=lambda p: np.hypot(*(p[0] - p[1])))
            return _two(*pair)
        ux = ((ax_**2 + ay_**2) * (by - cy) + (bx**2 + by**2) * (cy - ay_)
              + (cx**2 + cy**2) * (ay_ - by)) / dd
        uy = ((ax_**2 + ay_**2) * (cx - bx) + (bx**2 + by**2) * (ax_ - cx)
              + (cx**2 + cy**2) * (bx - ax_)) / dd
        cen = np.array([ux, uy])
        return cen, float(np.hypot(*(a - cen)))

    def _inside(c, r, p):
        return np.hypot(*(p - c)) <= r * (1 + 1e-12) + 1e-12

    if len(pts) == 1:
        return float(pts[0, 0]), float(pts[0, 1]), 0.0
    c, r = pts[0], 0.0
    for i in range(1, len(pts)):
        if _inside(c, r, pts[i]):
            continue
        c, r = pts[i], 0.0
        for j in range(i):
            if _inside(c, r, pts[j]):
                continue
            c, r = _two(pts[i], pts[j])
            for k in range(j):
                if not _inside(c, r, pts[k]):
                    c, r = _three(pts[i], pts[j], pts[k])
    return float(c[0]), float(c[1]), float(r)


def _convex_hull_cw(pts: np.ndarray) -> np.ndarray:
    """Indices of the convex hull, clockwise -- the order ``grDevices::chull`` gives."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(pts)
    return np.asarray(hull.vertices[::-1])  # scipy is counter-clockwise in 2-D


def _sq_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared distance from each row of ``p`` to segment ``ab``."""
    d = b - a
    dd = float(d @ d)
    if dd == 0:
        diff = p - a
        return np.einsum("ij,ij->i", diff, diff)
    t = np.clip(((p - a) @ d) / dd, 0.0, 1.0)
    proj = a + t[:, None] * d
    diff = p - proj
    return np.einsum("ij,ij->i", diff, diff)


def _orient(p, r, q) -> float:
    return float((r[1] - p[1]) * (q[0] - r[0]) - (r[0] - p[0]) * (q[1] - r[1]))


def concave_hull(points: np.ndarray, concavity: float = 2.0,
                 length_threshold: float = 0.0) -> np.ndarray:
    """Concave hull, a brute-force port of ``concaveman`` (Park & Oh 2012).

    ``ggforce::geom_mark_hull`` calls ``concaveman(points, concavity = 2)``.
    Starting from the clockwise convex hull, each edge is repeatedly bent
    inward to the nearest unused point, provided the new edges are no longer
    than ``edge / concavity``, the point is closer to this edge than to its
    neighbours, and no self-intersection results.  The R package does the
    nearest-point search with an R-tree; a vectorised scan finds the same
    point, so only ties can differ.  Returns the polygon vertices, open.
    """
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(pts) < 4:
        return pts
    hull_idx = list(_convex_hull_cw(pts))
    used = np.zeros(len(pts), dtype=bool)
    used[hull_idx] = True

    # doubly linked ring of point indices
    nxt: dict[int, int] = {}
    prv: dict[int, int] = {}
    for k, idx in enumerate(hull_idx):
        nxt[idx] = hull_idx[(k + 1) % len(hull_idx)]
        prv[idx] = hull_idx[k - 1]
    queue = list(hull_idx)
    sq_conc = concavity * concavity
    sq_thresh = length_threshold * length_threshold

    def _no_intersections(a_i: int, b_i: int) -> bool:
        a, b = pts[a_i], pts[b_i]
        e = hull_start
        while True:
            f = nxt[e]
            p1, q1 = e, f
            if p1 != b_i and q1 != a_i:
                if ((_orient(pts[p1], pts[q1], a) > 0) != (_orient(pts[p1], pts[q1], b) > 0)
                        and (_orient(a, b, pts[p1]) > 0) != (_orient(a, b, pts[q1]) > 0)):
                    return False
            e = f
            if e == hull_start:
                return True

    hull_start = hull_idx[0]
    while queue:
        node = queue.pop(0)
        a_i, b_i = node, nxt[node]
        a, b = pts[a_i], pts[b_i]
        sq_len = float((a - b) @ (a - b))
        if sq_len < sq_thresh:
            continue
        max_sq = sq_len / sq_conc
        free = np.flatnonzero(~used)
        if free.size == 0:
            break
        dist = _sq_seg_dist(pts[free], a, b)
        keep = dist <= max_sq
        cand = free[keep]
        cdist = dist[keep]
        order = np.argsort(cdist, kind="stable")
        prev_p, next_next_p = pts[prv[a_i]], pts[nxt[b_i]]
        chosen = -1
        for o in order:
            c = cand[o]
            d0 = _sq_seg_dist(pts[c][None, :], prev_p, a)[0]
            d1 = _sq_seg_dist(pts[c][None, :], b, next_next_p)[0]
            if (cdist[o] < d0 and cdist[o] < d1
                    and _no_intersections(a_i, c) and _no_intersections(b_i, c)):
                chosen = int(c)
                break
        if chosen < 0:
            continue
        pc = pts[chosen]
        if min(float((pc - a) @ (pc - a)), float((pc - b) @ (pc - b))) <= max_sq:
            nxt[a_i] = chosen
            prv[chosen] = a_i
            nxt[chosen] = b_i
            prv[b_i] = chosen
            used[chosen] = True
            queue.append(a_i)
            queue.append(chosen)

    ring = [hull_start]
    e = nxt[hull_start]
    while e != hull_start:
        ring.append(e)
        e = nxt[e]
    return pts[ring]


def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _miter_offset(poly: np.ndarray, delta: float, miter_limit: float = 1000.0) -> np.ndarray:
    """Offset every vertex along its bisector so each edge moves by ``delta``."""
    if len(poly) < 3 or delta == 0:
        return poly
    ccw = _signed_area(poly) > 0
    e_in = poly - np.roll(poly, 1, axis=0)
    e_out = np.roll(poly, -1, axis=0) - poly

    def _normal(e):
        n = np.column_stack([e[:, 1], -e[:, 0]]) if ccw else np.column_stack([-e[:, 1], e[:, 0]])
        ln = np.hypot(n[:, 0], n[:, 1])
        ln[ln == 0] = 1
        return n / ln[:, None]

    n1, n2 = _normal(e_in), _normal(e_out)
    denom = 1 + np.einsum("ij,ij->i", n1, n2)
    denom = np.maximum(denom, 2.0 / miter_limit**2)
    return poly + delta * (n1 + n2) / denom[:, None]


def _disk(r: float, n: int = 72) -> np.ndarray:
    t = 2 * np.pi * np.arange(n) / n
    return r * np.column_stack([np.cos(t), np.sin(t)])


def _minkowski_convex(poly: np.ndarray, r: float) -> np.ndarray:
    """``poly`` (convex) grown by a disk of radius ``r``: hull of the sum."""
    if r <= 0:
        return poly
    pts = (poly[:, None, :] + _disk(r)[None, :, :]).reshape(-1, 2)
    idx = _convex_hull_cw(pts)[::-1]
    return pts[idx]


def _round_offset_any(poly: np.ndarray, r: float) -> np.ndarray:
    """Round-joined outward offset of a possibly concave polygon.

    The boundary is the ``r``-level set of the distance to the polygon, traced
    with matplotlib's own contour engine -- exact up to the grid step, and
    with none of the self-intersection handling a vertex-offset needs.
    """
    import contourpy
    from matplotlib.path import Path

    step = r / 16.0
    lo = poly.min(axis=0) - r - 3 * step
    hi = poly.max(axis=0) + r + 3 * step
    gx = np.arange(lo[0], hi[0] + step, step)
    gy = np.arange(lo[1], hi[1] + step, step)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    dist = np.full(len(grid), np.inf)
    ring = np.vstack([poly, poly[:1]])
    for a, b in zip(ring[:-1], ring[1:], strict=True):
        dist = np.minimum(dist, _sq_seg_dist(grid, a, b))
    dist = np.sqrt(dist)
    inside = Path(poly).contains_points(grid)
    dist[inside] = 0.0
    gen = contourpy.contour_generator(gx, gy, dist.reshape(xx.shape))
    lines = gen.lines(r)
    if not lines:
        return _minkowski_convex(poly, r)
    best = max(lines, key=lambda a: len(a))
    return np.asarray(best)[:-1]


def offset_shape(poly: np.ndarray, expand: float, radius: float, *, convex: bool) -> np.ndarray:
    """``ggforce``'s ``makeContent.shape``: mitre by ``expand - radius``, then round by ``radius``.

    All lengths are in the same physical unit (mm).  Fewer than three distinct
    vertices are treated as ggforce treats points and lines: squared out by
    the mitre part, then rounded -- which for a point or segment is exactly
    the convex Minkowski sum used here.
    """
    poly = np.asarray(poly, dtype=float)
    delta = expand - radius
    if len(poly) < 3:
        if delta > 0:
            sq = np.array([[-delta, -delta], [delta, -delta], [delta, delta], [-delta, delta]])
            poly = (poly[:, None, :] + sq[None, :, :]).reshape(-1, 2)
        poly = poly if len(poly) >= 3 else np.vstack([poly, poly + 1e-9])
        hull = poly[_convex_hull_cw(poly)] if len(np.unique(poly, axis=0)) >= 3 else poly
        return _minkowski_convex(hull, max(radius, 1e-9))
    if delta != 0:
        poly = _miter_offset(poly, delta)
    if radius <= 0:
        return poly
    if convex:
        return _minkowski_convex(poly, radius)
    return _round_offset_any(poly, radius)


# --------------------------------------------------------------------------- #
# axis limits
# --------------------------------------------------------------------------- #
def expand_range(lo: float, hi: float, mult: float = 0.05) -> tuple[float, float]:
    """ggplot's default continuous expansion: ``mult`` of the range on each side.

    SCP sets ``scale_x_continuous(limits = range(...))`` (``SCP-plot.R:1636``),
    which keeps the default ``expansion(mult = 0.05)``, so its panels reach
    5 % past the data and no edge point is cut in half.  A zero-width range
    widens by 0.5 each way, as ``scales::expand_range4`` does.
    """
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return lo, hi
    if hi == lo:
        return lo - 0.5, hi + 0.5
    d = (hi - lo) * mult
    return lo - d, hi + d


# --------------------------------------------------------------------------- #
# graph edges
# --------------------------------------------------------------------------- #
def rescale(v: np.ndarray, to: tuple[float, float]) -> np.ndarray:
    """``scales::rescale``: linear map of ``range(v)`` onto ``to``; a constant maps to the middle."""
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return v
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    if hi == lo:
        return np.full(v.shape, (to[0] + to[1]) / 2)
    return to[0] + (v - lo) / (hi - lo) * (to[1] - to[0])


def graph_segments(graph, order: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Edges of ``graph`` among the cells ``order`` (positions into the graph), R-style.

    R subsets the adjacency to the panel's cells **in their drawing order**,
    blanks zeros and the strict upper triangle, and melts what is left.  So
    each stored entry ``(i, j)`` survives when ``i`` is drawn at or after
    ``j``; for a symmetric graph that is every undirected edge once (plus the
    diagonal, which is kept and is drawn as nothing).  Done on COO triplets --
    R densifies the whole matrix, which is O(n^2).

    Returns ``(row_pos, col_pos, value)``, positions into ``order``.
    """
    g = sp.csr_matrix(graph)
    order = np.asarray(order, dtype=np.int64)
    sub = g[order][:, order].tocoo()
    keep = (sub.data != 0) & np.isfinite(sub.data) & (sub.row >= sub.col)
    return sub.row[keep], sub.col[keep], np.asarray(sub.data[keep], dtype=float)


# --------------------------------------------------------------------------- #
# velocity on a grid
# --------------------------------------------------------------------------- #
def _knn_rows(D: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the ``k`` smallest entries in R's ``order(x)[1:k]`` order.

    ``order`` is stable, so ties go to the lower index; that is reproduced
    exactly, including a tie that straddles the ``k``-th place.
    """
    m, n = D.shape
    k = min(k, n)
    if k == n:
        idx = np.argsort(D, axis=1, kind="stable")
        return idx, np.take_along_axis(D, idx, 1)
    part = np.argpartition(D, k - 1, axis=1)[:, :k]
    pd_ = np.take_along_axis(D, part, 1)
    kth = pd_.max(axis=1)
    order = np.lexsort((part, pd_), axis=1)
    idx = np.take_along_axis(part, order, 1)
    dist = np.take_along_axis(pd_, order, 1)
    # rows where a tie at the k-th distance leaves the pick ambiguous: redo exactly
    for r in np.flatnonzero((D <= kth[:, None]).sum(axis=1) > k):
        cand = np.flatnonzero(D[r] <= kth[r])
        cand = cand[np.argsort(D[r, cand], kind="stable")][:k]
        idx[r], dist[r] = cand, D[r, cand]
    return idx, dist


def compute_velocity_on_grid(
    X_emb: np.ndarray,
    V_emb: np.ndarray,
    *,
    density: float = 1.0,
    smooth: float = 0.5,
    n_neighbors: int | None = None,
    min_mass: float = 1.0,
    scale: float = 1.0,
    adjust_for_stream: bool = False,
    cutoff_perc: float = 5.0,
):
    """A cell velocity field smoothed onto a regular grid.

    Port of SCP's ``compute_velocity_on_grid`` (``SCP-plot.R:6963-7039``) by
    Hao Zhang, https://github.com/zhanghao-njmu/SCP, itself a transliteration
    of scvelo's function of the same name.  It is reimplemented in numpy rather
    than delegated to scvelo because SCP's version differs from scvelo's in
    ways that change the picture:

    * no +-1 % range padding: the grid is ``ceil(50 * density)`` points from
      the minimum to the maximum of each axis;
    * ``n_neighbors`` defaults to ``ceil(n_obs / 50)`` (scvelo: ``int``);
    * no autoscaling; ``scale`` multiplies the grid vectors (grid mode only);
    * ``min_mass`` keeps grid points whose total Gaussian weight exceeds
      ``min_mass * quantile(p_mass, 0.99) / 100`` (grid mode), or is mapped to
      ``10 ** (min_mass - 6)`` once (stream mode; scvelo's own stream branch
      applies that mapping internally, so delegating applied it twice).

    Each grid point takes its ``n_neighbors`` nearest cells (ties to the lower
    index, as R's ``order``), weights them by a normal density with
    ``sd = mean(grid step) * smooth``, and averages their velocities, dividing
    by the total weight floored at 1.

    Returns ``(X_grid, V_grid)``.  Grid mode: ``(n, 2)`` positions and vectors
    of the points that pass the mass filter.  Stream mode
    (``adjust_for_stream=True``): ``X_grid`` is ``(2, ns)`` -- the x and y grid
    lines -- and ``V_grid`` is ``(2, ns, ns)`` indexed ``[component, y, x]``,
    the layout ``matplotlib.streamplot`` takes; points below the mass floor or
    the ``cutoff_perc`` percentile of neighbour speed are NaN in the first
    component only, as in R and scvelo.
    """
    X = np.asarray(X_emb, dtype=float)
    V = np.asarray(V_emb, dtype=float)
    n_obs, n_dim = X.shape
    k = int(n_neighbors) if n_neighbors is not None else math.ceil(n_obs / 50)
    k = max(1, min(k, n_obs))

    n_pts = math.ceil(50 * density)
    grs = [np.linspace(np.nanmin(X[:, d]), np.nanmax(X[:, d]), n_pts) for d in range(n_dim)]
    # expand.grid: the first axis varies fastest
    X_grid = np.column_stack([g.ravel(order="F") for g in np.meshgrid(*grs, indexing="ij")])

    # nearest cells per grid point, in chunks so the distance block stays small
    m = len(X_grid)
    neighbors = np.empty((m, k), dtype=np.intp)
    dists = np.empty((m, k))
    step = max(1, int(2e7 // max(n_obs * n_dim, 1)))
    for s in range(0, m, step):
        diff = X_grid[s:s + step, None, :] - X[None, :, :]
        D = np.sqrt(np.einsum("gcd,gcd->gc", diff, diff))
        neighbors[s:s + step], dists[s:s + step] = _knn_rows(D, k)

    sd = float(np.mean([g[1] - g[0] for g in grs])) * smooth
    weight = np.exp(-0.5 * (dists / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
    p_mass = weight.sum(axis=1)
    p_mass_v = np.maximum(p_mass, 1.0)
    neighbors_emb = V[neighbors]  # (m, k, n_dim)
    V_grid = (neighbors_emb * weight[:, :, None]).sum(axis=1) / p_mass_v[:, None]

    if adjust_for_stream:
        X_grid = np.vstack([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])])
        ns = int(math.floor(math.sqrt(len(V_grid))))
        V_grid = V_grid.T.reshape(2, ns, ns)  # C order: [component, y, x]
        mass = np.sqrt((V_grid ** 2).sum(axis=0))
        floor = 10 ** (min_mass - 6)  # default min_mass = 1 means 1e-5
        floor = min(floor, float(np.nanmax(mass)) * 0.9)
        cutoff = mass < floor
        length = np.abs(neighbors_emb).mean(axis=1).sum(axis=1).reshape(ns, ns)
        cutoff = cutoff | (length < np.quantile(length, cutoff_perc / 100))
        V_grid = V_grid.copy()
        V_grid[0][cutoff] = np.nan
        return X_grid, V_grid

    keep = p_mass > min_mass * np.quantile(p_mass, 0.99) / 100
    return X_grid[keep], V_grid[keep] * scale
