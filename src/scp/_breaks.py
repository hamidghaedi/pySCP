"""ggplot2's default axis breaks.

A continuous ggplot2 scale with no ``breaks`` argument asks
``scales::extended_breaks()`` for about five breaks, which is
``labeling::extended`` -- the Talbot, Lin and Hanrahan (2010) search over
"nice" step sizes scored for simplicity, coverage, density and legibility.
matplotlib's ``AutoLocator`` aims at up to nine ticks and so packs five-digit
rank labels edge to edge on a panel a few inches wide; these breaks are the
ones R would draw.

Port of ``labeling::extended`` (Justin Talbot, MIT licence) as
``scales::extended_breaks`` calls it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = ["extended_breaks"]

_Q = (1.0, 5.0, 2.0, 2.5, 4.0, 3.0)
_W = (0.25, 0.2, 0.5, 0.05)
_EPS = np.finfo(float).eps * 100


def _r_mod(a: float, b: float) -> float:
    """R's ``%%`` on doubles: the result takes the sign of ``b``."""
    return a - math.floor(a / b) * b


def _simplicity(i: int, n: int, j: int, lmin: float, lmax: float, lstep: float) -> float:
    m = _r_mod(lmin, lstep)
    v = 1.0 if ((m < _EPS or lstep - m < _EPS) and lmin <= 0 <= lmax) else 0.0
    return 1 - i / (n - 1) - j + v


def _simplicity_max(i: int, n: int, j: int) -> float:
    return 1 - i / (n - 1) - j + 1


def _coverage(dmin: float, dmax: float, lmin: float, lmax: float) -> float:
    r = dmax - dmin
    return 1 - 0.5 * ((dmax - lmax) ** 2 + (dmin - lmin) ** 2) / ((0.1 * r) ** 2)


def _coverage_max(dmin: float, dmax: float, span: float) -> float:
    r = dmax - dmin
    if span > r:
        half = (span - r) / 2
        return 1 - 0.5 * (2 * half**2) / ((0.1 * r) ** 2)
    return 1.0


def _density(k: int, m: int, dmin: float, dmax: float, lmin: float, lmax: float) -> float:
    r = (k - 1) / (lmax - lmin)
    rt = (m - 1) / (max(lmax, dmax) - min(dmin, lmin))
    return 2 - max(r / rt, rt / r)


def _density_max(k: int, m: int) -> float:
    return 2 - (k - 1) / (m - 1) if k >= m else 1.0


def extended_breaks(
    dmin: float,
    dmax: float,
    m: int = 5,
    *,
    q: Sequence[float] = _Q,
    only_loose: bool = False,
    w: Sequence[float] = _W,
) -> np.ndarray:
    """``labeling::extended(dmin, dmax, m)``: about ``m`` nice breaks covering
    ``[dmin, dmax]``.

    As in R, the breaks may reach past the data range (ggplot then drops the
    ones outside the scale limits; :func:`breaks_within` does that).
    """
    dmin, dmax = float(dmin), float(dmax)
    if not (math.isfinite(dmin) and math.isfinite(dmax)):
        return np.array([])
    if dmin > dmax:
        dmin, dmax = dmax, dmin
    if dmax - dmin < _EPS:
        return np.linspace(dmin, dmax, m)
    n = len(q)
    best_score = -2.0
    best: tuple[float, float, float] | None = None
    j = 1
    while True:
        stop = False
        for i, qq in enumerate(q):
            sm = _simplicity_max(i, n, j)
            if w[0] * sm + w[1] + w[2] + w[3] < best_score:
                stop = True
                break
            k = 2
            while True:
                dm = _density_max(k, m)
                if w[0] * sm + w[1] + w[2] * dm + w[3] < best_score:
                    break
                delta = (dmax - dmin) / (k + 1) / j / qq
                z = math.ceil(math.log10(delta))
                while True:
                    step = j * qq * 10.0**z
                    cm = _coverage_max(dmin, dmax, step * (k - 1))
                    if w[0] * sm + w[1] * cm + w[2] * dm + w[3] < best_score:
                        break
                    min_start = math.floor(dmax / step) * j - (k - 1) * j
                    max_start = math.ceil(dmin / step) * j
                    if min_start > max_start:
                        z += 1
                        continue
                    for start in range(int(min_start), int(max_start) + 1):
                        lmin = start * (step / j)
                        lmax = lmin + step * (k - 1)
                        s = _simplicity(i, n, j, lmin, lmax, step)
                        c = _coverage(dmin, dmax, lmin, lmax)
                        g = _density(k, m, dmin, dmax, lmin, lmax)
                        score = w[0] * s + w[1] * c + w[2] * g + w[3]
                        if score > best_score and (not only_loose or (lmin <= dmin and lmax >= dmax)):
                            best_score = score
                            best = (lmin, lmax, step)
                    z += 1
                k += 1
        if stop:
            break
        j += 1
    assert best is not None
    lmin, lmax, step = best
    # seq(from = lmin, to = lmax, by = step)
    count = int(math.floor((lmax - lmin) / step + 1e-10)) + 1
    return lmin + step * np.arange(count)


def breaks_within(lo: float, hi: float, m: int = 5) -> np.ndarray:
    """The breaks ggplot draws on a continuous scale with limits ``[lo, hi]``:
    ``extended_breaks`` over the limits, minus any outside them."""
    br = extended_breaks(lo, hi, m)
    tol = 1e-10 * max(abs(hi - lo), 1.0)
    return br[(br >= lo - tol) & (br <= hi + tol)]
