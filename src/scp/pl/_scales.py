"""Breaks and labels of ggplot2's log-transformed scales.

* :func:`log_breaks` -- ``scales::log_breaks()``, the breaks of a ``log10`` /
  ``log2`` transformed scale (``scale_y_continuous(trans = "log10")``); it
  falls back to ``scales::extended_breaks`` (:mod:`scp._breaks`) as R does;
* :func:`format_breaks` -- how such a scale labels them.

Breaks may lie outside the range they were given, as the R functions return
them; ggplot then drops those outside the limits.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .._breaks import extended_breaks

__all__ = ["log_breaks", "format_breaks"]

def _log(x, base: float):
    """R's ``log(x, base)``, which calls ``log10`` / ``log2`` for those bases:
    ``log(0.01) / log(10)`` is not exactly -2, and the break counts compare
    against powers of the base."""
    if base == 10:
        return np.log10(x)
    if base == 2:
        return np.log2(x)
    return np.log(x) / np.log(base)


def _log_sub_breaks(rng: tuple[float, float], n: int, base: float) -> list[float]:
    lo, hi = math.floor(rng[0]), math.ceil(rng[1])
    if base <= 2:
        return [base ** e for e in range(lo, hi + 1)]
    steps = [1.0]

    def delta(x: float) -> float:
        pts = _log(np.sort([x, *steps, base]), base)
        return float(np.min(np.diff(pts)))

    candidate = [float(c) for c in range(1, int(base) + 1) if 1 < c < base]
    breaks: list[float] = []
    relevant = 0
    while candidate:
        scores = [delta(c) for c in candidate]
        best = int(np.argmax(scores))
        steps.append(candidate.pop(best))
        breaks = [base ** e * s for s in steps for e in range(lo, hi + 1)]
        relevant = sum(base ** rng[0] <= b <= base ** rng[1] for b in breaks)
        if relevant >= n - 2:
            break
    if relevant >= n - 2:
        breaks = sorted(breaks)
        inside = [i for i, b in enumerate(breaks) if base ** rng[0] <= b <= base ** rng[1]]
        lower = max(inside[0] - 1, 0)
        upper = min(inside[-1] + 1, len(breaks) - 1)
        return breaks[lower:upper + 1]
    return [float(b) for b in extended_breaks(base ** rng[0], base ** rng[1], n)]


def log_breaks(lo: float, hi: float, n: int = 5, base: float = 10.0) -> list[float]:
    """``scales::log_breaks(n, base)(c(lo, hi))``: powers of ``base``, thinned
    or (when fewer than ``n - 2`` fall in range) filled in with the integer
    multiples that split the decades most evenly."""
    if not (lo > 0 and hi > 0 and np.isfinite(lo) and np.isfinite(hi)):
        return []
    rng = (float(_log(lo, base)), float(_log(hi, base)))
    emin, emax = math.floor(rng[0]), math.ceil(rng[1])
    if emax == emin:
        return [base ** emin]
    by = (emax - emin) // n + 1

    def powers(step: int) -> list[float]:
        return [base ** e for e in range(emin, emax + 1, step)]

    def n_relevant(bs: Sequence[float]) -> int:
        return sum(base ** rng[0] <= b <= base ** rng[1] for b in bs)

    breaks = powers(by)
    if n_relevant(breaks) >= n - 2:
        return breaks
    while by > 1:
        by -= 1
        breaks = powers(by)
        if n_relevant(breaks) >= n - 2:
            return breaks
    return _log_sub_breaks(rng, n, base)


def format_breaks(breaks: Sequence[float]) -> list[str]:
    """R's ``format(x, trim = TRUE)`` on a break vector: one common number of
    decimals, the fewest that tell every break apart at 15 significant
    digits (``c(0.5, 1, 3, 5)`` -> ``"0.5" "1.0" "3.0" "5.0"``)."""
    vals = [float(b) for b in breaks]
    if not vals:
        return []
    decimals = 0
    for v in vals:
        s = f"{v:.15g}"
        if "e" in s:
            return [f"{x:g}" for x in vals]
        if "." in s:
            decimals = max(decimals, len(s.split(".")[1]))
    return [f"{v:.{decimals}f}" for v in vals]
