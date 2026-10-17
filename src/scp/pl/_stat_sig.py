"""The significance layer behind ``feature_stat_plot``: tests and their labels.

SCP adds p-values with ``ggpubr::stat_compare_means``, which in turn uses
``compare_means`` (for labels placed per group) or ``ggsignif::geom_signif``
(for brackets).  Both end in plain calls to base R, and this module reproduces
those calls rather than the packages around them:

* ``wilcox.test`` — exact when both samples are under 50 and there are no ties,
  otherwise the normal approximation with the tie correction and a continuity
  correction of 0.5;
* ``t.test`` — Welch's unequal-variance test (R's default);
* ``kruskal.test`` — with the tie correction;
* ``anova(lm(y ~ g))`` — the one-way F test.

and the three ways a p-value becomes text:

* ``compare_means``: ``format.pval(p, digits = 2)`` — which formats the whole
  vector of p-values *jointly*, as R's ``format()`` does — and ``symnum()`` with
  the cut-points ``0, 1e-4, 1e-3, 0.01, 0.05, 1`` (intervals closed on the
  right, so ``p = 0.05`` is ``"*"``);
* ``geom_signif``: ``sprintf("%.2g", p)`` and a strict ``p < cut`` map, so
  ``p = 0.05`` is ``"ns"`` there.  The two packages genuinely disagree at the
  boundaries and SCP inherits both behaviours; so does this module.

Every function is pinned against R in
``tests/fixtures/parity/feature_stat/stat_tests.json`` (written by
``tools/parity/feature_stat_tests.R``).

Where R stops with an error (fewer than two observations for a t-test, a single
group for Kruskal-Wallis) these functions return ``nan`` instead, and the
caller labels the comparison the way the R packages label a missing p-value.

Port of the statistics used by ``ExpressionStatPlot`` (``SCP-plot.R:3723``) —
Hao Zhang, https://github.com/zhanghao-njmu/SCP.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import ROUND_HALF_EVEN, Decimal
from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy import stats as _st

__all__ = [
    "resolve_method",
    "wilcox_test",
    "t_test",
    "kruskal_test",
    "anova_test",
    "two_sample_p",
    "multigroup_p",
    "pairwise_p",
    "format_real",
    "format_pval",
    "symnum_signif",
    "signif_strict",
    "format_2g",
]

#: ggpubr's ``.method_info``: the accepted spellings and the canonical name.
_METHOD_ALIASES = {
    "t": "t.test", "t.test": "t.test", "student": "t.test",
    "wiloxon": "wilcox.test", "wilcox": "wilcox.test", "wilcox.test": "wilcox.test",
    "anova": "anova", "aov": "anova",
    "kruskal": "kruskal.test", "kruskal.test": "kruskal.test",
}
PAIRWISE_METHODS = ("t.test", "wilcox.test")
MULTIGROUP_METHODS = ("anova", "kruskal.test")
_EPS = float(np.finfo(float).eps)


def resolve_method(method: str) -> str:
    """Canonical test name for any spelling ggpubr accepts."""
    try:
        return _METHOD_ALIASES[method]
    except KeyError:
        raise ValueError(
            f"unsupported method {method!r}; allowed: {sorted(_METHOD_ALIASES)}"
        ) from None


def _finite(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


# --------------------------------------------------------------------------- #
# Wilcoxon rank-sum
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=256)
def _wilcox_counts(m: int, n: int) -> tuple[int, ...]:
    """Number of arrangements giving each value of W (0..m*n), exactly.

    The coefficients of the Gaussian binomial ``[m+n choose m]_q``.  Integers,
    not floats: the counts reach 1e28 at m = n = 49 and a float recursion with
    subtraction would lose them.
    """
    # prod_{i=1..m} (1 - q^(n+i)) / (1 - q^i), one factor at a time; each
    # partial product is itself a polynomial, so every division is exact.
    poly = np.zeros(m * n + m + 1, dtype=object)
    poly[:] = 0
    poly[0] = 1
    for i in range(1, m + 1):
        a = n + i
        poly[a:] = poly[a:] - poly[:-a]          # times (1 - q^a)
        for r in range(i):                       # divided by (1 - q^i)
            poly[r::i] = np.cumsum(poly[r::i])
    return tuple(int(c) for c in poly[: m * n + 1])


def _pwilcox(q: float, m: int, n: int) -> float:
    """R's ``pwilcox(q, m, n)``: P(W <= q)."""
    q = math.floor(q + 1e-7)
    if q < 0:
        return 0.0
    if q >= m * n:
        return 1.0
    counts = _wilcox_counts(m, n)
    return float(Fraction(sum(counts[: q + 1]), math.comb(m + n, n)))


def wilcox_test(x, y, *, exact: bool | None = None, correct: bool = True) -> float:
    """Two-sided p-value of R's ``wilcox.test(x, y)``.

    ``exact=None`` is R's default: exact when both samples have fewer than 50
    observations, unless there are ties, in which case R warns and falls back
    to the normal approximation.  Non-finite values are dropped first, as R
    does.  Returns ``nan`` where R stops (an empty sample).
    """
    x, y = _finite(x), _finite(y)
    nx, ny = x.size, y.size
    if nx < 1 or ny < 1:
        return float("nan")
    r = _st.rankdata(np.concatenate([x, y]))
    w = float(r[:nx].sum() - nx * (nx + 1) / 2)
    if exact is None:
        exact = nx < 50 and ny < 50
    _, tie_counts = np.unique(r, return_counts=True)
    ties = tie_counts.size != r.size
    if exact and not ties:
        if w > nx * ny / 2:
            p = 1.0 - _pwilcox(w - 1, nx, ny)
        else:
            p = _pwilcox(w, nx, ny)
        return min(2.0 * p, 1.0)
    z = w - nx * ny / 2
    nt = tie_counts.astype(float)
    sigma = math.sqrt(
        (nx * ny / 12)
        * ((nx + ny + 1) - float((nt**3 - nt).sum()) / ((nx + ny) * (nx + ny - 1)))
    )
    correction = float(np.sign(z)) * 0.5 if correct else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        zz = np.float64(z - correction) / np.float64(sigma)
    if not np.isfinite(zz):
        return float("nan")
    return float(2.0 * min(_st.norm.cdf(zz), _st.norm.sf(zz)))


# --------------------------------------------------------------------------- #
# t, Kruskal-Wallis, ANOVA
# --------------------------------------------------------------------------- #
def t_test(x, y) -> float:
    """Two-sided p-value of R's ``t.test(x, y)`` — Welch, unequal variances.

    Returns ``nan`` where R stops: fewer than two observations in either sample,
    or "data are essentially constant".
    """
    x, y = _finite(x), _finite(y)
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return float("nan")
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sx2, sy2 = vx / nx, vy / ny
    stderr = math.sqrt(sx2 + sy2)
    if stderr < 10 * _EPS * max(abs(mx), abs(my)):
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        df = stderr**4 / (sx2**2 / (nx - 1) + sy2**2 / (ny - 1))
        t = (mx - my) / stderr
    if not (np.isfinite(t) and np.isfinite(df)):
        return float("nan")
    return float(2.0 * _st.t.sf(abs(t), df))


def _clean_groups(groups: Sequence) -> list[np.ndarray]:
    return [g for g in (_finite(g) for g in groups) if g.size > 0]


def kruskal_test(groups: Sequence) -> float:
    """R's ``kruskal.test`` over a list of samples, with the tie correction.

    Empty samples are dropped (R's ``factor()`` drops unused levels); fewer than
    two non-empty samples, where R stops, gives ``nan``.
    """
    gs = _clean_groups(groups)
    if len(gs) < 2:
        return float("nan")
    x = np.concatenate(gs)
    n = x.size
    r = _st.rankdata(x)
    stat, start = 0.0, 0
    for g in gs:
        rs = r[start:start + g.size]
        stat += rs.sum() ** 2 / g.size
        start += g.size
    _, ties = np.unique(x, return_counts=True)
    ties = ties.astype(float)
    denom = 1.0 - float((ties**3 - ties).sum()) / (n**3 - n)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = (12.0 * stat / (n * (n + 1)) - 3.0 * (n + 1)) / denom
    if not np.isfinite(h):
        return float("nan")
    return float(_st.chi2.sf(h, len(gs) - 1))


def anova_test(groups: Sequence) -> float:
    """``anova(lm(y ~ g))$"Pr(>F)"[1]`` — the one-way F test."""
    gs = _clean_groups(groups)
    k = len(gs)
    if k < 2:
        return float("nan")
    x = np.concatenate(gs)
    n = x.size
    if n <= k:
        return float("nan")
    grand = x.mean()
    ssb = sum(g.size * (g.mean() - grand) ** 2 for g in gs)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in gs)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (ssb / (k - 1)) / (ssw / (n - k))
    if np.isnan(f):
        return float("nan")
    return float(_st.f.sf(f, k - 1, n - k))


def two_sample_p(x, y, method: str) -> float:
    """p-value of a pairwise ``method`` (``t.test`` or ``wilcox.test``)."""
    method = resolve_method(method)
    if method == "t.test":
        return t_test(x, y)
    if method == "wilcox.test":
        return wilcox_test(x, y)
    raise ValueError(f"{method!r} is not a two-sample test; use one of {PAIRWISE_METHODS}")


def multigroup_p(groups: Sequence, method: str) -> float:
    """p-value of an omnibus ``method`` (``kruskal.test`` or ``anova``)."""
    method = resolve_method(method)
    if method == "kruskal.test":
        return kruskal_test(groups)
    if method == "anova":
        return anova_test(groups)
    raise ValueError(f"{method!r} is not a multi-group test; use one of {MULTIGROUP_METHODS}")


def pairwise_p(groups: Sequence, method: str) -> list[tuple[int, int, float]]:
    """``pairwise.<method>(..., p.adjust.method = "none")`` as ggpubr reads it.

    Pairs come back in ``compare_means`` order — by first group, then second —
    and only for groups that have data, since R's ``factor()`` drops the rest.
    ``pairwise.t.test`` is called with ``pool.sd = FALSE``, so each pair is an
    independent Welch test.
    """
    idx = [i for i, g in enumerate(groups) if _finite(g).size > 0]
    out = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            out.append((i, j, two_sample_p(groups[i], groups[j], method)))
    return out


# --------------------------------------------------------------------------- #
# p-value text
# --------------------------------------------------------------------------- #
def _scientific(x: float, digits: int) -> tuple[int, int, bool]:
    """R's ``scientific()`` (``format.c``): exponent, significant digits, widening."""
    if x == 0.0:
        return 0, 1, False
    r = abs(x)
    kp = math.floor(math.log10(r)) - digits + 1
    # exact decimal scaling stands in for R's long-double arithmetic
    r_prec = Decimal(r) * (Decimal(10) ** (-kp))
    if r_prec < Decimal(10) ** (digits - 1):
        r_prec *= 10
        kp -= 1
    alpha = int(r_prec.to_integral_value(rounding=ROUND_HALF_EVEN))
    nsig = digits
    for _ in range(digits):
        if alpha % 10 == 0:
            nsig -= 1
            alpha //= 10
        else:
            break
    if nsig == 0 and digits > 0:
        nsig = 1
        kp += 1
    kpower = kp + digits - 1
    rgt = min(max(digits - kpower, 0), 22)
    fuzz = 0.5 / 10**rgt
    widens = 0 < kpower <= 22 and r < 10**kpower - fuzz
    return kpower, nsig, widens


def format_real(values: Sequence[float], digits: int) -> list[str]:
    """R's ``format(x, digits = digits)`` for a vector of finite numbers.

    The vector is formatted *jointly*: one choice of fixed or scientific
    notation and one number of decimals for every element, with the fixed
    strings right-justified to a common width.  That is why
    ``format(c(0.5, 0.012), digits = 2)`` is ``"0.500" "0.012"``.
    """
    xs = [float(v) for v in values]
    if not xs:
        return []
    neg = any(v < 0 for v in xs)
    rgt = mxsl = mxns = -(10**9)
    mxl = -(10**9)
    mnl = 10**9
    for v in xs:
        kpower, nsig, widens = _scientific(v, digits)
        left = kpower + 1
        if widens:
            left -= 1
        sleft = (v < 0) + (1 if left <= 0 else left)
        right = nsig - left
        rgt, mxl, mnl = max(rgt, right), max(mxl, left), min(mnl, left)
        mxsl, mxns = max(mxsl, sleft), max(mxns, nsig)
    if mxl < 0:
        mxsl = 1 + neg
    rgt = max(rgt, 0)
    w_fixed = neg + mxsl + rgt + (rgt != 0)
    e = 2 if (mxl > 100 or mnl <= -99) else 1
    d = mxns - 1
    w_sci = neg + (d > 0) + d + 4 + e
    if w_fixed <= w_sci:  # R's scipen defaults to 0
        return [f"{v:>{w_fixed}.{rgt}f}" for v in xs]
    out = []
    for v in xs:
        s = f"{v:.{d}e}"
        mant, _, ex = s.partition("e")
        sign, ex = ex[0], ex[1:].lstrip("0")
        out.append(f"{mant}e{sign}{ex.rjust(e + 1, '0')}".rjust(w_sci))
    return out


def format_pval(pvalues: Sequence[float], digits: int = 2) -> list[str]:
    """R's ``format.pval(pv, digits = 2)``, as ``compare_means`` calls it.

    p-values from 1e-4 up (with ``digits > 1``) are formatted together, smaller
    ones together, and anything under machine epsilon becomes ``"<2e-16"``.
    """
    pv = [float(p) for p in pvalues]
    out = ["NA"] * len(pv)
    ok = [i for i, p in enumerate(pv) if not math.isnan(p)]
    tiny = [i for i in ok if pv[i] < _EPS]
    rest = [i for i in ok if pv[i] >= _EPS]
    rr: dict[int, str] = {}
    if rest:
        expo = {i: math.floor(math.log10(pv[i] if pv[i] > 0 else 1e-50)) for i in rest}
        fixp = [i for i in rest if expo[i] >= -3 or (expo[i] == -4 and digits > 1)]
        sci = [i for i in rest if i not in fixp]
        for block in (fixp, sci):
            if block:
                for i, s in zip(block, format_real([pv[i] for i in block], digits), strict=True):
                    rr[i] = s
        for i, s in rr.items():
            out[i] = s
    if tiny:
        dig = max(1, digits - 2)
        if rest:
            nc = max(len(s) for s in rr.values())
            if dig > 1 and dig + 6 > nc:
                dig = max(1, nc - 7)
            sep = "" if (dig == 1 and nc <= 6) else " "
        else:
            sep = "" if dig == 1 else " "
        eps_s = format_real([_EPS], dig)[0]
        for i in tiny:
            out[i] = f"<{sep}{eps_s}"
    return out


_SYMNUM_CUTS = (0.0, 1e-4, 1e-3, 1e-2, 0.05, 1.0)
_SYMNUM_SYMBOLS = ("****", "***", "**", "*", "ns")


def symnum_signif(pvalues: Sequence[float]) -> list[str]:
    """``symnum(p, cutpoints = c(0, 1e-4, 1e-3, .01, .05, 1), ...)`` — compare_means.

    Intervals are closed on the right and the lowest is closed on the left, so
    ``p = 0.05`` maps to ``"*"``.  A missing p-value is R's ``"?"``.
    """
    out = []
    for p in pvalues:
        p = float(p)
        if math.isnan(p) or p < 0 or p > 1:
            out.append("?")
            continue
        k = int(np.searchsorted(_SYMNUM_CUTS[1:], p, side="left"))
        out.append(_SYMNUM_SYMBOLS[min(k, 4)])
    return out


_STRICT_MAP = (("****", 1e-4), ("***", 1e-3), ("**", 1e-2), ("*", 0.05), ("ns", math.inf))


def signif_strict(p: float) -> str:
    """ggsignif's ``map_signif_level``: the smallest cut strictly above ``p``.

    So ``p = 0.05`` is ``"ns"`` here, unlike :func:`symnum_signif`; a missing
    p-value is ggsignif's ``"NS."``.
    """
    p = float(p)
    if math.isnan(p):
        return "NS."
    for sym, cut in _STRICT_MAP:
        if cut > p:
            return sym
    return "NS."


def format_2g(p: float) -> str:
    """ggsignif's p-value text: ``sprintf("%.2g", p)``, or ``"p < 2.22e-16"``."""
    p = float(p)
    if math.isnan(p):
        return "NA"
    if p < _EPS:
        return f"p < {_EPS:.2e}"
    return f"{p:.2g}"
