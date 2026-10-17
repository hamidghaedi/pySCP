"""Query-vs-reference similarity for ``cell_cor_heatmap`` — proxyC, ported.

``CellCorHeatmap`` (``SCP-plot.R`` 10113) gets its matrix from
``RunKNNPredict(nn_method = "raw")`` (``SCP-cell_annotation.R`` 266-410), which
calls ``proxyC::simil`` / ``proxyC::dist`` between the reference rows and the
query rows and then shows

* ``simil(ref, query)`` for a similarity metric, titled ``"<Metric> similarity"``;
* ``1 - d / max(d)`` for a distance, titled ``"1-dist[<m>]/max(dist[<m>])"``;

with infinities clamped to the largest finite magnitude and NaN set to 0.
Every formula below was pinned against proxyC 0.5 on SCP's own centroids
(``tests/fixtures/parity/heatmap/cell_cor_metrics.csv``).

Binary metrics (``jaccard``, ``dice``, ``hamann``, ``simple matching``,
``faith``) treat every non-zero value as 1.  ``chisquared`` is Pearson's
chi-squared of the 2 x n table ``[x, y]`` divided by its total; ``kullback`` is
``sum(p * log(p / q))`` on the rows normalised to proportions (``p`` from the
reference).  Both are NaN for any pair with a zero entry -- proxyC's
``use_nan`` behaviour with ``smooth = 0``, which on log-expression centroids
blanks most of the matrix; SCP passes no smoothing, so neither does this.

Two R bugs are fixed rather than reproduced:

* SCP lists ``"hamman"``, proxyC spells it ``"hamann"``, so the SCP metric
  errors; both spellings are accepted here;
* proxyC's ``euclidean`` expands ``|x|^2 + |y|^2 - 2xy`` and loses identical
  pairs to cancellation (a self-distance of 1e-7, or ``sqrt`` of a negative
  number, i.e. NaN -> similarity 0 on the diagonal); the direct
  ``sqrt(sum((x - y)^2))`` is used, so the two agree to ~1e-7 except on those
  cells.  ``minkowski`` (``p = 2``) is computed directly in proxyC and agrees.

Upstream: Hao Zhang, SCP (GPL-3), https://github.com/zhanghao-njmu/SCP; Kohei
Watanabe, proxyC (GPL-3), https://github.com/koheiw/proxyC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .enrichment import capitalize
from .matrix import clean_nonfinite

__all__ = ["SIMIL_METHODS", "DIST_METHODS", "proxy_simil", "proxy_dist", "cell_similarity"]

SIMIL_METHODS = (
    "cosine", "pearson", "spearman", "correlation", "jaccard", "ejaccard", "dice", "edice",
    "hamann", "simple matching", "faith",
)
DIST_METHODS = (
    "euclidean", "chisquared", "kullback", "manhattan", "maximum", "canberra", "minkowski",
    "hamming",
)


def _norm_rows(A: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(A, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return A / np.where(n == 0, np.nan, n)


def proxy_simil(x: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
    """``proxyC::simil(x, y, method, use_nan = TRUE)``: rows of ``x`` vs rows of ``y``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if method == "hamman":
        method = "hamann"
    if method == "cosine":
        return _norm_rows(x) @ _norm_rows(y).T
    if method == "correlation":
        return _norm_rows(x - x.mean(axis=1, keepdims=True)) @ \
            _norm_rows(y - y.mean(axis=1, keepdims=True)).T
    if method in ("ejaccard", "edice"):
        e = x @ y.T
        sx = (x ** 2).sum(axis=1)[:, None]
        sy = (y ** 2).sum(axis=1)[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            return e / (sx + sy - e) if method == "ejaccard" else 2 * e / (sx + sy)
    bx = (x != 0).astype(float)
    by = (y != 0).astype(float)
    n = x.shape[1]
    a = bx @ by.T                      # both 1
    d = (1 - bx) @ (1 - by).T          # both 0
    with np.errstate(invalid="ignore", divide="ignore"):
        if method == "jaccard":
            return a / (bx.sum(1)[:, None] + by.sum(1)[None, :] - a)
        if method == "dice":
            return 2 * a / (bx.sum(1)[:, None] + by.sum(1)[None, :])
        if method == "simple matching":
            return (a + d) / n
        if method == "hamann":
            same = a + d
            return (same - (n - same)) / n
        if method == "faith":
            return (a + 0.5 * d) / n
    raise ValueError(f"unknown similarity method {method!r}")


def proxy_dist(x: np.ndarray, y: np.ndarray, method: str, *, p: float = 2) -> np.ndarray:
    """``proxyC::dist(x, y, method, use_nan = TRUE)``: rows of ``x`` vs rows of ``y``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x[:, None, :] - y[None, :, :]
    if method == "euclidean":
        return np.sqrt((diff ** 2).sum(axis=2))
    if method == "minkowski":
        return (np.abs(diff) ** p).sum(axis=2) ** (1 / p)
    if method == "manhattan":
        return np.abs(diff).sum(axis=2)
    if method == "maximum":
        return np.abs(diff).max(axis=2)
    if method == "hamming":
        return (diff != 0).sum(axis=2).astype(float)
    if method == "canberra":
        den = np.abs(x)[:, None, :] + np.abs(y)[None, :, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            term = np.where(den > 0, np.abs(diff) / den, 0.0)
        return term.sum(axis=2)
    if method in ("chisquared", "kullback"):
        zero = (x == 0).any(axis=1)[:, None] | (y == 0).any(axis=1)[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            if method == "chisquared":
                # Pearson's chi-squared of the 2 x n table [x, y], over its total
                sx = x.sum(axis=1)[:, None, None]
                sy = y.sum(axis=1)[None, :, None]
                col = x[:, None, :] + y[None, :, :]
                tot = sx + sy
                ex, ey = col * sx / tot, col * sy / tot
                chi = ((x[:, None, :] - ex) ** 2 / ex + (y[None, :, :] - ey) ** 2 / ey).sum(axis=2)
                out = chi / tot[..., 0]
            else:
                px = x / x.sum(axis=1, keepdims=True)
                py = y / y.sum(axis=1, keepdims=True)
                out = (px[:, None, :] * np.log(px[:, None, :] / py[None, :, :])).sum(axis=2)
        out[zero] = np.nan
        return out
    raise ValueError(f"unknown distance method {method!r}")


def _rank_rows(A: np.ndarray) -> np.ndarray:
    return np.vstack([pd.Series(r).rank().to_numpy() for r in A])


def cell_similarity(query: np.ndarray, ref: np.ndarray, metric: str = "cosine"):
    """``(query x ref similarity, legend title)`` exactly as ``CellCorHeatmap`` shows it.

    ``pearson`` and ``spearman`` become ``correlation`` (on per-row ranks for
    spearman, as ``t(apply(x, 1, rank))``), so their title reads
    ``"Correlation similarity"`` as in R.
    """
    m = "hamann" if metric == "hamman" else metric
    if m not in SIMIL_METHODS and m not in DIST_METHODS:
        raise ValueError(f"{metric} method is invalid.")
    q = np.asarray(query, dtype=float)
    r = np.asarray(ref, dtype=float)
    if m in SIMIL_METHODS:
        if m in ("pearson", "spearman"):
            if m == "spearman":
                q, r = _rank_rows(q), _rank_rows(r)
            m = "correlation"
        s = proxy_simil(r, q, m).T
        name = f"{capitalize([m])[0]} similarity"
    else:
        d = proxy_dist(r, q, m).T
        with np.errstate(invalid="ignore", divide="ignore"):
            s = 1 - d / np.nanmax(d)
        name = f"1-dist[{m}]/max(dist[{m}])"
    return clean_nonfinite(s, na_fill="zero"), name
