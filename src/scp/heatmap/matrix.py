"""Matrix preprocessing shared by every heatmap — ``matrix_process`` and friends.

Pure numpy, no plotting. Port and pin this first: it is fully testable against
R with golden fixtures, and every heatmap's appearance depends on it.

Reference: ``SCP/R/SCP-plot.R`` lines 7196--7230.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

__all__ = ["matrix_process", "lib_normalize", "clean_nonfinite", "color_limits", "ExpMethod"]

ExpMethod = Literal["raw", "zscore", "fc", "log2fc", "log1p"]


def matrix_process(matrix: np.ndarray, method: ExpMethod | Callable = "zscore", **kwargs) -> np.ndarray:
    """Row-wise normalisation of a ``features x observations`` matrix.

    ==========  =====================================  ==============================
    method      R                                      numpy
    ==========  =====================================  ==============================
    ``raw``     identity                               ``M``
    ``fc``      ``M / rowMeans(M)``                    ``M / M.mean(1, keepdims=True)``
    ``log2fc``  ``log2(M / rowMeans(M))``              ``np.log2(M / M.mean(1, ...))``
    ``log1p``   ``log1p(M)``                           ``np.log1p(M)``
    ``zscore``  ``t(scale(t(M)))``                     ``(M - mean) / std(ddof=1)``
    ==========  =====================================  ==============================

    .. warning::
       ``scale()`` uses the **sample** standard deviation.  ``np.std`` defaults
       to ``ddof=0``; passing the default makes every z-score off by
       ``sqrt(n/(n-1))``.

    Rows of all zeros produce NaN/-inf under ``fc``/``log2fc``, and constant
    rows produce NaN under ``zscore``.  R does not handle this here — the
    callers do, via :func:`clean_nonfinite`.  Keep that split so the transform
    stays pure.
    """
    M = np.asarray(matrix, dtype=float)
    if callable(method):
        out = method(M, **kwargs)
    elif method == "raw":
        out = M
    elif method == "fc":
        out = M / M.mean(axis=1, keepdims=True)
    elif method == "log2fc":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.log2(M / M.mean(axis=1, keepdims=True))
    elif method == "log1p":
        out = np.log1p(M)
    elif method == "zscore":
        center = kwargs.get("center", True)
        scale = kwargs.get("scale", True)
        out = M - M.mean(axis=1, keepdims=True) if center else M.copy()
        if scale:
            with np.errstate(divide="ignore", invalid="ignore"):
                out = out / M.std(axis=1, ddof=1, keepdims=True)
    else:
        raise ValueError(f"unknown exp_method {method!r}")
    out = np.asarray(out)
    if out.shape != M.shape:
        raise ValueError("matrix_process must preserve shape")
    return out


def lib_normalize(matrix: np.ndarray, libsize: np.ndarray, *, gene_rows: np.ndarray | None = None) -> np.ndarray:
    """CPM-like normalisation, but scaled to the **median** library size.

    R::

        libsize <- colSums(counts)
        if (any(libsize %% 1 != 0)) libsize <- rep(1, n)   # non-integer -> no-op
        mat[genes, ] <- t(t(mat[genes, ]) / libsize * median(libsize))

    Rows sourced from ``obs`` columns (QC scores plotted as pseudo-features)
    are deliberately left alone — pass their positions in ``gene_rows`` as a
    boolean mask of rows that *should* be normalised.
    """
    M = np.asarray(matrix, dtype=float).copy()
    ls = np.asarray(libsize, dtype=float)
    if np.any(ls % 1 != 0):
        ls = np.ones_like(ls)
    scale = np.median(ls)
    mask = np.ones(M.shape[0], dtype=bool) if gene_rows is None else np.asarray(gene_rows, dtype=bool)
    M[mask] = M[mask] / ls[None, :] * scale
    return M


def clean_nonfinite(matrix: np.ndarray, *, na_fill: Literal["mean", "zero"] = "mean") -> np.ndarray:
    """Infinities to the largest finite magnitude (sign-preserving); NaN to the mean.

    R does this in each caller rather than inside ``matrix_process``::

        mat[is.infinite(mat)] <- max(abs(mat[!is.infinite(mat)])) * sign(...)
        mat[is.na(mat)]       <- mean(mat, na.rm = TRUE)

    The ``grouping.var`` path of ``GroupHeatmap`` fills NaN with 0 instead.
    """
    M = np.asarray(matrix, dtype=float).copy()
    inf = np.isinf(M)
    if inf.any():
        finite = M[np.isfinite(M)]
        cap = float(np.max(np.abs(finite))) if finite.size else 0.0
        M[inf] = cap * np.sign(M[inf])
    nan = np.isnan(M)
    if nan.any():
        M[nan] = 0.0 if na_fill == "zero" else float(np.nanmean(M))
    return M


def color_limits(
    matrix: np.ndarray,
    exp_method: ExpMethod = "zscore",
    limits: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Derive the color-scale limits exactly as SCP does.

    Two rules that are easy to get subtly wrong:

    * for the symmetric methods (``zscore``, ``log2fc``) the bound is the
      **smaller** of ``|q01|`` and ``|q99|``, rounded *up* to the nearest 0.5;
    * otherwise the limits are the 1st and 99th percentiles as-is.

    Values outside the range clamp to the terminal colors — set
    ``cmap.set_over`` / ``set_under`` rather than masking.
    """
    if limits is not None:
        return float(limits[0]), float(limits[1])
    q01, q99 = np.nanquantile(matrix, [0.01, 0.99])
    if exp_method in ("zscore", "log2fc"):
        b = np.ceil(min(abs(q01), abs(q99)) * 2) / 2
        return -float(b), float(b)
    return float(q01), float(q99)
