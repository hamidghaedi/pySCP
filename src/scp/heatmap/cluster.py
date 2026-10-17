"""Feature splitting and axis ordering for the heatmap family.

Pure numpy/scipy, no plotting.  This is the part of SCP's heatmaps whose output
is a *permutation*, and a wrong permutation is the classic silent-wrongness bug:
the figure still looks like a heatmap.  Every function here therefore mirrors a
specific piece of R and is pinned against SCP's own output in
``tests/fixtures/parity/heatmap/`` (``tools/parity/heatmap_fixtures.R``).

Three layers of R are reproduced:

* SCP's splitters -- ``kmeans`` / ``hclust`` / ``mfuzz`` and the DynamicHeatmap
  ``*-peaktime`` variants -- and the rule that names the resulting clusters
  ``C1, C2, ...`` (``SCP-plot.R`` 8526-8581, 9590-9645, 11533-11591);
* ComplexHeatmap's ``make_cluster``: complete-linkage euclidean ``hclust`` per
  slice, ``reorder.dendrogram`` by ``-rowMeans``, and the optional clustering
  of slice means (``cluster_row_slices`` / ``cluster_column_slices``);
* ``cluster_within_group`` (ComplexHeatmap) and SCP's
  ``cluster_within_group2`` (``SCP-plot.R`` 7250), which cluster inside each
  group and order the groups by a tree of their centroids.

Upstream: Hao Zhang, SCP (GPL-3), https://github.com/zhanghao-njmu/SCP; and
Zuguang Gu, ComplexHeatmap (MIT), https://github.com/jokergoo/ComplexHeatmap.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "SplitMethod",
    "hclust",
    "dend_order",
    "reorder_dendrogram",
    "cutree",
    "standardise",
    "mestimate",
    "split_features",
    "name_splits_by_group",
    "name_splits_by_value",
    "AxisOrder",
    "axis_order",
    "cluster_within_group",
    "relabel_first_appearance",
]

SplitMethod = Literal["kmeans", "hclust", "mfuzz", "kmeans-peaktime", "hclust-peaktime"]


# --------------------------------------------------------------------------- #
# trees
# --------------------------------------------------------------------------- #
def hclust(X: np.ndarray) -> np.ndarray:
    """``hclust(dist(X))``: complete linkage on euclidean distances between rows.

    scipy's linkage matrix follows R's ``merge`` conventions -- a singleton
    before a cluster, the smaller id first, an earlier merge before a later one
    -- so ``leaves_list(Z)`` equals ``hclust(...)$order`` whenever the merge
    sequence agrees, which it does up to exact distance ties.
    """
    from scipy.cluster.hierarchy import linkage

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return linkage(X, method="complete", metric="euclidean")


def _leaf_sequences(Z: np.ndarray, n: int):
    """Yield, per merge, the two children ids; leaves are ``0..n-1``."""
    for i in range(Z.shape[0]):
        yield i, int(Z[i, 0]), int(Z[i, 1])


def dend_order(Z: np.ndarray) -> np.ndarray:
    """Left-to-right leaf order of a linkage matrix (``order.dendrogram``)."""
    from scipy.cluster.hierarchy import leaves_list

    return np.asarray(leaves_list(Z), dtype=int)


def reorder_dendrogram(
    Z: np.ndarray, wts: np.ndarray, agglo: Callable[[np.ndarray], float] = np.mean
) -> tuple[np.ndarray, np.ndarray]:
    """``stats::reorder.dendrogram(dend, wts, agglo.FUN)`` on a linkage matrix.

    A leaf's value is ``wts[leaf]``; a node's value is ``agglo`` of its
    children's *values* (not of its leaves); at every node the children are put
    in increasing order of value, ties keeping their current order
    (``sort.list`` is stable).  Returns the linkage with children swapped where
    needed -- so scipy draws the reordered tree -- and the new leaf order.

    Iterative rather than recursive: a per-cell heatmap can have thousands of
    leaves, deeper than Python's recursion limit.
    """
    Z = np.array(Z, dtype=float, copy=True)
    n = Z.shape[0] + 1
    wts = np.asarray(wts, dtype=float)
    value: dict[int, float] = {i: float(wts[i]) for i in range(n)}
    leaves: dict[int, list[int]] = {i: [i] for i in range(n)}
    for i, a, b in _leaf_sequences(Z, n):
        va, vb = value[a], value[b]
        if vb < va:  # strict: equal values keep the current order
            Z[i, 0], Z[i, 1] = b, a
            a, b, va, vb = b, a, vb, va
        value[n + i] = float(agglo(np.array([va, vb])))
        leaves[n + i] = leaves.pop(a) + leaves.pop(b)
    order = np.asarray(leaves[2 * n - 2] if n > 1 else [0], dtype=int)
    return Z, order


def relabel_first_appearance(labels: np.ndarray) -> np.ndarray:
    """Renumber cluster labels ``1..k`` in order of first appearance."""
    labels = np.asarray(labels)
    mapping: dict = {}
    for lab in labels:
        if lab not in mapping:
            mapping[lab] = len(mapping) + 1
    return np.array([mapping[lab] for lab in labels], dtype=int)


def cutree(Z: np.ndarray, k: int) -> np.ndarray:
    """``cutree(hc, k)``: ``k`` clusters, numbered by first appearance as R does."""
    from scipy.cluster.hierarchy import fcluster

    return relabel_first_appearance(fcluster(Z, t=k, criterion="maxclust"))


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def standardise(data: np.ndarray) -> np.ndarray:
    """Mfuzz's ``standardise``: every row to mean 0, sd 1 (``ddof=1``, as ``scale``)."""
    X = np.asarray(data, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, ddof=1, keepdims=True)


def mestimate(data: np.ndarray) -> float:
    """Schwaemmle & Jensen's estimate of the fuzzifier ``m`` (Mfuzz ``mestimate``)."""
    N, D = np.asarray(data).shape
    return float(
        1 + (1418 / N + 22.05) * D ** (-2)
        + (12.33 / N + 0.243) * D ** (-0.0406 * np.log(N) - 0.1134)
    )


def _kmeans(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "split_method='kmeans' needs scikit-learn: `pip install scp-plot[stats]`"
        ) from e
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] == 1:
        return relabel_first_appearance(_kmeans_1d(X[:, 0], k))
    # R: kmeans(x, centers = k, iter.max = 1e4, nstart = 20)
    km = KMeans(n_clusters=k, n_init=20, max_iter=10_000, random_state=seed)
    return relabel_first_appearance(_hartigan_refine(X, km.fit_predict(X)))


def _kmeans_1d(y: np.ndarray, k: int) -> np.ndarray:
    """Globally optimal 1-D k-means by dynamic programming (Wang & Song 2011).

    ``kmeans-peaktime`` clusters a single number per feature.  In one dimension
    the optimum is exactly computable, and it is what R's Hartigan-Wong with 20
    starts lands on in practice; Lloyd with random starts does get stuck there.
    """
    o = np.argsort(y, kind="stable")
    s = y[o]
    n = len(s)
    c1 = np.concatenate([[0.0], np.cumsum(s)])
    c2 = np.concatenate([[0.0], np.cumsum(s ** 2)])

    def sse(i, j):  # rows i..j-1 of the sorted values
        m = j - i
        tot = c1[j] - c1[i]
        return (c2[j] - c2[i]) - tot * tot / m

    D = np.full((k + 1, n + 1), np.inf)
    B = np.zeros((k + 1, n + 1), dtype=int)
    D[0, 0] = 0.0
    for q in range(1, k + 1):
        for j in range(q, n + 1):
            best, arg = np.inf, q - 1
            for i in range(q - 1, j):
                v = D[q - 1, i] + sse(i, j)
                if v < best - 1e-12:
                    best, arg = v, i
            D[q, j], B[q, j] = best, arg
    lab_sorted = np.empty(n, dtype=int)
    j = n
    for q in range(k, 0, -1):
        i = B[q, j]
        lab_sorted[i:j] = q - 1
        j = i
    out = np.empty(n, dtype=int)
    out[o] = lab_sorted
    return out


def _hartigan_refine(X: np.ndarray, labels: np.ndarray, max_pass: int = 1000) -> np.ndarray:
    """Single-point Hartigan moves on top of Lloyd's solution.

    R's default ``kmeans`` is Hartigan-Wong, whose optimality condition is
    stricter than Lloyd's: a point moves whenever that lowers the total
    within-cluster sum of squares, accounting for the centroid shifts
    (``n_j / (n_j + 1) * d_j^2 < n_i / (n_i - 1) * d_i^2``).  Applying these
    moves until none is left lifts scikit-learn's Lloyd solutions out of the
    local optima Hartigan-Wong does not stop in, which is what brings the two
    partitions together.
    """
    lab = np.asarray(labels).copy()
    k = int(lab.max()) + 1
    for _ in range(max_pass):
        moved = False
        for i in range(X.shape[0]):
            src = lab[i]
            n = np.bincount(lab, minlength=k).astype(float)
            if n[src] <= 1:
                continue
            cent = np.vstack([X[lab == c].mean(axis=0) if n[c] else np.full(X.shape[1], np.inf)
                              for c in range(k)])
            d = ((X[i] - cent) ** 2).sum(axis=1)
            gain = n / (n + 1) * d
            gain[src] = n[src] / (n[src] - 1) * d[src]
            dst = int(np.argmin(np.where(np.arange(k) == src, np.inf, gain)))
            if gain[dst] < gain[src] - 1e-12:
                lab[i] = dst
                moved = True
        if not moved:
            break
    return lab


def _mfuzz(X: np.ndarray, k: int, seed: int, fuzzification: float | None) -> np.ndarray:
    try:
        import skfuzzy
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "split_method='mfuzz' needs scikit-fuzzy: `pip install scp-plot[fuzzy]`"
        ) from e
    std = np.nan_to_num(standardise(X), nan=0.0)
    m_min = mestimate(std)
    if fuzzification is None:
        fuzzification = m_min + 0.1
    elif fuzzification <= m_min:
        warnings.warn(
            f"fuzzification value is smaller than estimated: {m_min:.2f}", stacklevel=3
        )
    # e1071::cmeans(x, centers = k, method = "cmeans", m = fuzzification);
    # skfuzzy wants observations in columns.
    _cntr, u, *_ = skfuzzy.cmeans(
        std.T, c=k, m=fuzzification, error=1e-6, maxiter=10_000, seed=seed
    )
    if not np.all(np.isfinite(u)):
        raise ValueError(
            f"Clustering with mfuzz failed (fuzzification={fuzzification:.2f}). "
            "Please set a larger fuzzification parameter manually."
        )
    return relabel_first_appearance(np.argmax(u, axis=0))


def split_features(
    matrix: np.ndarray,
    n_split: int,
    method: SplitMethod = "kmeans",
    *,
    seed: int = 11,
    fuzzification: float | None = None,
    peak: np.ndarray | None = None,
) -> np.ndarray:
    """Cluster features into ``n_split`` groups; labels are ``1..n_split``.

    ``matrix`` is SCP's ``mat_split`` (features x columns, already processed).
    The ``*-peaktime`` methods cluster the 1-D ``peak`` vector (the
    ``order_by`` column of DynamicHeatmap's feature metadata) instead.

    * ``kmeans`` -- R's Hartigan-Wong with ``nstart = 20``; here scikit-learn's
      Lloyd with ``n_init = 20`` followed by Hartigan single-point moves, and
      for 1-D input (``kmeans-peaktime``) the exact optimum by dynamic
      programming.  The RNGs cannot be shared, so it is the *partition* that
      is compared with R (adjusted Rand index); label numbers are arbitrary on
      both sides and only feed the ``C1..Cn`` naming, which re-sorts them.
    * ``hclust`` -- complete-linkage euclidean, cut at ``k``; deterministic and
      identical to R.
    * ``mfuzz`` -- fuzzy c-means on the row-standardised matrix with
      ``m = mestimate + 0.1`` (``e1071::cmeans`` in R, ``skfuzzy.cmeans``
      here), hard-assigned by maximum membership.  Random initialisation
      differs between the two, so again only the partition is comparable.
    """
    X = np.asarray(matrix, dtype=float)
    if n_split == 1:
        return np.ones(X.shape[0], dtype=int)
    if method in ("kmeans-peaktime", "hclust-peaktime"):
        if peak is None:
            raise ValueError(f"split_method={method!r} needs the per-feature peak times")
        y = np.asarray(peak, dtype=float).reshape(-1, 1)
        return _kmeans(y, n_split, seed) if method == "kmeans-peaktime" else cutree(hclust(y), n_split)
    if method == "kmeans":
        return _kmeans(X, n_split, seed)
    if method == "hclust":
        return cutree(hclust(X), n_split)
    if method == "mfuzz":
        return _mfuzz(X, n_split, seed, fuzzification)
    raise ValueError(f"unknown split_method {method!r}")


def _finish_names(
    raw_ids: list[int], labels: np.ndarray, split_order: Sequence[int] | None
) -> tuple[pd.Categorical, pd.Categorical]:
    if split_order is not None:
        so = [int(i) - 1 for i in split_order]  # R's split_order is 1-based
        raw_ids = [raw_ids[i] for i in so]
    short = np.empty(len(labels), dtype=object)
    long = np.empty(len(labels), dtype=object)
    short_levels, long_levels = [], []
    for i, rid in enumerate(raw_ids, start=1):
        members = labels == rid
        short[members] = f"C{i}"
        long[members] = f"C{i}({int(members.sum())})"
        short_levels.append(f"C{i}")
        long_levels.append(f"C{i}({int(members.sum())})")
    return (pd.Categorical(short, categories=short_levels),
            pd.Categorical(long, categories=long_levels))


def name_splits_by_group(
    labels: np.ndarray,
    mat_split: np.ndarray,
    column_groups: Sequence[str],
    group_levels: Sequence[str],
    *,
    decreasing: bool = False,
    split_order: Sequence[int] | None = None,
    sort_groups: Literal["alphabetical", "levels"] = "alphabetical",
) -> tuple[pd.Categorical, pd.Categorical]:
    """Name clusters ``C1..Cn`` by the group each one peaks in.

    GroupHeatmap / FeatureHeatmap rule: every feature's *max group* is the
    column group with the highest mean in ``mat_split``; each cluster's
    representative is the most common max group among its members (ties to the
    earlier level); clusters are sorted by that group's position in
    ``group_levels`` (ties keep cluster-id order), then permuted by the 1-based
    ``split_order``.

    ``sort_groups`` reproduces one R detail that only matters for ties in
    ``which.max``: GroupHeatmap aggregates by a *character* vector, so its group
    means come out alphabetically sorted; FeatureHeatmap aggregates by a factor
    and keeps level order.

    Returns ``(feature_split, row_split)`` aligned with the input rows:
    ``C1..Cn`` and the display labels ``"C1(37)"``.
    """
    labels = np.asarray(labels)
    cg = np.asarray([str(x) for x in column_groups])
    levels = [str(x) for x in group_levels]
    present = [lv for lv in levels if (cg == lv).any()]
    uniq = sorted(present) if sort_groups == "alphabetical" else present
    M = np.asarray(mat_split, dtype=float)
    gm = np.vstack([M[:, cg == g].mean(axis=1) for g in uniq])  # groups x features
    maxgroup = [uniq[int(np.argmax(gm[:, f]))] for f in range(M.shape[0])]
    raw = sorted(set(labels.tolist()))
    key = []
    for rid in raw:
        members = [maxgroup[f] for f in np.flatnonzero(labels == rid)]
        counts = [members.count(lv) for lv in levels]
        mode = levels[int(np.argmax(counts))]
        key.append(levels.index(mode) + 1)
    sign = -1 if decreasing else 1
    idx = sorted(range(len(raw)), key=lambda i: sign * key[i])  # stable
    return _finish_names([raw[i] for i in idx], labels, split_order)


def name_splits_by_value(
    labels: np.ndarray,
    values: np.ndarray,
    *,
    decreasing: bool = False,
    split_order: Sequence[int] | None = None,
) -> tuple[pd.Categorical, pd.Categorical]:
    """DynamicHeatmap rule: order clusters by the mean of their members' ``values``
    (peak or valley time); ties keep cluster-id order."""
    labels = np.asarray(labels)
    vals = np.asarray(values, dtype=float)
    raw = sorted(set(labels.tolist()))
    means = [float(np.mean(vals[labels == rid])) for rid in raw]
    sign = -1 if decreasing else 1
    idx = sorted(range(len(raw)), key=lambda i: sign * means[i])
    return _finish_names([raw[i] for i in idx], labels, split_order)


# --------------------------------------------------------------------------- #
# axis orders
# --------------------------------------------------------------------------- #
@dataclass
class AxisOrder:
    """The display order of one heatmap axis.

    ``order`` is a permutation of the input positions; ``slices`` lists the
    split levels in display order (empty when there is no split);
    ``slice_linkages`` holds one linkage per displayed slice (``None`` for a
    single-item slice), already reordered so scipy draws it as displayed.
    """

    order: np.ndarray
    slices: list[str] = field(default_factory=list)
    slice_linkages: list[np.ndarray | None] = field(default_factory=list)
    linkage: np.ndarray | None = None


def _blocks(split: pd.Categorical | None, n: int) -> list[tuple[str, np.ndarray]]:
    if split is None:
        return [("", np.arange(n))]
    s = pd.Categorical(split)
    vals = np.asarray(s.astype(object))
    out = []
    for lv in s.categories:
        idx = np.flatnonzero(vals == lv)
        if idx.size:
            out.append((str(lv), idx))
    return out


def _snap(w: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Round reorder weights to 1e-10 of the data scale.

    The rows of a z-scored matrix have mean exactly 0, so ``-rowMeans`` is pure
    rounding noise of order 1e-17 and R's ``reorder`` then flips children on the
    sign of that noise -- an order no other implementation (nor R on another
    BLAS) reproduces.  Snapping turns those into exact ties, which
    ``reorder.dendrogram`` resolves by keeping the ``hclust`` order.  Weights
    that carry information (``raw``, ``log1p``, column means) are untouched at
    this resolution.
    """
    scale = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 1.0
    tol = 1e-10 * max(scale, 1e-300)
    return np.round(np.asarray(w, dtype=float) / tol) * tol


def axis_order(
    matrix: np.ndarray,
    split: pd.Categorical | None = None,
    *,
    cluster: bool = False,
    cluster_slices: bool = False,
) -> AxisOrder:
    """ComplexHeatmap's ordering of the rows of ``matrix`` given a split.

    Without clustering, slices follow the split's level order and rows keep
    their input order inside each slice.  With ``cluster=True`` each slice is
    clustered on its own (complete linkage, euclidean) and the tree reordered by
    ``-rowMeans`` of the whole matrix -- ComplexHeatmap's ``row_dend_reorder``
    default.  ``cluster_slices=True`` then orders the slices by a tree of the
    slice column-means, reordered with ComplexHeatmap's own weights: it passes
    the slice-mean *matrix* as ``wts``, so each slice is weighted by the first
    slice's mean in the column of the same index.  That quirk is reproduced.

    One deliberate divergence: the ``-rowMeans`` weights are snapped to 1e-10
    of the data scale (:func:`_snap`).  For z-scored rows they are pure
    rounding noise, on which R flips children arbitrarily; here they tie and
    the ``hclust`` order stands.  Same tree, deterministic leaf order.

    Pass ``matrix.T`` to order columns.
    """
    M = np.asarray(matrix, dtype=float)
    blocks = _blocks(split, M.shape[0])
    names = [b[0] for b in blocks]
    orders = [b[1] for b in blocks]
    links: list[np.ndarray | None] = [None] * len(blocks)
    if cluster:
        wts = _snap(-np.nanmean(M, axis=1), M)
        for i, idx in enumerate(orders):
            if len(idx) > 1:
                Z, o = reorder_dendrogram(hclust(M[idx]), wts[idx])
                orders[i] = idx[o]
                links[i] = Z
        if len(orders) > 1 and cluster_slices:
            slice_mean = np.column_stack([np.nanmean(M[idx], axis=0) for idx in orders])
            Zs = hclust(slice_mean.T)
            _, so = reorder_dendrogram(Zs, slice_mean.ravel(order="F")[: len(orders)])
            orders = [orders[i] for i in so]
            names = [names[i] for i in so]
            links = [links[i] for i in so]
    order = np.concatenate(orders) if orders else np.arange(0)
    full = links[0] if (split is None and cluster) else None
    return AxisOrder(order=order, slices=names if split is not None else [],
                     slice_linkages=links if cluster else [], linkage=full)


def cluster_within_group(
    matrix: np.ndarray,
    groups: pd.Categorical,
    *,
    keep_heights: bool = False,
) -> AxisOrder:
    """Cluster rows inside each group, then order the groups by their centroids.

    ``keep_heights=False`` is ComplexHeatmap's ``cluster_within_group``, which
    flattens every within-group tree to height 0 (used by SCP for
    ``cluster_row_slices`` / ``cluster_column_slices``); ``keep_heights=True``
    is SCP's ``cluster_within_group2`` (``SCP-plot.R`` 7250, used for
    ``cluster_features_by`` with a split), which keeps them and lifts the
    parent tree above the tallest child.  The display order is the same
    construct in both: each group's own ``hclust`` order, groups concatenated
    in the order of an ``hclust`` of the group centroids.  Neither reorders a
    tree by weights.

    R's version takes ``t(matrix)`` and clusters its columns; this takes the
    items as rows.
    """
    M = np.asarray(matrix, dtype=float)
    blocks = _blocks(groups, M.shape[0])
    orders: list[np.ndarray] = []
    links: list[np.ndarray | None] = []
    for _name, idx in blocks:
        if len(idx) > 1:
            Z = hclust(M[idx])
            orders.append(idx[dend_order(Z)])
            if not keep_heights:
                Z = Z.copy()
                Z[:, 2] = 0.0
            links.append(Z)
        else:
            orders.append(idx)
            links.append(None)
    names = [b[0] for b in blocks]
    if len(blocks) > 1:
        centroids = np.vstack([M[o].mean(axis=0) for o in orders])
        parent = hclust(centroids)
        po = dend_order(parent)
    else:
        parent, po = None, np.arange(1)
    lifted = max((float(Z[:, 2].max()) for Z in links if Z is not None), default=0.0)
    if parent is not None:
        parent = parent.copy()
        parent[:, 2] += lifted
    return AxisOrder(
        order=np.concatenate([orders[i] for i in po]),
        slices=[names[i] for i in po],
        slice_linkages=[links[i] for i in po],
        linkage=parent,
    )
