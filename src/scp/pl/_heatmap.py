"""Heatmaps — the ComplexHeatmap family.

Every function here builds a :class:`scp.heatmap.HeatmapSpec` and renders it.
Keep the split: the wrapper decides *what* the tracks and matrices are, the
renderer decides *where* they go.  That is the only way multi-panel,
split-aware heatmaps stay maintainable.

Spec: ``docs/porting_briefs/heatmaps.md``.
"""

from __future__ import annotations

from typing import Callable, Literal, Sequence

import numpy as np

from ..heatmap.matrix import ExpMethod
from ..heatmap.spec import HeatmapResult

__all__ = [
    "group_heatmap",
    "feature_heatmap",
    "dynamic_heatmap",
    "cell_cor_heatmap",
    "feature_cor_heatmap",
]

SplitMethod = Literal["kmeans", "hclust", "mfuzz", "kmeans-peaktime", "hclust-peaktime"]


def group_heatmap(
    adata,
    features: Sequence[str] | None = None,
    group_by: str | Sequence[str] = None,  # type: ignore[assignment]
    *,
    split_by: str | None = None,
    cells: Sequence[str] | None = None,
    layer: str | None = "counts",
    # normalisation
    exp_method: ExpMethod | Callable = "zscore",
    exp_cutoff: float = 0.0,
    lib_normalize: bool | None = None,
    limits: tuple[float, float] | None = None,
    aggregate_fun: Callable = np.mean,
    grouping_var: str | None = None,
    numerator: str | None = None,
    # splitting / clustering
    n_split: int | None = None,
    split_method: SplitMethod = "kmeans",
    feature_split: Sequence[str] | None = None,
    feature_split_by: Sequence[str] | None = None,
    split_order: Sequence[int] | None = None,
    decreasing: bool = False,
    cluster_rows: bool = False,
    cluster_columns: bool = False,
    cluster_row_slices: bool = False,
    cluster_column_slices: bool = False,
    cluster_features_by: Sequence[str] | None = None,
    # body overlays
    add_dot: bool = False,
    add_bg: bool = False,
    add_reticle: bool = False,
    dot_size_mm: float = 8.0,
    bg_alpha: float = 0.5,
    reticle_color: str = "#808080",
    # annotation
    cell_annotation: Sequence[str] | None = None,
    cell_annotation_palette: Sequence[str] | None = None,
    feature_annotation: Sequence[str] | None = None,
    feature_annotation_palette: Sequence[str] | None = None,
    nlabel: int = 20,
    features_label: Sequence[str] | None = None,
    show_row_names: bool = False,
    show_column_names: bool = True,
    # enrichment tracks (pluggable backend, see docs/porting_briefs/heatmaps.md §6)
    anno_terms: bool = False,
    anno_keys: bool = False,
    anno_features: bool = False,
    enrichment_fn: Callable | None = None,
    top_term: int = 5,
    # sizing
    width: float | Sequence[float] | None = None,
    height: float | None = None,
    flip: bool = False,
    palette: str = "RdBu",
    palcolor: Sequence[str] | None = None,
    seed: int = 11,
) -> HeatmapResult:
    """Group-averaged expression heatmap. Milestone 7.

    R original: ``GroupHeatmap``.  Pipeline, in order:

    1. fetch ``features x cells`` (features may be ``var_names`` *or* numeric
       ``obs`` columns, so QC scores can be plotted as pseudo-features);
    2. ``lib_normalize`` when the layer is counts and the matrix is non-negative
       — note this scales to the **median** library size, not 1e6;
    3. aggregate per group with ``aggregate_fun`` (default mean), and build a
       parallel "percent expressed" matrix with ``x > exp_cutoff``, which is the
       only consumer of ``exp_cutoff`` and the size channel for ``add_dot``;
    4. :func:`scp.heatmap.matrix_process`;
    5. clean non-finite values;
    6. derive color limits (:func:`scp.heatmap.color_limits`);
    7. split and cluster features;
    8. build the annotation track stack;
    9. lay out and render.

    ``grouping_var`` forces ``exp_method="log2fc"``, bypasses ``matrix_process``
    entirely and computes ``log2(group_true / group_false)`` per feature.
    """
    raise NotImplementedError("Milestone 7. Spec: docs/porting_briefs/heatmaps.md §1-§7.")


def feature_heatmap(adata, features=None, group_by=None, *, max_cells: int = 100, **kwargs) -> HeatmapResult:
    """Per-cell (unaggregated) expression heatmap. Milestone 7.

    Same pipeline as :func:`group_heatmap` minus the aggregation step, with
    cells downsampled to ``max_cells`` per group and variable column gaps so
    sub-blocks of the same group touch.
    """
    raise NotImplementedError("Milestone 7. Spec: docs/porting_briefs/heatmaps.md §3.")


def dynamic_heatmap(
    adata,
    lineages: Sequence[str],
    features: Sequence[str] | None = None,
    *,
    use_fitted: bool = False,
    cell_density: float = 1.0,
    cell_bins: int = 100,
    order_by: Literal["peaktime", "valleytime"] = "peaktime",
    reverse_ht: Sequence[str] | None = None,
    n_split: int | None = None,
    split_method: SplitMethod = "kmeans-peaktime",
    num_intersections: int | Sequence[int] | None = None,
    separate_annotation: Sequence | None = None,
    pseudotime_label: Sequence[float] | None = None,
    **kwargs,
) -> HeatmapResult:
    """Expression along pseudotime, one block per lineage. Milestone 9.

    Depends on a GAM backend (``pygam`` or ``statsmodels.gam``) to reproduce
    ``mgcv::gam(y ~ s(x, bs="cs") + offset(log(libsize)))``; ship it as an
    optional extra.  ``peaktime`` is the *median pseudotime among cells in the
    top 1% of fitted values*, not the argmax — ``kmeans-peaktime`` clusters on
    that scalar rather than on the expression matrix.

    Prefer reading fitted trends that ``scFates`` already stores over refitting.
    """
    raise NotImplementedError("Milestone 9. Spec: docs/porting_briefs/heatmaps.md §8.")


def cell_cor_heatmap(query, reference=None, **kwargs) -> HeatmapResult:
    """Query-vs-reference similarity heatmap. Milestone 10."""
    raise NotImplementedError("Milestone 10. Spec: docs/porting_briefs/heatmaps.md §9.")


def feature_cor_heatmap(adata, features: Sequence[str], **kwargs) -> HeatmapResult:
    """Gene-gene correlation heatmap.

    Note: the R function is an **empty stub** — there is nothing to port.  This
    is a clean-room implementation: ``np.corrcoef`` on the selected features,
    clustered on both axes with average linkage over ``1 - corr``.
    """
    raise NotImplementedError("Milestone 10. No R reference exists; implement directly.")
