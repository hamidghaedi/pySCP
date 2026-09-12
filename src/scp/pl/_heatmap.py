"""Heatmaps — the ComplexHeatmap family.

Every function here builds a :class:`scp.heatmap.HeatmapSpec` and renders it.
Keep the split: the wrapper decides *what* the tracks and matrices are, the
renderer decides *where* they go.  That is the only way multi-panel,
split-aware heatmaps stay maintainable.

Spec: ``docs/porting_briefs/heatmaps.md``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

from ..fetch import as_ordered_categorical, fetch_data, fetch_var, infer_data_type
from ..heatmap.matrix import ExpMethod, clean_nonfinite, color_limits, matrix_process
from ..heatmap.matrix import lib_normalize as lib_norm
from ..heatmap.render import MM, render
from ..heatmap.spec import (
    Dendrogram,
    HeatmapResult,
    HeatmapSpec,
    Layer,
    LegendSpec,
    PanelSpec,
    Track,
)
from ..palettes import continuous_palette, discrete_palette

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
    for unsupported, name in (
        (anno_terms, "anno_terms"), (anno_keys, "anno_keys"), (anno_features, "anno_features"),
        (grouping_var, "grouping_var"), (feature_split_by, "feature_split_by"),
        (cluster_features_by, "cluster_features_by"), (cluster_column_slices,
                                                       "cluster_column_slices"),
    ):
        if unsupported:
            raise NotImplementedError(
                f"`{name}` is specified in docs/porting_briefs/heatmaps.md but not yet "
                "implemented; see docs/07_milestones.md."
            )
    if group_by is None:
        raise ValueError("`group_by` is required")
    groups = [group_by] if isinstance(group_by, str) else list(group_by)
    if features is None:
        raise ValueError("`features` is required")
    features = list(features)

    frame = fetch_data(adata, features, layer=layer, on_missing="warn")
    features = [f for f in features if f in frame.columns]
    if not features:
        raise ValueError("none of `features` could be resolved")
    X = frame[features].to_numpy(dtype=float).T  # features x cells

    obs = adata.obs
    keep = np.ones(adata.n_obs, dtype=bool)
    if cells is not None:
        keep = np.isin(np.asarray(adata.obs_names), np.asarray(cells))
    X = X[:, keep]

    # 2. lib_normalize scales to the MEDIAN library size, not 1e6, and is only
    #    meaningful on counts -- SCP's guard warns and no-ops on non-integers.
    if lib_normalize is None:
        lib_normalize = layer is not None and infer_data_type(X) == "raw_counts"
    if lib_normalize:
        # Library size is the FULL transcriptome per cell, not the sum of the
        # selected features: with a dozen markers most cells would sum to zero
        # and the whole matrix would come back NaN.
        full = adata.layers[layer] if layer in (adata.layers or {}) else adata.X
        libsize = np.asarray(full.sum(axis=1)).ravel()[keep]
        if not np.allclose(X, np.round(X)):
            warnings.warn(
                "the matrix is not integer-valued; library sizes set to 1 "
                "(SCP does the same rather than normalising log data twice).",
                stacklevel=2,
            )
            libsize = np.ones_like(libsize)
        X = lib_norm(X, libsize)

    split_levels: list[str | None] = [None]
    split_vec = None
    if split_by is not None:
        split_cat = as_ordered_categorical(obs[split_by][keep])
        split_vec = np.asarray(split_cat)
        split_levels = list(split_cat.cat.categories)

    panels: list[PanelSpec] = []
    matrices: dict[str, pd.DataFrame] = {}
    raw_blocks: list[np.ndarray] = []
    pct_blocks: list[np.ndarray] = []
    col_meta: list[tuple[str, list[str]]] = []

    for g in groups:
        gcat = as_ordered_categorical(obs[g][keep])
        glevels = [lv for lv in gcat.cat.categories if (np.asarray(gcat) == lv).any()]
        for sp in split_levels:
            mask_sp = np.ones(X.shape[1], dtype=bool) if sp is None else (split_vec == sp)
            agg = np.empty((len(features), len(glevels)), dtype=float)
            pct = np.empty_like(agg)
            for j, lv in enumerate(glevels):
                m = mask_sp & (np.asarray(gcat) == lv)
                block = X[:, m]
                if block.shape[1] == 0:
                    agg[:, j] = np.nan
                    pct[:, j] = 0.0
                    continue
                agg[:, j] = aggregate_fun(block, axis=1)
                # the ONLY consumer of exp_cutoff, and the size channel for add_dot
                pct[:, j] = (block > exp_cutoff).mean(axis=1)
            name = g if sp is None else f"{sp}"
            raw_blocks.append(agg)
            pct_blocks.append(pct)
            col_meta.append((name, [str(lv) for lv in glevels]))

    # 4-6: transform, clean, then derive the colour limits from the result
    processed = [clean_nonfinite(matrix_process(b, exp_method)) for b in raw_blocks]
    pooled = np.concatenate([p.ravel() for p in processed])
    vmin, vmax = color_limits(pooled.reshape(1, -1), exp_method, limits)
    ramp = continuous_palette(palette=palette, palcolor=palcolor, n=100)
    cmap = LinearSegmentedColormap.from_list("scp_hm", ramp, N=256)
    norm = Normalize(vmin, vmax)

    # 7. split and order the features
    row_split = None
    row_dend = None
    order = np.arange(len(features))
    if feature_split is not None:
        fs = pd.Categorical([str(x) for x in feature_split])
        order = np.argsort(pd.Categorical(fs, categories=fs.categories).codes, kind="stable")
        row_split = pd.Categorical(np.asarray(fs)[order], categories=fs.categories)
    elif n_split:
        # clustering cannot see NaN; a feature absent from a whole panel would
        # otherwise abort the split rather than simply cluster on what it has
        ref = np.nan_to_num(processed[0], nan=0.0, posinf=0.0, neginf=0.0)
        labels = _split_features(ref, n_split, split_method, seed=seed)
        cats = [f"Cluster {i + 1}" for i in range(n_split)]
        order = np.argsort(labels, kind="stable")
        row_split = pd.Categorical([cats[labels[i]] for i in order], categories=cats)
    elif cluster_rows:
        from scipy.cluster.hierarchy import leaves_list, linkage
        Z = linkage(processed[0], method="complete")
        order = np.asarray(leaves_list(Z))
        row_dend = Dendrogram(linkage=Z, order=order)

    row_labels = pd.Index([features[i] for i in order])

    # With add_dot the cell IS the dot at 100%, so the body cell size is set by
    # dot_size rather than chosen independently -- otherwise the dots overflow
    # their cells and overlap.
    cell_in = dot_size_mm * MM if add_dot else 0.16

    for (name, glevels), proc, pct in zip(col_meta, processed, pct_blocks, strict=True):
        layers: list[Layer] = []
        if add_bg:
            layers.append(Layer(kind="bg", alpha=bg_alpha))
        if add_reticle:
            layers.append(Layer(kind="reticle", color=reticle_color))
        if add_dot:
            # dots replace the body, so the body is whited out underneath first
            layers.insert(0, Layer(kind="whiteout"))
            layers.append(Layer(kind="dot", size_matrix=pct[order, :],
                                base_size_in=dot_size_mm * MM))
        panels.append(PanelSpec(
            name=name, matrix=proc[order, :], row_labels=row_labels,
            col_labels=pd.Index(glevels), cmap=cmap, norm=norm,
            body_size_in=(width if isinstance(width, (int, float))
                          else cell_in * len(glevels)),
            layers=layers, title=name if len(col_meta) > 1 else "",
        ))
        matrices[name] = pd.DataFrame(proc[order, :], index=row_labels, columns=glevels)

    # 8. annotation tracks
    left_tracks: list[Track] = []
    # R titles the scale with the transform AND its input, so a reader can tell
    # a z-score of normalised counts from a z-score of raw ones.
    scale_title = f"{exp_method}(normalized counts)" if lib_normalize else str(exp_method)
    legends: list[LegendSpec] = [LegendSpec(
        title=scale_title, kind="continuous", cmap=cmap, vmin=vmin, vmax=vmax)]
    if add_dot:
        legends.append(LegendSpec(title=f"Percent > {exp_cutoff:g}", kind="size"))

    if row_split is not None:
        cats = [str(c) for c in row_split.categories]
        cols = discrete_palette(cats, palette="simspec")
        left_tracks.append(Track(name="Split", side="left", kind="block",
                                 colors=cols, size_in=4 * MM))

    if feature_annotation:
        for key in feature_annotation:
            vals = fetch_var(adata, key)[key].to_numpy()[order]
            cats = [str(x) for x in pd.unique(vals)]
            cols = discrete_palette(cats, palette="Set2", palcolor=feature_annotation_palette)
            left_tracks.append(Track(name=key, side="left", kind="simple",
                                     values=vals, colors=cols, size_in=4 * MM))
            legends.append(LegendSpec(title=key, kind="categorical", mapping=cols))

    # The group colour bar is not opt-in: GroupHeatmap always puts one above the
    # body, titled with the grouping variable, so the columns are readable
    # without tracing back to the axis labels.
    top_tracks: list[Track] = []
    gname = groups[0]
    gcats = [str(c) for c in as_ordered_categorical(obs[gname]).cat.categories]
    gcols = discrete_palette(gcats, palette="Paired")
    for panel in panels:
        panel.top_tracks = [Track(name=gname, side="top", kind="simple",
                                  values=np.array([str(x) for x in panel.col_labels]),
                                  colors=gcols, size_in=4 * MM, show_name=False)]
        panel.title = panel.title or gname
    legends.append(LegendSpec(title=gname, kind="categorical", mapping=gcols))

    if cell_annotation:
        for key in cell_annotation:
            per_group = []
            gcat = as_ordered_categorical(obs[groups[0]][keep])
            ann = as_ordered_categorical(obs[key][keep])
            for lv in col_meta[0][1]:
                m = np.asarray(gcat).astype(str) == lv
                vals = pd.Series(np.asarray(ann)[m])
                per_group.append(str(vals.mode().iloc[0]) if len(vals) else "NA")
            cats = [str(c) for c in as_ordered_categorical(obs[key]).cat.categories]
            cols = discrete_palette(cats, palette="Paired", palcolor=cell_annotation_palette)
            top_tracks.append(Track(name=key, side="top", kind="simple",
                                    values=np.array(per_group), colors=cols, size_in=4 * MM))
            legends.append(LegendSpec(title=key, kind="categorical", mapping=cols))
        for p in panels:
            p.top_tracks = p.top_tracks + list(top_tracks)

    # anno_mark: call out a subset of rows rather than printing every name
    if features_label is not None or (not show_row_names and len(row_labels) > nlabel):
        picks = (list(features_label) if features_label is not None
                 else list(row_labels[np.linspace(0, len(row_labels) - 1,
                                                  min(nlabel, len(row_labels))).astype(int)]))
        at = [int(np.flatnonzero(row_labels == p)[0]) for p in picks
              if (row_labels == p).any()]
        left_tracks.insert(0, Track(name="marks", side="left", kind="mark",
                                    at=at, labels=[str(row_labels[i]) for i in at],
                                    size_in=0.0))

    spec = HeatmapSpec(
        panels=panels, row_split=row_split, row_dend=row_dend,
        left_tracks=left_tracks, legends=legends,
        body_height_in=(height if height is not None
                        else max(cell_in * len(row_labels), 1.0)),
        flip=flip, show_row_names=show_row_names, show_column_names=show_column_names,
    )
    fig = render(spec)
    result = HeatmapResult(fig=fig, matrices=matrices,
                           feature_split=(pd.Series(np.asarray(row_split), index=row_labels)
                                          if row_split is not None else None))
    result.spec = spec  # escape hatch: retune the layout and re-render
    return result


def _split_features(matrix: np.ndarray, n_split: int, method: SplitMethod, *, seed: int):
    """Cluster feature rows into ``n_split`` groups."""
    if method == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=n_split, random_state=seed, n_init=10).fit_predict(matrix)
    if method == "hclust":
        from scipy.cluster.hierarchy import fcluster, linkage
        Z = linkage(matrix, method="complete")
        return fcluster(Z, t=n_split, criterion="maxclust") - 1
    raise NotImplementedError(
        f"split_method={method!r} is specified in docs/porting_briefs/heatmaps.md §3 "
        "but not yet implemented."
    )


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
