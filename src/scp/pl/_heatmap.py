"""Heatmaps — the ComplexHeatmap family.

Every function here builds a :class:`scp.heatmap.HeatmapSpec` and renders it.
Keep the split: the wrapper decides *what* the tracks and matrices are, the
renderer decides *where* they go.  That is the only way multi-panel,
split-aware heatmaps stay maintainable.

The three expression heatmaps share one pipeline (``docs/porting_briefs/
heatmaps.md`` §1): fetch, library-normalise, (aggregate,) ``matrix_process``,
clean, split the features, order rows and columns, stack the annotations.
The shared stages live in the private helpers below; the pure transforms they
call (:mod:`scp.heatmap.cluster`, :mod:`scp.heatmap.enrichment`,
:mod:`scp.heatmap.similarity`) are pinned against SCP in
``tests/fixtures/parity/heatmap/``.

Upstream: Hao Zhang, SCP (GPL-3), https://github.com/zhanghao-njmu/SCP —
``R/SCP-plot.R``.  Spec: ``docs/porting_briefs/heatmaps.md``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

from ..fetch import as_ordered_categorical, fetch_data, fetch_var
from ..heatmap import glyphs
from ..heatmap.cluster import (
    AxisOrder,
    SplitMethod,
    axis_order,
    cluster_within_group,
    hclust,
    name_splits_by_group,
    name_splits_by_value,
    reorder_dendrogram,
    split_features,
)
from ..heatmap.enrichment import (
    BoxKind,
    EnrichmentFn,
    enrichment_boxes,
    resolve_enrichment,
    select_enrichment,
)
from ..heatmap.matrix import ExpMethod, clean_nonfinite, color_limits, matrix_process
from ..heatmap.render import MM, render
from ..heatmap.similarity import cell_similarity
from ..heatmap.spec import (
    VIEWPORT_IN,
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

FeatureSplit: TypeAlias = "Sequence[str] | Mapping[str, str] | pd.Series"


# --------------------------------------------------------------------------- #
# shared stages
# --------------------------------------------------------------------------- #
def _cmap(palette: str, palcolor: Sequence[str] | None, name: str):
    ramp = continuous_palette(palette=palette, palcolor=palcolor, n=100)
    # values outside the limits clamp to the terminal colours, as colorRamp2 does
    return LinearSegmentedColormap.from_list(name, ramp, N=256).with_extremes(
        over=ramp[-1], under=ramp[0])


def _cells_mask(adata, cells: Sequence[str] | None) -> np.ndarray:
    if cells is None:
        return np.ones(adata.n_obs, dtype=bool)
    names = np.asarray(adata.obs_names).astype(str)
    want = set(map(str, cells))
    keep = np.array([n in want for n in names])
    if not keep.any():
        raise ValueError("No cells found.")
    if len(want - set(names)):
        warnings.warn("Some cells not found.", stacklevel=3)
    return keep


def _fetch_features(adata, features: Sequence[str] | None, layer: str | None):
    """``features x cells`` for every cell, plus the mask of rows that are genes.

    Features may be ``var_names`` or numeric ``obs`` columns (QC scores as
    pseudo-features); unknown names warn and drop, as in R.
    """
    if features is None:
        raise ValueError("No feature provided.")
    feats = list(dict.fromkeys(map(str, features)))
    frame = fetch_data(adata, feats, layer=layer, on_missing="warn")
    feats = [f for f in feats if f in frame.columns]
    if not feats:
        raise ValueError("none of `features` could be resolved")
    X = frame[feats].to_numpy(dtype=float).T
    var = set(map(str, adata.var_names))
    is_gene = np.array([f in var for f in feats])
    return feats, X, is_gene


def _library_sizes(adata, layer: str | None, libsize) -> np.ndarray:
    if libsize is not None:
        if isinstance(libsize, str):
            return pd.to_numeric(adata.obs[libsize]).to_numpy(dtype=float)
        return np.asarray(libsize, dtype=float)
    # R: colSums(slot(assay, "counts")) -- the counts slot, whatever `slot` is,
    # and the WHOLE transcriptome: with a dozen markers most cells would sum
    # to zero and the whole matrix would come back NaN.
    layers = adata.layers or {}
    full = layers["counts"] if "counts" in layers else (
        layers[layer] if layer in layers else adata.X)
    return np.asarray(full.sum(axis=1)).ravel().astype(float)


def _lib_normalize(adata, X, is_gene, cell_mask, *, layer, lib_normalize, libsize,
                   median_over: np.ndarray | None = None):
    """SCP's library normalisation: divide by library size, scale to the median.

    Applied only when requested and the matrix is non-negative; rows sourced
    from ``obs`` are left alone; non-integer library sizes fall back to 1 with a
    warning (and the legend loses its "normalized").  Returns ``(X, applied)``.
    """
    if not lib_normalize or not np.nanmin(X) >= 0:
        return X, False
    ls_all = _library_sizes(adata, layer, libsize)
    ls = ls_all[cell_mask]
    if np.any(ls % 1 != 0):
        warnings.warn("The values in the 'counts' slot are non-integer. Set the library "
                      "size to 1.", stacklevel=3)
        return X, False
    scale = np.median(ls_all if median_over is None else median_over)
    Y = X.copy()
    Y[is_gene] = Y[is_gene] / ls[None, :] * scale
    return Y, True


def _scale_title(exp_method, layer, normalized: bool) -> str:
    """``paste0(exp_method, "(", data_nm, ")")`` — e.g. ``zscore(normalized counts)``."""
    name = exp_method if isinstance(exp_method, str) else getattr(exp_method, "__name__", "custom")
    slot = layer if layer is not None else "data"
    data_nm = " ".join(x for x in ("normalized" if normalized else "", slot) if x)
    return f"{name}({data_nm})"


def _feature_split_input(feature_split, features: list[str], all_features: list[str]):
    """An explicit ``feature_split`` as a Categorical aligned with ``features``."""
    if isinstance(feature_split, Mapping | pd.Series):
        lab = [None if feature_split.get(f) is None else str(feature_split.get(f))
               for f in features]
    else:
        vals = [str(x) for x in feature_split]
        if len(vals) != len(all_features):
            raise ValueError("feature_split must be the same length as features")
        lookup = dict(zip(all_features, vals, strict=True))
        lab = [lookup[f] for f in features]
    cats = [c for c in dict.fromkeys(x for x in lab if x is not None)]
    if isinstance(feature_split, pd.Series) and isinstance(feature_split.dtype, pd.CategoricalDtype):
        cats = [str(c) for c in feature_split.cat.categories if str(c) in set(cats)]
    return pd.Categorical(lab, categories=cats)


@dataclass
class _Rows:
    order: np.ndarray
    row_split: pd.Categorical | None      # display labels, in display order
    feature_split: pd.Series | None       # C1..Cn per feature, input order
    dend: Dendrogram | None
    split_colors: dict[str, str] | None


def _rows(
    features: list[str],
    *,
    mat_main: np.ndarray,
    mat_split: np.ndarray,
    mat_list: Mapping[str, np.ndarray],
    feature_split,
    all_features: list[str],
    n_split: int | None,
    split_method: SplitMethod,
    fuzzification: float | None,
    seed: int,
    decreasing: bool,
    split_order: Sequence[int] | None,
    namer: Callable[[np.ndarray], tuple[pd.Categorical, pd.Categorical]],
    peak: np.ndarray | None,
    cluster_rows: bool,
    cluster_row_slices: bool,
    cluster_features_by: Sequence[str] | None,
    feature_split_palette: str,
    feature_split_palcolor: Sequence[str] | None,
) -> _Rows:
    """Feature split + row order (``SCP-plot.R`` 8526-8620, and its twins).

    Precedence, as in R: an explicit ``feature_split`` wins; otherwise
    ``n_split`` clusters when it is smaller than the number of features.  The
    row order then comes from, in order of precedence:

    1. ``cluster_row_slices`` without ``cluster_rows`` — ComplexHeatmap's
       ``cluster_within_group`` on ``mat_split``;
    2. ``cluster_rows`` with ``cluster_features_by`` — SCP's
       ``cluster_within_group2`` on those panels (with a split), else an
       ``hclust`` of them reordered by row means;
    3. ``cluster_rows`` — ComplexHeatmap clustering the first panel inside each
       slice, slices optionally ordered by ``cluster_row_slices``;
    4. otherwise the input order inside each slice.
    """
    n = len(features)
    feature_short = None
    row_split = None
    if feature_split is not None:
        row_split = _feature_split_input(feature_split, features, all_features)
        feature_short = row_split
    elif n_split is not None and n <= int(n_split):
        pass  # nrow(mat_split) <= n_split: no split, as in R
    elif n_split is not None:
        labels = split_features(mat_split, int(n_split), split_method, seed=seed,
                                fuzzification=fuzzification, peak=peak)
        feature_short, row_split = namer(labels)

    dend = None
    if row_split is not None and cluster_row_slices and not cluster_rows:
        ax = cluster_within_group(mat_split, row_split)
        dend = Dendrogram(linkage=ax.linkage, order=ax.order, within_group=True,
                          slice_linkages=ax.slice_linkages)
    elif cluster_rows and cluster_features_by:
        mat_cluster = np.hstack([mat_list[k] for k in cluster_features_by])
        if row_split is None:
            # R reorders this tree by colMeans(mat_cluster) -- one weight per
            # *column* for a tree whose leaves are rows, so most leaves get NA
            # weights and the order is an artefact.  The evident intent,
            # weighting each feature by its mean, is used instead.
            Z, order = reorder_dendrogram(hclust(mat_cluster), np.nanmean(mat_cluster, axis=1))
            ax = AxisOrder(order=order, linkage=Z)
            dend = Dendrogram(linkage=Z, order=order)
        else:
            ax = cluster_within_group(mat_cluster, row_split, keep_heights=True)
            dend = Dendrogram(linkage=ax.linkage, order=ax.order, within_group=True,
                              slice_linkages=ax.slice_linkages)
    elif cluster_rows:
        ax = axis_order(mat_main, row_split, cluster=True, cluster_slices=cluster_row_slices)
        dend = Dendrogram(linkage=ax.linkage, order=ax.order,
                          slice_linkages=ax.slice_linkages if row_split is not None else None)
    else:
        ax = axis_order(mat_main, row_split)

    order = np.asarray(ax.order, dtype=int)
    disp = None
    colors = None
    if row_split is not None:
        vals = np.asarray(row_split.astype(object))[order]
        disp = pd.Categorical(vals.astype(str), categories=ax.slices)
        levels = [str(c) for c in row_split.categories]
        colors = discrete_palette(levels, palette=feature_split_palette,
                                  palcolor=feature_split_palcolor)
    fs = None
    if feature_short is not None:
        fs = pd.Series(np.asarray(feature_short.astype(object)), index=pd.Index(features),
                       name="feature_split")
        fs = fs.astype(pd.CategoricalDtype([str(c) for c in feature_short.categories]))
    return _Rows(order=order, row_split=disp, feature_split=fs, dend=dend, split_colors=colors)


def _mark_track(row_labels: pd.Index, *, nlabel: int, features_label, show_row_names: bool):
    """``anno_mark`` positions, chosen exactly as SCP chooses them.

    Without ``features_label``: every feature when there are at most
    ``nlabel``, otherwise ``nlabel`` evenly spaced from
    ``ceiling((n / nlabel) / 2)`` to ``n`` (1-based, rounded).  SCP draws these
    marks even next to printed row names; here they replace the names, so with
    ``show_row_names=True`` marks appear only for an explicit
    ``features_label``.
    """
    n = len(row_labels)
    if features_label is not None:
        want = [str(f) for f in features_label]
        missing = [f for f in want if f not in set(map(str, row_labels))]
        if missing:
            warnings.warn(f"{','.join(missing)} was not found in the features", stacklevel=3)
        at = [i for i, f in enumerate(map(str, row_labels)) if f in set(want)]
    else:
        if show_row_names or nlabel <= 0:
            return None
        if n > nlabel:
            start = int(np.ceil((n / nlabel) / 2))
            at = sorted({int(x) - 1 for x in np.round(np.linspace(start, n, nlabel))})
        else:
            at = list(range(n))
    if not at:
        return None
    return Track(name="marks", side="left", kind="mark", at=at,
                 labels=[str(row_labels[i]) for i in at], size_in=0.0)


def _annotation_track(name: str, values, *, side, size_in, palette, palcolor, show_name=True,
                      reference=None):
    """A simple track plus its legend — the ``anno_simple`` type dispatch.

    Colours come from ``reference`` when given (the whole annotation column, as
    R builds its palette from ``cell_metadata``), so a level absent from one
    panel keeps its colour and the numeric scale is shared.
    """
    v = pd.Series(values)
    ref = v if reference is None else pd.Series(reference)
    if pd.api.types.is_numeric_dtype(ref) and not pd.api.types.is_bool_dtype(ref):
        arr = v.to_numpy(dtype=float)
        full = ref.to_numpy(dtype=float)
        lo, hi = float(np.nanmin(full)), float(np.nanmax(full))
        cmap = _cmap(palette, palcolor, f"anno_{name}")
        norm = Normalize(lo, hi)
        track = Track(name=name, side=side, kind="simple", values=arr, cmap=cmap, norm=norm,
                      size_in=size_in, show_name=show_name)
        legend = LegendSpec(title=name, kind="continuous", cmap=cmap, vmin=lo, vmax=hi)
    else:
        if pd.api.types.is_bool_dtype(ref):
            cat = pd.Series(pd.Categorical(ref, categories=[True, False]))
        else:
            cat = as_ordered_categorical(ref)
        levels = [str(c) for c in cat.cat.categories]
        cols = discrete_palette(levels, palette=palette, palcolor=palcolor)
        vals = np.array([str(x) if pd.notna(x) else "NA" for x in v.astype(object)])
        track = Track(name=name, side=side, kind="simple", values=vals, colors=cols,
                      na_color="none", size_in=size_in, show_name=show_name)
        legend = LegendSpec(title=name, kind="categorical", mapping=cols)
    return track, legend


def _per_item(arg, n: int, name: str) -> list:
    if arg is None or isinstance(arg, str):
        return [arg] * n
    out = list(arg)
    if len(out) == 1:
        out = out * n
    if len(out) != n:
        raise ValueError(f"{name} must be the same length as the annotations it colours")
    return out


def _feature_annotation_tracks(adata, keys, order_features, palettes, palcolors):
    tracks, legends = [], []
    if not keys:
        return tracks, legends
    keys = list(keys)
    pals = _per_item(palettes, len(keys), "feature_annotation_palette")
    pcs = _per_item(palcolors, len(keys), "feature_annotation_palcolor")
    var = fetch_var(adata, keys)
    for key, pal, pc in zip(keys, pals, pcs, strict=True):
        vals = var[key].reindex(order_features)
        t, lg = _annotation_track(key, vals.to_numpy(), side="right", size_in=5 * MM,
                                  palette=pal or "Dark2", palcolor=pc)
        tracks.append(t)
        legends.append(lg)
    return tracks, legends


def _check_enrichment_args(*, flip: bool, anno_terms: bool, anno_keys: bool,
                           anno_features: bool, enrichment, enrichment_fn) -> None:
    """Fail on bad enrichment arguments before any clustering is run.

    The same errors :func:`_enrichment_tracks` raises, checked up front so a
    caller's mistake does not wait for (or need the optional backend of) the
    feature split.
    """
    if not (anno_terms or anno_keys or anno_features):
        return
    if flip:
        raise ValueError(
            "anno_keys, anno_features and anno_terms can only be used when flip is FALSE.")
    if enrichment is None and enrichment_fn is None:
        resolve_enrichment([], pd.Series(dtype="category"))  # raises with the how-to


def _enrichment_tracks(
    *, features: list[str], feature_split: pd.Series | None, row_values: np.ndarray,
    split_colors: dict[str, str] | None, flip: bool,
    anno_terms: bool, anno_keys: bool, anno_features: bool,
    enrichment, enrichment_fn, db, padjust_cutoff, pvalue_cutoff, top_term, show_termid,
    top_word, words_excluded, terms_width, keys_width, features_width,
    terms_fontsize, keys_fontsize, features_fontsize, seed,
):
    """The right-hand text boxes (``heatmap_enrichment``, ``SCP-plot.R`` 7297).

    ``row_values`` maps every displayed row to its slice label (``C1(12)``),
    which is what the boxes align to; the enrichment ``Groups`` are the short
    names (``C1``).  Returns ``(tracks, enrichment_table)``.
    """
    if not (anno_terms or anno_keys or anno_features):
        return [], None
    if flip:
        raise ValueError(
            "anno_keys, anno_features and anno_terms can only be used when flip is FALSE.")
    if feature_split is None:
        groups = pd.Series(pd.Categorical(["1"] * len(features)), index=pd.Index(features))
    else:
        groups = feature_split.astype(str)
        groups = pd.Series(pd.Categorical(groups, categories=[str(c) for c in feature_split.cat.categories]),
                           index=pd.Index(features))
    table = resolve_enrichment(features, groups, enrichment=enrichment,
                               enrichment_fn=enrichment_fn)
    df, metric = select_enrichment(table, db=db, padjust_cutoff=padjust_cutoff,
                                   pvalue_cutoff=pvalue_cutoff)
    if df.empty:
        warnings.warn("No term enriched using the threshold: "
                      f"pvalueCutoff = {pvalue_cutoff}; padjustCutoff = {padjust_cutoff}",
                      stacklevel=3)
        return [], table
    wanted: tuple[tuple[BoxKind, bool], ...] = (
        ("terms", anno_terms), ("keys", anno_keys), ("features", anno_features))
    kinds: list[BoxKind] = [k for k, on in wanted if on]
    boxes = enrichment_boxes(
        df, metric, kinds=kinds, top_term=top_term, show_termid=show_termid,
        top_word=top_word, terms_fontsize=terms_fontsize, keys_fontsize=keys_fontsize,
        features_fontsize=features_fontsize, words_excluded=words_excluded, seed=seed)
    widths = {"terms": terms_width, "keys": keys_width, "features": features_width}
    # each displayed row -> the short group name its box is keyed by
    short = {}
    for f, s in zip(features, groups.astype(str), strict=True):
        short[f] = s
    expected = [str(c) for c in groups.cat.categories]
    tracks: list[Track] = []
    for db_name, per_kind in boxes.items():
        for kind in kinds:
            if kind not in per_kind:
                continue
            if not set(per_kind[kind]) & set(expected):
                # heatmap_enrichment builds a box track only when
                # intersect(geneID_groups, names(<kind>_list)) is non-empty
                # (SCP-plot.R 7385/7462/7506) and drops it silently otherwise;
                # SCP's own RunEnrichment cannot miss, a table passed in can
                found = sorted({str(g) for g in per_kind[kind]})
                warnings.warn(
                    f"anno_{kind}: no {db_name} enrichment group matches a feature cluster, "
                    f"so no {kind} box is drawn (as in SCP). enrichment['Groups'] holds "
                    f"{found[:6]}{'...' if len(found) > 6 else ''}; the clusters are "
                    f"{expected}.", stacklevel=3)
                continue
            tracks.append(Track(name=f"{kind}_empty_{db_name}", side="right", kind="custom",
                                size_in=0.05))
            tracks.append(Track(name=f"{kind}_split_{db_name}", side="right", kind="block",
                                colors=split_colors, size_in=0.1, show_labels=False))
            tracks.append(Track(
                name=f"{kind}_{db_name}", side="right", kind="textbox",
                values=np.asarray([short[f] for f in row_values]), text=per_kind[kind],
                text_mode="lines" if kind == "terms" else "flow",
                size_in=float(widths[kind]), title=f"{db_name} ({kind})", show_name=False))
    return tracks, table


def _var_meta(features, feature_split, order, extra: pd.DataFrame | None = None):
    index = np.empty(len(features), dtype=int)
    index[np.asarray(order)] = np.arange(1, len(features) + 1)
    vm = pd.DataFrame({"features": features}, index=pd.Index(features))
    if extra is not None:
        vm = vm.join(extra)
    vm["feature_split"] = feature_split if feature_split is not None else np.nan
    vm["index"] = index
    return vm


def _column_order(mat: np.ndarray, col_split: pd.Categorical | None, *, cluster_columns: bool,
                  cluster_column_slices: bool):
    """Column order of one panel, as ComplexHeatmap receives it from SCP."""
    if col_split is not None and cluster_column_slices and not cluster_columns:
        present = [c for c in col_split.categories if (np.asarray(col_split) == c).any()]
        if len(present) == 1:
            raise ValueError("cluster_column_slices=TRUE can not be used when there is only one group.")
        ax = cluster_within_group(mat.T, col_split)
        return ax, Dendrogram(linkage=ax.linkage, order=ax.order, within_group=True,
                              slice_linkages=ax.slice_linkages, size_in=0.3)
    if cluster_columns:
        ax = axis_order(mat.T, col_split, cluster=True, cluster_slices=cluster_column_slices)
        return ax, Dendrogram(linkage=ax.linkage, order=ax.order, size_in=0.3,
                              slice_linkages=ax.slice_linkages if col_split is not None else None)
    return axis_order(mat.T, col_split), None


def _fill_key(label: str, split: bool) -> str:
    """``fill.by = "group"``: the column's split level with a split, else its group."""
    parts = str(label).split(" : ")
    return parts[1] if split and len(parts) > 1 else parts[0]


def _glyph_annotation(key, ser: pd.Series, keep: np.ndarray, panels: list[PanelSpec],
                      groups: list[str], cell_lab: Mapping[str, np.ndarray], *, palette: str,
                      group_levels: Mapping[str, list[str]],
                      split_levels: list[str]) -> LegendSpec | None:
    """GroupHeatmap's ``cell_annotation``: a pie or a violin per column (8363-8484).

    Appends a 10 mm custom lane to every panel and returns the legend (pies
    only; R's violin legend is commented out).
    """
    numeric = pd.api.types.is_numeric_dtype(ser) and not pd.api.types.is_bool_dtype(ser)
    split = bool(split_levels)
    if numeric:
        vals = ser.to_numpy(dtype=float)[keep]
        for gi, (p, g) in enumerate(zip(panels, groups, strict=True)):
            lab = cell_lab[g]
            in_panel = np.array([not str(x).startswith("\0") for x in lab], dtype=bool)
            fin = vals[in_panel & np.isfinite(vals)]
            y_range = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
            fills = discrete_palette(split_levels if split else group_levels[g], palette=palette)
            per_col = [vals[lab == str(c)] for c in p.col_labels]
            colors = [fills.get(_fill_key(c, split), "#B3B3B3") for c in p.col_labels]

            def draw_violins(ax, positions, flip=False, per_col=per_col, colors=colors,
                     y_range=y_range):
                for k, j in enumerate(positions):
                    box = (0.0, 1.0, k, k + 1.0) if flip else (k, k + 1.0, 0.0, 1.0)
                    glyphs.draw_violin(ax, box, per_col[j], y_range, colors[j], flip=flip)

            p.top_tracks.append(Track(name=key, side="top", kind="custom", draw=draw_violins,
                                      size_in=10 * MM, show_name=gi == 0))
        return None
    if pd.api.types.is_bool_dtype(ser):
        cat = pd.Series(pd.Categorical(ser, categories=[True, False]), index=ser.index)
    else:
        cat = as_ordered_categorical(ser)
    levels = [str(c) for c in cat.cat.categories]
    codes = np.asarray(cat.cat.codes)[keep]
    has_na = bool((codes < 0).any())
    colors_map = discrete_palette(levels, palette=palette, include_na=has_na)
    wedge_colors = [colors_map[lv] for lv in levels] + ([colors_map["NA"]] if has_na else [])
    for gi, (p, g) in enumerate(zip(panels, groups, strict=True)):
        lab = cell_lab[g]
        fracs = []
        for c in p.col_labels:
            cc = codes[lab == str(c)]
            counts = np.bincount(cc[cc >= 0], minlength=len(levels)).astype(float)
            if has_na:
                counts = np.append(counts, float((cc < 0).sum()))
            fracs.append(counts / max(counts.sum(), 1.0))

        def draw_pies(ax, positions, flip=False, fracs=fracs):
            for k, j in enumerate(positions):
                box = (0.0, 1.0, k, k + 1.0) if flip else (k, k + 1.0, 0.0, 1.0)
                glyphs.draw_pie(ax, box, fracs[j], wedge_colors, flip=flip)

        p.top_tracks.append(Track(name=key, side="top", kind="custom", draw=draw_pies,
                                  size_in=10 * MM, show_name=gi == 0))
    return LegendSpec(title=key, kind="categorical",
                      mapping={lv: colors_map[lv] for lv in levels})


def _violin_values(adata, feats: list[str], keep: np.ndarray) -> pd.DataFrame:
    """``FeatureStatPlot(slot = "data")``'s values: ``adata.X`` (or obs), kept cells."""
    return fetch_data(adata, feats, layer=None).iloc[np.flatnonzero(keep)][feats]


def _violin_layer(values: pd.DataFrame, row_feats: list[str], cols: list[str],
                  cell_lab: np.ndarray, *, fill_by: str, fill_palette: str,
                  fill_palcolor, feature_order: list[str], group_levels: list[str],
                  split_levels: list[str]) -> Layer:
    """``add_violin``: a violin per body cell, one value range per panel (8768)."""
    split = bool(split_levels)
    in_panel = np.array([not str(x).startswith("\0") for x in cell_lab])
    V = values.to_numpy(dtype=float)
    fin = V[in_panel][np.isfinite(V[in_panel])]
    y_range = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
    fidx = {f: k for k, f in enumerate(values.columns)}
    col_cells = {c: np.flatnonzero(cell_lab == str(c)) for c in cols}
    if fill_by == "feature":
        fmap = discrete_palette(feature_order, palette=fill_palette, palcolor=fill_palcolor)

        def color(f, c):
            return fmap[f]
    elif fill_by == "group":
        gmap = discrete_palette(split_levels if split else group_levels, palette=fill_palette,
                                palcolor=fill_palcolor)

        def color(f, c):
            return gmap.get(_fill_key(c, split), "#B3B3B3")
    elif fill_by == "expression":
        med = {(f, c): float(np.nanmedian(V[col_cells[c], fidx[f]])) if len(col_cells[c])
               else np.nan for f in row_feats for c in cols}
        m = np.array([v for v in med.values() if np.isfinite(v)])
        ramp = continuous_palette(palette=fill_palette, palcolor=fill_palcolor, n=100)
        ecmap = LinearSegmentedColormap.from_list("fill_expr", ramp, N=256)
        enorm = Normalize(float(m.min()), float(m.max())) if m.size else Normalize(0, 1)

        def color(f, c):
            v = med[(f, c)]
            return ecmap(enorm(v)) if np.isfinite(v) else "#B3B3B3"
    else:
        raise ValueError("fill_by must be one of 'feature', 'group', 'expression'")

    def draw(ax, panel, rows, cols_idx, flip=False):
        for a, r in enumerate(rows):
            f = row_feats[r]
            for b, j in enumerate(cols_idx):
                c = cols[j]
                box = (a, a + 1.0, b, b + 1.0) if flip else (b, b + 1.0, a, a + 1.0)
                glyphs.draw_violin(ax, box, V[col_cells[c], fidx[f]], y_range, color(f, c),
                                   flip=flip)

    return Layer(kind="violin", draw=draw)


def _body_cap(height, show_row_names: bool) -> float | None:
    """``heatmap_fixsize``'s ceiling for a defaulted body (``SCP-plot.R`` 7608).

    With no ``height`` SCP fills 0.95 npc of the viewport whatever the number
    of features, so a 200-feature heatmap is exactly as tall as a 30-feature
    one: 6.65 in on R's 7-inch device (``heatmap_sizes.json``, read back from
    SCP).  The port keeps its 0.16 in per row for short heatmaps and takes that
    viewport as the ceiling instead.  Printed row names keep their rows -- a
    deliberate divergence: R squeezes them into the viewport and overprints.
    """
    return VIEWPORT_IN if height is None and not show_row_names else None


def _enrich_kwargs(local: dict) -> dict:
    keys = ("anno_terms", "anno_keys", "anno_features", "enrichment", "enrichment_fn", "db",
            "padjust_cutoff", "pvalue_cutoff", "top_term", "show_termid", "top_word",
            "words_excluded", "terms_width", "keys_width", "features_width", "terms_fontsize",
            "keys_fontsize", "features_fontsize", "seed")
    return {k: local[k] for k in keys}


# --------------------------------------------------------------------------- #
# group_heatmap
# --------------------------------------------------------------------------- #
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
    libsize: Sequence[float] | str | None = None,
    limits: tuple[float, float] | None = None,
    aggregate_fun: Callable = np.mean,
    grouping_var: str | None = None,
    numerator: str | None = None,
    # splitting / clustering
    n_split: int | None = None,
    split_method: Literal["kmeans", "hclust", "mfuzz"] = "kmeans",
    fuzzification: float | None = None,
    feature_split: FeatureSplit | None = None,
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
    add_violin: bool = False,
    fill_by: Literal["feature", "group", "expression"] = "feature",
    fill_palette: str = "Dark2",
    fill_palcolor: Sequence[str] | None = None,
    # annotation
    cell_annotation: Sequence[str] | None = None,
    cell_annotation_palette: str | Sequence[str] | None = None,
    feature_annotation: Sequence[str] | None = None,
    feature_annotation_palette: str | Sequence[str] | None = None,
    group_palette: str | Sequence[str] = "Paired",
    cell_split_palette: str = "simspec",
    feature_split_palette: str = "simspec",
    nlabel: int = 20,
    features_label: Sequence[str] | None = None,
    show_row_names: bool = False,
    show_column_names: bool = True,
    # enrichment tracks (pluggable backend, see docs/porting_briefs/heatmaps.md §6)
    anno_terms: bool = False,
    anno_keys: bool = False,
    anno_features: bool = False,
    enrichment: pd.DataFrame | None = None,
    enrichment_fn: EnrichmentFn | None = None,
    db: str | Sequence[str] | None = None,
    padjust_cutoff: float | None = 0.05,
    pvalue_cutoff: float | None = None,
    top_term: int = 5,
    show_termid: bool = False,
    top_word: int = 20,
    words_excluded: Sequence[str] | None = None,
    terms_width: float = 4.0,
    keys_width: float = 2.0,
    features_width: float = 2.0,
    terms_fontsize: float = 8,
    keys_fontsize: tuple[float, float] = (6, 10),
    features_fontsize: tuple[float, float] = (6, 10),
    # sizing
    width: float | Sequence[float] | None = None,
    height: float | None = None,
    flip: bool = False,
    palette: str = "RdBu",
    palcolor: Sequence[str] | None = None,
    seed: int = 11,
) -> HeatmapResult:
    """Group-averaged expression heatmap.

    R original: ``GroupHeatmap`` (``SCP-plot.R`` 7911), by Hao Zhang —
    https://github.com/zhanghao-njmu/SCP.  Pipeline, in order:

    1. fetch ``features x cells`` (features may be ``var_names`` *or* numeric
       ``obs`` columns, so QC scores can be plotted as pseudo-features);
    2. ``lib_normalize`` (default: ``layer == "counts"``) when the matrix is
       non-negative — gene rows only, scaled to the **median** library size of
       the selected cells, not 1e6;
    3. aggregate per group with ``aggregate_fun`` (default mean), and build a
       parallel "percent expressed" matrix with ``x > exp_cutoff``, which is the
       only consumer of ``exp_cutoff`` and the size channel for ``add_dot``;
       with ``split_by`` the columns are ``"group : split"``, one panel per
       ``group_by`` with column slices per group;
    4. :func:`scp.heatmap.matrix_process`, then non-finite cleanup;
    5. derive color limits (:func:`scp.heatmap.color_limits`);
    6. split features (``n_split`` / ``feature_split``) and order rows and
       columns (``cluster_*``, see :func:`scp.heatmap.cluster`);
    7. build the annotation tracks — group blocks, split blocks, feature
       annotations and the enrichment boxes on the right;
    8. lay out and render.

    ``grouping_var`` forces ``exp_method="log2fc"``, bypasses ``matrix_process``
    and shows ``log2(mean(numerator) / mean(other))`` per group; groups lacking
    either arm are dropped and NaN becomes 0.

    Splitting: ``split_method`` is ``kmeans`` / ``hclust`` / ``mfuzz``; clusters
    are named ``C1..Cn`` by the group they peak in, among the panels named in
    ``feature_split_by`` (default all).  ``feature_split`` takes explicit labels
    instead.  Enrichment boxes take a precomputed ``enrichment`` table or an
    ``enrichment_fn`` (:func:`scp.heatmap.enrichment.ora` builds one); see
    :mod:`scp.heatmap.enrichment`.  An enrichment database whose ``Groups``
    match none of the clusters gets no box, as in R, and a warning naming the
    expected cluster names (R's own ``RunEnrichment`` cannot miss).

    Sizing (``heatmap_rendersize`` / ``heatmap_fixsize``, 7543-7672): an
    explicit ``width`` / ``height`` is the body size in inches.  Without one,
    rows are 0.16 in (the dot size with ``add_dot``) rather than R's
    viewport-filling body, but R's viewport is the ceiling: with row names
    hidden, a defaulted body never makes the titles + annotations + body +
    names stack longer than 6.65 in, the 0.95 npc of R's 7-inch device SCP
    fills for 30 and 200 features alike.  The figure is then grown to every
    drawn extent plus ``scp.layout.MARGIN``, so nothing is clipped without
    ``bbox_inches="tight"``.

    Deliberate divergences from R, all bugs or R-only machinery:

    * with ``grouping_var`` *and* ``split_by``, R builds the column split from
      every ``group : split`` level while the matrix keeps only the two-armed
      ones, and aborts when any combination is one-armed; here the split
      follows the kept columns.  Likewise ``grouping_var`` with ``n_split``
      fails in R once a group is dropped (its group means still count every
      level); here clusters are named from the kept columns;
    * ``cluster_features_by`` without a split: R reorders the tree by
      ``colMeans`` (one weight per column for a tree of rows); row means are
      used;
    * with z-scored rows, ComplexHeatmap's ``-rowMeans`` reorder weights are
      rounding noise, so R's within-slice order is a noise-driven flip of the
      same tree; here such ties keep the ``hclust`` order
      (:func:`scp.heatmap.cluster.axis_order`);
    * ``kmeans`` / ``mfuzz`` agree with R as partitions, not label for label;
    * ``anno_keys`` on GO databases uses the tokenising keyword method, not
      ``simplifyEnrichment`` (see :mod:`scp.heatmap.enrichment`);
    * the violins' densities are the exact Gaussian sum rather than
      ``stats::density``'s binned FFT (about 1e-3 of the peak).

    ``cell_annotation`` is drawn as R draws it (``SCP-plot.R`` 8363-8484), one
    glyph per column in a 10 mm lane: a categorical annotation as a pie of its
    levels among the column's cells (``CellStatPlot(plot_type = "pie")``,
    with a legend), a numeric one as a violin of its values filled by the
    column's group -- or split level, with ``split_by`` -- on one value range
    per panel (``FeatureStatPlot(fill.by = "group", same.y.lims = TRUE)``, no
    legend).  Genes are read from ``adata.X``, as R reads ``@data``.
    ``add_violin`` draws, in every body cell, the violin of that feature in
    that column's cells (``adata.X`` again), filled by ``fill_by`` from
    ``fill_palette``, on one value range per panel over every feature; the
    body is whited out underneath and its colour legend dropped, as in R.

    ``flip=True`` transposes every panel and stacks the panels vertically.
    As in R, ``cluster_rows`` / ``cluster_columns`` (and the ``*_slices``
    pair) then name the *drawn* axes -- they are swapped on entry -- and the
    returned matrices are not transposed.  The enrichment boxes cannot be
    combined with ``flip``, as in R.
    """
    if flip:
        # SCP-plot.R 8094: the flags name the drawn axes once flipped
        cluster_rows, cluster_columns = cluster_columns, cluster_rows
        cluster_row_slices, cluster_column_slices = cluster_column_slices, cluster_row_slices
    if group_by is None:
        raise ValueError("`group_by` is required")
    groups = [group_by] if isinstance(group_by, str) else list(group_by)
    obs = adata.obs
    for g in groups + ([split_by] if split_by else []):
        if g not in obs.columns:
            raise KeyError(f"{g} is not in adata.obs")
    if grouping_var is not None:
        if exp_method != "log2fc":
            warnings.warn("When 'grouping_var' is specified, 'exp_method' can only be 'log2fc'",
                          stacklevel=2)
            exp_method = "log2fc"
        if split_by is not None and split_by == grouping_var:
            raise ValueError("'grouping_var' must be different from 'split_by'")
    if feature_split_by is None:
        feature_split_by = list(groups)
    if any(f not in groups for f in feature_split_by):
        raise ValueError("feature_split_by must be a subset of group_by")
    if cluster_features_by is not None and any(f not in groups for f in cluster_features_by):
        raise ValueError("cluster_features_by must be a subset of group_by")

    keep = _cells_mask(adata, cells)
    all_features = [str(f) for f in features] if features is not None else []
    feats, X_all, is_gene = _fetch_features(adata, features, layer)
    X = X_all[:, keep]

    if lib_normalize is None:
        lib_normalize = layer == "counts"
    # median over the selected cells, as R's median(libsize_use)
    ls_sel = _library_sizes(adata, layer, libsize)[keep] if lib_normalize else None
    X, normalized = _lib_normalize(adata, X, is_gene, keep, layer=layer,
                                   lib_normalize=lib_normalize, libsize=libsize,
                                   median_over=ls_sel)
    scale_title = _scale_title(exp_method, layer, normalized)

    gv_true = None
    if grouping_var is not None:
        gv = as_ordered_categorical(obs[grouping_var])
        levels = [str(c) for c in gv.cat.categories]
        if numerator is None:
            numerator = levels[0]
            warnings.warn("'numerator' is not specified. Use the first level in "
                          f"'grouping_var': {numerator}", stacklevel=2)
        elif str(numerator) not in levels:
            raise ValueError(f"'{numerator}' is not an element of the '{grouping_var}'")
        gv_true = (np.asarray(gv.astype(object)).astype(str) == str(numerator))[keep]
        add_dot = False
        scale_title = f"{numerator}/other\n{scale_title}"

    split_vals = None
    split_levels: list[str] = []
    if split_by is not None:
        sc = as_ordered_categorical(obs[split_by])
        split_levels = [str(c) for c in sc.cat.categories]
        split_vals = np.asarray(sc.astype(object)).astype(str)[keep]

    # 3. aggregate per panel -------------------------------------------------
    raw: dict[str, np.ndarray] = {}
    pct: dict[str, np.ndarray] = {}
    columns: dict[str, list[str]] = {}
    col_group: dict[str, list[str]] = {}
    group_levels: dict[str, list[str]] = {}
    for g in groups:
        gc = as_ordered_categorical(obs[g])
        gvals = np.asarray(gc.astype(object))[keep]
        ok = pd.notna(gvals)
        gvals = np.where(ok, gvals, "").astype(str)
        glevels = [str(c) for c in gc.cat.categories if (gvals[ok] == str(c)).any()]
        group_levels[g] = [str(c) for c in gc.cat.categories]
        labels = gvals if split_vals is None else np.char.add(np.char.add(gvals, " : "), split_vals)
        combos = (glevels if split_vals is None
                  else [f"{lv} : {s}" for lv in glevels for s in split_levels])
        cols_g, agg_cols, pct_cols = [], [], []
        for lab in combos:
            m = ok & (labels == lab)
            if not m.any():
                continue
            if gv_true is None:
                block = X[:, m]
                agg_cols.append(aggregate_fun(block, axis=1))
                # the ONLY consumer of exp_cutoff, and the size channel for add_dot
                pct_cols.append((block > exp_cutoff).mean(axis=1))
                cols_g.append(lab)
            else:
                mt, mf = m & gv_true, m & ~gv_true
                if not (mt.any() and mf.any()):
                    continue  # groups lacking both arms are dropped
                with np.errstate(divide="ignore", invalid="ignore"):
                    agg_cols.append(np.log2(aggregate_fun(X[:, mt], axis=1)
                                            / aggregate_fun(X[:, mf], axis=1)))
                pct_cols.append(np.zeros(X.shape[0]))
                cols_g.append(lab)
        if not cols_g:
            raise ValueError(f"no group of {g!r} has cells" +
                             (" in both arms of grouping_var" if gv_true is not None else ""))
        raw[g] = np.column_stack(agg_cols)
        pct[g] = np.column_stack(pct_cols)
        columns[g] = cols_g
        col_group[g] = [c.split(" : ")[0] for c in cols_g]
        if exp_method == "zscore" and gv_true is None and len(glevels) == 1:
            raise ValueError("'zscore' cannot be applied to the group(s) consisting of one "
                             f"element: {g}")

    # 4. transform and clean -------------------------------------------------
    processed: dict[str, np.ndarray] = {}
    for g in groups:
        if gv_true is None:
            processed[g] = clean_nonfinite(matrix_process(raw[g], exp_method))
        else:
            processed[g] = clean_nonfinite(raw[g], na_fill="zero")
    pooled = np.concatenate([processed[g].ravel() for g in groups])
    vmin, vmax = color_limits(pooled.reshape(1, -1),
                              exp_method if isinstance(exp_method, str) else "raw", limits)
    cmap = _cmap(palette, palcolor, "scp_hm")
    norm = Normalize(vmin, vmax)

    # 6. feature split and row order -----------------------------------------
    mat_split = np.hstack([processed[g] for g in feature_split_by])
    split_cols = sum((columns[g] for g in feature_split_by), [])
    level_union = list(dict.fromkeys(
        lv for g in feature_split_by for lv in _panel_levels(columns[g], group_levels[g],
                                                             split_levels)))
    _check_enrichment_args(flip=flip, anno_terms=anno_terms, anno_keys=anno_keys,
                           anno_features=anno_features, enrichment=enrichment,
                           enrichment_fn=enrichment_fn)
    rows = _rows(
        feats, mat_main=processed[groups[0]], mat_split=mat_split, mat_list=processed,
        feature_split=feature_split, all_features=all_features or feats, n_split=n_split,
        split_method=split_method, fuzzification=fuzzification, seed=seed,
        decreasing=decreasing, split_order=split_order,
        namer=lambda lab: name_splits_by_group(lab, mat_split, split_cols, level_union,
                                               decreasing=decreasing, split_order=split_order),
        peak=None, cluster_rows=cluster_rows, cluster_row_slices=cluster_row_slices,
        cluster_features_by=cluster_features_by, feature_split_palette=feature_split_palette,
        feature_split_palcolor=None)
    order = rows.order
    row_labels = pd.Index([feats[i] for i in order])

    # 8. panels ----------------------------------------------------------------
    cell_in = dot_size_mm * MM if add_dot else 0.16
    gpals = _per_item(group_palette, len(groups), "group_palette")
    split_colors = (discrete_palette(split_levels, palette=cell_split_palette)
                    if split_by is not None else None)
    # every kept cell's column label in each panel ("group" or "group : split")
    cell_lab: dict[str, np.ndarray] = {}
    for g in groups:
        gv = np.asarray(as_ordered_categorical(obs[g]).astype(object))[keep]
        clab = np.array([str(v) if pd.notna(v) else "\0" for v in gv], dtype=object)
        if split_vals is not None:
            clab = np.array([f"{a} : {b}" for a, b in zip(clab, split_vals, strict=True)],
                            dtype=object)
        cell_lab[g] = clab
    panels: list[PanelSpec] = []
    matrices: dict[str, pd.DataFrame] = {}
    col_dends: dict[str, Dendrogram] = {}
    legends: list[LegendSpec] = []
    if not add_violin:  # R drops the body's colour legend under violins
        legends.append(LegendSpec(title=scale_title, kind="continuous", cmap=cmap,
                                  vmin=vmin, vmax=vmax))
    if add_dot:
        legends.append(LegendSpec(title=f"Percent > {exp_cutoff:g}", kind="size",
                                  size_in=dot_size_mm * MM))
    violin_data = _violin_values(adata, feats, keep) if add_violin else None
    for gi, g in enumerate(groups):
        proc = processed[g][order, :]
        csplit = (pd.Categorical(col_group[g], categories=group_levels[g])
                  if split_by is not None else None)
        cax, cdend = _column_order(proc, csplit, cluster_columns=cluster_columns,
                                   cluster_column_slices=cluster_column_slices)
        corder = cax.order
        cols = [columns[g][j] for j in corder]
        disp_split = (pd.Categorical([col_group[g][j] for j in corder], categories=cax.slices)
                      if csplit is not None else None)
        if cdend is not None:
            col_dends[g] = cdend
        layers: list[Layer] = []
        if add_bg:
            layers.append(Layer(kind="bg", alpha=bg_alpha))
        if add_reticle:
            layers.append(Layer(kind="reticle", color=reticle_color))
        if add_dot or add_violin:
            # dots / violins replace the body, so it is whited out underneath first
            layers.insert(0, Layer(kind="whiteout"))
        if add_dot:
            layers.append(Layer(kind="dot", size_matrix=pct[g][order][:, corder],
                                base_size_in=dot_size_mm * MM))
        if violin_data is not None:
            layers.append(_violin_layer(
                violin_data, [feats[i] for i in order], cols, cell_lab[g],
                fill_by=fill_by, fill_palette=fill_palette, fill_palcolor=fill_palcolor,
                feature_order=feats, group_levels=group_levels[g], split_levels=split_levels))
        gcols = discrete_palette(group_levels[g], palette=gpals[gi])
        top = [Track(name=g, side="top", kind="simple",
                     values=np.array([c.split(" : ")[0] for c in cols]), colors=gcols,
                     size_in=4 * MM, show_name=True)]
        if split_by is not None:
            top.append(Track(name=split_by, side="top", kind="simple",
                             values=np.array([c.split(" : ")[1] for c in cols]),
                             colors=split_colors, size_in=4 * MM, show_name=gi == 0))
        legends.append(LegendSpec(title=g, kind="categorical", mapping=gcols))
        panels.append(PanelSpec(
            name=g, matrix=proc[:, corder], row_labels=row_labels, col_labels=pd.Index(cols),
            cmap=cmap, norm=norm, col_split=disp_split,
            body_size_in=(float(width) if isinstance(width, (int, float))
                          else (float(width[gi]) if width is not None
                                else cell_in * len(cols))),
            layers=layers, top_tracks=top, title=g))
        matrices[g] = pd.DataFrame(processed[g], index=pd.Index(feats), columns=columns[g])
    if split_by is not None and split_colors is not None:
        legends.append(LegendSpec(title=split_by, kind="categorical", mapping=split_colors))

    if cell_annotation:
        pals = _per_item(cell_annotation_palette, len(cell_annotation), "cell_annotation_palette")
        for key, pal in zip(cell_annotation, pals, strict=True):
            ser = (obs[key] if key in obs.columns
                   else fetch_data(adata, [key], layer=None)[key])  # genes: R reads @data
            lg = _glyph_annotation(key, ser, keep, panels, groups, cell_lab,
                                   palette=pal or "Paired", group_levels=group_levels,
                                   split_levels=split_levels)
            if lg is not None:
                legends.append(lg)

    # 7. side tracks -----------------------------------------------------------
    left_tracks: list[Track] = []
    if rows.row_split is not None:
        left_tracks.append(Track(name="Split", side="left", kind="block",
                                 colors=rows.split_colors, size_in=0.1 + 2 * MM))
        legends.append(LegendSpec(title="Cluster", kind="categorical", mapping={
            k: v for k, v in (rows.split_colors or {}).items()
            if k in set(rows.row_split.categories)}))
    mark = _mark_track(row_labels, nlabel=nlabel, features_label=features_label,
                       show_row_names=show_row_names)
    if mark is not None:
        left_tracks.insert(0, mark)
    right_tracks, f_legends = _feature_annotation_tracks(
        adata, feature_annotation, list(row_labels), feature_annotation_palette, None)
    legends.extend(f_legends)
    etracks, etable = _enrichment_tracks(
        features=feats, feature_split=rows.feature_split,
        row_values=np.asarray(row_labels), split_colors=_slice_colors(rows),
        flip=flip, **_enrich_kwargs(locals()))
    right_tracks.extend(etracks)

    spec = HeatmapSpec(
        panels=panels, row_split=rows.row_split, row_dend=rows.dend, col_dends=col_dends,
        left_tracks=left_tracks, right_tracks=right_tracks, legends=legends,
        body_height_in=(height if height is not None else max(cell_in * len(row_labels), 1.0)),
        body_cap_in=_body_cap(height, show_row_names),
        flip=flip, show_row_names=show_row_names, show_column_names=show_column_names,
    )
    fig = render(spec)
    return HeatmapResult(
        fig=fig, matrices=matrices, feature_split=rows.feature_split,
        var_meta=_var_meta(feats, rows.feature_split, order,
                           _var_extra(adata, feature_annotation, feats)),
        obs_meta=obs.loc[keep, [c for c in dict.fromkeys(
            groups + ([split_by] if split_by else []) + list(cell_annotation or []))
            if c in obs.columns]].copy(),
        enrichment=etable, spec=spec)


def _panel_levels(cols: list[str], glevels: list[str], split_levels: list[str]) -> list[str]:
    """The level order R's ``levels(cell_groups[[g]])`` has for one panel."""
    if not split_levels:
        return [lv for lv in glevels if lv in set(cols)]
    want = [f"{lv} : {s}" for lv in glevels for s in split_levels]
    return [c for c in want if c in set(cols)]


def _slice_colors(rows: _Rows) -> dict[str, str] | None:
    """Split colours keyed by the *display* slice labels (``C1(12)``)."""
    if rows.row_split is None or rows.split_colors is None:
        return None
    return {str(k): rows.split_colors.get(str(k), "#DDDDDD") for k in rows.row_split.categories}


def _var_extra(adata, keys, feats) -> pd.DataFrame | None:
    if not keys:
        return None
    return fetch_var(adata, list(keys)).reindex(feats)


# --------------------------------------------------------------------------- #
# feature_heatmap
# --------------------------------------------------------------------------- #
def feature_heatmap(
    adata,
    features: Sequence[str] | None = None,
    group_by: str | Sequence[str] | None = None,
    *,
    split_by: str | None = None,
    cells: Sequence[str] | None = None,
    max_cells: int = 100,
    cell_order: Sequence[str] | None = None,
    layer: str | None = "counts",
    exp_method: ExpMethod | Callable = "zscore",
    lib_normalize: bool | None = None,
    libsize: Sequence[float] | str | None = None,
    limits: tuple[float, float] | None = None,
    # splitting / clustering
    n_split: int | None = None,
    split_method: Literal["kmeans", "hclust", "mfuzz"] = "kmeans",
    fuzzification: float | None = None,
    feature_split: FeatureSplit | None = None,
    feature_split_by: Sequence[str] | None = None,
    split_order: Sequence[int] | None = None,
    decreasing: bool = False,
    cluster_rows: bool = False,
    cluster_columns: bool = False,
    cluster_row_slices: bool = False,
    cluster_column_slices: bool = False,
    cluster_features_by: Sequence[str] | None = None,
    # annotation
    cell_annotation: Sequence[str] | None = None,
    cell_annotation_palette: str | Sequence[str] | None = None,
    feature_annotation: Sequence[str] | None = None,
    feature_annotation_palette: str | Sequence[str] | None = None,
    group_palette: str | Sequence[str] = "Paired",
    cell_split_palette: str = "simspec",
    feature_split_palette: str = "simspec",
    nlabel: int = 20,
    features_label: Sequence[str] | None = None,
    show_row_names: bool = True,
    show_column_names: bool = False,
    # enrichment tracks
    anno_terms: bool = False,
    anno_keys: bool = False,
    anno_features: bool = False,
    enrichment: pd.DataFrame | None = None,
    enrichment_fn: EnrichmentFn | None = None,
    db: str | Sequence[str] | None = None,
    padjust_cutoff: float | None = 0.05,
    pvalue_cutoff: float | None = None,
    top_term: int = 5,
    show_termid: bool = False,
    top_word: int = 20,
    words_excluded: Sequence[str] | None = None,
    terms_width: float = 4.0,
    keys_width: float = 2.0,
    features_width: float = 2.0,
    terms_fontsize: float = 8,
    keys_fontsize: tuple[float, float] = (6, 10),
    features_fontsize: tuple[float, float] = (6, 10),
    # sizing
    width: float | Sequence[float] | None = None,
    height: float | None = None,
    flip: bool = False,
    palette: str = "RdBu",
    palcolor: Sequence[str] | None = None,
    seed: int = 11,
) -> HeatmapResult:
    """Per-cell (unaggregated) expression heatmap.

    R original: ``FeatureHeatmap`` (``SCP-plot.R`` 9114), by Hao Zhang —
    https://github.com/zhanghao-njmu/SCP.  The :func:`group_heatmap` pipeline
    minus the aggregation: up to ``max_cells`` cells are sampled per group
    (per group *and* split level, then again per group, with ``split_by``),
    each ``group_by`` gets a panel z-scored on its own, and columns are split
    by ``group`` (or ``group : split``) so one group's cells touch.  Feature
    clusters are named by the group whose cells they peak in.

    ``cell_order`` reorders (and restricts) the sampled cells, e.g. by a
    pseudotime.  ``cell_annotation`` entries may be ``obs`` columns or genes;
    genes are read from ``adata.X`` whatever ``layer`` is, reproducing R,
    which hardcodes ``@data`` there.

    Divergences: the sampling uses numpy's generator, so *which* cells are
    drawn differs from R's ``sample()`` (with ``max_cells`` above every group
    size nothing is dropped and the matrices match); kmeans/mfuzz agree as
    partitions; the GO keyword method and the noise-driven zscore reorder are
    as described in :func:`group_heatmap`.

    ``flip=True`` transposes the panels and stacks them vertically; as in R,
    the ``cluster_*`` flags then name the drawn axes (they are swapped on
    entry, ``SCP-plot.R`` 9261).

    Sizing follows :func:`group_heatmap`: R's 6.65 in viewport caps a
    defaulted body only while row names are hidden, and ``show_row_names``
    keeps the port's ``True`` default, so pass ``show_row_names=False`` for a
    long feature list.
    """
    if flip:
        cluster_rows, cluster_columns = cluster_columns, cluster_rows
        cluster_row_slices, cluster_column_slices = cluster_column_slices, cluster_row_slices
    if features is None:
        raise ValueError("`features` is required")
    obs = adata.obs
    if group_by is None:
        groups = ["All.groups"]
        obs = obs.assign(**{"All.groups": pd.Categorical([""] * adata.n_obs)})
    else:
        groups = [group_by] if isinstance(group_by, str) else list(group_by)
    for g in groups + ([split_by] if split_by else []):
        if g not in obs.columns:
            raise KeyError(f"{g} is not in adata.obs")
    if feature_split_by is None:
        feature_split_by = list(groups)
    if any(f not in groups for f in feature_split_by):
        raise ValueError("feature_split_by must be a subset of group_by")

    keep = _cells_mask(adata, cells)
    names = np.asarray(adata.obs_names).astype(str)
    rng = np.random.default_rng(seed)
    split_vals = split_levels = None
    if split_by is not None:
        sc = as_ordered_categorical(obs[split_by])
        split_levels = [str(c) for c in sc.cat.categories]
        split_vals = np.asarray(sc.astype(object)).astype(str)

    # sample cells per group, keeping R's structure (and its random order)
    picks: dict[str, list[int]] = {}
    labels: dict[str, list[str]] = {}
    group_levels: dict[str, list[str]] = {}
    for g in groups:
        gc = as_ordered_categorical(obs[g])
        gstr = np.array([str(v) if pd.notna(v) else "\0" for v in gc.astype(object)])
        group_levels[g] = [str(c) for c in gc.cat.categories]
        pk: list[int] = []
        lb: list[str] = []
        for lv in group_levels[g]:
            idx = np.flatnonzero(keep & (gstr == lv))
            if split_vals is None:
                take = rng.choice(idx, size=min(len(idx), max_cells), replace=False)
                pk.extend(int(i) for i in take)
                lb.extend([lv] * len(take))
            else:
                tmp: list[int] = []
                for sp in split_levels or []:
                    sidx = idx[split_vals[idx] == sp]
                    tmp.extend(int(i) for i in rng.choice(sidx, size=min(len(sidx), max_cells),
                                                          replace=False))
                take = rng.choice(np.asarray(tmp, dtype=int), size=min(len(tmp), max_cells),
                                  replace=False) if tmp else np.array([], dtype=int)
                pk.extend(int(i) for i in take)
                lb.extend(f"{lv} : {split_vals[i]}" for i in take)
        if cell_order is not None:
            pos = {n: k for k, n in enumerate(names[pk])}
            sel = [pos[c] for c in map(str, cell_order) if c in pos]
            pk = [pk[k] for k in sel]
            lb = [lb[k] for k in sel]
        picks[g] = pk
        labels[g] = lb

    all_features = [str(f) for f in features]
    feats, X_all, is_gene = _fetch_features(adata, features, layer)
    union = list(dict.fromkeys(i for g in groups for i in picks[g]))
    umask = np.zeros(adata.n_obs, dtype=bool)
    umask[union] = True
    if lib_normalize is None:
        lib_normalize = layer == "counts"
    ls_all = _library_sizes(adata, layer, libsize) if lib_normalize else None
    # R normalises the union of sampled cells, scaled to their median
    Xn, normalized = _lib_normalize(
        adata, X_all[:, union], is_gene, np.asarray(union), layer=layer,
        lib_normalize=lib_normalize, libsize=libsize,
        median_over=None if ls_all is None else ls_all[union])
    col_of = {c: k for k, c in enumerate(union)}
    scale_title = _scale_title(exp_method, layer, normalized)

    processed: dict[str, np.ndarray] = {}
    for g in groups:
        M = Xn[:, [col_of[i] for i in picks[g]]]
        processed[g] = clean_nonfinite(matrix_process(M, exp_method))
    pooled = np.concatenate([processed[g].ravel() for g in groups])
    vmin, vmax = color_limits(pooled.reshape(1, -1),
                              exp_method if isinstance(exp_method, str) else "raw", limits)
    cmap = _cmap(palette, palcolor, "fh")
    norm = Normalize(vmin, vmax)

    mat_split = np.hstack([processed[g] for g in feature_split_by])
    split_cols = sum((labels[g] for g in feature_split_by), [])
    level_union = list(dict.fromkeys(
        lv for g in feature_split_by
        for lv in _panel_levels(labels[g], group_levels[g], split_levels or [])))
    _check_enrichment_args(flip=flip, anno_terms=anno_terms, anno_keys=anno_keys,
                           anno_features=anno_features, enrichment=enrichment,
                           enrichment_fn=enrichment_fn)
    rows = _rows(
        feats, mat_main=processed[groups[0]], mat_split=mat_split, mat_list=processed,
        feature_split=feature_split, all_features=all_features, n_split=n_split,
        split_method=split_method, fuzzification=fuzzification, seed=seed,
        decreasing=decreasing, split_order=split_order,
        namer=lambda lab: name_splits_by_group(lab, mat_split, split_cols, level_union,
                                               decreasing=decreasing, split_order=split_order,
                                               sort_groups="levels"),
        peak=None, cluster_rows=cluster_rows, cluster_row_slices=cluster_row_slices,
        cluster_features_by=cluster_features_by, feature_split_palette=feature_split_palette,
        feature_split_palcolor=None)
    order = rows.order
    row_labels = pd.Index([feats[i] for i in order])

    gpals = _per_item(group_palette, len(groups), "group_palette")
    split_colors = (discrete_palette(split_levels, palette=cell_split_palette)
                    if split_levels else None)
    legends = [LegendSpec(title=scale_title, kind="continuous", cmap=cmap, vmin=vmin, vmax=vmax)]
    panels, matrices, col_dends = [], {}, {}
    ann_pals = _per_item(cell_annotation_palette, len(cell_annotation or []),
                         "cell_annotation_palette")
    ann_frames = {}
    for key in cell_annotation or []:
        if key in obs.columns:
            ann_frames[key] = obs[key]
        else:  # a gene: SCP reads @data here regardless of `slot`
            ann_frames[key] = fetch_data(adata, [key], layer=None)[key]
    for gi, g in enumerate(groups):
        proc = processed[g][order, :]
        lv_order = _panel_levels(labels[g], group_levels[g], split_levels or [])
        csplit = pd.Categorical(labels[g], categories=lv_order)
        cax, cdend = _column_order(proc, csplit, cluster_columns=cluster_columns,
                                   cluster_column_slices=cluster_column_slices)
        corder = cax.order
        cells_g = [picks[g][j] for j in corder]
        labs = [labels[g][j] for j in corder]
        if cdend is not None:
            col_dends[g] = cdend
        top: list[Track] = []
        if g != "All.groups":
            gcols = discrete_palette(group_levels[g], palette=gpals[gi])
            top.append(Track(name=g, side="top", kind="simple",
                             values=np.array([s.split(" : ")[0] for s in labs]), colors=gcols,
                             size_in=4 * MM, show_name=gi == 0))
            legends.append(LegendSpec(title=g, kind="categorical", mapping=gcols))
        if split_colors is not None:
            top.append(Track(name=str(split_by), side="top", kind="simple",
                             values=np.array([s.split(" : ")[1] for s in labs]),
                             colors=split_colors, size_in=4 * MM, show_name=gi == 0))
        for (key, ser), pal in zip(ann_frames.items(), ann_pals, strict=True):
            t, lg = _annotation_track(key, ser.iloc[cells_g], side="top", size_in=4 * MM,
                                      palette=pal or "Paired", palcolor=None,
                                      show_name=gi == 0, reference=ser.iloc[union])
            top.append(t)
            if gi == 0:
                legends.append(lg)
        panels.append(PanelSpec(
            name=g, matrix=proc[:, corder], row_labels=row_labels,
            col_labels=pd.Index(names[cells_g]), cmap=cmap, norm=norm,
            col_split=pd.Categorical(labs, categories=cax.slices),
            body_size_in=(float(width) if isinstance(width, (int, float))
                          else (float(width[gi]) if width is not None
                                else max(0.02 * len(cells_g), 1.0))),
            title=g if g != "All.groups" else "", top_tracks=top))
        matrices[g] = pd.DataFrame(processed[g], index=pd.Index(feats),
                                   columns=pd.Index(names[picks[g]]))
    if split_colors is not None:
        legends.append(LegendSpec(title=str(split_by), kind="categorical", mapping=split_colors))

    left_tracks: list[Track] = []
    if rows.row_split is not None:
        left_tracks.append(Track(name="Split", side="left", kind="block",
                                 colors=rows.split_colors, size_in=0.1 + 2 * MM))
        legends.append(LegendSpec(title="Cluster", kind="categorical", mapping={
            k: v for k, v in (rows.split_colors or {}).items()
            if k in set(rows.row_split.categories)}))
    mark = _mark_track(row_labels, nlabel=nlabel, features_label=features_label,
                       show_row_names=show_row_names)
    if mark is not None:
        left_tracks.insert(0, mark)
    right_tracks, f_legends = _feature_annotation_tracks(
        adata, feature_annotation, list(row_labels), feature_annotation_palette, None)
    legends.extend(f_legends)
    etracks, etable = _enrichment_tracks(
        features=feats, feature_split=rows.feature_split, row_values=np.asarray(row_labels),
        split_colors=_slice_colors(rows), flip=flip, **_enrich_kwargs(locals()))
    right_tracks.extend(etracks)

    spec = HeatmapSpec(
        panels=panels, row_split=rows.row_split, row_dend=rows.dend, col_dends=col_dends,
        left_tracks=left_tracks, right_tracks=right_tracks, legends=legends,
        body_height_in=height if height is not None else max(0.16 * len(feats), 1.0),
        body_cap_in=_body_cap(height, show_row_names),
        flip=flip, show_row_names=show_row_names,
        show_column_names=show_column_names,  # one column per cell: names are noise
    )
    fig = render(spec)
    obs_cols = [c for c in dict.fromkeys(groups + ([split_by] if split_by else [])) if c in obs]
    return HeatmapResult(
        fig=fig, matrices=matrices, feature_split=rows.feature_split,
        var_meta=_var_meta(feats, rows.feature_split, order,
                           _var_extra(adata, feature_annotation, feats)),
        obs_meta=obs.iloc[union][obs_cols].copy(), enrichment=etable, spec=spec)


# --------------------------------------------------------------------------- #
# dynamic_heatmap
# --------------------------------------------------------------------------- #
_DYN_COLS = ("exp_ncells", "r.sq", "dev.expl", "peaktime", "valleytime", "padjust")


def _dynamic_entry(adata, lineage: str, dynamic_features):
    """What ``srt@tools[["DynamicFeatures_<l>"]]`` holds for one lineage, if supplied.

    ``dynamic_features[lineage]`` (else ``adata.uns["DynamicFeatures_<l>"]``)
    is either the ``DynamicFeatures`` table itself or a mapping in SCP's
    layout: ``"DynamicFeatures"`` plus, optionally, ``"fitted_matrix"``,
    ``"upr_matrix"`` and ``"lwr_matrix"`` (cells x features).
    """
    if dynamic_features is not None and lineage in dynamic_features:
        return dynamic_features[lineage]
    return (adata.uns or {}).get(f"DynamicFeatures_{lineage}")


def _dynamic_table(adata, lineage: str, dynamic_features) -> pd.DataFrame | None:
    """``srt@tools[["DynamicFeatures_<l>"]][["DynamicFeatures"]]``, if supplied."""
    tab = _dynamic_entry(adata, lineage, dynamic_features)
    if isinstance(tab, Mapping):
        tab = tab.get("DynamicFeatures")
    if tab is None:
        return None
    df = pd.DataFrame(tab).copy()
    if "features" in df.columns:
        df = df.set_index(df["features"].astype(str))
    df.index = df.index.astype(str)
    return df


def _dynamic_fits(adata, lineage: str, dynamic_features) -> dict[str, pd.DataFrame] | None:
    """RunDynamicFeatures' fitted / upper / lower matrices, cells x features."""
    entry = _dynamic_entry(adata, lineage, dynamic_features)
    if not isinstance(entry, Mapping):
        return None
    keys = ("fitted_matrix", "upr_matrix", "lwr_matrix")
    if any(entry.get(k) is None for k in keys):
        return None
    out = {}
    for k in keys:
        df = pd.DataFrame(entry[k]).copy()
        df.index = df.index.astype(str)
        out[k] = df
    return out


def _r_rank(x: np.ndarray) -> np.ndarray:
    """``rank(x)``: average ranks for ties, 1-based."""
    from scipy.stats import rankdata

    return rankdata(x, method="average")


def _ridge_track(key: str, ser: pd.Series, cells: np.ndarray, t: np.ndarray, *,
                 reverse: bool, palette: str, palcolor, show_name: bool) -> Track:
    """``separate_annotation`` with a categorical column (SCP-plot.R 11428-11482).

    ``CellDensityPlot(cells = <lineage cells>, group.by = key, features =
    <lineage>, x_order = "rank", decreasing = TRUE, reverse = ...)``: one
    density ridge per group over the ranks of the lineage's pseudotime, groups
    stacked by decreasing median pseudotime (in level order when flipped).
    """
    cat = as_ordered_categorical(ser)
    levels = [str(c) for c in cat.cat.categories]
    colors = discrete_palette(levels, palette=palette, palcolor=palcolor)
    codes = np.asarray(cat.cat.codes)[cells]
    rank = _r_rank(t)
    present = [k for k in range(len(levels)) if (codes == k).any()]
    groups = {levels[k]: rank[codes == k] for k in present}
    med = {levels[k]: float(np.median(t[codes == k])) for k in present}
    by_median = sorted(present, key=lambda k: -med[levels[k]])  # stable: level order on ties
    n = len(t)
    a, b = (float(rank.max()), float(rank.min())) if reverse else (float(rank.min()),
                                                                    float(rank.max()))

    def geometry(flip: bool):
        order = present if flip else by_median
        pos = {levels[k]: float(i + 1) for i, k in enumerate(order)}
        ridges, top = glyphs.ridgelines(groups, pos)
        return ridges, pos, top

    def draw(ax, positions, flip=False):
        ridges, pos, top = geometry(flip)
        glyphs.draw_ridgelines(ax, ridges, pos, colors, x_range=(a, b), y_top=top,
                               n_cells=n, flip=flip)

    track = Track(name=f"{key}\n(separate)", side="top", kind="custom", draw=draw,
                  size_in=10 * MM, show_name=show_name)
    return track


def _curve_track(keys: list[str], fits: dict[str, pd.DataFrame], cell_names: np.ndarray,
                 t: np.ndarray, *, reverse: bool, palette: str, palcolor,
                 show_name: bool) -> tuple[Track, dict[str, str]]:
    """``separate_annotation`` with features (SCP-plot.R 11483-11536).

    ``DynamicPlot(compare_features = TRUE, x_order = "rank", add_rug =
    FALSE)``: each feature's fitted curve and its interval, ``log1p`` of
    RunDynamicFeatures' fitted / upper / lower values, against the rank of the
    lineage's pseudotime.
    """
    colors = discrete_palette(keys, palette=palette, palcolor=palcolor)
    rank = _r_rank(t)
    curves = []
    for f in keys:
        sel = [fits[m].reindex(cell_names)[f].to_numpy(dtype=float)
               for m in ("fitted_matrix", "lwr_matrix", "upr_matrix")]
        with np.errstate(invalid="ignore"):
            y, lwr, upr = (np.log1p(v) for v in sel)
        curves.append((rank, y, lwr, upr, colors[f]))
    n = len(t)

    def draw(ax, positions, flip=False):
        glyphs.draw_curves(ax, curves, n_cells=n, reverse=reverse, flip=flip)

    track = Track(name=f"{','.join(keys)}\n(separate)", side="top", kind="custom", draw=draw,
                  size_in=10 * MM, show_name=show_name)
    return track, colors


def _estimate_dynamics(t: np.ndarray, Y: np.ndarray) -> pd.DataFrame:
    """A GAM-free stand-in for ``RunDynamicFeatures``' peak and valley times.

    R fits ``mgcv::gam`` per feature and takes the median pseudotime among the
    cells whose *fitted* value is above the 99th (below the 1st) percentile.
    Here the fit is replaced by a centred running mean over 5% of the lineage's
    cells (ordered by pseudotime) and the same quantile rule is applied.  It is
    deterministic and tracks the location of the trend's extremes, but it is
    not mgcv: prefer passing ``dynamic_features`` computed in R (or by scFates).
    """
    o = np.argsort(t, kind="stable")
    ts = t[o]
    w = max(3, int(np.ceil(0.05 * len(t))))
    kernel = np.ones(w) / w
    rows = []
    for y in Y[:, o]:
        pad = np.pad(y, (w // 2, w - 1 - w // 2), mode="edge")
        f = np.convolve(pad, kernel, mode="valid")
        hi, lo = np.quantile(f, [0.99, 0.01])
        top = ts[f > hi] if (f > hi).any() else ts[f >= hi]
        bot = ts[f < lo] if (f < lo).any() else ts[f <= lo]
        rows.append((int((y > y.min()).sum()), float(np.median(top)), float(np.median(bot))))
    return pd.DataFrame(rows, columns=["exp_ncells", "peaktime", "valleytime"])


def dynamic_heatmap(
    adata,
    lineages: str | Sequence[str],
    features: Sequence[str] | None = None,
    *,
    use_fitted: bool = False,
    dynamic_features: Mapping[str, pd.DataFrame | Mapping[str, pd.DataFrame]] | None = None,
    min_expcells: int = 20,
    r_sq: float = 0.2,
    dev_expl: float = 0.2,
    padjust: float = 0.05,
    num_intersections: int | Sequence[int] | None = None,
    cell_density: float = 1.0,
    cell_bins: int = 100,
    order_by: Literal["peaktime", "valleytime"] = "peaktime",
    layer: str | None = "counts",
    exp_method: ExpMethod | Callable = "zscore",
    lib_normalize: bool | None = None,
    libsize: Sequence[float] | str | None = None,
    limits: tuple[float, float] | None = None,
    # splitting / clustering
    n_split: int | None = None,
    split_method: SplitMethod = "mfuzz",
    fuzzification: float | None = None,
    feature_split: FeatureSplit | None = None,
    feature_split_by: Sequence[str] | None = None,
    split_order: Sequence[int] | None = None,
    decreasing: bool = False,
    cluster_rows: bool = False,
    cluster_columns: bool = False,
    cluster_row_slices: bool = False,
    cluster_column_slices: bool = False,
    cluster_features_by: Sequence[str] | None = None,
    # annotation
    cell_annotation: Sequence[str] | None = None,
    cell_annotation_palette: str | Sequence[str] | None = None,
    feature_annotation: Sequence[str] | None = None,
    feature_annotation_palette: str | Sequence[str] | None = None,
    separate_annotation: Sequence[str | Sequence[str]] | None = None,
    separate_annotation_palette: str | Sequence[str] = "Paired",
    separate_annotation_palcolor: Sequence[str] | None = None,
    pseudotime_label: Sequence[float] | None = None,
    pseudotime_label_color: str = "black",
    pseudotime_label_linetype: str = "--",
    pseudotime_label_linewidth: float = 1.0,
    pseudotime_palette: str = "cividis",
    feature_split_palette: str = "simspec",
    reverse_ht: Sequence[str] | None = None,
    nlabel: int = 20,
    features_label: Sequence[str] | None = None,
    show_row_names: bool = False,
    show_column_names: bool = False,
    # enrichment tracks
    anno_terms: bool = False,
    anno_keys: bool = False,
    anno_features: bool = False,
    enrichment: pd.DataFrame | None = None,
    enrichment_fn: EnrichmentFn | None = None,
    db: str | Sequence[str] | None = None,
    padjust_cutoff: float | None = 0.05,
    pvalue_cutoff: float | None = None,
    top_term: int = 5,
    show_termid: bool = False,
    top_word: int = 20,
    words_excluded: Sequence[str] | None = None,
    terms_width: float = 4.0,
    keys_width: float = 2.0,
    features_width: float = 2.0,
    terms_fontsize: float = 8,
    keys_fontsize: tuple[float, float] = (6, 10),
    features_fontsize: tuple[float, float] = (6, 10),
    # sizing
    width: float | Sequence[float] | None = None,
    height: float | None = None,
    flip: bool = False,
    palette: str = "viridis",
    palcolor: Sequence[str] | None = None,
    seed: int = 11,
) -> HeatmapResult:
    """Expression along pseudotime, one panel per lineage.

    R original: ``DynamicHeatmap`` (``SCP-plot.R`` 11020), by Hao Zhang —
    https://github.com/zhanghao-njmu/SCP.  Columns are the cells of each
    lineage (an ``obs`` column of pseudotime, NaN off the lineage) sorted by
    pseudotime — descending for lineages in ``reverse_ht`` — and the matrix is
    library-normalised per lineage, concatenated, and passed through
    ``matrix_process`` **once**, so z-scores are joint across lineages.
    ``cell_density < 1`` thins dense stretches of pseudotime over ``cell_bins``
    equal-width bins.  Features are pre-sorted by their peak (or valley) time,
    the row-wise maximum over lineages.

    Peak and valley times come from ``dynamic_features`` — one table per
    lineage with ``RunDynamicFeatures``' columns (``peaktime``,
    ``valleytime``, ``exp_ncells``, ``r.sq``, ``dev.expl``, ``padjust``) — or
    from ``adata.uns["DynamicFeatures_<lineage>"]``.  With ``features=None``
    the features are those passing ``min_expcells`` / ``r_sq`` / ``dev_expl``
    / ``padjust``, kept when they pass in ``num_intersections`` lineages.
    Without a table, peak/valley times are estimated GAM-free (see
    :func:`_estimate_dynamics`) — a documented divergence.

    ``split_method`` adds ``kmeans-peaktime`` / ``hclust-peaktime`` (cluster
    the 1-D peak times) to the matrix methods, and defaults to ``mfuzz`` as in
    R; clusters are named ``C1..Cn`` in order of their mean peak time.

    ``use_fitted=True`` deliberately raises: mgcv's GCV smoothing selection is
    not reproducible with pygam (``docs/07_milestones.md`` §9).

    ``separate_annotation`` adds one 10 mm strip per entry across every
    lineage (``SCP-plot.R`` 11422-11536), drawn from the numbers R's ggplot
    would draw:

    * a categorical ``obs`` column -- ``CellDensityPlot(x_order = "rank",
      decreasing = TRUE)``: a density ridge per group over the ranks of the
      lineage's pseudotime, one joint ``bw.nrd0`` bandwidth, groups stacked by
      decreasing median pseudotime (level order when flipped), reversed for
      ``reverse_ht`` lineages, with a ``<column>\\n(separate)`` legend;
    * a feature, a list of features or a numeric ``obs`` column --
      ``DynamicPlot(compare_features = TRUE, x_order = "rank")``: each
      feature's fitted curve and interval, ``log1p`` of RunDynamicFeatures'
      fitted / upper / lower values.  Those values are an input, as mgcv's fit
      is not reproduced here: pass ``dynamic_features={lineage: {"DynamicFeatures":
      table, "fitted_matrix": ..., "upr_matrix": ..., "lwr_matrix": ...}}``
      (SCP's ``srt@tools`` layout, cells x features) or store that mapping in
      ``adata.uns["DynamicFeatures_<lineage>"]``; without it this raises.

    ``separate_annotation_palette`` / ``_palcolor`` are recycled over the
    entries, as R does.

    ``flip=True`` transposes the panels and stacks them vertically; as in R,
    the ``cluster_*`` flags then name the drawn axes (they are swapped on
    entry).

    With several lineages each gets a 5 mm lane right of the bodies
    (``SCP-plot.R`` 11721): ``#181830`` where the feature is dynamic on that
    lineage (its peak/valley time is known), transparent where it is not,
    named on top and without a legend, as R draws it.  Features passed
    explicitly are fitted on every lineage, so their lanes are solid.  Sizing
    follows :func:`group_heatmap`.

    Deliberate divergences: a lineage whose name is contained in a cell
    barcode no longer corrupts the cell ids (R strips the suffix with
    ``gsub``); with non-integer counts no median scaling is applied (R still
    multiplies by the median library size); the zscore-reorder and kmeans/
    mfuzz notes of :func:`group_heatmap` apply; the ridge densities are the
    exact Gaussian sum rather than ``stats::density``'s binned FFT.
    """
    if use_fitted:
        raise NotImplementedError(
            "`use_fitted=True` needs the GAM backend; see the milestone 9 note in "
            "docs/07_milestones.md -- mgcv's GCV smoothing selection is not "
            "reproducible with pygam, so that path is deliberately not attempted."
        )
    if flip:
        # SCP-plot.R 11168: the flags name the drawn axes once flipped
        cluster_rows, cluster_columns = cluster_columns, cluster_rows
        cluster_row_slices, cluster_column_slices = cluster_column_slices, cluster_row_slices
    lins = [lineages] if isinstance(lineages, str) else list(lineages)
    obs = adata.obs
    for lin in lins:
        if lin not in obs.columns:
            raise KeyError(f"lineages: {lin} is not in adata.obs")
    if feature_split_by is None:
        feature_split_by = list(lins)
    if any(f not in lins for f in feature_split_by):
        raise ValueError("'feature_split_by' must be a subset of lineages.")
    if cluster_features_by is not None and any(f not in lins for f in cluster_features_by):
        raise ValueError("'cluster_features_by' must be a subset of lineages.")
    names = np.asarray(adata.obs_names).astype(str)
    rng = np.random.default_rng(seed)

    PT = np.column_stack([pd.to_numeric(obs[lin], errors="coerce").to_numpy(dtype=float)
                          for lin in lins])
    union = np.flatnonzero(np.isfinite(PT).any(axis=1))
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assign = np.nanmean(PT[union], axis=1)
    keep_cells = union
    if cell_density != 1:
        edges = np.linspace(np.nanmin(assign), np.nanmax(assign), cell_bins)
        b = np.clip(np.searchsorted(edges, assign, side="left"), 1, len(edges) - 1)
        counts = np.bincount(b, minlength=len(edges))
        ncells = int(np.ceil(counts.max() * cell_density))
        kept: list[int] = []
        for k in range(1, len(edges)):
            idx = union[b == k]
            if len(idx):
                kept.extend(int(i) for i in rng.choice(idx, size=min(len(idx), ncells),
                                                       replace=False))
        keep_cells = np.asarray(kept, dtype=np.int64)
    reverse = set(reverse_ht or [])
    cell_order: dict[str, np.ndarray] = {}
    for li, lin in enumerate(lins):
        t = PT[keep_cells, li]
        ok = keep_cells[np.isfinite(t)]
        tt = PT[ok, li]
        o = np.argsort(-tt if lin in reverse else tt, kind="stable")
        cell_order[lin] = ok[o]

    # features and their dynamics -------------------------------------------
    tables = {lin: _dynamic_table(adata, lin, dynamic_features) for lin in lins}
    if features is None:
        if any(tab is None for tab in tables.values()):
            raise ValueError("features=None needs DynamicFeatures tables: pass "
                             "`dynamic_features` or store adata.uns['DynamicFeatures_<lineage>'].")
        picked: list[str] = []
        for lin in lins:
            tab = tables[lin]
            assert tab is not None
            sel = tab[(tab["exp_ncells"] > min_expcells) & (tab["r.sq"] > r_sq)
                      & (tab["dev.expl"] > dev_expl) & (tab["padjust"] < padjust)]
            picked.extend(map(str, sel.index))
        cand = picked
    else:
        cand = [str(f) for f in features]
    counts_tab = pd.Series(cand).value_counts()
    allowed = (set(range(1, len(lins) + 1)) if num_intersections is None else
               {int(num_intersections)} if isinstance(num_intersections, (int, np.integer))
               else set(map(int, num_intersections)))
    # table() sorts its names, so R's feature list comes out alphabetical
    feats0 = sorted(f for f, c in counts_tab.items() if c in allowed)
    if not feats0:
        raise ValueError("No dynamic features found.")
    feats, X_all, is_gene = _fetch_features(adata, feats0, layer)
    if lib_normalize is None:
        lib_normalize = layer == "counts"

    meta = pd.DataFrame(index=pd.Index(feats))
    blocks = []
    normalized = False
    ls_all = _library_sizes(adata, layer, libsize) if lib_normalize else None
    for li, lin in enumerate(lins):
        cols = cell_order[lin]
        mask = np.zeros(adata.n_obs, dtype=bool)
        mask[cols] = True
        Xl = X_all[:, cols]
        # R scales every lineage to the median library size of ALL cells
        Xl, normalized = _lib_normalize(adata, Xl, is_gene, cols, layer=layer,
                                        lib_normalize=lib_normalize, libsize=libsize,
                                        median_over=ls_all)
        blocks.append(Xl)
        tab = tables[lin]
        if tab is None:
            est = _estimate_dynamics(PT[cols, li], Xl)
            est.index = pd.Index(feats)
            tab = est
        tab = tab.reindex(feats)
        for c in _DYN_COLS:
            if c in tab.columns:
                meta[f"{lin}{c}"] = pd.to_numeric(tab[c], errors="coerce").to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        meta[order_by] = np.nanmax(meta[[f"{lin}{order_by}" for lin in lins]].to_numpy(), axis=1)
    pre = np.argsort(-meta[order_by].to_numpy() if decreasing else meta[order_by].to_numpy(),
                     kind="stable")
    feats = [feats[i] for i in pre]
    meta = meta.iloc[pre]
    blocks = [b[pre] for b in blocks]
    is_gene = is_gene[pre]
    scale_title = _scale_title(exp_method, layer, normalized)

    mat = clean_nonfinite(matrix_process(np.hstack(blocks), exp_method))
    bounds = np.cumsum([0] + [b.shape[1] for b in blocks])
    mat_list = {lin: mat[:, bounds[i]:bounds[i + 1]] for i, lin in enumerate(lins)}
    vmin, vmax = color_limits(mat, exp_method if isinstance(exp_method, str) else "raw", limits)
    cmap = _cmap(palette, palcolor, "dh")
    norm = Normalize(vmin, vmax)

    mat_split = np.hstack([mat_list[lin] for lin in feature_split_by])
    peak = meta[order_by].to_numpy(dtype=float)
    _check_enrichment_args(flip=flip, anno_terms=anno_terms, anno_keys=anno_keys,
                           anno_features=anno_features, enrichment=enrichment,
                           enrichment_fn=enrichment_fn)
    rows = _rows(
        feats, mat_main=mat_list[lins[0]], mat_split=mat_split, mat_list=mat_list,
        feature_split=feature_split, all_features=feats, n_split=n_split,
        split_method=split_method, fuzzification=fuzzification, seed=seed,
        decreasing=decreasing, split_order=split_order,
        namer=lambda lab: name_splits_by_value(lab, peak, decreasing=decreasing,
                                               split_order=split_order),
        peak=peak, cluster_rows=cluster_rows, cluster_row_slices=cluster_row_slices,
        cluster_features_by=cluster_features_by, feature_split_palette=feature_split_palette,
        feature_split_palcolor=None)
    order = rows.order
    row_labels = pd.Index([feats[i] for i in order])

    # panels ----------------------------------------------------------------
    pts = PT[keep_cells]
    pt_all = pts[np.isfinite(pts)]
    pcmap = _cmap(pseudotime_palette, None, "pseudotime")
    pnorm = Normalize(float(pt_all.min()), float(pt_all.max()))
    legends = [LegendSpec(title=scale_title, kind="continuous", cmap=cmap, vmin=vmin, vmax=vmax),
               LegendSpec(title="Pseudotime", kind="continuous", cmap=pcmap,
                          vmin=pnorm.vmin, vmax=pnorm.vmax)]
    ann_pals = _per_item(cell_annotation_palette, len(cell_annotation or []),
                         "cell_annotation_palette")
    ann_frames = {}
    for key in cell_annotation or []:
        ann_frames[key] = (obs[key] if key in obs.columns
                           else fetch_data(adata, [key], layer=None)[key])  # R: @data
    panels, matrices, col_dends = [], {}, {}
    separate: list[tuple[list[str], bool, str, Sequence[str] | None]] = []
    if separate_annotation is not None:
        entries = [[e] if isinstance(e, str) else [str(x) for x in e] for e in separate_annotation]
        spals = _per_item(separate_annotation_palette, len(entries), "separate_annotation_palette")
        spcs = (_per_item(None, len(entries), "separate_annotation_palcolor")
                if separate_annotation_palcolor is None
                else [separate_annotation_palcolor] * len(entries))
        for keys, pal, pc in zip(entries, spals, spcs, strict=True):
            col = obs[keys[0]] if len(keys) == 1 and keys[0] in obs.columns else None
            ridge = col is not None and not (pd.api.types.is_numeric_dtype(col)
                                              and not pd.api.types.is_bool_dtype(col))
            missing = [k for k in keys if k not in obs.columns and k not in set(adata.var_names)]
            if missing:
                raise KeyError(f"separate_annotation: {','.join(missing)} is not in adata")
            separate.append((keys, ridge, pal or "Paired", pc))
    for li, lin in enumerate(lins):
        M = mat_list[lin][order]
        cax, cdend = _column_order(M, None, cluster_columns=cluster_columns,
                                   cluster_column_slices=False)
        corder = cax.order
        cells_l = cell_order[lin][corder]
        if cdend is not None:
            col_dends[lin] = cdend
        top = [Track(name="Pseudotime", side="top", kind="simple", values=PT[cells_l, li],
                     cmap=pcmap, norm=pnorm, size_in=4 * MM, show_name=li == 0)]
        for (key, ser), pal in zip(ann_frames.items(), ann_pals, strict=True):
            t, lg = _annotation_track(key, ser.iloc[cells_l], side="top", size_in=4 * MM,
                                      palette=pal or "Paired", palcolor=None,
                                      show_name=li == 0, reference=ser.iloc[keep_cells])
            top.append(t)
            if li == 0:
                legends.append(lg)
        lin_cells = cell_order[lin]
        rev = lin in reverse
        for keys, ridge, pal, pc in separate:
            if ridge:
                top.append(_ridge_track(keys[0], obs[keys[0]], lin_cells, PT[lin_cells, li],
                                        reverse=rev, palette=pal, palcolor=pc,
                                        show_name=li == 0))
                if li == 0:
                    lv = [str(c) for c in as_ordered_categorical(obs[keys[0]]).cat.categories]
                    legends.append(LegendSpec(title=f"{keys[0]}\n(separate)", kind="categorical",
                                              mapping=discrete_palette(lv, palette=pal,
                                                                       palcolor=pc)))
                continue
            fits = _dynamic_fits(adata, lin, dynamic_features)
            if fits is None or any(k not in fits["fitted_matrix"].columns for k in keys):
                raise ValueError(
                    f"separate_annotation {keys} draws RunDynamicFeatures' fitted curves for "
                    f"{lin}, and mgcv's fit is not reproduced here: pass dynamic_features="
                    "{lineage: {'DynamicFeatures': table, 'fitted_matrix': ..., "
                    "'upr_matrix': ..., 'lwr_matrix': ...}} (cells x features) or store that "
                    f"mapping in adata.uns['DynamicFeatures_{lin}'].")
            track, fcols = _curve_track(keys, fits, names[lin_cells], PT[lin_cells, li],
                                        reverse=rev, palette=pal, palcolor=pc,
                                        show_name=li == 0)
            top.append(track)
            if li == 0:
                legends.append(LegendSpec(title="Features\n(separate)", kind="categorical",
                                          mapping=fcols))
        layers: list[Layer] = []
        if pseudotime_label is not None:
            t_sorted = PT[cell_order[lin], li]

            def _draw(ax, panel, rows_, cols_, flip=False, t_sorted=t_sorted):
                for pse in pseudotime_label:  # rank position, as R draws it
                    i = int(np.argmin(np.abs(t_sorted - float(pse)))) + 1
                    line = ax.axhline if flip else ax.axvline
                    line(i, color=pseudotime_label_color, linestyle=pseudotime_label_linetype,
                         linewidth=pseudotime_label_linewidth, zorder=5)
            layers.append(Layer(kind="custom", draw=_draw))
        panels.append(PanelSpec(
            name=lin, matrix=M[:, corder], row_labels=row_labels,
            col_labels=pd.Index(names[cells_l]), cmap=cmap, norm=norm,
            body_size_in=(float(width) if isinstance(width, (int, float))
                          else (float(width[li]) if width is not None
                                else max(min(0.004 * len(cells_l), 3.0), 1.0))),
            layers=layers, top_tracks=top, title=lin))
        matrices[lin] = pd.DataFrame(mat_list[lin], index=pd.Index(feats),
                                     columns=pd.Index(names[cell_order[lin]]))

    left_tracks: list[Track] = []
    if rows.row_split is not None:
        left_tracks.append(Track(name="Split", side="left", kind="block",
                                 colors=rows.split_colors, size_in=0.1 + 2 * MM))
        legends.append(LegendSpec(title="Cluster", kind="categorical", mapping={
            k: v for k, v in (rows.split_colors or {}).items()
            if k in set(rows.row_split.categories)}))
    mark = _mark_track(row_labels, nlabel=nlabel, features_label=features_label,
                       show_row_names=show_row_names)
    if mark is not None:
        left_tracks.insert(0, mark)
    right_tracks: list[Track] = []
    if len(lins) > 1:
        for lin in lins:
            onl = np.isnan(meta.loc[list(row_labels), f"{lin}{order_by}"].to_numpy(dtype=float))
            right_tracks.append(Track(
                name=lin, side="right", kind="simple",
                values=np.where(onl, "1", "0"), colors={"0": "#181830", "1": "none"},
                size_in=5 * MM))
    ftracks, f_legends = _feature_annotation_tracks(
        adata, feature_annotation, list(row_labels), feature_annotation_palette, None)
    right_tracks.extend(ftracks)
    legends.extend(f_legends)
    etracks, etable = _enrichment_tracks(
        features=feats, feature_split=rows.feature_split, row_values=np.asarray(row_labels),
        split_colors=_slice_colors(rows), flip=flip, **_enrich_kwargs(locals()))
    right_tracks.extend(etracks)

    spec = HeatmapSpec(
        panels=panels, row_split=rows.row_split, row_dend=rows.dend, col_dends=col_dends,
        left_tracks=left_tracks, right_tracks=right_tracks, legends=legends,
        body_height_in=height if height is not None else max(0.16 * len(feats), 1.0),
        body_cap_in=_body_cap(height, show_row_names),
        flip=flip, show_row_names=show_row_names, show_column_names=show_column_names,
    )
    fig = render(spec)
    obs_meta = pd.DataFrame({"cells": names[keep_cells]}, index=pd.Index(names[keep_cells]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        obs_meta["Pseudotime_assign"] = np.nanmean(PT[keep_cells], axis=1)
    for li, lin in enumerate(lins):
        obs_meta[lin] = PT[keep_cells, li]
    return HeatmapResult(
        fig=fig, matrices=matrices, feature_split=rows.feature_split,
        var_meta=_var_meta(feats, rows.feature_split, order, meta), obs_meta=obs_meta,
        enrichment=etable, spec=spec,
        cell_order={lin: [str(x) for x in names[cell_order[lin]]] for lin in lins})


# --------------------------------------------------------------------------- #
# correlation heatmaps
# --------------------------------------------------------------------------- #
def cell_cor_heatmap(
    query,
    reference=None,
    *,
    query_group: str | None = None,
    ref_group: str | None = None,
    features: Sequence[str] | None = None,
    nfeatures: int = 2000,
    query_collapsing: bool | None = None,
    ref_collapsing: bool = True,
    distance_metric: str = "cosine",
    layer: str | None = None,
    cluster_rows: bool = False,
    cluster_columns: bool = False,
    nlabel: int = 0,
    label_cutoff: float | None = 0.0,
    label_by: Literal["row", "column", "both"] = "row",
    show_row_names: bool = True,
    show_column_names: bool = True,
    palette: str = "RdBu",
    palcolor: Sequence[str] | None = None,
    width: float | None = None,
    height: float | None = None,
    flip: bool = False,
) -> HeatmapResult:
    """Query-vs-reference similarity heatmap.

    R original: ``CellCorHeatmap`` (``SCP-plot.R`` 10113), by Hao Zhang —
    https://github.com/zhanghao-njmu/SCP, with the matrix from
    ``RunKNNPredict(nn_method = "raw")``.  Rows are the query (cells or
    groups), columns the reference.  Unlike the expression heatmaps the colour
    limits are the plain min/max of the similarity, **not** quantile-clipped.

    Collapsing takes the mean in **linear** space and then applies ``log1p``,
    matching ``AverageExpression``; averaging the logs instead gives a
    geometric mean and a visibly different matrix.

    ``distance_metric`` accepts every metric SCP lists — similarities
    ``cosine``, ``pearson``, ``spearman``, ``correlation``, ``jaccard``,
    ``ejaccard``, ``dice``, ``edice``, ``hamann``, ``simple matching``,
    ``faith``; distances ``euclidean``, ``chisquared``, ``kullback``,
    ``manhattan``, ``maximum``, ``canberra``, ``minkowski``, ``hamming``
    (shown as ``1 - d / max(d)``).  See :func:`scp.heatmap.similarity.
    cell_similarity` for the formulas and the two R bugs fixed there
    (``"hamman"``, and proxyC's cancellation in ``euclidean``).

    As in R:

    * ``reference=None`` compares the query with itself and inherits
      ``query_group`` and ``query_collapsing`` for the reference
      (``SCP-plot.R`` 10150-10156);
    * uncollapsed rows (and columns) are grouped level by level, cells of a
      level in object order, so each group's cells sit together; cells whose
      group is NA are dropped (10218-10241);
    * ``features=None`` takes the stored highly variable genes of the query and
      of the reference (``var["highly_variable"]``, Seurat's
      ``VariableFeatures``) and intersects them (``RunKNNPredict``,
      SCP-cell_annotation.R:122-263).  An object without the flag falls back to
      its ``nfeatures`` most variable genes: R runs ``FindVariableFeatures``
      (vst) there, which needs the counts and a loess fit and is not ported;
    * nothing is labelled by default (``nlabel=0``); labels need a value of at
      least ``label_cutoff`` (default 0) and of the ``nlabel``-th largest in
      their row/column.

    ``flip=True`` draws the reference down the rows (R's ``t(simil_matrix)``).
    As in R, ``label_by`` then names the drawn axes, while ``cluster_rows``
    keeps clustering the query (R swaps the flags and transposes, which
    cancels out); the returned matrix is not transposed.

    Not ported: the query/reference cell annotations, ``bulk_ref``, the
    reduction inputs and ``features_type="DE"``.
    """
    if query_collapsing is None:
        query_collapsing = query_group is not None
    if reference is None:
        if ref_group is not None and ref_group != query_group:
            warnings.warn(f"reference=None compares the query with itself, so ref_group "
                          f"{ref_group!r} is replaced by query_group {query_group!r}, as in R; "
                          "pass the reference explicitly to group it differently.",
                          stacklevel=2)
        reference, ref_group, ref_collapsing = query, query_group, query_collapsing

    feats = list(features) if features is not None else None
    if feats is None:
        def _hvf(adata) -> list[str]:
            names = [str(f) for f in adata.var_names]
            if "highly_variable" in adata.var.columns:
                hv = adata.var["highly_variable"].fillna(False).to_numpy(dtype=bool)
                return [f for f, h in zip(names, hv, strict=True) if h]
            if len(names) <= nfeatures:
                return names
            # no stored HVF: the object's most variable genes stand in for vst
            sub = fetch_data(adata, names, layer=layer).to_numpy(dtype=float)
            order = np.argsort(-np.nanvar(sub, axis=0), kind="stable")[:nfeatures]
            return [names[i] for i in sorted(order)]

        hv_query = set(_hvf(query))
        in_query = set(map(str, query.var_names))
        # intersect(features_ref, features_query), in the reference's order
        feats = [f for f in _hvf(reference) if f in hv_query and f in in_query]
    if not feats:
        raise ValueError("query and reference share no features")

    def _matrix(adata, group, collapse):
        M = fetch_data(adata, feats, layer=layer).to_numpy(dtype=float)  # cells x features
        names = np.asarray(adata.obs_names).astype(str)
        if group is None:
            codes = np.zeros(adata.n_obs, dtype=int)
            levels = [""]
        else:
            cat = as_ordered_categorical(adata.obs[group])
            codes = np.asarray(cat.cat.codes)
            levels = [str(x) for x in cat.cat.categories]
        if not collapse:
            # level by level, object order within a level; NA groups dropped
            idx = np.flatnonzero(codes >= 0)
            idx = idx[np.argsort(codes[idx], kind="stable")]
            return M[idx], [str(x) for x in names[idx]]
        rows, kept = [], []
        for k, lv in enumerate(levels):
            m = codes == k
            if not m.any():
                continue  # levels[levels %in% cell_groups]: empty levels vanish
            # mean in LINEAR space, then log1p -- AverageExpression's convention
            rows.append(np.log1p(np.expm1(M[m]).mean(axis=0)))
            kept.append(lv)
        return np.vstack(rows), kept

    Q, q_labels = _matrix(query, query_group, query_collapsing)
    R, r_labels = _matrix(reference, ref_group, ref_collapsing)

    simil, simil_name = cell_similarity(Q, R, distance_metric)

    cmap = _cmap(palette, palcolor, "cor_hm")
    norm = Normalize(float(np.nanmin(simil)), float(np.nanmax(simil)))

    # ComplexHeatmap defaults: complete-linkage euclidean, reordered by -means
    order_r = np.arange(len(q_labels))
    row_dend = None
    if cluster_rows and len(q_labels) > 2:
        ax = axis_order(simil, None, cluster=True)
        order_r = ax.order
        row_dend = Dendrogram(linkage=ax.linkage, order=order_r)
    order_c = np.arange(len(r_labels))
    col_dends = {}
    if cluster_columns and len(r_labels) > 2:
        cax = axis_order(simil.T, None, cluster=True)
        order_c = cax.order
        col_dends["similarity"] = Dendrogram(linkage=cax.linkage, order=order_c, size_in=0.3)

    M = simil[np.ix_(order_r, order_c)]
    rl = pd.Index([q_labels[i] for i in order_r])
    cl = pd.Index([r_labels[i] for i in order_c])

    # Label the strongest matches, per row and/or column, above a cutoff.
    def _label(ax, panel, rows, cols, flip=False):
        cut = label_cutoff if label_cutoff is not None else -np.inf
        # label_by names the DRAWN axes: flipped, the drawn rows are the reference
        by = {"row": "column", "column": "row"}.get(label_by, label_by) if flip else label_by
        picks: set[tuple[int, int]] = set()
        if by in ("row", "both"):
            for i in range(M.shape[0]):
                thr = np.sort(M[i])[-min(nlabel, M.shape[1])]
                picks |= {(i, j) for j in range(M.shape[1]) if M[i, j] >= max(thr, cut)}
        if by in ("column", "both"):
            colpicks = set()
            for j in range(M.shape[1]):
                thr = np.sort(M[:, j])[-min(nlabel, M.shape[0])]
                colpicks |= {(i, j) for i in range(M.shape[0]) if M[i, j] >= max(thr, cut)}
            picks = (picks & colpicks) if by == "both" else picks | colpicks
        for i, j in picks:
            x, y = (i + 0.5, j + 0.5) if flip else (j + 0.5, i + 0.5)
            # grid.text(round(value, 2)): R prints 1, 0.5, 0.99 -- not 1.00
            t = ax.text(x, y, f"{round(float(M[i, j]), 2) + 0.0:g}", ha="center", va="center",
                        fontsize=6, color="black", zorder=6)
            t.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

    panel = PanelSpec(name="similarity", matrix=M, row_labels=rl, col_labels=cl,
                      cmap=cmap, norm=norm,
                      body_size_in=width if width is not None else 0.2 * len(cl),
                      layers=[Layer(kind="custom", draw=_label)] if nlabel > 0 else [])
    spec = HeatmapSpec(
        panels=[panel], row_dend=row_dend, col_dends=col_dends,
        legends=[LegendSpec(title=simil_name, kind="continuous", cmap=cmap,
                            vmin=norm.vmin, vmax=norm.vmax)],
        body_height_in=height if height is not None else max(0.2 * len(rl), 1.0),
        flip=flip, show_row_names=show_row_names, show_column_names=show_column_names,
    )
    fig = render(spec)
    result = HeatmapResult(fig=fig,
                           matrices={"similarity": pd.DataFrame(M, index=rl, columns=cl)})
    result.spec = spec
    return result


def feature_cor_heatmap(
    adata,
    features: Sequence[str],
    *,
    layer: str | None = None,
    cor_method: Literal["pearson", "spearman"] = "pearson",
    cluster: bool = True,
    palette: str = "RdBu",
    show_row_names: bool = True,
    show_column_names: bool = True,
) -> HeatmapResult:
    """Gene-gene correlation heatmap.

    **The R function is an empty stub** (``SCP-plot.R`` 9980-9982: a body of
    ``{ }``, unexported, no roxygen), so there is nothing to port and this is
    implemented clean-room: correlation over the selected features, optionally
    ordered by average linkage on ``1 - corr``.  SCP is by Hao Zhang —
    https://github.com/zhanghao-njmu/SCP.
    """
    feats = list(features)
    if len(feats) < 2:
        raise ValueError("`features` needs at least two entries")
    X = fetch_data(adata, feats, layer=layer)
    feats = [f for f in feats if f in X.columns]
    M = X[feats].to_numpy(dtype=float)
    if cor_method == "spearman":
        M = np.apply_along_axis(lambda c: pd.Series(c).rank().to_numpy(), 0, M)
    corr = np.corrcoef(M, rowvar=False)

    order = np.arange(len(feats))
    dend = None
    if cluster and len(feats) > 2:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
        d = squareform(np.clip(1 - corr, 0, 2), checks=False)
        Z = linkage(d, method="average")
        order = np.asarray(leaves_list(Z))
        dend = Dendrogram(linkage=Z, order=order)

    C = corr[np.ix_(order, order)]
    labels = pd.Index([feats[i] for i in order])
    cmap = _cmap(palette, None, "fcor")
    norm = Normalize(-1, 1)
    panel = PanelSpec(name="correlation", matrix=C, row_labels=labels, col_labels=labels,
                      cmap=cmap, norm=norm, body_size_in=0.2 * len(labels))
    spec = HeatmapSpec(
        panels=[panel], row_dend=dend,
        legends=[LegendSpec(title=f"{cor_method} r", kind="continuous", cmap=cmap,
                            vmin=-1, vmax=1)],
        body_height_in=0.2 * len(labels),
        show_row_names=show_row_names, show_column_names=show_column_names,
    )
    fig = render(spec)
    result = HeatmapResult(fig=fig,
                           matrices={"correlation": pd.DataFrame(C, index=labels,
                                                                 columns=labels)})
    result.spec = spec
    return result
