"""Flipped heatmaps and the drawn annotations, against SCP's own numbers.

The fixtures come from ``tools/parity/heatmap_fixtures.R`` (SCP 0.5.6 on
``pancreas_sub``, the same input as ``test_heatmap_parity.py``):

* ``*_flip_*.csv`` -- feature metadata of ``flip = TRUE`` calls.  R swaps
  ``cluster_rows`` / ``cluster_columns`` on entry, so these equal the unflipped
  fixtures with the flags swapped; the port must reproduce both facts;
* ``anno_pie*.csv`` / ``anno_violin.csv`` / ``add_violin.csv`` -- the built
  layer data of the ``CellStatPlot`` pies and ``FeatureStatPlot`` violins
  ``GroupHeatmap`` squeezes into its annotation lane and body cells;
* ``separate_*.csv`` -- ``DynamicHeatmap``'s ``separate_annotation`` strips:
  the ``CellDensityPlot`` ridges and the ``DynamicPlot`` curves, with
  RunDynamicFeatures' fitted / upper / lower matrices the curves come from.

Glyphs are compared in the lane's own normalised coordinates (0..1 across
the strip, cells along it), which is what R's grob-in-a-cell drawing fixes.
Densities differ by ``stats::density``'s binning, hence the tolerances there;
everything else is exact.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.patches import Polygon, Wedge

import scp
from scp.heatmap.render import _draw_custom_track

FX = Path(__file__).parent / "fixtures" / "parity" / "heatmap"
GB = ["CellType", "SubCellType"]
LINEAGES = ["Lineage1", "Lineage2"]
KW = {"libsize": "libsize"}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def panc():
    anndata = pytest.importorskip("anndata")
    levels = json.loads((FX / "input_levels.json").read_text())
    counts = pd.read_csv(FX / "input_counts.csv", index_col=0)
    obs = pd.read_csv(FX / "input_obs.csv", index_col=0)
    for key, lv in levels.items():
        obs[key] = pd.Categorical(obs[key], categories=lv)
    obs["Batch"] = pd.Categorical(obs["Batch"], categories=["b1", "b2"])
    obs["S_score"] = pd.read_csv(FX / "input_scores.csv", index_col=0).loc[obs.index, "S_score"]
    ad = anndata.AnnData(X=counts.to_numpy(dtype=float), obs=obs,
                         var=pd.DataFrame(index=counts.columns))
    ad.layers["counts"] = ad.X.copy()
    return ad


@pytest.fixture(scope="module")
def feats(panc):
    return list(panc.var_names)


@pytest.fixture(scope="module")
def dyn_tables():
    return {lin: pd.read_csv(FX / f"dynamic_features_{lin}.csv") for lin in LINEAGES}


def _meta(name: str) -> pd.DataFrame:
    return pd.read_csv(FX / name).set_index("feature")


def _hex(c) -> str:
    return to_hex(c, keep_alpha=False).upper()


def _lane(track, positions, *, size=(2.0, 0.4)):
    """Draw one custom lane alone, as the renderer does, and return its axes."""
    fig = plt.figure(figsize=size)
    ax = fig.add_axes((0, 0, 1, 1))
    _draw_custom_track(ax, track, np.asarray(positions), vertical=False)
    return ax


def _track(res, panel: int, name: str):
    return next(t for t in res.spec.panels[panel].top_tracks if t.name == name)


# --------------------------------------------------------------------------- #
# flip
# --------------------------------------------------------------------------- #
FLIP_CASES = [
    ("group", "group_flip_raw_cluster_columns.csv", "group_raw_cluster_rows.csv",
     {"cluster_columns": True, "exp_method": "raw"}, False),
    ("group", "group_flip_raw_cluster_columns_and_slices.csv",
     "group_raw_cluster_rows_and_slices.csv",
     {"n_split": 4, "split_method": "hclust", "cluster_columns": True,
      "cluster_column_slices": True, "exp_method": "raw"}, True),
    ("feature", "feature_flip_hclust.csv", "feature_hclust.csv",
     {"n_split": 4, "split_method": "hclust", "max_cells": 1000}, True),
    ("dynamic", "dynamic_flip_hclust_peaktime.csv", "dynamic_hclust_peaktime.csv",
     {"n_split": 4, "split_method": "hclust-peaktime"}, True),
]


def _call(kind, panc, feats, dyn_tables, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "group":
            return scp.pl.group_heatmap(panc, feats, GB, **kw, **KW)
        if kind == "feature":
            return scp.pl.feature_heatmap(panc, feats, "CellType", **kw, **KW)
        return scp.pl.dynamic_heatmap(panc, LINEAGES, feats, dynamic_features=dyn_tables,
                                      **kw, **KW)


@pytest.mark.parametrize("kind,name,unflipped,kw,split", FLIP_CASES,
                         ids=[c[1] for c in FLIP_CASES])
def test_flip_feature_order_and_split_match_r(panc, feats, dyn_tables, kind, name, unflipped,
                                              kw, split):
    """flip + cluster_columns orders the features as cluster_rows does unflipped."""
    r = _meta(name)
    pd.testing.assert_frame_equal(r, _meta(unflipped))  # R's own swap
    res = _call(kind, panc, feats, dyn_tables, flip=True, **kw)
    v = res.var_meta.loc[r.index]
    np.testing.assert_array_equal(v["index"].to_numpy(), r["index"].to_numpy())
    if split:
        np.testing.assert_array_equal(v["feature_split"].astype(str).to_numpy(),
                                      r["feature_split"].astype(str).to_numpy())


def test_flip_matrices_are_not_transposed(panc, feats):
    a = scp.pl.group_heatmap(panc, feats, GB, **KW)
    b = scp.pl.group_heatmap(panc, feats, GB, flip=True, **KW)
    for g in GB:
        pd.testing.assert_frame_equal(a.matrices[g], b.matrices[g])


def test_flip_draws_transposed_panels_stacked_vertically(panc, feats):
    res = scp.pl.group_heatmap(panc, feats, GB, n_split=3, split_method="hclust", flip=True,
                               show_row_names=True, **KW)
    fig = res.fig
    n_slices = len(res.spec.row_split.categories)
    bodies = [ax for ax in fig.axes if ax.collections and ax.get_xlim()[1] >= 1
              and ax.collections[0].get_array() is not None
              and ax.collections[0].get_array().ndim == 2 and ax.get_ylim()[0] > 1]
    # one body per (panel, feature slice); observations down, features across
    assert len(bodies) == len(GB) * n_slices
    ys = sorted({round(ax.get_position().y0, 6) for ax in bodies}, reverse=True)
    assert len(ys) == len(GB)  # two rows of bodies: CellType above SubCellType
    first = [ax for ax in bodies if round(ax.get_position().y0, 6) == ys[0]]
    first.sort(key=lambda ax: ax.get_position().x0)
    p = res.spec.panels[0]
    blocks = [np.flatnonzero(np.asarray(res.spec.row_split) == c)
              for c in res.spec.row_split.categories]
    for ax, fpos in zip(first, blocks, strict=True):
        got = np.asarray(ax.collections[0].get_array())
        np.testing.assert_allclose(got, p.matrix[fpos].T)  # groups x features
        assert ax.yaxis_inverted()  # first group on top
    labels = {t.get_text() for ax in fig.axes for t in ax.texts if t.get_rotation() == 90}
    assert set(feats) <= labels  # feature names stand under the bottom lanes


def test_flip_with_enrichment_boxes_raises_as_in_r(panc, feats):
    with pytest.raises(ValueError, match="flip is FALSE"):
        scp.pl.group_heatmap(panc, feats, "CellType", n_split=2, flip=True, anno_terms=True,
                             enrichment=pd.DataFrame(), **KW)


# --------------------------------------------------------------------------- #
# cell_annotation pies and violins (GroupHeatmap)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture,split_by", [("anno_pie.csv", None),
                                              ("anno_pie_split.csv", "Batch")])
def test_cell_annotation_pies_match_cellstatplot(panc, feats, fixture, split_by):
    """Wedge extents and fills of CellStatPlot(plot_type = "pie") per column."""
    r = pd.read_csv(FX / fixture)
    res = scp.pl.group_heatmap(panc, feats, "CellType", split_by=split_by,
                               cell_annotation=["Phase"], **KW)
    track = _track(res, 0, "Phase")
    assert track.size_in == pytest.approx(10 / 25.4)
    labels = list(res.spec.panels[0].col_labels)
    checked = 0
    for j, lab in enumerate(labels):
        parts = lab.split(" : ")
        sub = r[r.subplot == f"CellType:{parts[0]}:{parts[1] if len(parts) > 1 else ''}"]
        sub = sub[sub.ymax > sub.ymin]
        ax = _lane(track, [j], size=(0.6, 0.4))
        wedges = [p for p in ax.patches if isinstance(p, Wedge)]
        got = sorted((round(w.theta1, 9), round(w.theta2, 9), _hex(w.get_facecolor()))
                     for w in wedges)
        want = sorted((round(90 + 360 * (1 - hi), 9), round(90 + 360 * (1 - lo), 9), f.upper())
                      for f, lo, hi in zip(sub["fill"], sub.ymin, sub.ymax, strict=True))
        assert len(got) == len(want)
        for (a1, a2, ca), (b1, b2, cb) in zip(got, want, strict=True):
            assert ca == cb
            assert a1 == pytest.approx(b1, abs=1e-6) and a2 == pytest.approx(b2, abs=1e-6)
        # a circle 80% of the shorter side across: 0.4 in -> radius 0.16 in
        assert wedges[0].r == pytest.approx(0.4 * 0.4)
        checked += 1
    assert checked == len(labels)
    legend = next(lg for lg in res.spec.legends if lg.title == "Phase")
    assert list(legend.mapping) == ["G1", "S", "G2M"]


def _violin_outline(ax):
    """(t, half-width) of the violin polygon drawn in a one-cell lane."""
    (poly,) = [p for p in ax.patches if isinstance(p, Polygon)]
    xy = poly.get_xy()
    n = (len(xy) - 1) // 2 if np.allclose(xy[0], xy[-1]) else len(xy) // 2
    left = xy[:n]
    return left[:, 1], 0.5 - left[:, 0], _hex(poly.get_facecolor())


def _assert_violin(ax, sub, data_range, *, atol):
    t, half, fill = _violin_outline(ax)
    assert fill == sub["fill"].iloc[0].upper()
    lo, hi = sub.y_lo.iloc[0], sub.y_hi.iloc[0]
    # same.y.lims: the panel's data range, expanded 5% a side by coord_cartesian
    np.testing.assert_allclose(scp.heatmap.glyphs.violin_range(data_range), (lo, hi),
                               rtol=1e-9)
    t_r = (sub.y.to_numpy() - lo) / (hi - lo)
    # trim = TRUE: the outline spans exactly the cells' own range
    assert t.min() == pytest.approx(t_r.min(), abs=1e-9)
    assert t.max() == pytest.approx(t_r.max(), abs=1e-9)
    half_r = sub.violinwidth.to_numpy() * 0.45 / 1.2
    np.testing.assert_allclose(np.interp(t_r, t, half), half_r, atol=atol)


def test_numeric_cell_annotation_violins_match_featurestatplot(panc, feats):
    r = pd.read_csv(FX / "anno_violin.csv")
    res = scp.pl.group_heatmap(panc, feats, "CellType", cell_annotation=["S_score"], **KW)
    track = _track(res, 0, "S_score")
    assert all(lg.title != "S_score" for lg in res.spec.legends)  # R draws no legend
    s = panc.obs["S_score"].to_numpy()
    for j, lab in enumerate(res.spec.panels[0].col_labels):
        ax = _lane(track, [j], size=(0.5, 0.5))
        _assert_violin(ax, r[r.subplot == f"S_score:CellType:{lab}:"], (s.min(), s.max()),
                       atol=5e-4)


def test_add_violin_matches_featurestatplot(panc):
    r = pd.read_csv(FX / "add_violin.csv")
    feats = ["Ins1", "Gcg", "Sox9"]
    res = scp.pl.group_heatmap(panc, feats, "CellType", add_violin=True, **KW)
    panel = res.spec.panels[0]
    (layer,) = [lay for lay in panel.layers if lay.kind == "violin"]
    assert [lay.kind for lay in panel.layers] == ["whiteout", "violin"]
    assert all(lg.kind != "continuous" for lg in res.spec.legends)  # body legend dropped
    x = panc[:, feats].X  # R's data slot: the counts, in pancreas_sub
    for i, f in enumerate(panel.row_labels):
        for j, lab in enumerate(panel.col_labels):
            fig = plt.figure(figsize=(0.5, 0.5))
            ax = fig.add_axes((0, 0, 1, 1))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            layer.draw(ax, panel, np.array([i]), np.array([j]))
            sub = r[r.subplot == f"{f}:CellType:{lab}:"]
            if sub.empty:  # fewer than two values: ggplot drops it, so does the port
                assert not ax.patches
                continue
            _assert_violin(ax, sub, (x.min(), x.max()), atol=2e-3)


# --------------------------------------------------------------------------- #
# separate_annotation (DynamicHeatmap)
# --------------------------------------------------------------------------- #
def test_separate_ridges_match_celldensityplot(panc, feats, dyn_tables):
    r = pd.read_csv(FX / "separate_ridges.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = scp.pl.dynamic_heatmap(panc, LINEAGES, feats, dynamic_features=dyn_tables,
                                     separate_annotation=["SubCellType"],
                                     reverse_ht=["Lineage2"], **KW)
    for li, lin in enumerate(LINEAGES):
        rl = r[r.lineage == lin]
        n = len(res.cell_order[lin])
        track = _track(res, li, "SubCellType\n(separate)")
        ax = _lane(track, np.arange(n), size=(4.0, 0.5))
        polys = [p for p in ax.patches if isinstance(p, Polygon)]
        x_lo, x_hi = rl.x_lo.iloc[0], rl.x_hi.iloc[0]
        y_lo, y_hi = rl.y_lo.iloc[0], rl.y_hi.iloc[0]
        assert y_lo == 1
        fills = {_hex(p.get_facecolor()): p for p in polys}
        r_fills = [f.upper() for f in pd.unique(rl["fill"])]
        assert sorted(fills) == sorted(r_fills)
        for fill, g in rl.groupby("fill", sort=False):
            g = g.dropna(subset=["x"])
            poly = fills[fill.upper()]
            xy = poly.get_xy()
            if np.allclose(xy[0], xy[-1]):
                xy = xy[:-1]  # the closing vertex
            m = len(xy) // 2
            u, v = xy[:m, 0], xy[:m, 1]
            # the baseline is the group's slot, and the slots are R's (the top of
            # the strip is the tallest ridge, so it carries the density error)
            base = xy[-1, 1]
            assert base == pytest.approx((g.y.iloc[0] - y_lo) / (y_hi - y_lo), abs=1e-3)
            u_r = (g.x.to_numpy() - x_lo) / (x_hi - x_lo) * n
            v_r = (g.ymax.to_numpy() - y_lo) / (y_hi - y_lo)
            o = np.argsort(u)
            inside = (u_r >= u.min()) & (u_r <= u.max())
            np.testing.assert_allclose(np.interp(u_r[inside], u[o], v[o]), v_r[inside],
                                       atol=1e-3)


def test_separate_curves_match_dynamicplot(panc, feats, dyn_tables):
    """Fitted lines from RunDynamicFeatures' matrices, at DynamicPlot's scales."""
    curves = pd.read_csv(FX / "separate_curves_Lineage1.csv")
    fits = pd.read_csv(FX / "separate_fits_Lineage1.csv")
    entry = {"DynamicFeatures": dyn_tables["Lineage1"]}
    for k, g in fits.groupby("matrix"):
        entry[k] = g.set_index("cell")[["Ins1", "Gcg"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = scp.pl.dynamic_heatmap(panc, ["Lineage1"], feats,
                                     dynamic_features={"Lineage1": entry},
                                     separate_annotation=[["Ins1", "Gcg"]], **KW)
    n = len(res.cell_order["Lineage1"])
    track = _track(res, 0, "Ins1,Gcg\n(separate)")
    ax = _lane(track, np.arange(n), size=(4.0, 0.5))
    lines = ax.get_lines()
    assert len(lines) == 2
    legend = next(lg for lg in res.spec.legends if lg.title == "Features\n(separate)")
    lr = curves[curves.geom == "NewGeomLine"]
    x_lo, x_hi, y_lo, y_hi = lr[["x_lo", "x_hi", "y_lo", "y_hi"]].iloc[0]
    for line, (f, col) in zip(lines, legend.mapping.items(), strict=True):
        assert _hex(line.get_color()) == col.upper()
        g = lr[lr.colour.str.upper() == col.upper()]
        u_r = (g.x.to_numpy() - x_lo) / (x_hi - x_lo) * n
        v_r = (g.y.to_numpy() - y_lo) / (y_hi - y_lo)
        u, v = line.get_xdata(), line.get_ydata()
        np.testing.assert_allclose(np.interp(u_r, u, v), v_r, atol=1e-6, err_msg=f)


def test_heatmap_examples_exist():
    fx = Path(__file__).parent / "fixtures"
    for name in ("group_heatmap_enrichment", "group_heatmap_annotations",
                 "group_heatmap_flip", "dynamic_heatmap_separate"):
        assert (fx / f"example_{name}.png").exists()


def test_cell_cor_flip_labels_the_drawn_rows(panc, feats):
    """CellCorHeatmap(flip = TRUE) draws t(simil): label_by names the drawn axes."""
    def labels(res):
        return sorted(t.get_text() for ax in res.fig.axes for t in ax.texts
                      if t.get_text().replace(".", "").replace("-", "").isdigit())

    lp = panc.copy()
    lp.X = np.log1p(lp.X)  # the data slot is log-normalised
    kw = dict(query_group="SubCellType", ref_group="CellType", features=feats, nlabel=1)
    flipped = scp.pl.cell_cor_heatmap(lp, lp, label_by="row", flip=True, **kw)
    plain = scp.pl.cell_cor_heatmap(lp, lp, label_by="column", **kw)
    assert labels(flipped) == labels(plain) != []
    pd.testing.assert_frame_equal(flipped.matrices["similarity"], plain.matrices["similarity"])
