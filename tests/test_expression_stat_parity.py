"""``expression_stat_plot`` against what SCP's ``FeatureStatPlot`` builds.

``tools/parity/feature_stat_layers.R`` ran SCP 0.5.6 (ggplot2 4.0.3, ggpubr
0.6.0) on ``pancreas_sub`` after ``NormalizeData`` and dumped, per panel, the
plist name, the final y range (``coord_cartesian`` with its expansion), the y
breaks, and the computed data of the bracket / p-value / trend / count layers.
``cells.csv`` is the per-cell input it used, so these tests rebuild the same
frame and need neither R nor the h5ad.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import scp

FIX = Path(__file__).parent / "fixtures" / "parity" / "feature_stat"
TOL = dict(rel=1e-9, abs=1e-9)


@pytest.fixture(scope="module")
def r():
    return json.loads((FIX / "layers.json").read_text())


@pytest.fixture(scope="module")
def frame(r):
    df = pd.read_csv(FIX / "cells.csv").set_index("cell")
    for col, levels in r["levels"].items():
        df[col] = pd.Categorical(df[col], categories=levels)
    return df


CASES = {
    "violin": dict(stat_by=["G2M_score", "Neurog3"], group_by="CellType"),
    "box_brackets": dict(stat_by="G2M_score", group_by="CellType", plot_type="box",
                         comparisons=[("Ductal", "Ngn3 low EP"), ("Ngn3 high EP", "Endocrine"),
                                      ("Ductal", "Endocrine")]),
    "violin_brackets_format": dict(stat_by=["S_score", "Sox9"], group_by="CellType",
                                   comparisons=[("Ductal", "Ngn3 low EP"),
                                                ("Ngn3 low EP", "Ngn3 high EP")],
                                   sig_label="p.format"),
    "brackets_ttest_multi": dict(stat_by="G2M_score", group_by="CellType",
                                 comparisons=[("Ductal", "Pre-endocrine"),
                                              ("Ngn3 high EP", "Endocrine")],
                                 pairwise_method="t.test", multiplegroup_comparisons=True),
    "multi_only": dict(stat_by=["G2M_score", "Neurog3"], group_by="CellType",
                       multiplegroup_comparisons=True, sig_label="p.format"),
    "multi_anova": dict(stat_by="S_score", group_by="CellType", multiplegroup_comparisons=True,
                        multiple_method="anova"),
    "split_true_kruskal": dict(stat_by=["S_score", "Neurog3"], group_by="CellType",
                               split_by="Phase", comparisons=True, sig_label="p.format"),
    "split_true_pairwise_small": dict(stat_by=["G2M_score", "Sox9"], group_by="CellType",
                                      split_by="Phase", cells="small", comparisons=True,
                                      sig_label="p.format"),
    "small_brackets": dict(stat_by=["G2M_score", "Neurog3"], group_by="CellType", cells="small",
                           comparisons=[("Ductal", "Ngn3 low EP"), ("Ductal", "Endocrine")],
                           sig_label="p.format"),
    "bar": dict(stat_by="G2M_score", group_by="CellType", plot_type="bar"),
    "bar_trend_split": dict(stat_by="G2M_score", group_by="CellType", split_by="Phase",
                            plot_type="bar", add_trend=True),
    "violin_trend": dict(stat_by="G2M_score", group_by="CellType", add_trend=True),
    "violin_trend_split": dict(stat_by="G2M_score", group_by="CellType", split_by="Phase",
                               add_trend=True),
    "dot": dict(stat_by="G2M_score", group_by="CellType", plot_type="dot"),
    "individual": dict(stat_by="G2M_score", group_by="CellType", split_by="Phase",
                       individual=True),
    "by_feature": dict(stat_by=["Neurog3", "Sox9", "G2M_score"], group_by="CellType",
                       plot_by="feature"),
    "stack_rows": dict(stat_by=["Neurog3", "Sox9", "Ins1", "Gcg"], group_by="SubCellType",
                       stack=True),
}


def _layers(panel, kind):
    out = []
    for layer in panel["layers"]:
        if kind in layer["kind"]:
            out.extend(layer.get("data") or [])
    return out


def _build(frame, r, name):
    kw = dict(CASES[name])
    if kw.get("cells") == "small":
        kw["cells"] = r["small_cells"]
    fig = scp.pl.expression_stat_plot(frame, **kw)
    rec = fig._scp_stat_record
    return fig, rec


@pytest.mark.parametrize("name", list(CASES))
def test_panels_and_y_range_match_r(frame, r, name):
    fig, rec = _build(frame, r, name)
    want = r["cases"][name]
    names = want["names"] if isinstance(want["names"], list) else [want["names"]]
    assert list(rec) == names
    for key in names:
        lo, hi = want["panels"][key]["y_range"]
        if lo is None:  # an empty panel (no cells in that level x split)
            continue
        assert rec[key]["ylim"][0] == pytest.approx(lo, **TOL), key
        assert rec[key]["ylim"][1] == pytest.approx(hi, **TOL), key
    plt.close(fig)


@pytest.mark.parametrize("name", [n for n in CASES if "brackets" in n])
def test_brackets_match_ggsignif(frame, r, name):
    fig, rec = _build(frame, r, name)
    for key, panel in r["cases"][name]["panels"].items():
        rows = _layers(panel, "Signif")
        got = rec[key]["brackets"]
        assert len(rows) == 3 * len(got)
        # ggsignif emits each bracket from the group whose id equals its first
        # level's x position, so rows are not in comparison order; the step
        # index i is the suffix of `group`
        by_step = {}
        for row in rows:
            by_step.setdefault(int(row["group"].rsplit("-", 1)[1]), []).append(row)
        for i, (x1, x2, ytop, ytip, lab) in enumerate(got):
            tip, bar, _ = by_step[i + 1]
            assert sorted([bar["x"], bar["xend"]]) == [min(x1, x2) + 1, max(x1, x2) + 1]
            assert ytop == pytest.approx(bar["y"], **TOL)
            assert ytip == pytest.approx(tip["y"], **TOL)
            assert lab == bar["annotation"]
    plt.close(fig)


@pytest.mark.parametrize("name", [n for n in CASES if "multi" in n or "split_true" in n])
def test_p_value_labels_match_stat_compare_means(frame, r, name):
    fig, rec = _build(frame, r, name)
    for key, panel in r["cases"][name]["panels"].items():
        rows = _layers(panel, "CompareMeans")
        got = rec[key]["texts"]
        assert len(got) == len(rows), key
        for (x, y, lab), row in zip(got, rows, strict=True):
            assert x + 1 == row["x"]
            assert y == pytest.approx(row["y"], **TOL)
            assert lab == row["label"]
        # (level, p) rows; bracket rows are ((a, b), p, label) and checked above
        ps = [row[1] for row in rec[key]["sig"] if len(row) == 2]
        assert ps == pytest.approx([row["p"] for row in rows], rel=1e-9)
    plt.close(fig)


@pytest.mark.parametrize("name", ["bar_trend_split", "violin_trend", "violin_trend_split"])
def test_trend_points_match_r(frame, r, name):
    fig, rec = _build(frame, r, name)
    for key, panel in r["cases"][name]["panels"].items():
        rows = _layers(panel, "GeomLine")
        want = sorted((round(row["x"], 9), row["y"]) for row in rows)
        got = sorted((round(x + 1, 9), y) for x, y in rec[key]["trend"])
        # R's line layer only holds groups with >1 point when split; points are
        # drawn for every group, so compare R's points as a subset
        got_d = dict(got)
        for x, y in want:
            assert got_d[x] == pytest.approx(y, **TOL)
    plt.close(fig)


def test_dot_bins_and_counts_match_geom_count(frame, r):
    fig, rec = _build(frame, r, "dot")
    for key, panel in r["cases"]["dot"]["panels"].items():
        rows = _layers(panel, "GeomPoint")
        want = sorted((row["x"], round(row["y"], 9), row["n"]) for row in rows)
        got = sorted((x + 1, round(y, 9), n) for x, y, n in rec[key]["dots"])
        assert got == want
    plt.close(fig)


def test_stack_rows_have_two_ticks_at_r_breaks(frame, r):
    fig, rec = _build(frame, r, "stack_rows")
    want = r["cases"]["stack_rows"]
    panel_axes = [ax for ax in fig.axes if ax.get_visible()][: len(want["names"])]
    for ax, key in zip(panel_axes, want["names"], strict=True):
        p = want["panels"][key]
        assert list(ax.get_yticks()) == pytest.approx(p["y_breaks"], **TOL)
        assert [t.get_text() for t in ax.get_yticklabels()] == [
            f"{v:g}" for v in p["y_labels"]]
        assert ax.get_ylim() == pytest.approx(tuple(p["y_range"]), **TOL)
    plt.close(fig)


def test_calculate_coexp_matches_r(r):
    anndata = pytest.importorskip("anndata")
    df = pd.read_csv(FIX / "cells.csv").set_index("cell")
    ad = anndata.AnnData(X=df[["Neurog3", "Sox9", "Ins1", "Gcg"]].to_numpy(),
                         obs=df[["CellType"]].astype("category"))
    ad.var_names = ["Neurog3", "Sox9", "Ins1", "Gcg"]
    co = scp.fetch.calculate_coexp(ad, ["Neurog3", "Sox9"])
    assert co.to_numpy()[:50] == pytest.approx(np.asarray(r["coexp"], dtype=float), **TOL)
    fig = scp.pl.feature_stat_plot(ad, ["Neurog3", "Sox9"], group_by="CellType",
                                   calculate_coexp=True)
    assert [k.split(":")[0] for k in fig._scp_stat_record] == ["Neurog3", "Sox9", "CoExp"]
    plt.close(fig)
