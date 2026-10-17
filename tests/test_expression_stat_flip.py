"""``expression_stat_plot(flip=True)`` against SCP's ``coord_flip`` output.

``tools/parity/feature_stat_layers.R`` also builds each case below with
``flip = TRUE`` and writes ``layers_flip.json``: per panel, both axes' ranges
as drawn (after ``coord_flip`` the horizontal axis is the value axis), the
group labels bottom-up, the aspect ratio, and the computed data of the stripe,
bar, bracket, p-value, trend and count layers — which stay in aesthetic space
(x = group position, y = value).  The tests check the numbers *and* that every
layer was drawn transposed.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import scp

FIX = Path(__file__).parent / "fixtures" / "parity" / "feature_stat"
TOL = dict(rel=1e-9, abs=1e-9)
_STACK = ["Neurog3", "Sox9", "Ins1", "Gcg"]
CASES = {
    "violin": dict(stat_by=["G2M_score", "Neurog3"], group_by="CellType"),
    "box_brackets": dict(stat_by="G2M_score", group_by="CellType", plot_type="box",
                         comparisons=[("Ductal", "Ngn3 low EP"), ("Ngn3 high EP", "Endocrine"),
                                      ("Ductal", "Endocrine")]),
    "split_true_kruskal": dict(stat_by="S_score", group_by="CellType", split_by="Phase",
                               comparisons=True, sig_label="p.format"),
    "multi_only": dict(stat_by="G2M_score", group_by="CellType",
                       multiplegroup_comparisons=True, sig_label="p.format"),
    "bar_bg": dict(stat_by="G2M_score", group_by="SubCellType", bg_by="CellType",
                   plot_type="bar"),
    "violin_trend_split": dict(stat_by="G2M_score", group_by="CellType", split_by="Phase",
                               add_trend=True),
    "dot": dict(stat_by="G2M_score", group_by="CellType", plot_type="dot"),
    "col": dict(stat_by="G2M_score", group_by="CellType", cells="small", plot_type="col"),
    "sorted": dict(stat_by="Neurog3", group_by="SubCellType", sort=True, aspect_ratio=0.5),
    "by_feature": dict(stat_by=["Neurog3", "Sox9", "G2M_score"], group_by="CellType",
                       plot_by="feature"),
    "stack": dict(stat_by=_STACK, group_by="SubCellType", stack=True),
}
_R_COLORS = {"grey85": "#d9d9d9"}


@pytest.fixture(scope="module")
def r():
    return json.loads((FIX / "layers.json").read_text())


@pytest.fixture(scope="module")
def rf():
    return json.loads((FIX / "layers_flip.json").read_text())


@pytest.fixture(scope="module")
def frame(r):
    df = pd.read_csv(FIX / "cells.csv").set_index("cell")
    for col, levels in r["levels"].items():
        df[col] = pd.Categorical(df[col], categories=levels)
    return df


def _layers(panel, kind):
    out = []
    for layer in panel["layers"]:
        if kind in layer["kind"]:
            out.extend(layer.get("data") or [])
    return out


def _names(want):
    return want["names"] if isinstance(want["names"], list) else [want["names"]]


def _build(frame, r, name, *, flip=True):
    kw = dict(CASES[name], flip=flip)
    if kw.get("cells") == "small":
        kw["cells"] = r["small_cells"]
    fig = scp.pl.expression_stat_plot(frame, **kw)
    rec = fig._scp_stat_record
    axes = [ax for ax in fig.axes if ax.get_visible() and ax.get_label() != "<colorbar>"]
    return fig, rec, dict(zip(rec, axes, strict=True))


@pytest.mark.parametrize("name", list(CASES))
def test_flip_transposes_axes_as_coord_flip(frame, r, rf, name):
    """Value range on the horizontal axis, groups reversed up the vertical one."""
    fig, rec, axes = _build(frame, r, name)
    want = rf["cases"][name]
    assert list(rec) == _names(want)
    shift = 0 if name == "col" else 1  # R puts groups at 1..n; cells are 1..n in both
    for key in _names(want):
        p, ax = want["panels"][key], axes[key]
        assert ax.get_xlim() == pytest.approx(tuple(p["x_range"]), **TOL), key
        assert rec[key]["ylim"] == pytest.approx(tuple(p["x_range"]), **TOL), key
        assert ax.get_ylim() == pytest.approx(tuple(np.subtract(p["y_range"], shift)), **TOL)
        if name == "col":
            assert ax.get_yticks().size == 0
        elif name == "stack" and key != _names(want)[0]:
            pass  # R blanks axis.text.y right of the first column (checked below)
        else:
            # R lists the discrete breaks bottom-up: the factor, reversed
            assert [t.get_text() for t in ax.get_yticklabels()] == p["y_labels"], key
        if p["aspect_ratio"] is not None:
            assert ax.get_box_aspect() == pytest.approx(p["aspect_ratio"])
    plt.close(fig)


@pytest.mark.parametrize("name", ["violin", "bar_bg", "sorted", "col", "stack"])
def test_flip_stripes_are_horizontal_bands_at_r_positions(frame, r, rf, name):
    fig, rec, axes = _build(frame, r, name)
    shift = 0 if name == "col" else 1
    for key, p in rf["cases"][name]["panels"].items():
        rows = [row for row in _layers(p, "GeomRect") if row["fill"] != "transparent"]
        bands = [b for b in axes[key].patches if b.get_zorder() == 0.6]
        assert len(bands) == len(rows), key
        got = sorted((b.get_y(), b.get_y() + b.get_height(), to_hex(b.get_facecolor()))
                     for b in bands)
        want = sorted(rows, key=lambda q: q["xmax"] if q["xmax"] is not None else 1e9)
        for (lo, hi, col), row in zip(got, want, strict=True):
            assert col == _R_COLORS.get(row["fill"], row["fill"].lower())
            for v, rv in ((lo, row["xmin"]), (hi, row["xmax"])):
                if rv is None:  # -Inf / Inf: the outer band bleeds past the panel
                    assert abs(v) > 1e5
                elif name == "col" and rv == 0.5:
                    # R's `xmin == min(xmax)` typo leaves this edge finite; the
                    # port bleeds it, as it does without flip
                    assert v < -1e5
                else:
                    assert v == pytest.approx(rv - shift, **TOL)
    plt.close(fig)


def test_flip_brackets_are_drawn_transposed(frame, r, rf):
    fig, rec, axes = _build(frame, r, "box_brackets")
    for key, panel in rf["cases"]["box_brackets"]["panels"].items():
        by_step = {}
        for row in _layers(panel, "Signif"):
            by_step.setdefault(int(row["group"].rsplit("-", 1)[1]), []).append(row)
        drawn = [ln for ln in axes[key].lines if len(ln.get_xdata()) == 4]
        assert len(drawn) == len(rec[key]["brackets"]) == len(by_step)
        for i, ((x1, x2, ytop, ytip, lab), ln) in enumerate(
                zip(rec[key]["brackets"], drawn, strict=True)):
            tip, bar, _ = by_step[i + 1]
            # ggsignif's own row order within a bracket varies; the span does not
            assert sorted([x1 + 1, x2 + 1]) == sorted([bar["x"], bar["xend"]])
            assert ytop == pytest.approx(bar["y"], **TOL)
            assert ytip == pytest.approx(tip["y"], **TOL)
            assert lab == bar["annotation"]
            # value along x, group position along y
            assert list(ln.get_xdata()) == pytest.approx([ytip, ytop, ytop, ytip], **TOL)
            assert list(ln.get_ydata()) == [x1, x1, x2, x2]
    plt.close(fig)


def test_flip_compare_means_labels(frame, r, rf):
    fig, rec, axes = _build(frame, r, "split_true_kruskal")
    for key, panel in rf["cases"]["split_true_kruskal"]["panels"].items():
        rows = _layers(panel, "CompareMeans")
        got = rec[key]["texts"]
        assert len(got) == len(rows)
        anns = [t for t in axes[key].texts if t.get_text().startswith("p = ")]
        for (x, y, lab), row, ann in zip(got, rows, anns, strict=True):
            assert x + 1 == row["x"]
            assert y == pytest.approx(row["y"], **TOL)
            assert lab == row["label"]
            assert ann.xy == pytest.approx((y, x), **TOL)
    plt.close(fig)


def test_flip_multigroup_label_is_kept(frame, r, rf):
    """R bug fixed: under flip SCP reverses the x factor after building the
    group key, so ggpubr tests within each x, kruskal.test fails and R drops
    the label.  The port keeps it, at the first (bottom) position."""
    want = rf["cases"]["multi_only"]
    assert _layers(want["panels"][_names(want)[0]], "CompareMeans") == []
    fig, rec, _ = _build(frame, r, "multi_only")
    flat, rec0, _ = _build(frame, r, "multi_only", flip=False)
    key = next(iter(rec))
    ((x, y, lab),) = rec[key]["texts"]
    assert x == 0.0
    assert (y, lab) == (rec0[key]["texts"][0][1], rec0[key]["texts"][0][2])
    plt.close(fig)
    plt.close(flat)


def test_flip_trend_points_and_lines(frame, r, rf):
    fig, rec, axes = _build(frame, r, "violin_trend_split")
    for key, panel in rf["cases"]["violin_trend_split"]["panels"].items():
        got = {round(x + 1, 9): y for x, y in rec[key]["trend"]}
        for row in _layers(panel, "GeomLine"):
            assert got[round(row["x"], 9)] == pytest.approx(row["y"], **TOL)
        joins = [ln for ln in axes[key].lines if len(ln.get_xdata()) > 1]
        assert joins
        for ln in joins:  # medians along x, dodged positions along y
            for xv, yv in zip(ln.get_xdata(), ln.get_ydata(), strict=True):
                assert got[round(yv + 1, 9)] == pytest.approx(xv, **TOL)
    plt.close(fig)


def test_flip_dot_counts(frame, r, rf):
    fig, rec, _ = _build(frame, r, "dot")
    for key, panel in rf["cases"]["dot"]["panels"].items():
        want = sorted((row["x"], round(row["y"], 9), row["n"])
                      for row in _layers(panel, "StatSum"))
        assert sorted((x + 1, round(y, 9), n) for x, y, n in rec[key]["dots"]) == want
    plt.close(fig)


@pytest.mark.parametrize("name, shift", [("bar_bg", 1), ("col", 0)])
def test_flip_bars_are_horizontal(frame, r, rf, name, shift):
    """Bars become barh at R's positions (``col`` numbers the cells from the top)."""
    fig, rec, axes = _build(frame, r, name)
    for key, panel in rf["cases"][name]["panels"].items():
        bars = [b for b in axes[key].patches if b.get_zorder() == 2]
        got = sorted((round(b.get_y() + b.get_height() / 2 + shift, 9), b.get_width())
                     for b in bars)
        want = sorted((round(row["x"], 9), row["y"]) for row in _layers(panel, "GeomCol"))
        assert [g[0] for g in got] == [w[0] for w in want]
        assert [g[1] for g in got] == pytest.approx([w[1] for w in want], **TOL)
    plt.close(fig)


def test_flip_stack_columns_have_two_ticks_at_r_breaks(frame, r, rf):
    fig, rec, axes = _build(frame, r, "stack")
    want = rf["cases"]["stack"]
    first = None
    for key in _names(want):
        p, ax = want["panels"][key], axes[key]
        assert list(ax.get_xticks()) == pytest.approx(p["x_breaks"], **TOL)
        assert [t.get_text() for t in ax.get_xticklabels()] == [f"{v:g}" for v in p["x_labels"]]
        assert ax.get_xlim() == pytest.approx(tuple(p["x_range"]), **TOL)
        assert ax.get_ylim() == pytest.approx(tuple(np.subtract(p["y_range"], 1)), **TOL)
        first = first or ax
        # one row: the panels sit side by side, the group labels only at the left
        assert ax.get_position().y0 == pytest.approx(first.get_position().y0)
        assert ax.yaxis.get_tick_params()["labelleft"] is (ax is first)
    plt.close(fig)
