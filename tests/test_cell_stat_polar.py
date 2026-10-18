"""``cell_stat_plot(plot_type="pie" / "ring")`` against SCP's ``ggplot_build``.

The fixture ``tests/fixtures/parity/cell_stat/polar.json`` is written by
``tools/parity/cell_stat_polar.R`` on SCP 0.5.6's ``pancreas_sub``: per case the
aggregated input (combination counts and factor levels) and, per panel, every
``geom_col`` rectangle and ``geom_text_repel`` label in coord_polar's own units
(radius in ``[0, 0.4]`` of the panel, angle clockwise from twelve o'clock),
plus the angle and radius breaks.  R is not needed to run these tests.

Regression: a grouped pie used to sum every group's percentages into a single
pie; R draws one concentric band per group.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.patches import Wedge

import scp

FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "cell_stat" / "polar.json"
TOL = 1e-9


@pytest.fixture(scope="module")
def parity():
    return json.loads(FIXTURE.read_text())["cases"]


def _frame(case) -> pd.DataFrame:
    combos = pd.DataFrame(case["combos"])
    n = combos.pop("n").to_numpy()
    df = combos.loc[combos.index.repeat(n)].reset_index(drop=True)
    for c, lv in case["levels"].items():
        df[c] = pd.Categorical(df[c], categories=lv)
    df.index = [f"cell{i}" for i in range(len(df))]
    return df


def _plot(case):
    args = dict(case["args"])
    group_by = case["group_by"]
    return scp.pl.stat_plot(_frame(case), stat_by=case["stat_by"], group_by=group_by,
                            **args)


def _rows(table: dict) -> list[dict]:
    keys = list(table)
    n = len(table[keys[0]])
    rows = [{k: table[k][i] for k in keys} for i in range(n)]
    return [r for r in rows if r.get("x") is not None]  # the ring's dummy NA row


def _xy(r, theta):
    return 0.5 + r * np.sin(theta), 0.5 + r * np.cos(theta)


CASES = ["pie_label", "ring_label", "ring_count", "pie_nogroup", "pie_flip", "pie_individual",
         "ring_individual"]


@pytest.mark.parametrize("key", CASES)
def test_wedges_match_ggplot_build(parity, key):
    case = parity[key]
    fig = _plot(case)
    assert len(fig.axes) == len(case["panels"])
    for ax, (name, panel) in zip(fig.axes, case["panels"].items(), strict=True):
        want = sorted(
            (round(w["r_in"], 9), round(w["r_out"], 9), round(w["theta_0"], 9),
             round(w["theta_1"], 9), w["fill"].lower())
            for w in _rows(panel["wedges"]))
        got = []
        for p in ax.patches:
            if not isinstance(p, Wedge):
                continue
            r_out = p.r
            r_in = 0.0 if p.width is None else p.r - p.width
            th0 = np.radians(90 - p.theta2)
            th1 = np.radians(90 - p.theta1)
            got.append((round(r_in, 9), round(r_out, 9), round(th0, 9), round(th1, 9),
                        to_hex(p.get_facecolor()).lower()))
        got.sort()
        assert len(got) == len(want), name
        for g, w in zip(got, want, strict=True):
            assert np.allclose(g[:4], w[:4], atol=1e-7), (name, g, w)
            assert g[4] == w[4], (name, g, w)


@pytest.mark.parametrize("key", CASES)
def test_axes_match_ggplot_build(parity, key):
    case = parity[key]
    fig = _plot(case)
    for ax, panel in zip(fig.axes, case["panels"].values(), strict=True):
        # radial axis: the group levels (and the ring's blank dummy) on the left
        r_major = np.atleast_1d(panel["r_major"])
        r_labels = np.atleast_1d(panel["r_labels"]).tolist()
        assert np.allclose(ax.get_yticks(), 0.5 + r_major, atol=TOL)
        assert [t.get_text() for t in ax.get_yticklabels()] == r_labels
        # angle labels on the circle at 0.45; the closed circle's ends share one
        thetas = list(panel["theta_major"])
        labels = list(panel["theta_labels"])
        # (the fixture's angles carry 15 digits, so test the gap both ways)
        gap = (thetas[-1] - thetas[0]) % (2 * np.pi)
        if min(gap, 2 * np.pi - gap) < 0.05:
            labels[-1] = f"{labels[0]}/{labels[-1]}"
            labels, thetas = labels[1:], thetas[1:]
        texts = {t.get_text(): t.get_position() for t in ax.texts}
        for lab, th in zip(labels, thetas, strict=True):
            assert lab in texts, (lab, sorted(texts))
            assert np.allclose(texts[lab], _xy(0.45, th), atol=TOL)
        assert ax.get_xlabel() == ("Count" if case["args"].get("stat_type") == "count"
                                   else "Percentage")
        assert ax.get_ylabel() == (case["group_by"] or "")


@pytest.mark.parametrize("key", ["pie_label", "ring_label", "ring_count", "ring_individual"])
def test_labels_match_ggplot_build(parity, key):
    case = parity[key]
    fig = _plot(case)
    for ax, panel in zip(fig.axes, case["panels"].values(), strict=True):
        # R also prints the zeros, piled on a band's start; the port does not
        want = sorted((lab["label"], *(round(v, 9) for v in _xy(lab["r"], lab["theta"])))
                      for lab in _rows(panel["labels"]) if lab["label"] not in ("0", "0%"))
        axis_labels = {t for t in ax.texts if t.get_zorder() == 3}
        got = sorted((t.get_text(), *(round(v, 9) for v in t.get_position()))
                     for t in ax.texts if t not in axis_labels)
        assert [g[0] for g in got] == [w[0] for w in want]
        assert np.allclose([g[1:] for g in got], [w[1:] for w in want], atol=1e-7)


def test_grouped_pie_is_one_band_per_group(parity):
    """The regression itself: three groups give three bands, each a full circle."""
    case = parity["pie_label"]
    ax = _plot(case).axes[0]
    bands: dict[float, float] = {}
    for p in ax.patches:
        if isinstance(p, Wedge):
            bands[round(p.r, 9)] = bands.get(round(p.r, 9), 0.0) + (p.theta2 - p.theta1)
    assert len(bands) == 3
    assert np.allclose(list(bands.values()), 360.0)


def test_cell_stat_plot_string_and_list_stat_by_agree(parity):
    case = parity["pie_label"]
    df = _frame(case)
    a = scp.pl.stat_plot(df, stat_by="CellType", group_by="Phase", plot_type="pie")
    b = scp.pl.stat_plot(df, stat_by=["CellType"], group_by="Phase", plot_type="pie")
    wa = [(p.r, p.theta1, p.theta2) for p in a.axes[0].patches if isinstance(p, Wedge)]
    wb = [(p.r, p.theta1, p.theta2) for p in b.axes[0].patches if isinstance(p, Wedge)]
    assert wa == wb and wa
