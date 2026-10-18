"""``expression_stat_plot(y_trans="log10" / "log2")`` on data with zeros, vs SCP.

``tests/fixtures/parity/feature_stat/layers_log.json`` is written by
``tools/parity/feature_stat_layers.R``.  SCP stops on these calls (the raw
minimum is 0, ``log10(0)`` is ``-Inf`` and the break computation errors), so
the fixture holds what SCP draws once given ``y.min`` = the smallest positive
value: the limit ggplot arrives at after dropping what the transform makes
non-finite.  The port drops those values itself, so its *default* call must
match.

Regression: the lower limit used to be clamped to 1e-300, the 5 % expansion
then spanned hundreds of decades, and violins were computed on the linear
scale (rectangles on a log axis reaching 1e18).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection

import scp
from scp._breaks import extended_breaks
from scp.pl._scales import format_breaks, log_breaks

FIX = Path(__file__).parent / "fixtures" / "parity" / "feature_stat"
TOL = dict(rel=1e-9, abs=1e-9)
LEVELS = ["Ductal", "Ngn3 low EP", "Ngn3 high EP", "Pre-endocrine", "Endocrine"]


@pytest.fixture(scope="module")
def r():
    return json.loads((FIX / "layers_log.json").read_text())


def _frame(name: str) -> pd.DataFrame:
    df = pd.read_csv(FIX / name).set_index("cell")
    df["CellType"] = pd.Categorical(df["CellType"], categories=LEVELS)
    return df


CASES = {
    "log10_counts": ("counts.csv", dict(stat_by="Ins1", y_trans="log10")),
    "log10_data": ("cells.csv", dict(stat_by="Ins1", y_trans="log10")),
    "log2_box_counts": ("counts.csv", dict(stat_by=["Ins1", "Gcg"], y_trans="log2",
                                           plot_type="box")),
}


def _build(name):
    csv, kw = CASES[name]
    with pytest.warns(UserWarning, match="transformation introduced infinite values"):
        fig = scp.pl.expression_stat_plot(_frame(csv), group_by="CellType", **kw)
    return fig


def test_r_stops_where_the_port_drops(r):
    assert r["default_error"] == "'from' must be a finite number"


@pytest.mark.parametrize("name", list(CASES))
def test_limits_and_breaks_match_r(r, name):
    fig = _build(name)
    rec = fig._scp_stat_record
    want = r["cases"][name]
    names = want["names"] if isinstance(want["names"], list) else [want["names"]]
    assert list(rec) == names
    for ax, key in zip(fig.axes, names, strict=True):
        panel = want["panels"][key]
        lo, hi = panel["y_range"]
        assert rec[key]["ylim"][0] == pytest.approx(lo, **TOL), key
        assert rec[key]["ylim"][1] == pytest.approx(hi, **TOL), key
        breaks = [b for b in panel["y_breaks"] if b is not None]
        labels = [lab for lab in panel["y_labels"] if lab is not None]
        assert np.allclose(ax.get_yticks(), breaks, atol=1e-12)
        fig.canvas.draw()
        assert [t.get_text() for t in ax.get_yticklabels()] == labels
    plt.close(fig)


@pytest.mark.parametrize("name", ["log10_counts", "log10_data"])
def test_violins_span_the_kept_values(r, name):
    """stat_ydensity(trim = TRUE) runs on the log values with the zeros gone."""
    fig = _build(name)
    panel = next(iter(r["cases"][name]["panels"].values()))
    want = pd.DataFrame(panel["violin"])
    ax = fig.axes[0]
    got = {}
    for c in ax.collections:
        if isinstance(c, PolyCollection) and c.get_paths():
            v = c.get_paths()[0].vertices
            got[int(round(v[:, 0].mean()))] = (v[:, 1].min(), v[:, 1].max())
    for line in ax.lines:  # a group of one distinct value draws a flat line
        x, y = line.get_data()
        if len(x) == 2 and y[0] == y[1]:
            got.setdefault(int(round(np.mean(x))), (y[0], y[0]))
    for _, row in want.iterrows():
        lo, hi = got[int(row.x) - 1]
        assert lo == pytest.approx(row.ymin, abs=1e-9)
        assert hi == pytest.approx(row.ymax, abs=1e-9)
    plt.close(fig)


def test_boxes_are_computed_on_the_log_scale(r):
    fig = _build("log2_box_counts")
    for ax, panel in zip(fig.axes, r["cases"]["log2_box_counts"]["panels"].values(),
                         strict=True):
        want = pd.DataFrame(panel["box"])
        frame = _frame("counts.csv")
        feat = ax.get_title()
        for _, row in want.iterrows():
            v = frame.loc[frame.CellType == LEVELS[int(row.x) - 1], feat].to_numpy(float)
            v = np.log2(v[v > 0])
            q = np.quantile(v, [0.25, 0.5, 0.75])
            assert np.allclose(q, [row.lower, row.middle, row.upper], atol=1e-9), feat
    plt.close(fig)


# ---- the break algorithms, against scales 1.3 (R 4.3) ----------------------- #
# scales::extended_breaks()(x), scales::log_breaks()(x), log_breaks(base = 2)(x)
R_BREAKS = [
    ((0, 1), [0, 0.25, 0.5, 0.75, 1], (0.01, 1.01), [0.01, 0.1, 1, 10]),
    ((0, 362), [0, 100, 200, 300, 400], (0.01, 362.01), [0.01, 1, 100]),
    ((0.4345, 8.4615), [0, 2, 4, 6, 8], (0.4445, 8.4715), [0.3, 0.5, 1, 3, 5, 10]),
    ((-3.2, 17.9), [-5, 0, 5, 10, 15, 20], (3.21, 17.91), [3, 5, 7, 10, 20]),
    ((0.0012, 0.0371), [0, 0.01, 0.02, 0.03, 0.04], (0.0112, 0.0471),
     [0.01, 0.02, 0.03, 0.04, 0.05]),
    ((1, 6.4), [1, 2, 3, 4, 5, 6], (1.01, 6.41), [1, 2, 3, 5, 10]),
    ((-0.15, 3.2), [0, 1, 2, 3], (0.16, 3.21), [0.1, 0.3, 1, 3, 10]),
]


@pytest.mark.parametrize("ext_in,ext_out,log_in,log_out", R_BREAKS)
def test_breaks_match_scales(ext_in, ext_out, log_in, log_out):
    assert np.allclose(extended_breaks(*ext_in), ext_out, atol=1e-12)
    assert np.allclose(log_breaks(*log_in), log_out, rtol=1e-12)


def test_log2_breaks_and_labels_match_scales():
    assert log_breaks(0.4445, 8.4715, base=2) == [0.25, 0.5, 1, 2, 4, 8, 16]
    assert log_breaks(2.01, 1e5 + 0.01, base=2) == [2, 32, 512, 8192, 131072]
    assert format_breaks([0.5, 1, 3, 5]) == ["0.5", "1.0", "3.0", "5.0"]
    assert format_breaks([1, 10, 100, 1000]) == ["1", "10", "100", "1000"]
