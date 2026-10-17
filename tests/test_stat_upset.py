"""``cell_stat_plot(plot_type="upset")`` against SCP's ``StatPlot`` + ggupset.

``tools/parity/cell_stat_upset.R`` ran SCP 0.5.6 with ggupset 0.4.1 on
``pancreas_sub`` (six logical set columns: SCP's own example's four plus two
genes, so that some blocks hit ggupset's 20-intersection cap) and dumped, per
split level, the intersections in bar order, the counts, the bar fills, the
count labels and the combination-matrix rows.  ``upset_cells.csv`` is the
per-cell input, so neither R nor the h5ad is needed here.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb

import scp
from scp.pl._stat_flow import _membership, resolve_stat_level, upset_layout

FIX = Path(__file__).parent / "fixtures" / "parity" / "cell_stat"
SETS4 = ["Progenitor", "G2M", "Sox9_Expressed", "Neurog3_Expressed"]
SETS6 = SETS4 + ["Ins1_Expressed", "Gcg_Expressed"]
CASES = {
    "four": dict(stat_by=SETS4),
    "six_split": dict(stat_by=SETS6, split_by="CellType"),
    "six": dict(stat_by=SETS6),
}


@pytest.fixture(scope="module")
def r():
    return json.loads((FIX / "upset.json").read_text())


@pytest.fixture(scope="module")
def cells(r):
    df = pd.read_csv(FIX / "upset_cells.csv").set_index("cell")
    df["CellType"] = pd.Categorical(df["CellType"],
                                    categories=r["cases"]["six_split"]["names"])
    return df


def _panel(want, name):
    # jsonlite names an unsplit plot (R list name "") by its position
    return want["panels"][name or "1"]


def _blocks(cells, case):
    kw = CASES[case]
    sets = kw["stat_by"]
    members = _membership(cells, sets, resolve_stat_level(cells, sets, "TRUE"))
    if "split_by" not in kw:
        return {"": upset_layout(members, sets)}
    sp = cells[kw["split_by"]]
    return {str(s): upset_layout(members.loc[(sp == s).to_numpy()], sets)
            for s in sp.cat.categories}


@pytest.mark.parametrize("case", list(CASES))
def test_upset_layout_matches_ggupset(cells, r, case):
    """Bar order (count, then C-collated name), counts and matrix rows."""
    want = r["cases"][case]
    got = _blocks(cells, case)
    assert list(got) == list(want["names"])
    for name, (bars, sets) in got.items():
        p = _panel(want, name)
        assert [list(t) for t in bars["intersection"]] == p["intersections"], name
        assert bars["count"].tolist() == p["bar_count"], name
        assert sets == p["sets"], name
        assert list(reversed(sets)) == p["matrix_rows_bottom_up"], name
    if case == "six":
        assert len(got[""][0]) == 20  # n_intersections = 20 truncates this one


@pytest.mark.parametrize("case", list(CASES))
def test_upset_figure_draws_one_block_per_split(cells, r, case):
    want = r["cases"][case]
    fig = scp.pl.stat_plot(cells, plot_type="upset", stat_level="TRUE", **CASES[case])
    axes = fig.axes
    assert len(axes) == 2 * len(want["names"])  # bars over matrix, per split level
    for k, name in enumerate(want["names"]):
        bar_ax, mat_ax = axes[2 * k], axes[2 * k + 1]
        p = _panel(want, name)
        bars = sorted(bar_ax.patches, key=lambda b: b.get_x())
        assert [b.get_height() for b in bars] == p["bar_count"]
        assert [b.get_x() + b.get_width() / 2 + 1 for b in bars] == p["bar_x"]
        # R interpolates the 100-step material-indigo ramp in Lab, here in RGB
        for b, fill in zip(bars, p["bar_fill"], strict=True):
            delta = np.abs(np.array(to_rgb(b.get_facecolor())) - to_rgb(fill)).max() * 255
            assert delta <= 6, (name, fill)
        assert [t.get_text() for t in bar_ax.texts] == p["label"]
        assert [t.get_text() for t in mat_ax.get_yticklabels()] == p["matrix_rows_bottom_up"]
        assert mat_ax.get_xlabel() == p["x_label"]
        assert bar_ax.get_ylabel() == "Intersection size"
        # the matrix dots: black where the set is in the intersection
        rows = p["matrix_rows_bottom_up"]
        dots = [c for c in mat_ax.collections if len(c.get_offsets()) == len(rows)]
        assert len(dots) == len(bars)
        for col, inter in zip(dots, p["intersections"], strict=True):
            on = [tuple(fc[:3]) == (0.0, 0.0, 0.0) for fc in col.get_facecolors()]
            assert on == [s in inter for s in rows]
    plt.close(fig)


def test_upset_goes_through_the_panel_grid(adata):
    """No private figure: nrow/ncol place the split blocks like any other panel."""
    adata.obs["in_a"] = adata.obs["leiden"].astype(str) == "A"
    adata.obs["in_b1"] = adata.obs["batch"].astype(str) == "b1"
    fig = scp.pl.cell_stat_plot(adata, ["in_a", "in_b1"], plot_type="upset",
                                stat_level=True, split_by="batch", ncol=1)
    n_split = adata.obs["batch"].nunique()
    assert len(fig.axes) == 2 * n_split
    bars = fig.axes[0::2]
    assert all(ax.patches for ax in bars)
    xs =[ax.get_position().x0 for ax in bars]
    ys = [ax.get_position().y0 for ax in bars]
    assert np.allclose(xs, xs[0]) and ys == sorted(ys, reverse=True)
    plt.close(fig)
