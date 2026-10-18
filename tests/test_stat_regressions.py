"""Regressions from the statistics family's end-to-end gallery run.

Each test pins one figure that used to come out wrong-but-plausible: a legend
missing levels, overlapping labels, labels outside their panel, tick labels
colliding across panels.  The parity-backed fixes (grouped pie / ring, log
``y_trans``) have their own files: ``test_cell_stat_polar.py`` and
``test_feature_stat_log.py``.
"""

import itertools

import numpy as np
import pandas as pd
import pytest
from matplotlib.text import Text

import scp
from scp.pl._stat import repel_labels


def _boxes(fig, ax):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # Text.get_window_extent: an Annotation's own would include its leader
    return [(t, Text.get_window_extent(t, r)) for t in ax.texts if t.get_text()]


def _corners(t, r):
    """A text's box as the rotated rectangle it is, not its axis-aligned hull."""
    box = Text.get_window_extent(t, r)
    w, h, _d = r.get_text_width_height_descent(t.get_text(), t.get_fontproperties(),
                                               ismath=False)
    a = np.radians(t.get_rotation())
    u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    c = np.array([(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2])
    return np.array([c + sx * w / 2 * u + sy * h / 2 * v for sx, sy in
                     [(-1, -1), (1, -1), (1, 1), (-1, 1)]])


def _clash(p, q) -> bool:
    """Separating-axis test for two convex quadrilaterals."""
    for poly in (p, q):
        for k in range(4):
            e = poly[(k + 1) % 4] - poly[k]
            n = np.array([-e[1], e[0]])
            a, b = p @ n, q @ n
            if a.max() <= b.min() or b.max() <= a.min():
                return False
    return True


def _rotated_clashes(fig, ax, keep=lambda s: True):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = [(t.get_text(), _corners(t, r)) for t in ax.texts if t.get_text() and
             keep(t.get_text())]
    return [(a, b) for (a, p), (b, q) in itertools.combinations(items, 2) if _clash(p, q)], items


def test_stat_plot_takes_a_string_stat_by(adata):
    df = adata.obs[["leiden", "batch"]]
    fig = scp.pl.stat_plot(df, stat_by="leiden", group_by="batch")
    ref = scp.pl.stat_plot(df, stat_by=["leiden"], group_by="batch")
    got = [(p.get_x(), p.get_height()) for p in fig.axes[0].patches]
    assert got == [(p.get_x(), p.get_height()) for p in ref.axes[0].patches] and got


def test_individual_legend_lists_every_level_once(adata):
    fig = scp.pl.feature_stat_plot(adata, "g0", group_by="leiden", individual=True)
    assert len(fig.legends) == 1
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert labels == [str(c) for c in adata.obs["leiden"].cat.categories]


def test_individual_legend_skips_levels_without_cells(adata):
    ad = adata[adata.obs["leiden"] != "B"].copy()
    fig = scp.pl.feature_stat_plot(ad, "g0", group_by="leiden", individual=True)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert labels == ["A", "C", "D"]


@pytest.mark.parametrize("plot_type", ["dot", "bar"])
def test_flip_turns_the_bottom_tick_labels(adata, plot_type):
    """axis.text.x = element_text(angle = 45, hjust = 1) styles the bottom axis,
    which coord_flip gives to the stat levels (dot) or the values (bar)."""
    fig = scp.pl.cell_stat_plot(adata, "leiden", group_by="batch", plot_type=plot_type,
                                flip=True)
    ax = fig.axes[0]
    fig.canvas.draw()
    xt = [t for t in ax.get_xticklabels() if t.get_text()]
    assert xt and all(t.get_rotation() == 45 and t.get_ha() == "right" for t in xt)
    assert all(t.get_rotation() == 0 for t in ax.get_yticklabels())
    if plot_type == "dot":
        assert [t.get_text() for t in xt] == ["A", "B", "C", "D"]


def test_chord_tick_labels_stay_clear_of_each_other(adata):
    fig = scp.pl.cell_stat_plot(adata, ["leiden", "batch"], plot_type="chord")
    clashes, items = _rotated_clashes(fig, fig.axes[0])
    assert not clashes
    assert {"A", "B", "C", "D", "b1", "b2"} <= {s for s, _ in items}


def test_chord_turns_crowded_names_radial():
    # small sectors beside one huge one: their names cannot all follow the circle
    rng = np.random.default_rng(1)
    big = ["a_rather_long_level_name"] * 900
    small = [f"tiny_level_{i}" for i in range(6) for _ in range(50)]
    df = pd.DataFrame({"x": pd.Categorical(big + small),
                       "y": pd.Categorical(rng.choice(["u", "v"], len(big) + len(small)))})
    df.index = [f"c{i}" for i in range(len(df))]
    fig = scp.pl.stat_plot(df, stat_by=["x", "y"], plot_type="chord")
    ax = fig.axes[0]
    tiny = [t for t in ax.texts if t.get_text().startswith("tiny")]
    assert len(tiny) == 6
    clashes, _ = _rotated_clashes(fig, ax, keep=lambda s: not s[0].isdigit())
    assert not clashes


@pytest.fixture
def de_table():
    rng = np.random.default_rng(3)
    rows = []
    for g in ["g1", "g2"]:
        for i in range(400):
            lfc = rng.normal(0, 1.5)
            rows.append(dict(group1=g, gene=f"gene{i}_{g}", avg_log2FC=lfc,
                             p_val_adj=10 ** -abs(rng.normal(0, 8) * (1 + abs(lfc))),
                             **{"pct.1": rng.uniform(0, 1), "pct.2": rng.uniform(0, 1)}))
    df = pd.DataFrame(rows)
    df["group1"] = pd.Categorical(df["group1"], categories=["g1", "g2"])
    return df


@pytest.mark.parametrize("x_metric", ["diff_pct", "avg_log2FC"])
def test_volcano_labels_stay_inside_their_panel_and_apart(de_table, x_metric):
    fig = scp.pl.volcano_plot(de_table, x_metric=x_metric, nlabel=8)
    r = None
    for ax in fig.axes[:2]:
        boxes = _boxes(fig, ax)
        r = fig.canvas.get_renderer()
        pane = ax.get_window_extent(r)
        assert len(boxes) == 16
        for _t, b in boxes:
            assert pane.x0 - 0.5 <= b.x0 and b.x1 <= pane.x1 + 0.5
            assert pane.y0 - 0.5 <= b.y0 and b.y1 <= pane.y1 + 0.5
        assert not [1 for (_, a), (_, b) in itertools.combinations(boxes, 2) if a.overlaps(b)]
    assert r is not None


def test_repel_labels_is_deterministic_and_separates_coincident_labels():
    x = np.zeros(5)
    y = np.zeros(5)
    texts = ["aaaa"] * 5
    kw = dict(xlim=(-1, 1), ylim=(-1, 1), size_in=(2.0, 2.0), fontsize=7)
    a = repel_labels(x, y, texts, **kw)
    b = repel_labels(x, y, texts, **kw)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    pos = np.column_stack(a) * 72  # 1 data unit = 72 pt here
    w = 4 * 7 * 0.6
    for i, j in itertools.combinations(range(5), 2):
        dx, dy = np.abs(pos[i] - pos[j])
        assert dx >= w - 1e-6 or dy >= 7 * 1.2 - 1e-6


def test_upset_small_counts_clear_the_panel_edge(adata):
    ad = adata.copy()
    ad.obs["s1"] = ad.obs["leiden"] == "A"
    ad.obs["s2"] = ad.obs["batch"] == "b1"
    ad.obs["s3"] = (ad.obs["leiden"] == "B") & (ad.obs_names.isin(ad.obs_names[:12]))
    with pytest.warns(UserWarning, match="stat_level"):
        fig = scp.pl.cell_stat_plot(ad, ["s1", "s2", "s3"], plot_type="upset")
    bar_ax = fig.axes[0]
    pane = bar_ax.get_window_extent(fig.canvas.get_renderer())
    boxes = _boxes(fig, bar_ax)
    assert boxes
    assert all(b.y0 >= pane.y0 - 0.5 and b.y1 <= pane.y1 + 0.5 for _t, b in boxes)


def test_flipped_stack_leaves_rs_gap_between_columns(adata):
    fig = scp.pl.feature_stat_plot(adata, ["g0", "g1", "g2"], group_by="leiden", stack=True,
                                   flip=True)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axes = sorted([a for a in fig.axes if a.get_visible()], key=lambda a: a.get_position().x0)
    gap_pt = (5.5 - 0.5 * 72.27 / 25.4) / 72.27 * 72
    for a, b in itertools.pairwise(axes):
        wa, wb = a.get_window_extent(r), b.get_window_extent(r)
        assert (wb.x0 - wa.x1) * 72 / fig.dpi == pytest.approx(gap_pt, abs=0.2)
        # the end labels of neighbouring columns no longer touch
        last = Text.get_window_extent(a.get_xticklabels()[-1], r)
        first = Text.get_window_extent(b.get_xticklabels()[0], r)
        assert first.x0 - last.x1 > 0
