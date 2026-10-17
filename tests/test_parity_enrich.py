"""Numeric parity of the enrichment family against SCP's R output.

The fixtures under ``tests/fixtures/parity/audit/enrich`` are written by
``tools/parity/audit_enrich.R`` from SCP's ``EnrichmentPlot``, ``GSEAPlot`` and
``gseaScores`` (Hao Zhang, NJMU, GPL-3; https://github.com/zhanghao-njmu/SCP).
They carry the *input* tables R plotted as well as the frames R derived from
them, so this module needs neither R nor the parity ``.h5ad``.

Where the port agrees the assertion is tight.  Where it does not, the test is
``xfail(strict=True)`` with a one-line diagnosis, so the suite stays green and
the fix flips it.  The diagnoses are written out in ``docs/parity_audit.md``.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import scp

matplotlib.use("Agg")

FIX = Path(__file__).parent / "fixtures" / "parity" / "audit" / "enrich"


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(FIX / name, keep_default_na=False, na_values=["NA"])


def _key(text) -> str:
    """Whitespace- and case-insensitive key, so labels match however they wrap."""
    return re.sub(r"\s+", "", str(text)).lower()


def _true(col: pd.Series) -> pd.Series:
    return col.astype(str).str.upper() == "TRUE"


def _rgb(h: str) -> np.ndarray:
    h = to_hex(h).lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)])


def _panels(fig):
    return [a for a in fig.axes if a.get_label() != "<colorbar>"]


def _panel_for(fig, group: str):
    for ax in _panels(fig):
        if ax.get_title().split(" | ")[-1] == group:
            return ax
    raise AssertionError(f"no panel for {group!r}")


def _tick_ids(ax, table: pd.DataFrame) -> list[str]:
    """Term IDs on the y axis, bottom to top."""
    lookup = {_key(d): i for d, i in zip(table["Description"], table["ID"], strict=True)}
    return [lookup[_key(t.get_text())] for t in ax.get_yticklabels()]


def _r_group(panel: str) -> str:
    return panel.split(".", 1)[1]  # R's split() names panels "<Database>.<Group>"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def enr() -> pd.DataFrame:
    return _csv("enrichment_input.csv")


@pytest.fixture(scope="module")
def edge() -> pd.DataFrame:
    return _csv("edge_input.csv")


@pytest.fixture(scope="module")
def edge_expected() -> pd.DataFrame:
    return _csv("edge_expected.csv")


# --------------------------------------------------------------------------
# enrichment_plot: what already agrees
# --------------------------------------------------------------------------

@pytest.mark.parametrize("plot_type", ["bar", "dot", "lollipop"])
def test_term_selection_per_group_matches(enr, plot_type):
    """padjust < 0.05, top 6 per (Database x Group): the same terms survive."""
    r = _csv(f"enrichment_{plot_type}.csv")
    fig = scp.pl.enrichment_plot(enr, plot_type=plot_type, top_term=6)
    for panel, rr in r.groupby("panel", sort=False):
        ax = _panel_for(fig, _r_group(panel))
        assert sorted(_tick_ids(ax, enr)) == sorted(rr["ID"])


def test_bar_order_and_metric_match(enr):
    """Bar length is -log10(p.adjust) and the most significant term is on top."""
    r = _csv("enrichment_bar.csv")
    fig = scp.pl.enrichment_plot(enr, plot_type="bar", top_term=6)
    for panel, rr in r.groupby("panel", sort=False):
        ax = _panel_for(fig, _r_group(panel))
        assert _tick_ids(ax, enr) == list(rr.sort_values("axis_pos")["ID"])
        widths = [p.get_width() for p in ax.patches]
        np.testing.assert_allclose(widths, rr.sort_values("axis_pos")["value"], rtol=1e-12)


@pytest.mark.parametrize("plot_type", ["dot", "lollipop"])
def test_point_fill_follows_the_metric_gradient(enr, plot_type):
    """fill = -log10(p.adjust) on Spectral over the panel's own range: within 2/255."""
    r = _csv(f"enrichment_{plot_type}.csv")
    fig = scp.pl.enrichment_plot(enr, plot_type=plot_type, top_term=6)
    worst = 0
    for panel, rr in r.groupby("panel", sort=False):
        ax = _panel_for(fig, _r_group(panel))
        fills = dict(zip(rr["ID"], rr["fill"], strict=True))
        coll = ax.collections[-1]
        for tid, fc in zip(_tick_ids(ax, enr), coll.get_facecolors(), strict=True):
            worst = max(worst, int(np.abs(_rgb(fc) - _rgb(fills[tid])).max()))
    # ggplot interpolates the 100-step ramp in CIELAB, matplotlib in sRGB
    assert worst <= 2


# --------------------------------------------------------------------------
# enrichment_plot: known divergences
# --------------------------------------------------------------------------

def test_panel_order_is_first_appearance(enr):
    r = _csv("enrichment_bar.csv")
    fig = scp.pl.enrichment_plot(enr, plot_type="bar", top_term=6)
    want = [_r_group(p) for p in pd.unique(r["panel"])]
    assert [a.get_title().split(" | ")[-1] for a in _panels(fig)] == want


@pytest.mark.xfail(strict=True, reason="labels are wrapped with textwrap.fill (breaks on "
                   "hyphens) and never capitalised; R does capitalize() then str_wrap()")
def test_labels_are_capitalised_and_wrapped_like_stringr(enr):
    r = _csv("enrichment_bar.csv")
    fig = scp.pl.enrichment_plot(enr, plot_type="bar", top_term=6)
    for panel, rr in r.groupby("panel", sort=False):
        ax = _panel_for(fig, _r_group(panel))
        got = [t.get_text() for t in ax.get_yticklabels()]
        assert got == list(rr.sort_values("axis_pos")["Description"])


def test_python_wrap_matches_stringr_once_hyphens_are_kept():
    """The fix for the test above: textwrap with break_on_hyphens=False is str_wrap."""
    meta = json.loads((FIX / "meta.json").read_text())
    for raw, wrapped in meta["stringr_wrap_examples"].items():
        cap = raw[:1].upper() + raw[1:]
        assert textwrap.fill(cap, 50, break_on_hyphens=False) == wrapped


@pytest.mark.xfail(strict=True, reason="wrapped labels are not italicised; R sets "
                   "face='italic' on every level containing a newline")
def test_wrapped_labels_are_italic(enr):
    r = _csv("enrichment_bar.csv")
    fig = scp.pl.enrichment_plot(enr, plot_type="bar", top_term=6)
    for panel, rr in r.groupby("panel", sort=False):
        ax = _panel_for(fig, _r_group(panel))
        got = [t.get_fontstyle() == "italic" for t in ax.get_yticklabels()]
        assert got == list(_true(rr.sort_values("axis_pos")["italic"]))


@pytest.mark.xfail(strict=True, reason="bar fill is the -log10(p) gradient; R fills by "
                   "color_by (Database) from palette_scp(levels, palette)")
def test_bar_fill_is_color_by(enr):
    r = _csv("enrichment_bar.csv")
    fig = scp.pl.enrichment_plot(enr, plot_type="bar", top_term=6)
    for panel, rr in r.groupby("panel", sort=False):
        ax = _panel_for(fig, _r_group(panel))
        for p, f in zip(ax.patches, rr.sort_values("axis_pos")["fill"], strict=True):
            assert to_hex(p.get_facecolor()) == f.lower()


@pytest.mark.xfail(strict=True, reason="no geom_text Count labels at the bar ends and no "
                   "value-axis limit of 1.3 x max (R: scale_y_continuous(limits=c(0, 1.3*max)))")
def test_bar_count_labels_and_axis_limit(enr):
    r = _csv("enrichment_bar.csv")
    fig = scp.pl.enrichment_plot(enr, plot_type="bar", top_term=6)
    for panel, rr in r.groupby("panel", sort=False):
        ax = _panel_for(fig, _r_group(panel))
        assert ax.get_xlim()[1] == pytest.approx(rr["ylim_hi"].iloc[0], rel=1e-9)
        counts = sorted(int(t.get_text()) for t in ax.texts if t.get_text().isdigit())
        assert counts == sorted(rr["Count"].tolist() * 2)  # white underlay + black text


@pytest.mark.xfail(strict=True, reason="dot x is -log10(p.adjust) and rows are ordered by "
                   "it; R plots GeneRatio and orders rows by GeneRatio")
def test_dot_x_is_gene_ratio(edge, edge_expected):
    want = edge_expected.query("plot_type == 'dot' and cutoff == 'padj'")
    fig = scp.pl.enrichment_plot(edge, plot_type="dot", top_term=2)
    for panel, rr in want.groupby("panel", sort=False):
        ax = _panel_for(fig, _r_group(panel))
        rr = rr.sort_values("axis_pos")
        assert _tick_ids(ax, edge) == list(rr["ID"])
        np.testing.assert_allclose(ax.collections[-1].get_offsets()[:, 0], rr["value"])


@pytest.mark.xfail(strict=True, reason="lollipop x is -log10(p.adjust); R plots "
                   "FoldEnrichment = GeneRatio/BgRatio and orders rows by it")
def test_lollipop_x_is_fold_enrichment(enr):
    r = _csv("enrichment_lollipop.csv")
    fig = scp.pl.enrichment_plot(enr, plot_type="lollipop", top_term=6)
    for panel, rr in r.groupby("panel", sort=False):
        ax = _panel_for(fig, _r_group(panel))
        rr = rr.sort_values("axis_pos")
        assert _tick_ids(ax, enr) == list(rr["ID"])
        np.testing.assert_allclose(ax.collections[-1].get_offsets()[:, 0], rr["value"])


@pytest.mark.xfail(strict=True, reason="point area is 20+80*Count/max (linear, zero-anchored); "
                   "R's scale_size(range=c(3,6)) maps sqrt-rescaled Count onto 3..6 mm")
def test_dot_size_mapping(edge, edge_expected):
    want = edge_expected.query("plot_type == 'dot' and cutoff == 'padj'")
    fig = scp.pl.enrichment_plot(edge, plot_type="dot", top_term=2)
    for panel, rr in want.groupby("panel", sort=False):
        ax = _panel_for(fig, _r_group(panel))
        sizes = dict(zip(rr["ID"], rr["size"], strict=True))
        got = ax.collections[-1].get_sizes()
        ref = np.array([sizes[i] for i in _tick_ids(ax, edge)]) ** 2  # area ~ diameter^2
        np.testing.assert_allclose(got / got.max(), ref / ref.max(), rtol=1e-6)


@pytest.mark.xfail(strict=True, reason="ties in p.adjust land in reverse: sort_values "
                   "ascending puts the first-listed term at the bottom; R's rev(order()) puts it on top")
def test_ties_keep_table_order_top_down(edge, edge_expected):
    want = edge_expected.query("plot_type == 'bar' and cutoff == 'padj'")
    fig = scp.pl.enrichment_plot(edge, plot_type="bar", top_term=2)
    for panel, rr in want.groupby("panel", sort=False):
        ax = _panel_for(fig, _r_group(panel))
        assert _tick_ids(ax, edge) == list(rr.sort_values("axis_pos")["ID"])


def test_cutoff_is_strict(edge):
    fig = scp.pl.enrichment_plot(edge, plot_type="bar", top_term=None)
    ids = [i for ax in _panels(fig) for i in _tick_ids(ax, edge)]
    assert "T3" not in ids


def test_pvalue_path_switches_the_metric(edge, edge_expected):
    want = edge_expected.query("plot_type == 'bar' and cutoff == 'pval'")
    fig = scp.pl.enrichment_plot(edge, plot_type="bar", padjust_cutoff=None,
                                 pvalue_cutoff=0.003, top_term=2)
    for panel, rr in want.groupby("panel", sort=False):
        ax = _panel_for(fig, _r_group(panel))
        rr = rr.sort_values("axis_pos")
        assert sorted(_tick_ids(ax, edge)) == sorted(rr["ID"])
        np.testing.assert_allclose(sorted(p.get_width() for p in ax.patches),
                                   sorted(rr["value"]), rtol=1e-12)


@pytest.mark.xfail(strict=True, reason="a panel whose metric is constant gets the ramp's "
                   "first colour (Normalize(v, v) -> 0); ggplot maps a zero-width range to the midpoint")
def test_constant_metric_panel_gets_mid_colour(edge, edge_expected):
    want = edge_expected.query("plot_type == 'dot' and cutoff == 'padj' and panel == 'DB.G2'")
    fig = scp.pl.enrichment_plot(edge, plot_type="dot", top_term=2)
    ax = _panel_for(fig, "G2")
    for fc in ax.collections[-1].get_facecolors():
        assert np.abs(_rgb(fc) - _rgb(want["fill"].iloc[0])).max() <= 2


def test_comparison_is_one_group_by_term_matrix(enr):
    r = _csv("enrichment_comparison.csv")
    lev = json.loads((FIX / "enrichment_comparison_levels.json").read_text())
    fig = scp.pl.enrichment_plot(enr, plot_type="comparison", top_term=6)
    axes = _panels(fig)
    assert len(axes) == 1
    ax = axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == lev["group_levels"]
    assert len(ax.collections) and sum(len(c.get_offsets()) for c in ax.collections) == len(r)


# --------------------------------------------------------------------------
# gsea_scores
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gsea_lists():
    gl = _csv("gsea_genelists.csv")
    t2g = _csv("gsea_term2gene.csv")
    lists = {k: pd.Series(g["metric"].to_numpy(), index=g["gene"].to_numpy())
             for k, g in gl.groupby("key", sort=False)}
    sets = t2g.groupby("Term")["symbol"].apply(list).to_dict()
    return lists, sets


def test_gsea_scores_match_gseaScores(gsea_lists):
    """Running score to 1e-12 at every hit, the peak and every 50th rank."""
    lists, sets = gsea_lists
    cur = _csv("gsea_curves.csv")
    n = 0
    for (key, tid), rr in cur.groupby(["key", "ID"], sort=False):
        py = scp.pl.gsea_scores(lists[key], sets[tid], exponent=1.0).set_index("x")
        sub = py.loc[rr["x"].to_numpy()]
        np.testing.assert_allclose(sub["running_score"], rr["runningScore"], atol=1e-12)
        np.testing.assert_array_equal(sub["position"], rr["position"])
        # every hit R recorded, and no other: the full hit set must agree too
        assert set(py.index[py["position"] == 1]) == set(rr.loc[rr["position"] == 1, "x"])
        # gsInfo's peak is which.max(abs(runningScore)), the first maximum
        assert int(np.argmax(np.abs(py["running_score"].to_numpy())) + 1) == \
            int(rr.loc[_true(rr["peak"]), "x"].iloc[0])
        n += 1
    assert n >= 20


# --------------------------------------------------------------------------
# gsea_plot
# --------------------------------------------------------------------------

def _gsea_input(table: pd.DataFrame, group: str, gsea_lists):
    lists, sets = gsea_lists
    tab = table[table["Groups"] == group] if "Groups" in table else table
    gl = lists[f"{group}-custom"]
    curves = {i: scp.pl.gsea_scores(gl, sets[i], 1.0) for i in tab["ID"]}
    return {"table": tab, "curves": curves}, tab


def test_gsea_line_draws_gseaScores_and_hits(gsea_lists):
    """Each selected set: the plotted curve and the hit ticks equal R's."""
    sel = _csv("gsea_selection.csv").query("direction == 'pos'")
    cur = _csv("gsea_curves.csv")
    gt = _csv("gsea_input.csv")
    checked = 0
    for key in pd.unique(sel["key"]):
        group = key.rsplit("-", 1)[0]
        res, tab = _gsea_input(gt, group, gsea_lists)
        fig = scp.pl.gsea_plot(res, plot_type="line", top_term=6)
        tops = fig.axes[0::3]
        mids = fig.axes[1::3]
        lookup = {_key(textwrap.fill(str(d), 46)): i
                  for d, i in zip(tab["Description"], tab["ID"], strict=True)}
        for top, mid in zip(tops, mids, strict=True):
            tid = lookup[_key(top.get_title())]
            rr = cur[(cur["key"] == key) & (cur["ID"] == tid)]
            if rr.empty:  # curves were only dumped for direction="both" picks
                continue
            x, y = top.lines[0].get_data()
            got = pd.Series(y, index=x).loc[rr["x"].to_numpy()]
            np.testing.assert_allclose(got, rr["runningScore"], atol=1e-12)
            ticks = sorted(int(round(s[0][0])) for s in mid.collections[0].get_segments())
            assert ticks == sorted(rr.loc[rr["position"] == 1, "x"])
            checked += 1
    assert checked >= 10


def test_gsea_pos_selection_agrees_on_real_data(gsea_lists):
    """On this data |NES| and p.adjust rank the positive sets identically."""
    sel = _csv("gsea_selection.csv").query("direction == 'pos'")
    gt = _csv("gsea_input.csv")
    for key, ss in sel.groupby("key", sort=False):
        res, tab = _gsea_input(gt, key.rsplit("-", 1)[0], gsea_lists)
        fig = scp.pl.gsea_plot(res, plot_type="bar", direction="pos", top_term=6)
        assert sorted(_tick_ids(fig.axes[0], tab)) == sorted(ss["ID"])


@pytest.mark.xfail(strict=True, reason="terms are ranked by |NES| and filtered with <=; R "
                   "filters p.adjust < cutoff and ranks by p.adjust")
def test_gsea_selection_ranks_by_padjust(gsea_lists):
    exp = _csv("gsea_edge_expected.csv").query("direction == 'pos'")
    tab = _csv("gsea_edge_input.csv")
    res, _ = _gsea_input(tab, "Ductal", gsea_lists)
    fig = scp.pl.gsea_plot(res, plot_type="bar", direction="pos", top_term=3)
    assert sorted(_tick_ids(fig.axes[0], tab)) == sorted(exp["ID"])


@pytest.mark.xfail(strict=True, reason="direction='both' takes the top |NES| of either sign; "
                   "R takes ceiling(topTerm/2) up then ceiling(topTerm/2) down, by p.adjust")
def test_gsea_both_splits_up_and_down(gsea_lists):
    sel = _csv("gsea_selection.csv").query("direction == 'both'")
    gt = _csv("gsea_input.csv")
    for key, ss in sel.groupby("key", sort=False):
        res, tab = _gsea_input(gt, key.rsplit("-", 1)[0], gsea_lists)
        fig = scp.pl.gsea_plot(res, plot_type="bar", direction="both", top_term=6)
        assert sorted(_tick_ids(fig.axes[0], tab)) == sorted(ss["ID"])


@pytest.mark.xfail(strict=True, reason="bars stack in |NES|-descending order from the bottom; "
                   "R orders by NES ascending (most negative at the bottom)")
def test_gsea_bar_order_is_nes_ascending(gsea_lists):
    exp = _csv("gsea_bar_pos.csv")
    gt = _csv("gsea_input.csv")
    for key, rr in exp.groupby("key", sort=False):
        res, tab = _gsea_input(gt, key.rsplit("-", 1)[0], gsea_lists)
        fig = scp.pl.gsea_plot(res, plot_type="bar", direction="pos", top_term=6)
        assert _tick_ids(fig.axes[0], tab) == list(rr.sort_values("axis_pos")["ID"])


@pytest.mark.xfail(strict=True, reason="Pos/Neg are hard-coded #B2182B/#2166AC with no alpha; "
                   "R uses palette_scp(c('Neg','Pos'), palette) and alpha = -log10(p.adjust)")
def test_gsea_bar_fill_and_alpha(gsea_lists):
    exp = _csv("gsea_bar_both.csv")
    gt = _csv("gsea_input.csv")
    for key, rr in exp.groupby("key", sort=False):
        res, tab = _gsea_input(gt, key.rsplit("-", 1)[0], gsea_lists)
        fig = scp.pl.gsea_plot(res, plot_type="bar", direction="both", top_term=6)
        ax = fig.axes[0]
        want = rr.set_index("ID")
        for tid, p in zip(_tick_ids(ax, tab), ax.patches, strict=True):
            assert to_hex(p.get_facecolor(), keep_alpha=False) == want.loc[tid, "fill"].lower()
            assert p.get_alpha() == pytest.approx(want.loc[tid, "alpha"], abs=1e-6)


@pytest.mark.xfail(strict=True, reason="the ranked-metric panel has no dashed zero-cross "
                   "line; R draws geom_vline at median(x[y == y[which.min(abs(y))]])")
def test_gsea_line_marks_the_zero_cross(gsea_lists):
    zc = _csv("gsea_zero_cross.csv").set_index("key")["zero_cross"]
    gt = _csv("gsea_input.csv")
    res, _ = _gsea_input(gt, "Ductal", gsea_lists)
    fig = scp.pl.gsea_plot(res, plot_type="line", top_term=1)
    bot = fig.axes[2]
    xs = [ln.get_xdata()[0] for ln in bot.lines if ln.get_linestyle() == "--"]
    assert xs and xs[0] == pytest.approx(zc["Ductal-custom"])


@pytest.mark.xfail(strict=True, reason="no peak marker: R annotates a single-set panel with "
                   "a triangle at which.max(abs(runningScore)) plus dashed crosshairs")
def test_gsea_line_marks_the_peak(gsea_lists):
    cur = _csv("gsea_curves.csv")
    sel = _csv("gsea_selection.csv").query("key == 'Ductal-custom' and direction == 'pos'")
    tid = sel.sort_values("rank")["ID"].iloc[0]
    peak = cur[(cur["key"] == "Ductal-custom") & (cur["ID"] == tid)
               & _true(cur["peak"])]
    gt = _csv("gsea_input.csv")
    res, _ = _gsea_input(gt, "Ductal", gsea_lists)
    fig = scp.pl.gsea_plot(res, plot_type="line", top_term=1)
    top = fig.axes[0]
    markers = [c.get_offsets() for c in top.collections if len(c.get_offsets()) == 1]
    assert markers and markers[0][0][0] == pytest.approx(peak["x"].iloc[0])
