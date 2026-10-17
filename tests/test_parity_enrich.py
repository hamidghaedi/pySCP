"""Numeric parity of the enrichment family against SCP's R output.

The fixtures under ``tests/fixtures/parity/audit/enrich`` are written by
``tools/parity/audit_enrich.R`` from SCP's ``EnrichmentPlot``, ``GSEAPlot`` and
``gseaScores`` (Hao Zhang, NJMU, GPL-3; https://github.com/zhanghao-njmu/SCP).
They carry the *input* tables R plotted as well as the frames R derived from
them, so this module needs neither R nor the parity ``.h5ad``.

Every assertion is held to the fixture's precision.  The audit marked each
disagreement ``xfail(strict=True)``; each marker was removed as its fix landed,
and the diagnoses and fixes are written out in ``docs/parity_audit.md``.
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
# enrichment_plot: the divergences the audit found, now fixed
# --------------------------------------------------------------------------

def test_panel_order_is_first_appearance(enr):
    r = _csv("enrichment_bar.csv")
    fig = scp.pl.enrichment_plot(enr, plot_type="bar", top_term=6)
    want = [_r_group(p) for p in pd.unique(r["panel"])]
    assert [a.get_title().split(" | ")[-1] for a in _panels(fig)] == want


def test_labels_are_capitalised_and_wrapped_like_stringr(enr):
    r = _csv("enrichment_bar.csv")
    fig = scp.pl.enrichment_plot(enr, plot_type="bar", top_term=6)
    for panel, rr in r.groupby("panel", sort=False):
        ax = _panel_for(fig, _r_group(panel))
        got = [t.get_text() for t in ax.get_yticklabels()]
        assert got == list(rr.sort_values("axis_pos")["Description"])


def test_python_wrap_matches_stringr_once_hyphens_are_kept():
    """textwrap with break_on_hyphens=False reproduces stringr::str_wrap.  That
    is *not* what SCP plots: SCP's own str_wrap (utils.R:872) masks stringr and
    is base strwrap, which keeps lines strictly shorter than the width; see
    the next test."""
    meta = json.loads((FIX / "meta.json").read_text())
    for raw, wrapped in meta["stringr_wrap_examples"].items():
        cap = raw[:1].upper() + raw[1:]
        assert textwrap.fill(cap, 50, break_on_hyphens=False) == wrapped


def test_plotted_labels_are_scp_strwrap_not_stringr():
    """Every label R drew (enrichment bar/dot/lollipop and GSEA bar) is
    capitalize() + base strwrap; on four descriptions that differs from
    stringr's wrapping, and R's plots side with strwrap on each one plotted."""
    meta = json.loads((FIX / "meta.json").read_text())["stringr_wrap_examples"]
    raw = {}
    for name in ("enrichment_input.csv", "gsea_input.csv"):
        t = _csv(name)
        raw.update(zip(t["ID"], t["Description"], strict=True))
    n = 0
    for name in ("enrichment_bar.csv", "enrichment_dot.csv", "enrichment_lollipop.csv",
                 "gsea_bar_both.csv", "gsea_bar_pos.csv"):
        r = _csv(name)
        for tid, d in zip(r["ID"], r["Description"], strict=True):
            assert scp.pl._enrich._wrap_desc([raw[tid]], 50) == [d]
            n += 1
    differ = [k for k, v in meta.items()
              if scp.pl._enrich._wrap_desc([k], 50)[0] != v]
    assert len(differ) == 4 and n > 50


def test_wrapped_labels_are_italic(enr):
    r = _csv("enrichment_bar.csv")
    fig = scp.pl.enrichment_plot(enr, plot_type="bar", top_term=6)
    for panel, rr in r.groupby("panel", sort=False):
        ax = _panel_for(fig, _r_group(panel))
        got = [t.get_fontstyle() == "italic" for t in ax.get_yticklabels()]
        assert got == list(_true(rr.sort_values("axis_pos")["italic"]))


def test_bar_fill_is_color_by(enr):
    r = _csv("enrichment_bar.csv")
    fig = scp.pl.enrichment_plot(enr, plot_type="bar", top_term=6)
    for panel, rr in r.groupby("panel", sort=False):
        ax = _panel_for(fig, _r_group(panel))
        for p, f in zip(ax.patches, rr.sort_values("axis_pos")["fill"], strict=True):
            assert to_hex(p.get_facecolor()) == f.lower()


def test_bar_count_labels_and_axis_limit(enr):
    r = _csv("enrichment_bar.csv")
    fig = scp.pl.enrichment_plot(enr, plot_type="bar", top_term=6)
    for panel, rr in r.groupby("panel", sort=False):
        ax = _panel_for(fig, _r_group(panel))
        assert ax.get_xlim()[1] == pytest.approx(rr["ylim_hi"].iloc[0], rel=1e-9)
        counts = sorted(int(t.get_text()) for t in ax.texts if t.get_text().isdigit())
        assert counts == sorted(rr["Count"].tolist() * 2)  # white underlay + black text


def test_dot_x_is_gene_ratio(edge, edge_expected):
    want = edge_expected.query("plot_type == 'dot' and cutoff == 'padj'")
    fig = scp.pl.enrichment_plot(edge, plot_type="dot", top_term=2)
    for panel, rr in want.groupby("panel", sort=False):
        ax = _panel_for(fig, _r_group(panel))
        rr = rr.sort_values("axis_pos")
        assert _tick_ids(ax, edge) == list(rr["ID"])
        np.testing.assert_allclose(ax.collections[-1].get_offsets()[:, 0], rr["value"])


def test_lollipop_x_is_fold_enrichment(enr):
    r = _csv("enrichment_lollipop.csv")
    fig = scp.pl.enrichment_plot(enr, plot_type="lollipop", top_term=6)
    for panel, rr in r.groupby("panel", sort=False):
        ax = _panel_for(fig, _r_group(panel))
        rr = rr.sort_values("axis_pos")
        assert _tick_ids(ax, enr) == list(rr["ID"])
        np.testing.assert_allclose(ax.collections[-1].get_offsets()[:, 0], rr["value"])


def test_dot_size_mapping(edge, edge_expected):
    want = edge_expected.query("plot_type == 'dot' and cutoff == 'padj'")
    fig = scp.pl.enrichment_plot(edge, plot_type="dot", top_term=2)
    for panel, rr in want.groupby("panel", sort=False):
        ax = _panel_for(fig, _r_group(panel))
        sizes = dict(zip(rr["ID"], rr["size"], strict=True))
        got = ax.collections[-1].get_sizes()
        ref = np.array([sizes[i] for i in _tick_ids(ax, edge)]) ** 2  # area ~ diameter^2
        np.testing.assert_allclose(got / got.max(), ref / ref.max(), rtol=1e-6)


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


def _gsea_bar_ids(ax, table: pd.DataFrame) -> list[str]:
    """Term IDs of a GSEA bar panel, bottom to top.  R blanks the y axis and
    writes each description inside the panel at x = 0, so read those."""
    lookup = {_key(d): i for d, i in zip(table["Description"], table["ID"], strict=True)}
    texts = sorted((t for t in ax.texts if _key(t.get_text()) in lookup),
                   key=lambda t: t.get_position()[1])
    return [lookup[_key(t.get_text())] for t in texts]


def _documented_input(gt: pd.DataFrame, gsea_lists, keys=None):
    """``{"enrichment": df, "results": {key: {"result", "geneList", "geneSets",
    "exponent"}}}``: the gseaResult slots, as the docstring describes."""
    lists, sets = gsea_lists
    results = {}
    for key in keys or lists:
        group = key.rsplit("-", 1)[0]
        results[key] = {"result": gt[gt["Groups"] == group], "geneList": lists[key],
                        "geneSets": sets, "exponent": 1.0}
    return {"enrichment": gt, "results": results}


def test_gsea_line_draws_gseaScores_and_hits(gsea_lists):
    """Every selected set is overlaid in its key's block: the plotted curve and
    the hit ticks equal R's, and the sets are R's selection in R's order."""
    sel = _csv("gsea_selection.csv").query("direction == 'pos'")
    cur = _csv("gsea_curves.csv")
    gt = _csv("gsea_input.csv")
    checked = 0
    for key, ss in sel.groupby("key", sort=False):
        group = key.rsplit("-", 1)[0]
        res, tab = _gsea_input(gt, group, gsea_lists)
        fig = scp.pl.gsea_plot(res, plot_type="line", top_term=6)
        assert len(fig.axes) == 3  # one key -> one 3-row block
        top, mid, _ = fig.axes
        ids = [ln.get_gid() for ln in top.lines if ln.get_gid()]
        assert ids == list(ss.sort_values("rank")["ID"])
        bands = {c.get_gid(): c for c in mid.collections if c.get_gid()}
        # the first set's band is on top
        assert [bands[i].get_segments()[0][0][1] for i in ids] == \
            list(range(len(ids) - 1, -1, -1))
        for ln in (ln for ln in top.lines if ln.get_gid()):
            tid = ln.get_gid()
            rr = cur[(cur["key"] == key) & (cur["ID"] == tid)]
            if rr.empty:  # curves were only dumped for direction="both" picks
                continue
            x, y = ln.get_data()
            got = pd.Series(y, index=x).loc[rr["x"].to_numpy()]
            np.testing.assert_allclose(got, rr["runningScore"], atol=1e-12)
            ticks = sorted(int(round(s[0][0])) for s in bands[tid].get_segments())
            assert ticks == sorted(rr.loc[rr["position"] == 1, "x"])
            checked += 1
    assert checked >= 10


def test_gsea_line_is_one_block_per_key_with_a_term_legend(gsea_lists):
    """R loops names(res): one block per "<Group>-<Database>", every term
    overlaid, the key down the right, and a "Term" legend per block."""
    gt = _csv("gsea_input.csv")
    sel = _csv("gsea_selection.csv").query("direction == 'both'")
    fig = scp.pl.gsea_plot(_documented_input(gt, gsea_lists), plot_type="line",
                           direction="both", top_term=6)
    keys = list(pd.unique(sel["key"]))
    assert len(fig.axes) == 3 * len(keys)
    for i, key in enumerate(keys):
        top = fig.axes[3 * i]
        assert [ln.get_gid() for ln in top.lines if ln.get_gid()] == \
            list(sel[sel["key"] == key].sort_values("rank")["ID"])
        assert key in [t.get_text() for t in top.texts]
    titles = [lg.get_title().get_text() for lg in fig.legends]
    assert titles == [f"Term ({k}):" for k in keys]


def test_gsea_line_computes_curves_from_the_documented_input(gsea_lists):
    """Given the geneList / geneSets / exponent per key, the running scores are
    computed and equal R's gsInfo to 1e-12."""
    cur = _csv("gsea_curves.csv")
    gt = _csv("gsea_input.csv")
    fig = scp.pl.gsea_plot(_documented_input(gt, gsea_lists, ["Ductal-custom"]),
                           plot_type="line", direction="both", top_term=6)
    lines = [ln for ln in fig.axes[0].lines if ln.get_gid()]
    assert len(lines) == 6
    for ln in lines:
        rr = cur[(cur["key"] == "Ductal-custom") & (cur["ID"] == ln.get_gid())]
        x, y = ln.get_data()
        np.testing.assert_allclose(pd.Series(y, index=x).loc[rr["x"].to_numpy()],
                                   rr["runningScore"], atol=1e-12)


def test_gsea_line_accepts_the_bare_per_key_mapping(gsea_lists):
    gt = _csv("gsea_input.csv")
    doc = _documented_input(gt, gsea_lists, ["Ductal-custom"])["results"]
    fig = scp.pl.gsea_plot(doc, plot_type="line", top_term=2)
    assert len(fig.axes) == 3
    assert len([ln for ln in fig.axes[0].lines if ln.get_gid()]) == 2


def test_gsea_pos_selection_agrees_on_real_data(gsea_lists):
    """On this data |NES| and p.adjust rank the positive sets identically."""
    sel = _csv("gsea_selection.csv").query("direction == 'pos'")
    gt = _csv("gsea_input.csv")
    for key, ss in sel.groupby("key", sort=False):
        res, tab = _gsea_input(gt, key.rsplit("-", 1)[0], gsea_lists)
        fig = scp.pl.gsea_plot(res, plot_type="bar", direction="pos", top_term=6)
        assert sorted(_gsea_bar_ids(fig.axes[0], tab)) == sorted(ss["ID"])


def test_gsea_selection_ranks_by_padjust(gsea_lists):
    exp = _csv("gsea_edge_expected.csv").query("direction == 'pos'")
    tab = _csv("gsea_edge_input.csv")
    res, _ = _gsea_input(tab, "Ductal", gsea_lists)
    fig = scp.pl.gsea_plot(res, plot_type="bar", direction="pos", top_term=3)
    assert sorted(_gsea_bar_ids(fig.axes[0], tab)) == sorted(exp["ID"])


def test_gsea_selection_edge_case_both_directions(gsea_lists):
    """The discriminating table again, direction="both": R's bar order."""
    exp = _csv("gsea_edge_expected.csv").query("direction == 'both'")
    tab = _csv("gsea_edge_input.csv")
    res, _ = _gsea_input(tab, "Ductal", gsea_lists)
    fig = scp.pl.gsea_plot(res, plot_type="bar", direction="both", top_term=3)
    assert _gsea_bar_ids(fig.axes[0], tab) == list(exp.sort_values("axis_pos")["ID"])


def test_gsea_both_splits_up_and_down(gsea_lists):
    sel = _csv("gsea_selection.csv").query("direction == 'both'")
    gt = _csv("gsea_input.csv")
    for key, ss in sel.groupby("key", sort=False):
        res, tab = _gsea_input(gt, key.rsplit("-", 1)[0], gsea_lists)
        fig = scp.pl.gsea_plot(res, plot_type="bar", direction="both", top_term=6)
        assert sorted(_gsea_bar_ids(fig.axes[0], tab)) == sorted(ss["ID"])


def test_gsea_bar_order_is_nes_ascending(gsea_lists):
    exp = _csv("gsea_bar_pos.csv")
    gt = _csv("gsea_input.csv")
    for key, rr in exp.groupby("key", sort=False):
        res, tab = _gsea_input(gt, key.rsplit("-", 1)[0], gsea_lists)
        fig = scp.pl.gsea_plot(res, plot_type="bar", direction="pos", top_term=6)
        assert _gsea_bar_ids(fig.axes[0], tab) == list(rr.sort_values("axis_pos")["ID"])


def test_gsea_bar_fill_and_alpha(gsea_lists):
    exp = _csv("gsea_bar_both.csv")
    gt = _csv("gsea_input.csv")
    for key, rr in exp.groupby("key", sort=False):
        res, tab = _gsea_input(gt, key.rsplit("-", 1)[0], gsea_lists)
        fig = scp.pl.gsea_plot(res, plot_type="bar", direction="both", top_term=6)
        ax = fig.axes[0]
        want = rr.set_index("ID")
        for tid, p in zip(_gsea_bar_ids(ax, tab), ax.patches, strict=True):
            assert to_hex(p.get_facecolor(), keep_alpha=False) == want.loc[tid, "fill"].lower()
            assert p.get_alpha() == pytest.approx(want.loc[tid, "alpha"], abs=1e-6)


def test_gsea_bar_panels_per_key_on_one_grid(gsea_lists):
    """Every key gets a panel of one figure, laid out by PanelGrid (constrained
    layout), with the description text inside the panel and no y tick labels."""
    exp = _csv("gsea_bar_both.csv")
    gt = _csv("gsea_input.csv")
    fig = scp.pl.gsea_plot(_documented_input(gt, gsea_lists), plot_type="bar",
                           direction="both", top_term=6)
    keys = list(pd.unique(exp["key"]))
    assert len(fig.axes) == len(keys)
    assert fig.get_layout_engine() is not None
    for ax, key in zip(fig.axes, keys, strict=True):
        rr = exp[exp["key"] == key]
        assert ax.get_title() == key.rsplit("-", 1)[0]
        assert not [t for t in ax.get_yticklabels() if t.get_text()]
        assert _gsea_bar_ids(ax, gt[gt["Groups"] == ax.get_title()]) == \
            list(rr.sort_values("axis_pos")["ID"])
        lim = rr["NES"].abs().max()
        assert ax.get_xlim() == pytest.approx((-1.1 * lim, 1.1 * lim))


def test_gsea_line_marks_the_zero_cross(gsea_lists):
    zc = _csv("gsea_zero_cross.csv").set_index("key")["zero_cross"]
    gt = _csv("gsea_input.csv")
    res, _ = _gsea_input(gt, "Ductal", gsea_lists)
    fig = scp.pl.gsea_plot(res, plot_type="line", top_term=1)
    bot = fig.axes[2]
    xs = [ln.get_xdata()[0] for ln in bot.lines if ln.get_linestyle() == "--"]
    assert xs and xs[0] == pytest.approx(zc["Ductal-custom"])
    assert f"Zero cross at {zc['Ductal-custom']:g}" in [t.get_text() for t in bot.texts]


def test_gsea_zero_cross_matches_R_on_every_key(gsea_lists):
    zc = _csv("gsea_zero_cross.csv").set_index("key")["zero_cross"]
    lists, _ = gsea_lists
    for key, v in zc.items():
        assert scp.pl._enrich.gsea_zero_cross(lists[key].to_numpy()) == v


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
    assert markers[0][0][1] == pytest.approx(peak["runningScore"].iloc[0], abs=1e-12)
    # the dashed crosshair: (0, peak) -> (x_peak, peak) and (x_peak, 0) -> (x_peak, peak)
    dashed = [ln.get_xydata() for ln in top.lines if ln.get_linestyle() == "--"]
    px, py = peak["x"].iloc[0], peak["runningScore"].iloc[0]
    assert any(np.allclose(d, [[0, py], [px, py]]) for d in dashed)
    assert any(np.allclose(d, [[px, 0], [px, py]]) for d in dashed)


def test_gsea_line_shades_the_half_planes(gsea_lists):
    gt = _csv("gsea_input.csv")
    res, _ = _gsea_input(gt, "Ductal", gsea_lists)
    fig = scp.pl.gsea_plot(res, plot_type="line", top_term=1)
    fills = {to_hex(p.get_facecolor(), keep_alpha=False): p.get_alpha()
             for p in fig.axes[0].patches}
    assert fills == {"#c40003": 0.2, "#1d008f": 0.2}


def test_gsea_heat_strip_matches_R(gsea_lists):
    """The single-set ranking-metric strip, from R's quantile/cut/colorRampPalette:
    the same rectangles, colours within one 8-bit step."""
    r = _csv("gsea_heat_strip.csv")
    lists, _ = gsea_lists
    for key, rr in r.groupby("key", sort=False):
        got = scp.pl._enrich.gsea_heat_strip(lists[key].to_numpy())
        assert [(a, b) for a, b, _ in got] == list(zip(rr["xmin"], rr["xmax"], strict=True))
        worst = max(int(np.abs(_rgb(c) - _rgb(w)).max())
                    for (_, _, c), w in zip(got, rr["col"], strict=True))
        assert worst <= 1


def test_gsea_line_draws_the_heat_strip(gsea_lists):
    r = _csv("gsea_heat_strip.csv").query("key == 'Ductal-custom'")
    gt = _csv("gsea_input.csv")
    res, _ = _gsea_input(gt, "Ductal", gsea_lists)
    fig = scp.pl.gsea_plot(res, plot_type="line", top_term=1)
    strip = [c for c in fig.axes[1].collections if not c.get_gid()]
    assert len(strip) == 1
    xs = [(p.vertices[:, 0].min(), p.vertices[:, 0].max()) for p in strip[0].get_paths()]
    assert xs == list(zip(r["xmin"], r["xmax"], strict=True))
