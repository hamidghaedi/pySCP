"""Structure of the network / enrichmap / wordcloud / comparison figures and
golden values for the small pure helpers behind them.

Spec: docs/porting_briefs/trajectory_enrichment.md §8-§9.  Parity of the
graphs with SCP itself is in test_enrich_parity.py.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.patches import Polygon

import scp
from scp.pl import _enrich as en
from scp.pl import _enrich_draw as dr
from scp.pl import _enrich_graph as eg


def _beside(fig, artist, ax) -> bool:
    """``artist`` (a legend or colourbar axes) sits right of ``ax``'s panel
    and level with it, as a patchwork panel's own guide does."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    a, b = artist.get_window_extent(r), ax.get_window_extent(r)
    mid = (a.y0 + a.y1) / 2
    return a.x0 >= b.x1 and b.y0 <= mid <= b.y1


def toy_enrichment() -> pd.DataFrame:
    """Two groups; within each, two blocks of terms sharing genes."""
    rng = np.random.default_rng(3)
    rows = []
    for g in ("Beta", "Alpha"):  # deliberately not alphabetical
        for i in range(8):
            block = i % 2
            genes = [f"{'AB'[block]}{k}" for k in rng.choice(12, 6, replace=False)]
            rows.append(dict(
                ID=f"GO:{g}{i}", Description=f"term {i} of {'immune' if block else 'neural'} process",
                GeneRatio=f"{len(genes)}/50", BgRatio="20/2000", pvalue=10.0 ** -(9 - i),
                **{"p.adjust": 10.0 ** -(8 - i)}, geneID="/".join(genes), Count=len(genes),
                Groups=g, Database="GO_BP"))
    return pd.DataFrame(rows)


def toy_gsea() -> pd.DataFrame:
    df = toy_enrichment().rename(columns={"geneID": "core_enrichment", "Count": "setSize"})
    df["NES"] = np.where(np.arange(len(df)) % 3 == 0, -1.5, 2.0)
    df["enrichmentScore"] = df["NES"] / 3
    return df.drop(columns=["GeneRatio", "BgRatio"])


# --------------------------------------------------------------------------- #
# golden helpers
# --------------------------------------------------------------------------- #
def test_capitalize_follows_scp():
    assert eg.capitalize("regulation of x") == "Regulation of x"
    assert eg.capitalize("mRNA splicing") == "mRNA splicing"
    assert eg.capitalize("T cell activation") == "T cell activation"
    assert eg.capitalize("gamma-delta T cell") == "Gamma-delta T cell"
    assert eg.capitalize("") == ""


@pytest.mark.parametrize("width,expected", [
    # base R: paste0(strwrap(s, width), collapse = "\n")
    (50, "regulation of transcription by RNA polymerase II\nin response to stress xx"),
    (20, "regulation of\ntranscription by\nRNA polymerase II\nin response to\nstress xx"),
    (10, "regulation\nof\ntranscription\nby RNA\npolymerase\nII in\nresponse\nto stress\nxx"),
])
def test_r_strwrap_matches_base_r(width, expected):
    s = "regulation of transcription by RNA polymerase II in response to stress xx"
    assert eg.r_strwrap(s, width) == expected
    assert eg.r_strwrap("a  b", 5) == "a b"
    assert eg.r_strwrap("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa bb", 20) == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nbb"


def test_r_format_sci_and_round_match_r():
    # format(x, digits = 3, scientific = TRUE)
    assert en.r_format_sci([1e-5, 2e-3]) == ["1e-05", "2e-03"]
    assert en.r_format_sci([1.234e-5, 2e-3, 0.5]) == ["1.23e-05", "2.00e-03", "5.00e-01"]
    assert en.r_format_sci([1e-120, 2e-3]) == ["1e-120", " 2e-03"]
    # as.character(round(x, 3)), R >= 4 rounding
    got = [en._r_num(v) for v in (1.8049, 3.0365, 2.0625, 0.0005, 1.0005, -2.0, 0.0015)]
    assert got == ["1.805", "3.037", "2.062", "0", "1", "-2", "0.002"]


def test_holm_matches_p_adjust():
    np.testing.assert_allclose(eg._p_adjust_holm(np.array([0.01, 0.04, 0.03, 0.2])),
                               [0.04, 0.09, 0.09, 0.2])


def test_overlap_edges_are_shared_gene_counts():
    sets = [["a", "b", "c"], ["b", "c", "d"], ["x"], ["a", "a", "c"]]
    assert eg.overlap_edges(sets) == [(0, 1, 2), (0, 3, 2), (1, 3, 1)]


def test_network_blends_gene_colours_from_their_terms():
    df = pd.DataFrame({"Description": ["alpha", "beta"], "p.adjust": [1e-4, 1e-2],
                       "geneID": ["g1/g2", "g2/g3"]})
    nt = eg.network_tables(df)
    assert nt.nodes["name"].tolist() == ["Alpha", "Beta", "g1", "g2", "g3"]
    c = dict(zip(nt.nodes["name"], nt.nodes["color"], strict=True))
    assert c["g1"] == c["Alpha"] and c["g3"] == c["Beta"]
    assert c["g2"] == scp.colors.blendcolors([c["Alpha"], c["Beta"]])
    assert nt.nodes["label"].tolist()[:2] == [1, 2]
    assert len(nt.edges) == 4


def test_adjust_layout_fixes_rs_one_sided_push():
    """A neighbour to the right of its nearest box is separated too (R's sign
    bug only ever separated neighbours to the left / below)."""
    lay = np.array([[0.0, 0.0], [1.0, 0.0], [60.0, 60.0]])
    w = np.array([8.0, 8.0, 2.0])
    for seed in range(5):
        out = scp.pl.adjust_layout(lay, w, edges=[], scale=60.0, iterations=3, seed=seed)
        dx, dy = np.abs(out[0] - out[1])
        assert dx >= 8.0 - 1e-9 or dy >= 2.0 - 1e-9
    a = scp.pl.adjust_layout(lay, w, [(0, 1)], seed=4)
    b = scp.pl.adjust_layout(lay, w, [(0, 1)], seed=4)
    np.testing.assert_array_equal(a, b)


def test_adjust_layout_pushes_neighbours_off_hubs():
    lay = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [60.0, 60.0]])
    w = np.array([8.0, 4.0, 4.0, 2.0])
    out = scp.pl.adjust_layout(lay, w, [(0, 1), (0, 2)], scale=60.0, iterations=0)
    # the hub (vertex 0, degree 2) keeps its place; its neighbours are pushed to r + nr
    np.testing.assert_allclose(out[0], [0.0, 0.0])
    assert np.hypot(*(out[1] - out[0])) == pytest.approx(6.0)
    assert np.hypot(*(out[2] - out[0])) == pytest.approx(6.0)


# --------------------------------------------------------------------------- #
# panels, level order, selection
# --------------------------------------------------------------------------- #
def test_panels_follow_first_appearance_order_and_top_term():
    _, panels, metric, value, top = en._prepare_enrichment(
        toy_enrichment(), plot_type="network", split_by=("Groups",), group_use=None,
        id_use=None, pvalue_cutoff=None, padjust_cutoff=0.05, top_term=None)
    # network forces Database x Groups; groups keep first-appearance order
    assert [k for k, _, _ in panels] == ["GO_BP | Beta", "GO_BP | Alpha"]
    assert top == 6 and all(len(df) == 6 for _, _, df in panels)
    # strict '<' as in R, sorted by the metric
    df = panels[0][2]
    assert (df[metric] < value).all() and df[metric].is_monotonic_increasing


def test_id_use_lifts_top_term():
    ids = [f"GO:Beta{i}" for i in range(7)]
    _, panels, *_ = en._prepare_enrichment(
        toy_enrichment(), plot_type="network", split_by=("Database", "Groups"),
        group_use=None, id_use=ids, pvalue_cutoff=None, padjust_cutoff=0.05, top_term=2)
    assert len(panels) == 1 and len(panels[0][2]) == 7


def test_gsea_select_ids_both_takes_half_of_each_direction():
    res = toy_gsea()
    res = res[res["Groups"] == "Beta"]
    ids = en.gsea_select_ids(res, metric="p.adjust", metric_value=0.05, top_term=4,
                             direction="both")
    nes = res.set_index("ID").loc[ids, "NES"]
    assert len(ids) == 4 and (nes > 0).sum() == 2 and (nes < 0).sum() == 2
    neg = en.gsea_select_ids(res, metric="p.adjust", metric_value=0.05, top_term=4,
                             direction="neg")
    assert neg[:2] == [i for i in ids if nes[i] < 0]
    assert (res.set_index("ID").loc[neg, "NES"] < 0).all()
    # the wordcloud keeps every significant set of the direction
    every = en.gsea_select_ids(res, metric="p.adjust", metric_value=0.05, top_term=None,
                               direction="both")
    assert len(every) == int((res["p.adjust"] < 0.05).sum())


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def test_network_figure_structure():
    pytest.importorskip("igraph")
    df = toy_enrichment()
    # distinct term names per group, so the two "Term" legends differ
    df["Description"] = df["Description"] + " in " + df["Groups"]
    fig = scp.pl.enrichment_plot(df, plot_type="network", top_term=3)
    assert len(fig.axes) == 2
    ax = fig.axes[0]
    nt = eg.network_tables(en._prepare_enrichment(
        df, plot_type="network", split_by=("Database", "Groups"), group_use=None, id_use=None,
        pvalue_cutoff=None, padjust_cutoff=0.05, top_term=3)[1][0][2])
    segs = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(segs) == 1 and len(segs[0].get_segments()) == len(nt.edges)
    texts = [t.get_text() for t in ax.texts]
    assert {"1", "2", "3"} <= set(texts)
    assert set(nt.nodes.loc[nt.nodes["class"] == "gene", "name"]) <= set(texts)
    # one "Term" legend per panel, in term order, registered once on the
    # figure and drawn beside its own panel (wrap_plots without collect)
    titles = [lg.get_title().get_text() for lg in fig.legends]
    assert titles == ["Term:", "Term:"]
    assert [t.get_text() for t in fig.legends[0].get_texts()] == nt.term_levels
    assert _beside(fig, fig.legends[0], fig.axes[0])
    assert _beside(fig, fig.legends[1], fig.axes[1])
    assert not any(a.get_legend() for a in fig.axes)
    matplotlib.pyplot.close(fig)


def test_enrichmap_figure_structure():
    pytest.importorskip("igraph")
    fig = scp.pl.enrichment_plot(toy_enrichment(), plot_type="enrichmap", group_use=["Alpha"])
    ax = fig.axes[0]
    _, panels, *_ = en._prepare_enrichment(
        toy_enrichment(), plot_type="enrichmap", split_by=("Database", "Groups"),
        group_use=["Alpha"], id_use=None, pvalue_cutoff=None, padjust_cutoff=0.05, top_term=None)
    et = eg.enrichmap_tables(panels[0][2])
    k = len(et.cluster_levels)
    assert k >= 2  # the two gene blocks separate
    assert sum(isinstance(p, Polygon) for p in ax.patches) == 2 * k  # fill + outline
    assert sum(isinstance(a, AnnotationBbox) for a in ax.artists) == k
    titles = [lg.get_title().get_text() for lg in fig.legends]
    assert titles == ["Count:", "Intersection:", "Feature:"]
    assert [t.get_text() for t in fig.legends[2].get_texts()] == list(et.legend_labels.values())
    # enrichmap_expand = 0.5: the data span is half of each axis
    xy = ax.collections[-1].get_offsets()
    span = np.ptp(np.asarray(xy)[:, 0])
    assert np.diff(ax.get_xlim())[0] == pytest.approx(2 * span)
    matplotlib.pyplot.close(fig)


def test_enrichmap_hull_mark_and_feature_label():
    pytest.importorskip("igraph")
    fig = scp.pl.enrichment_plot(toy_enrichment(), plot_type="enrichmap", group_use=["Beta"],
                                 enrichmap_mark="hull", enrichmap_label="feature",
                                 enrichmap_cluster="walktrap")
    assert fig.legends[2].get_title().get_text() == "Term:"
    matplotlib.pyplot.close(fig)


def test_wordcloud_places_words_without_overlap():
    fig = scp.pl.enrichment_plot(toy_enrichment(), plot_type="wordcloud", group_use=["Beta"],
                                 word_type="feature")
    ax = fig.axes[0]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(r) for t in ax.texts]
    assert len(boxes) >= 5
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            inter = (min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0))
            assert min(inter) <= 1.0  # at most a pixel of anti-aliasing
    words = {t.get_text() for t in ax.texts}
    assert words <= {g for s in toy_enrichment()["geneID"] for g in s.split("/")}
    matplotlib.pyplot.close(fig)


def test_wordcloud_go_terms_use_frequency_substitute_or_background():
    df = toy_enrichment()
    sub = df[df["Groups"] == "Beta"]
    t = eg.wordcloud_table(sub)
    # SCP drops "process" (words_excluded); the frequency substitute does too
    assert "process" not in set(t["keyword"]) and "immune" in set(t["keyword"])
    bg = {f"GO:bg{i}": f"neural crest {i} migration" for i in range(40)}
    bg.update(dict(zip(sub["ID"], sub["Description"], strict=True)))
    t2 = eg.wordcloud_table(sub, background=bg)
    # with a background the GO path is the keyword-enrichment test
    assert set(t2["keyword"]) <= {"term", "neural", "immune"}
    assert (t2["count"] >= 2).all()


def test_comparison_one_panel_per_database():
    df = toy_enrichment()
    df2 = df.assign(Database="KEGG", ID=df["ID"] + "k")
    fig = scp.pl.enrichment_plot(pd.concat([df, df2]), plot_type="comparison", top_term=2)
    panels = [a for a in fig.axes if a.get_title()]
    assert [a.get_title() for a in panels] == ["GO_BP", "KEGG"]
    assert [t.get_text() for t in panels[0].get_xticklabels()] == ["Beta", "Alpha"]
    assert any(lg.get_title().get_text() == "Non-sig:" for lg in fig.legends)
    matplotlib.pyplot.close(fig)


@pytest.mark.parametrize("plot_type", ["comparison", "network", "enrichmap", "wordcloud"])
def test_gsea_plot_new_types_render(plot_type):
    if plot_type in ("network", "enrichmap"):
        pytest.importorskip("igraph")
    fig = scp.pl.gsea_plot(toy_gsea(), plot_type=plot_type, direction="both")
    n_keys = 1 if plot_type == "comparison" else 2
    assert len([a for a in fig.axes if a.get_title()]) == n_keys
    matplotlib.pyplot.close(fig)


def test_gsea_network_term_names_carry_the_stats_line():
    pytest.importorskip("igraph")
    fig = scp.pl.gsea_plot(toy_gsea(), plot_type="network", direction="pos", group_use=["Beta"])
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert labels and all("\n(NES=2, p.adjust=" in s for s in labels)
    matplotlib.pyplot.close(fig)


def test_unknown_layout_and_cluster_raise():
    pytest.importorskip("igraph")
    with pytest.raises(ValueError, match="layout"):
        scp.pl.enrichment_plot(toy_enrichment(), plot_type="network", network_layout="gem")
    with pytest.raises(ValueError, match="enrichmap_cluster"):
        scp.pl.enrichment_plot(toy_enrichment(), plot_type="enrichmap",
                               enrichmap_cluster="fluid_communities")


def test_bar_fill_follows_color_by_levels_of_the_whole_table():
    """palette_scp(levels(df[[color_by]])): first-appearance levels over the
    input, before group_use filters it, so a group keeps its colour."""
    df = toy_enrichment()
    pal = scp.discrete_palette(["Beta", "Alpha"], "Spectral")
    fig = scp.pl.enrichment_plot(df, plot_type="bar", color_by="Groups", group_use=["Alpha"],
                                 top_term=3)
    (ax,) = [a for a in fig.axes if a.get_title()]
    assert {matplotlib.colors.to_hex(p.get_facecolor()) for p in ax.patches} == \
        {pal["Alpha"].lower()}
    matplotlib.pyplot.close(fig)
    with pytest.raises(ValueError, match="color_by"):
        scp.pl.enrichment_plot(df, plot_type="bar", color_by="nope")


def test_dot_and_lollipop_need_their_ratio_columns():
    df = toy_enrichment().drop(columns=["BgRatio"])
    fig = scp.pl.enrichment_plot(df, plot_type="dot", top_term=2)  # GeneRatio is enough
    matplotlib.pyplot.close(fig)
    with pytest.raises(ValueError, match="BgRatio"):
        scp.pl.enrichment_plot(df, plot_type="lollipop", top_term=2)


def test_dot_panels_carry_their_own_colourbar_and_size_legend():
    """Each panel is its own ggplot and SCP wraps them without collecting
    guides: distinct scales give every panel its own size legend (first,
    order = 1) and colourbar, stacked beside it."""
    df = toy_enrichment()
    alpha = df["Groups"] == "Alpha"
    df.loc[alpha, "p.adjust"] *= 1e-3
    df.loc[alpha, "Count"] = np.arange(alpha.sum()) + 2
    fig = scp.pl.enrichment_plot(df, plot_type="dot", top_term=3)
    panels = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    bars = [a for a in fig.axes if a.get_label() == "<colorbar>"]
    assert len(bars) == 2
    assert [lg.get_title().get_text() for lg in fig.legends] == ["Count:", "Count:"]
    for ax, leg, cb in zip(panels, fig.legends, bars, strict=True):
        assert _beside(fig, leg, ax) and _beside(fig, cb, ax)
        r = fig.canvas.get_renderer()
        lb, bb = leg.get_window_extent(r), cb.get_window_extent(r)
        # the size legend first: above the colourbar, or in the column before
        # it when the stack is taller than the panel and wraps
        assert lb.y0 > bb.y1 or lb.x1 < bb.x0
    matplotlib.pyplot.close(fig)


def test_guides_identical_in_every_panel_are_drawn_once():
    """A guide that is the same in every panel (same title, keys and scale) is
    drawn once in the legend column -- patchwork's collect behaviour --
    instead of N times; the toy groups share their counts and p-values."""
    fig = scp.pl.enrichment_plot(toy_enrichment(), plot_type="dot", top_term=3)
    panels = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    bars = [a for a in fig.axes if a.get_label() == "<colorbar>"]
    assert len(panels) == 2 and len(bars) == 1
    assert [lg.get_title().get_text() for lg in fig.legends] == ["Count:"]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    right = max(a.get_window_extent(r).x1 for a in panels)
    assert fig.legends[0].get_window_extent(r).x0 > right
    matplotlib.pyplot.close(fig)


def test_wordcloud_size_keys_do_not_overlap():
    """SCP's size key is a black "G" at each break's text size; each key's row
    grows to hold its glyph, as ggplot's legend keys do."""
    fig = scp.pl.enrichment_plot(toy_enrichment(), plot_type="wordcloud", group_use=["Beta"],
                                 word_type="feature")
    (leg,) = [lg for lg in fig.legends if lg.get_title().get_text() == "Count:"]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    glyphs = [t for t in leg.findobj(matplotlib.text.Text) if t.get_text() == "G"]
    assert len(glyphs) == len(leg.get_texts()) >= 2
    boxes = sorted((g.get_window_extent(r) for g in glyphs), key=lambda b: -b.y1)
    for upper, lower in zip(boxes[:-1], boxes[1:], strict=True):
        assert lower.y1 <= upper.y0 + 0.5
    # each glyph is level with its own label
    for g, t in zip(glyphs, leg.get_texts(), strict=True):
        gb, tb = g.get_window_extent(r), t.get_window_extent(r)
        assert gb.y0 <= (tb.y0 + tb.y1) / 2 <= gb.y1
    matplotlib.pyplot.close(fig)


def test_wordcloud_grid_search_equals_the_exhaustive_one():
    """The pixel grid only settles candidates the exact box test would settle
    the same way, so the first free candidate never changes."""
    rng = np.random.default_rng(7)
    W, H = 300.0, 240.0
    cand = dr._square_spiral(2.0, max(W, H) / 2)
    placed = np.zeros((0, 4))
    n_placed = 0
    for _ in range(150):
        w, h = rng.uniform(4, 90), rng.uniform(4, 30)
        x0, y0 = cand[:, 0] - w / 2, cand[:, 1] - h / 2
        inside = np.flatnonzero((x0 >= -W / 2) & (x0 + w <= W / 2)
                                & (y0 >= -H / 2) & (y0 + h <= H / 2))
        a = dr._first_free(x0[inside], y0[inside], w, h, placed)
        b = dr._first_free_grid(x0[inside], y0[inside], w, h, placed, W, H)
        assert a == b
        if a is not None:
            ox, oy = x0[inside[a]], y0[inside[a]]
            placed = np.vstack([placed, [ox, oy, ox + w, oy + h]])
            n_placed += 1
    assert 20 < n_placed < 150  # the panel fills up: both outcomes exercised


GOLDEN_ADJUST = np.array([[-10.862479090720004, -23.237920947404298],
                          [-3.3641566373263205, 6.9625262051326935],
                          [15.388200682313464, 1.8167257254791562]])
GOLDEN_ADJUST_SUM = -269.70731569609296


def test_adjust_layout_sweep_is_unchanged_by_vectorising():
    """Golden: the label-decollision sweep on a fixed graph (pinned before the
    nearest-neighbour search was vectorised)."""
    rng = np.random.default_rng(5)
    lay = rng.normal(size=(40, 2))
    lay[3] = lay[4]
    widths = rng.integers(3, 12, 40).astype(float)
    edges = [(int(a), int(b)) for a, b in rng.integers(0, 40, (60, 2)) if a != b]
    out = en.adjust_layout(lay, widths, edges, seed=2)
    assert out.shape == (40, 2)
    np.testing.assert_allclose(out[:3], GOLDEN_ADJUST, rtol=0, atol=1e-12)
    assert float(out.sum()) == pytest.approx(GOLDEN_ADJUST_SUM, abs=1e-9)


def test_gsea_line_rank_axis_uses_ggplot_breaks_and_labels_stay_inside():
    """ggplot's extended breaks on the rank axis (0, 4000, ... for ~16k
    genes), so five-digit labels never touch; the core-gene labels stay inside
    the panel, as ggrepel keeps them, clear of the key strip on the right."""
    rng = np.random.default_rng(1)
    n = 15836
    genes = [f"g{i}" for i in range(n)]
    gl = pd.Series(np.sort(rng.normal(size=n))[::-1] * 8, index=genes)
    core = genes[-12:]  # hits at the far right, where labels would overflow
    gs = core + list(rng.choice(genes[:-12], 40, replace=False))
    res = pd.DataFrame(dict(ID=["S1"], Description=["set one"], NES=[-1.5], pvalue=[1e-3],
                            **{"p.adjust": [1e-2]}, setSize=[len(gs)],
                            core_enrichment=["/".join(core)], Groups=["G"], Database=["DB"]))
    fig = scp.pl.gsea_plot({"G-DB": {"result": res, "geneList": gl, "geneSets": {"S1": gs}}},
                           direction="neg")
    top, mid, bot = fig.axes[:3]
    assert list(bot.get_xticks()) == [0, 4000, 8000, 12000]
    assert list(top.get_xticks()) == list(bot.get_xticks())
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    labels = [t for t in bot.get_xticklabels() if t.get_visible() and t.get_text()]
    boxes = sorted((t.get_window_extent(r) for t in labels), key=lambda b: b.x0)
    assert len(boxes) == 4
    for a, b in zip(boxes[:-1], boxes[1:], strict=True):
        assert b.x0 - a.x1 > 5  # clear gaps, in pixels
    panel = top.get_window_extent(r)
    names = [t for t in top.texts if t.get_text() in core]
    assert len(names) >= 9
    for t in names:
        bb = t.get_window_extent(r)
        assert panel.x0 <= bb.x0 and bb.x1 <= panel.x1
        assert panel.y0 <= bb.y0 and bb.y1 <= panel.y1
    matplotlib.pyplot.close(fig)


def test_gsea_bar_palcolor_is_read_as_pos_then_neg():
    """R passes palette_scp(c("Neg", "Pos"), palcolor = rev(palcolor))."""
    fig = scp.pl.gsea_plot(toy_gsea(), plot_type="bar", direction="both", group_use=["Beta"],
                           top_term=4, palcolor=["#FF0000", "#0000FF"])
    ax = fig.axes[0]
    got = {matplotlib.colors.to_hex(p.get_facecolor(), keep_alpha=False): p.get_width() > 0
           for p in ax.patches}
    assert got == {"#ff0000": True, "#0000ff": False}
    matplotlib.pyplot.close(fig)


def test_gsea_line_without_running_scores_names_what_to_pass():
    with pytest.raises(ValueError, match="geneList"):
        scp.pl.gsea_plot(toy_gsea(), plot_type="line", top_term=1)


def test_gsea_line_and_bar_compose_on_a_panel_grid():
    """One figure per call, one panel (bar) or one 3-row block (line) per key."""
    rng = np.random.default_rng(0)
    df = toy_gsea()
    gl = pd.Series(np.sort(rng.normal(size=60))[::-1],
                   index=[f"{'AB'[i % 2]}{i // 2}" for i in range(60)])
    sets = {i: g.split("/") for i, g in zip(df["ID"], df["core_enrichment"], strict=True)}
    res = {f"{g}-GO_BP": {"result": df[df["Groups"] == g], "geneList": gl, "geneSets": sets}
           for g in ("Beta", "Alpha")}
    bar = scp.pl.gsea_plot(res, plot_type="bar", direction="both", top_term=2)
    assert [a.get_title() for a in bar.axes] == ["Beta", "Alpha"]
    line = scp.pl.gsea_plot(res, plot_type="line", direction="both", top_term=2)
    assert len(line.axes) == 6
    assert line.get_layout_engine() is not None
    matplotlib.pyplot.close("all")


def test_gsea_heat_strip_and_zero_cross_golden():
    y = np.array([3.0, 2.0, 1.0, 0.5, -0.5, -0.5, -2.0, -4.0])
    # min |y| is first reached at rank 4 (0.5); only that value counts
    assert en.gsea_zero_cross(y) == 4.0
    assert en.gsea_zero_cross(np.array([2.0, -0.5, 0.5, -0.5, -1.0])) == 3.0
    strip = en.gsea_heat_strip(np.array([2.0, 1.0, 0.0, -1.0, -2.0]))
    # quantile clipping keeps the ends; each sign spans its own 99-bin ramp
    assert [(a, b) for a, b, _ in strip] == [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    cols = [c for *_, c in strip]
    assert cols[2] == "white"
    assert cols[0] == scp.palettes.color_ramp(["#F5DCDC", "#C40003"], 100)[98]
    assert cols[4] == scp.palettes.color_ramp(["#1D008F", "#DDDCF5"], 100)[0]


def test_rendered_example_exists():
    for name in ("enrichment_network", "enrichment_dot", "gsea_line"):
        assert (Path(__file__).parent / "fixtures" / f"example_{name}.png").exists()
