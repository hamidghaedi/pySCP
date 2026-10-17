"""``cell_stat_plot`` sankey / chord / venn and ``individual``, against SCP.

The fixture ``tests/fixtures/parity/cell_stat/flow_parity.json`` is written by
``tools/parity/cell_stat_flow.R`` on SCP 0.5.6's ``pancreas_sub``. It holds
the *aggregated* input of each case (combination counts plus factor levels),
which is all the layouts depend on, next to what R computed from it: ggsankey's
node and flow geometry from ``ggplot_build``, circlize's sector degrees and
link positions from ``chordDiagram``'s return value, ggVennDiagram's region
counts, and ``StatPlot(individual = TRUE)``'s panels. R is not needed to run
these tests.

Spec: docs/porting_briefs/stat_plots.md section 2.3.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.patches import PathPatch, Polygon, Rectangle

import scp
from scp.pl._stat_flow import (
    _membership,
    chord_layout,
    resolve_stat_level,
    sankey_layout,
    venn_regions,
)

FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "cell_stat" / "flow_parity.json"


@pytest.fixture(scope="module")
def parity():
    return json.loads(FIXTURE.read_text())


def _frame(case) -> pd.DataFrame:
    """Rebuild a cell-level frame from a case's combination counts."""
    combos = pd.DataFrame(case["combos"])
    n = combos.pop("n").to_numpy()
    df = combos.loc[combos.index.repeat(n)].reset_index(drop=True)
    for c, lv in case["levels"].items():
        if set(lv) <= {"TRUE", "FALSE"}:
            df[c] = df[c] == "TRUE"
        else:
            df[c] = pd.Categorical(df[c], categories=lv)
    df.index = [f"cell{i}" for i in range(len(df))]
    return df


# --------------------------------------------------------------------------- #
# sankey
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["sankey", "sankey_alpha"])
def test_sankey_colors_are_the_union_of_levels(parity, key):
    case = parity[key]
    names = list(case["colors"])
    assert names[-1] == "NA"
    expected = scp.discrete_palette(names[:-1], palette="Paired")
    assert {k: v.upper() for k, v in expected.items()} == \
        {k: v for k, v in case["colors"].items() if k != "NA"}


@pytest.mark.parametrize("key", ["sankey", "sankey_alpha"])
def test_sankey_layout_matches_ggsankey(parity, key):
    """Nodes and flows to rounding error, with SCP's own flow order."""
    case = parity[key]
    df = _frame(case).astype(str)
    nodes, flows = sankey_layout(df, case["stat_by"], list(case["colors"]), flow_order="r")

    r_nodes = pd.DataFrame(case["nodes"])
    m = r_nodes.merge(nodes, on=["column", "node"], suffixes=("_r", ""))
    assert len(m) == len(r_nodes) == len(nodes)
    for c in ("xmin", "xmax", "ymin", "ymax", "freq"):
        np.testing.assert_allclose(m[c], m[f"{c}_r"], atol=1e-9)

    r_flows = pd.DataFrame(case["flows"])
    keys = ["source_column", "source", "target_column", "target"]
    mf = r_flows.merge(flows, on=keys, suffixes=("_r", ""))
    assert len(mf) == len(r_flows) == len(flows)
    for c in ("start_ymin", "start_ymax", "end_ymin", "end_ymax", "x_start", "x_end"):
        np.testing.assert_allclose(mf[c], mf[f"{c}_r"], atol=1e-6)


def test_sankey_default_order_only_reorders_flows_inside_their_source(parity):
    """The fix moves ribbons within a source node; nothing else changes."""
    case = parity["sankey"]
    df = _frame(case).astype(str)
    order = list(case["colors"])
    n_r, f_r = sankey_layout(df, case["stat_by"], order, flow_order="r")
    n_fix, f_fix = sankey_layout(df, case["stat_by"], order)
    pd.testing.assert_frame_equal(n_r, n_fix)
    keys = ["source_column", "source", "target_column", "target"]
    m = f_r.merge(f_fix, on=keys, suffixes=("_r", ""))
    np.testing.assert_allclose(m.value, m.value_r)
    np.testing.assert_allclose(m.end_ymax, m.end_ymax_r)
    # outgoing ribbons now leave each node in their targets' top-to-bottom order
    rank = {n: i for i, n in enumerate(order)}
    for _, grp in f_fix.groupby(["source_column", "source"]):
        by_height = grp.sort_values("start_ymax", ascending=False)
        assert [rank[t] for t in by_height.target] == sorted(rank[t] for t in grp.target)


def test_sankey_figure_structure(parity):
    case = parity["sankey_alpha"]
    df = _frame(case)
    fig = scp.pl.stat_plot(df, stat_by=case["stat_by"], plot_type="sankey", alpha=0.5)
    ax = fig.axes[0]
    rects = [p for p in ax.patches if isinstance(p, Rectangle)]
    ribbons = [p for p in ax.patches if isinstance(p, Polygon)]
    assert len(rects) == len(case["nodes"]["node"])
    assert len(ribbons) == len(case["flows"]["source"])
    assert all(abs(p.get_facecolor()[3] - 0.5) < 1e-9 for p in ribbons)
    assert [t.get_text() for t in ax.get_xticklabels()] == case["stat_by"]
    # one legend per stat_by column, titled by it, listing its levels
    titles = [lg.get_title().get_text() for lg in fig.legends]
    assert titles == [f"{c}:" for c in case["stat_by"]]
    first = [t.get_text() for t in fig.legends[0].get_texts()]
    assert first == case["levels"][case["stat_by"][0]]


def test_sankey_node_colors_follow_the_shared_palette(parity):
    case = parity["sankey"]
    fig = scp.pl.stat_plot(_frame(case), stat_by=case["stat_by"], plot_type="sankey")
    want = sorted(c.lower() for c in case["nodes"]["fill"])
    got = sorted(to_hex(p.get_facecolor()).lower() for p in fig.axes[0].patches
                 if isinstance(p, Rectangle))
    assert got == want


def test_sankey_keeps_missing_values_as_a_node():
    df = pd.DataFrame({"a": list("xxyy"), "b": ["p", None, "q", "q"]})
    fig = scp.pl.stat_plot(df, stat_by=["a", "b"], plot_type="sankey")
    rects = [p for p in fig.axes[0].patches if isinstance(p, Rectangle)]
    assert len(rects) == 2 + 3  # x, y | p, NA, q
    assert "#bebebe" in {to_hex(p.get_facecolor()) for p in rects}


# --------------------------------------------------------------------------- #
# chord
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["chord", "chord_shared"])
def test_chord_layout_matches_circlize(parity, key):
    """Sector order and degrees, and every link's two ends, against circlize."""
    case = parity[key]
    mat = case["matrix"]
    tab = pd.DataFrame(np.asarray(mat["values"]), index=mat["rows"], columns=mat["cols"])
    sectors, links = chord_layout(tab)

    r_sec = pd.DataFrame(case["sectors"])
    assert list(sectors.name) == list(r_sec.name)
    np.testing.assert_allclose(sectors.start_degree, r_sec.start_degree, atol=1e-9)
    np.testing.assert_allclose(sectors.end_degree, r_sec.end_degree, atol=1e-9)
    np.testing.assert_allclose(sectors.xmax, r_sec.xmax)

    r_links = pd.DataFrame(case["links"])
    assert list(zip(links.rn, links.cn, strict=True)) == list(zip(r_links.rn, r_links.cn,
                                                                  strict=True))
    np.testing.assert_allclose(links.value, r_links.value)
    np.testing.assert_allclose(links.x1, r_links.x1)
    np.testing.assert_allclose(links.x2, r_links.x2)


def test_chord_matrix_is_the_cross_tab(parity):
    case = parity["chord"]
    df = _frame(case)
    tab = pd.crosstab(df["CellType"], df["Phase"]).reindex(
        index=case["matrix"]["rows"], columns=case["matrix"]["cols"])
    np.testing.assert_array_equal(tab.to_numpy(), np.asarray(case["matrix"]["values"]))


def test_chord_shared_names_merge_into_one_sector(parity):
    """CellType and SubCellType share four level names: one sector each."""
    case = parity["chord_shared"]
    names = list(pd.DataFrame(case["sectors"]).name)
    assert len(names) == len(set(names)) == 9
    assert names[:5] == case["levels"]["CellType"]


def test_chord_figure_structure(parity):
    case = parity["chord"]
    fig = scp.pl.stat_plot(_frame(case), stat_by=case["stat_by"], plot_type="chord")
    ax = fig.axes[0]
    links = [p for p in ax.patches if isinstance(p, PathPatch) and p.get_fill()]
    n_nonzero = sum(v > 0 for v in case["links"]["value"])
    assert len(links) == n_nonzero
    # link fill: its row sector's color at transparency 0.2
    want = {c.lower() for c in case["links"]["col"]}
    got = {to_hex(p.get_facecolor(), keep_alpha=True).lower() for p in links}
    assert got <= want
    texts = {t.get_text() for t in ax.texts}
    assert set(pd.DataFrame(case["sectors"]).name) <= texts
    assert ax.get_aspect() == 1.0
    assert not fig.legends


def test_chord_needs_exactly_two_columns(adata):
    with pytest.raises(ValueError, match="exactly 2"):
        scp.pl.cell_stat_plot(adata, ["leiden"], plot_type="chord")


def test_chord_split_grid_follows_par_mfrow(adata):
    ad = adata.copy()
    ad.obs["trio"] = pd.Categorical(np.resize(["u", "v", "w"], ad.n_obs))
    fig = scp.pl.cell_stat_plot(ad, ["leiden", "batch"], split_by="trio", plot_type="chord")
    assert [ax.get_title() for ax in fig.axes] == ["u", "v", "w"]
    rows = {round(ax.get_position().y0, 3) for ax in fig.axes}
    assert len(rows) == 2  # nrow = ceiling(sqrt(3)) = 2


def test_chord_names_a_missing_row_level():
    df = pd.DataFrame({"a": ["x", None, "x", "y"], "b": ["p", "q", "q", None]})
    fig = scp.pl.stat_plot(df, stat_by=["a", "b"], plot_type="chord")
    texts = [t.get_text() for t in fig.axes[0].texts]
    # R names only the column NA; naming both makes them one shared sector,
    # exactly as any other level present on both sides
    assert texts.count("NA") == 1
    tab = pd.DataFrame([[1, 1, 0], [0, 1, 0], [0, 0, 1]], index=["x", "NA", "y"],
                       columns=["p", "q", "NA"])
    sectors, _ = chord_layout(tab)
    assert list(sectors.name) == ["x", "NA", "y", "p", "q"]


# --------------------------------------------------------------------------- #
# venn
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["venn4", "venn5"])
def test_venn_region_counts_match_ggvenndiagram(parity, key):
    case = parity[key]
    df = _frame(case)
    levels = resolve_stat_level(df, case["stat_by"], "TRUE")
    mem = _membership(df, case["stat_by"], levels)
    sets = {c: set(mem.index[mem[c].to_numpy()]) for c in case["stat_by"]}
    assert {c: len(s) for c, s in sets.items()} == case["set_sizes"]
    reg = venn_regions(sets)
    r = pd.DataFrame(case["regions"])
    assert list(reg.id) == list(r.id)
    assert list(reg.name) == list(r.name)
    assert list(reg["count"]) == list(r["count"])
    np.testing.assert_allclose(reg.pct, r.pct)


@pytest.mark.parametrize("key", ["venn4", "venn5"])
def test_venn_figure_labels_every_region(parity, key):
    case = parity[key]
    fig = scp.pl.stat_plot(_frame(case), stat_by=case["stat_by"], plot_type="venn",
                           stat_level="TRUE")
    assert len(fig.axes) == 1
    texts = [t.get_text() for t in fig.axes[0].texts]
    r = pd.DataFrame(case["regions"])
    pct = [f"{p:.15g}" for p in r.pct]
    region_labels = [f"{c}\n{p}%" for c, p in zip(r["count"], pct, strict=True)]
    assert sorted(t for t in texts if "%" in t) == sorted(region_labels)
    for name, size in case["set_sizes"].items():
        assert f"{name}\n({size})" in texts
    # set outlines in the set colors
    lines = {to_hex(ln.get_color()).upper() for ln in fig.axes[0].lines}
    assert lines == set(case["set_colors"].values())


@pytest.mark.parametrize("n", [1, 2, 3, 6, 7])
def test_venn_draws_every_supported_set_count(n):
    rng = np.random.default_rng(n)
    df = pd.DataFrame(rng.random((300, n)) < 0.4, columns=[f"s{i}" for i in range(n)])
    fig = scp.pl.stat_plot(df, stat_by=list(df.columns), plot_type="venn", stat_level="TRUE")
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert sum("%" in t for t in texts) == 2 ** n - 1


def test_venn_caps_at_seven_sets():
    df = pd.DataFrame(np.ones((5, 8), dtype=bool), columns=[f"s{i}" for i in range(8)])
    with pytest.raises(ValueError, match="at most 7"):
        scp.pl.stat_plot(df, stat_by=list(df.columns), plot_type="venn", stat_level="TRUE")


def test_venn_split_draws_one_panel_per_level(adata):
    fig = scp.pl.cell_stat_plot(adata, ["leiden", "batch"], plot_type="venn",
                                split_by="batch", stat_level={"leiden": "A", "batch": "b1"})
    assert len(fig.axes) == adata.obs["batch"].nunique()


def test_stat_level_vector_is_positional(adata):
    """R names an unnamed stat_level vector by stat_by, so it is positional."""
    lv = resolve_stat_level(adata.obs, ["leiden", "batch"], ["A", "b1"])
    assert lv == {"leiden": ["A"], "batch": ["b1"]}
    with pytest.raises(ValueError, match="length 1"):
        resolve_stat_level(adata.obs, ["leiden", "batch"], ["A", "B", "C"])


def test_stat_level_default_for_logical_is_false():
    df = pd.DataFrame({"flag": [True, False, True]})
    with pytest.warns(UserWarning, match="flag=FALSE"):
        assert resolve_stat_level(df, ["flag"], None) == {"flag": ["FALSE"]}


def test_set_types_warn_that_group_by_is_ignored(adata):
    with pytest.warns(UserWarning, match="group_by"):
        scp.pl.cell_stat_plot(adata, ["leiden", "batch"], group_by="batch", plot_type="sankey")


# --------------------------------------------------------------------------- #
# individual
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["individual", "individual_count"])
def test_individual_panels_match_statplot(parity, key):
    case = parity[key]
    df = _frame(case)
    fig = scp.pl.stat_plot(df, stat_by=[case["stat_by"]], group_by=case["group_by"],
                           split_by=case["split_by"], plot_type="bar",
                           stat_type=case["stat_type"], individual=True)
    panels = case["panels"]
    assert len(fig.axes) - 0 == len(panels)
    assert [ax.get_title() for ax in fig.axes] == [p["title"] for p in panels]
    colors = scp.discrete_palette(case["levels"][case["stat_by"]], palette="Paired")
    by_color = {v.lower(): k for k, v in colors.items()}
    for ax, p in zip(fig.axes, panels, strict=True):
        ticks = [t.get_text() for t in ax.get_xticklabels()]
        assert ticks == sorted(set(p["group"]))
        heights = {by_color[to_hex(r.get_facecolor()).lower()]: r.get_height()
                   for r in ax.patches if isinstance(r, Rectangle)}
        want = dict(zip(p["stat"], p["value"], strict=True))
        assert heights.keys() == want.keys()
        for lv, v in want.items():
            assert heights[lv] == pytest.approx(v)


def test_individual_gives_one_panel_per_group_and_split(adata):
    fig = scp.pl.cell_stat_plot(adata, "leiden", group_by="batch", split_by="batch",
                                individual=True)
    n = adata.obs["batch"].nunique()
    assert len(fig.axes) == n * n


def test_empty_group_keeps_its_slot_only_in_count_mode(adata):
    """R filters NA values, and only percent of an empty group is NA."""
    ad = adata.copy()
    ad.obs["grp"] = ad.obs["batch"].cat.add_categories(["unused"])
    pct = scp.pl.cell_stat_plot(ad, "leiden", group_by="grp")
    cnt = scp.pl.cell_stat_plot(ad, "leiden", group_by="grp", stat_type="count")
    assert "unused" not in [t.get_text() for t in pct.axes[0].get_xticklabels()]
    assert "unused" in [t.get_text() for t in cnt.axes[0].get_xticklabels()]
