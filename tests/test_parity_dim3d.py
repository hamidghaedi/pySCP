"""Numeric parity of the 3-D embedding plots against SCP's R output.

The fixtures under ``tests/fixtures/parity/audit/dim3d`` are written by
``tools/parity/audit_dim3d.R`` from SCP's ``CellDimPlot3D`` and
``FeatureDimPlot3D`` (Hao Zhang, NJMU, GPL-3;
https://github.com/zhanghao-njmu/SCP): the ``plotly_build()`` traces R hands to
plotly.js, plus the per-cell inputs, so this module needs neither R nor the
parity ``.h5ad``.

Where the port agrees the assertion is tight.  Where it does not, the test is
``xfail(strict=True)`` with a one-line diagnosis, so the suite stays green and
the fix flips it.  The diagnoses are written out in ``docs/parity_audit.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scp

pytest.importorskip("plotly")
anndata = pytest.importorskip("anndata")

FIX = Path(__file__).parent / "fixtures" / "parity" / "audit" / "dim3d"


def _json(name: str) -> dict:
    return json.loads((FIX / name).read_text())


@pytest.fixture(scope="module")
def pancreas3d():
    inp = pd.read_csv(FIX / "input.csv", keep_default_na=False, na_values=[""])
    levels = _json("input_levels.json")["CellType"]
    obs = pd.DataFrame(index=pd.Index(inp["cell"].astype(str)))
    obs["CellType"] = pd.Categorical(inp["CellType"].to_numpy(), categories=levels)
    obs["CellTypeNA"] = pd.Categorical(inp["CellTypeNA"].to_numpy(), categories=levels)
    obs["S_score"] = inp["S_score"].to_numpy()
    a = anndata.AnnData(X=inp[["Ins1", "Gcg"]].to_numpy(), obs=obs,
                        var=pd.DataFrame(index=["Ins1", "Gcg"]))
    a.obsm["X_pca"] = inp[["PC_1", "PC_2", "PC_3"]].to_numpy()
    return a


def _rgba_hex(s: str) -> str:
    r, g, b = (int(v) for v in s[s.index("(") + 1:-1].split(",")[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def _pos(a) -> dict[str, int]:
    return {c: i for i, c in enumerate(a.obs_names)}


# --------------------------------------------------------------- cell_dim_plot_3d
def test_cell3d_traces_match_r(pancreas3d):
    """One trace per level, in level order, with R's palette colour."""
    r = _json("cell3d.json")
    fig = scp.pl.cell_dim_plot_3d(pancreas3d, "CellType", reduction="X_pca")
    assert [t.name for t in fig.data] == [t["name"] for t in r["traces"]]
    for pt, rt in zip(fig.data, r["traces"], strict=True):
        assert pt.marker.color.upper() == _rgba_hex(rt["marker_color"])
        assert pt.mode == rt["mode"] == "markers"


def test_cell3d_trace_membership_and_coordinates(pancreas3d):
    """Each trace holds exactly R's cells, in R's order, at the embedding."""
    r = _json("cell3d.json")
    fig = scp.pl.cell_dim_plot_3d(pancreas3d, "CellType", reduction="X_pca")
    pos, emb = _pos(pancreas3d), pancreas3d.obsm["X_pca"]
    for pt, rt in zip(fig.data, r["traces"], strict=True):
        xyz = emb[[pos[c] for c in rt["cells"]]]
        np.testing.assert_allclose(np.c_[pt.x, pt.y, pt.z], xyz, rtol=0, atol=0)


def test_cell3d_na_cells_drawn_by_default(pancreas3d):
    r = _json("cell3d_na.json")
    fig = scp.pl.cell_dim_plot_3d(pancreas3d, "CellTypeNA", reduction="X_pca")
    assert [t.name for t in fig.data] == [t["name"] for t in r["traces"]]


def test_cell3d_na_trace_uses_bg_color(pancreas3d):
    r = _json("cell3d_na.json")
    fig = scp.pl.cell_dim_plot_3d(pancreas3d, "CellTypeNA", reduction="X_pca", show_na=True)
    na = [t for t in fig.data if t.name == "NA"][0]
    r_na = [t for t in r["traces"] if t["name"] == "NA"][0]
    assert na.marker.color.upper() == _rgba_hex(r_na["marker_color"]) == "#CCCCCC"


def test_cell3d_na_trace_membership_when_shown(pancreas3d):
    r = _json("cell3d_na.json")
    fig = scp.pl.cell_dim_plot_3d(pancreas3d, "CellTypeNA", reduction="X_pca", show_na=True)
    pos, emb = _pos(pancreas3d), pancreas3d.obsm["X_pca"]
    for pt, rt in zip(fig.data, r["traces"], strict=True):
        assert pt.name == rt["name"]
        np.testing.assert_array_equal(pt.x, emb[[pos[c] for c in rt["cells"]], 0])


def test_cell3d_default_marker_size(pancreas3d):
    r = _json("cell3d.json")
    fig = scp.pl.cell_dim_plot_3d(pancreas3d, "CellType", reduction="X_pca")
    assert fig.data[0].marker.size == r["traces"][0]["marker_size"]


def test_cell3d_scene_ranges(pancreas3d):
    r = _json("cell3d.json")
    fig = scp.pl.cell_dim_plot_3d(pancreas3d, "CellType", reduction="X_pca")
    sc = fig.layout.scene
    for ax, key in ((sc.xaxis, "xaxis"), (sc.yaxis, "yaxis"), (sc.zaxis, "zaxis")):
        assert ax.range is not None
        np.testing.assert_allclose(ax.range, r[key]["range"], rtol=1e-6)


def test_cell3d_title(pancreas3d):
    r = _json("cell3d.json")
    fig = scp.pl.cell_dim_plot_3d(pancreas3d, "CellType", reduction="X_pca")
    assert fig.layout.title.text == r["title"]


# ------------------------------------------------------------ feature_dim_plot_3d
def test_feature3d_values_match_r_where_drawn(pancreas3d):
    """Per-feature colour values equal R's dropdown arrays wherever Python keeps them."""
    r = _json("feature3d.json")
    feats = r["button_labels"]
    fig = scp.pl.feature_dim_plot_3d(pancreas3d, feats, reduction="X_pca")
    assert [t.name for t in fig.data] == feats
    order = [_pos(pancreas3d)[c] for c in r["cells"]]
    assert order == list(range(pancreas3d.n_obs))  # R keeps object order
    for t, rc in zip(fig.data, r["button_colors"], strict=True):
        pc = np.asarray(t.marker.color, dtype=float)
        rc = np.asarray([np.nan if v is None else v for v in rc], dtype=float)
        keep = np.isfinite(pc)
        np.testing.assert_allclose(pc[keep], rc[keep], rtol=1e-12)


def test_feature3d_no_masking_by_default(pancreas3d):
    r = _json("feature3d.json")
    fig = scp.pl.feature_dim_plot_3d(pancreas3d, r["button_labels"], reduction="X_pca")
    for t, rc in zip(fig.data, r["button_colors"], strict=True):
        pc = np.asarray(t.marker.color, dtype=float)
        rc = np.asarray([np.nan if v is None else v for v in rc], dtype=float)
        np.testing.assert_allclose(pc, rc, rtol=1e-12)
        assert np.nanmin(pc) == pytest.approx(np.nanmin(rc))


def test_feature3d_colorscale_is_the_documented_divergence(pancreas3d):
    """R leaves the colorscale to plotly.js (documented divergence: Python uses Spectral)."""
    r = _json("feature3d.json")
    assert not r["colorscale"]  # R passes none
    fig = scp.pl.feature_dim_plot_3d(pancreas3d, "Ins1", reduction="X_pca")
    ramp = scp.continuous_palette(palette="Spectral", n=100)
    assert [c for _, c in fig.data[0].marker.colorscale] == ramp


# ---------------------------------------------- highlight, lineages and split.by
@pytest.fixture(scope="module")
def pancreas3d_extra(pancreas3d):
    """The same object plus Phase and RunSlingshot's three lineages."""
    a = pancreas3d.copy()
    ex = pd.read_csv(FIX / "extra_input.csv", keep_default_na=False, na_values=[""])
    assert list(ex["cell"].astype(str)) == list(a.obs_names)
    a.obs["Phase"] = pd.Categorical(ex["Phase"].to_numpy(),
                                    categories=_json("extra_levels.json")["Phase"])
    for lin in ("Lineage1", "Lineage2", "Lineage3"):
        a.obs[lin] = ex[lin].to_numpy(dtype=float)
    return a


def _r_highlight(name):
    r = _json(name)
    return r["trace_names"], r["highlight"]


def test_cell3d_highlight_list_matches_r(pancreas3d_extra):
    """A highlight trace after the levels: R's cells in R's (the caller's) order."""
    names, rh = _r_highlight("cell3d_highlight.json")
    a = pancreas3d_extra
    fig = scp.pl.cell_dim_plot_3d(a, "CellType", reduction="X_pca", cells_highlight=rh["cells"])
    assert [t.name for t in fig.data] == names
    h = fig.data[-1]
    assert (h.name, h.mode) == (rh["name"], rh["mode"])
    assert (h.marker.color, h.marker.size, h.marker.symbol) == \
        (rh["marker_color"], rh["marker_size"], rh["marker_symbol"])
    xyz = a.obsm["X_pca"][[_pos(a)[c] for c in rh["cells"]]]
    np.testing.assert_array_equal(np.c_[h.x, h.y, h.z], xyz)


def test_cell3d_highlight_true_matches_r(pancreas3d_extra):
    """``True`` highlights every cell with a group (the NA trace's cells are left out)."""
    names, rh = _r_highlight("cell3d_highlight_true.json")
    a = pancreas3d_extra
    fig = scp.pl.cell_dim_plot_3d(a, "CellTypeNA", reduction="X_pca", cells_highlight=True,
                                  sizes_highlight=3, cols_highlight="red")
    assert [t.name for t in fig.data] == names
    h = fig.data[-1]
    assert (h.marker.color, h.marker.size, h.marker.symbol) == \
        (rh["marker_color"], rh["marker_size"], rh["marker_symbol"])
    np.testing.assert_array_equal(h.x, a.obsm["X_pca"][[_pos(a)[c] for c in rh["cells"]], 0])


def test_cell3d_highlight_validation(pancreas3d_extra):
    with pytest.raises(ValueError, match="No cells"):
        scp.pl.cell_dim_plot_3d(pancreas3d_extra, "CellType", reduction="X_pca",
                                cells_highlight=["nope"])
    with pytest.warns(UserWarning, match="not found"):
        scp.pl.cell_dim_plot_3d(pancreas3d_extra, "CellType", reduction="X_pca",
                                cells_highlight=[pancreas3d_extra.obs_names[0], "nope"])


def test_cell3d_lineage_lines_match_r(pancreas3d_extra):
    """Per-axis degree-2 LOESS on every lineage cell, range-filtered and de-duplicated."""
    pytest.importorskip("skmisc")
    r = _json("cell3d_lineages.json")
    lins = [t["name"] for t in r]
    fig = scp.pl.cell_dim_plot_3d(pancreas3d_extra, "CellType", reduction="X_pca",
                                  lineages=lins)
    lines = [t for t in fig.data if t.mode == "lines"]
    assert [t.name for t in lines] == lins == [t.name for t in fig.data[-3:]]
    worst = 0.0
    for t, rt in zip(lines, r, strict=True):
        got = np.c_[t.x, t.y, t.z]
        want = np.c_[rt["x"], rt["y"], rt["z"]]
        assert got.shape == want.shape
        worst = max(worst, float(np.abs(got - want).max()))
        assert t.line.color.upper() == rt["line_color"].upper()
        assert t.line.width == rt["line_width"]
    assert worst < 1e-10, worst  # 1.3e-13 when written


def test_cell3d_missing_lineage_raises(pancreas3d_extra):
    with pytest.raises(KeyError, match="lineages"):
        scp.pl.cell_dim_plot_3d(pancreas3d_extra, "CellType", reduction="X_pca",
                                lineages=["nope"])


def test_cell3d_axis_labs(pancreas3d_extra):
    fig = scp.pl.cell_dim_plot_3d(pancreas3d_extra, "CellType", reduction="X_pca",
                                  axis_labs=["a", "b", "c"])
    sc = fig.layout.scene
    assert (sc.xaxis.title.text, sc.yaxis.title.text, sc.zaxis.title.text) == ("a", "b", "c")


@pytest.fixture(scope="module")
def feature_split(pancreas3d_extra):
    r = _json("feature3d_split.json")
    hl = r["traces"][1]["cells"]
    fig = scp.pl.feature_dim_plot_3d(pancreas3d_extra, r["feature_labels"], reduction="X_pca",
                                     split_by="Phase", cells_highlight=hl)
    return fig, r


def test_feature3d_title_and_ranges_match_r(feature_split):
    fig, r = feature_split
    assert fig.layout.title.text == r["title"]
    sc = fig.layout.scene
    for ax, k in ((sc.xaxis, "x"), (sc.yaxis, "y"), (sc.zaxis, "z")):
        np.testing.assert_allclose(ax.range, r["ranges"][k], rtol=1e-6)


def test_feature3d_highlight_trace_matches_r(feature_split, pancreas3d_extra):
    fig, r = feature_split
    rh = r["traces"][1]
    h = fig.data[-1]
    assert [t.name for t in fig.data] == [*r["feature_labels"], "highlight"]
    assert (h.name, h.marker.size, h.marker.color, h.marker.symbol) == \
        (rh["name"], rh["marker_size"], rh["marker_color"], rh["marker_symbol"])
    a = pancreas3d_extra
    np.testing.assert_array_equal(h.x, a.obsm["X_pca"][[_pos(a)[c] for c in rh["cells"]], 0])


def test_feature3d_feature_dropdown_keeps_the_highlight(feature_split):
    fig, r = feature_split
    menu = fig.layout.updatemenus[-1]
    assert [b.label for b in menu.buttons] == r["feature_labels"]
    assert [b.args[0]["visible"] for b in menu.buttons] == [[True, False, True],
                                                             [False, True, True]]


def test_feature3d_split_dropdown_matches_r(feature_split):
    """Labels, titles and the per-level mask of every trace, as R's split buttons."""
    fig, r = feature_split
    menu = fig.layout.updatemenus[0]
    rb = r["split_buttons"]
    assert [b.label for b in menu.buttons] == [b["label"] for b in rb]
    assert [b.args[1]["title"]["text"] for b in menu.buttons] == [b["title"] for b in rb]
    nfeat = len(r["feature_labels"])
    for pb, b in list(zip(menu.buttons, rb, strict=True))[1:]:
        xs = pb.args[0]["x"]
        assert len(xs) == nfeat + 1
        main, hl = b["x"]
        for x in xs[:nfeat]:
            assert [v is None for v in x] == [v is None for v in main]
            # R's JSON carries 15 significant digits
            np.testing.assert_allclose([v for v in x if v is not None],
                                       [v for v in main if v is not None], rtol=1e-12)
        assert [v is None for v in xs[-1]] == [v is None for v in hl]


def test_feature3d_all_cells_button_restores_every_cell(feature_split, pancreas3d_extra):
    """R bug, fixed: with split.by set, R's All.cells button carries no coordinates."""
    fig, r = feature_split
    assert r["split_buttons"][0]["x"] == [[], []]  # R: the plot empties
    b = fig.layout.updatemenus[0].buttons[0]
    assert b.label == "All.cells"
    x = pancreas3d_extra.obsm["X_pca"][:, 0]
    for xs in b.args[0]["x"][:-1]:
        np.testing.assert_array_equal(xs, x)


def test_feature3d_too_many_features_needs_force(pancreas3d_extra):
    a = pancreas3d_extra
    feats = ["Ins1"] + [f"PCX{i}" for i in range(55)]
    b = a.copy()
    for f in feats[1:]:
        b.obs[f] = 1.0
    with pytest.raises(ValueError, match="force=True"):
        scp.pl.feature_dim_plot_3d(b, feats, reduction="X_pca")
    assert len(scp.pl.feature_dim_plot_3d(b, feats, reduction="X_pca", force=True).data) == 56
