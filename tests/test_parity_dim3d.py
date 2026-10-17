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


@pytest.mark.xfail(strict=True, reason="NA cells are dropped unless show_na=True; R always draws "
                   "them as an 'NA' trace (CellDimPlot3D has no show_na switch)")
def test_cell3d_na_cells_drawn_by_default(pancreas3d):
    r = _json("cell3d_na.json")
    fig = scp.pl.cell_dim_plot_3d(pancreas3d, "CellTypeNA", reduction="X_pca")
    assert [t.name for t in fig.data] == [t["name"] for t in r["traces"]]


@pytest.mark.xfail(strict=True, reason="show_na=True colours the NA level from the palette "
                   "(#FF7F00) instead of bg_color, as R's palette_scp(NA_color=bg_color) does")
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


@pytest.mark.xfail(strict=True, reason="pt_size defaults to 2.0; R's pt.size default is 1.5")
def test_cell3d_default_marker_size(pancreas3d):
    r = _json("cell3d.json")
    fig = scp.pl.cell_dim_plot_3d(pancreas3d, "CellType", reduction="X_pca")
    assert fig.data[0].marker.size == r["traces"][0]["marker_size"]


@pytest.mark.xfail(strict=True, reason="scene axis ranges are left to plotly autorange; R pins "
                   "each to the observed min/max")
def test_cell3d_scene_ranges(pancreas3d):
    r = _json("cell3d.json")
    fig = scp.pl.cell_dim_plot_3d(pancreas3d, "CellType", reduction="X_pca")
    sc = fig.layout.scene
    for ax, key in ((sc.xaxis, "xaxis"), (sc.yaxis, "yaxis"), (sc.zaxis, "zaxis")):
        assert ax.range is not None
        np.testing.assert_allclose(ax.range, r[key]["range"], rtol=1e-6)


@pytest.mark.xfail(strict=True, reason="no figure title; R titles it 'Total (nCells:N)'")
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


@pytest.mark.xfail(strict=True, reason="bg_cutoff=0 NaN-masks every value <= 0 (76-80% of cells, "
                   "all negative S_score); FeatureDimPlot3D has no bg_cutoff and masks nothing")
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
