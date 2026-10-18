"""The heatmap size engine: everything inside the figure, R's viewport ceiling.

Geometric assertions in figure inches, on figures saved *without*
``bbox_inches="tight"`` -- the way a user saving ``res.fig`` gets them.  The
viewport numbers are read back from SCP's own ``heatmap_rendersize`` /
``heatmap_fixsize`` by ``tools/parity/heatmap_fixtures.R``.
"""

from __future__ import annotations

import io
import json
import pathlib

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.text import Text

import scp
from scp.heatmap.spec import VIEWPORT_IN
from scp.layout import MARGIN

SIZES = pathlib.Path(__file__).parent / "fixtures" / "parity" / "heatmap" / "heatmap_sizes.json"
TOL = 0.01  # inches

LONG = {"A": "Ngn3 low EP", "B": "Pre-endocrine", "C": "Ngn3 high EP", "D": "Endocrine"}


@pytest.fixture()
def wide(adata):
    """The synthetic object with pancreas-length names, which is what clipped."""
    ad = adata.copy()
    ad.var_names = pd.Index([f"Gm{4930000 + i}Rik" for i in range(ad.n_vars)])
    lab = np.asarray(ad.obs["leiden"]).astype(str)
    ad.obs["CellType"] = pd.Categorical([LONG[x] for x in lab], categories=list(LONG.values()))
    ad.obs["SubCellTypeName"] = ad.obs["CellType"].copy()
    t = np.asarray(ad.obsm["X_umap"][:, 0], dtype=float)
    t = t - t.min()
    ad.obs["Lineage1"] = np.where(lab != "D", t, np.nan)
    ad.obs["Lineage2"] = np.where(lab != "A", t, np.nan)
    return ad


def _genes(ad, n=10):
    return list(ad.var_names[:n])


def _inches(fig, bb):
    return bb.transformed(fig.dpi_scale_trans.inverted())


def _saved(fig):
    """Save as a user would -- no bbox_inches -- and hand back the renderer."""
    fig.savefig(io.BytesIO(), format="png", dpi=100)
    return fig.canvas.get_renderer()


def _texts_outside(fig) -> list[tuple[str, tuple[float, ...]]]:
    r = _saved(fig)
    W, H = fig.get_size_inches()
    bad = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = _inches(fig, t.get_window_extent(r))
        if bb.x0 < -TOL or bb.y0 < -TOL or bb.x1 > W + TOL or bb.y1 > H + TOL:
            bad.append((t.get_text(), tuple(round(v, 3) for v in bb.extents)))
    return bad


CASES = {
    "group": lambda ad: scp.pl.group_heatmap(
        ad, _genes(ad), "CellType", layer=None, cell_annotation=["SubCellTypeName"],
        show_row_names=True),
    "group_split_by": lambda ad: scp.pl.group_heatmap(
        ad, _genes(ad), "CellType", split_by="batch", layer=None),
    "group_nsplit_marks": lambda ad: scp.pl.group_heatmap(
        ad, _genes(ad, 30), "CellType", layer=None, n_split=3, split_method="hclust"),
    "group_flip": lambda ad: scp.pl.group_heatmap(
        ad, _genes(ad), "CellType", layer=None, flip=True, show_row_names=True),
    "feature": lambda ad: scp.pl.feature_heatmap(
        ad, _genes(ad), "CellType", max_cells=20, layer=None, cell_annotation=["batch"],
        show_row_names=True),
    "feature_flip": lambda ad: scp.pl.feature_heatmap(
        ad, _genes(ad), "CellType", max_cells=20, layer=None, flip=True),
    "dynamic_two": lambda ad: scp.pl.dynamic_heatmap(
        ad, ["Lineage1", "Lineage2"], _genes(ad), layer=None, show_row_names=True),
    "dynamic_flip": lambda ad: scp.pl.dynamic_heatmap(
        ad, ["Lineage1"], _genes(ad), layer=None, flip=True),
    "cell_cor": lambda ad: scp.pl.cell_cor_heatmap(ad, query_group="CellType"),
    "cell_cor_flip": lambda ad: scp.pl.cell_cor_heatmap(ad, query_group="CellType", flip=True),
    "feature_cor": lambda ad: scp.pl.feature_cor_heatmap(ad, _genes(ad, 12)),
}


@pytest.mark.parametrize("case", list(CASES))
def test_every_text_lies_inside_the_saved_figure(wide, case):
    # column names were clipped at the bottom edge ("Ngn3 low EP" -> "gn3 low EP")
    # and top-track names on the left ("SubCellType" -> "ubCellType")
    res = CASES[case](wide)
    assert _texts_outside(res.fig) == []


@pytest.mark.parametrize("case", list(CASES))
def test_content_keeps_the_margin_on_every_side(wide, case):
    fig = CASES[case](wide).fig
    r = _saved(fig)
    W, H = fig.get_size_inches()
    bb = fig.get_tightbbox(r)
    assert bb.x0 == pytest.approx(MARGIN, abs=TOL)
    assert bb.y0 == pytest.approx(MARGIN, abs=TOL)
    assert W - bb.x1 == pytest.approx(MARGIN, abs=TOL)
    assert H - bb.y1 == pytest.approx(MARGIN, abs=TOL)


@pytest.mark.parametrize("case", ["feature", "dynamic_two"])
def test_bodies_are_not_flush_with_the_bottom(wide, case):
    fig = CASES[case](wide).fig
    r = _saved(fig)
    lowest = min(_inches(fig, ax.get_window_extent(r)).y0 for ax in fig.axes)
    assert lowest >= MARGIN - TOL


def _body_axes(fig):
    return [ax for ax in fig.axes if any(isinstance(c, QuadMesh) for c in ax.get_children())]


def test_explicit_size_is_the_body_size(wide):
    # heatmap_fixsize pins the body at width x height, as SCP does for group_fixed
    res = scp.pl.group_heatmap(wide, _genes(wide), "CellType", layer=None, width=2, height=3)
    r = _saved(res.fig)
    body = max(_body_axes(res.fig), key=lambda ax: ax.get_window_extent(r).height)
    bb = _inches(res.fig, body.get_window_extent(r))
    assert bb.width == pytest.approx(2.0, abs=TOL)
    assert bb.height == pytest.approx(3.0, abs=TOL)


@pytest.fixture(scope="module")
def tall():
    """200 features: far more than R's viewport holds at 0.16 in a row."""
    anndata = pytest.importorskip("anndata")
    rng = np.random.default_rng(3)
    n, g = 200, 200
    lab = pd.Categorical(rng.choice(list("ABCD"), n))
    return anndata.AnnData(
        X=rng.gamma(2.0, 1.0, size=(n, g)).astype(np.float32),
        obs=pd.DataFrame({"leiden": lab}, index=pd.Index([f"c{i}" for i in range(n)])),
        var=pd.DataFrame(index=pd.Index([f"g{i}" for i in range(g)])),
    )


def test_r_fills_the_viewport_whatever_the_feature_count():
    s = json.loads(SIZES.read_text())
    assert s["device_in"] == [7, 7]
    for case in ("group_split30", "group_split200", "feature_split200"):
        assert s[case]["ht_height"] == pytest.approx(VIEWPORT_IN, abs=1e-9)
    assert s["group_default"]["height_sum"] == pytest.approx(VIEWPORT_IN, abs=1e-9)


@pytest.mark.parametrize("fn", ["group", "feature", "dynamic"])
def test_a_defaulted_body_is_capped_at_r_viewport(tall, fn):
    s = json.loads(SIZES.read_text())
    feats = list(tall.var_names)
    if fn == "group":
        res = scp.pl.group_heatmap(tall, feats, "leiden", layer=None)
        r_height = s["group_split200"]["ht_height"]
    elif fn == "feature":
        res = scp.pl.feature_heatmap(tall, feats, "leiden", max_cells=20, layer=None,
                                     show_row_names=False)  # the port defaults to True
        r_height = s["feature_split200"]["ht_height"]
    else:
        tall = tall.copy()
        tall.obs["pt"] = np.linspace(0, 1, tall.n_obs)
        res = scp.pl.dynamic_heatmap(tall, ["pt"], feats, layer=None)
        r_height = s["group_split200"]["ht_height"]
    # titles + annotations + body + names fill R's viewport: the legend column
    # spans that whole stack, so it is the tallest axes
    r = _saved(res.fig)
    grid = max(_inches(res.fig, ax.get_window_extent(r)).height for ax in res.fig.axes)
    assert grid == pytest.approx(r_height, abs=TOL)
    # the margin is ours (plus any label that hangs past the grid's ends)
    assert res.fig.get_size_inches()[1] < r_height + 2 * MARGIN + 0.1
    assert _texts_outside(res.fig) == []


def test_printed_row_names_keep_their_rows(tall):
    # the deliberate divergence: R would overprint 200 names in 6.65 in
    res = scp.pl.group_heatmap(tall, list(tall.var_names), "leiden", layer=None,
                               show_row_names=True)
    r = _saved(res.fig)
    body = max(_body_axes(res.fig), key=lambda ax: ax.get_window_extent(r).height)
    assert _inches(res.fig, body.get_window_extent(r)).height == pytest.approx(0.16 * 200, abs=TOL)


def test_explicit_height_is_never_capped(tall):
    res = scp.pl.group_heatmap(tall, list(tall.var_names), "leiden", layer=None, height=10)
    r = _saved(res.fig)
    body = max(_body_axes(res.fig), key=lambda ax: ax.get_window_extent(r).height)
    assert _inches(res.fig, body.get_window_extent(r)).height == pytest.approx(10.0, abs=TOL)


def test_lineage_lanes_are_what_r_draws(wide):
    # DynamicHeatmap (SCP-plot.R 11721): one 5 mm anno_simple per lineage, dark
    # (#181830) where the feature is dynamic on that lineage, transparent where
    # it is not, named on top, and no legend
    genes = _genes(wide, 8)

    def table(fs, peak):
        return pd.DataFrame({"features": fs, "exp_ncells": 50, "r.sq": 0.5, "dev.expl": 0.5,
                             "peaktime": peak, "valleytime": 0.0, "padjust": 0.01})

    dyn = {"Lineage1": table(genes[:6], np.arange(6.0)),
           "Lineage2": table(genes[3:], np.arange(5.0))}
    res = scp.pl.dynamic_heatmap(wide, ["Lineage1", "Lineage2"], None, layer=None,
                                 dynamic_features=dyn)
    lanes = res.spec.right_tracks
    assert [t.name for t in lanes] == ["Lineage1", "Lineage2"]
    rows = list(res.spec.panels[0].row_labels)
    for t, members in zip(lanes, (genes[:6], genes[3:]), strict=True):
        assert t.size_in == pytest.approx(5 / 25.4)
        assert t.colors == {"0": "#181830", "1": "none"}
        assert list(t.values) == ["0" if f in members else "1" for f in rows]
    assert not any(lg.title.startswith("Lineage") for lg in res.spec.legends)
    assert _texts_outside(res.fig) == []


def test_enrichment_groups_that_match_no_cluster_warn_and_draw_nothing(adata):
    def by_name(features, groups):  # keyed by cell type, not by C1..Cn
        return pd.DataFrame([{"ID": "T1", "Description": "regulation of insulin secretion",
                              "pvalue": 1e-4, "p.adjust": 1e-3, "geneID": "/".join(features),
                              "Count": len(features), "Groups": "Ductal",
                              "Database": "GO_BP"}])

    with pytest.warns(UserWarning, match=r"no GO_BP enrichment group.*\['C1', 'C2'\]"):
        res = scp.pl.group_heatmap(adata, _genes(adata), "leiden", layer=None, n_split=2,
                                   split_method="hclust", anno_terms=True,
                                   enrichment_fn=by_name)
    # heatmap_enrichment builds no annotation at all, as SCP does
    assert [t for t in res.spec.right_tracks if t.name.startswith("terms")] == []
    assert list(res.enrichment["Groups"]) == ["Ductal"]
