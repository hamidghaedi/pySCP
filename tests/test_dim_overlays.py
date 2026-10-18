"""Structural tests for the optional layers of ``cell_dim_plot`` / ``feature_dim_plot``.

R originals: ``CellDimPlot`` and ``FeatureDimPlot`` (``SCP-plot.R:1328``,
``:2042``).  Numbers are checked against SCP in ``test_dim_parity.py``; these
assert what each layer must *be* -- present, at the right depth, on the
right cells, with the right legend -- on the synthetic fixture.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.contour import ContourSet
from matplotlib.image import AxesImage

import scp
from scp.pl import _dim_geom as geom
from scp.pl import _dim_layers as layers


@pytest.fixture(scope="module")
def ov(adata):
    """The session fixture plus a graph, a pseudotime, PAGA and a velocity field."""
    ad = adata.copy()
    rng = np.random.default_rng(1)
    X = ad.obsm["X_umap"]
    d = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(d, axis=1)[:, 1:9]
    rows = np.repeat(np.arange(ad.n_obs), 8)
    g = sp.coo_matrix((np.ones(rows.size), (rows, nn.ravel())), shape=(ad.n_obs, ad.n_obs))
    ad.obsp["connectivities"] = ((g + g.T) > 0).astype(float).tocsr()
    pt = X[:, 0] - X[:, 0].min()
    pt[np.asarray(ad.obs["leiden"]) == "A"] = np.nan
    ad.obs["pt"] = pt
    ad.obsm["velocity_umap"] = rng.normal(size=(ad.n_obs, 2)) * 0.2
    conn = np.array([[0, .5, .1, 0], [.5, 0, .3, .2], [.1, .3, 0, .6], [0, .2, .6, 0]])
    ad.uns["paga"] = {"connectivities": sp.csr_matrix(conn), "groups": "leiden",
                      "transitions_confidence": sp.csr_matrix(conn * 0.5)}
    return ad


def _panel(fig):
    return [a for a in fig.axes if a.get_label() != "<colorbar>"]


def _points_z(ax):
    return max(c.get_zorder() for c in ax.collections if isinstance(c, PathCollection))


# --------------------------------------------------------------------------- #
# cell_dim_plot
# --------------------------------------------------------------------------- #
def test_density_contours_sit_under_the_points(ov):
    fig = scp.pl.cell_dim_plot(ov, "leiden", add_density=True)
    ax = fig.axes[0]
    cs = [a for a in ax.get_children() if isinstance(a, ContourSet)]
    assert len(cs) == 1 and len(cs[0].levels) >= 3
    assert cs[0].get_zorder() < _points_z(ax)


def test_filled_density_is_a_raster_under_the_points(ov):
    fig = scp.pl.cell_dim_plot(ov, "leiden", add_density=True, density_filled=True)
    ax = fig.axes[0]
    img = [a for a in ax.get_children() if isinstance(a, AxesImage)]
    assert len(img) == 1 and img[0].get_array().shape == (100, 100)
    assert img[0].get_zorder() < _points_z(ax)


def test_density_keeps_the_global_limits(ov):
    base = scp.pl.cell_dim_plot(ov, "leiden").axes[0]
    fig = scp.pl.cell_dim_plot(ov, "leiden", add_density=True, density_filled=True,
                               split_by="batch")
    for ax in fig.axes:
        assert ax.get_xlim() == base.get_xlim() and ax.get_ylim() == base.get_ylim()


@pytest.mark.parametrize("kind", ["hull", "ellipse", "rect", "circle"])
def test_marks_enclose_each_group_with_the_expand_margin(ov, kind):
    fig = scp.pl.cell_dim_plot(ov, "leiden", add_mark=True, mark_type=kind)
    fig.canvas.draw()
    ax = fig.axes[0]
    marks = [p for p in ax.patches if isinstance(p, layers._MarkPatch)]
    assert len(marks) == 4
    from matplotlib.path import Path

    px_to_mm = 25.4 / fig.dpi
    for m in marks:
        poly = m.shape_mm(px_to_mm)
        pts = np.round(ax.transData.transform(m._xy) * px_to_mm, 2)
        assert Path(poly).contains_points(pts).all()
        # every point is at least ~expand from the edge (2.5 mm for rounded rect/hull
        # corners, 3 mm elsewhere; allow the polygon sampling error)
        ring = np.vstack([poly, poly[:1]])
        dmin = min(np.sqrt(geom._sq_seg_dist(pts, a, b)).min()
                   for a, b in zip(ring[:-1], ring[1:], strict=True))
        assert dmin > 2.4, dmin
        assert m.get_zorder() < _points_z(ax)
        assert m.get_facecolor()[3] == pytest.approx(0.1)


def test_graph_edges_are_one_linecollection_under_the_points(ov):
    fig = scp.pl.cell_dim_plot(ov, "leiden", graph="connectivities")
    ax = fig.axes[0]
    lc = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(lc) == 1
    nnz_lower = sp.tril(ov.obsp["connectivities"], k=-1).nnz
    assert len(lc[0].get_segments()) == nnz_lower
    assert lc[0].get_zorder() < _points_z(ax)


def test_graph_segments_never_densify():
    # 50k cells, 10 edges each: the dense matrix would be 20 GB
    n = 50_000
    rng = np.random.default_rng(0)
    rows = np.repeat(np.arange(n), 10)
    g = sp.csr_matrix((np.ones(rows.size), (rows, rng.integers(0, n, rows.size))), shape=(n, n))
    r, c, w = geom.graph_segments(g, np.arange(n))
    assert (r >= c).all() and w.size > 0


def test_graph_lower_triangle_follows_drawing_order():
    g = sp.csr_matrix(np.array([[0, 2.0], [0, 0]]))  # one directed entry (0 -> 1)
    assert geom.graph_segments(g, np.array([0, 1]))[2].size == 0  # row drawn first: dropped
    assert geom.graph_segments(g, np.array([1, 0]))[2].size == 1  # row drawn last: kept


def test_missing_graph_raises(ov):
    with pytest.raises(KeyError, match="obsp"):
        scp.pl.cell_dim_plot(ov, "leiden", graph="nope")


def test_hex_replaces_points_and_disables_highlight(ov):
    fig = scp.pl.cell_dim_plot(ov, "leiden", hex=True, cells_highlight=True)
    ax = fig.axes[0]
    assert not [c for c in ax.collections if isinstance(c, PathCollection)]
    polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(polys) == 4  # one per group, binned separately
    assert all(p.get_paths()[0].vertices.shape[0] >= 6 for p in polys)


def test_hex_background_group_draws_last(ov):
    fig = scp.pl.cell_dim_plot(ov, "leiden", hex=True, split_by="batch")
    polys = [c for c in fig.axes[0].collections if isinstance(c, PolyCollection)]
    assert len(polys) == 5  # four groups, then the other split's cells, as ggplot orders them
    from matplotlib.colors import to_hex

    assert to_hex(polys[-1].get_edgecolor()[0]) == scp.palettes.NA_COLOR_DEFAULT.lower()


def test_stat_by_puts_one_inset_per_group_and_a_legend(ov):
    fig = scp.pl.cell_dim_plot(ov, "leiden", stat_by="batch")
    ax = fig.axes[0]
    assert len(ax.child_axes) == 4
    titles = [leg.get_title().get_text() for leg in fig.legends]
    assert titles == ["leiden:", "batch:"]
    for iax in ax.child_axes:
        wedges = [p for p in iax.patches if hasattr(p, "theta1")]
        assert sum(w.theta2 - w.theta1 for w in wedges) == pytest.approx(360)


@pytest.mark.parametrize("ptype", ["ring", "bar", "rose", "trend", "area", "dot"])
@pytest.mark.parametrize("position", ["stack", "dodge"])
def test_stat_by_other_inset_types(ov, ptype, position):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = scp.pl.cell_dim_plot(ov, "leiden", stat_by="batch", stat_plot_type=ptype,
                                   stat_plot_position=position, split_by="batch",
                                   stat_plot_label=True, stat_type="count")
    assert [str(w.message).startswith("stat_plot_type='area'") for w in caught] == \
        [True] * (ptype == "area")
    panels = [ax for ax in fig.axes if ax.child_axes]
    assert len(panels) == 2 and all(len(ax.child_axes) == 4 for ax in panels)
    for iax in panels[0].child_axes:
        assert not iax.axison
        assert iax.name == ("polar" if ptype == "rose" else "rectilinear")
        # every type labels its non-zero counts, which sum to the group size
        labels = [float(t.get_text()) for t in iax.texts if t.get_text()]
        assert labels and all(v > 0 for v in labels)
    # one split's insets show only that split's cells: here, one batch level
    first = [float(t.get_text()) for iax in panels[0].child_axes
             for t in iax.texts if t.get_text()]
    assert sum(first) == (ov.obs["batch"] == ov.obs["batch"].cat.categories[0]).sum()


def test_lineages_draw_a_cased_curve_and_a_legend(ov):
    pytest.importorskip("skmisc")
    fig = scp.pl.cell_dim_plot(ov, "leiden", lineages=["pt"], lineages_whiskers=True)
    ax = fig.axes[0]
    lines = [ln for ln in ax.lines if ln.get_zorder() >= layers.Z_LINEAGE]
    assert len(lines) == 2 and lines[0].get_color() == "white"
    assert lines[0].get_linewidth() > lines[1].get_linewidth()
    assert [lg.get_title().get_text() for lg in fig.legends] == ["leiden:", "Lineages:"]
    xs = lines[1].get_xdata()
    assert len(xs) > 100


def test_missing_lineage_raises(ov):
    with pytest.raises(KeyError, match="lineages"):
        scp.pl.cell_dim_plot(ov, "leiden", lineages=["nope"])


def test_paga_nodes_take_the_cell_colours(ov):
    from matplotlib.colors import to_hex

    fig = scp.pl.cell_dim_plot(ov, "leiden", paga=True, paga_show_transition=True)
    ax = fig.axes[0]
    nodes = [c for c in ax.collections if isinstance(c, PathCollection)][-1]
    assert nodes.get_offsets().shape == (4, 2)
    want = scp.discrete_palette(list(ov.obs["leiden"].cat.categories), palette="Paired")
    assert [to_hex(c) for c in nodes.get_facecolors()] == [v.lower() for v in want.values()]
    assert nodes.get_zorder() > layers.Z_POINTS
    # same groups as the cells: no second, duplicate legend
    assert [lg.get_title().get_text() for lg in fig.legends] == ["leiden:"]


def test_paga_and_velocity_refuse_split_panels(ov):
    with pytest.raises(ValueError, match="non-split"):
        scp.pl.cell_dim_plot(ov, "leiden", paga=True, split_by="batch")
    with pytest.raises(ValueError, match="non-split"):
        scp.pl.cell_dim_plot(ov, "leiden", velocity="velocity", split_by="batch")


def test_raw_velocity_is_one_arrow_per_cell(ov):
    fig = scp.pl.cell_dim_plot(ov, "leiden", velocity="velocity")
    arrows = [a for a in fig.axes[0].artists if isinstance(a, layers._ArrowField)]
    assert len(arrows) == 1 and arrows[0]._seg.shape == (ov.n_obs, 4)
    fig.canvas.draw()  # the heads are built at draw time


def test_raw_velocity_density_subsamples(ov):
    fig = scp.pl.cell_dim_plot(ov, "leiden", velocity="velocity", velocity_density=0.25)
    arrows = [a for a in fig.axes[0].artists if isinstance(a, layers._ArrowField)]
    assert arrows[0]._seg.shape[0] == int(np.ceil(0.25 * ov.n_obs))


def test_missing_velocity_embedding_names_scvelo(ov):
    with pytest.raises(KeyError, match="velocity_embedding"):
        scp.pl.cell_dim_plot(ov, "leiden", velocity="stochastic")


def _fake_grid(X, V, *, density, smooth, n_neighbors, min_mass, cutoff_perc, adjust_for_stream):
    """A precomputed grid in ``compute_velocity_on_grid``'s two return shapes."""
    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
    ys = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
    gx, gy = np.meshgrid(xs, ys)
    u, v = -(gy - gy.mean()), gx - gx.mean()
    if adjust_for_stream:
        u[:3, :3] = np.nan  # low-mass points are masked in u only, as in R
        return np.stack([xs, ys]), np.stack([u, v])
    return np.column_stack([gx.ravel(), gy.ravel()]), np.column_stack([u.ravel(), v.ravel()])


def test_grid_velocity_scales_the_grid_field(ov, monkeypatch):
    monkeypatch.setattr(layers, "_velocity_grid", _fake_grid)
    fig = scp.pl.cell_dim_plot(ov, "leiden", velocity="velocity", velocity_plot_type="grid",
                               velocity_scale=2.0)
    arrows = [a for a in fig.axes[0].artists if isinstance(a, layers._ArrowField)][0]
    gx, gv = _fake_grid(ov.obsm["X_umap"], None, density=1, smooth=0.5, n_neighbors=1,
                        min_mass=1, cutoff_perc=5, adjust_for_stream=False)
    assert np.allclose(arrows._seg[:, 2:] - arrows._seg[:, :2], 2.0 * gv)


def test_stream_velocity_gets_a_velocity_colourbar(ov, monkeypatch):
    monkeypatch.setattr(layers, "_velocity_grid", _fake_grid)
    fig = scp.pl.cell_dim_plot(ov, "leiden", velocity="velocity", velocity_plot_type="stream")
    cbars = [a for a in fig.axes if a.get_label() == "<colorbar>"]
    assert len(cbars) == 1 and cbars[0].get_ylabel() == "Velocity"
    fig.savefig(__import__("io").BytesIO(), format="png")


@pytest.mark.parametrize("plot_type", ["grid", "stream"])
def test_grid_velocity_needs_no_scvelo(ov, monkeypatch, plot_type):
    """The overlay's grid is the numpy port of SCP's function: no optional dependency.

    ``sys.modules["scvelo"] = None`` makes any ``import scvelo`` raise, so this
    holds whether or not scvelo happens to be installed.
    """
    monkeypatch.setitem(__import__("sys").modules, "scvelo", None)
    fig = scp.pl.cell_dim_plot(ov, "leiden", velocity="velocity", velocity_plot_type=plot_type)
    fig.savefig(__import__("io").BytesIO(), format="png")
    if plot_type == "grid":
        arrows = [a for a in fig.axes[0].artists if isinstance(a, layers._ArrowField)][0]
        gx, gv = scp.pl.compute_velocity_on_grid(ov.obsm["X_umap"], ov.obsm["velocity_umap"])
        np.testing.assert_allclose(arrows._seg[:, :2], gx, rtol=0, atol=1e-12)
        np.testing.assert_allclose(arrows._seg[:, 2:] - arrows._seg[:, :2], gv, atol=1e-12)


def test_cells_highlight_validation(ov):
    with pytest.raises(ValueError, match="No cells"):
        scp.pl.cell_dim_plot(ov, "leiden", cells_highlight=["nope"])
    with pytest.warns(UserWarning, match="not found"):
        scp.pl.cell_dim_plot(ov, "leiden", cells_highlight=["c0", "nope"])


def test_cells_highlight_draws_stroke_then_fill(ov):
    fig = scp.pl.cell_dim_plot(ov, "leiden", cells_highlight=list(ov.obs_names[:10]),
                               alpha_highlight=0.5)
    hl = [c for c in fig.axes[0].collections
          if isinstance(c, PathCollection) and c.get_zorder() > layers.Z_POINTS]
    assert len(hl) == 2 and all(len(c.get_offsets()) == 10 for c in hl)
    assert hl[0].get_sizes()[0] > hl[1].get_sizes()[0]
    assert hl[0].get_alpha() == 0.5


def test_label_repel_moves_labels_and_draws_leaders(ov):
    ad = ov.copy()
    ad.obsm["X_umap"] = ad.obsm["X_umap"] * 0.01  # crowd the centres together
    fig = scp.pl.cell_dim_plot(ad, "leiden", label=True, label_repel=True)
    texts = [t for t in fig.axes[0].texts if t.get_text() in {"1", "2", "3", "4"}]
    assert len(texts) == 4
    offsets = np.array([t.xyann for t in texts])
    assert np.abs(offsets).max() > 1


def test_overlay_legends_stack_after_the_group_legend(ov):
    pytest.importorskip("skmisc")
    ad = ov.copy()
    ad.obs["coarse"] = pd.Categorical(np.where(np.isin(ad.obs["leiden"], ["A", "B"]), "x", "y"))
    paga = {"connectivities": np.array([[0, .4], [.4, 0]]), "groups": "coarse"}
    fig = scp.pl.cell_dim_plot(ad, "leiden", stat_by="batch", lineages=["pt"], paga=paga)
    assert [lg.get_title().get_text() for lg in fig.legends] == \
        ["leiden:", "batch:", "Lineages:", "coarse:"]


# --------------------------------------------------------------------------- #
# feature_dim_plot
# --------------------------------------------------------------------------- #
def test_feature_density_and_graph_draw_under_points(ov):
    fig = scp.pl.feature_dim_plot(ov, "g0", add_density=True, graph="connectivities")
    ax = _panel(fig)[0]
    assert [a for a in ax.get_children() if isinstance(a, ContourSet)]
    assert [c for c in ax.collections if isinstance(c, LineCollection)]
    # the value scatter stays the last collection, so draw order tests still hold
    assert isinstance(ax.collections[-1], PathCollection)


def test_feature_graph_only_joins_cells_of_the_split(ov):
    fig = scp.pl.feature_dim_plot(ov, "g0", split_by="batch", graph="connectivities",
                                  bg_cutoff=None)
    b = np.asarray(ov.obs["batch"])
    total = 0
    for ax in _panel(fig):
        total += len([c for c in ax.collections if isinstance(c, LineCollection)][0].get_segments())
    g = sp.tril(ov.obsp["connectivities"], k=-1).tocoo()
    assert total == int((b[g.row] == b[g.col]).sum())


def test_feature_hex_is_mean_value_hexagons(ov):
    fig = scp.pl.feature_dim_plot(ov, "g0", hex=True)
    ax = _panel(fig)[0]
    polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(polys) >= 1
    arr = polys[-1].get_array()
    assert arr is not None and arr.size == len(polys[-1].get_paths())


def test_feature_lineages_use_every_cell(ov):
    pytest.importorskip("skmisc")
    a = scp.pl.feature_dim_plot(ov, "g0", lineages=["pt"])
    b = scp.pl.feature_dim_plot(ov, "g0", lineages=["pt"], cells=list(ov.obs_names[:200]))
    la = [ln for ln in _panel(a)[0].lines if ln.get_color() != "white"][0]
    lb = [ln for ln in _panel(b)[0].lines if ln.get_color() != "white"][0]
    assert np.allclose(la.get_xdata(), lb.get_xdata())
    assert "Lineages:" in [lg.get_title().get_text() for lg in a.legends]


def test_calculate_coexp_appends_a_panel(ov):
    fig = scp.pl.feature_dim_plot(ov, ["g0", "g1"], calculate_coexp=True)
    panels = _panel(fig)
    assert len(panels) == 3 and panels[-1].get_title().endswith("CoExp")


def test_calculate_coexp_drops_non_genes_with_a_warning(ov):
    with pytest.warns(UserWarning, match="score"):
        fig = scp.pl.feature_dim_plot(ov, ["g0", "g1", "score"], calculate_coexp=True,
                                      bg_cutoff=None)
    assert len(_panel(fig)) == 4


def test_calculate_coexp_joins_the_blend(ov):
    fig = scp.pl.feature_dim_plot(ov, ["g0", "g1"], calculate_coexp=True, compare_features=True)
    assert _panel(fig)[0].get_title().endswith("g0|g1|CoExp")


def test_calculate_coexp_without_genes_warns(ov):
    with pytest.warns(UserWarning, match="gene features"):
        fig = scp.pl.feature_dim_plot(ov, ["score"], calculate_coexp=True, bg_cutoff=None)
    assert len(_panel(fig)) == 1


def test_feature_highlight_draws_stroke_then_value(ov):
    fig = scp.pl.feature_dim_plot(ov, "g0", cells_highlight=list(ov.obs_names[:5]))
    hl = [c for c in _panel(fig)[0].collections if c.get_zorder() > layers.Z_POINTS]
    assert len(hl) == 2 and hl[1].get_array() is not None


def test_feature_label_marks_the_high_band(ov):
    fig = scp.pl.feature_dim_plot(ov, "g0", label=True)
    texts = [t for t in _panel(fig)[0].texts if t.get_text() == "g0"]
    assert len(texts) == 1


def test_keep_scale_feature_pools_every_cell_even_with_cells(ov):
    """R pools over dat_exp, built before `cells` subsets."""
    full = scp.pl.feature_dim_plot(ov, "g0")
    sub = scp.pl.feature_dim_plot(ov, "g0", cells=list(ov.obs_names[:100]))

    def lim(f):
        return [a for a in f.axes if a.get_label() == "<colorbar>"][0].get_ylim()

    assert lim(full) == pytest.approx(lim(sub))


def test_feature_dim_plot_3d_coexp_adds_a_trace(ov):
    pytest.importorskip("plotly")
    ad = ov.copy()
    ad.obsm["X_umap3"] = np.column_stack([ad.obsm["X_umap"], np.zeros(ad.n_obs)])
    fig = scp.pl.feature_dim_plot_3d(ad, ["g0", "g1"], reduction="X_umap3",
                                     calculate_coexp=True)
    assert [t.name for t in fig.data] == ["g0", "g1", "CoExp"]


# --------------------------------------------------------------------------- #
# pure geometry
# --------------------------------------------------------------------------- #
def test_concave_hull_contains_every_point_and_is_simple():
    from matplotlib.path import Path

    rng = np.random.default_rng(3)
    t = rng.uniform(0, 1.5 * np.pi, 400)
    r = rng.uniform(4, 5, 400)
    pts = np.column_stack([r * np.cos(t), r * np.sin(t)])  # a C shape
    hull = geom.concave_hull(pts, concavity=2)
    assert len(hull) > 10
    big = geom.offset_shape(hull, 0.01, 0.0, convex=False)
    assert Path(big).contains_points(pts, radius=1e-9).all()
    # concave: the C's mouth stays open
    assert not Path(hull).contains_point((4.5 * np.cos(1.75 * np.pi), 4.5 * np.sin(1.75 * np.pi)))
    from scipy.spatial import ConvexHull

    assert abs(geom._signed_area(hull)) < ConvexHull(pts).volume


def test_round_offset_of_a_square_is_a_rounded_square():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    out = geom.offset_shape(sq, 3.0, 2.5, convex=False)
    exact = geom.offset_shape(sq, 3.0, 2.5, convex=True)
    # mitre 0.5 then round 2.5: extent 13.5 along each axis edge
    assert out[:, 0].max() == pytest.approx(13.0, abs=0.05)
    assert abs(geom._signed_area(out)) == pytest.approx(abs(geom._signed_area(exact)), rel=5e-3)


def test_hexagon_offsets_tile():
    off = geom.hexagon_offsets((1.0, 2.0))
    assert off[:, 0].max() - off[:, 0].min() == pytest.approx(1.0)  # flat-to-flat = width
    radius = off[:, 1].max()
    # rows sit height * sqrt(3) / 2 apart, which is 1.5 radii: pointy tops interlock
    assert 2.0 * np.sqrt(3) / 2 == pytest.approx(1.5 * radius)


def test_rescale_constant_maps_to_the_middle():
    assert np.allclose(geom.rescale(np.array([2.0, 2.0]), (0.05, 0.5)), 0.275)
