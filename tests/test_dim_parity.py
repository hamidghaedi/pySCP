"""The dim-plot overlays against numbers dumped from SCP 0.5.6.

Fixtures come from ``tools/parity/dim_overlays.R`` on SCP's ``pancreas_sub``;
``inputs.csv`` carries the coordinates, groups and genes those calls read, so
nothing here needs R or the parity h5ad.  Each test compares an intermediate
-- a KDE grid, hexagon counts, edge list, fitted curve, inset box -- rather
than pixels.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import scp
from scp.pl import _dim_geom as geom
from scp.pl import _dim_layers as layers

FIX = Path(__file__).parent / "fixtures" / "parity" / "dim"
MM = 72.27 / 25.4 * 0.75  # ggplot linewidth mm -> matplotlib pt


def _csv(name):
    return pd.read_csv(FIX / name)


@pytest.fixture(scope="module")
def panc():
    anndata = pytest.importorskip("anndata")
    inp = _csv("inputs.csv").set_index("cell")
    lev = json.loads((FIX / "levels.json").read_text())
    genes = ["Tpt1", "Rps2", "Rpl8", "Ins1"]
    obs = pd.DataFrame(index=inp.index.astype(str))
    for c in ("CellType", "SubCellType", "Phase"):
        obs[c] = pd.Categorical(inp[c], categories=lev[c])
    obs["S_score"] = inp["S_score"].to_numpy()
    ad = anndata.AnnData(X=inp[genes].to_numpy(dtype=float), obs=obs,
                         var=pd.DataFrame(index=genes))
    ad.obsm["X_umap"] = inp[["UMAP_1", "UMAP_2"]].to_numpy(dtype=float)
    return ad


def _xy(ad):
    return ad.obsm["X_umap"][:, 0], ad.obsm["X_umap"][:, 1]


def _lims(ad):
    x, y = _xy(ad)
    return (x.min(), x.max(), y.min(), y.max())


# --------------------------------------------------------------------------- #
def test_r_pretty_matches_known_r_output():
    # values from R: pretty(c(lo, hi), 10)
    assert np.allclose(geom.r_pretty(0.0, 0.0093, 10), np.arange(0, 0.0101, 0.001))
    assert np.allclose(geom.r_pretty(1.3, 97.2, 10), np.arange(0, 101, 10))
    assert np.allclose(geom.r_pretty(-3.2, 4.7, 10), np.arange(-4, 5.1, 1))


def test_density_grid_and_levels_match_r(panc):
    x, y = _xy(panc)
    gx, gy, z = geom.kde2d(x, y, lims=_lims(panc), n=100)
    r = _csv("density_raster.csv")
    ix = np.abs(gx[None, :] - r["x"].to_numpy()[:, None]).argmin(axis=1)
    iy = np.abs(gy[None, :] - r["y"].to_numpy()[:, None]).argmin(axis=1)
    rel = np.abs(z[ix, iy] - r["density"]) / r["density"].abs().max()
    assert rel.max() < 1e-6, rel.max()

    want = json.loads((FIX / "density_levels.json").read_text())["levels"]
    br = geom.density_breaks(z)
    assert np.allclose(br[(br > z.min()) & (br < z.max())], want)

    # every R contour vertex lies on our surface at its own level
    from scipy.interpolate import RegularGridInterpolator

    c = _csv("density_contour.csv")
    at = RegularGridInterpolator((gx, gy), z)(c[["x", "y"]].to_numpy())
    assert np.abs(at - c["level"]).max() < 2e-6


def test_density_layer_draws_those_levels(panc):
    from matplotlib.contour import ContourSet

    fig = scp.pl.cell_dim_plot(panc, "CellType", add_density=True)
    cs = [a for a in fig.axes[0].get_children() if isinstance(a, ContourSet)]
    want = json.loads((FIX / "density_levels.json").read_text())["levels"]
    assert len(cs) == 1 and np.allclose(cs[0].levels, want)


def test_hex_bins_match_ggplot(panc):
    x, y = _xy(panc)
    xl, yl = _lims(panc)[:2], _lims(panc)[2:]
    bw = ((xl[1] - xl[0]) / 50, (yl[1] - yl[0]) / 50)
    r = _csv("hex_cell.csv")
    ct = np.asarray(panc.obs["CellType"])
    tabs = []
    for gi, lv in enumerate(panc.obs["CellType"].cat.categories, start=1):
        m = ct == lv
        hb = geom.hex_bin(x[m], y[m], bw)
        hb["group"] = gi
        tabs.append(hb)
    ours = pd.concat(tabs)
    # the alpha scale sees every hexagon; the x/y scale limits then censor the
    # ones centred outside, which R reports as NA coordinates
    alpha = geom.rescale(ours["value"].to_numpy(), (0.1, 1.0))
    assert np.allclose(np.sort(alpha), np.sort(r["alpha"]), atol=1e-7)
    inside = layers._in_limits(ours, xl, yl)
    assert (~inside).sum() == r[["x", "y"]].isna().any(axis=1).sum()
    ours, r = ours[inside], r.dropna()
    key = ["group", "x", "y"]
    a = ours.assign(x=ours.x.round(5), y=ours.y.round(5)).set_index(key)["value"].sort_index()
    b = r.assign(x=r.x.round(5), y=r.y.round(5)).set_index(key)["count"].sort_index()
    assert len(a) == len(b) and (a.index == b.index).all()
    assert (a.to_numpy() == b.to_numpy()).all()


def test_feature_hex_is_the_mean_clipped_value(panc):
    x, y = _xy(panc)
    v = np.asarray(panc[:, "Ins1"].X).ravel().astype(float)
    v[v <= 0] = np.nan
    fin = v[np.isfinite(v)]
    hi = np.quantile(fin, 0.99) + 0.001
    v = np.clip(v, fin.min(), hi)
    xl, yl = _lims(panc)[:2], _lims(panc)[2:]
    bw = ((xl[1] - xl[0]) / 50, (yl[1] - yl[0]) / 50)
    ok = np.isfinite(v)
    hv = geom.hex_bin(x[ok], y[ok], bw, z=v[ok], fun="mean")
    hv = hv[layers._in_limits(hv, xl, yl)]
    r = _csv("hex_feature.csv").dropna()
    a = hv.assign(x=hv.x.round(5), y=hv.y.round(5)).set_index(["x", "y"])["value"].sort_index()
    b = r.assign(x=r.x.round(5), y=r.y.round(5)).set_index(["x", "y"])["value"].sort_index()
    assert (a.index == b.index).all()
    assert np.allclose(a.to_numpy(), b.to_numpy(), rtol=1e-7)
    bg = geom.hex_bin(x[~ok], y[~ok], bw)
    bg = bg[layers._in_limits(bg, xl, yl)]
    rb = _csv("hex_feature_bg.csv").dropna()
    assert len(bg) == len(rb) and bg["value"].sum() == rb["count"].sum()


def _graph_adata(panc):
    cells = (FIX / "graph_cells.txt").read_text().split()
    t = _csv("graph_triplets.csv")
    pos = panc.obs_names.get_indexer(cells)
    g = sp.coo_matrix((t["x"], (pos[t["i"] - 1], pos[t["j"] - 1])),
                      shape=(panc.n_obs, panc.n_obs)).tocsr()
    ad = panc.copy()
    ad.obsp["RNA_snn"] = g
    return ad, cells


def _undirected(x0, y0, x1, y1, w):
    a = np.round(np.column_stack([x0, y0]), 5)
    b = np.round(np.column_stack([x1, y1]), 5)
    swap = (a[:, 0] > b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1]))
    lo = np.where(swap[:, None], b, a)
    hi = np.where(swap[:, None], a, b)
    df = pd.DataFrame({"x0": lo[:, 0], "y0": lo[:, 1], "x1": hi[:, 0], "y1": hi[:, 1],
                       "w": np.round(w, 5)})
    df = df[(df.x0 != df.x1) | (df.y0 != df.y1)]
    return df.sort_values(list(df.columns)).reset_index(drop=True)


def test_graph_edges_match_r(panc):
    from matplotlib.collections import LineCollection

    ad, cells = _graph_adata(panc)
    fig = scp.pl.cell_dim_plot(ad, "CellType", cells=cells, graph="RNA_snn")
    lc = [c for c in fig.axes[0].collections if isinstance(c, LineCollection)][0]
    segs = np.asarray(lc.get_segments())
    w = np.asarray(lc.get_linewidths()) / MM
    ours = _undirected(segs[:, 0, 0], segs[:, 0, 1], segs[:, 1, 0], segs[:, 1, 1], w)
    r = _csv("graph_segments.csv")
    theirs = _undirected(r.x, r.y, r.xend, r.yend, r.linewidth)
    assert len(ours) == len(theirs)
    pd.testing.assert_frame_equal(ours, theirs, atol=2e-5)


def test_lineage_curves_match_loess(panc):
    pytest.importorskip("skmisc")
    x, y = _xy(panc)
    ct = np.asarray(panc.obs["CellType"]).astype(str)
    pts = {
        "Lineage1": np.where(ct != "Ductal", x - x.min(), np.nan),
        "Lineage2": np.where(np.isin(ct, ["Ductal", "Ngn3 low EP", "Ngn3 high EP"]),
                             y - y.min(), np.nan),
    }
    curves = layers.fit_lineages(panc.obsm["X_umap"], pts, trim=(0.01, 0.99), span=0.75)
    r = _csv("lineage_paths.csv")
    for lin, col in (("Lineage1", "#1B9E77"), ("Lineage2", "#D95F02")):
        want = r[r.colour == col][["x", "y"]].to_numpy()
        got = curves[lin][["x", "y"]].to_numpy()
        assert got.shape == want.shape
        assert np.abs(got - want).max() < 1e-6, np.abs(got - want).max()


def test_paga_nodes_and_edges_match_r(panc):
    from matplotlib.collections import LineCollection, PathCollection

    conn = np.array(json.loads((FIX / "paga_input.json").read_text())["connectivities"])
    paga = {"connectivities": conn, "groups": "CellType"}
    fig = scp.pl.cell_dim_plot(panc, "SubCellType", paga=paga)
    ax = fig.axes[0]
    nodes = [c for c in ax.collections if isinstance(c, PathCollection)][-1]
    rn = _csv("paga_nodes.csv")
    assert np.allclose(nodes.get_offsets(), rn[["x", "y"]].to_numpy(), atol=1e-6)
    from matplotlib.colors import to_hex

    assert [to_hex(c).upper() for c in nodes.get_facecolors()] == list(rn["colour"].str.upper())
    edges = [c for c in ax.collections if isinstance(c, LineCollection)][-1]
    segs = np.asarray(edges.get_segments())
    w = np.asarray(edges.get_linewidths()) / MM
    ours = _undirected(segs[:, 0, 0], segs[:, 0, 1], segs[:, 1, 0], segs[:, 1, 1], w)
    re = _csv("paga_edges.csv")
    theirs = _undirected(re.x, re.y, re.xend, re.yend, re.linewidth)
    pd.testing.assert_frame_equal(ours, theirs, atol=2e-5)


def test_raw_velocity_arrows_match_r(panc):
    ad = panc.copy()
    x, y = _xy(ad)
    ad.obsm["stochastic_umap"] = 0.1 * np.column_stack([-(y - y.mean()), x - x.mean()])
    fig = scp.pl.cell_dim_plot(ad, "CellType", velocity="stochastic")
    field = [a for a in fig.axes[0].artists if isinstance(a, layers._ArrowField)][0]
    got = pd.DataFrame(field._seg, columns=["x", "y", "xend", "yend"]).round(6)
    want = _csv("velocity_raw.csv").round(6)
    key = ["x", "y"]
    pd.testing.assert_frame_equal(got.sort_values(key).reset_index(drop=True),
                                  want.sort_values(key).reset_index(drop=True), atol=2e-6)


def test_stat_pies_sit_where_r_puts_them(panc):
    fig = scp.pl.cell_dim_plot(panc, "CellType", stat_by="Phase")
    ax = fig.axes[0]
    rc = _csv("stat_pie_centres.csv").set_index("label")
    rv = _csv("stat_pie_values.csv")
    insets = ax.child_axes
    assert len(insets) == len(rc)
    for iax, (grp, row) in zip(insets, rc.iterrows(), strict=True):
        x0, y0, w, h = iax.get_axes_locator()._bounds
        assert np.isclose(x0 + w / 2, row.x, atol=1e-6) and np.isclose(y0 + h / 2, row.y, atol=1e-6)
        assert np.isclose(w, row.x_range * 0.15, rtol=1e-6)
        assert np.isclose(h, row.y_range * 0.15, rtol=1e-6)
        wedges = [p for p in iax.patches if hasattr(p, "theta1")]
        frac = [(p.theta2 - p.theta1) / 360 for p in wedges]
        want = rv[rv.group == grp].set_index("stat").loc[["G1", "S", "G2M"], "value"]
        assert np.allclose(frac, want.to_numpy(), atol=1e-7)


def _insets(panc, ptype, position="stack", stat_type="percent", **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # area's "no area" warning
        fig = scp.pl.cell_dim_plot(panc, "CellType", stat_by="Phase", stat_plot_type=ptype,
                                   stat_plot_position=position, stat_type=stat_type, **kw)
    return fig.axes[0].child_axes


def _norm(v, lim):
    return (np.asarray(v, dtype=float) - lim[0]) / (lim[1] - lim[0])


def _r_rows(name, ptype, position, stat_type, group):
    r = _csv(name)
    return r[(r.type == ptype) & (r.position == position) & (r.stat_type == stat_type)
             & (r.group == group)]


@pytest.mark.parametrize("ptype", ["pie", "ring", "bar", "rose", "trend", "area", "dot"])
def test_every_inset_type_sits_in_rs_box(panc, ptype):
    # the boxes CellDimPlot hands annotation_custom, traced in R per type
    want = _csv("stat_inset_boxes.csv").query("type == @ptype")
    insets = _insets(panc, ptype, stat_plot_size=0.2)
    assert len(insets) == len(want) == 5
    for iax, (_, row) in zip(insets, want.iterrows(), strict=True):
        x0, y0, w, h = iax.get_axes_locator()._bounds
        assert np.allclose([x0, x0 + w, y0, y0 + h],
                           row[["xmin", "xmax", "ymin", "ymax"]].to_numpy(float), atol=1e-6)


@pytest.mark.parametrize("ptype,position,stat_type", [
    ("bar", "stack", "percent"), ("bar", "stack", "count"), ("bar", "dodge", "count"),
    ("trend", "stack", "percent"),
])
def test_inset_bars_match_rs_panel(panc, ptype, position, stat_type):
    from matplotlib.colors import to_hex
    from matplotlib.patches import Rectangle

    levels = panc.obs["CellType"].cat.categories
    for grp, iax in zip(levels, _insets(panc, ptype, position, stat_type), strict=True):
        assert not iax.axison  # theme_void
        r = _r_rows("stat_inset_marks.csv", ptype, position, stat_type, grp)
        r = r.assign(fill=r.fill.str.lower()).set_index("fill")
        bars = [p for p in iax.patches if isinstance(p, Rectangle)]
        assert len(bars) == len(r)
        xl, yl = iax.get_xlim(), iax.get_ylim()
        for b in bars:
            want = r.loc[to_hex(b.get_facecolor())]
            x0, x1 = _norm([b.get_x(), b.get_x() + b.get_width()], xl)
            y0, y1 = _norm([b.get_y(), b.get_y() + b.get_height()], yl)
            assert np.allclose([y0, y1], want[["y0", "y1"]].to_numpy(float), atol=1e-7)
            assert np.isclose((x0 + x1) / 2, (want.x0 + want.x1) / 2, atol=1e-7)
            # stacked bars fill R's expand = 0 range; dodged ones keep
            # stat_plot's 0.95 of the slot where position_dodge2 leaves 0.9
            k = 1 if position == "stack" else 0.95 / 0.9
            assert np.isclose(x1 - x0, (want.x1 - want.x0) * k, rtol=1e-6)


def test_inset_dots_match_rs_panel(panc):
    from matplotlib.colors import to_hex

    levels = panc.obs["CellType"].cat.categories
    for grp, iax in zip(levels, _insets(panc, "dot", stat_type="count"), strict=True):
        r = _r_rows("stat_inset_marks.csv", "dot", "stack", "count", grp)
        r = r.assign(fill=r.fill.str.lower()).set_index("fill")
        assert len(iax.collections) == len(r)
        for pc in iax.collections:
            want = r.loc[to_hex(pc.get_facecolor()[0])]
            (x, y), = pc.get_offsets()
            assert np.isclose(_norm(x, iax.get_xlim()), want.x0, atol=1e-7)
            assert np.isclose(_norm(y, iax.get_ylim()), want.y0, atol=1e-7)
            # scale_size_area(max_size = 12): the same 12 * sqrt(v / max)
            assert np.isclose(np.sqrt(pc.get_sizes()[0]), want["size"], rtol=1e-6)


def test_inset_polar_marks_match_rs_panel(panc):
    from matplotlib.patches import Rectangle

    levels = panc.obs["CellType"].cat.categories
    marks = {t: _insets(panc, t, stat_type=st)
             for t, st in [("rose", "percent"), ("pie", "percent"), ("ring", "count")]}
    for i, grp in enumerate(levels):
        # rose: coord_polar puts the radius range on 0..0.4 of the box, so each
        # ring is 0.8 x its normalised thickness of the inset's half-width
        r = _r_rows("stat_inset_marks.csv", "rose", "stack", "percent", grp)
        iax = marks["rose"][i]
        top = iax.get_ylim()[1]
        bars = [p for p in iax.patches if isinstance(p, Rectangle)]
        got = sorted(b.get_height() / top for b in bars)
        assert np.allclose(got, sorted(0.8 * (r.y1 - r.y0)), atol=1e-7)
        assert np.isclose(max((b.get_y() + b.get_height()) / top for b in bars), 0.8)
        # pie and ring: theta is the stacked value, the radius R's bar extent
        for t, st in [("pie", "percent"), ("ring", "count")]:
            r = _r_rows("stat_inset_marks.csv", t, "stack", st, grp)
            wedges = [p for p in marks[t][i].patches if hasattr(p, "theta1")]
            assert np.allclose(sorted((w.theta2 - w.theta1) / 360 for w in wedges),
                               sorted(r.y1 - r.y0), atol=1e-7)
            for w in wedges:  # outer edge at 1; R's ring hole is 0.6 / 1.4
                inner = 0.0 if w.width is None else 1 - w.width
                assert np.allclose([inner, 1.0], [r.x0.iloc[0], r.x1.iloc[0]], atol=1e-7)


def test_area_insets_are_empty_like_r(panc):
    r = _csv("stat_inset_marks.csv").query("type == 'area'")
    assert r.fill.isna().all() and len(r) == 5  # ggplot built no rows at all
    with pytest.warns(UserWarning, match="no area"):
        fig = scp.pl.cell_dim_plot(panc, "CellType", stat_by="Phase", stat_plot_type="area")
    for iax in fig.axes[0].child_axes:
        assert not iax.patches and not iax.collections and not iax.texts
    assert all(not iax.patches for iax in _insets(panc, "area", stat_plot_label=True))


@pytest.mark.parametrize("ptype,position,stat_type", [
    ("bar", "stack", "percent"), ("bar", "stack", "count"), ("bar", "dodge", "count"),
    ("trend", "stack", "percent"), ("area", "stack", "percent"), ("dot", "stack", "count"),
    ("rose", "stack", "percent"), ("pie", "stack", "percent"), ("ring", "stack", "count"),
])
def test_inset_labels_match_r(panc, ptype, position, stat_type):
    levels = panc.obs["CellType"].cat.categories
    insets = _insets(panc, ptype, position, stat_type, stat_plot_label=True)
    for grp, iax in zip(levels, insets, strict=True):
        r = _r_rows("stat_inset_labels.csv", ptype, position, stat_type, grp)
        r = r[~r.label.isin(["0", "0%"])]  # R also prints zeros; the port does not
        texts = [t for t in iax.texts if t.get_text()]  # ax.pie adds empty ones
        assert sorted(t.get_text() for t in texts) == sorted(r.label)
        if ptype == "rose":
            continue  # stat_plot's rose stacks the other way round (docstring)
        got, want = [], []
        for t in texts:
            x, y = t.get_position()
            if ptype in ("pie", "ring"):
                # ax.pie's unit radius; anticlockwise from 12 o'clock is
                # 1 - R's clockwise theta
                frac = (np.degrees(np.arctan2(y, x)) - 90) % 360 / 360
                got.append((t.get_text(), np.hypot(x, y), 1 - frac))
            else:
                got.append((t.get_text(), _norm(x, iax.get_xlim()), _norm(y, iax.get_ylim())))
        for _, row in r.iterrows():
            # R's x range is zero-wide for area, where the label is centred
            want.append((row.label, 0.5 if np.isnan(row.x) else row.x, row.y))
        got, want = sorted(got), sorted(want)
        assert [g[0] for g in got] == [w[0] for w in want]
        assert np.allclose([g[1:] for g in got], [w[1:] for w in want], atol=1e-7)


def test_mark_enclosures_match_ggforce(panc):
    x, y = _xy(panc)
    mx, my = np.round(x * 3 + 60, 2), np.round(y * 3 + 60, 2)
    ct = np.asarray(panc.obs["CellType"])
    re, rc = _csv("mark_ellipse.csv"), _csv("mark_circle.csv")
    flips = 0
    for k, lv in enumerate(panc.obs["CellType"].cat.categories):
        m = ct == lv
        x0, y0, r = geom.enclose_circle(mx[m], my[m])
        assert np.allclose([x0, y0, r], rc.iloc[k][["x0", "y0", "r"]], atol=1e-6)
        ex = geom.enclose_ellipse(mx[m], my[m], tol=0.01)
        want = re.iloc[k]
        assert np.allclose(ex[:4], want[["x0", "y0", "a", "b"]], atol=1e-6)
        # ggforce's angle is acos() of the axis' x-component, so it loses the
        # sign of the y-component: equal to ours, or ours mirrored
        same = np.isclose(np.cos(2 * (ex[4] - want["angle"])), 1, atol=1e-9)
        mirrored = np.isclose(np.cos(2 * (ex[4] + want["angle"])), 1, atol=1e-9)
        assert same or mirrored
        flips += int(mirrored and not same)
        d = np.column_stack([mx[m] - ex[0], my[m] - ex[1]])
        u = d @ [np.cos(ex[4]), np.sin(ex[4])]
        v = d @ [-np.sin(ex[4]), np.cos(ex[4])]
        assert ((u / ex[2]) ** 2 + (v / ex[3]) ** 2).max() < 1.1  # ours encloses
    assert flips == 4  # pinned: ggforce mirrors four of the five CellType ellipses


def test_coexp_panel_matches_r(panc):
    with pytest.warns(UserWarning, match="S_score"):
        fig = scp.pl.feature_dim_plot(panc, ["Tpt1", "Rps2", "Rpl8", "S_score"],
                                      calculate_coexp=True)
    panels = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    assert len(panels) == 5 and panels[-1].get_title().endswith("CoExp")
    r = _csv("coexp.csv").set_index("cell")
    co = scp.fetch.calculate_coexp(panc, ["Tpt1", "Rps2", "Rpl8"])
    assert np.allclose(co.loc[r.index].to_numpy(), r["CoExp"].to_numpy(), rtol=1e-8)
    drawn = np.sort(np.asarray(panels[-1].collections[-1].get_array(), dtype=float))
    assert np.allclose(drawn, np.sort(r["value"].dropna().to_numpy()), rtol=1e-8)
