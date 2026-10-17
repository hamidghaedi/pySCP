"""Numeric parity of the correlation figures against SCP's R output.

The fixtures under ``tests/fixtures/parity/audit/cor`` are written by
``tools/parity/audit_cor.R`` from SCP's ``CellCorHeatmap`` and
``FeatureCorPlot`` (Hao Zhang, NJMU, GPL-3;
https://github.com/zhanghao-njmu/SCP).  They carry the log-normalised inputs
(pancreas_sub as query, gene-renamed panc8_sub as reference, restricted to 50
shared highly variable genes) as well as R's similarity matrices, clustering
orders, drawn labels and FeatureCorPlot's transforms, so this module needs
neither R nor the parity ``.h5ad``.

``feature_cor_heatmap`` has no fixture: SCP's ``FeatureCorHeatmap`` is an
empty stub (SCP-plot.R:9980), so there is nothing in R to compare against.

Where the port agrees the assertion is tight.  Where it does not, the test is
``xfail(strict=True)`` with a one-line diagnosis, so the suite stays green and
the fix flips it.  The diagnoses are written out in ``docs/parity_audit.md``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex, to_rgb
from scipy.cluster.hierarchy import linkage, to_tree

import scp

anndata = pytest.importorskip("anndata")

FIX = Path(__file__).parent / "fixtures" / "parity" / "audit" / "cor"
FEATURES = json.loads((FIX / "features.json").read_text())["features"]
META = json.loads((FIX / "cellcor_meta.json").read_text())


def _r(name: str) -> pd.DataFrame:
    return pd.read_csv(FIX / f"cellcor_{name}.csv", index_col=0)


def _adata(expr: str, obs: str, cols: list[str]):
    e = pd.read_csv(FIX / expr, index_col=0)
    e.index = e.index.astype(str)
    o = pd.read_csv(FIX / obs, index_col=0)
    o.index = o.index.astype(str)
    return anndata.AnnData(X=e[FEATURES].to_numpy(), obs=o.loc[e.index, cols].astype(str),
                           var=pd.DataFrame(index=FEATURES))


@pytest.fixture(scope="module")
def query():
    q = _adata("query_expr.csv", "query_obs.csv", ["CellType", "SubCellType", "Phase"])
    levels = list(_r("self_cosine").index)  # SubCellType's declared R levels
    q.obs["SubCellType"] = pd.Categorical(q.obs["SubCellType"], categories=levels)
    return q


@pytest.fixture(scope="module")
def panc8():
    # celltype is a plain character column in R: levels by first appearance
    return _adata("panc8_expr.csv", "panc8_obs.csv", ["celltype"])


def _heatmap(*args, **kw):
    res = scp.pl.cell_cor_heatmap(*args, features=FEATURES, **kw)
    plt.close(res.fig)
    return res


def _labels(res) -> list[float]:
    vals = []
    for ax in res.fig.axes:
        for t in ax.texts:
            if re.fullmatch(r"-?\d+(\.\d+)?", t.get_text()):
                vals.append(float(t.get_text()))
    return sorted(vals)


# --------------------------------------------------------------- cell_cor_heatmap
@pytest.mark.parametrize("case, metric", [("self_cosine", "cosine"),
                                          ("self_pearson", "pearson"),
                                          ("self_spearman", "spearman")])
def test_cellcor_self_matrix(query, case, metric):
    """Group centroids (mean in linear space, then log1p) and the similarity agree."""
    R = _r(case)
    res = _heatmap(query, query, query_group="SubCellType", ref_group="SubCellType",
                   distance_metric=metric)
    P = res.matrices["similarity"]
    assert list(P.index) == list(R.index) and list(P.columns) == list(R.columns)
    np.testing.assert_allclose(P.to_numpy(), R.to_numpy(), rtol=0, atol=1e-7)
    # colour limits are the plain min/max, not quantile-clipped
    lg = res.spec.legends[0]
    np.testing.assert_allclose([lg.vmin, lg.vmax], META[case]["limits"], atol=1e-7)


@pytest.mark.xfail(strict=True, reason="reference=None does not inherit query_group / "
                   "query_collapsing into the reference as R does (SCP-plot.R:10150-10156); "
                   "ref_group stays None and fetching it raises KeyError")
def test_cellcor_reference_none_inherits_query_grouping(query):
    R = _r("self_cosine")
    res = _heatmap(query, query_group="SubCellType")
    np.testing.assert_allclose(res.matrices["similarity"].to_numpy(), R.to_numpy(), atol=1e-7)


@pytest.mark.parametrize("case, metric", [("panc8_cosine", "cosine"),
                                          ("panc8_spearman", "spearman")])
def test_cellcor_panc8_matrix(query, panc8, case, metric):
    R = _r(case)
    res = _heatmap(query, panc8, query_group="SubCellType", ref_group="celltype",
                   distance_metric=metric)
    P = res.matrices["similarity"]
    assert list(P.index) == list(R.index) and list(P.columns) == list(R.columns)
    np.testing.assert_allclose(P.to_numpy(), R.to_numpy(), rtol=0, atol=1e-7)
    lg = res.spec.legends[0]
    np.testing.assert_allclose([lg.vmin, lg.vmax], META[case]["limits"], atol=1e-7)


def test_cellcor_uncollapsed_values(query, panc8):
    """Per-cell rows: every query cell's similarities agree once aligned by name."""
    R = _r("panc8_cells")
    res = _heatmap(query, panc8, query_group="SubCellType", query_collapsing=False,
                   ref_group="celltype")
    P = res.matrices["similarity"]
    assert set(P.index) == set(R.index.astype(str))
    np.testing.assert_allclose(P.loc[R.index.astype(str), R.columns].to_numpy(), R.to_numpy(),
                               rtol=0, atol=1e-7)


@pytest.mark.xfail(strict=True, reason="uncollapsed query rows stay in object order; R orders "
                   "them by query_group level (SCP-plot.R:10218-10241), so the group blocks touch")
def test_cellcor_uncollapsed_row_order(query, panc8):
    R = _r("panc8_cells")
    res = _heatmap(query, panc8, query_group="SubCellType", query_collapsing=False,
                   ref_group="celltype")
    assert list(res.matrices["similarity"].index) == list(R.index.astype(str))


@pytest.mark.parametrize("case, kw", [
    ("self_label_row", dict(nlabel=2, label_cutoff=0.5)),
    ("self_label_col", dict(nlabel=1, label_by="column")),
    ("self_label_both", dict(nlabel=2, label_by="both")),
])
def test_cellcor_labelled_cells(query, case, kw):
    """The same cells are labelled (compared as values; R prints round(x, 2))."""
    res = scp.pl.cell_cor_heatmap(query, query, query_group="SubCellType",
                                  ref_group="SubCellType", features=FEATURES, **kw)
    py = _labels(res)
    plt.close(res.fig)
    r = sorted(float(x) for x in META[case]["labels"])
    np.testing.assert_allclose(py, r, atol=0.005)


@pytest.mark.xfail(strict=True, reason="nlabel defaults to 3 (and label_cutoff to -inf); R's "
                   "defaults are nlabel=0, label_cutoff=0, so R draws no labels")
def test_cellcor_default_draws_no_labels(query):
    res = scp.pl.cell_cor_heatmap(query, query, query_group="SubCellType",
                                  ref_group="SubCellType", features=FEATURES)
    py = _labels(res)
    plt.close(res.fig)
    assert py == [float(x) for x in META["self_cosine"]["labels"]] == []


CLUSTER_CASES = [("self_cosine", "cosine", False), ("self_spearman", "spearman", False),
                 ("panc8_cosine", "cosine", True), ("panc8_spearman", "spearman", True)]


@pytest.mark.parametrize("case, metric, use_panc8", CLUSTER_CASES)
def test_cellcor_cluster_order(query, panc8, case, metric, use_panc8):
    ref, rg = (panc8, "celltype") if use_panc8 else (query, "SubCellType")
    res = _heatmap(query, ref, query_group="SubCellType", ref_group=rg, distance_metric=metric,
                   cluster_rows=True, cluster_columns=True)
    P = res.matrices["similarity"]
    assert list(P.index) == META[case]["cluster_rows"]
    assert list(P.columns) == META[case]["cluster_cols"]


def _complexheatmap_order(m: np.ndarray) -> list[int]:
    """hclust(dist(m), "complete") then reorder.dendrogram(wts = -rowMeans, mean)."""
    w = -m.mean(axis=1)

    def walk(node):
        if node.is_leaf():
            return w[node.id], [node.id]
        a, b = walk(node.left), walk(node.right)
        lo, hi = (a, b) if a[0] <= b[0] else (b, a)
        return (lo[0] + hi[0]) / 2, lo[1] + hi[1]

    return walk(to_tree(linkage(m, method="complete", metric="euclidean")))[1]


@pytest.mark.parametrize("case", [c for c, _, _ in CLUSTER_CASES])
def test_proposed_dendrogram_reorder_reproduces_r(case):
    """The fix proposed in docs/parity_audit.md, checked against R's own orders."""
    R = _r(case)
    m = R.to_numpy()
    assert [R.index[i] for i in _complexheatmap_order(m)] == META[case]["cluster_rows"]
    assert [R.columns[i] for i in _complexheatmap_order(m.T)] == META[case]["cluster_cols"]


# --------------------------------------------------------------- feature_cor_plot
FC = json.loads((FIX / "featcor.json").read_text())
FC_FEATS = FC["features"]
CELLTYPE_LEVELS = ["Ductal", "Ngn3 low EP", "Ngn3 high EP", "Pre-endocrine", "Endocrine"]


@pytest.fixture(scope="module")
def featcor_adata():
    inp = pd.read_csv(FIX / "featcor_input.csv", index_col=0)
    obs = pd.DataFrame({"CellType": pd.Categorical(inp["CellType"], categories=CELLTYPE_LEVELS),
                        "Phase": inp["Phase"].astype(str).to_numpy()},
                       index=inp.index.astype(str))
    return anndata.AnnData(X=inp[FC_FEATS].to_numpy(), obs=obs,
                           var=pd.DataFrame(index=FC_FEATS))


def _grid(fig, n):
    return fig.axes[: n * n]  # plt.subplots order: row-major, colorbar appended after


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_featcor_lower_blocks(featcor_adata, method):
    """Correlation text and block colour, at R's grid position (row = f2, col = f1)."""
    n = len(FC_FEATS)
    fig = scp.pl.feature_cor_plot(featcor_adata, FC_FEATS, group_by="CellType", cor_method=method)
    grid = _grid(fig, n)
    for pr in FC[method]["lower"]:
        i, j = FC_FEATS.index(pr["f1"]), FC_FEATS.index(pr["f2"])
        ax = grid[j * n + i]
        (txt,) = [t.get_text() for t in ax.texts]
        assert txt.split("\n")[:2] == [pr["f1"], pr["f2"]]
        assert float(txt.split("Cor: ")[1]) == pytest.approx(pr["cor"], abs=5e-4)
        # R bins r into 100 palette steps, Python interpolates: within a few 8-bit steps
        delta = np.abs(np.array(to_rgb(ax.patches[0].get_facecolor()))
                       - np.array(to_rgb(pr["fill"]))).max() * 255
        assert delta <= 6, (pr, to_hex(ax.patches[0].get_facecolor()))
    plt.close(fig)


def test_featcor_upper_scatter_and_lm(featcor_adata):
    n = len(FC_FEATS)
    fig = scp.pl.feature_cor_plot(featcor_adata, FC_FEATS, group_by="CellType")
    grid = _grid(fig, n)
    X = featcor_adata.to_df()
    for lm in FC["lm"]:
        i, j = FC_FEATS.index(lm["f1"]), FC_FEATS.index(lm["f2"])
        ax = grid[j * n + i]
        np.testing.assert_array_equal(ax.collections[0].get_offsets(),
                                      X[[lm["f1"], lm["f2"]]].to_numpy())
        xs, ys = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
        slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
        assert slope == pytest.approx(lm["slope"], rel=1e-9)
        assert ys[0] - slope * xs[0] == pytest.approx(lm["intercept"], rel=1e-9)
        (txt,) = [t.get_text() for t in ax.texts]
        r2 = float(re.search(r"\$r\^2\$ = (\S+)", txt).group(1))
        p = float(re.search(r"p = (\S+)", txt).group(1))
        assert r2 == pytest.approx(float(lm["r2_text"]), rel=1e-12)
        assert p == pytest.approx(float(lm["p_text"]), rel=1e-12)
    plt.close(fig)


def _half_widths(ax, k):
    """Normalised half-width profile of the k-th violin body."""
    verts = ax.collections[k].get_paths()[0].vertices
    right = verts[: len(verts) // 2]
    o = np.argsort(right[:, 1])
    w = np.abs(right[o, 0] - k)
    return right[o, 1], w / w.max()


def test_featcor_violin_extent(featcor_adata):
    """trim=TRUE: each violin spans exactly the data range, as R's does."""
    n = len(FC_FEATS)
    fig = scp.pl.feature_cor_plot(featcor_adata, FC_FEATS, group_by="CellType")
    grid = _grid(fig, n)
    for d, f in enumerate(FC_FEATS):
        rv = pd.DataFrame(FC["violin"][f])
        for k, g in enumerate(CELLTYPE_LEVELS):
            y, _ = _half_widths(grid[d * n + d], k)
            sub = rv[rv["group"] == g]
            assert (y.min(), y.max()) == pytest.approx((sub["y"].min(), sub["y"].max()), abs=1e-9)
    plt.close(fig)


def test_featcor_violin_shape(featcor_adata):
    n = len(FC_FEATS)
    fig = scp.pl.feature_cor_plot(featcor_adata, FC_FEATS, group_by="CellType")
    grid = _grid(fig, n)
    for d, f in enumerate(FC_FEATS):
        rv = pd.DataFrame(FC["violin"][f])
        for k, g in enumerate(CELLTYPE_LEVELS):
            y, w = _half_widths(grid[d * n + d], k)
            sub = rv[rv["group"] == g]
            np.testing.assert_allclose(np.interp(sub["y"], y, w), sub["violinwidth"], atol=0.01)
    plt.close(fig)


def test_featcor_split_by(featcor_adata):
    n = len(FC_FEATS)
    fig = scp.pl.feature_cor_plot(featcor_adata, FC_FEATS, group_by="CellType", split_by="Phase")
    texts = [t.get_text() for ax in fig.axes for t in ax.texts if "Cor: " in t.get_text()]
    want = []
    for s, sim in FC["split"].items():
        assert s
        S = np.asarray(sim)
        want += [S[FC_FEATS.index(a), FC_FEATS.index(b)]
                 for a in FC_FEATS for b in FC_FEATS if FC_FEATS.index(a) > FC_FEATS.index(b)]
    got = sorted(float(t.split("Cor: ")[1]) for t in texts)
    assert len(got) == len(want) == len(FC["split"]) * n * (n - 1) // 2
    np.testing.assert_allclose(got, sorted(want), atol=5e-4)


def test_featcor_stats_without_smooth(featcor_adata):
    n = len(FC_FEATS)
    fig = scp.pl.feature_cor_plot(featcor_adata, FC_FEATS, group_by="CellType", add_smooth=False)
    ax = _grid(fig, n)[1 * n + 0]
    assert any("p = " in t.get_text() for t in ax.texts)
    plt.close(fig)
