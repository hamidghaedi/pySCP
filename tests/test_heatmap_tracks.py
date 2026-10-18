"""Structural tests for feature_heatmap, dynamic_heatmap, cell_cor_heatmap and
the pure heatmap helpers (splitting, ordering, enrichment boxes, similarity).

Numeric agreement with SCP lives in ``test_heatmap_parity.py``; this file pins
shapes, level order, track stacks and the error paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scp
from scp.heatmap import cluster as C
from scp.heatmap import enrichment as E
from scp.heatmap.similarity import DIST_METHODS, SIMIL_METHODS, cell_similarity


@pytest.fixture(autouse=True)
def _close_figures():
    import matplotlib.pyplot as plt
    yield
    plt.close("all")


@pytest.fixture()
def lin(adata):
    """Two pseudotime lineages over the synthetic object."""
    ad = adata.copy()
    t = np.asarray(ad.obsm["X_umap"][:, 0], dtype=float)
    t = t - t.min()
    lab = np.asarray(ad.obs["leiden"])
    ad.obs["L1"] = np.where(lab != "D", t, np.nan)
    ad.obs["L2"] = np.where(lab != "A", t, np.nan)
    return ad


def _genes(adata, n=8):
    return list(adata.var_names[:n])


# --------------------------------------------------------------------------- #
# feature_heatmap
# --------------------------------------------------------------------------- #
def test_feature_heatmap_samples_at_most_max_cells_per_group(adata):
    r = scp.pl.feature_heatmap(adata, _genes(adata), "leiden", max_cells=20, layer=None)
    m = r.matrices["leiden"]
    assert m.shape == (8, 80)
    panel = r.spec.panels[0]
    assert list(panel.col_split.categories) == list("ABCD")
    assert list(pd.Series(np.asarray(panel.col_split)).value_counts().sort_index()) == [20] * 4


def test_feature_heatmap_split_by_makes_group_split_slices(adata):
    r = scp.pl.feature_heatmap(adata, _genes(adata), "leiden", split_by="batch",
                               max_cells=30, layer=None)
    panel = r.spec.panels[0]
    assert list(panel.col_split.categories) == [f"{g} : {b}" for g in "ABCD"
                                                for b in ("b1", "b2")]
    assert [t.name for t in panel.top_tracks] == ["leiden", "batch"]


def test_feature_heatmap_one_panel_per_group_by(adata):
    r = scp.pl.feature_heatmap(adata, _genes(adata), ["leiden", "batch"], max_cells=15,
                               layer=None, n_split=2, split_method="hclust",
                               feature_split_by=["batch"])
    assert list(r.matrices) == ["leiden", "batch"]
    assert len(r.spec.panels) == 2
    assert set(r.feature_split.cat.categories) == {"C1", "C2"}


def test_feature_heatmap_cell_annotation_reads_genes_from_x(adata):
    r = scp.pl.feature_heatmap(adata, _genes(adata), "leiden", max_cells=10, layer="counts",
                               cell_annotation=["batch", "g20"])
    names = [t.name for t in r.spec.panels[0].top_tracks]
    assert names == ["leiden", "batch", "g20"]
    track = r.spec.panels[0].top_tracks[2]
    cells = list(r.spec.panels[0].col_labels)
    np.testing.assert_allclose(track.values, adata[cells, "g20"].X.ravel())


def test_feature_heatmap_cell_order_restricts_and_orders(adata):
    want = list(adata.obs_names[::-7])
    r = scp.pl.feature_heatmap(adata, _genes(adata), "leiden", max_cells=1000,
                               layer=None, cell_order=want)
    cols = list(r.matrices["leiden"].columns)
    assert cols == [c for c in want if c in set(cols)]


def test_feature_heatmap_cluster_columns_keeps_slices(adata):
    r = scp.pl.feature_heatmap(adata, _genes(adata), "leiden", max_cells=15, layer=None,
                               cluster_columns=True)
    panel = r.spec.panels[0]
    assert list(panel.col_split.categories) == list("ABCD")
    assert "leiden" in r.spec.col_dends


# --------------------------------------------------------------------------- #
# dynamic_heatmap
# --------------------------------------------------------------------------- #
def test_dynamic_heatmap_orders_cells_by_pseudotime(lin):
    pytest.importorskip("sklearn")
    r = scp.pl.dynamic_heatmap(lin, ["L1", "L2"], _genes(lin), layer=None, n_split=2,
                               split_method="kmeans-peaktime")
    for name in ("L1", "L2"):
        t = lin.obs.loc[r.cell_order[name], name].to_numpy()
        assert np.all(np.diff(t) >= 0)
        assert list(r.matrices[name].columns) == r.cell_order[name]
    # z-scored once across both lineages: each row is centred jointly
    joint = np.hstack([r.matrices[n].to_numpy() for n in ("L1", "L2")])
    assert np.allclose(joint.mean(axis=1), 0, atol=1e-8)
    assert [t.name for t in r.spec.right_tracks] == ["L1", "L2"]
    assert [t.name for t in r.spec.panels[0].top_tracks] == ["Pseudotime"]


def test_dynamic_heatmap_reverse_ht_descends(lin):
    r = scp.pl.dynamic_heatmap(lin, ["L1", "L2"], _genes(lin), layer=None, reverse_ht=["L2"])
    t = lin.obs.loc[r.cell_order["L2"], "L2"].to_numpy()
    assert np.all(np.diff(t) <= 0)


def test_dynamic_heatmap_presorts_by_peak_time(lin):
    r = scp.pl.dynamic_heatmap(lin, ["L1"], _genes(lin), layer=None)
    peak = r.var_meta["peaktime"].to_numpy()
    assert np.all(np.diff(peak) >= 0)
    assert list(r.var_meta.index) == list(r.matrices["L1"].index)


def test_dynamic_heatmap_reads_dynamic_features_and_filters(lin):
    genes = _genes(lin, 6)
    tab = pd.DataFrame({"features": genes, "exp_ncells": [50, 50, 50, 5, 50, 50],
                        "r.sq": [0.5] * 6, "dev.expl": [0.5] * 6,
                        "peaktime": [3, 1, 2, 0, 5, 4.0], "valleytime": [0.0] * 6,
                        "padjust": [0.01, 0.01, 0.01, 0.01, 0.01, 0.5]})
    r = scp.pl.dynamic_heatmap(lin, ["L1"], None, layer=None, dynamic_features={"L1": tab})
    # exp_ncells and padjust thresholds drop two; the rest sort by peak time
    assert list(r.var_meta.index) == ["g1", "g2", "g0", "g4"]


def test_dynamic_heatmap_mfuzz_is_the_default(lin):
    pytest.importorskip("skfuzzy")
    r = scp.pl.dynamic_heatmap(lin, ["L1"], _genes(lin, 12), layer=None, n_split=2)
    assert r.feature_split is not None


def test_pseudotime_label_draws_a_line_per_value(lin):
    r = scp.pl.dynamic_heatmap(lin, ["L1"], _genes(lin), layer=None,
                               pseudotime_label=[1.0, 3.0])
    body = r.fig.axes[0]
    dashed = [ln for ln in body.get_lines() if ln.get_linestyle() == "--"]
    assert len(dashed) == 2


def test_feature_split_by_must_be_lineages(lin):
    with pytest.raises(ValueError, match="subset of lineages"):
        scp.pl.dynamic_heatmap(lin, ["L1"], _genes(lin), layer=None, feature_split_by=["L2"])


# --------------------------------------------------------------------------- #
# cell_cor_heatmap
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("metric", SIMIL_METHODS + DIST_METHODS + ("hamman",))
def test_cell_cor_heatmap_accepts_every_scp_metric(adata, metric):
    # an explicit reference: with reference=None R makes ref_group = query_group
    r = scp.pl.cell_cor_heatmap(adata, adata, query_group="leiden", ref_group="batch",
                                features=_genes(adata, 10), distance_metric=metric)
    m = r.matrices["similarity"]
    assert m.shape == (4, 2)
    assert np.isfinite(m.to_numpy()).all()


def test_similarity_titles_follow_r():
    x = np.abs(np.random.default_rng(0).normal(size=(3, 5))) + 0.1
    assert cell_similarity(x, x, "pearson")[1] == "Correlation similarity"
    assert cell_similarity(x, x, "simple matching")[1] == "Simple matching similarity"
    assert cell_similarity(x, x, "canberra")[1] == "1-dist[canberra]/max(dist[canberra])"
    s, _ = cell_similarity(x, x, "euclidean")
    np.testing.assert_allclose(np.diag(s), 1.0)  # no cancellation on the diagonal
    with pytest.raises(ValueError, match="invalid"):
        cell_similarity(x, x, "nope")


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_reorder_dendrogram_puts_light_children_first():
    X = np.array([[0.0], [0.1], [5.0], [5.1]])
    Z = C.hclust(X)
    _, order = C.reorder_dendrogram(Z, np.array([4.0, 3.0, 2.0, 1.0]))
    assert list(order) == [3, 2, 1, 0]


def test_cutree_numbers_by_first_appearance():
    X = np.array([[9.0], [0.0], [9.1], [0.1], [4.0]])
    assert list(C.cutree(C.hclust(X), 3)) == [1, 2, 1, 2, 3]


def test_mestimate_formula():
    N, D = 30, 13
    want = 1 + (1418 / N + 22.05) * D ** -2 + (12.33 / N + 0.243) * D ** (
        -0.0406 * np.log(N) - 0.1134)
    assert C.mestimate(np.zeros((N, D))) == pytest.approx(want)


def test_name_splits_by_value_orders_by_mean_peak():
    fs, rs = C.name_splits_by_value(np.array([1, 2, 1, 2, 3]), np.array([5.0, 1, 5, 1, 3]))
    assert list(fs) == ["C3", "C1", "C3", "C1", "C2"]
    assert list(rs.categories) == ["C1(2)", "C2(1)", "C3(2)"]


def test_axis_order_snaps_noise_weights_to_ties():
    """z-scored rows have mean 0 up to rounding; that noise must not reorder."""
    rng = np.random.default_rng(1)
    M = rng.normal(size=(12, 6))
    M = (M - M.mean(1, keepdims=True)) / M.std(1, ddof=1, keepdims=True)
    order = C.axis_order(M, cluster=True).order
    np.testing.assert_array_equal(order, C.dend_order(C.hclust(M)))


def test_bh_matches_r():
    p = np.array([0.01, 0.04, 0.03, 0.04, 0.2])
    # p.adjust(c(.01,.04,.03,.04,.2), "BH")
    np.testing.assert_allclose(E.p_adjust_bh(p), [0.05, 0.05, 0.05, 0.05, 0.2])


def test_capitalize_leaves_mixed_case_first_words():
    assert E.capitalize(["mRNA processing", "cell cycle", "DNA repair", "x-linked gene"]) == [
        "mRNA processing", "Cell cycle", "DNA repair", "X-linked gene"]
