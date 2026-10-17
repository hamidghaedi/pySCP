"""The heatmap family against SCP's own numbers.

Every fixture under ``tests/fixtures/parity/heatmap/`` was written by
``tools/parity/heatmap_fixtures.R`` (SCP 0.5.6 on ``pancreas_sub``) and
carries its own input -- 30 genes' counts for the 1000 cells, the obs columns,
their factor levels and full-transcriptome library sizes -- so this suite
rebuilds a small AnnData from CSV and never needs R or the h5ad.

What is compared, and how tightly:

* deterministic transforms (matrices, hclust splits, every row/column order
  that does not depend on noise, similarity metrics, ORA p-values, the
  enrichment word boxes) -- to the precision the CSVs were written at;
* kmeans -- R's Hartigan-Wong and scikit-learn's Lloyd cannot share an RNG, so
  only the partition is compared (adjusted Rand index), plus equality of the
  within-cluster sum of squares where both reach the optimum;
* ComplexHeatmap's reorder of z-scored rows is driven by rounding noise
  (every z-scored row has mean 0), so there the check is that R's order is a
  leaf order of the *same tree*.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scp
from scp.heatmap import cluster as C
from scp.heatmap import enrichment as E
from scp.heatmap.similarity import cell_similarity

FX = Path(__file__).parent / "fixtures" / "parity" / "heatmap"
GB = ["CellType", "SubCellType"]
LINEAGES = ["Lineage1", "Lineage2"]


@pytest.fixture(autouse=True)
def _close_figures():
    import matplotlib.pyplot as plt
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def panc():
    anndata = pytest.importorskip("anndata")
    levels = json.loads((FX / "input_levels.json").read_text())
    counts = pd.read_csv(FX / "input_counts.csv", index_col=0)
    obs = pd.read_csv(FX / "input_obs.csv", index_col=0)
    for key, lv in levels.items():
        obs[key] = pd.Categorical(obs[key], categories=lv)
    obs["Batch"] = pd.Categorical(obs["Batch"], categories=["b1", "b2"])
    ad = anndata.AnnData(X=counts.to_numpy(dtype=float), obs=obs,
                         var=pd.DataFrame(index=counts.columns))
    ad.layers["counts"] = ad.X.copy()
    return ad


@pytest.fixture(scope="module")
def feats(panc):
    return list(panc.var_names)


def _wide(name: str) -> pd.DataFrame:
    """Back from ``as.data.frame(as.table(m))``: column-major, duplicates kept."""
    ms = pd.read_csv(FX / name)
    rows = list(dict.fromkeys(ms.feature))
    cols = list(ms.column[:: len(rows)])
    return pd.DataFrame(ms.value.to_numpy().reshape(len(cols), len(rows)).T, index=rows,
                        columns=cols)


def _meta(name: str) -> pd.DataFrame:
    return pd.read_csv(FX / name).set_index("feature")


def _assert_meta(res, name: str, *, split: bool = True):
    r = _meta(name)
    v = res.var_meta.loc[r.index]
    np.testing.assert_array_equal(v["index"].to_numpy(), r["index"].to_numpy())
    if split:
        np.testing.assert_array_equal(v["feature_split"].astype(str).to_numpy(),
                                      r["feature_split"].astype(str).to_numpy())


def _leaf_order_of(Z: np.ndarray, order: np.ndarray) -> bool:
    """Is ``order`` a left-to-right leaf order of tree ``Z`` (children swapped freely)?"""
    n = Z.shape[0] + 1
    pos = {leaf: k for k, leaf in enumerate(order)}
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    for i, (a, b) in enumerate(Z[:, :2].astype(int)):
        m = members.pop(a) + members.pop(b)
        p = sorted(pos[x] for x in m)
        if p[-1] - p[0] != len(p) - 1:
            return False
        members[n + i] = m
    return True


KW = {"libsize": "libsize"}  # the full-transcriptome library size R used


# --------------------------------------------------------------------------- #
# group_heatmap
# --------------------------------------------------------------------------- #
def test_split_by_matrix_and_column_order(panc, feats):
    r = _wide("group_split_by.csv")
    m = scp.pl.group_heatmap(panc, feats, "CellType", split_by="Phase", **KW).matrices["CellType"]
    assert list(m.columns) == list(r.columns)  # "group : split", group-major
    assert np.max(np.abs(m.loc[r.index].to_numpy() - r.to_numpy())) < 1e-6


@pytest.mark.parametrize("name,kw", [
    ("group_grouping_var.csv", {"grouping_var": "Phase", "numerator": "G2M"}),
    ("group_grouping_var_split.csv", {"grouping_var": "Batch", "numerator": "b2",
                                      "split_by": "Phase"}),
])
def test_grouping_var_log2fc(panc, feats, name, kw):
    r = _wide(name)
    cells = None
    if "split_by" in kw:  # the fixture had to drop R's one-armed combination
        obs = panc.obs
        cells = list(obs.index[~((obs.CellType == "Pre-endocrine") & (obs.Phase == "S"))])
    with pytest.warns(UserWarning, match="log2fc"):
        res = scp.pl.group_heatmap(panc, feats, "CellType", cells=cells, **kw, **KW)
    m = res.matrices["CellType"]
    assert list(m.columns) == list(r.columns)  # one-armed groups dropped
    assert np.max(np.abs(m.loc[r.index].to_numpy() - r.to_numpy())) < 1e-6


def test_grouping_var_with_split_keeps_only_two_armed_columns(panc, feats):
    """R aborts here (column_split built from every level); the port drops the
    one-armed Pre-endocrine : S combination and carries on."""
    with pytest.warns(UserWarning):
        res = scp.pl.group_heatmap(panc, feats, "CellType", split_by="Phase",
                                   grouping_var="Batch", numerator="b2", **KW)
    assert "Pre-endocrine : S" not in res.matrices["CellType"].columns


def test_grouping_var_with_n_split_names_by_kept_columns(panc, feats):
    """R errors here ("arguments must have same length"): its group means are
    taken over every group level, including the one-armed groups the log2
    matrix dropped.  The port names clusters from the columns it kept."""
    with pytest.warns(UserWarning):
        res = scp.pl.group_heatmap(panc, feats, "CellType", grouping_var="Phase",
                                   numerator="G2M", n_split=3, split_method="hclust", **KW)
    assert list(res.matrices["CellType"].columns) == ["Ductal", "Ngn3 low EP", "Ngn3 high EP"]
    assert list(res.feature_split.cat.categories) == ["C1", "C2", "C3"]


@pytest.mark.parametrize("name,kw", [
    ("group_hclust.csv", {}),
    ("group_hclust_split_by.csv", {"feature_split_by": ["SubCellType"]}),
    ("group_hclust_decreasing.csv", {"decreasing": True}),
    ("group_cluster_row_slices.csv", {"cluster_row_slices": True}),
    ("group_cluster_features_by.csv", {"cluster_rows": True,
                                       "cluster_features_by": ["SubCellType"]}),
    ("group_raw_cluster_rows_and_slices.csv", {"cluster_rows": True,
                                               "cluster_row_slices": True,
                                               "exp_method": "raw"}),
])
def test_hclust_splits_and_row_orders_match_r(panc, feats, name, kw):
    res = scp.pl.group_heatmap(panc, feats, GB, n_split=4, split_method="hclust", **kw, **KW)
    _assert_meta(res, name)


def test_raw_cluster_rows_matches_r(panc, feats):
    res = scp.pl.group_heatmap(panc, feats, GB, cluster_rows=True, exp_method="raw", **KW)
    _assert_meta(res, "group_raw_cluster_rows.csv", split=False)


@pytest.mark.parametrize("name,kw", [
    ("group_cluster_rows.csv", {}),
    ("group_cluster_rows_split.csv", {"n_split": 4, "split_method": "hclust"}),
])
def test_zscore_cluster_rows_is_the_same_tree(panc, feats, name, kw):
    """R's z-score reorder flips children on rounding noise; the trees agree."""
    res = scp.pl.group_heatmap(panc, feats, GB, cluster_rows=True, **kw, **KW)
    r = _meta(name)
    r_order = np.argsort(r.loc[feats, "index"].to_numpy())
    mat = res.matrices["CellType"].loc[feats].to_numpy()
    split = res.var_meta.loc[feats, "feature_split"]
    groups = [np.arange(len(feats))] if split.isna().all() else \
        [np.flatnonzero(split.astype(str).to_numpy() == g) for g in split.cat.categories]
    for idx in groups:
        sub_r = [i for i in r_order if i in set(idx)]
        local = {g: k for k, g in enumerate(idx)}
        assert _leaf_order_of(C.hclust(mat[idx]), np.array([local[i] for i in sub_r]))


def test_kmeans_partition_agrees_with_r(panc, feats):
    from sklearn.metrics import adjusted_rand_score

    res = scp.pl.group_heatmap(panc, feats, GB, n_split=4, split_method="kmeans", **KW)
    r = _meta("group_kmeans.csv")
    ari = adjusted_rand_score(r.feature_split, res.var_meta.loc[r.index, "feature_split"])
    # 1.0 on this data once Lloyd is followed by Hartigan moves (0.953 without:
    # plain Lloyd stops in an optimum Hartigan-Wong does not).  Only the
    # partition is guaranteed comparable, so the bound is loose.
    assert ari > 0.9


def test_column_orders_match_r(panc, feats):
    res = scp.pl.group_heatmap(panc, feats, "CellType", split_by="Phase",
                               cluster_column_slices=True, **KW)
    want = pd.read_csv(FX / "group_cluster_column_slices.csv").column.tolist()
    assert list(res.spec.panels[0].col_labels) == want
    for name, slices in (("group_cluster_columns.csv", False),
                         ("group_cluster_columns_and_slices.csv", True)):
        res = scp.pl.group_heatmap(panc, feats, "CellType", split_by="Phase",
                                   cluster_columns=True, cluster_column_slices=slices, **KW)
        want = pd.read_csv(FX / name).column.tolist()
        assert list(res.spec.panels[0].col_labels) == want


# --------------------------------------------------------------------------- #
# feature_heatmap
# --------------------------------------------------------------------------- #
def test_feature_heatmap_matrix_and_split(panc, feats):
    res = scp.pl.feature_heatmap(panc, feats, "CellType", max_cells=1000, n_split=4,
                                 split_method="hclust", **KW)
    _assert_meta(res, "feature_hclust.csv")
    r = _wide("feature_matrix.csv")
    m = res.matrices["CellType"]
    assert np.max(np.abs(m.loc[r.index, r.columns].to_numpy() - r.to_numpy())) < 1e-8


# --------------------------------------------------------------------------- #
# dynamic_heatmap
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def dyn_tables():
    return {lin: pd.read_csv(FX / f"dynamic_features_{lin}.csv") for lin in LINEAGES}


def _dyn(panc, feats, dyn_tables, **kw):
    return scp.pl.dynamic_heatmap(panc, LINEAGES, feats, dynamic_features=dyn_tables,
                                  libsize="libsize", **kw)


def test_dynamic_cell_order_and_joint_matrix(panc, feats, dyn_tables):
    res = _dyn(panc, feats, dyn_tables, n_split=4, split_method="hclust-peaktime")
    co = pd.read_csv(FX / "dynamic_cell_order.csv")
    for lin in LINEAGES:
        want = [c[: -len(lin)] for c in co.column[co.lineage == lin]]
        assert res.cell_order[lin] == want
    dm = pd.read_csv(FX / "dynamic_matrix.csv")
    diffs = []
    for feat, col, val in dm[["feature", "column", "value"]].itertuples(index=False):
        lin = next(lin for lin in LINEAGES if col.endswith(lin))
        diffs.append(abs(res.matrices[lin].loc[feat, col[: -len(lin)]] - val))
    assert max(diffs) < 1e-8  # z-scored jointly across both lineages


@pytest.mark.parametrize("name,kw", [
    ("dynamic_hclust_peaktime.csv", {"n_split": 4, "split_method": "hclust-peaktime"}),
    ("dynamic_hclust.csv", {"n_split": 4, "split_method": "hclust"}),
    ("dynamic_hclust_valleytime.csv", {"n_split": 3, "split_method": "hclust-peaktime",
                                       "order_by": "valleytime", "reverse_ht": ["Lineage2"]}),
    ("dynamic_cluster_features_by.csv", {"n_split": 4, "split_method": "hclust",
                                         "cluster_rows": True,
                                         "cluster_features_by": ["Lineage1"]}),
    ("dynamic_feature_split_by.csv", {"n_split": 4, "split_method": "hclust",
                                      "feature_split_by": ["Lineage2"]}),
])
def test_dynamic_splits_and_orders_match_r(panc, feats, dyn_tables, name, kw):
    _assert_meta(_dyn(panc, feats, dyn_tables, **kw), name)


def test_kmeans_peaktime_reaches_rs_optimum(panc, feats, dyn_tables):
    """1-D kmeans is solved exactly, which is the optimum R's 20 Hartigan-Wong
    starts reach: same within-cluster sum of squares, same partition (Lloyd
    with 20 starts stopped at 45.27 against R's 44.52 here)."""
    res = _dyn(panc, feats, dyn_tables, n_split=4, split_method="kmeans-peaktime")
    r = _meta("dynamic_kmeans_peaktime.csv")
    vm = res.var_meta.loc[r.index]
    _assert_meta(res, "dynamic_kmeans_peaktime.csv")

    def wss(lab):
        y = vm["peaktime"]
        return sum(((y[lab == g] - y[lab == g].mean()) ** 2).sum() for g in set(lab))

    assert wss(vm["feature_split"].astype(str).to_numpy()) == pytest.approx(
        wss(r["feature_split"].to_numpy()), rel=1e-9)


# --------------------------------------------------------------------------- #
# cell_cor_heatmap similarity
# --------------------------------------------------------------------------- #
def test_every_distance_metric_matches_proxyc():
    R = pd.read_csv(FX / "cor_ref.csv", index_col=0)
    Q = pd.read_csv(FX / "cor_query.csv", index_col=0)
    cm = pd.read_csv(FX / "cell_cor_metrics.csv")
    assert set(cm.metric) == set(scp.heatmap.similarity.SIMIL_METHODS) | set(
        scp.heatmap.similarity.DIST_METHODS)
    for metric, sub in cm.groupby("metric"):
        want = sub.pivot(index="query", columns="ref", values="value").loc[Q.index, R.index]
        got, title = cell_similarity(Q.to_numpy(), R.to_numpy(), metric)
        assert title == sub.name.iloc[0]
        diff = np.abs(got - want.to_numpy())
        if metric == "euclidean":
            # proxyC loses the Ngn3-high-EP self-pair to cancellation (NaN -> 0)
            assert (diff > 1e-6).sum() == 1 and diff.max() == pytest.approx(1.0)
            assert np.sort(diff.ravel())[-2] < 1e-7
        else:
            assert diff.max() < 1e-9, metric


def test_hamman_spelling_is_accepted():
    x = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0]])
    a, _ = cell_similarity(x, x, "hamman")
    b, name = cell_similarity(x, x, "hamann")
    np.testing.assert_array_equal(a, b)
    assert name == "Hamann similarity"


# --------------------------------------------------------------------------- #
# enrichment
# --------------------------------------------------------------------------- #
def test_ora_reproduces_run_enrichment():
    t2g = pd.read_csv(FX / "enrich_term2gene.csv")
    t2n = pd.read_csv(FX / "enrich_term2name.csv")
    fs = _meta("group_hclust.csv")
    groups = pd.Series(pd.Categorical(fs.feature_split, categories=["C1", "C2", "C3", "C4"]))
    py = E.ora(t2g, t2n)(list(fs.index), groups)
    r = pd.read_csv(FX / "enrich_table.csv")
    assert (py[["ID", "Groups"]].to_numpy() == r[["ID", "Groups"]].to_numpy()).all()
    for col in ("GeneRatio", "BgRatio", "geneID", "Description", "Count"):
        assert (py[col].astype(str).to_numpy() == r[col].astype(str).to_numpy()).all(), col
    for col in ("pvalue", "p.adjust"):
        np.testing.assert_allclose(py[col], r[col], rtol=1e-9)


def test_word_boxes_match_r():
    r = pd.read_csv(FX / "enrich_table.csv")
    df, metric = E.select_enrichment(r, db="custom", padjust_cutoff=1)
    boxes = E.enrichment_boxes(df, metric, kinds=["terms", "keys", "features"])["custom"]
    want = pd.read_csv(FX / "enrich_boxes.csv")
    for (kind, group), sub in want.groupby(["kind", "group"], sort=False):
        got = boxes[kind][group]
        assert list(got.keyword) == list(sub.keyword), (kind, group)
        assert list(got.col) == list(sub.col), (kind, group)
        np.testing.assert_allclose(got.fontsize.to_numpy(float), sub.fontsize.to_numpy(float))


def test_words_excluded_is_scps_list():
    assert tuple(pd.read_csv(FX / "words_excluded.csv").word) == E.WORDS_EXCLUDED


def test_mestimate_matches_mfuzz():
    """Mfuzz's fuzzifier estimate on SCP's standardised mat_split."""
    r = json.loads((FX / "mfuzz_mestimate.json").read_text())
    mat = _wide("group_mat_split.csv").to_numpy()
    assert C.mestimate(C.standardise(mat)) == pytest.approx(r["mestimate"], rel=1e-12)
