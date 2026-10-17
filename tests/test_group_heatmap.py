"""Structural and numeric tests for the heatmap engine and ``group_heatmap``.

Spec: docs/porting_briefs/heatmaps.md.
"""

import numpy as np
import pandas as pd
import pytest

import scp
from scp.heatmap.render import _repel, _slices


@pytest.fixture(autouse=True)
def _close_figures():
    import matplotlib.pyplot as plt
    yield
    plt.close("all")


def _genes(adata, n=6):
    return list(adata.var_names[:n])


def test_figsize_is_the_sum_of_its_declared_parts():
    """The size engine is the point: ratios are inches and must add up."""
    import pandas as pd
    from matplotlib.colors import Normalize

    from scp.heatmap.spec import HeatmapSpec, PanelSpec, Track

    panel = PanelSpec(name="p", matrix=np.zeros((4, 3)), row_labels=pd.Index(list("abcd")),
                      col_labels=pd.Index(list("xyz")), cmap=None, norm=Normalize(0, 1),
                      body_size_in=2.0)
    spec = HeatmapSpec(panels=[panel], body_height_in=3.0,
                       left_tracks=[Track(name="t", side="left", kind="simple", size_in=0.25)])
    w, h = spec.figsize()
    assert w == pytest.approx(2.0 + 0.25)
    assert h == pytest.approx(3.0 + spec.measured.get("titles_h", 0.25))


def test_slices_follow_declared_level_order():
    import pandas as pd

    split = pd.Categorical(["b", "a", "b", "a"], categories=["b", "a"])
    assert [lab for lab, _ in _slices(split, 4)] == ["b", "a"]


def test_repel_preserves_order_and_minimum_gap():
    placed = _repel(np.array([1.0, 1.1, 1.2]), min_gap=1.0, lo=0.0, hi=10.0)
    assert np.all(np.diff(placed) >= 1.0 - 1e-9)
    assert np.all(np.diff(placed) > 0)


def test_returns_one_matrix_per_panel(adata):
    r = scp.pl.group_heatmap(adata, features=_genes(adata), group_by="leiden", layer=None)
    assert list(r.matrices) == ["leiden"]
    m = r.matrices["leiden"]
    assert m.shape == (6, adata.obs["leiden"].nunique())


def test_split_by_gives_group_split_columns_in_one_panel(adata):
    """R: one panel per group_by, columns "group : split", sliced by group."""
    r = scp.pl.group_heatmap(adata, features=_genes(adata), group_by="leiden",
                             split_by="batch", layer=None)
    assert list(r.matrices) == ["leiden"]
    cols = list(r.matrices["leiden"].columns)
    assert cols == [f"{g} : {b}" for g in "ABCD" for b in ("b1", "b2")]
    panel = r.spec.panels[0]
    assert list(panel.col_split.categories) == list("ABCD")
    assert [t.name for t in panel.top_tracks] == ["leiden", "batch"]


def test_columns_follow_declared_group_order(adata):
    r = scp.pl.group_heatmap(adata, features=_genes(adata), group_by="leiden", layer=None)
    declared = list(adata.obs["leiden"].cat.categories)
    assert list(r.matrices["leiden"].columns) == declared


def test_zscore_rows_are_centred(adata):
    """matrix_process("zscore") is row-wise with ddof=1."""
    r = scp.pl.group_heatmap(adata, features=_genes(adata), group_by="leiden", layer=None)
    m = r.matrices["leiden"].to_numpy()
    assert np.allclose(np.nanmean(m, axis=1), 0, atol=1e-8)


def test_n_split_names_clusters_like_r(adata):
    r = scp.pl.group_heatmap(adata, features=_genes(adata), group_by="leiden",
                             layer=None, n_split=2, split_method="hclust")
    assert r.feature_split is not None
    assert list(r.feature_split.cat.categories) == ["C1", "C2"]
    assert len(r.feature_split) == 6
    # the slices carry the member count, "C1(4)"
    counts = r.feature_split.value_counts()
    assert list(r.spec.row_split.categories) == [f"C{i}({counts[f'C{i}']})" for i in (1, 2)]


def test_library_size_uses_the_whole_transcriptome(adata):
    """Regression: summing only the selected features makes most cells zero,
    and the entire matrix comes back NaN."""
    r = scp.pl.group_heatmap(adata, features=_genes(adata, 3), group_by="leiden",
                             layer="counts", lib_normalize=True)
    assert np.isfinite(r.matrices["leiden"].to_numpy()).all()


def test_add_dot_sizes_the_body_cell_to_the_dot(adata):
    """With add_dot the cell IS the dot at 100%, else the dots overflow."""
    r = scp.pl.group_heatmap(adata, features=_genes(adata), group_by="leiden",
                             layer=None, add_dot=True, dot_size_mm=8.0)
    panel = r.spec.panels[0]
    assert panel.body_size_in == pytest.approx(8.0 / 25.4 * adata.obs["leiden"].nunique())


def test_group_colour_track_is_always_present(adata):
    """GroupHeatmap always puts a group colour bar above the body."""
    r = scp.pl.group_heatmap(adata, features=_genes(adata), group_by="leiden", layer=None)
    assert [t.name for t in r.spec.panels[0].top_tracks] == ["leiden"]


def test_grouping_var_forces_log2fc_and_disables_dots(adata):
    with pytest.warns(UserWarning, match="log2fc"):
        r = scp.pl.group_heatmap(adata, features=_genes(adata), group_by="leiden",
                                 layer=None, grouping_var="batch", numerator="b2",
                                 add_dot=True)
    assert not any(lay.kind == "dot" for lay in r.spec.panels[0].layers)
    assert r.spec.legends[0].title.startswith("b2/other\nlog2fc")
    with pytest.raises(ValueError, match="different from"):
        scp.pl.group_heatmap(adata, features=_genes(adata), group_by="leiden", layer=None,
                             grouping_var="batch", split_by="batch", exp_method="log2fc")


def test_explicit_feature_split_and_split_order(adata):
    genes = _genes(adata)
    lab = ["x", "y", "x", "z", "y", "x"]
    r = scp.pl.group_heatmap(adata, features=genes, group_by="leiden", layer=None,
                             feature_split=lab)
    assert list(r.spec.row_split.categories) == ["x", "y", "z"]
    as_map = dict(zip(genes, lab, strict=True))
    r2 = scp.pl.group_heatmap(adata, features=genes, group_by="leiden", layer=None,
                              feature_split=as_map)
    assert list(r2.feature_split) == lab
    r3 = scp.pl.group_heatmap(adata, features=genes, group_by="leiden", layer=None,
                              n_split=3, split_method="hclust", split_order=[3, 1, 2])
    base = scp.pl.group_heatmap(adata, features=genes, group_by="leiden", layer=None,
                                n_split=3, split_method="hclust")
    # split_order permutes which raw cluster becomes C1..C3
    assert sorted(r3.feature_split.value_counts()) == sorted(base.feature_split.value_counts())


def test_mfuzz_split_uses_scikit_fuzzy(adata):
    pytest.importorskip("skfuzzy")
    ad = adata.copy()
    X = np.asarray(ad.X, dtype=float).copy()
    lab = np.asarray(ad.obs["leiden"])
    for k, g in enumerate("ABC"):  # three blocks of genes, each peaking in one group
        X[np.ix_(lab == g, np.arange(4 * k, 4 * k + 4))] += 6.0
    ad.X = X
    r = scp.pl.group_heatmap(ad, features=list(ad.var_names[:12]), group_by="leiden",
                             layer=None, n_split=3, split_method="mfuzz")
    assert set(r.feature_split.cat.categories) == {"C1", "C2", "C3"}
    # clusters are named by the group they peak in: A's block becomes C1
    assert list(r.feature_split.iloc[[0, 4, 8]]) == ["C1", "C2", "C3"]


def test_cluster_rows_draws_one_tree_per_slice(adata):
    r = scp.pl.group_heatmap(adata, features=_genes(adata, 12), group_by="leiden",
                             layer=None, n_split=3, split_method="hclust", cluster_rows=True)
    d = r.spec.row_dend
    assert d is not None and len(d.slice_linkages) == 3


def test_cluster_column_slices_needs_two_groups(adata):
    sub = adata[adata.obs["leiden"] == "A"].copy()
    with pytest.raises(ValueError, match="only one group"):
        scp.pl.group_heatmap(sub, features=_genes(adata), group_by="leiden", split_by="batch",
                             layer=None, exp_method="raw", cluster_column_slices=True)


def test_feature_annotation_goes_on_the_right(adata):
    ad = adata.copy()
    ad.var["kind"] = ["TF" if i % 2 else "other" for i in range(ad.n_vars)]
    r = scp.pl.group_heatmap(ad, features=_genes(ad), group_by="leiden", layer=None,
                             feature_annotation=["kind"])
    assert [t.name for t in r.spec.right_tracks] == ["kind"]
    assert "kind" in r.var_meta.columns


def test_marks_follow_scps_spacing(adata):
    genes = list(adata.var_names[:25])
    r = scp.pl.group_heatmap(adata, features=genes, group_by="leiden", layer=None, nlabel=5)
    mark = next(t for t in r.spec.left_tracks if t.kind == "mark")
    # ceiling((25 / 5) / 2) = 3 .. 25, five points, 1-based -> 0-based
    want = sorted({int(x) - 1 for x in np.round(np.linspace(3, 25, 5))})
    assert list(mark.at) == want


def _toy_enrichment(features, groups):
    rows = []
    for g in groups.cat.categories:
        members = [f for f, x in zip(features, groups, strict=True) if x == g]
        rows.append({"ID": f"T{g}", "Description": f"regulation of {g} signaling",
                     "GeneRatio": "1/2", "BgRatio": "3/9", "pvalue": 1e-4, "p.adjust": 1e-3,
                     "geneID": "/".join(members), "Count": len(members), "Groups": g,
                     "Database": "custom"})
    return pd.DataFrame(rows)


def test_enrichment_boxes_need_a_table(adata):
    with pytest.raises(ValueError, match="enrichment"):
        scp.pl.group_heatmap(adata, features=_genes(adata), group_by="leiden", layer=None,
                             n_split=2, anno_terms=True)
    with pytest.raises(ValueError, match="flip"):
        scp.pl.group_heatmap(adata, features=_genes(adata), group_by="leiden", layer=None,
                             n_split=2, anno_terms=True, enrichment_fn=_toy_enrichment,
                             flip=True)


def test_enrichment_boxes_are_right_tracks(adata):
    r = scp.pl.group_heatmap(adata, features=_genes(adata), group_by="leiden", layer=None,
                             n_split=2, split_method="hclust", anno_terms=True,
                             anno_keys=True, anno_features=True,
                             enrichment_fn=_toy_enrichment)
    boxes = [t for t in r.spec.right_tracks if t.kind == "textbox"]
    assert [t.title for t in boxes] == ["custom (terms)", "custom (keys)", "custom (features)"]
    # every box is preceded by a spacer and a split-coloured block, as in R
    kinds = [t.kind for t in r.spec.right_tracks]
    assert kinds == ["custom", "block", "textbox"] * 3
    keys = boxes[1].text["C1"]
    assert "regulation" not in set(keys.keyword)  # a stop word
    assert set(keys.keyword) == {"c1", "signaling"}
    assert set(r.enrichment["Groups"]) == {"C1", "C2"}


def test_unported_options_raise_with_a_pointer(adata):
    adata = adata.copy()
    adata.obs["pt"] = np.linspace(0, 1, adata.n_obs)
    # a categorical separate_annotation is drawn; fitted curves need R's fits
    r = scp.pl.dynamic_heatmap(adata, ["pt"], _genes(adata), layer=None,
                               separate_annotation=["leiden"])
    assert [t.name for t in r.spec.panels[0].top_tracks][-1] == "leiden\n(separate)"
    with pytest.raises(ValueError, match="dynamic_features"):
        scp.pl.dynamic_heatmap(adata, ["pt"], _genes(adata), layer=None,
                               separate_annotation=[_genes(adata)[:2]])
    with pytest.raises(NotImplementedError, match="milestone 9"):
        scp.pl.dynamic_heatmap(adata, ["pt"], _genes(adata), layer=None, use_fitted=True)
