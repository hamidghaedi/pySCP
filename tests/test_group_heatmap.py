"""Structural and numeric tests for the heatmap engine and ``group_heatmap``.

Spec: docs/porting_briefs/heatmaps.md.
"""

import numpy as np
import pytest

import scp
from scp.heatmap.render import _repel, _slices


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


def test_split_by_produces_one_panel_per_level(adata):
    r = scp.pl.group_heatmap(adata, features=_genes(adata), group_by="leiden",
                             split_by="batch", layer=None)
    assert len(r.matrices) == adata.obs["batch"].nunique()


def test_columns_follow_declared_group_order(adata):
    r = scp.pl.group_heatmap(adata, features=_genes(adata), group_by="leiden", layer=None)
    declared = list(adata.obs["leiden"].cat.categories)
    assert list(r.matrices["leiden"].columns) == declared


def test_zscore_rows_are_centred(adata):
    """matrix_process("zscore") is row-wise with ddof=1."""
    r = scp.pl.group_heatmap(adata, features=_genes(adata), group_by="leiden", layer=None)
    m = r.matrices["leiden"].to_numpy()
    assert np.allclose(np.nanmean(m, axis=1), 0, atol=1e-8)


def test_n_split_labels_every_feature(adata):
    r = scp.pl.group_heatmap(adata, features=_genes(adata), group_by="leiden",
                             layer=None, n_split=2)
    assert r.feature_split is not None
    assert set(r.feature_split.unique()) <= {"Cluster 1", "Cluster 2"}
    assert len(r.feature_split) == 6


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


def test_unimplemented_options_raise_with_a_pointer(adata):
    for kw in ({"anno_terms": True}, {"grouping_var": "batch"}, {"cluster_column_slices": True}):
        with pytest.raises(NotImplementedError, match="heatmaps.md"):
            scp.pl.group_heatmap(adata, features=_genes(adata), group_by="leiden",
                                 layer=None, **kw)
