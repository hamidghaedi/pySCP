"""Structural tests for ``feature_dim_plot`` (R original: ``FeatureDimPlot``).

Asserts the behaviours docs/porting_briefs/dim_plots.md section 2 calls out as
defining, rather than pixels.
"""

import numpy as np
import pytest

import scp


def _panels(fig):
    return [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]


def test_one_panel_per_feature(adata):
    fig = scp.pl.feature_dim_plot(adata, ["g0", "g1", "g2"])
    assert len(_panels(fig)) == 3


def test_split_partitions_rather_than_masking(adata):
    """Unlike cell_dim_plot, a split panel holds ONLY its own cells."""
    fig = scp.pl.feature_dim_plot(adata, "g0", split_by="batch", bg_cutoff=None)
    panels = _panels(fig)
    assert len(panels) == adata.obs["batch"].nunique()
    drawn = sum(c.get_offsets().shape[0] for ax in panels for c in ax.collections)
    assert drawn == adata.n_obs  # partitioned, so each cell appears exactly once


def test_split_panels_share_axis_limits(adata):
    fig = scp.pl.feature_dim_plot(adata, "g0", split_by="batch")
    panels = _panels(fig)
    assert panels[0].get_xlim() == panels[1].get_xlim()
    assert panels[0].get_ylim() == panels[1].get_ylim()


def test_bg_cutoff_masks_at_or_below_zero(adata):
    """bg_cutoff=0 is <=, so exact zeros are background too."""
    ad = adata.copy()
    ad.X = np.asarray(ad.X).copy()
    ad.X[: ad.n_obs // 2, 0] = 0.0
    fig = scp.pl.feature_dim_plot(ad, ad.var_names[0])
    title = _panels(fig)[0].get_title()
    assert f"nPos:{ad.n_obs - ad.n_obs // 2}" in title


def test_bg_cutoff_none_keeps_signed_values(adata):
    """Signed scores need bg_cutoff=None or half the cells vanish."""
    fig = scp.pl.feature_dim_plot(adata, "score", bg_cutoff=None)
    assert f"nPos:{adata.n_obs}" in _panels(fig)[0].get_title()


def test_draw_order_is_nan_first_then_ascending(adata):
    """Highest expressers draw last so they land on top."""
    fig = scp.pl.feature_dim_plot(adata, "score", bg_cutoff=None)
    coll = _panels(fig)[0].collections[-1]
    drawn = coll.get_array()
    assert np.all(np.diff(np.asarray(drawn)) >= 0)


def test_force_guard_fires_above_fifty_features(adata):
    """The R code guards at >50, despite its roxygen text saying 100."""
    with pytest.raises(ValueError, match="force=True"):
        # The guard counts requested names and fires before resolution,
        # so these need not all exist in the fixture's 30 genes.
        scp.pl.feature_dim_plot(adata, [f"g{i}" for i in range(51)])


def test_compare_features_blends_into_one_panel(adata):
    fig = scp.pl.feature_dim_plot(adata, ["g0", "g1"], compare_features=True)
    assert len(_panels(fig)) == 1


def test_named_features_become_subtitles(adata):
    fig = scp.pl.feature_dim_plot(adata, {"markers": ["g0", "g1"]})
    assert "markers" in _panels(fig)[0].get_title()
