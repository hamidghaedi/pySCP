"""Structural tests for ``feature_stat_plot`` (R: ``FeatureStatPlot``).

Spec: docs/porting_briefs/stat_plots.md section 1.
"""

import numpy as np
import pytest

import scp


def test_one_panel_per_feature(adata):
    fig = scp.pl.feature_stat_plot(adata, ["g0", "g1", "g2"], group_by="leiden")
    assert len(fig.axes) == 3


@pytest.mark.parametrize("plot_type", ["violin", "box", "bar", "dot", "col"])
def test_every_plot_type_draws(adata, plot_type):
    fig = scp.pl.feature_stat_plot(adata, "g0", group_by="leiden", plot_type=plot_type)
    ax = fig.axes[0]
    assert ax.collections or ax.patches or ax.lines


def test_x_order_follows_declared_levels_not_appearance(adata):
    import pandas as pd

    ad = adata.copy()
    declared = ["D", "C", "B", "A"]
    ad.obs["ordered"] = pd.Categorical(ad.obs["leiden"].astype(str), categories=declared)
    fig = scp.pl.feature_stat_plot(ad, "g0", group_by="ordered")
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == declared


def test_sort_reorders_x_by_median(adata):
    fig = scp.pl.feature_stat_plot(adata, "g0", group_by="leiden", sort="increasing")
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    medians = [
        float(np.median(np.asarray(adata[adata.obs["leiden"] == lv, "g0"].X).ravel()))
        for lv in labels
    ]
    assert medians == sorted(medians)


def test_default_striping_is_drawn_without_bg_by(adata):
    """With no bg_by, R still stripes by group_by in alternating grey85."""
    fig = scp.pl.feature_stat_plot(adata, "g0", group_by="leiden")
    stripes = [p for p in fig.axes[0].patches if getattr(p, "get_width", None)]
    assert stripes, "expected alternating background stripes"


def test_bg_by_must_be_coarser_than_group_by(adata):
    """R: \"'group.by' must be a part of 'bg.by'\" -- a group may not straddle two."""
    with pytest.raises(ValueError, match="part of"):
        scp.pl.feature_stat_plot(adata, "g0", group_by="leiden", bg_by="batch")


def test_same_y_lims_shares_scale_across_features(adata):
    fig = scp.pl.feature_stat_plot(adata, ["g0", "g1"], group_by="leiden", same_y_lims=True)
    assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()


def test_split_by_dodges_within_group(adata):
    fig = scp.pl.feature_stat_plot(adata, "g0", group_by="leiden", split_by="batch",
                                   plot_type="bar")
    xs = sorted({round(float(p.get_x() + p.get_width() / 2), 3) for p in fig.axes[0].patches})
    assert len(xs) >= adata.obs["leiden"].nunique() * 2


def test_unimplemented_features_raise_with_a_pointer(adata):
    for kw in ({"stack": True}, {"plot_by": "feature"}, {"comparisons": True},
               {"add_trend": True}, {"individual": True}):
        with pytest.raises(NotImplementedError, match="stat_plots.md"):
            scp.pl.feature_stat_plot(adata, "g0", group_by="leiden", **kw)
