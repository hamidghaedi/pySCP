"""Structural tests for ``feature_stat_plot`` (R: ``FeatureStatPlot``).

Spec: docs/porting_briefs/stat_plots.md section 1.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import scp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


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


def test_flip_puts_groups_on_y_first_level_on_top(adata):
    """coord_flip: values along x, the factor reversed up the y axis."""
    levels = list(scp.fetch.as_ordered_categorical(adata.obs["leiden"]).cat.categories)
    flat = scp.pl.feature_stat_plot(adata, "g0", group_by="leiden")
    fig = scp.pl.feature_stat_plot(adata, "g0", group_by="leiden", flip=True)
    ax, ax0 = fig.axes[0], flat.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == levels[::-1]
    assert ax.get_xlim() == pytest.approx(ax0.get_ylim())
    assert ax.get_ylim() == pytest.approx(ax0.get_xlim())
    assert ax.get_xlabel() == "Expression level" and ax.get_ylabel() == "leiden"


# --------------------------------------------------------------------------- #
# stack, plot_by="feature", individual, trend, coexp, significance
# --------------------------------------------------------------------------- #
def _panels(fig):
    return [ax for ax in fig.axes if ax.get_visible() and ax.get_label() != "<colorbar>"]


def test_stack_is_one_shared_x_column_with_two_ticks_per_row(adata):
    feats = ["g0", "g1", "g2", "g3"]
    fig = scp.pl.feature_stat_plot(adata, feats, group_by="leiden", stack=True)
    axes = _panels(fig)
    assert len(axes) == len(feats)
    top = axes[0]
    for ax in axes:
        assert len(ax.get_yticks()) == 2
        assert ax.get_xlim() == top.get_xlim()
        assert ax.get_ylabel() == ""
    # x tick labels only on the last row; strips carry the feature names
    for ax in axes[:-1]:
        assert not ax.xaxis.get_tick_params()["labelbottom"]
    assert [ax.texts[-1].get_text() for ax in axes] == feats
    assert fig._supylabel is not None
    assert fig._supylabel.get_text() == "Expression level"
    assert len(fig.legends) == 1


def test_stack_rows_have_their_own_y_limits(adata):
    ad = adata.copy()
    ad.X[:, 1] *= 10
    fig = scp.pl.feature_stat_plot(ad, ["g0", "g1"], group_by="leiden", stack=True)
    a0, a1 = _panels(fig)
    assert a1.get_ylim()[1] > 5 * a0.get_ylim()[1]


def test_plot_by_feature_puts_features_on_x_one_panel_per_group(adata):
    feats = ["g0", "g1", "g2"]
    fig = scp.pl.feature_stat_plot(adata, feats, group_by="leiden", plot_by="feature")
    axes = _panels(fig)
    assert len(axes) == adata.obs["leiden"].nunique()
    for ax in axes:
        assert [t.get_text() for t in ax.get_xticklabels()] == feats
    # R loops over unique(): panels follow first appearance in the cells
    appearance = list(dict.fromkeys(adata.obs["leiden"].astype(str)))
    assert [ax.get_title() for ax in axes] == appearance


def test_plot_by_feature_rejects_several_group_columns(adata):
    with pytest.raises(ValueError, match="length of 1"):
        scp.pl.feature_stat_plot(adata, ["g0", "g1"], group_by=["leiden", "batch"],
                                 plot_by="feature")


def test_individual_gives_one_panel_per_level_and_split(adata):
    fig = scp.pl.feature_stat_plot(adata, "g0", group_by="leiden", split_by="batch",
                                   individual=True)
    keys = list(fig._scp_stat_record)
    assert len(_panels(fig)) == len(keys) == 4 * 2
    # R's order: the group level varies fastest
    assert keys[0].endswith(":A:b1") and keys[1].endswith(":B:b1")


def test_add_trend_passes_through_the_group_medians(adata):
    fig = scp.pl.feature_stat_plot(adata, "g0", group_by="leiden", add_trend=True)
    rec = next(iter(fig._scp_stat_record.values()))
    meds = [float(np.median(np.asarray(adata[adata.obs["leiden"] == lv, "g0"].X).ravel()))
            for lv in adata.obs["leiden"].cat.categories]
    assert [y for _x, y in rec["trend"]] == pytest.approx(meds)
    assert any(len(line.get_xdata()) == 4 for line in fig.axes[0].lines)


def test_sort_reorders_x_but_keeps_each_groups_colour(adata):
    """R keys fill colours by level name, so sorting moves bars, not colours."""
    def colours(fig):
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_xticklabels()]
        bars = [p for p in ax.patches if type(p).__name__ == "Rectangle"
                and 0 < p.get_width() < 1]
        bars = sorted(bars, key=lambda b: b.get_x())
        return dict(zip(labels, [tuple(b.get_facecolor()) for b in bars], strict=True))

    plain = scp.pl.feature_stat_plot(adata, "g0", group_by="leiden", plot_type="bar")
    srt = scp.pl.feature_stat_plot(adata, "g0", group_by="leiden", plot_type="bar",
                                   sort="increasing")
    assert colours(plain) == colours(srt)


def test_same_y_lims_accepts_a_quantile_string(adata):
    """R errors here (it reads `values`, the pool is `valus`); the port does not."""
    fig = scp.pl.feature_stat_plot(adata, ["g0", "g1"], group_by="leiden",
                                   same_y_lims=True, y_max="q90")
    a0, a1 = _panels(fig)
    assert a0.get_ylim() == a1.get_ylim()
    pool = np.asarray(adata[:, ["g0", "g1"]].X).ravel()
    assert a0.get_ylim()[1] < pool.max()


def test_comparisons_true_needs_split_by(adata):
    with pytest.raises(ValueError, match="split.by"):
        scp.pl.feature_stat_plot(adata, "g0", group_by="leiden", comparisons=True)


def test_comparison_brackets_step_up_and_raise_the_ceiling(adata):
    comps = [("A", "B"), ("A", "C"), ("A", "D")]
    fig = scp.pl.feature_stat_plot(adata, "g0", group_by="leiden", comparisons=comps)
    rec = next(iter(fig._scp_stat_record.values()))
    tops = [b[2] for b in rec["brackets"]]
    steps = np.diff(tops)
    assert len(tops) == 3 and np.all(steps > 0)
    assert list(steps) == pytest.approx([steps[0]] * 2)
    assert rec["ylim"][1] > max(tops)
    assert all(b[4] in {"****", "***", "**", "*", "ns"} for b in rec["brackets"])


def test_ref_group_compares_every_other_level_against_it(adata):
    fig = scp.pl.feature_stat_plot(adata, "g0", group_by="leiden", ref_group="B",
                                   sig_label="p.format")
    rec = next(iter(fig._scp_stat_record.values()))
    assert [row[0] for row in rec["sig"]] == [("B", "A"), ("B", "C"), ("B", "D")]


def test_split_comparisons_label_each_group(adata):
    fig = scp.pl.feature_stat_plot(adata, "g0", group_by="leiden", split_by="batch",
                                   comparisons=True, sig_label="p.format")
    rec = next(iter(fig._scp_stat_record.values()))
    assert len(rec["texts"]) == 4
    assert all(t[2].startswith("p = ") for t in rec["texts"])


def test_col_drops_overlays_and_comparisons_with_a_warning(adata):
    with pytest.warns(UserWarning, match="Cannot add other layers"):
        scp.pl.feature_stat_plot(adata, "g0", group_by="leiden", plot_type="col",
                                 add_point=True)
    with pytest.warns(UserWarning, match="Cannot add comparison"):
        scp.pl.feature_stat_plot(adata, "g0", group_by="leiden", plot_type="col",
                                 multiplegroup_comparisons=True)


def test_calculate_coexp_adds_a_coexp_panel(adata):
    with pytest.warns(UserWarning, match="not used when calculating co-expression"):
        fig = scp.pl.feature_stat_plot(adata, ["g0", "g1", "score"], group_by="leiden",
                                       calculate_coexp=True)
    assert [k.split(":")[0] for k in fig._scp_stat_record] == ["g0", "g1", "score", "CoExp"]


def test_feature_cor_plot_calculate_coexp_appends_coexp(adata):
    fig = scp.pl.feature_cor_plot(adata, ["g0", "g1"], calculate_coexp=True)
    assert len([ax for ax in fig.axes if ax.get_label() != "<colorbar>"]) == 9


def test_expression_stat_plot_is_the_dataframe_engine():
    import pandas as pd

    df = pd.DataFrame({"v": np.arange(40.0), "grp": list("ab") * 20})
    fig = scp.pl.expression_stat_plot(df, "v", group_by="grp", plot_type="box")
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["a", "b"]
