"""Structural tests for ``cell_stat_plot`` / ``stat_plot``.

Spec: docs/porting_briefs/stat_plots.md section 2. The aggregation core is a
complete cross-tab; every plot_type is a view of it, so the table is what the
tests pin down.
"""

import pandas as pd
import pytest

import scp


@pytest.mark.parametrize("plot_type", ["bar", "trend", "area", "dot", "rose", "ring", "pie"])
def test_every_cartesian_and_polar_type_draws(adata, plot_type):
    fig = scp.pl.cell_stat_plot(adata, "leiden", group_by="batch", plot_type=plot_type)
    ax = fig.axes[0]
    assert ax.patches or ax.collections or ax.lines


def test_percent_sums_to_one_within_each_group(adata):
    fig = scp.pl.cell_stat_plot(adata, "leiden", group_by="batch", plot_type="bar")
    # stacked bars reach exactly 1.0 in every group
    tops = {}
    for p in fig.axes[0].patches:
        x = round(p.get_x() + p.get_width() / 2, 3)
        tops[x] = max(tops.get(x, 0.0), p.get_y() + p.get_height())
    assert all(abs(v - 1.0) < 1e-6 for v in tops.values())


def test_count_mode_totals_the_group_size(adata):
    fig = scp.pl.cell_stat_plot(adata, "leiden", group_by="batch",
                                plot_type="bar", stat_type="count")
    tops = {}
    for p in fig.axes[0].patches:
        x = round(p.get_x() + p.get_width() / 2, 3)
        tops[x] = max(tops.get(x, 0.0), p.get_y() + p.get_height())
    expected = sorted(adata.obs["batch"].value_counts().tolist())
    assert sorted(round(v) for v in tops.values()) == expected


def test_first_level_stacks_on_top(adata):
    """ggplot's position_stack fills top-down; the first level is the top band.

    Regression: accumulating in declared order flips the figure upside down.
    """
    fig = scp.pl.cell_stat_plot(adata, "leiden", group_by="batch", plot_type="bar")
    first = str(adata.obs["leiden"].cat.categories[0])
    colors = scp.discrete_palette([str(c) for c in adata.obs["leiden"].cat.categories],
                                  palette="Paired")
    from matplotlib.colors import to_hex
    xs = sorted({round(p.get_x() + p.get_width() / 2, 3) for p in fig.axes[0].patches})
    top = max((p for p in fig.axes[0].patches
               if abs(p.get_x() + p.get_width() / 2 - xs[0]) < 1e-6),
              key=lambda p: p.get_y() + p.get_height())
    assert to_hex(top.get_facecolor()).lower() == colors[first].lower()


def test_split_by_gives_one_panel_per_level(adata):
    fig = scp.pl.cell_stat_plot(adata, "leiden", group_by="batch", split_by="batch")
    assert len(fig.axes) == adata.obs["batch"].nunique()


def test_dot_puts_the_category_on_y(adata):
    """plot_type='dot' is a matrix of group x category, not a value axis."""
    fig = scp.pl.cell_stat_plot(adata, "leiden", group_by="batch", plot_type="dot")
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels == [str(c) for c in adata.obs["leiden"].cat.categories]


def test_na_stat_keeps_missing_as_its_own_level(adata):
    ad = adata.copy()
    vals = ad.obs["leiden"].astype(str).to_numpy().copy()
    vals[:50] = None
    ad.obs["withna"] = pd.Categorical(vals)
    fig = scp.pl.cell_stat_plot(ad, "withna", group_by="batch", na_stat=True)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "NA" in labels


def test_several_stat_by_requires_a_set_type(adata):
    with pytest.raises(ValueError, match="sankey"):
        scp.pl.cell_stat_plot(adata, ["leiden", "batch"], plot_type="bar")


def test_upset_counts_intersections(adata):
    fig = scp.pl.cell_stat_plot(adata, ["leiden", "batch"], plot_type="upset",
                                stat_level={"leiden": "A", "batch": "b1"})
    heights = [p.get_height() for p in fig.axes[0].patches]
    assert heights and all(h > 0 for h in heights)
    assert sum(heights) <= adata.n_obs


def test_unimplemented_types_raise_with_a_pointer(adata):
    for t in ("sankey", "chord"):
        with pytest.raises(NotImplementedError, match="stat_plots.md"):
            scp.pl.cell_stat_plot(adata, ["leiden", "batch"], plot_type=t)
