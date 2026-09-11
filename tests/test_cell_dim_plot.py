"""Structural assertions for the reference implementation.

The pattern here is the one every future plot function should copy: assert the
*shape* of what was produced, not its pixels.
"""

import pytest

import scp


def test_single_panel(adata):
    fig = scp.pl.cell_dim_plot(adata, "leiden")
    assert len(fig.axes) == 1


def test_one_panel_per_split_level(adata):
    fig = scp.pl.cell_dim_plot(adata, "leiden", split_by="batch")
    assert len(fig.axes) == adata.obs["batch"].nunique()


def test_split_panels_share_axis_limits(adata):
    # SCP locks limits to the global embedding range so panels are comparable.
    fig = scp.pl.cell_dim_plot(adata, "leiden", split_by="batch")
    a, b = fig.axes[0], fig.axes[1]
    assert a.get_xlim() == b.get_xlim()
    assert a.get_ylim() == b.get_ylim()


def test_every_split_panel_draws_all_cells(adata):
    # Defining behaviour of CellDimPlot: cells outside the split are masked to
    # bg_color, not dropped. feature_dim_plot is the one that partitions.
    fig = scp.pl.cell_dim_plot(adata, "leiden", split_by="batch")
    n = fig.axes[0].collections[0].get_offsets().shape[0]
    assert n == adata.n_obs


def test_multiple_group_by_gives_a_panel_each(adata):
    fig = scp.pl.cell_dim_plot(adata, ["leiden", "batch"])
    assert len(fig.axes) == 2


def test_label_writes_numbers_by_default(adata):
    fig = scp.pl.cell_dim_plot(adata, "leiden", label=True)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert set(texts) >= {"1", "2", "3", "4"}


def test_label_insitu_writes_names(adata):
    fig = scp.pl.cell_dim_plot(adata, "leiden", label=True, label_insitu=True)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert set(texts) >= set(adata.obs["leiden"].cat.categories)


def test_theme_blank_removes_the_frame(adata):
    fig = scp.pl.cell_dim_plot(adata, "leiden", theme=scp.theme_blank())
    ax = fig.axes[0]
    assert not any(ax.spines[s].get_visible() for s in ("top", "right", "bottom", "left"))
    assert ax.get_xticks().size == 0


def test_force_guard_raises_instead_of_prompting(adata):
    ad = adata.copy()
    ad.obs["many"] = [f"lvl{i}" for i in range(ad.n_obs)]
    with pytest.raises(ValueError, match="force=True"):
        scp.pl.cell_dim_plot(ad, "many")


@pytest.mark.parametrize(
    "kw", [{"add_density": True}, {"graph": "connectivities"}, {"stat_by": "batch"}]
)
def test_unimplemented_overlays_raise_with_a_pointer(adata, kw):
    with pytest.raises(NotImplementedError, match="milestones"):
        scp.pl.cell_dim_plot(adata, "leiden", **kw)


def test_declared_level_order_survives(adata):
    """A column's declared level order must drive the legend, not the order the
    values happen to appear in.

    Regression: the frame stored `as_ordered_categorical(...).to_numpy()`, which
    drops the Categorical, so the levels were re-derived from raw values further
    down and came back in first-appearance order. Every palette index and legend
    position shifted by that permutation.
    """
    import pandas as pd

    ad = adata.copy()
    declared = ["D", "C", "B", "A"]  # deliberately not the data's appearance order
    ad.obs["ordered"] = pd.Categorical(ad.obs["leiden"].astype(str), categories=declared)

    fig = scp.pl.cell_dim_plot(ad, "ordered")
    # show_stat is on by default, so labels read "D(141)"; compare the level only.
    labels = [t.get_text().split("(")[0] for t in fig.legends[0].get_texts()]
    assert labels == declared, f"legend followed appearance order, not declared: {labels}"
