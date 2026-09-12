"""Tests for volcano_plot, feature_cor_plot and cell_density_plot.

Spec: docs/porting_briefs/stat_plots.md sections 3-5.
"""

import numpy as np
import pandas as pd
import pytest

import scp


@pytest.fixture
def de_table():
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame({
        "gene": [f"g{i}" for i in range(n)],
        "group1": pd.Categorical(rng.choice(["A", "B"], n), categories=["A", "B"]),
        "group2": "others",
        "avg_log2FC": rng.normal(0, 2, n),
        "p_val": rng.uniform(1e-40, 1, n),
        "p_val_adj": rng.uniform(1e-30, 1, n),
        "pct.1": rng.uniform(0, 1, n),
        "pct.2": rng.uniform(0, 1, n),
    })


def test_volcano_one_panel_per_group(de_table):
    fig = scp.pl.volcano_plot(de_table)
    panels = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    assert len(panels) == 2


def test_volcano_y_is_signed(de_table):
    """Down-regulated genes hang BELOW zero -- this is not a standard volcano."""
    fig = scp.pl.volcano_plot(de_table)
    ax = [a for a in fig.axes if a.get_label() != "<colorbar>"][0]
    ys = np.concatenate([c.get_offsets()[:, 1] for c in ax.collections if len(c.get_offsets())])
    assert ys.min() < 0 < ys.max()


def test_volcano_y_axis_reads_as_magnitude(de_table):
    """scale_y_continuous(labels=abs): the axis shows magnitude both ways."""
    fig = scp.pl.volcano_plot(de_table)
    ax = [a for a in fig.axes if a.get_label() != "<colorbar>"][0]
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
    assert all(not lab.startswith("-") and "−" not in lab for lab in labels)


def test_volcano_reports_missing_detection_fractions(de_table):
    """scanpy omits pct unless pts=True; an all-NaN x must say so, not draw
    an empty panel."""
    t = de_table.copy()
    t["pct.1"] = np.nan
    t["pct.2"] = np.nan
    with pytest.raises(ValueError, match="pts=True"):
        scp.pl.volcano_plot(t)


def test_volcano_threshold_is_an_expression(de_table):
    fig = scp.pl.volcano_plot(de_table, de_threshold="avg_log2FC > 3 and p_val_adj < 0.01")
    assert fig is not None


def test_cor_plot_is_a_full_splom(adata):
    fig = scp.pl.feature_cor_plot(adata, ["g0", "g1", "g2"], group_by="leiden")
    panels = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    assert len(panels) == 9  # n^2


def test_cor_plot_guards_above_ten_features(adata):
    with pytest.raises(ValueError, match="force=True"):
        scp.pl.feature_cor_plot(adata, [f"g{i}" for i in range(11)])


def test_spearman_is_pearson_on_ranks(adata):
    """proxyC::simil ranks the rows then takes Pearson."""
    x = np.asarray(adata[:, "g0"].X).ravel()
    y = np.asarray(adata[:, "g1"].X).ravel()
    from scipy.stats import rankdata, spearmanr
    assert np.isclose(np.corrcoef(rankdata(x), rankdata(y))[0, 1], spearmanr(x, y).statistic)


def test_density_default_order_is_reversed(adata):
    """Level 1 sits at the TOP of the ridge stack."""
    fig = scp.pl.cell_density_plot(adata, "g0", group_by="leiden")
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels == list(reversed([str(c) for c in adata.obs["leiden"].cat.categories]))


def test_density_decreasing_sorts_by_median(adata):
    fig = scp.pl.cell_density_plot(adata, "g0", group_by="leiden", decreasing=True)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    meds = [float(np.median(np.asarray(adata[adata.obs["leiden"] == lv, "g0"].X)))
            for lv in labels]
    assert meds == sorted(meds, reverse=True)


def test_density_rank_mode_spreads_values(adata):
    fig = scp.pl.cell_density_plot(adata, "g0", group_by="leiden", x_order="rank")
    assert fig.axes[0].get_xlim()[1] > adata.n_obs * 0.5
