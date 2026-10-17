"""Tests for the trajectory, projection, enrichment and dynamic families.

Specs: docs/porting_briefs/trajectory_enrichment.md.
"""

import numpy as np
import pandas as pd
import pytest

import scp
from scp.pl._traj import shorten_segments


def test_shorten_segments_is_pure_geometry():
    x, y, xe, ye = shorten_segments(np.array([0.0]), np.array([0.0]),
                                    np.array([10.0]), np.array([0.0]),
                                    shorten_start=1.0, shorten_end=2.0)
    assert x[0] == pytest.approx(1.0)
    assert xe[0] == pytest.approx(8.0)


def test_offset_displaces_perpendicular():
    x, y, xe, ye = shorten_segments(np.array([0.0]), np.array([0.0]),
                                    np.array([10.0]), np.array([0.0]), offset=1.0)
    assert y[0] == pytest.approx(1.0) and ye[0] == pytest.approx(1.0)


def test_graph_plot_draws_net_transitions_only(adata):
    """Only the net flow between a pair is drawn, as one arrow."""
    node = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "g": ["a", "b"]},
                        index=["a", "b"])
    edge = np.array([[0.0, 1.0], [1.0, 0.0]])
    trans = np.array([[0.0, 0.9], [0.2, 0.0]])   # net favours a -> b
    ax = scp.pl.graph_plot(node, edge, transition=trans, node_group="g")
    assert len(ax.collections) >= 1  # edges + nodes drawn
    # annotate() hangs its arrow off the Annotation, not the axes
    arrows = [a for a in ax.texts if getattr(a, "arrow_patch", None) is not None]
    assert len(arrows) == 1


def test_graph_plot_rejects_mismatched_edge(adata):
    node = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}, index=["a", "b"])
    with pytest.raises(ValueError, match="square"):
        scp.pl.graph_plot(node, np.zeros((3, 3)))


def test_enrichment_plot_requires_the_column_contract():
    with pytest.raises(ValueError, match="ENRICHMENT_COLUMNS"):
        scp.pl.enrichment_plot(pd.DataFrame({"foo": [1]}))


def test_enrichment_plot_filters_by_padjust():
    df = pd.DataFrame({"ID": ["a", "b"], "Description": ["x", "y"],
                       "p.adjust": [0.001, 0.9], "Count": [5, 5],
                       "Groups": ["g", "g"], "Database": ["d", "d"]})
    fig = scp.pl.enrichment_plot(df, padjust_cutoff=0.05)
    assert len(fig.axes[0].patches) == 1


def test_enrichment_plot_raises_when_nothing_survives():
    df = pd.DataFrame({"ID": ["a"], "Description": ["x"], "p.adjust": [0.9],
                       "Count": [1], "Groups": ["g"], "Database": ["d"]})
    with pytest.raises(ValueError, match="no terms"):
        scp.pl.enrichment_plot(df)


def test_gsea_scores_running_score_returns_to_zero():
    """A weighted KS walk ends at Phit - Pmiss = 0 by construction."""
    rng = np.random.default_rng(0)
    gl = pd.Series(rng.normal(size=200),
                   index=[f"g{i}" for i in range(200)]).sort_values(ascending=False)
    cur = scp.pl.gsea_scores(gl, [f"g{i}" for i in range(0, 60, 3)])
    assert cur["running_score"].iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_adjust_layout_separates_overlapping_boxes():
    layout = np.array([[0.0, 0.0], [0.05, 0.0]])
    out = scp.pl.adjust_layout(layout, np.array([1.0, 1.0]), edges=[], scale=1.0)
    assert abs(out[1, 0] - out[0, 0]) > abs(layout[1, 0] - layout[0, 0])


def test_velocity_plot_names_its_extra(adata):
    ad = adata.copy()
    with pytest.raises(KeyError, match="velocity_embedding"):
        scp.pl.velocity_plot(ad, basis="umap")


# --------------------------------------------------------------------------- #
# the trajectory layers: drawn into a borrowed axes, legends collected there
# --------------------------------------------------------------------------- #
def _with_pseudotime(adata):
    ad = adata.copy()
    x = ad.obsm["X_umap"][:, 0]
    ad.obs["pt"] = (x - x.min()) / (x.max() - x.min()) * 10
    ad.obs.loc[ad.obs["leiden"] == "D", "pt"] = np.nan
    v = ad.obsm["X_umap"]
    ad.obsm["velocity_umap"] = 0.05 * np.column_stack([-(v[:, 1] - v[:, 1].mean()),
                                                      v[:, 0] - v[:, 0].mean()])
    return ad


def test_trajectory_layers_draw_into_a_borrowed_panel(adata):
    import warnings

    from matplotlib.collections import PathCollection

    from scp.layout import LegendColumn

    ad = _with_pseudotime(adata)
    fig = scp.pl.cell_dim_plot(ad, "leiden")
    ax = fig.axes[0]
    n_cells = len([c for c in ax.collections if isinstance(c, PathCollection)])
    xlabel = ax.get_xlabel()
    leg = LegendColumn()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # skmisc may be missing
        scp.pl.lineage_plot(ad, ["pt"], reduction="X_umap", ax=ax, legend=leg)
    scp.pl.velocity_plot(ad, basis="umap", group_by="leiden", ax=ax, legend=leg)
    # neither layer adds a cell scatter, and the host panel keeps its labels
    assert len([c for c in ax.collections if isinstance(c, PathCollection)]) == n_cells
    assert ax.get_xlabel() == xlabel
    assert [t for t, _h in leg.entries] == ["Lineages", "leiden"]
    assert [h.get_label() for h in leg.entries[1][1]] == list(ad.obs["leiden"].cat.categories)


def test_velocity_stream_runs_without_scvelo(adata):
    ad = _with_pseudotime(adata)
    ax = scp.pl.velocity_plot(ad, basis="umap", plot_type="stream", density=0.5)
    cbars = [a for a in ax.figure.axes if a.get_label() == "<colorbar>"]
    assert len(cbars) == 1 and cbars[0].get_ylabel() == "Velocity"
    ax.figure.savefig(__import__("io").BytesIO(), format="png")


def test_compute_velocity_on_grid_shapes(adata):
    ad = _with_pseudotime(adata)
    X, V = ad.obsm["X_umap"], ad.obsm["velocity_umap"]
    gx, gv = scp.pl.compute_velocity_on_grid(X, V, density=0.4)
    assert gx.shape == gv.shape and gx.shape[1] == 2 and len(gx) <= 20 * 20
    sx, sv = scp.pl.compute_velocity_on_grid(X, V, density=0.4, adjust_for_stream=True)
    assert sx.shape == (2, 20) and sv.shape == (2, 20, 20)
    assert np.isnan(sv[0]).any() and not np.isnan(sv[1]).any()


def test_dynamic_plot_panel_layouts(adata):
    import warnings

    ad = _with_pseudotime(adata)
    ad.obs["pt2"] = ad.obs["pt"][::-1].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pygam may be missing
        f1 = scp.pl.dynamic_plot(ad, ["pt", "pt2"], ["g0", "g1"], group_by="leiden")
        f2 = scp.pl.dynamic_plot(ad, ["pt", "pt2"], ["g0", "g1"], compare_lineages=False)
        f3 = scp.pl.dynamic_plot(ad, ["pt", "pt2"], ["g0", "g1"], compare_lineages=False,
                                 compare_features=True)
        f4 = scp.pl.dynamic_plot(ad, ["pt", "pt2"], ["g0", "g1"], compare_features=True)
    # R's panel grid: one per feature, per (lineage, feature), per lineage, one
    assert [a.get_title() for a in f1.axes] == ["g0", "g1"]
    assert len(f2.axes) == 4
    assert len(f3.axes) == 2 and not any(a.get_title() for a in f3.axes)
    assert len(f4.axes) == 1
    # lines are coloured by feature there, so only the feature legend is drawn
    assert [lg.get_title().get_text() for lg in f4.legends] == ["Features:"]
    assert f1.axes[0].get_ylabel() == "log1p(normalized counts)"
    titles = [lg.get_title().get_text() for lg in f1.legends]
    assert titles == ["leiden:", "Lineages:"]


def test_trajectory_examples_exist():
    from pathlib import Path

    fx = Path(__file__).parent / "fixtures"
    for name in ("lineage_plot", "paga_plot", "velocity_plot_stream", "dynamic_plot"):
        assert (fx / f"example_{name}.png").exists()
