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


def test_unimplemented_enrichment_types_raise_with_a_pointer():
    df = pd.DataFrame({"ID": ["a"], "Description": ["x"], "p.adjust": [0.01],
                       "Count": [1], "Groups": ["g"], "Database": ["d"]})
    for t in ("network", "enrichmap", "wordcloud"):
        with pytest.raises(NotImplementedError, match="trajectory_enrichment.md"):
            scp.pl.enrichment_plot(df, plot_type=t)


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
