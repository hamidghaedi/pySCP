"""Regressions from an end-to-end run of the dim and trajectory plots.

Each test names the failure it guards against: a named reduction that never
resolved, 3-D plots that picked a 2-D embedding, group columns that are not
strings losing their legend, overlay titles printed over the host panel,
the projection legends, and scanpy's stored colours being ignored.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.text import Annotation

import scp
from scp.pl import _dim_layers as layers


def _titles(fig):
    return [lg.get_title().get_text() for lg in fig.legends]


def _labels(fig, i=0):
    return [t.get_text() for t in fig.legends[i].get_texts()]


def _panels(fig):
    return [a for a in fig.axes if a.get_label() != "<colorbar>" and a.get_xlabel()]


# --------------------------------------------------------------------------- #
# 1, 2. reduction resolution
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["pca", "PCA", "X_pca", "Pca"])
def test_cell_dim_plot_resolves_a_named_reduction(adata, name):
    fig = scp.pl.cell_dim_plot(adata, "leiden", reduction=name)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "PC_1"
    x = ax.collections[0].get_offsets()[:, 0]
    assert np.allclose(np.sort(x), np.sort(adata.obsm["X_pca"][:, 0]))


@pytest.mark.parametrize("name", ["umap", "UMAP", "X_umap"])
def test_every_dim_and_trajectory_plot_resolves_a_named_reduction(adata, name):
    ad = adata.copy()
    ad.obsm["velocity_umap"] = np.ones((ad.n_obs, 2)) * 0.1
    ad.obs["pt"] = ad.obsm["X_umap"][:, 0] - ad.obsm["X_umap"][:, 0].min()
    assert _panels(scp.pl.feature_dim_plot(ad, "g0", reduction=name))[0].get_xlabel() == "UMAP_1"
    assert scp.pl.lineage_plot(ad, ["pt"], reduction=name).get_xlabel() == "UMAP_1"
    assert scp.pl.velocity_plot(ad, basis=name).get_xlabel() == "UMAP_1"
    ad.uns["paga"] = {"connectivities": np.eye(4) * 0 + 0.5, "groups": "leiden"}
    assert scp.pl.paga_plot(ad, reduction=name).get_xlabel() == "UMAP_1"
    fig = scp.pl.cell_dim_plot(ad, "leiden", reduction=name, velocity="velocity")
    assert [a for a in fig.axes[0].artists if isinstance(a, layers._ArrowField)]


def test_an_unmatched_reduction_warns_before_falling_back(adata):
    with pytest.warns(UserWarning, match="matches no embedding"):
        assert scp.default_reduction(adata, pattern="tsne") == "X_umap"


def test_3d_plots_default_to_a_3d_embedding(adata):
    pytest.importorskip("plotly")
    ad = adata.copy()
    ad.obsm["X_umap3d"] = np.random.default_rng(0).normal(size=(ad.n_obs, 3))
    # X_umap (2-D) wins at min_dim = 2; R uses min_dim = 3, which never
    # yields a 2-D embedding -- here R's exact "pca" outranks "umap3D"
    assert scp.default_reduction(ad) == "X_umap"
    assert scp.default_reduction(ad, min_dim=3) == "X_pca"
    del ad.obsm["X_pca"]
    assert scp.default_reduction(ad, min_dim=3) == "X_umap3d"
    fig = scp.pl.cell_dim_plot_3d(ad, "leiden")
    assert np.allclose(np.sort(np.concatenate([t.z for t in fig.data])),
                       np.sort(ad.obsm["X_umap3d"][:, 2]))
    fig = scp.pl.feature_dim_plot_3d(ad, "g0", reduction="UMAP")
    assert np.allclose(fig.data[0].z, ad.obsm["X_umap3d"][:, 2])


# --------------------------------------------------------------------------- #
# 3. PAGA transitions without velocity PAGA
# --------------------------------------------------------------------------- #
def test_paga_show_transition_without_velocity_paga_says_what_is_missing(adata):
    ad = adata.copy()
    ad.uns["paga"] = {"connectivities": np.full((4, 4), 0.5), "groups": "leiden"}
    with pytest.warns(UserWarning, match=r"scv\.tl\.paga"):
        ax = scp.pl.paga_plot(ad, show_transition=True)
    assert ax is not None
    with pytest.raises(KeyError, match=r"scv\.tl\.paga"):
        scp.io.read_paga(ad, type="transitions_confidence")


# --------------------------------------------------------------------------- #
# 4. group columns that are not strings
# --------------------------------------------------------------------------- #
def _with_columns(adata):
    ad = adata.copy()
    codes = ad.obs["leiden"].cat.codes.to_numpy()
    ad.obs["int_col"] = codes.astype(int)                     # plain int64
    ad.obs["bool_col"] = codes % 2 == 0                       # bool
    ad.obs["intcat"] = pd.Categorical(codes + 1, categories=[4, 3, 2, 1])
    ad.obs["floatcat"] = pd.Categorical(codes / 2.0)
    ad.obs["mixed"] = pd.Series(np.where(codes < 2, codes, "x"), dtype=object,
                                index=ad.obs_names).map(lambda v: int(v) if v != "x" else v)
    ad.obs["split_int"] = (np.arange(ad.n_obs) % 3).astype(int)
    return ad


@pytest.mark.parametrize("col", ["int_col", "bool_col", "intcat", "floatcat", "mixed"])
def test_non_string_groups_keep_legend_labels_and_marks(adata, col):
    ad = _with_columns(adata)
    want = [str(c) for c in scp.fetch.as_ordered_categorical(ad.obs[col]).cat.categories]
    fig = scp.pl.cell_dim_plot(ad, col, label=True, add_mark=True, mark_type="ellipse",
                               show_stat=False)
    ax = fig.axes[0]
    assert _labels(fig) == [f"{i}: {lv}" for i, lv in enumerate(want, start=1)]
    marks = [p for p in ax.patches if isinstance(p, layers._MarkPatch)]
    assert len(marks) == len(want)
    texts = [a.get_text() for a in ax.texts if isinstance(a, Annotation)]
    assert texts == [str(i) for i in range(1, len(want) + 1)]
    # every cell is coloured by its group, none fall back to the background
    cols = {to_hex(c) for c in ax.collections[0].get_facecolors()}
    assert to_hex(scp.palettes.NA_COLOR_DEFAULT) not in cols


def test_int_categories_keep_their_declared_order(adata):
    ad = _with_columns(adata)
    fig = scp.pl.cell_dim_plot(ad, "intcat", show_stat=False)
    assert _labels(fig) == ["4", "3", "2", "1"]


def test_non_string_split_and_stat_by(adata):
    ad = _with_columns(adata)
    fig = scp.pl.cell_dim_plot(ad, "int_col", split_by="split_int", stat_by="bool_col")
    axes = _panels(fig)
    assert len(axes) == 3
    subs = [a.get_title(loc="left") for a in axes]
    assert [s.split(" ")[0] for s in subs] == ["0", "1", "2"]
    assert "bool_col:" in _titles(fig)
    stat = fig.legends[_titles(fig).index("bool_col:")]
    want = [str(c) for c in scp.fetch.as_ordered_categorical(ad.obs["bool_col"]).cat.categories]
    assert [t.get_text() for t in stat.get_texts()] == want
    # the legend counts each level in the first split panel: none are zero
    counts = [int(t.split("(")[1].rstrip(")")) for t in _labels(fig)]
    assert sum(counts) == int((ad.obs["split_int"] == 0).sum()) and min(counts) > 0
    # split_by on feature_dim_plot too: one panel per level, partitioned
    fig = scp.pl.feature_dim_plot(ad, "g0", split_by="split_int", bg_cutoff=None)
    n = [a.collections[-1].get_offsets().shape[0] for a in _panels(fig)]
    assert sum(n) == ad.n_obs


def test_paga_layer_on_an_int_group(adata):
    ad = _with_columns(adata)
    ad.obs["intcat"] = ad.obs["intcat"].cat.reorder_categories([1, 2, 3, 4])
    ad.uns["paga"] = {"connectivities": np.full((4, 4), 0.5), "groups": "intcat"}
    fig = scp.pl.cell_dim_plot(ad, "intcat", paga=True)
    assert _labels(fig) == [f"{lv}({int((ad.obs['intcat'] == lv).sum())})" for lv in (1, 2, 3, 4)]
    nodes = fig.axes[0].collections[-1]  # the coloured node discs
    assert nodes.get_offsets().shape[0] == 4


# --------------------------------------------------------------------------- #
# 6. overlays drawn into a borrowed ax set no title
# --------------------------------------------------------------------------- #
def test_overlays_do_not_title_a_borrowed_axes(adata):
    ad = adata.copy()
    ad.obsm["velocity_umap"] = np.ones((ad.n_obs, 2)) * 0.1
    ad.uns["paga"] = {"connectivities": np.full((4, 4), 0.5), "groups": "leiden"}
    fig = scp.pl.cell_dim_plot(ad, "leiden")
    ax = fig.axes[0]
    host = (ax.get_title(), ax.get_title(loc="left"), ax.get_xlabel(), ax.get_ylabel())
    scp.pl.paga_plot(ad, ax=ax)
    scp.pl.velocity_plot(ad, plot_type="grid", ax=ax)
    assert (ax.get_title(), ax.get_title(loc="left"), ax.get_xlabel(), ax.get_ylabel()) == host
    # their own figures keep R's titles
    assert scp.pl.paga_plot(ad).get_title() == "PAGA"
    assert scp.pl.velocity_plot(ad).get_title() == "Cell velocity"


# --------------------------------------------------------------------------- #
# 5. velocity_plot draws R's arrows, the same layer as the overlay
# --------------------------------------------------------------------------- #
def test_velocity_plot_and_overlay_draw_the_same_arrows(adata):
    ad = adata.copy()
    ad.obsm["velocity_umap"] = np.random.default_rng(1).normal(size=(ad.n_obs, 2)) * 0.2
    a = scp.pl.velocity_plot(ad)
    b = scp.pl.cell_dim_plot(ad, "leiden", velocity="velocity").axes[0]
    fa = [x for x in a.artists if isinstance(x, layers._ArrowField)][0]
    fb = [x for x in b.artists if isinstance(x, layers._ArrowField)][0]
    order_a = np.lexsort(fa._seg.T[::-1])
    order_b = np.lexsort(fb._seg.T[::-1])
    np.testing.assert_allclose(fa._seg[order_a], fb._seg[order_b])
    np.testing.assert_allclose(fa._head[order_a], fb._head[order_b])
    # the standalone panel still frames its arrows
    x0, x1 = a.get_xlim()
    ends = np.r_[fa._seg[:, 0], fa._seg[:, 2]]
    assert x0 <= ends.min() and x1 >= ends.max()


# --------------------------------------------------------------------------- #
# 8. feature_dim_plot's default grid
# --------------------------------------------------------------------------- #
def test_many_features_wrap_into_patchworks_grid(adata):
    fig = scp.pl.feature_dim_plot(adata, [f"g{i}" for i in range(12)])
    axes = _panels(fig)
    xs = sorted({round(a.get_position().x0, 3) for a in axes})
    ys = sorted({round(a.get_position().y0, 3) for a in axes})
    assert (len(ys), len(xs)) == (3, 4)  # wrap_dims(12)


def test_split_features_wrap_as_one_flat_list(adata):
    fig = scp.pl.feature_dim_plot(adata, ["g0", "g1", "g2"], split_by="batch")
    axes = _panels(fig)
    ys = sorted({round(a.get_position().y0, 3) for a in axes})
    xs = sorted({round(a.get_position().x0, 3) for a in axes})
    assert (len(ys), len(xs)) == (2, 3)  # wrap_dims(6), not 2 x 3 by construction
    fig = scp.pl.feature_dim_plot(adata, ["g0", "g1"], split_by="batch")
    assert len({round(a.get_position().y0, 3) for a in _panels(fig)}) == 2  # wrap_dims(4)


# --------------------------------------------------------------------------- #
# 9. projection legends
# --------------------------------------------------------------------------- #
def test_projection_plot_registers_ref_and_query_legends(adata):
    ref = adata[adata.obs["batch"] == "b1"].copy()
    qry = adata[adata.obs["batch"] == "b2"].copy()
    ax = scp.pl.projection_plot(qry, ref, query_group="leiden", ref_group="leiden")
    fig = ax.figure
    assert _titles(fig) == ["Ref: leiden:", "Query: leiden:"]
    assert _labels(fig, 0) == list(ref.obs["leiden"].cat.categories)
    q = fig.legends[1].legend_handles
    assert all(h.get_markeredgecolor() == "black" for h in q)
    ax = scp.pl.projection_plot(qry, ref)
    assert not ax.figure.legends


# --------------------------------------------------------------------------- #
# 10. scanpy's stored colours
# --------------------------------------------------------------------------- #
def _uns_colours(adata):
    ad = adata.copy()
    ad.uns["leiden_colors"] = ["#111111", "#222222", "#333333", "#444444"]
    return ad


def test_cell_dim_plot_reads_uns_colours_unless_told_otherwise(adata):
    ad = _uns_colours(adata)
    cats = list(ad.obs["leiden"].cat.categories)
    fig = scp.pl.cell_dim_plot(ad, "leiden")
    got = [to_hex(h.get_markerfacecolor()) for h in fig.legends[0].legend_handles]
    assert got == ad.uns["leiden_colors"]
    for kw in ({"use_uns_colors": False}, {"palette": "Set1"}, {"palcolor": ["red"] * 4}):
        fig = scp.pl.cell_dim_plot(ad, "leiden", **kw)
        got = [to_hex(h.get_markerfacecolor()) for h in fig.legends[0].legend_handles]
        assert got != ad.uns["leiden_colors"], kw
    want = scp.palettes.discrete_palette(cats, palette="Paired")
    fig = scp.pl.cell_dim_plot(ad, "leiden", use_uns_colors=False)
    got = [to_hex(h.get_markerfacecolor()) for h in fig.legends[0].legend_handles]
    assert got == [to_hex(want[c]) for c in cats]


def test_misaligned_uns_colours_are_ignored_with_a_warning(adata):
    ad = adata.copy()
    ad.uns["leiden_colors"] = ["#111111", "#222222"]
    with pytest.warns(UserWarning, match="ignored"):
        assert scp.palettes.uns_palcolor(ad, "leiden") is None
    ad.obs["plain"] = ad.obs["leiden"].astype(str)
    ad.uns["plain_colors"] = ["#111111"] * 4
    assert scp.palettes.uns_palcolor(ad, "plain") is None  # not categorical: no alignment


def test_uns_colours_reach_paga_velocity_projection_and_3d(adata):
    ad = _uns_colours(adata)
    ad.obsm["velocity_umap"] = np.ones((ad.n_obs, 2)) * 0.1
    ad.uns["paga"] = {"connectivities": np.full((4, 4), 0.5), "groups": "leiden"}
    want = ad.uns["leiden_colors"]
    ax = scp.pl.paga_plot(ad)
    assert [to_hex(c) for c in ax.collections[-1].get_facecolors()] == want
    ax = scp.pl.velocity_plot(ad, group_by="leiden")
    assert [to_hex(h.get_color()) for h in ax.figure.legends[0].legend_handles] == want
    ref = ad[ad.obs["batch"] == "b1"].copy()
    ax = scp.pl.projection_plot(ref, ref, ref_group="leiden")
    assert [to_hex(h.get_markerfacecolor()) for h in ax.figure.legends[0].legend_handles] == want
    pytest.importorskip("plotly")
    ad.obsm["X_umap3d"] = np.zeros((ad.n_obs, 3))
    fig = scp.pl.cell_dim_plot_3d(ad, "leiden")
    assert [t.marker.color for t in fig.data] == want
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scp.pl.cell_dim_plot(ad, "leiden", use_uns_colors=False)
