"""scanpy's stored colours (``adata.uns[f"{key}_colors"]``) outside the dim plots.

Every AnnData-taking function that colours by a categorical ``obs`` column
reads the stored list through :func:`scp.palettes.level_colors` (directly, or
:func:`scp.palettes.uns_level_palcolor` for a DataFrame engine) under one set
of rules, pinned here per function:

* the stored list is used as is;
* a list that does not match the categories warns and is ignored;
* an explicit ``palette`` / ``palcolor`` wins;
* ``use_uns_colors=False`` ignores it.

The last three are checked against the same call on an object without stored
colours, so they also pin that the R palette is untouched.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import scp
from scp.heatmap import glyphs

LEIDEN = ["#111111", "#222222", "#333333", "#444444"]
BATCH = ["#aa0000", "#00aa00"]


@pytest.fixture(autouse=True)
def _close_figures():
    import matplotlib.pyplot as plt
    yield
    plt.close("all")


@pytest.fixture()
def plain(adata):
    """The synthetic object plus two lineages, without stored colours."""
    ad = adata.copy()
    t = np.asarray(ad.obsm["X_umap"][:, 0], dtype=float)
    t = t - t.min()
    lab = np.asarray(ad.obs["leiden"])
    ad.obs["L1"] = np.where(lab != "D", t, np.nan)
    ad.obs["coarse"] = pd.Categorical(np.where(np.isin(lab, ["A", "B"]), "AB", "CD"))
    for k in [k for k in ad.uns if k.endswith("_colors")]:
        del ad.uns[k]
    return ad


def _stored(ad, leiden=LEIDEN, batch=BATCH):
    ad = ad.copy()
    ad.uns["leiden_colors"] = list(leiden)
    ad.uns["batch_colors"] = list(batch)
    ad.uns["coarse_colors"] = ["#0000aa", "#aaaa00"]
    return ad


# --------------------------------------------------------------------------- #
# reading the colours back
# --------------------------------------------------------------------------- #
def _fig_legend(fig, key: str) -> dict[str, str]:
    for lg in fig.legends:
        if lg.get_title().get_text().rstrip(":") != key:
            continue
        out = {}
        for h, t in zip(lg.legend_handles, lg.get_texts(), strict=True):
            for get in ("get_facecolor", "get_markerfacecolor", "get_color"):
                if hasattr(h, get):
                    out[t.get_text()] = to_hex(getattr(h, get)())
                    break
        return out
    raise AssertionError(f"no legend {key!r}: {[lg.get_title().get_text() for lg in fig.legends]}")


def _spec_legend(result, title: str) -> dict[str, str]:
    for lg in result.spec.legends:
        if lg.kind == "categorical" and lg.title == title:
            return {k: to_hex(v) for k, v in lg.mapping.items()}
    raise AssertionError(f"no legend {title!r}")


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*pygam.*")
        return fn(*args, **kwargs)


G3 = ["g0", "g1", "g2"]

# name -> (draw(ad, **kw) -> {level: colour}, stored colours, explicit overrides)
CASES = {
    "feature_stat_plot group_by": (
        lambda ad, **kw: _fig_legend(scp.pl.feature_stat_plot(ad, "g0", group_by="leiden",
                                                              **kw), "leiden"),
        LEIDEN, [{"palette": "Set1"}, {"palcolor": ["red", "blue", "green", "black"]}]),
    "feature_stat_plot split_by": (
        lambda ad, **kw: _fig_legend(scp.pl.feature_stat_plot(
            ad, "g0", group_by="leiden", split_by="batch", **kw), "batch"),
        BATCH, [{"palette": "Set1"}, {"palcolor": ["red", "blue"]}]),
    "cell_stat_plot": (
        lambda ad, **kw: _fig_legend(scp.pl.cell_stat_plot(ad, "leiden", group_by="batch",
                                                           **kw), "leiden"),
        LEIDEN, [{"palette": "Set1"}, {"palcolor": ["red", "blue", "green", "black"]}]),
    "cell_stat_plot sankey": (
        lambda ad, **kw: _fig_legend(scp.pl.cell_stat_plot(ad, ["leiden", "batch"],
                                                           plot_type="sankey", **kw), "batch"),
        BATCH, [{"palette": "Set1"}, {"palcolor": ["red", "blue"] * 3}]),
    "cell_density_plot": (
        lambda ad, **kw: _fig_legend(scp.pl.cell_density_plot(ad, "g0", group_by="leiden",
                                                              **kw), "leiden"),
        LEIDEN, [{"palette": "Set1"}, {"palcolor": ["red", "blue", "green", "black"]}]),
    "feature_cor_plot": (
        lambda ad, **kw: _fig_legend(scp.pl.feature_cor_plot(ad, ["g0", "g1"], group_by="leiden",
                                                             **kw), "leiden"),
        LEIDEN, [{"palette": "Set1"}, {"palcolor": ["red", "blue", "green", "black"]}]),
    "dynamic_plot": (
        lambda ad, **kw: _fig_legend(_quiet(scp.pl.dynamic_plot, ad, ["L1"], ["g0"],
                                            group_by="leiden", **kw), "leiden"),
        LEIDEN, [{"point_palette": "Set1"},
                 {"point_palcolor": ["red", "blue", "green", "black"]}]),
    "group_heatmap group": (
        lambda ad, **kw: _spec_legend(scp.pl.group_heatmap(ad, G3, "leiden", layer=None, **kw),
                                      "leiden"),
        LEIDEN, [{"group_palette": "Set1"}]),
    "group_heatmap split": (
        lambda ad, **kw: _spec_legend(scp.pl.group_heatmap(ad, G3, "leiden", split_by="batch",
                                                           layer=None, **kw), "batch"),
        BATCH, [{"cell_split_palette": "Set1"}]),
    "group_heatmap cell_annotation": (
        lambda ad, **kw: _spec_legend(scp.pl.group_heatmap(ad, G3, "batch", layer=None,
                                                           cell_annotation=["leiden"], **kw),
                                      "leiden"),
        LEIDEN, [{"cell_annotation_palette": "Set1"}]),
    "feature_heatmap group": (
        lambda ad, **kw: _spec_legend(scp.pl.feature_heatmap(ad, G3, "leiden", layer=None,
                                                             max_cells=20, **kw), "leiden"),
        LEIDEN, [{"group_palette": "Set1"}]),
    "feature_heatmap split": (
        lambda ad, **kw: _spec_legend(scp.pl.feature_heatmap(
            ad, G3, "leiden", split_by="batch", layer=None, max_cells=20, **kw), "batch"),
        BATCH, [{"cell_split_palette": "Set1"}]),
    "feature_heatmap cell_annotation": (
        lambda ad, **kw: _spec_legend(scp.pl.feature_heatmap(
            ad, G3, "batch", layer=None, max_cells=20, cell_annotation=["leiden"], **kw),
            "leiden"),
        LEIDEN, [{"cell_annotation_palette": "Set1"}]),
    "dynamic_heatmap cell_annotation": (
        lambda ad, **kw: _spec_legend(scp.pl.dynamic_heatmap(
            ad, ["L1"], G3, layer=None, cell_annotation=["leiden"], **kw), "leiden"),
        LEIDEN, [{"cell_annotation_palette": "Set1"}]),
    "dynamic_heatmap separate_annotation": (
        lambda ad, **kw: _spec_legend(scp.pl.dynamic_heatmap(
            ad, ["L1"], G3, layer=None, separate_annotation=["leiden"], **kw),
            "leiden\n(separate)"),
        LEIDEN, [{"separate_annotation_palette": "Set1"},
                 {"separate_annotation_palcolor": ["red", "blue", "green", "black"]}]),
}


def _want(stored, levels) -> dict[str, str]:
    return dict(zip(levels, stored, strict=True))


@pytest.mark.parametrize("name", list(CASES))
def test_stored_colours_are_used(plain, name):
    draw, stored, _ = CASES[name]
    key = "leiden" if stored is LEIDEN else "batch"
    full = _want(stored, [str(c) for c in plain.obs[key].cat.categories])
    got = draw(_stored(plain))
    assert got and got == {lv: full[lv] for lv in got}
    if name != "dynamic_plot":  # the D cells are off the lineage, so D has no legend entry
        assert got == full


@pytest.mark.parametrize("name", list(CASES))
def test_misaligned_list_warns_and_is_ignored(plain, name):
    draw, stored, _ = CASES[name]
    bad = plain.copy()
    key = "leiden" if stored is LEIDEN else "batch"
    bad.uns[f"{key}_colors"] = ["#123456"]
    with pytest.warns(UserWarning, match="ignored"):
        got = draw(bad)
    assert got == draw(plain)


@pytest.mark.parametrize("name", list(CASES))
def test_explicit_palette_or_palcolor_wins(plain, name):
    draw, stored, explicit = CASES[name]
    ad = _stored(plain)
    for kw in explicit:
        got = draw(ad, **kw)
        assert got == draw(plain, **kw), kw
        assert list(got.values()) != stored, kw


@pytest.mark.parametrize("name", list(CASES))
def test_use_uns_colors_false_restores_the_palette(plain, name):
    draw, stored, _ = CASES[name]
    got = draw(_stored(plain), use_uns_colors=False)
    assert got == draw(plain)
    assert list(got.values()) != stored


# --------------------------------------------------------------------------- #
# what a legend does not show
# --------------------------------------------------------------------------- #
def test_feature_stat_plot_bg_by_stripes_and_engine_handoff(plain, monkeypatch):
    import scp.pl._stat as stat

    seen = []
    real = stat.expression_stat_plot

    def spy(*args, **kwargs):
        seen.append((kwargs["palcolor"], kwargs["bg_palcolor"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(stat, "expression_stat_plot", spy)
    ad = _stored(plain)
    scp.pl.feature_stat_plot(ad, "g0", group_by="leiden", bg_by="coarse")
    assert seen[-1] == (_want(LEIDEN, list("ABCD")), {"AB": "#0000aa", "CD": "#aaaa00"})
    scp.pl.feature_stat_plot(ad, "g0", group_by="leiden", bg_by="coarse", use_uns_colors=False)
    assert seen[-1] == (None, None)
    scp.pl.feature_stat_plot(ad, "g0", group_by="leiden", bg_by="coarse", bg_palette="Set1")
    assert seen[-1][1] is None
    # fill_by="feature" / "expression" colour features, not a column
    scp.pl.feature_stat_plot(ad, ["g0", "g1"], group_by="leiden", fill_by="feature")
    assert seen[-1][0] is None
    scp.pl.feature_stat_plot(plain, "g0", group_by="leiden", bg_by="coarse")
    assert seen[-1] == (None, None)


def test_feature_stat_plot_several_group_by_share_one_mapping(plain):
    fig = scp.pl.feature_stat_plot(_stored(plain), "g0", group_by=["leiden", "batch"])
    assert list(_fig_legend(fig, "leiden").values()) == LEIDEN
    assert list(_fig_legend(fig, "batch").values()) == BATCH


def test_cell_stat_plot_na_level_keeps_na_color(plain):
    ad = _stored(plain)
    ad.obs["leiden"] = ad.obs["leiden"].astype(object)
    ad.obs.loc[ad.obs_names[:5], "leiden"] = np.nan
    ad.obs["leiden"] = pd.Categorical(ad.obs["leiden"], categories=list("ABCD"))
    got = _fig_legend(scp.pl.cell_stat_plot(ad, "leiden", na_color="#BEBEBE"), "leiden")
    assert got == {**_want(LEIDEN, list("ABCD")), "NA": "#bebebe"}


def test_group_heatmap_violins_take_stored_colours(plain, monkeypatch):
    ad = _stored(plain)
    seen: list[str] = []
    real = glyphs.draw_violin

    def spy(ax, box, values, y_range, color, **kw):
        seen.append(to_hex(color))
        return real(ax, box, values, y_range, color, **kw)

    monkeypatch.setattr(glyphs, "draw_violin", spy)
    scp.pl.group_heatmap(ad, G3, "leiden", layer=None, add_violin=True, fill_by="group")
    assert set(seen) == set(LEIDEN)
    seen.clear()
    scp.pl.group_heatmap(ad, G3, "leiden", layer=None, add_violin=True, fill_by="group",
                         use_uns_colors=False)
    assert not set(seen) & set(LEIDEN)
    seen.clear()
    # a numeric cell_annotation: a violin per column, filled by the column's group
    scp.pl.group_heatmap(ad, G3, "leiden", layer=None, cell_annotation=["score"])
    assert set(seen) == set(LEIDEN)


def test_pooled_columns_that_disagree_on_a_level_are_ignored(plain):
    ad = plain.copy()
    ad.obs["other"] = pd.Categorical(np.asarray(ad.obs["leiden"]))
    ad.uns["leiden_colors"] = LEIDEN
    ad.uns["other_colors"] = ["#999999"] * 4
    cols = {"leiden": list("ABCD"), "other": list("ABCD")}
    with pytest.warns(UserWarning, match="two colours"):
        got = scp.palettes.uns_level_palcolor(ad, cols, palette="Paired", palcolor=None,
                                              default_palette="Paired", pooled=True)
    assert got is None
    ad.uns["other_colors"] = LEIDEN
    got = scp.palettes.uns_level_palcolor(ad, cols, palette="Paired", palcolor=None,
                                          default_palette="Paired", pooled=True)
    assert got == _want(LEIDEN, list("ABCD"))
