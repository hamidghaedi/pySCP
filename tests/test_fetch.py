"""The data contract: resolution order, categorical order, reduction picking."""

import numpy as np
import pandas as pd
import pytest

import scp
from scp.fetch import as_ordered_categorical, embedding_coord_map, reduction_key


def test_category_order_is_first_appearance_not_sorted():
    # pandas sorts by default; R does not. Sorting here silently reshuffles
    # every palette and legend relative to the R output.
    s = pd.Series(["z", "a", "z", "m"])
    assert list(as_ordered_categorical(s).cat.categories) == ["z", "a", "m"]


def test_existing_categorical_order_is_trusted():
    s = pd.Series(pd.Categorical(["b", "a"], categories=["b", "a"]))
    assert list(as_ordered_categorical(s).cat.categories) == ["b", "a"]


def test_show_na_appends_na_as_a_real_level():
    s = pd.Series(["a", None, "b"])
    out = as_ordered_categorical(s, show_na=True)
    assert list(out.cat.categories) == ["a", "b", "NA"]
    assert out.iloc[1] == "NA"


def test_reduction_key():
    assert reduction_key("X_umap") == "UMAP_"
    assert reduction_key("X_pca") == "PC_"
    # fallback mirrors adata_to_srt: strip X_, drop underscores, append _
    assert reduction_key("X_my_embedding") == "myembedding_"


def test_default_reduction_prefers_umap(adata):
    assert scp.default_reduction(adata) == "X_umap"


def test_default_reduction_short_circuits_on_a_single_candidate(adata):
    ad = adata.copy()
    del ad.obsm["X_umap"]
    # X_pca is ninth in the priority list, but it is the only candidate
    assert scp.default_reduction(ad) == "X_pca"


def test_default_reduction_breaks_ties_on_fewest_dims(adata):
    ad = adata.copy()
    ad.obsm["X_umap3d"] = np.zeros((ad.n_obs, 3))
    assert scp.default_reduction(ad) == "X_umap"


def test_coord_map_accepts_both_grammars(adata):
    m = embedding_coord_map(adata)
    assert m["UMAP_1"] == ("X_umap", 0)   # 1-based, R-style
    assert m["X_umap:0"] == ("X_umap", 0)  # 0-based, unambiguous


def test_fetch_data_resolution_order_and_block_order(adata):
    df = scp.fetch_data(adata, ["score", "g0", "UMAP_1"])
    # columns come back in resolution-block order (genes, obs, coords), not
    # the caller's order -- matching R's cbind
    assert list(df.columns) == ["g0", "score", "UMAP_1"]
    assert len(df) == adata.n_obs


def test_fetch_data_gene_wins_a_collision(adata):
    ad = adata.copy()
    ad.obs["g0"] = 999.0
    with pytest.warns(UserWarning, match="both"):
        df = scp.fetch_data(ad, ["g0"])
    assert not (df["g0"] == 999.0).all()


def test_fetch_data_warns_and_drops_missing(adata):
    with pytest.warns(UserWarning, match="not found"):
        df = scp.fetch_data(adata, ["g0", "nope"])
    assert list(df.columns) == ["g0"]


def test_fetch_data_raises_only_when_everything_is_missing(adata):
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="no valid variables"):
        scp.fetch_data(adata, ["nope", "also_nope"])


def test_fetch_data_cells_keeps_object_order(adata):
    wanted = [adata.obs_names[5], adata.obs_names[1], "ghost"]
    df = scp.fetch_data(adata, ["g0"], cells=wanted)
    assert list(df.index) == [adata.obs_names[1], adata.obs_names[5]]


def test_resolve_matrix_layer_semantics(adata):
    X, _ = scp.fetch.resolve_matrix(adata, layer=None)
    assert X is adata.X
    C, _ = scp.fetch.resolve_matrix(adata, layer="counts")
    assert np.allclose(C, adata.layers["counts"])
    with pytest.raises(KeyError):
        scp.fetch.resolve_matrix(adata, layer="nope")
