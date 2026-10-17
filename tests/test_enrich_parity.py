"""The enrichment graphs, partitions and word tables against SCP itself.

Fixtures come from ``tools/parity/enrich_parity.R``: SCP 0.5.6's own
``RunEnrichment`` / ``RunGSEA`` on ``pancreas_sub`` with a custom GO-BP gene-set
database, then the layer data of the ggplots ``EnrichmentPlot`` and
``GSEAPlot`` return.  What must agree is the *graph* -- vertex set and order,
edge set and weights, colours, community partition and the cluster labels --
and the word tables.  Layout coordinates are seeded from different RNGs on the
two sides and are not compared.

The suite never needs R: the fixtures are committed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scp.pl import _enrich as en
from scp.pl import _enrich_graph as eg

FIX = Path(__file__).parent / "fixtures" / "parity" / "enrich"
pytestmark = pytest.mark.skipif(not (FIX / "enrichment.csv.gz").exists(),
                                reason="enrichment parity fixtures not generated")


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(FIX / f"{name}.csv.gz", keep_default_na=False, na_values=[""])


def ari(a, b) -> float:
    """Adjusted Rand index, without scikit-learn."""
    a, b = np.asarray(a), np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(table, (ai, bi), 1)

    def comb2(x):
        return (x * (x - 1) / 2).sum()

    idx = comb2(table)
    ea, eb = comb2(table.sum(1)), comb2(table.sum(0))
    expected = ea * eb / comb2(np.array([len(a)]))
    top = (ea + eb) / 2
    return 1.0 if top == expected else float((idx - expected) / (top - expected))


@pytest.fixture(scope="module")
def enrichment():
    return read("enrichment")


@pytest.fixture(scope="module")
def gsea():
    return read("gsea")


def _panels(enrichment, plot_type, **kw):
    _, panels, metric, value, _ = en._prepare_enrichment(
        enrichment, plot_type=plot_type, split_by=("Database", "Groups"), group_use=None,
        id_use=None, pvalue_cutoff=None, padjust_cutoff=0.05, top_term=kw.get("top_term"))
    return panels, metric, value


def _rkey(key: str) -> str:
    """'GO_BP | Ductal' -> R's split() name 'GO_BP.Ductal'."""
    return key.replace(" | ", ".")


# --------------------------------------------------------------------------- #
# network
# --------------------------------------------------------------------------- #
def test_network_graph_matches_scp(enrichment):
    pytest.importorskip("igraph")
    rn, re_ = read("network_nodes"), read("network_edges")
    panels, metric, _ = _panels(enrichment, "network")
    assert [_rkey(k) for k, _, _ in panels] == list(dict.fromkeys(rn["panel"]))
    for key, _, df in panels:
        nt = eg.network_tables(df, metric=metric,
                               descriptions=en._wrap_desc(df["Description"], 50))
        r = rn[rn["panel"] == _rkey(key)].reset_index(drop=True)
        e = re_[re_["panel"] == _rkey(key)].reset_index(drop=True)
        assert nt.nodes["name"].tolist() == r["name"].tolist()
        assert nt.nodes["class"].tolist() == r["class"].tolist()
        assert nt.nodes["color"].str.upper().tolist() == r["color"].str.upper().tolist()
        assert nt.nodes["label_color"].tolist() == r["label_color"].tolist()
        np.testing.assert_allclose(nt.nodes["metric"], r["metric"], rtol=0, atol=1e-10)
        terms = r["class"] == "term"
        assert nt.nodes.loc[terms, "label"].astype(int).tolist() == r.loc[terms, "label"].astype(int).tolist()
        assert list(zip(nt.edges["from"], nt.edges["to"], strict=True)) == list(
            zip(e["from"], e["to"], strict=True))
        assert (nt.edges["weight"] == e["weight"]).all()
        assert nt.edges["color"].str.upper().tolist() == e["color"].str.upper().tolist()


def test_gsea_network_graph_matches_scp(gsea):
    pytest.importorskip("igraph")
    rn, re_ = read("gsea_network_nodes"), read("gsea_network_edges")
    _, per, _ = en._gsea_tables(gsea)
    assert list(per) == read("gsea_keys")["key"].tolist()
    for key, res in per.items():
        ids = en.gsea_select_ids(res, metric="p.adjust", metric_value=0.05, top_term=6,
                                 direction="both")
        df = res.set_index("ID").loc[ids].reset_index()
        desc = en.gsea_network_descriptions(df, "p.adjust")
        nt = eg.network_tables(df, gene_col="core_enrichment", descriptions=desc)
        r = rn[rn["panel"] == key].reset_index(drop=True)
        e = re_[re_["panel"] == key].reset_index(drop=True)
        # term names carry the formatted NES / p.adjust / stars line
        assert nt.nodes["name"].tolist() == r["name"].tolist()
        assert nt.nodes["color"].str.upper().tolist() == r["color"].str.upper().tolist()
        assert nt.nodes["label_color"].tolist() == r["label_color"].tolist()
        assert list(zip(nt.edges["from"], nt.edges["to"], strict=True)) == list(
            zip(e["from"], e["to"], strict=True))


# --------------------------------------------------------------------------- #
# enrichmap
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prefix,kw", [
    ("enrichmap", {}),
    ("enrichmap_kw", dict(show_keyword=True, label="feature", cluster="walktrap")),
])
def test_enrichmap_graph_and_partition_match_scp(enrichment, prefix, kw):
    pytest.importorskip("igraph")
    rn, re_, rl = read(f"{prefix}_nodes"), read(f"{prefix}_edges"), read(f"{prefix}_legend")
    panels, metric, _ = _panels(enrichment, "enrichmap")
    assert [_rkey(k) for k, _, _ in panels] == list(dict.fromkeys(rn["panel"]))
    n_clusters = 0
    for key, _, df in panels:
        et = eg.enrichmap_tables(df, metric=metric, **kw)
        r = rn[rn["panel"] == _rkey(key)].reset_index(drop=True)
        e = re_[re_["panel"] == _rkey(key)].reset_index(drop=True)
        leg = rl[rl["panel"] == _rkey(key)].reset_index(drop=True)
        assert et.nodes["ID"].tolist() == r["ID"].tolist()
        # similarity = shared-gene count, every pair with > 0
        assert set(zip(et.edges["from"], et.edges["to"], et.edges["weight"].astype(int),
                       strict=True)) == set(zip(e["from"], e["to"], e["weight"].astype(int),
                                                strict=True))
        assert ari(et.nodes["clusters"], r["clusters"]) == pytest.approx(1.0)
        # not only the same partition: the same C1, C2, ... names, which pick colours
        assert et.nodes["clusters"].tolist() == r["clusters"].tolist()
        assert et.nodes["keyword1"].fillna("").tolist() == r["keyword1"].fillna("").tolist()
        assert et.nodes["keyword2"].fillna("").tolist() == r["keyword2"].fillna("").tolist()
        assert list(et.legend_labels.values()) == leg["label"].tolist()
        assert f"{et.legend_title}:" == leg["name"].iloc[0]
        n_clusters += len(et.cluster_levels)
    assert n_clusters > len(panels)  # the partition is not trivial


def test_enrichmap_similarity_matrix_matches_scp(enrichment):
    """The full term-by-term overlap matrix, zero entries included."""
    re_ = read("enrichmap_edges")
    panels, _, _ = _panels(enrichment, "enrichmap")
    worst = 0.0
    for key, _, df in panels:
        ids = df["ID"].tolist()
        pos = {i: k for k, i in enumerate(ids)}
        mine = np.zeros((len(ids), len(ids)))
        for i, j, w in eg.overlap_edges([s.split("/") for s in df["geneID"]]):
            mine[i, j] = mine[j, i] = w
        theirs = np.zeros_like(mine)
        e = re_[re_["panel"] == _rkey(key)]
        for f, t, w in zip(e["from"], e["to"], e["weight"], strict=True):
            theirs[pos[f], pos[t]] = theirs[pos[t], pos[f]] = w
        worst = max(worst, float(np.abs(mine - theirs).max()))
    assert worst == 0.0


def test_gsea_enrichmap_matches_scp(gsea):
    pytest.importorskip("igraph")
    rn, re_, rl = (read("gsea_enrichmap_nodes"), read("gsea_enrichmap_edges"),
                   read("gsea_enrichmap_legend"))
    _, per, _ = en._gsea_tables(gsea)
    for key, res in per.items():
        ids = en.gsea_select_ids(res, metric="p.adjust", metric_value=0.05, top_term=100,
                                 direction="both")
        df = res.set_index("ID").loc[ids].reset_index()
        et = eg.enrichmap_tables(
            df, gene_col="core_enrichment",
            descriptions=en._wrap_desc(df["Description"], 50),
            counts=[len(s.split("/")) for s in df["core_enrichment"]])
        r = rn[rn["panel"] == key].reset_index(drop=True)
        e = re_[re_["panel"] == key]
        assert et.nodes["ID"].tolist() == r["ID"].tolist()
        assert et.nodes["Count"].tolist() == r["Count"].tolist()
        assert set(zip(et.edges["from"], et.edges["to"], et.edges["weight"].astype(int),
                       strict=True)) == set(zip(e["from"], e["to"], e["weight"].astype(int),
                                                strict=True))
        assert et.nodes["clusters"].tolist() == r["clusters"].tolist()
        assert et.nodes["keyword1"].tolist() == r["keyword1"].tolist()
        assert et.nodes["keyword2"].tolist() == r["keyword2"].tolist()
        assert list(et.legend_labels.values()) == rl[rl["panel"] == key]["label"].tolist()


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("only_sig", [False, True])
def test_enrichment_comparison_frame_matches_scp(enrichment, only_sig):
    r = read("comparison_sig" if only_sig else "comparison")
    full, panels, metric, value, _ = en._prepare_enrichment(
        enrichment, plot_type="comparison", split_by=("Database", "Groups"), group_use=None,
        id_use=None, pvalue_cutoff=None, padjust_cutoff=0.05, top_term=3)
    frame, levels = en.enrichment_comparison_frame(full, panels, metric=metric,
                                                   metric_value=value,
                                                   compare_only_sig=only_sig)
    assert list(zip(frame["Groups"], frame["ID"], strict=True)) == list(
        zip(r["Groups"], r["ID"], strict=True))
    assert frame["Description"].tolist() == r["Description"].tolist()
    np.testing.assert_allclose(frame["GeneRatio"], r["GeneRatio"], atol=1e-12)
    # factor level of each row: the y position
    assert [levels.index(d) + 1 for d in frame["Description"]] == r["level"].tolist()


@pytest.mark.parametrize("direction", ["pos", "both"])
def test_gsea_comparison_frame_matches_scp(gsea, direction):
    r = read(f"gsea_comparison_{direction}")
    frame, levels = en.gsea_comparison_frame(gsea, metric="p.adjust", metric_value=0.05,
                                             top_term=3, direction=direction)
    assert list(zip(frame["Groups"], frame["ID"], strict=True)) == list(
        zip(r["Groups"], r["ID"], strict=True))
    assert frame["Description"].tolist() == r["Description"].tolist()
    assert frame["Significant"].tolist() == \
        r["Significant"].astype(str).str.upper().eq("TRUE").tolist()
    assert [levels.index(d) + 1 for d in frame["Description"]] == r["level"].tolist()


# --------------------------------------------------------------------------- #
# wordcloud
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("word_type,name", [("term", "wordcloud_term"),
                                            ("feature", "wordcloud_feature")])
def test_wordcloud_tables_match_scp(enrichment, word_type, name):
    r = read(name)
    sig = enrichment[enrichment["p.adjust"] < 0.05]
    sig = sig.iloc[np.argsort(sig["p.adjust"].to_numpy(), kind="stable")]
    for g in dict.fromkeys(sig["Groups"]):
        sub = sig[sig["Groups"] == g].assign(Database="custom")  # the non-GO path
        t = eg.wordcloud_table(sub, word_type=word_type)
        rr = r[r["Groups"] == g]
        assert t["keyword"].tolist() == rr["keyword"].astype(str).tolist()
        assert t["count"].tolist() == rr["count"].tolist()
        np.testing.assert_allclose(t["score"], rr["score"], rtol=1e-12)


@pytest.mark.parametrize("word_type,name", [("term", "gsea_wordcloud_term"),
                                            ("feature", "gsea_wordcloud_feature")])
def test_gsea_wordcloud_tables_match_scp(gsea, word_type, name):
    r = read(name)
    _, per, _ = en._gsea_tables(gsea)
    for key, res in per.items():
        ids = en.gsea_select_ids(res, metric="p.adjust", metric_value=0.05, top_term=None,
                                 direction="both")
        df = res.set_index("ID").loc[ids].reset_index().assign(Database="custom")
        t = eg.wordcloud_table(df, word_type=word_type, gene_col="core_enrichment")
        rr = r[r["key"] == key]
        assert t["keyword"].tolist() == rr["keyword"].astype(str).tolist()
        assert t["count"].tolist() == rr["count"].tolist()
        np.testing.assert_allclose(t["score"], rr["score"], rtol=1e-12)


def test_keyword_enrichment_matches_simplifyenrichment(enrichment):
    """simplifyEnrichment:::keyword_enrichment on the same term-document matrix:
    the Fisher p-values, the Holm adjustment and the ordering."""
    r = read("wordcloud_keyword_substitute")
    bg = dict(zip(enrichment["ID"], enrichment["Description"], strict=False))
    tdm = eg.term_document_counts(bg)
    sig = enrichment[enrichment["p.adjust"] < 0.05]
    worst = 0.0
    for g in dict.fromkeys(sig["Groups"]):
        k = eg.keyword_enrichment(sig[sig["Groups"] == g]["ID"].tolist(), tdm, list(bg))
        rr = r[r["Groups"] == g].reset_index(drop=True)
        assert k["keyword"].tolist() == rr["keyword"].tolist()
        assert k["n_term"].tolist() == rr["n_term"].tolist()
        assert k["n_bg"].tolist() == rr["n_bg"].tolist()
        worst = max(worst, float(np.abs(k["p"] - rr["p"]).max()),
                    float(np.abs(k["padj"] - rr["padj"]).max()))
    assert worst < 1e-12


def test_go_wordcloud_substitute_covers_scp_keywords(enrichment):
    """The documented divergence, measured: SCP's GO path ranks keywords by
    simplifyEnrichment's test against all of GO; without that matrix the
    substitute scores word frequency.  The scores differ in kind, but the
    vocabulary should not: on pancreas_sub 52 of R's 53 keywords appear."""
    r = read("wordcloud_go_keyword_R")
    excluded = {w.lower() for w in eg.WORDS_EXCLUDED}
    r = r[~r["keyword"].str.lower().isin(excluded)]
    sig = enrichment[enrichment["p.adjust"] < 0.05]
    found = total = 0
    for g in dict.fromkeys(sig["Groups"]):
        words = set(eg.wordcloud_table(sig[sig["Groups"] == g])["keyword"])
        theirs = set(r[r["Groups"] == g]["keyword"])
        found += len(theirs & words)
        total += len(theirs)
    assert total > 0 and found / total >= 0.95
