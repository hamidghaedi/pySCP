"""Adapters between community result formats and this package's table contracts.

The plotting layer never reaches into ``adata.uns`` directly.  Every reader
here follows the same rule: try the explicit key, then the **native** scanpy /
scvelo / decoupler key, then our own namespace.  That keeps the package usable
on objects that have never touched it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "DE_COLUMNS",
    "ENRICHMENT_COLUMNS",
    "GSEA_COLUMNS",
    "de_from_rank_genes_groups",
    "enrichment_from_gseapy",
    "read_paga",
    "read_uns",
]

#: Column contract for :func:`scp.pl.volcano_plot` and the DE-driven heatmaps.
DE_COLUMNS = ("gene", "group1", "group2", "avg_log2FC", "p_val", "p_val_adj", "pct.1", "pct.2")

#: Column contract for :func:`scp.pl.enrichment_plot`.
ENRICHMENT_COLUMNS = (
    "ID", "Description", "GeneRatio", "BgRatio", "pvalue", "p.adjust",
    "geneID", "Count", "Groups", "Database",
)

#: Column contract for :func:`scp.pl.gsea_plot`.
GSEA_COLUMNS = (
    "ID", "Description", "setSize", "enrichmentScore", "NES", "pvalue",
    "p.adjust", "core_enrichment", "Groups", "Database",
)


def read_uns(adata, key: str | None, native: str | Sequence[str] = (), scp_key: str | None = None) -> Any:
    """Resolve an ``uns`` entry: explicit key, then native keys, then our namespace."""
    if key is not None:
        return adata.uns[key]
    for k in (native,) if isinstance(native, str) else native:
        if k in adata.uns:
            return adata.uns[k]
    if scp_key and "scp" in adata.uns and scp_key in adata.uns["scp"]:
        return adata.uns["scp"][scp_key]
    raise KeyError(
        f"none of {list((native,) if isinstance(native, str) else native)} found in adata.uns; "
        "pass an explicit key."
    )


def de_from_rank_genes_groups(
    adata, key: str = "rank_genes_groups", *, reference: str = "rest"
) -> pd.DataFrame:
    """Flatten scanpy's DE result into the :data:`DE_COLUMNS` contract.

    scanpy stores a structured array of arrays; SCP's plots want one tidy row
    per (gene, group).  Column renaming: ``names`` -> ``gene``,
    ``logfoldchanges`` -> ``avg_log2FC``, ``pvals`` -> ``p_val``,
    ``pvals_adj`` -> ``p_val_adj``, ``pts``/``pts_rest`` -> ``pct.1``/``pct.2``.
    """
    res = adata.uns[key]
    groups = list(res["names"].dtype.names)
    frames = []
    for g in groups:
        df = pd.DataFrame(
            {
                "gene": np.asarray(res["names"][g]),
                "group1": g,
                "group2": reference,
                "avg_log2FC": np.asarray(res["logfoldchanges"][g], dtype=float),
                "p_val": np.asarray(res["pvals"][g], dtype=float),
                "p_val_adj": np.asarray(res["pvals_adj"][g], dtype=float),
            }
        )
        if "pts" in res:
            pts = res["pts"]
            df["pct.1"] = pts[g].reindex(df["gene"]).to_numpy() if hasattr(pts, "reindex") else np.nan
            df["pct.2"] = (
                res["pts_rest"][g].reindex(df["gene"]).to_numpy()
                if "pts_rest" in res and hasattr(res["pts_rest"], "reindex")
                else np.nan
            )
        else:
            df["pct.1"] = np.nan
            df["pct.2"] = np.nan
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["group1"] = pd.Categorical(out["group1"], categories=groups)
    return out


def enrichment_from_gseapy(res, *, group: str = "all", database: str = "custom") -> pd.DataFrame:
    """Map a ``gseapy`` enrich/prerank result onto :data:`ENRICHMENT_COLUMNS`."""
    df = res.res2d.copy() if hasattr(res, "res2d") else pd.DataFrame(res).copy()
    rename = {
        "Term": "Description",
        "Adjusted P-value": "p.adjust",
        "FDR q-val": "p.adjust",
        "P-value": "pvalue",
        "NOM p-val": "pvalue",
        "Overlap": "GeneRatio",
        "Genes": "geneID",
        "Lead_genes": "core_enrichment",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["ID"] = df.get("ID", df["Description"])
    df["Groups"] = df.get("Groups", group)
    df["Database"] = df.get("Database", database)
    if "geneID" in df.columns:
        df["geneID"] = df["geneID"].astype(str).str.replace(";", "/", regex=False)
        df["Count"] = df["geneID"].str.count("/") + 1
    return df


def read_paga(adata, *, type: str = "connectivities") -> tuple[np.ndarray, str, list[str]]:
    """``(matrix, groups_column, categories)`` from ``adata.uns['paga']``.

    Keys are scanpy's own: ``connectivities``, ``connectivities_tree``,
    ``groups``; scvelo adds ``transitions_confidence``.  Rows and columns are
    ordered by ``adata.obs[groups].cat.categories``.
    """
    paga = adata.uns["paga"]
    groups = str(paga["groups"])
    mat = paga[type]
    mat = np.asarray(mat.todense()) if hasattr(mat, "todense") else np.asarray(mat)
    cats = list(adata.obs[groups].cat.categories)
    if mat.shape[0] != len(cats):
        raise ValueError(
            f"paga['{type}'] is {mat.shape[0]}x{mat.shape[1]} but obs['{groups}'] has "
            f"{len(cats)} categories."
        )
    return mat, groups, cats
