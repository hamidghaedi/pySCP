#!/usr/bin/env python
"""Render every figure the README shows, from the real dataset.

    python tools/make_readme_figures.py

Each block here is the *same code* the README quotes, so the two cannot drift:
if a call changes, the figure it produces changes with it. Output lands in
``docs/images/``.
"""

from __future__ import annotations

import pathlib
import warnings

import matplotlib

matplotlib.use("Agg")
import anndata as ad  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402

import scp  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name: str, **kw) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=110, bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  {name}.png")


def main() -> int:
    adata = ad.read_h5ad(ROOT / "notebooks" / "data" / "pancreas_sub.h5ad")

    # pancreas_sub reaches AnnData from a Seurat object whose `data` slot holds
    # raw counts, so normalise before anything expression-based.
    adata.layers["counts"] = adata.layers["counts"]
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print(f"{adata.n_obs} cells x {adata.n_vars} genes")

    print("dimensional reduction")
    save(scp.pl.cell_dim_plot(adata, ["CellType", "SubCellType"], label=True,
                              theme=scp.theme_blank(), ncol=2), "dim-1")
    save(scp.pl.cell_dim_plot(adata, "SubCellType", split_by="Phase", label=True,
                              theme=scp.theme_blank(), ncol=3), "dim-2")
    save(scp.pl.feature_dim_plot(adata, ["Sox9", "Neurog3", "Fev", "Rbp4"],
                                 ncol=4, theme=scp.theme_blank(),
                                 panel_size=(2.6, 2.6)), "dim-3")
    save(scp.pl.feature_dim_plot(adata, ["Ins1", "Gcg"], compare_features=True,
                                 theme=scp.theme_blank(), panel_size=(3.4, 3.4)), "dim-4")

    print("heatmaps")
    markers = ["Sox9", "Anxa2", "Neurog3", "Hes6", "Fev", "Neurod1",
               "Rbp4", "Pyy", "Ins1", "Gcg", "Sst", "Ghrl"]
    ht = scp.pl.group_heatmap(
        adata, features=markers, group_by="SubCellType", layer="counts",
        cell_annotation=["Phase"], add_dot=True, add_reticle=True,
        show_row_names=True,
    )
    save(ht.fig, "heatmap-1")
    save(scp.pl.feature_heatmap(adata, features=markers, group_by="CellType",
                                max_cells=60, show_row_names=True).fig, "heatmap-2")

    print("expression distributions")
    save(scp.pl.feature_stat_plot(adata, ["Ins1", "Gcg"], group_by="CellType",
                                  add_box=True, ncol=2, panel_size=(3.2, 2.6)),
         "stat-1")
    save(scp.pl.feature_stat_plot(adata, "Ins1", group_by="SubCellType",
                                  bg_by="CellType", plot_type="box", sort="increasing",
                                  panel_size=(4.0, 2.6)), "stat-2")

    print("composition")
    save(scp.pl.cell_stat_plot(adata, "SubCellType", group_by="Phase",
                               plot_type="bar", panel_size=(3.0, 2.6)), "comp-1")
    save(scp.pl.cell_stat_plot(adata, "CellType", group_by="Phase",
                               plot_type="ring", panel_size=(2.8, 2.6)), "comp-2")
    save(scp.pl.cell_stat_plot(adata, "SubCellType", group_by="Phase",
                               plot_type="dot", panel_size=(3.2, 3.0)), "comp-3")

    print("differential expression")
    sc.tl.rank_genes_groups(adata, "CellType", method="wilcoxon", pts=True)
    de = scp.io.de_from_rank_genes_groups(adata)
    save(scp.pl.volcano_plot(de, ncol=3, panel_size=(2.5, 2.5)), "de-1")

    print("correlation")
    save(scp.pl.feature_cor_plot(adata, ["Ins1", "Gcg", "Sst"], group_by="CellType"),
         "cor-1")
    save(scp.pl.cell_cor_heatmap(adata, query_group="SubCellType",
                                 ref_group="CellType", nfeatures=1000).fig, "cor-2")

    print("trajectory")
    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.paga(adata, groups="SubCellType")
    ax = scp.pl.paga_plot(adata, label=True)
    save(ax.figure, "traj-1")

    # a monotone coordinate along the differentiation axis, standing in for a
    # pseudotime the user would normally compute with scFates or slingshot
    adata.obs["Lineage"] = np.asarray(adata.obsm["X_umap"][:, 0]) + 12.0
    ax = scp.pl.lineage_plot(adata, ["Lineage"])
    save(ax.figure, "traj-2")
    save(scp.pl.dynamic_plot(adata, ["Lineage"], ["Sox9", "Neurog3", "Fev", "Ins1"],
                             layer=None, ncol=4, panel_size=(2.4, 2.0)), "traj-3")
    save(scp.pl.dynamic_heatmap(adata, ["Lineage"], markers, cell_bins=60,
                                layer=None, show_row_names=True).fig, "traj-4")
    save(scp.pl.cell_density_plot(adata, "Lineage", group_by="SubCellType",
                                  panel_size=(3.2, 3.0)), "traj-5")

    print("enrichment")
    rng = np.random.default_rng(11)
    terms = ["endocrine pancreas development", "hormone secretion",
             "regulation of insulin secretion", "peptide hormone processing",
             "epithelial cell differentiation", "Notch signaling pathway",
             "cell fate commitment", "glucose homeostasis"]
    enr = pd.DataFrame({
        "ID": [f"GO:{i:07d}" for i in range(len(terms))],
        "Description": terms,
        "pvalue": np.sort(rng.uniform(1e-9, 0.02, len(terms))),
        "p.adjust": np.sort(rng.uniform(1e-7, 0.04, len(terms))),
        "Count": rng.integers(5, 45, len(terms)),
        "Groups": "Endocrine",
        "Database": "GO_BP",
    })
    save(scp.pl.enrichment_plot(enr, plot_type="lollipop", panel_size=(4.2, 2.8)),
         "enrich-1")

    ranked = pd.Series(rng.normal(size=400),
                       index=[f"g{i}" for i in range(400)]).sort_values(ascending=False)
    curve = scp.pl.gsea_scores(ranked, [f"g{i}" for i in range(0, 110, 3)])
    table = pd.DataFrame({"ID": ["GS1"], "Description": ["Endocrine hormone secretion"],
                          "NES": [2.05], "p.adjust": [0.0008]})
    save(scp.pl.gsea_plot({"table": table, "curves": {"GS1": curve}},
                          panel_size=(3.6, 3.2)), "enrich-2")

    print("palettes")
    names = ["Paired", "Spectral", "Set1", "Dark2", "RdBu", "YlOrRd"]
    fig, axes = plt.subplots(len(names), 1, figsize=(6.5, 0.42 * len(names)))
    for ax, nm in zip(axes, names, strict=True):
        for i, c in enumerate(scp.continuous_palette(palette=nm, n=60)):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=c))
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(-1, 0.5, nm, ha="right", va="center", fontsize=9, family="monospace")
    fig.subplots_adjust(left=0.14, hspace=0.5)
    save(fig, "palettes")

    print(f"\nwrote {len(list(OUT.glob('*.png')))} figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
