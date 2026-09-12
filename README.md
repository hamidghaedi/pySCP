# scp-plot

**SCP's plotting grammar, for scanpy.**

A port of the plotting layer of the R package
[SCP](https://github.com/zhanghao-njmu/SCP) (Hao Zhang, Nanjing Medical
University, GPL-3) — 229 curated palettes, a consistent theme, and ~30 figure
functions — reading **AnnData** directly instead of Seurat.

> **Status: first pass complete, pre-alpha.** 27 of the 28 functions in
> `scp.pl` are implemented; 117 tests pass. Options that are not ported raise
> `NotImplementedError` naming the brief section that specifies them, rather
> than silently doing something else — see the table at the end of
> [`docs/07_milestones.md`](docs/07_milestones.md).
>
> *Implemented* and *verified against R* are different claims, and this README
> keeps them apart. See [Parity](#parity) below.

```python
import scanpy as sc
import scp

adata = sc.datasets.pbmc3k_processed()

scp.pl.cell_dim_plot(adata, "louvain", label=True, theme=scp.theme_blank())
scp.pl.feature_dim_plot(adata, ["CD3E", "MS4A1"], compare_features=True)   # soon
scp.pl.group_heatmap(adata, features=genes, group_by="louvain", add_dot=True)  # soon
```

## Why

scanpy's plotting is deliberately minimal. SCP's is not: it has a coherent
grammar for grouping, splitting, faceting, labelling and legend composition
that carries across every figure type, and the results are publication-ready
without post-processing. That grammar is worth having in Python, but it is
locked behind Seurat and 14,794 lines of R.

## What exists today

```python
# dimensional reduction
scp.pl.cell_dim_plot(adata, "leiden", split_by="batch", label=True)
scp.pl.feature_dim_plot(adata, ["CD3E", "MS4A1"], compare_features=True)
scp.pl.cell_dim_plot_3d(adata, "leiden")                       # plotly

# distributions and composition
scp.pl.feature_stat_plot(adata, "CD3E", group_by="leiden", plot_type="violin")
scp.pl.cell_stat_plot(adata, "leiden", group_by="batch", plot_type="ring")
scp.pl.volcano_plot(scp.io.de_from_rank_genes_groups(adata))
scp.pl.cell_density_plot(adata, "pseudotime", group_by="leiden")
scp.pl.feature_cor_plot(adata, ["CD3E", "MS4A1", "LYZ"])

# heatmaps
scp.pl.group_heatmap(adata, features=genes, group_by="leiden", add_dot=True)
scp.pl.feature_heatmap(adata, features=genes, group_by="leiden")
scp.pl.cell_cor_heatmap(query, reference, ref_group="celltype")

# trajectory and enrichment
scp.pl.paga_plot(adata, label=True)
scp.pl.lineage_plot(adata, ["Lineage1"])
scp.pl.enrichment_plot(enrichment_df, plot_type="lollipop")
scp.pl.gsea_plot({"table": tab, "curves": curves})

# the layer everything is built on
scp.list_palettes()                          # 229, extracted from SCP's own .rda
scp.discrete_palette(["A", "B"], "Paired")   # verbatim slice, not interpolation
scp.blendcolors(["#FF0000", "#00FF00"], "screen")
scp.matrix_process(M, "zscore")              # ddof=1, matching R's scale()
scp.fetch_data(adata, ["CD3E", "leiden", "UMAP_1"])
```

## Parity

`notebooks/01_parity_foundation.ipynb` runs SCP and this port side by side on
`pancreas_sub`, the dataset behind 21 of the 22 plotting examples in
`SCP-plot.R`, and compares them — numerically wherever a number is available,
because the risk in a port like this is a figure that looks plausible and is
quietly wrong.

**Checked against R, and agreeing:**

| | |
|---|---|
| Categorical level order | identical end to end |
| Discrete palettes | byte-identical (verbatim-slice rule) |
| Interpolated ramp | within 1/255, the documented `colorRampPalette` tolerance |
| `blendcolors` (4 modes), `adjcolors` | identical |
| `matrix_process` `zscore` (ddof=1), `log2fc` | identical |
| `cell_dim_plot` legend | identical text, order and counts |
| `feature_dim_plot` `nPos` and percentages | identical on all four genes tested |
| `group_heatmap` matrix vs R's `matrix_list` | identical to 8.6e-07 |
| `cell_stat_plot` cross-tab and stacking order | identical |

**Implemented but not yet checked against R:** the trajectory family, the
enrichment and GSEA views, the 3-D plots, the correlation heatmaps, and
`volcano_plot` / `cell_density_plot` / `feature_cor_plot`. Several of these need
SCP's own analysis pipeline (`RunSlingshot`, `RunDEtest`, `RunEnrichment`) to
produce comparable input; standing that up would test the pipeline rather than
the plotting layer.

**Known not to match, by decision:** anything depending on a fitted GAM.
`mgcv` selects its smoothing parameter by GCV/REML and `pygam` does not
reproduce that criterion, so `dynamic_plot`'s smooth differs from R's by more
than numerical noise and `dynamic_heatmap(use_fitted=True)` raises rather than
pretending otherwise. The binned form of the figure needs no fit and is
provided. Reasoning is in `docs/07_milestones.md` §9.

## Install

```bash
pip install -e ".[dev]"
```

Core deps are anndata, matplotlib, numpy, pandas, scipy. Everything heavier is
an extra: `[gam]`, `[graph]`, `[enrich]`, `[velocity]`, `[interactive]`,
`[raster]`, `[fuzzy]`, or `[all]`.

## Documentation

| | |
|---|---|
| [Overview](docs/00_overview.md) | what this is, decisions already made |
| [Architecture](docs/01_architecture.md) | the draw-into-axes inversion, sizing, legends |
| [Data contract](docs/02_data_contract.md) | Seurat → AnnData, `fetch_data`, gotchas |
| [Theme & palette](docs/03_theme_and_palette.md) | why the color layer looks the way it does |
| [API mapping](docs/04_api_mapping.md) | every R function → its Python name and status |
| [Porting briefs](docs/porting_briefs/) | per-family implementation detail |
| [Testing](docs/06_testing.md) | golden fixtures over image diffs |
| [Milestones](docs/07_milestones.md) | ordered work plan |
| [**Agent handoff**](docs/08_agent_handoff.md) | **start here to contribute** |

## Credit and licence

All design credit belongs to **Hao Zhang** and the SCP authors. This is a
derivative work of SCP's plotting layer and is licensed **GPL-3.0-or-later**,
as SCP is.

If you use it, cite SCP: <https://github.com/zhanghao-njmu/SCP>
