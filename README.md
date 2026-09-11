# scp-plot

**SCP's plotting grammar, for scanpy.**

A port of the plotting layer of the R package
[SCP](https://github.com/zhanghao-njmu/SCP) (Hao Zhang, Nanjing Medical
University, GPL-3) — 229 curated palettes, a consistent theme, and ~30 figure
functions — reading **AnnData** directly instead of Seurat.

> **Status: scaffold + spec.** The foundation (palettes, color algebra, theme,
> data access, panel layout, heatmap matrix transforms) is implemented and
> tested. `cell_dim_plot` works as the reference implementation. Everything
> else carries its final signature and raises `NotImplementedError` with a
> pointer to its spec. See [`docs/07_milestones.md`](docs/07_milestones.md).

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
scp.list_palettes()                          # 229, extracted from SCP's own .rda
scp.discrete_palette(["A", "B"], "Paired")   # verbatim slice, not interpolation
scp.blendcolors(["#FF0000", "#00FF00"], "screen")
scp.matrix_process(M, "zscore")              # ddof=1, matching R's scale()
scp.fetch_data(adata, ["CD3E", "leiden", "UMAP_1"])
scp.default_reduction(adata)
scp.pl.cell_dim_plot(adata, "leiden", split_by="batch", label=True)
```

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
