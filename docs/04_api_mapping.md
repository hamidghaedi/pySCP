# R → Python API mapping

Status: **done** = implemented and tested · **stub** = final signature landed,
body raises · **n/a** = nothing to port.

## Plotting functions

| R (`SCP-plot.R`) | Line | Python | Status | Milestone |
|---|---|---|---|---|
| `CellDimPlot` | 1328 | `scp.pl.cell_dim_plot` | done (core; overlays stub) | 2 |
| `FeatureDimPlot` | 2042 | `scp.pl.feature_dim_plot` | stub | 3 |
| `CellDimPlot3D` | 2843 | `scp.pl.cell_dim_plot_3d` | stub | 8 |
| `FeatureDimPlot3D` | 3066 | `scp.pl.feature_dim_plot_3d` | stub | 8 |
| `FeatureStatPlot` | 3524 | `scp.pl.feature_stat_plot` | stub | 4 |
| `ExpressionStatPlot` | 3723 | `scp.pl.expression_stat_plot` | stub | 4 |
| `CellStatPlot` | 4463 | `scp.pl.cell_stat_plot` | stub | 5 |
| `StatPlot` | 4564 | `scp.pl.stat_plot` | stub | 5 |
| `FeatureCorPlot` | 5232 | `scp.pl.feature_cor_plot` | stub | 6 |
| `CellDensityPlot` | 5679 | `scp.pl.cell_density_plot` | stub | 6 |
| `VolcanoPlot` | 7088 | `scp.pl.volcano_plot` | stub | 6 |
| `LineagePlot` | 5918 | `scp.pl.lineage_plot` | stub | 12 |
| `PAGAPlot` | 6121 | `scp.pl.paga_plot` | stub | 11 |
| `GraphPlot` | 6287 | `scp.pl.graph_plot` | stub | 11 |
| `segementsDf` | 6666 | `scp.pl.shorten_segments` | stub | 11 |
| `VelocityPlot` | 6745 | `scp.pl.velocity_plot` | stub | 12 |
| `compute_velocity_on_grid` | 6963 | delegate to **scvelo** | stub | 12 |
| `GroupHeatmap` | 7911 | `scp.pl.group_heatmap` | stub | 7 |
| `FeatureHeatmap` | 9114 | `scp.pl.feature_heatmap` | stub | 7 |
| `FeatureCorHeatmap` | 9980 | `scp.pl.feature_cor_heatmap` | stub — **R version is an empty stub**, implement clean-room | 10 |
| `CellCorHeatmap` | 10113 | `scp.pl.cell_cor_heatmap` | stub | 10 |
| `DynamicHeatmap` | 11020 | `scp.pl.dynamic_heatmap` | stub | 9 |
| `DynamicPlot` | 12102 | `scp.pl.dynamic_plot` | stub | 9 |
| `GroupTreePlot` | 12472 | — | **n/a — the R function is empty**; use `sc.tl.dendrogram` | — |
| `ProjectionPlot` | 12511 | `scp.pl.projection_plot` | stub | 13 |
| `EnrichmentPlot` | 12727 | `scp.pl.enrichment_plot` | stub | 14 |
| `adjustlayout` | 13441 | `scp.pl.adjust_layout` | stub | 14 |
| `GSEAPlot` | 13563 | `scp.pl.gsea_plot` | stub | 15 |
| `gseaScores` | 14602 | `scp.pl.gsea_scores` | **done** | 15 |

## Infrastructure

| R | Python | Status |
|---|---|---|
| `theme_scp` | `scp.theme_scp` + `scp.apply_theme` | done |
| `theme_blank` | `scp.theme_blank` + `scp.theme.add_coord_arrows` | done |
| `palette_scp` | `scp.palette_scp` / `discrete_palette` / `continuous_palette` | done |
| `show_palettes` | `scp.list_palettes` | done |
| `palette_list` (data) | `scp.palettes.PALETTES` (229) | done |
| `adjcolors` | `scp.adjcolors` | done |
| `blendcolors`, `Blend2Color`, `BlendRGBList`, `RGBA2RGB` | `scp.colors.*` | done |
| `matrix_process`, `zscore_matrix`, `fc_matrix`, `log2fc_matrix`, `log1p_matrix` | `scp.matrix_process` | done |
| `heatmap_rendersize` / `heatmap_fixsize` | `HeatmapSpec.figsize()` | done (sizing); renderer todo |
| `panel_fix` / `panel_fix_overall` | `scp.layout.PanelGrid` + `panel_size_for` | done |
| `cluster_within_group2` | `scp.heatmap` (todo) | stub |
| `heatmap_enrichment` | pluggable `enrichment_fn` | stub |
| `get_legend` / `add_grob` | `scp.layout.LegendColumn` | done |
| `DefaultReduction` | `scp.default_reduction` | done |
| `FetchData` (SCP's own resolver) | `scp.fetch_data` | done |
| `check_DataType` | `scp.fetch.infer_data_type` | done |
| `drop_data` / `slim_data` | — n/a (no ggplot data duplication problem) | — |
| `as_grob` / `as_gtable` / `build_patchwork` | — n/a | — |

## Argument renaming rules

Applied consistently; deviate only with a note in the function docstring.

| R | Python | Note |
|---|---|---|
| `srt` | `adata` | positional |
| `group.by` | `group_by` | all dots → underscores |
| `slot` | `layer` | `"data"`→`None`, `"counts"`→`"counts"`, `"scale.data"`→`"scaled"` |
| `assay` | *dropped* | the AnnData is the assay; use MuData for multimodal |
| `cells` | `cells` | obs-name sequence |
| `pt.size` | `pt_size` | |
| `cells.highlight` | `cells_highlight` | |
| `label.bg.r` | `label_bg_r` | |
| `theme_use` + `theme_args` | `theme=` (a `Theme` object) | strings-as-function-names don't belong in a Python API |
| `aspect.ratio` | `aspect_ratio` | |
| `combine` / `nrow` / `ncol` / `byrow` | `nrow` / `ncol` / `byrow` | `combine` dropped — see architecture doc |
| `force` | `force` | turns a hard error into a warning, never an interactive prompt |
| `seed` | `seed` | default 11, matching R |
| `NA_color` / `NA_keep` | `na_color` / `na_keep` | |
| `show_na` | `show_na` | |
| `exp_method` | `exp_method` | |
| `save` | *dropped* | use `fig.savefig` |
| `return_layer` | `ax=` | pass an Axes to draw an overlay into an existing panel |

## Things deliberately not ported

* `combine=FALSE` returning a named list — `fig.axes` already gives access.
* `drop_data` / `slim_data` — these exist because ggplot objects embed their
  data; matplotlib artists don't have the problem.
* `save=` on every function — one obvious way (`fig.savefig`) beats thirty.
* The interactive `askYesNo()` guards — replaced by `force=True`.
* The `values`/`valus` typo bug in `ExpressionStatPlot`'s `same.y.lims`
  quantile branch, and the undefined `status` in `FeatureCorPlot`'s
  `calculate_coexp` branch. Both are live R bugs; fix them and note it.
* `srt@assays[setdiff(...)] <- NULL` object mutation in
  `ExpressionStatPlot(plot.by="feature")` — an R-only hack to steer
  `FetchData`; just pass `layer=`.
