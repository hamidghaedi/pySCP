# R → Python API mapping

Status: **done** = implemented and tested · **stub** = final signature landed,
body raises · **n/a** = nothing to port.

## Plotting functions

| R (`SCP-plot.R`) | Line | Python | Status | Milestone |
|---|---|---|---|---|
| `CellDimPlot` | 1328 | `scp.pl.cell_dim_plot` | done (all overlays; `stat_plot_type` limited to pie/ring/bar) | 2 |
| `FeatureDimPlot` | 2042 | `scp.pl.feature_dim_plot` | done (standard + blend, all overlays, `calculate_coexp`) | 3 |
| `CellDimPlot3D` | 2843 | `scp.pl.cell_dim_plot_3d` | done (traces, NA trace, ranges, title verified vs R; `cells.highlight`/`lineages` not ported) | 8 |
| `FeatureDimPlot3D` | 3066 | `scp.pl.feature_dim_plot_3d` | done (incl. `calculate_coexp`; values verified vs R; Spectral scale deliberate) | 8 |
| `FeatureStatPlot` | 3524 | `scp.pl.feature_stat_plot` | done (all but `flip`; `stack`, `plot_by='feature'`, significance layer and `calculate_coexp` verified vs R) | 4 |
| `ExpressionStatPlot` | 3723 | `scp.pl.expression_stat_plot` | done (the DataFrame engine; `flip` stub) | 4 |
| `CellStatPlot` | 4463 | `scp.pl.cell_stat_plot` | done (11/11 types + `individual`; sankey/chord/venn verified vs R) | 5 |
| `StatPlot` | 4564 | `scp.pl.stat_plot` | done (11/11 types + `individual`; sankey/chord/venn verified vs R) | 5 |
| `FeatureCorPlot` | 5232 | `scp.pl.feature_cor_plot` | done (incl. `calculate_coexp`) | 6 |
| `CellDensityPlot` | 5679 | `scp.pl.cell_density_plot` | done | 6 |
| `VolcanoPlot` | 7088 | `scp.pl.volcano_plot` | done | 6 |
| `LineagePlot` | 5918 | `scp.pl.lineage_plot` | done (curves, trim, whiskers, legend verified vs R) | 12 |
| `PAGAPlot` | 6121 | `scp.pl.paga_plot` | done (verified vs R; `cells=` R bug not reproduced) | 11 |
| `GraphPlot` | 6287 | `scp.pl.graph_plot` | done (verified vs R; curved lines, highlights, `node_alpha` not ported) | 11 |
| `segementsDf` | 6666 | `scp.pl.shorten_segments` | done | 11 |
| `VelocityPlot` | 6745 | `scp.pl.velocity_plot` | done (raw and grid verified vs R; stream not checkable, metR) | 12 |
| `compute_velocity_on_grid` | 6963 | `scp.pl.compute_velocity_on_grid` | done — numpy port of SCP's (not scvelo's); grid and stream verified vs R | 12 |
| `GroupHeatmap` | 7911 | `scp.pl.group_heatmap` | done — matrices, `split_by`, `grouping_var`, splits and row/column orders verified vs R; enrichment tracks from a table / `enrichment_fn`; `cell_annotation` drawn as per-group lanes, not pies/violins | 7 |
| `FeatureHeatmap` | 9114 | `scp.pl.feature_heatmap` | done — matrix and splits verified vs R (the sampling RNG differs) | 7 |
| `FeatureCorHeatmap` | 9980 | `scp.pl.feature_cor_heatmap` | done — clean-room; **the R version is an empty stub** | 10 |
| `CellCorHeatmap` | 10113 | `scp.pl.cell_cor_heatmap` | done — all 19 `distance_metric`s verified vs proxyC | 10 |
| `DynamicHeatmap` | 11020 | `scp.pl.dynamic_heatmap` | done — per-cell columns, joint z-score and peaktime splits verified vs R given R's `DynamicFeatures`; `use_fitted` deliberately raises; `separate_annotation` stub | 9 |
| `DynamicPlot` | 12102 | `scp.pl.dynamic_plot` | done (points, lib-normalisation, panels verified vs R; the mgcv fit is not) | 9 |
| `GroupTreePlot` | 12472 | — | **n/a — the R function is empty**; use `sc.tl.dendrogram` | — |
| `ProjectionPlot` | 12511 | `scp.pl.projection_plot` | done | 13 |
| `EnrichmentPlot` | 12727 | `scp.pl.enrichment_plot` | done (all 7 types; bar/dot/lollipop frames, labels, encodings and sizes verified vs R, see `docs/parity_audit.md`; network/enrichmap graphs, partitions and word tables verified vs R; GO wordcloud keywords substituted) | 14 |
| `adjustlayout` | 13441 | `scp.pl.adjust_layout` | done (transcribed; step-2 sign bug fixed) | 14 |
| `GSEAPlot` | 13563 | `scp.pl.gsea_plot` | done (all 6 types; line/bar selection, order, fill, alpha, peak, zero cross and heat strip verified vs R; comparison/network/enrichmap/wordcloud tables verified vs R) | 15 |
| `gseaScores` | 14570 | `scp.pl.gsea_scores` | **done** (verified vs R to 1e-12) | 15 |

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
| `heatmap_rendersize` / `heatmap_fixsize` | `HeatmapSpec.figsize()` + `heatmap.render` | done |
| `panel_fix` / `panel_fix_overall` | `scp.layout.PanelGrid` + `panel_size_for` | done |
| `cluster_within_group2` (plus ComplexHeatmap's `cluster_within_group` and `make_cluster` ordering) | `scp.heatmap.cluster` | done |
| `standardise` / `mestimate` (Mfuzz) | `scp.heatmap.cluster` | done |
| `heatmap_enrichment` | `enrichment=` table or `enrichment_fn=`; `scp.heatmap.enrichment` (word boxes, `ora`) | done |
| `RunKNNPredict` similarity (`proxyC::simil` / `dist`) | `scp.heatmap.similarity` | done |
| `get_legend` / `add_grob` | `scp.layout.LegendColumn` | done |
| `theme_sankey`, `theme_alluvial`, `theme_sankey_bump` (vendored `ggsankey.R`) | — n/a: exported by SCP but used by no SCP function; `StatPlot`'s sankey draws on `theme_void()` + x-axis text, which `stat_plot(plot_type="sankey")` reproduces on `theme_blank(add_coord=False)` | — |
| `make_long` / `geom_sankey` (vendored `ggsankey.R`) | `scp.pl._stat_flow.sankey_layout` (node/flow geometry) | done |
| `circlize::chordDiagram` (as `StatPlot` calls it) | `scp.pl._stat_flow.chord_layout` (sectors, link positions) | done |
| `ggVennDiagram::process_data` (1.2.x shapes) | `scp.pl._stat_flow.venn_regions` + `scp.pl._venn_shapes` | done |
| `DefaultReduction` | `scp.default_reduction` | done |
| `FetchData` (SCP's own resolver) | `scp.fetch_data` | done |
| `check_DataType` | `scp.fetch.infer_data_type` | done |
| `drop_data` / `slim_data` | — n/a (no ggplot data duplication problem) | — |
| `as_grob` / `as_gtable` / `build_patchwork` | — n/a | — |

## Signature changes made during the port

| Function | Change | Why |
|---|---|---|
| `feature_stat_plot` | added `title`, `subtitle`, `xlab`, `ylab="Expression level"` | The scaffold signature omitted them; they are parameters of the R original and `ylab`'s default is visible on every panel. |
| `expression_stat_plot` | the stub's `(data, *args, **kwargs)` became `feature_stat_plot`'s signature with `adata`→`data` and without `layer`, `use_raw`, `calculate_coexp` | It is the DataFrame engine, so the matrix-selection arguments belong to the wrapper; `CoExp` needs the gene matrix, which a frame does not carry. |
| `feature_cor_plot` | added `calculate_coexp=False` | A parameter of `FeatureCorPlot` the scaffold omitted. In R it always errors (undefined `status`); here it goes through `scp.fetch.calculate_coexp`. |
| `cell_density_plot` | added `y_min`, `y_max`, `y_nbreaks`, `keep_empty`, `aspect_ratio`, `legend_position`, `nrow`, `ncol`, `byrow` | Same omission; all nine are parameters of `CellDensityPlot` and the layout ones are needed for multi-feature figures. |
| `enrichment_plot` | added `compare_only_sig`, `network_labelsize`, `network_adjscale`, `network_adjiter`, `enrichmap_layout`, `enrichmap_label`, `enrichmap_labelsize`, `enrichmap_show_keyword`, `enrichmap_expand`, `keyword_background`; `panel_size` now defaults to `None` (per-type size) | The first eight are `EnrichmentPlot` parameters the scaffold left out. `enrichmap_nlabel` keeps its corrected spelling (R: `enrlichmap_nlabel`). `keyword_background` is new: a `{GO ID: term name}` map that lets the GO wordcloud run simplifyEnrichment's keyword test, whose term-document matrix only ships with the R package. |
| `gsea_plot` | added `group_use`, `id_use`, `pvalue_cutoff`, `compare_only_sig`, `character_width`, `palcolor`, the whole `network_*` / `enrichmap_*` / wordcloud block of `enrichment_plot`, `keyword_background`, `nrow`, `ncol`; `top_term` now defaults to `None` (6, or 100 for `enrichmap`, as R's `ifelse`); `panel_size` defaults to `None` | All are `GSEAPlot` parameters needed by the types it shares with `EnrichmentPlot`. Also accepts SCP's own `{"enrichment": df, "results": {key: df}}` shape, where a key's entry may also be `{"result": df, "geneList": Series, "geneSets": {ID: genes}, "exponent": 1.0}` (the `gseaResult` slots `gsInfo` reads, from which `line` computes its running scores), or that per-key mapping bare; `{"table": df, "curves": ...}` takes curves keyed by ID or by key then ID. No signature change came with the parity fix: `line` and `bar` now draw one panel per result key on a `PanelGrid`, as R loops `names(res)`. |
| `cell_dim_plot` | added the settings of every overlay: `hex`, `hex_linewidth`, `hex_count`, `hex_bins`, `hex_binwidth`; `label_point_size`, `label_point_color`, `label_segment_color`; `alpha_highlight`; `density_color`, `density_filled_palette`, `density_filled_palcolor`; `mark_expand`, `mark_alpha`, `mark_linetype`; `edge_size`, `edge_alpha`, `edge_color`; the eleven `lineages_*`, eleven `paga_*` and nine `velocity_*` parameters; `streamline_width`, `streamline_alpha`, `streamline_color`, `streamline_palette`, `streamline_palcolor`, `streamline_bg_color`, `streamline_bg_stroke`; `stat_type`, `stat_plot_type`, `stat_plot_position`, `stat_plot_size`, `stat_plot_palette`, `stat_palcolor`, `stat_plot_alpha`, `stat_plot_label`, `stat_plot_label_size` | The scaffold declared the overlay switches but not their settings. All are `CellDimPlot` parameters with R's defaults, keyword-only, so no existing call changes. Lengths stay ggplot millimetres; `mark_expand` is a number of mm rather than a `unit`, and `lineages_arrow` is the head length in inches (or `None`) rather than a grid `arrow`. `velocity_n_neighbors=None` means R's `ceiling(n_cells / 50)`. metR's `streamline_L`, `streamline_minL`, `streamline_res` and `streamline_n` are not added, because matplotlib's `streamplot` has nothing they could control. `velocity` names the scvelo key: `velocity="velocity"` reads `obsm["velocity_umap"]`. |
| `feature_dim_plot` | added `hex`, `hex_linewidth`, `hex_color`, `hex_bins`, `hex_binwidth`; `label_repulsion`, `label_point_size`, `label_point_color`, `label_segment_color`; `alpha_highlight`; `density_color`, `density_filled_palette`, `density_filled_palcolor`; `edge_size`, `edge_alpha`, `edge_color`; the eleven `lineages_*` parameters | The same omission; all are `FeatureDimPlot` parameters with R's defaults and units. |
| `feature_dim_plot_3d` | added `calculate_coexp` | A `FeatureDimPlot3D` parameter the scaffold left out. It appends a `CoExp` trace through the shared `scp.fetch.calculate_coexp`. |
| `feature_dim_plot_3d` | `bg_cutoff` default `0.0` → `None`; `pt_size` `2.0` → `1.5` | `FeatureDimPlot3D` has no cutoff and passes every value to plotly; the copied 2-D default NaN-masked 76-80% of a sparse gene and every negative signed score. `bg_cutoff` stays as a Python-only opt-in. 1.5 is R's `pt.size`. |
| `cell_dim_plot_3d` | `show_na` default `False` → `True`; `pt_size` `2.0` → `1.5` | `CellDimPlot3D` has no `show_na` switch: it always draws NA cells as an `NA` trace in `bg_color`. `show_na=False` remains as a Python-only way to drop them. |
| `graph_plot` | added `node_palcolor`, `transition_size=(0.2, 1)`, `transition_color`, `transition_alpha`, `legend_position`, `legend` | `GraphPlot` parameters the scaffold left out; without `transition_size` the net flows could not be rescaled as R does. `legend=` takes a `LegendColumn` to register the node legend in when drawing into a borrowed `ax`. |
| `lineage_plot` | added `cells`, `palcolor`, `lineages_arrow=0.1`, `whiskers_linewidth`, `whiskers_alpha`, `legend_position`, `legend`; `linewidth` is now ggplot mm | `LineagePlot` parameters; `whiskers` was accepted but could not be drawn without them. `lineages_arrow` is the head length in inches (or `None`), as in `cell_dim_plot`. The function no longer draws a grey layer of every cell: it is a layer, as `LineagePlot` is. |
| `velocity_plot` | added `cells`, `group_palette`, `group_palcolor`, `arrow_angle`, the seven `streamline_*` settings, `legend_position`, `legend`, `seed` | `VelocityPlot` parameters. `group_by` colours the arrows, so it needs its palette; stream mode now goes through the overlay layer `cell_dim_plot` uses, which takes the streamline settings. No cell layer is drawn any more. |
| `dynamic_plot` | added `cells`, `lib_normalize`, `libsize`, `line_size`, `line_palette`, `line_palcolor`, `pt_size`, `point_palette`, `point_palcolor`, `legend_position`, `seed`; `palette=` (via `**kwargs`) stays as an alias of `line_palette` | `DynamicPlot` parameters. `lib_normalize` (default: `layer == "counts"`) and `libsize` are what R's default `slot = "counts"` path needs; the palettes are R's two separate ones. |
| `group_heatmap` | added `libsize`, `fuzzification`, `enrichment`, `db`, `padjust_cutoff`, `pvalue_cutoff`, `show_termid`, `top_word`, `words_excluded`, `terms_width` / `keys_width` / `features_width`, `terms_fontsize` / `keys_fontsize` / `features_fontsize`, `group_palette`, `cell_split_palette`, `feature_split_palette`; `split_method` narrowed to `kmeans` / `hclust` / `mfuzz` | All are `GroupHeatmap` parameters the splitting and enrichment paths need. `enrichment` stands in for R's database download: the table is an input, `enrichment_fn` the alternative. The `*-peaktime` methods only exist in `DynamicHeatmap`. |
| `feature_heatmap` | `**kwargs` replaced by the explicit `FeatureHeatmap` parameters (`split_by`, `cells`, `cell_order`, `libsize`, and the split / cluster / annotation / enrichment set shared with `group_heatmap`) | `**kwargs` silently swallowed misspelt options. `show_row_names` keeps its `True` default. |
| `dynamic_heatmap` | `**kwargs` replaced by explicit parameters; added `dynamic_features`, `min_expcells`, `r_sq`, `dev_expl`, `padjust`, `feature_split`, `feature_split_by`, `fuzzification`, `cluster_*`, the annotation and enrichment sets; `split_method` default `"kmeans-peaktime"` -> `"mfuzz"`; `palette` default `"viridis"` | Both defaults are R's. `dynamic_features` carries `RunDynamicFeatures`' table (peak and valley times), since this package does not refit mgcv. `cell_bins` keeps R's meaning — the bins for `cell_density` thinning — rather than binning the columns. |

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
