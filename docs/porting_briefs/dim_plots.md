# Porting brief: dimensional-reduction plots

**Source**

- **Upstream:** `SCP/R/SCP-plot.R` — `CellDimPlot` L1328, `FeatureDimPlot` L2042, `CellDimPlot3D` L2843, `FeatureDimPlot3D` L3066
- **Target:** `src/scp/pl/_dim.py`
- **Milestones:** 2 (`cell_dim_plot` overlays), 3 (`feature_dim_plot`), 8 (3D)

Supporting upstream helpers: `palette_scp` (L168), `theme_scp` (L16), `theme_blank` (L84), `adjcolors` (L1000), `blendcolors` / `Blend2Color` / `BlendRGBList` / `RGBA2RGB` (L1000–1129), `LineagePlot` (L5918), `PAGAPlot` (L6121), `VelocityPlot` (L6745), `CellStatPlot`→`StatPlot` (L4463/L4564), `get_legend` / `add_grob` (L14743/L14755), `DefaultReduction` (`SCP/R/SCP-workflow.R:1034`).

---

## 0. Shared infrastructure

### 0.1 `DefaultReduction(srt, pattern=NULL, min_dim=2)`

Fuzzy reduction resolver. Filters `srt@reductions` to those with `ncol(cell.embeddings) >= min_dim`. If only one, returns it. If `pattern` is `NULL`, uses `srt@misc[["Default_reduction"]]` if present, else the ordered list `c("umap","tsne","dm","phate","pacmap","trimap","largevis","fr","pca","svd","ica","nmf","mds","glmpca")`. It then appends `paste0(pattern, min_dim, "D")` variants, tries exact match, then `grep(ignore.case=TRUE)`, then `agrep(max.distance=0.1)` (approximate/fuzzy string match), and finally breaks ties by preferring names matching the default patterns plus `"2D"`/`"3D"` and choosing the reduction with the *fewest* columns.

**Python equivalent:** already implemented as `scp.fetch.default_reduction` / `reduction_key` — resolve `reduction="umap"` → `adata.obsm["X_umap"]` with the same case-insensitive priority order; the fuzzy `agrep` step is dropped. See `docs/02_data_contract.md` §4 for the resolution order and the `"X_umap"` → `"UMAP_"` key synthesis rule that supplies axis labels.

### 0.2 `palette_scp`

The single upstream color entry point: name → color vector from `palette_list` (~255 palettes, each tagged `"discrete"` or `"continuous"`), with discrete levels taken in *order of appearance*, an exact-slice-when-it-fits / interpolate-otherwise rule, and `NA_color` handling. Two SCP-specific values matter downstream: `"Spectral"` is **reversed relative to RColorBrewer** (`#5E4FA2 … #9E0142`, so low = purple/blue, high = red) and is `FeatureDimPlot`'s default; `"Paired"` (standard Brewer, 12 colors) is `CellDimPlot`'s default, and `"Set1"` is the blend-mode default.

This is implemented — `scp.palettes.discrete_palette` / `continuous_palette`. See `docs/03_theme_and_palette.md`; do not re-derive it here.

### 0.3 Themes

`theme_scp` is a white-background, black-1pt-panel-border theme with a fixed type scale (title 14, subtitle 13 left-aligned, axis title 13, axis text 12, strip 12.5, legend key 10pt), all scaled by `base_size/12`. `theme_blank` blanks the frame entirely and instead draws two corner arrows (15% of the panel along each axis, 2pt, with a 0.02-npc head) labelled with the axis names — the "little UMAP axis arrows" idiom.

Both are implemented as `scp.theme.theme_scp` / `Theme(blank=True)`; see `docs/03_theme_and_palette.md`. The only behaviour the plot functions must remember is that `show_stat` defaults to `FALSE` under a blank theme, and that the blank theme consumes `xlab`/`ylab` into the arrow labels rather than axis labels.

### 0.4 `get_legend` / `add_grob`

Upstream composes **multiple independent legends** (base scale + lineages + paga + velocity + stat pies) by rendering each from a throwaway ggplot, extracting its `guide-box` grob, `cbind`/`rbind`-ing the gtables and splicing the result onto the plot gtable. In Python this is just "build N legend artists and stack them in one column".

Implemented as `scp.layout.LegendColumn` (plus `PanelGrid` for the panel assembly); see `docs/01_architecture.md`.

---

## 1. `CellDimPlot` (L1328–1851)

### 1.0 Implementation status

`src/scp/pl/_dim.py::cell_dim_plot` is the **reference implementation** for the whole package. Already done and not to be re-litigated:

- reduction/key resolution, `dims`, `cells` subsetting, appearance-ordered categoricals, `show_na`;
- the per-panel frame with **NA-masking rather than dropping** of other-split cells (§1.2);
- global axis limits computed before `cells` subsetting (§1.8);
- `pt.size = min(3000/n, 0.5)` auto-size and the `raster` threshold at 1e5 cells (§1.8);
- the seeded background-first / shuffled-foreground draw order (§1.8);
- discrete coloring with `bg_color` for NA, per-panel legend restriction to present levels, `show_stat` counts in legend labels and the `nCells:` subtitle (§1.4);
- marginal-median label placement, `label_insitu` numbering and the `"1: Name"` legend, halo text via `path_effects` (§1.5);
- the two-layer highlight stroke, `cells_highlight=True` sentinel (§1.6);
- panel naming `"<split>:<group>"`, `nrow`/`ncol`/`byrow` assembly, `LegendColumn` (§1.7).

**Remaining milestone-2 work** is everything in §1.6 that currently raises `NotImplementedError`: `add_density` (+ `density_filled`), `add_mark`, `graph` edges, `lineages`, `paga`, `velocity`, `stat_by` pies. Each must be inserted at its exact position in the layer order of §1.3 — the overlays are drawn *under* the points (`add_mark`, `graph`, `add_density`) or *over* them (`stat_by`, `lineages`, `paga`, `velocity`), and labels always come last.

**Known signature gaps** to close while doing milestone 2: `hex`/`hex.count`/`hex.bins`/`hex.binwidth`/`hex.linewidth`, `alpha.highlight`, `label_point_size` / `label_point_color` / `label_segment_color`, and the `mark_expand` / `mark_alpha` / `mark_linetype` trio. `label_repel` is declared but currently ignored (labels are drawn unrepelled); wire it to an actual repulsion pass (`adjustText` or a hand-rolled force loop) when the overlays land.

### 1.1 Full parameter list (with defaults)

**Data selection (Seurat-specific except `dims`/`cells`)**

| arg | default | notes |
|---|---|---|
| `srt` | — | Seurat object (→ AnnData) |
| `group.by` | — | one or more `meta.data` column names (→ `adata.obs`) |
| `reduction` | `NULL` | resolved via `DefaultReduction` |
| `dims` | `c(1,2)` | 1-based dim indices |
| `split.by` | `NULL` | one `meta.data` column; `NULL` → synthetic `"All.groups"` factor with single level `""` |
| `cells` | `NULL` | cell-name subset |
| `graph` | `NULL` | name in `srt@graphs` |
| `paga` | `NULL` | usually `srt@misc$paga` |
| `velocity` | `NULL` | mode string, e.g. `"stochastic"`; looks up reduction `paste0(velocity,"_",reduction)` |
| `lineages` | `NULL` | `meta.data` columns of pseudotime |
| `stat.by` | `NULL` | `meta.data` column for pies |

**Point aesthetics**: `show_na=FALSE`, `pt.size=NULL` (auto), `pt.alpha=1`, `palette="Paired"`, `palcolor=NULL`, `bg_color="grey80"`, `raster=NULL` (auto), `raster.dpi=c(512,512)`, `hex=FALSE`, `hex.linewidth=0.5`, `hex.count=TRUE`, `hex.bins=50`, `hex.binwidth=NULL`.

**Labels**: `label=FALSE`, `label.size=4`, `label.fg="white"`, `label.bg="black"`, `label.bg.r=0.1`, `label_insitu=FALSE`, `label_repel=FALSE`, `label_repulsion=20`, `label_point_size=1`, `label_point_color="black"`, `label_segment_color="black"`.

**Highlight**: `cells.highlight=NULL` (vector or `TRUE`), `cols.highlight="black"`, `sizes.highlight=1`, `alpha.highlight=1`, `stroke.highlight=0.5`.

**Statistics / annotation**: `show_stat=ifelse(identical(theme_use,"theme_blank"),FALSE,TRUE)`, `add_density=FALSE`, `density_color="grey80"`, `density_filled=FALSE`, `density_filled_palette="Greys"`, `density_filled_palcolor=NULL`, `add_mark=FALSE`, `mark_type=c("hull","ellipse","rect","circle")`, `mark_expand=unit(3,"mm")`, `mark_alpha=0.1`, `mark_linetype=1`.

**Stat pies**: `stat_type="percent"`, `stat_plot_type="pie"`, `stat_plot_position=c("stack","dodge")`, `stat_plot_size=0.15`, `stat_plot_palette="Set1"`, `stat_palcolor=NULL`, `stat_plot_alpha=1`, `stat_plot_label=FALSE`, `stat_plot_label_size=3`.

**Graph edges**: `edge_size=c(0.05,0.5)`, `edge_alpha=0.1`, `edge_color="grey40"`.

**Lineages**: `lineages_trim=c(0.01,0.99)`, `lineages_span=0.75`, `lineages_palette="Dark2"`, `lineages_palcolor=NULL`, `lineages_arrow=arrow(length=unit(0.1,"inches"))`, `lineages_linewidth=1`, `lineages_line_bg="white"`, `lineages_line_bg_stroke=0.5`, `lineages_whiskers=FALSE`, `lineages_whiskers_linewidth=0.5`, `lineages_whiskers_alpha=0.5`.

**PAGA**: `paga_type="connectivities"`, `paga_node_size=4`, `paga_edge_threshold=0.01`, `paga_edge_size=c(0.2,1)`, `paga_edge_color="grey40"`, `paga_edge_alpha=0.5`, `paga_transition_threshold=0.01`, `paga_transition_size=c(0.2,1)`, `paga_transition_color="black"`, `paga_transition_alpha=1`, `paga_show_transition=FALSE`.

**Velocity**: `velocity_plot_type="raw"` (`raw`/`grid`/`stream`), `velocity_n_neighbors=ceiling(ncol(srt@assays[[1]])/50)`, `velocity_density=1`, `velocity_smooth=0.5`, `velocity_scale=1`, `velocity_min_mass=1`, `velocity_cutoff_perc=5`, `velocity_arrow_color="black"`, `velocity_arrow_angle=20`, plus streamline params `streamline_L=5, streamline_minL=1, streamline_res=1, streamline_n=15, streamline_width=c(0,0.8), streamline_alpha=1, streamline_color=NULL, streamline_palette="RdYlBu", streamline_palcolor=NULL, streamline_bg_color="white", streamline_bg_stroke=0.5`.

**Theme / faceting / combination**: `aspect.ratio=1`, `title=NULL`, `subtitle=NULL`, `xlab=NULL`, `ylab=NULL`, `legend.position="right"`, `legend.direction="vertical"`, `theme_use="theme_scp"`, `theme_args=list()`, `combine=TRUE`, `nrow=NULL`, `ncol=NULL`, `byrow=TRUE`, `force=FALSE`, `seed=11`.

### 1.2 Data extraction

Slots touched: `srt@meta.data`, `srt@reductions[[reduction]]@key`, `@cell.embeddings`, `srt@assays[[1]]` (only for `colnames()` = cell names and `ncol()` = n cells), `srt@graphs[[graph]]`, `srt@misc` (indirectly via `DefaultReduction` and the `paga` argument).

Preprocessing, in order:

1. If `split.by` is `NULL`, inject `srt@meta.data$All.groups <- factor("")`.
2. Every `group.by`/`split.by` column that isn't a factor is coerced with `levels = unique(x)` (**appearance order, not alphabetical** — preserve this; `pd.Categorical(x, categories=pd.unique(x))`).
3. `show_na=TRUE` rewrites `NA` to the literal string `"NA"` and appends `"NA"` as the last level, so it gets a palette color. `show_na=FALSE` leaves `NA` alone → those points get `bg_color` via `na.value`.
4. Validate lineages exist, `graph <- srt@graphs[[graph]]`, resolve reduction, intersect `cells.highlight` with cell names (error if none match, warn if partial).
5. `dat_meta <- srt@meta.data[, unique(c(group.by, split.by))]`; if any column has `>100` levels and `!force`, warn + `askYesNo` prompt (interactive gate — in Python make it a `force: bool` that raises).

Intermediate frame:

```r
reduction_key <- srt@reductions[[reduction]]@key          # e.g. "UMAP_"
dat_dim <- srt@reductions[[reduction]]@cell.embeddings
colnames(dat_dim) <- paste0(reduction_key, seq_len(ncol(dat_dim)))  # UMAP_1, UMAP_2, ...
rownames(dat_dim) <- rownames(dat_dim) %||% colnames(srt@assays[[1]])
dat_use <- cbind(dat_dim, dat_meta[row.names(dat_dim), , drop = FALSE])
if (!is.null(cells)) dat_use <- dat_use[intersect(rownames(dat_use), cells), ]
```

**Shape of `dat_use`:** `n_cells × (n_reduction_dims + n_group_cols + 1 split col)`, row names = cell barcodes, all embedding columns renamed `"<key><i>"`. Note the columns are renamed by *position*, so `dims=c(1,2)` selects `"<key>1"`, `"<key>2"`.

Auto point size and rasterization:

```r
if (is.null(pt.size)) pt.size <- min(3000 / nrow(dat_use), 0.5)
raster <- raster %||% (nrow(dat_use) > 1e5)
```

`raster.dpi` must be a length-2 numeric.

Per-panel frame (inside the `lapply`), for group `g` and split level `s`:

```r
dat <- dat_use
dat[[g]][dat[[split.by]] != s] <- NA     # cells of other splits become NA, NOT dropped
dat[["x"]] <- dat[[paste0(reduction_key, dims[1])]]
dat[["y"]] <- dat[[paste0(reduction_key, dims[2])]]
dat[["group.by"]] <- dat[[g]]
dat[, "split.by"] <- s                    # constant column, used as the facet strip
```

So **every panel contains all cells**; cells not in the current split are rendered as grey background. This is a defining visual behavior — reproduce it.

### 1.3 Rendering pipeline (exact layer order)

Layers are appended in this order; ggplot draws them bottom-to-top in the same order.

1. **`mark`** (if `add_mark`): `geom_mark_hull` / `_ellipse` / `_rect` / `_circle` from **ggforce**, data = non-NA rows only, `aes(x,y,color=group.by,fill=group.by)`, `expand=mark_expand`, `alpha=mark_alpha`, `linetype=mark_linetype`, `show.legend=FALSE`, then `scale_fill_manual`/`scale_color_manual` with `colors[names(labels_tb)]`, then `new_scale_fill()` + `new_scale_color()` (ggnewscale) so the marks own a private scale.
2. **`net`** (if `graph`): see §1.6.
3. **`density`** (if `add_density`): see §1.6.
4. `labs(title, subtitle=subtitle_use, x=xlab, y=ylab)`.
5. `scale_x_continuous(limits=range(dat_dim[,"<key><dims[1]>"]))`, same for y. **Limits come from `dat_dim`, i.e. all cells before the `cells` subset** — so all split panels share identical axes.
6. `do.call(theme_use, theme_args)` + `theme(aspect.ratio, legend.position, legend.direction)`.
7. `facet_grid(. ~ split.by)` if `split.by != "All.groups"`.
8. **Points**, mutually exclusive branches:
   - `raster`: two `scattermore::geom_scattermore` layers — first the NA rows in flat `bg_color`, then the non-NA rows colored by `group.by`; `pointsize=ceiling(pt.size)`, `pixels=raster.dpi`.
   - `hex`: `geom_hex(aes(x,y,fill=group.by,color=group.by, alpha=after_stat(count)))` (alpha dropped if `hex.count=FALSE`), `linewidth=hex.linewidth`, `bins=hex.bins`, `binwidth=hex.binwidth`.
   - else: a single `geom_point(aes(x,y,color=group.by), size=pt.size, alpha=pt.alpha)`.
9. **Highlight** (if `cells.highlight` and not `hex`): two stacked layers on the highlighted subset — an under-layer of size `sizes.highlight + stroke.highlight` in `cols.highlight` (the "stroke"), then an over-layer of size `sizes.highlight` colored by `group.by`. Raster variant uses `floor(sizes.highlight)+stroke.highlight` / `floor(sizes.highlight)`.
10. `scale_color_manual` + `scale_fill_manual` (see §1.4). `p_base <- p` is snapshotted **here** — the base legend is extracted from `p_base`, before any overlay steals the legend.
11. **stat pies** (`annotation_custom` grobs), then **lineages**, **paga**, **velocity** layers — each prefixed with `new_scale_color()` (velocity also `new_scale("size")`), each contributing a separately-rendered legend gtable into `legend_list`, and each followed by `theme(legend.position="none")`.
12. **Labels** (`geom_text_repel`), last, so text is on top.
13. If `legend_list` is non-empty: `legend_base <- get_legend(p_base + theme_scp(legend.position="bottom", ...))`; combine with `cbind` (vertical direction) or `rbind` (horizontal); render `p` with no legend to a gtable; `add_grob(gtable, legend, legend.position)`; wrap in `patchwork::wrap_plots` → the panel is now a **patchwork, not a ggplot**.

### 1.4 Color logic

```r
colors <- palette_scp(levels(dat_use[[g]]), palette=palette, palcolor=palcolor, NA_keep=TRUE)
```

Called on the **full level set** of the group column (computed from `dat_use`, i.e. across all splits), so a given group keeps the same color in every split panel. Because `levels(...)` never contains `NA`, `NA_keep=TRUE` is a no-op here.

`labels_tb <- table(dat[[g]]); labels_tb <- labels_tb[labels_tb != 0]` — only levels **present in this panel** enter the scale:

```r
scale_color_manual(name = paste0(g, ":"), values = colors[names(labels_tb)],
                   labels = label_use, na.value = bg_color,
                   guide = guide_legend(title.hjust=0, order=1,
                                        override.aes=list(size=4, alpha=1)))
```

A matching `scale_fill_manual` (no `override.aes`) is added for the hex/mark fills. So:

- Background / out-of-split / `NA` points → `na.value = bg_color` (`"grey80"`).
- `show_na=TRUE` promotes `NA` to a real `"NA"` level that gets a palette color instead.
- Legend keys are forced to `size=4, alpha=1` regardless of `pt.size`/`pt.alpha`.
- Legend title is `"<group.by>:"`.

Legend *labels* (`label_use`) encode the statistics:

- `label_insitu=TRUE` → `"Name(count)"` if `show_stat` else `"Name"`.
- `label=TRUE, label_insitu=FALSE` → `"1: Name(count)"` (index prefix matching the in-plot numbers).
- otherwise → `"Name(count)"` / `"Name"`.

### 1.5 Label placement

```r
label_df <- aggregate(p$data[, c("x","y")], by=list(p$data[["group.by"]]), FUN=median)
```

**Per-group median of x and median of y, computed independently** (a marginal median, not a geometric median / medoid / density peak). `aggregate` drops the `NA` group automatically. Rows with `NA` label are dropped explicitly. If `!label_insitu`, the label text is replaced with `seq_len(nrow(label_df))` — i.e. **1, 2, 3 … in level order**, and the legend carries `"1: Name"` so the plot stays readable with many groups.

Repel parameters:

```r
# label_repel = TRUE
geom_point(data=label_df, aes(x,y), color=label_point_color, size=label_point_size)  # center dot
geom_text_repel(..., fontface="bold", min.segment.length=0, segment.color=label_segment_color,
                point.size=label_point_size, max.overlaps=100, force=label_repulsion,
                color=label.fg, bg.color=label.bg, bg.r=label.bg.r, size=label.size,
                inherit.aes=FALSE)
# label_repel = FALSE: same geom but point.size = NA, force = 0  -> no repulsion, no segment
```

Note that even in the non-repel case `geom_text_repel` is still used (with `force=0`) purely to get the shadow-text rendering.

**The `label.bg.r` shadow-text trick:** ggrepel's `bg.color`/`bg.r` draw the string 8 extra times, offset radially by `bg.r × fontsize` in 8 compass directions, in `bg.color`, before drawing the real glyphs in `color`. The default `label.fg="white"`, `label.bg="black"`, `label.bg.r=0.1` yields white bold text with a thin black halo that remains legible over any point color. In matplotlib this is exactly `matplotlib.patheffects.withStroke(linewidth=..., foreground=label_bg)` applied to a white bold `Text`; `linewidth ≈ 2 × bg.r × fontsize_pt`.

### 1.6 Statistical / annotation overlays

**`show_stat`** — no geometry; it only (a) appends `"(n)"` counts to legend labels, and (b) sets `subtitle_use <- subtitle %||% paste0(s, " nCells:", sum(!is.na(dat[["group.by"]])))`. Default is off under `theme_blank`.

**`add_density`** — two modes:

```r
# density_filled = FALSE
geom_density_2d(aes(x,y), color=density_color, inherit.aes=FALSE, show.legend=FALSE)
# density_filled = TRUE
stat_density_2d(geom="raster", aes(x,y,fill=after_stat(density)), contour=FALSE,
                inherit.aes=FALSE, show.legend=FALSE)
scale_fill_gradientn(name="Density", colours=palette_scp(palette=density_filled_palette, ...))
new_scale_fill()
```

The filled variant is a raster heat-map of a 2D KDE over **all cells in the panel** (including the NA/other-split ones), drawn *under* the points, with the "Greys" ramp by default. Python: `scipy.stats.gaussian_kde` on a grid → `imshow`, or `sns.kdeplot`.

**`hex`** — replaces points entirely (requires the `hexbin` package). In `CellDimPlot`, hexes are grouped by the discrete `group.by` aesthetic, so bins are computed within each group and overlap; `hex.count=TRUE` maps bin `count` to alpha. Highlighting is disabled when `hex=TRUE`.

**`stat.by` pie charts** — delegates to `CellStatPlot(..., individual=TRUE, combine=FALSE)`, whose returned list is keyed `"<group_name>:<group_level>:<split_level>"`. Then:

```r
coor_df <- aggregate(p$data[, c("x","y")], by=list(p$data[["group.by"]]), FUN=median)
x_range <- diff(layer_scales(p)$x$range$range); y_range <- diff(...)
annotation_custom(as_grob(stat_plot[[grp]] + theme_void() + theme(legend.position="none")),
                  xmin = x - x_range*stat_plot_size/2, xmax = x + x_range*stat_plot_size/2,
                  ymin = y - y_range*stat_plot_size/2, ymax = y + y_range*stat_plot_size/2)
```

i.e. a mini plot (pie / ring / bar / rose per `stat_plot_type`) is embedded as an inset centered on each group's median position, sized as a fraction (`stat_plot_size`, default 0.15) of the axis range. Its legend is extracted once and appended to `legend_list` (the code uses the loop variable `i` after the loop, so it grabs the *last* group's legend — harmless because all pies share a palette). Python: `mpl_toolkits.axes_grid1.inset_locator.inset_axes` or `fig.add_axes` in data coordinates.

**`graph` edges** — the kNN/SNN adjacency is subset to the plotted cells, zeros → NA, upper triangle → NA (dedupe), melted to long form, and joined to the coordinate table:

```r
net_mat <- as_matrix(graph)[rownames(dat), rownames(dat)]
net_mat[net_mat == 0] <- NA; net_mat[upper.tri(net_mat)] <- NA
net_df <- reshape2::melt(net_mat, na.rm = TRUE)
# x/y from Var1, xend/yend from Var2
geom_segment(aes(x,y,xend,yend,linewidth=value), color=edge_color, alpha=edge_alpha, show.legend=FALSE)
scale_linewidth_continuous(range = edge_size)   # c(0.05, 0.5)
```

Edge weight drives line width, rescaled into `edge_size`. **This densifies the whole graph matrix — O(n²) memory.** In Python use the sparse COO triplets directly and a `LineCollection` with `linewidths` rescaled to `edge_size`.

**`lineages`** — delegates to `LineagePlot(..., return_layer=TRUE)` and keeps only `curve_layer`. Algorithm per lineage column `l` (a pseudotime vector with NAs for cells off the lineage):

```r
trim_pass <- dat[[l]] > quantile(dat[[l]], trim[1]) & dat[[l]] < quantile(dat[[l]], trim[2])  # 1%..99%
index <- which(trim_pass & !is.na(dat[[l]]));  index <- index[order(dat[index, l])]
fitted <- loess(<axis> ~ <lineage>, data=dat_sub, span=span, degree=2)$fitted   # one per axis
```

so each embedding axis is separately LOESS-smoothed (degree 2, `span=lineages_span`, default 0.75) against pseudotime, giving a principal curve in the plane, ordered by pseudotime. Rendering per lineage:

- optional **whiskers**: `geom_segment` from each fitted point to its raw cell coordinate, colored by lineage, `linewidth=lineages_whiskers_linewidth`, `alpha=lineages_whiskers_alpha`;
- a background `geom_path` in `lineages_line_bg` (white) with `linewidth = lineages_linewidth + lineages_line_bg_stroke` and the arrow head — the white outline;
- the colored `geom_path` with `linewidth = lineages_linewidth` and the same arrow.

Colors come from `palette_scp(lineages, palette="Dark2")`. The arrow default is `arrow(length=unit(0.1,"inches"))` at the terminal end.

**`paga`** — only allowed when `split.by == "All.groups"` (errors otherwise). Delegates to `PAGAPlot(..., return_layer=TRUE)` passing the *same* node palette as the cells, so node colors match group colors; `paga_edge_threshold` filters connectivities, edge widths rescale into `paga_edge_size`; `paga_show_transition=TRUE` adds directed transition arrows from the velocity-derived transition matrix. Its legend is only collected when `g != paga$groups` (otherwise it would duplicate the cell legend).

**`velocity`** — also requires `split.by == "All.groups"`. `VelocityPlot` reads `srt@reductions[[paste0(velocity,"_",reduction)]]@cell.embeddings` as the velocity field `V_emb` alongside `X_emb`:

- `plot_type="raw"`: one `geom_segment` arrow per cell (optionally subsampled to `density`×n when `0<density<1`), `V_emb` scaled by `velocity_scale`; **arrow head length is per-segment**, `unit(length/global_size, "npc")` where `global_size = sqrt(max(x)^2+max(y)^2)`. If `group_by` is supplied the arrows are colored by group with the same palette.
- `plot_type="grid"`: `compute_velocity_on_grid` (a port of scvelo's `velocity_embedding_grid`: build a regular grid, Gaussian-weight the `n_neighbors` nearest cells with `smooth`-scaled bandwidth, mask by `min_mass`), then one arrow per grid node.
- `plot_type="stream"`: `compute_velocity_on_grid(..., adjust_for_stream=TRUE, cutoff_perc=...)` then three stacked `metR::geom_streamline` layers — a fat background stroke in `streamline_bg_color`, the colored streamline (`size=after_stat(step)` mapped through `scale_size(range=streamline_width)`, `color=sqrt(dx²+dy²)` through `scale_color_gradientn(palette="RdYlBu")`), and a third linetype-0 layer that draws only the arrowheads in `velocity_arrow_color`.

**Highlight logic** — `cells.highlight=TRUE` is a sentinel meaning "highlight every cell that is *not* background in this panel": `cells.highlight_use <- rownames(dat)[!is.na(dat[[g]])]`. Combined with `split.by`, this outlines only the current split's cells. The two-layer stroke trick (§1.3 step 9) is the mechanism.

### 1.7 Faceting & combination

```r
comb <- expand.grid(split = levels(dat_use[[split.by]]), group = group.by)
rownames(comb) <- paste0(comb$split, ":", comb$group)
plist <- lapply(setNames(rownames(comb), rownames(comb)), function(i) { ... })
```

One panel per `(split level × group.by column)` pair, named `"<split>:<group>"`, **split varying fastest**. Within a panel, `facet_grid(. ~ split.by)` produces a single strip showing the split level (a cosmetic label, since the constant `split.by` column has one value).

Return type:

```r
if (combine) { if (length(plist)>1) wrap_plots(plotlist=plist, nrow=nrow, ncol=ncol, byrow=byrow) else plist[[1]] }
else plist
```

So: **`combine=TRUE` + 1 panel → a bare ggplot** (or a patchwork if extra legends were glued on); **`combine=TRUE` + >1 → a patchwork**; **`combine=FALSE` → a named list**. `nrow`/`ncol`/`byrow` only apply to the patchwork. `aspect.ratio` and `theme_use`/`theme_args` are applied per panel, not to the assembly.

`force=FALSE` triggers the interactive `askYesNo` guard when any grouping column has >100 levels; declining returns `invisible(NULL)`.

### 1.8 Edge cases and defaults worth preserving

- **Point draw order** is deliberately randomized so no group systematically overplots another:

```r
dat <- dat[order(dat[,"group.by"], decreasing=FALSE, na.last=FALSE), ]
naindex <- which(is.na(dat[,"group.by"]));  naindex <- ifelse(length(naindex)>0, max(naindex), 1)
dat <- dat[c(1:naindex, sample((min(naindex+1, nrow(dat))):nrow(dat))), ]
```

NA/background cells first (drawn at the bottom), then all colored cells in a **seeded random permutation** (`set.seed(seed)` with `seed=11` at function entry). Quirk: when there are *no* NA cells `naindex=1`, so the first row stays pinned at the front and only rows 2..n are shuffled.

- `pt.size = min(3000/n_cells, 0.5)` — so at ≥6000 cells the size saturates at 0.5 (ggplot mm units); at 1000 cells it's 3.0.
- `raster` auto-enables above **100,000** cells; rasterized point size is `ceiling(pt.size)` (so sub-1 sizes become 1 pixel), highlight sizes become `floor(sizes.highlight)`.
- Axis labels default to `paste0(reduction_key, dims[i])` → e.g. `"UMAP_1"`, derived from the Seurat reduction `@key`. In AnnData there is no key; synthesize one from the obsm name (`X_umap` → `"UMAP_"`).
- Axis limits are locked to the **global** embedding range (from `dat_dim`, pre-`cells` subsetting) so panels are directly comparable.
- `split.by=NULL` creates a factor whose only level is the empty string — the subtitle then reads `" nCells:1234"` with a leading space.
- The synthetic `"All.groups"` / `"split.by"` / `"group.by"` / `"x"` / `"y"` column names are injected into the user's data frame; a real metadata column named `x`, `y` or `split.by` would be shadowed.
- `guide_legend(order=1)` pins the group legend first when other legends exist.

---

## 2. `FeatureDimPlot` (L2042–2843)

### 2.1 Full parameter list (with defaults)

**Data selection**: `srt`, `features` (character vector **or named list**, names become subtitles), `reduction=NULL`, `dims=c(1,2)`, `split.by=NULL`, `cells=NULL`, `slot="data"` (Seurat-specific: `counts`/`data`/`scale.data`), `assay=NULL` (→ `DefaultAssay`), `graph=NULL`, `lineages=NULL`.

**Aesthetics / scaling**: `show_stat=ifelse(identical(theme_use,"theme_blank"),FALSE,TRUE)`, `palette=ifelse(isTRUE(compare_features),"Set1","Spectral")`, `palcolor=NULL`, `pt.size=NULL`, `pt.alpha=1`, `bg_cutoff=0`, `bg_color="grey80"`, `keep_scale="feature"` (or `"all"` or `NULL`), `lower_quantile=0`, `upper_quantile=0.99`, `lower_cutoff=NULL`, `upper_cutoff=NULL`, `raster=NULL`, `raster.dpi=c(512,512)`, `hex=FALSE`, `hex.linewidth=0.5`, `hex.color="grey90"`, `hex.bins=50`, `hex.binwidth=NULL`.

**Multi-feature**: `calculate_coexp=FALSE`, `compare_features=FALSE`, `color_blend_mode=c("blend","average","screen","multiply")`.

**Density**: `add_density=FALSE`, `density_color="grey80"`, `density_filled=FALSE`, `density_filled_palette="Greys"`, `density_filled_palcolor=NULL`.

**Highlight**: identical to `CellDimPlot` (`cells.highlight=NULL`, `cols.highlight="black"`, `sizes.highlight=1`, `alpha.highlight=1`, `stroke.highlight=0.5`).

**Labels**: identical names/defaults to `CellDimPlot` (`label=FALSE`, `label.size=4`, `label.fg="white"`, `label.bg="black"`, `label.bg.r=0.1`, `label_insitu=FALSE`, `label_repel=FALSE`, `label_repulsion=20`, `label_point_size=1`, `label_point_color="black"`, `label_segment_color="black"`).

**Lineages / edges / theme / combination**: same as `CellDimPlot`. No `add_mark`, no `paga`, no `velocity`, no `stat.by`, no `show_na`.

### 2.2 Data extraction

Slots touched: `srt@assays[[assay]]` via `methods::slot(assay_obj, slot)` (the expression matrix), `rownames(srt@assays[[assay]])` (gene names), `srt@meta.data`, `srt@reductions[[*]]@cell.embeddings` (both for the plotting reduction **and** to enumerate all embedding column names), `Seurat::FetchData` (only for embedding-valued features), `DefaultAssay`, `DefaultReduction`.

Feature resolution — three disjoint sources, checked in this order:

```r
embeddings <- lapply(srt@reductions, function(x) colnames(x@cell.embeddings))
embeddings <- setNames(rep(names(embeddings), sapply(embeddings, length)), nm = unlist(embeddings))
features_gene      <- features[features %in% rownames(srt@assays[[assay]])]
features_meta      <- features[features %in% colnames(srt@meta.data)]
features_embedding <- features[features %in% names(embeddings)]
```

A name present in both genes and metadata triggers a warning (the gene column wins in `cbind` order). Unknown names are dropped with a warning; zero valid features is an error. A **named list** input is flattened with `setNames(unlist(features), rep(names(features), lengths(features)))` and the names become per-panel subtitles.

`calculate_coexp=TRUE` computes a geometric-mean co-expression column across `features_gene`, dispatching on `check_DataType`:

```r
raw_counts / raw_normalized_counts : exp(mean(log(x)))
log_normalized_counts              : log1p(exp(mean(log(expm1(x)))))
```

stored as `meta.data$CoExp` and appended to the feature list.

Matrix assembly:

```r
dat_gene      <- t(as_matrix(slot(srt@assays[[assay]], slot)[features_gene, ]))   # dense!
dat_meta      <- as_matrix(srt@meta.data[, features_meta])
dat_embedding <- as_matrix(FetchData(srt, vars = features_embedding))
dat_exp <- as_matrix(cbind(dat_gene, dat_meta, dat_embedding))   # n_cells × n_features
dat_exp[, features][dat_exp[, features] <= bg_cutoff] <- NA      # background masking
```

**`bg_cutoff` (default 0) is applied globally and destructively before any plotting**: every value ≤ 0 becomes `NA` and will render in `bg_color`. For signed scores (e.g. `S_score`, PCs) you must pass `bg_cutoff = -Inf`. Note the comparison is `<=`, so exact zeros are background.

Coordinates:

```r
dat_dim <- srt@reductions[[reduction]]@cell.embeddings  # renamed "<key><i>"
dat_sp  <- srt@meta.data[, split.by, drop=FALSE]
dat_use <- cbind(dat_dim, dat_sp[rownames(dat_dim), , drop=FALSE])   # n_cells × (n_dims + 1)
dat_all <- cbind(dat_use, dat_exp[rownames(dat_use), features])
dat_split <- split.data.frame(dat_all, dat_all[[split.by]])
```

Unlike `CellDimPlot`, `FeatureDimPlot` **actually partitions the cells by split level** (`split.data.frame`) rather than NA-masking them — so a split panel contains only its own cells, and there are no grey "other split" points. Axis limits still come from the full `dat_use`, so panels stay aligned.

`pt.size`, `raster` auto-detection, and `raster.dpi` validation are identical to `CellDimPlot`. The `force` guard fires at **>50 features** (the roxygen text says 100; the code says 50).

Per-panel frame in the standard path: columns of `dat_use` plus the single feature column `f`, plus `x`, `y`, `value` (= `dat[[f]]`), `features` (= `f`, the facet strip variable).

### 2.3 Rendering pipeline — standard (one feature per panel)

1. Infinite values are pulled in to the finite extremes: `dat[,f][dat[,f]==max] <- max(finite)` and likewise for min.
2. Sort: `dat[order(dat[,"value"], method="radix", decreasing=FALSE, na.last=FALSE), ]` — **NA (background) first, then ascending value**, so the highest-expressing cells are drawn last, on top. This is the single most important visual convention to preserve.
3. `colors_value` computed (§2.4), values clipped into that range.
4. `net` (graph edges, identical code to `CellDimPlot`), then `density`, then `labs`, `scale_x/y_continuous(limits=range(dat_use[,...]))`, theme.
5. Points: `raster` → two `geom_scattermore` layers (NA rows then non-NA rows, both mapping `color=value`); `hex` → a `geom_hex` of the NA cells filled flat `bg_color` with `hex.color` borders, plus a `stat_summary_hex(aes(z=value))` (default summary = **mean**) for non-NA cells with `scale_fill_gradientn(colours=colors, values=rescale(colors_value), limits=range(colors_value), na.value=bg_color)` then `new_scale_fill()`; else a single `geom_point(aes(color=value), size=pt.size, alpha=pt.alpha)`.
6. Highlight (two-layer stroke, skipped when `hex`).
7. Facet: `facet_grid(. ~ features)` when unsplit, else `facet_grid(<split.by> ~ features)` — so the **feature name becomes the column strip and the split level the row strip**.
8. Color scale + `guides(color = guide_colorbar(frame.colour="black", ticks.colour="black", title.hjust=0, order=1))`. `p_base` snapshotted here.
9. `lineages` layers (with `new_scale_color()` prefix) + their legend.
10. Label annotation (§2.5).
11. If extra legends exist, the same gtable splice as `CellDimPlot` → patchwork.

### 2.4 Color logic — continuous scale, clipping, and blending

**Continuous ramp.** `colors <- palette_scp(type="continuous", palette=palette, palcolor=palcolor)` is computed **once outside the panel loop** → 100 colors interpolated from "Spectral" (blue→red, see §0.2). The mapping is:

```r
scale_color_gradientn(name="", colours=colors, values=rescale(colors_value),
                      limits=range(colors_value), na.value=bg_color, aesthetics=c("color"))
```

If the feature is all-NA in this panel, it degrades to `scale_colour_gradient(name="", na.value=bg_color)`.

**`colors_value`** is a 100-point vector defining both the limits and the (here linear) interpolation positions:

```r
# keep_scale = NULL  -> per-panel (per split × feature)
seq(lower_cutoff %||% quantile(dat[is.finite(dat$value),"value"], lower_quantile),
    upper_cutoff %||% quantile(dat[is.finite(dat$value),"value"], upper_quantile) + 0.001, length.out=100)
# keep_scale = "feature" (default) -> per feature, pooled across splits, using the global dat_exp column
# keep_scale = "all"               -> pooled across all features and splits
```

Then hard clipping into the scale (not dropping):

```r
dat[dat$value > max(colors_value), "value"] <- max(colors_value)
dat[dat$value < min(colors_value), "value"] <- min(colors_value)
```

So `lower_quantile`/`upper_quantile` (defaults 0 and **0.99**) and the absolute `lower_cutoff`/`upper_cutoff` (which take precedence, via `%||%`) *winsorize* the data onto the color scale — values beyond the limits are saturated, never greyed out. Two details: `+0.001` is added to the upper bound so the range can never be degenerate; and the `keep_scale="all"` branch forgets the `is.finite` filter on the upper quantile (`quantile(all_values, upper_quantile)`), an asymmetry you should fix rather than port.

Note that `bg_cutoff` (NA masking → `bg_color`) and the quantile clipping are independent mechanisms: the first removes cells from the color scale entirely, the second compresses the remaining ones.

**Multi-feature blending (`compare_features=TRUE` and >1 feature).** Taken when both hold; otherwise the standard path runs. Per split level:

1. `colors <- palette_scp(features, type="discrete", palette=palette, palcolor=palcolor)` — one base hue per feature (default "Set1"); `palcolor=c("red","green")` is the documented two-feature idiom.
2. For each feature a **private 2-stop ramp** is built from a 10%-alpha-over-white tint of the hue up to the full hue:

```r
colors_list[[i]] <- palette_scp(dat[, feature_i], type="continuous", NA_color=NA, NA_keep=TRUE,
                                matched=TRUE, palcolor=c(adjcolors(colors[i], 0.1), colors[i]))
```

`matched=TRUE` → a **per-cell color vector**; `NA_color=NA` → cells that are NA for that feature contribute *nothing* (literal `NA`, not a color).
`adjcolors(c, 0.1)` = composite `c` at alpha 0.1 over white, i.e. `0.1*c + 0.9*white`.

3. Per-cell blend across features:

```r
for (j in seq_len(nrow(dat)))
  dat[j, "color_blend"] <- blendcolors(sapply(colors_list, function(x) x[j]), mode=color_blend_mode)
```

`blendcolors` drops `NA`s; 0 colors → `NA`; 1 color → passthrough; otherwise `BlendRGBList`.

4. `BlendRGBList` reduces the list pairwise, repeatedly folding the last element into each earlier one with weights `a1*(1-1/N)` and `a2*(1/N)` (so with N inputs the last gets weight 1/N), until one remains, then flattens RGBA over white via `RGBA2RGB(RGB*A + white*(1-A))`. `Blend2Color(C1,C2,mode)` with `A = 1-(1-a1)(1-a2)`:
   - `blend` (default): `out = (c1*a1 + c2*a2*(1-a1))/A`, `A←1` — standard source-over alpha compositing.
   - `average`: `out = (c1+c2)/2`, clamped at 1.
   - `screen`: `out = 1-(1-c1)(1-c2)` — brighter, additive-ish.
   - `multiply`: `out = c1*c2` — darker, subtractive.
5. Draw-order key and background:

```r
dat["color_value"] <- colSums(col2rgb(dat[,"color_blend"]))         # R+G+B, 0..765
dat[rowSums(is.na(dat[, features])) == length(features), "color_value"] <- NA
dat <- dat[order(dat$color_value, decreasing=TRUE, na.last=FALSE), ]
dat[rowSums(is.na(dat[, features])) == length(features), "color_blend"] <- bg_color
```

Cells with no signal on *any* feature sort first (bottom) and are painted `bg_color`; the rest are sorted brightest-first so the **darkest / most saturated (most co-expressing) points end up on top**.

6. Points are drawn with `scale_color_identity()` (colors are literal hex strings in the data) followed by `new_scale_color()`, in `geom_point` or the two `geom_scattermore` layers (background subset first). `facet_grid(. ~ features)` / `facet_grid(split.by ~ features)` with the strip text `paste(features, collapse="|")`.
7. `cells.highlight=TRUE` here means `rownames(dat)[dat$color_blend != bg_color]`.

**The blend legend** is not a normal guide — since the plot uses identity colors, SCP builds **one colorbar per feature** by rendering a throwaway ggplot per feature:

```r
temp_geom[[i]] <- geom_point(aes(color = .data[[feature_i]]))  +
                  scale_color_gradientn(colours=pal_list[[i]], values=rescale(value_list[[i]]),
                                        na.value=bg_color,
                                        guide=guide_colorbar(frame.colour="black", ticks.colour="black"))
legend_list[[i]] <- get_legend(ggplot(dat, aes(x,y)) + temp_geom[[i]] + theme...)
```

(with `pal_list[[i]]` the unmatched version of the same two-stop ramp and `value_list[[i]]` 100 points spanning the feature's range). The legend gtables are then tiled: `legend_nrow <- min(ceiling(sqrt(n_features)), 3)` colorbars are `cbind`ed per row, rows are `rbind`ed, and a short final row is padded with `gtable_add_cols` to match the previous row's width. The lineage legend, if any, is glued on top. The composite is spliced onto the plot gtable with `add_grob(..., legend.position)` and wrapped in `wrap_plots` — so **the blend path always returns a patchwork**, never a plain ggplot.

### 2.5 Label placement

Labels mark the **high-expression centroid**, not the group centroid:

```r
label_df <- p$data %>%
  filter(value >= quantile(value[is.finite(value)], 0.95, na.rm=TRUE) &
         value <= quantile(value[is.finite(value)], 0.99, na.rm=TRUE)) %>%
  reframe(x = median(x), y = median(y))
```

i.e. take cells in the **95th–99th percentile band** of the feature (excluding the extreme top 1%, which are usually outliers) and place the label at their marginal median. In the standard path this is a single row labeled with the feature name, drawn via `annotate(geom = GeomTextRepel, ...)` (using `annotate` rather than a layer avoids inheriting the plot data). In the blend path the same computation is grouped by feature (`melt` to long form, `group_by(variable)`), producing one label per feature.

Repel/shadow parameters mirror `CellDimPlot`: `fontface="bold"`, `min.segment.length=0`, `max.overlaps=100`, `force = label_repulsion` when `label_repel` else `0`, `point.size = pt.size + 1` (repel) or `NA`, plus `color=label.fg` / `bg.color=label.bg` / `bg.r=label.bg.r` for the halo.

The blend path's `label_insitu` switch differs from `CellDimPlot`'s:

- `label_insitu=TRUE` → the **feature name** is written at the centroid, and `scale_color_manual(name="Label:", values=adjcolors(colors[label], 0.5))`.
- `label_insitu=FALSE` → a **rank number** is written, and a second legend (`legend2`) maps `"1: Feature"` with `guides(colour=guide_legend(override.aes=list(color=colors[label]), order=1))`; that legend is spliced in separately with its own `add_grob` call.

### 2.6 Statistical / annotation overlays

`show_stat` here produces a **positive-fraction subtitle** instead of a count:

```r
subtitle_use <- subtitle[f] %||% paste0(s, " nPos:", sum(dat$value > 0, na.rm=TRUE), ", ",
                                        round(sum(dat$value > 0, na.rm=TRUE)/nrow(dat)*100, 2), "%")
```

Note it is `%||%`-guarded by the per-feature subtitle, so a named `features` list (or an explicit `subtitle`) **suppresses** the statistic. `subtitle` may be length 1 (recycled) or length `n_features` (named by feature); any other length is an error.

`add_density`, `graph`/edges, `lineages`, and the highlight mechanism are byte-for-byte the same as in `CellDimPlot` (§1.6). There is no `add_mark`, `paga`, `velocity`, or `stat.by` in this function; `bg_cutoff`/`bg_color` replace `show_na`.

### 2.7 Faceting & combination

Standard path:

```r
comb <- expand.grid(split = levels(dat_sp[[split.by]]), feature = features)
rownames(comb) <- paste0(comb$split, ":", comb$feature)
```

→ one panel per `(split × feature)`, split varying fastest, named `"<split>:<feature>"`. Blend path → one panel per split, named `"<split>:<f1|f2|…>"`.

Return contract is identical to `CellDimPlot`: `combine=TRUE` collapses via `wrap_plots(plotlist, nrow, ncol, byrow)` when >1 panel, returns the single object otherwise; `combine=FALSE` returns the named list. A panel is a plain **ggplot** unless lineages/blend legends forced the gtable splice, in which case it is a **patchwork**. `keep_scale` is the cross-panel coupling knob: `"feature"` makes all splits of one feature comparable, `"all"` makes every panel comparable, `NULL` makes each panel independently scaled (and explicitly warns in the docs that colors are then not comparable).

### 2.8 Edge cases and defaults worth preserving

- **Point order**: ascending value with NAs first — high expression on top. (Contrast with `CellDimPlot`'s seeded shuffle.) The blend path orders by descending RGB sum with NAs first.
- `±Inf` values are snapped to the finite extremes before scaling, per panel.
- `dat_exp` is materialized **dense** (`as_matrix` on the sparse assay). Requesting all genes triggers `t(as_matrix(slot(assay, slot)))` over the whole matrix. In Python, slice the AnnData columns first and only densify the selected features.
- `features` are de-duplicated (`unique`) but the original input (with its names) is retained as `feature_input` for subtitles.
- `assay` and `slot` are pure Seurat concepts; the Python analogue is `layer=` (`None` → `adata.X`) plus `use_raw`.
- Axis labels derive from the reduction `@key` exactly as in `CellDimPlot`; `theme_blank` injects them into the corner arrow grob rather than the axis.
- `pt.size = min(3000/n, 0.5)`, `raster` above 1e5 cells, `pointsize = ceiling(pt.size)` — same as `CellDimPlot`.
- The colorbar guide is always `name=""` (no legend title) with black frame and black ticks.

---

## 3. `CellDimPlot3D` (L2843–3065) and `FeatureDimPlot3D` (L3066–3369) — plotly

Both return a `plotly` htmlwidget, not a ggplot, and share a much smaller feature set.

### 3.1 `CellDimPlot3D`

Signature: `(srt, group.by, reduction=NULL, dims=c(1,2,3), axis_labs=NULL, palette="Paired", palcolor=NULL, bg_color="grey80", pt.size=1.5, cells.highlight=NULL, cols.highlight="black", shape.highlight="circle-open", sizes.highlight=2, lineages=NULL, lineages_palette="Dark2", span=0.75, width=NULL, height=NULL, save=NULL, force=FALSE)`.

- `bg_color` and `cols.highlight` are normalized through `col2hex` (plotly needs hex).
- Reduction resolution uses `DefaultReduction(srt, min_dim = 3)`; a reduction with `<3` columns is an error.
- `dat_use` = renamed embeddings `cbind` full `meta.data`; the `>100 levels` `askYesNo` guard applies.
- **NA handling differs from 2D**: NAs are *always* converted to a literal `"NA"` level appended to the factor levels, and `palette_scp(..., NA_color = bg_color, NA_keep = TRUE)` gives that level the background color. There is no `show_na` switch.
- Axis columns are duplicated under the suffixed names `"<key><dim>All_cells"` — these are the actual trace inputs (a vestige of the split machinery used in `FeatureDimPlot3D`).
- One `add_trace(type="scatter3d", mode="markers", color=<group>, colors=colors, marker=list(size=pt.size))`, with a hover `text` of `"Cell:<barcode>\ngroup.by:<grp>\ncolor:<grp>"`.
- `cells.highlight=TRUE` → all cells whose group is not `"NA"`; rendered as a second trace named `"highlight"` with `marker=list(size=sizes.highlight, color=cols.highlight, symbol=shape.highlight)` (default `"circle-open"`, i.e. a ring).
- **Lineages**: per lineage, cells with non-NA pseudotime are sorted by it and each of the three axes is `loess(axis ~ lineage, span=span, degree=2)`-fitted; fitted points outside the observed range of any axis are dropped, then `unique(na.omit(...))`, and drawn as a `mode="lines"` trace of `width=6` colored by `palette_scp(lineages, palette=lineages_palette)[l]`. No arrow, no background stroke, no whiskers (unlike the 2D `LineagePlot`).
- `layout`: title `"Total (nCells:N)"`, legend at `x=1, y=0.5, itemsizing="constant"`, and `scene` with per-axis titles (`axis_labs` if length 3, else `"<key><dim>"`) and explicit `range` = observed min/max, `aspectratio = list(x=1,y=1,z=1)`, `autosize=FALSE`.
- `save="foo.html"` writes via `htmlwidgets::saveWidget` and then deletes the `foo_files` sidecar directory.

### 3.2 `FeatureDimPlot3D`

Signature: `(srt, features, reduction=NULL, dims=c(1,2,3), axis_labs=NULL, split.by=NULL, slot="data", assay=NULL, calculate_coexp=FALSE, pt.size=1.5, cells.highlight=NULL, cols.highlight="black", shape.highlight="circle-open", sizes.highlight=2, width=NULL, height=NULL, save=NULL, force=FALSE)`.

- Feature resolution, the `features_gene`/`features_meta`/`features_embedding` split, `calculate_coexp`, and the dense `dat_exp` assembly are **identical to `FeatureDimPlot`** (§2.2) — except there is **no `bg_cutoff`**, so nothing is masked to NA, and no quantile clipping at all. The `>50 features` guard applies.
- `split.by=NULL` creates a single-level `"All.cells"` factor. Per split level, three extra columns `"<key><dim><level>"` hold the coordinate or `NA` (`ifelse(split == level, coord, NA)`), plus the `"All_cells"` triple. Toggling splits is thus a pure client-side data swap.
- A single scatter3d trace is created for `features[1]`, colored by the raw feature values with `marker$colorbar` titled with the feature name (`len = 0.5`) and `showscale=TRUE`. The color scale is plotly's default — **`palette_scp` is not used here at all**, so 3D feature plots do not match 2D Spectral coloring.
- Optional highlight trace as in `CellDimPlot3D` (`cells.highlight=TRUE` → all cells).
- Two `updatemenus` dropdowns are built:
  - `split_option` (at `y=0.67`): one `method="update"` button per split level (plus `"All.cells"`), swapping `x`/`y`/`z` (and the highlight trace's coordinates) and rewriting the title to `"<level> (nCells:n)"`.
  - `genes_option` (at `y=0.33`): one button per feature, swapping `marker` (color vector + colorbar title) and the hover `text` to `"Cell:<barcode>\nExp:<rounded value>"`.
- Layout parallels `CellDimPlot3D` but with the legend below the plot (`y=-0.2, x=0.5, xanchor="center"`), fixed axis ranges, unit aspect ratio, `autosize=FALSE`, and the same `save` handling.

**Python note:** both 3D functions map cleanly onto `plotly.graph_objects.Scatter3d` with `updatemenus`; the only non-trivial port is the per-axis LOESS lineage smoothing (use `statsmodels.nonparametric.lowess` with `frac=span`, or a local quadratic fit to match `degree=2`).

---

## 4. Consolidated porting checklist

1. **Export `palette_list`** to a Python dict of `{name: (colors, kind)}`; note SCP's `Spectral` is reversed vs. Brewer.
2. Implement `palette_scp` semantics: appearance-ordered categorical levels, exact-slice-vs-interpolate rule (`n_levels <= len(palette)` → slice), `NA_color` handling, `matched` per-element mode.
3. Two point-ordering policies: **seeded shuffle after NA** (categorical) vs. **ascending value with NA first** (continuous) vs. **descending RGB sum with NA first** (blend).
4. `pt.size = min(3000/n, 0.5)`, raster above 1e5 points (`datashader`/`matplotlib` rasterized `scatter`), `raster.dpi` → pixel grid.
5. Label placement = per-group marginal median (categorical) or marginal median of the 95–99th percentile band (continuous); numbered labels with a `"i: Name"` legend when `label_insitu=False`; halo via `path_effects.withStroke`.
6. Background semantics: `bg_color` for NA/other-split/`<= bg_cutoff` points; `show_na` promotes NA to a real colored level (categorical only).
7. Winsorizing color limits from `lower/upper_quantile` (0, 0.99) or absolute `lower/upper_cutoff`, with `keep_scale ∈ {None, "feature", "all"}` controlling the pooling scope.
8. Blend mode math from `Blend2Color` (`blend`/`average`/`screen`/`multiply`) with the `1/N`-weighted pairwise fold of `BlendRGBList`, per-feature ramps from `adjcolors(hue, 0.1)` → `hue`, and an N-colorbar legend grid (`min(ceil(sqrt(N)), 3)` per row).
9. Shared axis limits from the *global* embedding range; axis labels `"<KEY><dim>"`; `theme_blank` = corner arrows at 15% of the panel with labels, no frame.
10. Panel naming `"<split>:<group|feature>"` (split fastest), `combine` → grid vs. list, `nrow`/`ncol`/`byrow` for the grid.
