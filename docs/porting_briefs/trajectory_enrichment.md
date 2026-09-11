# Porting brief: trajectory, graph and enrichment plots

**Source**

- **Upstream:** `SCP/R/SCP-plot.R` — `LineagePlot` L5918, `PAGAPlot` L6121, `GraphPlot` L6287 (+ `segementsDf` L6666), `VelocityPlot` L6745 (+ `compute_velocity_on_grid` L6963), `DynamicPlot` L12102, `GroupTreePlot` L12472 (stub), `ProjectionPlot` L12511, `EnrichmentPlot` L12727 (+ `adjustlayout` L13441), `GSEAPlot` L13563 (+ `gsInfo` L14545, `gseaScores` L14570)
- **Target:** `src/scp/pl/_traj.py` and `src/scp/pl/_enrich.py`
- **Milestones:** 11 (`graph_plot` / `paga_plot`), 12 (`lineage_plot` / `velocity_plot`), 13 (`projection_plot`), 14 (`enrichment_plot`), 15 (`gsea_plot`)

Supporting upstream code: `SCP/R/SCP-analysis.R` (`RunSlingshot` L4307, `RunDynamicFeatures` L4948–5170, `RunEnrichment` L3766–3975, `RunGSEA` L4030–4260, `adata_to_srt` L5521–5681), `SCP/inst/python/SCP_analysis.py` (the scanpy/scvelo wrappers), `SCP/R/utils.R` (`capitalize`, `str_wrap`).

Two helpers are **already implemented in the scaffold** and are described here only so the surrounding logic makes sense: `segementsDf` → `scp.pl._traj.shorten_segments`, and `gseaScores` → `scp.pl._enrich.gsea_scores`.

---

## 0. Shared skeleton

All plots in this family share a common skeleton worth porting once:

- `palette_scp(x, n=100, palette="Paired", palcolor=NULL, type=c("auto","discrete","continuous"), matched, reverse, NA_keep, NA_color)` (`SCP-plot.R:168`) — palette resolver keyed off `SCP::palette_list`. Discrete → named vector keyed by level; continuous → `colorRampPalette` of `n`. See `docs/03_theme_and_palette.md`.
- `theme_use` / `theme_args` — a string naming a theme function (`"theme_scp"`, `"theme_blank"`) called via `do.call`. When `theme_use == "theme_blank"`, `xlab`/`ylab` are pushed **into** `theme_args` (the blank theme draws its own mini-axis arrows).
- `return_layer = TRUE` — returns a named list of ggplot layer lists instead of a plot, so `CellDimPlot` can compose them (this is how PAGA/lineage/velocity overlays get onto a UMAP scatter). The Python port wants the equivalent: functions that take/return a matplotlib `Axes` rather than always creating a figure.
- `combine` / `nrow` / `ncol` / `byrow` — `patchwork::wrap_plots`. Port → `matplotlib` gridspec or a small facet helper.
- `get_legend()` / `add_grob()` / `as_grob()` (`SCP-plot.R:14710–14790`) — legend extraction and re-attachment onto a gtable edge. Used by `DynamicPlot`, `ProjectionPlot`, `GSEAPlot`. Port → shared `fig.legend` with handles collected from sub-axes.

---

## 1. `LineagePlot` (line 5918)

### Parameters

| Group | Params (defaults) |
|---|---|
| Data | `srt`, `lineages` (required), `reduction=NULL`, `dims=c(1,2)`, `cells=NULL` |
| Fit | `trim=c(0.01,0.99)`, `span=0.75` |
| Line style | `palette="Dark2"`, `palcolor=NULL`, `lineages_arrow=arrow(length=unit(0.1,"inches"))`, `linewidth=1`, `line_bg="white"`, `line_bg_stroke=0.5` |
| Whiskers | `whiskers=FALSE`, `whiskers_linewidth=0.5`, `whiskers_alpha=0.5` |
| Cosmetic | `aspect.ratio=1`, `title`, `subtitle`, `xlab`, `ylab`, `legend.position="right"`, `legend.direction="vertical"`, `theme_use="theme_scp"`, `theme_args=list()`, `return_layer=FALSE`, `seed=11` |

### Input data contract

- `srt@reductions[[reduction]]@cell.embeddings` → coordinate matrix; columns renamed `<key><i>` from `@key`. For the AnnData equivalent of the reduction accessors and the `key_` convention, see `docs/02_data_contract.md`.
- `srt@meta.data[, lineages]` — **one numeric pseudotime column per lineage**. Naming convention comes from `RunSlingshot` (`SCP-analysis.R:4307`): `slingshot::slingPseudotime(sl)` columns are `Lineage1`, `Lineage2`, … optionally prefixed (`prefix_Lineage1`). `RunSlingshot` also writes `BranchID` and stashes the raw slingshot object in `srt@tools[["Slingshot_<group.by>_<reduction>"]]`.
- Cells not on a lineage have `NA` pseudotime — this is load-bearing (see trimming).

### Rendering pipeline

Per lineage `l`:

1. Trim: keep cells with `quantile(t, 0.01) < t < quantile(t, 0.99)` and `!is.na(t)`; order by pseudotime.
2. Fit **two independent LOESS curves**, one per axis, against pseudotime:
   `loess(Axis_k ~ l, weights = 1, span = span, degree = 2)$fitted`. (A commented-out `weights` argument hints at planned weighted fits.)
3. Layers, in draw order:
   - (optional) `geom_segment` whiskers connecting each fitted point back to its raw cell coordinate, colored by lineage.
   - `geom_path` **background/halo** in `line_bg` at `linewidth + line_bg_stroke`, with arrow.
   - `geom_path` foreground colored by `Lineages`, with arrow.
4. `scale_color_manual(values = palette_scp(lineages, "Dark2"))`.

Returned layer list: `curve_layer`, `lab_layer`, `theme_layer`.

### Python port

- Slingshot is R-only. Replacements: **scFates** (`scf.tl.curve`/`circle`, principal-graph), **Palantir**, **CellRank**, or slingshot-in-python (`pyslingshot`). Whatever produces it, the plot only needs a per-cell pseudotime column per lineage → `adata.obs["Lineage1"]`, etc.
- LOESS: `statsmodels.nonparametric.smoothers_lowess.lowess(endog=axis, exog=pseudotime, frac=span, it=0)` — note statsmodels lowess is degree-1; for degree-2 parity use `skmisc.loess.loess(..., degree=2, span=span)` (the same C/Fortran `loess` the R function wraps) — this is the closest match.
- Arrowed paths: `matplotlib.patches.FancyArrowPatch` or `ax.annotate` per segment; the halo effect is `path_effects.withStroke`.

---

## 2. `PAGAPlot` (line 6121)

`PAGAPlot` is a thin adapter: it reshapes PAGA output into a node data.frame + edge matrix and delegates **everything** to `GraphPlot`.

### Parameters

`srt`, `paga = srt@misc$paga`, `type="connectivities"`, `reduction=NULL`, `dims=c(1,2)`, `cells=NULL`, `show_transition=FALSE`, then the entire `GraphPlot` node/label/edge/transition parameter set forwarded verbatim (`node_palette="Paired"`, `node_palcolor=NULL`, `node_size=4`, `node_alpha=1`, `node_highlight=NULL`, `node_highlight_color="red"`, `label=FALSE`, `label.size=3.5`, `label.fg="white"`, `label.bg="black"`, `label.bg.r=0.1`, `label_insitu=FALSE`, `label_repel=FALSE`, `label_repulsion=20`, `label_point_size=1`, `label_point_color="black"`, `label_segment_color="black"`, `edge_threshold=0.01`, `edge_line=c("straight","curved")`, `edge_line_curvature=0.3`, `edge_line_angle=90`, `edge_size=c(0.2,1)`, `edge_color="grey40"`, `edge_alpha=0.5`, `edge_shorten=0`, `edge_offset=0`, `edge_highlight=NULL`, `edge_highlight_color="red"`, `transition_threshold=0.01`, `transition_line=c("straight","curved")`, `transition_line_curvature=0.3`, `transition_line_angle=90`, `transition_size=c(0.2,1)`, `transition_color="black"`, `transition_alpha=1`, `transition_arrow_type="closed"`, `transition_arrow_angle=20`, `transition_arrow_length=unit(0.02,"npc")`, `transition_shorten=0.05`, `transition_offset=0`, `transition_highlight=NULL`, `transition_highlight_color="red"`), plus `aspect.ratio=1`, `title="PAGA"`, and the standard cosmetic block.

### The PAGA object — exact keys

`srt@misc$paga` is a **verbatim R translation of `adata.uns['paga']`** — `adata_to_srt` copies every `adata.uns` key into `srt@misc[[key]]` with no renaming (see `docs/02_data_contract.md`). Keys read:

```r
connectivities <- paga[[type]]          # "connectivities" or "connectivities_tree"
transition     <- paga[["transitions_confidence"]]
groups         <- paga[["groups"]]       # a STRING: the obs column name
```

So the Python port reads natively (this is what `scp.io.read_paga` exists for):

| Key | Producer | Shape / type | Used for |
|---|---|---|---|
| `adata.uns['paga']['connectivities']` | `sc.tl.paga` | sparse `(n_groups, n_groups)`, symmetric | default edge weights |
| `adata.uns['paga']['connectivities_tree']` | `sc.tl.paga` | sparse `(n_groups, n_groups)`, spanning-tree subset | `type="connectivities_tree"` |
| `adata.uns['paga']['groups']` | `sc.tl.paga` | str — name of the `obs` categorical | maps rows/cols → categories |
| `adata.uns['paga']['pos']` | `sc.pl.paga` | `(n_groups, 2)` float | **ignored by SCP** |
| `adata.uns['paga']['transitions_confidence']` | `scv.tl.paga` (scvelo) | sparse `(n_groups, n_groups)`, **asymmetric/directed** | arrows when `show_transition=TRUE` |

**Critical porting note:** SCP does *not* use `paga['pos']`. It recomputes node coordinates as the **per-group median of the chosen embedding**:

```r
dat <- aggregate(dat_dim[, paste0(reduction_key, dims)], by = list(dat_dim[[groups]]), FUN = median)
dat[["GroupSize"]] <- as.numeric(table(dat_dim[[groups]])[rownames(dat)])
```

This is why `PAGAPlot(srt, reduction="UMAP")` and `reduction="PCA"` both work — the graph is projected onto whatever embedding you name. `GroupSize` (cells per group) becomes an addressable node-size column (`node_size="GroupSize"`).

**Row/column ordering:** `colnames(connectivities) <- rownames(connectivities) <- rownames(dat)`, and `rownames(dat)` is the `aggregate` output ordered by factor level. The port must therefore index the connectivity matrix by `adata.obs[groups].cat.categories` order — scanpy's own convention, so this matches directly.

Guard: `nlevels(meta[[groups]]) != nrow(connectivities)` → error. Subsetting by `cells` also subsets the connectivity matrix to the surviving groups.

Triangular handling:

```r
if (type == "connectivities_tree") { use_triangular <- "both"; edge_threshold <- 0 }
else                               { use_triangular <- "upper" }
```

`connectivities_tree` is not symmetric-complete, so both triangles are read and the threshold is disabled.

### Python replacement

Essentially free: `adata.uns['paga']` is already native. `sc.pl.paga` exists but draws its own layout; the SCP behaviour (graph over an embedding, median node positions) is `sc.pl.paga_compare`-ish but with far more styling control. Port as a standalone function over `networkx` + matplotlib, reading `uns['paga']` directly.

---

## 3. `GraphPlot` (line 6287) and `segementsDf` (line 6666)

This is the generic engine. Anything that can be expressed as (node table, edge matrix, optional directed transition matrix) routes through it.

### Parameters (grouped)

- **Data:** `node` (data.frame), `edge` (square matrix), `transition=NULL` (square matrix), `node_coord=c("x","y")`, `node_group=NULL`.
- **Nodes:** `node_palette="Paired"`, `node_palcolor=NULL`, `node_size=4`, `node_alpha=1`, `node_highlight=NULL`, `node_highlight_color="red"`.
- **Labels:** `label=FALSE`, `label.size=3.5`, `label.fg="white"`, `label.bg="black"`, `label.bg.r=0.1`, `label_insitu=FALSE`, `label_repel=FALSE`, `label_repulsion=20`, `label_point_size=1`, `label_point_color="black"`, `label_segment_color="black"`.
- **Edges:** `edge_threshold=0.01`, `use_triangular=c("upper","lower","both")`, `edge_line=c("straight","curved")`, `edge_line_curvature=0.3`, `edge_line_angle=90`, `edge_color="grey40"`, `edge_size=c(0.2,1)`, `edge_alpha=0.5`, `edge_shorten=0`, `edge_offset=0`, `edge_highlight=NULL`, `edge_highlight_color="red"`.
- **Transitions:** `transition_threshold=0.01`, `transition_line=c("straight","curved")`, `transition_line_curvature=0.3`, `transition_line_angle=90`, `transition_color="black"`, `transition_size=c(0.2,1)`, `transition_alpha=1`, `transition_arrow_type="closed"`, `transition_arrow_angle=20`, `transition_arrow_length=unit(0.02,"npc")`, `transition_shorten=0.05`, `transition_offset=0`, `transition_highlight=NULL`, `transition_highlight_color="red"`.
- **Cosmetic:** `aspect.ratio=1`, `title`, `subtitle`, `xlab`, `ylab`, `legend.position="right"`, `legend.direction="vertical"`, `theme_use="theme_scp"`, `theme_args=list()`, `return_layer=FALSE`.

### Input data contract / validation

`node` must be a data.frame, `edge` a square matrix, `nrow(edge) == nrow(node)`, rownames matched (else a warning and positional correspondence). `node_size` and `node_alpha` are **polymorphic**: if the value names a column of `node`, that column drives a continuous/discrete scale; otherwise it's an identity constant.

```r
if (!node_size %in% colnames(node)) { node[["node_size"]] <- node_size; scale_size <- scale_size_identity() }
else { node[["node_size"]] <- node[[node_size]]; scale_size <- scale_size_continuous(name = node_size) }
```

`global_size <- sqrt(max(x)^2 + max(y)^2)` — a scalar used to convert the fractional `edge_shorten` / `edge_offset` / `transition_shorten` into data units.

### Edge construction

```r
edge[edge <= edge_threshold] <- NA
upper/lower triangle nulled per use_triangular
edge_df <- melt(edge, na.rm = TRUE)      # from, to, size
# joined against node[x,y] → x,y,xend,yend; rowname = "from-to"
edge_df <- segementsDf(edge_df, global_size*edge_shorten, global_size*edge_shorten, global_size*edge_offset)
```

`edge_name` is `"<from>-<to>"` — that string is what `edge_highlight` matches (see the roxygen example `paste("Pre-endocrine", levels(...), sep="-")`).

Edge linetype: **solid when no transition matrix, dashed (`linetype=2`) when transitions are being drawn** — so connectivity vs. directed flow read differently.

### Transition construction (net directed flow)

```r
trans2 <- trans1 <- as_matrix(transition)
trans1[lower.tri(trans1)] <- 0
trans2[upper.tri(trans2)] <- 0
trans <- t(trans1) - trans2                 # trans[i,j] = T[j,i] - T[i,j] for i>j
trans[abs(trans) <= transition_threshold] <- NA
# then any negative value flips (from,to) and takes the absolute value
```

i.e. only the **net** flow between each pair is drawn, as a single arrow in the dominant direction. Port this exactly — a naive "draw all directed edges" gives a visually different plot.

### Layer order

`edge_layer` → `trans_layer` → `node_layer` → `label_layer` → `lab_layer` → `theme_layer`.

- `node_layer` draws each node **twice**: a black point at `node_size*1.2` (outline) then the colored point at `node_size`. Highlights add a third/fourth pair at `*1.3` in `node_highlight_color`.
- Legend labels: when `label=TRUE && !label_insitu`, legend entries become `"1: Ductal"`, `"2: Ngn3 low EP"`, … and the on-plot label is the **integer index** rather than the name — a compact legend idiom worth reproducing.
- `label_repel=TRUE` adds a small anchor point plus `geom_text_repel` with `force=label_repulsion`; `label_repel=FALSE` uses `geom_text_repel(force=0, point.size=NA)` purely for the shadow-text background effect (`bg.color`, `bg.r`).
- Each of `edge` / `transition` ends with `scale_linewidth_continuous(range=range(edge_size), guide="none")` + `ggnewscale::new_scale("linewidth")` so the three linewidth scales don't collide.

### `segementsDf(data, shorten_start, shorten_end, offset)` (line 6666) — already implemented as `shorten_segments`

Pure geometry:

```r
dx, dy = xend-x, yend-y;  dist = hypot(dx,dy);  px, py = dx/dist, dy/dist
x     += px*shorten_start ;  y    += py*shorten_start
xend  -= px*shorten_end   ;  yend -= py*shorten_end
x,xend -= py*offset       ;  y,yend += px*offset      # perpendicular displacement
```

Shortening pulls segment ends back so arrowheads don't sit under node circles; `offset` shifts the whole segment perpendicular to itself, which is how the forward and reverse transitions between the same pair avoid overlapping (`transition_offset=0.02` in the examples).

### Python port

`networkx` + `matplotlib` (`LineCollection` for edges, `FancyArrowPatch` for transitions, `ax.scatter` for nodes, `adjustText` for repelled labels). Curved edges → `FancyArrowPatch(connectionstyle="arc3,rad=curvature")`. Note R's `geom_curve(curvature, angle)` semantics differ from `arc3,rad` — `rad ≈ curvature/2` is a reasonable visual match.

---

## 4. `VelocityPlot` (line 6745) + `compute_velocity_on_grid` (line 6963)

### Parameters

| Group | Params |
|---|---|
| Data | `srt`, `reduction` (required), `dims=c(1,2)`, `cells=NULL`, `velocity="stochastic"`, `plot_type=c("raw","grid","stream")` |
| Grouping | `group_by=NULL`, `group_palette="Paired"`, `group_palcolor=NULL` |
| Grid/field | `n_neighbors=ceiling(ncol(srt@assays[[1]])/50)`, `density=1`, `smooth=0.5`, `scale=1`, `min_mass=1`, `cutoff_perc=5` |
| Arrows | `arrow_angle=20`, `arrow_color="black"` |
| Streamlines | `streamline_L=5`, `streamline_minL=1`, `streamline_res=1`, `streamline_n=15`, `streamline_width=c(0,0.8)`, `streamline_alpha=1`, `streamline_color=NULL`, `streamline_palette="RdYlBu"`, `streamline_palcolor=NULL`, `streamline_bg_color="white"`, `streamline_bg_stroke=0.5` |
| Cosmetic | standard block, `title="Cell velocity"`, `seed=11` |

### Input data contract

```r
V_reduction <- paste0(velocity, "_", reduction)     # e.g. "stochastic_UMAP"
X_emb <- srt@reductions[[reduction]]@cell.embeddings[, dims]
V_emb <- srt@reductions[[V_reduction]]@cell.embeddings[, dims]
```

This is **`adata.obsm[vkey + "_" + basis]`** as written by `scv.tl.velocity_embedding(adata, basis=basis, vkey=vkey)` (`SCP_analysis.py:187`). For how `obsm` keys become Seurat reductions, see `docs/02_data_contract.md`; the upshot is that `obsm["stochastic_umap"]` surfaces as reduction `stochastic_UMAP`. **The Python port reads `adata.obsm` directly; nothing needs reconstructing.** `velocity` values track scvelo's `vkey` (`"stochastic"`, `"deterministic"`, `"dynamical"`, plus SCP's `"dynamical_kinetics"`, `"dynamical_denoise"`).

`check_R("metR")` is invoked unconditionally at function entry even for `plot_type="raw"`.

### `plot_type` behaviours

- **`raw`** — per-cell arrows. `density < 1` randomly subsamples `ceiling(density*n)` cells. `V_emb *= scale`. Arrowhead length is proportional to vector magnitude: `arrow(length = unit(length/global_size, "npc"))` where `global_size = sqrt(max(x)^2 + max(y)^2)`. If `group_by` is given, arrows are colored by that `obs` column with a discrete legend; otherwise a flat `arrow_color`.
- **`grid`** — `compute_velocity_on_grid(..., scale=scale)`, then the same segment+arrow drawing over grid points.
- **`stream`** — `compute_velocity_on_grid(..., scale=1, cutoff_perc=cutoff_perc, adjust_for_stream=TRUE)`, then `metR::geom_streamline`. Three stacked streamline layers: a fat background stroke in `streamline_bg_color`, the colored line, and a `linetype=0` layer that contributes **only arrowheads** in `arrow_color`. When `streamline_color` is `NULL`, the middle layer maps `size = after_stat(step)` (tapering along the streamline) and `color = sqrt(after_stat(dx)^2 + after_stat(dy)^2)` (speed) through `scale_color_gradientn(palette="RdYlBu")`.

### `compute_velocity_on_grid` — direct port of scvelo

Documented in the roxygen as a port of `scvelo/plotting/velocity_embedding_grid.py`. Signature: `(X_emb, V_emb, density=NULL, smooth=NULL, n_neighbors=NULL, min_mass=NULL, scale=1, adjust_for_stream=FALSE, cutoff_perc=NULL)`, defaults filled in as `density=1, smooth=0.5, n_neighbors=ceiling(n_obs/50), min_mass=1, cutoff_perc=5`.

Algorithm, step by step:

1. **Grid construction.** Per dimension, `seq(min, max, length.out = ceiling(50*density))`; `X_grid = expand.grid(...)` → `(50*density)^n_dim × n_dim`. (scvelo pads the range by ±1%; SCP has that padding **commented out** — a deliberate divergence to note.)
2. **KNN.** Full pairwise Euclidean distance `proxyC::dist(X_emb, X_grid)`; for each grid point take the `n_neighbors` nearest cells (`neighbors`, `dists`, each `n_grid × n_neighbors`).
3. **Gaussian weights.**
   ```r
   weight <- dnorm(dists, sd = mean(sapply(grs, function(g) g[2]-g[1])) * smooth)
   ```
   σ = (mean grid spacing) × `smooth`. Matches scvelo's `normal.pdf(dists, scale=mean(grid_spacing)*smooth)`.
4. **Mass.** `p_mass <- rowSums(weight)`; a clipped copy `p_mass_V` with values `< 1` raised to 1 is used as the divisor.
5. **Weighted mean velocity.** `V_grid = Σ_k (V_emb[neighbor_k] * weight_k) / p_mass_V`, computed via a 3-D array reduce `apply(neighbors_emb * c(weight), c(1,3), sum)`.
6. **Masking — two modes:**
   - *grid mode* (`adjust_for_stream=FALSE`): `min_mass <- min_mass * quantile(p_mass, 0.99) / 100`; keep grid points with `p_mass > min_mass`; then `V_grid *= scale`.
   - *stream mode* (`adjust_for_stream=TRUE`): reshape to `(2, ns, ns)` with `ns = floor(sqrt(n_grid))`; `X_grid` collapses to a `2 × ns` matrix of unique axis values.
     ```r
     mass   <- sqrt(apply(V_grid**2, c(2,3), sum))
     min_mass <- 10**(min_mass - 6)                 # default 1 → 1e-5
     min_mass[min_mass > max(mass)*0.9] <- max(mass)*0.9
     cutoff <- mass < min_mass
     length <- t(apply(apply(abs(neighbors_emb), c(1,3), mean), 1, sum))
     cutoff <- cutoff | length < quantile(length, cutoff_perc/100)
     V_grid[1,,][cutoff] <- NA
     ```
     Two masks: low vector magnitude, and low mean neighbor-velocity magnitude (bottom `cutoff_perc` percent). Note the `10**(min_mass-6)` reparameterisation — `min_mass=1` means scvelo's `1e-5`.

Returns `list(X_grid, V_grid)`. Stream mode returns the transposed `2×ns` / `(2,ns,ns)` layout that `metR::geom_streamline` expects.

### Python port

**Do not reimplement.** `scvelo.plotting.velocity_embedding_grid.compute_velocity_on_grid` is the original; call it, or copy it from scvelo. For rendering: `ax.quiver` (raw/grid), `ax.streamplot` (stream — `density`, `linewidth`, `color` arrays map cleanly onto the `metR` parameters; `streamline_L`/`minL`/`res`/`n` have no direct streamplot analogue and would need `start_points` + `maxlength` tuning). `scv.pl.velocity_embedding_stream` already does most of this if exact styling parity isn't required.

Two small bugs to not replicate: `ggplot() + velocity_layer + lab_layer + lab_layer + theme_layer` adds `lab_layer` twice; and in `GraphPlot`'s curved-transition branch the highlight guard tests `!is.null(edge_highlight)` instead of `transition_highlight`.

---

## 5. `DynamicPlot` (line 12102)

### Parameters

| Group | Params |
|---|---|
| Data | `srt`, `lineages`, `features`, `group.by=NULL`, `cells=NULL`, `slot="counts"`, `assay=NULL`, `family=NULL` |
| Normalisation | `exp_method=c("log1p","raw","zscore","fc","log2fc")` (or a function), `lib_normalize=identical(slot,"counts")`, `libsize=NULL` |
| Faceting | `compare_lineages=TRUE`, `compare_features=FALSE` |
| Line | `add_line=TRUE`, `add_interval=TRUE`, `line.size=1`, `line_palette="Dark2"`, `line_palcolor=NULL` |
| Points | `add_point=TRUE`, `pt.size=1`, `point_palette="Paired"`, `point_palcolor=NULL` |
| Axes | `add_rug=TRUE`, `flip=FALSE`, `reverse=FALSE`, `x_order=c("value","rank")` |
| Cosmetic | `aspect.ratio=NULL`, standard block + `combine/nrow/ncol/byrow`, `seed=11` |

### Input data contract

Reads `srt@tools[["DynamicFeatures_<lineage>"]]`, written by `RunDynamicFeatures` (`SCP-analysis.R:5155`):

```r
list(DynamicFeatures = <df: features, pvalue, padjust, r.sq, dev.expl, ...>,
     raw_matrix    = <cells × (1 + n_features), col 1 = "pseudotime">,
     fitted_matrix = <same layout>,
     upr_matrix    = <same>,          # GAM upper CI
     lwr_matrix    = <same>,          # GAM lower CI
     libsize, lineages, family)
```

The plot drops column 1 (`[, -1]`) from each. **If a requested feature is missing, `DynamicPlot` silently calls `RunDynamicFeatures` on the fly** for just that feature — a GAM fit per feature per lineage. Features may be genes (`rownames(assay)`) or `meta.data` columns; both are accepted and concatenated (for the AnnData equivalents of `assay`/`slot`/`meta.data` see `docs/02_data_contract.md`).

Pseudotime per lineage again comes from `srt@meta.data[, lineages]`; `x_assign = rowMeans(meta[cells, lineages], na.rm=TRUE)` is the x used for **raw points and rugs** (a single shared x when comparing lineages), while the fitted line uses the lineage's own `Pseudotime`.

### Expression transforms

Library normalisation (only if `lib_normalize` and the matrix is non-negative): `raw[,gene] / libsize * median(libsize)` where `libsize = colSums(counts)`; non-integer counts trigger a warning and `libsize := 1`. Then `exp_method` ∈ `raw` / `zscore` (center & scale by the **raw** matrix's column means/sds, applied to all four matrices) / `fc` (divide by raw colMeans) / `log2fc` / `log1p`. Infinities are clamped to `±max(finite)`. The y-axis label is built as `"<exp_method>(normalized counts)"`.

### Rendering pipeline (per panel)

Long-format frame with `Cell, Pseudotime, x_assign, Value ∈ {raw, fitted}, Features, exp, upr, lwr, Lineages, LineagesFeatures`; rows shuffled before plotting so overplotting isn't biased.

Layers in order: `scale_x_continuous(trans = reverse|identity)` → **raw points** (colored by `group.by`, suppressed entirely when `compare_features=TRUE`) → **rug** on x → **ribbon** `lwr..upr` (α 0.4, `grey90` outline, filled by `fill_by`) → **line** on `fitted` → `facet_grid(<lineages_formula> ~ <features_formula>)` → theme → optional `coord_flip()`.

Faceting logic: `compare_lineages=TRUE` ⇒ all lineages overlaid (row facet `.`), else faceted by `Lineages`. Same for features. `fill_by` is `"Lineages"` unless lineages aren't a legend dimension and `compare_features=TRUE`, in which case `"Features"`. `ggnewscale::new_scale_color()/new_scale_fill()` is inserted after each scale because points, rug, ribbon and line each need their own color/fill mapping.

Legend is captured once (`get_legend(p + theme(legend.position="bottom"))`) from the first panel, all panels are then stripped of legends, and the legend is re-attached to the combined gtable via `add_grob`.

### Python port

- GAM fitting: `pyGAM` (`LinearGAM`/`PoissonGAM` with `.prediction_intervals()`), or `statsmodels.gam`. `scFates.tl.test_association` + `scFates.pl.trends` covers the same ground natively and stores fitted values in `adata.layers`/`adata.uns`.
- Everything downstream is pandas + seaborn/matplotlib: `fill_between` for the interval, `ax.plot` for the trend, `ax.scatter` + `sns.rugplot` for cells.

---

## 6. `GroupTreePlot` (line 12472)

```r
GroupTreePlot <- function() {

}
```

**An empty stub.** Not exported in `NAMESPACE`, no `man/GroupTreePlot.Rd`, no callers anywhere in the package. There is no tree layout to port. If a group dendrogram is wanted in the Python version, the nearest existing SCP behaviour is the `group.by` dendrogram inside `GroupHeatmap`/`FeatureHeatmap` (hierarchical clustering of group means); in Python that's `scipy.cluster.hierarchy.linkage/dendrogram` or `sc.tl.dendrogram` → `adata.uns['dendrogram_<groupby>']` (keys `linkage`, `categories_ordered`, `dendrogram_info`), which scanpy already computes. **Flag as: nothing to port.**

---

## 7. `ProjectionPlot` (line 12511)

### Parameters

`srt_query`, `srt_ref`, `query_group=NULL`, `ref_group=NULL`, `query_reduction="ref.embeddings"`, `ref_reduction = srt_query[[query_reduction]]@misc[["reduction.model"]] %||% NULL`, `query_param=list(palette="Set1", cells.highlight=TRUE)`, `ref_param=list(palette="Paired")`, `xlim=NULL`, `ylim=NULL`, `pt.size=0.8`, `stroke.highlight=0.5`.

### Input data contract

- `srt_ref[[ref_reduction]]@cell.embeddings`, `srt_query[[query_reduction]]@cell.embeddings` (reduction accessors → `docs/02_data_contract.md`).
- `srt_query[[query_reduction]]@misc[["reduction.model"]]` — set by `RunKNNMap`/`RunPCAMap` (`SCP-projection.R`), records which reference reduction the query was projected into. Errors if absent and `ref_reduction` not supplied.
- `query_param` / `ref_param` are **splatted into `CellDimPlot`** (`do.call(CellDimPlot, c(srt=..., reduction=..., group.by=..., param))`), so any `CellDimPlot` argument is legal there. `show_stat` is forced `FALSE` for both.

### Rendering pipeline

1. Shared axis limits = union of both embeddings' ranges (unless overridden).
2. `p1` = reference `CellDimPlot`; legend title rewritten to `"Ref: <ref_group>"`; legend extracted.
3. `p2` = query `CellDimPlot` clipped to `xlim`/`ylim`. Its **rendered** point colors are harvested with `ggplot_build(p2)$data[[1]]` — the plot is built purely to steal `colour` per row. Legend keys are then overridden to `shape=21, color="black", fill=<harvested colors>` (outlined dots) and titled `"Query: <query_group>"`.
4. `p3` = `p1` + `new_scale_fill()/new_scale_color()` + a black halo `geom_point(size = pt.size + stroke.highlight)` + the colored `geom_point(size = pt.size)` using `scale_color_identity()`, `facet_null()`, legend off.
5. The two legends are `cbind`-ed as gtables and re-attached to the right of `p3`.

### Python port

Straightforward: two `ax.scatter` calls on one axis, query points drawn with `edgecolors='black', linewidths=stroke.highlight`. The `ggplot_build` color-harvesting trick is unnecessary — compute the query palette directly. Reference-mapping itself is `sc.tl.ingest` or `scarches`/`symphonypy`, both of which write query coords into `adata_query.obsm['X_umap']`.

---

## 8. `EnrichmentPlot` (line 12727) and `adjustlayout` (line 13441)

### Parameters

| Group | Params |
|---|---|
| Source | `srt`, `db="GO_BP"`, `group_by=NULL`, `test.use="wilcox"`, `res=NULL` |
| Type | `plot_type=c("bar","dot","lollipop","network","enrichmap","wordcloud","comparison")` |
| Facet/colour | `split_by=c("Database","Groups")`, `color_by="Database"` |
| Filtering | `group_use=NULL`, `id_use=NULL`, `pvalueCutoff=NULL`, `padjustCutoff=0.05`, `topTerm = ifelse(plot_type=="enrichmap", 100, 6)`, `compare_only_sig=FALSE` |
| Wordcloud | `topWord=100`, `word_type=c("term","feature")`, `word_size=c(2,8)`, `words_excluded=NULL` (→ `SCP::words_excluded`) |
| Network | `network_layout="fr"`, `network_labelsize=5`, `network_blendmode="blend"`, `network_layoutadjust=TRUE`, `network_adjscale=60`, `network_adjiter=100` |
| Enrichmap | `enrichmap_layout="fr"`, `enrichmap_cluster="fast_greedy"`, `enrichmap_label=c("term","feature")`, `enrichmap_labelsize=5`, `enrlichmap_nlabel=4` *(sic — typo in the API)*, `enrichmap_show_keyword=FALSE`, `enrichmap_mark=c("ellipse","hull")`, `enrichmap_expand=c(0.5,0.5)` |
| Text | `character_width=50`, `lineheight=0.5` |
| Cosmetic | `palette="Spectral"`, `palcolor=NULL`, `aspect.ratio=1`, standard block, `combine/nrow/ncol/byrow`, `seed=11` |

### Input data contract — `srt@tools$Enrichment_<group_by>_<test.use>`

Slot name is `paste("Enrichment", group_by, test.use, sep="_")`. The list has `enrichment`, `results`, `geneMap`, `input`, `DE_threshold`. Only `[["enrichment"]]` is used (a flat `rbind` of all `clusterProfiler::enricher` result tables). **Exact columns** — this is what `scp.io.ENRICHMENT_COLUMNS` must carry:

| Column | Origin | Notes |
|---|---|---|
| `ID` | enricher | e.g. `"GO:0002181"` |
| `Description` | enricher | term name |
| `GeneRatio` | enricher | **string** `"k/n"` — parsed via `strsplit(x,"/")` then divided |
| `BgRatio` | enricher | **string** `"K/N"` — same parsing |
| `pvalue`, `p.adjust`, `qvalue` | enricher | |
| `geneID` | enricher, **remapped by SCP to `result_IDtype`** | `"/"`-joined gene symbols |
| `Count` | enricher | int |
| `Groups` | SCP | the DE group (`group1` from `RunDEtest`) |
| `Database` | SCP | e.g. `"GO_BP"`, `"GO_BP_sim"`, `"KEGG"`, `"Combined"` |
| `Version` | SCP | DB version string |

Upstream, `RunEnrichment` pulls DE genes from `srt@tools[["DEtest_<group_by>"]][["AllMarkers_<test.use>"]]`, filtered by `DE_threshold = "avg_log2FC > 0 & p_val_adj < 0.05"`, taking `gene` and `group1` — i.e. `scp.io.DE_COLUMNS`, produced in Python by `scp.io.de_from_rank_genes_groups`.

Derived quantities computed **inside the plot function**:

- `metric <- ifelse(is.null(padjustCutoff), "pvalue", "p.adjust")` — `padjustCutoff` wins; `pvalueCutoff` only applies when `padjustCutoff=NULL`.
- `df[["metric"]] <- -log10(df[[metric]])`
- `FoldEnrichment` / `EnrichmentScore` `= GeneRatio / BgRatio`
- `Description` → `capitalize()` then `str_wrap(width=character_width)`; wrapped labels are italicised (`face = ifelse(grepl("\n", levels), "italic", "plain")`) — a neat "this label was wrapped" cue.

Splitting/faceting:

```r
df_list <- split(enrichment_sig, formula(paste0("~", split_by, collapse = "+")))
facet <- switch(paste0(split_by, collapse="~"),
  "Groups"   = formula("Database ~ Groups"),
  "Database" = formula("Groups ~ Database"),
  formula(paste0(split_by, collapse="~")))       # default: Database ~ Groups
```

Within each split, `split(df, list(Database, Groups))` then `head(topTerm)` per sub-group — so `topTerm` is per (Database × Group), not global. `id_use` (vector, or named list keyed by group) forces `topTerm <- Inf` and selects exact term IDs.

`"GO_sim"` / `"GO_BP_sim"` etc. are handled specially: the *significant* term IDs are selected from the simplified table but plotted against the **unsimplified** `enrichment_sim` rows, so terms dropped by `simplify()` in other groups still appear.

### Plot-type vocabulary

| `plot_type` | Marks | Axes / encodings |
|---|---|---|
| `bar` | `geom_bar(stat="identity", color="black")` + two `geom_text` (white bold underlay + black overlay, `hjust=-0.5`) showing `Count`; `coord_flip()` | x = Description (rev order), y = `-log10(metric)`, fill = `color_by` (legend suppressed), y-limit `1.3×max` |
| `dot` | `geom_point(shape=21)`; `coord_flip()` | x = Description, y = `GeneRatio`, fill = `-log10(metric)` (gradientn), size = `Count` (range 3–6) |
| `lollipop` | `geom_blank` + black `geom_segment(linewidth=2)` + colored `geom_segment(linewidth=1)` + `geom_point(shape=21)`; `coord_flip()` | x = Description ordered by `FoldEnrichment`, y = `FoldEnrichment`, color+fill = `-log10(metric)`, size = `GeneRatio` |
| `network` | bipartite **term–gene** graph: `geom_segment` edges, `geom_label` for gene nodes, double `geom_point` for term nodes, `geom_text_repel` numeric term labels | see below |
| `enrichmap` | term–term graph: `geom_mark_ellipse`/`geom_mark_hull` cluster hulls, `geom_segment` (linewidth = `weight`), `geom_point(shape=21, size=Count, fill=cluster)` | see below |
| `wordcloud` | `ggwordcloud::geom_text_wordcloud(rm_outside=TRUE, eccentricity=1, shape="square", grid_margin=3)`; `coord_flip()` | size = `count`, color = `score`, `angle` = 90° with p=0.4 |
| `comparison` | `geom_point(shape=21)` dotplot across groups | x = `Groups`, y = `Description`, size = `GeneRatio` (`scale_size_area`, max 6), fill = raw `metric` **with `limits=c(0, metric_value)`** so non-significant cells fall to `na.value="grey80"`; a dummy `scale_color_manual(values=NA, na.value="black")` produces the "Non-sig" legend key |

### Network construction

```r
df$geneID <- strsplit(df$geneID, "/"); df_unnest <- unnest(df, cols="geneID")
nodes <- rbind(data.frame(ID=Description, class="term", metric=-log10(p)),
               data.frame(ID=unique(geneID),  class="gene", metric=0))
edges <- df_unnest[, c("Description","geneID")]; edges$weight <- 1
graph <- graph_from_data_frame(edges, vertices=nodes, directed=FALSE)
```

Colouring: terms get `palette_scp(levels(Description), "Spectral")`; each **gene** node is coloured by `blendcolors(colors[terms_it_belongs_to], mode=network_blendmode)` (`"blend"`/`"average"`/`"screen"`/`"multiply"`, alpha-composited RGB, `SCP-plot.R:1026`). Label text colour flips on luminance: `ifelse(colSums(col2rgb(colors)) > 255*2, "black", "white")`. Term nodes are labelled with integers 1..k and the legend carries the full term names via a custom `key_glyph` (`draw_key_cust`) that draws a numbered circle.

### Enrichmap construction

```r
edges <- t(combn(nodes$ID, 2))
edges$weight <- |intersect(geneID[from], geneID[to])|   # shared-gene count
edges <- edges[weight > 0, ]
clusters <- cluster_<enrichmap_cluster>(graph)          # fast_greedy by default
df_nodes$clusters <- factor(paste0("C", clusters$membership))
```

Cluster labels: if `enrichmap_show_keyword=TRUE`, term descriptions are tokenised on `\s|\n`, scored by `sum(-log10(metric))` per keyword × cluster, stoplist-filtered (`words_excluded`, bracketed tokens, empty), top `enrlichmap_nlabel` joined by spaces. Otherwise the top-`enrlichmap_nlabel` term descriptions by `metric` are joined by newlines. A parallel `df_keyword2` does the same over **genes** (`geneID`). Then `enrichmap_label="term"` puts term keywords in the hull annotation and gene keywords in the legend; `="feature"` swaps them.

### `adjustlayout(graph, layout, width, height=2, scale=100, iter=100)`

**Problem it solves:** in the network plot, gene nodes are rendered as *text labels* whose horizontal extent is proportional to the gene-symbol length. A force-directed layout treats nodes as points, so long labels overlap badly. `adjustlayout` post-processes the layout to separate label *boxes*.

Algorithm:

1. Rescale both axes to span `scale` (default 60 when called from `EnrichmentPlot`): `layout[,k] <- layout[,k]/diff(range(layout[,k])) * scale`. Half-widths `w = width/2`, where `width = nchar(node_name)` for genes and a fixed `8` for term nodes.
2. **Pass 1 — degree-ordered neighbour push.** Visit vertices in decreasing degree. For each not-yet-adjusted neighbour whose centre distance `ndist` is below the required `expect = r + nr`, push the neighbour radially outward along the connecting line by the deficit:
   ```r
   dx <- (x-nx)*(expect-ndist)/ndist ;  layout[neighbor,1] <- nx - dx    # (same for y)
   ```
   Each vertex is pushed at most once (tracked in `adjusted`) — high-degree hubs keep their positions.
3. **Pass 2 — `iter` rounds of nearest-neighbour box separation.** Recompute the full distance matrix, find each vertex's nearest neighbour(s), and for any pair whose boxes overlap (`|Δx| < r+nr` **and** `|Δy| < height`), displace the neighbour by the full overlap in *either* x or y — chosen at random per collision (`if (sample(c(1,0),1)==1) dx <- 0 else dy <- 0`). This is a stochastic rectangle-decollision relaxation, effectively a crude label-repulsion solver.

Return: adjusted layout matrix.

**Python replacement:** `networkx.spring_layout` (≙ `layout_with_fr`) for step 0, then either port `adjustlayout` verbatim (it is ~40 lines and deterministic apart from the coin flips — seed it) or use `adjustText.adjust_text` with `avoid_self=True`, or `netgraph`'s label-aware layouts. Given the port must match visually, a direct transcription is the safer choice.

### igraph layouts and clusterings exposed

Layouts (both `network_layout` and `enrichmap_layout`): special-cased `"circle"` → `layout_in_circle`, `"tree"` → `layout_as_tree`, `"grid"` → `layout_on_grid`; anything else is dispatched as `layout_with_<name>` — the roxygen imports name `dh, drl, fr, gem, graphopt, kk, lgl, mds`. Clustering (`enrichmap_cluster`) dispatches `cluster_<name>`: imported are `fast_greedy, infomap, leiden, louvain, spinglass, walktrap, fluid_communities`.

`networkx` equivalents: `fr`→`spring_layout`, `kk`→`kamada_kawai_layout`, `circle`→`circular_layout`, `grid`→manual, `tree`→`graphviz_layout(prog="dot")`, `mds`→`sklearn.manifold.MDS`; `drl/lgl/dh/gem/graphopt` have no direct equivalent — `igraph`'s own Python bindings (`python-igraph`) provide all of them and are the pragmatic choice.

---

## 9. `GSEAPlot` (line 13563) + `gsInfo` (14545) + `gseaScores` (14570)

### Parameters

Shares most of `EnrichmentPlot`'s block via `@inheritParams`. Distinct:
`plot_type=c("line","bar","network","enrichmap","wordcloud","comparison")`, `direction=c("pos","neg","both")`, `line_width=1.5`, `line_alpha=1`, `line_color="#6BB82D"`, `n_coregene=10`, `sample_coregene=FALSE`, `features_label=NULL`, `label.fg="black"`, `label.bg="white"`, `label.bg.r=0.1`, `label.size=4`, `aspect.ratio=NULL`. Hardcoded internals (not parameters): `subplots <- 1:3`, `rel_heights <- c(1.5, 0.5, 1)`, `rel_width <- 3`.

### Input data contract — `srt@tools$GSEA_<group_by>_<test.use>`

Two components are used (unlike `EnrichmentPlot`, which only needs the flat table):

- `[["enrichment"]]` — flat data.frame, used by `comparison`. Columns from `clusterProfiler::GSEA` (`DOSE::gseaResult@result`): `ID`, `Description`, `setSize`, `enrichmentScore`, `NES`, `pvalue`, `p.adjust`, `qvalue`, `rank`, `leading_edge`, `core_enrichment` (`"/"`-joined, symbol-remapped by SCP) — **plus** SCP's `Groups`, `Database`, `Version`. This is `scp.io.GSEA_COLUMNS`.
- `[["results"]]` — a **named list of `gseaResult` S4 objects**, keys `"<Group>-<Database>"` (e.g. `"Ductal-GO_BP"`, `"Ductal-GO_BP_sim"`). Selected by `expand.grid(group_use, db)`. All plot types except `comparison` iterate this list and produce one panel per key.

`gseaResult` slots consumed: `@result` (the df above), `@geneList` (named numeric, **sorted descending** — the ranking metric, `avg_log2FC` by default), `@geneSets[[id]]` (character vector of gene IDs), `@params[["exponent"]]`, `@gene2Symbol` (parallel to `@geneList`).

### Term selection (shared by every plot type)

```r
filter p < metric_value; order by metric
ID_up   <- top topTerm of NES > 0
ID_down <- top topTerm of NES < 0
switch(direction,
  pos  = head(ID_up, topTerm),
  neg  = head(ID_down, topTerm),
  both = head(c(head(ID_up, ceiling(topTerm/2)), head(ID_down, ceiling(topTerm/2))), topTerm))
```

### `gseaScores(geneList, geneSet, exponent=1)` — already implemented as `gsea_scores`

The classic GSEA running score:

```r
geneSet <- intersect(geneSet, names(geneList));  N <- length(geneList); Nh <- length(geneSet)
hits <- names(geneList) %in% geneSet
Phit[hits]  <- abs(geneList[hits])^exponent;  NR <- sum(Phit);  Phit <- cumsum(Phit/NR)
Pmiss[!hits] <- 1/(N-Nh);                     Pmiss <- cumsum(Pmiss)
runningES <- Phit - Pmiss
```

Returns `data.frame(x = 1..N, runningScore, position = as.integer(hits), gene)`. This is Subramanian et al.'s weighted KS statistic, identical to `gseapy`'s internal `enrichment_score`. **Port note:** `gseapy.gsea`/`prerank` returns `res2d` plus per-term `RES` arrays — no need to recompute.

### `gsInfo(object, id_use)`

Wraps `gseaScores` and appends the plotting columns: `ymin`/`ymax` = `±diff(range(runningScore))/20` at hit positions (tick heights), `geneList` (the ranking metric, aligned), `Description`, `CoreGene` (= `core_enrichment` string), and `GeneName` (`@gene2Symbol` if lengths match, else the raw gene ID).

### `plot_type="line"` — the classic three-panel GSEA figure

Per result key, panels assembled with `gtable`/`patchwork` at relative heights `1.5 : 0.5 : 1`, width 3:

**p1 — enrichment score curve.** Background `geom_rect` shading: red `alpha("#C40003",0.2)` above y=0, blue `alpha("#1D008F",0.2)` below. `geom_hline(0)`, then `geom_line(y=runningScore, color=DescriptionP)` at `line_width`. Legend title blanked; the legend entry `DescriptionP` is the wrapped description plus `"\n(NES=…, p.adjust=…, ****)"` where significance stars come from:

```r
> 0.05 → "ns  "; ≤0.05 → "*   "; ≤0.01 → "**  "; ≤0.001 → "*** "; ≤1e-4 → "****"
```

When exactly one gene set is plotted, p1 additionally gets: dashed crosshair segments to the peak `|runningScore|`, a triangle marker at the peak (`shape=24` filled `#F52323` if NES>0, `shape=25` filled `#5E34F5` if NES<0), a subtitle with the stats, and optional core-gene labels — `n_coregene` genes taken either randomly (`sample_coregene=TRUE`) or as the first `n` in rank order from `CoreGene`, drawn as `geom_point` + `geom_text_repel` nudged ±5% of the axis ranges away from the curve. `features_label` overrides the automatic selection (missing genes produce a warning).

**p2 — hit ticks.** `geom_linerange(ymin, ymax, color=DescriptionP)`. For multiple gene sets the ticks are stacked in bands: each term's ticks are reassigned `ymin=i, ymax=i+1`. For a single gene set, a **ranking-metric heat strip** is appended: `geneList` is winsorised to its 2nd/98th percentiles, positive values mapped through `colorRampPalette(c("#F5DCDC","#C40003"))(100)`, negatives through `colorRampPalette(c("#1D008F","#DDDCF5"))(100)`, then run-length-compressed into `geom_rect` blocks occupying the bottom 30% of the tick band.

**p3 — ranked metric.** `geom_segment(x, y=geneList[x], yend=0)` in `grey30`, with annotations `"Positively correlated"` (top-left, `#C81A1F`) / `"Negtively correlated "` (bottom-right, `#3C298C`) *(sic)* conditioned on the sign range, plus a dashed `geom_vline` at the zero-crossing rank labelled `"Zero cross at <x>"`.

Panels are converted to grobs, height/width-fixed via `panel_fix_overall(..., units="null", respect=TRUE)`, `rbind`-ed with `size="first"`, the shared colour legend attached when >1 gene set, and the result key `nm` added as a rotated right-hand strip label (`textGrob(rot=-90)`).

**p1 colour:** `line_color` is used only if `length(line_color) == length(geneSetID_use)`; otherwise `palette_scp(levels(DescriptionP), palette, palcolor)`. Single-set plots hide the legend and force the p2 ticks to black.

### Other `plot_type`s

- **`bar`** — horizontal bars of `NES`, `geom_vline(0)`, `geom_col(fill = Direction ∈ {Pos, Neg}, alpha = -log10(metric))`, with the description drawn **inside the plot** via `geom_text` anchored at x=0 and `hjust` flipped by NES sign (axis text is blanked). `coord_cartesian(xlim = ±max|NES|)`, `facet_grid(Database ~ Groups)`.
- **`comparison`** — dotplot, x = `Groups`, y = `Description`, size = `setSize` (`scale_size_area`, max 6), fill = `NES` on a symmetric `±max|NES|` gradient, **stroke colour** encodes significance (`black` vs `grey90`) rather than fading the fill.
- **`network`**, **`enrichmap`**, **`wordcloud`** — structurally identical to `EnrichmentPlot`'s, with two substitutions: the gene list comes from `core_enrichment` instead of `geneID`, and `Description` is suffixed with `"\n(NES=…, p.adjust=…, stars)"` in the network variant. `enrichmap` additionally derives `Count = length(geneID)` (GSEA has no `Count` column) and a `Direction` factor. Faceting is `Database ~ Groups` (not the `split_by` machinery).

---

## 10. R-only dependency map and Python replacements

| R package | Used by | What for | Python replacement |
|---|---|---|---|
| **slingshot** | `LineagePlot` (upstream `RunSlingshot`) | lineage pseudotime columns `Lineage1..k`, `BranchID` | **scFates** (`scf.tl.curve`, `scf.tl.pseudotime`), **Palantir**, **CellRank** `GPCCA`, `pyslingshot`. Only the resulting `obs` columns are needed. |
| **scanpy (via reticulate)** | `PAGAPlot` | `uns['paga']` | **native** — `sc.tl.paga` |
| **scvelo (via reticulate)** | `VelocityPlot`, PAGA transitions | `obsm[vkey_basis]`, `uns['paga']['transitions_confidence']` | **native** — `scv.tl.velocity_embedding`, `scv.tl.paga` |
| **proxyC** | `compute_velocity_on_grid` | sparse pairwise distances | `sklearn.neighbors.NearestNeighbors` / `scipy.spatial.cKDTree` (faster than the full dense matrix SCP builds) |
| **reticulate** (`array_reshape`) | `compute_velocity_on_grid` | C-order reshape | `numpy.reshape(order='C')` — SCP uses it precisely because R is Fortran-order; in Python this disappears |
| **metR** | `VelocityPlot` stream | `geom_streamline` | `matplotlib.pyplot.streamplot`, or `scv.pl.velocity_embedding_stream` |
| **igraph** | `EnrichmentPlot`/`GSEAPlot` network + enrichmap | layouts, community detection | **python-igraph** (1:1 API) or `networkx` (layouts) + `python-louvain`/`leidenalg` (communities) |
| **ggraph/ggforce** | enrichmap hulls | `geom_mark_ellipse`, `geom_mark_hull` | `matplotlib.patches.Ellipse` fitted per cluster (PCA of member coords), `scipy.spatial.ConvexHull` + `shapely.buffer` for the hull; SCP itself needs `concaveman` for the hull variant |
| **clusterProfiler / DOSE / fgsea** | upstream `RunEnrichment`, `RunGSEA`; `GSEAPlot` reads `gseaResult` slots | ORA + GSEA | **gseapy** (`gp.enrich`, `gp.prerank` — `res2d` has `Term, ES, NES, NOM p-val, FDR q-val, Lead_genes`; see `scp.io.enrichment_from_gseapy`), **decoupler** (`dc.get_ora_df`, `dc.run_gsea`), **blitzgsea** |
| **simplifyEnrichment** | `wordcloud` (`keyword_enrichment_from_GO`) | GO keyword enrichment → `keyword, padj, n_term` | no direct port; approximate with TF-IDF over term descriptions (`sklearn.feature_extraction.text.TfidfVectorizer`) + the `words_excluded` stoplist, or call the R function via `rpy2`. **Flag as a fidelity gap.** |
| **GOSemSim** (via `clusterProfiler::simplify`) | `*_sim` databases | Wang semantic similarity, cutoff 0.7 | `pygosemsim`, or `GOATOOLS` semantic similarity; or precompute in R and ship the reduced term list |
| **ggwordcloud** | `wordcloud` | packed word layout with rotation | **wordcloud** (Python) with `prefer_horizontal=0.6` to match the 60/40 rotation prior, or manual box-packing for exact size/colour control (the SCP version maps size→`count` and colour→`score` independently, which `wordcloud` does not do out of the box) |
| **ggrepel** | all label layers, shadow-text | repulsion + `bg.color`/`bg.r` halo | **adjustText**; halo via `matplotlib.patheffects.withStroke(linewidth=bg.r*size, foreground=bg)` |
| **ggnewscale** | every multi-scale plot | multiple colour/fill scales in one plot | unnecessary in matplotlib — each artist carries its own colormap |
| **patchwork / gtable / grid** | combine, legend re-attachment, `panel_fix_overall` | fixed-aspect panel composition | `matplotlib.gridspec` + `fig.legend`; `panel_fix_overall` ≈ `set_box_aspect` / `height_ratios` |
| **monocle / ggVennDiagram / ggupset** | *not used by any function in this family* | — | (only relevant elsewhere in SCP: monocle in `RunMonocle2/3`, Venn/upset in `VennDiagram`/`FeatureStatPlot` helpers) |

---

## 11. What the Python port can read straight from AnnData

This family reimplements several things that scanpy/scvelo already persist. Reading them natively removes most of the R-side bookkeeping:

1. **`adata.uns['paga']`** — `connectivities`, `connectivities_tree`, `groups`, `pos` (scanpy) and `transitions_confidence` (scvelo). `paga_plot` needs no conversion layer at all; it only re-derives node positions (per-group median of `adata.obsm['X_<basis>']`) and group sizes (`adata.obs[groups].value_counts()`), both one-liners. `scp.io.read_paga` already covers the read. Note that scanpy's `pos` is available if you prefer a graph-native layout.
2. **`adata.obsm['<vkey>_<basis>']`** — the velocity embedding. SCP round-trips this through a Seurat `DimReduc` named `"stochastic_UMAP"`; in Python it's just an `obsm` key. Similarly `adata.obs['<vkey>_length']`, `['<vkey>_confidence']`, `['velocity_pseudotime']`, `['root_cells']`, `['end_points']` are all already present from `RunSCVELO`.
3. **`compute_velocity_on_grid`** — verbatim available as `scvelo.plotting.velocity_embedding_grid.compute_velocity_on_grid`. The R version is a transliteration (the roxygen says so). Only divergence: SCP commented out scvelo's ±1% range padding, so grids differ slightly at the border.
4. **`adata.uns['neighbors']` / `adata.obsp['connectivities']`, `['distances']`** — `adata_to_srt` converts these into Seurat `Graph` objects; nothing downstream in this family needs the Seurat form.
5. **`adata.uns['dendrogram_<key>']`** — would cover the never-implemented `GroupTreePlot`.
6. **`adata.uns['rank_genes_groups']`** — the DE tables that feed `RunEnrichment`/`RunGSEA`. SCP uses Seurat's `FindAllMarkers` output (`gene`, `group1`, `avg_log2FC`, `p_val_adj`); scanpy's equivalent is `sc.get.rank_genes_groups_df(adata, group=...)` with columns `names`, `logfoldchanges`, `pvals_adj`. `scp.io.de_from_rank_genes_groups` is that adapter, and it makes the enrichment plotting layer source-agnostic.
7. **Enrichment results** — there is no AnnData convention here, so the port defines one. Recommendation, matching `scp.io.ENRICHMENT_COLUMNS` / `GSEA_COLUMNS`: store the flat table at `adata.uns['enrichment'][f'{group_by}_{test}']` with SCP's exact column names (`ID, Description, GeneRatio, BgRatio, pvalue, p.adjust, qvalue, geneID, Count, Groups, Database, Version`) and, for GSEA, a parallel `adata.uns['gsea'][...]` holding `{'enrichment': DataFrame, 'results': {f'{group}-{db}': {'result': df, 'geneList': Series, 'geneSets': dict, 'exponent': float}}}`. `gseapy.prerank` output maps onto this with modest renaming (`Term`→`ID`/`Description`, `FDR q-val`→`p.adjust`, `Lead_genes`→`core_enrichment` with `;`→`/`).

### Suggested port ordering

`shorten_segments` (done) → `graph_plot` (unlocks `paga_plot` for free) → `compute_velocity_on_grid` + `velocity_plot` (mostly delegation to scvelo) → `lineage_plot` → `projection_plot` → `enrichment_plot` bar/dot/lollipop/comparison (pure tabular, no graph deps) → `adjust_layout` + network/enrichmap → `gsea_plot` line (the most intricate single figure) → wordcloud (lowest fidelity, blocked on `simplifyEnrichment`) → `dynamic_plot` (blocked on a GAM backend). `GroupTreePlot` is a no-op.
