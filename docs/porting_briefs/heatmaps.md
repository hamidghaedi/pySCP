# Porting brief: heatmaps

**Source**

- **Upstream:** `SCP/R/SCP-plot.R` — helpers L7196–7911, `GroupHeatmap` L7911, `FeatureHeatmap` L9114, `FeatureCorHeatmap` L9980 (stub), `CellCorHeatmap` L10113, `DynamicHeatmap` L11020
- **Target:** `src/scp/pl/_heatmap.py` + `src/scp/heatmap/`
- **Milestones:** 7 (`group_heatmap` / `feature_heatmap`), 9 (`dynamic_heatmap`), 10 (`cell_cor_heatmap`)

Supporting upstream code: `SCP/R/SCP-analysis.R` (`RunDynamicFeatures` L4948–5170 — the GAM machinery; `RunEnrichment` L3766–3975), `SCP/R/SCP-cell_annotation.R` (`RunKNNPredict` L98–480 — the correlation machinery).

This family is built on **ComplexHeatmap**, not ggplot2, and is therefore the hardest part of SCP to port: there is no Python equivalent of ComplexHeatmap's physical-unit track layout engine.

---

## 1. Shared architecture

All four working heatmaps (`GroupHeatmap`, `FeatureHeatmap`, `DynamicHeatmap`, and to a lesser degree `CellCorHeatmap`) follow one pipeline. Roughly 60% of their bodies are literally copy-pasted between functions — the Python port should factor this into one engine.

**Stage 0 — argument normalization.** `set.seed(seed)` (default 11); `match.arg` on `exp_method` / `split_method`; the legend title is built as `paste0(exp_method, "(", data_nm, ")")` where `data_nm <- paste(c(ifelse(lib_normalize,"normalized",""), slot))`, e.g. `"zscore(normalized counts)"`. Every `group.by` / `split.by` / annotation column is coerced to a factor with `levels = unique(x)` (i.e. **order of first appearance**, not sorted — important for reproducing column order in Python; use `pd.Categorical(x, categories=pd.unique(x))`).

**Stage 0b — the `flip` swap.** Very early, if `flip=TRUE` the four clustering flags are transposed:

```r
cluster_rows <- cluster_columns_raw; cluster_columns <- cluster_rows_raw
cluster_row_slices <- cluster_column_slices_raw; cluster_column_slices <- cluster_row_slices_raw
```

Then at the very end, the matrix itself is transposed (`matrix = if (flip) t(mat) else mat`) and every annotation's `which =` is flipped between `"row"` and `"column"`. Net effect: the *user-facing* semantics stay "rows = features, columns = cells/groups", while the *rendered* orientation is transposed. In Python, keep a single canonical `features × observations` matrix and apply `flip` only at the drawing layer.

**Stage 1 — fetch expression.** Features may live in `rownames(assay)` **or** in `colnames(meta.data)` (so QC scores like `G2M_score` can be plotted as "features"). The matrix is built as:

```r
mat_raw <- as_matrix(rbind(slot(srt@assays[[assay]], slot)[gene, cells],
                           t(srt@meta.data[cells, meta])))[features, ]
rownames(mat_raw) <- features_unique   # make.unique(features)
```

For the AnnData equivalents of `slot`, `assay`, `@meta.data`, `@meta.features` and the reduction accessors, see `docs/02_data_contract.md`.

**Stage 2 — library normalization.** Applied only when `lib_normalize=TRUE` (default `identical(slot,"counts")`) **and** `min(mat_raw) >= 0`:

```r
libsize_use <- colSums(counts[, cells])
if (any(libsize_use %% 1 != 0)) libsize_use <- rep(1, n)   # non-integer counts -> no normalization
mat_raw[genes, ] <- t(t(mat_raw[genes, ]) / libsize_use * median(libsize_use))
```

i.e. **CPM-like, but scaled to the median library size instead of 1e6**, and meta.data-derived rows are left untouched. Note the silent fallback: non-integer counts → all library sizes set to 1 (and the legend title drops the "normalized" prefix in `GroupHeatmap`). Already implemented as `lib_normalize()` in `src/scp/heatmap/matrix.py`.

**Stage 3 — aggregation (GroupHeatmap only).** For each `cell_group` in `group.by`:

```r
mat_tmp <- t(aggregate(t(mat_raw), by = list(cell_groups[[cell_group]]), FUN = aggregate_fun))
```

`aggregate_fun` defaults to `base::mean`. A parallel "percent expressed" matrix is computed with `FUN = function(x) sum(x > exp_cutoff)/length(x)` — this is the **only** use of `exp_cutoff`, and it feeds `add_dot`. `FeatureHeatmap` skips aggregation entirely (columns are individual cells, downsampled to `max_cells=100` per group). `DynamicHeatmap` skips aggregation but orders columns by pseudotime.

**Stage 4 — `matrix_process`** (§2).

**Stage 5 — non-finite cleanup** (done in each caller, *not* inside `matrix_process`):

```r
mat[is.infinite(mat)] <- max(abs(mat[!is.infinite(mat)]), na.rm=TRUE) * ifelse(mat[is.infinite(mat)] > 0, 1, -1)
mat[is.na(mat)] <- mean(mat, na.rm = TRUE)
```

(The `grouping.var` branch of `GroupHeatmap` uses `0` instead of the mean for NAs.) Already implemented as `clean_nonfinite(..., na_fill="mean"|"zero")`.

**Stage 6 — color mapping.** `circlize::colorRamp2(breaks, colors)` with exactly 100 breaks and 100 colors from `palette_scp(palette=..., palcolor=...)`:

```r
if (is.null(limits)) {
  if (exp_method %in% c("zscore","log2fc")) {
    b <- ceiling(min(abs(quantile(mat, c(0.01,0.99), na.rm=TRUE))) * 2) / 2   # symmetric, snapped to 0.5
    breaks <- seq(-b, b, length = 100)
  } else {
    b <- quantile(mat, c(0.01,0.99), na.rm=TRUE); breaks <- seq(b[1], b[2], length = 100)
  }
} else breaks <- seq(limits[1], limits[2], length = 100)
```

Two things to reproduce exactly: (a) the symmetric branch takes the **smaller** of |q01| and |q99| and rounds *up* to the nearest 0.5; (b) `colorRamp2` interpolates in **CIE-LAB** space by default, and values outside the break range clamp to the terminal colors. Python: `matplotlib.colors.LinearSegmentedColormap.from_list(...)` over the palette, with `Normalize(vmin=breaks[0], vmax=breaks[-1])` and `cmap.set_over/set_under` to the terminal colors; use `colorspacious`/`colour` for LAB interpolation if exact fidelity matters, otherwise sRGB is close enough at 100 stops. Already implemented as `color_limits()` in `src/scp/heatmap/matrix.py`.

**Stage 7 — row/column split + clustering** (§3), **Stage 8 — annotation stack** (§4), **Stage 9 — `ComplexHeatmap::Heatmap` construction**. One `Heatmap` object is built *per* `group.by` entry (or per lineage) and concatenated:

```r
if (flip) ht_list <- ht_list %v% Heatmap(...)   # vertical stack
else      ht_list <- ht_list + Heatmap(...)     # horizontal concatenation
```

Only the **first** panel gets `left_annotation = ha_left` (split blocks, gene marks); only the **last** gets `right_annotation = ha_right` (feature annotations, enrichment text boxes). Every panel gets its own `top_annotation` (`ha_top_list[[cell_group]]`). `show_heatmap_legend = FALSE` on all panels — legends are collected manually in `lgd` and passed to `draw(ht_list, annotation_legend_list = lgd)`.

**Stage 10 — sizing + grid draw** (§7), **Stage 11 — return list** (§10).

---

## 2. `matrix_process` in full detail

```r
fc_matrix     <- function(m) m / rowMeans(m)
zscore_matrix <- function(m, ...) t(scale(t(m), ...))
log2fc_matrix <- function(m) log2(m / rowMeans(m))
log1p_matrix  <- function(m) log1p(m)
matrix_process <- function(matrix, method = c("raw","zscore","fc","log2fc","log1p"), ...) { ... }
```

Exact numpy equivalents (matrix is `features × observations`; all operations are **row-wise**):

| `exp_method` | R | numpy |
|---|---|---|
| `raw` | identity | `M` |
| `fc` | `M / rowMeans(M)` | `M / M.mean(axis=1, keepdims=True)` |
| `log2fc` | `log2(M / rowMeans(M))` | `np.log2(M / M.mean(axis=1, keepdims=True))` |
| `log1p` | `log1p(M)` | `np.log1p(M)` |
| `zscore` | `t(scale(t(M)))` | `(M - M.mean(1,keepdims=True)) / M.std(1, ddof=1, keepdims=True)` |

**Critical detail:** `scale()` uses the **sample** standard deviation (`ddof=1`), not the population sd. `np.std` defaults to `ddof=0` — you must pass `ddof=1` or results differ by `sqrt(n/(n-1))`.

`method` may also be an arbitrary function; `...` is forwarded (so `zscore` accepts `center=`/`scale=`). The function asserts `identical(dim(processed), dim(matrix))` and errors otherwise.

Edge cases the caller must handle (SCP does, post hoc): rows of all zeros → `rowMeans==0` → `fc`/`log2fc` produce `NaN`/`-Inf`; constant rows → `sd==0` → `zscore` produces `NaN`. Both are swept up by the Stage-5 cleanup. A guard exists upstream: `zscore` errors out if any `group.by` variable has only one level (`"'zscore' cannot be applied to the group(s) consisting of one element"`), because the aggregated matrix would have one column.

**`grouping.var` mode (GroupHeatmap only).** When `grouping.var` is set, `exp_method` is forced to `log2fc` and `matrix_process` is bypassed. Instead cell groups are suffixed `"<group> ; TRUE|FALSE"` by whether `meta[[grouping.var]] == numerator`, aggregated, and then:

```r
mat_tmp <- log2(mat_tmp[, group_TRUE] / mat_tmp[, group_FALSE])
```

Groups lacking both arms are dropped. `add_dot` is force-disabled. Legend title becomes `"<numerator>/other\nlog2fc(normalized counts)"`.

This whole section is already implemented as `matrix_process()` in `src/scp/heatmap/matrix.py`; the `grouping.var` branch is **not** yet and belongs in the `group_heatmap` wrapper.

---

## 3. Row/column splitting and clustering

### Feature (row) splitting

Precedence: an explicit `feature_split` factor wins; otherwise if `n_split` is given **and** `n_split < nrow(mat_split)`, features are clustered. `mat_split` is the horizontal concatenation of the processed matrices for the groups named in `feature_split_by` (default: all of `group.by` / all lineages):

```r
mat_split <- do.call(cbind, mat_list[feature_split_by])
```

`split_method` options:

- **`kmeans`** — `kmeans(mat_split, centers=n_split, iter.max=1e4, nstart=20)`. Lloyd/Hartigan-Wong on the *processed* matrix. Python: `sklearn.cluster.KMeans(n_clusters=n_split, n_init=20, max_iter=10000)`.
- **`hclust`** — `hclust(as.dist(proxyC::dist(mat_split)))`, then `cutree(hc, k=n_split)`. `proxyC::dist` defaults to **euclidean**; `hclust` defaults to **complete** linkage. Python: `scipy.cluster.hierarchy.linkage(M, method="complete", metric="euclidean")` + `fcluster(Z, n_split, criterion="maxclust")`.
- **`mfuzz`** — soft c-means via `e1071::cmeans`. Data is first row-standardized with `standardise()` (`t(apply(data,1,scale))`, same ddof=1 z-score). Fuzzification defaults to `mestimate(data) + 0.1` where

  ```r
  m.sj <- 1 + (1418/N + 22.05) * D^(-2) + (12.33/N + 0.243) * D^(-0.0406*log(N) - 0.1134)
  ```

  with `N = nrow`, `D = ncol`. This is the Schwämmle–Jensen estimator. Falls back to `kmeans` with a warning if `e1071` is absent. Python: `skfuzzy.cmeans(data.T, c=n_split, m=fuzzification, error=1e-5, maxiter=...)`, hard assignment = `argmax(u, axis=0)`.
- **`kmeans-peaktime` / `hclust-peaktime`** (DynamicHeatmap only) — cluster the **1-D** vector `feature_metadata[, order_by]` (peak or valley pseudotime) rather than the expression matrix.

### Split ordering and naming

`GroupHeatmap` / `FeatureHeatmap`: for each feature, find the group with the maximum mean (`groupmean` = per-group means of `mat_split`; `maxgroup = which.max`). Each cluster's representative group is the **mode** of its members' `maxgroup`. Clusters are then sorted by that group's factor level index (`decreasing=` controls direction), optionally permuted by `split_order`. Clusters are renamed to `C1, C2, …` with display labels `"C1(37)"` carrying the member count:

```r
level <- paste0("C", i, "(", sum(row_split == raw_nm), ")")
```

`DynamicHeatmap` instead orders clusters by the **mean peaktime/valleytime** of their members: `df_order <- aggregate(df, by=list(row_split), FUN=mean)` then ordered by `order_by`.

### Column splitting

- `GroupHeatmap`: `column_split = factor(gsub(" : .*", "", levels(cell_groups)))` — i.e. split by `group.by` level when `split.by` is used, so cells are labelled `"CellTypeA : G1"` and blocks group by cell type with sub-blocks by phase.
- `FeatureHeatmap`: `column_split = cell_groups` (the full `"group : split"` factor). It additionally computes **variable gaps** so that sub-blocks of the same group touch:

  ```r
  gaps <- rep(unit(1,"mm"), n); gaps[same_group_adjacent] <- unit(0,"mm")
  ```
- `DynamicHeatmap`: `column_split = NULL` (columns are a pseudotime-ordered continuum).

### Clustering flags

`cluster_rows`, `cluster_columns`, `cluster_row_slices`, `cluster_column_slices` all default `FALSE`. Two special paths:

1. **`cluster_*_slices=TRUE` without `cluster_*=TRUE`** → build a constrained dendrogram with `ComplexHeatmap::cluster_within_group(mat, factor)` and pass it as the clustering object, replacing `row_split` with the *integer count* of slices (`row_split <- length(unique(row_split_raw))`), which makes ComplexHeatmap cut the dendrogram into that many slices.
2. **`cluster_rows=TRUE` + `cluster_features_by` non-NULL** → cluster on a sub-matrix `mat_cluster <- do.call(cbind, mat_list[cluster_features_by])`. If there is no split: `reorder(as.dendrogram(hclust(as.dist(dist(mat_cluster)))), wts = colMeans(mat_cluster), agglo.FUN = mean)` — a weighted dendrogram reordering. If there *is* a split: `cluster_within_group2(t(mat_cluster), row_split_raw)`.

### `cluster_within_group2` vs. ComplexHeatmap's `cluster_within_group`

```r
cluster_within_group2 <- function(mat, factor) {
  for (le in levels(factor)) {
    m <- mat[, factor == le, drop = FALSE]
    hc1 <- hclust(as.dist(dist(t(m))))               # proxyC::dist, euclidean; complete linkage
    dend_list[[le]] <- as.dendrogram(hc1)
    order_list[[le]] <- which(factor == le)[order.dendrogram(dend_list[[le]])]
    dendextend::order.dendrogram(dend_list[[le]]) <- order_list[[le]]   # store GLOBAL indices
  }
  parent <- as.dendrogram(hclust(as.dist(dist(t(sapply(order_list,
              function(x) rowMeans(mat[, x, drop = FALSE])))))))        # cluster the centroids
  dend_list <- lapply(dend_list, function(d) dendrapply(d, function(node) {
      if (is.null(attr(node,"height"))) attr(node,"height") <- 0; node }))
  dend <- merge_dendrogram(parent, dend_list)
  order.dendrogram(dend) <- unlist(order_list[order.dendrogram(parent)])
}
```

Differences from the stock function: (a) it uses **`proxyC::dist`** (sparse-aware, euclidean) rather than `stats::dist`; (b) it force-sets `height = 0` on any node missing a height — stock `cluster_within_group` can produce `NULL` heights for singleton groups, which makes `merge_dendrogram` fail; (c) it handles the single-column group case by building a bare leaf `structure(idx, class="dendrogram", leaf=TRUE, members=1)`; (d) it explicitly reassigns global leaf indices via `dendextend::order.dendrogram<-`. Semantically it is the same construct: cluster within each group independently, cluster the group *centroids* to get a parent tree, graft the sub-dendrograms onto the parent's leaves.

Python: compute per-group `scipy.linkage`, get leaf orders, compute a centroid linkage over group means, and emit the concatenated leaf order plus a synthetic merged linkage matrix for dendrogram drawing (or simply draw the parent and sub-dendrograms as separate axes).

### Names, titles, dendrograms

`row_names_side` defaults to `ifelse(flip,"left","right")`, `column_names_side` to `ifelse(flip,"bottom","top")`, `row_names_rot=0`, `column_names_rot=90`, `column_title_rot = ifelse(flip, 90, 0)`. Titles per panel: the panel title is the `group.by` column name (or lineage name), placed as `column_title` when not flipped and `row_title` when flipped; `"All.groups"` (the synthetic single-group case) renders as `""`. Dendrogram widths/heights are read back out of the fitted `Heatmap` objects in `heatmap_rendersize` (`ht@row_dend_param$width`, `ht@column_dend_param$height`).

---

## 4. The annotation system

This is the part with no Python analogue. ComplexHeatmap's model: a `HeatmapAnnotation` is an ordered list of *annotation functions* (`anno_*`), each occupying one track of fixed physical thickness, stacked on one of four sides of the heatmap body, sharing the body's row/column coordinate system and its split/gap structure. `c(ha1, ha2)` concatenates tracks.

| SCP feature | ComplexHeatmap construct | Side | Size |
|---|---|---|---|
| `group.by` block | `anno_block(align_to = split(seq_along(levels), group), panel_fun = grid.rect(fill=palette[nm]))` | top (left if flip) | auto |
| `split.by` block | same, `align_to` on the split component | top (left if flip) | auto |
| `cell_annotation` — categorical, **GroupHeatmap** | `anno_customize(x, graphics = list(<pie chart grob per group>))` built from `CellStatPlot(..., plot_type="pie")` | top | `cell_annotation_params` default `unit(10,"mm")` |
| `cell_annotation` — numeric, **GroupHeatmap** | `anno_customize` with **violin** grobs from `FeatureStatPlot(...)` | top | idem |
| `cell_annotation` — categorical, **FeatureHeatmap/DynamicHeatmap** | `anno_simple(x=as.character(v), col=palette_scp(v), na_col="transparent", border=TRUE)` | top | default `unit(5,"mm")` |
| `cell_annotation` — numeric, ditto | `anno_simple(x=v, col=colorRamp2(seq(min,max,length=100), palette))` | top | idem |
| `feature_annotation` categorical/numeric | `anno_simple` identical to above | right (bottom if flip) | `feature_annotation_params` default `unit(5,"mm")` |
| feature split blocks | `anno_block(align_to=split(seq_along(row_split_raw), row_split_raw), width/height = unit(0.1,"in"))` | left | `0.1 in` |
| `nlabel` / `features_label` | `anno_mark(at=..., labels=..., side=..., labels_gp=gpar(fontsize=label_size, col=label_color))` | left (top if flip) | auto (text width) |
| `anno_terms` / `anno_keys` / `anno_features` | `anno_empty(0.05in)` + `anno_block(0.1in)` + `anno_textbox(align_to=groups, text=df, max_width=...)` | right | `terms_width=4in`, `keys_width=2in`, `features_width=2in` |
| `separate_annotation` (DynamicHeatmap) | `anno_block(panel_fun = draw a full ggplot grob)` — a density/dynamic line plot spanning the whole axis | top | `unit(10,"mm")` |
| lineage membership (DynamicHeatmap, >1 lineage) | `anno_simple(x = is.na(peaktime)+0, col=c("0"="#181830","1"="transparent"), 5mm)` | right | `5 mm` |
| `Pseudotime` (DynamicHeatmap) | `anno_simple(x=pseudotime, col=colorRamp2(..., pseudotime_palette="cividis"))` | top | auto |

Palette/palcolor arguments are recycled to the length of the annotation vector; mismatched lengths are a hard error. `cell_annotation` entries are resolved against **both** `colnames(meta.data)` and `rownames(assay)` (so a gene name is a legal cell annotation, rendered from `@data`). `feature_annotation` entries must be columns of `assay@meta.features` (populated by `AnnotateFeatures`). See `docs/02_data_contract.md` for how these resolve against AnnData.

Type dispatch is uniform everywhere:

```r
if (!is.numeric(v)) {
  if (is.logical(v)) v <- factor(v, levels = c(TRUE, FALSE))
  else if (!is.factor(v)) v <- factor(v, levels = unique(v))
  # -> discrete palette + Legend(labels=levels, legend_gp=gpar(fill=...))
} else {
  col_fun <- colorRamp2(seq(min(v), max(v), length = 100), palette_scp(palette, palcolor))
  # -> Legend(col_fun = col_fun)
}
```

`show_annotation_name` is `TRUE` only on the first panel (`cell_group == group.by[1]` / `l == lineages[1]`), with `annotation_name_side = ifelse(flip, "top", "left")`.

The `anno_customize` path in `GroupHeatmap` is the exotic one: SCP renders a *complete ggplot* (pie chart or violin) per heatmap column-block, strips it with `+ facet_null() + theme_void() + theme(legend.position="none")`, converts to a grob, and registers it in a `graphics` list keyed by `"<cell_group>:<level>:<split_level>"`. Note the code generates these closures via `eval(parse(text = ...))` string building — a workaround for R's lazy-evaluation capture in loops. In Python this is just a list of callables `(ax) -> None` (the `Track.draw` field).

---

## 5. `add_dot` mechanics

`add_dot` (GroupHeatmap only) turns the heatmap into a dot plot: **color** encodes the processed expression value (same color map as the heatmap), **area/diameter** encodes the fraction of cells above `exp_cutoff`.

The `layer_fun` is assembled as a *string* and `eval(parse(...))`-ed, composing up to five layers in this fixed order:

```r
# 1. whiteout (present when add_dot OR add_violin)
grid.rect(x, y, width=width, height=height, gp=gpar(col='white', lwd=1, fill='white'));
# 2. background tiles (add_bg)
grid.rect(x, y, width=width, height=height, gp=gpar(col=fill, lwd=1, fill=adjcolors(fill, bg_alpha)));
# 3. reticle (add_reticle)
ind_mat = restore_matrix(j, i, x, y);
for (col in seq_len(ncol(ind_mat)))
  grid.lines(x=unit(rep(x[ind_mat[1,][col]],2),'npc'), y=unit(c(0,1),'npc'), gp=gpar(col=reticle_color, lwd=1.5));
for (row in seq_len(nrow(ind_mat)))
  grid.lines(x=unit(c(0,1),'npc'), y=unit(rep(y[ind_mat[,1][row]],2),'npc'), gp=gpar(col=reticle_color, lwd=1.5));
# 4. dots (add_dot)
perc <- pindex(mat_perc_list[[cell_group]], i, j);
grid.points(x, y, pch = 21, size = dot_size * perc, gp = gpar(col='black', lwd=1, fill=fill));
# 5. violins (add_violin) - draws pre-rendered ggplot grobs at each cell
```

Semantics:

- `fill` is the already-color-mapped value for that cell, supplied by ComplexHeatmap.
- `add_bg` paints the cell with `adjcolors(fill, bg_alpha)` (alpha-blended, `bg_alpha=0.5`) with the undiluted `fill` as the border — a soft background behind the dot.
- `dot_size` defaults to `unit(8,"mm")` and is the **diameter at 100% expressed**. `size = dot_size * perc` is a linear diameter scale, so visual *area* scales as `perc²`. The legend confirms this: five points at `seq(0.2, 1, length.out = 5)` labelled `"20%" … "100%"`.
- `mat_perc` is transposed up-front when `flip=TRUE` so that `pindex(mat_perc, i, j)` (row `i`, column `j` lookup) lines up with the transposed body.
- `add_reticle` draws crosshair guide lines at every cell center of the first row and first column of each slice — `restore_matrix(j,i,x,y)` reconstitutes the slice's index grid from ComplexHeatmap's flattened vectors.

Python: a single `ax.scatter(xx, yy, s=(dot_size_pt * perc)**2 * np.pi/4, c=values, cmap=cmap, norm=norm, edgecolors='black', linewidths=1)` over the cell-center grid, plus `ax.pcolormesh` for the `add_bg` layer and `ax.vlines`/`ax.hlines` for the reticle. Take care converting `dot_size` from mm to points² for `scatter`'s area-based `s`.

---

## 6. Enrichment annotation — `heatmap_enrichment`

Signature at line 7297. Triggered when any of `anno_terms` / `anno_keys` / `anno_features` is `TRUE`. **Hard error if `flip=TRUE`** (these are row-side text boxes only).

Flow:

1. `geneID_groups` (= `feature_metadata$feature_split`) is factorized; if all-NA every gene is put in one group.
2. `RunEnrichment(geneID, geneID_groups, IDtype, species, db, TERM2GENE, TERM2NAME, minGSSize=10, maxGSSize=500, GO_simplify, …)` → `list(enrichment = <data.frame>, results = <list of enrichResult>, geneMap, input)`. `enrichment` is a stacked `clusterProfiler::enricher` result with columns `ID, Description, GeneRatio, BgRatio, pvalue, p.adjust, qvalue, geneID, Count, Database, Groups`.
3. Filtering: `metric <- ifelse(is.null(padjustCutoff), "pvalue", "p.adjust")`; keep rows where `df[[metric]] < metric_value`; sort ascending by metric; `split.data.frame(df, ~ Database + Groups)`.
4. For each database, three optional text-box annotations are built per feature cluster:
   - **`anno_terms`** — `head(df$Description, topTerm)` (default 5), `capitalize()`d (or `paste(ID, Description)` if `show_termid`). Color = `palette_scp(-log10(p), "Spectral", matched=TRUE)` blended 50/50 with black (`blendcolors(c(x,"black"))`). Fixed `terms_fontsize = 8`.
   - **`anno_keys`** — keyword cloud. For GO databases it calls `simplifyEnrichment::keyword_enrichment_from_GO(df$ID)` and uses `score = -log10(padj)`, `count = n_term`. For non-GO it tokenizes `Description` on whitespace, groups by word, `score = sum(-log10(metric))`, `count = n()`. Then filters out bracketed tokens (`\\[.*\\]`), empty strings, and a stop-word list (`SCP::words_excluded`); takes `topWord = 20` by score; font size is `rescale(count, to = keys_fontsize)` (default 6–10); a random 0/90° rotation is assigned with probability 60/40.
   - **`anno_features`** — same construction but tokenizing the `geneID` column on `"/"`, so the box shows the member genes sized by how many enriched terms contain them.
5. Each is wrapped as `anno_empty(0.05in) + anno_block(0.1in, fill = split color) + anno_textbox(align_to = geneID_groups, text = <list of data.frames with keyword/col/fontsize/angle>, max_width = …, background_gp = gpar(fill="grey98", col="black"), round_corners = TRUE)`, `which="row"`, `gap = unit(0,"points")`. Names are suffixed `_<db>` so the caller can later `decorate_annotation()` a header label `"<db> (<terms|keys|features>)"` 2.5 mm above the box.

**Python replacement needed.** The R-only Bioconductor dependencies and their suggested substitutes:

| R component | Role | Python substitute |
|---|---|---|
| `clusterProfiler::enricher` | hypergeometric ORA against TERM2GENE | `gseapy.enrich` / `gseapy.enrichr` (offline mode with a custom gene-set dict), or `decoupler.run_ora`, or `goatools.GOEnrichmentStudyNS` |
| GO DAG + `GO_simplify` (`simplify_method="Wang"`, `simplify_similarityCutoff=0.7`) | semantic-similarity redundancy pruning | `goatools` (`semantic_similarity`, Resnik/Lin) + greedy pruning; **Wang similarity has no maintained Python implementation** — either port it (~80 lines over the GO DAG with edge weights `is_a=0.8, part_of=0.6`) or substitute Lin/Resnik and document the difference |
| `simplifyEnrichment::keyword_enrichment_from_GO` | word-level enrichment over GO term names | reimplement: tokenize all GO names, hypergeometric test of each token in the selected term set vs. all GO terms (`scipy.stats.hypergeom`) |
| SCP's own `db_update`/`db_version`/`Ensembl_version`/`convert_species` ID machinery | AnnotationHub / biomaRt gene-set download and cross-species orthology mapping | `pybiomart` or `mygene`; for orthology `pyensembl` or a cached ortholog table. **Flag: the SCP database-caching layer is entirely R-specific and should be replaced by a simple `dict[str, set[str]]` gene-set registry loaded from GMT files.** |
| `ComplexHeatmap::anno_textbox` | word-cloud-like wrapped text block with per-word color/size/rotation | custom matplotlib: greedy line-wrap over `TextPath`-measured widths, `ax.text` per token with `rotation=angle`, plus a `FancyBboxPatch(boxstyle="round")` background |

Recommendation: make enrichment a **pluggable callback** in the Python API (`enrichment_fn: Callable[[list[str], pd.Series], pd.DataFrame]`) returning a frame with `ID, Description, pvalue, padj, geneID, Group`, so the plotting layer is decoupled from whichever backend the user installs.

---

## 7. Sizing engine — `heatmap_rendersize` + `heatmap_fixsize`

This is SCP's ComplexHeatmap analogue of `panel_fix`: it makes the **heatmap body** a fixed physical size, then derives the total device size by adding up every peripheral component.

### `heatmap_rendersize` (7543–7605)

```r
width_sum  <- sum(width  %||% convertWidth(unit(1,"in"), units, valueOnly=TRUE))   # !flip
height_sum <- height[1] %||% convertHeight(unit(1,"in"), units, valueOnly=TRUE)
```

So `width` is a **per-panel vector** (named by `group.by` / lineage) that is *summed* along the concatenation axis, while `height` is scalar (taken from `[1]`) along the shared axis. `flip=TRUE` swaps the two roles. Default body size when unspecified is 1 inch.

Then it accumulates:

- `height_annotation` += `height.HeatmapAnnotation(ha_top_list[[1]])` (top tracks; only the first panel's, since all panels share the same stack height)
- `width_annotation` += `width.HeatmapAnnotation(ha_left)` + `width.HeatmapAnnotation(ha_right)`
- `dend_width = max(ht@row_dend_param$width)` over all panels; `dend_height = max(ht@column_dend_param$height)`
- `name_width = max(ht@row_names_param$max_width)`; `name_height = max(ht@column_names_param$max_height)`
- `lgd_width = sum(width.Legends(lgd_i))` over all legends

Final:

```r
width_sum  <- convertWidth(width_sum + width_annotation + dend_width + name_width, units) + sum(lgd_width)
height_sum <- max(convertHeight(height_sum + height_annotation + dend_height + name_height, units),
                  convertHeight(unit(0.95,"npc"), units))
```

Note the asymmetry: legends are added to **width only** (they are stacked in a right-hand column), and the height has a floor of 95% of the current viewport height.

### `heatmap_fixsize` (7608–7672)

Called only when `fix=TRUE`. It does a *trial render* inside `grid.grabExpr(..., width=width_sum, height=height_sum)` purely to (a) resolve `npc` units into absolute units and (b) read back ComplexHeatmap's computed layout slots. Inside the trial it rewrites each panel's body size to `unit(width_fix %||% ncol, "null")` so panels share space proportionally to their matrix dimensions when no explicit size is given. Then:

```r
if (is.null(width))  ht_width  <- max(max_left_component_width + max_right_component_width
                                      + sum(max_title_component_width)
                                      + annotation_legend_param$size[1] + 1in,  0.95npc)
if (is.null(height)) ht_height <- max(max_top_component_height + max_bottom_component_height
                                      + sum(max_title_component_height) + 1in,
                                      annotation_legend_param$size[2], 0.95npc)
```

i.e. when the user gives no size, the body falls back to 1 inch and everything else is measured.

### The `fix` decision

```r
fix <- (!is.null(row_split) && length(index) > 0) ||
       any(c(anno_terms, anno_keys, anno_features)) ||
       !is.null(width) || !is.null(height)
```

Rationale (stated in the emitted message): `anno_mark` link lines and `anno_textbox` boxes are **not scalable** — they are drawn in absolute units and break if the device is resized. When `fix=TRUE` the result is passed through `panel_fix_overall(gTree, width, height, units)` which pins the gtable cells; otherwise it is simply `wrap_plots(gTree)`.

**Python mapping.** The whole thing collapses to: given `body_w_in`, `body_h_in`, and measured track thicknesses, compute

```
fig_w = left_tracks + row_dend_w + row_names_w + Σ body_w_i + (n_panels-1)*gap + right_tracks + legend_col_w
fig_h = col_title_h + top_tracks + col_dend_h + body_h + col_names_h + bottom_tracks
```

and build a `GridSpec` with `width_ratios` / `height_ratios` in inches — i.e. `fig.add_gridspec(..., width_ratios=[...])` on a figure whose `figsize=(fig_w, fig_h)`. Because matplotlib ratios are relative, absolute-inch sizing is exact as long as `left=0, right=1, top=1, bottom=0` and all padding is folded into the ratios.

---

## 8. `DynamicHeatmap` specifics

**No `trajectory_type` parameter exists** — the task brief's mention of it is not in this version of the source. The relevant knobs are `lineages`, `reverse_ht`, `use_fitted`, `cell_density`, `cell_bins`, `order_by`, `pseudotime_label*`, `separate_annotation`, `border`, `flip`.

### Lineage resolution

`lineages` are column names of `meta.data` holding per-cell pseudotime (NA for cells off that lineage), typically produced by `RunSlingshot`. If a name is missing from `meta.data`, it is recovered from `srt@tools[["DynamicFeatures_<l>"]][["lineages"]]`.

### Cell selection and ordering

```r
cell_union <- cells with at least one non-NA lineage pseudotime
Pseudotime_assign <- rowMeans(meta[cell_union, lineages], na.rm = TRUE)
```

`cell_density != 1` performs **density-equalizing subsampling**: bin `Pseudotime_assign` into `cell_bins=100` equal-width bins with `cut(..., breaks = seq(min,max,length.out=cell_bins))`, set `ncells <- ceiling(max(table(bins)) * cell_density)`, then `sample(cells_in_bin, min(n, ncells))`. With `cell_density=1` (default) nothing is dropped; `< 1` thins dense regions of pseudotime so the heatmap doesn't over-represent them.

Per lineage, cells are sorted by that lineage's pseudotime — **descending** if the lineage is in `reverse_ht`, ascending otherwise — and renamed by appending the lineage name (`paste0(cellname, l)`) so that multiple lineages sharing cells can be `cbind`-ed into one matrix without collisions. This suffix is stripped with `gsub(pattern=l, replacement="", x=...)` whenever the original cell ID is needed — a fragile trick that will misbehave if a cell barcode contains the lineage name; the Python port should use a MultiIndex `(lineage, cell)` instead.

### Feature selection

If `features=NULL`, per lineage read `srt@tools[["DynamicFeatures_<l>"]][["DynamicFeatures"]]` and filter:

```r
exp_ncells > min_expcells(20) & r.sq > r.sq(0.2) & dev.expl > dev.expl(0.2) & padjust < padjust(0.05)
```

Then `num_intersections` controls multi-lineage logic: `features_tab <- table(features)` across lineages, keep features whose count is in `num_intersections %||% seq_along(lineages)` — e.g. `num_intersections = 2` with 2 lineages keeps only shared dynamic features; `1` keeps lineage-exclusive ones.

`feature_metadata` accumulates, per lineage, `<l>peaktime`, `<l>exp_ncells`, `<l>r.sq`, `<l>dev.expl`, `<l>padjust`; the global ordering key is `order_by` = row-wise `max` of the per-lineage peak/valley times. Rows are pre-sorted by it (`decreasing=`).

### GAM fitting (`RunDynamicFeatures`, SCP-analysis.R:5069–5105)

```r
mod <- mgcv::gam(y ~ s(x, bs = "cs") + offset(log(l_libsize)),
                 family = family_use,   # "nb" for raw counts, else "gaussian"
                 data = data.frame(y = Y_ordered[f, ], x = t_ordered, l_libsize = l_libsize))
pre <- predict(mod, type = "link", se.fit = TRUE)
upr <- linkinv(pre$fit + 2*pre$se.fit);  lwr <- linkinv(pre$fit - 2*pre$se.fit)
sizefactor    <- median(Y_libsize) / l_libsize
fitted.values <- fitted(mod) * sizefactor
r.sq <- summary(mod)$r.sq;  dev.expl <- summary(mod)$dev.expl;  pvalue <- summary(mod)$s.table[[4]]
exp_ncells <- sum(Y_ordered[f, ] > min(Y_ordered[f, ]))
peaktime   <- median(t_ordered[fitted.values > quantile(fitted.values, 0.99)])
valleytime <- median(t_ordered[fitted.values < quantile(fitted.values, 0.01)])
padjust    <- p.adjust(pvalue)   # Holm, R's default
```

Key details: cubic-spline-with-shrinkage basis (`bs="cs"`), log-library-size offset, negative-binomial family auto-selected when the data type is `raw_counts` (`check_DataType`), Gaussian otherwise; a per-feature fallback to Gaussian if any value is negative. `peaktime` is the *median pseudotime among cells in the top 1% of fitted values*, not the argmax — this is the value `kmeans-peaktime` clusters on.

Python replacement: `pygam.GAM(s(0, basis='cp'|'ps', lam=...), distribution='poisson'|'normal', link='log'|'identity')` with `exposure`, or `statsmodels.gam.GLMGam` with `BSplines`, or `scipy.interpolate.UnivariateSpline` for a quick approximation. `mgcv`'s GCV-based automatic smoothing-parameter selection is the hardest part to match; `pygam`'s `gridsearch(lam=...)` is the closest analogue. `r.sq` / `dev.expl` must be computed manually: `dev.expl = 1 - deviance/null_deviance`.

### Matrix construction

- `use_fitted=TRUE`: take `srt@tools[["DynamicFeatures_<l>"]][["fitted_matrix"]][,-1]` (drop the pseudotime column), transpose → `features × cells`, columns suffixed with `l`. This is the **smoothed** matrix.
- `use_fitted=FALSE` (default): raw `slot` data for the pseudotime-ordered cells, lib-normalized as in §1, columns suffixed with `l`.

Then `mat_raw <- do.call(cbind, mat_list)` across lineages, `matrix_process` is applied **once to the concatenated matrix** (so z-scores are computed across *all* lineages jointly — important), then infinite/NA cleanup. `mat_split <- mat[, unlist(cell_order_list[feature_split_by])]`.

### Decorations

`pseudotime_label` (numeric vector) draws dashed guide lines at given pseudotime values, per lineage, per row slice:

```r
pseudotime <- cell_metadata[stripped_cell_ids, l]
i <- which.min(abs(pseudotime - pse))
x <- i / length(pseudotime)                      # !flip: vertical line at fraction x
x <- 1 - (i / length(pseudotime))                # flip:  horizontal line
grid.lines(..., gp = gpar(lty = pseudotime_label_linetype, lwd = ..., col = ...))
```

Note the position is a **rank fraction**, not a pseudotime fraction — consistent with the columns being rank-ordered cells.

`separate_annotation` is a list; each element is either one categorical metadata column (→ a `CellDensityPlot` of group densities along the lineage, `x_order="rank"`, reversed for `reverse_ht` lineages) or a vector of feature names (→ a `DynamicPlot` overlaying fitted curves for those features). Each is rendered as a full ggplot grob inside an `anno_block` spanning the entire axis, `unit(10,"mm")` thick.

---

## 9. `FeatureCorHeatmap` / `CellCorHeatmap`

**`FeatureCorHeatmap` is an unimplemented stub** (lines 9980–9982: `function(srt, features, cells) { }`). It is not exported and has no roxygen. Nothing to port; the intent was presumably a gene–gene correlation heatmap. If needed, implement directly: `np.corrcoef(X[:, features].T)` → symmetric heatmap with `scipy.linkage(1 - corr, method='average')` on both axes.

**`CellCorHeatmap`** is a query-vs-reference similarity heatmap, delegating all computation to `RunKNNPredict(..., nn_method="raw", return_full_distance_matrix=TRUE)`.

Matrix construction:

1. Feature selection: `features_type ∈ {HVF, DE}`, `feature_source ∈ {query, ref, both}`, `nfeatures = 2000`; DE uses `RunDEtest(max.cells.per.ident=200, test.use="wilcox")` filtered by `DE_threshold = "p_val_adj < 0.05"`. Final set is `Reduce(intersect, list(features, rownames(query), rownames(ref)))`.
2. Collapsing: if `query_collapsing` / `ref_collapsing` (query default `!is.null(query_group)`, ref default `TRUE`), replace cells by group centroids: `t(log1p(AverageExpression(obj, slot="data", group.by=...)))` — note the mean is taken in **linear** space then `log1p`-ed. Otherwise use `t(GetAssayData(slot="data"))` per cell. Alternatively, with `query_reduction`/`ref_reduction`, use the first 30 embedding dims.
3. Distance:

   ```r
   simil_method <- c("cosine","pearson","spearman","correlation","jaccard","ejaccard","dice","edice","hamman","simple matching","faith")
   dist_method  <- c("euclidean","chisquared","kullback","manhattan","maximum","canberra","minkowski","hamming")
   if (metric %in% c("pearson","spearman")) { if (spearman) {ref <- t(apply(ref,1,rank)); query <- t(apply(query,1,rank))}; metric <- "correlation" }
   d <- 1 - proxyC::simil(as.sparse(ref), as.sparse(query), method = metric, use_nan = TRUE)   # or proxyC::dist(...)
   ```

   Default `distance_metric = "cosine"`.
4. Back in the plotting function:

   ```r
   if (metric %in% simil_method) { simil_matrix <- t(as_matrix(1 - distance_matrix));  simil_name <- "<Metric> similarity" }
   else                          { simil_matrix <- t(as_matrix(1 - distance_matrix / max(distance_matrix)));
                                   simil_name <- "1-dist[m]/max(dist[m])" }
   ```

   After the double negation, for similarity metrics `simil_matrix` is simply the **query × ref similarity**. Infinite → clamped to `max(abs(finite)) * sign`; NA → `0`. Rows are subset to `levels(query_group)` (collapsed) or cell names (uncollapsed); likewise columns for ref.

So: **rows = query (cells or groups), columns = reference**, clustered with plain `cluster_rows` / `cluster_columns` booleans (ComplexHeatmap defaults: euclidean distance, complete linkage) — there is no `cluster_within_group` path and no feature splitting here. Color limits default to `seq(min(simil), max(simil), length=100)` (not quantile-clipped, unlike the expression heatmaps).

An interesting detail: when the group is *not* collapsed **and** that axis is clustered, the group block annotation would be meaningless, so the group variable is silently demoted into the per-cell `cell_annotation` list:

```r
if (isFALSE(query_collapsing) && ((!flip && cluster_rows) || (flip && cluster_columns))) {
  query_cell_annotation <- c(query_group, query_cell_annotation); ...
}
```

Cell labeling (`nlabel`, `label_cutoff`, `label_by ∈ {row, column, both}`) is done in `layer_fun`: per row (and/or column), keep indices whose value is ≥ the `nlabel`-th largest **and** ≥ `label_cutoff`; `label_by="both"` intersects the two sets (`inds[duplicated(inds)]`). Labels are drawn with a manual white halo — 16 offset copies at `theta = seq(pi/8, 2*pi, length.out=16)`, radius `label_size/30` mm, in white, then the black text on top. Python: `matplotlib.patheffects.withStroke(linewidth=2, foreground="white")`.

Return: `list(plot, features, simil_matrix, simil_name, cell_metadata)`.

---

## 10. Return value contract

| Function | Returned list |
|---|---|
| `GroupHeatmap` | `plot` (patchwork/gtable), `matrix_list` (named list, one processed `features × groups` matrix per `group.by`), `feature_split` (factor `C1..Cn` named by feature, or `NULL`), `cell_metadata` (data.frame: `cells`, all `group.by` cols, all `cell_annotation` cols incl. gene expression pulled from `@data`), `feature_metadata` (data.frame: `features`, `features_uique` [sic], `feature_annotation` cols, `duplicated` flag, `feature_split`, `index` = final row order), `enrichment` (the full `RunEnrichment` result list, or `NULL`) |
| `FeatureHeatmap` | identical shape to `GroupHeatmap` (`matrix_list` columns are cells, not groups) |
| `DynamicHeatmap` | `plot`, `matrix` (single `features × (cells × lineages)` matrix), `cell_order` (named list of suffixed cell-ID vectors per lineage), `feature_split`, `cell_metadata` (incl. `Pseudotime_assign` and one column per lineage), `feature_metadata` (incl. `<l>peaktime`, `<l>r.sq`, `<l>dev.expl`, `<l>padjust`, `index`), `enrichment` |
| `CellCorHeatmap` | `plot`, `features`, `simil_matrix`, `simil_name`, `cell_metadata` (row names prefixed `query_`/`ref_`, columns prefixed likewise) |
| `FeatureCorHeatmap` | — (stub) |

Note the roxygen for `GroupHeatmap` documents `cell_metadata` twice and omits `feature_metadata`; the actual returns are as above. `feature_metadata$index` is computed by building a throwaway `Heatmap` and calling `row_order()` on it — i.e. it is the *post-clustering* row order, available before the final render.

This maps onto the `HeatmapResult` dataclass in `src/scp/heatmap/spec.py`.

---

## 11. Python porting strategy

ComplexHeatmap's value is: (1) physical-unit layout of a body plus arbitrary stacked tracks, (2) split-aware coordinate systems shared across tracks, (3) multi-panel concatenation, (4) automatic legend column. Nothing in Python does this. `seaborn.clustermap` is a dead end (single body, 4 fixed slots, no splits). Build it on `matplotlib.gridspec`.

### What already exists

The data model is **done**. `src/scp/heatmap/spec.py` holds `Track`, `Layer`, `LegendSpec`, `Dendrogram`, `PanelSpec`, `HeatmapSpec` and `HeatmapResult`; `src/scp/heatmap/matrix.py` holds the pure-numpy transforms `matrix_process`, `lib_normalize`, `clean_nonfinite` and `color_limits` (§1 Stages 2, 4, 5, 6 and all of §2). `src/scp/pl/_heatmap.py` has the wrapper stubs.

What remains is **`src/scp/heatmap/render.py`** — the layout/draw engine described below — plus the wrappers that assemble a `HeatmapSpec` from an AnnData.

### GridSpec layout

```
Figure (figsize in inches, computed by the size engine)
└── GridSpec (nested)
    ├── row 0: column titles
    ├── row 1: top annotation tracks   (one sub-row per track)
    ├── row 2: column dendrogram
    ├── row 3: BODY ROW
    │     ├── col 0: row-mark / anno_mark axes
    │     ├── col 1: row dendrogram
    │     ├── col 2: left annotation tracks (split blocks)
    │     ├── col 3..N: panel bodies (one per group.by / lineage)
    │     ├── col N+1: right annotation tracks
    │     └── col N+2: LEGEND COLUMN
    └── row 4: column names
```

Splits are handled by making the body itself a **nested GridSpec**: `n_row_slices × n_col_slices` sub-axes with `height_ratios` proportional to slice sizes and `hspace/wspace` driven by `row_gap`/`column_gap` in inches. Every annotation track that runs along the split axis gets the *same* nested subdivision, so slices stay aligned. This is exactly what ComplexHeatmap does internally.

### Rendering algorithm

1. **Measure.** For each track, `size_in` is known. For text elements (row names, `anno_mark` labels, titles), measure with `TextPath(...).get_extents()` or a throwaway `fig.canvas.get_renderer()` pass; this replaces ComplexHeatmap's `max_width` slots.
2. **Size.** `HeatmapSpec.figsize()` sums: `Σ panel.body_size_in + (n-1)*panel_gap + left_tracks + right_tracks + dend_w + names_w + legend_col_w` (width) and `body_height_in + top_tracks + dend_h + titles + names_h` (height). Reproduce the 1-inch default and the `max(..., 0.95 * viewport)` floors only if you care about bug-for-bug fidelity; otherwise drop them.
3. **Lay out.** One top-level `GridSpec(nrows, ncols, width_ratios=[...in...], height_ratios=[...in...], wspace=0, hspace=0)` on a figure of exactly `figsize`. Every ratio is an inch value; because the figure has no margins (`left=0, right=1, top=1, bottom=0`), ratios map 1:1 to inches.
4. **Draw bodies.** `ax.pcolormesh(M, cmap=cmap, norm=norm, edgecolors='none', rasterized=use_raster)` — `pcolormesh` with `rasterized=True` is the direct analogue of `use_raster=TRUE` / `raster_device="png"`. Set `ax.invert_yaxis()` so row 0 is at the top (matching ComplexHeatmap). Then run each `Layer` over the cell-center grid.
5. **Draw tracks.** `simple` → `pcolormesh` of a 1×n or n×1 array; `block` → `ax.axvspan`/`Rectangle` per block plus centered text; `mark` → `ConnectionPatch` leader lines from evenly-redistributed label y-positions to the true feature rows (implement the standard "repel labels along one axis" algorithm: sort targets, then push apart to a minimum spacing); `textbox` → wrapped colored tokens in a `FancyBboxPatch`; `custom` → hand the sub-axes to the `Track.draw` callback (this replaces `anno_customize` with pie/violin ggplot grobs — in Python you just plot into the axes directly, which is far cleaner than SCP's `eval(parse(...))` grob-capture).
6. **Dendrograms.** `scipy.cluster.hierarchy.dendrogram(Z, ax=dend_ax, orientation='left'|'top', no_labels=True, link_color_func=lambda k: 'black')`. For `cluster_within_group2`, draw one dendrogram per slice in the slice's own axes, plus the parent centroid tree in a thin outer axes.
7. **Legends.** A dedicated axes column; stack `LegendSpec` objects top-down. Categorical → `ax.legend(handles=[Patch(facecolor=c, label=l) ...], loc='upper left', frameon=False)`; continuous → `fig.colorbar(ScalarMappable(norm, cmap), cax=inset)`. The dot-size legend is a scatter of 5 markers labelled `20%…100%`, exactly as SCP's `Legend(type="points", pch=21, size=dot_size*seq(0.2,1,length.out=5))`.

### Recommended API surface

Mirror the R entry points as thin functions over the engine, with AnnData-idiomatic names (the signatures already stubbed in `src/scp/pl/_heatmap.py`):

```python
group_heatmap(adata, features, groupby, *, split_by=None, layer="counts",
              exp_method="zscore", exp_cutoff=0, limits=None, lib_normalize=None,
              aggregate_fun=np.mean, n_split=None, split_method="kmeans",
              feature_split=None, decreasing=False,
              cluster_rows=False, cluster_columns=False, cluster_row_slices=False,
              add_dot=False, add_bg=False, add_reticle=False, dot_size=8,
              cell_annotation=None, feature_annotation=None,
              nlabel=20, features_label=None,
              width=None, height=None, flip=False, seed=11) -> HeatmapResult
```

`HeatmapResult` carries `fig`, `matrices: dict[str, pd.DataFrame]`, `feature_split: pd.Series`, `obs_meta: pd.DataFrame`, `var_meta: pd.DataFrame`, `enrichment: pd.DataFrame | None`, plus a `.spec: HeatmapSpec` escape hatch so users can retune the layout and re-`render()`.

### Porting order (remaining work)

1. ~~`matrix_process` + lib-normalization + color-limit derivation~~ — **done** (`src/scp/heatmap/matrix.py`). Still needs `palette_scp` wiring and golden-fixture tests against R.
2. `render.py`: size engine + `HeatmapSpec.render()` with only `simple` tracks, one panel, no splits.
3. Splits + gaps + multi-panel concatenation.
4. Clustering (`kmeans`/`hclust`/`cmeans`, `cluster_within_group2`) + dendrogram axes.
5. `add_dot`/`add_bg`/`add_reticle` layers; `anno_mark` label repulsion.
6. `group_heatmap` and `feature_heatmap` wrappers (milestone 7).
7. `dynamic_heatmap` (milestone 9) — needs the GAM port, the single largest dependency risk; consider shipping it as an optional extra requiring `pygam`.
8. `cell_cor_heatmap` (milestone 10) — straightforward once the engine exists.
9. Enrichment tracks last, behind a pluggable backend interface.

Items 7 and 9 are the only ones with genuine R-only dependencies (`mgcv`, `clusterProfiler`/`simplifyEnrichment`/GO semantic similarity). Everything else is a faithful, deterministic re-implementation.
