# Porting brief: statistical / distribution plots

**Source**

- **Upstream:** `SCP/R/SCP-plot.R` — `FeatureStatPlot` L3524, `ExpressionStatPlot` L3723, `CellStatPlot` L4463, `StatPlot` L4564, `FeatureCorPlot` L5232, `CellDensityPlot` L5679, `VolcanoPlot` L7088
- **Target:** `src/scp/pl/_stat.py`
- **Milestones:** 4 (`feature_stat_plot`), 5 (`cell_stat_plot`), 6 (`cor` / `density` / `volcano`)

Supporting upstream code: `SCP/R/ggsankey.R` (vendored ggsankey), `SCP/R/SCP-analysis.R` (`RunDEtest`, which produces VolcanoPlot's input).

**Scope note up front:** the task brief mentions `dumbbell`, `heatmap`, and `alluvial` as `plot_type` values. They do **not** exist in this version. The complete vocabulary is `violin/box/bar/dot/col` (ExpressionStatPlot) and `bar/rose/ring/pie/trend/area/dot/sankey/chord/venn/upset` (StatPlot). `sankey` is the alluvial equivalent.

---

## 0. Shared architecture

Every function in this family follows the same skeleton, which should be factored into one Python helper:

1. **Seurat unwrapping layer** (`FeatureStatPlot`, `CellStatPlot`) — pulls `srt@meta.data` and `slot(srt@assays[[assay]], slot)`, then delegates to a pure-dataframe engine (`ExpressionStatPlot`, `StatPlot`). These engines are the real reimplementation targets. For the AnnData equivalents of these accessors see `docs/02_data_contract.md`.
2. **Null-group normalization**: `group.by <- "All.groups"` / `split.by <- "All.groups"` with a single-level factor `""` when the user passes NULL. If *both* collapse to `All.groups`, `legend.position` is forced to `"none"`.
3. **Factorization**: every `group.by`/`split.by`/`bg.by`/`stat.by` column is coerced with `factor(x, levels = unique(x))` — **level order is first-appearance order, not sorted**. This is load-bearing for palette assignment and must be reproduced (pandas `Categorical(..., categories=pd.unique(x))`).
4. **Level explosion guard**: `nlev <- sapply(dat, nlevels); nlev <- nlev[nlev > 100]` → interactive `askYesNo` unless `force = TRUE`.
5. **Combination grid** (`comb`) built via `expand.grid` over `(stat_name, group_name [, group_element, split_name])`, one subplot per row, named `"stat:group:elements:splits"`.
6. **Return**: `combine=TRUE` → `patchwork::wrap_plots(plotlist, nrow, ncol, byrow)` if >1 plot else the bare ggplot; `combine=FALSE` → the named list.

`palette_scp` (line 168) is the universal color factory:
- `type="auto"` → `"continuous"` if `is.numeric(x)` else `"discrete"`.
- Discrete: `n_x = nlevels(x)`; if `n_x <= length(palcolor)` take the first `n_x` colors, else `colorRampPalette(palcolor)(n_x)`. Returns a **named** vector keyed by level.
- Continuous: cuts `1:100` (or `x` if `matched=TRUE`) into `n=100` bins between `min(x)`/`max(x)`, names the colors by the interval strings `"(a,b]"`. `FeatureCorPlot` parses these names back into numeric bounds.
- `NA_keep`/`NA_color` appends a `"NA"`-named entry.

---

## 1. `FeatureStatPlot` (3524) → `ExpressionStatPlot` (3723)

### 1.1 Parameter list (identical for both, except `srt/cells/slot/assay/combine/nrow/ncol/byrow` which live only on the wrapper)

**Data selection (Seurat-specific: `srt`, `slot`, `assay` — see `docs/02_data_contract.md`)**
`srt`, `stat.by` (features and/or numeric meta.data columns), `group.by=NULL`, `split.by=NULL`, `bg.by=NULL`, `plot.by=c("group","feature")`, `fill.by=c("group","feature","expression")`, `cells=NULL`, `slot="data"`, `assay=NULL`, `keep_empty=FALSE`, `individual=FALSE`, `calculate_coexp=FALSE`.

**Geometry**: `plot_type=c("violin","box","bar","dot","col")`.

**Primary color**: `palette="Paired"`, `palcolor=NULL`, `alpha=1`.
**Background stripes**: `bg_palette="Paired"`, `bg_palcolor=NULL`, `bg_alpha=0.2`.

**Overlays**
- `add_box=FALSE`, `box_color="black"`, `box_width=0.1`, `box_ptsize=2`
- `add_point=FALSE`, `pt.color="grey30"`, `pt.size=NULL`, `pt.alpha=1`, `jitter.width=0.4`, `jitter.height=0.1` (roxygen says 0.5 for width — the **code default is 0.4**; trust the code)
- `add_trend=FALSE`, `trend_color="black"`, `trend_linewidth=1`, `trend_ptsize=2`
- `add_stat=c("none","mean","median")`, `stat_color="black"`, `stat_size=1`, `stat_stroke=1`, `stat_shape=25`
- `add_line=NULL` (numeric y-intercept), `line_color="red"`, `line_size=1`, `line_type=1`

**Highlighting (Seurat cell-name based — see `docs/02_data_contract.md`)**: `cells.highlight=NULL` (TRUE = all), `cols.highlight="red"`, `sizes.highlight=1`, `alpha.highlight=1`.

**Axis / scaling**: `same.y.lims=FALSE`, `y.min=NULL`, `y.max=NULL` (numeric or `"qNN"` quantile string), `y.trans="identity"` (`"log2"` etc.), `y.nbreaks=5`, `sort=FALSE|TRUE|"increasing"|"decreasing"`, `stack=FALSE`, `flip=FALSE`.

**Statistics**: `comparisons=NULL` (list of length-2 vectors, or `TRUE`), `ref_group=NULL`, `pairwise_method="wilcox.test"`, `multiplegroup_comparisons=FALSE`, `multiple_method="kruskal.test"`, `sig_label=c("p.signif","p.format")` (code default = first = `"p.signif"`; roxygen incorrectly says `p.format`), `sig_labelsize=3.5`.

**Cosmetics**: `aspect.ratio`, `title`, `subtitle`, `xlab`, `ylab="Expression level"`, `legend.position="right"`, `legend.direction="vertical"`, `theme_use="theme_scp"`, `theme_args=list()`, `combine=TRUE`, `nrow`, `ncol`, `byrow=TRUE`, `force=FALSE`, `seed=11`.

### 1.2 `plot.by = "feature"` — the reshape trick

This is the single most surprising mechanic. When `plot.by="feature"`, x becomes the feature axis instead of the group axis:

```r
meta.reshape <- FetchData(srt, vars = c(stat.by, group.by, split.by), ...)
meta.reshape <- melt(meta.reshape, measure.vars = stat.by,
                     variable.name = "Features", value.name = "Stat.by")
exp.data <- matrix(0, nrow = 1, ncol = nrow(meta.reshape), dimnames = list("Stat.by", rownames(meta.reshape)))
```

It long-melts the cells×features matrix into `n_cells * n_features` pseudo-rows keyed `"<cell>-<feature>"`, builds a **dummy all-zero 1-row expression matrix** (never read, because `stat.by` resolves against `colnames(meta.data)`), renames the value column `Stat.by` → `<group level>`, and then calls `ExpressionStatPlot` **once per level of the original `group.by`** with `group.by="Features"`, `stat.by=<that level>`. `bg.by` is ignored (with a message) and `group.by` must be length 1. Python equivalent: `melt` then loop groups; each subplot is one cell group with features on x.

### 1.3 `plot_type` vocabulary (ExpressionStatPlot)

Base mapping for all non-`col` types: `aes(x = group.by, y = value, fill = fill.by)`. Data is **per-cell, unaggregated**, except where noted.

| `plot_type` | geoms built | x / y | pre-aggregation |
|---|---|---|---|
| `violin` | `geom_violin(scale="width", trim=TRUE, alpha, position_dodge())` | x=group, y=raw value | none (KDE per group × split) |
| `box` | `geom_boxplot(aes(group=group.unique), position_dodge(0.9), color="black", width=0.8, outlier.shape=NA)` + `stat_summary(fun=median, geom="point", shape=21, fill="white", size=1.5)` | x=group, y=raw | quartiles per (group,split); `add_box` is force-disabled |
| `bar` | `stat_summary(fun=mean, geom="col", position_dodge(0.9), width=0.8, color="black")` + `stat_summary(fun.data=mean_sdl, mult=1, geom="errorbar", width=0.2)` | x=group, y=mean ± 1 SD | mean and sd per (group,split). Adds `geom_hline(0, linetype=2)`. `y_min_use` reset to `layer_scales(p)$y$range$range[1]` |
| `dot` | `geom_count(aes(y=bins), shape=21, position_dodge(0.9))` + `scale_size_area(max_size=6, n.breaks=4)` | x=group, y=**binned** value | values cut into **14 equal-width bins** over `[min,max]`; each bin replaced by its midpoint; dot size = count in (group,bin) |
| `col` | `geom_col()` on one bar per cell | x=`cell` (integer 1..n in group order; reversed if flip), y=raw | none — one column per cell. Adds `geom_vline` dashed separators at group boundaries and blanks the x axis text |

`col` rejects all overlays (`add_box/add_point/add_trend/add_stat` warn and reset to FALSE) and rejects all comparisons.

The dot binning snippet:
```r
bins <- cut(dat$value, breaks = seq(min(dat$value), max(dat$value), length.out = 15), include.lowest = TRUE)
bins_median <- sapply(strsplit(levels(bins), ","), function(x) median(as.numeric(gsub("\\(|\\)|\\[|\\]","",x))))
dat[["bins"]] <- bins_median[bins]
```

### 1.4 Aggregation / grouping pipeline (pseudocode)

```
# --- global (once) ---
dat_exp   = cbind(t(exp.data[features_gene, ]), as_matrix(meta.data[, features_meta]))
dat_group = meta.data[, c("cells", group.by, bg.by, split.by)]
dat_use   = cbind(dat_group, dat_exp[rownames(dat_group), ])
if cells: subset rows
pt.size   = pt.size or min(3000 / n_cells, 0.5)

if calculate_coexp and len(features_gene) > 1:
    status = check_DataType(exp.data)
    if raw:  CoExp = geometric mean over genes          # exp(mean(log(x)))
    if log:  CoExp = log1p(exp(mean(log(expm1(x)))))    # geometric mean in linear space
    stat.by += ["CoExp"]

if same.y.lims:
    v = finite values over ALL stat.by columns
    y.max = y.max or max(v);  y.min = y.min or min(v)   # "qNN" -> quantile

# --- subplot grid ---
comb = expand.grid(group_name=group.by, stat_name=stat.by)
if individual: cross with (group_element = each level of g, split_name = each level of split.by)
else:          cross with (group_element = ALL levels as a list, split_name = ALL levels as a list)
# -> individual=TRUE yields one panel per (feature, group, group level, split level)

# --- per subplot (g, f, single_group, sp) ---
dat = dat_use[ dat_use[g] in single_group & dat_use[split.by] in sp , c(group cols, f) ]
dat[g] = droplevels(dat[g])
dat.value = dat[f];  dat.group.by = dat[g];  dat.split.by = dat[split.by]
if split.by == g: dat.split.by = dat.group.by
dat.features = f
dat["bg.by"] = dat[bg.by or g]

if sort:                                  # median-based reorder of x
    med = aggregate(value ~ group.by, FUN=median)
    decreasing = not (sort == "increasing")
    levels(group.by) = med.sort(decreasing)

# fill channel
if fill.by == "feature":    fill = f;                           keynm = "Features"
if fill.by == "group":      fill = split.by if split else group; keynm = split.by or g
if fill.by == "expression": fill = median_values[(group,split), f]; keynm = "Median expression"

# the dodge key: every (split, group) cell gets a unique factor level
dat["group.unique"] = factor("sp-<split>-gp-<group>",
    levels = paste over expand.grid(x=levels(split.by), y=levels(group.by)))
dat = dat[order(group.unique), ]

# per-panel y limits (overridden by same.y.lims values if set)
y_max_use = y.max or max(finite values);  y_min_use = y.min or min(finite values)

if flip: levels(group.by) = rev(levels(group.by)); aspect.ratio = 1/aspect.ratio
```

`stack` + `sort` is illegal: "Set sort to FALSE when stack is TRUE".

### 1.5 Faceting and `stack`

Each subplot always gets a facet on the feature column:
```r
if (stack && !flip) p + facet_grid(features ~ .) + theme(strip.text.y = element_text(angle = 0))
else                p + facet_grid(. ~ features)
```
When `stack=TRUE` and `length(stat.by) > 1` and `individual=FALSE`, `FeatureStatPlot` post-processes: it groups `plist` by the `g` component of the name, extracts **one legend** via `get_legend(plist_g[[1]])`, strips axis titles/text/panel grid/margins from all but the edge panel, converts each to a grob with `as_grob`, and `rbind`s (or `cbind`s when flip) them into a single gtable. A rotated `textGrob(ylab)` is attached to the left (bottom when flipped), the shared legend is attached at `legend.position`, then 1 cm padding and `wrap_plots(gtable)`. Under `stack`, the y scale gets only two breaks:
```r
scale_y_continuous(trans=y.trans, breaks=c(y_min_use, y_max_use), labels=round(...,1))
```
This is the "stacked violin" figure. In Python this is a shared-x subplot grid with per-row y-limits and two ticks, one figure legend.

### 1.6 Statistical testing layer

All tests go through `ggpubr::stat_compare_means` (checked via `check_R("ggpubr")`). `comparisons` entries must all be length 2. Three mutually composable modes:

**(a) `comparisons = TRUE`** — requires `split.by`. Compares **split levels within each x group**:
```r
group_use <- names(which(rowSums(table(group.by, split.by) >= 2) >= 2))
method <- if (any(rowSums(table(...) >= 2) >= 3)) multiple_method else pairwise_method
stat_compare_means(data = dat[group.by %in% group_use, ],
  mapping = aes(x=group.by, y=value, group=group.unique),
  label = sig_label, label.y = y_max_use, size = sig_labelsize,
  step.increase = 0.1, tip.length = 0.03, vjust = 1, method = method)
y_max_use <- layer_scales(p)$y$range$range[2]
```
i.e. only groups with ≥2 split levels having ≥2 observations are tested; ≥3 such levels anywhere switches to Kruskal-Wallis.

**(b) `comparisons = list(c("A","B"), ...)`** — bracketed pairwise comparisons across x positions, `method = pairwise_method`, `ref.group = ref_group`, `vjust = 0`. Then the ceiling is expanded:
`y_max_use <- ymin + (ymax - ymin) * 1.15`.

**(c) `multiplegroup_comparisons = TRUE`** — a single omnibus label (`multiple_method`, default `kruskal.test`) per facet at `label.y = y_max_use`, `vjust = 1.2`, `hjust = 0`; same 1.15 expansion.

`sig_label` is passed straight to ggpubr: `"p.signif"` → stars, `"p.format"` → formatted p-value. Bracket stacking is `step.increase = 0.1` (fraction of range per additional bracket), tips `0.03`.

**Overlay layers (order matters — they are added after the comparison layers, so they draw on top):**
- `add_point`: `geom_point(..., position = position_jitterdodge(jitter.width, jitter.height, dodge.width = 0.9, seed = 11))`, `show.legend = FALSE`. A throwaway `linetype = rep(f, n)` aesthetic is used only to keep grouping consistent. `cells.highlight` adds a second `geom_point` over the same jitter (same seed → identical positions) in `cols.highlight`. Note the jitter seed is hardcoded `11`, not `seed`.
- `add_box`: filled `geom_boxplot(width = box_width, color = box_color, fill = box_color, outlier.shape = NA)` grouped by `group.unique`, plus a white median dot (`shape 21`, `size = box_ptsize`).
- `add_trend` (only `violin`/`box`/`bar`): connects per-group summaries (median for violin/box, mean for bar). With >1 split level the dodged x positions are not directly knowable, so it renders a throwaway layer and **harvests its computed positions**:
```r
p_data <- p + stat_summary(fun=median, geom="point", aes(group=split.by, color=group.by), position=position_dodge(0.9), ...)
p <- p + geom_line(data = layer_data(p_data, length(p_data$layers)),
                   aes(x = x, y = y, group = colour), color = trend_color, linewidth = trend_linewidth,
                   inherit.aes = FALSE) + <white summary point layer>
```
With one split level it simply uses `stat_summary(geom="line")`. In Python: compute dodged x offsets explicitly and `plot()` the polyline.
- `add_stat`: `stat_summary(fun = add_stat, geom = "point", aes(shape = stat_shape))` + `scale_shape_identity()` — shape 25 is a filled down-triangle marker.
- `add_line`: `geom_hline(yintercept = add_line, color/linetype/linewidth = line_*)`.

### 1.7 Background striping (`bg.by`)

`bg.by` must be a **coarser** partition than `group.by` — enforced by:
```r
df_table <- table(meta.data[[g]], meta.data[[bg.by]])
if (max(rowSums(df_table > 0)) > 1) stop("'group.by' must be a part of 'bg.by'")
bg_map[[g]] <- setNames(colnames(df_table)[apply(df_table, 1, function(x) which(x > 0))], rownames(df_table))
```
so `bg_map[g]` maps each group level → its single parent level. When `bg.by` is NULL, `bg_map` is identity and `bg_color` defaults to `rep(c("transparent","grey85"), nlevels)` — alternating zebra stripes. With `bg.by` set, colors come from `palette_scp(levels(bg), palette = bg_palette, palcolor = bg_palcolor)`.

Rectangles (only when `individual = FALSE`):
```r
bg_data$x    <- as.numeric(group.by)             # 1..k discrete positions
bg_data$xmin <- ifelse(x == min(x), -Inf, x - 0.5)
bg_data$xmax <- ifelse(x == max(x),  Inf, x + 0.5)
bg_data$ymin <- -Inf; bg_data$ymax <- Inf
bg_data$fill <- bg_color[bg_map[[g]][as.character(group.by)]]
geom_rect(data = bg_data, xmin=..., xmax=..., ymin=..., ymax=..., fill=..., alpha = bg_alpha, inherit.aes = FALSE)
```
Note `fill`/`xmin`/… are passed as **vectorized layer parameters**, not aesthetics, so they never touch the fill scale (this is the trick that lets the violin fill scale stay free). The outermost stripes bleed to ±Inf so there is no white gutter at the panel edge. For `plot_type = "col"` the same is done in continuous cell-index space: `range()` of the cell indices in each group, ±0.5, with the same ±Inf edge handling. **Python equivalent**: `ax.axvspan(xmin, xmax, color=c, alpha=bg_alpha, zorder=0)` drawn before the main artists.

### 1.8 Color and scale assembly

- `fill.by="feature"` → `palette_scp(stat.by, palette, palcolor)`, `levels_order = unique(stat.by)`.
- `fill.by="group"` → keyed on `split.by` levels if splitting, otherwise on `g` levels.
- `fill.by="expression"` → continuous: medians are precomputed once per panel via
  `aggregate(dat_use[, stat.by], by = list(group, split), FUN = median)`, keyed `"<group>-<split>"`; colors = `palette_scp(..., type="continuous")`; `colors_limits = range(median_values)`. Rendered with `scale_fill_gradientn(limits = colors_limits)` + `guide_colorbar(frame.colour="black", ticks.colour="black")`.
- Discrete case: `scale_fill_manual(name = paste0(keynm, ":"), values = colors, breaks = levels_order, drop = FALSE)` plus a matching `scale_color_manual`; `stack=TRUE` additionally sets `limits = levels_order` so empty levels still occupy the legend. Legend override: `guide_legend(order = 1, override.aes = list(size = 4, color = "black", alpha = 1))`.
- y limits are applied as **clipping, not filtering**: `coord_cartesian(ylim = c(y_min_use, y_max_use))` (or `coord_flip(ylim = ...)` when flipped).
- `same.y.lims` computes global min/max across all `stat.by` columns before the loop, so every panel shares limits.

⚠ **Bug to not port**: in the `same.y.lims` block, the character-quantile branch calls `quantile(values, q.max)` but the local variable is named `valus` — `values` is undefined at that point. So `same.y.lims=TRUE` combined with a `"qNN"` string errors out. The per-panel branch (which uses a correctly-named `values`) is fine.

---

## 2. `CellStatPlot` (4463) → `StatPlot` (4564)

`CellStatPlot` is a thin wrapper: `cells <- cells %||% colnames(srt@assays[[1]])`, `meta.data <- srt@meta.data[cells, ]`, then `StatPlot(...)`. Only `srt` and `cells` are Seurat-specific — see `docs/02_data_contract.md`.

### 2.1 Parameters

`meta.data`, `stat.by` (1 categorical column, or 2–7 for sankey/chord/venn/upset), `group.by=NULL`, `split.by=NULL`, `bg.by=NULL`, `flip=FALSE`, `NA_color="grey"`, `NA_stat=TRUE`, `keep_empty=FALSE`, `individual=FALSE`, `stat_level=NULL`, `plot_type=c("bar","rose","ring","pie","trend","area","dot","sankey","chord","venn","upset")`, `stat_type=c("percent","count")`, `position=c("stack","dodge")`, `palette="Paired"`, `palcolor=NULL`, `alpha=1`, `bg_palette="Paired"`, `bg_palcolor=NULL`, `bg_alpha=0.2`, `label=FALSE`, `label.size=3.5`, `label.fg="black"`, `label.bg="white"`, `label.bg.r=0.1`, `aspect.ratio`, `title`, `subtitle`, `xlab`, `ylab`, `legend.position="right"`, `legend.direction="vertical"`, `theme_use="theme_scp"`, `theme_args=list()`, `combine=TRUE`, `nrow`, `ncol`, `byrow=TRUE`, `force=FALSE`, `seed=11`.

Validation rules: `length(stat.by) >= 2` ⇒ plot_type must be sankey/chord/venn/upset; chord requires exactly 2; venn ≤ 7. `group.by` is ignored (warning) for those four; `stat_type` is forced to `"count"` for them. `rose`/`ring`/`pie` force `aspect.ratio <- 1`.

### 2.2 The aggregation core

```r
dat_all   <- meta.data[, unique(c(stat.by, group.by, split.by, bg.by))]
dat_split <- split.data.frame(dat_all, dat_all[[split.by]])   # one panel per split level

# percent
dat_use <- dat_split[[sp]] %>%
  xtabs(formula = paste0("~", stat.by, "+", g), addNA = NA_stat) %>% as.data.frame() %>%
  group_by(across(all_of(g)), .drop = FALSE) %>% mutate(groupn = sum(Freq)) %>%
  group_by(across(all_of(c(stat.by, g))), .drop = FALSE) %>% mutate(value = Freq / groupn) %>%
  as.data.frame()
# count
dat_use <- ... xtabs(...) %>% as.data.frame() %>% mutate(value = Freq)
```
So: **complete cross-tab of `stat.by × group.by` within each split level** (`addNA = NA_stat` keeps NA as its own category), then `value = Freq / sum(Freq within group)` for percent. Empty cells are retained (`.drop = FALSE`) then filtered with `dat[!is.na(value), ]`. Pandas equivalent: `pd.crosstab(stat, group, dropna=not NA_stat)` → `melt` → `value = Freq / Freq.groupby(group).transform('sum')`.

The subplot grid mirrors ExpressionStatPlot: `expand.grid(stat_name, group_name)` crossed with either every `(group level, split level)` pair (`individual=TRUE`) or the full level list × each split level; names are `"group:elements:split"`.

### 2.3 `plot_type` vocabulary (StatPlot)

All of the first seven share `x = group.by`, `y = value`, `fill = stat.by`, with data already aggregated.

| `plot_type` | geoms | coordinate system | notes |
|---|---|---|---|
| `bar` | `geom_col(width=0.8, color="black", alpha, position=position_use)` | cartesian | canonical stacked/dodged composition bar |
| `trend` | `geom_area` on an *expanded* frame + `geom_col(width=0.6)` | cartesian | each row duplicated; the x of the pair is set to `x−0.3` and `x+0.3` (group coerced to numeric), producing a ribbon that connects adjacent bars. Area drawn at `alpha/2`, `color="grey50"`. Grid major is blanked when `stat_type=="percent"` |
| `rose` | `geom_col` | `coord_polar(theta="x", start = if flip pi/2 else 0)` | Nightingale rose; group levels around the circle |
| `ring` | `geom_col` | `coord_polar(theta="y", ...)` | donut: a dummy level `"   "` is prepended to the group factor and an all-NA row appended, which reserves the inner hole:<br>`dat[[g]] <- factor(dat[[g]], levels=c("   ", levels(dat[[g]]))); dat <- rbind(dat, dat[nrow(dat)+1, ]); dat[nrow(dat), g] <- "   "` |
| `pie` | `geom_col` | `coord_polar(theta="y", ...)` | identical to ring **minus** the hole row |
| `area` | `geom_area(color="black", alpha, position)` | cartesian | dodge uses `position_dodge2(width=0.9, preserve="total")` (all others use `preserve="single"`) |
| `dot` | `geom_point(aes(fill=stat.by, size=value), shape=21, color="black")` + `scale_size_area(name=capitalize(stat_type), max_size=12)` | cartesian | **y = `stat.by`, not value** — a dot matrix of group × category with size = count/percent; `position_identity()`, `scale_x_discrete(drop=!keep_empty)` |
| `sankey` | `make_long(dat_use, stat.by)` + `geom_sankey(color="black", flow.alpha=alpha)` | `theme_void` | multi-column alluvial across ≥2 `stat.by` columns; NAs recoded to a literal `"NA"` level |
| `chord` | `circlize::chordDiagram(table(stat.by[1], stat.by[2]), grid.col=colors, transparency=0.2, link.lwd=1, link.lty=1, link.border=1)` | base graphics | returns `recordPlot()` — **not a ggplot** |
| `venn` | `ggVennDiagram::process_data` → `geom_sf` regions + `geom_sf` set edges + two `geom_text_repel` layers | sf | region fill = `blendcolors(member set colors, mode="blend")`; region label = `"<count>\n<pct>%"`; set label = `"<name>\n(<size>)"` |
| `upset` | `geom_bar(aes(fill=after_stat(count)))` + `ggupset::scale_x_upset(sets=stat.by, n_intersections=20)` + `theme_combmatrix` | cartesian | builds a list-column `intersection` of the `stat.by` names that are TRUE for each row; rows with empty intersections dropped |

### 2.4 `position`, `label`, `flip`

```r
if (plot_type == "dot")        position_use <- position_identity();  scalex <- scale_x_discrete(drop=!keep_empty)
else if (position == "stack")  position_use <- position_stack(vjust = 0.5)
                               scalex <- scale_x_discrete(drop=!keep_empty, expand=c(0,0))
                               scaley <- scale_y_continuous(labels = number|percent, expand=c(0,0))
else /* dodge */               position_use <- position_dodge2(width=0.9, preserve = "total"|"single")
                               scaley <- scale_y_continuous(limits = c(0, max(value)*1.1), labels=..., expand=c(0,0))
```
`vjust = 0.5` on the stack position is what centers the `geom_text_repel` labels in the middle of each stacked segment (the same `position_use` object is reused for the label layer). Labels are `value` for count, `paste0(round(value*100,1), "%")` for percent, drawn with `geom_text_repel(colour=label.fg, size=label.size, bg.color=label.bg, bg.r=label.bg.r, point.size=NA, max.overlaps=100, min.segment.length=0, force=0)` — `force=0` means no repulsion, it's purely a halo-text renderer.

**Background stripes exist only when `position != "stack"`** (`bg_layer <- NULL` under stack) and use exactly the same rect recipe as §1.7, with `bg_map` computed the same way including the "group.by must be a part of bg.by" nesting check.

`flip=TRUE` reverses group levels, inverts `aspect.ratio`, adds `coord_flip()` for everything except `pie`/`rose` (for those it rotates the polar start to `pi/2` instead).

### 2.5 `stat_level` (venn/upset only)

```r
if (is.null(stat_level)) stat_level <- first level of each stat.by column   # with a message
if (length(stat_level) == 1) stat_level <- rep(stat_level, length(stat.by))
for (i in stat.by) meta.data[[i]] <- meta.data[[i]] %in% stat_level[[i]]
```
It binarizes each `stat.by` column into set membership. It may be a named list (`list(CellType = c("Ductal","Ngn3 low EP"), Phase = "S")`) or a single scalar broadcast to all columns (`stat_level = "TRUE"` for logical columns). It has **no effect on any other plot type**.

### 2.6 Fill scale

```r
colors <- palette_scp(dat_all[[stat.by]], palette, palcolor, NA_color = NA_color, NA_keep = TRUE)
colors_use <- colors[names(colors) %in% dat_split[[sp]][[stat.by]]]      # per-panel subset
if (any NA present && NA_stat) colors_use <- c(colors_use, colors["NA"])
scale_fill_manual(name = paste0(stat.by, ":"), values = colors_use,
                  na.value = colors_use["NA"], drop = FALSE,
                  limits = names(colors_use), na.translate = TRUE)
```
Sankey/chord recompute `colors` over the union of all `stat.by` levels plus NA so node colors are shared across columns. Sankey builds one legend **per `stat.by` variable** by rendering a throwaway `geom_col` plot per variable and harvesting `get_legend`, then `cbind`ing (vertical) or `rbind`ing (horizontal) the legend grobs and `add_grob`-ing them onto the sankey gtable — the final object is `wrap_plots(gtable)`.

Chord under `combine=TRUE` is special-cased entirely: it opens a temporary `png()` device with `dev.control("enable")`, sets `par(mfrow = c(nrow, ncol))` (auto `nrow = ceiling(sqrt(nlev))`), draws each split's chord diagram into the grid, and returns a single `recordPlot()`. In Python this maps to a plain matplotlib figure with subplots (e.g. a chord/`holoviews`/custom bezier implementation), not to the ggplot-analog pipeline.

---

## 3. `FeatureCorPlot` (5232)

A full pairwise SPLOM (scatter-plot matrix) over `features`.

**Parameters**: `srt`, `features` (≥2; genes and/or numeric meta columns), `group.by=NULL`, `split.by=NULL`, `cells=NULL`, `slot="data"`, `assay=NULL` (all Seurat-specific except `features` — see `docs/02_data_contract.md`), `cor_method="pearson"`, `adjust=1` (violin bandwidth adjust), `margin=1` (panel margin in pt), `reverse=FALSE`, `add_equation=FALSE`, `add_r2=TRUE`, `add_pvalue=TRUE`, `add_smooth=TRUE`, `palette="Paired"`, `palcolor=NULL`, `cor_palette="RdBu"`, `cor_palcolor=NULL`, `cor_range=c(-1,1)`, `pt.size=NULL`, `pt.alpha=1`, `cells.highlight`/`cols.highlight="black"`/`sizes.highlight=1`/`alpha.highlight=1`/`stroke.highlight=0.5`, `calculate_coexp=FALSE`, `raster=NULL`, `raster.dpi=c(512,512)`, `aspect.ratio=1`, `title`, `subtitle`, `legend.position="right"`, `legend.direction="vertical"`, `theme_use`, `theme_args`, `combine=TRUE`, `nrow`, `ncol`, `byrow`, `force=FALSE` (guard at >10 features → >50 panels), `seed=11`.

**Correlation**: computed with `proxyC::simil` on the feature × cell matrix, **not** `stats::cor`:
```r
feature_mat <- t(dat_exp[rownames(dat), features])
if (cor_method == "spearman") feature_mat <- t(apply(feature_mat, 1, rank))   # rank then Pearson
cor_method <- "correlation"
pair_sim <- proxyC::simil(x = feature_mat, method = cor_method)
```
So Spearman = Pearson on row-ranked values. Correlations are computed **per split level**, over all cells in that split (no per-`group.by` correlation).

**Panel layout** — `mapply` over `pair_expand = expand.grid(features, features)` (column-major), producing `length(features)^2` panels. `f1_index`/`f2_index` are the factor codes:
- **Diagonal (`f1_index == f2_index`)**: `geom_violin(aes(x = group.by, y = feature, fill = group.by), scale="width", adjust=adjust, trim=TRUE)` — the distribution of that feature per group.
- **Upper (`f1_index < f2_index`)**: scatter of `f1` vs `f2` colored by `group.by` (`scattermore::geom_scattermore` when `raster`, auto-enabled if `n_cells * choose(n_features,2) > 1e5`). Plus `geom_smooth(method="lm", formula = y ~ x, color="red", alpha=0.5)` when `add_smooth`. Annotation via `lm(dat[[f2]] ~ dat[[f1]])`:
  - `eq1`: `italic(y) == a + b %.% italic(x)` from `coef(m)` (sign-aware, emits `a - b·x` when slope < 0)
  - `eq2`: `italic(r)^2 == summary(m)$r.squared`
  - `eq3`: `italic(p) == summary(m)$coefficients[2, 4]` (slope p-value)
  All formatted `digits = 2`, drawn as `annotate(geom = GeomTextRepel, x = -Inf, y = Inf, ..., hjust = -0.05, vjust = c(1.3, 2.6, 5.2)[selected], parse = TRUE)` — top-left corner, one line per enabled annotation.
  `cells.highlight` overplots a halo point (`size = sizes.highlight + stroke.highlight`, `cols.highlight`) then a group-colored point on top.
- **Lower (`f1_index > f2_index`)**: a solid color block encoding the correlation — `annotate("rect", xmin/xmax/ymin/ymax = ±Inf, fill = <binned color>)` plus a centered bold repel label `"f1\nf2\nCor: <round(r,3)>"`. The fill is looked up by binning `r` into the 200-color `cor_colors` ramp:
```r
cor_colors <- palette_scp(seq(cor_range[1], cor_range[2], length.out = 200), palette = cor_palette, palcolor = cor_palcolor)
df_bound  <- interval bounds parsed out of names(cor_colors); df_bound[1,1] <- df_bound[1,1] - 0.01
fill <- rownames(df_bound)[df_bound[,1] < r & df_bound[,2] >= r]
```

**Axis decoration**: axes are blanked globally; ticks/text are restored only on the first column (`f1_index == 1`) and last row (`f2_index == length(features)`); feature names become the outer axis titles. `reverse=TRUE` reverses both orders and moves the scales to the `top`/`right` positions.

**Assembly**: panels → `as_grob` → `cbind` row-wise in chunks of `length(features)` → `rbind` → one gtable. Two legends are built from throwaway plots and merged (`cbind` if vertical, `rbind` if horizontal): the correlation colorbar (`limits = cor_range`, `n.breaks = 3`) and the `group.by` discrete legend. Split name, subtitle and title are attached as `textGrob`s on top. Final `wrap_plots(gtable)` per split; then the standard `combine` logic.

---

## 4. `CellDensityPlot` (5679)

A **ridgeline** plot — `ggridges::geom_density_ridges()`, i.e. per-group Gaussian KDE with ggridges' default bandwidth (`stats::bw.nrd0`) and default `scale`/overlap. Not `stats::density` called directly, not a violin.

**Parameters**: `srt`, `features`, `group.by=NULL`, `split.by=NULL`, `assay=NULL`, `slot="data"` (Seurat-specific: `srt`, `assay`, `slot` — see `docs/02_data_contract.md`), `flip=FALSE`, `reverse=FALSE`, `x_order=c("value","rank")`, `decreasing=NULL`, `palette="Paired"`, `palcolor=NULL`, `cells=NULL`, `keep_empty=FALSE`, `y.nbreaks=4`, `y.min=NULL`, `y.max=NULL`, `same.y.lims=FALSE`, `aspect.ratio`, `title`, `subtitle`, `legend.position="right"`, `legend.direction="vertical"`, `theme_use`, `theme_args`, `combine=TRUE`, `nrow`, `ncol`, `byrow=TRUE`, `force=FALSE` (guard at >50 features).

**Loop structure**: triple nested `for (f in features) for (g in group.by) for (s in levels(split.by))` → `plist[["<f>:<g>:<s>"]]`.

**Per panel**:
1. Infinities are winsorized to the finite extremes: the max value is replaced by the max finite value, likewise the min. NA rows dropped.
2. `x_order`: `"value"` → `value = dat[, f]`; `"rank"` → `value = rank(dat[, f])`. Rank mode is what makes pseudotime ridges evenly spread.
3. Limits: `y_max_use <- y.max %||% max(finite value)`, `y_min_use <- y.min %||% min(...)`; `same.y.lims` precomputes global `y.max`/`y.min` over all features before the loop.
4. **Ordering** (`decreasing`):
```r
if (!is.null(decreasing)) {
  levels <- dat %>% group_by_at(g) %>% summarise_at(.funs = median, .vars = f, na.rm = TRUE) %>%
            arrange_at(.vars = f, .funs = if (decreasing) desc else list()) %>% pull(g) %>% as.character()
  dat[["order"]] <- factor(dat[[g]], levels = levels)
} else {
  dat[["order"]] <- factor(dat[[g]], levels = rev(levels(dat[[g]])))   # default: reversed so level 1 is on top
}
if (flip) { dat[["order"]] <- factor(dat[[g]], levels = levels(dat[[g]])); aspect.ratio <- 1/aspect.ratio }
```
`decreasing=TRUE` sorts groups by **descending median of `f`**, `FALSE` ascending, `NULL` keeps factor order but reversed. `flip` overrides ordering back to natural order and then `coord_flip()`s.
5. Mapping: `aes(x = value, y = order, fill = group.by)` + `geom_density_ridges()`. Note the fill is `group.by` while the y is `order` — same variable, different level ordering, so the palette stays keyed to the original level order.
6. Scales: `scale_y_discrete(drop = !keep_empty, expand = c(0,0))`; `scale_x_continuous(limits, trans = y.trans, n.breaks = y.nbreaks, expand = c(0,0))` where
```r
y.trans <- ifelse(flip, "reverse", "identity")
y.trans <- ifelse(reverse, setdiff(c("reverse","identity"), y.trans), y.trans)
limits  <- if (y.trans == "reverse") c(y_max_use, y_min_use) else c(y_min_use, y_max_use)
```
i.e. `flip` and `reverse` XOR into the axis direction.
7. `facet_grid(. ~ split.by)` only when `split.by` is real. Labels: `x = f`, `y = g`.

---

## 5. `VolcanoPlot` (7088)

**Input contract.** Despite taking `srt`, it reads a DE **result table**, not expression data:
```r
slot   <- paste0("DEtest_", group_by %||% "custom")
index  <- grep(paste0("AllMarkers_", test.use), names(srt@tools[[slot]]))[1]
de_df  <- srt@tools[[slot]][[ names(...)[index] ]]
```
The table is produced by `RunDEtest` (`SCP-analysis.R` ~1914–1946) and has these columns — **this is the exact contract to require of a Python DataFrame**:

| column | type | meaning |
|---|---|---|
| `gene` | chr | feature name (used for labels) |
| `group1` | **factor** | the tested group; drives per-panel iteration and `facet_wrap(~group1)` |
| `group2` | chr | reference (`"others"` for AllMarkers) |
| `p_val` | num | raw p-value from `FindMarkers` |
| `avg_log2FC` | num | log2 fold change |
| `pct.1`, `pct.2` | num | detection fraction in group1 / group2 |
| `p_val_adj` | num | `p.adjust(p_val, method = p.adjust.method)` |
| `test_group_number`, `test_group` | int / chr | bookkeeping, unused by VolcanoPlot |

Derived in-function: `diff_pct = pct.1 - pct.2`, `-log10padj`, `DE`, `border`, `x`, `y`, `distance`, `label`.

**Parameters**: `srt`, `group_by=NULL`, `test.use="wilcox"` (all three Seurat/`RunDEtest`-specific — they are only a lookup key; see `docs/02_data_contract.md`), `DE_threshold = "avg_log2FC > 0 & p_val_adj < 0.05"`, `x_metric = "diff_pct"` (or `"avg_log2FC"`), `palette="RdBu"`, `palcolor=NULL`, `pt.size=1`, `pt.alpha=1`, `cols.highlight="black"`, `sizes.highlight=1`, `alpha.highlight=1`, `stroke.highlight=0.5`, `nlabel=5`, `features_label=NULL`, `label.fg="black"`, `label.bg="white"`, `label.bg.r=0.1`, `label.size=4`, `aspect.ratio=NULL`, `xlab=x_metric`, `ylab="-log10(p-adjust)"`, `theme_use`, `theme_args`, `combine=TRUE`, `nrow`, `ncol`, `byrow=TRUE`.

**Thresholding**: `DE_threshold` is an **R expression string** evaluated against the data frame:
```r
de_df[with(de_df, eval(rlang::parse_expr(DE_threshold))), "DE"] <- TRUE
```
Python equivalent: `df.query(...)` with a pandas-syntax string.

**x clipping / "border" points**:
```r
x_upper <- quantile(finite avg_log2FC, c(0.99, 1)); x_upper <- if (x_upper[1] > 0) x_upper[1] else x_upper[2]
x_lower <- quantile(finite avg_log2FC, c(0.01, 0)); x_lower <- if (x_lower[1] < 0) x_lower[1] else x_lower[2]
if (x_upper > 0 & x_lower < 0) { v <- min(abs(c(x_upper, x_lower))); x_upper <- v; x_lower <- -v }  # symmetrize
border <- avg_log2FC outside [x_lower, x_upper]; those values are clamped to the bound
```
Clamped points are later drawn with `position_jitter(width = 0.2, height = 0.2, seed = 11)` so the pile-up at the clip boundary fans out visually.

**The two-sided y trick** — this plot is *not* a standard volcano. y is signed:
```r
y <- -log10(p_val_adj)
if (x_metric == "diff_pct")   { x <- diff_pct;   y[avg_log2FC < 0] <- -y[avg_log2FC < 0]; sort rows by abs(avg_log2FC) asc }
if (x_metric == "avg_log2FC") { x <- avg_log2FC; y[diff_pct  < 0] <- -y[diff_pct  < 0];  sort rows by abs(diff_pct)  asc }
distance <- x^2 + y^2
```
Down-regulated genes hang below `y = 0`; `scale_y_continuous(labels = abs)` relabels so the axis reads as magnitude in both directions. A solid `geom_hline(0)` and a dashed grey `geom_vline(0)` are drawn. Row sorting ascending by the *other* metric means the most extreme points are plotted last (on top).

**Layer order** — six `geom_point` layers, split on `DE × border`, so highlighted points get a halo:
non-DE/non-border → non-DE/border (jittered) → DE/non-border halo (`cols.highlight`, `size = sizes.highlight + stroke.highlight`) → DE/border halo (jittered) → DE/non-border colored → DE/border colored (jittered).

**Labeling**: top `nlabel` by `distance` **separately for the y ≥ 0 and y < 0 halves**:
```r
df[df$y >= 0, ][head(order(df[df$y >= 0, "distance"], decreasing = TRUE), nlabel), "label"] <- TRUE
df[df$y <  0, ][head(order(df[df$y <  0, "distance"], decreasing = TRUE), nlabel), "label"] <- TRUE
```
overridden entirely by `features_label` (label those genes by `gene`). Repel params: `min.segment.length = 0`, `max.overlaps = 100`, `segment.colour = "grey40"`, `force = 20`, `nudge_x = ±diff(range(x)) * 0.05` (negative for the upper half, positive for the lower).

**Color**: points are colored by the *other* metric — `color_by <- ifelse(x_metric == "diff_pct", "avg_log2FC", "diff_pct")` — with
`scale_color_gradientn(colors = palette_scp(palette = palette, palcolor = palcolor), values = rescale(unique(c(min(c(col,0)), 0, max(col)))))`, i.e. a **zero-anchored diverging ramp**. Legend name is the other metric's label.

**Faceting**: one ggplot per level of `group1`, each with `facet_wrap(~group1)` so the panel carries a strip title; then combined with `wrap_plots`.

---

## 6. Return types — summary table

| function | `combine=TRUE` | `combine=FALSE` | list keys |
|---|---|---|---|
| `ExpressionStatPlot` | n/a (always returns list) | list of ggplot | `"<feature>:<group.by>:<group levels,>:<split levels,>"` |
| `FeatureStatPlot` | ggplot (1 panel) / patchwork; **patchwork-wrapped gtable** when `stack=TRUE` & >1 feature | named list | as above, or one entry per `group.by` under stack |
| `StatPlot` / `CellStatPlot` | ggplot or patchwork; **`recordPlot()` (base graphics)** for `chord`; patchwork-wrapped gtable for `sankey` | named list keyed by split level (or the comb key for the cartesian types) | |
| `FeatureCorPlot` | patchwork-wrapped gtable (one per split), combined with `wrap_plots` | named list keyed by split level | |
| `CellDensityPlot` | ggplot or patchwork | named list | `"<feature>:<group.by>:<split level>"` |
| `VolcanoPlot` | ggplot or patchwork | named list | `group1` level |

`combine` logic is literally identical in all five:
```r
if (isTRUE(combine)) { if (length(plist) > 1) wrap_plots(plotlist = plist, nrow = nrow, ncol = ncol, byrow = byrow) else plist[[1]] } else plist
```
`nrow`/`ncol` are patchwork's grid controls; if both NULL patchwork picks a near-square. Only `StatPlot`'s chord branch reinterprets them (`par(mfrow = c(nrow, ncol))`, defaulting to `nrow = ceiling(sqrt(n_splits))`).

**Python mapping recommendation**: make every engine return `list[Axes]` or a `(fig, axes)` pair with the same key strings, and implement one shared `combine(plots, nrow, ncol, byrow)` that lays them into a `GridSpec`. The "stacked violin" and `FeatureCorPlot` cases need a real shared-figure implementation (single figure, subplot grid, one figure-level legend and one shared axis label) rather than naive per-axes composition, because R achieves them through gtable surgery, not through patchwork.

## 7. Portability notes / gotchas

- **Factor level order = first appearance**, everywhere. Sorting will silently change colors.
- **Seurat-only args** to drop or re-map: `srt`, `assay`, `slot`, `cells`, `cells.highlight`, and `VolcanoPlot`'s `group_by`+`test.use` which only address `srt@tools[["DEtest_<group_by>"]][["AllMarkers_<test.use>"]]` — in Python just accept the DataFrame directly. For the AnnData equivalents of all of these see `docs/02_data_contract.md`.
- `set.seed(seed)` at engine entry, but jitter layers hardcode `seed = 11`.
- `y.max`/`y.min` accept `"qNN"` strings parsed as `as.numeric(sub("(^q)(\\d+)", "\\2", y.max))/100` then `quantile(...)`.
- Limits are applied by **coordinate clipping** (`coord_cartesian`), never by filtering data — a Python port must use `set_ylim`, not a data mask, or box/violin statistics will differ.
- `keep_empty` maps to `scale_x_discrete(drop = !keep_empty)` — whether empty categorical levels reserve an x slot.
- Two live bugs not to port: the `values`/`valus` typo in `ExpressionStatPlot`'s `same.y.lims` quantile branch (3946–3958), and `FeatureCorPlot`'s `calculate_coexp` branch referencing an undefined `status` variable (5292).
