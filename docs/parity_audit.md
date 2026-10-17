# R parity audit: trajectory, enrichment, 3-D and correlation

These four families were implemented without ever being compared numerically
with SCP (Hao Zhang, NJMU, GPL-3, <https://github.com/zhanghao-njmu/SCP>).
This document records that comparison. Each function and its main options were
run on the same input in R (SCP 0.5.6, R 4.5.3, `pancreas_sub`) and in Python
(`src/` at commit `2604bc6`). Where the comparison could not be numeric, that
is stated, with the reason.

The method is the one in `docs/06_testing.md`: compare **the data the plot is
about to draw**, not pictures. On the R side that means `ggplot_build(p)$data`,
`ht$matrix`, `plotly_build(p)$x$data` and the like. On the Python side it means
the returned figure's artists, or the package's own transform functions.

**Reproducing.** Each family has an R script that writes its fixtures, and a
test module that reads them. The tests need neither R nor the `.h5ad`, because
every fixture directory carries its own inputs.

| Family | R script | Fixtures | Tests |
|---|---|---|---|
| trajectory | `tools/parity/audit_traj.R` | `tests/fixtures/parity/audit/traj/` | `tests/test_parity_traj.py` |
| enrichment | `tools/parity/audit_enrich.R` | `tests/fixtures/parity/audit/enrich/` | `tests/test_parity_enrich.py` |
| 3-D | `tools/parity/audit_dim3d.R` | `tests/fixtures/parity/audit/dim3d/` | `tests/test_parity_dim3d.py` |
| correlation | `tools/parity/audit_cor.R` | `tests/fixtures/parity/audit/cor/` | `tests/test_parity_cor.py` |

```bash
Rscript tools/parity/audit_traj.R       # and audit_enrich.R, audit_dim3d.R, audit_cor.R
pytest -q tests/test_parity_*.py
```

**How to read the tests.** Where Python agrees with R, the assertion is held to
the fixture's precision. Where it disagrees, the test is marked
`xfail(strict=True)` with a one-line diagnosis, so the suite stays green, and
fixing the code turns the xfail into an XPASS, which fails the run until the
marker is removed. Each xfail was checked with `--runxfail` to confirm that it
fails for the reason it states, and not because of a bug in the test.

**Environment notes.** These affect anyone rerunning the scripts:

* SCP 0.5.6 still calls `GetAssayData(slot = ...)` inside
  `RunDynamicFeatures`, `RunDEtest` and `check_DataType`. SeuratObject 5 makes
  that an error rather than a warning, so SCP's own analysis pipeline cannot
  run as shipped against SeuratObject 5. The scripts work around it in the
  open: a `slot=`→`layer=` shim in SCP's import environment, and
  `Seurat::FindAllMarkers` in place of `RunDEtest`.
* `RunSlingshot` needed `DelayedMatrixStats`. That is the only package
  installed into the `scp_parity` env, from Bioconductor.
* `org.Mm.eg.db` and `msigdbr` are not installed, so the enrichment gene sets
  are built from the data and passed through `TERM2GENE`.
* `RunPAGA` and `RunSCVELO` route through reticulate to scanpy and scvelo. The
  PAGA input is therefore built in R, and the velocity field is synthetic.
* scvelo and pygam are not installed in the Python env. Velocity `grid` and
  `stream`, and the fitted curve of `dynamic_plot`, are therefore not
  executable on the Python side.

## Summary

| Function | Agrees with R | Differs (count, strict xfail) | Not checkable |
|---|---|---|---|
| `lineage_plot` | LOESS curves to 5e-8, all options tried | 4, all fixed: trim strictness, whiskers, extra cell layer, legend/arrow | — |
| `paga_plot` / `graph_plot` | node medians to 1e-8, colours, edge selection and widths, net-transition logic, `segementsDf` to 1e-9 | 5, all fixed: `global_size`, transition widths, default node size, size mapping, legend | — |
| `velocity_plot` | raw arrows to 5e-8 (both scales); grid to 5e-8 after the numpy port | 2, both fixed: `group_by`, extra cell layer; the 5 grid/stream findings fixed by porting `compute_velocity_on_grid` | stream drawing (no metR); its grid is checked |
| `dynamic_plot` | point x positions | 3, all fixed: no lib-normalisation, `group_by`, `compare_lineages` | the fitted curve (mgcv; documented) |
| `dynamic_heatmap` | the package's `matrix_process` reproduces R's matrix to 4e-7 | 3: binning, row order, palette | — |
| `projection_plot` | reference and query colours/positions, exact | 1 deliberate (R drops out-of-range query cells, an R bug) | — |
| `enrichment_plot` | term selection; bar metric to 1e-12; dot/lollipop fill within 2/255; labels, italics, encodings, axis limits, sizes, ties and constant panels since the parity fix | 13 at audit time; all fixed (4 by the enrichment port, 9 by the parity fix) | network, enrichmap, wordcloud: ported since, verified separately |
| `gsea_plot` | line: curve and hit ticks, exact; selection, bar order/fill/alpha, zero cross, peak, heat strip (to one 8-bit step) and per-key layout since the parity fix | 4 at audit time; all fixed | comparison and graph types: ported since, verified separately |
| `gsea_scores` | running score to 1e-12; hits and peak exact | — | — |
| `cell_dim_plot_3d` | traces, level order, colours, membership, xyz, exact | 3, all fixed: NA handling, NA colour, defaults/layout | highlight, lineages (not ported) |
| `feature_dim_plot_3d` | colour values to 1e-12 | 1, fixed: `bg_cutoff` default masks 76-80% | split dropdown (not ported) |
| `cell_cor_heatmap` | similarity to 1.6e-9 (cosine, pearson, spearman; self and panc8 reference), collapsing, limits, labelled cells | 5: `reference=None`, uncollapsed row order, cluster leaf order, label defaults, default features | annotations, `bulk_ref` (not ported) |
| `feature_cor_plot` | placement, scatter, r to 5e-4, lm fit to 1e-9, r²/p | 4: violin bandwidth, `split_by`, stats tied to `add_smooth`, docstring | `calculate_coexp` (broken in R) |
| `feature_cor_heatmap` | — | — | R's `FeatureCorHeatmap` is an empty stub |

**None of the mismatches is a documented deliberate divergence.** Several
contradict their own docstrings. The documented divergences that were found
are the 3-D Spectral scale, the scvelo grid padding and the mgcv fit, and the
tests assert those as intended behaviour rather than xfailing them.

**R bugs found**, none of them reproduced in Python:

* `ProjectionPlot` drops every query cell outside the reference range
  (SCP-plot.R:12546, together with `CellDimPlot`'s limits at 1636).
* `PAGAPlot(cells=)` intersects group names with cell names (6188).
* `VelocityPlot` adds `lab_layer` twice (6941-6942; already known).
* `!is.factor(enrichment["Groups"])` is always TRUE (12773, 13621).
* "Negtively" is misspelled (13948).
* `CellCorHeatmap` cannot run under Seurat 5 for three reasons: a group-named
  metadata write, `GetAssayData(slot=)`, and `_`→`-` renaming
  (SCP-cell_annotation.R:473, 271/305; SCP-plot.R:10244).
* SCP's `GetAssayData(slot=)` calls are defunct under SeuratObject 5.

**Cross-cutting fixes worth doing once:**

* a ComplexHeatmap-style dendrogram reorder helper. It is verified against 8
  R orders and is needed by every heatmap that clusters.
* an R `bw.nrd0` bandwidth for every violin (`_stat.py:270`, `:844`).
* `LegendColumn` registration in `graph_plot`, `lineage_plot` and
  `velocity_plot`. None of them has a legend. *Done:* all three register
  through `legend=` (or their own `PanelGrid` without `ax=`).
* library-size normalisation in `dynamic_plot` and `dynamic_heatmap`.
  *Done in `dynamic_plot`*; `dynamic_heatmap` is still open.
* a data-contract note: AnnData drops unused categories when an object is
  subset and Seurat does not, which shifts palette indices (see
  `projection_plot` below).

## Trajectory family

Fixtures: `tools/parity/audit_traj.R` → `tests/fixtures/parity/audit/traj/`
(about 1 MB), plus `tools/parity/velocity_grid.R` for two extra
`compute_velocity_on_grid` cases; tests: `tests/test_parity_traj.py`.

*At audit time:* 28 pass, 21 strict xfail, 1 skipped because scvelo was not
installed. *Since the trajectory fixes:* 49 pass and 3 strict xfail; the three
left are `dynamic_heatmap`'s, which lives with the heatmap engine. The
fixture's `r_adata` folds the counts of the genes it does not ship into one
`_rest` column, so `colSums(counts)` is R's library size.

**How the R side was built.** `pancreas_sub` with `RunSlingshot(group.by =
"SubCellType", reduction = "UMAP")`, exactly as SCP's examples call it, giving
`Lineage1..3`. The fixture carries the UMAP and first two PCs, both factors with
their R level order, the three pseudotimes, the library sizes and the raw counts
of seven genes, so the Python side rebuilds the same object from CSV. Several
SCP steps could not run unmodified here, and the script works around each one
in the open:

* slingshot needed `DelayedMatrixStats`. That is the only package installed
  into `scp_parity` (from Bioconductor).
* `RunDynamicFeatures` calls `GetAssayData(slot = ...)`, which SeuratObject 5
  makes defunct. The script maps `slot=` onto `layer=` in SCP's import
  environment. Nothing else changes.
* `VelocityPlot` calls `check_R("metR")` unconditionally, which tries to
  install metR and sf. Only `plot_type = "stream"` uses metR, so raw and grid
  run through a copy with that guard removed. Stream is not checked on either
  side.
* `RunPAGA` and `RunSCVELO` go through reticulate to scanpy and scvelo. Rather
  than run them, the script builds a PAGA-shaped input in R: group
  connectivities from the SNN graph, a maximum spanning tree stored
  one-directionally (as scanpy stores `connectivities_tree`), and an
  asymmetric transition matrix oriented by median pseudotime. Both sides read
  the same matrices, so where they came from does not matter for a plotting
  parity check.
* The velocity embedding is synthetic: a rotational field stored as the
  `stochastic_UMAP` reduction on the R side and as `obsm["velocity_umap"]` on
  the Python side.
* `ProjectionPlot` returns a patchwork, which cannot be built. The script
  captures the composite ggplot (`p3`) by stopping the function just before
  that step.

### lineage_plot (`LineagePlot`, SCP-plot.R:5918)

| Option / aspect | Status |
|---|---|
| degree-2 LOESS curve, all three lineages (default `span`, `trim`) | **matches to 5e-8** (fixture precision); same point count, same Dark2 colours |
| `span = 0.3, trim = (0.05, 0.95)` | **matches to 5e-8** |
| trim on tied pseudotime | fixed; matches (760 cells, strict `>`/`<`) |
| `whiskers = TRUE` | fixed; matches (one segment per fitted cell) |
| layer content | fixed; matches (curves only) |
| legend, arrowhead | fixed; matches (`Lineages` legend labels, arrowhead per path) |
| casing width, aspect, axis key | casing and aspect fixed (`linewidth + line_bg_stroke` mm, square panel); axis key is the known `UMAP_`/`umap_` divergence |

1. **Trim is inclusive.** `_traj.py:337` keeps `pt >= lo` and `pt <= hi`. R
   keeps `>` and `<` (`SCP-plot.R:5959`). The two agree on continuous
   slingshot pseudotime because the quantiles interpolate between data values.
   They part as soon as a quantile lands on a value: with
   `floor(Lineage1)`, R drops the cells at the minimum and the maximum, and
   Python keeps them. *Fix:* `m = m & (pt > lo) & (pt < hi)`. Real bug.
2. **`whiskers` is a no-op.** The parameter at `_traj.py:302` is never read.
   R draws, before the casing, one segment per fitted cell from its fitted
   point to its raw embedding position, coloured by lineage, with
   `whiskers_linewidth = 0.5` and `whiskers_alpha = 0.5` (`SCP-plot.R:5984-5992`).
   *Fix:* after the fit, add
   `LineCollection(np.stack([np.c_[sx, sy], np.c_[px, py]], 1), colors=colour,
   linewidths=whiskers_linewidth, alpha=whiskers_alpha, zorder=2)`. Also expose
   `whiskers_linewidth` and `whiskers_alpha`. Real bug.
3. **Extra cell layer.** `_traj.py:327` scatters every cell in `#D9D9D9`.
   `LineagePlot` draws curves only. It is meant to go on top of a
   `CellDimPlot`, which is how `cell_dim_plot(lineages=...)` uses it. *Fix:*
   drop the scatter, or gate it behind an opt-in `show_cells=False`. Real
   divergence. It doubles the cells whenever the curve is laid over a
   `cell_dim_plot` axes.
4. **No legend and no arrow.** R maps colour to `Lineages` through
   `scale_color_manual`, which produces a legend (`SCP-plot.R:6008`).
   Every path also ends in `arrow(length = unit(0.1, "inches"))` (`5920`,
   `5997`). Python sets `label=` on the line (`_traj.py:363`) but registers
   nothing with `LegendColumn`, and draws a plain line. *Fix:* register one
   handle per lineage with `layout.LegendColumn`, and finish each path with a
   `FancyArrowPatch(arrowstyle="-|>", mutation_scale=...)` over its last two
   points. Expose `lineages_arrow=True`. Real bug.
5. *Cosmetic, not tested.* The casing is `linewidth + 2*line_bg_stroke`
   (`_traj.py:362`); R uses `linewidth + line_bg_stroke` (`5997`). R also sets
   `aspect.ratio = 1`, and the axis key comes from Seurat's `umap_` rather
   than the regenerated `UMAP_`, which is the known divergence noted in the
   notebook.

**Fixed since the audit** (items 1-4, and the casing of 5): `lineage_plot` now
draws through `cell_dim_plot`'s own lineage layer (`_dim_layers.fit_lineages`
and `draw_lineages`), so the two produce the same curves, with R's strict
trim, whiskers, arrowheads (`lineages_arrow=0.1` inches) and a `Lineages`
legend registered through `LegendColumn`. It draws no cells, takes `ax=` and
`legend=`, and allocates a square themed panel through `PanelGrid` otherwise.

### paga_plot / graph_plot (`PAGAPlot` 6121, `GraphPlot` 6287, `segementsDf` 6666)

| Option / aspect | Status |
|---|---|
| node position = per-group median of the embedding (UMAP and PCA) | **matches to 1e-8** |
| node colours (Paired over the factor levels) | **identical** |
| edge selection: threshold, upper triangle; tree reads both triangles at threshold 0 | **identical** (13 edges, and 7 for the tree) |
| edge width rescale onto `edge_size = (0.2, 1)` | **identical to 1e-8** |
| edges dashed when transitions are drawn | **matches** |
| net transitions: pair kept, direction, offset direction | **matches** (midpoints and unit vectors to 1e-6) |
| node label text (`1..n`) | **matches** |
| `segementsDf` ↔ `shorten_segments` | **matches to 1e-9** |
| `edge_shorten`, `transition_shorten` (default 0.05), `transition_offset` | fixed; matches to 1e-8 (R's signed `global_size`) |
| transition arrow widths | fixed; matches to 1e-6 (rescaled onto `transition_size`) |
| default node size | fixed; matches (constant 4, outline 4.8) |
| `node_size = "GroupSize"` relative sizes | fixed; matches to 1e-6 (`area_pal(1, 6)` on both layers) |
| legend (including `"1: Ductal"` under `label = TRUE`) | fixed; labels match exactly |

1. **`global_size`.** `_traj.py:124` computes
   `hypot(max|x|, max|y|)`. R computes
   `sqrt(max(x)^2 + max(y)^2)` on the **signed** coordinates
   (`SCP-plot.R:6389`). Every shortened or offset segment is scaled by this
   value. On the synthetic graph, whose coordinates are mostly negative, R's
   value is 3.61 and Python's is 7.81. On pancreas UMAP the gap is about 0.1
   data units per arrow. A passing test shows that `shorten_segments` with R's
   value reproduces R to 5e-9, so the geometry itself is right. *Fix:*
   `global_size = float(np.hypot(np.nanmax(X), np.nanmax(Y))) or 1.0`. R's
   definition is odd (it can collapse when coordinates are negative), but it
   is not on the known-bug list. Either match it, or keep the abs form and
   document it as a deliberate fix. As it stands it is an undocumented
   divergence.
2. **Transition widths.** Every arrow uses `linewidth=edge_size[1]`
   (`_traj.py:169`). R maps net flow onto `transition_size = c(0.2, 1)` with
   `scale_linewidth_continuous` (`SCP-plot.R:6542`), the same linear rescale
   that the edges already get right. *Fix:* add `transition_size=(0.2, 1.0)`.
   Rescale `|net|` over the drawn arrows the way `widths` is computed at
   `_traj.py:137`, and pass each value as that arrow's `linewidth`. Real bug.
   The parameter is missing from the signature too.
3. **Default node size.** `paga_plot` passes `node_size="GroupSize"` to
   `graph_plot` unconditionally (`_traj.py:286`) and ignores its own
   `node_size=4.0`. `PAGAPlot` forwards `node_size` unchanged (`6194`) with a
   default of 4, which gives every node the same size. *Fix:*
   `node_size=node_size`. Real bug.
4. **Size mapping.** `_traj.py:184` uses area `∝ size/max*200` for size
   columns over 20, and area `size**2*8` otherwise. The outline disc is
   `area*1.2` (`_traj.py:186`), which is a diameter ratio of 1.095. R uses
   `scale_size_identity` for a constant: outline diameter 4.8 against 4, a
   ratio of 1.2. For a numeric column R uses `scale_size_continuous`, that is
   `area_pal(range = c(1, 6))`, trained on **both** `node_size` and
   `node_size*1.2` (`6551-6552`). *Fix:* constant → `s = (k*size)**2`, and
   `(k*1.2*size)**2` for the outline. Column → pool `v` and `1.2*v`, take
   `d = 1 + 5*sqrt((v - min)/(max - min))`, `s = (k*d)**2`. Here `k` is the
   ggplot-mm to points factor `72.27/25.4`. Real bug.
5. **No legend.** `graph_plot` never registers its node colours. R builds a
   colour legend titled by `node_group`, with labels
   `paste0(1:n, ": ", levels)` when `label = TRUE` and
   `label_insitu = FALSE` (`6385`, `6564-6573`). *Fix:* register the level
   handles with `LegendColumn`, using the same `"i: name"` relabelling that
   `cell_dim_plot` already has. Real bug.
6. **R bug, not reproduced and correctly so.** `PAGAPlot(cells = ...)`
   intersects the *group names* with the cell names
   (`dat[intersect(rownames(dat), cells), ]`, `SCP-plot.R:6188`). In practice
   that drops every node. It also never subsets `transition`. Python filters
   cells before taking the medians (`_traj.py:250-262`), which is the evident
   intent. Worth one sentence in the docstring.

**Fixed since the audit** (items 1-5; item 6 is now in the docstring).
`global_size` follows R's signed definition rather than a documented
deviation: it is what every SCP figure is drawn with, and it is not a known
bug. Edge widths stay in the units the edge test pins (the mm value as
matplotlib points); transition widths use the same units, so their ratios match
R's. Node sizes are ggplot mm drawn as `(mm * 72.27 / 25.4) ** 2` pt^2, which
moves `cell_dim_plot(paga=...)` nodes from 128 to 129.5 pt^2, a change too small
to see.

### velocity_plot (`VelocityPlot` 6745, `compute_velocity_on_grid` 6963)

| Option / aspect | Status |
|---|---|
| raw arrows: start and end, `scale = 1` and `2` | **match to 5e-8** |
| raw `group_by` | fixed; arrow colours match exactly, with a legend |
| raw layer content | fixed; matches (arrows only) |
| `grid` | fixed; matches R's pinned grid to 5e-8 (positions and ends), and to 3e-11 with `density`, `smooth`, `n_neighbors`, `min_mass` and `scale` all moved (`velocity_grid_opts.csv`) |
| `stream` | the grid is checked: `(2, ns)` lines, `(2, ns, ns)` field and NA mask match R to 5e-13 (`velocity_stream_grid.csv`). The drawing is not checkable (metR is not installable here) |

1. **`group_by`.** `_traj.py:456-463` colours a scatter of the cells, and the
   arrows stay `arrow_color`. R maps `color = group_by` on the **segments**,
   draws no cells (`SCP-plot.R:6801-6812`), and adds a legend. *Fix:* pass the
   per-cell colour vector as `color=` to `ax.quiver`, drop the scatter, and
   register the levels with `LegendColumn`. Real bug.
2. **Raw-mode options missing** (code review, not tested). `density` in
   (0, 1) subsamples the arrows in R (`6783-6788`); Python ignores it in raw
   mode. R's arrowhead length is proportional to each arrow's magnitude
   (`length_perc = length / global_size`, `6796`). Python's quiver heads are
   a fixed multiple of the shaft width. The `cells=` argument is absent.
3. **Grid and stream** (code review only, not executable here). The claims
   about scvelo come from its upstream `compute_velocity_on_grid` source.
   scvelo is not installed in this environment, so confirm them when it is:
   * the `scale` argument is never passed to `compute_velocity_on_grid`
     (`_traj.py:470-473`). R multiplies `V_grid` by `scale` (`7035`);
   * `autoscale=not adjust_for_stream` (`_traj.py:405`) turns on scvelo's
     quiver autoscaling for grid. R does not autoscale;
   * `_traj.py:403` already maps `min_mass → 10**(min_mass-6)` for stream, and
     scvelo applies the same mapping internally in its `adjust_for_stream`
     branch. The floor is then `10**(1e-5 - 6)` instead of `1e-5`;
   * in stream mode scvelo returns `X_grid` as `(2, ns)` and `V_grid` as
     `(2, ns, ns)`. `_traj.py:479` sets `side = int(sqrt(len(gx))) = 1`, so the
     reshape that follows raises. Stream mode cannot have run;
   * `n_neighbors=None` becomes scvelo's `int(n/50)`. R's default is
     `ceiling(n/50)` (`6747`);
   * the ±1% grid padding is the one **documented** divergence (the docstring
     says "match scvelo, not SCP").

   *Fix:* port R's ~60-line `compute_velocity_on_grid` (`6963-7039`) into numpy
   instead of delegating. That removes the scvelo dependency, all four
   divergences above, and the shape bug at once. The pinned
   `velocity_grid_layers.csv` is then the test.

   **Fixed since the audit**: ported as proposed. The ±1% padding note is gone
   with it, since the port follows SCP rather than scvelo. Neighbour ties go to
   the lower index, as R's stable `order`.
4. **R bug.** `VelocityPlot` adds `lab_layer` twice (`6941-6942`). It is
   harmless, and it is already on the known-bug list in `docs/08_agent_handoff.md`.

**Fixed since the audit**: items 1 and 3, and from item 2 the `density`
subsample and `cells=` (numpy's generator, so a different random subset than
R's). Still open from item 2: R's arrowheads scale with each arrow
(`length / global_size` npc); `quiver` heads scale with the shaft, so they are
uniform. The overlay layer `cell_dim_plot(velocity=...)` uses does size them
per arrow, but its artist is not a `Quiver`, which the raw-arrow tests read.
Stream mode now draws through that overlay layer, fed by the port.

### dynamic_plot (`DynamicPlot` 12102)

Exact parity of the fitted curve is out of scope by design (mgcv vs pygam,
documented). pygam is not installed here either, so Python falls back to a
rolling mean. Only the raw points and the layout are compared.

| Option / aspect | Status |
|---|---|
| point x = lineage pseudotime (single lineage) | **matches to 1e-6** |
| point y | fixed; matches to 1e-6 (`log1p(counts / libsize * median(libsize))`) |
| point colour by `group_by` | fixed; matches (Paired over `group_by`) |
| `compare_lineages = TRUE` (the R default) | fixed; matches (2 panels, lineages overlaid, points at `x_assign`) |
| fitted line and ribbon | not checkable (mgcv); a documented divergence |

1. **No library-size normalisation.** `_traj.py:546-554` takes the counts and
   applies `log1p`. With `slot = "counts"`, R defaults to
   `lib_normalize = TRUE`, and the raw matrix becomes
   `counts / libsize * median(libsize)` before `exp_method`
   (`SCP-plot.R:12184-12197`; `libsize = colSums(counts)`, with the
   non-integer guard). A passing test pins that transform against R's points.
   *Fix:* add `lib_normalize: bool | None = None` (default: `layer ==
   "counts"`) and `libsize=`. Apply the same `heatmap.matrix.lib_normalize`
   that `group_heatmap` uses to the gene columns before the transform. Use
   `ylabel = f"{exp_method}(normalized counts)"` to match R's `exp_name`.
   Real bug.
2. **`group_by` ignored.** `_traj.py:557` hard-codes `#BFBFBF`. R colours the
   points (and the rug) with `point_palette = "Paired"` over `group_by`
   (`12106`, `12337-12350`). *Fix:* use the `discrete_palette` over
   `as_ordered_categorical(adata.obs[group_by])` and register a legend. Real
   bug.
3. **`compare_lineages` / `compare_features` ignored.** `_traj.py:535` always
   builds one panel per (lineage, feature). R's default
   `compare_lineages = TRUE` builds one panel per feature, with every lineage
   overlaid and coloured by `line_palette` (`12296-12310`). Points sit at
   `x_assign = rowMeans(pseudotime over all lineages)` (`12170`), not at the
   lineage's own pseudotime. *Fix:* key the panels by feature when
   `compare_lineages` is set (and by lineage when `compare_features` is set),
   draw every lineage's line and ribbon into that panel, and compute
   `x_assign` as the row mean over `lineages` for the points. Real bug.

**Fixed since the audit** (items 1-3), as proposed. One deliberate divergence
is documented in the docstring: when the non-integer guard fires, R sets
`libsize` to 1 but still multiplies by the median column sum; here the guard
makes the normalisation a no-op, as `group_heatmap` does.

### dynamic_heatmap, binned form (`DynamicHeatmap(use_fitted = FALSE)` 11020)

| Option / aspect | Status |
|---|---|
| R's matrix = per-cell, lib-normalised, z-scored jointly across lineages | **reproduced to 4e-7** by the package's own `matrix_process` (a passing test that pins the fix) |
| `dynamic_heatmap` matrix | **differs**: 100 bins per lineage vs 799 + 693 cells |
| feature order | **differs**: Isl1 lands 3rd, R puts it last |
| default palette | **differs**: RdBu; R uses viridis |
| colour limit `b` | equal here (1.0), but by coincidence, since each side computes it from a different matrix |

1. **The binning is not in R.** `_heatmap.py:470-481` averages cells into
   `cell_bins` quantile bins, z-scores each lineage separately, and never
   lib-normalises. R's `use_fitted = FALSE` path involves **no fit**. It keeps
   every cell, ordered by pseudotime (descending for `reverse_ht`). It
   lib-normalises the counts, `cbind`s the lineages, and applies
   `matrix_process` **once** to the concatenation, so the z-score is shared
   across lineages (`SCP-plot.R:11281-11306`). `cell_bins` exists only for the
   density-equalising subsample when `cell_density != 1` (`11184-11194`). The
   README and docstring present the binned form as "the honest version that
   needs no fit". In fact R's own default needs no fit for the matrix, so it
   is reproducible exactly. *Fix:* build `mat_raw` per cell as R does (reuse
   `lib_normalize`). Run `matrix_process` once on `np.hstack` across
   lineages, split the columns back per lineage for the panels, and keep
   `cell_bins` for `cell_density` subsampling only. If a binned view is still
   wanted for very large data, make it an explicit opt-in. Real bug, and the
   docs overstate it as a parity-safe form.
2. **Row order.** `_heatmap.py:482-485` orders by `argmax` of the first
   lineage's binned z-score. R orders by `max` over lineages of each
   feature's `peaktime` from `RunDynamicFeatures`: the median pseudotime of
   cells in the top 1% of the **GAM-fitted** values (`11256-11265`). That one
   step does depend on mgcv. *Fix:* take `peaktime` from a user-supplied
   column or scFates' stored fits when present, otherwise from a documented
   proxy (median pseudotime of the cells above the 99th percentile of a
   rolling-mean smooth). Take the max over lineages, not the first lineage.
   Then document that the row order, and only the row order, can differ from
   R.
3. **Palette.** `_heatmap.py:465` defaults to `RdBu`. R's
   `heatmap_palette = "viridis"` (`11041`). *Fix:* default `palette="viridis"`.
   `docs/04_api_mapping.md` has no entry pinning RdBu here, so this is a slip
   rather than a decision.

### projection_plot (`ProjectionPlot` 12511)

| Option / aspect | Status |
|---|---|
| reference cells: positions and Paired colours over `ref_group` | **identical** (638/638) |
| query cells: Set1 colours over `query_group` | **identical**, once the categories are restored (see note) |
| query cells outside the reference range | **differs on purpose**: R silently drops them (an R bug), Python keeps them |
| point sizes, ggplot's 5% axis expansion | differ, cosmetic (not tested) |

1. **R bug.** `ProjectionPlot` computes the union of both ranges
   (`SCP-plot.R:12522-12531`) but applies it only to the throwaway query plot
   `p2` (`12546`). The composite `p3` is `p1`, the reference `CellDimPlot`,
   and that already carries `scale_x/y_continuous(limits = range(ref))`
   (`1636-1637`). So every query cell beyond the reference range is censored
   to NA and dropped without a warning: 111 of the 362 query cells in this
   fixture. Python applies the union (`_traj.py:648-649`). That is right by
   R's evident intent, and a test pins it. *Suggestion:* say so in the
   docstring, and pad by ggplot's 5% so that edge points are not cut in half.
2. **Data-contract note, not a bug in `projection_plot`.** AnnData drops
   unused categories when an object is subset, and Seurat keeps the factor
   levels. A query built as `adata[mask].copy()` therefore indexes Set1 from
   its first *present* level. Here that is Beta = `#E41A1C`, where R uses Beta
   = level 5 = `#FF7F00`, so every query colour shifts. The tests restore the
   full categories, and then the colours match exactly. This belongs in
   `docs/02_data_contract.md` as a gotcha: subsetting changes palette
   indices. Every `discrete_palette` caller is exposed to it, not only this
   one.
3. *Cosmetic.* R draws the reference at `CellDimPlot`'s auto `pt.size` and
   the query at `pt.size = 0.8` with a 1.3 halo. Python sizes the reference
   from the query's `pt_size` (`_traj.py:634`).

## Enrichment family

Fixtures: `tools/parity/audit_enrich.R` → `tests/fixtures/parity/audit/enrich/`
(588 KB, byte-identical across reruns); tests: `tests/test_parity_enrich.py`
(39 pass, no xfail, after the parity fix; it was 10 pass and 19 strict xfail at
audit time, 14 and 15 after the enrichment port).

**How the R side was built.** `org.Mm.eg.db` and `msigdbr` are not installed,
so no species database is used. Gene sets are built from the data and passed to
`RunEnrichment` / `RunGSEA` through `TERM2GENE`: per CellType a top-25 marker
set, a rank-26..70 band and a "diluted" set (8 markers plus 22 random genes),
and 12 random 30-gene sets (`set.seed(11)`). Descriptions are lower-case and of
different lengths, so `capitalize()` and `str_wrap()` both get exercised.
SCP 0.5.6's `RunDEtest` cannot run on SeuratObject 5: `check_DataType` calls
`GetAssayData(slot=)`, which is now defunct. The marker table therefore comes
from `Seurat::FindAllMarkers(only.pos=FALSE, logfc.threshold=0.25)`, stored at
`srt@tools$DEtest_CellType$AllMarkers_wilcox` with RunDEtest's column names.
`BiocParallel::SerialParam()` keeps fgsea deterministic. `RunEnrichment`
lower-cases the database name (`db="Custom"` is stored as `"custom"`). Two
small hand-built cases make up the rest. `edge_input.csv` has ties, a term at
exactly the cutoff, `topTerm` per group, and group order G2 before G1.
`gsea_edge_input.csv` is the real Ductal GSEA result with p.adjust reassigned
so that p.adjust order and |NES| order disagree. They exist because, on the
real data, several rules can't be told apart: the two orderings happen to agree
there.

No R package was installed.

### enrichment_plot

| Option / aspect | Status | Evidence |
|---|---|---|
| term selection (padjust < 0.05, `top_term` per Database x Group), bar/dot/lollipop | **matches** (same ID sets in all 5 panels) | `test_term_selection_per_group_matches` |
| bar: length = -log10(p.adjust), most significant on top | **matches to 1e-12** | `test_bar_order_and_metric_match` |
| dot/lollipop: fill = -log10(p.adjust) on Spectral over each panel's range | **matches within 2/255** (ggplot interpolates in CIELAB) | `test_point_fill_follows_the_metric_gradient` |
| panel order | **matches** (first appearance) since the network/enrichmap/wordcloud port | `test_panel_order_is_first_appearance` |
| label text: capitalize + str_wrap(50) | **fixed; matches exactly** (every label R drew, enrichment and GSEA) | `test_labels_are_capitalised_and_wrapped_like_stringr`, `test_plotted_labels_are_scp_strwrap_not_stringr` |
| italic for wrapped labels | **fixed; matches exactly** | `test_wrapped_labels_are_italic` |
| bar fill = `color_by` (Database) | **fixed; matches exactly** (`#5E4FA2`) | `test_bar_fill_is_color_by` |
| bar Count labels, value axis = 1.3 x max | **fixed; matches to 1e-9** (axis limit; labels exact) | `test_bar_count_labels_and_axis_limit` |
| dot: x = GeneRatio, rows ordered by GeneRatio | **fixed; matches to 1e-7** (order exact) | `test_dot_x_is_gene_ratio` |
| dot/lollipop size mapping | **fixed; matches to 1e-6** (relative area) | `test_dot_size_mapping` |
| lollipop: x = FoldEnrichment, ordered by it | **fixed; matches to 1e-7** (order exact) | `test_lollipop_x_is_fold_enrichment` |
| ties in the metric | **fixed; matches exactly** | `test_ties_keep_table_order_top_down` |
| strict cutoff `<` | **matches** since the network/enrichmap/wordcloud port | passing test |
| `padjust_cutoff=None` → metric = pvalue | **matches** since the network/enrichmap/wordcloud port | passing test |
| constant-metric panel colour | **fixed; matches within 2/255** (#FEFEBC) | `test_constant_metric_panel_gets_mid_colour` |
| comparison | **matches** since the network/enrichmap/wordcloud port (one Groups x terms matrix) | passing test; frames also pinned in `tests/test_enrich_parity.py` |
| network / enrichmap / wordcloud | ported after this audit; graphs, communities and word tables verified in `tests/test_enrich_parity.py` (GO wordcloud uses a documented substitute) | — |

Not one of the mismatches below is a documented deliberate divergence. The
docstring at `_enrich.py:70-77` actually *promises* the R behaviour for the
cutoff precedence, capitalisation, wrapping, italics and FoldEnrichment. The
code does not deliver it. All are real bugs. Line numbers are against
`src/scp/pl/_enrich.py` at 2604bc6.

**Status: all thirteen are fixed.** Items 1, 9, 10 and 12 landed with the
network/enrichmap/wordcloud port; the rest with the parity fix, which draws
bar/dot/lollipop from `enrichment_panel_frame` (R's factor levels, parsed
ratios) through `_enrich_draw.draw_enrichment_bar` / `draw_enrichment_points`.
Where the fix differs from the proposal below, the item says so.

1. **Panel order.** *Fixed (enrichment port).* `_enrich.py:111-114`: `df.groupby(panels, observed=True)`
   on plain string columns sorts the keys, which gives Ductal, Endocrine,
   Ngn3 high EP. R (`SCP-plot.R:12773-12779`, `12824`) factors Groups and
   Database with `levels = unique(x)` and `split()`s in that order.
   *Fix:* turn each `split_by` column into
   `pd.Categorical(col, categories=pd.unique(col))` (or call
   `fetch.as_ordered_categorical`) before `groupby(..., sort=True, observed=True)`.
2. **Labels.** *Fixed, but not as proposed.* SCP's own `str_wrap`
   (`utils.R:872`) masks stringr's and is base `strwrap`, which keeps every
   line strictly shorter than the width; textwrap and stringr allow a line of
   exactly the width. On 4 of the 27 fixture descriptions the two disagree, and
   every label R drew sides with `strwrap`. The port uses the existing
   `_enrich_graph.r_strwrap`. Original diagnosis: `_enrich.py:130`, `161`: `textwrap.fill(d, character_width)`
   with no capitalisation. textwrap also breaks on hyphens: "pre-\nendocrine"
   where stringr writes "\npre-endocrine". R: `12909-12910` (and 12863-12864,
   12958-12959, 13014-13015). *Fix:* `d = d[:1].upper() + d[1:]`, then
   `textwrap.fill(d, character_width, break_on_hyphens=False)`. With that
   change Python reproduces `str_wrap` exactly on every description in the
   fixture (`test_python_wrap_matches_stringr_once_hyphens_are_kept`).
3. **Italic wrapped labels.** *Fixed.* `_enrich.py:167`: missing. R: `12937`.
   *Fix:* after `set_yticklabels`, call `t.set_fontstyle("italic")` on every
   label whose text contains `"\n"`.
4. **Bar encodings.** *Fixed.* `_enrich.py:135-138`: the fill is the score gradient.
   R (`12913-12926`) has `fill = color_by` through
   `palette_scp(levels(df[[color_by]]), palette)`, Count drawn as text twice
   (a white bold underlay, then black, at `hjust=-0.5`), and
   `ylim = c(0, 1.3*max(metric))`, no legend.
   *Fix:* use `discrete_palette(levels_of(color_by), palette)` per row;
   `ax.text(v, y, f" {count}")` with a white halo; `ax.set_xlim(0, 1.3*max)`.
   `scp.discrete_palette(["custom"], "Spectral")` already returns R's
   `#5E4FA2`; the bar just never calls it, and draws the gradient's
   `#4175B4` instead.
5. **Dot.** *Fixed.* `_enrich.py:140`: `x = _score`, rows sorted by score. R
   (`12950-12976`): `GeneRatio = k/n` parsed from the string; rows ordered by
   `GeneRatio` descending then `rev()`, so the highest is on top; `x = GeneRatio`;
   `xlim = (0, 1.3*max)`; `fill = metric`; `size = Count`. *Fix:* parse
   GeneRatio, `g = g.sort_values("GeneRatio", kind="stable")`, scatter
   `(GeneRatio, y)`.
6. **Lollipop.** *Fixed.* `_enrich.py:143-145`: `x = _score`. R (`13001-13030`):
   `FoldEnrichment = GeneRatio/BgRatio`, rows ordered by FoldEnrichment
   ascending, a black 2-wide segment under a 1-wide segment coloured by metric,
   `size = GeneRatio`, `xlim = (0, 1.2*max)`. *Fix:* compute FoldEnrichment
   (the docstring already says to). Sort by it with a stable sort and draw two
   `hlines` from 0 to FE: black, then the metric colour.
7. **Size mapping** (dot, lollipop, comparison). *Fixed* (comparison by the
   enrichment port). `_enrich.py:140`, `144`,
   `155`: area = `20 + 80*v/max`. R `scale_size(range=c(3,6))` maps
   diameter = `3 + 3*sqrt((v-min)/(max-min))` mm (`12967`, `13031`).
   `scale_size_area(max_size=6)` in comparison maps diameter =
   `6*sqrt(v/max)` mm (`12872`). *Fix:*
   `d = 3 + 3*np.sqrt(rescale(v))`, `s = (d * 72/25.4)**2`, with ggplot's
   point-size factor `.pt = 72.27/25.4`.
8. **Ties.** *Fixed.* `_enrich.py:117`+`128`: `nlargest` keeps table order, then
   `sort_values("_score")` ascending puts the *first* of a tie at the bottom.
   R `order()` is stable and `levels = unique(rev(...))` (`12816`, `12911`)
   puts the first listed on top. *Fix:* order by the raw metric with
   `kind="stable"`, take `head(top_term)`, then reverse for drawing
   (`y = arange(n)[::-1]`).
9. **Cutoff.** *Fixed (enrichment port).* `_enrich.py:96-99`: `<=`, and `pvalue_cutoff` applies even
   when `padjust_cutoff` is set. R `12804-12815`:
   `metric = "pvalue" if padjustCutoff is None else "p.adjust"`, then
   `metric < metric_value` (strict). *Fix:*
   `metric, cut = ("p.adjust", padjust_cutoff) if padjust_cutoff is not None else ("pvalue", pvalue_cutoff)`;
   `df = df[df[metric] < cut]`.
10. **Metric under the p-value path.** *Fixed (enrichment port).* `_enrich.py:109` and `168` hardcode
    `p.adjust`. R `12908`: `-log10(df[[metric]])`, axis label
    `-log10(<metric>)`. *Fix:* use the `metric` from item 9 in both places.
11. **Zero-width colour range.** *Fixed* (the norm is centred on the value). `_enrich.py:134`: `Normalize(v, v)` maps
    everything to 0, the first ramp colour. ggplot's `rescale` maps a zero
    range to 0.5, giving `#FEFEBC` on Spectral. *Fix:*
    `if vmin == vmax: cvec = cmap(np.full(n, 0.5))`.
12. **Comparison.** *Fixed (enrichment port).* `_enrich.py:146-164` draws one matrix per `split_by`
    panel. Under the default `split_by=("Database","Groups")` that is five
    one-column panels. R (`12833-12898`) behaves differently:
    * it collects the per-panel top-term IDs (`ids`) and plots *every* row of
      the unfiltered table with those IDs, non-significant ones included;
    * it draws ONE dot matrix, x = Groups (first appearance) and
      y = `unique(rev(terms[ids]))`;
    * fill is the raw metric on `limits = c(0, min(cutoff, 1))` with
      `palette_scp(..., reverse=TRUE)`, and values beyond the limits go to
      `grey80`, which is the "Non-sig" legend key;
    * size = GeneRatio (`scale_size_area`, max 6);
    * the result is faceted by Database.

    *Fix:* special-case `comparison` before the per-panel loop. Build `ids`
    exactly as R does, then draw a single axes (per Database) with the
    encodings above.
13. **Defaults and options.** *`top_term` and `compare_only_sig` (enrichment
    port) and `color_by` (parity fix) are done; there is still no `db=`, by
    design: the input is a table, so filter it first.* These diverge too, though the fixture does not
    exercise them:
    * `top_term` defaults to `None` (all terms); R defaults to 6 (100 for
      enrichmap);
    * there is no `db=` filter (R's default is `"GO_BP"`) and no
      `compare_only_sig`;
    * `color_by` is accepted but never used.

### gsea_plot

| Option / aspect | Status | Evidence |
|---|---|---|
| line: plotted running score | **matches to 1e-12** (it is `gsea_scores` output) | `test_gsea_line_draws_gseaScores_and_hits` |
| line: hit ticks | **matches exactly** (every hit position) | same |
| `direction="pos"` term set on the real data | **matches** (by coincidence: \|NES\| and p.adjust order agree there) | `test_gsea_pos_selection_agrees_on_real_data` |
| term selection rule (p.adjust < cutoff, ranked by p.adjust) | **fixed; matches exactly** on the discriminating case, both directions | `test_gsea_selection_ranks_by_padjust`, `test_gsea_selection_edge_case_both_directions` |
| `direction="both"` | **fixed; matches exactly** in 5/5 groups | `test_gsea_both_splits_up_and_down` |
| bar order (NES ascending) | **fixed; matches exactly** | `test_gsea_bar_order_is_nes_ascending` |
| bar fill (Spectral Pos/Neg) and alpha = -log10(p.adjust) | **fixed; matches** (fill exact, alpha to 1e-6) | `test_gsea_bar_fill_and_alpha` |
| line: zero-cross dashed line | **fixed; matches exactly** (all 5 keys) | `test_gsea_line_marks_the_zero_cross`, `test_gsea_zero_cross_matches_R_on_every_key` |
| line: peak marker and crosshair | **fixed; matches to 1e-12** | `test_gsea_line_marks_the_peak` |
| line: one panel per (group, db) with all terms overlaid | **fixed**: one 3-row block per key, terms overlaid in R's order, first set's tick band on top, "Term" legend; curves computed from the documented `geneList`/`geneSets` input match R to 1e-12 | `test_gsea_line_draws_gseaScores_and_hits`, `test_gsea_line_is_one_block_per_key_with_a_term_legend`, `test_gsea_line_computes_curves_from_the_documented_input` |
| line: half-plane shading, ranking-metric heat strip | **fixed**: shading as R's `bg_dat`; heat strip rectangles exact, colours within one 8-bit step (`gsea_heat_strip.csv`, replayed from R) | `test_gsea_line_shades_the_half_planes`, `test_gsea_heat_strip_matches_R`, `test_gsea_line_draws_the_heat_strip` |
| `bar` / `line` layout | **fixed**: both go through `PanelGrid` (one panel or block per key) instead of calling `plt.figure` | `test_gsea_bar_panels_per_key_on_one_grid`, `test_gsea_line_and_bar_compose_on_a_panel_grid` |
| comparison / network / enrichmap / wordcloud | ported after this audit; verified in `tests/test_enrich_parity.py` | — |

1. **Selection.** *Fixed:* every type goes through `gsea_select_ids` per key. `_enrich.py:220-228`: `p.adjust <= cutoff`, then sort by
   `|NES|` descending, `head(top_term)`. R (`13743-13756`, `14013-14026`):
   `metric < cutoff`, `order(metric)`, top `topTerm` of NES>0 and of NES<0.
   `direction="both"` takes `head(c(head(up, ceil(k/2)), head(down, ceil(k/2))), k)`.
   *Fix:* port that block literally. Use `kind="stable"` sorts on `metric`,
   and apply the same `metric` switch as enrichment item 9.
2. **Bar.** *Fixed* (`gsea_bar_frame` + `_enrich_draw.draw_gsea_bar`; the
   in-panel text is clipped to the panel, as ggplot's coord clips it). `_enrich.py:236-246`: rows sit in selection order, coloured
   `#B2182B`/`#2166AC`, with no alpha and term names on the y axis. R
   (`14041-14078`):
   * rows are `order(NES)` ascending;
   * fill is `palette_scp(c("Neg","Pos"), palette)`, which on Spectral is
     Neg `#5E4FA2`, Pos `#9E0142`;
   * `alpha = -log10(p.adjust)` through ggplot's default `scale_alpha`
     range (0.1, 1);
   * the description is drawn *inside* the panel at x=0 with `hjust` flipped
     by sign, and axis text is blanked;
   * `xlim = ±max|NES|`, `facet Database ~ Groups`.

   *Fix:* `table.sort_values("NES", kind="stable")`, colours from
   `discrete_palette(["Neg","Pos"], palette)` (verified: it returns R's
   `#5E4FA2` / `#9E0142`),
   `alpha = 0.1 + 0.9*rescale(-log10 p)`, then the text and limits above.
3. **Line, missing decorations.** *Fixed, all four,* plus the core-gene
   labels (nudged as ggrepel nudges them, then spread apart vertically rather
   than repelled). The heat strip is pinned by a new fixture,
   `gsea_heat_strip.csv`, which `audit_enrich.R` now writes by replaying
   `SCP-plot.R:13895-13923`. The docstring (`_enrich.py:199-205`)
   promises these, but the code (`268-291`) draws none:
   * the red/blue half-plane shading (R `13803`);
   * the peak triangle and dashed crosshair at `which.max(abs(runningScore))`
     (R `13840-13858`);
   * the ranking-metric heat strip (R `13896-13931`);
   * the dashed zero-cross line at `median(x[y == y[which.min(abs(y))]])`
     with its label (R `13936-13952`).

   Python's bottom panel is a `fill_between` area. R draws `geom_segment`
   stems in grey30. *Fix:* add each element; the zero-cross is
   `x[np.abs(y) == np.abs(y).min()]`, then the median.
4. **Line, layout and input contract.** *Fixed:* one block per key with the
   terms overlaid; the documented per-key `{"result", "geneList",
   "geneSets", "exponent"}` input is accepted (inside SCP's
   `{"enrichment", "results"}` or bare), and running scores are computed from
   it; `{"table", "curves"}` still works, with `curves` keyed by ID or by key
   then ID. `bar` and `line` now go through `PanelGrid`. R plots one figure per result key
   (`"<Group>-<Database>"`) with every selected term overlaid in p1 and
   stacked tick bands in p2 (`13740-14006`). Python draws one block per term
   and has no notion of groups. On top of that, the docstring describes an
   input of `{f"{group}-{db}": {"result", "geneList", "geneSets", "exponent"}}`
   (`_enrich.py:207-210`), while the code reads
   `{"table": df, "curves": {id: df}}` (`216-217`). *Fix:* accept the
   documented per-key mapping, loop keys, overlay terms. Short of that,
   correct the docstring.

### gsea_scores

| Aspect | Status |
|---|---|
| running score, 30 (group, set) curves, at every hit + peak + every 50th rank | **matches to 1e-12** |
| hit positions | **identical** |
| peak = first `which.max(abs(runningScore))` | **identical** |

`gsea_scores` agrees with `gseaScores` (`SCP-plot.R:14570`). One latent
difference: R computes `Nh` from `intersect(geneSet, names(geneList))`, while
Python counts hits along the list (`_enrich.py:313-314`). The two differ only
if `gene_list` has duplicated names, which a ranked list from `RunGSEA` never
has.

### R-side notes

* `!is.factor(enrichment["Groups"])` (`SCP-plot.R:12773`, `13621`)
  single-brackets a data.frame, so the test is always TRUE. A
  user-supplied factor order on `Groups` is always replaced by first-appearance
  order. Harmless for RunEnrichment output, but it is a bug.
* `"Negtively correlated "` (`13948`) and `corss_x`: a typo, cosmetic. The
  port spells the label correctly.
* SCP's `str_wrap` (`utils.R:872`) is base `strwrap`, not stringr's, so a
  line never reaches `character_width` characters. The audit's proposed
  textwrap fix would have matched stringr and missed four descriptions.
* With a single positive (or negative) ranked value, `cut(..., breaks =
  seq(min, max, len = 100))` stops on non-unique breaks, so `GSEAPlot`'s
  single-set line plot errors. The port colours that value with its ramp's
  first colour.
* The three sub-plots of the line figure train their x scales separately
  (p1 includes the crosshair's x = 0, p2 the heat strip's N + 1), so their
  ticks disagree by up to one rank. The port shares one x range.
* `RunDEtest` is broken on SeuratObject >= 5 (`GetAssayData(slot=)`), so
  SCP's documented enrichment pipeline cannot run in this environment as
  shipped.
* `docs/04_api_mapping.md` gave `gseaScores` at line 14602; it is at 14570
  (corrected).

## 3-D and correlation families

Fixtures: `tools/parity/audit_dim3d.R` → `tests/fixtures/parity/audit/dim3d/`
(228 KB), `tools/parity/audit_cor.R` → `tests/fixtures/parity/audit/cor/`
(~990 KB). Tests: `tests/test_parity_dim3d.py` (5 pass, 6 xfail at audit
time; all 11 pass since the 3-D fixes),
`tests/test_parity_cor.py` (17 pass, 10 xfail).

Inputs: both sides read `pancreas_sub` after `NormalizeData` (its stored
`data` slot holds raw counts). The fixtures carry the log-normalised values, so
the Python side reads exactly the matrix R used. `pancreas_sub` has only a
2-D UMAP, so the 3-D plots use PCA dims 1:3 (`reduction="PCA"` /
`"X_pca"`). The cell-correlation reference is `panc8_sub`, with SCP's README
gene renaming (`make.unique(capitalize(rownames, force_tolower = TRUE))`) and
`NormalizeData`. Both sides are passed an explicit list of 50 query HVGs that
the reference also carries, so feature selection is not part of the matrix
comparison; it is checked on its own below.

### R-side incompatibilities found (SCP 0.5.6, Seurat 5.5.1)

None of these is a Python bug, but each one stops `CellCorHeatmap` running
under a current Seurat. `audit_cor.R` works around them and says so in
comments. All the workarounds come after the similarity is computed, or are
pure renames.

1. **Every collapsed `CellCorHeatmap` call fails** with "No cell overlap
   between new meta data and Seurat object". `RunKNNPredict`
   (SCP-cell_annotation.R:473) writes `match_best[query_index]` into
   meta.data. With `query_collapsing = TRUE` that vector is named by *group*,
   and SeuratObject 5 matches those names against cell barcodes. The script
   wraps the three trailing writes in `unname()`, using `assignInNamespace`.
2. **The uncollapsed path fails** because `GetAssayData(..., slot = "data")`
   (SCP-cell_annotation.R:271, 305) is rejected by the installed
   SeuratObject. The script rewrites it to `layer = "data"`, which reads the
   same matrix.
3. **Any reference group containing `_` fails.** Seurat 5's
   `AverageExpression` rewrites `_` to `-` in group names, so the later
   `simil_matrix[, levels(ref_group)]` (SCP-plot.R:10244) is out of bounds
   for panc8's `activated_stellate` and `quiescent_stellate`. The fixture
   uses the dashed spelling on both sides.
4. The `calculate_coexp` undefined-`status` bug in `FeatureCorPlot`
   (SCP-plot.R:5298) is already known (porting rule 6). It was not hit here.

### cell_dim_plot_3d

| Option | Status |
|---|---|
| traces: one per level, level order, names | **matches** exactly |
| trace colours (Paired, verbatim slice) | **matches** exactly (hex-identical) |
| trace membership and cell order; x/y/z | **matches** exactly (atol 0) |
| NA cells, default | fixed; matches (an `NA` trace, last) |
| NA cells, `show_na=True` | fixed; membership and colour (`#CCCCCC`) match |
| `pt_size` default | fixed; matches (1.5) |
| scene axis ranges | fixed; matches to 1e-6 (pinned, cube aspect) |
| title | fixed; matches exactly |
| axis titles | `PC_1` vs R `pca_1`. This is the known key divergence (the notebook's `UMAP_`/`umap_`), not re-flagged |
| `cells.highlight`, `lineages` | not checkable: not ported (no parameters) |

**NA handling** (`src/scp/pl/_dim.py:605`, `:622-633`). R's
`CellDimPlot3D` has no `show_na` switch. It always turns NA into an `"NA"`
level and colours it with `palette_scp(..., NA_color = bg_color,
NA_keep = TRUE)` (SCP-plot.R:2919-2926). Python defaults to
`show_na=False`, so the 100 NA cells disappear. With `show_na=True`, `"NA"` is
appended to `levels` *before* `discrete_palette`, so it takes the sixth
Paired colour, and `colors.get(lv, bg_color)` never falls through. This is a
real bug.
Fix: build `colors = discrete_palette([l for l in levels if l != "NA"], ...)`,
then set `colors["NA"] = bg_color`, and default `show_na=True` (or drop the
switch) to match R.

**Defaults and layout** (`_dim.py:604`, `:635-641`). These are small,
undocumented divergences.
Fix: set `pt_size=1.5`. Add `title=f"Total (nCells:{n})"`. Pass
`range=[min, max]` per axis along with `aspectratio=dict(x=1, y=1, z=1)`, as
SCP-plot.R:3027-3030 does.

**Fixed since the audit**, as proposed. `show_na` now defaults to `True` and
is kept as a Python-only way to drop NA cells; the palette spans the real
levels and `NA` takes `bg_color`.

### feature_dim_plot_3d

| Option | Status |
|---|---|
| per-feature colour values | **match** exactly where Python keeps them (rtol 1e-12; object order) |
| `bg_cutoff` default | fixed; matches to 1e-12 (default `None`: nothing masked, the range starts at 0 / -0.356 as in R). At audit time Python NaN-masked 76-80% of the cells, including every negative S_score |
| colour scale | Spectral vs R's plotly.js default. **Documented deliberate divergence** (docstring), so it is asserted as Python's documented behaviour |
| multiple features | Python: one trace per feature plus a visibility dropdown. R: one trace plus a marker-swap dropdown. Same values per feature, so this is structural only |
| `split.by` dropdown, `cells.highlight`, `calculate_coexp` | not checkable: not ported |

**Masking** (`src/scp/pl/_dim.py:653`, `:688-689`). `FeatureDimPlot3D`
has no `bg_cutoff` and no quantile clipping (dim_plots brief §3.2;
SCP-plot.R:3239-3242 pass the raw values as `marker.color`). The port copied the 2-D default
`bg_cutoff=0.0`, which hides most of a sparse gene and every negative signed
score. The docstring documents only the palette change. This is a real bug.
Fix: default `bg_cutoff=None` (or remove the parameter).

**Fixed since the audit**: the default is `None` and the parameter stays as a
Python-only opt-in; `pt_size` also moved to R's 1.5.

### cell_cor_heatmap

| Option | Status |
|---|---|
| collapsed query x collapsed ref, `cosine` / `pearson` / `spearman` (self) | **matches** to 1.6e-9 (1e-10 at full fixture precision); row/col order identical |
| panc8 reference, `cosine` / `spearman` | **matches** to 6.7e-9; ref columns in first-appearance order, as R |
| uncollapsed query (`query_collapsing=False`) | values **match** to 1.4e-8 once aligned by cell; **row order differs** |
| collapsing convention (`log1p(mean(expm1(x)))`) | **matches**; confirmed against Seurat 5 `AverageExpression(slot = "data")` |
| colour limits (plain min/max) | **match** to 1e-9 |
| labelled cells, `label_by` row / column / both, `nlabel`, `label_cutoff` | **match**: the same cells are labelled. Text format differs (`1.00` vs R `round(x, 2)` printing `1`), cosmetic |
| default `nlabel` / `label_cutoff` | **differs**: 3 / -inf vs R 0 / 0 (R draws no labels; Python draws 24 on the 8x8 case) |
| `cluster_rows` / `cluster_columns` | **differs**: same tree, different leaf order (all 4 cases, rows and columns) |
| `srt_ref = NULL` with `query_group` | **differs**: Python raises `KeyError: None` |
| default feature selection | **differs**: max abs 0.157 on the default 8x8 matrix; 3.6e-10 once Python is given R's HVF list (checked on the full h5ad, not in the suite) |
| legend title | `Pearson similarity` / `Spearman similarity` vs R `Correlation similarity`. Cosmetic: R's `RunKNNPredict` renames both metrics to `"correlation"` before the title is built |
| query/ref cell annotations, `bulk_ref`, reductions, distance metrics other than the 3 | not checkable: not ported |

**`reference=None`** (`src/scp/pl/_heatmap.py:543-546`). R copies
`ref_group <- query_group` and `ref_collapsing <- query_collapsing` when no
reference is given (SCP-plot.R:10150-10156). Python only aliases the object,
so `ref_group` stays `None` and `as_ordered_categorical(adata.obs[None])`
raises. This is a real bug. Fix:
`if reference is None: reference, ref_group, ref_collapsing = query, query_group, query_collapsing`
(after resolving `query_collapsing`).

**Uncollapsed row order** (`_heatmap.py:563`). R orders query cells level by
level (`cell_groups[["query_group"]]`, SCP-plot.R:10218-10241), so each
group's cells sit together under the group block annotation. Python keeps
`obs_names` order. The values match, but the picture is a shuffle. This is a
real bug.
Fix: when not collapsing and `group` is given, reorder the rows by
`np.argsort(cat.cat.codes, kind="stable")` (dropping NA). Apply the same to
the reference.

**Clustering order** (`_heatmap.py:591`, `:596`). ComplexHeatmap clusters with
`hclust(dist(m), "complete")`, then reorders each dendrogram with
`reorder(dend, -rowMeans(m), mean)` (or `-colMeans`), because
`row_dend_reorder` defaults to TRUE for numeric matrices. scipy's
`leaves_list` is not reordered. The fix is to walk `scipy.cluster.hierarchy.to_tree(Z)`,
put the child with the smaller weight first (a leaf's weight is
`-rowMeans`, a node's is the mean of its two children), and concatenate the
leaves. `test_proposed_dendrogram_reorder_reproduces_r` checks this recipe
against R's own orders, and it reproduces all 8 orders exactly. Use the
same helper for `feature_cor_heatmap` and the other heatmaps if they cluster
through scipy.

**Defaults** (`_heatmap.py:522-523`). R's signature is `nlabel = 0,
label_cutoff = 0` (SCP-plot.R:10128). Python's `nlabel=3` labels every
heatmap by default, and `label_cutoff=None → -inf` would let negative
similarities through.
Fix: `nlabel=0, label_cutoff=0.0`.

**Default features** (`_heatmap.py:550-555`). R uses
`intersect(VariableFeatures(query), VariableFeatures(ref))`
(SCP-cell_annotation.R:126, 220, 263), which here is all 3,467 stored HVFs,
and runs `FindVariableFeatures(nfeatures)` only when none are stored.
Python ignores `var["highly_variable"]` and takes the 2,000 shared genes with
the largest variance.
Fix: use `query.var["highly_variable"] & reference.var["highly_variable"]`
when both exist. Otherwise compute HVGs (`sc.pp.highly_variable_genes(...,
n_top_genes=nfeatures)`), and keep the variance ranking only as the last
fallback.

Minor, not tested: Python clamps `+inf` to 1, while R uses `max(|finite|) *
sign` (SCP-plot.R:10202), which differs only for distance metrics. R also
drops reference cells whose `ref_group` is NA (SCP-cell_annotation.R:200);
Python does not.

### feature_cor_plot

| Option | Status |
|---|---|
| grid placement (row = f2, column = f1; scatter where f1 < f2, correlation block where f1 > f2) | **matches** R |
| correlation values, `pearson` and `spearman` (proxyC `correlation` on rank rows) | **match** to the 3 printed decimals (abs 5e-4). Text `-0.020` vs R `-0.02` is cosmetic |
| correlation block colour | **matches** within 6/255 per channel. R bins r into 100 palette steps; Python interpolates |
| scatter points | **match** exactly |
| `lm` fit line slope/intercept | **match** to 1e-9 relative |
| r² / p annotation values | **match** at R's `format(digits = 2)`. Text `0.0004` vs R `4e-04` is cosmetic |
| violin extent (`trim = TRUE`) | **matches** exactly (data min/max) |
| violin shape | **differs**: up to 0.46 in normalised half-width |
| `split_by` | **differs**: ignored |
| `add_smooth=False` | **differs**: r²/p text disappears |
| `cells.highlight`, `calculate_coexp`, `reverse`, `adjust`, raster | not checkable: not ported (`calculate_coexp` is broken in R anyway) |

**Violin bandwidth** (`src/scp/pl/_stat.py:844`). `ax.violinplot` uses
matplotlib's Scott factor (`n^-1/5 · sd`). ggplot's `geom_violin` uses
`density(bw = "nrd0")` = `0.9 · min(sd, IQR/1.34) · n^-1/5`, falling back to
`sd` when the IQR is 0. On this zero-inflated data that is a factor 0.9 at
best, and the Endocrine violins are visibly different (0.46). This is a real
bug, and the same call is at `_stat.py:270` in `feature_stat_plot`, which
was outside this audit.
Fix: pass `bw_method=lambda kde: bw_nrd0(v) / np.std(v, ddof=1)` (times
`adjust`). Checked offline: that reproduces R's `violinwidth` to 4e-4 for all
15 violins.

**`split_by` ignored** (`_stat.py:773`). It is in the signature but never
read. R builds one SPLOM per split level, with the correlations computed
within the split (SCP-plot.R:5371-5385). This is a real bug: the parameter
silently does nothing.
Fix: loop over the split levels, compute `corr` on each split's cells, and
put the SPLOMs on a `PanelGrid`, or raise `NotImplementedError` until then.

**Stats tied to `add_smooth`** (`_stat.py:856`). R draws the
equation/r²/p annotation whenever `add_equation`, `add_r2` or `add_pvalue` is
set (SCP-plot.R:5439, outside the `add_smooth` block at 5434). Python
computes `linregress` and the text only inside `if add_smooth`.
Fix: compute `linregress` unconditionally and gate only the red line on
`add_smooth`.

**Docstring** (`_stat.py:793`). It says "Lower triangle = solid correlation
blocks ... upper triangle = scatter". The code (correctly, like R) draws the
scatter at row > column, which is the lower-left triangle, and the blocks
upper-right. Only the docstring needs fixing.

### feature_cor_heatmap

Not checkable. SCP's `FeatureCorHeatmap` is an empty stub (SCP-plot.R:9980:
`function(srt, features, cells) { }`, unexported), so there is no R output to
compare. The port is clean-room (documented in its docstring and in
`docs/04_api_mapping.md`). If it clusters through scipy's `leaves_list`, its
order will also differ from what a ComplexHeatmap user would expect; see the
reorder fix above.
