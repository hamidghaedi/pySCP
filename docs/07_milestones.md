# Milestones

Ordered so that each one unlocks the next, and so that the highest-use
functions land early. Milestone 1 is done.

Effort is a rough size, not a schedule: **S** = a session, **M** = a few, **L**
= a substantial piece of work with real design decisions inside it.

---

### 1 — Foundation ✅ done

`palettes.py`, `colors.py`, `theme.py`, `fetch.py`, `layout.py`,
`heatmap/matrix.py`, `heatmap/spec.py`, `io.py`, and `cell_dim_plot`'s core
path as the reference implementation.

Verified: 229 palettes load, verbatim-slice rule holds, blend modes match,
`zscore` uses ddof=1, `fetch_data` resolves genes/obs/coords in the right
order, `default_reduction` picks `X_umap`, a two-panel split figure renders.

---

### 2 — `cell_dim_plot` overlays — M ✅ done

Brief: `porting_briefs/dim_plots.md` §1.6.

Landed, in SCP's layer order: `add_mark` (hull/ellipse/rect/circle) · `graph`
edges (COO triplets into one `LineCollection`; the adjacency is never
densified) · `add_density` (contours and the filled KDE raster) · `hex` ·
`cells_highlight` validation and `alpha_highlight` · `stat_by` insets (pie,
ring, bar) · `lineages` · `paga` · `velocity` (raw, grid, stream) · a
`label_repel` pass. The numeric cores are in `pl/_dim_geom.py`, the layers in
`pl/_dim_layers.py`.

Verified against SCP 0.5.6 on `pancreas_sub` (`tools/parity/dim_overlays.R`
writes the fixtures, `tests/test_dim_parity.py` compares, no R needed): the
KDE grid to 6e-9 relative and the contour levels exactly; every R contour
vertex lies on the Python surface at its level to 1e-10; hexagon cells and
counts identical (670 of 670, including the 5 ggplot censors at the limits)
and alpha to 1e-7; graph edges identical (1536 segments, widths to 2e-5 mm);
LOESS lineage curves to 8.5e-8; PAGA node positions, colours and edge widths
exact; raw velocity arrows exact; inset centres and sizes exact and pie
fractions to 1e-7; ggforce's enclosing circle to 1e-6 mm and ellipse centre
and semi-axes to 4e-8 mm.

Found along the way, and documented in the docstrings: ggforce's ellipse angle
loses the sign of the axis' y-component, so 4 of the 5 `CellType` ellipses it
draws are mirrored and do not enclose their points (the port draws the
enclosing one); `PAGAPlot(cells=)` intersects group names with cell names and
drops every node.

Grid and stream velocity no longer go through scvelo. The overlay smooths the
field with `compute_velocity_on_grid`, the numpy port `velocity_plot` uses
(milestone 12), so nothing in the package imports scvelo and the `velocity`
extra is gone. Against the trajectory audit's R fixtures the overlay's grid
matches `VelocityPlot`'s to 5e-8 (positions and arrow ends, the fixture's
precision), and the stream field it hands to `streamplot` matches R's
`(2, ns, ns)` field and NA mask to 5e-13 (`tests/test_parity_traj.py`). The
lineage, PAGA and velocity layers now share their code with `lineage_plot`,
`paga_plot` and `velocity_plot` (one curve fit, one PAGA node table and
`graph_plot` drawing, one velocity lookup, grid and stream layer), so a layer
and its standalone plot cannot drift apart; every fixture above still matches.

Not verified against R, and why: the `hull` mark ports `concaveman`, which is
not installed in the parity environment; the stream drawing is matplotlib's
`streamplot`, not metR, so it is not a numeric match by construction (its
input field is). `stat_plot_type` beyond pie/ring/bar raises.

### 3 — `feature_dim_plot` — M ✅ done

Brief: `porting_briefs/dim_plots.md` §2.

Landed: both halves, verified against R in `notebooks/01_parity_foundation.ipynb`
(`nPos` counts and percentages agree exactly), and since then `add_density`,
`graph`, `lineages`, `hex` (mean value per hexagon over flat background
hexagons), labels at the 95th–99th percentile band, highlight over-layers and
`calculate_coexp` through the shared `scp.fetch.calculate_coexp`. The CoExp
values match `FeatureDimPlot(calculate_coexp = TRUE)` to 4e-9 relative and the
per-hexagon means match `stat_summary_hex` to 1e-7.
`keep_scale="feature"` now pools over every cell before `cells` subsets, as
R's `dat_exp` does.

SCP 0.5.6 cannot run `calculate_coexp` as shipped under SeuratObject 5:
`check_DataType` calls `GetAssayData(slot = )`, which is defunct. The parity
script shims that one call.

Two halves. The standard path is the easier one, but three details define how
it looks: ascending-value draw order with NaN first, `bg_cutoff=0` NaN-masking
*before* plotting, and winsorized limits from `upper_quantile=0.99` with
`keep_scale` controlling the pooling scope.

The blend path (`compare_features=True`) is where `colors.py` pays off:
per-feature two-stop ramps from `adjcolors(hue, 0.1)` → `hue`, per-cell
`blendcolors`, draw order by descending RGB sum, and **one colorbar per
feature** tiled `min(ceil(sqrt(N)), 3)` per row.

### 4 — `feature_stat_plot` / `expression_stat_plot` — L ✅ done

Brief: `porting_briefs/stat_plots.md` §1.

`expression_stat_plot` (`pl/_stat_feature.py`) is the DataFrame engine and
`feature_stat_plot` the AnnData wrapper over it, as in R. All five
`plot_type`s, `bg_by` striping, `sort`, `split_by` dodging, `same_y_lims`, the
overlays including `add_trend`, `individual`, `plot_by="feature"`, `stack=True`
(one shared-x GridSpec column per `group_by`, two y ticks per row, one y label,
one legend), `fill_by="expression"`, `calculate_coexp` (through
`scp.fetch.calculate_coexp`), and the significance layer: `comparisons=True`,
bracketed `comparisons`, `ref_group`, `multiplegroup_comparisons`, both
`sig_label`s. The tests and their text live in `pl/_stat_sig.py`; ggpubr and
ggsignif are reproduced at the level of the base-R calls they make.

Verified against R: `tools/parity/feature_stat_tests.R` pins `wilcox.test`
(exact and normal branches, ties, n at 49/50), Welch `t.test`,
`kruskal.test`, `anova`, `format.pval`, `symnum` and `sprintf("%.2g")`;
`tools/parity/feature_stat_layers.R` dumps SCP 0.5.6's built layers on
`pancreas_sub` (panel names and order, final y ranges, bracket coordinates and
labels, `stat_compare_means` positions/p-values/labels, trend points, dot
counts, stacked-row breaks, `CoExp`). Both fixtures sit in
`tests/fixtures/parity/feature_stat/` and agree to 1e-9.

Geometry now follows ggplot2: `position_dodge(0.9)` counted per x, the
discrete axis's 0.6 padding, `coord_cartesian`'s 5 % expansion, `bw.nrd0`
violins, `cut()`-label bin centres for `dot`, jitter of `0.4/(ndodge+2)`, and
the bar plot's lower limit taken from its error bars. Fill colours are keyed by
level name, so `sort` no longer recolours the groups.

`flip=True` is R's `coord_flip`: every layer is drawn through one
(position, value) adapter (`_Ax` in `pl/_stat_feature.py`) that transposes the
geometry, the stripes, the reference lines, the significance layer, the
limits and the ticks together; the group factor is reversed (first level on
top), `aspect_ratio` inverted, and `col` numbers its cells from the top. Under
`stack` the panels sit side by side (R's `cbind`), one row per group column,
group labels once at the left, two value ticks per column turned 90 degrees,
strips on top. `feature_stat_layers.R` writes `layers_flip.json` from SCP's own
`coord_flip` plots (eleven cases: violin, box with brackets, split
`comparisons=True`, `multiplegroup_comparisons`, `bar` with `bg_by`, split
trend, dot, `col`, `sort` with `aspect.ratio`, `plot_by="feature"`, `stack`);
`tests/test_expression_stat_flip.py` matches both axes' ranges, the group
labels bottom-up, the aspect ratio, stripe edges and colours, bracket, p-value,
trend, dot and bar data to 1e-9, and checks that each artist was drawn
transposed. Two R behaviours are not copied, both documented: under flip
SCP reverses the factor after building ggpubr's group key, so
`multiplegroup_comparisons` without a split loses its label in R (kruskal.test
fails per x); the port keeps it at the bottom position. ggsignif places a
flipped bracket's label above the bracket's upper end; the port puts it beside
the bracket. Making `col` match R also fixed its unflipped cell-axis range,
which now follows R's (bars and finite stripe edges, padded by 0.6).

### 5 — `cell_stat_plot` / `stat_plot` — L ✅ done

Brief: `porting_briefs/stat_plots.md` §2.

Landed: the aggregation core (complete cross-tab, empty cells retained,
percent within group) plus all eleven `plot_type`s and `individual`. The
cartesian and polar types were verified against R in
`notebooks/01_parity_foundation.ipynb` §9.

`sankey`, `chord` and `venn` are hand-rolled in `pl/_stat_flow.py`, each as a
pure layout function plus a renderer into the panel's `Axes`. They are pinned
against SCP 0.5.6 on `pancreas_sub` by `tests/test_stat_flow.py`, from a
fixture written by `tools/parity/cell_stat_flow.R`:

* sankey -- ggsankey's node rectangles and flow ribbons from `ggplot_build`,
  agreeing to 4e-13 (16 nodes, 23 flows over three columns; 8 and 12 over
  two);
* chord -- circlize's sector order, degrees (to 4e-14) and every link's two
  end positions (exact), for a table with disjoint level names (two-group
  layout) and one whose sides share names (merged sectors, self links);
* venn -- region ids, exclusive counts and percentages for 4 and 5 sets,
  exact, computed by ggVennDiagram's own set operations; outlines are
  ggVennDiagram 1.2.2's shapes for 1-7 sets (`tools/parity/cell_stat_venn_shapes.R`
  generates `pl/_venn_shapes.py`), and R's cap of seven sets is kept;
* individual -- `StatPlot(individual = TRUE)`'s panel order, titles and
  per-panel values, with and without a split.

Deliberate divergences, each documented in the docstrings: sankey ribbons
leave a node in target order (SCP sorts them alphabetically because
`next_node` is never made a factor, so ribbons cross); split sankey and chord
panels carry the split level as a title; chord names a missing row level
`NA` as it does a column one and never reinterprets the table as an edge list.
In this R stack `StatPlot(plot_type = "sankey")` itself errors while splicing
the legends onto the gtable, after the plot is built; the fixture script
rebuilds that plot from SCP's own lines.

`upset` (`upset_layout` + `upset_plot` in `pl/_stat_flow.py`) no longer builds a
private figure: each `split_by` level gets one `PanelGrid` panel, divided into
the count bars over ggupset's combination matrix. `tools/parity/cell_stat_upset.R`
(ggupset 0.4.1, installed in a scratch library) dumps SCP's plots for four and
six sets, with and without a five-level split, and
`tests/test_stat_upset.py` matches, per block, the bar order (count
descending, ties in C-collated name order, the first 20), every count, the
matrix rows and their order, the count labels and the material-indigo fills
(within 6/255; R interpolates in Lab).

`flip=True` on the cartesian types (`bar`, `area`, `trend`, `dot`) now
transposes the panel through the same adapter as `feature_stat_plot`; it
previously only reversed the group order. This is structural (tested in
`test_cell_stat_plot.py`), not pinned against an R dump.

### 6 — `feature_cor_plot`, `cell_density_plot`, `volcano_plot` — M ✅ done

Brief: `porting_briefs/stat_plots.md` §3–§5.

All three landed; see `notebooks/01_parity_foundation.ipynb` §10.
`feature_cor_plot`'s four audit findings are fixed (`docs/parity_audit.md`):
the diagonal violins use `bw.nrd0` (to 5e-4 of R's `violinwidth`), `split_by`
draws one matrix per level on a `PanelGrid` with per-split correlations (to the
printed 3 decimals), the r²/p text no longer depends on `add_smooth`, and the
docstring's triangles are right. `volcano_plot`
raises a pointed error when the detection fractions are missing, since scanpy
only fills them for `rank_genes_groups(..., pts=True)` and an all-NaN x would
otherwise render as an empty panel.

Three independent, self-contained figures. `volcano_plot` is the highest-value
one and takes a DataFrame, so it needs nothing from AnnData —
`io.de_from_rank_genes_groups` already builds the input. Remember the signed-y
trick: down-regulated genes hang **below** zero and the axis is relabelled with
`abs`.

`cell_density_plot` is a ridgeline; `joypy` exists but hand-rolling over
`scipy.stats.gaussian_kde` gives control over the ordering rules, which matter
here.

### 7 — The heatmap engine + `group_heatmap` / `feature_heatmap` — L ✅ done

Brief: `porting_briefs/heatmaps.md`. This is the biggest single piece.

Landed: `heatmap/render.py` (inch-exact GridSpec engine, nested split subgrids,
simple/block/mark/textbox tracks on every side, whiteout/bg/reticle/dot
layers, legend column, per-slice dendrograms), `heatmap/cluster.py`
(SCP's splitters and `C1..Cn` naming, ComplexHeatmap's slice clustering and
`reorder.dendrogram`, `cluster_within_group` / `cluster_within_group2`),
`heatmap/enrichment.py` (`heatmap_enrichment`'s word boxes plus an `ora`
backend) and both wrappers, with `split_by`, `grouping_var`,
`feature_split_by`, `cluster_features_by`, `cluster_row_slices`,
`cluster_column_slices` and `split_method` `kmeans`/`hclust`/`mfuzz`.

Verified against SCP 0.5.6 on `pancreas_sub` by
`tools/parity/heatmap_fixtures.R`, pinned in `tests/test_heatmap_parity.py`:
`split_by` and `grouping_var` matrices to <7e-07; every `hclust` split and
every deterministic row/column order exactly (`cluster_within_group2`,
`cluster_row_slices`, `cluster_column_slices`, raw-scale `cluster_rows`);
`kmeans` with ARI 1.0; `feature_heatmap`'s per-cell matrix to 5e-10;
`RunEnrichment`'s ORA p-values to 5e-12 (relative) and the terms / keys /
features boxes word for word and colour for colour.

Not reproducible, by construction: ComplexHeatmap reorders z-scored rows by
`-rowMeans`, which is rounding noise, so there only the tree is compared;
`mfuzz` has no R fixture (e1071 is not in the parity environment) and its
random start differs anyway; `anno_keys` on GO uses the tokenising keyword
method, not `simplifyEnrichment`.

Since then the last stubs have landed, in `heatmap/glyphs.py` and the
renderer:

* `group_heatmap`'s `cell_annotation` is drawn as R draws it. A categorical
  annotation gets a pie per column (`CellStatPlot(plot_type = "pie")`). A
  numeric one gets a violin per column, filled by group (or split level), on
  one range per panel (`FeatureStatPlot(fill.by = "group", same.y.lims =
  TRUE)`).
* `add_violin` puts a violin of each feature into every body cell, with
  `fill_by`, `fill_palette` and `fill_palcolor`. The body is whited out and
  its legend dropped, as in R.
* `flip=True` works for `group_heatmap`, `feature_heatmap`,
  `dynamic_heatmap` and `cell_cor_heatmap`. Panels are transposed and stacked
  vertically, annotations move to the sides ComplexHeatmap gives them, and
  the `cluster_*` flags are swapped on entry, as R does. Enrichment boxes
  raise with `flip`, as in R.

Checked against R in `tests/test_heatmap_glyphs_parity.py`, on the
fixtures `heatmap_fixtures.R` now also writes:

* pie wedges and fills exactly, with and without `split_by`;
* violin outlines to 3.5e-4 of the maximum width (`add_violin` 2.8e-3; the
  rest is `stats::density`'s binning) on exactly R's 5%-expanded ranges;
* flipped feature orders and splits exactly, for all three functions.
  R's own flipped fixtures equal its unflipped ones with the flags swapped.

The dot layer now scales the dot's *diameter* with the percentage, as
`grid.points(size = dot_size * perc)` does (it scaled the area), with a black
outline. The reticle runs through the cell centres. The legend column is
stacked by measured heights, and a short heatmap grows to fit it.

Build `heatmap/render.py` bottom-up and do not skip a step:

1. one panel, `simple` tracks only, no splits;
2. splits + gaps (nested GridSpec, shared subdivision across body and tracks);
3. multi-panel concatenation;
4. clustering + dendrogram axes, including `cluster_within_group2`;
5. `add_dot` / `add_bg` / `add_reticle` layers;
6. `anno_mark` with 1-D label repulsion and leader lines;
7. the wrappers.

`matrix_process`, `lib_normalize`, `clean_nonfinite` and `color_limits` are
already done and tested — start from them.

### 8 — 3D plots — S ✅ done

`plotly.graph_objects.Scatter3d` plus `updatemenus`. Nearly mechanical. Note
that `FeatureDimPlot3D` does **not** use `palette_scp` at all in R — it falls
back to plotly's default scale, so 3D feature plots don't match the 2D
Spectral coloring. Fix that rather than porting it.

Verified against SCP 0.5.6 (`tools/parity/audit_dim3d.R` →
`tests/test_parity_dim3d.py`, no xfails left): traces, level order, colours,
membership and coordinates match exactly. The audit's six mismatches are
fixed: NA cells are always drawn as a last `NA` trace in `bg_color` (R has no
`show_na` switch; the Python one now defaults to `True`), `pt_size` is R's 1.5,
the scene axes are pinned to the data range in a cube, the title is
`Total (nCells:N)`, and `feature_dim_plot_3d` masks nothing by default
(`bg_cutoff=None`; R has no cutoff), so its colour values match R's to 1e-12.

The remaining options are ported and verified the same way (`audit_dim3d.R`
now also runs `RunSlingshot` as `audit_traj.R` does):

* `cell_dim_plot_3d(cells_highlight=)` — a `highlight` trace after the levels
  with R's size, colour and `circle-open` symbol; its cells and their order
  match exactly, for a cell list and for `True` (every cell with a group).
* `cell_dim_plot_3d(lineages=)` — R's per-axis degree-2 LOESS over every
  lineage cell (no trim, unlike 2-D), range filter and de-duplication; the
  three line traces match R's to 1.3e-13 with the same point counts (792,
  685, 623), Dark2 colours and width.
* `feature_dim_plot_3d(split_by=, cells_highlight=)` — the split dropdown's
  labels, titles and per-trace masks match R's buttons; the highlight trace
  matches; the title `All_cells (nCells:N)` and the pinned cube scene, which
  the port lacked, now match too. R bug fixed and documented: with
  `split.by` set, R's `All.cells` button points at coordinate columns that do
  not exist, so pressing it empties the plot; here it restores every cell.
* `axis_labs` and `force` (over 100 levels / 50 features) for both.

Still structural, not numeric: R keeps one `All_cells` trace and swaps its
marker per feature, the port keeps one trace per feature (same values).

### 9 — `dynamic_heatmap` / `dynamic_plot` — L (raw form done; fitted form deliberately not)

Brief: `porting_briefs/heatmaps.md` §8, `trajectory_enrichment.md` §5.

`dynamic_heatmap` now follows R's `use_fitted = FALSE` path: one column per
cell ordered by pseudotime (`reverse_ht` descending), library-normalised per
lineage, z-scored once across all lineages, features pre-sorted by peak (or
valley) time, every `split_method` including `kmeans-peaktime` /
`hclust-peaktime` (default `mfuzz`, as in R), lineage-membership and
pseudotime tracks, `pseudotime_label`, enrichment boxes. Peak times come from
R's `RunDynamicFeatures` table (`dynamic_features=` or
`adata.uns["DynamicFeatures_<lineage>"]`); without one a GAM-free running-mean
estimate stands in. Given R's table, the cell order, the joint matrix (to
2e-09) and every split and row order match SCP exactly — 1-D kmeans is solved
exactly, which is the optimum R reaches. The audit's row-order finding is
closed the same way: `audit_traj.R` exports the `DynamicFeatures` tables, and
with them the row order and the per-lineage peak times match R exactly.

`separate_annotation` is ported:

* A categorical column becomes `CellDensityPlot`'s density ridges over the
  pseudotime ranks: one joint `bw.nrd0` bandwidth, groups by decreasing
  median, reversed for `reverse_ht`. It matches R's ridges to 2.5e-4 of the
  strip height.
* Features become `DynamicPlot`'s fitted curves and intervals. Those need
  mgcv's fits, so they are read from R's `fitted_matrix` / `upr_matrix` /
  `lwr_matrix`, passed with the table in SCP's `srt@tools` layout. They then
  match R to 5e-10.
* Without the fits, this raises and says what to pass.

`flip=True` works too: panels stacked vertically, flags swapped as in R.

Blocked on a GAM backend. `pygam` is the closest to `mgcv::gam(y ~ s(x,
bs="cs") + offset(log(libsize)))`; matching mgcv's GCV smoothing-parameter
selection is the hard part (`gridsearch(lam=...)`). Ship it as the
`scp-plot[gam]` extra.

Before writing a fitter, check whether `scFates` already stores what you need —
it computes and persists fitted trends, and reading them is strictly better
than refitting.

> **Known risk — exact parity here is probably not achievable.** `mgcv` selects
> its smoothing parameter by GCV/REML over a penalised spline basis; `pygam`
> selects by grid search over a different basis and does not reproduce mgcv's
> criterion. Two fits of the same data will therefore differ by more than
> floating-point noise, and the difference shows up as a visibly different
> smooth. Treat this milestone as **best-effort with a documented divergence**
> rather than a parity target: pin the *shape* (monotone trend, peak position,
> ordering of features by peak time) rather than the fitted values, and say so
> in the docstring. Do not spend the budget chasing mgcv's GCV.

> Note also that reading `scFates`' persisted fits sidesteps the problem
> entirely where the user has them, and that path should be preferred.

### 10 — `cell_cor_heatmap`, `feature_cor_heatmap` — M ✅ done

Brief: `porting_briefs/heatmaps.md` §9. Straightforward once the engine exists.
`feature_cor_heatmap` has no R reference (the R function is an empty stub), so
implement it clean-room.

Every `distance_metric` SCP lists is implemented in `heatmap/similarity.py`
and matches proxyC to 5e-11 on SCP's own centroids, with two R bugs fixed:
SCP's `"hamman"` (proxyC spells it `"hamann"`, so R errors) and proxyC's
cancellation in `euclidean`, which turns a self-similarity of 1 into 0.

`cell_cor_heatmap` has no open audit finding left
(`tests/test_parity_cor.py`):

* `reference=None` inherits `query_group` and `query_collapsing`, and matches
  R to 1.2e-9.
* Uncollapsed rows are grouped level by level, as R orders them, with
  NA-group cells dropped.
* Labels default to `nlabel=0`, `label_cutoff=0`, printed as R's
  `round(x, 2)`.
* `features=None` takes the intersection of the stored
  `var["highly_variable"]` flags. Checked end to end on R's centroids, it picks
  exactly the 3,467 HVFs, and the matrix matches R's default to 1e-11.

Still not ported: the query/reference cell annotations, `bulk_ref`, the
reduction inputs and `features_type="DE"`.

### 11 — `graph_plot` + `paga_plot` — M ✅ done

Brief: `porting_briefs/trajectory_enrichment.md` §2–§3.

Build `graph_plot` first; `paga_plot` is then nearly free, because
`adata.uns['paga']` has exactly the keys SCP reads. Two things to port
precisely: the **net** transition reduction (only the dominant direction is
drawn) and the node positions as the per-group *median* of the embedding
rather than `uns['paga']['pos']`.

Verified against SCP 0.5.6 (`tools/parity/audit_traj.R` →
`tests/test_parity_traj.py`): node medians, colours, edge selection and
widths, the net-transition logic and `segementsDf` all match, and the audit's
five mismatches are fixed. `global_size` is R's signed
`sqrt(max(x)^2 + max(y)^2)`, so shortened and offset segments match to 1e-8;
net flows rescale onto the new `transition_size`; `paga_plot` passes its own
`node_size` (R's constant 4) instead of forcing `GroupSize`; sizes follow
ggplot (`scale_size_identity`, or `area_pal(1, 6)` trained on `node_size` and
`node_size * 1.2` for a column); and the node legend is registered, with
`"1: Ductal"` labels under `label=True`.

### 12 — `lineage_plot`, `velocity_plot` — M ✅ done

Brief: `porting_briefs/trajectory_enrichment.md` §1, §4.

For LOESS with `degree=2`, use `skmisc.loess` (the same underlying
C/Fortran routine R wraps); `statsmodels.lowess` is degree-1 and will not
match.

`compute_velocity_on_grid` was first delegated to scvelo. The audit showed
SCP's version is not scvelo's — no ±1% padding, `ceiling(n/50)` neighbours, no
autoscale, `scale` applied, the stream `min_mass` mapping applied once — and
the delegation also broke stream mode's reshape. It is now a numpy port of
R's 60 lines (`SCP-plot.R:6963-7039`), so `velocity_plot` needs no scvelo, and
it matches R's grid to 5e-8 (VelocityPlot's default, the fixture's precision),
3e-11 with every option moved, and stream mode's `(2, ns, ns)` field and NA
mask to 5e-13 (`tools/parity/velocity_grid.R`).

Verified against SCP 0.5.6 (`tests/test_parity_traj.py`): `lineage_plot`'s
curves match to 5e-8, with R's strict trim, `whiskers`, the end arrowhead and
a `Lineages` legend; it shares `cell_dim_plot`'s lineage layer, so the two draw
the same curves. Raw velocity arrows match to 5e-8, and `group_by` now colours
the arrows. Neither function draws cells any more: both are layers that take
`ax=` and an optional `legend=` (`LegendColumn`). Not matched: R's raw/grid
arrowheads scale with each arrow, `quiver`'s do not; metR's streamline
parameters have no `streamplot` counterpart.

`cell_dim_plot(velocity=...)` now computes its grid through the same port, so
both are R-exact and scvelo is needed nowhere: the overlay's grid matches R's
to 5e-8 and its stream field to 5e-13, and one test pins the field that
`velocity_plot` and the overlay both hand to their shared stream layer. The
port moved into `pl/_dim_geom.py` with the other numeric cores; it is still
`scp.pl.compute_velocity_on_grid`.

### 13 — `projection_plot` — S ✅ done

Two scatters on one axes with shared limits. The R code harvests rendered point
colors out of `ggplot_build`; unnecessary here — compute the palette directly.

Verified against SCP 0.5.6 (`tests/test_parity_traj.py`): reference and query
colours and positions match exactly. R bug, deliberately not reproduced and
now stated in the docstring: `ProjectionPlot` applies the union of the two
ranges only to its throwaway query plot, so the composite inherits the
reference's limits and silently drops every query cell outside them (111 of
362 in the fixture); here every query cell is drawn. The two mypy errors in it
are fixed with annotations only. Without `ax=` it now allocates its square
themed panel through `PanelGrid` like every other figure (porting rule 1).
Still open: R's point sizes and 5% axis expansion are not matched.

### 14 — `enrichment_plot` — L ✅ done

Brief: `porting_briefs/trajectory_enrichment.md` §8.

All seven types are in. `network` and `enrichmap` use `python-igraph` (extra
`graph`) because its layouts and community detection are R igraph's C core:
the term-gene graph, the term-term overlap graph (shared-gene counts), the
fast-greedy / walktrap partitions *and their C1, C2, ... numbering*, node and
gene-blend colours, and both cluster label sets agree exactly with what
`EnrichmentPlot` builds on `pancreas_sub` (`tests/test_enrich_parity.py`,
fixtures from `tools/parity/enrich_parity.R`). Coordinates do not: the layouts
are seeded from Python's RNG, not R's. `adjust_layout` is a transcription of
`adjustlayout` with its step-2 sign bug fixed.

`wordcloud`'s frequency and feature paths match SCP's word tables exactly.
The GO path calls `simplifyEnrichment::keyword_enrichment_from_GO`, whose GO
term-document matrix ships only with the R package; by default GO panels use
the frequency path instead (documented divergence: on `pancreas_sub` 52 of R's
53 GO keywords appear in it, but the scores differ in kind). Passing
`keyword_background` runs the same Fisher/Holm test, which is verified
against simplifyEnrichment's function to 1e-15. Word placement is a square
spiral, not ggwordcloud's packer; mark labels are stacked down the panel
edges rather than placed by ggforce; the `hull` mark is convex, not
`concaveman`'s concave hull.

`bar`, `dot` and `lollipop` were then checked against `EnrichmentPlot`'s
plotted frames (`docs/parity_audit.md`, enrichment family) and every mismatch
fixed: labels are SCP's `capitalize()` + `strwrap` (not stringr's wrap, which
differs on lines of exactly the width) and italic when wrapped; the bar is
filled by `color_by` with the Count at its end and a `1.3 x max` axis; the dot
plots `GeneRatio` and the lollipop `FoldEnrichment`, each ordered by it;
sizes follow `scale_size(range = c(3, 6))`; ties keep table order top-down;
and a constant-metric panel takes the palette's middle colour.

Keep enrichment computation **out** of the plotting layer: accept a DataFrame
matching `io.ENRICHMENT_COLUMNS`, with a pluggable `enrichment_fn` for the
heatmap tracks.

### 15 — `gsea_plot` — M ✅ done

Brief: `porting_briefs/trajectory_enrichment.md` §9. `line` and `bar` as
before; `comparison`, `network`, `enrichmap` and `wordcloud` reuse milestone
14's machinery per result key as R does, with SCP's term selection
(`direction="both"` takes `ceiling(top_term / 2)` of each sign), its
`(NES=..., p.adjust=..., stars)` term names -- formatted with R's `round` and
`format(digits = 3)` rules -- and core-enrichment genes. The comparison frame,
both graphs, the partitions and the word tables match `GSEAPlot`'s on
`pancreas_sub`.

The parity fix brought `line` and `bar` in line with R: SCP's selection
(`metric < cutoff`, ranked by the metric, `ceiling(k/2)` per sign for
`"both"`) for every type; `bar` in `order(NES)` with Spectral Pos/Neg fill,
alpha `-log10(p.adjust)` and the description inside the panel; `line` as one
block per result key with every selected set overlaid, the half-plane shading,
the single-set peak crosshair, core-gene labels and ranking-metric heat strip,
and the dashed zero-cross line. Both now go through `PanelGrid` instead of
creating their own figure. `line` computes its running scores from a key's
`geneList` / `geneSets` / `exponent`, the input its docstring describes.

---

## Dependency graph

```
1 ──┬─ 2 ─ 3 ──────────────── 8
    ├─ 4 ─ 5 ─ 6
    ├─ 7 ──┬─ 9
    │      └─ 10
    ├─ 11 ─ 12 ─ 13
    └─ 14 ─ 15
```

Branches under 1 are independent; 2/3, 4/5/6, 7 and 11 can proceed in parallel
if more than one person (or agent) is working.

## Suggested first cut for a usable release

Milestones 1–7 give `cell_dim_plot`, `feature_dim_plot`, `feature_stat_plot`,
`cell_stat_plot`, `volcano_plot` and `group_heatmap` — which is the ninety
percent of day-to-day use, and enough to publish `0.1.0`.


---

## Status at the end of the first pass

Every milestone has landed at least its main path. What remains is listed here
rather than left implicit:

| Not ported | Why |
|---|---|
| `sankey`, `chord` (`cell_stat_plot`) | no good Python equivalent; a bad one is worse than `NotImplementedError` |
| significance brackets, `stack=True`, `plot_by="feature"` (`feature_stat_plot`) | the largest remaining piece of milestone 4 |
| `network`, `enrichmap`, `wordcloud` (`enrichment_plot`) | need igraph layouts / a keyword-enrichment backend |
| `dynamic_heatmap(use_fitted=True)`, mgcv-matched smooths | see the milestone 9 note: not reproducible with `pygam`, so not attempted |
| `cell_cor_heatmap` cell annotations, `bulk_ref`, reductions, `features_type="DE"` | milestone 10 extras; the heatmap engine, enrichment tracks, `cluster_within_group2`, pies, violins, `separate_annotation` and `flip` have all landed |
| `expression_stat_plot` | the DataFrame-level engine; the AnnData path covers the same grammar |

Every one of these raises `NotImplementedError` naming its brief section, rather
than silently doing something else.
