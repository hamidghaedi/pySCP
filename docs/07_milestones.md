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

Not verified against R, and why: the `hull` mark ports `concaveman`, which is
not installed in the parity environment; grid and stream velocity delegate to
scvelo, which is not installed here (tested with a precomputed grid in its two
return shapes); the stream drawing is matplotlib's `streamplot`, not metR, so
it is not a numeric match by construction. `stat_plot_type` beyond
pie/ring/bar raises.

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

### 4 — `feature_stat_plot` / `expression_stat_plot` — L ✅ done (except `flip`)

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

Still stub: `flip` (`coord_flip`), which raises `NotImplementedError`.

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

### 6 — `feature_cor_plot`, `cell_density_plot`, `volcano_plot` — M ✅ done

Brief: `porting_briefs/stat_plots.md` §3–§5.

All three landed; see `notebooks/01_parity_foundation.ipynb` §10. `volcano_plot`
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

### 7 — The heatmap engine + `group_heatmap` — L (engine + group_heatmap done)

Brief: `porting_briefs/heatmaps.md`. This is the biggest single piece.

Landed: `heatmap/render.py` (inch-exact GridSpec engine, nested split subgrids,
simple/block/mark tracks, whiteout/bg/reticle/dot layers, legend column,
dendrogram axes) and `group_heatmap`. Its matrix matches R's `matrix_list` to
8.6e-07 — see `notebooks/01_parity_foundation.ipynb` §8.

Still to do: `feature_heatmap` (per-cell, downsampled), `cluster_within_group2`,
enrichment tracks, and the `mfuzz`/peaktime split methods.

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

### 9 — `dynamic_heatmap` / `dynamic_plot` — L (binned form done; fitted form deliberately not)

Brief: `porting_briefs/heatmaps.md` §8, `trajectory_enrichment.md` §5.

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

### 11 — `graph_plot` + `paga_plot` — M ✅ done

Brief: `porting_briefs/trajectory_enrichment.md` §2–§3.

Build `graph_plot` first; `paga_plot` is then nearly free, because
`adata.uns['paga']` has exactly the keys SCP reads. Two things to port
precisely: the **net** transition reduction (only the dominant direction is
drawn) and the node positions as the per-group *median* of the embedding
rather than `uns['paga']['pos']`.

### 12 — `lineage_plot`, `velocity_plot` — M ✅ done

Brief: `porting_briefs/trajectory_enrichment.md` §1, §4.

Delegate `compute_velocity_on_grid` to scvelo rather than reimplementing it —
the R version is a transliteration of scvelo's, with one deliberate divergence
(no ±1% grid padding). `ax.quiver` and `ax.streamplot` do the drawing.

For LOESS with `degree=2`, use `skmisc.loess` (the same underlying
C/Fortran routine R wraps); `statsmodels.lowess` is degree-1 and will not
match.

### 13 — `projection_plot` — S ✅ done

Two scatters on one axes with shared limits. The R code harvests rendered point
colors out of `ggplot_build`; unnecessary here — compute the palette directly.

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
| `cluster_within_group2`, heatmap enrichment tracks | milestone 7 extras |
| `expression_stat_plot` | the DataFrame-level engine; the AnnData path covers the same grammar |

Every one of these raises `NotImplementedError` naming its brief section, rather
than silently doing something else.
