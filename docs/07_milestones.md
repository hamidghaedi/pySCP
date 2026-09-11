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

### 2 — `cell_dim_plot` overlays — M

Brief: `porting_briefs/dim_plots.md` §1.6.

Add, in this order: `cells_highlight` polish · `add_density` (both the contour
and the filled-KDE variants) · `graph` edges (use the sparse COO triplets and a
`LineCollection`; **do not** densify the adjacency as R does) · `add_mark`
(hull/ellipse/rect/circle via `scipy.spatial.ConvexHull` and
`matplotlib.patches`) · `stat_by` pie insets (`inset_axes` at each group's
median position, sized as a fraction of the axis range) · `hex` binning.

Depends on nothing beyond milestone 1.

### 3 — `feature_dim_plot` — M

Brief: `porting_briefs/dim_plots.md` §2.

Two halves. The standard path is the easier one, but three details define how
it looks: ascending-value draw order with NaN first, `bg_cutoff=0` NaN-masking
*before* plotting, and winsorized limits from `upper_quantile=0.99` with
`keep_scale` controlling the pooling scope.

The blend path (`compare_features=True`) is where `colors.py` pays off:
per-feature two-stop ramps from `adjcolors(hue, 0.1)` → `hue`, per-cell
`blendcolors`, draw order by descending RGB sum, and **one colorbar per
feature** tiled `min(ceil(sqrt(N)), 3)` per row.

### 4 — `feature_stat_plot` / `expression_stat_plot` — L

Brief: `porting_briefs/stat_plots.md` §1.

The five `plot_type`s are straightforward; the work is in the surrounding
grammar. Order: violin → box → bar → dot → col, then `bg_by` striping
(`axvspan` at `zorder=0`, with the nesting validation), then the overlays
(`add_point` jitterdodge, `add_box`, `add_trend`, `add_stat`, `add_line`), then
`stack=True` (a real shared-x GridSpec, not composited panels), then the
statistics layer.

For significance brackets there is no `ggpubr`; `statannotations` is the
closest, or hand-roll — a bracket is two `hlines` and a `text`, and the
`step.increase=0.1` stacking rule is in the brief.

### 5 — `cell_stat_plot` / `stat_plot` — L

Brief: `porting_briefs/stat_plots.md` §2.

Eleven `plot_type`s. Ship `bar`, `trend`, `area`, `dot` first (cartesian),
then `rose`/`ring`/`pie` (polar — note the dummy `"   "` level that reserves
the donut hole), then `sankey` (`plotly` or hand-rolled bezier ribbons),
`venn` (`matplotlib-venn` up to 3 sets, custom beyond), `upset`
(`UpSetPlot`), `chord` (no good Python library — consider deferring or
bezier-by-hand).

The aggregation core is a complete cross-tab with empty cells retained; get
that right before any drawing.

### 6 — `feature_cor_plot`, `cell_density_plot`, `volcano_plot` — M

Brief: `porting_briefs/stat_plots.md` §3–§5.

Three independent, self-contained figures. `volcano_plot` is the highest-value
one and takes a DataFrame, so it needs nothing from AnnData —
`io.de_from_rank_genes_groups` already builds the input. Remember the signed-y
trick: down-regulated genes hang **below** zero and the axis is relabelled with
`abs`.

`cell_density_plot` is a ridgeline; `joypy` exists but hand-rolling over
`scipy.stats.gaussian_kde` gives control over the ordering rules, which matter
here.

### 7 — The heatmap engine + `group_heatmap` / `feature_heatmap` — L

Brief: `porting_briefs/heatmaps.md`. This is the biggest single piece.

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

### 8 — 3D plots — S

`plotly.graph_objects.Scatter3d` plus `updatemenus`. Nearly mechanical. Note
that `FeatureDimPlot3D` does **not** use `palette_scp` at all in R — it falls
back to plotly's default scale, so 3D feature plots don't match the 2D
Spectral coloring. Fix that rather than porting it.

### 9 — `dynamic_heatmap` / `dynamic_plot` — L

Brief: `porting_briefs/heatmaps.md` §8, `trajectory_enrichment.md` §5.

Blocked on a GAM backend. `pygam` is the closest to `mgcv::gam(y ~ s(x,
bs="cs") + offset(log(libsize)))`; matching mgcv's GCV smoothing-parameter
selection is the hard part (`gridsearch(lam=...)`). Ship it as the
`scp-plot[gam]` extra.

Before writing a fitter, check whether `scFates` already stores what you need —
it computes and persists fitted trends, and reading them is strictly better
than refitting.

### 10 — `cell_cor_heatmap`, `feature_cor_heatmap` — M

Brief: `porting_briefs/heatmaps.md` §9. Straightforward once the engine exists.
`feature_cor_heatmap` has no R reference (the R function is an empty stub), so
implement it clean-room.

### 11 — `graph_plot` + `paga_plot` — M

Brief: `porting_briefs/trajectory_enrichment.md` §2–§3.

Build `graph_plot` first; `paga_plot` is then nearly free, because
`adata.uns['paga']` has exactly the keys SCP reads. Two things to port
precisely: the **net** transition reduction (only the dominant direction is
drawn) and the node positions as the per-group *median* of the embedding
rather than `uns['paga']['pos']`.

### 12 — `lineage_plot`, `velocity_plot` — M

Brief: `porting_briefs/trajectory_enrichment.md` §1, §4.

Delegate `compute_velocity_on_grid` to scvelo rather than reimplementing it —
the R version is a transliteration of scvelo's, with one deliberate divergence
(no ±1% grid padding). `ax.quiver` and `ax.streamplot` do the drawing.

For LOESS with `degree=2`, use `skmisc.loess` (the same underlying
C/Fortran routine R wraps); `statsmodels.lowess` is degree-1 and will not
match.

### 13 — `projection_plot` — S

Two scatters on one axes with shared limits. The R code harvests rendered point
colors out of `ggplot_build`; unnecessary here — compute the palette directly.

### 14 — `enrichment_plot` — L

Brief: `porting_briefs/trajectory_enrichment.md` §8.

Ship `bar`, `dot`, `lollipop`, `comparison` first: pure tabular, no graph
dependency, and they cover most use. Then `network` and `enrichmap` (needs
`python-igraph` for layout and community detection — its API maps 1:1 onto
R's igraph, which `networkx` does not). `wordcloud` last; it is the lowest
fidelity because `simplifyEnrichment::keyword_enrichment_from_GO` has no
Python equivalent.

Keep enrichment computation **out** of the plotting layer: accept a DataFrame
matching `io.ENRICHMENT_COLUMNS`, with a pluggable `enrichment_fn` for the
heatmap tracks.

### 15 — `gsea_plot` — M

Brief: `porting_briefs/trajectory_enrichment.md` §9. `gsea_scores` is already
implemented. The `line` type is one intricate three-panel figure; the rest
reuse milestone 14's machinery.

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
