# Architecture

## The central inversion

R:

```r
p1 <- CellDimPlot(srt, "CellType")    # returns a complete, self-contained plot
p2 <- FeatureDimPlot(srt, "Ins1")
p1 / p2                                # patchwork glues them afterwards
```

matplotlib has no equivalent of that last line. A `Figure` cannot be nested
inside another `Figure`. So the port cannot mirror the R control flow.

**Rule: a drawing function takes an `Axes` and draws into it. A layout object
owns figure creation and sizing.**

```python
def _draw_dim_panel(ax, dat, colors, ...): ...   # private, draws, returns nothing

def cell_dim_plot(adata, group_by, ...) -> Figure:
    pg = PanelGrid(n_panels, panel_size=..., keys=..., theme=...)
    for key, ax in pg.axes.items():
        _draw_dim_panel(ax, ...)
    return pg.finish()
```

Every plot function follows this shape. `cell_dim_plot` in
[`src/scp/pl/_dim.py`](../src/scp/pl/_dim.py) is the worked reference — copy its
structure.

## Module layout

```
src/scp/
├── __init__.py        public surface
├── palettes.py        229 palettes + palette_scp semantics          [done]
├── colors.py          adjcolors / blendcolors / Blend2Color algebra [done]
├── theme.py           theme_scp / theme_blank as objects            [done]
├── fetch.py           fetch_data, default_reduction, categoricals   [done]
├── layout.py          PanelGrid, LegendColumn, combine              [done]
├── io.py              adapters: rank_genes_groups, gseapy, paga     [done]
├── data/              palette_list.json                             [done]
├── heatmap/
│   ├── matrix.py      matrix_process, lib_normalize, color_limits   [done]
│   ├── spec.py        HeatmapSpec / PanelSpec / Track / Layer       [done]
│   └── render.py      the GridSpec renderer                         [todo]
└── pl/
    ├── _dim.py        cell_dim_plot [done], feature_dim_plot, 3D
    ├── _stat.py       feature_stat_plot, cell_stat_plot, cor, density, volcano
    ├── _heatmap.py    group/feature/dynamic/cell_cor heatmaps
    ├── _traj.py       graph, paga, lineage, velocity, dynamic, projection
    └── _enrich.py     enrichment_plot, gsea_plot
```

Why `heatmap/` is its own subpackage: the heatmap family is not a ggplot
analogue at all. It needs a declarative layout model with physical sizing, and
mixing that into `pl/` would drag layout concerns into every other function.

## Sizing: panels, not figures

The thing with a known size must be the **panel**, and the figure grows to fit
its decorations. This is what `panel_fix()` buys R users, and it is why SCP
figures compose cleanly into manuscripts.

```python
fig_w = ncol * panel_w + (ncol + 1) * margin + legend_w
fig_h = nrow * panel_h + (nrow + 1) * margin
```

Setting `figsize` and letting the panel float — the default matplotlib habit —
gives you panels of different sizes whenever the legend or tick labels change
length. Do not do it.

For heatmaps the same principle applies with more terms; see
`HeatmapSpec.figsize()` and [`porting_briefs/heatmaps.md`](porting_briefs/heatmaps.md) §7.

## Legends

SCP composes several *independent* legends onto one plot — group colors, then
lineage colors, then PAGA, then velocity, then the stat pies. In R this is done
by rendering each legend from a throwaway ggplot, `cbind`ing the gtables and
splicing the result onto the panel with `add_grob`.

Here: artists register handles with a `LegendColumn`, which lays them out once
against the figure.

```python
legend = LegendColumn(position="right")
legend.add("leiden", [Line2D([], [], marker="o", ...), ...])
legend.add("Lineages", [...])
```

Do **not** call `ax.legend()` inside a drawing function. A per-axes legend
cannot be shared across panels and will be duplicated N times in a grid.

## Return values

| R | Python |
|---|---|
| ggplot | `Figure` (or the `Axes` you passed in, for single-panel overlays) |
| patchwork | `Figure` |
| named list (`combine=FALSE`) | `Figure`; reach for `fig.axes` or use the `ax=` argument |
| heatmap named list | `HeatmapResult` dataclass |

SCP's `combine=FALSE` exists because R users needed to post-process individual
ggplots. Here that is unnecessary — `fig.axes` is already mutable — so the
argument is dropped rather than ported.

## Rasterization

`raster=True` should set `rasterized=True` on the point/mesh artist, **not**
render to a bitmap and re-embed. Vector text and axes survive; only the marks
are rasterized. This is better than what R does with `panel_fix(raster=TRUE)`,
which needs grob surgery to keep labels vector.

`raster` auto-enables above 100,000 cells, matching SCP. For genuinely huge
objects, `datashader` is the optional escalation (extra: `scp-plot[raster]`).

## Dependencies

Core: anndata, matplotlib, numpy, pandas, scipy. Everything else is an extra —
see `pyproject.toml`. A user who only wants `cell_dim_plot` should not be made
to install `pygam` and `python-igraph`.

Import heavy optional deps **inside** the function that needs them, and raise a
message naming the extra:

```python
try:
    import igraph
except ImportError as e:
    raise ImportError("network layouts need `pip install scp-plot[graph]`") from e
```
