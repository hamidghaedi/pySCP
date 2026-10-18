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
│   └── render.py      the GridSpec renderer                         [done]
└── pl/
    ├── _dim.py        cell_dim_plot, feature_dim_plot, 3D              [done]
    ├── _stat.py       feature_stat_plot, cell_stat_plot, cor, density,
    │                  volcano                                          [done]
    ├── _heatmap.py    group/feature/dynamic/cell_cor heatmaps          [done]
    ├── _traj.py       graph, paga, lineage, velocity, dynamic,
    │                  projection                                       [done]
    └── _enrich.py     enrichment_plot, gsea_plot                       [done]
```

Why `heatmap/` is its own subpackage: the heatmap family is not a ggplot
analogue at all. It needs a declarative layout model with physical sizing, and
mixing that into `pl/` would drag layout concerns into every other function.

## Sizing: panels, not figures

The thing with a known size must be the **panel**, and the figure grows to fit
its decorations. This is what `panel_fix()` buys R users, and it is why SCP
figures compose cleanly into manuscripts.

Setting `figsize` and letting the panel float — the default matplotlib habit,
and what the constrained layout engine does — gives you panels of different
sizes whenever the legend or tick labels change length, and leaves aspect-fixed
panels floating in cells wider than they are. `PanelGrid` does not use it.
Instead `PanelGrid.finish()` runs a deterministic two-pass layout:

1. **Allocate.** Every panel is created at exactly `panel_size` inches on a
   provisional canvas (one `GridSpec` whose rows and columns alternate panel
   tracks and gap tracks). When the theme fixes `aspect_ratio` and
   `panel_size` disagrees with it, the panel is the largest box of that ratio
   inside `panel_size` — `panel_fix` likewise derives the missing dimension
   from the aspect. Drawing functions draw into these axes; a cell may be
   subdivided with `pg.axes[key].get_subplotspec().subgridspec(...)` (the
   `feature_cor_plot` matrix, the upset bar/matrix pair, the stacked GSEA
   block), and the sub-axes stay inside the exact panel box.
2. **Measure.** After a draw, each cell's reach past its panel box — tick
   labels, axis titles, panel titles, strips written in axes coordinates,
   inset axes, and colourbars attached to that panel — is read from the axes'
   tight bounding boxes, in inches. Inch extents do not depend on where the
   panel sits, so one measurement is enough.
3. **Place.** Each column gap is the widest right-hand reach in the column on
   its left plus the widest left-hand reach in the column on its right plus a
   pad (`wspace`, 11 pt by default — ggplot's spacing); rows likewise. A pad
   of exactly `0` makes the panels abut (the stacked violins). Figure-level
   `suptitle` / `supxlabel` / `supylabel` get their own measured margin. The
   legend column is measured and placed, and the figure size is set to the sum
   of all of it plus a 0.1 in outer margin.

```text
| margin | legend? | supylabel? | L0 | panel | gap0 | panel | ... | R_last | legend? | margin |
gap_c = R[c] + wspace + L[c+1]        R[c] = max over the rows of column c
```

So the figure size is an *output*, nothing is drawn outside it, and saving
with or without `bbox_inches="tight"` gives the same picture (the outer margin
equals matplotlib's default `pad_inches`). After `finish()` the figure carries
a marker layout engine that does nothing at draw time but makes
`fig.subplots_adjust()` refuse to move the panels.

Colourbars on a `PanelGrid` figure go through `pg.colorbar(mappable, ax=...)`,
never `fig.colorbar`, which steals width from the panel. A bar whose `ax` lies
in one panel is that panel's own guide (each panel is its own ggplot in SCP)
and sits beside it on the legend side; with no `ax`, or axes spread over
several panels, it is a figure-level guide in the legend column. Either way it
is ggplot's default colourbar: one legend key (10 pt) thick and five long.

Borrowed axes (`ax=`) are untouched: the caller owns that figure's size.

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

Under `PanelGrid` the column lives **inside** the figure and is sized from what
it actually draws, for every `legend_position`:

* `right` / `left` — one `fig.legend` per registered entry, then any
  figure-level colourbars, stacked top to bottom with 11 pt between them, the
  stack centred vertically on the panels. When the stack is taller than the
  panel block it wraps into further columns rather than running off the
  figure; a single legend taller than the panels makes the figure taller.
* `top` / `bottom` — legends laid out horizontally (one row of keys each,
  fewer columns when that is wider than the panels), side by side, wrapping
  into further rows; the block is centred horizontally.
* `none` — no legends; a colourbar a function still asks for goes on the
  right.

Legend titles carry R's trailing colon (`"leiden:"`), and `fig.legends` keeps
registration order. `LegendColumn.draw(fig)` remains for caller-owned figures:
it stacks the legends down the inside right edge without resizing anything.

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
objects, `datashader` would be the escalation; it is not wired in, so there is
no extra for it yet.

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
