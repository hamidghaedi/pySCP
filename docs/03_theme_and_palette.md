# Theme, palettes and color algebra

This layer is **already implemented and verified**
([`palettes.py`](../src/scp/palettes.py), [`colors.py`](../src/scp/colors.py),
[`theme.py`](../src/scp/theme.py)). This document explains why it looks the way
it does, so you don't "fix" something deliberate.

## 1. The palettes

All 229 palettes were extracted from `SCP/data/palette_list.rda` by parsing R's
XDR serialization directly (`tools/extract_palettes.py`; no R involved). They
are shipped as `data/palette_list.json`, each carrying the `type` attribute R
used (`"discrete"` or `"continuous"`) — 85 discrete, 144 continuous — because
`palette_scp` branches on it.

Sources, per SCP's own documentation: RColorBrewer, ggsci, Redmonder,
rcartocolor, nord, viridis, pals (ocean.*), dichromat, jcolors, plus three
custom (`jet`, `simspec`, `GdRd`).

**Two palettes are not what their name suggests:**

* `Spectral` is **reversed** relative to RColorBrewer — SCP reverses every
  Brewer diverging palette at build time. So low = purple/blue, high = red.
  This is `feature_dim_plot`'s default, and it is why SCP feature plots look
  the way they do.
* `Paired` is reordered: `brewer.pal(12,"Paired")[c(1:4, 7, 8, 5, 6, 9:12)]`.
  It is `cell_dim_plot`'s default.

Defaults elsewhere: `Set1` (blend path), `Dark2` (lineages), `RdYlBu`
(streamlines), `Greys` (filled density), `RdBu` (heatmaps, correlation),
`Spectral` (enrichment).

## 2. `palette_scp` semantics

### Discrete

```
n = len(levels)
if palette is declared "continuous":  interpolate to n
elif n <= len(colors):                take the FIRST n colors verbatim
else:                                 interpolate to n
```

The verbatim-slice rule is load-bearing. Interpolating a 12-color Paired down
to 3 gives three colors that are *not* Paired's first three, and every
categorical figure shifts. `tests/test_palettes.py` asserts this.

Returns a `dict[level, color]` — keyed by level, so the mapping survives
subsetting and faceting.

### Continuous

```
if n <= len(colors):  take the FIRST n colors verbatim
else:                 colorRampPalette(colors)(n)
```

The same verbatim-slice rule, applied whatever the palette's declared type.
It matters for the 100-colour viridis family (`viridis`, `magma`, `cividis`,
`turbo`, `jet`, ...): at the default `n = 100` R uses their own colours,
alpha suffix included (`#440154FF`), not a re-interpolation of them. A
5-colour request takes the first five, which is what R does too.

R cuts the literal vector `1:100` rather than the data when `matched=False`,
which only ever yields `n` levels, so `continuous_palette()` returns the `n`
colours directly.

`matched=True` returns one colour per element of `x`: R's `cut(x, seq(min,
max, length.out = n + 1), include.lowest = TRUE)` — `n` equal-width,
right-closed bins, the first closed on both sides — and bin `k` takes the
`k`-th of the `n` colours. `np.linspace` reproduces `seq`'s breaks bit for
bit, and a `searchsorted(side="left")` reproduces `.bincode`. Rounding the
value onto the ramp instead (what this port did at first) moves values that
sit near a bin edge by one colour.

### `colorRampPalette` truncates

`color_ramp` is `grDevices::colorRampPalette` operation for operation:
channels as `col2rgb / 255`, `approx1`'s linear interpolation (an exact hit on
a stop returns the stop), clamped and multiplied by 255, then
`rgb(..., maxColorValue = 255)`, which coerces to integer — **truncation**,
not rounding. A black-to-white ramp of 3 has `#7F7F7F` in the middle, not
`#808080`; rounding lands one 8-bit step above R on 87 of Spectral's 100
colours. `tests/fixtures/parity/palettes/palettes.json`
(`tools/parity/palettes.R`) pins the ramps, the continuous branch and the
matched bins against SCP 0.5.6.

### NA

`na_color` defaults to `"grey80"` = `#CCCCCC`. `na_keep=False` (the default)
strips the `"NA"` entry from the returned mapping; `show_na=True` upstream
promotes NaN to a real level instead. With `matched=True` an NA element comes
back as `None` — what R's `NA_keep = FALSE` filter leaves there — or as
`na_color` with `na_keep=True`.

Two R edge cases are deliberately not reproduced: a constant `x` makes
`palette_scp(matched = TRUE)` return a *single* colour whatever the length of
`x` (here it is repeated per element), and `reverse = TRUE` with
`matched = TRUE` reverses the per-element vector, so colours no longer line
up with their elements (here the scale is reversed).

### scanpy's stored colours (`uns[f"{key}_colors"]`)

R knows nothing of `uns`, so reading it is a **deliberate divergence**, and it
is on by default: a user who has drawn `sc.pl.umap(adata, color="CellType")`
expects `cell_dim_plot(adata, "CellType")` to use the same colours, and
`docs/02_data_contract.md` §6 settles that the stored list is read first.

* `uns_palcolor(adata, key, levels)` returns `{level: colour}` from the list,
  or `None`. It is used only when `obs[key]` is categorical and the list has
  exactly one colour per category (a misaligned list warns and is ignored;
  a non-categorical column has no defined alignment, so it is skipped).
* `level_colors(adata, key, levels, palette=, palcolor=, default_palette=,
  use_uns_colors=True)` is what a plot calls: `discrete_palette` as R does,
  then the stored colours on top for the levels they cover — *only* when the
  caller left `palette` at the function's default and `palcolor` unset.
  Levels the list does not cover (`show_na`'s `"NA"`) keep the palette's
  colour.
* Every plot that reads it has `use_uns_colors=True`; `False` gives R's
  colours exactly. Passing the default palette's name explicitly cannot be
  told apart from leaving it alone, so `palette="Paired"` still yields the
  stored colours — use `use_uns_colors=False`.
* Nothing is written back to `uns`: the stored list wins over the default
  palette, so writing one call's colours would fix the palette of every
  later call.

Wired so far into `cell_dim_plot` (groups, `stat_by`, the PAGA nodes),
`cell_dim_plot_3d`, `paga_plot`, `velocity_plot(group_by=)` and
`projection_plot`. The R parity fixtures build objects without stored
colours, so every parity test runs on R's palette.

## 3. Color algebra

Four functions, ported verbatim, that make multi-feature blending work:

| R | Python | What |
|---|---|---|
| `adjcolors(c, a)` | `adjcolors` | composite `c` at alpha `a` over white |
| `RGBA2RGB` | `rgba_to_rgb` | flatten RGBA onto a background |
| `Blend2Color` | `blend2color` | two colors under blend/average/screen/multiply |
| `BlendRGBList` | `blend_rgb_list` | fold a list down to one RGB |

`blend_rgb_list`'s reduction is unusual and must be exact: while more than one
element remains, the **last** is folded into every earlier one with weights
`a_i * (1 - 1/N)` and `a_last * (1/N)`. With three inputs the last contributes
1/3, then the new last contributes 1/2.

The four modes:

```
blend    out = (c1*a1 + c2*a2*(1-a1)) / A,  A = 1   # source-over compositing
average  out = clip((c1 + c2) / 2, 0, 1)
screen   out = 1 - (1-c1)*(1-c2)                    # brighter
multiply out = c1 * c2                              # darker
```

`luminance_text_color` ports SCP's label rule: `sum(channels_0_255) > 510 →
black else white`.

## 4. Themes

### `theme_scp`

White panel and figure, **black 1pt closed border on all four sides, no
protruding axis line**, no grid. Font sizes at `base_size=12`: text 12, title
14, subtitle 13 (left-aligned), axis title 13, axis text 12, strip 12.5, legend
title 12, legend text 11 — all scaled by `base_size / 12`.

Split across two mechanisms because matplotlib has two:

* `theme_scp_rc(theme)` → an rcParams dict, used with `mpl.rc_context` around
  figure construction so fonts are right from the start;
* `apply_theme(ax, theme)` → fixes what rcParams cannot express: the four-sided
  border, `set_box_aspect` for `aspect_ratio`, facecolor.

`aspect_ratio` in ggplot is panel height / panel width in *panel* units, which
is exactly `Axes.set_box_aspect`. Not `set_aspect`, which is in data units.

### `theme_blank`

The published-UMAP idiom: no frame, no ticks, and two little arrows in the
bottom-left corner spanning 15% of the panel, labelled outside it. In R this is
a `grobTree` of two `linesGrob`s with arrowheads plus two `textGrob`s, injected
via `annotation_custom`. Here it is `ax.annotate` in axes-fraction coordinates
(`add_coord_arrows`).

Both plot families special-case it in R:

```r
if (identical(theme_use, "theme_blank")) { theme_args$xlab <- xlab; theme_args$ylab <- ylab }
show_stat <- ifelse(identical(theme_use, "theme_blank"), FALSE, TRUE)
```

i.e. the axis labels move into the arrow grob and the stat subtitle is
suppressed. Keep both.

### Shadow text

ggrepel's `bg.color` / `bg.r` redraws a string eight times, offset radially by
`bg.r * fontsize`, before drawing the real glyphs. A single stroke of
`2 * bg.r * fontsize` is visually equivalent and far cheaper:

```python
text.set_path_effects([patheffects.withStroke(linewidth=2*bg_r*size, foreground=bg)])
```

That is `scp.theme.halo`. Defaults `label_fg="white"`, `label_bg="black"`,
`label_bg_r=0.1` give white bold text with a thin black halo — legible over any
point color, which is the whole point.

## 5. Label repulsion

ggrepel has no exact Python equivalent. Options, in order of preference:

1. **`adjustText`** — closest in spirit; `force_text`, `arrowprops` map onto
   `force`/`min.segment.length`.
2. **Hand-rolled 1-D repulsion** for heatmap `anno_mark` tracks, where labels
   move along one axis only. Sort targets, push apart to a minimum spacing,
   draw leader lines with `ConnectionPatch`. This is deterministic and better
   than a general solver for that case.
3. `force=0` (SCP's non-repel path) needs nothing — ggrepel is used there
   purely for the halo, which we get from path effects.
