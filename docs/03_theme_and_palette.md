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

R cuts the literal vector `1:100` rather than the data when `matched=False`,
which is a bug that happens to be harmless because the result is only ever used
as an evenly spaced ramp. `continuous_palette()` returns the ramp directly.

`matched=True` returns one color per element of `x` — used by the blend path
and by the correlation blocks in `feature_cor_plot`.

### NA

`na_color` defaults to `"grey80"` = `#CCCCCC`. `na_keep=False` (the default)
strips the `"NA"` entry from the returned mapping; `show_na=True` upstream
promotes NaN to a real level instead.

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
