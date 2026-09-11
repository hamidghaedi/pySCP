# Testing strategy

The risk in this port is not crashes — it is figures that look plausible and
are subtly wrong: a palette shifted by one, a z-score off by
`sqrt(n/(n-1))`, a legend ordered alphabetically instead of by appearance.
Those never raise. So the test suite is weighted towards **numeric fixtures**,
not image comparison.

## Three layers

### 1. Golden numeric fixtures (the important ones)

Every pure transform gets its expected output pinned as JSON/npy, generated
once from R and committed. These catch the silent-wrongness class.

| What | Assert |
|---|---|
| `discrete_palette(levels, "Paired")` | for `n <= 12`, equals `PALETTES["Paired"].colors[:n]` **exactly** — verbatim slice, not interpolation |
| `discrete_palette` with `n > len(palette)` | length `n`, endpoints equal the palette endpoints |
| `color_ramp` | matches `grDevices::colorRampPalette` within one 8-bit step |
| `matrix_process(M, "zscore")` | uses **ddof=1**; `[[1,2,3]]` → `[-1, 0, 1]` |
| `matrix_process(M, "log2fc")` | `log2(M / rowmean)` including the `-inf` on all-zero rows |
| `color_limits(M, "zscore")` | symmetric, `ceil(min(|q01|,|q99|) * 2) / 2` |
| `blendcolors(["#FF0000","#00FF00"], mode)` | all four modes, against R |
| `adjcolors("#FF0000", 0.1)` | `"#FFE6E6"` |
| `blend_rgb_list` with 3+ inputs | the `1/N` pairwise fold, not a naive mean |
| `gsea_scores` | running score matches `fgsea` / `gseapy` for a known set |
| `shorten_segments` | pure geometry, exact |

Generating fixtures from R, once, on a machine with SCP installed:

```r
library(SCP); library(jsonlite)
out <- list(
  paired3  = unname(palette_scp(c("a","b","c"), palette = "Paired")),
  spectral = unname(palette_scp(palette = "Spectral", n = 100)),
  blend    = sapply(c("blend","average","screen","multiply"),
                    function(m) blendcolors(c("red","green"), mode = m)),
  zscore   = as.vector(matrix_process(matrix(c(1,2,3,4,4,4), 2, 3, byrow=TRUE), "zscore"))
)
write_json(out, "tests/fixtures/r_golden.json", auto_unbox = TRUE, digits = 12)
```

Commit `r_golden.json`; do not require R to run the suite.

### 2. Structural assertions (cheap, catch most regressions)

For each plot function, assert the *shape* of what it produced rather than its
pixels:

```python
fig = scp.pl.cell_dim_plot(adata, "leiden", split_by="batch")
assert len(fig.axes) == adata.obs["batch"].nunique()
assert fig.axes[0].get_xlim() == fig.axes[1].get_xlim()      # shared limits
assert fig.axes[0].collections[0].get_offsets().shape[0] == adata.n_obs  # all cells, incl. background
```

Things worth asserting this way:

* panel count and grid shape for every `split_by` × `group_by` combination;
* **shared axis limits** across split panels (SCP locks them to the global
  embedding range);
* `cell_dim_plot` draws *all* cells in every split panel; `feature_dim_plot`
  draws only the split's own;
* legend labels carry counts when `show_stat=True` and the `"1: Name"` prefix
  when `label=True, label_insitu=False`;
* level order equals `pd.unique`, not `sorted`;
* `bg_cutoff=0` really does NaN-mask zeros in `feature_dim_plot`;
* `force=False` raises on >100 levels / >50 features instead of prompting;
* heatmap `figsize()` is the sum of its declared parts, to within a float
  epsilon.

### 3. Image comparison (last, and sparingly)

`pytest-mpl` with a **generous** tolerance (`tolerance=20`) on a handful of
canonical figures, one per family. Purpose: catch gross layout breakage, not
to pin appearance. Tight image tests on a package under active development
become a maintenance tax and get `--mpl-generate-path`'d away, which is worse
than not having them.

Always pin the backend and the font in `conftest.py`, or the baselines will
differ across machines:

```python
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
```

## Fixture data

`scanpy.datasets.pbmc3k_processed()` for most things — it has `X_umap`,
`X_pca`, `louvain`, and is small. For trajectory and velocity work, the
pancreas dataset SCP itself uses is the natural match:
`scv.datasets.pancreas()` (it is where `pancreas_sub` came from).

A synthetic 600-cell AnnData built with a seeded RNG is enough for everything
structural, runs in milliseconds, and does not need a network. Build it in
`conftest.py` and prefer it.

## Comparing against R directly

Where a figure's correctness is genuinely in question, the highest-value check
is not an image diff — it is comparing the **data the plot is about to draw**.
Have the R side write out the intermediate frame (`p$data` for a ggplot,
`matrix_list` for a heatmap) as CSV, and assert the Python equivalent matches.
That localises a discrepancy to a transform instead of to "the picture looks
different".

## CI

```
ruff check . && mypy src/scp && pytest -q
```

Image tests behind a marker (`pytest -m mpl`) so the default run stays fast and
deterministic.
