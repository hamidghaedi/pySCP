# Agent handoff

You are picking up a port of the R package
[SCP](https://github.com/zhanghao-njmu/SCP)'s plotting layer to Python/AnnData.
The inspection is finished. The foundation is built and tested. Your job is to
work through the milestones.

## Before you write code

1. Read [`00_overview.md`](00_overview.md), [`01_architecture.md`](01_architecture.md)
   and [`02_data_contract.md`](02_data_contract.md). Roughly 15 minutes. Do not
   skip the data contract — it is where the silent-wrongness bugs live.
2. Read [`../src/scp/pl/_dim.py`](../src/scp/pl/_dim.py)'s `cell_dim_plot`. It
   is the reference pattern; every other function follows its shape.
3. Pick your milestone from [`07_milestones.md`](07_milestones.md) and read the
   matching brief in [`porting_briefs/`](porting_briefs/) **in full** before
   writing anything. The briefs are long because the detail is the point.

Clone the R source for reference:

```bash
git clone --depth 1 https://github.com/zhanghao-njmu/SCP.git /tmp/SCP
# the whole plotting layer is one file:
sed -n '1328,2042p' /tmp/SCP/R/SCP-plot.R
```

## Ground rules

**Signatures are the contract.** Every stub carries its final signature,
derived from reading the R source. If you change one, update
[`04_api_mapping.md`](04_api_mapping.md) in the same commit and say why.

**Draw into an `Axes`, never create a `Figure` in a drawing function.**
`PanelGrid` owns figure creation and sizing. This is the one architectural rule
that, if broken, makes multi-panel figures impossible to fix later.

**Never call `ax.legend()`.** Register handles with the `LegendColumn`. A
per-axes legend gets duplicated N times in a grid.

**Always go through `fetch_data` / `as_ordered_categorical`.** Reaching into
`adata.obs[...]` directly is how level order gets silently sorted and every
palette shifts.

**Port the behaviour, not the implementation.** Much of SCP's R is worked
around ggplot/gtable limitations that don't exist here — legend grob surgery,
`eval(parse(text=...))` closure building, throwaway plots rendered just to
harvest computed positions. Reproduce what the reader sees; use native tools to
get there. The briefs flag which is which.

**Fix the known R bugs, and say so in the docstring.** The briefs list them:
the `values`/`valus` typo in `same.y.lims`, the undefined `status` in
`calculate_coexp`, the `keep_scale="all"` branch that drops its `is.finite`
filter, the duplicated `lab_layer` in `VelocityPlot`, the
`edge_highlight`/`transition_highlight` mix-up in `GraphPlot`.

**Optional dependencies stay optional.** Import heavy deps inside the function
and raise a message naming the extra (`pip install scp-plot[graph]`). A user
who wants `cell_dim_plot` should not be made to install `pygam`.

**Never block on input.** R's `askYesNo()` guards become exceptions gated by
`force=True`.

## Definition of done, per function

- [ ] Signature matches `04_api_mapping.md`
- [ ] Docstring names the R original, its line number, and any deliberate
      divergence
- [ ] Structural tests: panel count, shared axis limits, level order, legend
      labels (see [`06_testing.md`](06_testing.md))
- [ ] Numeric fixtures for any new pure transform
- [ ] One rendered example committed under `tests/fixtures/`
- [ ] `ruff check . && mypy src/scp && pytest -q` clean
- [ ] `04_api_mapping.md` status flipped from **stub** to **done**

## When you get stuck

The most common failure is trying to reproduce an R *mechanism* rather than an
R *outcome*. If you find yourself building something to extract and re-glue
legend grobs, or rendering a throwaway plot to read positions back out, stop —
matplotlib can compute those positions directly, and the brief will usually say
how.

The second most common is a figure that looks right but is subtly wrong. When
that is a possibility, do not reach for an image diff. Have the R side dump the
intermediate data frame (`p$data`, or `matrix_list` for a heatmap) as CSV and
compare the numbers. That localises the discrepancy to a transform instead of
to "the picture looks different".

## Licence

SCP is GPL-3; this port is a derivative work and must stay GPL-3. Credit Hao
Zhang and link the upstream repository in the README and in the docstring of
every ported function.
