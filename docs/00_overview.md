# Porting SCP's plotting layer to scanpy — overview

## What this is

[SCP](https://github.com/zhanghao-njmu/SCP) (Hao Zhang, NJMU, GPL-3) is an R
single-cell pipeline whose plotting layer is unusually good: 229 curated
palettes, a consistent theme, and about thirty figure functions that share one
grammar for grouping, splitting, faceting, labelling and legend composition.
The whole layer is one 14,794-line file, `R/SCP-plot.R`, built on ggplot2 +
ComplexHeatmap + patchwork, and it reads Seurat objects.

This repository is the spec and scaffold for a Python package that reproduces
that plotting layer on top of **AnnData**, so it can be used beside scanpy.

## Decisions already made

| Question | Decision |
|---|---|
| Rendering stack | **matplotlib** (+ scipy/sklearn for stats). Not plotnine. |
| Scope | All of it — core plots, heatmaps, trajectory, enrichment. |
| API style | Standalone `scp.pl.*` namespace, snake_case, AnnData-native, returns matplotlib objects. Coexists with `sc.pl.*`. |
| Deliverable | These docs plus a runnable scaffold with the palette/theme/data layer already working. |

The matplotlib choice has one consequence that shapes everything else: **there
is no "combine finished plots" operation.** R builds N independent ggplots and
glues them with patchwork; matplotlib cannot do that. So the architecture is
inverted — plotting functions draw *into* an `Axes` that a layout object has
already sized. Build [`scp.layout.PanelGrid`](../src/scp/layout.py) before any
plot function, not after.

## Read these in order

| Doc | What it settles |
|---|---|
| [`01_architecture.md`](01_architecture.md) | Module layout, the draw-into-axes inversion, legend composition, sizing |
| [`02_data_contract.md`](02_data_contract.md) | `fetch_data`, `default_reduction`, layer/slot semantics, categorical order, `uns` conventions |
| [`03_theme_and_palette.md`](03_theme_and_palette.md) | `theme_scp`, `theme_blank`, the palette resolver, color algebra |
| [`04_api_mapping.md`](04_api_mapping.md) | Every R function → its Python name, signature and status |
| [`porting_briefs/`](porting_briefs/) | Per-family implementation detail, read when you pick up that milestone |
| [`06_testing.md`](06_testing.md) | Golden fixtures, image comparison, what to assert |
| [`07_milestones.md`](07_milestones.md) | Ordered work plan with dependencies |
| [`08_agent_handoff.md`](08_agent_handoff.md) | **Start here if you are an agent picking this up** |

## What already works

The first pass over the whole layer is complete: 27 of the 28 functions in
`scp.pl` are implemented. `docs/04_api_mapping.md` carries the per-function
status, and the table at the end of `docs/07_milestones.md` lists what is
deliberately not ported and why.

Options within an implemented function that are not ported raise
`NotImplementedError` naming the brief section that specifies them. That is
deliberate: the signatures are the contract, they were derived from reading the
R source, and a missing option should announce itself rather than quietly do
something else.

`notebooks/01_parity_foundation.ipynb` runs both packages on `pancreas_sub` and
compares them; see the README's Parity section for what has actually been
checked against R and what has only been implemented.

## Licence

SCP is GPL-3. This port is a derivative work of its plotting layer and must
also be GPL-3. Credit Hao Zhang and link the upstream repository in the README
and in the docs of every ported function.
