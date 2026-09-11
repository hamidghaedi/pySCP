# The data contract: Seurat → AnnData

Everything the plotting layer reads goes through
[`src/scp/fetch.py`](../src/scp/fetch.py). This document is the reasoning
behind it.

## 1. SCP already has a conversion layer — read it first

`SCP/R/SCP-analysis.R` lines 5351–5681 hold `srt_to_adata` and `adata_to_srt`.
They tell you exactly what the R code assumes about AnnData.

### `adata_to_srt` (AnnData → Seurat)

| AnnData | Seurat | Note |
|---|---|---|
| `X` | `RNA@counts` **and** `@data` | transposed; **`X` lands in `counts` regardless of whether it is log-normalized** |
| `obs` | `meta.data` | column names put through `make.names()` — `n-counts` → `n.counts` |
| `layers[k]` | a separate Assay named `k`, in its `counts` slot | |
| `obsm[k]` | reduction `sub("^X_", "", k)`, key `gsub("_","",k) + "_"` | so `X_umap` → name `umap`, key `umap_`, columns `umap_1`, `umap_2` |
| `obsp[k]` | `@graphs[[k]]` | |
| `var` | `RNA@meta.features` | |
| `varm[k]` | `RNA@misc$feature.loadings[[k]]` | **not** into `@reductions[[..]]@feature.loadings` |
| `uns[k]` | `@misc[[k]]` | recursive `py_to_r`; failures are warned and skipped |
| `raw` | — | **never read** |
| images | — | no spatial support |

### `srt_to_adata` (Seurat → AnnData)

Reduction names are kept **verbatim**, with no `X_` prefix added — the reverse
of the import path. `X` is forced dense float32. Feature loadings, `stdev` and
images are all commented out and lost.

**Consequence for us:** after a Seurat→AnnData round trip, a log-normalized
matrix is sitting in what everyone calls `counts`. `group_heatmap`'s
`lib_normalize` must keep SCP's guard: if the "counts" matrix is non-integer,
set all library sizes to 1 and warn.

## 2. The `key_` convention

Seurat's `@key` (`"UMAP_"`) has no AnnData counterpart, and two conventions
coexist in SCP: native SCP reductions use uppercase (`RunUMAP2` → `UMAP_`),
while round-tripped ones get lowercase (`umap_`). So `X_umap` does **not**
reliably become `UMAP_1`/`UMAP_2`.

Treat the axis-label prefix as a separate, settable attribute. `REDUCTION_KEY`
in `fetch.py` is a lookup table with a deterministic fallback
(`re.sub(r"^X_", "", k).replace("_", "") + "_"`), and callers may override.

One real latent bug in R worth not reproducing: `FeatureDimPlot` builds its
embedding-name map from the *stored* column names but reads coordinates using
*regenerated* ones. These disagree for any obsm key containing an underscore
(`X_draw_graph_fr` → stored `draw_graph_fr_1`, regenerated `drawgraphfr_1`).
We use the regenerated form in both places.

## 3. `fetch_data` — resolution order

**SCP does not use Seurat's `FetchData` order.** Seurat resolves keyed vars →
`meta.data` → features. SCP's plotting layer hand-builds
`cbind(dat_gene, dat_meta, dat_embedding)`, so the order is:

1. `adata.var_names` (features)
2. `adata.obs.columns`
3. embedding coordinates

A name in both `var_names` and `obs` is **not** an error — both columns are
materialised, `cbind` produces duplicate names, and downstream `[, features]`
takes the first, so **the gene wins** with only a warning. Get this backwards
and every collision flips.

Other rules, all faithful:

* unresolvable names are **warned and dropped**, not an error; only an
  all-missing set raises;
* columns come back in resolution-block order, not the caller's order;
* `cells=` intersects, so unknown cell names are silently dropped and the
  survivors keep **object order**, not request order;
* a factor `obs` column passed as a *feature* (i.e. where a number is required)
  is a hard error.

Coordinate names accept two grammars:

| Form | Index base | Example |
|---|---|---|
| `<KEY><n>` | 1-based | `UMAP_1`, `PC_2` |
| `<obsm_key>:<i>` | 0-based | `X_umap:0` |

The second is unambiguous and is the recommended internal representation.

## 4. `default_reduction`

Reproduces `SCP::DefaultReduction`, including two behaviours that surprise
people:

* a single qualifying reduction is returned **without** consulting the priority
  list — so an object with only `X_pca` gets `X_pca` even though PCA is ninth;
* ties break on **fewest columns** — a 2-column UMAP beats a 3-column one, and
  any 2-column embedding beats a 50-column PCA.

Priority order: `umap, tsne, dm, diffmap, phate, pacmap, trimap, largevis, fr,
draw_graph_fr, pca, svd, ica, nmf, mds, glmpca`, each probed bare and with an
`X_` prefix. Override with `adata.uns["default_reduction"]` (the analogue of
`srt@misc[["Default_reduction"]]`, which already round-trips through `uns`).

R's fuzzy `agrep` step is dropped; exact then case-insensitive substring is
enough, and `difflib` fuzziness in a plotting API is a footgun.

## 5. assay / slot → layer

| Seurat | Here |
|---|---|
| `slot="data"` | `layer=None` → `adata.X` |
| `slot="counts"` | `layer="counts"` |
| `slot="scale.data"` | `layer="scaled"` |
| `assay="spliced"` | `layer="spliced"` |
| `assay=` (multimodal, different `var_names`) | a separate AnnData / MuData modality |
| `DefaultAssay(srt)` | **nothing** — the AnnData *is* the assay |

So: port every SCP `assay=` argument to nothing and every `slot=` to `layer=`,
with `use_raw` as a separate boolean.

Defaults differ by family and must be kept: the dim/stat plots default to
`slot="data"` (→ `layer=None`), the heatmaps to `slot="counts"` (→
`layer="counts"`) because they then do their own normalization.

One faithful ugliness: heatmap **cell-annotation tracks hardcode `@data`**,
ignoring the user's `slot=` (`SCP-plot.R:8250, 9396, 10298`). Reproduce it
(hardcode `adata.X`) or fix it — but document which you chose.

`infer_data_type()` ports `check_DataType`; the heatmaps genuinely depend on
it for the `lib_normalize` guard.

## 6. Categorical order

```r
factor(x, levels = unique(x))   # order of FIRST APPEARANCE, not sorted
```

R deliberately bypasses its own sorting. pandas sorts by default, so
`pd.Categorical(s)` will silently reshuffle **every palette and legend**
relative to the R output. Always go through
`scp.fetch.as_ordered_categorical`, which trusts an existing Categorical's
order and otherwise uses `pd.unique`.

Level order then drives palette index, legend order, facet order, heatmap
column order and x-axis order. `show_na=True` promotes missing values to a
real `"NA"` level appended last, so the palette gives it a color; otherwise
NaN falls through to `bg_color`.

scanpy already persists `adata.uns[f"{key}_colors"]`, a list aligned to
`obs[key].cat.categories`. Read it first when present — it is the AnnData-native
equivalent of a named `palcolor` — and write it back, so scanpy and this
package agree.

## 7. Where computed results live

Every reader tries, in order: (1) an explicit key, (2) the native
scanpy/scvelo/decoupler key, (3) `adata.uns["scp"][...]`. That keeps the
package usable on objects that have never touched it. See
[`src/scp/io.py`](../src/scp/io.py).

| SCP slot | Native AnnData key | Adapter |
|---|---|---|
| `@tools$DEtest_<g>$AllMarkers_<test>` | `uns["rank_genes_groups"]` | `io.de_from_rank_genes_groups` |
| `@tools$Enrichment_<g>_<test>$enrichment` | none — define `uns["scp"]["enrichment_<g>"]` | `io.enrichment_from_gseapy` |
| `@tools$GSEA_<g>_<test>` | none — `uns["scp"]["gsea_<g>"]` | `io.enrichment_from_gseapy` |
| `@misc$paga` | `uns["paga"]` — **identical keys** | `io.read_paga` |
| `@graphs[[..]]` | `obsp["connectivities"]` / `["distances"]` | direct |
| reduction `"<mode>_<RED>"` (velocity) | `obsm[f"{vkey}_{basis}"]` | direct |
| `DynamicFeatures_<lineage>` | `uns["scp"]["dynamic_features_<l>"]` | — |
| slingshot pseudotime | `obs` columns, whatever produced them | — |

`adata.uns["paga"]` deserves emphasis: SCP reads `connectivities`,
`connectivities_tree`, `groups` and `transitions_confidence`, which are exactly
scanpy's and scvelo's own keys. `paga_plot` needs no conversion at all.

## 8. Gotchas that will silently break a naive port

1. Resolution order is **genes first**, not obs first.
2. Transpose everywhere: Seurat is genes × cells, AnnData is cells × genes.
3. `dims` is **1-based** in the API; subtract one for numpy.
4. SCP takes cell names from `srt@assays[[1]]` — the *first* assay, not the
   default one. Collapses to `adata.obs_names` here; just don't reintroduce it.
5. Coordinate column names are **regenerated**, not read (§2).
6. After a round trip, log-normalized data sits in `counts` (§1).
7. `obs` column names are `make.names()`-mangled on the Python→R path.
8. Level order is first-appearance (§6).
9. Missing features warn-and-drop; missing cells silently intersect (§3).
10. `default_reduction` short-circuits on one reduction and tie-breaks on
    fewest dims (§4).
11. `bg_cutoff` defaults to **0** in `feature_dim_plot` and NaN-masks
    everything `<= 0` *before* plotting. Signed scores need `bg_cutoff=None`.
12. R's `>50 features` / `>100 levels` guards are interactive `askYesNo()`
    prompts. In a library these must be exceptions gated by `force=True`,
    never a blocking prompt.
13. `utils.R:831`'s `as_matrix()` fast path is dead code (it tests the base
    `matrix` function). Don't port the "optimization"; it never ran.
