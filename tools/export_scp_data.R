#!/usr/bin/env Rscript
# Export SCP's development Seurat objects to an intermediate form that
# tools/build_parity_h5ad.py assembles into .h5ad.
#
# This replicates the mapping SCP's own srt_to_adata() performs
# (SCP-analysis.R:5351) rather than calling it, because that function routes
# through reticulate and will try to build its own Python environment when one
# is not already registered. The mapping itself is short, and writing it out
# explicitly makes it auditable against docs/02_data_contract.md.
#
# One deliberate divergence: X carries the `data` slot, not `counts`.
# srt_to_adata defaults to slot_X = "counts", but every plotting function in
# SCP-plot.R defaults to slot = "data", and pySCP reads adata.X when
# layer = None. Writing counts into X would leave the two sides plotting
# different numbers for the same call. Counts are kept as a layer.
#
# Reduction names are kept verbatim, as srt_to_adata does: pancreas_sub's
# reductions are `PCA` and `UMAP`, so the obsm keys are `PCA` and `UMAP` and
# NOT the scanpy-conventional `X_pca` / `X_umap`.

suppressPackageStartupMessages({
  library(Seurat); library(Matrix); library(jsonlite)
})

# SCP itself is only needed for its data/. Set SCP_RDA_DIR to a checkout's
# data/ directory to run this without installing the package and its 43
# imports; otherwise the installed SCP is used.
rda_dir <- Sys.getenv("SCP_RDA_DIR", unset = NA)
load_dataset <- function(name) {
  e <- new.env()
  if (!is.na(rda_dir)) {
    load(file.path(rda_dir, paste0(name, ".rda")), envir = e)
  } else {
    suppressPackageStartupMessages(library(SCP))
    utils::data(list = name, package = "SCP", envir = e)
  }
  e[[name]]
}

args <- commandArgs(trailingOnly = TRUE)
out_root <- if (length(args) > 0) args[[1]] else "notebooks/data/_raw"

# Seurat 5 renamed slot= to layer=; support both so this runs on either.
get_mat <- function(srt, assay, which) {
  if (utils::packageVersion("SeuratObject") >= "5.0.0") {
    SeuratObject::LayerData(srt, assay = assay, layer = which)
  } else {
    Seurat::GetAssayData(srt, assay = assay, slot = which)
  }
}

# A factor's level ORDER is the contract: it drives palette index, legend order
# and facet order on both sides. write.csv would drop it, so levels go out
# separately and are restored as an ordered pandas Categorical.
factor_levels <- function(df) {
  out <- list()
  for (nm in colnames(df)) if (is.factor(df[[nm]])) out[[nm]] <- levels(df[[nm]])
  out
}

write_mtx <- function(m, path) writeMM(as(m, "CsparseMatrix"), path)

export_one <- function(name) {
  message("== ", name)
  srt <- load_dataset(name)
  d <- file.path(out_root, name)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)

  assay <- DefaultAssay(srt)
  # transpose: Seurat is genes x cells, AnnData is cells x genes
  write_mtx(t(get_mat(srt, assay, "data")),   file.path(d, "X_data.mtx"))
  write_mtx(t(get_mat(srt, assay, "counts")), file.path(d, "layer_counts.mtx"))

  extra <- setdiff(names(srt@assays), assay)
  for (a in extra) {
    ok <- tryCatch({ write_mtx(t(get_mat(srt, a, "counts")),
                               file.path(d, paste0("layer_", a, ".mtx"))); TRUE },
                   error = function(err) FALSE)
    message("   assay ", a, if (ok) " -> layer" else " -> skipped")
  }

  obs <- srt@meta.data
  write.csv(obs, file.path(d, "obs.csv"), row.names = TRUE)
  var <- srt[[assay]]@meta.features
  vf <- VariableFeatures(srt, assay = assay)
  if (length(vf) > 0) var[["highly_variable"]] <- rownames(var) %in% vf
  write.csv(var, file.path(d, "var.csv"), row.names = TRUE)

  reductions <- names(srt@reductions)
  for (r in reductions) {
    write.csv(srt[[r]]@cell.embeddings, file.path(d, paste0("obsm_", r, ".csv")),
              row.names = TRUE)
  }

  meta <- list(
    dataset = name,
    default_assay = assay,
    n_obs = ncol(srt), n_var = nrow(srt[[assay]]),
    obs_names = colnames(srt),
    var_names = rownames(srt[[assay]]),
    obs_levels = factor_levels(obs),
    reductions = reductions,
    # Seurat's @key has no AnnData counterpart; carry it so the axis labels
    # on both sides can be made to agree (docs/02_data_contract.md section 2).
    reduction_keys = setNames(lapply(reductions, function(r) Key(srt[[r]])), reductions),
    variable_features = vf,
    misc = names(srt@misc),
    scp_version = tryCatch(as.character(utils::packageVersion("SCP")),
                           error = function(e) NA_character_),
    seurat_version = as.character(utils::packageVersion("Seurat"))
  )
  write_json(meta, file.path(d, "meta.json"), auto_unbox = TRUE, digits = NA)
  message("   ", ncol(srt), " cells x ", nrow(srt[[assay]]), " genes | reductions: ",
          paste(reductions, collapse = ", "))
}

dir.create(out_root, recursive = TRUE, showWarnings = FALSE)
for (nm in c("pancreas_sub", "panc8_sub")) export_one(nm)
message("done -> ", normalizePath(out_root))
