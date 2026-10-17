#!/usr/bin/env Rscript
# Parity fixtures for the 3-D embedding plots: CellDimPlot3D and
# FeatureDimPlot3D from SCP (Hao Zhang, NJMU, GPL-3;
# https://github.com/zhanghao-njmu/SCP), checked against
# scp.pl.cell_dim_plot_3d / scp.pl.feature_dim_plot_3d.
#
# Regenerates tests/fixtures/parity/audit/dim3d/:
#   input.csv          per-cell inputs the Python side needs: PCA dims 1-3,
#                      CellType, CellTypeNA (every 10th cell set to NA), and
#                      the log-normalised Ins1 / Gcg plus the signed S_score
#   input_levels.json  declared factor levels of CellType
#   cell3d.json        plotly_build() traces of CellDimPlot3D (name, mode,
#                      marker colour/size, member cells in trace order) and
#                      the scene layout (titles, ranges, title text)
#   cell3d_na.json     the same for a group column containing NA
#   feature3d.json     the plotly_build() trace of FeatureDimPlot3D: marker
#                      colour values (NA kept as null), colorscale, cmin/cmax,
#                      plus the per-feature dropdown colour arrays
#
# pancreas_sub carries only a 50-column PCA and a 2-column UMAP, so the 3-D
# plots use PCA dims 1:3. Its `data` slot holds raw counts, so the object is
# log-normalised first (NormalizeData) and both sides read that matrix.
#
# Usage, from the repository root:
#   Rscript tools/parity/audit_dim3d.R [out_dir]

suppressPackageStartupMessages({
  library(SCP); library(Seurat); library(plotly); library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) > 0) args[[1]] else "tests/fixtures/parity/audit/dim3d"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

data("pancreas_sub", package = "SCP")
srt <- NormalizeData(pancreas_sub, verbose = FALSE)
cells <- colnames(srt)

na_idx <- seq(1, ncol(srt), by = 10)
v <- as.character(srt$CellType)
v[na_idx] <- NA
srt$CellTypeNA <- factor(v, levels = levels(srt$CellType))

emb <- Embeddings(srt, "PCA")[, 1:3]
dat <- LayerData(srt, layer = "data")
input <- data.frame(
  cell = cells,
  PC_1 = emb[, 1], PC_2 = emb[, 2], PC_3 = emb[, 3],
  CellType = as.character(srt$CellType),
  CellTypeNA = as.character(srt$CellTypeNA),
  Ins1 = dat["Ins1", cells], Gcg = dat["Gcg", cells],
  S_score = srt$S_score,
  check.names = FALSE
)
write.csv(input, file.path(out, "input.csv"), row.names = FALSE, na = "")
write_json(list(CellType = levels(srt$CellType)), file.path(out, "input_levels.json"))

# plotly_build rewrites the hover "\n" as "<br />"; keep only the barcode
cell_of <- function(text) sub("(<br />|\n).*$", "", sub("^Cell:", "", text))

dump_cell3d <- function(p) {
  b <- plotly_build(p)$x
  traces <- lapply(b$data, function(tr) {
    list(
      name = tr$name, mode = tr$mode, type = tr$type,
      marker_color = tr$marker$color, marker_size = tr$marker$size,
      cells = cell_of(tr$text)
    )
  })
  sc <- b$layout$scene
  list(
    traces = traces,
    title = b$layout$title$text,
    xaxis = list(title = sc$xaxis$title, range = unlist(sc$xaxis$range)),
    yaxis = list(title = sc$yaxis$title, range = unlist(sc$yaxis$range)),
    zaxis = list(title = sc$zaxis$title, range = unlist(sc$zaxis$range))
  )
}

p1 <- CellDimPlot3D(srt, group.by = "CellType", reduction = "PCA")
write_json(dump_cell3d(p1), file.path(out, "cell3d.json"), auto_unbox = TRUE, digits = NA)

p2 <- CellDimPlot3D(srt, group.by = "CellTypeNA", reduction = "PCA")
write_json(dump_cell3d(p2), file.path(out, "cell3d_na.json"), auto_unbox = TRUE, digits = NA)

feats <- c("Ins1", "Gcg", "S_score")
p3 <- FeatureDimPlot3D(srt, features = feats, reduction = "PCA")
b3 <- plotly_build(p3)$x
tr <- b3$data[[1]]
menus <- b3$layout$updatemenus
gene_buttons <- menus[[2]]$buttons
sc <- b3$layout$scene
feature3d <- list(
  trace_names = vapply(b3$data, function(t) t$name, ""),
  name = tr$name,
  marker_color = tr$marker$color,
  colorscale = tr$marker$colorscale,
  cmin = tr$marker$cmin, cmax = tr$marker$cmax,
  colorbar_title = tr$marker$colorbar$title$text,
  marker_size = tr$marker$size,
  cells = cell_of(tr$text),
  button_labels = vapply(gene_buttons, function(bt) bt$label, ""),
  button_colors = lapply(gene_buttons, function(bt) bt$args[[1]]$marker$color),
  title = b3$layout$title$text,
  xaxis = list(title = sc$xaxis$title, range = unlist(sc$xaxis$range))
)
write_json(feature3d, file.path(out, "feature3d.json"), auto_unbox = TRUE, digits = NA,
           na = "null")
message("dim3d fixtures -> ", normalizePath(out))
