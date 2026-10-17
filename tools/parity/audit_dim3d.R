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
#   extra_input.csv    Phase (the split column) and the three slingshot
#                      pseudotimes, with Phase's declared levels in
#                      extra_levels.json
#   cell3d_highlight.json, cell3d_highlight_true.json
#                      CellDimPlot3D with cells.highlight given as a cell
#                      vector (in a non-object order) and as TRUE: the trace
#                      names and the highlight trace
#   cell3d_lineages.json
#                      CellDimPlot3D(lineages = Lineage1..3): the LOESS line
#                      traces (x/y/z, colour, width) after the range filter
#   feature3d_split.json
#                      FeatureDimPlot3D(split.by = "Phase", cells.highlight =)
#                      traces, title, scene ranges and both dropdowns
#
# The lineages come from RunSlingshot(group.by = "SubCellType", reduction =
# "UMAP") with seed 11, exactly as tools/parity/audit_traj.R makes them; any
# pseudotime would do, since CellDimPlot3D only smooths the 3-D coordinates
# against it.
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

# --- highlight, lineages, split --------------------------------------------------
set.seed(11)
srt <- RunSlingshot(srt, group.by = "SubCellType", reduction = "UMAP", show_plot = FALSE)
lineages <- paste0("Lineage", 1:3)
extra <- data.frame(cell = cells, Phase = as.character(srt$Phase), check.names = FALSE)
for (l in lineages) extra[[l]] <- srt@meta.data[[l]]
write.csv(extra, file.path(out, "extra_input.csv"), row.names = FALSE, na = "")
write_json(list(Phase = levels(srt$Phase)), file.path(out, "extra_levels.json"))

hl <- rev(cells[seq(3, length(cells), by = 41)])
# the level traces are cell3d.json's again; keep their names and the highlight trace
dump_highlight <- function(p) {
  b <- plotly_build(p)$x
  tr <- b$data[[length(b$data)]]
  list(
    trace_names = vapply(b$data, function(t) t$name, ""),
    highlight = list(
      name = tr$name, mode = tr$mode, marker_color = tr$marker$color,
      marker_size = tr$marker$size, marker_symbol = tr$marker$symbol,
      cells = cell_of(tr$text)
    )
  )
}
write_json(dump_highlight(CellDimPlot3D(srt, group.by = "CellType", reduction = "PCA",
                                        cells.highlight = hl)),
           file.path(out, "cell3d_highlight.json"), auto_unbox = TRUE, digits = NA)
write_json(dump_highlight(CellDimPlot3D(srt, group.by = "CellTypeNA", reduction = "PCA",
                                        cells.highlight = TRUE, sizes.highlight = 3,
                                        cols.highlight = "red")),
           file.path(out, "cell3d_highlight_true.json"), auto_unbox = TRUE, digits = NA)

p4 <- CellDimPlot3D(srt, group.by = "CellType", reduction = "PCA", lineages = lineages)
b4 <- plotly_build(p4)$x
lin <- Filter(function(tr) identical(tr$mode, "lines"), b4$data)
write_json(lapply(lin, function(tr) list(
  name = tr$name, x = tr$x, y = tr$y, z = tr$z,
  line_color = tr$line$color, line_width = tr$line$width
)), file.path(out, "cell3d_lineages.json"), auto_unbox = TRUE, digits = NA)

p5 <- FeatureDimPlot3D(srt, features = c("Ins1", "Gcg"), reduction = "PCA",
                       split.by = "Phase", cells.highlight = hl)
b5 <- plotly_build(p5)$x
menus <- b5$layout$updatemenus
sc <- b5$layout$scene
write_json(list(
  traces = lapply(b5$data, function(tr) list(
    name = tr$name, cells = cell_of(tr$text), marker_size = tr$marker$size,
    marker_color = if (is.numeric(tr$marker$color)) NULL else tr$marker$color,
    marker_symbol = tr$marker$symbol
  )),
  title = b5$layout$title$text,
  ranges = list(x = unlist(sc$xaxis$range), y = unlist(sc$yaxis$range),
                z = unlist(sc$zaxis$range)),
  split_buttons = lapply(menus[[1]]$buttons, function(bt) list(
    label = bt$label, name = bt$args[[1]]$name, title = bt$args[[2]]$title$text,
    # y and z are masked exactly as x is, so x carries the split
    x = bt$args[[1]]$x
  )),
  feature_labels = vapply(menus[[2]]$buttons, function(bt) bt$label, "")
), file.path(out, "feature3d_split.json"), auto_unbox = TRUE, digits = NA, na = "null")

message("dim3d fixtures -> ", normalizePath(out))
