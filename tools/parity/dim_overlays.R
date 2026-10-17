#!/usr/bin/env Rscript
# Dump the numbers behind CellDimPlot / FeatureDimPlot's optional layers, so
# tests/test_dim_parity.py can compare the Python port against SCP without R.
#
#   Rscript tools/parity/dim_overlays.R tests/fixtures/parity/dim
#
# Every fixture is an intermediate frame (ggplot_build layer data, a stat
# subplot's data, a ggforce enclosure), never an image. pancreas_sub is SCP's
# own development object; the Python side reads the same cells from the h5ad
# that tools/export_scp_data.R + tools/build_parity_h5ad.py produce.
#
# Inputs that pancreas_sub does not carry (a pseudotime, a PAGA result, a
# velocity embedding) are built here from quantities both sides already have,
# so the Python test can rebuild them identically rather than read them.

suppressPackageStartupMessages({
  library(SCP); library(Seurat); library(ggplot2); library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) > 0) args[[1]] else "tests/fixtures/parity/dim"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
# nine significant digits: far below every tolerance, and it halves the files
w <- function(df, name) {
  num <- vapply(df, is.numeric, logical(1))
  df[num] <- lapply(df[num], signif, digits = 9)
  write.csv(df, file.path(out, name), row.names = FALSE)
}
layer_of <- function(p, cls) {
  b <- ggplot_build(p)
  i <- which(vapply(p$layers, function(l) inherits(l$geom, cls), logical(1)))
  b$data[[i[1]]]
}

data(pancreas_sub, package = "SCP")
srt <- pancreas_sub
emb <- Embeddings(srt, "UMAP")

# The inputs every comparison below starts from, so the Python test is
# self-contained: no R, and no dependency on the (untracked) h5ad.
genes <- c("Tpt1", "Rps2", "Rpl8", "Ins1")
dat <- GetAssayData(srt, layer = "data")[genes, ]
w(data.frame(cell = colnames(srt), UMAP_1 = emb[, 1], UMAP_2 = emb[, 2],
             CellType = as.character(srt$CellType), SubCellType = as.character(srt$SubCellType),
             Phase = as.character(srt$Phase), S_score = srt$S_score,
             t(as.matrix(dat)), check.names = FALSE),
  "inputs.csv")
write_json(list(CellType = levels(srt$CellType), SubCellType = levels(srt$SubCellType),
                Phase = levels(srt$Phase)), file.path(out, "levels.json"))

# ---- density: contour levels and the KDE grid behind them ------------------
p <- CellDimPlot(srt, "CellType", reduction = "UMAP", add_density = TRUE)
d <- layer_of(p, "GeomDensity2d")
w(d[, c("x", "y", "level", "piece", "group")][seq(1, nrow(d), by = 7), ], "density_contour.csv")
write_json(list(levels = sort(unique(d$level))), file.path(out, "density_levels.json"), digits = NA)
p <- CellDimPlot(srt, "CellType", reduction = "UMAP", add_density = TRUE, density_filled = TRUE)
d <- layer_of(p, "GeomRaster")
keep <- (seq_len(nrow(d)) - 1) %% 97 == 0
w(d[keep, c("x", "y", "density")], "density_raster.csv")

# ---- hexagons ----------------------------------------------------------------
p <- CellDimPlot(srt, "CellType", reduction = "UMAP", hex = TRUE)
d <- layer_of(p, "GeomHex")
w(d[, c("x", "y", "count", "group", "alpha")], "hex_cell.csv")
p <- FeatureDimPlot(srt, "Ins1", reduction = "UMAP", hex = TRUE)
b <- ggplot_build(p)
k <- which(vapply(p$layers, function(l) inherits(l$stat, "StatSummaryHex"), logical(1)))
w(b$data[[k]][, c("x", "y", "value")], "hex_feature.csv")
k <- which(vapply(p$layers, function(l) inherits(l$stat, "StatBinhex"), logical(1)))
w(b$data[[k]][, c("x", "y", "count")], "hex_feature_bg.csv")

# ---- graph edges on a 200-cell subset ---------------------------------------
srt <- FindNeighbors(srt, reduction = "PCA", dims = 1:30, verbose = FALSE)
cells <- colnames(srt)[1:200]
g <- srt@graphs[["RNA_snn"]][cells, cells]
gt <- Matrix::summary(as(g, "TsparseMatrix"))
w(data.frame(i = gt$i, j = gt$j, x = gt$x), "graph_triplets.csv")
p <- CellDimPlot(srt, "CellType", reduction = "UMAP", cells = cells, graph = "RNA_snn")
d <- layer_of(p, "GeomSegment")
w(d[, c("x", "y", "xend", "yend", "linewidth")], "graph_segments.csv")
writeLines(cells, file.path(out, "graph_cells.txt"))

# ---- lineages: two synthetic pseudotimes ------------------------------------
u1 <- emb[, 1]; u2 <- emb[, 2]
srt$Lineage1 <- ifelse(srt$CellType != "Ductal", u1 - min(u1), NA)
srt$Lineage2 <- ifelse(srt$CellType %in% c("Ductal", "Ngn3 low EP", "Ngn3 high EP"), u2 - min(u2), NA)
p <- LineagePlot(srt, lineages = c("Lineage1", "Lineage2"), reduction = "UMAP")
b <- ggplot_build(p)
paths <- do.call(rbind, lapply(seq_along(b$data), function(k) {
  d <- b$data[[k]]
  if (!inherits(p$layers[[k]]$geom, "GeomPath") || !"colour" %in% names(d)) return(NULL)
  if (all(d$colour == "white")) return(NULL)
  data.frame(layer = k, x = d$x, y = d$y, colour = d$colour)
}))
w(paths, "lineage_paths.csv")

# ---- PAGA on CellType, drawn over a SubCellType plot -------------------------
conn <- matrix(c(0, .8, .1, 0, 0,
                 .8, 0, .6, .05, 0,
                 .1, .6, 0, .7, .2,
                 0, .05, .7, 0, .9,
                 0, 0, .2, .9, 0), 5, 5)
paga <- list(connectivities = conn, connectivities_tree = conn, groups = "CellType")
p <- PAGAPlot(srt, paga = paga, reduction = "UMAP")
b <- ggplot_build(p)
pts <- b$data[[which(vapply(p$layers, function(l) inherits(l$geom, "GeomPoint"), logical(1)))[2]]]
w(pts[, c("x", "y", "fill", "colour")], "paga_nodes.csv")
seg <- b$data[[which(vapply(p$layers, function(l) inherits(l$geom, "GeomSegment"), logical(1)))[1]]]
names(seg)[grep("^linewidth", names(seg))] <- "linewidth"  # ggnewscale renames it
w(seg[, c("x", "y", "xend", "yend", "linewidth")], "paga_edges.csv")
write_json(list(connectivities = conn), file.path(out, "paga_input.json"), digits = NA)

# ---- raw velocity from a synthetic swirl ------------------------------------
V <- 0.1 * cbind(-(u2 - mean(u2)), u1 - mean(u1))
colnames(V) <- c("stochasticUMAP_1", "stochasticUMAP_2")
srt[["stochastic_UMAP"]] <- CreateDimReducObject(V, key = "stochasticUMAP_", assay = "RNA")
# VelocityPlot's check_R("metR") tries to install metR, which raw mode never
# uses. Stub it for this session so the dump never touches the R library.
assignInNamespace("check_R", function(...) invisible(NULL), ns = "SCP")
p <- VelocityPlot(srt, reduction = "UMAP", velocity = "stochastic", plot_type = "raw")
d <- layer_of(p, "GeomSegment")
w(d[, c("x", "y", "xend", "yend")], "velocity_raw.csv")

# ---- stat_by insets: StatPlot's per-group pies and the label positions -------
sub <- CellStatPlot(srt, stat.by = "Phase", group.by = "CellType", plot_type = "pie",
                    individual = TRUE, combine = FALSE)
props <- do.call(rbind, lapply(names(sub), function(nm) {
  d <- sub[[nm]]$data
  data.frame(panel = nm, stat = as.character(d$Phase), group = as.character(d$CellType),
             value = d$value)
}))
w(props, "stat_pie_values.csv")
p <- CellDimPlot(srt, "CellType", reduction = "UMAP", label = TRUE, label_insitu = TRUE)
d <- layer_of(p, "GeomTextRepel")
rng <- layer_scales(p)
w(data.frame(label = d$label, x = d$x, y = d$y,
             x_range = diff(rng$x$range$range), y_range = diff(rng$y$range$range)),
  "stat_pie_centres.csv")

# ---- ggforce enclosures on fixed point sets (in mm) -------------------------
ct <- as.character(srt$CellType)
mm <- data.frame(x = round(emb[, 1] * 3 + 60, 2), y = round(emb[, 2] * 3 + 60, 2),
                 id = as.integer(factor(ct, levels = levels(srt$CellType))))
mm <- mm[order(mm$id), ]
ell <- ggforce:::enclose_ellip_points(mm$x, mm$y, mm$id, 0.01)
w(ell, "mark_ellipse.csv")
cir <- ggforce:::enclose_points(mm$x, mm$y, mm$id)
w(cir, "mark_circle.csv")

# ---- calculate_coexp ----------------------------------------------------------
# SCP 0.5.6's check_DataType calls GetAssayData(slot = ), which SeuratObject 5
# made defunct, so FeatureDimPlot(calculate_coexp = TRUE) cannot run as
# shipped. Hand it the matrix through `data =` instead; the decision logic
# that follows is SCP's own and untouched.
orig_check <- SCP::check_DataType
assignInNamespace("check_DataType", function(srt, data = NULL, slot = "data", assay = NULL) {
  if (is.null(data)) data <- GetAssayData(srt, layer = slot, assay = assay %||% DefaultAssay(srt))
  orig_check(srt, data = data)
}, ns = "SCP")
pl <- FeatureDimPlot(srt, c("Tpt1", "Rps2", "Rpl8", "S_score"), reduction = "UMAP",
                     calculate_coexp = TRUE, combine = FALSE)
d <- pl[[grep(":CoExp$", names(pl))]]$data
w(data.frame(cell = rownames(d), CoExp = d$CoExp, value = d$value), "coexp.csv")

message("wrote ", length(list.files(out)), " files to ", normalizePath(out))
