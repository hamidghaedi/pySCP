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
  library(SCP); library(Seurat); library(ggplot2); library(jsonlite); library(grid)
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

# ---- panel ranges: scale limits plus ggplot's default 5 % expansion ---------
pr_row <- function(p, case) {
  pp <- ggplot_build(p)$layout$panel_params[[1]]
  data.frame(case = case, x_lo = pp$x.range[1], x_hi = pp$x.range[2],
             y_lo = pp$y.range[1], y_hi = pp$y.range[2])
}
sub_cells <- colnames(srt)[srt$CellType %in% c("Ductal", "Endocrine")]
cdp_split <- CellDimPlot(srt, "CellType", split.by = "Phase", reduction = "UMAP", combine = FALSE)
w(rbind(pr_row(CellDimPlot(srt, "CellType", reduction = "UMAP"), "cell_umap"),
        pr_row(CellDimPlot(srt, "CellType", reduction = "UMAP", cells = sub_cells), "cell_subset"),
        pr_row(cdp_split[[length(cdp_split)]], "cell_split_last"),
        pr_row(FeatureDimPlot(srt, "Ins1", reduction = "UMAP"), "feature_umap")),
  "panel_ranges.csv")

# ---- DefaultReduction: how a reduction name or pattern resolves ------------
# a throwaway object carrying reductions named to exercise every step:
# exact names, case-insensitive grep, agrep's one-edit match, the
# fewest-columns tie-break and min_dim
red <- srt
set.seed(1)
for (nm in c("umap_harmony", "tsne", "UMAP3D", "pcaX", "lrgvis")) {
  k <- c(umap_harmony = 2, tsne = 2, UMAP3D = 3, pcaX = 10, lrgvis = 2)[[nm]]
  m <- matrix(rnorm(ncol(red) * k), ncol = k, dimnames = list(colnames(red), paste0(nm, "_", 1:k)))
  red[[nm]] <- CreateDimReducObject(m, key = paste0(gsub("[^A-Za-z]", "", nm), "_"), assay = "RNA")
}
cases <- expand.grid(pattern = c("<default>", "umap", "UMAP", "Umap", "pca", "PCA", "tsne",
                                 "TSNE", "harmony", "largevis", "3D", "umap3d"),
                     min_dim = c(2, 3), stringsAsFactors = FALSE)
cases$result <- mapply(function(p, d) {
  DefaultReduction(red, pattern = if (p == "<default>") NULL else p, min_dim = d)
}, cases$pattern, cases$min_dim)
w(cases, "default_reduction.csv")
w(data.frame(name = names(red@reductions),
             ncol = vapply(red@reductions, function(r) ncol(r@cell.embeddings), numeric(1))),
  "default_reduction_inputs.csv")

# ---- patchwork's default grid (wrap_plots -> ggplot2::wrap_dims) ------------
dims_n <- t(vapply(1:60, function(n) ggplot2::wrap_dims(n), numeric(2)))
w(data.frame(n = 1:60, nrow = dims_n[, 1], ncol = dims_n[, 2]), "wrap_dims.csv")

# ---- grid's arrow length: an npc length is taken on min(width, height) -----
# drawn into viewports of known size on an uncompressed pdf, whose path
# operators give the head's vertices in points
arrow_pdf <- tempfile(fileext = ".pdf")
vps <- data.frame(w_in = c(4, 2, 3, 5), h_in = c(2, 4, 3, 1.5), npc = c(0.25, 0.1, 0.2, 0.3))
pdf(arrow_pdf, width = 7, height = 6, compress = FALSE)
for (i in seq_len(nrow(vps))) {
  grid.newpage()
  pushViewport(viewport(x = unit(1, "in"), y = unit(1, "in"), width = unit(vps$w_in[i], "in"),
                        height = unit(vps$h_in[i], "in"), just = c(0, 0)))
  grid.segments(0.1, 0.5, 0.9, 0.5, gp = gpar(fill = "black", lwd = 0.01),
                arrow = arrow(length = unit(vps$npc[i], "npc"), type = "closed", angle = 20))
}
invisible(dev.off())
ops <- readLines(arrow_pdf, warn = FALSE)
ops <- ops[grepl("^[0-9.]+ [0-9.]+ [ml]$", ops, useBytes = TRUE)]
xy <- do.call(rbind, lapply(strsplit(ops, " "), function(t) as.numeric(t[1:2])))
# per page the shaft is one "x y m x y l S" line, then the closed head is
# three lines: a back corner (m), the tip (l), the other back corner (l)
stopifnot(nrow(xy) == 3 * nrow(vps))
heads <- lapply(seq_len(nrow(vps)), function(i) xy[(i - 1) * 3 + 1:3, , drop = FALSE])
vps$head_side_in <- vapply(heads, function(h) sqrt(sum((h[1, ] - h[2, ])^2)) / 72, numeric(1))
vps$head_back_in <- vapply(heads, function(h) abs(h[2, 1] - h[1, 1]) / 72, numeric(1))
w(vps, "arrow_npc.csv")

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
# each arrow's head length, unit(length / global_size, "npc")
k <- which(vapply(p$layers, function(l) inherits(l$geom, "GeomSegment"), logical(1)))[1]
w(data.frame(head_npc = as.numeric(p$layers[[k]]$geom_params$arrow$length)), "velocity_raw_heads.csv")

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

# ---- stat_by insets of every type, as CellDimPlot pastes them ----------------
# Last, because of the stub: SCP 0.5.6 composes its legends with cbind() on
# gtables, which ggplot2 4 turns into a list matrix that add_grob() rejects,
# so every CellDimPlot(stat.by = ) call fails at that final step. Skipping only
# the legend lets the insets be traced; nothing they draw is touched.
assignInNamespace("add_grob", function(gtable, grob, ...) gtable, ns = "SCP")
srt <- pancreas_sub
boxes <- list()
trace("annotation_custom", where = asNamespace("SCP"), print = FALSE,
      tracer = quote(assign("boxes", c(get("boxes", envir = .GlobalEnv),
                                       list(c(xmin, xmax, ymin, ymax))), envir = .GlobalEnv)))
inset_types <- c("pie", "ring", "bar", "rose", "trend", "area", "dot")
box_df <- do.call(rbind, lapply(inset_types, function(pt) {
  assign("boxes", list(), envir = .GlobalEnv)
  p <- suppressWarnings(CellDimPlot(srt, "CellType", reduction = "UMAP", stat.by = "Phase",
                                    stat_plot_type = pt, stat_plot_size = 0.2))
  b <- do.call(rbind, get("boxes", envir = .GlobalEnv))
  data.frame(type = pt, group = levels(srt$CellType), xmin = b[, 1], xmax = b[, 2],
             ymin = b[, 3], ymax = b[, 4])
}))
untrace("annotation_custom", where = asNamespace("SCP"))
w(box_df, "stat_inset_boxes.csv")

# Each subplot as annotation_custom receives it: + theme_void(), built, with
# every mark and label normalised to the panel range ggplot trained on that
# one group (0 = left/bottom/start, 1 = right/top/end). For coord_polar the
# x aesthetic is the radius under theta = "y" (pie, ring) and the angle under
# theta = "x" (rose).
nrm <- function(v, r) if (diff(r) == 0) rep(NA_real_, length(v)) else (v - r[1]) / diff(r)
inset_cfg <- data.frame(
  type = c("bar", "bar", "bar", "rose", "trend", "area", "dot", "pie", "ring"),
  position = c("stack", "stack", "dodge", "stack", "stack", "stack", "stack", "stack", "stack"),
  stat_type = c("percent", "count", "count", "percent", "percent", "percent", "count",
                "percent", "count"))
marks <- list(); labs <- list()
for (k in seq_len(nrow(inset_cfg))) {
  cf <- inset_cfg[k, ]
  sub <- CellStatPlot(srt, stat.by = "Phase", group.by = "CellType", plot_type = cf$type,
                      position = cf$position, stat_type = cf$stat_type, palette = "Set1",
                      label = TRUE, label.size = 3, individual = TRUE, combine = FALSE)
  for (grp in levels(srt$CellType)) {
    q <- sub[[paste0("CellType:", grp, ":")]] + theme_void() + theme(legend.position = "none")
    b <- suppressWarnings(ggplot_build(q))
    pp <- b$layout$panel_params[[1]]
    if (is.null(pp$theta.range)) {
      xr <- pp$x.range; yr <- pp$y.range
    } else if (q$coordinates$theta == "y") {
      xr <- pp$r.range; yr <- pp$theta.range
    } else {
      xr <- pp$theta.range; yr <- pp$r.range
    }
    for (i in seq_along(q$layers)) {
      geom <- class(q$layers[[i]]$geom)[1]
      d <- b$data[[i]]
      if (geom == "GeomTextRepel") {
        d <- d[!is.na(d$x), ]
        labs[[length(labs) + 1]] <- data.frame(cf, group = grp, label = d$label,
                                                x = nrm(d$x, xr), y = nrm(d$y, yr))
      } else if (geom %in% c("GeomCol", "GeomPoint", "GeomArea")) {
        if (nrow(d) == 0) {
          marks[[length(marks) + 1]] <- data.frame(cf, group = grp, geom = geom, fill = NA,
                                                   x0 = NA, x1 = NA, y0 = NA, y1 = NA, size = NA)
          next
        }
        if (geom == "GeomArea") next  # trend's ribbon: hidden under its own bar
        d <- d[!is.na(d$fill), ]
        pt <- geom == "GeomPoint"
        marks[[length(marks) + 1]] <- data.frame(
          cf, group = grp, geom = geom, fill = d$fill,
          x0 = nrm(if (pt) d$x else d$xmin, xr), x1 = nrm(if (pt) d$x else d$xmax, xr),
          y0 = nrm(if (pt) d$y else d$ymin, yr), y1 = nrm(if (pt) d$y else d$ymax, yr),
          size = if (pt) d$size else NA)
      }
    }
  }
}
w(do.call(rbind, marks), "stat_inset_marks.csv")
w(do.call(rbind, labs), "stat_inset_labels.csv")

message("wrote ", length(list.files(out)), " files to ", normalizePath(out))
