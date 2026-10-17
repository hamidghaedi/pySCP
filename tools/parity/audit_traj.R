#!/usr/bin/env Rscript
# Regenerate the trajectory-family parity fixtures under
# tests/fixtures/parity/audit/traj/ from SCP (Hao Zhang, NJMU, GPL-3 --
# https://github.com/zhanghao-njmu/SCP).
#
#   Rscript tools/parity/audit_traj.R [out_dir]
#
# What is written, and why it is shaped the way it is:
#
# * inputs.csv / levels.json -- everything the Python side needs to rebuild the
#   same object: UMAP and the first two PCs, SubCellType/CellType, the three
#   slingshot lineages (RunSlingshot exactly as SCP's examples call it), the
#   library size and the raw counts of a handful of genes. tests/ never needs R
#   or the .h5ad.
# * lineage_*.csv -- the built (ggplot_build) path and whisker layers of
#   LineagePlot: the LOESS-fitted curve per lineage.
# * paga_*.csv -- a deterministic PAGA-shaped input (group connectivities from
#   the SNN graph, a maximum spanning tree, and an asymmetric transition matrix
#   oriented by median pseudotime). Its provenance does not matter -- both
#   sides read the same matrices -- and building it here keeps the export free
#   of scanpy/scvelo, which RunPAGA/RunSCVELO would route through reticulate.
#   For each PAGAPlot option the built node, edge and transition layers are
#   dumped together with the legend labels.
# * graph_*.csv -- GraphPlot on a small synthetic graph with negative
#   coordinates, which is where SCP's global_size convention shows.
# * segments.csv -- segementsDf on random segments (pure geometry).
# * velocity_*.csv -- VelocityPlot on a synthetic velocity embedding stored as
#   the `stochastic_UMAP` reduction: raw segments (x2 scales, grouped) and the
#   grid field from SCP's compute_velocity_on_grid.
# * dynamic_*.csv -- DynamicPlot point/line/ribbon layers (combine = FALSE).
# * dynheat_*.csv -- DynamicHeatmap(use_fitted = FALSE): the processed matrix,
#   the per-lineage cell order and the feature order/peaktime, plus the
#   RunDynamicFeatures tables it took the peak times from
#   (dynheat_dynamic_features_<lineage>.csv).
# * projection_*.csv -- the composite layer of ProjectionPlot (p3) and its
#   panel ranges.

suppressPackageStartupMessages({
  library(SCP); library(Seurat); library(ggplot2); library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) > 0) args[[1]] else "tests/fixtures/parity/audit/traj"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
grDevices::pdf(NULL)

wcsv <- function(df, name) {
  write.csv(df, file.path(out, name), row.names = FALSE)
}
num <- function(x) signif(x, 12)

# --- layer dumper ------------------------------------------------------------
# One row per drawn primitive; geom class recorded so the Python side can pick
# the layer it is comparing without relying on layer index alone.
dump_layers <- function(p, prefix, layers = NULL, cols = NULL, digits = 9) {
  b <- ggplot_build(p)
  rows <- list()
  use <- layers %||% seq_along(b$data)
  # ggnewscale renames aesthetics (linewidth -> linewidth_ggnewscale_1, ...), so match by
  # prefix and strip the suffix
  pat <- "^(x|y|xend|yend|ymin|ymax|linewidth|size|colour|fill|alpha|linetype|label|group|PANEL)(_new|_ggnewscale_[0-9]+)*$"
  for (i in use) {
    d <- b$data[[i]]
    d <- d[, grepl(pat, colnames(d)), drop = FALSE]
    colnames(d) <- sub("(_new|_ggnewscale_[0-9]+)+$", "", colnames(d))
    if (!is.null(cols)) d <- d[, intersect(cols, colnames(d)), drop = FALSE]
    for (k in intersect(c("x", "y", "xend", "yend", "ymin", "ymax", "linewidth", "size"), colnames(d))) {
      d[[k]] <- signif(as.numeric(d[[k]]), digits)
    }
    d$layer <- i
    d$geom <- class(p$layers[[i]]$geom)[1]
    rows[[length(rows) + 1]] <- d
  }
  all_cols <- unique(unlist(lapply(rows, colnames)))
  rows <- lapply(rows, function(d) {
    for (k in setdiff(all_cols, colnames(d))) d[[k]] <- NA
    d[, all_cols]
  })
  wcsv(do.call(rbind, rows), paste0(prefix, "_layers.csv"))
  invisible(b)
}

guide_labels <- function(p, aes = "colour") {
  as.character(ggplot2::get_guide_data(p, aes)$.label)
}

# SCP 0.5.6 still calls GetAssayData(slot = ...) inside RunDynamicFeatures,
# which SeuratObject >= 5 makes defunct (an error, not a warning). Map slot= onto
# layer= in SCP's import environment so DynamicPlot / DynamicHeatmap run
# unmodified otherwise.
local({
  imp <- parent.env(asNamespace("SCP"))
  shim <- function(object, slot = NULL, assay = NULL, layer = NULL, ...) {
    SeuratObject::GetAssayData(object, assay = assay, layer = layer %||% slot %||% "data", ...)
  }
  if (exists("GetAssayData", envir = imp, inherits = FALSE)) {
    unlockBinding("GetAssayData", imp)
    assign("GetAssayData", shim, envir = imp)
    lockBinding("GetAssayData", imp)
  }
})

data("pancreas_sub", package = "SCP")
srt <- pancreas_sub
set.seed(11)
srt <- RunSlingshot(srt, group.by = "SubCellType", reduction = "UMAP", show_plot = FALSE)
lineages <- paste0("Lineage", 1:3)

genes <- c("Nnat", "Irx1", "Sox9", "Neurod2", "Isl1", "Rbp4", "Pyy")
counts <- SeuratObject::LayerData(srt, layer = "counts")
umap <- Embeddings(srt, "UMAP")
pca <- Embeddings(srt, "PCA")

# --- inputs ------------------------------------------------------------------
inputs <- data.frame(
  cell = colnames(srt),
  UMAP_1 = num(umap[, 1]), UMAP_2 = num(umap[, 2]),
  PC_1 = num(pca[, 1]), PC_2 = num(pca[, 2]),
  SubCellType = as.character(srt$SubCellType),
  CellType = as.character(srt$CellType),
  stringsAsFactors = FALSE
)
for (l in lineages) inputs[[l]] <- num(srt@meta.data[[l]])
inputs$libsize <- Matrix::colSums(counts)
for (g in genes) inputs[[paste0("gene_", g)]] <- as.numeric(counts[g, ])
wcsv(inputs, "inputs.csv")
write_json(list(SubCellType = levels(srt$SubCellType), CellType = levels(srt$CellType),
                genes = genes, lineages = lineages,
                reduction_key = Key(srt[["UMAP"]]),
                scp_version = as.character(packageVersion("SCP")),
                ggplot2_version = as.character(packageVersion("ggplot2")),
                slingshot_version = as.character(packageVersion("slingshot"))),
           file.path(out, "levels.json"), auto_unbox = FALSE, pretty = TRUE)

# --- LineagePlot ---------------------------------------------------------------
p <- LineagePlot(srt, lineages = lineages)
dump_layers(p, "lineage_default", layers = c(2, 4, 6), cols = c("x", "y", "colour"))
p <- LineagePlot(srt, lineages = lineages, whiskers = TRUE)
dump_layers(p, "lineage_whiskers", layers = 1, cols = c("x", "y", "xend", "yend", "colour"))
p <- LineagePlot(srt, lineages = "Lineage1", span = 0.3, trim = c(0.05, 0.95))
dump_layers(p, "lineage_span03", layers = 2, cols = c("x", "y", "colour"))
# how many cells survive the trim, per lineage (R uses strict > and <)
trim_n <- sapply(lineages, function(l) {
  x <- srt@meta.data[[l]]
  q <- quantile(x, c(0.01, 0.99), na.rm = TRUE)
  sum(x > q[1] & x < q[2], na.rm = TRUE)
})
wcsv(data.frame(lineage = lineages, n_used = trim_n), "lineage_trim_n.csv")
# a tied (integer) pseudotime, where the trim quantiles land exactly on data
# values and the strict inequality decides which cells are fitted
srt$LineageInt <- floor(srt$Lineage1)
p <- LineagePlot(srt, lineages = "LineageInt")
dump_layers(p, "lineage_int", layers = 2, cols = c("x", "y"))

# --- PAGA-shaped input ---------------------------------------------------------
srt <- FindNeighbors(srt, reduction = "PCA", dims = 1:30, verbose = FALSE)
snn <- srt@graphs[["RNA_snn"]]
grp <- srt$SubCellType
lev <- levels(grp)
G <- sapply(lev, function(b) sapply(lev, function(a) sum(snn[grp == a, grp == b])))
sz <- as.numeric(table(grp)[lev])
conn <- G / sqrt(outer(sz, sz))
diag(conn) <- 0
conn <- conn / max(conn)
dimnames(conn) <- list(lev, lev)
# maximum spanning tree on the connectivities, stored one-directional as
# scanpy's connectivities_tree is
ig <- igraph::graph_from_adjacency_matrix(conn, mode = "undirected", weighted = TRUE)
mst <- igraph::mst(ig, weights = 1 - igraph::E(ig)$weight)
tree <- matrix(0, length(lev), length(lev), dimnames = list(lev, lev))
el <- igraph::as_edgelist(mst, names = FALSE)
for (k in seq_len(nrow(el))) {
  i <- max(el[k, ]); j <- min(el[k, ])
  tree[i, j] <- conn[i, j]
}
pt_med <- sapply(lev, function(a) median(srt$Lineage1[grp == a], na.rm = TRUE))
pt_med[is.na(pt_med)] <- max(pt_med, na.rm = TRUE) + 1
trans <- conn * plogis(outer(pt_med, pt_med, function(a, b) b - a) / 5)
dimnames(trans) <- list(lev, lev)
wcsv(data.frame(group = lev, conn, check.names = FALSE), "paga_connectivities.csv")
wcsv(data.frame(group = lev, tree, check.names = FALSE), "paga_connectivities_tree.csv")
wcsv(data.frame(group = lev, trans, check.names = FALSE), "paga_transitions.csv")

srt@misc$paga <- list(connectivities = conn, connectivities_tree = tree,
                      transitions_confidence = trans, groups = "SubCellType")
paga_cases <- list(
  default = list(),
  tree = list(type = "connectivities_tree"),
  shorten = list(edge_shorten = 0.05),
  transition = list(show_transition = TRUE),
  transition_offset = list(show_transition = TRUE, transition_offset = 0.02),
  groupsize = list(node_size = "GroupSize"),
  label = list(label = TRUE),
  pca = list(reduction = "PCA")
)
for (nm in names(paga_cases)) {
  p <- do.call(PAGAPlot, c(list(srt = srt), paga_cases[[nm]]))
  dump_layers(p, paste0("paga_", nm))
  writeLines(guide_labels(p), file.path(out, paste0("paga_", nm, "_legend.txt")))
}

# --- GraphPlot on a synthetic graph with negative coordinates -----------------
node <- data.frame(x = c(-5, -2, 3, 1, -1), y = c(-4, 2, -1, -3, -6),
                   g = c("a", "b", "c", "d", "e"), row.names = c("n1", "n2", "n3", "n4", "n5"))
E <- matrix(c(0, .8, .2, 0, .5,
              .8, 0, .6, .3, 0,
              .2, .6, 0, .9, .05,
              0, .3, .9, 0, .4,
              .5, 0, .05, .4, 0), 5, 5, byrow = TRUE, dimnames = list(rownames(node), rownames(node)))
Tm <- matrix(c(0, .9, .1, 0, .3,
               .2, 0, .7, .1, 0,
               .3, .1, 0, .6, .2,
               0, .5, .2, 0, .1,
               .1, 0, .4, .3, 0), 5, 5, byrow = TRUE, dimnames = dimnames(E))
wcsv(data.frame(node = rownames(node), node), "graph_node.csv")
wcsv(data.frame(node = rownames(node), E, check.names = FALSE), "graph_edge.csv")
wcsv(data.frame(node = rownames(node), Tm, check.names = FALSE), "graph_transition.csv")
p <- GraphPlot(node = node, edge = E, node_group = "g", edge_shorten = 0.1)
dump_layers(p, "graph_edges")
p <- GraphPlot(node = node, edge = E, transition = Tm, node_group = "g")
dump_layers(p, "graph_transition")

# --- segementsDf ---------------------------------------------------------------
set.seed(11)
seg <- data.frame(x = rnorm(20), y = rnorm(20), xend = rnorm(20), yend = rnorm(20))
s2 <- SCP:::segementsDf(seg, shorten_start = 0.3, shorten_end = 0.2, offset = 0.1)
wcsv(data.frame(x0 = num(seg$x), y0 = num(seg$y), xend0 = num(seg$xend), yend0 = num(seg$yend),
                x = num(s2$x), y = num(s2$y), xend = num(s2$xend), yend = num(s2$yend)),
     "segments.csv")

# --- VelocityPlot on a synthetic velocity embedding ----------------------------
# VelocityPlot calls check_R("metR") unconditionally, which tries to install
# metR (and sf) even though only plot_type = "stream" uses it. The raw and grid
# paths are exercised through a copy with that guard removed; stream is not
# exported, so it is not checked here.
vp <- SCP::VelocityPlot
src <- deparse(body(vp))
src <- src[!grepl('check_R("metR")', src, fixed = TRUE)]
body(vp) <- parse(text = paste(src, collapse = "\n"))[[1]]
environment(vp) <- asNamespace("SCP")
V <- cbind(-umap[, 2], umap[, 1]) * 0.05 + 0.1
colnames(V) <- c("stochasticUMAP_1", "stochasticUMAP_2")
srt[["stochastic_UMAP"]] <- CreateDimReducObject(V, key = "stochasticUMAP_", assay = "RNA")
wcsv(data.frame(cell = colnames(srt), V1 = num(V[, 1]), V2 = num(V[, 2])), "velocity_input.csv")
dump_layers(vp(srt, reduction = "UMAP"), "velocity_raw", cols = c("x", "y", "xend", "yend", "colour"))
dump_layers(vp(srt, reduction = "UMAP", scale = 2), "velocity_raw_scale2", cols = c("xend", "yend"))
dump_layers(vp(srt, reduction = "UMAP", group_by = "SubCellType"), "velocity_raw_group", cols = "colour")
dump_layers(vp(srt, reduction = "UMAP", plot_type = "grid"), "velocity_grid", cols = c("x", "y", "xend", "yend"))

# --- DynamicPlot ---------------------------------------------------------------
bp <- BiocParallel::SerialParam()
pl <- DynamicPlot(srt, lineages = "Lineage1", features = c("Nnat", "Irx1"),
                  group.by = "SubCellType", combine = FALSE)
# only the raw-point layer is pinned: the ribbon and line come from mgcv,
# which the Python side does not attempt to reproduce
writeLines(names(pl), file.path(out, "dynamic_panels.txt"))
dump_layers(pl[[1]], "dynamic_Lineage1_Nnat", layers = 1, cols = c("x", "y", "colour"))
pl2 <- DynamicPlot(srt, lineages = c("Lineage1", "Lineage2"), features = c("Nnat", "Irx1"),
                   group.by = "SubCellType", combine = FALSE)
writeLines(names(pl2), file.path(out, "dynamic_compare_panels.txt"))
b2 <- ggplot_build(pl2[[1]])
wcsv(do.call(rbind, lapply(seq_along(b2$data), function(i) {
  d <- b2$data[[i]]
  data.frame(layer = i, geom = class(pl2[[1]]$layers[[i]]$geom)[1], n = nrow(d),
             colours = paste(unique(d[[grep("^colour", colnames(d))[1]]] %||% NA), collapse = ";"),
             fills = paste(unique(d[[grep("^fill", colnames(d))[1]]] %||% NA), collapse = ";"))
})), "dynamic_compare_first_summary.csv")

# --- DynamicHeatmap (use_fitted = FALSE) ---------------------------------------
dh_genes <- c("Sox9", "Neurod2", "Isl1", "Rbp4", "Pyy", "Nnat", "Irx1")
ht <- DynamicHeatmap(srt, lineages = c("Lineage1", "Lineage2"), features = dh_genes)
M <- ht$matrix
# columns are written unnamed; dynheat_cell_order.csv maps column i to
# (lineage, cell) in the same order
Mw <- signif(as.matrix(M), 8)
colnames(Mw) <- seq_len(ncol(Mw))
wcsv(data.frame(feature = rownames(M), Mw, check.names = FALSE), "dynheat_matrix.csv")
co <- do.call(rbind, lapply(names(ht$cell_order), function(l)
  data.frame(lineage = l, cell = sub(paste0(l, "$"), "", ht$cell_order[[l]]))))
stopifnot(identical(paste0(co$cell, co$lineage), colnames(M)))
wcsv(co, "dynheat_cell_order.csv")
fm <- ht$feature_metadata
wcsv(data.frame(feature = rownames(fm), fm[, grep("peaktime$|^features$", colnames(fm)), drop = FALSE],
                check.names = FALSE), "dynheat_features.csv")
# the RunDynamicFeatures tables DynamicHeatmap computed those peak times from
# (it fits them on a local copy and does not return them): what
# scp.pl.dynamic_heatmap(dynamic_features=) takes, one CSV per lineage
srt_dyn <- RunDynamicFeatures(srt, lineages = c("Lineage1", "Lineage2"), features = dh_genes)
for (l in c("Lineage1", "Lineage2")) {
  df <- srt_dyn@tools[[paste0("DynamicFeatures_", l)]][["DynamicFeatures"]]
  stopifnot(isTRUE(all.equal(df[rownames(fm), "peaktime"], fm[[paste0(l, "peaktime")]])))
  wcsv(df, paste0("dynheat_dynamic_features_", l, ".csv"))
}

# --- ProjectionPlot ------------------------------------------------------------
# ProjectionPlot returns a patchwork once a legend exists, so the composite
# ggplot (p3) is captured by stopping the function just before that step.
pp <- SCP::ProjectionPlot
src <- deparse(body(pp))
src <- sub("if (is.null(legend)) {", "if (TRUE) {", src, fixed = TRUE)
body(pp) <- parse(text = paste(src, collapse = "\n"))[[1]]
environment(pp) <- asNamespace("SCP")
ref <- srt[, srt$CellType != "Endocrine"]
qry <- srt[, srt$CellType == "Endocrine"]
p3 <- pp(qry, ref, query_group = "SubCellType", ref_group = "CellType",
         query_reduction = "UMAP", ref_reduction = "UMAP")
b <- dump_layers(p3, "projection", cols = c("x", "y", "colour", "size"), digits = 8)
pr <- b$layout$panel_params[[1]]
wcsv(data.frame(axis = c("x", "y"), lo = num(c(pr$x.range[1], pr$y.range[1])),
                hi = num(c(pr$x.range[2], pr$y.range[2]))), "projection_ranges.csv")
wcsv(data.frame(cell = colnames(ref)), "projection_ref_cells.csv")

message("done -> ", normalizePath(out))
