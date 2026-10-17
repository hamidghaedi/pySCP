# What SCP's FeatureStatPlot actually builds, layer by layer, on pancreas_sub.
#
# Writes into the output directory:
#   cells.csv     the per-cell input (normalised genes, scores, groups) so the
#                 Python test can rebuild the same frame without R or the h5ad
#   layers.json   per case and panel: the plist names, the final y range of the
#                 panel (coord_cartesian with its expansion), the y breaks, and
#                 the computed data of the significance / trend / count layers
#
#   Rscript tools/parity/feature_stat_layers.R tests/fixtures/parity/feature_stat
#
# Needs SCP (0.5.x), Seurat, ggplot2, jsonlite, and ggpubr + ggsignif for the
# significance cases.  If ggpubr lives in a separate library, point
# SCP_PARITY_LIB at it.

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args)) args[[1]] else "tests/fixtures/parity/feature_stat"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
if (nzchar(Sys.getenv("SCP_PARITY_LIB"))) .libPaths(c(Sys.getenv("SCP_PARITY_LIB"), .libPaths()))

suppressPackageStartupMessages({
  library(SCP)
  library(Seurat)
  library(ggplot2)
})
data("pancreas_sub", package = "SCP")
pn <- NormalizeData(pancreas_sub, verbose = FALSE)

genes <- c("Neurog3", "Sox9", "Ins1", "Gcg")
expr <- as.matrix(GetAssayData(pn, layer = "data")[genes, ])
meta <- pn@meta.data
cells_df <- data.frame(
  cell = colnames(pn),
  CellType = as.character(meta$CellType),
  SubCellType = as.character(meta$SubCellType),
  Phase = as.character(meta$Phase),
  G2M_score = meta$G2M_score,
  S_score = meta$S_score,
  t(expr),
  check.names = FALSE
)
write.csv(cells_df, file.path(out_dir, "cells.csv"), row.names = FALSE)
levels_df <- list(CellType = levels(meta$CellType), SubCellType = levels(meta$SubCellType),
                  Phase = levels(meta$Phase))

set.seed(11)
small <- sample(colnames(pn), 60)

layer_kind <- function(l) {
  paste(class(l$geom)[1], class(l$stat)[1], sep = "/")
}
dump_plot <- function(p) {
  b <- ggplot_build(p)
  pp <- b$layout$panel_params[[1]]
  layers <- lapply(seq_along(p$layers), function(i) {
    k <- layer_kind(p$layers[[i]])
    d <- b$data[[i]]
    keep <- intersect(colnames(d), c("x", "xend", "y", "yend", "annotation", "label", "p",
                                     "p.format", "p.signif", "group", "n", "size", "ymin", "ymax"))
    if (!grepl("Signif|CompareMeans|GeomLine|GeomPoint|StatSum|GeomErrorbar|GeomCol", k)) {
      return(list(kind = k))
    }
    list(kind = k, data = d[, keep, drop = FALSE])
  })
  list(y_range = pp$y.range, y_breaks = pp$y$breaks, y_labels = pp$y$get_labels(),
       layers = layers)
}

run <- function(...) {
  pl <- FeatureStatPlot(pn, ..., combine = FALSE)
  if (inherits(pl, "ggplot")) pl <- list(pl)
  list(names = names(pl), panels = lapply(pl, function(p) {
    if (inherits(p, "ggplot")) dump_plot(p) else list(kind = class(p)[1])
  }))
}
run_engine <- function(...) {
  # ExpressionStatPlot directly: FeatureStatPlot's stack output is a gtable,
  # but the per-row panels (limits, two breaks) are what we compare
  pl <- SCP:::ExpressionStatPlot(exp.data = GetAssayData(pn, layer = "data"),
                                 meta.data = cbind(pn@meta.data, cells = colnames(pn)), ...)
  list(names = names(pl), panels = lapply(pl, dump_plot))
}

run_by_feature <- function(stat.by, group.by, ...) {
  # FeatureStatPlot's plot.by = "feature" branch (SCP-plot.R:3555), verbatim
  # except FetchData(layer =): SCP 0.5.6 passes slot =, which SeuratObject 5
  # made defunct, so the shipped branch errors in a current install.
  meta.reshape <- FetchData(pn, vars = c(stat.by, group.by), cells = colnames(pn), layer = "data")
  meta.reshape[["cells"]] <- rownames(meta.reshape)
  meta.reshape <- reshape2::melt(meta.reshape, measure.vars = stat.by,
                                 variable.name = "Features", value.name = "Stat.by")
  rownames(meta.reshape) <- paste0(meta.reshape[["cells"]], "-", meta.reshape[["Features"]])
  exp.data <- matrix(0, nrow = 1, ncol = nrow(meta.reshape),
                     dimnames = list("Stat.by", rownames(meta.reshape)))
  plist <- list()
  for (g in unique(meta.reshape[[group.by]])) {
    meta.use <- meta.reshape
    meta.use[[group.by]] <- NULL
    colnames(meta.use)[colnames(meta.use) == "Stat.by"] <- g
    p <- SCP:::ExpressionStatPlot(exp.data = exp.data, meta.data = meta.use, stat.by = g,
      group.by = "Features", cells = rownames(meta.reshape)[meta.reshape[[group.by]] == g], ...)
    plist <- append(plist, p)
  }
  list(names = names(plist), panels = lapply(plist, dump_plot))
}

cases <- list()
cases$violin <- run(stat.by = c("G2M_score", "Neurog3"), group.by = "CellType")
cases$box_brackets <- run(stat.by = "G2M_score", group.by = "CellType", plot_type = "box",
  comparisons = list(c("Ductal", "Ngn3 low EP"), c("Ngn3 high EP", "Endocrine"),
                     c("Ductal", "Endocrine")))
cases$violin_brackets_format <- run(stat.by = c("S_score", "Sox9"), group.by = "CellType",
  comparisons = list(c("Ductal", "Ngn3 low EP"), c("Ngn3 low EP", "Ngn3 high EP")),
  sig_label = "p.format")
cases$brackets_ttest_multi <- run(stat.by = "G2M_score", group.by = "CellType",
  comparisons = list(c("Ductal", "Pre-endocrine"), c("Ngn3 high EP", "Endocrine")),
  pairwise_method = "t.test", multiplegroup_comparisons = TRUE)
cases$multi_only <- run(stat.by = c("G2M_score", "Neurog3"), group.by = "CellType",
  multiplegroup_comparisons = TRUE, sig_label = "p.format")
cases$multi_anova <- run(stat.by = "S_score", group.by = "CellType",
  multiplegroup_comparisons = TRUE, multiple_method = "anova")
cases$split_true_kruskal <- run(stat.by = c("S_score", "Neurog3"), group.by = "CellType",
  split.by = "Phase", comparisons = TRUE, sig_label = "p.format")
cases$split_true_pairwise_small <- run(stat.by = c("G2M_score", "Sox9"), group.by = "CellType",
  split.by = "Phase", cells = small, comparisons = TRUE, sig_label = "p.format")
cases$split_multi <- run(stat.by = "G2M_score", group.by = "CellType", split.by = "Phase",
  multiplegroup_comparisons = TRUE, sig_label = "p.format")
cases$small_brackets <- run(stat.by = c("G2M_score", "Neurog3"), group.by = "CellType",
  cells = small, comparisons = list(c("Ductal", "Ngn3 low EP"), c("Ductal", "Endocrine")),
  sig_label = "p.format")
cases$bar <- run(stat.by = "G2M_score", group.by = "CellType", plot_type = "bar")
cases$bar_trend_split <- run(stat.by = "G2M_score", group.by = "CellType", split.by = "Phase",
  plot_type = "bar", add_trend = TRUE)
cases$violin_trend <- run(stat.by = "G2M_score", group.by = "CellType", add_trend = TRUE)
cases$violin_trend_split <- run(stat.by = "G2M_score", group.by = "CellType", split.by = "Phase",
  add_trend = TRUE)
cases$dot <- run(stat.by = "G2M_score", group.by = "CellType", plot_type = "dot")
cases$individual <- run(stat.by = "G2M_score", group.by = "CellType", split.by = "Phase",
  individual = TRUE)
cases$by_feature <- run_by_feature(stat.by = c("Neurog3", "Sox9", "G2M_score"), group.by = "CellType")
cases$stack_rows <- run_engine(stat.by = c("Neurog3", "Sox9", "Ins1", "Gcg"),
  group.by = "SubCellType", stack = TRUE)
cases$coexp <- run(stat.by = c("Neurog3", "Sox9"), group.by = "CellType", calculate_coexp = TRUE)

coexp_values <- {
  x <- expm1(expr[c("Neurog3", "Sox9"), ])
  apply(x, 2, function(v) log1p(exp(mean(log(v)))))
}

res <- list(
  r_version = R.version.string,
  scp_version = as.character(packageVersion("SCP")),
  ggplot2_version = as.character(packageVersion("ggplot2")),
  ggpubr_version = as.character(packageVersion("ggpubr")),
  levels = levels_df, small_cells = small,
  coexp = unname(coexp_values[1:50]),
  cases = cases
)
jsonlite::write_json(res, file.path(out_dir, "layers.json"), digits = NA, auto_unbox = TRUE,
                     null = "null", na = "null", force = TRUE)
cat("wrote", file.path(out_dir, "layers.json"), "\n")
