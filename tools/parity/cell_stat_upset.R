#!/usr/bin/env Rscript
# Dump what SCP's StatPlot builds for plot_type = "upset" on pancreas_sub, so
# tests/test_stat_upset.py can compare the bar order, the counts and the
# combination-matrix rows without R.
#
#   Rscript tools/parity/cell_stat_upset.R [out_dir]
#
# out_dir defaults to tests/fixtures/parity/cell_stat.  Writes
#   upset_cells.csv   the per-cell input: the logical set columns and the split
#   upset.json        per case and split level: the intersections in bar order
#                     (ggupset's scale limits), each bar's count and fill, the
#                     count labels, and the matrix rows top to bottom
#
# Needs SCP (0.5.x), ggplot2, jsonlite and ggupset (0.4.x).  ggupset is not in
# the scp_parity library; install it into a scratch library and point
# SCP_PARITY_LIB at it (several libraries may be joined with ":").
#
# The matrix rows are not in the built plot: ggupset draws them from the axis
# guide at render time.  make_combination_matrix_plot() is traced to capture
# the label_set it is handed while ggplotGrob() renders each plot.

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args)) args[[1]] else "tests/fixtures/parity/cell_stat"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
if (nzchar(Sys.getenv("SCP_PARITY_LIB"))) {
  .libPaths(c(strsplit(Sys.getenv("SCP_PARITY_LIB"), ":")[[1]], .libPaths()))
}

suppressPackageStartupMessages({
  library(SCP)
  library(Seurat)
  library(ggplot2)
  library(ggupset)
  library(jsonlite)
})
pdf(NULL)  # ggplotGrob() below needs a device; do not leave an Rplots.pdf behind
data(pancreas_sub, package = "SCP")
srt <- pancreas_sub
counts <- GetAssayData(srt, layer = "counts")
md <- srt@meta.data
# the sets of SCP's own CellStatPlot example (SCP-plot.R:4450), plus two more
# genes so that some splits have more than ggupset's 20 shown intersections
md$Progenitor <- md$CellType %in% c("Ngn3 low EP", "Ngn3 high EP")
md$G2M <- md$Phase == "G2M"
md$Sox9_Expressed <- counts["Sox9", ] > 0
md$Neurog3_Expressed <- counts["Neurog3", ] > 0
md$Ins1_Expressed <- counts["Ins1", ] > 0
md$Gcg_Expressed <- counts["Gcg", ] > 0
sets4 <- c("Progenitor", "G2M", "Sox9_Expressed", "Neurog3_Expressed")
sets6 <- c(sets4, "Ins1_Expressed", "Gcg_Expressed")
srt@meta.data <- md

cells <- data.frame(cell = rownames(md), CellType = as.character(md$CellType),
                    md[, sets6], check.names = FALSE)
write.csv(cells, file.path(out_dir, "upset_cells.csv"), row.names = FALSE)

captured <- NULL
trace(ggupset:::make_combination_matrix_plot, quote({
  captured <<- as.character(label_set)
}), print = FALSE, where = asNamespace("ggupset"))

dump_upset <- function(p) {
  b <- ggplot_build(p)
  sc <- b$layout$panel_scales_x[[1]]
  lim <- sc$get_limits()
  sep <- sc$internal_text_separator
  bars <- b$data[[1]]
  labels <- b$data[[2]]
  captured <<- NULL
  invisible(ggplotGrob(p))
  list(
    intersections = lapply(strsplit(as.character(lim), sep, fixed = TRUE), as.list),
    bar_x = bars$x, bar_count = bars$count, bar_fill = bars$fill,
    label_x = labels$x, label = as.character(labels$label),
    # make_combination_matrix_plot's discrete y: first level at the bottom
    matrix_rows_bottom_up = as.list(captured),
    sets = as.list(as.character(sc$sets)),
    y_label = p$labels$y, x_label = p$labels$x %||% ""
  )
}

run <- function(...) {
  pl <- CellStatPlot(srt, plot_type = "upset", stat_level = "TRUE", combine = FALSE, ...)
  if (inherits(pl, "ggplot")) pl <- list(pl)
  list(names = as.list(names(pl)), panels = lapply(pl, dump_upset))
}

cases <- list(
  four = run(stat.by = sets4),
  six_split = run(stat.by = sets6, split.by = "CellType"),
  six = run(stat.by = sets6)
)
res <- list(
  r_version = R.version.string,
  scp_version = as.character(packageVersion("SCP")),
  ggupset_version = as.character(packageVersion("ggupset")),
  ggplot2_version = as.character(packageVersion("ggplot2")),
  locale = Sys.getlocale("LC_COLLATE"),
  cases = cases
)
write_json(res, file.path(out_dir, "upset.json"), digits = NA, auto_unbox = TRUE,
           null = "null", na = "null", force = TRUE)
cat("wrote", file.path(out_dir, "upset.json"), "\n")
