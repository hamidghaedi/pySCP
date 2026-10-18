#!/usr/bin/env Rscript
# Dump what SCP's StatPlot builds for plot_type = "pie" and "ring" on
# pancreas_sub, so tests/test_cell_stat_polar.py can compare the drawn wedges
# and labels with numbers rather than pictures.
#
#   Rscript tools/parity/cell_stat_polar.R [out_dir] [png_dir]
#
# out_dir defaults to tests/fixtures/parity/cell_stat.  Writes polar.json:
# per case, the aggregated input (combination counts and factor levels -- all
# the layout depends on) and, per panel of StatPlot(..., combine = FALSE):
#
#   wedges   one row per geom_col rectangle, in coord_polar's own space: the
#            radius of its inner and outer edge (r_rescale: the discrete x
#            range mapped onto [0, 0.4] of the panel) and the angle of its two
#            edges (theta_rescale before the modulo: the y range mapped onto
#            [0, 2 pi], plus `start`), clockwise from twelve o'clock
#   labels   the geom_text_repel layer at the same transform (force = 0, so
#            the label sits where position_stack(vjust = 0.5) put it)
#   axes     the theta breaks and labels (percent / count around the circle)
#            and the r breaks and labels (the group levels), both rescaled
#
# A ring is the same plot with a dummy "   " level prepended to the group
# factor: that level holds one all-NA row, which geom_col drops, so it only
# reserves the innermost slot -- the hole.
#
# Needs SCP (0.5.x), ggplot2, ggrepel and jsonlite.  An optional second
# argument writes each case's figure as a PNG for eyeballing.

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[[1]] else "tests/fixtures/parity/cell_stat"
png_dir <- if (length(args) >= 2) args[[2]] else NA
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
if (nzchar(Sys.getenv("SCP_PARITY_LIB"))) {
  .libPaths(c(strsplit(Sys.getenv("SCP_PARITY_LIB"), ":")[[1]], .libPaths()))
}

suppressPackageStartupMessages({
  library(SCP); library(ggplot2); library(jsonlite)
})

data(pancreas_sub, package = "SCP")
md <- pancreas_sub@meta.data

combos <- function(df, cols) {
  key <- do.call(paste, c(lapply(cols, function(c) as.character(df[[c]])), sep = "\r"))
  u <- unique(key)
  first <- df[match(u, key), cols, drop = FALSE]
  out <- lapply(first, as.character)
  out$n <- as.integer(table(factor(key, levels = u)))
  as.data.frame(out, stringsAsFactors = FALSE, check.names = FALSE)
}

# coord_polar's two rescalings (ggplot2 coord-polar.R), without the final
# modulo so that a wedge ending at the full circle reads 2 pi, not 0
r_rescale <- function(x, range) scales::rescale(x, c(0, 0.4), range)
theta_rescale <- function(y, range, start) scales::rescale(y, c(0, 2 * pi), range) + start

dump_panel <- function(p) {
  b <- ggplot_build(p)
  pp <- b$layout$panel_params[[1]]
  start <- p$coordinates$start
  col <- b$data[[1]]
  wedges <- data.frame(
    fill = col$fill, x = col$x,
    r_in = r_rescale(col$xmin, pp$r.range), r_out = r_rescale(col$xmax, pp$r.range),
    theta_0 = theta_rescale(col$ymin, pp$theta.range, start),
    theta_1 = theta_rescale(col$ymax, pp$theta.range, start),
    ymin = col$ymin, ymax = col$ymax
  )
  out <- list(
    start = start, r_range = pp$r.range, theta_range = pp$theta.range,
    theta_major = theta_rescale(pp$theta.major, pp$theta.range, start),
    theta_labels = as.character(pp$theta.labels),
    r_major = r_rescale(pp$r.major, pp$r.range),
    r_labels = as.character(pp$r.labels),
    wedges = wedges
  )
  if (length(b$data) >= 2) {
    lab <- b$data[[2]]
    out$labels <- data.frame(
      label = as.character(lab$label), x = lab$x, y = lab$y,
      r = r_rescale(lab$x, pp$r.range), theta = theta_rescale(lab$y, pp$theta.range, start)
    )
  }
  out
}

polar_case <- function(stat.by, group.by = NULL, ..., png_name = NULL) {
  cols <- c(stat.by, group.by)
  pl <- StatPlot(md, stat.by = stat.by, group.by = group.by, combine = FALSE, ...)
  if (!is.na(png_dir) && !is.null(png_name)) {
    ggsave(file.path(png_dir, paste0(png_name, ".png")),
           StatPlot(md, stat.by = stat.by, group.by = group.by, ...), width = 10, height = 8)
  }
  list(
    stat_by = stat.by, group_by = group.by %||% NA, args = list(...),
    levels = lapply(setNames(cols, cols), function(c) levels(md[[c]])),
    combos = combos(md, cols),
    names = names(pl),
    panels = lapply(pl, dump_panel)
  )
}

cases <- list(
  pie_label = polar_case("CellType", "Phase", plot_type = "pie", label = TRUE,
                         png_name = "pie_label"),
  ring_label = polar_case("CellType", "Phase", plot_type = "ring", label = TRUE,
                          png_name = "ring_label"),
  ring_count = polar_case("Phase", "CellType", plot_type = "ring", stat_type = "count",
                          label = TRUE, png_name = "ring_count"),
  pie_nogroup = polar_case("CellType", NULL, plot_type = "pie", png_name = "pie_nogroup"),
  pie_flip = polar_case("CellType", "Phase", plot_type = "pie", flip = TRUE,
                        png_name = "pie_flip"),
  pie_individual = polar_case("CellType", "Phase", plot_type = "pie", individual = TRUE,
                              png_name = "pie_individual"),
  ring_individual = polar_case("CellType", "Phase", plot_type = "ring", individual = TRUE,
                               label = TRUE, png_name = "ring_individual")
)

res <- list(
  r_version = R.version.string,
  scp_version = as.character(packageVersion("SCP")),
  ggplot2_version = as.character(packageVersion("ggplot2")),
  cases = cases
)
write_json(res, file.path(out_dir, "polar.json"), digits = NA, auto_unbox = TRUE,
           null = "null", na = "null", force = TRUE, dataframe = "columns")
cat("wrote", file.path(out_dir, "polar.json"), "\n")
