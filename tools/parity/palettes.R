#!/usr/bin/env Rscript
# Golden values for the palette layer: grDevices::colorRampPalette and SCP's
# palette_scp (Hao Zhang, NJMU, GPL-3; https://github.com/zhanghao-njmu/SCP,
# SCP-plot.R:168-297), checked by tests/test_palettes.py.
#
#   Rscript tools/parity/palettes.R [out_dir]
#
# Writes tests/fixtures/parity/palettes/palettes.json:
#   ramps       colorRampPalette(<colors>)(n) for several palettes and n
#   continuous  palette_scp(palette = p, n = n): the unmatched continuous
#               branch, which takes the first n colours verbatim when n does
#               not exceed the palette
#   matched     palette_scp(x, type = "continuous", matched = TRUE, ...): the
#               per-value colours, cut() into n equal-width bins
#   discrete    palette_scp(x, type = "discrete", matched = TRUE, ...)
# NA colours are written as null.

suppressPackageStartupMessages({
  library(SCP)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) > 0) args[[1]] else "tests/fixtures/parity/palettes"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
pl <- SCP::palette_list

ramp_sets <- list(
  Spectral = pl[["Spectral"]], RdBu = pl[["RdBu"]], Paired = pl[["Paired"]],
  viridis = pl[["viridis"]], cividis = pl[["cividis"]], simspec = pl[["simspec"]],
  Set1 = pl[["Set1"]], Dark2 = pl[["Dark2"]],
  black_white = c("#000000", "#FFFFFF"),
  gsea_pos = c("#F5DCDC", "#C40003"), gsea_neg = c("#1D008F", "#DDDCF5")
)
ns <- c(1, 2, 3, 5, 10, 11, 12, 20, 50, 100, 200, 256)
ramps <- lapply(names(ramp_sets), function(nm) {
  list(name = nm, colors = unname(as.character(ramp_sets[[nm]])),
       ramps = setNames(lapply(ns, function(n) colorRampPalette(ramp_sets[[nm]])(n)),
                        as.character(ns)))
})

continuous <- list()
for (p in c("Spectral", "RdBu", "viridis", "Paired")) {
  for (n in c(5, 11, 12, 100)) {
    continuous[[length(continuous) + 1]] <- list(
      palette = p, n = n, colors = unname(palette_scp(palette = p, n = n)))
  }
}

set.seed(11)
xs <- list(
  ramp = seq(-1, 1, length.out = 37),
  runif = round(runif(200, -3, 5), 6),
  breaks = seq(0, 1, length.out = 101),
  with_na = c(0.1, NA, 0.7, 0.35, NA, 1.9, -0.4),
  ties = c(3, 1, 2, 2, 1, 3, 3),
  constant = c(2, 2, NA, 2),
  integers = 1:30
)
matched <- list()
for (nm in names(xs)) {
  for (p in c("Spectral", "RdBu")) {
    for (n in c(100, 10)) {
      for (keep in c(FALSE, TRUE)) {
        col <- palette_scp(xs[[nm]], type = "continuous", palette = p, n = n,
                           matched = TRUE, NA_keep = keep)
        matched[[length(matched) + 1]] <- list(
          x = nm, palette = p, n = n, NA_keep = keep,
          values = xs[[nm]], colors = unname(as.character(col)))
      }
    }
  }
}

discrete <- list()
dx <- list(letters = c("b", "a", NA, "c", "a", "b"),
           many = paste0("L", c(1:15, 3, NA, 7)))
for (nm in names(dx)) {
  for (keep in c(FALSE, TRUE)) {
    col <- palette_scp(dx[[nm]], type = "discrete", palette = "Paired",
                       matched = TRUE, NA_keep = keep)
    discrete[[length(discrete) + 1]] <- list(
      x = nm, NA_keep = keep, values = dx[[nm]], colors = unname(as.character(col)))
  }
}

write_json(list(scp_version = as.character(packageVersion("SCP")),
                r_version = R.version.string,
                ramps = ramps, continuous = continuous, matched = matched,
                discrete = discrete),
           file.path(out, "palettes.json"), auto_unbox = TRUE, digits = NA,
           na = "null", null = "null", pretty = FALSE)
message("palette fixtures -> ", normalizePath(out))
