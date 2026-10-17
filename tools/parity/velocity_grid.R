#!/usr/bin/env Rscript
# Pin SCP's compute_velocity_on_grid (SCP-plot.R:6963-7039; SCP by Hao Zhang,
# NJMU, GPL-3 -- https://github.com/zhanghao-njmu/SCP) on the trajectory audit's
# synthetic velocity field, for the numpy port in scp.pl.compute_velocity_on_grid.
#
#   Rscript tools/parity/velocity_grid.R [out_dir]
#
# Reads the inputs that audit_traj.R already wrote (inputs.csv for the UMAP,
# velocity_input.csv for the field), so both sides start from the same
# 12-significant-digit numbers rather than R's in-memory doubles.
#
# * velocity_grid_opts.csv -- grid mode with every knob moved off its default
#   (density = 0.5, smooth = 1, min_mass = 3, scale = 2, n_neighbors = 15).
#   VelocityPlot's own default grid is velocity_grid_layers.csv (audit_traj.R).
# * velocity_stream_grid.csv -- adjust_for_stream = TRUE at density = 0.6, the
#   (2, ns, ns) field flattened with its x/y grid indices. R's NA mask (first
#   component only, as in scvelo) is kept as NA.
#
# compute_velocity_on_grid calls reticulate::array_reshape in stream mode,
# which needs numpy through reticulate; nothing else leaves R.

suppressPackageStartupMessages(library(SCP))

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) > 0) args[[1]] else "tests/fixtures/parity/audit/traj"

inp <- read.csv(file.path(out, "inputs.csv"))
vel <- read.csv(file.path(out, "velocity_input.csv"))
stopifnot(identical(inp$cell, vel$cell))
X <- as.matrix(inp[, c("UMAP_1", "UMAP_2")])
V <- as.matrix(vel[, c("V1", "V2")])

num <- function(x) signif(x, 12)

res <- SCP:::compute_velocity_on_grid(X, V, density = 0.5, smooth = 1, n_neighbors = 15,
                                      min_mass = 3, scale = 2)
write.csv(data.frame(x = num(res$X_grid[, 1]), y = num(res$X_grid[, 2]),
                     u = num(res$V_grid[, 1]), v = num(res$V_grid[, 2])),
          file.path(out, "velocity_grid_opts.csv"), row.names = FALSE)

res <- SCP:::compute_velocity_on_grid(X, V, density = 0.6, smooth = 0.5, n_neighbors = 20,
                                      min_mass = 1, scale = 1, cutoff_perc = 5,
                                      adjust_for_stream = TRUE)
ns <- ncol(res$X_grid)
# V_grid[d, j, i]: row j of the y grid, column i of the x grid
ij <- expand.grid(i = seq_len(ns), j = seq_len(ns))
write.csv(data.frame(i = ij$i, j = ij$j,
                     x = num(res$X_grid[1, ij$i]), y = num(res$X_grid[2, ij$j]),
                     u = num(res$V_grid[cbind(1, ij$j, ij$i)]),
                     v = num(res$V_grid[cbind(2, ij$j, ij$i)])),
          file.path(out, "velocity_stream_grid.csv"), row.names = FALSE, na = "")

message("done -> ", normalizePath(out))
