#!/usr/bin/env Rscript
# Dump what SCP's StatPlot builds for plot_type = "sankey", "chord", "venn" and
# for individual = TRUE, on pancreas_sub, so tests/test_stat_flow.py can compare
# numbers rather than pictures.
#
#   Rscript tools/parity/cell_stat_flow.R [out_dir] [ggVennDiagram_src]
#
# out_dir defaults to tests/fixtures/parity/cell_stat. The output is one JSON
# file holding, per case, the *aggregated* input (combination counts and factor
# levels -- enough for the Python side to rebuild an equivalent frame without
# the 20 MB dataset) next to what R computed from it.
#
# * sankey -- StatPlot draws a ggplot it never returns (it is wrapped in a
#   gtable to splice the legends on), so the few lines that build it
#   (SCP-plot.R:5072-5113) are repeated verbatim below and the plot is then
#   ggplot_build()-ed. Node rectangles come from layer 2; flow ribbons from
#   layer 1, whose polygons are two 300-point sigmoids (upper edge out, lower
#   edge back), so rows 1/300/301/600 of each group are the four corners.
# * chord -- the chordDiagram() call of SCP-plot.R:5115-5124, verbatim; its
#   invisible return value carries every link's position, and the sector
#   degrees are read back with get.cell.meta.data() before circos.clear().
# * venn -- ggVennDiagram is not installed here (and SCP's venn code needs the
#   sf-era 1.2.x API, which no current ggVennDiagram provides). When a source
#   checkout is given, its Venn class and set operations are sourced so the
#   region counts come from the package's own code; otherwise a base-R
#   equivalent of process_region_data(specific = TRUE) is used.
# * individual -- StatPlot(..., individual = TRUE, combine = FALSE) is called
#   for real; panel names and each panel's p$data are dumped.

suppressPackageStartupMessages({
  library(SCP); library(ggplot2); library(dplyr); library(circlize); library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[[1]] else "tests/fixtures/parity/cell_stat"
gvd_src <- if (length(args) >= 2) args[[2]] else NA
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

data(pancreas_sub, package = "SCP")
md <- pancreas_sub@meta.data
counts <- pancreas_sub[["RNA"]]@counts
md$Progenitor <- md$CellType %in% c("Ngn3 low EP", "Ngn3 high EP")
md$G2M <- md$Phase == "G2M"
md$Sox9_Expressed <- counts["Sox9", rownames(md)] > 0
md$Neurog3_Expressed <- counts["Neurog3", rownames(md)] > 0
md$S_phase <- md$Phase == "S"

combos <- function(df, cols) {
  # counts of every observed combination, in first-appearance order
  key <- do.call(paste, c(lapply(cols, function(c) as.character(df[[c]])), sep = "\r"))
  u <- unique(key)
  first <- df[match(u, key), cols, drop = FALSE]
  out <- lapply(first, as.character)
  out$n <- as.integer(table(factor(key, levels = u)))
  as.data.frame(out, stringsAsFactors = FALSE, check.names = FALSE)
}
levels_of <- function(df, cols) {
  lapply(setNames(cols, cols), function(c) {
    x <- df[[c]]
    if (is.factor(x)) levels(x) else as.character(unique(x))
  })
}

# ---- sankey ---------------------------------------------------------------- #
sankey_case <- function(stat.by, palette = "Paired", alpha = 1, NA_color = "grey") {
  dat_all <- md[, stat.by, drop = FALSE]
  dat_use <- dat_all
  # SCP-plot.R:5073
  colors <- palette_scp(c(unique(unlist(lapply(dat_all[, stat.by, drop = FALSE], levels))), NA),
                        palette = palette, NA_keep = TRUE, NA_color = NA_color)
  for (l in stat.by) {
    if (any(is.na(dat_use[[l]]))) {
      raw_levels <- levels(dat_use[[l]])
      dat_use[[l]] <- as.character(dat_use[[l]])
      dat_use[[l]][is.na(dat_use[[l]])] <- "NA"
      dat_use[[l]] <- factor(dat_use[[l]], levels = c(raw_levels, "NA"))
    }
  }
  dat <- suppressWarnings(make_long(dat_use, all_of(stat.by)))
  dat$node <- factor(dat$node, levels = rev(names(colors)))
  p0 <- ggplot(dat, aes(x = x, next_x = next_x, node = node, next_node = next_node, fill = node)) +
    geom_sankey(color = "black", flow.alpha = alpha, show.legend = FALSE, na.rm = FALSE) +
    scale_fill_manual(values = colors, drop = FALSE) +
    scale_x_discrete(expand = c(0, 0.2)) +
    theme_void() +
    theme(axis.text.x = element_text())
  b <- ggplot_build(p0)
  fill_to_name <- setNames(names(colors), toupper(colors))

  nd <- b$data[[2]]
  nodes <- data.frame(
    column = stat.by[round(nd$x)], node = as.character(nd$node),
    xmin = nd$xmin, xmax = nd$xmax, ymin = nd$ymin, ymax = nd$ymax,
    fill = toupper(nd$fill), freq = nd$freq, stringsAsFactors = FALSE
  )
  fl <- b$data[[1]]
  # The sigmoid runs over logistic(-smooth..smooth), which never quite reaches
  # 0 or 1, so the first/last points sit a hair inside the true corners. Invert
  # that to recover the exact flow_start/flow_end values the stat computed.
  s0 <- 1 / (1 + exp(8)); s1 <- 1 / (1 + exp(-8))
  ends <- function(y_first, y_last) {
    span <- (y_last - y_first) / (s1 - s0)
    c(from = y_first - s0 * span, to = y_first - s0 * span + span)
  }
  flows <- do.call(rbind, lapply(split(fl, fl$group), function(g) {
    stopifnot(nrow(g) == 600)
    xs <- g$x[1]; xe <- g$x[300]
    top <- ends(g$y[1], g$y[300]); bot <- ends(g$y[301], g$y[600])
    sy_top <- top[["from"]]; ey_top <- top[["to"]]; ey_bot <- bot[["from"]]; sy_bot <- bot[["to"]]
    tol <- 1e-6
    on_src <- nodes[abs(nodes$xmax - xs) < tol & nodes$ymin <= sy_bot + tol & nodes$ymax >= sy_top - tol, ]
    on_dst <- nodes[abs(nodes$xmin - xe) < tol & nodes$ymin <= ey_bot + tol & nodes$ymax >= ey_top - tol, ]
    stopifnot(nrow(on_src) == 1, nrow(on_dst) == 1)
    data.frame(
      source_column = on_src$column[1], source = on_src$node[1],
      target_column = on_dst$column[1], target = on_dst$node[1],
      start_ymin = sy_bot, start_ymax = sy_top, end_ymin = ey_bot, end_ymax = ey_top,
      x_start = xs, x_end = xe, fill = toupper(g$fill[1]), alpha = g$alpha[1],
      stringsAsFactors = FALSE
    )
  }))
  rownames(flows) <- NULL
  list(
    stat_by = stat.by, levels = levels_of(dat_all, stat.by),
    combos = combos(dat_all, stat.by),
    colors = as.list(toupper(colors)),
    x_range = b$layout$panel_params[[1]]$x.range,
    nodes = nodes, flows = flows
  )
}

# ---- chord ----------------------------------------------------------------- #
chord_case <- function(stat.by, palette = "Paired", NA_color = "grey") {
  dat_all <- md[, stat.by, drop = FALSE]
  dat_use <- dat_all
  # SCP-plot.R:5116-5124
  colors <- palette_scp(c(unique(unlist(lapply(dat_all[, stat.by, drop = FALSE], levels))), NA),
                        palette = palette, NA_keep = TRUE, NA_color = NA_color)
  M <- table(dat_use[[stat.by[1]]], dat_use[[stat.by[2]]], useNA = "ifany")
  m <- matrix(M, ncol = ncol(M), dimnames = dimnames(M))
  colnames(m)[is.na(colnames(m))] <- "NA"
  png(tempfile(fileext = ".png"))
  df <- chordDiagram(m, grid.col = colors, transparency = 0.2, link.lwd = 1, link.lty = 1, link.border = 1)
  sectors <- get.all.sector.index()
  sec <- do.call(rbind, lapply(sectors, function(s) {
    data.frame(
      name = s,
      start_degree = get.cell.meta.data("cell.start.degree", sector.index = s, track.index = 1),
      end_degree = get.cell.meta.data("cell.end.degree", sector.index = s, track.index = 1),
      xmax = get.cell.meta.data("xlim", sector.index = s, track.index = 1)[2],
      stringsAsFactors = FALSE
    )
  }))
  circos.clear()
  dev.off()
  list(
    stat_by = stat.by, levels = levels_of(dat_all, stat.by),
    combos = combos(dat_all, stat.by),
    colors = as.list(toupper(colors)),
    matrix = list(rows = rownames(m), cols = colnames(m), values = unname(as.matrix(m))),
    sectors = sec,
    links = data.frame(rn = df$rn, cn = df$cn, value = df$value1, x1 = df$x1, x2 = df$x2,
                       col = toupper(df$col), stringsAsFactors = FALSE)
  )
}

# ---- venn ------------------------------------------------------------------ #
region_counts <- NULL
if (!is.na(gvd_src)) {
  ok <- tryCatch({
    suppressPackageStartupMessages(library(methods))
    for (f in c("classVenn.R", "setOperations.R", "regions.R", "process_data.R")) {
      source(file.path(gvd_src, "R", f), local = globalenv())
    }
    region_counts <- function(sets) {
      rd <- process_region_data(Venn(sets))
      data.frame(id = rd$id, name = rd$name, count = rd$count, stringsAsFactors = FALSE)
    }
    TRUE
  }, error = function(e) {
    message("ggVennDiagram source not usable (", conditionMessage(e), "); using base R")
    FALSE
  })
}
venn_source <- if (is.null(region_counts)) "base R" else "ggVennDiagram sources"
if (is.null(region_counts)) {
  region_counts <- function(sets) {
    n <- length(sets)
    ids <- character(); names <- character(); counts <- integer()
    universe <- Reduce(union, sets)
    for (k in seq_len(n)) {
      cb <- combn(n, k)
      for (j in seq_len(ncol(cb))) {
        idx <- cb[, j]
        inside <- Reduce(intersect, sets[idx])
        outside <- if (k < n) Reduce(union, sets[-idx]) else character()
        ids <- c(ids, paste(idx, collapse = "/"))
        names <- c(names, paste(names(sets)[idx], collapse = "/"))
        counts <- c(counts, length(setdiff(inside, outside)))
      }
    }
    data.frame(id = ids, name = names, count = counts, stringsAsFactors = FALSE)
  }
}

venn_case <- function(stat.by, stat_level) {
  meta <- md
  if (length(stat_level) == 1) stat_level <- rep(stat_level, length(stat.by))
  if (is.null(names(stat_level))) names(stat_level) <- stat.by
  dat_raw <- meta[, stat.by, drop = FALSE]
  for (i in stat.by) meta[[i]] <- meta[[i]] %in% stat_level[[i]]   # SCP-plot.R:4666
  dat_use <- meta[, stat.by, drop = FALSE]
  sets <- lapply(setNames(stat.by, stat.by), function(x) rownames(dat_use)[dat_use[[x]]])
  rc <- region_counts(sets)
  rc$pct <- round(rc$count / sum(rc$count) * 100, 1)                  # SCP-plot.R:4994
  list(
    stat_by = stat.by, stat_level = lapply(stat_level, as.character),
    levels = levels_of(dat_raw, stat.by),
    combos = combos(dat_raw, stat.by),
    set_sizes = as.list(sapply(sets, length)),
    set_colors = as.list(toupper(palette_scp(stat.by, palette = "Paired"))),
    regions = rc
  )
}

# ---- individual ------------------------------------------------------------ #
individual_case <- function(stat.by, group.by, split.by, stat_type) {
  pl <- suppressWarnings(StatPlot(md, stat.by = stat.by, group.by = group.by, split.by = split.by,
                                  plot_type = "bar", stat_type = stat_type, individual = TRUE,
                                  combine = FALSE))
  panels <- lapply(names(pl), function(nm) {
    d <- pl[[nm]]$data
    list(name = nm, title = pl[[nm]]$labels$title %||% "",
         stat = as.character(d[[stat.by]]), group = as.character(d[[group.by]]),
         value = d$value)
  })
  list(stat_by = stat.by, group_by = group.by, split_by = split.by, stat_type = stat_type,
       levels = levels_of(md, c(stat.by, group.by, split.by)),
       combos = combos(md, c(stat.by, group.by, split.by)),
       panels = panels)
}

# Whether StatPlot itself gets through these paths in this R stack. The sankey
# one fails after the plot is built, while splicing the per-variable legends
# onto the gtable (add_grob on a cbind-ed legend list); the replicated code
# above stops before that step, so the geometry is unaffected.
try_statplot <- function(...) {
  pdf(NULL)  # StatPlot draws; keep it from leaving an Rplots.pdf behind
  on.exit(dev.off())
  tryCatch({
    suppressWarnings(StatPlot(md, ...))
    "ok"
  }, error = function(e) paste("error:", conditionMessage(e)))
}
statplot_runs <- list(
  sankey = try_statplot(stat.by = c("CellType", "SubCellType", "Phase"), plot_type = "sankey"),
  chord = try_statplot(stat.by = c("CellType", "Phase"), plot_type = "chord")
  # venn is deliberately not called: StatPlot's check_R() tries to *install*
  # ggVennDiagram and sf into the library when they are missing.
)
print(statplot_runs)

out <- list(
  provenance = list(
    R = R.version.string, SCP = as.character(packageVersion("SCP")),
    circlize = as.character(packageVersion("circlize")),
    ggplot2 = as.character(packageVersion("ggplot2")), venn_counts = venn_source,
    dataset = "pancreas_sub (SCP)", statplot_runs = statplot_runs
  ),
  sankey = sankey_case(c("CellType", "SubCellType", "Phase")),
  sankey_alpha = sankey_case(c("Phase", "CellType"), alpha = 0.5),
  chord = chord_case(c("CellType", "Phase")),
  chord_shared = chord_case(c("CellType", "SubCellType")),
  venn4 = venn_case(c("Progenitor", "G2M", "Sox9_Expressed", "Neurog3_Expressed"), "TRUE"),
  venn5 = venn_case(c("Progenitor", "G2M", "Sox9_Expressed", "Neurog3_Expressed", "S_phase"), "TRUE"),
  individual = individual_case("Phase", "CellType", "Phase", "percent"),
  individual_count = individual_case("Phase", "CellType", NULL, "count")
)
path <- file.path(out_dir, "flow_parity.json")
writeLines(toJSON(out, dataframe = "columns", digits = NA, auto_unbox = TRUE, null = "null"), path)
message("wrote ", path)
