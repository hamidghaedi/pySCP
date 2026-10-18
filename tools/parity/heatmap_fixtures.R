#!/usr/bin/env Rscript
# Pin the heatmap family's deterministic intermediates from SCP, as small CSVs.
#
#   Rscript tools/parity/heatmap_fixtures.R [out_dir]
#
# Writes to tests/fixtures/parity/heatmap/ by default. The Python suite reads
# these files and never needs R: every fixture carries its own *input* as well
# as SCP's output, so tests rebuild a small AnnData from the CSVs instead of
# reading pancreas_sub.h5ad.
#
# Two packages SCP reaches for are absent from the parity environment and SCP's
# check_R() would try to *install* them, so neither path is allowed to call it:
#
# * dendextend -- cluster_within_group2() only needs its `order.dendrogram<-`
#   setter. When dendextend is missing, a stub package providing exactly that
#   is built in tempdir() and put first on .libPaths(); nothing is installed
#   into the environment.
# * e1071 -- split_method = "mfuzz" is never requested here (SCP would fall
#   back to kmeans with a warning anyway), so mfuzz has no R fixture.

suppressPackageStartupMessages({
  library(SCP)
  library(Seurat)
  library(ComplexHeatmap)
  library(Matrix)
  library(jsonlite)
})
# ComplexHeatmap draws in order to lay out; route every device it opens to a
# null PDF so no Rplots.pdf is left behind
options(device = function(...) grDevices::pdf(NULL))
grDevices::pdf(NULL)

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) > 0) args[[1]] else "tests/fixtures/parity/heatmap"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
wcsv <- function(x, name, ...) {
  utils::write.csv(x, file.path(out, name), quote = TRUE, ...)
}

# --- dendextend stub ---------------------------------------------------------
stub_lib <- file.path(tempdir(), "stublib")
if (!requireNamespace("dendextend", quietly = TRUE)) {
  pkg <- file.path(tempdir(), "dendextend")
  dir.create(file.path(pkg, "R"), recursive = TRUE, showWarnings = FALSE)
  writeLines(c(
    "Package: dendextend", "Version: 0.0.0.9000", "Title: stub",
    "Description: order.dendrogram<- only.", "License: GPL-3",
    "Author: parity stub", "Maintainer: parity stub <stub@localhost>"
  ), file.path(pkg, "DESCRIPTION"))
  writeLines("export(\"order.dendrogram<-\")", file.path(pkg, "NAMESPACE"))
  writeLines(c(
    "`order.dendrogram<-` <- function(object, ..., value) {",
    "  i <- 0L",
    "  stats::dendrapply(object, function(node) {",
    "    if (stats::is.leaf(node)) { i <<- i + 1L; node[[1]] <- value[i] }",
    "    node",
    "  })",
    "}"
  ), file.path(pkg, "R", "order.R"))
  dir.create(stub_lib, showWarnings = FALSE)
  install.packages(pkg, lib = stub_lib, repos = NULL, type = "source", quiet = TRUE)
  .libPaths(c(stub_lib, .libPaths()))
}
# cluster_within_group2() calls `order.dendrogram<-` unqualified and SCP does
# not import it, so it only resolves when dendextend is *attached* -- an R bug
# that makes cluster_features_by-with-a-split fail in a fresh session.
suppressPackageStartupMessages(library(dendextend))

# SCP 0.5.6 still calls GetAssayData(slot=) / AverageExpression(slot=), which
# SeuratObject 5 made defunct. Translate slot -> layer inside SCP's import
# environment only; the plotting functions under test are untouched.
local({
  imp <- parent.env(asNamespace("SCP"))
  shim <- function(name, fun) {
    if (exists(name, envir = imp, inherits = FALSE)) {
      unlockBinding(name, imp)
      assign(name, fun, envir = imp)
      lockBinding(name, imp)
    }
  }
  shim("GetAssayData", function(object, slot = "data", assay = NULL, ...) {
    SeuratObject::GetAssayData(object, layer = slot, assay = assay, ...)
  })
  shim("AverageExpression", function(object, ..., slot = "data") {
    Seurat::AverageExpression(object, ..., layer = slot)
  })
})

data(pancreas_sub, package = "SCP")
srt <- pancreas_sub

features <- c(
  "Sox9", "Anxa2", "Neurog3", "Hes6", "Fev", "Neurod1", "Rbp4", "Pyy", "Ins1",
  "Gcg", "Sst", "Ghrl", "Krt19", "Spp1", "Mki67", "Top2a", "Cdk1", "Pax4",
  "Chga", "Chgb", "Iapp", "Pcsk2", "Nkx6-1", "Pdx1", "Mafb", "Arx", "Cck",
  "Gast", "Vim", "Hhex"
)

# Two synthetic lineages: pseudotime is UMAP_1 shifted positive, restricted to
# the cells on the path towards Beta (Lineage1) or Alpha (Lineage2).
umap <- Embeddings(srt, "UMAP")
pt <- umap[, 1] - min(umap[, 1]) + 0.1
on1 <- !srt$SubCellType %in% c("Alpha", "Delta", "Epsilon")
on2 <- !srt$SubCellType %in% c("Beta", "Delta", "Epsilon")
srt$Lineage1 <- ifelse(on1, pt, NA)
srt$Lineage2 <- ifelse(on2, pt, NA)
# a deterministic two-level condition, interleaved so every group has both arms
srt$Batch <- factor(rep(c("b1", "b2"), length.out = ncol(srt)), levels = c("b1", "b2"))

# --- inputs ------------------------------------------------------------------
counts <- GetAssayData(srt, assay = "RNA", layer = "counts")
wcsv(as.matrix(t(counts[features, ])), "input_counts.csv")
obs <- data.frame(
  row.names = colnames(srt),
  CellType = as.character(srt$CellType),
  SubCellType = as.character(srt$SubCellType),
  Phase = as.character(srt$Phase),
  Batch = as.character(srt$Batch),
  libsize = colSums(counts),
  Lineage1 = srt$Lineage1,
  Lineage2 = srt$Lineage2
)
wcsv(obs, "input_obs.csv")
# a numeric cell annotation for the GroupHeatmap violins (anno_violin.csv)
wcsv(data.frame(row.names = colnames(srt), S_score = signif(srt$S_score, 12)), "input_scores.csv")
writeLines(toJSON(list(
  CellType = levels(srt$CellType),
  SubCellType = levels(srt$SubCellType),
  Phase = levels(srt$Phase)
), pretty = TRUE), file.path(out, "input_levels.json"))

long <- function(m, name) {
  df <- as.data.frame(as.table(as.matrix(m)), stringsAsFactors = FALSE)
  colnames(df) <- c("feature", "column", "value")
  df$value <- signif(df$value, 10)
  wcsv(df, name, row.names = FALSE)
}
fmeta <- function(ht, name) {
  fm <- ht$feature_metadata
  df <- data.frame(
    feature = rownames(fm),
    feature_split = as.character(fm$feature_split),
    index = as.integer(fm$index)
  )
  wcsv(df, name, row.names = FALSE)
}

# --- GroupHeatmap: split.by gives "group : split" columns --------------------
ht <- GroupHeatmap(srt, features = features, group.by = "CellType", split.by = "Phase")
long(ht$matrix_list$CellType, "group_split_by.csv")

# column order under cluster_column_slices: SCP hands ComplexHeatmap the stock
# cluster_within_group() of the processed matrix and the group factor
m <- ht$matrix_list$CellType
csplit <- factor(gsub(" : .*", "", colnames(m)), levels = levels(srt$CellType))
dend <- ComplexHeatmap::cluster_within_group(m, csplit)
wcsv(data.frame(column = colnames(m)[order.dendrogram(dend)]),
     "group_cluster_column_slices.csv", row.names = FALSE)
# cluster_columns = TRUE with that split: ComplexHeatmap clusters inside each
# column slice and reorders by -colMeans (SCP passes these to Heatmap() as is)
ord <- function(...) unlist(suppressWarnings(column_order(Heatmap(m, ...))))
wcsv(data.frame(column = colnames(m)[ord(column_split = csplit, cluster_columns = TRUE,
                                         cluster_column_slices = FALSE)]),
     "group_cluster_columns.csv", row.names = FALSE)
wcsv(data.frame(column = colnames(m)[ord(column_split = csplit, cluster_columns = TRUE,
                                         cluster_column_slices = TRUE)]),
     "group_cluster_columns_and_slices.csv", row.names = FALSE)

# --- GroupHeatmap: grouping.var -> log2(numerator / other) -------------------
ht <- suppressWarnings(GroupHeatmap(srt, features = features, group.by = "CellType",
                                    grouping.var = "Phase", numerator = "G2M"))
long(ht$matrix_list$CellType, "group_grouping_var.csv")
# R bug: with split.by, column_split is built from every "group : split" level
# while the log2 matrix keeps only the combinations holding both arms, so any
# one-armed combination aborts with "Length or ncol of `column_split` should be
# same as ncol of `matrix`". Pre-endocrine has a single S-phase cell, so the
# fixture drops it; the Python port builds the split from the kept columns.
keep <- colnames(srt)[!(srt$CellType == "Pre-endocrine" & srt$Phase == "S")]
ht <- suppressWarnings(GroupHeatmap(srt, features = features, group.by = "CellType",
                                    split.by = "Phase", grouping.var = "Batch",
                                    numerator = "b2", cells = keep))
long(ht$matrix_list$CellType, "group_grouping_var_split.csv")

# --- GroupHeatmap: feature splits and row orders ------------------------------
gb <- c("CellType", "SubCellType")
ht <- GroupHeatmap(srt, features = features, group.by = gb, n_split = 4,
                   split_method = "hclust")
fmeta(ht, "group_hclust.csv")
long(do.call(cbind, ht$matrix_list), "group_mat_split.csv")
# the Mfuzz fuzzifier estimate SCP's mfuzz split would use on this mat_split
# (e1071 itself is absent, see the header)
mt <- SCP:::standardise(do.call(cbind, ht$matrix_list))
writeLines(toJSON(list(mestimate = SCP:::mestimate(mt)), digits = NA, auto_unbox = TRUE),
           file.path(out, "mfuzz_mestimate.json"))
ht <- GroupHeatmap(srt, features = features, group.by = gb, n_split = 4,
                   split_method = "hclust", feature_split_by = "SubCellType")
fmeta(ht, "group_hclust_split_by.csv")
ht <- GroupHeatmap(srt, features = features, group.by = gb, n_split = 4,
                   split_method = "hclust", decreasing = TRUE)
fmeta(ht, "group_hclust_decreasing.csv")
ht <- GroupHeatmap(srt, features = features, group.by = gb, n_split = 4,
                   split_method = "kmeans")
fmeta(ht, "group_kmeans.csv")
ht <- GroupHeatmap(srt, features = features, group.by = gb, n_split = 4,
                   split_method = "hclust", cluster_rows = TRUE)
fmeta(ht, "group_cluster_rows_split.csv")
ht <- GroupHeatmap(srt, features = features, group.by = gb, cluster_rows = TRUE)
fmeta(ht, "group_cluster_rows.csv")
ht <- GroupHeatmap(srt, features = features, group.by = gb, n_split = 4,
                   split_method = "hclust", cluster_row_slices = TRUE)
fmeta(ht, "group_cluster_row_slices.csv")
ht <- GroupHeatmap(srt, features = features, group.by = gb, n_split = 4,
                   split_method = "hclust", cluster_rows = TRUE, cluster_row_slices = TRUE)
fmeta(ht, "group_cluster_rows_and_slices.csv")
ht <- GroupHeatmap(srt, features = features, group.by = gb, n_split = 4,
                   split_method = "hclust", cluster_rows = TRUE,
                   cluster_features_by = "SubCellType")
fmeta(ht, "group_cluster_features_by.csv")
# (cluster_features_by without a split has no fixture: R reorders that tree by
# colMeans, one weight per column for a tree of rows, and the port uses row
# means instead.)
# A z-scored row has mean 0, so ComplexHeatmap's -rowMeans reorder weights are
# floating-point noise and the zscore orders above are noise-driven flips of
# the tree. exp_method = "raw" gives weights that mean something.
ht <- GroupHeatmap(srt, features = features, group.by = gb, cluster_rows = TRUE,
                   exp_method = "raw")
fmeta(ht, "group_raw_cluster_rows.csv")
ht <- GroupHeatmap(srt, features = features, group.by = gb, n_split = 4,
                   split_method = "hclust", cluster_rows = TRUE,
                   cluster_row_slices = TRUE, exp_method = "raw")
fmeta(ht, "group_raw_cluster_rows_and_slices.csv")

# --- FeatureHeatmap: per cell; max_cells above every group size so nothing is
# subsampled (sample() still permutes, so compare by cell name) ---------------
ht <- FeatureHeatmap(srt, features = features, group.by = "CellType", max_cells = 1000,
                     n_split = 4, split_method = "hclust")
fmeta(ht, "feature_hclust.csv")
m <- ht$matrix_list$CellType
long(m[c("Ins1", "Sox9"), sort(colnames(m))[1:200]], "feature_matrix.csv")

# --- DynamicHeatmap: per-cell columns, z-scored jointly across lineages ------
srt <- RunDynamicFeatures(srt, lineages = c("Lineage1", "Lineage2"), features = features,
                          BPPARAM = BiocParallel::SerialParam())
for (l in c("Lineage1", "Lineage2")) {
  df <- srt@tools[[paste0("DynamicFeatures_", l)]][["DynamicFeatures"]]
  df <- df[, c("features", "exp_ncells", "r.sq", "dev.expl", "peaktime", "valleytime",
               "pvalue", "padjust")]
  df[, -1] <- signif(df[, -1], 10)
  wcsv(df, paste0("dynamic_features_", l, ".csv"), row.names = FALSE)
}
dyn <- function(name, ...) {
  ht <- DynamicHeatmap(srt, lineages = c("Lineage1", "Lineage2"), features = features, ...)
  fmeta(ht, name)
  ht
}
ht <- dyn("dynamic_hclust_peaktime.csv", n_split = 4, split_method = "hclust-peaktime")
m <- ht$matrix
pick <- unlist(lapply(ht$cell_order, function(x) x[seq(1, length(x), by = 7)]))
long(m[c("Ins1", "Gcg", "Sox9"), pick], "dynamic_matrix.csv")
wcsv(data.frame(lineage = rep(names(ht$cell_order), lengths(ht$cell_order)),
                column = unlist(ht$cell_order)),
     "dynamic_cell_order.csv", row.names = FALSE)
dyn("dynamic_kmeans_peaktime.csv", n_split = 4, split_method = "kmeans-peaktime")
dyn("dynamic_hclust.csv", n_split = 4, split_method = "hclust")
dyn("dynamic_hclust_valleytime.csv", n_split = 3, split_method = "hclust-peaktime",
    order_by = "valleytime", reverse_ht = "Lineage2")
dyn("dynamic_cluster_features_by.csv", n_split = 4, split_method = "hclust",
    cluster_rows = TRUE, cluster_features_by = "Lineage1")
dyn("dynamic_feature_split_by.csv", n_split = 4, split_method = "hclust",
    feature_split_by = "Lineage2")

# --- CellCorHeatmap: every distance_metric SCP lists --------------------------
# CellCorHeatmap delegates to RunKNNPredict(nn_method = "raw"), whose metadata
# write-back fails under SeuratObject 5 ("No cell overlap between new meta data
# and Seurat object"), so the matrix is rebuilt here with the exact lines
# RunKNNPredict and CellCorHeatmap run (SCP-cell_annotation.R 266-410,
# SCP-plot.R 10183-10194). Query = SubCellType centroids, reference = CellType
# centroids, both log1p(AverageExpression(data)) on log-normalised data.
#
# "hamman" is in SCP's list but proxyC spells it "hamann", so SCP errors on it
# (an R bug); the fixture asks proxyC for "hamann".
srt_n <- NormalizeData(srt, verbose = FALSE)
avg <- function(g) {
  a <- AverageExpression(srt_n, features = features, layer = "data", group.by = g,
                         assays = "RNA", verbose = FALSE)[[1]]
  t(log1p(as.matrix(a)))
}
ref <- avg("CellType")
qry <- avg("SubCellType")
wcsv(signif(ref, 12), "cor_ref.csv")
wcsv(signif(qry, 12), "cor_query.csv")
simil_method <- c("cosine", "pearson", "spearman", "correlation", "jaccard", "ejaccard",
                  "dice", "edice", "hamann", "simple matching", "faith")
dist_method <- c("euclidean", "chisquared", "kullback", "manhattan", "maximum", "canberra",
                 "minkowski", "hamming")
cor_out <- list()
for (mt in c(simil_method, dist_method)) {
  r <- ref
  q <- qry
  m <- mt
  if (m %in% simil_method) {
    if (m %in% c("pearson", "spearman")) {
      if (m == "spearman") {
        r <- t(apply(r, 1, rank))
        q <- t(apply(q, 1, rank))
      }
      m <- "correlation"
    }
    d <- 1 - proxyC::simil(x = as.sparse(r), y = as.sparse(q), method = m, use_nan = TRUE)
    s <- t(as.matrix(1 - d))
    nm <- paste0(SCP:::capitalize(m), " similarity")
  } else {
    d <- proxyC::dist(x = as.sparse(r), y = as.sparse(q), method = m, use_nan = TRUE)
    s <- t(as.matrix(1 - d / max(d, na.rm = TRUE)))
    nm <- paste0("1-dist[", m, "]/max(dist[", m, "])")
  }
  s[is.infinite(s)] <- max(abs(s[!is.infinite(s)]), na.rm = TRUE) * ifelse(s[is.infinite(s)] > 0, 1, -1)
  s[is.na(s)] <- 0
  df <- as.data.frame(as.table(s), stringsAsFactors = FALSE)
  colnames(df) <- c("query", "ref", "value")
  df$metric <- mt
  df$name <- nm
  cor_out[[mt]] <- df
}
cor <- do.call(rbind, cor_out)
cor$value <- signif(cor$value, 10)
wcsv(cor, "cell_cor_metrics.csv", row.names = FALSE)

# --- heatmap_enrichment: a synthetic TERM2GENE so no database is downloaded ---
set.seed(1)
pool <- unique(c(features, sample(rownames(srt), 300)))
words <- c("insulin", "secretion", "hormone", "endocrine", "pancreas", "development",
           "cell", "cycle", "division", "mitotic", "regulation", "signaling",
           "glucose", "response", "differentiation", "of", "the", "[obsolete]",
           "beta", "alpha", "transport", "process", "positive", "negative")
t2g <- do.call(rbind, lapply(seq_len(40), function(i) {
  n <- sample(10:40, 1)
  data.frame(Term = sprintf("T%02d", i),
             symbol = unique(c(sample(features, sample(2:8, 1)), sample(pool, n))))
}))
t2n <- data.frame(Term = sprintf("T%02d", 1:40),
                  Name = vapply(1:40, function(i) paste(sample(words, sample(2:5, 1)), collapse = " "),
                                character(1)))
wcsv(t2g, "enrich_term2gene.csv", row.names = FALSE)
wcsv(t2n, "enrich_term2name.csv", row.names = FALSE)
fs <- read.csv(file.path(out, "group_hclust.csv"))
res <- SCP:::heatmap_enrichment(
  geneID = fs$feature, geneID_groups = factor(fs$feature_split, levels = paste0("C", 1:4)),
  anno_terms = TRUE, anno_keys = TRUE, anno_features = TRUE,
  db = "custom", TERM2GENE = t2g, TERM2NAME = t2n, padjustCutoff = 1, topTerm = 5,
  topWord = 20
)
en <- res$res$enrichment
en <- en[, c("ID", "Description", "GeneRatio", "BgRatio", "pvalue", "p.adjust", "geneID",
             "Count", "Groups", "Database")]
en$pvalue <- signif(en$pvalue, 12)
en$p.adjust <- signif(en$p.adjust, 12)
wcsv(en, "enrich_table.csv", row.names = FALSE)
boxes <- list()
for (kind in c("terms", "keys", "features")) {
  # anno_textbox wraps anno_zoom; the per-group data frames live in the
  # closure of its panel_fun
  fun <- res$ha_right@anno_list[[paste0(kind, "_custom")]]@fun
  txt <- environment(fun@var_env$panel_fun)$text
  for (g in names(txt)) {
    d <- txt[[g]]
    if (is.null(d) || nrow(d) == 0) next
    boxes[[paste(kind, g)]] <- data.frame(
      kind = kind, group = g, keyword = d$keyword, col = d$col,
      fontsize = signif(d$fontsize, 10),
      score = if ("score" %in% colnames(d)) signif(d$score, 10) else NA,
      count = if ("count" %in% colnames(d)) d$count else NA
    )
  }
}
wcsv(do.call(rbind, boxes), "enrich_boxes.csv", row.names = FALSE)
wcsv(data.frame(word = SCP::words_excluded), "words_excluded.csv", row.names = FALSE)

# --- flip = TRUE ---------------------------------------------------------------
# Every heatmap swaps cluster_rows / cluster_columns (and the *_slices pair) on
# entry when flip = TRUE, so the user's flags name the DRAWN axes; the returned
# matrices are not transposed. flip + cluster_columns therefore orders the
# features as cluster_rows does without flip.
ht <- GroupHeatmap(srt, features = features, group.by = gb, flip = TRUE, cluster_columns = TRUE,
                   exp_method = "raw")
fmeta(ht, "group_flip_raw_cluster_columns.csv")
ht <- GroupHeatmap(srt, features = features, group.by = gb, n_split = 4, flip = TRUE,
                   split_method = "hclust", cluster_columns = TRUE, cluster_column_slices = TRUE,
                   exp_method = "raw")
fmeta(ht, "group_flip_raw_cluster_columns_and_slices.csv")
ht <- FeatureHeatmap(srt, features = features, group.by = "CellType", max_cells = 1000,
                     n_split = 4, split_method = "hclust", flip = TRUE)
fmeta(ht, "feature_flip_hclust.csv")
invisible(dyn("dynamic_flip_hclust_peaktime.csv", n_split = 4, split_method = "hclust-peaktime", flip = TRUE))

# --- cell_annotation in GroupHeatmap: a pie / violin per column ----------------
# GroupHeatmap draws CellStatPlot(plot_type = "pie") for a categorical
# annotation and FeatureStatPlot(fill.by = "group", same.y.lims = TRUE) for a
# numeric one, one subplot per column, each squeezed into the column's cell
# (SCP-plot.R:8363-8484); add_violin draws FeatureStatPlot(fill.by = fill.by)
# into every body cell (8768). The built layer data of those subplots is what
# the annotation shows: wedge extents and fills, violin outlines and the panel
# ranges they are drawn in.
# every k-th point of each outline (and its last) is plenty to compare shapes
thin <- function(d, by, k) {
  if (k <= 1) return(d)
  key <- if (length(by) == 1 && by %in% colnames(d)) d[[by]] else by
  i <- ave(seq_len(nrow(d)), key, FUN = seq_along)
  n <- ave(seq_len(nrow(d)), key, FUN = length)
  d[i %% k == 1 | i == n, , drop = FALSE]
}
built <- function(plots, name, keep = function(d) d, every = 1) {
  rows <- list()
  for (nm in names(plots)) {
    b <- ggplot2::ggplot_build(plots[[nm]])
    d <- b$data[[1]]
    d <- tryCatch(keep(d), error = function(e) NULL)
    if (is.null(d) || nrow(d) == 0) next  # a combination without cells
    d$subplot <- nm
    d$y_lo <- b$layout$panel_params[[1]]$y.range[1]
    d$y_hi <- b$layout$panel_params[[1]]$y.range[2]
    rows[[nm]] <- d
  }
  out <- thin(do.call(rbind, rows), "subplot", every)
  num <- vapply(out, is.numeric, logical(1))
  out[num] <- lapply(out[num], signif, 10)
  wcsv(out, name, row.names = FALSE)
}
pies <- CellStatPlot(srt, plot_type = "pie", stat.by = "Phase", group.by = "CellType",
                     palette = "Paired", title = NULL, individual = TRUE, combine = FALSE)
built(pies, "anno_pie.csv", function(d) d[, c("fill", "ymin", "ymax", "xmin", "xmax")])
pies <- CellStatPlot(srt, plot_type = "pie", stat.by = "Phase", group.by = "CellType",
                     split.by = "Batch", palette = "Paired", title = NULL,
                     individual = TRUE, combine = FALSE)
built(pies, "anno_pie_split.csv", function(d) d[, c("fill", "ymin", "ymax", "xmin", "xmax")])
# ggplot2 4.x appends repeated rows after the 512-point grid: sort and de-duplicate
violin_cols <- function(d) {
  d <- d[order(d$y), c("fill", "y", "violinwidth", "xmin", "xmax")]
  d[!duplicated(d$y), ]
}
vl <- FeatureStatPlot(srt, slot = "data", stat.by = "S_score", group.by = "CellType",
                      palette = "Paired", fill.by = "group", same.y.lims = TRUE,
                      individual = TRUE, combine = FALSE)
built(vl, "anno_violin.csv", violin_cols, every = 8)
vl <- FeatureStatPlot(srt, slot = "data", stat.by = c("Ins1", "Gcg", "Sox9"), group.by = "CellType",
                      palette = "Dark2", fill.by = "feature", same.y.lims = TRUE,
                      individual = TRUE, combine = FALSE)
built(vl, "add_violin.csv", violin_cols, every = 8)

# --- separate_annotation in DynamicHeatmap -------------------------------------
# A categorical column becomes CellDensityPlot(cells = <lineage cells>,
# features = <lineage>, x_order = "rank", decreasing = TRUE, reverse = <in
# reverse_ht>) and a feature vector DynamicPlot(compare_features = TRUE,
# x_order = "rank", add_rug = FALSE) (SCP-plot.R:11422-11536). The ridges and
# the fitted curves are dumped with their panel ranges, and the fits come with
# RunDynamicFeatures' fitted / upr / lwr matrices, which is what the Python side
# draws them from.
sep_cells <- function(l, decreasing = FALSE) {
  cl <- colnames(srt)[!is.na(srt[[l, drop = TRUE]])]
  cl[order(srt[[l, drop = TRUE]][!is.na(srt[[l, drop = TRUE]])], decreasing = decreasing)]
}
ridges <- list()
for (l in c("Lineage1", "Lineage2")) {
  p <- suppressMessages(CellDensityPlot(srt = srt, cells = sep_cells(l, l == "Lineage2"),
                                        group.by = "SubCellType", features = l, decreasing = TRUE,
                                        x_order = "rank", palette = "Paired",
                                        reverse = l == "Lineage2"))
  b <- suppressMessages(ggplot2::ggplot_build(p))
  d <- b$data[[1]][, c("x", "y", "ymin", "ymax", "fill", "group")]
  d$lineage <- l
  d$x_lo <- b$layout$panel_params[[1]]$x.range[1]
  d$x_hi <- b$layout$panel_params[[1]]$x.range[2]
  d$y_lo <- b$layout$panel_params[[1]]$y.range[1]
  d$y_hi <- b$layout$panel_params[[1]]$y.range[2]
  ridges[[l]] <- d
}
ridges <- do.call(rbind, ridges)
ridges <- thin(ridges, paste(ridges$lineage, ridges$group), 4)
num <- vapply(ridges, is.numeric, logical(1))
ridges[num] <- lapply(ridges[num], signif, 10)
wcsv(ridges, "separate_ridges.csv", row.names = FALSE)

sep_feats <- c("Ins1", "Gcg")
fits <- lapply(c("fitted_matrix", "upr_matrix", "lwr_matrix"), function(k) {
  m <- srt@tools[["DynamicFeatures_Lineage1"]][[k]][, sep_feats, drop = FALSE]
  data.frame(cell = rownames(m), matrix = k, signif(as.matrix(m), 9), check.names = FALSE)
})
wcsv(do.call(rbind, fits), "separate_fits_Lineage1.csv", row.names = FALSE)
p <- DynamicPlot(srt = srt, cells = sep_cells("Lineage1"), lineages = "Lineage1", group.by = NULL,
                 features = sep_feats, line_palette = "Paired", add_rug = FALSE,
                 legend.position = "none", compare_features = TRUE, x_order = "rank")
b <- ggplot2::ggplot_build(p)
curves <- list()
for (i in seq_along(b$data)) {
  d <- b$data[[i]]
  geom <- class(p$layers[[i]]$geom)[1]
  colnames(d) <- sub("_ggnewscale_[0-9]+$", "", colnames(d))
  keepc <- intersect(c("x", "y", "ymin", "ymax", "colour", "fill", "group"), colnames(d))
  d <- d[, keepc, drop = FALSE]
  d$geom <- geom
  curves[[i]] <- d
}
allc <- unique(unlist(lapply(curves, colnames)))
curves <- do.call(rbind, lapply(curves, function(d) {
  for (k in setdiff(allc, colnames(d))) d[[k]] <- NA
  d[, allc]
}))
curves <- thin(curves, paste(curves$geom, curves$group), 8)
curves$x_lo <- b$layout$panel_params[[1]]$x.range[1]
curves$x_hi <- b$layout$panel_params[[1]]$x.range[2]
curves$y_lo <- b$layout$panel_params[[1]]$y.range[1]
curves$y_hi <- b$layout$panel_params[[1]]$y.range[2]
num <- vapply(curves, is.numeric, logical(1))
curves[num] <- lapply(curves[num], signif, 10)
wcsv(curves, "separate_curves_Lineage1.csv", row.names = FALSE)

# --- the size engine: heatmap_rendersize / heatmap_fixsize -------------------
# The figure sizes SCP computes, read back by tracing the two internal helpers.
# rendersize's height has a floor of 0.95 npc of the current viewport -- here
# the 7 x 7 in null PDF device every Rscript session opens -- and fixsize's
# default (height = NULL) body is whatever that viewport leaves: neither grows
# with the number of features.
.sizes <- new.env()
trace("heatmap_rendersize", where = asNamespace("SCP"), print = FALSE,
      exit = quote(assign("rs", returnValue(), envir = .GlobalEnv$.sizes)))
trace("heatmap_fixsize", where = asNamespace("SCP"), print = FALSE,
      exit = quote(assign("fs", returnValue(), envir = .GlobalEnv$.sizes)))
sized <- function(call) {
  rm(list = ls(.sizes), envir = .sizes)
  suppressMessages(suppressWarnings(force(call)))
  fs <- if (exists("fs", envir = .sizes)) .sizes$fs else NULL
  list(
    width_sum = .sizes$rs$width_sum, height_sum = .sizes$rs$height_sum,
    fix = !is.null(fs),
    ht_width = if (is.null(fs)) NULL else as.numeric(fs$ht_width),
    ht_height = if (is.null(fs)) NULL else as.numeric(fs$ht_height)
  )
}
cnt <- GetAssayData(srt, assay = "RNA", layer = "counts")
many <- head(rownames(cnt)[order(-Matrix::rowSums(cnt > 0))], 200)
sizes <- list(
  device_in = as.numeric(grDevices::dev.size("in")),
  group_default = sized(GroupHeatmap(srt, features = features[1:12], group.by = "CellType")),
  group_fixed = sized(GroupHeatmap(srt, features = features[1:12], group.by = "CellType",
                                   width = 2, height = 3)),
  group_split30 = sized(GroupHeatmap(srt, features = features, group.by = "CellType",
                                     n_split = 4, split_method = "hclust")),
  group_split200 = sized(GroupHeatmap(srt, features = many, group.by = "CellType",
                                      n_split = 4, split_method = "hclust")),
  feature_split200 = sized(FeatureHeatmap(srt, features = many, group.by = "CellType",
                                          n_split = 4, split_method = "hclust"))
)
untrace("heatmap_rendersize", where = asNamespace("SCP"))
untrace("heatmap_fixsize", where = asNamespace("SCP"))
writeLines(toJSON(sizes, digits = NA, auto_unbox = TRUE, null = "null", pretty = TRUE),
           file.path(out, "heatmap_sizes.json"))

message("done: ", out)
