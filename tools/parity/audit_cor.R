#!/usr/bin/env Rscript
# Parity fixtures for the correlation figures: CellCorHeatmap and
# FeatureCorPlot from SCP (Hao Zhang, NJMU, GPL-3;
# https://github.com/zhanghao-njmu/SCP), checked against
# scp.pl.cell_cor_heatmap / scp.pl.feature_cor_plot.
# FeatureCorHeatmap is an empty stub in SCP (SCP-plot.R:9980) and has no
# fixture.
#
# Regenerates tests/fixtures/parity/audit/cor/:
#   query_expr.csv / query_obs.csv   pancreas_sub, log-normalised, restricted to
#                                    the 50 features below (cells x genes)
#   panc8_expr.csv / panc8_obs.csv   panc8_sub with SCP's README gene renaming
#                                    (capitalize, force_tolower), log-normalised,
#                                    same 50 features
#   features.json                    the explicit feature list both sides use,
#                                    and how many features R picks by default
#   cellcor_self_default_features.csv  R's matrix with features = NULL, for
#                                    checking default selection on the full data
#   cellcor_default_centroids.csv    R's collapsed SubCellType centroids for
#   cellcor_default_hvf.json         every stored HVF plus 500 genes that are
#                                    not, and the HVF list: default selection
#                                    checked without the full matrix
#   cellcor_<case>.csv               CellCorHeatmap()$simil_matrix per case
#                                    (rows = query, columns = reference, in R's
#                                    row/column order before clustering)
#   cellcor_meta.json                per case: simil_name, colour limits,
#                                    clustered row/column order, and the label
#                                    texts drawn by layer_fun
#   featcor_input.csv                FeatureCorPlot input: Ins1 / Ins2 / Gcg,
#                                    CellType and Phase per cell
#   featcor.json                     FeatureCorPlot transforms: proxyC
#                                    similarity, lower-panel fill colours, lm
#                                    fit stats, violin outlines, per-split
#                                    similarity for split.by = "Phase"
#
# pancreas_sub's `data` slot holds raw counts, so both objects are
# log-normalised with NormalizeData first, as SCP's own examples do
# (Standard_SCP / check_srtMerge normalise before CellCorHeatmap).
#
# Usage, from the repository root:
#   Rscript tools/parity/audit_cor.R [out_dir]

suppressPackageStartupMessages({
  library(SCP); library(Seurat); library(jsonlite); library(ComplexHeatmap)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) > 0) args[[1]] else "tests/fixtures/parity/audit/cor"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
grDevices::pdf(NULL)
set.seed(11)

# R bug (SCP 0.5.6 on SeuratObject >= 5): RunKNNPredict ends by writing
# `match_best[query_index]` into meta.data. With query_collapsing = TRUE that
# vector is named by *group*, SeuratObject 5 matches names against cell
# barcodes, finds none and stops with "No cell overlap between new meta data
# and Seurat object" -- so every collapsed CellCorHeatmap call fails. The
# similarity matrix is already computed by then; unname() the three trailing
# metadata writes so the call completes. Nothing upstream of them changes.
local({
  f <- SCP:::RunKNNPredict
  src <- deparse(f)
  src <- gsub("match_best[query_index]", "unname(match_best[query_index])", src, fixed = TRUE)
  src <- gsub("apply(match_prob, 1, max)[query_index]", "unname(apply(match_prob, 1, max)[query_index])", src, fixed = TRUE)
  src <- gsub("(1 - distance)[query_index]", "unname((1 - distance)[query_index])", src, fixed = TRUE)
  # Third: the uncollapsed branches call GetAssayData(slot = "data"), which the
  # installed SeuratObject rejects outright; layer = is the same matrix.
  src <- gsub('GetAssayData(srt_query, slot = "data"', 'GetAssayData(srt_query, layer = "data"', src, fixed = TRUE)
  src <- gsub('GetAssayData(srt_ref, slot = "data"', 'GetAssayData(srt_ref, layer = "data"', src, fixed = TRUE)
  g <- eval(parse(text = src))
  environment(g) <- environment(f)
  utils::assignInNamespace("RunKNNPredict", g, ns = "SCP")
})

data("pancreas_sub", package = "SCP")
data("panc8_sub", package = "SCP")
srt <- NormalizeData(pancreas_sub, verbose = FALSE)

genenames <- make.unique(capitalize(rownames(panc8_sub), force_tolower = TRUE))
panc8 <- RenameFeatures(panc8_sub, newnames = genenames)
panc8 <- NormalizeData(panc8, verbose = FALSE)
# Second R incompatibility: Seurat 5's AverageExpression rewrites "_" to "-"
# in group names, so CellCorHeatmap's later simil_matrix[, levels(ref_group)]
# is out of bounds for panc8's "activated_stellate" / "quiescent_stellate".
# Use the dashed spelling on both sides.
panc8$celltype <- gsub("_", "-", panc8$celltype)

# 50 highly variable query genes that the renamed reference also carries
features <- head(intersect(VariableFeatures(srt), rownames(panc8)), 50)

write_expr <- function(obj, path) {
  m <- as.matrix(LayerData(obj, layer = "data")[features, ])
  df <- data.frame(cell = colnames(obj), t(signif(m, 8)), check.names = FALSE)
  write.csv(df, path, row.names = FALSE)
}
write_expr(srt, file.path(out, "query_expr.csv"))
write.csv(data.frame(cell = colnames(srt), CellType = as.character(srt$CellType),
                     SubCellType = as.character(srt$SubCellType),
                     Phase = as.character(srt$Phase)),
          file.path(out, "query_obs.csv"), row.names = FALSE)
write_expr(panc8, file.path(out, "panc8_expr.csv"))
write.csv(data.frame(cell = colnames(panc8), celltype = as.character(panc8$celltype)),
          file.path(out, "panc8_obs.csv"), row.names = FALSE)

# ---------------------------------------------------------------- CellCorHeatmap
# The label texts layer_fun draws. They exist only while ComplexHeatmap draws
# (grid.grabExpr records them as recordedGrobs), so grid.text is traced inside
# SCP's namespace for the duration of each call and the black copies kept --
# the 16 white copies around each one are the halo. A label is keyed by its
# text and position so that a second draw pass does not count it twice.
.cap <- new.env()
.cap$acc <- character()
trace("grid.text", where = asNamespace("SCP"), print = FALSE, tracer = quote({
  if (identical(gp$col, "black")) {
    key <- paste(as.character(label), as.character(x), as.character(y), sep = "|")
    assign("acc", c(get("acc", envir = .cap), key), envir = .cap)
  }
}))
label_texts <- function() {
  keys <- unique(.cap$acc)
  .cap$acc <- character()
  sub("\\|.*$", "", keys)
}

# ComplexHeatmap's own clustering of the returned matrix: what CellCorHeatmap
# draws with cluster_rows / cluster_columns = TRUE (euclidean, complete,
# dendrogram reordered by row means).
cluster_order <- function(m) {
  ht <- draw(Heatmap(m, cluster_rows = TRUE, cluster_columns = TRUE))
  list(rows = rownames(m)[row_order(ht)], cols = colnames(m)[column_order(ht)])
}

meta <- list()
run_case <- function(name, ..., query = srt, ref = NULL) {
  .cap$acc <- character()
  res <- CellCorHeatmap(srt_query = query, srt_ref = ref, features = features, ...)
  m <- res$simil_matrix
  write.csv(data.frame(row = rownames(m), signif(m, 10), check.names = FALSE),
            file.path(out, paste0("cellcor_", name, ".csv")), row.names = FALSE)
  labs <- label_texts()
  co <- if (nrow(m) > 2 && ncol(m) > 2 && nrow(m) <= 50) cluster_order(m) else NULL
  meta[[name]] <<- list(simil_name = res$simil_name, limits = range(m),
                        n_features = length(res$features),
                        cluster_rows = co$rows, cluster_cols = co$cols,
                        labels = sort(labs))
  invisible(res)
}

# srt_ref = NULL: the reference is the query itself, collapsing inherited
run_case("self_cosine", query_group = "SubCellType")
run_case("self_pearson", query_group = "SubCellType", distance_metric = "pearson")
run_case("self_spearman", query_group = "SubCellType", distance_metric = "spearman")
run_case("self_label_row", query_group = "SubCellType", nlabel = 2, label_cutoff = 0.5)
run_case("self_label_col", query_group = "SubCellType", nlabel = 1, label_by = "column")
run_case("self_label_both", query_group = "SubCellType", nlabel = 2, label_by = "both")

# a separate reference: panc8_sub, renamed to mouse symbols
run_case("panc8_cosine", query_group = "SubCellType", ref = panc8, ref_group = "celltype")
run_case("panc8_spearman", query_group = "SubCellType", ref = panc8, ref_group = "celltype",
         distance_metric = "spearman")
# uncollapsed query: one row per cell, grouped by SubCellType level
run_case("panc8_cells", query_group = "SubCellType", query_collapsing = FALSE,
         ref = panc8, ref_group = "celltype")

# what R selects when features = NULL: HVF(query) intersect HVF(ref). The
# selection is not returned by CellCorHeatmap (its `features` element echoes
# the argument), so read it off the RunKNNPredict call it delegates to.
knn <- SCP:::RunKNNPredict(srt_query = srt, srt_ref = srt, query_group = "SubCellType",
                     ref_group = "SubCellType", nn_method = "raw",
                     return_full_distance_matrix = TRUE)
default_features <- knn@tools$KNNPredict_classification$features
write_json(list(features = features, default_n_features = length(default_features),
                default_features_head = head(default_features, 20),
                query_hvf_n = length(VariableFeatures(srt))),
           file.path(out, "features.json"), auto_unbox = TRUE)
res_default <- CellCorHeatmap(srt_query = srt, query_group = "SubCellType")
md <- res_default$simil_matrix
write.csv(data.frame(row = rownames(md), md, check.names = FALSE),
          file.path(out, "cellcor_self_default_features.csv"), row.names = FALSE)
# ... and what it needs to be checked without the full matrix: the collapsed
# centroids R correlates (log1p of AverageExpression on the data layer, one row
# per SubCellType level) for every stored HVF plus 500 genes that are not HVFs,
# in object gene order, with the HVF flag. One pseudo-cell per group collapses
# to itself, so default selection runs on these and must pick the HVFs alone.
hvf <- VariableFeatures(srt)
set.seed(11)
cent_genes <- rownames(srt)[rownames(srt) %in% c(hvf, sample(setdiff(rownames(srt), hvf), 500))]
cent <- log1p(as.matrix(AverageExpression(srt, features = cent_genes, layer = "data",
                                          group.by = "SubCellType", verbose = FALSE)[[1]]))
stopifnot(identical(colnames(cent), levels(srt$SubCellType)))
write.csv(data.frame(row = colnames(cent), t(signif(cent, 10)), check.names = FALSE),
          file.path(out, "cellcor_default_centroids.csv"), row.names = FALSE)
write_json(list(hvf = hvf, genes = cent_genes), file.path(out, "cellcor_default_hvf.json"))
write_json(meta, file.path(out, "cellcor_meta.json"), auto_unbox = TRUE, digits = NA)

# ---------------------------------------------------------------- FeatureCorPlot
fc_feats <- c("Ins1", "Ins2", "Gcg")
p_fc <- FeatureCorPlot(srt, features = fc_feats, group.by = "CellType")  # must build
dat_exp <- t(as.matrix(LayerData(srt, layer = "data")[fc_feats, ]))
write.csv(data.frame(cell = colnames(srt), CellType = as.character(srt$CellType),
                     Phase = as.character(srt$Phase), signif(dat_exp, 10), check.names = FALSE),
          file.path(out, "featcor_input.csv"), row.names = FALSE)

# the transforms FeatureCorPlot applies, as written at SCP-plot.R:5389-5560
sim_of <- function(cells, method) {
  fm <- t(dat_exp[cells, fc_feats])
  if (method == "spearman") fm <- t(apply(fm, 1, rank))
  as.matrix(proxyC::simil(x = as(fm, "CsparseMatrix"), method = "correlation"))
}
cor_colors <- palette_scp(x = seq(-1, 1, length.out = 200), palette = "RdBu")
bound <- do.call(rbind, lapply(strsplit(gsub("\\(|\\)|\\[|\\]", "", names(cor_colors)), ","), as.numeric))
rownames(bound) <- cor_colors
bound[1, 1] <- bound[1, 1] - 0.01
fill_of <- function(r) rownames(bound)[bound[, 1] < r & bound[, 2] >= r]

all_cells <- colnames(srt)
fc <- list(features = fc_feats)
for (method in c("pearson", "spearman")) {
  s <- sim_of(all_cells, method)
  pairs <- list()
  for (i in seq_along(fc_feats)) for (j in seq_along(fc_feats)) if (i > j) {
    f1 <- fc_feats[i]; f2 <- fc_feats[j]
    pairs[[length(pairs) + 1]] <- list(f1 = f1, f2 = f2, cor = s[f1, f2],
                                       label = round(s[f1, f2], 3), fill = fill_of(s[f1, f2]))
  }
  fc[[method]] <- list(simil = s, lower = pairs)
}
lmfits <- list()
for (i in seq_along(fc_feats)) for (j in seq_along(fc_feats)) if (i < j) {
  f1 <- fc_feats[i]; f2 <- fc_feats[j]
  m <- lm(dat_exp[, f2] ~ dat_exp[, f1])
  lmfits[[length(lmfits) + 1]] <- list(
    f1 = f1, f2 = f2, intercept = unname(coef(m)[1]), slope = unname(coef(m)[2]),
    r2 = summary(m)$r.squared, p = summary(m)$coefficients[2, 4],
    r2_text = format(summary(m)$r.squared, digits = 2),
    p_text = format(summary(m)$coefficients[2, 4], digits = 2)
  )
}
fc$lm <- lmfits

# diagonal violins exactly as built (scale = "width", adjust = 1, trim = TRUE)
dat <- cbind(srt@meta.data[, "CellType", drop = FALSE], dat_exp)
viol <- list()
for (f in fc_feats) {
  b <- ggplot_build(ggplot(dat) + geom_violin(aes(x = CellType, y = .data[[f]], fill = CellType),
                                              scale = "width", adjust = 1, trim = TRUE, na.rm = TRUE))$data[[1]]
  b <- b[, c("group", "y", "violinwidth")]
  b$group <- levels(dat$CellType)[b$group]
  # ggplot2 4.x appends repeated y = 0 rows after the 512-point grid; sort and
  # de-duplicate, then keep every 8th point plus each violin's top end
  b <- b[order(b$group, b$y), ]
  b <- b[!duplicated(b[, c("group", "y")]), ]
  k <- ave(seq_len(nrow(b)), b$group, FUN = seq_along)
  last <- ave(seq_len(nrow(b)), b$group, FUN = length)
  b <- b[k %% 8 == 1 | k == last, ]
  viol[[f]] <- b
  viol[[paste0(f, "_bw")]] <- tapply(dat[[f]], dat$CellType, function(v) bw.nrd0(v))
}
fc$violin <- viol

# split.by = "Phase": one SPLOM per split, similarity computed within the split
p_split <- FeatureCorPlot(srt, features = fc_feats, group.by = "CellType", split.by = "Phase")
fc$split <- lapply(setNames(levels(factor(srt$Phase, levels = unique(srt$Phase))),
                            levels(factor(srt$Phase, levels = unique(srt$Phase)))),
                   function(s) sim_of(colnames(srt)[srt$Phase == s], "pearson"))
write_json(fc, file.path(out, "featcor.json"), auto_unbox = TRUE, digits = NA, matrix = "rowmajor")
message("cor fixtures -> ", normalizePath(out))
