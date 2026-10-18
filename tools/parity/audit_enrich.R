#!/usr/bin/env Rscript
# Parity fixtures for the enrichment family: EnrichmentPlot, GSEAPlot and
# gseaScores from SCP (Hao Zhang, NJMU, GPL-3; https://github.com/zhanghao-njmu/SCP).
#
# Regenerate from the repository root with
#
#     Rscript tools/parity/audit_enrich.R [out_dir]
#
# (default out_dir: tests/fixtures/parity/audit/enrich). The output is read by
# tests/test_parity_enrich.py, which needs neither R nor the .h5ad.
#
# The species databases (org.Mm.eg.db, msigdbr) are not assumed to be
# installed, so the gene sets are built from the data itself and passed to
# RunEnrichment / RunGSEA through their TERM2GENE argument. The plot layer is
# what is under audit, not the statistics: whatever enrichment table R ends up
# with is written out verbatim as the *input* for the Python side, and the
# plotted frames R derives from it are written out as the *expected* output.
#
# Everything is deterministic: set.seed() for the random gene sets and
# BiocParallel::SerialParam() so fgsea's permutations do not depend on workers.

suppressPackageStartupMessages({
  library(SCP); library(Seurat); library(ggplot2); library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) > 0) args[[1]] else "tests/fixtures/parity/audit/enrich"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
wcsv <- function(df, name) write.csv(df, file.path(out, name), row.names = FALSE)
BP <- BiocParallel::SerialParam()

data("pancreas_sub", package = "SCP")
srt <- NormalizeData(pancreas_sub, verbose = FALSE)
# SCP 0.5.6's RunDEtest calls GetAssayData(slot=), which SeuratObject 5 made
# defunct, so the marker table is produced with Seurat directly and stored where
# RunEnrichment / RunGSEA look for it (srt@tools$DEtest_<group_by>$AllMarkers_<test>),
# with RunDEtest's column names (gene, group1).
# FindAllMarkers takes a few minutes; set AUDIT_CACHE to a scratch .rds to reuse it.
Idents(srt) <- "CellType"
cache <- Sys.getenv("AUDIT_CACHE", unset = "")
if (nzchar(cache) && file.exists(cache)) {
  de <- readRDS(cache)
} else {
  de <- FindAllMarkers(srt, only.pos = FALSE, logfc.threshold = 0.25, min.pct = 0.1,
                       verbose = FALSE)
  if (nzchar(cache)) saveRDS(de, cache)
}
de$group1 <- factor(as.character(de$cluster), levels = levels(srt$CellType))
de$group2 <- "others"
rownames(de) <- NULL
srt@tools$DEtest_CellType <- list(AllMarkers_wilcox = de)

# ---- gene sets -------------------------------------------------------------
# Per CellType: two marker-rank bands and one diluted set, plus random sets.
# Descriptions are lower-case (exercises capitalize()) and of varied length
# (exercises str_wrap at character_width = 50).
set.seed(11)
universe <- unique(de$gene)
t2g <- list(); t2n <- list()
groups <- levels(srt$CellType)
for (g in groups) {
  up <- de[de$group1 == g & de$avg_log2FC > 0 & de$p_val_adj < 0.05, ]
  up <- up[order(-up$avg_log2FC), "gene"]
  tag <- gsub("[^A-Za-z0-9]", "", g)
  sets <- list(
    list(id = paste0("SET:", tag, "_top"), genes = head(up, 25),
         desc = paste0("genes most strongly up in ", tolower(g), " cells")),
    list(id = paste0("SET:", tag, "_band"), genes = up[26:min(70, length(up))],
         desc = paste0("second band of markers distinguishing the ", tolower(g),
                       " compartment from the other pancreatic lineages")),
    list(id = paste0("SET:", tag, "_mix"),
         genes = c(head(up, 8), sample(setdiff(universe, up), 22)),
         desc = paste0("diluted ", tolower(g), " signature"))
  )
  for (s in sets) {
    s$genes <- s$genes[!is.na(s$genes)]
    t2g[[s$id]] <- data.frame(Term = s$id, symbol = s$genes)
    t2n[[s$id]] <- data.frame(Term = s$id, Name = s$desc)
  }
}
for (k in 1:12) {
  id <- sprintf("SET:random%02d", k)
  t2g[[id]] <- data.frame(Term = id, symbol = sample(universe, 30))
  t2n[[id]] <- data.frame(Term = id, Name = sprintf("random gene set number %d drawn from the expressed universe", k))
}
TERM2GENE <- do.call(rbind, t2g); TERM2NAME <- do.call(rbind, t2n)
rownames(TERM2GENE) <- NULL; rownames(TERM2NAME) <- NULL

# ---- enrichment ------------------------------------------------------------
srt <- RunEnrichment(srt, group_by = "CellType", db = "custom", TERM2GENE = TERM2GENE,
                     TERM2NAME = TERM2NAME, minGSSize = 5, maxGSSize = 500, BPPARAM = BP)
enr <- srt@tools$Enrichment_CellType_wilcox$enrichment
wcsv(enr, "enrichment_input.csv")

dump_panels <- function(plist, tag) {
  rows <- list()
  for (nm in names(plist)) {
    p <- plist[[nm]]
    d <- p$data
    b <- ggplot_build(p)
    # the layer that carries the term marks: bar -> 1, dot -> 1, lollipop -> 4
    li <- switch(tag, bar = 1, dot = 1, lollipop = 4, comparison = 1, 1)
    bd <- b$data[[li]]
    lev <- levels(d$Description)
    # ggplot_build x is the factor position (1 = bottom after coord_flip)
    pos <- if (tag == "comparison") bd$y else bd$x
    o <- data.frame(
      panel = nm,
      Description = lev[pos],
      axis_pos = pos,
      value = if (tag == "comparison") NA else bd$y,
      fill = bd$fill,
      size = if ("size" %in% names(bd)) bd$size else NA,
      italic = grepl("\n", lev[pos])
    )
    m <- d[match(o$Description, as.character(d$Description)), ]
    o$ID <- m$ID
    o$Groups <- as.character(m$Groups)
    o$Count <- m$Count
    if ("metric" %in% names(m)) o$metric <- m$metric
    if (is.numeric(m$GeneRatio)) o$GeneRatio_num <- m$GeneRatio
    if ("FoldEnrichment" %in% names(m)) o$FoldEnrichment <- m$FoldEnrichment
    o$ylim_hi <- b$layout$panel_params[[1]]$x.range[2]
    rows[[nm]] <- o
  }
  do.call(rbind, rows)
}

for (pt in c("bar", "dot", "lollipop")) {
  pl <- EnrichmentPlot(srt, db = "custom", group_by = "CellType", plot_type = pt, combine = FALSE)
  wcsv(dump_panels(pl, pt), paste0("enrichment_", pt, ".csv"))
}
# pvalue-only path: metric switches to pvalue when padjustCutoff is NULL
pl <- EnrichmentPlot(srt, db = "custom", group_by = "CellType", plot_type = "bar",
                     padjustCutoff = NULL, pvalueCutoff = 0.01, topTerm = 4, combine = FALSE)
wcsv(dump_panels(pl, "bar"), "enrichment_bar_pvalue.csv")

pc <- EnrichmentPlot(srt, db = "custom", group_by = "CellType", plot_type = "comparison",
                     combine = FALSE)
cmp <- pc[[1]]
bc <- ggplot_build(cmp)$data[[1]]
dc <- cmp$data
wcsv(data.frame(
  Groups = levels(dc$Groups)[bc$x], Description = levels(dc$Description)[bc$y],
  x = bc$x, y = bc$y, fill = bc$fill, size = bc$size,
  ID = dc$ID[match(paste(levels(dc$Groups)[bc$x], levels(dc$Description)[bc$y]),
                   paste(dc$Groups, dc$Description))],
  p.adjust = dc[["p.adjust"]][match(paste(levels(dc$Groups)[bc$x], levels(dc$Description)[bc$y]),
                                    paste(dc$Groups, dc$Description))],
  GeneRatio_num = dc$GeneRatio[match(paste(levels(dc$Groups)[bc$x], levels(dc$Description)[bc$y]),
                                     paste(dc$Groups, dc$Description))]
), "enrichment_comparison.csv")
write_json(list(group_levels = levels(dc$Groups),
                term_levels = levels(dc$Description)),
           file.path(out, "enrichment_comparison_levels.json"), auto_unbox = FALSE)

# ---- edge cases on a hand-built table ----------------------------------------
# Ties in p.adjust (R's order() is stable), a term at exactly the cutoff (R
# filters with a strict <), topTerm applied per group, and groups whose
# first-appearance order (G2 before G1) is not alphabetical.
edge <- data.frame(
  ID = paste0("T", 1:8),
  Description = c("alpha tie first", "beta tie second", "gamma at the cutoff",
                  "delta g1", "epsilon g1", "zeta g1 best", "eta not significant",
                  "theta g2 third"),
  GeneRatio = c("5/50", "9/50", "4/50", "6/40", "3/40", "12/40", "2/40", "7/50"),
  BgRatio = c("50/1000", "60/1000", "40/1000", "30/1000", "90/1000", "100/1000",
              "20/1000", "35/1000"),
  pvalue = c(1e-4, 1e-4, 0.004, 0.002, 0.003, 1e-6, 0.03, 0.0009),
  p.adjust = c(0.001, 0.001, 0.05, 0.02, 0.04, 0.0001, 0.2, 0.01),
  geneID = "a/b/c", Count = c(5L, 9L, 4L, 6L, 3L, 12L, 2L, 7L),
  Groups = c("G2", "G2", "G2", "G1", "G1", "G1", "G1", "G2"),
  Database = "DB"
)
wcsv(edge, "edge_input.csv")
edge_rows <- list()
for (pt in c("bar", "dot", "lollipop")) {
  for (cut in c("padj", "pval")) {
    pl <- if (cut == "padj") {
      EnrichmentPlot(res = list(enrichment = edge), db = "DB", plot_type = pt,
                     topTerm = 2, combine = FALSE)
    } else {
      EnrichmentPlot(res = list(enrichment = edge), db = "DB", plot_type = pt,
                     padjustCutoff = NULL, pvalueCutoff = 0.003, topTerm = 2, combine = FALSE)
    }
    o <- dump_panels(pl, pt)
    o$plot_type <- pt; o$cutoff <- cut; o$panel_index <- match(o$panel, names(pl))
    edge_rows[[paste(pt, cut)]] <- o[, c("plot_type", "cutoff", "panel", "panel_index", "ID",
                                         "Description", "axis_pos", "value", "fill", "size")]
  }
}
wcsv(do.call(rbind, edge_rows), "edge_expected.csv")

# ---- GSEA ------------------------------------------------------------------
srt <- RunGSEA(srt, group_by = "CellType", db = "custom", TERM2GENE = TERM2GENE,
               TERM2NAME = TERM2NAME, minGSSize = 5, maxGSSize = 500, BPPARAM = BP)
gs <- srt@tools$GSEA_CellType_wilcox
gtab <- gs$enrichment
wcsv(gtab, "gsea_input.csv")

# ranked lists and gene sets, so Python can recompute the running scores
gl_rows <- list()
for (nm in names(gs$results)) {
  r <- gs$results[[nm]]
  gl_rows[[nm]] <- data.frame(key = nm, gene = names(r@geneList), metric = unname(r@geneList),
                              exponent = r@params[["exponent"]])
}
wcsv(do.call(rbind, gl_rows), "gsea_genelists.csv")
wcsv(TERM2GENE, "gsea_term2gene.csv")

# term selection for line / bar, verbatim from GSEAPlot (SCP-plot.R:13762-13784);
# the line figure is assembled as grobs, so its data cannot be read back from
# the plot object and the selection is replayed here instead.
select_ids <- function(res_enrich, direction, topTerm = 6, metric = "p.adjust", metric_value = 0.05) {
  f <- res_enrich@result[res_enrich@result[[metric]] < metric_value, , drop = FALSE]
  f <- f[order(f[[metric]]), , drop = FALSE]
  up <- f[f[["NES"]] > 0, , drop = FALSE]; up <- up[head(order(up[[metric]]), topTerm), "ID"]
  dn <- f[f[["NES"]] < 0, , drop = FALSE]; dn <- dn[head(order(dn[[metric]]), topTerm), "ID"]
  switch(direction,
    pos = unique(head(up, topTerm)), neg = unique(head(dn, topTerm)),
    both = unique(head(c(head(up, ceiling(topTerm / 2)), head(dn, ceiling(topTerm / 2))), topTerm)))
}
sel <- list(); curves <- list(); crosses <- list()
for (nm in names(gs$results)) {
  r <- gs$results[[nm]]
  for (dirn in c("pos", "neg", "both")) {
    ids <- select_ids(r, dirn)
    if (length(ids)) sel[[paste(nm, dirn)]] <- data.frame(key = nm, direction = dirn,
                                                          rank = seq_along(ids), ID = ids)
  }
  ids <- select_ids(r, "both")
  for (id in ids) {
    gi <- SCP:::gsInfo(r, id)
    pk <- which.max(abs(gi$runningScore))
    curves[[paste(nm, id)]] <- data.frame(key = nm, ID = id, x = gi$x,
                                          runningScore = gi$runningScore,
                                          position = gi$position, ymin = gi$ymin,
                                          ymax = gi$ymax, peak = seq_along(gi$x) == pk)
  }
  y <- r@geneList
  min_y <- y[which.min(abs(y))]
  crosses[[nm]] <- data.frame(key = nm, zero_cross = median(seq_along(y)[y == min_y]))
}
wcsv(do.call(rbind, sel), "gsea_selection.csv")
cv <- do.call(rbind, curves)
# keep the fixture small: hits, the peak, and every 50th position
cv <- cv[cv$position == 1 | cv$peak | (cv$x %% 50 == 1), ]
wcsv(cv, "gsea_curves.csv")
wcsv(do.call(rbind, crosses), "gsea_zero_cross.csv")

# the ranking-metric heat strip of a single-set line plot, replayed verbatim
# from GSEAPlot (SCP-plot.R:13895-13923) for the same reason as the selection
strips <- list()
for (nm in names(gs$results)) {
  y <- y_raw <- unname(gs$results[[nm]]@geneList)
  y[y > quantile(y_raw, 0.98)] <- quantile(y_raw, 0.98)
  y[y < quantile(y_raw, 0.02)] <- quantile(y_raw, 0.02)
  col <- rep("white", length(y))
  y_pos <- which(y > 0)
  if (length(y_pos) > 0) {
    y_pos_i <- cut(y[y_pos], breaks = seq(min(y[y_pos], na.rm = TRUE), max(y[y_pos], na.rm = TRUE),
                                          len = 100), include.lowest = TRUE)
    col[y_pos] <- colorRampPalette(c("#F5DCDC", "#C40003"))(100)[y_pos_i]
  }
  y_neg <- which(y < 0)
  if (length(y_neg) > 0) {
    y_neg_i <- cut(y[y_neg], breaks = seq(min(y[y_neg], na.rm = TRUE), max(y[y_neg], na.rm = TRUE),
                                          len = 100), include.lowest = TRUE)
    col[y_neg] <- colorRampPalette(c("#1D008F", "#DDDCF5"))(100)[y_neg_i]
  }
  xmin <- which(!duplicated(col))
  xmax <- xmin + as.numeric(table(col)[as.character(unique(col))])
  strips[[nm]] <- data.frame(key = nm, xmin = xmin, xmax = xmax, col = unique(col))
}
wcsv(do.call(rbind, strips), "gsea_heat_strip.csv")

for (dirn in c("pos", "both")) {
  pb <- GSEAPlot(srt, db = "custom", group_by = "CellType", plot_type = "bar",
                 direction = dirn, combine = FALSE)
  rows <- list()
  for (nm in names(pb)) {
    p <- pb[[nm]]; d <- p$data; b <- ggplot_build(p)$data[[2]]
    rows[[nm]] <- data.frame(key = nm, ID = d$ID, NES = d$NES, axis_pos = as.integer(d$Description),
                             Description = as.character(d$Description),
                             Direction = as.character(d$Direction),
                             alpha_raw = -log10(d[["p.adjust"]]),
                             fill = b$fill[match(as.integer(d$Description), b$y)],
                             alpha = b$alpha[match(as.integer(d$Description), b$y)])
  }
  wcsv(do.call(rbind, rows), paste0("gsea_bar_", dirn, ".csv"))
}

# A discriminating GSEA case: on the real data |NES| order and p.adjust order
# happen to agree, which would hide a selection rule keyed on the wrong column.
# Re-assign p.adjust so the *weakest* |NES| among the significant positive sets
# is the most significant, and put one set exactly at the cutoff.
r2 <- gs$results[["Ductal-custom"]]
res2 <- r2@result
pos_sig <- which(res2$NES > 0 & res2$p.adjust < 0.05)
o <- pos_sig[order(abs(res2$NES[pos_sig]))]          # weakest |NES| first
res2$p.adjust[o] <- 10^-(length(o):1 + 1)            # ... gets the smallest p.adjust
res2$p.adjust[o[length(o)]] <- 0.05                  # strongest |NES| sits at the cutoff
r2@result <- res2
wcsv(res2, "gsea_edge_input.csv")
edge_sel <- list()
for (dirn in c("pos", "both")) {
  ids <- select_ids(r2, dirn, topTerm = 3)
  pb <- GSEAPlot(res = list(enrichment = res2, results = list("Ductal-custom" = r2)),
                 db = "custom", plot_type = "bar", direction = dirn, topTerm = 3, combine = FALSE)
  d <- pb[[1]]$data
  edge_sel[[dirn]] <- data.frame(direction = dirn, selection_rank = match(d$ID, ids),
                                 ID = d$ID, NES = d$NES, axis_pos = as.integer(d$Description))
}
wcsv(do.call(rbind, edge_sel), "gsea_edge_expected.csv")

# ggplot2's default continuous breaks, scales::extended_breaks() ->
# labeling::extended(): the rank axis of GSEAPlot's line plot uses them.
set.seed(1)
ext <- lapply(1:300, function(i) {
  a <- signif(rnorm(1, 0, 10^sample(-3:5, 1)), 4)
  b <- a + signif(abs(rnorm(1, 0, 10^sample(-3:5, 1))), 4)
  m <- sample(3:8, 1)
  data.frame(dmin = sprintf("%.17g", a), dmax = sprintf("%.17g", b), m = m,
             breaks = paste(sprintf("%.17g", labeling::extended(a, b, m)), collapse = " "))
})
wcsv(do.call(rbind, ext), "extended_breaks.csv")

write_json(list(scp = as.character(packageVersion("SCP")),
                clusterProfiler = as.character(packageVersion("clusterProfiler")),
                stringr_wrap_examples = setNames(
                  as.list(stringr::str_wrap(capitalize(unique(TERM2NAME$Name)), width = 50)),
                  unique(TERM2NAME$Name))),
           file.path(out, "meta.json"), auto_unbox = TRUE, pretty = TRUE)
message("done -> ", normalizePath(out))
