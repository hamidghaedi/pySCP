#!/usr/bin/env Rscript
# Parity fixtures for the enrichment family: network, enrichmap, wordcloud and
# GSEA comparison/network/enrichmap/wordcloud.
#
#   Rscript tools/parity/enrich_parity.R tests/fixtures/parity/enrich
#
# SCP's RunEnrichment / RunGSEA need a gene annotation database that PrepareDB
# downloads, which is not possible offline. Both accept a custom TERM2GENE, so
# the database is built here instead: real GO biological-process IDs and names
# from GO.db, each given a gene set drawn (seeded) from pancreas_sub's marker
# genes plus background genes. That gives SCP's own pipeline -- RunEnrichment
# (clusterProfiler::enricher) and RunGSEA (fgsea) -- realistic inputs with
# overlapping gene sets, so the term-term graph has structure. The custom
# database comes back labelled "custom"; it is relabelled "GO_BP" so
# EnrichmentPlot takes its GO code paths.
#
# What is dumped is what SCP's plot functions actually computed: the layer data
# of the ggplot objects EnrichmentPlot / GSEAPlot return (combine = FALSE).
# The wordcloud branch cannot be run here because ggwordcloud is not installed
# (check_R would try to install it), so its data step is transcribed from
# SCP-plot.R:13308-13392 verbatim below and run on the same input.
#
# Every table is written gzip-compressed; the suite reads them without R.
# Needs: SCP 0.5.6, GO.db, simplifyEnrichment, slam, tm. About three minutes.

suppressPackageStartupMessages({
  library(SCP); library(GO.db); library(BiocParallel); library(dplyr); library(tidyr)
  library(slam); library(simplifyEnrichment)
})

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args)) args[1] else "tests/fixtures/parity/enrich"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
wcsv <- function(x, f) {
  con <- gzfile(file.path(out, paste0(f, ".gz")), "w")
  write.csv(x, con, row.names = FALSE)
  close(con)
}

set.seed(11)
data(pancreas_sub, package = "SCP")
# SCP::RunDEtest calls GetAssayData(slot = ...), which SeuratObject 5 made
# defunct, so the DE step uses Seurat directly (Wilcoxon, both directions, no
# fold-change floor -- what RunDEtest(only.pos = FALSE, fc.threshold = 1)
# asks for) and RunEnrichment / RunGSEA are fed its genes through their
# geneID / geneScore / geneID_groups interface, which skips the Seurat tools.
Seurat::Idents(pancreas_sub) <- "CellType"
groups <- levels(factor(pancreas_sub$CellType))
de <- Seurat::FindAllMarkers(pancreas_sub, features = Seurat::VariableFeatures(pancreas_sub),
                             logfc.threshold = 0, min.pct = 0.1, only.pos = FALSE,
                             verbose = FALSE)
de$group1 <- factor(as.character(de$cluster), levels = groups)
markers <- lapply(groups, function(g) {
  d <- de[de$group1 == g & de$avg_log2FC > 0 & de$p_val_adj < 0.05, ]
  head(d[order(d$avg_log2FC, decreasing = TRUE), "gene"], 60)
})
names(markers) <- groups
universe <- rownames(pancreas_sub)

# ---- the custom "GO_BP" database ------------------------------------------
ont <- AnnotationDbi::Ontology(GOTERM)
bp <- sort(names(ont)[ont == "BP"])
ids <- sort(sample(bp, 160))
t2n <- data.frame(Term = ids, Name = unname(AnnotationDbi::Term(GOTERM)[ids]))
t2g <- do.call(rbind, lapply(seq_along(ids), function(i) {
  g <- groups[(i %% length(groups)) + 1]
  n <- sample(15:50, 1)
  k <- round(n * runif(1, 0.3, 0.8))
  genes <- unique(c(sample(markers[[g]], min(k, length(markers[[g]]))),
                    sample(universe, n - k)))
  data.frame(Term = ids[i], symbol = genes)
}))

up <- de[de$avg_log2FC > 0 & de$p_val_adj < 0.05, ]   # RunEnrichment's DE_threshold
enr_res <- RunEnrichment(geneID = up$gene, geneID_groups = up$group1,
                         TERM2GENE = t2g, TERM2NAME = t2n, BPPARAM = SerialParam())
enrichment <- enr_res$enrichment
enrichment$Database <- "GO_BP"
rownames(enrichment) <- NULL
keep <- c("ID", "Description", "GeneRatio", "BgRatio", "pvalue", "p.adjust", "qvalue",
          "geneID", "Count", "Groups", "Database")
wcsv(enrichment[, keep], "enrichment.csv")

sig_de <- de[de$p_val_adj < 0.05, ]                    # RunGSEA's DE_threshold
gs_res <- RunGSEA(geneID = sig_de$gene, geneScore = sig_de$avg_log2FC,
                  geneID_groups = sig_de$group1, TERM2GENE = t2g, TERM2NAME = t2n,
                  BPPARAM = SerialParam())
results <- gs_res$results
names(results) <- sub("-custom$", "-GO_BP", names(results))
for (nm in names(results)) {
  results[[nm]]@result$Database <- "GO_BP"
}
gs_enrichment <- gs_res$enrichment
gs_enrichment$Database <- "GO_BP"
rownames(gs_enrichment) <- NULL
gkeep <- c("ID", "Description", "setSize", "enrichmentScore", "NES", "pvalue", "p.adjust",
           "core_enrichment", "Groups", "Database")
wcsv(gs_enrichment[, gkeep], "gsea.csv")
# the per-key result tables GSEAPlot iterates, in the order it iterates them
wcsv(data.frame(key = names(results)), "gsea_keys.csv")

# ---- EnrichmentPlot(network) ----------------------------------------------
res <- list(enrichment = enrichment)
net <- EnrichmentPlot(res = res, db = "GO_BP", plot_type = "network", combine = FALSE)
dump_network <- function(plist, prefix) {
  nodes <- list(); edges <- list()
  for (nm in names(plist)) {
    p <- plist[[nm]]
    e <- p$layers[[1]]$data
    nd <- rbind(p$layers[[4]]$data, p$layers[[2]]$data)
    edges[[nm]] <- data.frame(panel = nm, from = e$from, to = e$to, weight = e$weight,
                              color = e$color)
    nodes[[nm]] <- data.frame(panel = nm, name = nd$name, class = nd$class, metric = nd$metric,
                              color = nd$color, label_color = nd$label_color, label = nd$label)
  }
  wcsv(do.call(rbind, nodes), paste0(prefix, "_nodes.csv"))
  wcsv(do.call(rbind, edges), paste0(prefix, "_edges.csv"))
}
dump_network(net, "network")

# ---- EnrichmentPlot(enrichmap) --------------------------------------------
dump_enrichmap <- function(plist, prefix) {
  nodes <- list(); edges <- list(); legend <- list()
  for (nm in names(plist)) {
    p <- plist[[nm]]
    nd <- p$layers[[1]]$data
    e <- p$layers[[2]]$data
    nodes[[nm]] <- data.frame(panel = nm, ID = nd$name, Description = nd$Description,
                              Count = nd$Count, metric = nd$metric, clusters = as.character(nd$clusters),
                              keyword1 = nd$keyword1, keyword2 = nd$keyword2)
    edges[[nm]] <- data.frame(panel = nm, from = e$from, to = e$to, weight = e$weight)
    sc <- p$scales$get_scales("fill")
    legend[[nm]] <- data.frame(panel = nm, name = sc$name, label = unname(sc$labels))
  }
  wcsv(do.call(rbind, nodes), paste0(prefix, "_nodes.csv"))
  wcsv(do.call(rbind, edges), paste0(prefix, "_edges.csv"))
  wcsv(do.call(rbind, legend), paste0(prefix, "_legend.csv"))
}
emap <- EnrichmentPlot(res = res, db = "GO_BP", plot_type = "enrichmap", combine = FALSE)
dump_enrichmap(emap, "enrichmap")
emap_kw <- EnrichmentPlot(res = res, db = "GO_BP", plot_type = "enrichmap", combine = FALSE,
                          enrichmap_show_keyword = TRUE, enrichmap_label = "feature",
                          enrichmap_cluster = "walktrap")
dump_enrichmap(emap_kw, "enrichmap_kw")

# ---- EnrichmentPlot(comparison) -------------------------------------------
for (only_sig in c(FALSE, TRUE)) {
  cmp <- EnrichmentPlot(res = res, db = "GO_BP", plot_type = "comparison", topTerm = 3,
                        compare_only_sig = only_sig)
  d <- cmp$data
  wcsv(data.frame(Groups = as.character(d$Groups), ID = d$ID,
                  Description = as.character(d$Description), GeneRatio = d$GeneRatio,
                  p.adjust = d$p.adjust, level = as.integer(d$Description)),
       paste0("comparison", if (only_sig) "_sig" else "", ".csv"))
}

# ---- GSEAPlot ---------------------------------------------------------------
gres <- list(enrichment = gs_enrichment, results = results)
gnet <- GSEAPlot(res = gres, db = "GO_BP", plot_type = "network", direction = "both", combine = FALSE)
dump_network(gnet, "gsea_network")
gemap <- GSEAPlot(res = gres, db = "GO_BP", plot_type = "enrichmap", direction = "both", combine = FALSE)
dump_enrichmap(gemap, "gsea_enrichmap")
for (dirn in c("pos", "both")) {
  gcmp <- GSEAPlot(res = gres, db = "GO_BP", plot_type = "comparison", direction = dirn, topTerm = 3)
  d <- gcmp$data
  wcsv(data.frame(Groups = as.character(d$Groups), ID = d$ID, Description = as.character(d$Description),
                  NES = d$NES, setSize = d$setSize, Significant = as.character(d$Significant),
                  level = as.integer(d$Description)),
       paste0("gsea_comparison_", dirn, ".csv"))
}

# ---- wordcloud data step (SCP-plot.R:13308-13392, transcribed) -------------
words_excluded <- SCP::words_excluded
set.seed(11)
metric <- "p.adjust"
sig <- enrichment[enrichment[[metric]] < 0.05, ]
sig <- sig[order(sig[[metric]]), ]
freq_path <- function(df_sub) {
  df_sub %>%
    mutate(keyword = strsplit(tolower(as.character(.data[["Description"]])), " ")) %>%
    unnest(cols = "keyword") %>%
    group_by(.data[["keyword"]], Database, Groups) %>%
    reframe(
      keyword = .data[["keyword"]],
      score = sum(-(log10(.data[[metric]]))),
      count = n(),
      Database = .data[["Database"]],
      Groups = .data[["Groups"]],
      .groups = "keep"
    ) %>%
    filter(!grepl(pattern = "\\[.*\\]", x = .data[["keyword"]])) %>%
    filter(nchar(.data[["keyword"]]) >= 1) %>%
    filter(!tolower(.data[["keyword"]]) %in% tolower(words_excluded)) %>%
    distinct() %>%
    as.data.frame()
}
feature_path <- function(df) {
  df %>%
    mutate(keyword = strsplit(as.character(.data[["geneID"]]), "/")) %>%
    unnest(cols = "keyword") %>%
    group_by(.data[["keyword"]], Database, Groups) %>%
    reframe(
      keyword = .data[["keyword"]],
      score = sum(-(log10(.data[[metric]]))),
      count = n(),
      Database = .data[["Database"]],
      Groups = .data[["Groups"]],
      .groups = "keep"
    ) %>%
    distinct() %>%
    as.data.frame()
}
wf <- list(); wg <- list(); wk <- list(); ws <- list()
# keyword_enrichment is checked on a matrix both sides can build: the names of
# every term in the input table, tokenised as scp.pl._enrich_graph does it
bg <- unique(enrichment[, c("ID", "Description")])
tok <- function(x) {
  x <- tolower(x)
  x <- gsub("[[:punct:]]", "", x)
  x <- gsub("[[:digit:]]", "", x)
  w <- strsplit(x, "[[:space:]]+")
  lapply(w, function(v) v[nchar(v) >= 2 & !v %in% gsub("'", "", tm::stopwords("en"))])
}
toks <- tok(bg$Description)
vocab <- sort(unique(unlist(toks)), method = "radix")
ii <- unlist(lapply(toks, function(v) match(v, vocab)))
jj <- rep(seq_along(toks), lengths(toks))
# term frequencies: a word repeated inside one name counts twice, as in tm
trip <- aggregate(v ~ i + j, data = data.frame(i = ii, j = jj, v = 1), FUN = sum)
tdm <- simple_triplet_matrix(i = trip$i, j = trip$j, v = trip$v,
                             nrow = length(vocab), ncol = nrow(bg),
                             dimnames = list(vocab, bg$ID))
for (g in unique(as.character(sig$Groups))) {
  df_sub <- sig[sig$Groups == g, ]
  a <- freq_path(df_sub)
  wf[[g]] <- a[head(order(a$score, decreasing = TRUE), 100), c("keyword", "score", "count", "Groups")]
  b <- feature_path(df_sub)
  wg[[g]] <- b[head(order(b$score, decreasing = TRUE), 100), c("keyword", "score", "count", "Groups")]
  k <- simplifyEnrichment::keyword_enrichment_from_GO(df_sub$ID)
  if (nrow(k) > 0) wk[[g]] <- data.frame(Groups = g, k)
  s <- simplifyEnrichment:::keyword_enrichment(df_sub$ID, tdm)
  if (nrow(s) > 0) ws[[g]] <- data.frame(Groups = g, s)
}
wcsv(do.call(rbind, wf), "wordcloud_term.csv")
wcsv(do.call(rbind, wg), "wordcloud_feature.csv")
wcsv(do.call(rbind, wk), "wordcloud_go_keyword_R.csv")
wcsv(do.call(rbind, ws), "wordcloud_keyword_substitute.csv")

# GSEAPlot's wordcloud runs the same steps on every significant set of the
# direction (SCP-plot.R:14418-14520), with core_enrichment as the genes
gw <- list(); gf <- list()
for (nm in names(results)) {
  r <- results[[nm]]@result
  r <- r[r[[metric]] < 0.05, ]
  r <- r[order(r[[metric]]), ]
  use <- unique(c(r[r$NES > 0, "ID"], r[r$NES < 0, "ID"]))
  if (!length(use)) next
  df <- results[[nm]][use, , drop = FALSE]
  df$geneID <- df$core_enrichment
  df$Database <- "custom"   # the frequency path, which SCP takes for non-GO data
  a <- freq_path(df)
  gw[[nm]] <- data.frame(key = nm, a[head(order(a$score, decreasing = TRUE), 100),
                                     c("keyword", "score", "count")])
  b <- feature_path(df)
  gf[[nm]] <- data.frame(key = nm, b[head(order(b$score, decreasing = TRUE), 100),
                                     c("keyword", "score", "count")])
}
wcsv(do.call(rbind, gw), "gsea_wordcloud_term.csv")
wcsv(do.call(rbind, gf), "gsea_wordcloud_feature.csv")

writeLines(c(
  paste("SCP", as.character(packageVersion("SCP"))),
  paste("igraph", as.character(packageVersion("igraph"))),
  paste("clusterProfiler", as.character(packageVersion("clusterProfiler"))),
  paste("simplifyEnrichment", as.character(packageVersion("simplifyEnrichment"))),
  R.version.string
), file.path(out, "VERSIONS.txt"))
message("done: ", out)
