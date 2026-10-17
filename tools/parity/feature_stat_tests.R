# Golden p-values for the significance layer of feature_stat_plot.
#
# ggpubr's stat_compare_means and ggsignif's geom_signif both end in plain
# calls to base R: wilcox.test / t.test (via pairwise.*.test or directly),
# kruskal.test, and anova(lm()). Their labels come from format.pval(digits = 2),
# symnum() with ggpubr's cut-points, or sprintf("%.2g"). This script pins all
# of those on crafted inputs -- small samples with and without ties, samples at
# and past the exact-test threshold of 50, constant data -- so the Python port
# can be checked without R.
#
#   Rscript tools/parity/feature_stat_tests.R tests/fixtures/parity/feature_stat
#
# Needs only base R and jsonlite.

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args)) args[[1]] else "tests/fixtures/parity/feature_stat"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(20261017)
safe_p <- function(expr) tryCatch(suppressWarnings(expr$p.value), error = function(e) NA_real_)

two_sample <- list(
  exact_small = list(x = rnorm(5), y = rnorm(7, 1)),
  exact_unbalanced = list(x = rnorm(3), y = rnorm(11, 0.5)),
  exact_one_vs_many = list(x = rnorm(1), y = rnorm(4)),
  exact_49_10 = list(x = rnorm(49), y = rnorm(10, 0.4)),
  exact_extreme = list(x = rnorm(8, 3), y = rnorm(9)),
  ties_small = list(x = c(1, 2, 2, 3, 5, 5, 5, 8), y = c(2, 3, 3, 6, 7, 9)),
  ties_zero_inflated = list(x = c(rep(0, 20), round(rexp(15), 1)),
                            y = c(rep(0, 8), round(rexp(25, 0.7), 1))),
  normal_50_55 = list(x = rnorm(50), y = rnorm(55, 0.3)),
  normal_120_80 = list(x = rnorm(120), y = rnorm(80, 0.2)),
  normal_ties_large = list(x = round(rnorm(200), 1), y = round(rnorm(150, 0.15), 1)),
  constant = list(x = rep(1, 6), y = rep(1, 5)),
  wide_spread = list(x = rnorm(30, 0, 0.01), y = rnorm(40, 0, 10))
)

wilcox <- lapply(two_sample, function(d) list(
  x = d$x, y = d$y,
  p = safe_p(wilcox.test(d$x, d$y)),
  p_exact_false = safe_p(wilcox.test(d$x, d$y, exact = FALSE)),
  p_pairwise = tryCatch(suppressWarnings(pairwise.wilcox.test(
    c(d$x, d$y), rep(c("a", "b"), c(length(d$x), length(d$y))),
    p.adjust.method = "none")$p.value[1, 1]), error = function(e) NA_real_)
))
ttest <- lapply(two_sample, function(d) list(
  x = d$x, y = d$y,
  p = safe_p(t.test(d$x, d$y)),
  p_pairwise = tryCatch(suppressWarnings(pairwise.t.test(
    c(d$x, d$y), rep(c("a", "b"), c(length(d$x), length(d$y))),
    pool.sd = FALSE, p.adjust.method = "none")$p.value[1, 1]), error = function(e) NA_real_)
))

multi <- list(
  three_continuous = list(rnorm(12), rnorm(15, 0.5), rnorm(9, 1)),
  four_ties = list(round(rnorm(20), 0), round(rnorm(25, 0.3), 0),
                   round(rnorm(18, 0.6), 0), round(rnorm(30), 0)),
  two_groups = list(rnorm(10), rnorm(13, 1)),
  zero_inflated = list(c(rep(0, 30), rexp(10)), c(rep(0, 12), rexp(30)),
                       c(rep(0, 5), rexp(40, 0.5)))
)
multigroup <- lapply(multi, function(gs) {
  x <- unlist(gs)
  g <- factor(rep(seq_along(gs), lengths(gs)))
  list(groups = gs,
       kruskal = safe_p(kruskal.test(x, g)),
       anova = anova(lm(x ~ g))$`Pr(>F)`[1])
})

pvec <- list(
  c(0.5), c(1), c(0.05), c(0.0499), c(0.012), c(0.00012), c(1e-5), c(3e-20), c(0),
  c(0.5, 0.012), c(0.00012, 0.5), c(1, 0.05), c(0.9, 0.0009), c(2.1e-7, 3.4e-12),
  c(1e-30, 0.2), c(0.123456, 0.0123456, 0.00123456), c(0.99999), c(0.0001),
  c(0.001), c(0.01), c(5e-4, 4e-4), c(1, 1, 0.3), c(0.995), c(0.0995), c(1e-300, 1e-17)
)
symnum_args <- list(cutpoints = c(0, 1e-04, 0.001, 0.01, 0.05, 1),
                    symbols = c("****", "***", "**", "*", "ns"))
formatting <- lapply(pvec, function(p) list(
  p = p,
  format_pval = format.pval(p, digits = 2),
  symnum = as.character(do.call(stats::symnum, c(list(x = p), symnum_args))),
  sprintf_2g = sprintf("%.2g", p)
))

res <- list(
  r_version = R.version.string,
  wilcox = wilcox, ttest = ttest, multigroup = multigroup, formatting = formatting
)
jsonlite::write_json(res, file.path(out_dir, "stat_tests.json"),
                     digits = NA, auto_unbox = TRUE, pretty = FALSE, na = "null")
cat("wrote", file.path(out_dir, "stat_tests.json"), "\n")
