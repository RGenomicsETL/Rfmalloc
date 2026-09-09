## statgen_snp_cor(): windowed LD correlation over an fmalloc genotype tensor,
## packed into a banded 'ld' store. Oracle 1: stats::cor() on the decoded matrix,
## restricted to the window (equal within the int8/int16 quantization tolerance).
## Oracle 2 (guarded): bigsnpr::snp_cor() on the same complete-data matrix.
##
## Wrapped in one function so a single on.exit() cleans up the fmalloc runtime.

library(Rfmalloc)
library(RfmallocStatgen)

(function() {
    set.seed(20260710)

    n <- 200L
    m <- 40L
    ## complete genotypes (no missing) so mean-imputation is a no-op and the
    ## estimator equals a plain Pearson correlation, matching both oracles.
    g <- matrix(sample(0:2, n * m, replace = TRUE, prob = c(0.25, 0.5, 0.25)),
                nrow = n, ncol = m)
    ## guarantee no monomorphic variant (cor() would be NaN there)
    for (j in seq_len(m)) if (length(unique(g[, j])) == 1L) g[1L, j] <- (g[1L, j] + 1L) %% 3L
    storage.mode(g) <- "integer"

    tmp <- tempfile(fileext = ".bin")
    rt <- open_fmalloc(tmp, mode = "scratch", size_gb = 0.2)
    on.exit({ cleanup_fmalloc(rt); unlink(tmp) }, add = TRUE)

    tn <- fmalloc_bed(g, runtime = rt)

    size <- 5L
    corr <- statgen_snp_cor(tn, size = size)

    ## -- shape / type contract -------------------------------------------
    expect_true(inherits(corr, "fmalloc_ld"))
    expect_equal(ld_ncol(corr), m)
    expect_equal(dim(corr), c(m, m))

    ## -- oracle 1: stats::cor() on the decoded matrix, within the window --
    R_full <- stats::cor(g)
    tol8 <- 1 / 127 + 1e-8

    max_abs_diff <- 0
    for (j in seq_len(m)) {
        cc <- ld_col(corr, j)
        lo <- max(1L, j - size)
        hi <- min(m, j + size)
        expect_equal(cc$lo, lo)
        expect_equal(cc$hi, hi)
        expect_equal(length(cc$x), hi - lo + 1L)
        d <- max(abs(cc$x - R_full[lo:hi, j]))
        max_abs_diff <- max(max_abs_diff, d)
        expect_true(d <= tol8,
                    info = sprintf("column %d: band vs cor() differ by %.5f", j, d))
    }
    message(sprintf("  snp_cor vs stats::cor(): max |diff| over all bands = %.6g (int8 tol %.6g)",
                    max_abs_diff, tol8))

    ## diagonal is exactly 1; a pair well outside the window is absent (0)
    expect_equal(ld_pair(corr, 10L, 10L), 1)
    expect_equal(ld_pair(corr, 1L, m), 0)
    ## symmetry
    expect_equal(ld_pair(corr, 3L, 6L), ld_pair(corr, 6L, 3L))

    ## -- int16 store: tighter tolerance -----------------------------------
    corr16 <- statgen_snp_cor(tn, size = size, bits = 16L)
    tol16 <- 1 / 32767 + 1e-8
    for (j in c(1L, 20L, m)) {
        cc <- ld_col(corr16, j)
        lo <- max(1L, j - size); hi <- min(m, j + size)
        expect_true(max(abs(cc$x - R_full[lo:hi, j])) <= tol16)
    }

    ## -- thr_r2 zeroes sub-threshold correlations within the band ---------
    corr_thr <- statgen_snp_cor(tn, size = size, thr_r2 = 0.05)
    for (j in seq_len(m)) {
        cc <- ld_col(corr_thr, j)
        lo <- max(1L, j - size); hi <- min(m, j + size)
        keep <- R_full[lo:hi, j]^2 >= 0.05
        keep[(lo:hi) == j] <- TRUE      # diagonal always kept
        ## kept entries match cor(), dropped entries are exactly 0
        expect_true(all(abs(cc$x[!keep]) == 0))
        expect_true(max(abs(cc$x[keep] - R_full[lo:hi, j][keep])) <= tol8)
    }

    ## -- position-based window (bigsnpr kb semantics) ---------------------
    pos <- as.double(sort(sample.int(500000L, m)))
    win_kb <- 50
    corr_pos <- statgen_snp_cor(tn, size = win_kb, infos_pos = pos)
    for (j in seq_len(m)) {
        in_win <- which(abs(pos - pos[j]) <= win_kb * 1000)
        cc <- ld_col(corr_pos, j)
        expect_equal(cc$lo, min(in_win))
        expect_equal(cc$hi, max(in_win))
        expect_true(max(abs(cc$x - R_full[cc$lo:cc$hi, j])) <= tol8)
    }

    ## -- input validation -------------------------------------------------
    expect_error(statgen_snp_cor(matrix(1, 3, 3), 5), "fmalloc_tensor")
    expect_error(statgen_snp_cor(tn, 5, bits = 7L), "bits")
    expect_error(statgen_snp_cor(tn, 5, infos_pos = pos[-1L]), "one position")

    ## -- dosage tensor: same estimator on continuous dosages --------------
    d <- g + 0
    storage.mode(d) <- "double"
    td <- fmalloc_dosage(d, runtime = rt)
    corr_d <- statgen_snp_cor(td, size = size)
    for (j in c(1L, 15L, m)) {
        cc <- ld_col(corr_d, j)
        lo <- max(1L, j - size); hi <- min(m, j + size)
        expect_true(max(abs(cc$x - R_full[lo:hi, j])) <= tol8)
    }

    ## -- optional oracle: bigsnpr::snp_cor(), when installed --------------
    if (requireNamespace("bigsnpr", quietly = TRUE) &&
        requireNamespace("bigstatsr", quietly = TRUE)) {
        ## snp_cor() wants an FBM.code256 (a genotype FBM with a 0/1/2/NA code)
        G256 <- bigstatsr::FBM.code256(n, m, code = bigsnpr::CODE_012, init = g)
        ## bigsnpr default alpha = 1 -> threshold 0, keeps every in-window pair
        big_corr <- bigsnpr::snp_cor(G256, size = size, alpha = 1, thr_r2 = 0)
        big_dense <- as.matrix(big_corr)

        max_big_diff <- 0
        for (j in seq_len(m)) {
            cc <- ld_col(corr, j)
            lo <- cc$lo; hi <- cc$hi
            max_big_diff <- max(max_big_diff, max(abs(cc$x - big_dense[lo:hi, j])))
        }
        message(sprintf("  snp_cor vs bigsnpr::snp_cor(): max |diff| over all bands = %.6g",
                        max_big_diff))
        expect_true(max_big_diff <= tol8)
    } else {
        message("  (bigsnpr not installed: skipped the bigsnpr::snp_cor oracle)")
    }

    invisible(NULL)
})()

## The band is written into the store one column at a time, so a run whose
## variant count is many times its window exercises the flush cycle repeatedly:
## a column must be stored exactly once, after the last step that can touch it,
## and never before. Compared against the same stats::cor() oracle.
(function() {
    set.seed(20260910)

    n <- 60L
    m <- 400L
    size <- 7L
    g <- matrix(sample(0:2, n * m, replace = TRUE, prob = c(0.25, 0.5, 0.25)),
                nrow = n, ncol = m)
    for (j in seq_len(m)) if (length(unique(g[, j])) == 1L) g[1L, j] <- (g[1L, j] + 1L) %% 3L
    storage.mode(g) <- "integer"

    tmp <- tempfile(fileext = ".bin")
    rt <- open_fmalloc(tmp, mode = "scratch", size_gb = 0.2)
    on.exit({ cleanup_fmalloc(rt); unlink(tmp) }, add = TRUE)

    corr <- statgen_snp_cor(fmalloc_bed(g, runtime = rt), size = size)
    expect_equal(ld_ncol(corr), m)

    R_full <- stats::cor(g)
    tol8 <- 1 / 127 + 1e-8
    worst <- 0
    for (j in seq_len(m)) {
        band <- ld_col(corr, j)
        rows <- seq.int(band$lo, band$hi)
        expect_equal(length(band$x), length(rows))
        worst <- max(worst, max(abs(band$x - R_full[rows, j])))
    }
    expect_true(worst <= tol8)
    ## outside the window the store is exactly zero, not merely small
    expect_equal(ld_pair(corr, 1L, m), 0)
    expect_equal(ld_pair(corr, 1L, size + 2L), 0)
    ## the diagonal survives every flush
    expect_equal(vapply(seq_len(m), function(j) ld_pair(corr, j, j), numeric(1)),
                 rep(1, m))
    cat(sprintf("  streamed %d columns at window +/-%d: max |diff| = %.6f\n",
                m, size, worst))
})()
