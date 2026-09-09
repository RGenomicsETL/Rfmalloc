#' Windowed LD (correlation) matrix from a genotype tensor
#'
#' Computes the (Pearson) correlations between nearby variants of a `bed` or
#' `dosage` [Rfmalloc::fmalloc_tensor] and stores them as a compressed, banded
#' [Rfmalloc::fmalloc_ld] LD matrix. This is the fused form
#' `bigsnpr::snp_cor()` uses (Prive et al. 2018,
#' \doi{10.1093/bioinformatics/bty185}): the genotype tensor is streamed one
#' variant column at a time through [Rfmalloc::fmalloc_tensor]'s decode (never
#' materializing the dense samples x variants matrix), each column is
#' mean-imputed, centred and normalized to unit L2 norm, and the correlation of
#' two nearby variants is then the dot product of their unit columns. Each band
#' column is written into the store as soon as no later variant can add to it,
#' so what stays resident is a sliding window of unit columns and of in-flight
#' band columns, both bounded by the window width rather than by the variant
#' count; the band itself lives in the store's own fmalloc storage.
#'
#' Because the variants are assumed position-sorted, each variant's in-window
#' neighbours are a contiguous index range, so the full symmetric band is kept
#' per column with no explicit neighbour indices (see [Rfmalloc::fmalloc_ld]).
#' The diagonal is `1`. The result is storage-agnostic: read it with
#' [Rfmalloc::ld_col()] / [Rfmalloc::ld_pair()], or hand it to LDpred2.
#'
#' Missing genotype calls are mean-imputed per variant before the correlation
#' (matching [Rfmalloc::fmalloc_bed_standardize()]). On complete data this is
#' the same estimator as `bigsnpr::snp_cor()`; with missingness the two differ
#' (bigsnpr deletes pairwise-incomplete samples instead of imputing).
#'
#' @param genotypes A `"bed"` or `"dosage"` [Rfmalloc::fmalloc_tensor]
#'   (samples x variants), the variants position-sorted.
#' @param size Window half-width. With `infos_pos = NULL` (the default) this is
#'   a window in **number of variants** each side (variant `j` is correlated
#'   with variants `j - size .. j + size`). When `infos_pos` is supplied, it is
#'   a window in **kilobases** of physical distance (bigsnpr's convention: the
#'   window in position units is `size * 1000`).
#' @param thr_r2 Squared-correlation threshold: correlations with `r^2 < thr_r2`
#'   are set to `0` within the band. Default `0` (keep every in-window
#'   correlation).
#' @param infos_pos `NULL` (window by variant index) or a numeric vector of
#'   physical positions, one per variant, **sorted** (window by genomic
#'   distance).
#' @param bits Quantization width for the stored correlations, `8` (int8, the
#'   default; resolution `~1/127`) or `16` (int16; resolution `~3e-5`).
#'
#' @return An [Rfmalloc::fmalloc_ld] banded LD matrix (`ncol(genotypes)` x
#'   `ncol(genotypes)`).
#'
#' @seealso [Rfmalloc::fmalloc_ld], [Rfmalloc::ld_col()],
#'   [Rfmalloc::ld_pair()], [statgen_ldpred2_inf()]
#' @examples
#' rt <- Rfmalloc::open_fmalloc(tempfile(fileext = ".bin"), mode = "scratch",
#'                              size_gb = 0.1)
#' set.seed(1)
#' n <- 100L; m <- 20L
#' g <- matrix(sample(0:2, n * m, replace = TRUE), nrow = n, ncol = m)
#' storage.mode(g) <- "integer"
#' tn <- Rfmalloc::fmalloc_bed(g, runtime = rt)
#' corr <- statgen_snp_cor(tn, size = 3)
#' Rfmalloc::ld_pair(corr, 1, 2)
#' Rfmalloc::cleanup_fmalloc(rt)
#' @export
#' @useDynLib RfmallocStatgen, .registration = TRUE
statgen_snp_cor <- function(genotypes, size, thr_r2 = 0, infos_pos = NULL,
                            bits = 8L) {
    if (!inherits(genotypes, "fmalloc_tensor") ||
        !(attr(genotypes, "rfm_dtype") %in% c("bed", "dosage"))) {
        stop("genotypes must be a 'bed' or 'dosage' fmalloc_tensor (see ",
             "Rfmalloc::fmalloc_bed()/fmalloc_dosage())")
    }
    dims <- dim(genotypes)
    if (length(dims) != 2L) {
        stop("genotypes must be a 2-dimensional tensor (samples x variants)")
    }
    n <- dims[1L]
    m <- dims[2L]

    size <- as.numeric(size)
    if (length(size) != 1L || is.na(size) || size < 0) {
        stop("size must be a single non-negative number")
    }
    thr_r2 <- as.numeric(thr_r2)
    if (length(thr_r2) != 1L || is.na(thr_r2) || thr_r2 < 0) {
        stop("thr_r2 must be a single non-negative number")
    }
    bits <- as.integer(bits)
    if (!bits %in% c(8L, 16L)) {
        stop("bits must be 8 or 16")
    }
    if (!is.null(infos_pos)) {
        infos_pos <- as.double(infos_pos)
        if (length(infos_pos) != m) {
            stop(sprintf("infos_pos (%d) must have one position per variant (%d)",
                         length(infos_pos), m))
        }
        if (anyNA(infos_pos) || is.unsorted(infos_pos)) {
            stop("infos_pos must be sorted and free of missing values")
        }
    }

    payload <- .Call(C_statgen_snp_cor, genotypes, as.integer(n), as.integer(m),
                     size, thr_r2, infos_pos, bits)
    class(payload) <- c("fmalloc_ld", "fmalloc")
    payload
}
