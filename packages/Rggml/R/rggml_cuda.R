#' CUDA GPU backend availability
#'
#' Reports the NVIDIA CUDA devices GGML can see. The backend is opt-in at
#' installation because it requires the CUDA toolkit and compiles native GPU
#' kernels:
#'
#' \preformatted{
#'   install.packages("Rggml", configure.args = "--with-cuda")
#'   R CMD INSTALL --configure-args=--with-cuda .
#' }
#'
#' Set `CUDA_HOME` when the toolkit is outside the search path. By default the
#' source installation targets the GPU visible at build time. Set
#' `RGGML_CUDA_ARCH` to an explicit nvcc architecture such as `sm_89` or
#' `sm_120a` when building for another machine.
#'
#' A build without CUDA returns zero devices rather than failing, so callers
#' can probe and fall back to Vulkan, BLAS or CPU computation.
#'
#' @return A list with `n_devices` (integer) and `device` (the description of
#'   device 0, or `NA` when there is none).
#' @examples
#' rggml_cuda_info()
#' @export
rggml_cuda_info <- function() {
    .Call("RC_rggml_cuda_info")
}

#' @rdname rggml_cuda_info
#' @return `rggml_has_cuda()` returns `TRUE` when at least one CUDA device is
#'   usable.
#' @export
rggml_has_cuda <- function() {
    rggml_cuda_info()$n_devices > 0L
}

#' Reduced-precision F32 products on CUDA devices
#'
#' cuBLAS computes single-precision matrix products on TF32 tensor cores by
#' default on Ampere and later devices. TF32 keeps ten mantissa bits rather
#' than twenty-four, so a device result stops agreeing with the CPU backend to
#' F32 accumulation error: one OpenSpliceAI forward pass on an RTX 5050
#' differed from its CPU result by 7.3e-4 with the default and by 1.0e-6 with
#' TF32 off, for 19% less throughput.
#'
#' This is a property of the CUDA libraries, not of GGML, and they read it when
#' they initialize. Call this before anything in the session touches CUDA,
#' including [rggml_cuda_info()]; afterwards only a fresh R session can change
#' it, and this function says so rather than pretending to have worked.
#'
#' @param enabled `TRUE` to leave cuBLAS its default reduced-precision path,
#'   `FALSE` (the default) to require full F32.
#'
#' @return The previous setting of `NVIDIA_TF32_OVERRIDE` (`""` when unset),
#'   invisibly.
#' @examples
#' rggml_cuda_tf32(FALSE)
#' @export
rggml_cuda_tf32 <- function(enabled = FALSE) {
    if (!is.logical(enabled) || length(enabled) != 1L || is.na(enabled)) {
        stop("enabled must be TRUE or FALSE")
    }
    previous <- Sys.getenv("NVIDIA_TF32_OVERRIDE", unset = "")
    if (isTRUE(.Call("RC_rggml_cuda_touched"))) {
        warning("CUDA is already initialized in this session; ",
                "NVIDIA_TF32_OVERRIDE takes effect only in a fresh R session")
    }
    if (enabled) {
        Sys.unsetenv("NVIDIA_TF32_OVERRIDE")
    } else {
        Sys.setenv(NVIDIA_TF32_OVERRIDE = "0")
    }
    invisible(previous)
}
