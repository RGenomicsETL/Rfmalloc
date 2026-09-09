library(tinytest)
library(Rggml)

# The knob is process-level state that the CUDA libraries read when they
# initialize, so it must be settable before anything touches CUDA and must
# report the previous value rather than silently overwriting it.
old <- Sys.getenv("NVIDIA_TF32_OVERRIDE", unset = NA_character_)
on.exit({
    if (is.na(old)) Sys.unsetenv("NVIDIA_TF32_OVERRIDE") else
        Sys.setenv(NVIDIA_TF32_OVERRIDE = old)
}, add = TRUE)

Sys.unsetenv("NVIDIA_TF32_OVERRIDE")
expect_equal(suppressWarnings(rggml_cuda_tf32(FALSE)), "")
expect_equal(Sys.getenv("NVIDIA_TF32_OVERRIDE"), "0")
expect_equal(suppressWarnings(rggml_cuda_tf32(TRUE)), "0")
expect_equal(Sys.getenv("NVIDIA_TF32_OVERRIDE", unset = ""), "")
expect_error(rggml_cuda_tf32("no"), "must be TRUE or FALSE")
expect_error(rggml_cuda_tf32(NA), "must be TRUE or FALSE")

# Asking CUDA anything marks the session, and the knob then warns instead of
# quietly failing. A build without CUDA never marks it.
touched_before <- .Call("RC_rggml_cuda_touched")
expect_true(is.logical(touched_before))
invisible(rggml_cuda_info())
if (rggml_has_cuda()) {
    expect_true(.Call("RC_rggml_cuda_touched"))
    expect_warning(rggml_cuda_tf32(FALSE), "already initialized")
} else {
    expect_false(.Call("RC_rggml_cuda_touched"))
}

message("CUDA reduced-precision control tests completed")
