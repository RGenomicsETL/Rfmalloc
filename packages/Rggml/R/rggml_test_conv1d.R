rggml_test_conv1d_f32 <- function(kernel, input, bias, stride = 1L,
                                  padding = 0L, dilation = 1L) {
  storage.mode(kernel) <- "double"
  storage.mode(input) <- "double"
  storage.mode(bias) <- "double"
  .Call(
    "RC_rggml_test_conv1d_f32", kernel, input, bias,
    as.integer(stride), as.integer(padding), as.integer(dilation)
  )
}

rggml_test_conv1d_f32_bounds <- function() {
  .Call("RC_rggml_test_conv1d_f32_bounds")
}
