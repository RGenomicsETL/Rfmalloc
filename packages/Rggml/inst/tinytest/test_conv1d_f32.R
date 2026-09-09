library(tinytest)
library(Rggml)

# GGUF/GGML layout is [K, IC, OC], while the direct primitive returns the
# AST-preserving [Nout, OC, B]. Each case also forces the scalar callback and
# compares it with the runtime-dispatched implementation. AVX2/FMA changes
# F32 rounding through fused accumulation, so 2e-5 is intentionally wider than
# the observed 352-term convolution error while remaining far below model scale.
check_conv <- function(k, ic, oc, n, batch, stride, padding, dilation) {
  set.seed(k * 10000L + ic * 1000L + oc * 10L + batch)
  kernel <- array(rnorm(k * ic * oc, sd = 0.2), dim = c(k, ic, oc))
  input <- array(rnorm(n * ic * batch), dim = c(n, ic, batch))
  bias <- rnorm(oc, sd = 0.1)
  got <- Rggml:::rggml_test_conv1d_f32(
    kernel, input, bias, stride = stride, padding = padding, dilation = dilation
  )
  nout <- floor((n + 2L * padding - dilation * (k - 1L) - 1L) / stride) + 1L
  expect_equal(dim(got$direct), c(nout, oc, batch))
  expect_equal(got$scalar, got$im2col, tolerance = 2e-5)
  expect_equal(got$direct, got$scalar, tolerance = 2e-5)
  expect_equal(got$direct, got$im2col, tolerance = 2e-5)
}

# K=1, tails in IC/OC, batch one and strided output.
check_conv(1L, 3L, 5L, 9L, 1L, 2L, 0L, 1L)
# K=3, nonzero padding and batch > 1.
check_conv(3L, 5L, 7L, 11L, 2L, 1L, 1L, 1L)
# Production-shaped K=11 with both channel tails and dilation > 1.
check_conv(11L, 7L, 13L, 29L, 3L, 2L, 10L, 2L)
# Vector-width OC, so the tail-free path is also independently checked.
check_conv(3L, 9L, 16L, 17L, 2L, 1L, 2L, 2L)

expect_true(all(Rggml:::rggml_test_conv1d_f32_bounds()))
expect_error(
  Rggml:::rggml_test_conv1d_f32(
    array(1, c(3L, 2L, 1L)), array(1, c(2L, 2L, 1L)), 0,
    stride = 1L, padding = 0L, dilation = 1L
  ),
  "dimensions are invalid"
)

message("packed direct F32 conv1d differential tests completed")
