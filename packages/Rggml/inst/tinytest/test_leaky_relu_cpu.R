library(tinytest)
library(Rggml)

# Upstream GGML computes leaky ReLU on thread zero alone. The worker-split
# variant must agree with it exactly, at any thread count, for both signs and
# for element counts that do not divide evenly among the workers.
check_leaky <- function(dims, slope, threads) {
  set.seed(sum(dims) + threads)
  x <- array(rnorm(prod(dims)), dim = dims)
  x[seq_len(min(5L, length(x)))] <- 0
  got <- Rggml:::rggml_test_leaky_relu_cpu(x, slope, threads)
  expect_identical(got$parallel, got$official)
  reference <- as.vector(ifelse(x > 0, x, x * slope))
  expect_equal(got$parallel, reference, tolerance = 1e-7)
}

for (threads in c(1L, 3L, 8L)) {
  check_leaky(c(7L, 5L, 3L), 0.01, threads)
  check_leaky(c(64L, 32L, 2L), 0.2, threads)
  check_leaky(c(1L, 1L, 1L), 0.5, threads)
}

message("worker-split leaky ReLU differential tests completed")
