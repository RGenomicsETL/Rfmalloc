# Quantize a matrix into an Rfmalloc-backed quantized tensor

Encodes a dense numeric matrix into a GGUF quantized block format and
stores the compressed payload in Rfmalloc-backed (file-backed,
memory-mapped) storage, returning an `fmalloc_tensor`. This is the
write-side counterpart to Rgguf's `gguf_tensor(..., as = "native")`: the
resulting tensor keeps its quantized storage density and, when
multiplied as the right-hand operand of `dense %*% tensor` with the ggml
backend active (see
[`rllm_use_ggml()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_use_ggml.md)),
is contracted natively in quantized space by GGML's SIMD-dispatched
kernels without ever being decoded to double.

## Usage

``` r
rllm_quantize_tensor(x, dtype = "q4_k", runtime = NULL)
```

## Arguments

- x:

  A numeric matrix to quantize.

- dtype:

  Target quantized codec, one of `"q4_0"`, `"q4_1"`, `"q5_0"`, `"q5_1"`,
  `"q8_0"`, `"q2_0"`, `"q2_k"`, `"q3_k"`, `"q4_k"` (default), `"q5_k"`,
  `"q6_k"`.

- runtime:

  Optional Rfmalloc runtime handle (see
  [`Rfmalloc::open_fmalloc()`](https://rdrr.io/pkg/Rfmalloc/man/open_fmalloc.html));
  if `NULL`, Rfmalloc's default runtime is used.

## Value

An `fmalloc_tensor` of the given `dtype` with `dim(x)`.

## Details

The number of rows of `x` is the quantized (per-row) dimension and must
be a multiple of the codec's block size: 256 for the K-quants (`"q2_k"`,
`"q3_k"`, `"q4_k"`, `"q5_k"`, `"q6_k"`) and 32 for `"q4_0"`, `"q4_1"`,
`"q5_0"`, `"q5_1"`, `"q8_0"`; `"q2_0"` uses GGML's group-64 ternary
blocks.

## See also

[`rllm_use_ggml()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_use_ggml.md),
[`Rfmalloc::create_fmalloc_tensor()`](https://rdrr.io/pkg/Rfmalloc/man/fmalloc_tensor.html)

## Examples

``` r
rt <- Rfmalloc::open_fmalloc(tempfile(fileext = ".bin"))
set.seed(1)
W <- matrix(rnorm(256 * 4, sd = 0.4), nrow = 256)  # 256 must divide nrow
Wt <- rllm_quantize_tensor(W, "q4_k", runtime = rt)
X <- matrix(rnorm(3 * 256), nrow = 3)              # 3 x 256
Y <- X %*% Wt                                      # native quantized product
dim(Y)
#> [1] 3 4
```
