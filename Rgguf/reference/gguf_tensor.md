# Read and Dequantize a GGUF Tensor

Reads a single named tensor from a GGUF file, dequantizing it if needed,
into a fresh Rfmalloc-backed ALTREP matrix or array of doubles.

## Usage

``` r
gguf_tensor(x, name, runtime = NULL, as = c("numeric", "native", "view"))
```

## Arguments

- x:

  Either a path to a GGUF file, or a `"gguf_ctx"` object returned by
  [`gguf_open()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rgguf/reference/gguf_open.md).

- name:

  Tensor name, as it appears in
  [`gguf_tensors()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rgguf/reference/gguf_tensors.md)'s
  `name` column.

- runtime:

  Optional `Rfmalloc` runtime handle (see
  [`Rfmalloc::open_fmalloc()`](https://rdrr.io/pkg/Rfmalloc/man/open_fmalloc.html)).
  If `NULL`, `Rfmalloc`'s own default-runtime resolution is used.

- as:

  `"numeric"` (default) dequantizes the whole tensor into an fmalloc
  double matrix/array. `"native"` copies the tensor's raw, still-encoded
  payload into fmalloc storage and returns an
  [`Rfmalloc::create_fmalloc_tensor()`](https://rdrr.io/pkg/Rfmalloc/man/fmalloc_tensor.html)
  typed tensor: it keeps the GGUF storage density (e.g. 4.5 bits/weight
  for `q4_k`) and is decoded in bounded panels only when used in matrix
  products. `"view"` returns the same typed tensor without copying: it
  borrows the exact read-only span in the original GGUF mapping and
  keeps that mapping alive. Native mode requires two dimensions; view
  mode accepts any rank. Both require a registered Rfmalloc codec.

## Value

For `as = "numeric"`, an `Rfmalloc`-backed ALTREP matrix (if the tensor
has 2 dimensions) or array (otherwise) of doubles, with
[`dim()`](https://rdrr.io/r/base/dim.html) equal to the tensor's GGUF
dimensions in `c(dim[0], dim[1], ...)` order. For `as = "native"` or
`"view"`, an `fmalloc_tensor` with the same dims.

## Details

GGUF tensor dimensions are stored with `dim[0]` as the fastest-varying
(contiguous) dimension. R arrays are column-major, i.e. the first
dimension varies fastest, so `dim[0]` maps directly onto R's first
dimension with no transposition needed:
`dim(result) == c(dim[0], dim[1], ...)`. See
`inst/tinytest/test_gguf_roundtrip.R` for a test that verifies this
mapping by writing known matrices/arrays and reading them back.

## Examples

``` r
tmp <- tempfile(fileext = ".gguf")
gguf_write_tensors(tmp, list(w = matrix(1:6 + 0.5, nrow = 2)))
rt <- Rfmalloc::open_fmalloc(tempfile(fileext = ".bin"))
w <- gguf_tensor(tmp, "w", runtime = rt)
w
#>      [,1] [,2] [,3]
#> [1,]  1.5  3.5  5.5
#> [2,]  2.5  4.5  6.5
#> attr(,"class")
#> [1] "fmalloc_matrix" "matrix"         "fmalloc"        "numeric"       
Rfmalloc::is_fmalloc_vector(w)
#> [1] TRUE
unlink(tmp)
```
