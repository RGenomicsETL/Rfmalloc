# Reduced-precision F32 products on CUDA devices

cuBLAS computes single-precision matrix products on TF32 tensor cores by
default on Ampere and later devices. TF32 keeps ten mantissa bits rather
than twenty-four, so a device result stops agreeing with the CPU backend
to F32 accumulation error: one OpenSpliceAI forward pass on an RTX 5050
differed from its CPU result by 7.3e-4 with the default and by 1.0e-6
with TF32 off, for 19% less throughput.

## Usage

``` r
rggml_cuda_tf32(enabled = FALSE)
```

## Arguments

- enabled:

  `TRUE` to leave cuBLAS its default reduced-precision path, `FALSE`
  (the default) to require full F32.

## Value

The previous setting of `NVIDIA_TF32_OVERRIDE` (`""` when unset),
invisibly.

## Details

This is a property of the CUDA libraries, not of GGML, and they read it
when they initialize. Call this before anything in the session touches
CUDA, including
[`rggml_cuda_info()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rggml/reference/rggml_cuda_info.md);
afterwards only a fresh R session can change it, and this function says
so rather than pretending to have worked.

## Examples

``` r
rggml_cuda_tf32(FALSE)
```
