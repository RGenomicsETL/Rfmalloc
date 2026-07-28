# Create program-shaped state for incremental decoding

Allocates the persistent state declared by the bound program. Attention
layers receive key/value slabs, short-convolution layers receive causal
history, and gated-delta layers receive both convolution history and
recurrent matrices. State is plain R memory by default or
Rfmalloc-backed when `runtime` is supplied.

## Usage

``` r
rllm_kv_cache(model, n_ctx = 512L, runtime = NULL)
```

## Arguments

- model:

  An `rllm_model` from
  [`rllm_gguf_model()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_gguf_model.md).

- n_ctx:

  Maximum number of positions the cache can hold.

- runtime:

  Optional
  [`Rfmalloc::open_fmalloc()`](https://rdrr.io/pkg/Rfmalloc/man/open_fmalloc.html)
  runtime for the slabs.

## Value

An environment of class `rllm_kv_cache` with per-layer `k`, `v`, `conv`
and `recurrent` state, plus `n_ctx` and `n_past`.

## Details

The returned object is an environment, so
[`rllm_forward()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_forward.md)
can advance its `n_past` by reference.

## See also

[`rllm_forward()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_forward.md),
[`rllm_generate()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_generate.md)
