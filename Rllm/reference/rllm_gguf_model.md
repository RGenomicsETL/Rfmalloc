# Load a GGUF model as a bound semantic program

Normalizes the model-family metadata into a typed
[`rllm_program()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_input.md),
validates every tensor role and shape, then binds each required weight
directly to the original GGUF mapping. Quantized and floating-point
payloads keep their on-disk encoding; no second weight store is created.
The same program is the inspectable architecture and the source consumed
by the native GGML lowering.

## Usage

``` r
rllm_gguf_model(path, runtime = NULL, rope_mode = NULL)
```

## Arguments

- path:

  Path to a GGUF file.

- runtime:

  [`Rfmalloc::open_fmalloc()`](https://rdrr.io/pkg/Rfmalloc/man/open_fmalloc.html)
  runtime attached to the borrowed tensor views, or `NULL` to use the
  default established by
  [`Rfmalloc::init_fmalloc()`](https://rdrr.io/pkg/Rfmalloc/man/init_fmalloc.html).
  It supplies the allocation context for operations which produce
  fmalloc results; the weight bytes remain in the GGUF mapping.

- rope_mode:

  Optional RoPE override: `0` for normal/interleaved or `2` for NEOX.
  The architecture program supplies the default.

## Value

An object of class `rllm_model` containing its hyperparameters, borrowed
weight payloads, tokenizer metadata, and model-owned backend contexts
created lazily by
[`rllm_forward()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_forward.md).

## Details

Architecture definitions are data ASTs rather than native model-family
branches. Native GGML lowering covers llama, Qwen3.5, LFM2MoE and
EmbeddingGemma. ESM-2 and OpenSpliceAI are numerically executable
through
[`rllm_execute()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_execute.md),
but their multi-input or floating-point sequence-input programs are not
accepted by this native model loader. Models with tied embeddings reuse
`token_embd.weight` as the output projection.

## See also

[`rllm_forward()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_forward.md)
