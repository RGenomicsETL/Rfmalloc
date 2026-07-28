# Generate tokens with a KV cache (greedy or sampled)

The bytes-in/bytes-out generation entry point: prefills a KV cache with
the prompt (ids or raw bytes), then decodes one token per step - each
step is a single-token
[`rllm_forward()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_forward.md)
over the cache, not a full re-forward. Stops after `n_new` tokens or at
the model's EOS token. Decoding is greedy by default
(`temperature = 0`); set a positive temperature for temperature-scaled
sampling, optionally narrowed by `top_k` and/or `top_p`.

## Usage

``` r
rllm_generate(
  model,
  prompt,
  n_new = 32L,
  temperature = 0,
  top_k = 0L,
  top_p = 1,
  seed = NULL,
  cache = NULL,
  runtime = NULL,
  backend = c("cpu", "cuda")
)
```

## Arguments

- model:

  An `rllm_model` from
  [`rllm_gguf_model()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_gguf_model.md).

- prompt:

  Integer vector of 0-based token ids, or a raw vector (encoded with
  [`rllm_encode()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_decode.md);
  a single string is converted via
  [`charToRaw()`](https://rdrr.io/r/base/rawConversion.html) as a
  convenience). A textual prompt at the start of a sequence receives the
  GGUF tokenizer's BOS token when one is declared. Integer prompts
  remain exact and are never modified.

- n_new:

  Maximum number of tokens to generate.

- temperature:

  Sampling temperature. `0` (default) is greedy/argmax; larger values
  flatten the distribution (more diverse).

- top_k:

  Keep only the `top_k` highest-logit tokens before sampling (`0`,
  default, disables the cutoff). Ignored when greedy.

- top_p:

  Nucleus sampling: keep the smallest set of tokens whose probabilities
  sum to at least `top_p` (`1`, default, disables it). Ignored when
  greedy.

- seed:

  Optional integer seed, for reproducible sampling.

- cache:

  Optional
  [`rllm_kv_cache()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_kv_cache.md)
  to continue from; by default a fresh cache sized
  `length(prompt) + n_new` is created.

- runtime:

  Optional
  [`Rfmalloc::open_fmalloc()`](https://rdrr.io/pkg/Rfmalloc/man/open_fmalloc.html)
  runtime for the cache slabs (file-backed cache).

- backend:

  Compute backend passed to
  [`rllm_forward()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_forward.md).
  `"cuda"` requires a CUDA-enabled Rggml build.

## Value

A list with `ids` (prompt + generated, 0-based), `new_ids` (the
generated tokens), and `raw` (the generated tokens decoded to bytes, or
`NULL` when the model has no tokenizer metadata).
