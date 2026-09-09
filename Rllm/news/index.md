# Changelog

## Rllm 0.1.0 (unreleased)

- The native F32 lowering’s CUDA branch ran on real hardware for the
  first time. On the RTX 5050 rig (driver 595.95, CUDA 13.2, `sm_120a`)
  it borrows the model-owned weight backend, keeps the official im2col
  composition, and reuses its retained device context across calls.
  Against the CPU path the real OpenSpliceAI 80 nt and 400 nt
  checkpoints agree to 1.0e-6 and 1.2e-7 with `NVIDIA_TF32_OVERRIDE=0`;
  cuBLAS’s default TF32 tensor cores widen that to 7.3e-4, which Rggml’s
  README now documents. The device reached 16,678 samples/s at batch 256
  on the 80 nt model against that laptop CPU’s 7,004.

- The native F32 CPU lowering folds a batch normalization and a leaky
  rectification into the convolution that is their only reader, so an
  OpenSpliceAI residual block emits two nodes where it used to emit
  seven. A convolution that cannot take the fusion, and every device
  graph, still emits the operators as before.

- The native F32 CPU lowering now uses Rggml’s worker-split leaky ReLU,
  its retiled convolution kernel and the folded gather. On the i5-13500
  the real MANE 400 nt OpenSpliceAI checkpoint at `[4, 416, 128]` went
  from 260.1 ms to 41.6 ms per batch. With both implementations pinned
  to the same four idle performance cores, four threads each, it runs
  1.61x upstream PyTorch at 400 nt and batch 128, 1.92x at 400 nt and
  batch 32, 1.89x at 2000 nt, 1.36x at 10000 nt, and 0.77x to 1.09x at
  80 nt, where per-node launch cost still dominates. All four released
  MANE checkpoints agree with upstream PyTorch: 5e-7 at 80 nt, 6.0e-8 at
  400 nt and 2000 nt, 1.5e-15 at 10000 nt.

- Fixed a resource leak on rejected native execution. A malformed
  program is rejected by a validation helper deep inside graph
  construction, and an R error unwinds past local cleanup, so both
  native paths now own their transient GGML backend, graph contexts,
  backend buffer and packed convolution plans through a protected
  external pointer whose finalizer releases whatever was built.
  Rejecting one malformed OpenSpliceAI program leaked 0.6 to 1.3 MB of
  live allocation per call and now leaks none.

- Added an internal fixed-shape F32 dataflow lowering for bound programs
  whose single input is a named `[channel, sequence, batch]` array. It
  retains one CPU backend, its graph context, backend buffer and input
  and output staging across calls, and packs each generic convolution’s
  validated `[K, IC, OC]` F32 weights once into Rggml’s direct CPU plan;
  CUDA keeps the official im2col composition because a custom CPU
  callback is not a CUDA operation. The real MANE 80 nt OpenSpliceAI
  rs10 checkpoint stays within 5e-7 of both the dense R oracle and
  upstream PyTorch. On an i5-13500 at 14 threads and batch 128 it
  reaches 2,018.3 samples/s, 2.68 times the prior persistent CPU path
  but 24.2% of upstream PyTorch, so it remains unexported.

- Native inference now consumes a bound `rllm_program`: the serializable
  AST, its typed GGUF parameter bindings and its validated GGML lowering
  travel as one object. Forward execution, embedding, CUDA upload and
  persistent state no longer read or store a separate model plan or
  tensor-binding alias; `rllm_plan(model)` is a derived inspection view.
  GGUF metadata adapters now trace that program and declare its
  parameters directly instead of first allocating a layer plan and
  tensor table. Program mutations, not legacy plan mutations, drive the
  native semantic tests. Operator fields such as masks, routing, scale
  and state live directly on AST nodes rather than inside a second
  opaque specification wrapper.

- Named each sibling package explicitly in `Remotes`, so dependency
  installers replace stale monorepo dependencies instead of mistaking a
  shared repository commit for an already installed package.

- Attaching Rllm is quiet. Backend registration remains automatic and
  its selected state is available through
  [`rllm_backend_enabled()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_use_ggml.md).

- Added complete Qwen3.5 execution against the real Ternary Bonsai 27B
  metadata and tensor layout. Its serializable R program distinguishes
  48 gated-delta recurrent blocks from 16 gated-attention blocks,
  represents both recurrent and KV state, and binds all 851 parameters
  without a C++ model branch. Whole-batch and incremental paths match a
  pure-R oracle and run on CUDA with plain and fmalloc-backed state; all
  248,320 token-zero logits from the real checkpoint match a portable
  upstream llama.cpp build bit-for-bit.

- Added official GGML group-64 `q2_0` to quantization and the
  typed-product bridge. The codec stays encoded in Rfmalloc storage and
  uses GGML’s ternary decoder and CPU dot product.

- Added a data-only architecture program traced from ordinary R modules
  and base pipes. It preserves several typed inputs, several named
  operator results, parameter identities, representation taps and
  structured shared loops.
  [`rllm_execute()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_execute.md)
  runs dataflow and carried loops through explicit R lowerings; its
  dense oracle pins the Tiny Recursive Model recurrence against direct
  iteration. The ESM-2 8M probe records all six layers and its real
  attention-map contact path. The Evo 2 7B probe records all 32 HCS,
  HCM, HCL and attention blocks with their distinct state geometries.
  Unsupported operator lowerings fail explicitly.

- Added a direct ESM-2 GGUF adapter and dense semantic execution for
  token dropout, key-padded NEOX attention with attention-map results,
  tied projection and contact regression. `tools/convert_esm2.R` reads
  the official ESM-2 8M safetensors checkpoint and writes its 106
  unmodified F32 tensors through Rgguf’s upstream GGUF implementation.
  Its 69-node program matches upstream logits, representations,
  attention probabilities and contacts on a fixed protein. Native GGML
  execution rejects the unsupported two-input grammar explicitly instead
  of selecting a hidden model-family path.

- Added an OpenSpliceAI GGUF adapter and dense semantic program for
  dilated one-dimensional convolution, inference batch normalization,
  leaky ReLU, residual and accumulated skip paths, context cropping and
  per-position softmax. The converter uses Python only to unwrap an
  upstream PyTorch state dictionary into safetensors, then writes 56 F32
  tensors through Rgguf. The complete output of the MANE 80 nt rs10
  checkpoint matches upstream PyTorch within 5e-7; unsupported native
  GGML lowering remains an explicit failure.

- GGUF metadata is normalized into semantic programs for llama, LFM2MoE
  and EmbeddingGemma before any weight is borrowed. The model-neutral
  lowerer covers causal, bidirectional and symmetric-window attention,
  normal and NEOX RoPE, post-branch RMS normalization, SwiGLU, GEGLU,
  short convolution, sparse routed experts, mean pooling and dense
  embedding projections.

- Added
  [`rllm_embed()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_embed.md)
  and pinned EmbeddingGemma against both a pure-R f32 oracle and a real
  300M Q8_0 model compared with upstream GGML execution.

- Added the CUDA transformer path over Rggml’s official backend. A
  model-owned context uploads codec-native weights once and reuses them;
  mutable inputs, cache state and logits use the backend-neutral
  transfer API. The RTX 5050 suite pins whole-batch and incremental
  logits, plain and fmalloc caches, and CPU/CUDA cache handoff. A
  12-token prompt plus 128 greedy tokens measured a median 40.2 tok/s on
  CPU and 69.7 tok/s on CUDA after upload. A persistent GGML scheduler
  and graph-capture experiment measured 63.7 and 62.6 tok/s and was
  removed rather than retained as unused machinery.

- [`rllm_gguf_model()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_gguf_model.md)
  now borrows every weight directly from the read-only GGUF mapping. It
  no longer copies all two-dimensional weights into an Rfmalloc file or
  dequantizes and repacks one-dimensional norms. The model retains the
  mapping, and GGML receives the original encoded pointers.

- [`rllm_generate()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_generate.md)
  gained **sampling**: `temperature` (0 = greedy, the default, so
  existing behaviour is unchanged), `top_k`, `top_p` (nucleus), and
  `seed` for reproducibility. Greedy stays deterministic; sampled
  decoding is reproducible under a fixed seed. The sampler is pure logic
  over the logits vector (`test_sampling.R`).

- Added **incremental decoding with a KV cache** and the
  **bytes-boundary generation API**.
  [`rllm_kv_cache()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_kv_cache.md)
  allocates per-layer f32 cache slabs (plain R memory, or fmalloc-backed
  when given a runtime - the cache as a disk citizen; a quantized cache
  codec can later replace the slabs without touching the graph).
  `rllm_forward(cache =)` appends new keys/values and attends over
  everything cached (llama.cpp’s classic layout: K flat, V transposed),
  advancing `n_past` by reference.
  [`rllm_generate()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_generate.md)
  prefills the cache and decodes greedily one token per step; on
  SmolLM2-135M the current CPU path measures 40.2 tok/s for a 128-token
  controlled run. The correctness invariant - incremental logits equal
  whole-batch logits at every position - is pinned on the synthetic
  model for plain and fmalloc cache backings (`test_kv_cache.R`).

- The model I/O boundary is **bytes, not text**:
  [`rllm_encode()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_decode.md)/
  [`rllm_decode()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_decode.md)
  convert raw bytes to and from token ids using only the GGUF’s own
  byte-level BPE metadata (`tokenizer.ggml.tokens`/`merges`, GPT-2 byte
  alphabet) - no external tokenizer;
  [`rawToChar()`](https://rdrr.io/r/base/rawConversion.html) is the
  caller’s interpretation. Encoding applies merge ranks without GPT-2’s
  regex pre-tokenizer, so splits may occasionally differ from
  llama.cpp’s canonical ones while always decoding back to the same
  bytes. Real-model record: “The capital of France is Paris. The capital
  of Germany is” continues ” Berlin. The capital of Italy is Rome. The
  capital of Spain is Madrid.”

- **Validated on a real model** (SmolLM2-135M `Q4_K_M`, 30 layers, GQA
  9:3, 272 tensors in a `q4_k`/`q5_0`/`q6_k`/`q8_0`/`f32` mix): with the
  model’s decoded weights, the GGML graph matches a pure-R reference
  forward to 1e-06 relative (f32-twin roundtrip written with Rgguf’s own
  writer), and the native quantized path agrees on the argmax; its ~0.19
  relative logit deviation is Q8_K-activation/quantized-weight
  arithmetic compounded over 30 layers, not graph error. An opt-in smoke
  test (`RLLM_TEST_GGUF=<path>`, `test_real_model.R`) exercises the
  loader and graph on real files without affecting CI/CRAN.

- `q5_0`, `q5_1`, `q3_k`, and `q5_k` remain accepted by quantization and
  the typed-GEMM bridge. Their Rfmalloc codec registration now lives
  with every other GGUF storage codec in Rgguf and uses Rggml’s
  reference decoder.

- Added the **llama-architecture forward pass**:
  [`rllm_gguf_model()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_gguf_model.md)
  loads a GGUF model’s hyperparameters and exposes weights in their
  native `q4_k`/`f32`/… encoding, and
  [`rllm_forward()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_forward.md)
  assembles the GGML compute graph (RMSNorm, RoPE, causal self-attention
  with grouped-query support, SwiGLU) from Rggml’s graph-op C-callables
  over those weights zero-copy, computing the logits for every position
  of a token batch on the GGML CPU backend. Quantized weights are
  contracted through the SIMD-dispatched quantized kernels without ever
  being decoded to double. Verified against a pure-R reference
  implementation of the same arithmetic on a synthetic GGUF model
  written at test time - logits agree to float accumulation error (\<
  1e-4 relative) for both multi-head and grouped-query configurations,
  plus causality probes (`test_llama_forward.R`).

- Initial release. Rllm is the composition layer of the Rfmalloc
  ecosystem, wiring together Rfmalloc (file-backed storage), Rgguf (GGUF
  weights as fmalloc tensors) and Rggml (vendored GGML compute with
  runtime-SIMD-dispatched quantized kernels).

- Registers Rggml as an **Rfmalloc codec-aware (“typed”) matrix-multiply
  backend** named `"ggml"`, selected on load (toggle with
  [`rllm_use_ggml()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_use_ggml.md)).
  When active, a product `dense %*% quantized_tensor` (the tensor being
  an `fmalloc_tensor` of a GGUF quantized codec: `q4_0`, `q4_1`, `q8_0`,
  `q2_k`, `q4_k`, `q6_k`) is computed natively in quantized space:
  Rfmalloc hands the raw compressed payload to Rllm, which points a GGML
  tensor at it zero-copy and contracts each weight row through GGML’s
  SIMD-dispatched `vec_dot`, quantizing the dense operand on the fly -
  no decode to `double`. Orientations and codecs the ggml path cannot
  serve (tensor on the left, non-quantized codecs) are declined and fall
  back to Rfmalloc’s decode-then-BLAS path, so results are always
  correct regardless of the selected backend.

- [`rllm_quantize_tensor()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_quantize_tensor.md)
  encodes a dense matrix into a GGUF quantized block format and stores
  the payload in Rfmalloc-backed storage, returning an
  `fmalloc_tensor` - the write-side counterpart to Rgguf’s
  `gguf_tensor(..., as = "native")`.
