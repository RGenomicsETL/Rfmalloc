# Rggml 0.1.0 (unreleased)

- Documented that cuBLAS computes F32 products on TF32 tensor cores by default
  on Ampere and later, and what that costs in agreement with the CPU backend.

- The packed F32 conv1d plan can absorb a per-input-channel affine, a leaky
  rectification, or both, through `Rggml_conv_1d_f32_plan_fuse_input()`. A
  unit-stride convolution gathers each input channel as one window, activates
  it once, and reads every tap as an overlapping slice of it, so the folded
  operators cost one pass over the input rather than one graph node, one
  barrier and two passes over the activation each. Padding stays zero under a
  folded affine, which is what the unfolded operators compute, and a focused
  differential test pins the two forms against each other.

- Rewrote the packed F32 conv1d kernel around output tiles. A work item now
  gathers its own `[tap * channel][position]` tile, so padding, stride and
  dilation leave the inner loop, and the AVX2 form keeps twelve accumulators,
  three input vectors and one broadcast live, which turns one fused
  multiply-add per weight load into four. The plan writes the AST layout
  directly, so the permutation and the materializing copy that followed every
  convolution are gone. Workers claim tiles from a shared counter instead of
  taking a fixed share, because performance cores retire this kernel about
  twice as fast as efficiency cores and a static split left every node waiting
  on the slowest thread.

- Added `Rggml_leaky_relu_cpu()`, a leaky ReLU spread over every CPU worker.
  Upstream GGML computes that operator on thread zero alone, which stalled
  every other worker at the next barrier for the whole of it. The new operator
  is bit-identical to the official one at any thread count; device graphs keep
  the official operator because a custom CPU callback is not a device
  operation.

- Added a generic persistent packed F32 conv1d CPU C-callable plan. It copies
  validated `[K, IC, OC]` weights and optional bias once, stores output-channel
  blocks contiguously, and builds a CPU-only custom graph operation with scalar
  fallback plus runtime-gated x86 AVX2/FMA. CUDA retains the official F32
  im2col/matrix-product graph. Added leaky ReLU and CPU thread control
  C-callables for bound dataflow programs.

- Fixed webR and Apple Silicon portability. wasm builds use GGML's official
  SIMD128 quant kernels and omit the incompatible Fortran BLAS bridge; dense
  products continue through the CPU backend. Target-macro classification also
  prevents Apple clang from accepting and then ignoring x86 AVX2 flags on
  arm64. Default package metadata no longer makes r-universe provision the
  optional CUDA toolkit for CPU-only builds.

- Updated the generated engine to official GGML v0.16.0. This adds the
  upstream group-64 `Q2_0` ternary block format and lets Rggml consume it
  through the official GGUF and CPU implementations on one content-pinned
  source tree. Upstream does not yet provide a `Q2_0` CUDA or Vulkan product.

- Extended the backend-neutral graph C-callables with L2 normalization,
  softplus, MRoPE, gated-delta recurrence, and four-dimensional reshape and
  view operations. Rllm composes these reusable operators for Qwen3.5 without
  linking GGML or adding a native model-family class.

- Corrected CUDA launch geometry for recurrent models. SSM convolution selects
  a thread width that divides the channel count, while gated-delta kernels
  support power-of-two state widths from 1 through 128 without partial-warp
  out-of-bounds access. The Qwen3.5 whole-batch and incremental graph tests run
  these paths on the RTX rig.

- Removed the unused C-callable API counter and its accumulated version strata.
  The installed header is the one current monorepo contract. The official GGUF
  writer service also accepts string-array metadata, which makes hermetic
  tokenizer-bearing model fixtures possible without another writer.

- Corrected the aarch64 fallback map so the generic Q1 dot product keeps its
  `_generic` name while GGML's ARM translation unit owns the canonical NEON
  symbol. Rggml now links cleanly on both Apple Silicon and Linux aarch64.

- Added an opt-in `--with-cuda` build from the content-pinned official GGML
  CUDA sources. `rggml_cuda_info()` probes without making CUDA a hard
  dependency, and dense F32 plus Q4_K products use the same backend-owned
  buffer allocation, upload, compute and download path as Vulkan. The build
  accepts `CUDA_HOME` and `RGGML_CUDA_ARCH`; ordinary CPU builds never invoke
  `nvcc`. Extended Blackwell targets such as `sm_120a` emit matching PTX and
  SASS. Optional CUDA graph capture is gated by `RGGML_CUDA_GRAPHS=1` and is
  disabled by default after a slower full-generation measurement on the rig.

- Added GGML's official `gguf.cpp` to the generated vendor recipe and exposed
  its indexed reader and writer as opaque C-callable services. Rgguf now owns
  only its R surface and read-only tensor mapping; the partial second parser
  and duplicate quantized decoders are gone.

- Added `rggml_mul_mat(A, B, backend)`, a public GEMM (`crossprod(A, B)`) on the
  `"cpu"`, `"blas"`, or `"vulkan"` backend via GGML's device-buffer residency
  path - the supported entry point for a dense single-precision matrix multiply
  on the GPU, promoted from the internal test routine.

- Added `rggml_cpu_info()`, reporting what `configure` actually compiled:
  `arch_kernels` (`"arm"` for GGML's native NEON kernels, `"wasm"` for its
  SIMD128 kernels, `"generic"` for the portable ones), `simd_dispatch`,
  `blas`, `sgemm`, `vulkan`, and `cuda`. Every branch `configure`
  can take is numerically correct, so no numerical test can tell them apart: a
  build that silently fell back from the NEON kernels, or from `sgemm_` to the
  `dgemm_` promotion, passes the whole suite. `test_cpu_info.R` now asserts on
  this, including the invariant that `arch_kernels == "arm"` and
  `simd_dispatch` are mutually exclusive (both define the canonical
  `ggml_vec_dot_q4_K_q8_K`, and the linker would silently pick one), and that
  real aarch64 CI runners land on `"arm"`.

- **On aarch64, GGML's own hand-tuned NEON kernels are compiled in** instead of
  the portable scalar reference: 23 `vec_dot`s covering every quantized type,
  not just the one `q4_K` variant the SIMD dispatcher stages. NEON is a
  mandatory baseline of the architecture, so `ggml-cpu/arch/arm/quants.c` needs
  no ISA flag and cannot produce a binary that will not run on the machine that
  built it; `configure` therefore drops `-DGGML_CPU_GENERIC` and skips the
  dispatcher there (`arch/arm` defines the canonical symbols it would
  duplicate). x86 keeps the dispatcher on purpose: `arch/x86`'s kernels are
  selected by compile-time `#if defined(__AVX2__)`, so a `-mavx2` build of them
  would *require* AVX2 at runtime - which is why GGML upstream needs
  `GGML_CPU_ALL_VARIANTS` plus `dlopen` and we do not.
- **Fixed a BLAS link failure on Windows** that was really a portability bug.
  R guarantees only the **double-precision** BLAS: `<R_ext/BLAS.h>` declares no
  single-precision routine and `Rblas.dll` exports no `sgemm_`. Most Linux
  reference BLAS builds happen to export it, which is why the `cblas_sgemm`
  shim linked here but not there. `configure` now link-probes `sgemm_` against
  R's own `BLAS_LIBS`/`FLIBS`; where it is missing, the shim promotes its
  operands to double, calls `dgemm_` (still an optimized BLAS), and demotes the
  result - correct to float precision, with a naive triple loop only if the
  temporaries cannot be allocated. Set `RGGML_NO_SGEMM=1` to force the promoted
  path anywhere.
- **Rggml now builds on Windows** (Rtools/MinGW); `OS_type: unix` is gone. R
  runs `configure.win`, which re-execs the single `configure` with
  `RGGML_WINDOWS=1` rather than duplicating the SIMD probe, the Vulkan shader
  pipeline and the `libggml.a` build. The Windows branch writes
  `src/Makevars.win` without `-ldl` (Windows has no libdl) and locates the
  Vulkan SDK under its Windows layout (`$VULKAN_SDK/Bin/glslc.exe`,
  `-lvulkan-1`). A `windows-latest` CI job verifies it. The vendored GGML was
  already Windows-hardened: the by-pointer `ggml_backend_buffer_i` and
  never-destroyed-teardown-singleton patches exist precisely because passing
  that POD by value, and running those destructors at exit, crashed on
  Windows/MinGW.
- Added the **Vulkan GPU backend**, vendored from the same
  pinned official GGML source as the CPU core through `tools/vendor-ggml`. It
  is **opt-in at build time**, never auto-detected:

  ```
  install.packages("Rggml", configure.args = "--with-vulkan")
  R CMD INSTALL --configure-args=--with-vulkan .
  ```

  Auto-detection would be hostile: GGML embeds 156 compiled SPIR-V shaders as
  C++ translation units, and the largest (`mul_mm`, a 141 MB array literal)
  needs ~5 GB of RAM to compile - independent of `-O` level, since the cost is
  parsing the literal. `configure` probes `glslc`'s optional shader features
  (coopmat, coopmat2, integer_dot, bfloat16), builds the shader generator, and
  compiles the shader set those features imply; it errors out clearly if
  `--with-vulkan` is given without `glslc`/Vulkan headers. Without the flag,
  nothing about the build changes. New C-callables:
  `Rggml_backend_vulkan_init`, `Rggml_backend_vulkan_device_count`, and
  `Rggml_backend_vulkan_device_description`; they report zero devices instead
  of failing when the backend was not built, so callers can probe and fall
  back. R-level: `rggml_vulkan_info()`, `rggml_has_vulkan()`.
- Added the **device-buffer residency C-callables**:
  `Rggml_backend_alloc_ctx_tensors`, `Rggml_backend_buffer_free`,
  `Rggml_backend_tensor_set`, `Rggml_backend_tensor_get`. The CPU and BLAS
  backends compute on host memory, so pointing a tensor at an R buffer works;
  a GPU backend's tensors must live in device memory. These wrap GGML's
  backend-agnostic path - allocate a no_alloc context's tensors into one
  backend buffer, upload, compute, download - which is identical for CPU, BLAS
  Vulkan and CUDA, and is what `test_vulkan.R` uses to assert the GPU backend
  computes the same product as the CPU one.
- Added **view and copy C-callables**: `Rggml_view_1d`/
  `_2d`/`_3d` (byte offsets/strides, as in ggml) and `Rggml_cpy` - the
  building blocks of a KV cache: cpy nodes write new keys/values into views
  of a persistent cache tensor, expanded into the graph ahead of the
  attention nodes that read other views of the same cache (see Rllm's
  incremental decoding).
- Added the transformer **graph-op C-callables**: the ops a
  forward pass composes - `Rggml_get_rows`, `Rggml_rms_norm`, `Rggml_mul`,
  `Rggml_add`, `Rggml_silu`, `Rggml_scale`, `Rggml_soft_max`,
  `Rggml_diag_mask_inf`, `Rggml_rope` (wrapping `ggml_rope_ext` with YaRN
  off), and the shape ops `Rggml_reshape_2d`/`_3d`, `Rggml_permute`,
  `Rggml_cont`, `Rggml_transpose`. Downstream packages can now assemble and
  compute full transformer graphs (see Rllm's llama forward pass) without
  linking GGML.
- Added quantization C-callables: `Rggml_quantize` wraps
  `ggml_quantize_chunk()` so downstream packages can encode f32 rows into any
  GGUF block format (the output is byte-compatible with GGUF tensor payloads),
  and `Rggml_dequantize` wraps GGML's type-traits `to_float` - the
  authoritative reference dequantizer, used by the Rfmalloc ecosystem to
  cross-validate its codec decoders (this cross-validation caught and pinned a
  Q4_K decode bug in Rgguf's vendored gguflib; see that package's NEWS).
- Verified the full quantized-weight compute path over an external payload:
  a `Q4_K` tensor pointed zero-copy at a heap buffer standing in for an
  mmap'd GGUF payload, multiplied by dense F32 activations via
  `ggml_mul_mat()` on the CPU backend, routes each weight row through the
  runtime-SIMD-dispatched `vec_dot` (activations quantized to `Q8_K` on the
  fly, as at llama.cpp inference) and tracks the true product to q4_K
  accuracy (`test_mul_mat_q4k.R`).
- Initial release. Rggml is a carrier package: it vendors a CPU-only,
  architecture-generic build of the 'GGML' tensor library as a static
  library (`inst/ggml/libggml.a`), installs its headers, and exposes GGML's
  tensor-context/matrix-multiply compute path through `R_RegisterCCallable`
  C-callable entry points, declared for downstream use in
  `inst/include/Rggml.h`.
- Registered C-callables: context lifecycle (`Rggml_context_create`,
  `Rggml_context_free`, plus `Rggml_used_mem`/`Rggml_tensor_overhead`/
  `Rggml_graph_overhead` for sizing), tensor creation with an optional
  zero-copy external data pointer (`Rggml_new_tensor`, `Rggml_tensor_data`,
  `Rggml_tensor_set_data`, `Rggml_tensor_ne`, `Rggml_tensor_nb`), the CPU
  backend (`Rggml_backend_cpu_init`, `Rggml_backend_free`,
  `Rggml_backend_graph_compute`), graph building (`Rggml_new_graph`,
  `Rggml_build_forward_expand`), matrix multiply (`Rggml_mul_mat`, and the
  canned single-op `Rggml_compute_mul_mat` that also copies the F32 result
  into a caller-provided `float*`/`double*` buffer), and type/size
  introspection (`Rggml_type_size`, `Rggml_row_size`, `Rggml_blck_size`,
  `Rggml_nbytes`, `Rggml_nelements`, `Rggml_type_name`).
- Added runtime SIMD dispatch for `q4_K x q8_K`. On x86, `configure` compiles
  GGML's official `arch/x86/quants.c` translation unit under private names and
  a CPUID dispatcher selects its Q4_K dot only when the complete upstream AVX2
  feature set is present. Aarch64 and wasm compile GGML's complete official
  architecture sources directly. ISA flags stay out of R's recorded package
  flags, and the portable GGML implementation remains the fallback.
- Enabled GGML's **BLAS backend** (`Rggml_backend_blas_init`,
  `Rggml_backend_blas_set_n_threads`): dense F32 `mul_mat` offloads to
  whatever BLAS the R build links against, since BLAS is universal in R.
  GGML's backend calls the C `cblas_sgemm`, which R does not provide, so
  Rggml bridges it with a small portable shim (`inst/ggml/cblas.h` +
  `rggml_cblas.c`) forwarding to the Fortran BLAS via `F77_NAME()`, linking
  `$(BLAS_LIBS) $(FLIBS)`. The BLAS backend is a drop-in `backend` for the
  existing `Rggml_compute_mul_mat`/`Rggml_backend_graph_compute` path.
- Added `ggml_version()`, returning the vendored GGML library's own runtime
  version string.
- Verified and documented the `ggml_mul_mat()` convention: loading two R
  matrices directly into GGML tensors with `ne = dim(matrix)` (the raw
  column-major bytes, no copy or transpose), `ggml_mul_mat(ctx, A, B)`
  produces a result tensor of dim `(ncol(A), ncol(B))` equal to
  `crossprod(A, B)` (`t(A) %*% B`). Covered by the tinytest suite with both
  hand-checkable small cases and larger random matrices, through both the
  ggml-managed and the zero-copy (externally-owned buffer) tensor creation
  paths, exercised via the registered C-callables (not the internal
  implementation functions directly).
- CPU kernels are architecture-generic (no `-march` SIMD flags, no OpenMP,
  no hand-written x86 SIMD kernel files) for CRAN-facing build portability,
  plus the BLAS backend above for dense products. No Vulkan/CUDA/Metal GPU
  backends. Runtime-SIMD-dispatched quantized kernels
  (`GGML_CPU_ALL_VARIANTS`) and a Vulkan backend are planned; see README.md.
