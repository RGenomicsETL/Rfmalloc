
# Rggml

<!-- badges: start -->

[![R-CMD-check](https://github.com/sounkou-bioinfo/Rggml/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/sounkou-bioinfo/Rggml/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

Rggml is a low-level **carrier package** for
[GGML](https://github.com/ggml-org/ggml). It vendors the core, CPU
backend, native BLAS bridge, optional Vulkan and CUDA backends, and
official GGUF implementation as one generated static library. It exposes
them through `R_RegisterCCallable()` so sibling packages can build
tensor graphs, inspect GGUF files, and compute without separately
vendoring or linking GGML. Its small R surface provides diagnostics and
direct matrix multiplication; model composition belongs in Rllm and
GGUF’s R-facing storage integration belongs in Rgguf.

``` r
library(Rggml)
ggml_version()
#> [1] "0.16.0"
```

## What’s vendored

A generated, pinned subset of GGML: tensor/context/graph core, the
official `gguf.cpp` reader and writer, tensor-level quantize/dequantize,
CPU kernels, the **BLAS backend**, and the opt-in Vulkan and CUDA
sources. See `inst/COPYRIGHTS` and `tools/vendor-ggml/PROVENANCE.md` for
the exact recipe.

On native R targets, GGML’s BLAS backend (`-DGGML_USE_BLAS`) offloads
F32 `mul_mat` to whatever BLAS the R build links against (reference
`libRblas`, OpenBLAS, MKL, Accelerate, …) - the BLAS interface is
universal in native R, so this needs no extra system dependency. GGML’s
BLAS backend is written against the C `cblas_sgemm` interface, which R
does *not* guarantee (R guarantees only the Fortran `sgemm_`); Rggml
bridges the gap with a small portable shim (`inst/ggml/cblas.h` +
`rggml_cblas.c`) that forwards `cblas_sgemm` to Fortran `sgemm_` via R’s
`F77_NAME()` convention, linking `$(BLAS_LIBS) $(FLIBS)`.

webR omits that bridge because its flang ABI adds hidden
character-length arguments which do not match the native R ABI. Dense
products remain available through GGML’s CPU backend, so this is a
backend capability difference rather than a missing operation. This is
GGML’s supported CPU-only WebAssembly shape, using its wasm SIMD kernels
and requiring no WebGPU backend; see the upstream [WebAssembly
discussion](https://github.com/ggml-org/llama.cpp/discussions/17237).

The quantized dequant-dot path that BLAS cannot cover is
architecture-aware. On aarch64, GGML’s complete NEON kernel set is the
mandatory baseline. wasm uses GGML’s official SIMD128 quant kernels. On
x86, hot kernels are compiled by `configure` into ISA-specific variants,
beginning with `q4_K x q8_K` under `-mavx2 -mfma -O3`; a CPUID
dispatcher (`tools/simd/`) picks the best at runtime and otherwise uses
GGML’s scalar reference. This follows the
[RsimdDispatch](https://github.com/sounkou-bioinfo/RsimdDispatch)
strategy: the ISA flags live in `configure`, never in R’s recorded
package flags, so R CMD check raises no “non-portable flags” NOTE, and a
variant is only ever called after its ISA is confirmed at runtime
(single `.so`, no `dlopen`). On x86 the AVX2 q4_K dot measures ~6-7x
faster than the scalar reference.

(GGML’s own `GGML_CPU_ALL_VARIANTS` is deliberately *not* used: it
requires `GGML_BACKEND_DL` + separate per-variant shared libraries
loaded via `dlopen`, which does not fit a single-`.so` CRAN package.)

The GGML tree is **generated, not a mystery copy**: `tools/vendor-ggml/`
in the monorepo selects 479 files from official GGML v0.16.0 commit
`524f974bb21a`, verifies their paths and contents, applies the small
local patch set, and lets CI derive the tree again. Core, BLAS, GGUF,
CPU, Vulkan and CUDA therefore move as one engine. The only
ggmlR-derived component left is its MIT-licensed R-safe diagnostic shim.
See `inst/COPYRIGHTS` and `tools/vendor-ggml/PROVENANCE.md` for the
complete derivation.

## Vulkan GPU backend (opt-in)

GGML’s Vulkan backend is vendored and builds **on request**, on any
vendor’s GPU (NVIDIA, AMD, Intel):

    install.packages("Rggml", configure.args = "--with-vulkan")
    R CMD INSTALL --configure-args=--with-vulkan .

It needs `glslc` and the Vulkan headers at build time (`libvulkan-dev` +
`glslc` on Debian/Ubuntu; on Windows the LunarG Vulkan SDK with
`VULKAN_SDK` set), and a Vulkan driver at run time.

It is deliberately **not** auto-detected. GGML embeds its shaders as
compiled SPIR-V: `mul_mm.comp` is a 464-line GLSL *template* that the
shader generator compiles into 1,214 variants (every dtype × alignment ×
accumulator × cooperative-matrix combination), emitted as one C array
per variant. The resulting `mul_mm.comp.cpp` is a 141 MB
brace-initializer that costs ~5 GB of RAM to compile - identical at
`-O0`, because the cost is the compiler *parsing* the literal. Enabling
that on any machine that happens to have `glslc` installed would ambush
CRAN builders; without the flag, nothing about the build changes.

`rggml_vulkan_info()` reports the devices GGML can see, and a build
without the backend reports zero devices rather than failing, so callers
can probe and fall back. A GPU backend’s tensors must live in device
memory, which is what the
`Rggml_backend_alloc_ctx_tensors`/`tensor_set`/`tensor_get` C-callables
are for; that path is backend-agnostic, so the same code drives CPU,
BLAS, Vulkan and CUDA. Correctness is pinned against the CPU backend and
can be exercised without a GPU at all through Mesa’s software driver,
since `GGML_VK_ALLOW_CPU=1` opts in to the CPU-type Vulkan devices
upstream refuses.

Metal is not built.

## CUDA GPU backend (opt-in)

The CUDA backend is built from the same pinned GGML source as the core.
It is NVIDIA-only and opt-in because it requires a local CUDA toolkit
and a native `nvcc` build:

    R CMD INSTALL --configure-args="--with-cuda=/usr/local/cuda" packages/Rggml

`CUDA_HOME` supplies the toolkit when no path is passed.
`RGGML_CUDA_ARCH` can pin an architecture such as `sm_89` or `sm_120a`;
otherwise `nvcc` chooses its native default. `rggml_cuda_info()` reports
visible devices, and a build without CUDA reports zero rather than
failing. CUDA uses the same backend-owned allocation, tensor upload,
graph compute and download API as Vulkan. Rllm exercises that path with
a complete transformer graph, not only a standalone matrix product.
Extended Blackwell targets such as `sm_120a` emit both matching PTX and
SASS rather than silently losing the architecture suffix.

cuBLAS decides for itself how to compute an F32 product. On Ampere and
later it uses TF32 tensor cores for `cublasSgemm` by default, which
keeps ten mantissa bits rather than twenty-four. On the RTX 5050 rig one
OpenSpliceAI forward pass differed from the CPU result by 7.3e-4 with
that default and by 1.0e-6 with `NVIDIA_TF32_OVERRIDE=0`, for 19% less
throughput at batch 256 (16,678 against 13,989 samples/s). Set that
variable before the first CUDA call when a device result has to agree
with the CPU reference to F32 accumulation error.

CUDA graph capture is compiled only when `RGGML_CUDA_GRAPHS=1` is set
during configuration. It is off by default because Rllm’s current decode
graph changes its attention extent on every token: on the RTX 5050 rig
capture measured 62.6 tokens/s versus 69.7 for ordinary launches. This
is an experiment switch, not a performance claim. The rig’s full Rggml
suite passes 85 checks and Rllm’s real-model stack passes whole-graph
and cache correctness on the device.

## C-callable API

Declared in the installed header `inst/include/Rggml.h`, registered from
`src/rggml_init.c`, implemented in `src/rggml_api.c`:

``` c
/* upstream identity */
const char    *Rggml_version(void);

/* context lifecycle */
struct ggml_context *Rggml_context_create(size_t mem_size, int no_alloc);
void                  Rggml_context_free(struct ggml_context *ctx);
size_t                Rggml_used_mem(const struct ggml_context *ctx);
size_t                Rggml_tensor_overhead(void);
size_t                Rggml_graph_overhead(size_t size);

/* tensor creation / introspection (zero-copy: pass a non-NULL `data`) */
struct ggml_tensor *Rggml_new_tensor(struct ggml_context *ctx, enum ggml_type type,
                                      int n_dims, const int64_t *ne, void *data);
void    *Rggml_tensor_data(const struct ggml_tensor *tensor);
void     Rggml_tensor_set_data(struct ggml_tensor *tensor, void *data);
int64_t  Rggml_tensor_ne(const struct ggml_tensor *tensor, int dim);
size_t   Rggml_tensor_nb(const struct ggml_tensor *tensor, int dim);

/* CPU backend */
ggml_backend_t Rggml_backend_cpu_init(void);
void            Rggml_backend_free(ggml_backend_t backend);
int             Rggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph *cgraph);

/* BLAS backend (dense F32 mul_mat via R's BLAS; drop-in `backend` above) */
ggml_backend_t Rggml_backend_blas_init(void);
void            Rggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, int n_threads);

/* graph building */
struct ggml_cgraph *Rggml_new_graph(struct ggml_context *ctx, size_t size);
void                 Rggml_build_forward_expand(struct ggml_cgraph *cgraph, struct ggml_tensor *tensor);

/* matrix multiply */
struct ggml_tensor *Rggml_mul_mat(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b);
int Rggml_compute_mul_mat(struct ggml_context *ctx, ggml_backend_t backend,
                           struct ggml_tensor *a, struct ggml_tensor *b,
                           float *out_f32, double *out_f64);

/* quantize / dequantize (GGUF-byte-compatible payloads; dequantize is GGML's
 * authoritative type-traits to_float reference decoder) */
size_t Rggml_quantize(enum ggml_type type, const float *src, void *dst,
                      int64_t nrows, int64_t n_per_row);
int    Rggml_dequantize(enum ggml_type type, const void *src, float *dst, int64_t n);

/* type/size introspection */
size_t       Rggml_type_size(enum ggml_type type);
size_t       Rggml_row_size(enum ggml_type type, int64_t ne);
int64_t      Rggml_blck_size(enum ggml_type type);
size_t       Rggml_nbytes(const struct ggml_tensor *tensor);
int64_t      Rggml_nelements(const struct ggml_tensor *tensor);
const char  *Rggml_type_name(enum ggml_type type);
```

Every one of these has a `<Name>_ptr()` accessor in `Rggml.h` that
resolves it via `R_GetCCallable()` and returns a function pointer, e.g.
`Rggml_context_create_ptr()` returns an `Rggml_context_create_fun`.

### The `ggml_mul_mat()` convention (verified, not just asserted)

If you load an R matrix directly into a GGML tensor with
`ne = c(nrow(M), ncol(M))` pointing at `M`’s own raw column-major memory
(no copy, no transpose - this is exactly the zero-copy path
`Rggml_new_tensor(..., data)` is for), then:

    ggml_mul_mat(ctx, A, B)  ==  crossprod(A, B)  ==  t(A) %*% B

where the result, read back the same way (dim `c(ncol(A), ncol(B))`), is
always `GGML_TYPE_F32` regardless of the input types. This is exercised
in `inst/tinytest/test_mul_mat.R` with hand-computed small cases and
larger random matrices, through both the ggml-managed and the zero-copy
tensor paths, driven entirely through the *registered* C-callables (via
`.Call()` into `src/rggml_test.c`, which itself only uses the
`Rggml.h`/`R_GetCCallable()` path) - the same path a downstream
`LinkingTo` package would use.

## How a downstream package uses it

    # DESCRIPTION
    Imports: Rggml
    LinkingTo: Rggml

``` c
/* your_package_file.c */
#include <Rggml.h>   /* pulls in ggml.h, ggml-alloc.h, ggml-backend.h, ggml-cpu.h too */

void do_something(void) {
    Rggml_context_create_fun ctx_create = Rggml_context_create_ptr();
    Rggml_backend_cpu_init_fun backend_init = Rggml_backend_cpu_init_ptr();
    Rggml_compute_mul_mat_fun compute_mul_mat = Rggml_compute_mul_mat_ptr();
    /* ... build tensors with struct ggml_tensor *, run ggml_mul_mat, ... */
}
```

No GGML re-vendoring, no linking against `Rggml.so` at the linker
level - only `R_GetCCallable()` symbol resolution at run time (standard
R C-callable convention, same as e.g. `Rfmalloc`/`Rgguf`).

## Validation surface

The tinytest suite pins dense BLAS products, every exposed quantized
decoder, Q4_K contraction, runtime SIMD selection and the graceful
absence of optional device backends. CUDA and Vulkan products are
compared with the CPU backend when those devices are present.

The ordinary build targets Linux, macOS, Windows and aarch64. On
Windows, `configure.win` re-executes the shared configure logic with
`RGGML_WINDOWS=1`, writes `src/Makevars.win` without `-ldl`, and locates
an optional Vulkan SDK through `$VULKAN_SDK`. The CUDA correctness and
performance path is exercised separately on the RTX 5050 rig because CI
has no NVIDIA device.
