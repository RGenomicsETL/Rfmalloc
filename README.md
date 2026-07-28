
# Rfmalloc: out-of-core arrays for R

<!-- badges: start -->

[![R-CMD-check](https://github.com/sounkou-bioinfo/Rfmalloc/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/sounkou-bioinfo/Rfmalloc/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

**One repo, one argument: computation should close over typed storage
without forcing every representation through a dense R object.**

Out-of-core computation is not a file format. It is the refusal to treat
RAM as the boundary of an array. Rfmalloc begins with a patched
[fmalloc](https://github.com/yasukata/fmalloc) allocator behind R’s
ALTREP, so a memory-mapped file can be an ordinary R vector whenever an
ordinary pointer is a truthful representation. RAM then stops being a
hard cutoff and becomes one fast level in a spectrum that continues
through the page cache and storage.

That first move exposes its own limit. A PLINK hardcall occupies two
bits, a GGUF weight may occupy four and a half, and a phased haplotype
is not a number waiting to become a double. None of them can honestly
masquerade as a `double*`. The abstraction that survives is therefore
larger than mmap: a storage span carries bytes, extent, ownership and
runtime context; a codec gives those bytes numeric meaning when a
numeric algorithm asks for bounded panels; a typed accessor preserves
non-numeric meaning when it does not; and a compute backend chooses how
to consume the representation. These decisions remain independent.

The independence is enforced by a small but decisive contract. A backend
may decline any product. Rfmalloc then decodes bounded panels and hands
them to BLAS. Specialization may change where bytes live and how quickly
a result arrives, but never whether the result exists. The abstraction
was not imposed in advance. It is what remained when the same
materialization problem appeared in scientific arrays, genotype formats
and quantized language models.

## A pointer is enough until it is not

A cells by genes matrix can live in an fmalloc mapping while R continues
to see a matrix. Variance reduction and PCA stream over it without
constructing a second dense payload, and their products still pass
through the backend registry.

``` r
library(Rfmalloc)

rt <- open_fmalloc(tempfile(fileext = ".bin"), size_gb = 1)
set.seed(1)
factors  <- matrix(rnorm(3000 * 4), 3000, 4)
loadings <- matrix(rnorm(4 * 600), 4, 600)
X <- create_fmalloc_matrix("numeric", nrow = 3000, ncol = 600, runtime = rt)
X[] <- 3 * (factors %*% loadings) + matrix(rnorm(3000 * 600), 3000, 600)

vars <- fmalloc_colVars(X)
length(which(vars > quantile(vars, 0.9)))
#> [1] 60

pca <- fmalloc_pca(X, k = 6)
round(pca$sdev, 2)
#> [1] 77.74 74.67 72.43 66.61  1.44  1.44
dim(pca$x)
#> [1] 3000    6
```

The more revealing case is code that knows nothing about this
repository. `bigPCAcpp` and `bigPLSR` link bigmemory, while
[`pcaone`](https://github.com/Zilong-Li/PCAoneR) takes an
`Eigen::Map<MatrixXd>`. They consume a bare pointer. An fmalloc ALTREP
matrix is such a pointer into an mmap, so the same randomized SVD can
work over a payload larger than the R heap without being rewritten
around a new container.

``` r
library(pcaone)

rt2 <- open_fmalloc(tempfile(fileext = ".bin"), size_gb = 3)
m <- 2e6L
n <- 40L
G <- create_fmalloc_matrix("numeric", nrow = m, ncol = n, runtime = rt2)
set.seed(1)
for (j in seq_len(n)) G[, j] <- rnorm(m)

invisible(gc(reset = TRUE))
sv_mapped <- pcaone(G, k = 5)$d
peak_mapped <- gc()["Vcells", "max used"] * 8 / 2^20

invisible(gc(reset = TRUE))
sv_heap <- pcaone(matrix(G[], m, n), k = 5)$d
peak_heap <- gc()["Vcells", "max used"] * 8 / 2^20

round(c(
  payload_MB = m * n * 8 / 2^20,
  peak_heap_MB = peak_heap,
  peak_mmap_MB = peak_mapped
))
#>   payload_MB peak_heap_MB peak_mmap_MB 
#>          610          697           87
identical(sv_mapped, sv_heap)
#> [1] TRUE
```

The payload never enters the R heap, and `pcaone` never has to know why.
This is the clean pointer case. Compressed storage begins exactly where
that case ends.

``` r
rt3 <- open_fmalloc(tempfile(fileext = ".bin"), size_gb = 1)
set.seed(2)
g <- matrix(sample(c(0L, 1L, 2L), 5e6, replace = TRUE), 5000, 1000)
tn <- fmalloc_bed(g, runtime = rt3)

c(
  bits_per_genotype = 8 * length(unclass(tn)) / length(g),
  vs_double = length(g) * 8 / length(unclass(tn))
)
#> bits_per_genotype         vs_double 
#>          2.000038         31.999386

Omega <- matrix(rnorm(1000 * 5), 1000, 5)
observed <- matrix((tn %*% Omega)[], 5000, 5)
reference <- matrix(as.numeric(g), 5000, 1000) %*% Omega
max(abs(observed - reference))
#> [1] 0
```

No pointer into the encoded payload could have given BLAS the matrix it
expects. The codec instead decodes only the panel needed by the product.
`fmalloc_bed` stores PLINK hardcalls at two bits and `fmalloc_dosage`
stores fractional dosages at one byte. Both can fuse centering, mean
imputation and scaling into the lookup that decodes each value. Missing
calls map directly to zero after centering, so standardization does not
require an intermediate genotype matrix.

Phase and multiple alleles force the next distinction. They are not
merely compressed numbers, so `fmalloc_haplotypes` is deliberately not a
matrix codec. It stores locus-major phased rows at one bit per haplotype
and exposes typed access for Li and Stephens or
[`kalis`](https://kalis.louisaslett.com/)-class HMM kernels. The
physical layout follows the consumer’s traversal instead of pretending
every future algorithm wants column-major doubles. Numeric panels and
haplotype rows are two interfaces over the same storage substrate.

`Rpgen` closes the reader side of this argument. Its native PLINK2
import closure reads PGEN, BED, PED/MAP, TPED/TFAM, BGEN, VCF/BCF, GEN,
HAPS/legend, EIGENSTRAT and legacy dosage. Each parser emits bounded
records into the same destination context, which chooses hardcall,
dosage, phased-haplotype or dense layout. A format is parsed once; it is
not serialized through a temporary PGEN merely to be decoded again.

## A model is another typed array program

GGUF presents the same contradiction with different economics. Its
tensor directory describes weights whose encoded bytes already live in a
model file. Copying them into a second backing store gains nothing.
`Rgguf` uses GGML’s official GGUF implementation, and `Rllm` borrows
each tensor’s exact read-only span from the original mapping. The model
keeps quantized weights quantized; GGML consumes those blocks directly;
only the KV cache and transient graph state require new storage.

``` r
library(Rllm)

local({
  backing <- tempfile(fileext = ".bin")
  rt <- Rfmalloc::open_fmalloc(backing, mode = "scratch", size_gb = 2)
  on.exit({
    Rfmalloc::cleanup_fmalloc(rt)
    unlink(backing)
  }, add = TRUE)

  model_path <- Sys.getenv(
    "RLLM_README_GGUF",
    "LFM2.5-8B-A1B-Q4_K_M.gguf"
  )
  model <- rllm_gguf_model(model_path, runtime = rt)
  gen <- rllm_generate(
    model,
    charToRaw("The capital of France is"),
    n_new = 16L,
    runtime = rt
  )
  rawToChar(gen$raw)
})
```

The evaluated run uses LFM2.5-8B-A1B `Q4_K_M`, not a toy model shaped to
emit a convenient answer. Set `RLLM_README_GGUF` when rendering the Rmd.
If the file is absent, the chunk is not evaluated and no transcript is
invented. Knitr keys its cache by the model file’s size and modification
time.

The I/O boundary is bytes, not a hidden theory of text. The tokenizer is
an edge codec built from GGUF metadata, generation returns raw output,
and `rawToChar()` is the caller’s interpretation. On the recorded
SmolLM2-135M `Q4_K_M` run on the RTX 5050 rig, a 12-token prompt
followed by 128 greedy decode tokens reached a median 40.2 tokens per
second on CPU and 69.7 on CUDA after the one-time weight upload. These
are medians of three complete generations, not a kernel microbenchmark.
The logits are checked against a pure-R reference forward to the
tolerance imposed by the quantized weights, and cached CUDA logits are
checked against both whole-batch CUDA and CPU cache handoffs.

The real `Q4_K_M` probe separates cache correctness from invariance to
batch shape. For an identical prefix shape, cached and uncached logits
are bit-identical on CPU and CUDA. CPU is also invariant when a
four-token prefix is evaluated whole or one token at a time. CUDA is
not: on the structural token sequence 1, 5, 9, 2, prefix-versus-whole
cosine similarity ranges from 0.761 to 0.997 and two of four argmaxes
differ. Individual products at the model’s real `q5_0`, `q4_k`, `q8_0`
and `q6_k` shapes remain within 1.25e-4 NMSE of CPU, while CUDA
one-column and four-column products agree exactly. This rules out cache
layout and weight upload as the source of that split. The graph-level
numerical contract against upstream GGML and llama.cpp across batch
shapes is therefore an explicit frontier, not a hidden correctness
claim.

ALP is a useful counterexample to protecting a beautiful idea from a bad
measurement. The decimal ALP codec is lossless and effective for
analytical values with decimal structure. On a model-like 2048 by 2048
Q4_K weight, however, it expands the values to 64.2 bits each and its
scalar decode plus BLAS takes 20.75 ms for batch one. Native Q4_K
occupies 4.5 bits and its GGML product takes 0.30 ms. The LLM version of
the lossless-compressed bet has to earn its way back through ALP-RD,
SIMD decode or a fused compressed dot kernel. The failed form is
evidence, not an API to preserve.

The graph description follows the same separation. GGUF metadata is
first normalized into named tensor roles, shapes, operators, state and
outputs. An R architecture program then expresses modules, residual
branches, representation taps and structured recurrence with ordinary
functions and the base pipe. Several typed inputs and several named
results stay explicit instead of being smuggled through string
attributes. The frozen program is data only. Rllm binds its declared
parameters directly to mapped storage and compiles its validated block
grammar to the GGML operator vocabulary. C does not receive a second
model plan and does not need another model-family class; it needs
reusable lowerings for attention, normalization, convolution, routing
and whatever operators survive the next model stress test.

This language is intentionally closer to `torch::nn_module()` and luz’s
pipe cadence than to a C++ architecture switch. EmbeddingGemma forces
non-causal and symmetric-window attention, post-branch normalization,
pooling and projection. The complete ESM-2 8M topology forces typed
token and padding inputs, attention maps as second results, arbitrary
representation taps, tied embeddings and a contact head over those maps.
OpenSpliceAI forces residual dilated 1-D convolutions, inference batch
normalization, accumulated skip projections, context cropping and
per-position softmax. The Evo 2 7B topology forces the real 32-layer
schedule: 27 HCS, HCM and HCL cascades with different FIR or IIR state,
plus five interleaved attention layers. Tiny Recursive Models force
shared modules and nested carried state.

`rllm_execute()` interprets the same program’s dataflow and loops
through an explicit operator table. A dense oracle executes the TRM
`z_H` and `z_L` recurrence across its three nested symbolic loops and
agrees with direct R iteration. It is the semantic oracle for the native
bound-program lowering, not a second GGUF runtime. Official ESM-2 and
OpenSpliceAI checkpoints now cross complete dense numerical oracles
through Rgguf; their native GGML lowerings remain explicit work. Evo
still needs a checkpoint importer and numerical lowering, while TRM
still needs its full bidirectional-attention lowering. Those absences
remain named operator failures instead of becoming architecture
switches.

[Rtinycc](https://github.com/sounkou-bioinfo/Rtinycc) gives this
separation a useful experimental path. A data-only program can generate
a small C reference lowering or ABI adapter, compile it in memory and
compare it with the R oracle before any operator is promoted into Rggml.
TinyCC is not thereby declared a tensor compiler, and it does not
replace tuned GGML kernels. It shortens the distance between a semantic
operator and a falsifiable native prototype.

## Compute follows storage

The storage representation does not dictate one machine. Dense products
use R’s BLAS. Quantized CPU products use GGML kernels, with runtime
CPUID dispatch on x86 and the mandatory NEON baseline on aarch64. The
ISA-specific flags are confined to staged objects and never leak into
R’s recorded package flags. Both paths operate over bounded or mapped
storage rather than demanding that the entire problem enter RAM.

Vulkan is the first device backend. It is built explicitly with
`configure.args = "--with-vulkan"` because compiling GGML’s embedded
SPIR-V set is itself expensive. `rggml_vulkan_info()` reports visible
devices, while a build without Vulkan reports none and leaves callers
free to fall back. The same correctness graph can run on a real GPU or,
with `GGML_VK_ALLOW_CPU=1`, through Mesa’s software driver. Device
allocation, tensor upload, graph execution and download pass through
backend-neutral GGML interfaces rather than Vulkan-shaped application
code.

CUDA comes from the exact same pinned GGML source as the core. It is an
opt-in `nvcc` build, and Rllm’s model-owned context uploads codec-native
weights once and reuses them while the host GGUF mapping remains
authoritative. The RTX 5050 rig passes the full Rggml, Rgguf and Rllm
suites, including whole-graph CUDA, plain and fmalloc KV caches, and
CPU-to-CUDA and CUDA-to-CPU cache handoffs.

The remaining gap is measured rather than euphemized. Upstream
`llama-bench` on the same model and rig reaches 156.1 tokens per second
with CUDA graphs disabled and 460.1 with them enabled. This stack
reaches 69.7. Replacing each pass’s activation buffer with GGML’s
persistent backend scheduler reduced it to 63.7, while enabling graph
capture reached 62.6 because the graph and attention extent change on
every token. That machinery was removed. Closing the gap requires
reusable execution, most likely through stable attention shapes and
device-resident mutable cache state; another storage API cannot solve
it.

## One monorepo because the abstractions cross packages

The package boundaries separate bets, not teams.
[Rfmalloc](https://sounkou-bioinfo.github.io/Rfmalloc/Rfmalloc/) owns
mapped storage, typed spans, codecs, backend dispatch, bounded panel
products and eviction.
[Rggml](https://sounkou-bioinfo.github.io/Rfmalloc/Rggml/) is the
compute and format authority: it carries the pinned GGML core, official
GGUF implementation, CPU kernels, BLAS and optional device backends.
[Rgguf](https://sounkou-bioinfo.github.io/Rfmalloc/Rgguf/) turns GGUF
metadata and tensor spans into an R-facing storage layer without
maintaining another parser.
[Rllm](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/) composes those
pieces into quantized products, semantic architecture programs,
persistent model state, embeddings and byte-level generation.

The statistical genetics side pushes the same contracts in directions an
LLM does not.
[Rpgen](https://github.com/sounkou-bioinfo/Rfmalloc/tree/main/packages/Rpgen)
owns the native format-reader closure and bounded record transfer.
[RfmallocStatgen](https://github.com/sounkou-bioinfo/Rfmalloc/tree/main/packages/RfmallocStatgen)
owns streamed regression and PCA, banded LD, LDpred2 and matrix-form
colocalisation. When one experiment changes a shared contract, every
consumer changes in the same commit. There is no external consumer that
justifies compatibility theatre while the abstraction is still being
discovered.

This is also why the repository keeps `experiments/` beside `packages/`.
Experiments are falsifiable probes of storage and compute bets; `tests/`
contains the cross-package consequences; `tools/` records how
third-party sources and fixtures are reproduced. Synthesis is allowed to
delete a failed materialization or merge two accidental interfaces. The
monorepo makes that movement visible and testable.

## Correctness is the invariant

Every quantized decoder is checked bit-for-bit against GGML’s own
`to_float` reference through fixtures in Rgguf and live cross-validation
in Rllm. That discipline caught a real Q4_K dequantization bug in the
earlier C port. The inference graphs are checked against pure-R
forwards; hermetic f32 state must equal whole-batch execution at every
position. A real quantized MoE adds a discontinuous top-k router after
batch-width-dependent kernels, so its real-file invariant is the
selected token, logit direction and upstream continuation rather than
bitwise equality across GEMV and GEMM geometries. Lossless codecs must
round-trip exactly, and every specialized backend must be free to
decline into the decode-and-BLAS reference path. Performance claims come
from measured paths, not from the presence of a backend name in a build.

## Install and provenance

The packages are available from
[r-universe](https://sounkou-bioinfo.r-universe.dev). Installing `Rllm`
pulls the composed CPU stack.

``` r
install.packages(
  "Rllm",
  repos = c("https://sounkou-bioinfo.r-universe.dev", getOption("repos"))
)
```

A package can also be installed directly from its monorepo subdirectory,
for example `pak::pak("sounkou-bioinfo/Rfmalloc/packages/Rllm")`. Linux,
macOS, Windows, Apple Silicon and Linux aarch64 remain part of the
ordinary test surface. Platform shims exist where the operating system
really differs, such as `CreateFileMapping` in Rgguf on Windows, but the
storage and compute contracts remain shared.

The vendored GGML core is generated from pinned sources by
`tools/vendor-ggml/vendorggml.R`; it is never edited in place. GGML,
fmalloc, pgenlib, PCAone and ALP retain their upstream licenses and
provenance in each package’s `inst/COPYRIGHTS`. The runtime SIMD staging
pattern is developed separately in
[RsimdDispatch](https://github.com/sounkou-bioinfo/RsimdDispatch) and
used here where it survives contact with the larger stack.

## License

The stack is GPL (\>= 2). Rpgen is GPL-3 because it combines PLINK2’s
GPL-3 import closure with LGPL (\>= 3) pgenlib. RfmallocStatgen is GPL-3
because its pinned PCAone workhorse is GPL-3. Vendored components keep
their own MIT, BSD, LGPL or GPL terms as recorded in the package
copyright files.
