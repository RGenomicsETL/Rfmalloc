# Phase synthesis

This repository is an exploration of one idea: computation should close over
typed storage without forcing every representation through a dense R object.
The current synthesis is not a frozen API or a release boundary. It is the
smallest set of abstractions that survived contact with genomics, statistical
genetics, GGUF models, and out-of-core linear algebra.

## The abstraction that remains

There are three independent decisions:

1. A source reader emits bounded records with explicit semantics. A record may
   contain hardcalls, fixed-point dosages, phased haplotypes, packed bits, or
   doubles.
2. A storage destination owns allocation, lifetime, alignment, packing, and
   physical layout. It may own bytes in an fmalloc file or borrow a read-only
   span owned by another mapping.
3. A compute consumer chooses an algorithm appropriate to the semantics. Dense
   and compressed numeric tensors enter matrix contraction. Haplotype HMMs and
   banded LD algorithms use typed accessors instead.

The record-panel context is the boundary between the first two decisions. It
does not erase the difference between compressed and uncompressed data. It
makes that difference a destination choice instead of duplicating every reader
API. The storage-span interface is the dual boundary on the read side: a codec
or backend receives a pointer, byte extent, lifetime owner, and runtime context
without caring whether the bytes are owned by fmalloc or borrowed from GGUF.

The materialization rule is simple: materialize only when the algorithm
requires a different representation, never merely because control crosses a
package boundary.

## What this resolves

- Rggml derives core, CPU, BLAS, GGUF, Vulkan and CUDA from one pinned official
  GGML v0.16.0 tree. The unused ggmlR engine fork and its split translation
  units are gone. Rgguf no longer owns a partial C port of the format; it is
  the R-facing adapter over the one official `gguf.cpp`.
- Rllm weights borrow their exact encoded spans from the original read-only
  GGUF mapping. Loading a 4.9 GB model created 256 views over 4.79 GiB in 34 ms
  and created no persistent fmalloc allocation records. CPU computation uses
  those spans directly. CUDA makes the one materialization the device actually
  requires: a model-owned execution context uploads the codec-native weights
  once and reuses them. Device residency is therefore context state, not a
  second storage format and not a compressed-versus-uncompressed API split.
- Rpgen keeps one persistent PGEN or BED reader and transfers bounded panels
  into hardcall, dosage, phased-haplotype, or dense destinations. PED/MAP,
  TPED/TFAM, BGEN, VCF/BCF, GEN, HAPS/legend, EIGENSTRAT, and legacy dosage
  retain PLINK2's own parsers, but `rpgen_ingest()` redirects every terminal
  `STPgenWriter` append shape to the same record sink. No genotype PGEN is
  serialized and decoded again. The file-producing `rpgen_import_*()` surface
  remains ordinary upstream PGEN output. PED/MAP keeps its bounded
  sample-major transpose scratch because the physical reordering is real;
  HAPS performs a bounded count pass because upstream supplies only a writer
  upper bound. Temporary metadata sidecars are cleaned on exit.
- Phased haplotypes are locus-major with a 64-byte-aligned row per variant.
  `Rfmalloc_haplotypes_data()` exposes the body, dimensions, meaningful row
  bytes and padded stride without decoding. This is the bit order and alignment
  kalis uses for its private `hap_locus` cache. A borrowed-cache method needs
  only a locus pointer table and an owning SEXP; PGEN, phased VCF/BCF and HAPS
  then reach kalis through Rpgen without another importer, integer matrix, or
  packed copy. Arbitrary haplotype subsetting still requires repacking because
  it changes bit positions; locus subsetting can remain pointer-only.
- A backend may decline a product. Correctness then falls back to bounded
  decode plus BLAS. Specialization changes speed, not meaning.

## Bets tested rather than protected

The decimal ALP implementation is not a credible LLM weight format.
On a real 2048 by 2048 Q4_K weight, lossless ALP expanded model-like values to
64.2 bits per value and its scalar decode plus BLAS took 20.75 ms for batch 1.
Native GGML Q4_K used 4.5 bits per value and took 0.30 ms. ALP compressed a
three-decimal control to 7.1 bits per value, so the implementation remains
interesting for analytical decimal data. The LLM hypothesis requires a
binary-float transform such as ALP-RD, a measured SIMD decoder, or a direct
compressed dot kernel. The benchmark is in `experiments/alp_gguf_cpu.R`.

The ds4 comparison exposes a different distinction. Its current SSD mode keeps
non-routed weights resident and uses a memory budget as a cache of complete
routed MoE experts, loading an expert from GGUF on a routing miss and preloading
known-hot experts. That cache policy is the central algorithm, not a property
of the allocator. fmalloc and borrowed mmap views solve storage, lifetime, and
page ownership; they do not schedule a model or predict sparse expert demand.
For a dense sequential model, `fmalloc_storage_advise()` can express prefetch
and release intentions, but a layer scheduler must produce the access sequence.
The fair experiment is therefore workload-specific: routing-aware expert cache
against an equivalent fmalloc cache for sparse MoE, and mmap plus advice against
`pread` double buffering for dense layer streaming, both with controlled cold
page-cache state. Cached view construction is not evidence about SSD
throughput. See [antirez/ds4](https://github.com/antirez/ds4#running-models-larger-than-ram)
for the design being compared.

The CUDA experiment separates residency from reusable execution. Keeping
codec-native weights in a model-owned device context is a clear win and does
not alter the storage contract. Keeping GGML's graph allocator in a persistent
scheduler is not: on the RTX 5050, a 12-token prompt followed by 128 greedy
tokens measured 69.7 tok/s on the retained direct path and 63.7 with the
scheduler. CUDA graph capture measured 62.6 because attention extents and graph
properties change at each token. Both experiments were removed. Upstream
`llama-bench` on the same model and rig measures 156.1 tok/s without graphs and
460.1 with them, so the remaining contradiction is graph stability and mutable
KV residency, not another buffer-transfer or storage API.

The real `Q4_K_M` probe also separates cache correctness from invariance to
batch shape. For an identical prefix shape, cached and uncached logits are
bit-identical on CPU and CUDA. CPU is invariant when the same four-token prefix
is evaluated whole or incrementally. CUDA is not: on token ids 1, 5, 9, 2,
prefix-versus-whole cosine similarity ranges from 0.761 to 0.997 and two of
four argmaxes differ. Products at the model's real `q5_0`, `q4_k`, `q8_0` and
`q6_k` shapes remain within 1.25e-4 NMSE of CPU, while CUDA one-column and
four-column products agree exactly. In this probe, cache layout and weight
upload are not the source of the split. The next numerical oracle must compare
Rllm with the matching upstream GGML and llama.cpp graph at each batch shape;
the measurements do not yet assign the graph-level divergence to either side.

## Architecture graphs are programs

The executable object is a bound program: the serializable architecture
AST, its typed storage bindings and a validated GGML lowering. Forward passes,
embedding, CUDA upload and persistent state allocation consume that object.
The C entry point never receives the older model plan and never dispatches on a
model-family name. The LFM2MoE tests mutate the program's routed-expert
semantics and rebind it; the native result changes or fails accordingly. This
is stronger than checking that the AST merely resembles the graph.

R is a useful surface syntax for this language, but arbitrary R closures would
make validation and compilation opaque. The constructors therefore freeze
ordinary modules and pipes into data. Binding validates every declared tensor
name and shape without copying its payload. Compilation validates dataflow,
residual structure, state, normalization and output semantics, then reduces
the accepted program to the native operator vocabulary. Llama, Qwen3.5,
LFM2MoE and EmbeddingGemma all cross that same boundary while retaining their
CPU, CUDA, cache and dense-equation oracles.

The earlier construction duplication is gone. GGUF metadata adapters now
trace the program and declare typed parameters directly. The tensor directory
validates those declarations before any payload is borrowed. No layer plan or
parallel tensor table is allocated while loading a model. `rllm_plan()` lowers
the program into a layer-oriented inspection view only when a caller asks for
one.

The native compiler recognizes a deliberately constrained transformer grammar:
embedding, repeated attention or state-space blocks with two residual joins,
and projection or pooled embedding output. ESM, OpenSpliceAI, TRM and Evo are
useful because they force this grammar to grow through reusable dataflow,
multi-result, convolution and state primitives rather than model-name
exceptions.

[Rtinycc](https://github.com/sounkou-bioinfo/Rtinycc) is a plausible lowering
target for the program, not the language itself. It can turn a declarative
recipe into C, compile and relocate it in memory, retain the live compiler
state with the callable, and recompile from the recipe after serialization. A
validated architecture AST could therefore emit one C graph-builder function
rather than interpret one R call per operator. The generated function should
call a narrow, opaque Rllm operator vtable instead of depending on GGML's
entire header surface. TinyCC does not need to optimize tensor arithmetic: the
generated code only assembles the graph, while official GGML kernels still
perform the work.

The R compiler and native operator builder remain the oracle and
fallback; generated C would be a cache derived from the AST, never another
source of truth.

The ESM-2 8M numerical proof is closed at the dense semantic boundary. The
official Facebook safetensors checkpoint converts through Rgguf to 106
unmodified F32 tensors in GGUF.
Its adapter emits a 69-node program directly. A fixed protein crosses the
second typed input, token dropout, padding semantics, six multi-result rotary
attention blocks, representation and attention taps, tied projection and the
contact head. Selected logits, representations, attention probabilities and
contacts agree with official execution within 0.003, 0.0002, 0.00003 and
0.0003. The reference interpreter learned reusable semantic operators; it did
not acquire an ESM executor branch. The native GGML compiler still rejects the
two-input grammar explicitly, so native ESM execution remains a compilation
proof rather than a semantic uncertainty.

OpenSpliceAI closes a second biological-model proof at the same dense boundary.
Its upstream MANE 80 nt rs10 checkpoint becomes 56 unchanged F32 tensors
through Rgguf. A 36-node program records four residual dilated-convolution
blocks, inference batch normalization, leaky ReLU, accumulated skip
projections, the context crop and per-position softmax. Its complete
16-position output agrees with the pinned upstream PyTorch model within 5e-7.
The reference vocabulary gained reusable one-dimensional convolution and
normalization operators, not an OpenSpliceAI executor. An internal constrained
native F32 dataflow lowering consumes that same bound program and its named
`[channel, sequence, batch]` input, with no C model-family branch. Its retained
fixed-shape context owns one CPU backend, no-allocation graph context, backend
buffer, input and output staging, while its external-pointer owner retains CPU
mapped spans. CUDA continues to borrow the model-owned weight backend and uses
the official F32 im2col/mul_mat graph because a GGML custom CPU callback is not
a CUDA operation. Fixed batch-normalization affine values upload once; a timed
pass copies only input and output.

The retained CPU convolution plan copies a validated F32 `[K, IC, OC]` span at
context construction and repacks it as contiguous `[K * IC, OC]` output-channel
blocks. A work item is one batch element and a block of output positions. It
gathers each input channel's window once, applies whatever the convolution
absorbed to it once, and reads every tap as an overlapping slice of that
window, so padding, stride and dilation leave the inner loop. The portable
callback is the oracle. On x86 an AVX2/FMA object staged outside R's recorded
flags keeps twelve accumulators, three input vectors and one broadcast live,
which is exactly the sixteen registers, so each gathered vector feeds four
fused multiply-adds rather than one. The plan writes the AST layout directly,
so no permutation and no materializing copy follow a convolution. ARM, wasm and
unsupported x86 CPUs use the portable form.

Three things then mattered more than the kernel. Workers claim output tiles
from a shared counter rather than taking a fixed share, because a performance
core retires this kernel about 2.2 times as fast as an efficiency core and a
static split made every node wait on the slowest thread. Upstream GGML computes
leaky ReLU on thread zero alone, which stalled every other worker at the next
barrier for a fifth of the single-thread work, so Rggml gained a worker-split
operator that is bit-identical to it. A convolution that is the only reader of
a normalization, of a rectification, or of both, absorbs them into its gather,
so an OpenSpliceAI residual block emits two nodes where it emitted seven.
Together those took the 400 nt checkpoint at `[4, 416, 128]` from 260.1 ms per
batch to 41.6 ms on the i5-13500.

Correctness is pinned per change, not at the end. The real MANE 80 nt rs10
output stays within 5e-7 of both the dense R oracle and the pinned upstream
values. Converting the other three released MANE checkpoints and running them
against upstream PyTorch on the same input gives 6.0e-8 at 400 nt, 6.0e-8 at
2000 nt and 1.5e-15 at 10000 nt, the last exercising the tap-chunked gather
that a 41-tap kernel at dilation 25 forces. A focused differential test pins
the folded activation against the same operators staged as graph nodes, and it
caught a real defect: a folded affine must not be applied to the convolution's
zero padding, which the crop in every OpenSpliceAI program would have hidden.

The comparison against upstream is on equal resources: both implementations
pinned to the same four idle performance cores of the i5-13500, four threads
each, the minimum of three interleaved runs, on the exact released checkpoints.

| Model | Batch | Rllm ms | Rllm samples/s | PyTorch ms | PyTorch samples/s | Ratio |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 80 nt | 32 | 8.2 | 3,902 | 6.3 | 5,079 | 0.77x |
| 80 nt | 128 | 27.9 | 4,588 | 30.4 | 4,211 | 1.09x |
| 400 nt | 32 | 17.2 | 1,860 | 33.0 | 970 | 1.92x |
| 400 nt | 128 | 76.8 | 1,667 | 123.5 | 1,036 | 1.61x |
| 2000 nt | 16 | 68.0 | 235 | 128.5 | 125 | 1.89x |
| 10000 nt | 4 | 212.0 | 18.9 | 288.5 | 13.9 | 1.36x |

On one pinned performance core at batch 32 the same comparison is 1.06x at
80 nt and 1.72x at 400 nt. A biallelic ref-plus-alt variant consumes two model
samples, so variants/s is mechanically `samples/s / 2`: the 400 nt column above
is 930 and 833 variants/s. The threshold this lowering was held to, that no
performance claim exists until the unchanged program beats the upstream path,
is met everywhere except the smallest checkpoint at small batches, where
per-node launch cost still dominates. The surface stays internal because its R
API is a separate design question, not because of its speed.

After these changes `perf record` at 14 threads and batch 128 attributes 36% of
cycles to the staged AVX2/FMA kernel, 12% to residual addition, 11% to the
worker-split rectification and 9% to the gather; GGML barriers fall out of the
top of the profile. Cached uncompressed FASTA+FAI 181-base slices already
sustain about 318,000 random and 468,000 coordinate-sorted fetches/s.
Annotation lookup, not FASTA format, needs the next source-side change: three
whole-array scans reach about 529 queries/s on 19,305 GRCh38 rows, versus about
708,000/s for a contig-partitioned sorted-start/prefix-max-end point index in
Python.

The device branch of the same lowering ran on the RTX 5050 rig (driver 595.95,
CUDA 13.2, `sm_120a`). It borrows the model-owned weight backend, keeps the
official im2col composition because a custom CPU callback is not a device
operation, and reuses its retained device context. Against the CPU path the
80 nt and 400 nt checkpoints agree to 1.0e-6 and 1.2e-7 with
`NVIDIA_TF32_OVERRIDE=0`. cuBLAS's default TF32 tensor cores widen that to
7.3e-4 for 19% more throughput, which is a trade a caller should make
knowingly rather than inherit.

A constrained vocabulary plus a deterministic validator makes both human and
LLM-authored programs reviewable; the prompt is not the artifact. This is the
useful discipline in [DSLs Enable Reliable Use of LLMs](https://martinfowler.com/articles/llm-and-dsls.html):
the semantic language and its validator become the maintained source of
truth. An optional Rtinycc proof must additionally produce the same graph and
logits as the interpreter, cross into native code once per graph build rather
than once per node, and demonstrate that compiler-state lifetime follows model
lifetime.

## Threat model for this phase

The working environment is a trusted researcher running local code over local
datasets and model files in a controlled build. Inputs may still be truncated,
corrupt, or unexpectedly large, so bounds checks, integer-overflow checks,
mapping lifetime, decoder correctness, and deterministic cleanup are part of
correctness.

This phase does not assume a hostile multi-tenant service, adversarial package
loading, signed model distribution, sandbox escape resistance, or secrecy from
other processes owned by the same user. Work justified only by those deployment
threats is out of scope until such a deployment exists. If untrusted remote
files or a service boundary enter the project, the threat model must be written
again from that boundary.

## The next contradictions to push

1. Lower the ESM and OpenSpliceAI semantic operators through GGML without
   changing either program or numerical oracle. Compare OpenSpliceAI with
   upstream PyTorch on identical production-length windows before making a CPU
   performance claim. Then close full TRM recurrence and Evo's genuinely new
   Hyena FIR/IIR operators.
2. Patch kalis to borrow the aligned haplotype view while preserving its cache
   ownership and SIMD invariants. Compare Forward/Backward output against its
   copied cache and assert that no integer matrix or second packed body exists.
3. Test ALP-RD and SIMD decode as independent storage-bandwidth experiments,
   then decide whether a fused compressed dot deserves to exist.
4. Build the layer access scheduler and compare mmap advice with explicit
   double buffering on cold storage.
5. Establish the upstream CUDA numerical oracle across one-token prefixes and
   equivalent whole batches, then make a decode graph genuinely reusable.
   Test stable or bucketed attention extents and device-resident mutable KV
   state against the host-authoritative cache, preserving explicit
   CPU/CUDA handoff. Persistent graph allocation without stable execution has
   already measured slower and does not deserve an API.
6. Decide whether importer metadata should become its own semantic sink. It is
   transient because compute paths consume genotype records only.

No compatibility shim or API-version ceremony is warranted while every
consumer lives in this monorepo. When an abstraction changes, change all of its
consumers in the same commit and keep only the clearer form.
