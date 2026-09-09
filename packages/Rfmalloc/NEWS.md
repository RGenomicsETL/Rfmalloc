# NEWS

- Fixed an incremental build that could link a stale object and still report
  success. Every implementation unit is an `.inc` textually included by
  `fmalloc.cpp`, whose own timestamp never moves, so `configure`'s generated
  `src/Makevars` now declares them as dependencies (behind an explicit
  `all: $(SHLIB)`, so the shared library stays the default goal). A rig install
  had reported DONE while its `Rfmalloc.so` was missing symbols added minutes
  earlier.

- Added `Rfmalloc_ld_alloc()` and `Rfmalloc_ld_write()` to the C-callable API:
  an `ld` store whose band geometry is fixed up front and whose correlations
  are then written one column at a time. A producer whose band does not fit in
  memory no longer has to assemble it in doubles before handing it over; the
  store itself is fmalloc storage. `Rfmalloc_ld_build()` is unchanged and is
  now the all-at-once form of the same thing.

## 0.1.0 (unreleased)

- Added a zero-copy C view over phased-haplotype stores. HMM consumers receive
  the 64-byte-aligned locus body, dimensions, meaningful row bytes, and padded
  stride while the owning R object remains alive. Full integer-matrix
  materialization is an explicit adapter rather than the native compute path.
  Removed the unused API-version counter: all C-callable consumers live in the
  monorepo and change with the contract in the same commit.

- Added borrowed read-only storage views. A typed tensor can now refer to bytes
  already owned by another mapping, keep that owner alive, and enter codec or
  backend compute without a copy into fmalloc. `fmalloc_storage_advise()` adds
  best-effort sequential, prefetch, and release intentions over the same span.

- Added one opaque record-panel transfer context for plain doubles, packed
  hardcalls, fixed-point dosages, and phased haplotypes. Sources now provide
  typed bounded panels while Rfmalloc owns destination allocation, packing,
  alignment, and layout. Compressed versus uncompressed storage is therefore a
  destination choice, not a separate reader API.

- Changed the phased-haplotype payload to locus-major storage: every variant's
  donor haplotypes occupy one contiguous bit row, with the body and each row
  aligned to 64 bytes. This matches the access direction of Li and Stephens
  HMM kernels and kalis Forward/Backward, while materialization still presents
  the conventional variants-by-haplotypes R matrix.

- Added `fmalloc_ld()` and the banded LD-matrix
  accessor API. An `fmalloc_ld` store is a compressed, mmap-backed **banded
  symmetric correlation (LD) matrix**, a SIBLING interface to the matmul tensor
  codec ABI (like the haplotype store), not a decode-to-`f64` matmul codec: an
  LD matrix is read one column's contiguous neighbour run at a time by a Gibbs
  sampler or ridge solve (LDpred2), never multiplied as a dense `p x p` double
  matrix. Because the variants are position-sorted, each column's in-window
  neighbours are a contiguous index range, so the store keeps the **full
  symmetric band per column with no explicit neighbour indices** (the row of a
  column's t-th value is `lo_j + t`), a per-column offset table for O(1) random
  access, and correlations quantized to int8 (`round(r*127)`, resolution
  `~1/127`) or int16 (`round(r*32767)`, resolution `~3e-5`). Build one from
  `(i, j, x)` triplets; read it with `ld_ncol()`, `ld_pair()` and `ld_col()`.
  The new C-callables `Rfmalloc_ld_ncol/bits/pair/col/col_raw/build` (declared
  in `inst/include/Rfmalloc.h`) let a consumer package (RfmallocStatgen's
  `statgen_snp_cor()` / LDpred2) build and read a store without knowing the
  private byte layout. Exercised by `inst/tinytest/test_fmalloc_ld.R`.

- Added `Rfmalloc_tensor_decode(tensor, elem_offset,
  n_elems, out)`, a streaming decode primitive that decodes one block-aligned
  element range of a typed tensor into a caller-owned `double*` buffer using the
  tensor's own codec. This is the read primitive an out-of-core consumer needs
  to pull one variant-column range at a time instead of materializing the whole
  matrix; because standardization is a property of the tensor (a standardized
  `bed`/`dosage` payload decodes centred/scaled and mean-imputed), no
  standardize flag is needed. It returns a status code rather than calling
  `Rf_error`, so a C++ caller can turn a failure into its own control flow with
  no longjmp through its stack. RfmallocStatgen's out-of-core PCA is the first
  consumer. Exercised by `inst/tinytest/test_tensor_decode_range.R`.

- Added `fmalloc_haplotypes()` / `fmalloc_hap_materialize()`: a phased-
  haplotype fmalloc store with a **one-bit-per-call body**, a SIBLING interface to the
  matmul tensor codec ABI rather than an instance of it. Haplotype HMM methods
  (Li and Stephens local-ancestry inference, e.g. the `kalis` package) are not
  linear algebra, so this store never calls `Rfmalloc_register_tensor_codec`
  and never participates in `%*%`; it is a second, independent typed accessor
  over the same fmalloc storage substrate. Materializing back to a `0`/`1`
  matrix decodes into fmalloc-backed storage (not an R-heap copy), and that
  matrix can be handed straight to `kalis::CacheHaplotypes()`: kalis's own
  Forward/Backward output is bit-identical whether it caches from the
  fmalloc-materialized matrix or from the original in-memory matrix (see
  `inst/tinytest/test_fmalloc_hap_kalis.R`). The body is asymptotically 32x
  tighter than an integer `0`/`1` matrix and 64x tighter than doubles, with
  per-locus alignment padding for direct SIMD access.

- Added the `"dosage"` tensor codec via `fmalloc_dosage()`: fractional genotype
  dosages in `[0, 2]` at **one byte each** (fixed point, `round(d * 127)`, with
  255 reserved for missing), the continuous sibling of `"bed"`. Eight times
  tighter than the doubles it decodes to, lossy at resolution `2/254` by design.
  It is the target a PLINK 2 `.pgen` dosage import re-encodes into, since pgen's
  own records are stateful and LD-compressed and so are decoded through pgenlibr
  once at import rather than stored raw. `fmalloc_dosage_standardize()` bakes
  per-variant mean and sd into the tensor in one streaming pass exactly as the
  `"bed"` codec does, so decode returns standardized, mean-imputed dosages (the
  missing code maps to the mean, hence to 0 after centering) with no dosage ever
  materialized as a double; `scale = "sd"` matches `scale()` on the mean-imputed
  matrix to quantization, `scale = "binomial"` uses `sqrt(2 p (1 - p))`. No ABI
  change: like `"bed"`, the stats ride behind a kind byte in the header.

- `fmalloc_bed_standardize()` bakes per-variant standardization into a `"bed"`
  tensor in one streaming pass, so every later decode returns standardized,
  mean-imputed genotypes with no genotype ever materialized as a double and no
  second pass. Decode, mean-imputation (the missing code maps to the variant
  mean, hence to 0 after centering), centering and scaling collapse into one
  four-entry lookup table per variant, built once from the variant's mean and
  sd and applied on the same sweep the matmul's first panel decode already
  takes. `scale = "sd"` matches `scale()` on the mean-imputed matrix to floating
  point (max abs diff 4e-16 on 2000 x 120); `scale = "binomial"` uses
  `sqrt(2 p (1 - p))`, `p = mean/2`, the allele-frequency scaling of GRM /
  SmartPCA / GCTA. Monomorphic variants standardize to 0, not `NaN`. Products
  against the standardized tensor are then centered-and-scaled genotype inputs,
  so a randomized SVD over `tn %*% Omega` and `crossprod(tn, .)` converges to
  the dense standardized SVD - genotype PCA with the genotypes still at 2 bits.

- Added the `"bed"` tensor codec: PLINK 1 genotypes at **2 bits each**, via
  `fmalloc_bed()`. A `.bed` is already the storage we want, because SNP-major
  layout stores each variant contiguously and fmalloc tensors are column-major,
  so a variant *is* a column. Four times tighter than a one-byte-per-genotype
  file-backed matrix, thirty-two times tighter than the doubles it decodes to
  (500k samples x 800k variants: 100 GB, not 3.2 TB). Products against the
  tensor decode bounded column panels and contract them with BLAS, so genotypes
  are never materialized; `X %*% Omega` and `crossprod(X, Y)`, the two kernels a
  randomized SVD needs, both work today.

  No ABI change was needed. PLINK pads every variant to a byte boundary, so a
  flat element index does not map affinely onto a byte offset unless
  `nrow %% 4 == 0`; but the tensor matmul only ever decodes whole columns, so
  `"bed"` registers as a *self-indexing* codec exactly as `"alp"` and `"sparse"`
  already do, and works out its own offsets from a header. The padded case
  (`nrow %% 4 != 0`) is what the round-trip test pins.

- Out-of-core `crossprod(X)` now picks its blocking from the shape, and no
  longer re-reads `X` once per column panel. The old kernel held a column panel
  of `X` resident and swept every other panel against it, so it moved
  `ceil(n/pw) * |X|` bytes while doing a fixed `n^2 * m` flops - and `pw` shrinks
  as `X` gets taller (`pw = tile_bytes / (m * 8)`), so the tall matrices this
  path exists for were the worst case: `m = 1e7`, `n = 2000` and a 256 MB tile
  gives `pw = 3`, i.e. 667 passes over a larger-than-RAM file. Measured at
  constant flops, time tracked the pass count exactly (0.05s to 1.64s as the
  tile shrank) at a flat ~5 GB/s, which is the signature of a kernel bound by
  data movement rather than arithmetic. When the `n x n` result fits the tile
  budget - the documented case for `fmalloc_pca()`, `n` moderate after feature
  selection - it now accumulates `C += Xb'Xb` over row blocks and reads `X`
  exactly once, holding the timing flat at 0.07s. Column panels remain for the
  case they actually win, a Gram matrix too large to keep resident. Row blocking
  preserves the order of the `k` summation, so the Gram is unchanged bit for bit.
- `fmalloc_pca(center = TRUE)` no longer makes a separate pass over `X` for
  `colMeans()`. The Gram sweep already has each block resident, so the column
  sums cost one add per element on a pass that was being paid for anyway; they
  are accumulated in `long double`, as R's own `colSums()` does, and are bitwise
  equal to it. Centering an out-of-core PCA therefore costs one pass over `X`
  instead of `ceil(n/pw) + 1`.

- Fixed native `Ops` edge semantics to match base R for logical `&`/`|` with `NA` and numeric `/` by zero (`Inf`/`NaN` instead of forcing `NA`), and corrected mixed-type native `Ops` regression tests.
- Optimized `[[` on fmalloc ALTREP vectors to bypass subset-copy for scalar extraction, removing a major per-element regression hotspot observed in scalar read loops.
- Added version 2 C API exports for zero-copy native interoperability: `Rfmalloc_is_fmalloc_vector`, `Rfmalloc_vector_type`, `Rfmalloc_vector_length`, and `Rfmalloc_vector_payload_ptr`.
- Added ALTREP regression tests ensuring scalar `[[` no longer returns wrapped fmalloc vectors and that out-of-bounds `[[` signals the expected bounds error.
- Added ALTREP attribute regression coverage for matrix/array/data.frame attribute roundtripping and set minimum R dependency to `R (>= 4.4.0)`.
- Added `create_fmalloc_matrix()` and `create_fmalloc_array()` constructors and `create_fmalloc_data_frame()` plus `as_fmalloc_matrix()`, `as_fmalloc_array()`, and `as_fmalloc_data_frame()` convenience converters for metadata-only reshaping.
- Switched `create_fmalloc_vector()` and matrix/array constructors to length/dimension validation that supports non-negative exact double lengths and long-vector-friendly `R_xlen_t` handling (up to `2^52`) while keeping dimension elements within `integer` limits.
- Added `copy = FALSE` mode to `as_fmalloc_matrix()` and `as_fmalloc_array()` to install shape metadata in-place on fmalloc ALTREP vectors via C-level attribute assignment, with new tests proving this mode avoids additional payload allocations.
- Added S3 class tagging for fmalloc vectors/matrices/arrays and method dispatch for core `Ops`, `Summary`, and matrix reduction families (`rowSums`, `colSums`, `rowMeans`, `colMeans`) so key operators and reductions execute on fmalloc-backed inputs through package-backed handlers.
- Replaced matrix summary/reduction fallback paths with explicit in-package kernels for `range`, `rowSums`, `colSums`, `rowMeans`, and `colMeans` so results match base R edge-case semantics while avoiding unnecessary large intermediate allocations.
- Clarified explicit fallback behavior: `rowSums()`, `colSums()`, `rowMeans()`, and `colMeans()` now warn and delegate to base R when inputs are not exact 2D matrices or `dims != 1L`; `Summary`/`Math`/`Math2` scalar or zero-length results remain ordinary R scalars by design.
- Added a runtime-sharing policy for opened runtimes in-process: opening the same
  backing file path now returns the existing shared runtime (with matching mode) and
  prevents accidental same-file multi-handle mode mismatches.
- Added `diagnose_fmalloc_runtime()` for runtime+catalog diagnostics, including record state counts, payload usage summaries, and an explicit compaction status note to explain why catalog compaction is not yet implemented.
- Implemented nested fmalloc list persistence by reference for persistent runtimes.
- Added recursive fmalloc list/container serialized-reference recovery in ALTREP
  serialization/unserialization.
- Added explicit vector destroy API `destroy_fmalloc_vector()` with mode-aware
  semantics and persistent unsafe reclaim mode.
- Added explicit reference-count failure behavior when destroying a vector that is
  still referenced by another fmalloc list.
- Added initial public R and C-callable API surface for runtime/vector
  introspection: `fmalloc_api_version()`, `fmalloc_default_runtime()`,
  `is_fmalloc_runtime()`, `is_fmalloc_vector()`, `fmalloc_runtime()`,
  `fmalloc_runtime_info()`, `fmalloc_vector_info()`, `fmalloc_vector_type()`,
  `fmalloc_vector_length()`, and `fmalloc_vector_payload_ptr()`, with runtime
  default synchronization for C-callable access.
- Added native C kernel implementations for fmalloc-backed linear algebra
  (`%*%`, `crossprod()`, and `tcrossprod()`), returning managed fmalloc matrix
  outputs with base-consistent shape behavior and name propagation.
- Typed/compressed tensors are now genuinely n-dimensional:
  `create_fmalloc_tensor()`/`as_fmalloc_tensor()` accept dims of any rank and
  `fmalloc_tensor_materialize()` returns an array of that shape (storage and
  decoding are dimension-agnostic). The matrix products (`%*%`, `crossprod()`,
  `tcrossprod()`) require exactly 2 dimensions and error clearly otherwise.
- Added a builtin `"sparse"` tensor codec for mostly-zero data (e.g.
  single-cell counts): `as_fmalloc_tensor(x, dtype = "sparse")` stores only the
  nonzeros of each 1024-element chunk, losslessly, and the resulting tensor
  participates in the panel-streamed matrix products like any other codec. At
  ~10% density it compresses ~6x versus dense `f64` while `%*%`/`crossprod`
  stay exact - the storage layer for out-of-core single-cell/genomics matrices.
- Added a builtin, lossless `"alp"` tensor codec and `as_fmalloc_tensor()`:
  double vectors/matrices are compressed into fmalloc storage as bit-packed
  decimal-scaled integers in independently decodable 1024-value chunks
  (Afroozeh et al., ALP, \doi{10.1145/3626717}; scalar core adapted from the
  MIT-licensed zap implementation - see `inst/COPYRIGHTS`), with exact-value
  patches, a raw escape hatch for incompressible chunks, and
  division-by-exact-power-of-ten decoding so decimal-rounded data
  round-trips with few patches. Compressed tensors participate in the
  panel-streamed matrix products like any other typed tensor.
- Added `fmalloc_matmul_ooc()`, an out-of-core matrix product for fmalloc
  matrices larger than RAM: `A %*% x` consumes `A` one contiguous column tile
  at a time through BLAS `dgemm`, then releases each tile's pages with
  `madvise(MADV_DONTNEED)` (and hints the payload `MADV_SEQUENTIAL`), so the
  resident set stays bounded by the tile budget. Demonstrated on a 62.6 GB
  matrix (equal to total RAM): peak resident memory 0.31 GB during the gemv,
  result exact vs the analytic reference.
- Added a genomics layer: `fmalloc_pca()` (out-of-core truncated PCA via the
  Gram matrix - `crossprod()` + eigendecomposition + projection) and
  `fmalloc_colVars()`/`fmalloc_rowVars()` (variance reductions for
  highly-variable-feature selection). The heavy steps dispatch through the
  pluggable matrix-multiply backend, so single-cell PCA on a larger-than-RAM,
  compressed count matrix runs on CPU BLAS today and on a registered GPU
  backend unchanged - the same call, composable across backends.
- Extended the backend registry with a codec-aware *typed* hook,
  `Rfmalloc_register_matmul_backend_ex`: a backend can register a
  `typed_gemm` that receives a compressed tensor's raw codec payload and dims
  and multiplies it by a dense operand *without* Rfmalloc decoding to f64 -
  enabling native quantized/on-device matmul (e.g. an fmalloc-mmap'd `q4_k`
  payload is byte-compatible with a ggml Q4_K tensor). A typed backend may
  decline a codec, in which case Rfmalloc falls back to the panel-decode path.
- Added a pluggable matrix-multiply backend registry: the matrix-product
  kernels (`%*%`, `crossprod()`, `tcrossprod()`, the out-of-core and
  typed-tensor products) dispatch their `dgemm` through a selectable backend.
  `fmalloc_matmul_backend(name)` selects one (default `"blas"` = R's BLAS) and
  `fmalloc_matmul_backends()` lists registered ones; downstream packages
  register a GEMM kernel (e.g. GPU or out-of-core-aware) through the new
  `Rfmalloc_register_matmul_backend` C-callable. Selection is
  Rfmalloc-scoped (base R's `%*%` is unaffected), and a backend may decline a
  call to fall back to BLAS. Together with the tensor codec registry this makes
  the stack pluggable at both tiers - how bytes decode and what hardware
  multiplies - over fmalloc storage.
- Added `fmalloc_sync()` to flush a persistent runtime's backing store to disk
  (`msync`/`fsync`). Writes to the `MAP_SHARED` mapping (including in-place
  mutations) are otherwise only written back asynchronously by the OS, so a
  crash before writeback loses unsynced data; `fmalloc_sync()` is the explicit
  durability barrier. No-op where `msync` is unavailable (Windows/Rtools).
- Added in-place (by-reference) mutation: `fmalloc_set(x, i, value)` and
  `fmalloc_fill(x, value)` write straight through the backing store, mutating
  by reference regardless of sharing. (An ordinary `x[i] <- value` on an
  *unshared* fmalloc vector already writes in place via the ALTREP data
  pointer; the copy that hurts is R's copy-on-modify when the vector is
  *shared* - these functions bypass it.) The functions are deliberately
  explicit (never a silent `[<-` method) because
  they break value semantics by design: all bindings to the same vector
  observe the change, and for a persistent runtime the durable store is
  updated. Supported for atomic logical/integer/numeric/complex/raw vectors.
  `fmalloc_add()`/`fmalloc_sub()`/`fmalloc_mul()`/`fmalloc_div()` compute
  `x <- x op y` in place (numeric vectors), for the accumulate-into-`x` pattern
  without per-step allocation.
- Added `fmalloc_tcrossprod_ooc()` and out-of-core routing for `tcrossprod(X)`
  (`X X'`): tiles over the reduction dimension (columns of `X`), accumulating
  rank-`kw` updates via `dgemm('N','T')`. Each contiguous column panel is read
  once and evicted, so it is a single streaming pass over `X` (input residency
  ~one panel) with an fmalloc-backed `m x m` result. Single-argument
  `tcrossprod(X)` on a real fmalloc matrix auto-routes above the threshold.
- Added `fmalloc_crossprod_ooc()` and out-of-core routing for `crossprod(X)`:
  the Gram matrix `X'X` is computed as pairs of contiguous column panels of `X`
  through BLAS `dgemm`, writing each block straight into the `n x n` fmalloc
  result and releasing panels with `madvise`. Input residency stays ~two
  panels (so `X` may exceed RAM) and the result is fmalloc-backed (so the
  `n x n` Gram matrix may too) - the covariance/Gram operation behind PCA,
  ridge, and GWAS on larger-than-RAM matrices. Single-argument `crossprod(X)`
  on a real fmalloc matrix auto-routes above `Rfmalloc.ooc_threshold_gb`;
  two-argument and complex cases keep the in-core path.
- `%*%` on an fmalloc matrix now auto-selects the out-of-core path when the
  left operand's payload reaches `getOption("Rfmalloc.ooc_threshold_gb")`
  (default: half of physical RAM, detected portably via POSIX `sysconf` or
  BSD/macOS `sysctl`), tiling with `getOption("Rfmalloc.ooc_tile_mb", 256)`;
  smaller products keep the in-core BLAS path unchanged. Elementwise `Ops` and
  matrix reductions are left as-is: they are already single-pass streaming, so
  forcing page eviction on them would only regress the in-core case.
  `crossprod()`/`tcrossprod()` are not auto-routed because their output can
  itself exceed RAM.
- Typed/compressed tensor matrix products stream out-of-core the same way:
  above the OOC threshold, each column panel's compressed source pages are
  released after decoding (fixed-geometry codecs such as `f16`/`bf16`/`f32`
  and the quantized GGUF formats), so a tensor whose decoded `f64` form
  exceeds RAM multiplies with a bounded resident set. Demonstrated on a matrix
  that is 74.5 GB as `f64` (larger than RAM) stored as 18.6 GB of `f16`:
  matrix-vector product at 0.13 GB peak resident memory, exact vs the analytic
  reference. Adds `rfm_raw_fill_pattern_impl` for building large payloads.
- Added typed fmalloc tensors: `create_fmalloc_tensor()` tags an fmalloc raw
  payload with a dtype codec (builtin `f64`/`f32`/`f16`/`bf16`; other packages
  register codecs through the new `Rfmalloc_register_tensor_codec` C-callable)
  and dims. Matrix products against dense double operands
  decode the payload in bounded block-aligned column panels streamed through
  BLAS `dgemm`, so the full double representation is never materialized;
  `fmalloc_tensor_materialize()` converts to a regular fmalloc matrix.
- Matrix products (`%*%`, `crossprod()`, `tcrossprod()`) now call BLAS `dgemm`
  for finite double operands, falling back to the managed native loops for
  `NA`/`NaN`/`Inf` values and logical/integer/complex operands (the same split
  base R's default matrix product uses).
- Fixed fmalloc vector recognition for R's generic ALTREP wrappers: attribute
  changes on referenced fmalloc vectors of length >= 64 (for example the class
  stripping done by dispatch helpers) made R substitute a wrapper object that
  native code no longer recognized, breaking `%*%`, `crossprod()`, and
  `tcrossprod()` for realistically sized matrices. Native lookup now unwraps
  such wrappers, and the linear algebra methods validate operands without
  stripping classes.
- Fixed list child validation to require the same runtime pointer (not only matching UUID) when storing fmalloc elements in fmalloc list containers.
- Hardened ALTREP subset behavior by only taking the fast native path for strictly
  positive integer indexes; all other index types/values (including `0`, negative,
  fractional `REALSXP`, and mixed modes) now fall back to base R semantics.
- Switched serialized persistent metadata fields (`offset`, `nbytes`, `catalog_offset`,
  `generation`) to fixed-width hex string encoding for exact 64-bit persistence; restoration
  now accepts both new hex and legacy numeric encodings.
- Clarified that full view-based subset support remains experimental (current
  behavior is subset-copy with copy-on-write duplication controls).
- Documented Simon Urbanek custom allocator PoC separately from Rfmalloc's prior
  allocator implementation in README references.
