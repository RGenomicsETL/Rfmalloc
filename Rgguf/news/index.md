# Changelog

## Rgguf 0.1.0 (unreleased)

- Named each sibling package explicitly in `Remotes`, so dependency
  installers distinguish monorepo subdirectories which share one
  repository commit.

- Registered official GGML `q2_0` group-64 ternary blocks as an Rfmalloc
  storage codec. Legacy group-128 files which reuse the same numeric
  type id remain unsupported rather than being decoded with the wrong
  geometry.

- Added `gguf_tensor(as = "view")` and `gguf_import(as = "view")`. A
  view borrows the tensor’s exact read-only span in the original GGUF
  mapping, keeps that mapping alive, and enters Rfmalloc/GGML compute
  without copying the encoded payload into a second file.

- Removed the partial vendored `gguflib` parser. Rggml now carries
  GGML’s official `gguf.cpp`; Rgguf uses its indexed metadata, tensor
  directory, writer, and type-traits decoders, while mapping tensor
  bytes read-only. Numeric decode uses bounded scratch, and the ten
  common quantized codecs are registered here rather than split between
  Rgguf and Rllm.

- `gguf_tensor(as = "native")` copies the raw encoded payload and only
  needs a registered Rfmalloc codec. `as = "numeric"` decodes through
  GGML’s reference implementation.

- Removed the inherited Q4_K scale-indexing bug by deleting the
  independent decoder. Codec outputs are pinned to GGML by
  `test_gguf_codec_ggml_ref.R`.

- Added native typed imports: `gguf_tensor(as = "native")` (and
  `gguf_import(as = "native")`) copy a 2-d tensor’s raw GGUF payload
  into fmalloc storage at its original density (e.g. 4.5 bits/weight for
  `q4_k`) and return an
  [`Rfmalloc::fmalloc_tensor`](https://rdrr.io/pkg/Rfmalloc/man/fmalloc_tensor.html),
  decoded in bounded panels only inside matrix products. Registers
  GGML-backed codecs for `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q2_k`,
  `q3_k`, `q4_k`, `q5_k`, and `q6_k`.

- Initial release.

- Added
  [`gguf_open()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rgguf/reference/gguf_open.md)
  to parse and map a GGUF file read-only and return a `gguf_ctx` handle
  with a closing finalizer.

- Added
  [`gguf_metadata()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rgguf/reference/gguf_metadata.md)
  to read all metadata key-value pairs (scalars and arrays of the
  supported GGUF value types) into a named R list.

- Added
  [`gguf_tensors()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rgguf/reference/gguf_tensors.md)
  to list the tensor directory (name, type, number of dimensions,
  dimensions, element count, byte size, file offset) as a data frame,
  without reading any tensor payload.

- Added
  [`gguf_tensor()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rgguf/reference/gguf_tensor.md)
  to read and dequantize a single named tensor directly into an
  `Rfmalloc`-backed, file-backed ALTREP matrix/array of doubles: R
  allocates the destination via
  [`Rfmalloc::create_fmalloc_matrix()`](https://rdrr.io/pkg/Rfmalloc/man/create_fmalloc_matrix.html)/[`create_fmalloc_array()`](https://rdrr.io/pkg/Rfmalloc/man/create_fmalloc_array.html),
  and native code fills it in place, so no ordinary R-sized copy of the
  tensor is ever materialized on the R heap.

- Added
  [`gguf_import()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rgguf/reference/gguf_import.md)
  to read some or all tensors from a file into a named list of
  `Rfmalloc`-backed matrices/arrays sharing one open file handle.

- Added
  [`gguf_write_tensors()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rgguf/reference/gguf_write_tensors.md),
  a minimal GGUF writer (F32 tensors plus simple string/numeric
  metadata), primarily to build small GGUF fixtures for this package’s
  own round-trip tests without shipping binary fixtures.

- Verified (with a dedicated round-trip test) that GGUF’s `dim[0]`
  (fastest-varying/contiguous dimension) maps directly onto R’s first,
  column-major dimension, so tensor dimensions need no transposition:
  `dim(gguf_tensor(x, name)) == c(dim[0], dim[1], ...)`.

- Numeric decoding uses GGML’s storage-type implementation for plain and
  quantized tensors.

- Documented (README, `inst/smoke_test.R`) a discovered `Rfmalloc` 0.1.0
  limitation, not an `Rgguf` bug: `%*%`/`crossprod`/`tcrossprod` on
  fmalloc-backed operands with \>= 64 elements currently fail once the
  operand has passed through any R closure call (effectively always),
  due to a generic R-core ALTREP attribute-wrapping behavior that
  `Rfmalloc` does not unwrap.
  [`gguf_tensor()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rgguf/reference/gguf_tensor.md)/[`gguf_import()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rgguf/reference/gguf_import.md)
  themselves are unaffected at any size.
