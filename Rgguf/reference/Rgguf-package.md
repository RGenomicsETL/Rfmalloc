# Rgguf: Official GGUF parsing and typed storage views

Rgguf reads 'GGUF' model files, the file format used by the 'llama.cpp'
project to store large language model tensors and metadata, and exposes
their tensors as decoded Rfmalloc arrays, owned native-codec copies, or
borrowed read-only views. Parsing, writing, and quantized decoding use
the official GGUF and type-traits implementation carried by the sibling
Rggml package.

## Main Functions

- [`gguf_open`](https://sounkou-bioinfo.github.io/Rfmalloc/Rgguf/reference/gguf_open.md):

  Open a GGUF file and return a context handle.

- [`gguf_metadata`](https://sounkou-bioinfo.github.io/Rfmalloc/Rgguf/reference/gguf_metadata.md):

  Read all metadata key-value pairs.

- [`gguf_tensors`](https://sounkou-bioinfo.github.io/Rfmalloc/Rgguf/reference/gguf_tensors.md):

  List the tensor directory as a data frame.

- [`gguf_tensor`](https://sounkou-bioinfo.github.io/Rfmalloc/Rgguf/reference/gguf_tensor.md):

  Read, copy, or borrow one tensor.

- [`gguf_import`](https://sounkou-bioinfo.github.io/Rfmalloc/Rgguf/reference/gguf_import.md):

  Apply the same storage choice to several tensors.

- [`gguf_write_tensors`](https://sounkou-bioinfo.github.io/Rfmalloc/Rgguf/reference/gguf_write_tensors.md):

  Write a minimal GGUF file (F32 tensors), mainly for round-trip
  testing.

## Design

Allocation happens on the R side: tensor destinations are created with
[`Rfmalloc::create_fmalloc_matrix()`](https://rdrr.io/pkg/Rfmalloc/man/create_fmalloc_matrix.html)/[`create_fmalloc_array()`](https://rdrr.io/pkg/Rfmalloc/man/create_fmalloc_array.html),
which returns a properly classed, file-backed ALTREP object with
Rfmalloc's full `Ops`/matrix-product dispatch already working. Native
code only ever fills that destination in place (dequantizing as it
goes); it never allocates R vectors of tensor size itself.

## Storage

- GGUF files are mapped read-only. Metadata and tensor geometry are
  parsed once by Rggml's official GGUF implementation.

- Numeric import decodes through GGML in bounded chunks into the fmalloc
  destination. Native import copies encoded bytes into owned fmalloc
  storage. View import borrows the original mapped span.

## See also

Useful links:

- <https://github.com/sounkou-bioinfo/Rfmalloc>

- <https://sounkou-bioinfo.github.io/Rfmalloc/Rgguf/>

- Report bugs at <https://github.com/sounkou-bioinfo/Rfmalloc/issues>

## Author

**Maintainer**: Sounkou Mahamane Toure <sounkoutoure@gmail.com>
