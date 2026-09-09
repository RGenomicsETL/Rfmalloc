#ifndef RFMALLOC_API_H
#define RFMALLOC_API_H

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Rfmalloc C-callable interface.
 *
 * These functions are resolved with R_GetCCallable(). Packages should import
 * Rfmalloc at runtime before calling them, for example by listing Rfmalloc in
 * LinkingTo and Imports. Returned SEXP objects follow normal R API ownership:
 * protect them if allocating further R objects, and preserve them if they must
 * outlive the current call.
 */

typedef SEXP (*Rfmalloc_default_runtime_fun)(void);
typedef SEXP (*Rfmalloc_open_runtime_fun)(const char *filepath, double size_gb,
                                          const char *mode);
typedef SEXP (*Rfmalloc_create_vector_fun)(SEXP runtime, int sexptype,
                                           R_xlen_t length);
typedef SEXP (*Rfmalloc_create_vector_ex_fun)(SEXP runtime, int sexptype,
                                              R_xlen_t length,
                                              int zero_initialize);
typedef void (*Rfmalloc_cleanup_runtime_fun)(SEXP runtime);
typedef SEXP (*Rfmalloc_set_default_runtime_fun)(SEXP runtime);
typedef SEXP (*Rfmalloc_list_allocations_fun)(SEXP runtime);
typedef SEXP (*Rfmalloc_is_runtime_fun)(SEXP runtime);
typedef SEXP (*Rfmalloc_is_fmalloc_vector_fun)(SEXP vector);
typedef SEXP (*Rfmalloc_runtime_of_vector_fun)(SEXP vector);
typedef SEXP (*Rfmalloc_runtime_info_fun)(SEXP runtime);
typedef SEXP (*Rfmalloc_vector_type_fun)(SEXP vector);
typedef SEXP (*Rfmalloc_vector_length_fun)(SEXP vector);
typedef SEXP (*Rfmalloc_vector_payload_ptr_fun)(SEXP vector);
typedef SEXP (*Rfmalloc_vector_info_fun)(SEXP vector);
typedef SEXP (*Rfmalloc_destroy_vector_fun)(SEXP vector, int unsafe);

/*
 * Tensor codec registration. A codec decodes a flat,
 * block-aligned element range of a typed tensor payload into doubles.
 * Ranges start on a block boundary and cover a whole number of blocks,
 * except possibly the final range for a payload; the codec must write
 * exactly n_elems doubles.
 */
typedef int (*Rfmalloc_tensor_decode_fn)(const void *payload,
                                         R_xlen_t elem_offset,
                                         R_xlen_t n_elems, double *out);
typedef int (*Rfmalloc_register_tensor_codec_fun)(const char *name,
                                                  unsigned int items_per_block,
                                                  unsigned int bytes_per_block,
                                                  Rfmalloc_tensor_decode_fn decode);

/*
 * Pluggable matrix-multiply backend. Register a GEMM kernel
 * and select it (from R via fmalloc_matmul_backend(name), or another package)
 * so Rfmalloc's matrix products dispatch to it instead of R's BLAS. The
 * kernel computes C = alpha op(A) op(B) + beta C (op = 'N'/'T'), returning 0
 * if it handled the call or non-zero to decline (fall back to BLAS). Names
 * "blas" and the empty string are reserved.
 */
typedef int (*Rfmalloc_gemm_fn)(const char *transa, const char *transb,
                                int m, int n, int k, double alpha,
                                const double *A, int lda,
                                const double *B, int ldb,
                                double beta, double *C, int ldc);
typedef int (*Rfmalloc_register_matmul_backend_fun)(const char *name,
                                                    Rfmalloc_gemm_fn fn);

/*
 * Codec-aware (typed) backend hook. Multiply a compressed
 * tensor by a dense double operand WITHOUT Rfmalloc decoding it to f64 first:
 * the backend receives the raw codec payload (byte-identical to how the codec
 * stored it - e.g. an fmalloc-mmap'd q4_k payload is a valid ggml Q4_K tensor)
 * and the dims, so a device/quantized engine can do native quantized matmul.
 * C = T x D when typed_on_left, else C = D x T. Return 0 if handled, non-zero
 * to decline (Rfmalloc then decodes panels and uses the dense gemm path).
 * Register with Rfmalloc_register_matmul_backend_ex (either fn may be NULL).
 */
typedef int (*Rfmalloc_typed_gemm_fn)(const char *codec,
                                      const void *payload, size_t payload_bytes,
                                      int tensor_nrow, int tensor_ncol,
                                      int typed_on_left,
                                      const double *dense, int dense_nrow, int dense_ncol,
                                      double *C);
typedef int (*Rfmalloc_register_matmul_backend_ex_fun)(const char *name,
                                                       Rfmalloc_gemm_fn fn,
                                                       Rfmalloc_typed_gemm_fn typed_fn);

/*
 * Streaming decode primitive. Decode a flat, block-aligned
 * element range [elem_offset, elem_offset + n_elems) of a typed tensor into a
 * caller-owned double buffer, using the tensor's own registered codec, so an
 * out-of-core consumer can decode one column range at a time instead of
 * materializing the whole matrix. Standardization is a property of the tensor
 * (a standardized 'bed'/'dosage' payload decodes centred/scaled and
 * mean-imputed), so no flag is needed: pass a standardized tensor to stream
 * standardized values. 'tensor' is the fmalloc tensor object (the ALTREP
 * payload with its 'rfm_dtype' and 'dim' attributes). Returns 0 on success and
 * non-zero on any error; it never calls Rf_error.
 */
typedef int (*Rfmalloc_tensor_decode_range_fun)(SEXP tensor, R_xlen_t elem_offset,
                                                R_xlen_t n_elems, double *out);

/* Record-panel transfer context. Sources provide semantic records while
 * Rfmalloc owns allocation, packing, alignment, and persistent layout. This
 * is the same interface for plain f64, packed hardcalls, fixed-point dosage,
 * and locus-major haplotype bits. */
struct Rfmalloc_buffer_context;
enum Rfmalloc_buffer_source_type {
    RFMALLOC_BUFFER_I32 = 1,
    RFMALLOC_BUFFER_F64 = 2,
    RFMALLOC_BUFFER_PACKED_BITS = 3
};
typedef struct Rfmalloc_buffer_context *(*Rfmalloc_buffer_open_fun)(
    SEXP runtime, const char *storage, R_xlen_t n_item, R_xlen_t n_record);
typedef int (*Rfmalloc_buffer_write_fun)(
    struct Rfmalloc_buffer_context *ctx, R_xlen_t first_record,
    R_xlen_t n_record, int source_type, const void *data,
    size_t record_stride);
typedef SEXP (*Rfmalloc_buffer_finish_fun)(struct Rfmalloc_buffer_context *ctx);
typedef void (*Rfmalloc_buffer_abort_fun)(struct Rfmalloc_buffer_context *ctx);

/* Direct read-only view of the locus-major phased-haplotype store. The caller
 * must keep `store` reachable while using `data`. Row `locus` begins at
 * data + locus * stride and carries n_haplotype bits, least-significant bit
 * first within each byte. row_bytes excludes zero padding, while stride
 * includes it. The body and every row are 64-byte aligned, so a SIMD HMM
 * cache can borrow them without repacking. */
struct Rfmalloc_haplotype_view {
    const uint8_t *data;
    R_xlen_t n_locus;
    R_xlen_t n_haplotype;
    size_t row_bytes;
    size_t stride;
};
typedef int (*Rfmalloc_haplotypes_data_fun)(
    SEXP store, struct Rfmalloc_haplotype_view *view);

/* Borrow bytes already owned by another mapped container. `owner` remains
 * reachable for the lifetime of the returned external pointer; `runtime` is
 * where Rfmalloc allocates decoded results. The view is read-only and does not
 * copy or free `data`. Rfmalloc_storage_data() also resolves ordinary raw and
 * fmalloc raw vectors, so compute packages need one payload accessor. */
typedef SEXP (*Rfmalloc_storage_view_fun)(SEXP owner, SEXP runtime,
                                          const void *data, size_t nbytes);
typedef int (*Rfmalloc_storage_data_fun)(SEXP object, const void **data,
                                         size_t *nbytes, SEXP *runtime);
enum Rfmalloc_storage_advice {
    RFMALLOC_STORAGE_SEQUENTIAL = 0,
    RFMALLOC_STORAGE_DONTNEED = 1,
    RFMALLOC_STORAGE_WILLNEED = 2
};
typedef int (*Rfmalloc_storage_advise_fun)(SEXP object, size_t offset,
                                           size_t nbytes, int advice);

/*
 * Banded LD-matrix accessor API. An "ld" store is a compressed,
 * mmap-backed banded symmetric correlation matrix built by fmalloc_ld() /
 * RfmallocStatgen's statgen_snp_cor(); it is a typed accessor sibling of the
 * tensor codec (like the haplotype store), not a decode-to-f64 matmul codec.
 * Consumers (LDpred2) read one column's contiguous neighbour run at a time.
 *   - Rfmalloc_ld_ncol: number of variants (columns == rows), or -1.
 *   - Rfmalloc_ld_bits: quantization width, 8 or 16, or -1.
 *   - Rfmalloc_ld_pair: r[i, j] (0-based), 0.0 outside the band.
 *   - Rfmalloc_ld_col: decode column j's band into out (>= *len doubles),
 *     setting *lo (band start row) and *len (band length); returns 0/-1.
 *   - Rfmalloc_ld_col_raw: zero-copy pointer to column j's raw int8/int16 codes
 *     (decode a code c as c / (bits==16 ? 32767 : 127)); returns 0/-1.
 *   - Rfmalloc_ld_build: build an ld store from computed per-column bands (lo,
 *     len 0-based, rvals the column-major concatenation of band correlations);
 *     returns the ALTREP raw payload SEXP - PROTECT it immediately.
 *   - Rfmalloc_ld_alloc / Rfmalloc_ld_write: the same store built one column at
 *     a time. Alloc fixes the band geometry and zeroes every correlation;
 *     write quantizes `count` values into column j starting `first` entries
 *     into its band. A producer whose band does not fit in memory uses these
 *     instead of Rfmalloc_ld_build, and only its in-flight columns are
 *     resident: the store itself is fmalloc storage.
 */
typedef R_xlen_t (*Rfmalloc_ld_ncol_fun)(SEXP store);
typedef int (*Rfmalloc_ld_bits_fun)(SEXP store);
typedef double (*Rfmalloc_ld_pair_fun)(SEXP store, R_xlen_t i, R_xlen_t j);
typedef int (*Rfmalloc_ld_col_fun)(SEXP store, R_xlen_t j, R_xlen_t *lo,
                                   R_xlen_t *len, double *out);
typedef int (*Rfmalloc_ld_col_raw_fun)(SEXP store, R_xlen_t j, R_xlen_t *lo,
                                       R_xlen_t *len, const void **values);
typedef SEXP (*Rfmalloc_ld_build_fun)(SEXP runtime, R_xlen_t n_variants, int bits,
                                      int window, const R_xlen_t *lo,
                                      const R_xlen_t *len, const double *rvals);
typedef SEXP (*Rfmalloc_ld_alloc_fun)(SEXP runtime, R_xlen_t n_variants, int bits,
                                      int window, const R_xlen_t *lo,
                                      const R_xlen_t *len);
typedef int (*Rfmalloc_ld_write_fun)(SEXP store, R_xlen_t j, R_xlen_t first,
                                     R_xlen_t count, const double *values);

static inline Rfmalloc_default_runtime_fun Rfmalloc_default_runtime_ptr(void)
{
    return (Rfmalloc_default_runtime_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_default_runtime");
}

static inline Rfmalloc_open_runtime_fun Rfmalloc_open_runtime_ptr(void)
{
    return (Rfmalloc_open_runtime_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_open_runtime");
}

static inline Rfmalloc_create_vector_fun Rfmalloc_create_vector_ptr(void)
{
    return (Rfmalloc_create_vector_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_create_vector");
}

static inline Rfmalloc_create_vector_ex_fun Rfmalloc_create_vector_ex_ptr(void)
{
    return (Rfmalloc_create_vector_ex_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_create_vector_ex");
}

static inline Rfmalloc_cleanup_runtime_fun Rfmalloc_cleanup_runtime_ptr(void)
{
    return (Rfmalloc_cleanup_runtime_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_cleanup_runtime");
}

static inline Rfmalloc_list_allocations_fun Rfmalloc_list_allocations_ptr(void)
{
    return (Rfmalloc_list_allocations_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_list_allocations");
}

static inline Rfmalloc_set_default_runtime_fun Rfmalloc_set_default_runtime_ptr(void)
{
    return (Rfmalloc_set_default_runtime_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_set_default_runtime");
}

static inline Rfmalloc_is_runtime_fun Rfmalloc_is_runtime_ptr(void)
{
    return (Rfmalloc_is_runtime_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_is_runtime");
}

static inline Rfmalloc_is_fmalloc_vector_fun Rfmalloc_is_fmalloc_vector_ptr(void)
{
    return (Rfmalloc_is_fmalloc_vector_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_is_fmalloc_vector");
}

static inline Rfmalloc_runtime_of_vector_fun Rfmalloc_runtime_of_vector_ptr(void)
{
    return (Rfmalloc_runtime_of_vector_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_runtime_of_vector");
}

static inline Rfmalloc_runtime_info_fun Rfmalloc_runtime_info_ptr(void)
{
    return (Rfmalloc_runtime_info_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_runtime_info");
}

static inline Rfmalloc_vector_type_fun Rfmalloc_vector_type_ptr(void)
{
    return (Rfmalloc_vector_type_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_vector_type");
}

static inline Rfmalloc_vector_length_fun Rfmalloc_vector_length_ptr(void)
{
    return (Rfmalloc_vector_length_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_vector_length");
}

static inline Rfmalloc_vector_payload_ptr_fun Rfmalloc_vector_payload_ptr_ptr(void)
{
    return (Rfmalloc_vector_payload_ptr_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_vector_payload_ptr");
}

static inline Rfmalloc_vector_info_fun Rfmalloc_vector_info_ptr(void)
{
    return (Rfmalloc_vector_info_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_vector_info");
}

static inline Rfmalloc_destroy_vector_fun Rfmalloc_destroy_vector_ptr(void)
{
    return (Rfmalloc_destroy_vector_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_destroy_vector");
}

static inline Rfmalloc_register_tensor_codec_fun Rfmalloc_register_tensor_codec_ptr(void)
{
    return (Rfmalloc_register_tensor_codec_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_register_tensor_codec");
}

static inline SEXP Rfmalloc_default_runtime(void)
{
    return Rfmalloc_default_runtime_ptr()();
}

static inline SEXP Rfmalloc_open_runtime(const char *filepath, double size_gb,
                                         const char *mode)
{
    return Rfmalloc_open_runtime_ptr()(filepath, size_gb, mode);
}

static inline SEXP Rfmalloc_create_vector(SEXP runtime, int sexptype,
                                          R_xlen_t length)
{
    return Rfmalloc_create_vector_ptr()(runtime, sexptype, length);
}

static inline SEXP Rfmalloc_create_vector_ex(SEXP runtime, int sexptype,
                                             R_xlen_t length,
                                             int zero_initialize)
{
    return Rfmalloc_create_vector_ex_ptr()(runtime, sexptype, length, zero_initialize);
}

static inline void Rfmalloc_cleanup_runtime(SEXP runtime)
{
    Rfmalloc_cleanup_runtime_ptr()(runtime);
}

static inline SEXP Rfmalloc_set_default_runtime(SEXP runtime)
{
    return Rfmalloc_set_default_runtime_ptr()(runtime);
}

static inline SEXP Rfmalloc_list_allocations(SEXP runtime)
{
    return Rfmalloc_list_allocations_ptr()(runtime);
}

static inline SEXP Rfmalloc_is_runtime(SEXP runtime)
{
    return Rfmalloc_is_runtime_ptr()(runtime);
}

static inline SEXP Rfmalloc_is_fmalloc_vector(SEXP vector)
{
    return Rfmalloc_is_fmalloc_vector_ptr()(vector);
}

static inline SEXP Rfmalloc_runtime_of_vector(SEXP vector)
{
    return Rfmalloc_runtime_of_vector_ptr()(vector);
}

static inline SEXP Rfmalloc_runtime_info(SEXP runtime)
{
    return Rfmalloc_runtime_info_ptr()(runtime);
}

static inline SEXP Rfmalloc_vector_type(SEXP vector)
{
    return Rfmalloc_vector_type_ptr()(vector);
}

static inline SEXP Rfmalloc_vector_length(SEXP vector)
{
    return Rfmalloc_vector_length_ptr()(vector);
}

static inline SEXP Rfmalloc_vector_payload_ptr(SEXP vector)
{
    return Rfmalloc_vector_payload_ptr_ptr()(vector);
}

static inline SEXP Rfmalloc_vector_info(SEXP vector)
{
    return Rfmalloc_vector_info_ptr()(vector);
}

static inline SEXP Rfmalloc_destroy_vector(SEXP vector, int unsafe)
{
    return Rfmalloc_destroy_vector_ptr()(vector, unsafe);
}

static inline int Rfmalloc_register_tensor_codec(const char *name,
                                                 unsigned int items_per_block,
                                                 unsigned int bytes_per_block,
                                                 Rfmalloc_tensor_decode_fn decode)
{
    return Rfmalloc_register_tensor_codec_ptr()(name, items_per_block,
                                                bytes_per_block, decode);
}

static inline Rfmalloc_register_matmul_backend_fun Rfmalloc_register_matmul_backend_ptr(void)
{
    return (Rfmalloc_register_matmul_backend_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_register_matmul_backend");
}

static inline int Rfmalloc_register_matmul_backend(const char *name, Rfmalloc_gemm_fn fn)
{
    return Rfmalloc_register_matmul_backend_ptr()(name, fn);
}

static inline Rfmalloc_register_matmul_backend_ex_fun Rfmalloc_register_matmul_backend_ex_ptr(void)
{
    return (Rfmalloc_register_matmul_backend_ex_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_register_matmul_backend_ex");
}

static inline int Rfmalloc_register_matmul_backend_ex(const char *name,
                                                      Rfmalloc_gemm_fn fn,
                                                      Rfmalloc_typed_gemm_fn typed_fn)
{
    return Rfmalloc_register_matmul_backend_ex_ptr()(name, fn, typed_fn);
}

static inline Rfmalloc_tensor_decode_range_fun Rfmalloc_tensor_decode_ptr(void)
{
    return (Rfmalloc_tensor_decode_range_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_tensor_decode");
}

static inline int Rfmalloc_tensor_decode(SEXP tensor, R_xlen_t elem_offset,
                                         R_xlen_t n_elems, double *out)
{
    return Rfmalloc_tensor_decode_ptr()(tensor, elem_offset, n_elems, out);
}

static inline Rfmalloc_buffer_open_fun Rfmalloc_buffer_open_ptr(void)
{
    return (Rfmalloc_buffer_open_fun)
        R_GetCCallable("Rfmalloc", "Rfmalloc_buffer_open");
}

static inline Rfmalloc_buffer_write_fun Rfmalloc_buffer_write_ptr(void)
{
    return (Rfmalloc_buffer_write_fun)
        R_GetCCallable("Rfmalloc", "Rfmalloc_buffer_write");
}

static inline Rfmalloc_buffer_finish_fun Rfmalloc_buffer_finish_ptr(void)
{
    return (Rfmalloc_buffer_finish_fun)
        R_GetCCallable("Rfmalloc", "Rfmalloc_buffer_finish");
}

static inline Rfmalloc_buffer_abort_fun Rfmalloc_buffer_abort_ptr(void)
{
    return (Rfmalloc_buffer_abort_fun)
        R_GetCCallable("Rfmalloc", "Rfmalloc_buffer_abort");
}

static inline Rfmalloc_haplotypes_data_fun Rfmalloc_haplotypes_data_ptr(void)
{
    return (Rfmalloc_haplotypes_data_fun)
        R_GetCCallable("Rfmalloc", "Rfmalloc_haplotypes_data");
}

static inline int Rfmalloc_haplotypes_data(
    SEXP store, struct Rfmalloc_haplotype_view *view)
{
    return Rfmalloc_haplotypes_data_ptr()(store, view);
}

static inline Rfmalloc_storage_view_fun Rfmalloc_storage_view_ptr(void)
{
    return (Rfmalloc_storage_view_fun)
        R_GetCCallable("Rfmalloc", "Rfmalloc_storage_view");
}

static inline Rfmalloc_storage_data_fun Rfmalloc_storage_data_ptr(void)
{
    return (Rfmalloc_storage_data_fun)
        R_GetCCallable("Rfmalloc", "Rfmalloc_storage_data");
}

static inline SEXP Rfmalloc_storage_view(SEXP owner, SEXP runtime,
                                         const void *data, size_t nbytes)
{
    return Rfmalloc_storage_view_ptr()(owner, runtime, data, nbytes);
}

static inline int Rfmalloc_storage_data(SEXP object, const void **data,
                                        size_t *nbytes, SEXP *runtime)
{
    return Rfmalloc_storage_data_ptr()(object, data, nbytes, runtime);
}

static inline Rfmalloc_storage_advise_fun Rfmalloc_storage_advise_ptr(void)
{
    return (Rfmalloc_storage_advise_fun)
        R_GetCCallable("Rfmalloc", "Rfmalloc_storage_advise");
}

static inline int Rfmalloc_storage_advise(SEXP object, size_t offset,
                                          size_t nbytes, int advice)
{
    return Rfmalloc_storage_advise_ptr()(object, offset, nbytes, advice);
}

static inline R_xlen_t Rfmalloc_ld_ncol(SEXP store)
{
    return ((Rfmalloc_ld_ncol_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_ld_ncol"))(store);
}

static inline int Rfmalloc_ld_bits(SEXP store)
{
    return ((Rfmalloc_ld_bits_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_ld_bits"))(store);
}

static inline double Rfmalloc_ld_pair(SEXP store, R_xlen_t i, R_xlen_t j)
{
    return ((Rfmalloc_ld_pair_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_ld_pair"))(store, i, j);
}

static inline int Rfmalloc_ld_col(SEXP store, R_xlen_t j, R_xlen_t *lo,
                                  R_xlen_t *len, double *out)
{
    return ((Rfmalloc_ld_col_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_ld_col"))(store, j, lo, len, out);
}

static inline int Rfmalloc_ld_col_raw(SEXP store, R_xlen_t j, R_xlen_t *lo,
                                      R_xlen_t *len, const void **values)
{
    return ((Rfmalloc_ld_col_raw_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_ld_col_raw"))(store, j, lo, len, values);
}

static inline SEXP Rfmalloc_ld_alloc(SEXP runtime, R_xlen_t n_variants, int bits,
                                     int window, const R_xlen_t *lo,
                                     const R_xlen_t *len)
{
    return ((Rfmalloc_ld_alloc_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_ld_alloc"))(
        runtime, n_variants, bits, window, lo, len);
}

static inline int Rfmalloc_ld_write(SEXP store, R_xlen_t j, R_xlen_t first,
                                    R_xlen_t count, const double *values)
{
    return ((Rfmalloc_ld_write_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_ld_write"))(
        store, j, first, count, values);
}

static inline SEXP Rfmalloc_ld_build(SEXP runtime, R_xlen_t n_variants, int bits,
                                     int window, const R_xlen_t *lo,
                                     const R_xlen_t *len, const double *rvals)
{
    return ((Rfmalloc_ld_build_fun) R_GetCCallable("Rfmalloc", "Rfmalloc_ld_build"))(
        runtime, n_variants, bits, window, lo, len, rvals);
}

#ifdef __cplusplus
}
#endif

#endif /* RFMALLOC_API_H */
