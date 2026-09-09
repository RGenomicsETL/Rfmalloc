/*
 * rggml_api.c - Rggml's C-callable compute API.
 *
 * Thin `extern "C"` wrapper functions around the vendored GGML engine.
 * These are the functions registered with R_RegisterCCallable() in
 * rggml_init.c and re-exposed to downstream packages via the inline
 * R_GetCCallable() wrappers in inst/include/Rggml.h.
 *
 * Design intent: downstream packages build and compute GGML tensor graphs
 * using real GGML types (struct ggml_context *, struct ggml_tensor *,
 * ggml_backend_t, ...) - Rggml.h includes the vendored public GGML headers
 * so those types are available - and never need to vendor or link against
 * GGML themselves.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#ifdef RGGML_HAVE_BLAS
#include <ggml-blas.h>
#endif
/* RGGML_HAVE_VULKAN is put into PKG_CPPFLAGS by configure only when the package
 * was built with --with-vulkan; libggml.a then contains the Vulkan backend. */
#ifdef RGGML_HAVE_VULKAN
#include <ggml-vulkan.h>
#endif
#ifdef RGGML_HAVE_CUDA
#include <ggml-cuda.h>
#endif

#include "rggml_api.h"

const char *Rggml_version(void)
{
    return ggml_version();
}

struct ggml_context *Rggml_context_create(size_t mem_size, int no_alloc)
{
    struct ggml_init_params params;
    memset(&params, 0, sizeof(params));
    params.mem_size = mem_size;
    params.mem_buffer = NULL;
    params.no_alloc = no_alloc != 0;
    return ggml_init(params);
}

void Rggml_context_free(struct ggml_context *ctx)
{
    if (ctx) ggml_free(ctx);
}

size_t Rggml_used_mem(const struct ggml_context *ctx)
{
    return ggml_used_mem(ctx);
}

size_t Rggml_tensor_overhead(void)
{
    return ggml_tensor_overhead();
}

size_t Rggml_graph_overhead(size_t size)
{
    return ggml_graph_overhead_custom(size, /*grads=*/false);
}

struct ggml_tensor *Rggml_new_tensor(struct ggml_context *ctx, enum ggml_type type,
                                      int n_dims, const int64_t *ne, void *data)
{
    if (!ctx || !ne || n_dims < 1 || n_dims > GGML_MAX_DIMS) return NULL;
    struct ggml_tensor *t = ggml_new_tensor(ctx, type, n_dims, ne);
    if (t && data) {
        /* Zero-copy path: point the tensor at caller-owned memory (e.g. an
         * mmap'd quantized payload). Intended for contexts created with
         * no_alloc = 1, where ggml_new_tensor() does not allocate a data
         * buffer of its own. */
        t->data = data;
    }
    return t;
}

void *Rggml_tensor_data(const struct ggml_tensor *tensor)
{
    return tensor ? tensor->data : NULL;
}

void Rggml_tensor_set_data(struct ggml_tensor *tensor, void *data)
{
    if (tensor) tensor->data = data;
}

int64_t Rggml_tensor_ne(const struct ggml_tensor *tensor, int dim)
{
    if (!tensor || dim < 0 || dim >= GGML_MAX_DIMS) return -1;
    return tensor->ne[dim];
}

size_t Rggml_tensor_nb(const struct ggml_tensor *tensor, int dim)
{
    if (!tensor || dim < 0 || dim >= GGML_MAX_DIMS) return 0;
    return tensor->nb[dim];
}

ggml_backend_t Rggml_backend_cpu_init(void)
{
    return ggml_backend_cpu_init();
}

void Rggml_backend_cpu_set_n_threads(ggml_backend_t backend, int n_threads)
{
    if (backend && n_threads > 0) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }
}

void Rggml_backend_free(ggml_backend_t backend)
{
    if (backend) ggml_backend_free(backend);
}

int Rggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph *cgraph)
{
    if (!backend || !cgraph) return (int)GGML_STATUS_FAILED;
    return (int)ggml_backend_graph_compute(backend, cgraph);
}

ggml_backend_t Rggml_backend_blas_init(void)
{
#ifdef RGGML_HAVE_BLAS
    return ggml_backend_blas_init();
#else
    return NULL;
#endif
}

/*
 * Vulkan backend. Present unconditionally as a C-callable so
 * downstream packages can probe for it at run time; it reports zero devices and
 * refuses to initialize when Rggml was built without --with-vulkan. Freeing is
 * the usual Rggml_backend_free().
 */
int Rggml_backend_vulkan_device_count(void)
{
#ifdef RGGML_HAVE_VULKAN
    return ggml_backend_vk_get_device_count();
#else
    return 0;
#endif
}

ggml_backend_t Rggml_backend_vulkan_init(int device)
{
#ifdef RGGML_HAVE_VULKAN
    if (device < 0 || device >= ggml_backend_vk_get_device_count()) return NULL;
    return ggml_backend_vk_init((size_t) device);
#else
    (void) device;
    return NULL;
#endif
}

/*
 * Device-buffer residency.
 *
 * The CPU and BLAS backends compute on ordinary host memory, so a tensor whose
 * ->data points at an R buffer just works. A GPU backend does not: its tensors
 * must live in device memory. These four wrap GGML's backend-agnostic path -
 * allocate every tensor of a no_alloc context in one backend buffer, upload
 * inputs, compute, download results - which is identical for CPU, BLAS and
 * Vulkan. Free the buffer before the context.
 */
ggml_backend_buffer_t Rggml_backend_alloc_ctx_tensors(struct ggml_context *ctx,
                                                       ggml_backend_t backend)
{
    if (!ctx || !backend) return NULL;
    return ggml_backend_alloc_ctx_tensors(ctx, backend);
}

void Rggml_backend_buffer_free(ggml_backend_buffer_t buffer)
{
    if (buffer) ggml_backend_buffer_free(buffer);
}

void Rggml_backend_tensor_set(struct ggml_tensor *tensor, const void *data,
                               size_t offset, size_t size)
{
    if (tensor && data) ggml_backend_tensor_set(tensor, data, offset, size);
}

void Rggml_backend_tensor_get(const struct ggml_tensor *tensor, void *data,
                               size_t offset, size_t size)
{
    if (tensor && data) ggml_backend_tensor_get(tensor, data, offset, size);
}

int Rggml_backend_vulkan_device_description(int device, char *buf, size_t buf_size)
{
#ifdef RGGML_HAVE_VULKAN
    if (!buf || buf_size == 0) return -1;
    if (device < 0 || device >= ggml_backend_vk_get_device_count()) return -1;
    ggml_backend_vk_get_device_description(device, buf, buf_size);
    return 0;
#else
    (void) device; (void) buf; (void) buf_size;
    return -1;
#endif
}

/* CUDA is the same optional-backend contract as Vulkan: the C-callables are
 * always present, while a non-CUDA build reports no devices and declines init. */
/* Set once anything in this process has asked CUDA a question. cuBLAS reads
 * NVIDIA_TF32_OVERRIDE when its library initializes, so a caller that wants to
 * change reduced-precision behaviour has to do it before that point, and
 * rggml_cuda_tf32() uses this to say so instead of failing silently. */
static int rggml_cuda_touched;

int Rggml_cuda_touched(void)
{
    return rggml_cuda_touched;
}

int Rggml_backend_cuda_device_count(void)
{
#ifdef RGGML_HAVE_CUDA
    rggml_cuda_touched = 1;
    return ggml_backend_cuda_get_device_count();
#else
    return 0;
#endif
}

ggml_backend_t Rggml_backend_cuda_init(int device)
{
#ifdef RGGML_HAVE_CUDA
    rggml_cuda_touched = 1;
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) return NULL;
    return ggml_backend_cuda_init(device);
#else
    (void) device;
    return NULL;
#endif
}

int Rggml_backend_cuda_device_description(int device, char *buf, size_t buf_size)
{
#ifdef RGGML_HAVE_CUDA
    if (!buf || buf_size == 0) return -1;
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) return -1;
    ggml_backend_cuda_get_device_description(device, buf, buf_size);
    return 0;
#else
    (void) device; (void) buf; (void) buf_size;
    return -1;
#endif
}

void Rggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, int n_threads)
{
#ifdef RGGML_HAVE_BLAS
    if (backend_blas) ggml_backend_blas_set_n_threads(backend_blas, n_threads);
#else
    (void) backend_blas;
    (void) n_threads;
#endif
}

struct ggml_cgraph *Rggml_new_graph(struct ggml_context *ctx, size_t size)
{
    if (!ctx) return NULL;
    return ggml_new_graph_custom(ctx, size > 0 ? size : GGML_DEFAULT_GRAPH_SIZE, /*grads=*/false);
}

void Rggml_build_forward_expand(struct ggml_cgraph *cgraph, struct ggml_tensor *tensor)
{
    if (cgraph && tensor) ggml_build_forward_expand(cgraph, tensor);
}

struct ggml_tensor *Rggml_mul_mat(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b)
{
    if (!ctx || !a || !b) return NULL;
    return ggml_mul_mat(ctx, a, b);
}

struct ggml_tensor *Rggml_mul_mat_id(struct ggml_context *ctx,
                                      struct ggml_tensor *as,
                                      struct ggml_tensor *b,
                                      struct ggml_tensor *ids)
{
    if (!ctx || !as || !b || !ids) return NULL;
    return ggml_mul_mat_id(ctx, as, b, ids);
}

/*
 * Rggml_compute_mul_mat(): build the one-op graph `ggml_mul_mat(ctx, a, b)`,
 * compute it on `backend`, and copy the F32 result into a caller-provided
 * buffer (out_f32 XOR out_f64; pass NULL for the one you don't want).
 *
 * ggml_mul_mat()'s result is always GGML_TYPE_F32 regardless of the input
 * types (this holds for quantized inputs too - dequantization happens
 * internally during the dot-product accumulation). This helper allocates
 * scratch memory for the result tensor itself (it does not require `ctx`
 * to have room for the result's *data*, only for its small tensor/graph
 * metadata) so it works whether `ctx` was created with no_alloc = 0 or 1.
 *
 * Returns 0 on success, a negative value on failure.
 */
int Rggml_compute_mul_mat(struct ggml_context *ctx, ggml_backend_t backend,
                           struct ggml_tensor *a, struct ggml_tensor *b,
                           float *out_f32, double *out_f64)
{
    if (!ctx || !backend || !a || !b) return -1;
    if (!out_f32 && !out_f64) return -1;

    struct ggml_tensor *result = ggml_mul_mat(ctx, a, b);
    if (!result) return -2;

    size_t nbytes = ggml_nbytes(result);
    void *result_data = malloc(nbytes > 0 ? nbytes : 1);
    if (!result_data) return -3;
    result->data = result_data;

    struct ggml_cgraph *gf = ggml_new_graph_custom(ctx, 8, /*grads=*/false);
    if (!gf) {
        free(result_data);
        return -4;
    }
    ggml_build_forward_expand(gf, result);

    enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        free(result_data);
        return -5;
    }

    int64_t n = ggml_nelements(result);
    if (out_f32) {
        memcpy(out_f32, result_data, (size_t)n * sizeof(float));
    }
    if (out_f64) {
        const float *src = (const float *)result_data;
        for (int64_t i = 0; i < n; i++) out_f64[i] = (double)src[i];
    }

    free(result_data);
    return 0;
}

size_t Rggml_quantize(enum ggml_type type, const float *src, void *dst,
                      int64_t nrows, int64_t n_per_row)
{
    if (!src || !dst || nrows < 1 || n_per_row < 1) return 0;
    /* Quantize `nrows` contiguous rows of `n_per_row` f32 values each into the
     * type's block format (e.g. GGML_TYPE_Q4_K), writing to `dst`. `dst` must
     * hold ggml_row_size(type, n_per_row) * nrows bytes. No importance matrix.
     * Returns the number of bytes written (0 on bad arguments). The result is
     * byte-compatible with a GGUF tensor of that type, i.e. a valid payload for
     * an Rfmalloc typed tensor / a ggml tensor pointed at it zero-copy. */
    return ggml_quantize_chunk(type, src, dst, 0, nrows, n_per_row, NULL);
}

int Rggml_dequantize(enum ggml_type type, const void *src, float *dst, int64_t n)
{
    /* Decode `n` elements (a whole number of blocks) of a quantized payload to
     * f32 through GGML's own type-traits to_float - the authoritative reference
     * dequantizer for every GGUF type. The caller must have initialized the CPU
     * backend once (Rggml_backend_cpu_init) so the fp16 lookup table used to
     * read block scales is populated. Returns 0 on success. */
    if (!src || !dst || n < 1) return -1;
    if (type < 0 || type >= GGML_TYPE_COUNT) return -2;
    const struct ggml_type_traits *traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float) return -2;
    if (n % ggml_blck_size(type) != 0) return -3;
    traits->to_float(src, dst, n);
    return 0;
}

int Rggml_can_dequantize(enum ggml_type type)
{
    if (type < 0 || type >= GGML_TYPE_COUNT) return 0;
    switch (type) {
    case GGML_TYPE_F32:
    case GGML_TYPE_F64:
    case GGML_TYPE_I8:
    case GGML_TYPE_I16:
    case GGML_TYPE_I32:
    case GGML_TYPE_I64:
        return 1;
    default: {
        const struct ggml_type_traits *traits = ggml_get_type_traits(type);
        return traits && traits->to_float;
    }
    }
}

int Rggml_dequantize_double(enum ggml_type type, const void *src, double *dst,
                            int64_t n)
{
    if (!src || !dst || n < 1) return -1;
    if (type < 0 || type >= GGML_TYPE_COUNT) return -2;
    switch (type) {
    case GGML_TYPE_F32: {
        const float *p = (const float *)src;
        for (int64_t i = 0; i < n; ++i) dst[i] = p[i];
        return 0;
    }
    case GGML_TYPE_F64:
        memcpy(dst, src, (size_t)n * sizeof(*dst));
        return 0;
    case GGML_TYPE_I8: {
        const int8_t *p = (const int8_t *)src;
        for (int64_t i = 0; i < n; ++i) dst[i] = p[i];
        return 0;
    }
    case GGML_TYPE_I16: {
        const int16_t *p = (const int16_t *)src;
        for (int64_t i = 0; i < n; ++i) dst[i] = p[i];
        return 0;
    }
    case GGML_TYPE_I32: {
        const int32_t *p = (const int32_t *)src;
        for (int64_t i = 0; i < n; ++i) dst[i] = p[i];
        return 0;
    }
    case GGML_TYPE_I64: {
        const int64_t *p = (const int64_t *)src;
        for (int64_t i = 0; i < n; ++i) dst[i] = (double)p[i];
        return 0;
    }
    default:
        break;
    }

    const struct ggml_type_traits *traits = ggml_get_type_traits(type);
    const int64_t block = ggml_blck_size(type);
    const size_t block_bytes = ggml_type_size(type);
    if (!traits || !traits->to_float || block < 1 || block_bytes == 0 ||
        n % block != 0) return -2;

    int64_t cap = 16384;
    if (cap < block) cap = block;
    cap -= cap % block;
    float *scratch = (float *)malloc((size_t)cap * sizeof(*scratch));
    if (!scratch) return -3;

    const unsigned char *p = (const unsigned char *)src;
    for (int64_t done = 0; done < n; ) {
        int64_t count = n - done;
        if (count > cap) count = cap;
        traits->to_float(p + (size_t)(done / block) * block_bytes,
                         scratch, count);
        for (int64_t i = 0; i < count; ++i) dst[done + i] = scratch[i];
        done += count;
    }
    free(scratch);
    return 0;
}

/*
 * Graph ops. Thin wrappers over the ggml ops a transformer
 * forward pass composes: embedding lookup, RMSNorm, elementwise, activation,
 * RoPE, masked softmax, and the shape ops. Scalar parameters cross the
 * C-callable boundary as double and are narrowed here. All return NULL on
 * NULL inputs so downstream builders can chain and check once.
 */
struct ggml_tensor *Rggml_get_rows(struct ggml_context *ctx, struct ggml_tensor *a,
                                    struct ggml_tensor *b)
{
    if (!ctx || !a || !b) return NULL;
    return ggml_get_rows(ctx, a, b);
}

struct ggml_tensor *Rggml_rms_norm(struct ggml_context *ctx, struct ggml_tensor *a,
                                    double eps)
{
    if (!ctx || !a) return NULL;
    return ggml_rms_norm(ctx, a, (float) eps);
}

struct ggml_tensor *Rggml_l2_norm(struct ggml_context *ctx, struct ggml_tensor *a,
                                   double eps)
{
    if (!ctx || !a) return NULL;
    return ggml_l2_norm(ctx, a, (float) eps);
}

struct ggml_tensor *Rggml_mul(struct ggml_context *ctx, struct ggml_tensor *a,
                               struct ggml_tensor *b)
{
    if (!ctx || !a || !b) return NULL;
    return ggml_mul(ctx, a, b);
}

struct ggml_tensor *Rggml_add(struct ggml_context *ctx, struct ggml_tensor *a,
                               struct ggml_tensor *b)
{
    if (!ctx || !a || !b) return NULL;
    return ggml_add(ctx, a, b);
}

struct ggml_tensor *Rggml_div(struct ggml_context *ctx, struct ggml_tensor *a,
                               struct ggml_tensor *b)
{
    if (!ctx || !a || !b) return NULL;
    return ggml_div(ctx, a, b);
}

struct ggml_tensor *Rggml_silu(struct ggml_context *ctx, struct ggml_tensor *a)
{
    if (!ctx || !a) return NULL;
    return ggml_silu(ctx, a);
}

struct ggml_tensor *Rggml_geglu(struct ggml_context *ctx,
                                struct ggml_tensor *gate,
                                struct ggml_tensor *up)
{
    if (!ctx || !gate || !up) return NULL;
    return ggml_geglu_split(ctx, gate, up);
}

struct ggml_tensor *Rggml_sigmoid(struct ggml_context *ctx, struct ggml_tensor *a)
{
    if (!ctx || !a) return NULL;
    return ggml_sigmoid(ctx, a);
}

struct ggml_tensor *Rggml_softplus(struct ggml_context *ctx, struct ggml_tensor *a)
{
    if (!ctx || !a) return NULL;
    return ggml_softplus(ctx, a);
}

struct ggml_tensor *Rggml_scale(struct ggml_context *ctx, struct ggml_tensor *a,
                                 double s)
{
    if (!ctx || !a) return NULL;
    return ggml_scale(ctx, a, (float) s);
}

struct ggml_tensor *Rggml_sum_rows(struct ggml_context *ctx, struct ggml_tensor *a)
{
    if (!ctx || !a) return NULL;
    return ggml_sum_rows(ctx, a);
}

struct ggml_tensor *Rggml_clamp(struct ggml_context *ctx, struct ggml_tensor *a,
                                 double min, double max)
{
    if (!ctx || !a) return NULL;
    return ggml_clamp(ctx, a, (float) min, (float) max);
}

struct ggml_tensor *Rggml_argsort_top_k(struct ggml_context *ctx,
                                         struct ggml_tensor *a, int k)
{
    if (!ctx || !a || k < 1 || k > a->ne[0]) return NULL;
    return ggml_argsort_top_k(ctx, a, k);
}

struct ggml_tensor *Rggml_concat(struct ggml_context *ctx,
                                  struct ggml_tensor *a,
                                  struct ggml_tensor *b, int dim)
{
    if (!ctx || !a || !b || dim < 0 || dim >= GGML_MAX_DIMS) return NULL;
    return ggml_concat(ctx, a, b, dim);
}

struct ggml_tensor *Rggml_ssm_conv(struct ggml_context *ctx,
                                    struct ggml_tensor *sx,
                                    struct ggml_tensor *kernel)
{
    if (!ctx || !sx || !kernel) return NULL;
    return ggml_ssm_conv(ctx, sx, kernel);
}

struct ggml_tensor *Rggml_conv_1d(struct ggml_context *ctx,
                                   struct ggml_tensor *kernel,
                                   struct ggml_tensor *data,
                                   int stride, int padding, int dilation)
{
    if (!ctx || !kernel || !data || stride < 1 || padding < 0 || dilation < 1) {
        return NULL;
    }
    if (kernel->type != GGML_TYPE_F32 || data->type != GGML_TYPE_F32 ||
        kernel->ne[0] < 1 || kernel->ne[1] != data->ne[1] ||
        data->ne[3] != 1) {
        return NULL;
    }
    /* Upstream ggml_conv_1d deliberately builds an F16 im2col tensor. The
     * bound F32-program contract instead needs F32 inference semantics, so
     * compose the same upstream im2col and matmul primitives with an F32
     * staging tensor. The layout is kernel [K, IC, OC], data [N, IC, B], and
     * result [N, OC, B]. */
    struct ggml_tensor *im2col = ggml_im2col(
        ctx, kernel, data, stride, 0, padding, 0, dilation, 0,
        false, GGML_TYPE_F32);
    if (!im2col) return NULL;
    /* A mapped CPU weight has data but deliberately no backend buffer. A
     * reshape would be a view whose source lacks that buffer, which the
     * retained all-tensor allocator correctly rejects. Make a metadata-only
     * two-dimensional alias in that case. Device weights do have a buffer, so
     * retain the ordinary GGML view path. */
    if (kernel->ne[0] > INT64_MAX / kernel->ne[1]) return NULL;
    struct ggml_tensor *flat_kernel;
    if (kernel->buffer == NULL) {
        int64_t flat_ne[2] = { kernel->ne[0] * kernel->ne[1], kernel->ne[2] };
        flat_kernel = ggml_new_tensor(ctx, kernel->type, 2, flat_ne);
        if (flat_kernel) flat_kernel->data = kernel->data;
    } else {
        flat_kernel = ggml_reshape_2d(ctx, kernel,
                                      kernel->ne[0] * kernel->ne[1], kernel->ne[2]);
    }
    if (!flat_kernel || im2col->ne[1] > INT64_MAX / im2col->ne[2]) return NULL;
    struct ggml_tensor *result = ggml_mul_mat(
        ctx, flat_kernel,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0],
                        im2col->ne[2] * im2col->ne[1]));
    if (!result) return NULL;
    result = ggml_reshape_3d(ctx, result, kernel->ne[2], im2col->ne[1],
                             im2col->ne[2]);
    return ggml_cont(ctx, ggml_permute(ctx, result, 1, 0, 2, 3));
}

struct ggml_tensor *Rggml_leaky_relu(struct ggml_context *ctx,
                                      struct ggml_tensor *a, double slope)
{
    if (!ctx || !a || !isfinite(slope) || slope < 0) return NULL;
    return ggml_leaky_relu(ctx, a, (float)slope, false);
}

struct ggml_tensor *Rggml_soft_max(struct ggml_context *ctx, struct ggml_tensor *a)
{
    if (!ctx || !a) return NULL;
    return ggml_soft_max(ctx, a);
}

struct ggml_tensor *Rggml_soft_max_ext(struct ggml_context *ctx,
                                       struct ggml_tensor *a,
                                       struct ggml_tensor *mask,
                                       double scale, double max_bias)
{
    if (!ctx || !a) return NULL;
    return ggml_soft_max_ext(ctx, a, mask, (float)scale, (float)max_bias);
}

struct ggml_tensor *Rggml_diag_mask_inf(struct ggml_context *ctx, struct ggml_tensor *a,
                                         int n_past)
{
    if (!ctx || !a) return NULL;
    return ggml_diag_mask_inf(ctx, a, n_past);
}

/*
 * RoPE with the parameters that vary between model families exposed and the
 * YaRN extension left off (ext_factor = 0, so beta_fast/beta_slow are inert;
 * attn_factor = 1, freq_scale = 1, n_ctx_orig = 0 - llama.cpp's defaults for
 * non-scaled contexts). `pos` is an I32 tensor of positions; `mode` is
 * GGML_ROPE_TYPE_NORMAL (0) or GGML_ROPE_TYPE_NEOX (2).
 */
struct ggml_tensor *Rggml_rope(struct ggml_context *ctx, struct ggml_tensor *a,
                                struct ggml_tensor *pos, int n_dims, int mode,
                                double freq_base)
{
    if (!ctx || !a || !pos) return NULL;
    return ggml_rope_ext(ctx, a, pos, NULL, n_dims, mode, /*n_ctx_orig=*/0,
                         (float) freq_base, /*freq_scale=*/1.0f,
                         /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                         /*beta_fast=*/32.0f, /*beta_slow=*/1.0f);
}

struct ggml_tensor *Rggml_rope_multi(struct ggml_context *ctx,
                                      struct ggml_tensor *a,
                                      struct ggml_tensor *pos, int n_dims,
                                      const int sections[GGML_MROPE_SECTIONS],
                                      int mode, double freq_base)
{
    if (!ctx || !a || !pos || !sections) return NULL;
    int copy[GGML_MROPE_SECTIONS];
    memcpy(copy, sections, sizeof(copy));
    return ggml_rope_multi(ctx, a, pos, NULL, n_dims, copy, mode,
                           /*n_ctx_orig=*/0, (float)freq_base,
                           /*freq_scale=*/1.0f, /*ext_factor=*/0.0f,
                           /*attn_factor=*/1.0f, /*beta_fast=*/32.0f,
                           /*beta_slow=*/1.0f);
}

struct ggml_tensor *Rggml_gated_delta_net(struct ggml_context *ctx,
                                           struct ggml_tensor *q,
                                           struct ggml_tensor *k,
                                           struct ggml_tensor *v,
                                           struct ggml_tensor *gate,
                                           struct ggml_tensor *beta,
                                           struct ggml_tensor *state,
                                           int64_t snapshots)
{
    if (!ctx || !q || !k || !v || !gate || !beta || !state ||
        snapshots < 1) return NULL;
    return ggml_gated_delta_net(ctx, q, k, v, gate, beta, state, snapshots);
}

struct ggml_tensor *Rggml_reshape_2d(struct ggml_context *ctx, struct ggml_tensor *a,
                                      int64_t ne0, int64_t ne1)
{
    if (!ctx || !a) return NULL;
    return ggml_reshape_2d(ctx, a, ne0, ne1);
}

struct ggml_tensor *Rggml_reshape_3d(struct ggml_context *ctx, struct ggml_tensor *a,
                                      int64_t ne0, int64_t ne1, int64_t ne2)
{
    if (!ctx || !a) return NULL;
    return ggml_reshape_3d(ctx, a, ne0, ne1, ne2);
}

struct ggml_tensor *Rggml_reshape_4d(struct ggml_context *ctx, struct ggml_tensor *a,
                                      int64_t ne0, int64_t ne1, int64_t ne2,
                                      int64_t ne3)
{
    if (!ctx || !a) return NULL;
    return ggml_reshape_4d(ctx, a, ne0, ne1, ne2, ne3);
}

struct ggml_tensor *Rggml_permute(struct ggml_context *ctx, struct ggml_tensor *a,
                                   int axis0, int axis1, int axis2, int axis3)
{
    if (!ctx || !a) return NULL;
    return ggml_permute(ctx, a, axis0, axis1, axis2, axis3);
}

struct ggml_tensor *Rggml_cont(struct ggml_context *ctx, struct ggml_tensor *a)
{
    if (!ctx || !a) return NULL;
    return ggml_cont(ctx, a);
}

struct ggml_tensor *Rggml_transpose(struct ggml_context *ctx, struct ggml_tensor *a)
{
    if (!ctx || !a) return NULL;
    return ggml_transpose(ctx, a);
}

/*
 * Views and copies - what a KV cache is made of: strided
 * views into a persistent cache tensor, written with ggml_cpy nodes expanded
 * into the graph ahead of the attention that reads them. Offsets and strides
 * are in bytes, as in ggml itself.
 */
struct ggml_tensor *Rggml_view_1d(struct ggml_context *ctx, struct ggml_tensor *a,
                                   int64_t ne0, size_t offset)
{
    if (!ctx || !a) return NULL;
    return ggml_view_1d(ctx, a, ne0, offset);
}

struct ggml_tensor *Rggml_view_2d(struct ggml_context *ctx, struct ggml_tensor *a,
                                   int64_t ne0, int64_t ne1, size_t nb1, size_t offset)
{
    if (!ctx || !a) return NULL;
    return ggml_view_2d(ctx, a, ne0, ne1, nb1, offset);
}

struct ggml_tensor *Rggml_view_3d(struct ggml_context *ctx, struct ggml_tensor *a,
                                   int64_t ne0, int64_t ne1, int64_t ne2,
                                   size_t nb1, size_t nb2, size_t offset)
{
    if (!ctx || !a) return NULL;
    return ggml_view_3d(ctx, a, ne0, ne1, ne2, nb1, nb2, offset);
}

struct ggml_tensor *Rggml_view_4d(struct ggml_context *ctx, struct ggml_tensor *a,
                                   int64_t ne0, int64_t ne1, int64_t ne2,
                                   int64_t ne3, size_t nb1, size_t nb2,
                                   size_t nb3, size_t offset)
{
    if (!ctx || !a) return NULL;
    return ggml_view_4d(ctx, a, ne0, ne1, ne2, ne3,
                        nb1, nb2, nb3, offset);
}

struct ggml_tensor *Rggml_cpy(struct ggml_context *ctx, struct ggml_tensor *a,
                               struct ggml_tensor *b)
{
    if (!ctx || !a || !b) return NULL;
    return ggml_cpy(ctx, a, b);
}

size_t Rggml_type_size(enum ggml_type type)
{
    return ggml_type_size(type);
}

size_t Rggml_row_size(enum ggml_type type, int64_t ne)
{
    return ggml_row_size(type, ne);
}

int64_t Rggml_blck_size(enum ggml_type type)
{
    return ggml_blck_size(type);
}

size_t Rggml_nbytes(const struct ggml_tensor *tensor)
{
    return tensor ? ggml_nbytes(tensor) : 0;
}

int64_t Rggml_nelements(const struct ggml_tensor *tensor)
{
    return tensor ? ggml_nelements(tensor) : 0;
}

const char *Rggml_type_name(enum ggml_type type)
{
    return ggml_type_name(type);
}
