#ifndef RGGML_API_PUBLIC_H
#define RGGML_API_PUBLIC_H

/*
 * Rggml.h - C-callable compute API for the Rggml carrier package.
 *
 * Rggml carries one pinned official GGML tree: core tensors and GGUF, CPU and
 * BLAS compute, plus opt-in Vulkan and CUDA backends. It exposes the subset
 * sibling packages compose through R_RegisterCCallable(). Other R packages
 * link to Rggml ('LinkingTo: Rggml' in DESCRIPTION) and #include this header
 * to build and compute GGML tensor graphs, including quantized types such as
 * GGML_TYPE_Q4_K, without re-vendoring GGML themselves.
 *
 * Usage:
 *   1. List Rggml under LinkingTo (and Imports, so it is loaded first) in
 *      your package's DESCRIPTION.
 *   2. #include <Rggml.h> in your C/C++ source. This header pulls in the
 *      vendored public GGML headers (ggml.h, ggml-alloc.h, ggml-backend.h,
 *      ggml-cpu.h), also installed by Rggml, so real GGML types
 *      (struct ggml_context *, struct ggml_tensor *, ggml_backend_t, ...)
 *      are available in your signatures.
 *   3. Every wrapped function below has a "<Name>_ptr()" accessor that
 *      resolves the symbol with R_GetCCallable() and returns a function
 *      pointer. Call it once (e.g. cache the pointer in a static, or in
 *      your package's .onLoad-triggered init routine) and invoke through
 *      it:
 *
 *          Rggml_context_create_fun ctx_create = Rggml_context_create_ptr();
 *          struct ggml_context *ctx = ctx_create(1 << 20, 0);
 *
 *      R_GetCCallable() aborts the R session if Rggml's namespace has not
 *      been loaded yet, so only call these accessors after Rggml is
 *      guaranteed to be attached/loaded (normal LinkingTo + Imports
 *      behavior guarantees this by the time your package's own code runs).
 *
 * Returned/accepted GGML pointers (ggml_context *, ggml_tensor *, ...)
 * follow plain GGML ownership rules, not R's SEXP protection rules: there
 * is no PROTECT/UNPROTECT here, only the usual ggml_init()/ggml_free() and
 * ggml_backend_*_init()/ggml_backend_free() lifecycle pairs.
 */

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <R_ext/Rdynload.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -- upstream identity --------------------------------------------------- */

typedef const char *(*Rggml_version_fun)(void);

/* -- context lifecycle ---------------------------------------------------- */

typedef struct ggml_context *(*Rggml_context_create_fun)(size_t mem_size, int no_alloc);
typedef void (*Rggml_context_free_fun)(struct ggml_context *ctx);
typedef size_t (*Rggml_used_mem_fun)(const struct ggml_context *ctx);
typedef size_t (*Rggml_tensor_overhead_fun)(void);
typedef size_t (*Rggml_graph_overhead_fun)(size_t size);

/* -- tensor creation / introspection -------------------------------------- */

typedef struct ggml_tensor *(*Rggml_new_tensor_fun)(struct ggml_context *ctx, enum ggml_type type,
                                                     int n_dims, const int64_t *ne, void *data);
typedef void *(*Rggml_tensor_data_fun)(const struct ggml_tensor *tensor);
typedef void (*Rggml_tensor_set_data_fun)(struct ggml_tensor *tensor, void *data);
typedef int64_t (*Rggml_tensor_ne_fun)(const struct ggml_tensor *tensor, int dim);
typedef size_t (*Rggml_tensor_nb_fun)(const struct ggml_tensor *tensor, int dim);

typedef struct Rggml_conv_1d_f32_plan Rggml_conv_1d_f32_plan;

/* -- CPU backend ----------------------------------------------------------- */

typedef ggml_backend_t (*Rggml_backend_cpu_init_fun)(void);
typedef void (*Rggml_backend_cpu_set_n_threads_fun)(ggml_backend_t backend,
                                                      int n_threads);
typedef void (*Rggml_backend_free_fun)(ggml_backend_t backend);
typedef int (*Rggml_backend_graph_compute_fun)(ggml_backend_t backend, struct ggml_cgraph *cgraph);

/* -- BLAS backend ---------------------------------------------------------- */
/*
 * GGML's BLAS backend offloads dense F32 (and F16/quantized, after an internal
 * to-float conversion) mul_mat to whatever BLAS the R build links against
 * (reference libRblas, OpenBLAS, MKL, Accelerate, ...). It computes on host
 * memory, so a backend from Rggml_backend_blas_init() is a drop-in `backend`
 * for Rggml_compute_mul_mat()/Rggml_backend_graph_compute() on mul_mat graphs.
 * Returns NULL if the BLAS backend is unavailable. Free with Rggml_backend_free().
 */
typedef ggml_backend_t (*Rggml_backend_blas_init_fun)(void);
typedef void (*Rggml_backend_blas_set_n_threads_fun)(ggml_backend_t backend_blas, int n_threads);

/* -- graph building ---------------------------------------------------------- */

typedef struct ggml_cgraph *(*Rggml_new_graph_fun)(struct ggml_context *ctx, size_t size);
typedef void (*Rggml_build_forward_expand_fun)(struct ggml_cgraph *cgraph, struct ggml_tensor *tensor);

/* -- matrix multiply --------------------------------------------------------- */

typedef struct ggml_tensor *(*Rggml_mul_mat_fun)(struct ggml_context *ctx, struct ggml_tensor *a,
                                                  struct ggml_tensor *b);
typedef struct ggml_tensor *(*Rggml_mul_mat_id_fun)(struct ggml_context *ctx,
                                                     struct ggml_tensor *as,
                                                     struct ggml_tensor *b,
                                                     struct ggml_tensor *ids);
typedef int (*Rggml_compute_mul_mat_fun)(struct ggml_context *ctx, ggml_backend_t backend,
                                         struct ggml_tensor *a, struct ggml_tensor *b,
                                         float *out_f32, double *out_f64);

/* -- quantization ------------------------------------------------------------- */

/* Quantize `nrows` rows of `n_per_row` f32 values into `type`'s block format
 * (e.g. GGML_TYPE_Q4_K); `dst` must hold Rggml_row_size(type, n_per_row) *
 * nrows bytes. Returns bytes written. The output is byte-compatible with a
 * GGUF tensor of that type. */
typedef size_t (*Rggml_quantize_fun)(enum ggml_type type, const float *src, void *dst,
                                     int64_t nrows, int64_t n_per_row);

/* Decode `n` elements (a whole number of blocks) of a quantized payload to f32
 * through GGML's type-traits to_float - the authoritative reference
 * dequantizer for every GGUF type. The caller must have initialized the CPU
 * backend once (Rggml_backend_cpu_init) so the fp16 table used to read block
 * scales is populated. Returns 0 on success. */
typedef int (*Rggml_dequantize_fun)(enum ggml_type type, const void *src,
                                    float *dst, int64_t n);
typedef int (*Rggml_can_dequantize_fun)(enum ggml_type type);
typedef int (*Rggml_dequantize_double_fun)(enum ggml_type type,
                                           const void *src, double *dst,
                                           int64_t n);

/* Official GGUF reader/writer owned by Rggml. Borrowed strings and data
 * pointers remain valid until their opaque context is closed. */
struct Rggml_gguf_context;
struct Rggml_gguf_writer;
struct Rggml_gguf_kv {
    const char *key;
    int type;
    int is_array;
    size_t n;
    const void *data;
};
struct Rggml_gguf_tensor {
    const char *name;
    const char *type_name;
    enum ggml_type type;
    int n_dims;
    int64_t ne[GGML_MAX_DIMS];
    int64_t n_elements;
    size_t nbytes;
    size_t offset;
};

typedef struct Rggml_gguf_context *(*Rggml_gguf_open_fun)(const char *path);
typedef void (*Rggml_gguf_close_fun)(struct Rggml_gguf_context *ctx);
typedef uint32_t (*Rggml_gguf_version_fun)(const struct Rggml_gguf_context *ctx);
typedef size_t (*Rggml_gguf_data_offset_fun)(const struct Rggml_gguf_context *ctx);
typedef int64_t (*Rggml_gguf_n_kv_fun)(const struct Rggml_gguf_context *ctx);
typedef int (*Rggml_gguf_kv_fun)(const struct Rggml_gguf_context *ctx,
                                 int64_t id, struct Rggml_gguf_kv *out);
typedef const char *(*Rggml_gguf_kv_string_fun)(
    const struct Rggml_gguf_context *ctx, int64_t id, size_t index);
typedef int64_t (*Rggml_gguf_n_tensors_fun)(const struct Rggml_gguf_context *ctx);
typedef int64_t (*Rggml_gguf_find_tensor_fun)(
    const struct Rggml_gguf_context *ctx, const char *name);
typedef int (*Rggml_gguf_tensor_fun)(const struct Rggml_gguf_context *ctx,
                                     int64_t id,
                                     struct Rggml_gguf_tensor *out);

typedef struct Rggml_gguf_writer *(*Rggml_gguf_writer_open_fun)(void);
typedef void (*Rggml_gguf_writer_close_fun)(struct Rggml_gguf_writer *ctx);
typedef int (*Rggml_gguf_writer_set_string_fun)(
    struct Rggml_gguf_writer *ctx, const char *key, const char *value);
typedef int (*Rggml_gguf_writer_set_strings_fun)(
    struct Rggml_gguf_writer *ctx, const char *key,
    const char **values, size_t n);
typedef int (*Rggml_gguf_writer_set_f64_fun)(struct Rggml_gguf_writer *ctx,
                                             const char *key, double value);
typedef int (*Rggml_gguf_writer_set_f64s_fun)(
    struct Rggml_gguf_writer *ctx, const char *key,
    const double *values, size_t n);
typedef int (*Rggml_gguf_writer_add_f32_fun)(
    struct Rggml_gguf_writer *ctx, const char *name, int n_dims,
    const int64_t *ne, const double *data);
typedef int (*Rggml_gguf_writer_write_fun)(struct Rggml_gguf_writer *ctx,
                                           const char *path);

/* -- graph ops --------------------------------------------------------------- */
/*
 * The ggml ops a transformer forward pass composes. Scalar parameters cross
 * the boundary as double. Rggml_rope wraps ggml_rope_ext with YaRN off
 * (ext_factor 0, attn_factor 1, freq_scale 1, n_ctx_orig 0); `pos` is an I32
 * positions tensor; mode is GGML_ROPE_TYPE_NORMAL (0) or _NEOX (2).
 */
typedef struct ggml_tensor *(*Rggml_get_rows_fun)(struct ggml_context *ctx,
                                                   struct ggml_tensor *a,
                                                   struct ggml_tensor *b);
typedef struct ggml_tensor *(*Rggml_rms_norm_fun)(struct ggml_context *ctx,
                                                   struct ggml_tensor *a, double eps);
typedef struct ggml_tensor *(*Rggml_l2_norm_fun)(struct ggml_context *ctx,
                                                  struct ggml_tensor *a, double eps);
typedef struct ggml_tensor *(*Rggml_mul_fun)(struct ggml_context *ctx,
                                              struct ggml_tensor *a,
                                              struct ggml_tensor *b);
typedef struct ggml_tensor *(*Rggml_add_fun)(struct ggml_context *ctx,
                                              struct ggml_tensor *a,
                                              struct ggml_tensor *b);
typedef struct ggml_tensor *(*Rggml_div_fun)(struct ggml_context *ctx,
                                              struct ggml_tensor *a,
                                              struct ggml_tensor *b);
typedef struct ggml_tensor *(*Rggml_silu_fun)(struct ggml_context *ctx,
                                               struct ggml_tensor *a);
typedef struct ggml_tensor *(*Rggml_geglu_fun)(struct ggml_context *ctx,
                                                struct ggml_tensor *gate,
                                                struct ggml_tensor *up);
typedef struct ggml_tensor *(*Rggml_sigmoid_fun)(struct ggml_context *ctx,
                                                  struct ggml_tensor *a);
typedef struct ggml_tensor *(*Rggml_softplus_fun)(struct ggml_context *ctx,
                                                   struct ggml_tensor *a);
typedef struct ggml_tensor *(*Rggml_scale_fun)(struct ggml_context *ctx,
                                                struct ggml_tensor *a, double s);
typedef struct ggml_tensor *(*Rggml_sum_rows_fun)(struct ggml_context *ctx,
                                                   struct ggml_tensor *a);
typedef struct ggml_tensor *(*Rggml_clamp_fun)(struct ggml_context *ctx,
                                                struct ggml_tensor *a,
                                                double min, double max);
typedef struct ggml_tensor *(*Rggml_argsort_top_k_fun)(struct ggml_context *ctx,
                                                        struct ggml_tensor *a,
                                                        int k);
typedef struct ggml_tensor *(*Rggml_concat_fun)(struct ggml_context *ctx,
                                                 struct ggml_tensor *a,
                                                 struct ggml_tensor *b,
                                                 int dim);
typedef struct ggml_tensor *(*Rggml_ssm_conv_fun)(struct ggml_context *ctx,
                                                   struct ggml_tensor *sx,
                                                   struct ggml_tensor *kernel);
typedef struct ggml_tensor *(*Rggml_conv_1d_fun)(struct ggml_context *ctx,
                                                  struct ggml_tensor *kernel,
                                                  struct ggml_tensor *data,
                                                  int stride, int padding,
                                                  int dilation);
/* Persistent CPU-only direct F32 conv1d plan. Creation validates and copies
 * GGUF-layout [K, IC, OC] weights plus optional OC bias into owned packed
 * storage. The caller owns the returned plan and must destroy it only after
 * every graph built from it and its backend buffer are released. The plan is
 * immutable and contains no R object, so GGML worker callbacks need no R API.
 * `max_output_channels` is an explicit per-plan bound. */
typedef Rggml_conv_1d_f32_plan *(*Rggml_conv_1d_f32_plan_create_fun)(
    const float *kernel, size_t kernel_bytes, const float *bias,
    size_t bias_bytes, int64_t kernel_size, int64_t input_channels,
    int64_t output_channels, int64_t stride, int64_t padding,
    int64_t dilation, int64_t max_output_channels);
typedef void (*Rggml_conv_1d_f32_plan_destroy_fun)(Rggml_conv_1d_f32_plan *plan);
typedef struct ggml_tensor *(*Rggml_conv_1d_f32_plan_apply_fun)(
    struct ggml_context *ctx, const Rggml_conv_1d_f32_plan *plan,
    struct ggml_tensor *input);
typedef struct ggml_tensor *(*Rggml_leaky_relu_fun)(struct ggml_context *ctx,
                                                     struct ggml_tensor *a,
                                                     double slope);
/* Leaky ReLU spread over every CPU worker. Upstream GGML computes this
 * operator on thread zero alone, which stalls every other worker at the next
 * barrier. Returns NULL when the input is not a contiguous F32 tensor, and a
 * device graph must keep the official operator because a custom CPU callback
 * is not a device operation. */
typedef struct ggml_tensor *(*Rggml_leaky_relu_cpu_fun)(struct ggml_context *ctx,
                                                         struct ggml_tensor *a,
                                                         double slope);
typedef struct ggml_tensor *(*Rggml_soft_max_fun)(struct ggml_context *ctx,
                                                   struct ggml_tensor *a);
typedef struct ggml_tensor *(*Rggml_soft_max_ext_fun)(struct ggml_context *ctx,
                                                       struct ggml_tensor *a,
                                                       struct ggml_tensor *mask,
                                                       double scale,
                                                       double max_bias);
typedef struct ggml_tensor *(*Rggml_diag_mask_inf_fun)(struct ggml_context *ctx,
                                                        struct ggml_tensor *a,
                                                        int n_past);
typedef struct ggml_tensor *(*Rggml_rope_fun)(struct ggml_context *ctx,
                                               struct ggml_tensor *a,
                                               struct ggml_tensor *pos,
                                               int n_dims, int mode, double freq_base);
typedef struct ggml_tensor *(*Rggml_rope_multi_fun)(struct ggml_context *ctx,
                                                     struct ggml_tensor *a,
                                                     struct ggml_tensor *pos,
                                                     int n_dims,
                                                     const int sections[GGML_MROPE_SECTIONS],
                                                     int mode, double freq_base);
typedef struct ggml_tensor *(*Rggml_gated_delta_net_fun)(
    struct ggml_context *ctx, struct ggml_tensor *q, struct ggml_tensor *k,
    struct ggml_tensor *v, struct ggml_tensor *gate,
    struct ggml_tensor *beta, struct ggml_tensor *state, int64_t snapshots);
typedef struct ggml_tensor *(*Rggml_reshape_2d_fun)(struct ggml_context *ctx,
                                                     struct ggml_tensor *a,
                                                     int64_t ne0, int64_t ne1);
typedef struct ggml_tensor *(*Rggml_reshape_3d_fun)(struct ggml_context *ctx,
                                                     struct ggml_tensor *a,
                                                     int64_t ne0, int64_t ne1, int64_t ne2);
typedef struct ggml_tensor *(*Rggml_reshape_4d_fun)(struct ggml_context *ctx,
                                                     struct ggml_tensor *a,
                                                     int64_t ne0, int64_t ne1,
                                                     int64_t ne2, int64_t ne3);
typedef struct ggml_tensor *(*Rggml_permute_fun)(struct ggml_context *ctx,
                                                  struct ggml_tensor *a,
                                                  int axis0, int axis1, int axis2, int axis3);
typedef struct ggml_tensor *(*Rggml_cont_fun)(struct ggml_context *ctx,
                                               struct ggml_tensor *a);
typedef struct ggml_tensor *(*Rggml_transpose_fun)(struct ggml_context *ctx,
                                                    struct ggml_tensor *a);

/* -- Vulkan backend ---------------------------------------------------------- */
/* Always resolvable; reports 0 devices and refuses to init unless Rggml was
 * built with --with-vulkan. Free with Rggml_backend_free(). */
typedef int (*Rggml_backend_vulkan_device_count_fun)(void);
typedef ggml_backend_t (*Rggml_backend_vulkan_init_fun)(int device);
typedef int (*Rggml_backend_vulkan_device_description_fun)(int device, char *buf,
                                                            size_t buf_size);

/* -- CUDA backend ---------------------------------------------------------- */
/* Always resolvable; reports 0 devices and refuses to init unless Rggml was
 * built with --with-cuda. Free with Rggml_backend_free(). */
typedef int (*Rggml_backend_cuda_device_count_fun)(void);
typedef ggml_backend_t (*Rggml_backend_cuda_init_fun)(int device);
typedef int (*Rggml_backend_cuda_device_description_fun)(int device, char *buf,
                                                          size_t buf_size);

/* -- device-buffer residency ------------------------------------------------- */
/* The backend-agnostic allocate/upload/compute/download path. CPU and BLAS
 * compute on host memory, but a GPU backend's tensors must live in device
 * memory: allocate every tensor of a no_alloc context into one backend buffer,
 * upload inputs, compute, download the result. Free the buffer before the ctx. */
typedef ggml_backend_buffer_t (*Rggml_backend_alloc_ctx_tensors_fun)(struct ggml_context *ctx,
                                                                      ggml_backend_t backend);
typedef void (*Rggml_backend_buffer_free_fun)(ggml_backend_buffer_t buffer);
typedef void (*Rggml_backend_tensor_set_fun)(struct ggml_tensor *tensor, const void *data,
                                              size_t offset, size_t size);
typedef void (*Rggml_backend_tensor_get_fun)(const struct ggml_tensor *tensor, void *data,
                                              size_t offset, size_t size);

/* -- views and copies -------------------------------------------------------- */
/* Strided views into a tensor (offsets/strides in bytes, as in ggml) and the
 * copy op - the building blocks of a KV cache: cpy nodes write new K/V into
 * views of a persistent cache tensor, expanded into the graph ahead of the
 * attention nodes that read other views of the same cache. */
typedef struct ggml_tensor *(*Rggml_view_1d_fun)(struct ggml_context *ctx,
                                                  struct ggml_tensor *a,
                                                  int64_t ne0, size_t offset);
typedef struct ggml_tensor *(*Rggml_view_2d_fun)(struct ggml_context *ctx,
                                                  struct ggml_tensor *a,
                                                  int64_t ne0, int64_t ne1,
                                                  size_t nb1, size_t offset);
typedef struct ggml_tensor *(*Rggml_view_3d_fun)(struct ggml_context *ctx,
                                                  struct ggml_tensor *a,
                                                  int64_t ne0, int64_t ne1, int64_t ne2,
                                                  size_t nb1, size_t nb2, size_t offset);
typedef struct ggml_tensor *(*Rggml_view_4d_fun)(struct ggml_context *ctx,
                                                  struct ggml_tensor *a,
                                                  int64_t ne0, int64_t ne1,
                                                  int64_t ne2, int64_t ne3,
                                                  size_t nb1, size_t nb2,
                                                  size_t nb3, size_t offset);
typedef struct ggml_tensor *(*Rggml_cpy_fun)(struct ggml_context *ctx,
                                              struct ggml_tensor *a,
                                              struct ggml_tensor *b);

/* -- type/size introspection -------------------------------------------------- */

typedef size_t (*Rggml_type_size_fun)(enum ggml_type type);
typedef size_t (*Rggml_row_size_fun)(enum ggml_type type, int64_t ne);
typedef int64_t (*Rggml_blck_size_fun)(enum ggml_type type);
typedef size_t (*Rggml_nbytes_fun)(const struct ggml_tensor *tensor);
typedef int64_t (*Rggml_nelements_fun)(const struct ggml_tensor *tensor);
typedef const char *(*Rggml_type_name_fun)(enum ggml_type type);

/* -------------------------------------------------------------------------- */
/* R_GetCCallable() accessors                                                 */
/* -------------------------------------------------------------------------- */

static inline Rggml_version_fun Rggml_version_ptr(void)
{
    return (Rggml_version_fun) R_GetCCallable("Rggml", "Rggml_version");
}

static inline Rggml_context_create_fun Rggml_context_create_ptr(void)
{
    return (Rggml_context_create_fun) R_GetCCallable("Rggml", "Rggml_context_create");
}

static inline Rggml_context_free_fun Rggml_context_free_ptr(void)
{
    return (Rggml_context_free_fun) R_GetCCallable("Rggml", "Rggml_context_free");
}

static inline Rggml_used_mem_fun Rggml_used_mem_ptr(void)
{
    return (Rggml_used_mem_fun) R_GetCCallable("Rggml", "Rggml_used_mem");
}

static inline Rggml_tensor_overhead_fun Rggml_tensor_overhead_ptr(void)
{
    return (Rggml_tensor_overhead_fun) R_GetCCallable("Rggml", "Rggml_tensor_overhead");
}

static inline Rggml_graph_overhead_fun Rggml_graph_overhead_ptr(void)
{
    return (Rggml_graph_overhead_fun) R_GetCCallable("Rggml", "Rggml_graph_overhead");
}

static inline Rggml_new_tensor_fun Rggml_new_tensor_ptr(void)
{
    return (Rggml_new_tensor_fun) R_GetCCallable("Rggml", "Rggml_new_tensor");
}

static inline Rggml_tensor_data_fun Rggml_tensor_data_ptr(void)
{
    return (Rggml_tensor_data_fun) R_GetCCallable("Rggml", "Rggml_tensor_data");
}

static inline Rggml_tensor_set_data_fun Rggml_tensor_set_data_ptr(void)
{
    return (Rggml_tensor_set_data_fun) R_GetCCallable("Rggml", "Rggml_tensor_set_data");
}

static inline Rggml_tensor_ne_fun Rggml_tensor_ne_ptr(void)
{
    return (Rggml_tensor_ne_fun) R_GetCCallable("Rggml", "Rggml_tensor_ne");
}

static inline Rggml_tensor_nb_fun Rggml_tensor_nb_ptr(void)
{
    return (Rggml_tensor_nb_fun) R_GetCCallable("Rggml", "Rggml_tensor_nb");
}

static inline Rggml_backend_cpu_init_fun Rggml_backend_cpu_init_ptr(void)
{
    return (Rggml_backend_cpu_init_fun) R_GetCCallable("Rggml", "Rggml_backend_cpu_init");
}

static inline Rggml_backend_cpu_set_n_threads_fun Rggml_backend_cpu_set_n_threads_ptr(void)
{
    return (Rggml_backend_cpu_set_n_threads_fun)
        R_GetCCallable("Rggml", "Rggml_backend_cpu_set_n_threads");
}

static inline Rggml_backend_free_fun Rggml_backend_free_ptr(void)
{
    return (Rggml_backend_free_fun) R_GetCCallable("Rggml", "Rggml_backend_free");
}

static inline Rggml_backend_graph_compute_fun Rggml_backend_graph_compute_ptr(void)
{
    return (Rggml_backend_graph_compute_fun) R_GetCCallable("Rggml", "Rggml_backend_graph_compute");
}

static inline Rggml_backend_blas_init_fun Rggml_backend_blas_init_ptr(void)
{
    return (Rggml_backend_blas_init_fun) R_GetCCallable("Rggml", "Rggml_backend_blas_init");
}

static inline Rggml_backend_blas_set_n_threads_fun Rggml_backend_blas_set_n_threads_ptr(void)
{
    return (Rggml_backend_blas_set_n_threads_fun) R_GetCCallable("Rggml", "Rggml_backend_blas_set_n_threads");
}

static inline Rggml_new_graph_fun Rggml_new_graph_ptr(void)
{
    return (Rggml_new_graph_fun) R_GetCCallable("Rggml", "Rggml_new_graph");
}

static inline Rggml_build_forward_expand_fun Rggml_build_forward_expand_ptr(void)
{
    return (Rggml_build_forward_expand_fun) R_GetCCallable("Rggml", "Rggml_build_forward_expand");
}

static inline Rggml_mul_mat_fun Rggml_mul_mat_ptr(void)
{
    return (Rggml_mul_mat_fun) R_GetCCallable("Rggml", "Rggml_mul_mat");
}

static inline Rggml_mul_mat_id_fun Rggml_mul_mat_id_ptr(void)
{
    return (Rggml_mul_mat_id_fun) R_GetCCallable("Rggml", "Rggml_mul_mat_id");
}

static inline Rggml_compute_mul_mat_fun Rggml_compute_mul_mat_ptr(void)
{
    return (Rggml_compute_mul_mat_fun) R_GetCCallable("Rggml", "Rggml_compute_mul_mat");
}

static inline Rggml_quantize_fun Rggml_quantize_ptr(void)
{
    return (Rggml_quantize_fun) R_GetCCallable("Rggml", "Rggml_quantize");
}

static inline Rggml_dequantize_fun Rggml_dequantize_ptr(void)
{
    return (Rggml_dequantize_fun) R_GetCCallable("Rggml", "Rggml_dequantize");
}

static inline Rggml_get_rows_fun Rggml_get_rows_ptr(void)
{
    return (Rggml_get_rows_fun) R_GetCCallable("Rggml", "Rggml_get_rows");
}

static inline Rggml_rms_norm_fun Rggml_rms_norm_ptr(void)
{
    return (Rggml_rms_norm_fun) R_GetCCallable("Rggml", "Rggml_rms_norm");
}

static inline Rggml_l2_norm_fun Rggml_l2_norm_ptr(void)
{
    return (Rggml_l2_norm_fun) R_GetCCallable("Rggml", "Rggml_l2_norm");
}

static inline Rggml_mul_fun Rggml_mul_ptr(void)
{
    return (Rggml_mul_fun) R_GetCCallable("Rggml", "Rggml_mul");
}

static inline Rggml_add_fun Rggml_add_ptr(void)
{
    return (Rggml_add_fun) R_GetCCallable("Rggml", "Rggml_add");
}

static inline Rggml_div_fun Rggml_div_ptr(void)
{
    return (Rggml_div_fun) R_GetCCallable("Rggml", "Rggml_div");
}

static inline Rggml_silu_fun Rggml_silu_ptr(void)
{
    return (Rggml_silu_fun) R_GetCCallable("Rggml", "Rggml_silu");
}

static inline Rggml_geglu_fun Rggml_geglu_ptr(void)
{
    return (Rggml_geglu_fun) R_GetCCallable("Rggml", "Rggml_geglu");
}

static inline Rggml_sigmoid_fun Rggml_sigmoid_ptr(void)
{
    return (Rggml_sigmoid_fun) R_GetCCallable("Rggml", "Rggml_sigmoid");
}

static inline Rggml_softplus_fun Rggml_softplus_ptr(void)
{
    return (Rggml_softplus_fun) R_GetCCallable("Rggml", "Rggml_softplus");
}

static inline Rggml_scale_fun Rggml_scale_ptr(void)
{
    return (Rggml_scale_fun) R_GetCCallable("Rggml", "Rggml_scale");
}

static inline Rggml_sum_rows_fun Rggml_sum_rows_ptr(void)
{
    return (Rggml_sum_rows_fun) R_GetCCallable("Rggml", "Rggml_sum_rows");
}

static inline Rggml_clamp_fun Rggml_clamp_ptr(void)
{
    return (Rggml_clamp_fun) R_GetCCallable("Rggml", "Rggml_clamp");
}

static inline Rggml_argsort_top_k_fun Rggml_argsort_top_k_ptr(void)
{
    return (Rggml_argsort_top_k_fun)
        R_GetCCallable("Rggml", "Rggml_argsort_top_k");
}

static inline Rggml_concat_fun Rggml_concat_ptr(void)
{
    return (Rggml_concat_fun) R_GetCCallable("Rggml", "Rggml_concat");
}

static inline Rggml_ssm_conv_fun Rggml_ssm_conv_ptr(void)
{
    return (Rggml_ssm_conv_fun) R_GetCCallable("Rggml", "Rggml_ssm_conv");
}

static inline Rggml_conv_1d_fun Rggml_conv_1d_ptr(void)
{
    return (Rggml_conv_1d_fun) R_GetCCallable("Rggml", "Rggml_conv_1d");
}

static inline Rggml_conv_1d_f32_plan_create_fun Rggml_conv_1d_f32_plan_create_ptr(void)
{
    return (Rggml_conv_1d_f32_plan_create_fun)
        R_GetCCallable("Rggml", "Rggml_conv_1d_f32_plan_create");
}

static inline Rggml_conv_1d_f32_plan_destroy_fun Rggml_conv_1d_f32_plan_destroy_ptr(void)
{
    return (Rggml_conv_1d_f32_plan_destroy_fun)
        R_GetCCallable("Rggml", "Rggml_conv_1d_f32_plan_destroy");
}

static inline Rggml_conv_1d_f32_plan_apply_fun Rggml_conv_1d_f32_plan_apply_ptr(void)
{
    return (Rggml_conv_1d_f32_plan_apply_fun)
        R_GetCCallable("Rggml", "Rggml_conv_1d_f32_plan_apply");
}

static inline Rggml_leaky_relu_fun Rggml_leaky_relu_ptr(void)
{
    return (Rggml_leaky_relu_fun)
        R_GetCCallable("Rggml", "Rggml_leaky_relu");
}

static inline Rggml_leaky_relu_cpu_fun Rggml_leaky_relu_cpu_ptr(void)
{
    return (Rggml_leaky_relu_cpu_fun)
        R_GetCCallable("Rggml", "Rggml_leaky_relu_cpu");
}

static inline Rggml_soft_max_fun Rggml_soft_max_ptr(void)
{
    return (Rggml_soft_max_fun) R_GetCCallable("Rggml", "Rggml_soft_max");
}

static inline Rggml_soft_max_ext_fun Rggml_soft_max_ext_ptr(void)
{
    return (Rggml_soft_max_ext_fun)
        R_GetCCallable("Rggml", "Rggml_soft_max_ext");
}

static inline Rggml_diag_mask_inf_fun Rggml_diag_mask_inf_ptr(void)
{
    return (Rggml_diag_mask_inf_fun) R_GetCCallable("Rggml", "Rggml_diag_mask_inf");
}

static inline Rggml_rope_fun Rggml_rope_ptr(void)
{
    return (Rggml_rope_fun) R_GetCCallable("Rggml", "Rggml_rope");
}

static inline Rggml_rope_multi_fun Rggml_rope_multi_ptr(void)
{
    return (Rggml_rope_multi_fun)
        R_GetCCallable("Rggml", "Rggml_rope_multi");
}

static inline Rggml_gated_delta_net_fun Rggml_gated_delta_net_ptr(void)
{
    return (Rggml_gated_delta_net_fun)
        R_GetCCallable("Rggml", "Rggml_gated_delta_net");
}

static inline Rggml_reshape_2d_fun Rggml_reshape_2d_ptr(void)
{
    return (Rggml_reshape_2d_fun) R_GetCCallable("Rggml", "Rggml_reshape_2d");
}

static inline Rggml_reshape_3d_fun Rggml_reshape_3d_ptr(void)
{
    return (Rggml_reshape_3d_fun) R_GetCCallable("Rggml", "Rggml_reshape_3d");
}

static inline Rggml_reshape_4d_fun Rggml_reshape_4d_ptr(void)
{
    return (Rggml_reshape_4d_fun)
        R_GetCCallable("Rggml", "Rggml_reshape_4d");
}

static inline Rggml_permute_fun Rggml_permute_ptr(void)
{
    return (Rggml_permute_fun) R_GetCCallable("Rggml", "Rggml_permute");
}

static inline Rggml_cont_fun Rggml_cont_ptr(void)
{
    return (Rggml_cont_fun) R_GetCCallable("Rggml", "Rggml_cont");
}

static inline Rggml_transpose_fun Rggml_transpose_ptr(void)
{
    return (Rggml_transpose_fun) R_GetCCallable("Rggml", "Rggml_transpose");
}

static inline Rggml_backend_vulkan_device_count_fun Rggml_backend_vulkan_device_count_ptr(void)
{
    return (Rggml_backend_vulkan_device_count_fun) R_GetCCallable("Rggml", "Rggml_backend_vulkan_device_count");
}

static inline Rggml_backend_vulkan_init_fun Rggml_backend_vulkan_init_ptr(void)
{
    return (Rggml_backend_vulkan_init_fun) R_GetCCallable("Rggml", "Rggml_backend_vulkan_init");
}

static inline Rggml_backend_vulkan_device_description_fun Rggml_backend_vulkan_device_description_ptr(void)
{
    return (Rggml_backend_vulkan_device_description_fun) R_GetCCallable("Rggml", "Rggml_backend_vulkan_device_description");
}

static inline Rggml_backend_cuda_device_count_fun Rggml_backend_cuda_device_count_ptr(void)
{
    return (Rggml_backend_cuda_device_count_fun) R_GetCCallable("Rggml", "Rggml_backend_cuda_device_count");
}

static inline Rggml_backend_cuda_init_fun Rggml_backend_cuda_init_ptr(void)
{
    return (Rggml_backend_cuda_init_fun) R_GetCCallable("Rggml", "Rggml_backend_cuda_init");
}

static inline Rggml_backend_cuda_device_description_fun Rggml_backend_cuda_device_description_ptr(void)
{
    return (Rggml_backend_cuda_device_description_fun) R_GetCCallable("Rggml", "Rggml_backend_cuda_device_description");
}

static inline Rggml_backend_alloc_ctx_tensors_fun Rggml_backend_alloc_ctx_tensors_ptr(void)
{
    return (Rggml_backend_alloc_ctx_tensors_fun) R_GetCCallable("Rggml", "Rggml_backend_alloc_ctx_tensors");
}

static inline Rggml_backend_buffer_free_fun Rggml_backend_buffer_free_ptr(void)
{
    return (Rggml_backend_buffer_free_fun) R_GetCCallable("Rggml", "Rggml_backend_buffer_free");
}

static inline Rggml_backend_tensor_set_fun Rggml_backend_tensor_set_ptr(void)
{
    return (Rggml_backend_tensor_set_fun) R_GetCCallable("Rggml", "Rggml_backend_tensor_set");
}

static inline Rggml_backend_tensor_get_fun Rggml_backend_tensor_get_ptr(void)
{
    return (Rggml_backend_tensor_get_fun) R_GetCCallable("Rggml", "Rggml_backend_tensor_get");
}

static inline Rggml_view_1d_fun Rggml_view_1d_ptr(void)
{
    return (Rggml_view_1d_fun) R_GetCCallable("Rggml", "Rggml_view_1d");
}

static inline Rggml_view_2d_fun Rggml_view_2d_ptr(void)
{
    return (Rggml_view_2d_fun) R_GetCCallable("Rggml", "Rggml_view_2d");
}

static inline Rggml_view_3d_fun Rggml_view_3d_ptr(void)
{
    return (Rggml_view_3d_fun) R_GetCCallable("Rggml", "Rggml_view_3d");
}

static inline Rggml_view_4d_fun Rggml_view_4d_ptr(void)
{
    return (Rggml_view_4d_fun) R_GetCCallable("Rggml", "Rggml_view_4d");
}

static inline Rggml_cpy_fun Rggml_cpy_ptr(void)
{
    return (Rggml_cpy_fun) R_GetCCallable("Rggml", "Rggml_cpy");
}

static inline Rggml_type_size_fun Rggml_type_size_ptr(void)
{
    return (Rggml_type_size_fun) R_GetCCallable("Rggml", "Rggml_type_size");
}

static inline Rggml_row_size_fun Rggml_row_size_ptr(void)
{
    return (Rggml_row_size_fun) R_GetCCallable("Rggml", "Rggml_row_size");
}

static inline Rggml_blck_size_fun Rggml_blck_size_ptr(void)
{
    return (Rggml_blck_size_fun) R_GetCCallable("Rggml", "Rggml_blck_size");
}

static inline Rggml_nbytes_fun Rggml_nbytes_ptr(void)
{
    return (Rggml_nbytes_fun) R_GetCCallable("Rggml", "Rggml_nbytes");
}

static inline Rggml_nelements_fun Rggml_nelements_ptr(void)
{
    return (Rggml_nelements_fun) R_GetCCallable("Rggml", "Rggml_nelements");
}

static inline Rggml_type_name_fun Rggml_type_name_ptr(void)
{
    return (Rggml_type_name_fun) R_GetCCallable("Rggml", "Rggml_type_name");
}

static inline Rggml_dequantize_double_fun Rggml_dequantize_double_ptr(void)
{
    return (Rggml_dequantize_double_fun)
        R_GetCCallable("Rggml", "Rggml_dequantize_double");
}

static inline Rggml_can_dequantize_fun Rggml_can_dequantize_ptr(void)
{
    return (Rggml_can_dequantize_fun)
        R_GetCCallable("Rggml", "Rggml_can_dequantize");
}

static inline Rggml_gguf_open_fun Rggml_gguf_open_ptr(void)
{
    return (Rggml_gguf_open_fun) R_GetCCallable("Rggml", "Rggml_gguf_open");
}
static inline Rggml_gguf_close_fun Rggml_gguf_close_ptr(void)
{
    return (Rggml_gguf_close_fun) R_GetCCallable("Rggml", "Rggml_gguf_close");
}
static inline Rggml_gguf_version_fun Rggml_gguf_version_ptr(void)
{
    return (Rggml_gguf_version_fun) R_GetCCallable("Rggml", "Rggml_gguf_version");
}
static inline Rggml_gguf_data_offset_fun Rggml_gguf_data_offset_ptr(void)
{
    return (Rggml_gguf_data_offset_fun)
        R_GetCCallable("Rggml", "Rggml_gguf_data_offset");
}
static inline Rggml_gguf_n_kv_fun Rggml_gguf_n_kv_ptr(void)
{
    return (Rggml_gguf_n_kv_fun) R_GetCCallable("Rggml", "Rggml_gguf_n_kv");
}
static inline Rggml_gguf_kv_fun Rggml_gguf_kv_ptr(void)
{
    return (Rggml_gguf_kv_fun) R_GetCCallable("Rggml", "Rggml_gguf_kv");
}
static inline Rggml_gguf_kv_string_fun Rggml_gguf_kv_string_ptr(void)
{
    return (Rggml_gguf_kv_string_fun)
        R_GetCCallable("Rggml", "Rggml_gguf_kv_string");
}
static inline Rggml_gguf_n_tensors_fun Rggml_gguf_n_tensors_ptr(void)
{
    return (Rggml_gguf_n_tensors_fun)
        R_GetCCallable("Rggml", "Rggml_gguf_n_tensors");
}
static inline Rggml_gguf_find_tensor_fun Rggml_gguf_find_tensor_ptr(void)
{
    return (Rggml_gguf_find_tensor_fun)
        R_GetCCallable("Rggml", "Rggml_gguf_find_tensor");
}
static inline Rggml_gguf_tensor_fun Rggml_gguf_tensor_ptr(void)
{
    return (Rggml_gguf_tensor_fun)
        R_GetCCallable("Rggml", "Rggml_gguf_tensor");
}
static inline Rggml_gguf_writer_open_fun Rggml_gguf_writer_open_ptr(void)
{
    return (Rggml_gguf_writer_open_fun)
        R_GetCCallable("Rggml", "Rggml_gguf_writer_open");
}
static inline Rggml_gguf_writer_close_fun Rggml_gguf_writer_close_ptr(void)
{
    return (Rggml_gguf_writer_close_fun)
        R_GetCCallable("Rggml", "Rggml_gguf_writer_close");
}
static inline Rggml_gguf_writer_set_string_fun Rggml_gguf_writer_set_string_ptr(void)
{
    return (Rggml_gguf_writer_set_string_fun)
        R_GetCCallable("Rggml", "Rggml_gguf_writer_set_string");
}
static inline Rggml_gguf_writer_set_strings_fun Rggml_gguf_writer_set_strings_ptr(void)
{
    return (Rggml_gguf_writer_set_strings_fun)
        R_GetCCallable("Rggml", "Rggml_gguf_writer_set_strings");
}
static inline Rggml_gguf_writer_set_f64_fun Rggml_gguf_writer_set_f64_ptr(void)
{
    return (Rggml_gguf_writer_set_f64_fun)
        R_GetCCallable("Rggml", "Rggml_gguf_writer_set_f64");
}
static inline Rggml_gguf_writer_set_f64s_fun Rggml_gguf_writer_set_f64s_ptr(void)
{
    return (Rggml_gguf_writer_set_f64s_fun)
        R_GetCCallable("Rggml", "Rggml_gguf_writer_set_f64s");
}
static inline Rggml_gguf_writer_add_f32_fun Rggml_gguf_writer_add_f32_ptr(void)
{
    return (Rggml_gguf_writer_add_f32_fun)
        R_GetCCallable("Rggml", "Rggml_gguf_writer_add_f32");
}
static inline Rggml_gguf_writer_write_fun Rggml_gguf_writer_write_ptr(void)
{
    return (Rggml_gguf_writer_write_fun)
        R_GetCCallable("Rggml", "Rggml_gguf_writer_write");
}

#ifdef __cplusplus
}
#endif

#endif /* RGGML_API_PUBLIC_H */
