#ifndef RGGML_API_H
#define RGGML_API_H

/*
 * rggml_api.h - internal declarations for Rggml's C-callable implementation
 * functions (src/rggml_api.c). Shared between rggml_init.c (registration)
 * and rggml_test.c (the .Call smoke-test routine).
 *
 * This is NOT the installed header - downstream packages use
 * inst/include/Rggml.h instead, which declares the same functions behind
 * R_GetCCallable() lookups.
 */

#include <ggml.h>
#include <ggml-backend.h>

#ifdef __cplusplus
extern "C" {
#endif

const char *Rggml_version(void);

struct ggml_context *Rggml_context_create(size_t mem_size, int no_alloc);
void Rggml_context_free(struct ggml_context *ctx);
size_t Rggml_used_mem(const struct ggml_context *ctx);
size_t Rggml_tensor_overhead(void);
size_t Rggml_graph_overhead(size_t size);

struct ggml_tensor *Rggml_new_tensor(struct ggml_context *ctx, enum ggml_type type,
                                      int n_dims, const int64_t *ne, void *data);
void *Rggml_tensor_data(const struct ggml_tensor *tensor);
void Rggml_tensor_set_data(struct ggml_tensor *tensor, void *data);
int64_t Rggml_tensor_ne(const struct ggml_tensor *tensor, int dim);
size_t Rggml_tensor_nb(const struct ggml_tensor *tensor, int dim);

ggml_backend_t Rggml_backend_cpu_init(void);
void Rggml_backend_cpu_set_n_threads(ggml_backend_t backend, int n_threads);
void Rggml_backend_free(ggml_backend_t backend);
int Rggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph *cgraph);

ggml_backend_t Rggml_backend_blas_init(void);
void Rggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, int n_threads);

/* Vulkan backend; reports 0 devices unless built --with-vulkan. */
int Rggml_backend_vulkan_device_count(void);
ggml_backend_t Rggml_backend_vulkan_init(int device);
int Rggml_backend_vulkan_device_description(int device, char *buf, size_t buf_size);

/* CUDA backend; reports 0 devices unless built --with-cuda */
int Rggml_backend_cuda_device_count(void);
int Rggml_cuda_touched(void);
ggml_backend_t Rggml_backend_cuda_init(int device);
int Rggml_backend_cuda_device_description(int device, char *buf, size_t buf_size);

/* Device-buffer residency: the backend-agnostic
 * allocate/upload/compute/download path, required by GPU backends */
ggml_backend_buffer_t Rggml_backend_alloc_ctx_tensors(struct ggml_context *ctx,
                                                       ggml_backend_t backend);
void Rggml_backend_buffer_free(ggml_backend_buffer_t buffer);
void Rggml_backend_tensor_set(struct ggml_tensor *tensor, const void *data,
                               size_t offset, size_t size);
void Rggml_backend_tensor_get(const struct ggml_tensor *tensor, void *data,
                               size_t offset, size_t size);

struct ggml_cgraph *Rggml_new_graph(struct ggml_context *ctx, size_t size);
void Rggml_build_forward_expand(struct ggml_cgraph *cgraph, struct ggml_tensor *tensor);

struct ggml_tensor *Rggml_mul_mat(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b);
struct ggml_tensor *Rggml_mul_mat_id(struct ggml_context *ctx,
                                      struct ggml_tensor *as,
                                      struct ggml_tensor *b,
                                      struct ggml_tensor *ids);
int Rggml_compute_mul_mat(struct ggml_context *ctx, ggml_backend_t backend,
                           struct ggml_tensor *a, struct ggml_tensor *b,
                           float *out_f32, double *out_f64);

size_t Rggml_quantize(enum ggml_type type, const float *src, void *dst,
                      int64_t nrows, int64_t n_per_row);
int Rggml_dequantize(enum ggml_type type, const void *src, float *dst, int64_t n);
int Rggml_can_dequantize(enum ggml_type type);
int Rggml_dequantize_double(enum ggml_type type, const void *src, double *dst,
                            int64_t n);

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

struct Rggml_gguf_context *Rggml_gguf_open(const char *path);
void Rggml_gguf_close(struct Rggml_gguf_context *ctx);
uint32_t Rggml_gguf_version(const struct Rggml_gguf_context *ctx);
size_t Rggml_gguf_data_offset(const struct Rggml_gguf_context *ctx);
int64_t Rggml_gguf_n_kv(const struct Rggml_gguf_context *ctx);
int Rggml_gguf_kv(const struct Rggml_gguf_context *ctx, int64_t id,
                  struct Rggml_gguf_kv *out);
const char *Rggml_gguf_kv_string(const struct Rggml_gguf_context *ctx,
                                 int64_t id, size_t index);
int64_t Rggml_gguf_n_tensors(const struct Rggml_gguf_context *ctx);
int64_t Rggml_gguf_find_tensor(const struct Rggml_gguf_context *ctx,
                               const char *name);
int Rggml_gguf_tensor(const struct Rggml_gguf_context *ctx, int64_t id,
                      struct Rggml_gguf_tensor *out);

struct Rggml_gguf_writer *Rggml_gguf_writer_open(void);
void Rggml_gguf_writer_close(struct Rggml_gguf_writer *ctx);
int Rggml_gguf_writer_set_string(struct Rggml_gguf_writer *ctx,
                                 const char *key, const char *value);
int Rggml_gguf_writer_set_strings(struct Rggml_gguf_writer *ctx,
                                  const char *key, const char **values,
                                  size_t n);
int Rggml_gguf_writer_set_f64(struct Rggml_gguf_writer *ctx, const char *key,
                              double value);
int Rggml_gguf_writer_set_f64s(struct Rggml_gguf_writer *ctx, const char *key,
                               const double *values, size_t n);
int Rggml_gguf_writer_add_f32(struct Rggml_gguf_writer *ctx, const char *name,
                              int n_dims, const int64_t *ne,
                              const double *data);
int Rggml_gguf_writer_write(struct Rggml_gguf_writer *ctx, const char *path);

/* Graph operations used to compose a transformer forward pass. */
struct ggml_tensor *Rggml_get_rows(struct ggml_context *ctx, struct ggml_tensor *a,
                                    struct ggml_tensor *b);
struct ggml_tensor *Rggml_rms_norm(struct ggml_context *ctx, struct ggml_tensor *a,
                                    double eps);
struct ggml_tensor *Rggml_l2_norm(struct ggml_context *ctx, struct ggml_tensor *a,
                                   double eps);
struct ggml_tensor *Rggml_mul(struct ggml_context *ctx, struct ggml_tensor *a,
                               struct ggml_tensor *b);
struct ggml_tensor *Rggml_add(struct ggml_context *ctx, struct ggml_tensor *a,
                               struct ggml_tensor *b);
struct ggml_tensor *Rggml_div(struct ggml_context *ctx, struct ggml_tensor *a,
                               struct ggml_tensor *b);
struct ggml_tensor *Rggml_silu(struct ggml_context *ctx, struct ggml_tensor *a);
struct ggml_tensor *Rggml_geglu(struct ggml_context *ctx, struct ggml_tensor *gate,
                                struct ggml_tensor *up);
struct ggml_tensor *Rggml_sigmoid(struct ggml_context *ctx, struct ggml_tensor *a);
struct ggml_tensor *Rggml_softplus(struct ggml_context *ctx, struct ggml_tensor *a);
struct ggml_tensor *Rggml_scale(struct ggml_context *ctx, struct ggml_tensor *a,
                                 double s);
struct ggml_tensor *Rggml_sum_rows(struct ggml_context *ctx, struct ggml_tensor *a);
struct ggml_tensor *Rggml_clamp(struct ggml_context *ctx, struct ggml_tensor *a,
                                 double min, double max);
struct ggml_tensor *Rggml_argsort_top_k(struct ggml_context *ctx,
                                         struct ggml_tensor *a, int k);
struct ggml_tensor *Rggml_concat(struct ggml_context *ctx,
                                  struct ggml_tensor *a,
                                  struct ggml_tensor *b, int dim);
struct ggml_tensor *Rggml_ssm_conv(struct ggml_context *ctx,
                                    struct ggml_tensor *sx,
                                    struct ggml_tensor *kernel);
struct ggml_tensor *Rggml_conv_1d(struct ggml_context *ctx,
                                   struct ggml_tensor *kernel,
                                   struct ggml_tensor *data,
                                   int stride, int padding, int dilation);
struct Rggml_conv_1d_f32_plan;
struct Rggml_conv_1d_f32_plan *Rggml_conv_1d_f32_plan_create(
    const float *kernel, size_t kernel_bytes, const float *bias,
    size_t bias_bytes, int64_t kernel_size, int64_t input_channels,
    int64_t output_channels, int64_t stride, int64_t padding,
    int64_t dilation, int64_t max_output_channels);
void Rggml_conv_1d_f32_plan_destroy(struct Rggml_conv_1d_f32_plan *plan);
int Rggml_conv_1d_f32_plan_fuse_input(struct Rggml_conv_1d_f32_plan *plan,
    const float *scale, const float *shift, size_t channel_bytes,
    int leaky, double slope);
struct ggml_tensor *Rggml_conv_1d_f32_plan_apply(
    struct ggml_context *ctx, const struct Rggml_conv_1d_f32_plan *plan,
    struct ggml_tensor *input);
struct ggml_tensor *Rggml_leaky_relu(struct ggml_context *ctx,
                                      struct ggml_tensor *a, double slope);
struct ggml_tensor *Rggml_leaky_relu_cpu(struct ggml_context *ctx,
                                          struct ggml_tensor *a, double slope);
struct ggml_tensor *Rggml_soft_max(struct ggml_context *ctx, struct ggml_tensor *a);
struct ggml_tensor *Rggml_soft_max_ext(struct ggml_context *ctx,
                                       struct ggml_tensor *a,
                                       struct ggml_tensor *mask,
                                       double scale, double max_bias);
struct ggml_tensor *Rggml_diag_mask_inf(struct ggml_context *ctx, struct ggml_tensor *a,
                                         int n_past);
struct ggml_tensor *Rggml_rope(struct ggml_context *ctx, struct ggml_tensor *a,
                                struct ggml_tensor *pos, int n_dims, int mode,
                                double freq_base);
struct ggml_tensor *Rggml_rope_multi(struct ggml_context *ctx,
                                      struct ggml_tensor *a,
                                      struct ggml_tensor *pos, int n_dims,
                                      const int sections[GGML_MROPE_SECTIONS],
                                      int mode, double freq_base);
struct ggml_tensor *Rggml_gated_delta_net(struct ggml_context *ctx,
                                           struct ggml_tensor *q,
                                           struct ggml_tensor *k,
                                           struct ggml_tensor *v,
                                           struct ggml_tensor *gate,
                                           struct ggml_tensor *beta,
                                           struct ggml_tensor *state,
                                           int64_t snapshots);
struct ggml_tensor *Rggml_reshape_2d(struct ggml_context *ctx, struct ggml_tensor *a,
                                      int64_t ne0, int64_t ne1);
struct ggml_tensor *Rggml_reshape_3d(struct ggml_context *ctx, struct ggml_tensor *a,
                                      int64_t ne0, int64_t ne1, int64_t ne2);
struct ggml_tensor *Rggml_reshape_4d(struct ggml_context *ctx, struct ggml_tensor *a,
                                      int64_t ne0, int64_t ne1, int64_t ne2,
                                      int64_t ne3);
struct ggml_tensor *Rggml_permute(struct ggml_context *ctx, struct ggml_tensor *a,
                                   int axis0, int axis1, int axis2, int axis3);
struct ggml_tensor *Rggml_cont(struct ggml_context *ctx, struct ggml_tensor *a);
struct ggml_tensor *Rggml_transpose(struct ggml_context *ctx, struct ggml_tensor *a);

/* Views and copies. */
struct ggml_tensor *Rggml_view_1d(struct ggml_context *ctx, struct ggml_tensor *a,
                                   int64_t ne0, size_t offset);
struct ggml_tensor *Rggml_view_2d(struct ggml_context *ctx, struct ggml_tensor *a,
                                   int64_t ne0, int64_t ne1, size_t nb1, size_t offset);
struct ggml_tensor *Rggml_view_3d(struct ggml_context *ctx, struct ggml_tensor *a,
                                   int64_t ne0, int64_t ne1, int64_t ne2,
                                   size_t nb1, size_t nb2, size_t offset);
struct ggml_tensor *Rggml_view_4d(struct ggml_context *ctx, struct ggml_tensor *a,
                                   int64_t ne0, int64_t ne1, int64_t ne2,
                                   int64_t ne3, size_t nb1, size_t nb2,
                                   size_t nb3, size_t offset);
struct ggml_tensor *Rggml_cpy(struct ggml_context *ctx, struct ggml_tensor *a,
                               struct ggml_tensor *b);

size_t Rggml_type_size(enum ggml_type type);
size_t Rggml_row_size(enum ggml_type type, int64_t ne);
int64_t Rggml_blck_size(enum ggml_type type);
size_t Rggml_nbytes(const struct ggml_tensor *tensor);
int64_t Rggml_nelements(const struct ggml_tensor *tensor);
const char *Rggml_type_name(enum ggml_type type);

#ifdef __cplusplus
}
#endif

#endif /* RGGML_API_H */
