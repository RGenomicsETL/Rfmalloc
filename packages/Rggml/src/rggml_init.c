/*
 * rggml_init.c - native routine registration and R_RegisterCCallable() calls
 * for Rggml. Mirrors the approach used by Rfmalloc/Rgguf: every .Call entry
 * point and every C-callable is registered explicitly, and dynamic symbol
 * lookup is disabled so R CMD check does not warn about unregistered native
 * symbols.
 */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

#include "rggml_api.h"
#include "rggml_cpu_ops.h"

#if defined(RGGML_SIMD_DISPATCH) && RGGML_SIMD_DISPATCH
void rggml_simd_dispatch_init(void);
void rggml_conv1d_simd_dispatch_init(void);
#endif

SEXP RC_rggml_version(void);
SEXP RC_rggml_test_mul_mat(SEXP A_sexp, SEXP B_sexp, SEXP zero_copy_sexp, SEXP use_blas_sexp);
SEXP RC_rggml_test_mul_mat_backend(SEXP A_sexp, SEXP B_sexp, SEXP backend_sexp);
SEXP RC_rggml_vulkan_info(void);
SEXP RC_rggml_cuda_info(void);
SEXP RC_rggml_cpu_info(void);
SEXP RC_rggml_test_mul_mat_q4k(SEXP A_sexp, SEXP B_sexp);
SEXP RC_rggml_test_mul_mat_quant_backend(SEXP A_sexp, SEXP B_sexp,
    SEXP type_sexp, SEXP backend_sexp);
SEXP RC_rggml_test_q4k_dot(SEXP nblocks_sexp);
SEXP RC_rggml_bench_q4k_dot(SEXP nblocks_sexp, SEXP iters_sexp);
SEXP RC_rggml_test_conv1d_f32(SEXP kernel_sexp, SEXP input_sexp,
                              SEXP bias_sexp, SEXP stride_sexp,
                              SEXP padding_sexp, SEXP dilation_sexp);
SEXP RC_rggml_test_conv1d_f32_bounds(void);
SEXP RC_rggml_test_leaky_relu_cpu(SEXP input_sexp, SEXP slope_sexp,
                                  SEXP threads_sexp);

static const R_CallMethodDef CallEntries[] = {
    {"RC_rggml_version",           (DL_FUNC) &RC_rggml_version,           0},
    {"RC_rggml_test_mul_mat",      (DL_FUNC) &RC_rggml_test_mul_mat,      4},
    {"RC_rggml_test_mul_mat_backend", (DL_FUNC) &RC_rggml_test_mul_mat_backend, 3},
    {"RC_rggml_vulkan_info",       (DL_FUNC) &RC_rggml_vulkan_info,       0},
    {"RC_rggml_cuda_info",         (DL_FUNC) &RC_rggml_cuda_info,         0},
    {"RC_rggml_cpu_info",          (DL_FUNC) &RC_rggml_cpu_info,          0},
    {"RC_rggml_test_mul_mat_q4k",  (DL_FUNC) &RC_rggml_test_mul_mat_q4k,  2},
    {"RC_rggml_test_mul_mat_quant_backend", (DL_FUNC) &RC_rggml_test_mul_mat_quant_backend, 4},
    {"RC_rggml_test_q4k_dot",      (DL_FUNC) &RC_rggml_test_q4k_dot,      1},
    {"RC_rggml_bench_q4k_dot",     (DL_FUNC) &RC_rggml_bench_q4k_dot,     2},
    {"RC_rggml_test_conv1d_f32",   (DL_FUNC) &RC_rggml_test_conv1d_f32,   6},
    {"RC_rggml_test_conv1d_f32_bounds", (DL_FUNC) &RC_rggml_test_conv1d_f32_bounds, 0},
    {"RC_rggml_test_leaky_relu_cpu", (DL_FUNC) &RC_rggml_test_leaky_relu_cpu, 3},
    {NULL, NULL, 0}
};

static void register_c_callables(DllInfo *dll)
{
    R_RegisterCCallable("Rggml", "Rggml_version",               (DL_FUNC) Rggml_version);

    R_RegisterCCallable("Rggml", "Rggml_context_create",        (DL_FUNC) Rggml_context_create);
    R_RegisterCCallable("Rggml", "Rggml_context_free",          (DL_FUNC) Rggml_context_free);
    R_RegisterCCallable("Rggml", "Rggml_used_mem",               (DL_FUNC) Rggml_used_mem);
    R_RegisterCCallable("Rggml", "Rggml_tensor_overhead",        (DL_FUNC) Rggml_tensor_overhead);
    R_RegisterCCallable("Rggml", "Rggml_graph_overhead",         (DL_FUNC) Rggml_graph_overhead);

    R_RegisterCCallable("Rggml", "Rggml_new_tensor",             (DL_FUNC) Rggml_new_tensor);
    R_RegisterCCallable("Rggml", "Rggml_tensor_data",            (DL_FUNC) Rggml_tensor_data);
    R_RegisterCCallable("Rggml", "Rggml_tensor_set_data",        (DL_FUNC) Rggml_tensor_set_data);
    R_RegisterCCallable("Rggml", "Rggml_tensor_ne",              (DL_FUNC) Rggml_tensor_ne);
    R_RegisterCCallable("Rggml", "Rggml_tensor_nb",              (DL_FUNC) Rggml_tensor_nb);

    R_RegisterCCallable("Rggml", "Rggml_backend_cpu_init",       (DL_FUNC) Rggml_backend_cpu_init);
    R_RegisterCCallable("Rggml", "Rggml_backend_cpu_set_n_threads", (DL_FUNC) Rggml_backend_cpu_set_n_threads);
    R_RegisterCCallable("Rggml", "Rggml_backend_free",           (DL_FUNC) Rggml_backend_free);
    R_RegisterCCallable("Rggml", "Rggml_backend_graph_compute",  (DL_FUNC) Rggml_backend_graph_compute);
    R_RegisterCCallable("Rggml", "Rggml_backend_blas_init",      (DL_FUNC) Rggml_backend_blas_init);
    R_RegisterCCallable("Rggml", "Rggml_backend_blas_set_n_threads", (DL_FUNC) Rggml_backend_blas_set_n_threads);
    R_RegisterCCallable("Rggml", "Rggml_backend_vulkan_device_count", (DL_FUNC) Rggml_backend_vulkan_device_count);
    R_RegisterCCallable("Rggml", "Rggml_backend_vulkan_init",       (DL_FUNC) Rggml_backend_vulkan_init);
    R_RegisterCCallable("Rggml", "Rggml_backend_vulkan_device_description", (DL_FUNC) Rggml_backend_vulkan_device_description);
    R_RegisterCCallable("Rggml", "Rggml_backend_cuda_device_count", (DL_FUNC) Rggml_backend_cuda_device_count);
    R_RegisterCCallable("Rggml", "Rggml_backend_cuda_init",       (DL_FUNC) Rggml_backend_cuda_init);
    R_RegisterCCallable("Rggml", "Rggml_backend_cuda_device_description", (DL_FUNC) Rggml_backend_cuda_device_description);
    R_RegisterCCallable("Rggml", "Rggml_backend_alloc_ctx_tensors", (DL_FUNC) Rggml_backend_alloc_ctx_tensors);
    R_RegisterCCallable("Rggml", "Rggml_backend_buffer_free",     (DL_FUNC) Rggml_backend_buffer_free);
    R_RegisterCCallable("Rggml", "Rggml_backend_tensor_set",      (DL_FUNC) Rggml_backend_tensor_set);
    R_RegisterCCallable("Rggml", "Rggml_backend_tensor_get",      (DL_FUNC) Rggml_backend_tensor_get);

    R_RegisterCCallable("Rggml", "Rggml_new_graph",              (DL_FUNC) Rggml_new_graph);
    R_RegisterCCallable("Rggml", "Rggml_build_forward_expand",   (DL_FUNC) Rggml_build_forward_expand);

    R_RegisterCCallable("Rggml", "Rggml_mul_mat",                (DL_FUNC) Rggml_mul_mat);
    R_RegisterCCallable("Rggml", "Rggml_mul_mat_id",             (DL_FUNC) Rggml_mul_mat_id);
    R_RegisterCCallable("Rggml", "Rggml_compute_mul_mat",        (DL_FUNC) Rggml_compute_mul_mat);

    R_RegisterCCallable("Rggml", "Rggml_quantize",              (DL_FUNC) Rggml_quantize);
    R_RegisterCCallable("Rggml", "Rggml_dequantize",            (DL_FUNC) Rggml_dequantize);
    R_RegisterCCallable("Rggml", "Rggml_can_dequantize",        (DL_FUNC) Rggml_can_dequantize);
    R_RegisterCCallable("Rggml", "Rggml_dequantize_double",     (DL_FUNC) Rggml_dequantize_double);

    R_RegisterCCallable("Rggml", "Rggml_gguf_open",             (DL_FUNC) Rggml_gguf_open);
    R_RegisterCCallable("Rggml", "Rggml_gguf_close",            (DL_FUNC) Rggml_gguf_close);
    R_RegisterCCallable("Rggml", "Rggml_gguf_version",          (DL_FUNC) Rggml_gguf_version);
    R_RegisterCCallable("Rggml", "Rggml_gguf_data_offset",      (DL_FUNC) Rggml_gguf_data_offset);
    R_RegisterCCallable("Rggml", "Rggml_gguf_n_kv",             (DL_FUNC) Rggml_gguf_n_kv);
    R_RegisterCCallable("Rggml", "Rggml_gguf_kv",               (DL_FUNC) Rggml_gguf_kv);
    R_RegisterCCallable("Rggml", "Rggml_gguf_kv_string",        (DL_FUNC) Rggml_gguf_kv_string);
    R_RegisterCCallable("Rggml", "Rggml_gguf_n_tensors",        (DL_FUNC) Rggml_gguf_n_tensors);
    R_RegisterCCallable("Rggml", "Rggml_gguf_find_tensor",      (DL_FUNC) Rggml_gguf_find_tensor);
    R_RegisterCCallable("Rggml", "Rggml_gguf_tensor",           (DL_FUNC) Rggml_gguf_tensor);
    R_RegisterCCallable("Rggml", "Rggml_gguf_writer_open",      (DL_FUNC) Rggml_gguf_writer_open);
    R_RegisterCCallable("Rggml", "Rggml_gguf_writer_close",     (DL_FUNC) Rggml_gguf_writer_close);
    R_RegisterCCallable("Rggml", "Rggml_gguf_writer_set_string", (DL_FUNC) Rggml_gguf_writer_set_string);
    R_RegisterCCallable("Rggml", "Rggml_gguf_writer_set_strings", (DL_FUNC) Rggml_gguf_writer_set_strings);
    R_RegisterCCallable("Rggml", "Rggml_gguf_writer_set_f64",   (DL_FUNC) Rggml_gguf_writer_set_f64);
    R_RegisterCCallable("Rggml", "Rggml_gguf_writer_set_f64s",  (DL_FUNC) Rggml_gguf_writer_set_f64s);
    R_RegisterCCallable("Rggml", "Rggml_gguf_writer_add_f32",   (DL_FUNC) Rggml_gguf_writer_add_f32);
    R_RegisterCCallable("Rggml", "Rggml_gguf_writer_write",     (DL_FUNC) Rggml_gguf_writer_write);

    R_RegisterCCallable("Rggml", "Rggml_get_rows",              (DL_FUNC) Rggml_get_rows);
    R_RegisterCCallable("Rggml", "Rggml_rms_norm",              (DL_FUNC) Rggml_rms_norm);
    R_RegisterCCallable("Rggml", "Rggml_l2_norm",               (DL_FUNC) Rggml_l2_norm);
    R_RegisterCCallable("Rggml", "Rggml_mul",                   (DL_FUNC) Rggml_mul);
    R_RegisterCCallable("Rggml", "Rggml_add",                   (DL_FUNC) Rggml_add);
    R_RegisterCCallable("Rggml", "Rggml_div",                   (DL_FUNC) Rggml_div);
    R_RegisterCCallable("Rggml", "Rggml_silu",                  (DL_FUNC) Rggml_silu);
    R_RegisterCCallable("Rggml", "Rggml_geglu",                 (DL_FUNC) Rggml_geglu);
    R_RegisterCCallable("Rggml", "Rggml_sigmoid",               (DL_FUNC) Rggml_sigmoid);
    R_RegisterCCallable("Rggml", "Rggml_softplus",              (DL_FUNC) Rggml_softplus);
    R_RegisterCCallable("Rggml", "Rggml_scale",                 (DL_FUNC) Rggml_scale);
    R_RegisterCCallable("Rggml", "Rggml_sum_rows",              (DL_FUNC) Rggml_sum_rows);
    R_RegisterCCallable("Rggml", "Rggml_clamp",                 (DL_FUNC) Rggml_clamp);
    R_RegisterCCallable("Rggml", "Rggml_argsort_top_k",         (DL_FUNC) Rggml_argsort_top_k);
    R_RegisterCCallable("Rggml", "Rggml_concat",                (DL_FUNC) Rggml_concat);
    R_RegisterCCallable("Rggml", "Rggml_ssm_conv",              (DL_FUNC) Rggml_ssm_conv);
    R_RegisterCCallable("Rggml", "Rggml_conv_1d",               (DL_FUNC) Rggml_conv_1d);
    R_RegisterCCallable("Rggml", "Rggml_conv_1d_f32_plan_create", (DL_FUNC) Rggml_conv_1d_f32_plan_create);
    R_RegisterCCallable("Rggml", "Rggml_conv_1d_f32_plan_destroy", (DL_FUNC) Rggml_conv_1d_f32_plan_destroy);
    R_RegisterCCallable("Rggml", "Rggml_conv_1d_f32_plan_apply", (DL_FUNC) Rggml_conv_1d_f32_plan_apply);
    R_RegisterCCallable("Rggml", "Rggml_leaky_relu",            (DL_FUNC) Rggml_leaky_relu);
    R_RegisterCCallable("Rggml", "Rggml_leaky_relu_cpu",        (DL_FUNC) Rggml_leaky_relu_cpu);
    R_RegisterCCallable("Rggml", "Rggml_soft_max",              (DL_FUNC) Rggml_soft_max);
    R_RegisterCCallable("Rggml", "Rggml_soft_max_ext",          (DL_FUNC) Rggml_soft_max_ext);
    R_RegisterCCallable("Rggml", "Rggml_diag_mask_inf",         (DL_FUNC) Rggml_diag_mask_inf);
    R_RegisterCCallable("Rggml", "Rggml_rope",                  (DL_FUNC) Rggml_rope);
    R_RegisterCCallable("Rggml", "Rggml_rope_multi",            (DL_FUNC) Rggml_rope_multi);
    R_RegisterCCallable("Rggml", "Rggml_gated_delta_net",       (DL_FUNC) Rggml_gated_delta_net);
    R_RegisterCCallable("Rggml", "Rggml_reshape_2d",            (DL_FUNC) Rggml_reshape_2d);
    R_RegisterCCallable("Rggml", "Rggml_reshape_3d",            (DL_FUNC) Rggml_reshape_3d);
    R_RegisterCCallable("Rggml", "Rggml_reshape_4d",            (DL_FUNC) Rggml_reshape_4d);
    R_RegisterCCallable("Rggml", "Rggml_permute",               (DL_FUNC) Rggml_permute);
    R_RegisterCCallable("Rggml", "Rggml_cont",                  (DL_FUNC) Rggml_cont);
    R_RegisterCCallable("Rggml", "Rggml_transpose",             (DL_FUNC) Rggml_transpose);
    R_RegisterCCallable("Rggml", "Rggml_view_1d",               (DL_FUNC) Rggml_view_1d);
    R_RegisterCCallable("Rggml", "Rggml_view_2d",               (DL_FUNC) Rggml_view_2d);
    R_RegisterCCallable("Rggml", "Rggml_view_3d",               (DL_FUNC) Rggml_view_3d);
    R_RegisterCCallable("Rggml", "Rggml_view_4d",               (DL_FUNC) Rggml_view_4d);
    R_RegisterCCallable("Rggml", "Rggml_cpy",                   (DL_FUNC) Rggml_cpy);

    R_RegisterCCallable("Rggml", "Rggml_type_size",              (DL_FUNC) Rggml_type_size);
    R_RegisterCCallable("Rggml", "Rggml_row_size",               (DL_FUNC) Rggml_row_size);
    R_RegisterCCallable("Rggml", "Rggml_blck_size",              (DL_FUNC) Rggml_blck_size);
    R_RegisterCCallable("Rggml", "Rggml_nbytes",                 (DL_FUNC) Rggml_nbytes);
    R_RegisterCCallable("Rggml", "Rggml_nelements",              (DL_FUNC) Rggml_nelements);
    R_RegisterCCallable("Rggml", "Rggml_type_name",              (DL_FUNC) Rggml_type_name);

    (void) dll;
}

void R_init_Rggml(DllInfo *dll)
{
#if defined(RGGML_SIMD_DISPATCH) && RGGML_SIMD_DISPATCH
    rggml_simd_dispatch_init();
    rggml_conv1d_simd_dispatch_init();
#endif
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    register_c_callables(dll);
    R_useDynamicSymbols(dll, FALSE);
}
