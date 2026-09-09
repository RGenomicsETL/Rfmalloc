/*
 * rllm_init.c - native routine registration for Rllm. As in the sibling
 * packages, every .Call entry point is registered explicitly and dynamic
 * symbol lookup is disabled. Rllm registers no C-callables of its own: it is
 * a consumer of Rfmalloc's and Rggml's.
 */
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

SEXP RC_rllm_register_backend(void);
SEXP RC_rllm_qtensor_nbytes(SEXP dtype_sexp, SEXP k_sexp, SEXP n_sexp);
SEXP RC_rllm_quantize_into(SEXP x, SEXP dtype_sexp, SEXP payload);
SEXP RC_rllm_dequantize(SEXP payload, SEXP dtype_sexp, SEXP n_sexp);
SEXP RC_rllm_cuda_model_context(SEXP tensors);
SEXP RC_rllm_program_forward(SEXP bound, SEXP tokens_sexp,
                             SEXP kcache, SEXP vcache, SEXP convcache,
                             SEXP recurrent_cache, SEXP n_ctx_sexp,
                             SEXP n_past_sexp,
                             SEXP backend_sexp, SEXP backend_context);
SEXP RC_rllm_f32_program_forward(SEXP execution, SEXP inputs,
                                 SEXP threads_sexp, SEXP backend_sexp,
                                 SEXP backend_context, SEXP context);

static const R_CallMethodDef CallEntries[] = {
    {"RC_rllm_register_backend", (DL_FUNC) &RC_rllm_register_backend, 0},
    {"RC_rllm_qtensor_nbytes",   (DL_FUNC) &RC_rllm_qtensor_nbytes,   3},
    {"RC_rllm_quantize_into",    (DL_FUNC) &RC_rllm_quantize_into,    3},
    {"RC_rllm_dequantize",       (DL_FUNC) &RC_rllm_dequantize,       3},
    {"RC_rllm_cuda_model_context", (DL_FUNC) &RC_rllm_cuda_model_context, 1},
    {"RC_rllm_program_forward",  (DL_FUNC) &RC_rllm_program_forward, 10},
    {"RC_rllm_f32_program_forward", (DL_FUNC) &RC_rllm_f32_program_forward, 6},
    {NULL, NULL, 0}
};

void R_init_Rllm(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
