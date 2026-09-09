/*
 * rggml_test.c - .Call() entry points used only by the tinytest smoke test.
 *
 * These intentionally go through the *installed* Rggml.h header and its
 * R_GetCCallable() accessors, exactly as a downstream package would, rather
 * than calling the static rggml_api.c functions directly - this is what
 * proves the registered C-callable path actually works end to end.
 */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include <R.h>
#include <Rinternals.h>

#include <Rggml.h>

#include "rggml_conv1d.h"

SEXP RC_rggml_version(void)
{
    Rggml_version_fun version_fn = Rggml_version_ptr();
    if (!version_fn) Rf_error("Rggml_version C-callable not registered");
    return Rf_mkString(version_fn());
}

/*
 * RC_rggml_test_q4k_dot(nblocks)
 *
 * Exercises the runtime-SIMD-dispatched q4_K x q8_K dot product. Builds
 * deterministic inputs of length nblocks*256, quantizes them to Q4_K and Q8_K
 * with GGML's own quantizers, then calls both the canonical (dispatched, i.e.
 * upstream x86 where staged) ggml_vec_dot_q4_K_q8_K and GGML's scalar reference
 * ggml_vec_dot_q4_K_q8_K_generic on identical bytes. Returns c(dispatched,
 * scalar); the tinytest asserts they agree, proving the staged ISA variant is
 * correct. These are GGML-internal symbols (not C-callables); this test file
 * is part of the package and links libggml.a directly, so it declares them.
 */
extern void quantize_row_q8_K(const float *x, void *y, int64_t k);
extern void ggml_vec_dot_q4_K_q8_K(int n, float *s, size_t bs,
        const void *vx, size_t bx, const void *vy, size_t by, int nrc);
extern void ggml_vec_dot_q4_K_q8_K_generic(int n, float *s, size_t bs,
        const void *vx, size_t bx, const void *vy, size_t by, int nrc);

SEXP RC_rggml_test_q4k_dot(SEXP nblocks_sexp)
{
    int nb = Rf_asInteger(nblocks_sexp);
    if (nb < 1) nb = 1;
    const int QKK = 256;
    int n = nb * QKK;

    /* Initializing the CPU backend runs ggml_cpu_init(), which populates the
     * fp16->fp32 lookup table the q4_K dot uses to read block scales. Without
     * it those scales read as 0 and the dot is (silently) 0. */
    Rggml_backend_cpu_init_fun backend_init = Rggml_backend_cpu_init_ptr();
    Rggml_backend_free_fun     backend_free = Rggml_backend_free_ptr();
    ggml_backend_t cpu = backend_init ? backend_init() : NULL;

    /* Deterministic pseudo-random inputs (LCG), no RNG-state dependency. */
    float *x = (float *) R_alloc((size_t) n, sizeof(float));
    float *y = (float *) R_alloc((size_t) n, sizeof(float));
    uint32_t st = 0x9e3779b9u;
    for (int i = 0; i < n; i++) {
        st = st * 1664525u + 1013904223u;
        x[i] = (float) ((int32_t) (st >> 8) / 8388608.0 - 1.0);   /* ~[-1,1) */
        st = st * 1664525u + 1013904223u;
        y[i] = (float) ((int32_t) (st >> 8) / 8388608.0 - 1.0);
    }

    size_t xb = ggml_row_size(GGML_TYPE_Q4_K, n);
    size_t yb = ggml_row_size(GGML_TYPE_Q8_K, n);
    void *qx = (void *) R_alloc(xb, 1);
    void *qy = (void *) R_alloc(yb, 1);

    ggml_quantize_chunk(GGML_TYPE_Q4_K, x, qx, 0, 1, n, NULL);
    quantize_row_q8_K(y, qy, n);

    float s_disp = 0.0f, s_gen = 0.0f;
    ggml_vec_dot_q4_K_q8_K(n, &s_disp, 0, qx, 0, qy, 0, 1);
    ggml_vec_dot_q4_K_q8_K_generic(n, &s_gen, 0, qx, 0, qy, 0, 1);

    if (cpu && backend_free) backend_free(cpu);

    SEXP out = PROTECT(Rf_allocVector(REALSXP, 2));
    REAL(out)[0] = (double) s_disp;
    REAL(out)[1] = (double) s_gen;
    UNPROTECT(1);
    return out;
}

/*
 * RC_rggml_bench_q4k_dot(nblocks, iters) - time the dispatched (staged ISA)
 * q4_K dot against GGML's scalar reference over `iters` repetitions on the
 * same quantized inputs. Returns c(dispatched_sec, scalar_sec). Not a
 * regression test (timings are machine-dependent); a helper to report the
 * SIMD speedup.
 */
static double rggml_now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

SEXP RC_rggml_bench_q4k_dot(SEXP nblocks_sexp, SEXP iters_sexp)
{
    int nb = Rf_asInteger(nblocks_sexp);
    if (nb < 1) nb = 1;
    int iters = Rf_asInteger(iters_sexp);
    if (iters < 1) iters = 1;
    const int QKK = 256;
    int n = nb * QKK;

    Rggml_backend_cpu_init_fun backend_init = Rggml_backend_cpu_init_ptr();
    Rggml_backend_free_fun     backend_free = Rggml_backend_free_ptr();
    ggml_backend_t cpu = backend_init ? backend_init() : NULL;

    float *x = (float *) R_alloc((size_t) n, sizeof(float));
    float *y = (float *) R_alloc((size_t) n, sizeof(float));
    uint32_t st = 0x9e3779b9u;
    for (int i = 0; i < n; i++) {
        st = st * 1664525u + 1013904223u;
        x[i] = (float) ((int32_t) (st >> 8) / 8388608.0 - 1.0);
        st = st * 1664525u + 1013904223u;
        y[i] = (float) ((int32_t) (st >> 8) / 8388608.0 - 1.0);
    }
    void *qx = (void *) R_alloc(ggml_row_size(GGML_TYPE_Q4_K, n), 1);
    void *qy = (void *) R_alloc(ggml_row_size(GGML_TYPE_Q8_K, n), 1);
    ggml_quantize_chunk(GGML_TYPE_Q4_K, x, qx, 0, 1, n, NULL);
    quantize_row_q8_K(y, qy, n);

    volatile float sink = 0.0f;
    float s;
    double t0 = rggml_now_sec();
    for (int it = 0; it < iters; it++) { ggml_vec_dot_q4_K_q8_K(n, &s, 0, qx, 0, qy, 0, 1); sink += s; }
    double t_disp = rggml_now_sec() - t0;

    t0 = rggml_now_sec();
    for (int it = 0; it < iters; it++) { ggml_vec_dot_q4_K_q8_K_generic(n, &s, 0, qx, 0, qy, 0, 1); sink += s; }
    double t_gen = rggml_now_sec() - t0;

    if (cpu && backend_free) backend_free(cpu);

    SEXP out = PROTECT(Rf_allocVector(REALSXP, 2));
    REAL(out)[0] = t_disp;
    REAL(out)[1] = t_gen;
    UNPROTECT(1);
    (void) sink;
    return out;
}

/*
 * RC_rggml_test_mul_mat(A, B, zero_copy)
 *
 * A, B: numeric matrices, nrow(A) == nrow(B) == the contracted dimension.
 * zero_copy: if TRUE, tensors are created in a no_alloc = 1 context and
 *   pointed at caller-owned float buffers (the mmap-style path); if FALSE,
 *   tensors are created in a normal (no_alloc = 0) context and filled via
 *   the ggml-owned data pointer.
 *
 * Returns a numeric matrix, dim (ncol(A), ncol(B)), computed via
 * ggml_mul_mat(ctx, A, B) on the CPU backend. Verified/documented
 * convention (see README.md / man/ggml_version.Rd): reinterpreting each
 * input tensor's raw column-major memory directly as an R matrix of dim
 * (ne[0], ne[1]), the result (dim (A_ne1, B_ne1)) equals crossprod(A, B),
 * i.e. t(A) %*% B.
 */
SEXP RC_rggml_test_mul_mat(SEXP A_sexp, SEXP B_sexp, SEXP zero_copy_sexp, SEXP use_blas_sexp)
{
    if (TYPEOF(A_sexp) != REALSXP || TYPEOF(B_sexp) != REALSXP) {
        Rf_error("A and B must be numeric matrices");
    }
    SEXP dimA = Rf_getAttrib(A_sexp, R_DimSymbol);
    SEXP dimB = Rf_getAttrib(B_sexp, R_DimSymbol);
    if (Rf_length(dimA) != 2 || Rf_length(dimB) != 2) {
        Rf_error("A and B must be matrices");
    }
    int64_t kA = INTEGER(dimA)[0];
    int64_t nA = INTEGER(dimA)[1];
    int64_t kB = INTEGER(dimB)[0];
    int64_t nB = INTEGER(dimB)[1];
    if (kA != kB) {
        Rf_error("nrow(A) must equal nrow(B)");
    }
    int zero_copy = Rf_asLogical(zero_copy_sexp);
    int use_blas  = Rf_asLogical(use_blas_sexp);

    Rggml_context_create_fun    ctx_create      = Rggml_context_create_ptr();
    Rggml_context_free_fun      ctx_free        = Rggml_context_free_ptr();
    Rggml_new_tensor_fun        new_tensor      = Rggml_new_tensor_ptr();
    Rggml_tensor_data_fun       tensor_data     = Rggml_tensor_data_ptr();
    Rggml_backend_cpu_init_fun  backend_init    = Rggml_backend_cpu_init_ptr();
    Rggml_backend_blas_init_fun blas_init       = Rggml_backend_blas_init_ptr();
    Rggml_backend_free_fun      backend_free    = Rggml_backend_free_ptr();
    Rggml_compute_mul_mat_fun   compute_mul_mat = Rggml_compute_mul_mat_ptr();
    Rggml_tensor_overhead_fun   tensor_overhead = Rggml_tensor_overhead_ptr();
    Rggml_graph_overhead_fun    graph_overhead  = Rggml_graph_overhead_ptr();

    if (!ctx_create || !ctx_free || !new_tensor || !tensor_data || !backend_init ||
        !backend_free || !compute_mul_mat || !tensor_overhead || !graph_overhead) {
        Rf_error("one or more Rggml C-callables were not found via R_GetCCallable()");
    }

    /* In the no_alloc = 0 path the two input tensors' *data* also lives in the
     * context pool, so size it to include their bytes (plus per-tensor alignment
     * slack); the zero_copy path keeps tensor data in external buffers. */
    size_t data_bytes = zero_copy ? 0
        : ((size_t)kA * (size_t)nA + (size_t)kB * (size_t)nB) * sizeof(float);
    size_t mem_size = (size_t)4 * tensor_overhead() + graph_overhead(8) + data_bytes + 8192;
    struct ggml_context *ctx = ctx_create(mem_size, zero_copy ? 1 : 0);
    if (!ctx) Rf_error("Rggml_context_create (ggml_init) failed");

    int64_t neA[2] = { kA, nA };
    int64_t neB[2] = { kB, nB };

    double *Ap = REAL(A_sexp);
    double *Bp = REAL(B_sexp);
    R_xlen_t nElemA = (R_xlen_t)kA * (R_xlen_t)nA;
    R_xlen_t nElemB = (R_xlen_t)kB * (R_xlen_t)nB;

    float *ext_A = NULL, *ext_B = NULL;
    struct ggml_tensor *tA, *tB;

    if (zero_copy) {
        /* Caller-owned buffers, e.g. standing in for an mmap'd payload the
         * downstream package does not want ggml to copy. */
        ext_A = (float *) malloc(sizeof(float) * (size_t) nElemA);
        ext_B = (float *) malloc(sizeof(float) * (size_t) nElemB);
        if (!ext_A || !ext_B) {
            free(ext_A); free(ext_B);
            ctx_free(ctx);
            Rf_error("allocation failure preparing zero-copy buffers");
        }
        for (R_xlen_t i = 0; i < nElemA; i++) ext_A[i] = (float) Ap[i];
        for (R_xlen_t i = 0; i < nElemB; i++) ext_B[i] = (float) Bp[i];

        tA = new_tensor(ctx, GGML_TYPE_F32, 2, neA, ext_A);
        tB = new_tensor(ctx, GGML_TYPE_F32, 2, neB, ext_B);
    } else {
        tA = new_tensor(ctx, GGML_TYPE_F32, 2, neA, NULL);
        tB = new_tensor(ctx, GGML_TYPE_F32, 2, neB, NULL);
        if (tA && tB) {
            float *tAd = (float *) tensor_data(tA);
            float *tBd = (float *) tensor_data(tB);
            for (R_xlen_t i = 0; i < nElemA; i++) tAd[i] = (float) Ap[i];
            for (R_xlen_t i = 0; i < nElemB; i++) tBd[i] = (float) Bp[i];
        }
    }

    if (!tA || !tB) {
        free(ext_A); free(ext_B);
        ctx_free(ctx);
        Rf_error("Rggml_new_tensor failed");
    }

    ggml_backend_t backend = use_blas ? (blas_init ? blas_init() : NULL) : backend_init();
    if (!backend) {
        free(ext_A); free(ext_B);
        ctx_free(ctx);
        Rf_error(use_blas ? "Rggml_backend_blas_init failed" : "Rggml_backend_cpu_init failed");
    }

    SEXP result = PROTECT(Rf_allocMatrix(REALSXP, (int) nA, (int) nB));
    int rc = compute_mul_mat(ctx, backend, tA, tB, NULL, REAL(result));

    backend_free(backend);
    free(ext_A);
    free(ext_B);
    ctx_free(ctx);

    if (rc != 0) {
        UNPROTECT(1);
        Rf_error("Rggml_compute_mul_mat failed with status %d", rc);
    }

    UNPROTECT(1);
    return result;
}

/*
 * RC_rggml_test_mul_mat_q4k(A, B)
 *
 * The quantized analogue of RC_rggml_test_mul_mat's zero-copy path, and the
 * exact call the Rfmalloc typed-GEMM bridge makes: the weight operand A is
 * quantized to Q4_K into a *separate* heap buffer that stands in for an
 * mmap'd GGUF q4_K payload, and a Q4_K tensor is pointed at it zero-copy
 * (no_alloc context, no ggml-owned copy). The dense F32 activations B are the
 * right operand. ggml_mul_mat() then contracts each Q4_K weight row against
 * B's columns via GGML's type-traits vec_dot for Q4_K - i.e. through the
 * runtime-SIMD-dispatched ggml_vec_dot_q4_K_q8_K (AVX2/NEON where staged) -
 * quantizing B's columns to Q8_K on the fly, exactly as llama.cpp does at
 * inference. This proves the full quantized weight -> compute path (not just
 * the isolated dot) over an external payload.
 *
 * A: numeric weight matrix, nrow(A) = the contracted dimension k, which must
 *    be a multiple of 256 (QK_K); ncol(A) = number of output features.
 * B: numeric activation matrix, nrow(B) = k, ncol(B) = number of columns.
 * Returns a numeric matrix of dim (ncol(A), ncol(B)) = crossprod(A, B) up to
 * q4_K weight + q8_K activation quantization error (same convention as the
 * F32 path). Column-major R storage maps a matrix's column r straight onto
 * ggml row r (ne[0] = k contiguous), so no transpose is needed before
 * quantization.
 */
SEXP RC_rggml_test_mul_mat_q4k(SEXP A_sexp, SEXP B_sexp)
{
    if (TYPEOF(A_sexp) != REALSXP || TYPEOF(B_sexp) != REALSXP) {
        Rf_error("A and B must be numeric matrices");
    }
    SEXP dimA = Rf_getAttrib(A_sexp, R_DimSymbol);
    SEXP dimB = Rf_getAttrib(B_sexp, R_DimSymbol);
    if (Rf_length(dimA) != 2 || Rf_length(dimB) != 2) {
        Rf_error("A and B must be matrices");
    }
    int64_t k  = INTEGER(dimA)[0];
    int64_t nA = INTEGER(dimA)[1];
    int64_t kB = INTEGER(dimB)[0];
    int64_t nB = INTEGER(dimB)[1];
    if (k != kB) {
        Rf_error("nrow(A) must equal nrow(B)");
    }
    if (k % 256 != 0) {
        Rf_error("nrow(A) must be a multiple of 256 (QK_K) to quantize weights to Q4_K; got %lld",
                 (long long) k);
    }

    Rggml_context_create_fun    ctx_create      = Rggml_context_create_ptr();
    Rggml_context_free_fun      ctx_free        = Rggml_context_free_ptr();
    Rggml_new_tensor_fun        new_tensor      = Rggml_new_tensor_ptr();
    Rggml_backend_cpu_init_fun  backend_init    = Rggml_backend_cpu_init_ptr();
    Rggml_backend_free_fun      backend_free    = Rggml_backend_free_ptr();
    Rggml_compute_mul_mat_fun   compute_mul_mat = Rggml_compute_mul_mat_ptr();
    Rggml_tensor_overhead_fun   tensor_overhead = Rggml_tensor_overhead_ptr();
    Rggml_graph_overhead_fun    graph_overhead  = Rggml_graph_overhead_ptr();

    if (!ctx_create || !ctx_free || !new_tensor || !backend_init ||
        !backend_free || !compute_mul_mat || !tensor_overhead || !graph_overhead) {
        Rf_error("one or more Rggml C-callables were not found via R_GetCCallable()");
    }

    /* Initializing the CPU backend runs ggml_cpu_init(), which populates the
     * fp16->fp32 table the q4_K dot needs to read block scales; without it the
     * dot silently returns 0. */
    ggml_backend_t backend = backend_init();
    if (!backend) Rf_error("Rggml_backend_cpu_init failed");

    R_xlen_t nElemA = (R_xlen_t) k * (R_xlen_t) nA;
    R_xlen_t nElemB = (R_xlen_t) k * (R_xlen_t) nB;

    /* Weights -> f32, then quantize to Q4_K into a heap buffer standing in for
     * an mmap'd GGUF payload. Column-major A already lays out ggml row r (= A's
     * column r) contiguously, which is exactly ggml_quantize_chunk's per-row
     * expectation, so no transpose. */
    float *af = (float *) R_alloc((size_t) nElemA, sizeof(float));
    double *Ap = REAL(A_sexp);
    for (R_xlen_t i = 0; i < nElemA; i++) af[i] = (float) Ap[i];

    size_t qa_bytes = ggml_row_size(GGML_TYPE_Q4_K, k) * (size_t) nA;
    void  *qa = malloc(qa_bytes > 0 ? qa_bytes : 1);
    float *bf = (float *) malloc(sizeof(float) * (size_t) (nElemB > 0 ? nElemB : 1));
    if (!qa || !bf) {
        free(qa); free(bf);
        backend_free(backend);
        Rf_error("allocation failure preparing the Q4_K payload / activations");
    }
    ggml_quantize_chunk(GGML_TYPE_Q4_K, af, qa, 0, nA, k, NULL);

    double *Bp = REAL(B_sexp);
    for (R_xlen_t i = 0; i < nElemB; i++) bf[i] = (float) Bp[i];

    size_t mem_size = (size_t) 4 * tensor_overhead() + graph_overhead(8) + 8192;
    struct ggml_context *ctx = ctx_create(mem_size, /*no_alloc=*/1);
    if (!ctx) {
        free(qa); free(bf);
        backend_free(backend);
        Rf_error("Rggml_context_create (ggml_init) failed");
    }

    int64_t neA[2] = { k, nA };
    int64_t neB[2] = { k, nB };
    struct ggml_tensor *tA = new_tensor(ctx, GGML_TYPE_Q4_K, 2, neA, qa);
    struct ggml_tensor *tB = new_tensor(ctx, GGML_TYPE_F32,  2, neB, bf);
    if (!tA || !tB) {
        ctx_free(ctx);
        free(qa); free(bf);
        backend_free(backend);
        Rf_error("Rggml_new_tensor failed");
    }

    SEXP result = PROTECT(Rf_allocMatrix(REALSXP, (int) nA, (int) nB));
    int rc = compute_mul_mat(ctx, backend, tA, tB, NULL, REAL(result));

    ctx_free(ctx);
    free(qa);
    free(bf);
    backend_free(backend);

    if (rc != 0) {
        UNPROTECT(1);
        Rf_error("Rggml_compute_mul_mat (Q4_K) failed with status %d", rc);
    }

    UNPROTECT(1);
    return result;
}

static enum ggml_type
rggml_quant_type(const char *name)
{
    if (!strcmp(name, "q4_0")) return GGML_TYPE_Q4_0;
    if (!strcmp(name, "q4_1")) return GGML_TYPE_Q4_1;
    if (!strcmp(name, "q5_0")) return GGML_TYPE_Q5_0;
    if (!strcmp(name, "q5_1")) return GGML_TYPE_Q5_1;
    if (!strcmp(name, "q8_0")) return GGML_TYPE_Q8_0;
    if (!strcmp(name, "q2_k")) return GGML_TYPE_Q2_K;
    if (!strcmp(name, "q3_k")) return GGML_TYPE_Q3_K;
    if (!strcmp(name, "q4_k")) return GGML_TYPE_Q4_K;
    if (!strcmp(name, "q5_k")) return GGML_TYPE_Q5_K;
    if (!strcmp(name, "q6_k")) return GGML_TYPE_Q6_K;
    return GGML_TYPE_COUNT;
}

/* Quantized device-residency proof. Unlike the host-pointer helper above,
 * this allocates quantized weights, F32 activations and the result in the
 * selected backend buffer, then uploads, computes and downloads through the
 * same interface used by a complete model graph. */
SEXP
RC_rggml_test_mul_mat_quant_backend(SEXP A_sexp, SEXP B_sexp,
    SEXP type_sexp, SEXP backend_sexp)
{
    if (TYPEOF(A_sexp) != REALSXP || TYPEOF(B_sexp) != REALSXP) {
        Rf_error("A and B must be numeric matrices");
    }
    if (TYPEOF(type_sexp) != STRSXP || Rf_xlength(type_sexp) != 1) {
        Rf_error("type must be one quantized GGML type name");
    }
    const char *type_name = CHAR(STRING_ELT(type_sexp, 0));
    enum ggml_type type = rggml_quant_type(type_name);
    if (type == GGML_TYPE_COUNT) {
        Rf_error("unsupported quantized GGML type '%s'", type_name);
    }
    SEXP dimA = Rf_getAttrib(A_sexp, R_DimSymbol);
    SEXP dimB = Rf_getAttrib(B_sexp, R_DimSymbol);
    if (Rf_length(dimA) != 2 || Rf_length(dimB) != 2) Rf_error("A and B must be matrices");
    int64_t k = INTEGER(dimA)[0], nA = INTEGER(dimA)[1];
    int64_t kB = INTEGER(dimB)[0], nB = INTEGER(dimB)[1];
    if (k != kB) Rf_error("nrow(A) must equal nrow(B)");
    int64_t block = ggml_blck_size(type);
    if (k % block != 0) {
        Rf_error("nrow(A) must be a multiple of %lld for %s",
            (long long) block, type_name);
    }
    int which = Rf_asInteger(backend_sexp);

    ggml_backend_t backend = NULL;
    switch (which) {
    case 0: backend = Rggml_backend_cpu_init_ptr()(); break;
    case 1: backend = Rggml_backend_blas_init_ptr()(); break;
    case 2: backend = Rggml_backend_vulkan_init_ptr()(0); break;
    case 3: backend = Rggml_backend_cuda_init_ptr()(0); break;
    default: Rf_error("backend must be 0 (cpu), 1 (blas), 2 (vulkan) or 3 (cuda)");
    }
    if (!backend) Rf_error("backend %d unavailable (GPU backends require an enabled build and a device)", which);

    /* Initialize CPU type traits before host-side quantization even when the
     * graph itself is destined for a GPU. */
    ggml_backend_t cpu = Rggml_backend_cpu_init_ptr()();
    if (!cpu) {
        Rggml_backend_free_ptr()(backend);
        Rf_error("CPU type-trait initialization failed");
    }

    R_xlen_t nElemA = (R_xlen_t) k * nA;
    R_xlen_t nElemB = (R_xlen_t) k * nB;
    float *af = (float *) R_alloc((size_t) nElemA, sizeof(float));
    float *bf = (float *) R_alloc((size_t) nElemB, sizeof(float));
    for (R_xlen_t i = 0; i < nElemA; i++) af[i] = (float) REAL(A_sexp)[i];
    for (R_xlen_t i = 0; i < nElemB; i++) bf[i] = (float) REAL(B_sexp)[i];

    size_t qa_bytes = ggml_row_size(type, k) * (size_t) nA;
    void *qa = R_alloc(qa_bytes > 0 ? qa_bytes : 1, 1);
    ggml_quantize_chunk(type, af, qa, 0, nA, k, NULL);
    Rggml_backend_free_ptr()(cpu);

    Rggml_context_create_fun ctx_create = Rggml_context_create_ptr();
    Rggml_context_free_fun ctx_free = Rggml_context_free_ptr();
    Rggml_backend_free_fun bfree = Rggml_backend_free_ptr();
    Rggml_tensor_overhead_fun t_over = Rggml_tensor_overhead_ptr();
    Rggml_graph_overhead_fun g_over = Rggml_graph_overhead_ptr();
    size_t mem = (size_t) 8 * t_over() + g_over(8) + 4096;
    struct ggml_context *ctx = ctx_create(mem, /*no_alloc=*/1);
    if (!ctx) { bfree(backend); Rf_error("context creation failed"); }

    int64_t neA[2] = { k, nA }, neB[2] = { k, nB };
    struct ggml_tensor *tA = Rggml_new_tensor_ptr()(ctx, type, 2, neA, NULL);
    struct ggml_tensor *tB = Rggml_new_tensor_ptr()(ctx, GGML_TYPE_F32, 2, neB, NULL);
    struct ggml_tensor *tC = (tA && tB) ? Rggml_mul_mat_ptr()(ctx, tA, tB) : NULL;
    if (!tC) { ctx_free(ctx); bfree(backend); Rf_error("graph construction failed"); }
    struct ggml_cgraph *gf = Rggml_new_graph_ptr()(ctx, 8);
    if (!gf) { ctx_free(ctx); bfree(backend); Rf_error("graph allocation failed"); }
    Rggml_build_forward_expand_ptr()(gf, tC);

    ggml_backend_buffer_t buf = Rggml_backend_alloc_ctx_tensors_ptr()(ctx, backend);
    if (!buf) { ctx_free(ctx); bfree(backend); Rf_error("backend buffer allocation failed"); }
    Rggml_backend_tensor_set_ptr()(tA, qa, 0, qa_bytes);
    Rggml_backend_tensor_set_ptr()(tB, bf, 0, (size_t) nElemB * sizeof(float));

    int rc = Rggml_backend_graph_compute_ptr()(backend, gf);
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, (int) nA, (int) nB));
    if (rc == 0) {
        R_xlen_t n = (R_xlen_t) nA * nB;
        float *cf = (float *) R_alloc((size_t) n, sizeof(float));
        Rggml_backend_tensor_get_ptr()(tC, cf, 0, (size_t) n * sizeof(float));
        for (R_xlen_t i = 0; i < n; i++) REAL(out)[i] = (double) cf[i];
    }
    Rggml_backend_buffer_free_ptr()(buf);
    ctx_free(ctx);
    bfree(backend);
    if (rc != 0) {
        UNPROTECT(1);
        Rf_error("%s graph compute failed with status %d", type_name, rc);
    }
    UNPROTECT(1);
    return out;
}

/*
 * RC_rggml_test_mul_mat_backend(A, B, backend)
 *
 * The backend-agnostic path: build the mul_mat graph in a no_alloc context,
 * let the backend allocate every tensor in one of its own buffers, upload the
 * inputs, compute, download the result. Identical code for CPU (0), BLAS (1),
 * Vulkan (2) and CUDA (3) - which is exactly the point: a GPU backend's
 * tensors live in device memory, so the host-pointer shortcut
 * Rggml_compute_mul_mat() takes cannot work there.
 *
 * Returns a numeric matrix, dim (ncol(A), ncol(B)) = crossprod(A, B).
 */
SEXP RC_rggml_test_mul_mat_backend(SEXP A_sexp, SEXP B_sexp, SEXP backend_sexp)
{
    if (TYPEOF(A_sexp) != REALSXP || TYPEOF(B_sexp) != REALSXP) {
        Rf_error("A and B must be numeric matrices");
    }
    SEXP dimA = Rf_getAttrib(A_sexp, R_DimSymbol);
    SEXP dimB = Rf_getAttrib(B_sexp, R_DimSymbol);
    if (Rf_length(dimA) != 2 || Rf_length(dimB) != 2) Rf_error("A and B must be matrices");
    int64_t k = INTEGER(dimA)[0], nA = INTEGER(dimA)[1];
    int64_t kB = INTEGER(dimB)[0], nB = INTEGER(dimB)[1];
    if (k != kB) Rf_error("nrow(A) must equal nrow(B)");
    int which = Rf_asInteger(backend_sexp);

    Rggml_context_create_fun   ctx_create = Rggml_context_create_ptr();
    Rggml_context_free_fun     ctx_free   = Rggml_context_free_ptr();
    Rggml_new_tensor_fun       new_tensor = Rggml_new_tensor_ptr();
    Rggml_mul_mat_fun          mul_mat    = Rggml_mul_mat_ptr();
    Rggml_new_graph_fun        new_graph  = Rggml_new_graph_ptr();
    Rggml_build_forward_expand_fun expand = Rggml_build_forward_expand_ptr();
    Rggml_backend_graph_compute_fun compute = Rggml_backend_graph_compute_ptr();
    Rggml_backend_free_fun     bfree      = Rggml_backend_free_ptr();
    Rggml_tensor_overhead_fun  t_over     = Rggml_tensor_overhead_ptr();
    Rggml_graph_overhead_fun   g_over     = Rggml_graph_overhead_ptr();
    Rggml_backend_alloc_ctx_tensors_fun alloc_tensors = Rggml_backend_alloc_ctx_tensors_ptr();
    Rggml_backend_buffer_free_fun buf_free = Rggml_backend_buffer_free_ptr();
    Rggml_backend_tensor_set_fun  t_set    = Rggml_backend_tensor_set_ptr();
    Rggml_backend_tensor_get_fun  t_get    = Rggml_backend_tensor_get_ptr();

    ggml_backend_t backend = NULL;
    switch (which) {
    case 0: backend = Rggml_backend_cpu_init_ptr()(); break;
    case 1: backend = Rggml_backend_blas_init_ptr()(); break;
    case 2: backend = Rggml_backend_vulkan_init_ptr()(0); break;
    case 3: backend = Rggml_backend_cuda_init_ptr()(0); break;
    default: Rf_error("backend must be 0 (cpu), 1 (blas), 2 (vulkan) or 3 (cuda)");
    }
    if (!backend) Rf_error("backend %d unavailable (GPU backends require an enabled build and a device)", which);

    size_t mem = (size_t) 8 * t_over() + g_over(8) + 4096;
    struct ggml_context *ctx = ctx_create(mem, /*no_alloc=*/1);
    if (!ctx) { bfree(backend); Rf_error("context creation failed"); }

    int64_t neA[2] = { k, nA }, neB[2] = { k, nB };
    struct ggml_tensor *tA = new_tensor(ctx, GGML_TYPE_F32, 2, neA, NULL);
    struct ggml_tensor *tB = new_tensor(ctx, GGML_TYPE_F32, 2, neB, NULL);
    struct ggml_tensor *tC = (tA && tB) ? mul_mat(ctx, tA, tB) : NULL;
    if (!tC) { ctx_free(ctx); bfree(backend); Rf_error("graph construction failed"); }

    struct ggml_cgraph *gf = new_graph(ctx, 8);
    if (!gf) { ctx_free(ctx); bfree(backend); Rf_error("graph alloc failed"); }
    expand(gf, tC);

    /* one backend buffer holds tA, tB and tC - device memory for a GPU backend */
    ggml_backend_buffer_t buf = alloc_tensors(ctx, backend);
    if (!buf) { ctx_free(ctx); bfree(backend); Rf_error("backend buffer allocation failed"); }

    R_xlen_t nElemA = (R_xlen_t) k * nA, nElemB = (R_xlen_t) k * nB;
    float *af = (float *) R_alloc((size_t) nElemA, sizeof(float));
    float *bf = (float *) R_alloc((size_t) nElemB, sizeof(float));
    for (R_xlen_t i = 0; i < nElemA; i++) af[i] = (float) REAL(A_sexp)[i];
    for (R_xlen_t i = 0; i < nElemB; i++) bf[i] = (float) REAL(B_sexp)[i];
    t_set(tA, af, 0, (size_t) nElemA * sizeof(float));
    t_set(tB, bf, 0, (size_t) nElemB * sizeof(float));

    int rc = compute(backend, gf);

    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, (int) nA, (int) nB));
    if (rc == 0) {
        R_xlen_t n = (R_xlen_t) nA * nB;
        float *cf = (float *) R_alloc((size_t) n, sizeof(float));
        t_get(tC, cf, 0, (size_t) n * sizeof(float));
        for (R_xlen_t i = 0; i < n; i++) REAL(out)[i] = (double) cf[i];
    }

    buf_free(buf);
    ctx_free(ctx);
    bfree(backend);
    if (rc != 0) { UNPROTECT(1); Rf_error("graph compute failed with status %d", rc); }
    UNPROTECT(1);
    return out;
}

/* RC_rggml_vulkan_info() -> list(n_devices, description of device 0 or NA) */
SEXP RC_rggml_vulkan_info(void)
{
    int n = Rggml_backend_vulkan_device_count_ptr()();
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(out, 0, Rf_ScalarInteger(n));
    if (n > 0) {
        char buf[256];
        buf[0] = '\0';
        if (Rggml_backend_vulkan_device_description_ptr()(0, buf, sizeof(buf)) == 0) {
            SET_VECTOR_ELT(out, 1, Rf_mkString(buf));
        } else {
            SET_VECTOR_ELT(out, 1, Rf_ScalarString(NA_STRING));
        }
    } else {
        SET_VECTOR_ELT(out, 1, Rf_ScalarString(NA_STRING));
    }
    SEXP nm = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(nm, 0, Rf_mkChar("n_devices"));
    SET_STRING_ELT(nm, 1, Rf_mkChar("device"));
    Rf_setAttrib(out, R_NamesSymbol, nm);
    UNPROTECT(2);
    return out;
}

/* RC_rggml_cuda_info() -> list(n_devices, description of device 0 or NA) */
SEXP RC_rggml_cuda_info(void)
{
    int n = Rggml_backend_cuda_device_count_ptr()();
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(out, 0, Rf_ScalarInteger(n));
    if (n > 0) {
        char buf[256];
        buf[0] = '\0';
        if (Rggml_backend_cuda_device_description_ptr()(0, buf, sizeof(buf)) == 0) {
            SET_VECTOR_ELT(out, 1, Rf_mkString(buf));
        } else {
            SET_VECTOR_ELT(out, 1, Rf_ScalarString(NA_STRING));
        }
    } else {
        SET_VECTOR_ELT(out, 1, Rf_ScalarString(NA_STRING));
    }
    SEXP nm = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(nm, 0, Rf_mkChar("n_devices"));
    SET_STRING_ELT(nm, 1, Rf_mkChar("device"));
    Rf_setAttrib(out, R_NamesSymbol, nm);
    UNPROTECT(2);
    return out;
}

/*
 * RC_rggml_cpu_info() ->
 *   list(arch_kernels, simd_dispatch, blas, sgemm, vulkan, cuda)
 *
 * Reports what configure actually compiled, read straight off the preprocessor
 * symbols it put in PKG_CPPFLAGS. This exists because a build that silently
 * misses its intended branch - GGML_CPU_GENERIC where NEON kernels were meant,
 * the dgemm_ promotion where sgemm_ was meant - still compiles, still passes
 * every numerical test, and is indistinguishable from the intended one. It is
 * cheaper to make the build state observable than to audit disassembly.
 */
SEXP RC_rggml_cpu_info(void)
{
    static const char *const kernels =
#ifdef RGGML_ARCH_ARM
        "arm";
#elif defined(RGGML_ARCH_WASM)
        "wasm";
#else
        "generic";
#endif
    const Rboolean dispatch =
#if defined(RGGML_SIMD_DISPATCH) && RGGML_SIMD_DISPATCH
        TRUE;
#else
        FALSE;
#endif
    const Rboolean sgemm =
#ifdef RGGML_HAVE_SGEMM
        TRUE;
#else
        FALSE;
#endif
    const Rboolean blas =
#ifdef RGGML_HAVE_BLAS
        TRUE;
#else
        FALSE;
#endif
    const Rboolean vulkan =
#ifdef RGGML_HAVE_VULKAN
        TRUE;
#else
        FALSE;
#endif
    const Rboolean cuda =
#ifdef RGGML_HAVE_CUDA
        TRUE;
#else
        FALSE;
#endif

    SEXP out = PROTECT(Rf_allocVector(VECSXP, 6));
    SET_VECTOR_ELT(out, 0, Rf_mkString(kernels));
    SET_VECTOR_ELT(out, 1, Rf_ScalarLogical(dispatch));
    SET_VECTOR_ELT(out, 2, Rf_ScalarLogical(blas));
    SET_VECTOR_ELT(out, 3, Rf_ScalarLogical(sgemm));
    SET_VECTOR_ELT(out, 4, Rf_ScalarLogical(vulkan));
    SET_VECTOR_ELT(out, 5, Rf_ScalarLogical(cuda));

    SEXP nm = PROTECT(Rf_allocVector(STRSXP, 6));
    SET_STRING_ELT(nm, 0, Rf_mkChar("arch_kernels"));
    SET_STRING_ELT(nm, 1, Rf_mkChar("simd_dispatch"));
    SET_STRING_ELT(nm, 2, Rf_mkChar("blas"));
    SET_STRING_ELT(nm, 3, Rf_mkChar("sgemm"));
    SET_STRING_ELT(nm, 4, Rf_mkChar("vulkan"));
    SET_STRING_ELT(nm, 5, Rf_mkChar("cuda"));
    Rf_setAttrib(out, R_NamesSymbol, nm);
    UNPROTECT(2);
    return out;
}

/* Focused differential proof for the persistent packed F32 conv1d primitive.
 * It intentionally resolves every public operation through R_GetCCallable(),
 * including plan creation/application, so this remains a downstream ABI test.
 */
static int
rggml_test_conv_output_length(int64_t input, int stride, int padding,
                              int dilation, int64_t kernel, int64_t *output)
{
    int64_t effective;
    if (input < 1 || stride < 1 || padding < 0 || dilation < 1 || kernel < 1 ||
        padding > (INT64_MAX - input) / 2 ||
        kernel - 1 > (INT64_MAX - 1) / dilation) {
        return -1;
    }
    effective = (kernel - 1) * dilation + 1;
    if (input + 2 * (int64_t) padding < effective) return -1;
    *output = (input + 2 * (int64_t) padding - effective) / stride + 1;
    return *output < 1 ? -1 : 0;
}

static int
rggml_test_conv_compute(const float *kernel, const float *bias, const float *input,
    int64_t k, int64_t ic, int64_t oc, int64_t n, int64_t batch,
    int stride, int padding, int dilation, int scalar_only, int upstream,
    float *output, size_t output_bytes)
{
    Rggml_context_create_fun context_create = Rggml_context_create_ptr();
    Rggml_context_free_fun context_free = Rggml_context_free_ptr();
    Rggml_tensor_overhead_fun tensor_overhead = Rggml_tensor_overhead_ptr();
    Rggml_graph_overhead_fun graph_overhead = Rggml_graph_overhead_ptr();
    Rggml_new_tensor_fun new_tensor = Rggml_new_tensor_ptr();
    Rggml_new_graph_fun new_graph = Rggml_new_graph_ptr();
    Rggml_build_forward_expand_fun expand = Rggml_build_forward_expand_ptr();
    Rggml_backend_alloc_ctx_tensors_fun alloc = Rggml_backend_alloc_ctx_tensors_ptr();
    Rggml_backend_buffer_free_fun buffer_free = Rggml_backend_buffer_free_ptr();
    Rggml_backend_tensor_set_fun tensor_set = Rggml_backend_tensor_set_ptr();
    Rggml_backend_tensor_get_fun tensor_get = Rggml_backend_tensor_get_ptr();
    Rggml_backend_cpu_init_fun cpu_init = Rggml_backend_cpu_init_ptr();
    Rggml_backend_free_fun backend_free = Rggml_backend_free_ptr();
    Rggml_backend_graph_compute_fun compute = Rggml_backend_graph_compute_ptr();
    Rggml_conv_1d_fun oracle_conv = Rggml_conv_1d_ptr();
    Rggml_add_fun add = Rggml_add_ptr();
    Rggml_conv_1d_f32_plan_create_fun plan_create = Rggml_conv_1d_f32_plan_create_ptr();
    Rggml_conv_1d_f32_plan_destroy_fun plan_destroy = Rggml_conv_1d_f32_plan_destroy_ptr();
    Rggml_conv_1d_f32_plan_apply_fun plan_apply = Rggml_conv_1d_f32_plan_apply_ptr();
    struct ggml_context *ctx = NULL;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer = NULL;
    Rggml_conv_1d_f32_plan *plan = NULL;
    struct ggml_tensor *x, *result;
    int64_t input_ne[3] = { n, ic, batch };
    int64_t kernel_ne[3] = { k, ic, oc };
    int64_t bias_ne[3] = { 1, oc, 1 };
    size_t metadata;
    int status = -1;

    if (n > INT64_MAX / ic || n * ic > INT64_MAX / batch ||
        (size_t) n * ic > SIZE_MAX / (size_t) batch ||
        (size_t) n * ic * batch > SIZE_MAX / sizeof(float) ||
        (size_t) k * ic > SIZE_MAX / (size_t) oc ||
        (size_t) k * ic * oc > SIZE_MAX / sizeof(float)) return -1;
    metadata = 16 * tensor_overhead() + graph_overhead(32) + 4096;
    ctx = context_create(metadata, 1);
    backend = cpu_init();
    if (!ctx || !backend) goto done;
    x = new_tensor(ctx, GGML_TYPE_F32, 3, input_ne, NULL);
    if (!x) goto done;
    if (upstream) {
        struct ggml_tensor *w = new_tensor(ctx, GGML_TYPE_F32, 3, kernel_ne, (void *) kernel);
        struct ggml_tensor *b = new_tensor(ctx, GGML_TYPE_F32, 3, bias_ne, NULL);
        if (!w || !b || !(result = oracle_conv(ctx, w, x, stride, padding, dilation))) goto done;
        result = add(ctx, result, b);
        if (!result) goto done;
    } else {
        plan = plan_create(kernel, (size_t) k * ic * oc * sizeof(float), bias,
                           (size_t) oc * sizeof(float), k, ic, oc, stride,
                           padding, dilation, oc);
        if (!plan) goto done;
        plan->scalar_only = scalar_only;
        result = plan_apply(ctx, plan, x);
        if (!result) goto done;
    }
    struct ggml_cgraph *graph = new_graph(ctx, 32);
    if (!graph) goto done;
    expand(graph, result);
    buffer = alloc(ctx, backend);
    if (!buffer) goto done;
    tensor_set(x, input, 0, (size_t) n * ic * batch * sizeof(float));
    if (upstream) {
        /* The bias is the only backend-owned leaf after allocation. It is
         * the second source of the final add node. */
        tensor_set(result->src[1], bias, 0, (size_t) oc * sizeof(float));
    }
    if (compute(backend, graph) != 0) goto done;
    tensor_get(result, output, 0, output_bytes);
    status = 0;
done:
    if (plan) plan_destroy(plan);
    if (buffer) buffer_free(buffer);
    if (backend) backend_free(backend);
    if (ctx) context_free(ctx);
    return status;
}

SEXP
RC_rggml_test_conv1d_f32(SEXP kernel_sexp, SEXP input_sexp, SEXP bias_sexp,
                         SEXP stride_sexp, SEXP padding_sexp, SEXP dilation_sexp)
{
    SEXP kd = Rf_getAttrib(kernel_sexp, R_DimSymbol);
    SEXP xd = Rf_getAttrib(input_sexp, R_DimSymbol);
    int stride = Rf_asInteger(stride_sexp), padding = Rf_asInteger(padding_sexp);
    int dilation = Rf_asInteger(dilation_sexp);
    int64_t k, ic, oc, n, batch, nout;
    size_t kernel_count, input_count, output_count, output_bytes;

    if (TYPEOF(kernel_sexp) != REALSXP || TYPEOF(input_sexp) != REALSXP ||
        TYPEOF(bias_sexp) != REALSXP || TYPEOF(kd) != INTSXP ||
        TYPEOF(xd) != INTSXP || XLENGTH(kd) != 3 || XLENGTH(xd) != 3 ||
        stride == NA_INTEGER || padding == NA_INTEGER || dilation == NA_INTEGER) {
        Rf_error("conv1d test needs numeric [K, IC, OC] kernel, [N, IC, B] input, and bias");
    }
    k = INTEGER(kd)[0]; ic = INTEGER(kd)[1]; oc = INTEGER(kd)[2];
    n = INTEGER(xd)[0]; batch = INTEGER(xd)[2];
    if (k < 1 || ic < 1 || oc < 1 || n < 1 || batch < 1 ||
        INTEGER(xd)[1] != ic || XLENGTH(bias_sexp) != oc ||
        rggml_test_conv_output_length(n, stride, padding, dilation, k, &nout) ||
        (size_t) k > SIZE_MAX / (size_t) ic ||
        (kernel_count = (size_t) k * ic) > SIZE_MAX / (size_t) oc ||
        (kernel_count *= (size_t) oc) != (size_t) XLENGTH(kernel_sexp) ||
        (size_t) n > SIZE_MAX / (size_t) ic ||
        (input_count = (size_t) n * ic) > SIZE_MAX / (size_t) batch ||
        (input_count *= (size_t) batch) != (size_t) XLENGTH(input_sexp) ||
        (size_t) nout > SIZE_MAX / (size_t) oc ||
        (output_count = (size_t) nout * oc) > SIZE_MAX / (size_t) batch ||
        (output_count *= (size_t) batch) > SIZE_MAX / sizeof(float)) {
        Rf_error("conv1d test dimensions are invalid or overflow");
    }
    output_bytes = output_count * sizeof(float);
    float *kernel = (float *) R_alloc(kernel_count, sizeof(*kernel));
    float *input = (float *) R_alloc(input_count, sizeof(*input));
    float *bias = (float *) R_alloc((size_t) oc, sizeof(*bias));
    float *direct = (float *) R_alloc(output_count, sizeof(*direct));
    float *scalar = (float *) R_alloc(output_count, sizeof(*scalar));
    float *oracle = (float *) R_alloc(output_count, sizeof(*oracle));
    for (size_t i = 0; i < kernel_count; ++i) kernel[i] = (float) REAL(kernel_sexp)[i];
    for (size_t i = 0; i < input_count; ++i) input[i] = (float) REAL(input_sexp)[i];
    for (int64_t i = 0; i < oc; ++i) bias[i] = (float) REAL(bias_sexp)[i];
    if (rggml_test_conv_compute(kernel, bias, input, k, ic, oc, n, batch,
            stride, padding, dilation, 0, 0, direct, output_bytes) ||
        rggml_test_conv_compute(kernel, bias, input, k, ic, oc, n, batch,
            stride, padding, dilation, 1, 0, scalar, output_bytes) ||
        rggml_test_conv_compute(kernel, bias, input, k, ic, oc, n, batch,
            stride, padding, dilation, 0, 1, oracle, output_bytes)) {
        Rf_error("native conv1d differential graph construction or compute failed");
    }
    SEXP dim = PROTECT(Rf_allocVector(INTSXP, 3));
    INTEGER(dim)[0] = (int) nout; INTEGER(dim)[1] = (int) oc; INTEGER(dim)[2] = (int) batch;
    SEXP result = PROTECT(Rf_allocVector(VECSXP, 3));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 3));
    const char *labels[] = { "direct", "scalar", "im2col" };
    float *sets[] = { direct, scalar, oracle };
    for (int set = 0; set < 3; ++set) {
        SEXP values = PROTECT(Rf_allocArray(REALSXP, dim));
        for (size_t i = 0; i < output_count; ++i) REAL(values)[i] = sets[set][i];
        SET_VECTOR_ELT(result, set, values);
        SET_STRING_ELT(names, set, Rf_mkChar(labels[set]));
        UNPROTECT(1);
    }
    Rf_setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(3);
    return result;
}

SEXP
RC_rggml_test_conv1d_f32_bounds(void)
{
    Rggml_conv_1d_f32_plan_create_fun create = Rggml_conv_1d_f32_plan_create_ptr();
    Rggml_conv_1d_f32_plan_destroy_fun destroy = Rggml_conv_1d_f32_plan_destroy_ptr();
    Rggml_conv_1d_f32_plan_apply_fun apply = Rggml_conv_1d_f32_plan_apply_ptr();
    Rggml_context_create_fun context_create = Rggml_context_create_ptr();
    Rggml_context_free_fun context_free = Rggml_context_free_ptr();
    Rggml_new_tensor_fun new_tensor = Rggml_new_tensor_ptr();
    Rggml_tensor_overhead_fun overhead = Rggml_tensor_overhead_ptr();
    float value[] = { 1.0f, 1.0f };
    int64_t ne[3] = { 5, 2, 1 };
    Rggml_conv_1d_f32_plan *plan = create(value, sizeof(value[0]), NULL, 0,
        1, 1, 1, 1, 0, 1, 1);
    struct ggml_context *ctx = context_create(4 * overhead() + 1024, 1);
    struct ggml_tensor *wrong_channels = ctx
        ? new_tensor(ctx, GGML_TYPE_F32, 3, ne, NULL) : NULL;
    int ok[] = {
        create(NULL, 0, NULL, 0, 1, 1, 1, 1, 0, 1, 1) == NULL,
        create(value, sizeof(value), NULL, 0, INT64_MAX, 2, 1, 1, 0, 1, 1) == NULL,
        create(value, sizeof(value), NULL, 0, 1, 1, 65537, 1, 0, 1, 65537) == NULL,
        create(value, sizeof(value), NULL, 0, 2, 1, 1, 1, 0,
               INT64_MAX, 1) == NULL,
        plan != NULL && wrong_channels != NULL && apply(ctx, plan, wrong_channels) == NULL
    };
    if (plan) destroy(plan);
    if (ctx) context_free(ctx);
    SEXP result = PROTECT(Rf_allocVector(LGLSXP, 5));
    for (int i = 0; i < 5; ++i) LOGICAL(result)[i] = ok[i];
    UNPROTECT(1);
    return result;
}
