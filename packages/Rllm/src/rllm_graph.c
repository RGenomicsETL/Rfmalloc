/*
 * rllm_graph.c - lower a bound semantic program through Rggml's C-callable
 * graph ops over borrowed GGUF weight spans. Quantized and float weights point
 * directly into the model's read-only mapping. Generative programs may append
 * to persistent state; embedding programs consume a complete sequence.
 *
 * CPU passes use a no_alloc weight context whose tensors point at R-owned
 * payloads. CUDA creates that context once per model, uploads the encoded
 * weights once, and reuses it; each pass builds a separate transient compute
 * context for inputs, cache mirrors, intermediates and logits.
 */
#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <R.h>
#include <Rinternals.h>

#include <Rfmalloc.h>
#include <Rggml.h>

/* From rllm_backend.c: codec-name -> quantized ggml type (declines floats). */
int rllm_ggml_type_from_codec(const char *codec, enum ggml_type *out);

/* Weight-tensor types additionally include the float storage formats. */
static int rllm_ggml_type_from_name(const char *name, enum ggml_type *out)
{
    if (!name) return -1;
    if (!strcmp(name, "f32")) { *out = GGML_TYPE_F32; return 0; }
    if (!strcmp(name, "f16")) { *out = GGML_TYPE_F16; return 0; }
    return rllm_ggml_type_from_codec(name, out);
}

static SEXP rllm_list_elt(SEXP list, const char *name)
{
    SEXP names = Rf_getAttrib(list, R_NamesSymbol);
    if (TYPEOF(names) != STRSXP) return R_NilValue;
    for (R_xlen_t i = 0; i < XLENGTH(list); i++) {
        if (!strcmp(CHAR(STRING_ELT(names, i)), name)) {
            return VECTOR_ELT(list, i);
        }
    }
    return R_NilValue;
}

static double rllm_hparam(SEXP hparams, const char *name)
{
    SEXP v = rllm_list_elt(hparams, name);
    if (v == R_NilValue) Rf_error("missing hyperparameter '%s'", name);
    return Rf_asReal(v);
}

static const char *rllm_string(SEXP list, const char *name)
{
    SEXP v = rllm_list_elt(list, name);
    if (TYPEOF(v) != STRSXP || XLENGTH(v) != 1 ||
        STRING_ELT(v, 0) == NA_STRING) {
        Rf_error("program field '%s' must be one string", name);
    }
    return CHAR(STRING_ELT(v, 0));
}

static const char *rllm_optional_string(SEXP list, const char *name)
{
    SEXP v = rllm_list_elt(list, name);
    if (v == R_NilValue) return NULL;
    if (TYPEOF(v) != STRSXP || XLENGTH(v) != 1 ||
        STRING_ELT(v, 0) == NA_STRING) {
        Rf_error("program field '%s' must be NULL or one string", name);
    }
    return CHAR(STRING_ELT(v, 0));
}

static int rllm_integer(SEXP list, const char *name)
{
    SEXP v = rllm_list_elt(list, name);
    int out = Rf_asInteger(v);
    if (v == R_NilValue || out == NA_INTEGER) {
        Rf_error("program field '%s' must be one integer", name);
    }
    return out;
}

static int rllm_named_integer(SEXP vector, const char *name)
{
    SEXP names = Rf_getAttrib(vector, R_NamesSymbol);
    if (TYPEOF(vector) != INTSXP && TYPEOF(vector) != REALSXP) {
        Rf_error("program vector '%s' must be numeric", name);
    }
    if (TYPEOF(names) != STRSXP || XLENGTH(names) != XLENGTH(vector)) {
        Rf_error("program vector must name '%s'", name);
    }
    for (R_xlen_t i = 0; i < XLENGTH(vector); i++) {
        if (STRING_ELT(names, i) == NA_STRING) {
            Rf_error("program vector has a missing field name");
        }
        if (strcmp(CHAR(STRING_ELT(names, i)), name)) continue;
        double value;
        if (TYPEOF(vector) == INTSXP) value = INTEGER(vector)[i];
        else if (TYPEOF(vector) == REALSXP) value = REAL(vector)[i];
        if (!R_FINITE(value) || value < 1 || value > INT_MAX ||
            value != floor(value)) {
            Rf_error("program vector '%s' must be a positive integer", name);
        }
        return (int)value;
    }
    Rf_error("program vector has no '%s' field", name);
    return 0;
}

static double rllm_number(SEXP list, const char *name)
{
    SEXP v = rllm_list_elt(list, name);
    double out = Rf_asReal(v);
    if (v == R_NilValue || !R_FINITE(out)) {
        Rf_error("program field '%s' must be one finite number", name);
    }
    return out;
}

static int rllm_boolean(SEXP list, const char *name)
{
    SEXP v = rllm_list_elt(list, name);
    if (TYPEOF(v) != LGLSXP || XLENGTH(v) != 1 ||
        LOGICAL(v)[0] == NA_LOGICAL) {
        Rf_error("program field '%s' must be TRUE or FALSE", name);
    }
    return LOGICAL(v)[0];
}

struct rllm_upload;
struct rllm_cuda_context;

static struct ggml_tensor *rllm_named_weight(
    struct ggml_context *wctx, SEXP tensors, const char *name,
    Rggml_new_tensor_fun new_tensor, Rfmalloc_storage_data_fun storage_data,
    struct rllm_upload *uploads, int *n_uploads, int max_uploads,
    struct rllm_cuda_context *cuda_ctx);

struct rllm_upload {
    struct ggml_tensor *tensor;
    const void *data;
    size_t bytes;
};

struct rllm_download {
    const struct ggml_tensor *tensor;
    void *data;
    size_t bytes;
    size_t tensor_offset;
    float *scatter;
    int64_t stride;
    int64_t scatter_offset;
    int64_t rows;
    int64_t cols;
};

struct rllm_cuda_context {
    ggml_backend_t backend;
    struct ggml_context *wctx;
    ggml_backend_buffer_t wbuf;
    struct ggml_tensor **weights;
    R_xlen_t n_weights;
    Rggml_backend_buffer_free_fun buffer_free;
    Rggml_context_free_fun context_free;
    Rggml_backend_free_fun backend_free;
};

static SEXP rllm_cuda_context_tag(void)
{
    return Rf_install("Rllm_cuda_model_context");
}

static void rllm_cuda_context_finalizer(SEXP ext)
{
    struct rllm_cuda_context *ctx =
        (struct rllm_cuda_context *)R_ExternalPtrAddr(ext);

    if (!ctx) return;
    if (ctx->wbuf && ctx->buffer_free) ctx->buffer_free(ctx->wbuf);
    if (ctx->wctx && ctx->context_free) ctx->context_free(ctx->wctx);
    if (ctx->backend && ctx->backend_free) ctx->backend_free(ctx->backend);
    free(ctx->weights);
    free(ctx);
    R_ClearExternalPtr(ext);
}

static struct rllm_cuda_context *rllm_cuda_context_get(SEXP ext,
                                                       SEXP tensors)
{
    if (TYPEOF(ext) != EXTPTRSXP ||
        R_ExternalPtrTag(ext) != rllm_cuda_context_tag()) {
        Rf_error("invalid CUDA model context");
    }
    if (R_ExternalPtrProtected(ext) != tensors) {
        Rf_error("CUDA model context belongs to a different tensor set");
    }
    struct rllm_cuda_context *ctx =
        (struct rllm_cuda_context *)R_ExternalPtrAddr(ext);
    if (!ctx || !ctx->backend || !ctx->wctx || !ctx->wbuf) {
        Rf_error("CUDA model context has been released");
    }
    return ctx;
}

static void rllm_add_upload(struct rllm_upload *uploads, int *n_uploads,
                            int max_uploads, struct ggml_tensor *tensor,
                            const void *data, size_t bytes)
{
    if (!uploads) return;
    if (*n_uploads >= max_uploads) Rf_error("internal upload table overflow");
    uploads[*n_uploads].tensor = tensor;
    uploads[*n_uploads].data = data;
    uploads[*n_uploads].bytes = bytes;
    ++*n_uploads;
}

/*
 * Wrap one named weight from the R side as a ggml tensor in `wctx`.
 * `tensors[[name]]` is list(payload = raw vector (fmalloc-backed or plain),
 * type = codec/type string, dims = integer vector, GGUF dim[0]-first order).
 */
static struct ggml_tensor *rllm_weight(struct ggml_context *wctx, SEXP tensors,
                                       const char *name,
                                       Rggml_new_tensor_fun new_tensor,
                                       Rfmalloc_storage_data_fun storage_data,
                                       struct rllm_upload *uploads,
                                       int *n_uploads, int max_uploads)
{
    SEXP w = rllm_list_elt(tensors, name);
    if (w == R_NilValue) Rf_error("missing weight tensor '%s'", name);

    SEXP payload = rllm_list_elt(w, "payload");
    SEXP type_s  = rllm_list_elt(w, "type");
    SEXP dims_s  = rllm_list_elt(w, "dims");
    if (TYPEOF(type_s) != STRSXP || XLENGTH(type_s) != 1 ||
        STRING_ELT(type_s, 0) == NA_STRING || TYPEOF(dims_s) != INTSXP) {
        Rf_error("weight '%s' must be list(payload, type = chr, dims = int)", name);
    }

    enum ggml_type type;
    if (rllm_ggml_type_from_name(CHAR(STRING_ELT(type_s, 0)), &type) != 0) {
        Rf_error("weight '%s' has unsupported type '%s'", name,
                 CHAR(STRING_ELT(type_s, 0)));
    }

    int n_dims = (int) XLENGTH(dims_s);
    if (n_dims < 1 || n_dims > GGML_MAX_DIMS) {
        Rf_error("weight '%s': expected 1 to %d dims, got %d",
                 name, GGML_MAX_DIMS, n_dims);
    }
    for (int i = 0; i < n_dims; ++i) {
        if (INTEGER(dims_s)[i] == NA_INTEGER || INTEGER(dims_s)[i] <= 0) {
            Rf_error("weight '%s': dimensions must be positive", name);
        }
    }
    int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
    for (int i = 0; i < n_dims; ++i) ne[i] = INTEGER(dims_s)[i];

    const void *payload_data;
    size_t payload_bytes;
    if (storage_data(payload, &payload_data, &payload_bytes, NULL) != 0) {
        Rf_error("weight '%s': unsupported storage payload", name);
    }
    Rggml_row_size_fun row_size = Rggml_row_size_ptr();
    const size_t one_row = row_size(type, ne[0]);
    if (one_row == 0) Rf_error("weight '%s': invalid row size", name);
    size_t required = one_row;
    for (int i = 1; i < n_dims; ++i) {
        if ((uint64_t)ne[i] > SIZE_MAX / required) {
            Rf_error("weight '%s': payload size overflows size_t", name);
        }
        required *= (size_t)ne[i];
    }
    if (payload_bytes < required) {
        Rf_error("weight '%s': payload too short", name);
    }

    struct ggml_tensor *t = new_tensor(wctx, type, n_dims, ne,
                                       uploads ? NULL : (void *)payload_data);
    if (!t) Rf_error("failed to wrap weight '%s'", name);
    rllm_add_upload(uploads, n_uploads, max_uploads, t, payload_data,
                    required);
    return t;
}

static struct ggml_tensor *rllm_cuda_weight(struct rllm_cuda_context *ctx,
                                            SEXP tensors, const char *name)
{
    SEXP names = Rf_getAttrib(tensors, R_NamesSymbol);

    if (TYPEOF(names) != STRSXP || XLENGTH(names) != ctx->n_weights) {
        Rf_error("model tensors must retain their names");
    }
    for (R_xlen_t i = 0; i < ctx->n_weights; i++) {
        if (!strcmp(CHAR(STRING_ELT(names, i)), name)) return ctx->weights[i];
    }
    Rf_error("missing CUDA weight tensor '%s'", name);
    return NULL;
}

static struct ggml_tensor *rllm_named_weight(
    struct ggml_context *wctx, SEXP tensors, const char *name,
    Rggml_new_tensor_fun new_tensor, Rfmalloc_storage_data_fun storage_data,
    struct rllm_upload *uploads, int *n_uploads, int max_uploads,
    struct rllm_cuda_context *cuda_ctx)
{
    return cuda_ctx
        ? rllm_cuda_weight(cuda_ctx, tensors, name)
        : rllm_weight(wctx, tensors, name, new_tensor, storage_data,
                      uploads, n_uploads, max_uploads);
}

/* Upload the immutable, codec-native model weights once. The external
 * pointer protects the R tensor list, which in turn keeps every borrowed GGUF
 * mapping alive for the lifetime of the device copy. */
SEXP RC_rllm_cuda_model_context(SEXP tensors)
{
    if (TYPEOF(tensors) != VECSXP || XLENGTH(tensors) < 1 ||
        XLENGTH(tensors) > INT_MAX) {
        Rf_error("tensors must be a non-empty named list");
    }
    SEXP names = Rf_getAttrib(tensors, R_NamesSymbol);
    if (TYPEOF(names) != STRSXP || XLENGTH(names) != XLENGTH(tensors)) {
        Rf_error("tensors must be a named list");
    }
    struct rllm_cuda_context *ctx =
        (struct rllm_cuda_context *)calloc(1, sizeof(*ctx));
    if (!ctx) Rf_error("CUDA model context allocation failed");

    SEXP ext = PROTECT(R_MakeExternalPtr(ctx, rllm_cuda_context_tag(), tensors));
    R_RegisterCFinalizerEx(ext, rllm_cuda_context_finalizer, FALSE);

    ctx->buffer_free = Rggml_backend_buffer_free_ptr();
    ctx->context_free = Rggml_context_free_ptr();
    ctx->backend_free = Rggml_backend_free_ptr();
    ctx->n_weights = XLENGTH(tensors);
    ctx->weights = (struct ggml_tensor **)calloc(
        (size_t)ctx->n_weights, sizeof(*ctx->weights));
    if (!ctx->weights) Rf_error("CUDA weight table allocation failed");

    Rggml_backend_cuda_init_fun cuda_init = Rggml_backend_cuda_init_ptr();
    Rggml_context_create_fun ctx_create = Rggml_context_create_ptr();
    Rggml_tensor_overhead_fun t_over = Rggml_tensor_overhead_ptr();
    Rggml_new_tensor_fun new_tensor = Rggml_new_tensor_ptr();
    Rggml_backend_alloc_ctx_tensors_fun alloc_tensors =
        Rggml_backend_alloc_ctx_tensors_ptr();
    Rggml_backend_tensor_set_fun tensor_set = Rggml_backend_tensor_set_ptr();
    Rfmalloc_storage_data_fun storage_data = Rfmalloc_storage_data_ptr();

    ctx->backend = cuda_init(0);
    if (!ctx->backend) {
        Rf_error("CUDA backend unavailable: install Rggml with --with-cuda "
                 "and use a visible NVIDIA device");
    }
    ctx->wctx = ctx_create((size_t)(ctx->n_weights + 8) * t_over() + 4096,
                           /*no_alloc=*/1);
    if (!ctx->wctx) Rf_error("CUDA weights context creation failed");

    struct rllm_upload *uploads = (struct rllm_upload *)R_alloc(
        (size_t)ctx->n_weights, sizeof(*uploads));
    int n_uploads = 0;
    for (R_xlen_t i = 0; i < ctx->n_weights; i++) {
        if (STRING_ELT(names, i) == NA_STRING) {
            Rf_error("tensor names cannot be missing");
        }
        ctx->weights[i] = rllm_weight(
            ctx->wctx, tensors, CHAR(STRING_ELT(names, i)), new_tensor,
            storage_data, uploads, &n_uploads, (int)ctx->n_weights);
    }

    ctx->wbuf = alloc_tensors(ctx->wctx, ctx->backend);
    if (!ctx->wbuf) Rf_error("CUDA weight-buffer allocation failed");
    for (int i = 0; i < n_uploads; i++) {
        tensor_set(uploads[i].tensor, uploads[i].data, 0, uploads[i].bytes);
    }

    UNPROTECT(1);
    return ext;
}

/* Lower a validated, bound semantic program to one GGML graph. The R compiler
 * has already checked its dataflow and reduced it to this native operator
 * vocabulary; this code never dispatches on a model-family name. */
SEXP RC_rllm_program_forward(SEXP bound, SEXP tokens_sexp,
                             SEXP kcache, SEXP vcache, SEXP convcache,
                             SEXP recurrent_cache, SEXP n_ctx_sexp,
                             SEXP n_past_sexp,
                             SEXP backend_sexp, SEXP backend_context)
{
    if (TYPEOF(bound) != VECSXP) {
        Rf_error("bound program must be a list");
    }
    SEXP lowering = rllm_list_elt(bound, "lowering");
    SEXP tensors = rllm_list_elt(bound, "bindings");
    if (TYPEOF(lowering) != VECSXP || TYPEOF(tensors) != VECSXP) {
        Rf_error("bound program must contain lowering and bindings lists");
    }
    if (TYPEOF(tokens_sexp) != INTSXP || XLENGTH(tokens_sexp) < 1 ||
        XLENGTH(tokens_sexp) > INT_MAX) {
        Rf_error("tokens must be a non-empty integer vector of 0-based ids");
    }

    SEXP hparams = rllm_list_elt(lowering, "symbols");
    SEXP layers = rllm_list_elt(lowering, "layers");
    SEXP input_spec = rllm_list_elt(lowering, "input");
    SEXP output_spec = rllm_list_elt(lowering, "output");
    if (TYPEOF(hparams) != VECSXP || TYPEOF(layers) != VECSXP ||
        TYPEOF(input_spec) != VECSXP || TYPEOF(output_spec) != VECSXP) {
        Rf_error("program lowering must contain symbols, input, layers and "
                 "output lists");
    }
    const char *input_op = rllm_string(input_spec, "op");
    if (strcmp(input_op, "embedding")) {
        Rf_error("unknown input operator '%s'", input_op);
    }
    const int n_layer = (int) rllm_hparam(hparams, "n_layer");
    const int n_embd  = (int) rllm_hparam(hparams, "n_embd");
    const int n_head  = (int) rllm_hparam(hparams, "n_head");
    const int n_ff    = (int) rllm_hparam(hparams, "n_ff");
    const int n_vocab = (int) rllm_hparam(hparams, "n_vocab");
    const double rms_eps = rllm_hparam(hparams, "rms_eps");
    const int backend_code = Rf_asInteger(backend_sexp);
    const int use_device   = backend_code == 3;
    const int S = (int) XLENGTH(tokens_sexp);

    if (backend_code != 0 && backend_code != 3) {
        Rf_error("backend must be 0 (cpu) or 3 (cuda)");
    }
    if (n_layer < 1 || n_embd < 1 || n_head < 1 || n_ff < 1 ||
        n_vocab < 1 || XLENGTH(layers) != n_layer || rms_eps <= 0) {
        Rf_error("invalid program symbols or layer count");
    }

    int has_shortconv = 0;
    int has_mrope = 0;
    double recurrent_elems = 0;
    int max_ff = n_ff;
    for (int il = 0; il < n_layer; ++il) {
        SEXP layer = VECTOR_ELT(layers, il);
        SEXP op = rllm_list_elt(layer, "operator");
        SEXP ffn = rllm_list_elt(layer, "feed_forward");
        const char *op_name = rllm_string(op, "op");
        const char *ffn_name = rllm_string(ffn, "op");
        if (!strcmp(op_name, "shortconv")) {
            has_shortconv = 1;
        } else if (!strcmp(op_name, "gated_attention")) {
            has_mrope = 1;
        } else if (!strcmp(op_name, "gated_delta_net")) {
            SEXP state = rllm_list_elt(layer, "state");
            SEXP matrix = rllm_list_elt(state, "matrix");
            recurrent_elems +=
                (double)rllm_named_integer(matrix, "row") *
                rllm_named_integer(matrix, "column") *
                rllm_named_integer(matrix, "head");
        } else if (strcmp(op_name, "attention")) {
            Rf_error("layer %d has unknown operator '%s'", il, op_name);
        }
        if (strcmp(ffn_name, "swiglu") && strcmp(ffn_name, "geglu") &&
            strcmp(ffn_name, "moe_swiglu"))
            Rf_error("layer %d has unknown feed-forward operator '%s'", il, ffn_name);
        int width = rllm_integer(ffn, "width");
        if (width < 1) Rf_error("layer %d has invalid feed-forward width", il);
        if (width > max_ff) max_ff = width;
    }
    if (use_device && has_shortconv) {
        Rf_error("CUDA lowering for short-convolution state is not implemented");
    }

    /* -- program-declared persistent state --------------------------------- */
    const int use_cache = kcache != R_NilValue;
    const int n_ctx = Rf_asInteger(n_ctx_sexp);
    const int n_past    = Rf_asInteger(n_past_sexp);
    if (n_ctx == NA_INTEGER || n_ctx < 0 || n_past == NA_INTEGER || n_past < 0) {
        Rf_error("n_ctx and n_past must be non-negative integers");
    }
    if (use_cache) {
        if (TYPEOF(kcache) != VECSXP || TYPEOF(vcache) != VECSXP ||
            TYPEOF(convcache) != VECSXP || TYPEOF(recurrent_cache) != VECSXP ||
            XLENGTH(kcache) != n_layer || XLENGTH(vcache) != n_layer ||
            XLENGTH(convcache) != n_layer ||
            XLENGTH(recurrent_cache) != n_layer) {
            Rf_error("k, v, convolution and recurrent state must be per-layer lists");
        }
        if (n_ctx < 1 || (int64_t) n_past + S > n_ctx) {
            Rf_error("KV cache too small: n_past %d + %d tokens > n_ctx %lld",
                     n_past, S, (long long) n_ctx);
        }
        for (int il = 0; il < n_layer; il++) {
            SEXP state = rllm_list_elt(VECTOR_ELT(layers, il), "state");
            const char *state_op = rllm_string(state, "op");
            R_xlen_t klen = XLENGTH(VECTOR_ELT(kcache, il));
            R_xlen_t vlen = XLENGTH(VECTOR_ELT(vcache, il));
            R_xlen_t clen = XLENGTH(VECTOR_ELT(convcache, il));
            R_xlen_t rlen = XLENGTH(VECTOR_ELT(recurrent_cache, il));
            if (TYPEOF(VECTOR_ELT(kcache, il)) != RAWSXP ||
                TYPEOF(VECTOR_ELT(vcache, il)) != RAWSXP ||
                TYPEOF(VECTOR_ELT(convcache, il)) != RAWSXP ||
                TYPEOF(VECTOR_ELT(recurrent_cache, il)) != RAWSXP) {
                Rf_error("state layer %d must contain raw vectors", il);
            }
            if (!strcmp(state_op, "kv")) {
                int width = rllm_integer(state, "width");
                double bytes = (double)n_ctx * width * sizeof(float);
                if (bytes > R_XLEN_T_MAX || klen != (R_xlen_t)bytes ||
                    vlen != (R_xlen_t)bytes || clen != 0 || rlen != 0) {
                    Rf_error("KV state layer %d has the wrong extent", il);
                }
            } else if (!strcmp(state_op, "conv")) {
                int width = rllm_integer(state, "width");
                int history = rllm_integer(state, "history");
                double bytes = (double)width * history * sizeof(float);
                if (bytes > R_XLEN_T_MAX || clen != (R_xlen_t)bytes ||
                    klen != 0 || vlen != 0 || rlen != 0) {
                    Rf_error("convolution state layer %d has the wrong extent", il);
                }
            } else if (!strcmp(state_op, "gated_delta")) {
                SEXP matrix = rllm_list_elt(state, "matrix");
                SEXP convolution = rllm_list_elt(state, "convolution");
                int rows = rllm_named_integer(matrix, "row");
                int columns = rllm_named_integer(matrix, "column");
                int heads = rllm_named_integer(matrix, "head");
                int width = rllm_named_integer(convolution, "width");
                int history = rllm_named_integer(convolution, "history");
                double cbytes = (double)width * history * sizeof(float);
                double rbytes = (double)rows * columns * heads * sizeof(float);
                if (cbytes > R_XLEN_T_MAX || rbytes > R_XLEN_T_MAX ||
                    clen != (R_xlen_t)cbytes || rlen != (R_xlen_t)rbytes ||
                    klen != 0 || vlen != 0) {
                    Rf_error("gated-delta state layer %d has the wrong extent", il);
                }
            } else if (!strcmp(state_op, "none")) {
                if (klen != 0 || vlen != 0 || clen != 0 || rlen != 0) {
                    Rf_error("stateless layer %d has non-empty state", il);
                }
            } else {
                Rf_error("layer %d has unknown state operator '%s'", il, state_op);
            }
        }
    } else if (n_ctx != 0 || n_past != 0) {
        Rf_error("n_ctx and n_past must be 0 without persistent state");
    }
    const int64_t n_kv = n_past + S;   /* positions attended over */

    /* -- resolve every C-callable up front ---------------------------------- */
    Rggml_context_create_fun     ctx_create = Rggml_context_create_ptr();
    Rggml_context_free_fun       ctx_free   = Rggml_context_free_ptr();
    Rggml_new_tensor_fun         new_tensor = Rggml_new_tensor_ptr();
    Rggml_tensor_data_fun        tdata      = Rggml_tensor_data_ptr();
    Rggml_tensor_overhead_fun    t_over     = Rggml_tensor_overhead_ptr();
    Rggml_graph_overhead_fun     g_over     = Rggml_graph_overhead_ptr();
    Rggml_new_graph_fun          new_graph  = Rggml_new_graph_ptr();
    Rggml_build_forward_expand_fun expand   = Rggml_build_forward_expand_ptr();
    Rggml_backend_cpu_init_fun   cpu_init   = Rggml_backend_cpu_init_ptr();
    Rggml_backend_free_fun       bfree      = Rggml_backend_free_ptr();
    Rggml_backend_graph_compute_fun compute = Rggml_backend_graph_compute_ptr();
    Rggml_backend_alloc_ctx_tensors_fun alloc_tensors =
        Rggml_backend_alloc_ctx_tensors_ptr();
    Rggml_backend_buffer_free_fun buf_free = Rggml_backend_buffer_free_ptr();
    Rggml_backend_tensor_set_fun tensor_set = Rggml_backend_tensor_set_ptr();
    Rggml_backend_tensor_get_fun tensor_get = Rggml_backend_tensor_get_ptr();
    Rggml_mul_mat_fun            mul_mat    = Rggml_mul_mat_ptr();
    Rggml_mul_mat_id_fun         mul_mat_id = Rggml_mul_mat_id_ptr();
    Rggml_get_rows_fun           get_rows   = Rggml_get_rows_ptr();
    Rggml_rms_norm_fun           rms_norm   = Rggml_rms_norm_ptr();
    Rggml_l2_norm_fun            l2_norm    = Rggml_l2_norm_ptr();
    Rggml_mul_fun                mul        = Rggml_mul_ptr();
    Rggml_add_fun                add        = Rggml_add_ptr();
    Rggml_div_fun                divide     = Rggml_div_ptr();
    Rggml_silu_fun               silu       = Rggml_silu_ptr();
    Rggml_geglu_fun              geglu      = Rggml_geglu_ptr();
    Rggml_sigmoid_fun            sigmoid    = Rggml_sigmoid_ptr();
    Rggml_softplus_fun           softplus   = Rggml_softplus_ptr();
    Rggml_scale_fun              scale      = Rggml_scale_ptr();
    Rggml_sum_rows_fun           sum_rows   = Rggml_sum_rows_ptr();
    Rggml_clamp_fun              clamp      = Rggml_clamp_ptr();
    Rggml_argsort_top_k_fun      top_k      = Rggml_argsort_top_k_ptr();
    Rggml_concat_fun             concat     = Rggml_concat_ptr();
    Rggml_ssm_conv_fun           ssm_conv   = Rggml_ssm_conv_ptr();
    Rggml_soft_max_ext_fun       soft_max_ext = Rggml_soft_max_ext_ptr();
    Rggml_rope_fun               rope       = Rggml_rope_ptr();
    Rggml_rope_multi_fun         rope_multi = Rggml_rope_multi_ptr();
    Rggml_gated_delta_net_fun    gated_delta = Rggml_gated_delta_net_ptr();
    Rggml_reshape_2d_fun         reshape_2d = Rggml_reshape_2d_ptr();
    Rggml_reshape_3d_fun         reshape_3d = Rggml_reshape_3d_ptr();
    Rggml_reshape_4d_fun         reshape_4d = Rggml_reshape_4d_ptr();
    Rggml_permute_fun            permute    = Rggml_permute_ptr();
    Rggml_cont_fun               cont       = Rggml_cont_ptr();
    Rggml_transpose_fun          transpose  = Rggml_transpose_ptr();
    Rggml_view_1d_fun            view_1d    = Rggml_view_1d_ptr();
    Rggml_view_2d_fun            view_2d    = Rggml_view_2d_ptr();
    Rggml_view_3d_fun            view_3d    = Rggml_view_3d_ptr();
    Rggml_view_4d_fun            view_4d    = Rggml_view_4d_ptr();
    Rggml_cpy_fun                cpy        = Rggml_cpy_ptr();
    Rfmalloc_storage_data_fun    storage_data = Rfmalloc_storage_data_ptr();

    struct rllm_cuda_context *cuda_ctx = use_device
        ? rllm_cuda_context_get(backend_context, tensors) : NULL;
    ggml_backend_t backend = use_device ? cuda_ctx->backend : cpu_init();
    if (!backend) Rf_error("failed to initialize the GGML CPU backend");

    /* CPU tensors borrow the mapped GGUF bytes directly. CUDA tensors already
     * occupy the model-owned context created on the first device call. */
    if (XLENGTH(tensors) > INT_MAX - 2 * n_layer - 16) {
        if (!use_device) bfree(backend);
        Rf_error("model has too many tensors");
    }
    const int n_weight_slots = (int)XLENGTH(tensors) + 2 * n_layer + 16;
    const int max_uploads = n_weight_slots + 8;
    int n_uploads = 0;
    int n_downloads = 0;
    struct rllm_upload *uploads = use_device
        ? (struct rllm_upload *)R_alloc((size_t)max_uploads, sizeof(*uploads))
        : NULL;
    struct rllm_download *downloads = use_device && use_cache
        ? (struct rllm_download *)R_alloc((size_t)(4 * n_layer), sizeof(*downloads))
        : NULL;
    struct ggml_context *wctx = use_device ? cuda_ctx->wctx :
        ctx_create((size_t)(n_weight_slots + 8) * t_over() + 4096,
                   /*no_alloc=*/1);
    if (!wctx) {
        if (!use_device) bfree(backend);
        Rf_error("weights context creation failed");
    }

    /* -- compute context: structured size estimate, 2x slack ---------------- */
    const char *output_op = rllm_string(output_spec, "op");
    int output_rows;
    int output_cols;
    if (!strcmp(output_op, "projection")) {
        output_rows = n_vocab;
        output_cols = S;
    } else if (!strcmp(output_op, "embedding")) {
        const char *pooling = rllm_string(output_spec, "pooling");
        output_rows = rllm_integer(output_spec, "dimension");
        if (!strcmp(pooling, "mean")) output_cols = 1;
        else if (!strcmp(pooling, "none")) output_cols = S;
        else {
            if (!use_device) {
                ctx_free(wctx);
                bfree(backend);
            }
            Rf_error("unknown embedding pooling operator '%s'", pooling);
        }
        if (output_rows < 1) {
            if (!use_device) {
                ctx_free(wctx);
                bfree(backend);
            }
            Rf_error("embedding output dimension must be positive");
        }
    } else {
        if (!use_device) {
            ctx_free(wctx);
            bfree(backend);
        }
        Rf_error("unknown output operator '%s'", output_op);
    }

    const size_t graph_sz = (size_t) n_layer * 128 + 96;
    double est =
        (double) n_layer * (4.0 * S * (28.0 * n_embd + 16.0 * max_ff)
                            + 4.0 * 3.5 * (double) n_head * S * (double) n_kv)
        + 4.0 * S * 6.0 * n_embd
        + 4.0 * 2.5 * recurrent_elems
        + 4.0 * 2.5 * (double) output_rows * output_cols;
    size_t cmem = (size_t)(2.0 * est)
        + ((size_t) n_layer * 128 + 96) * t_over()
        + g_over(graph_sz) + (1u << 20);
    struct ggml_context *cctx = ctx_create(cmem, /*no_alloc=*/use_device ? 1 : 0);
    if (!cctx) {
        if (!use_device) {
            ctx_free(wctx);
            bfree(backend);
        }
        Rf_error("compute context creation failed (%.1f MB)", cmem / 1048576.0);
    }
    ggml_backend_buffer_t cbuf = NULL;

    /* The CUDA model context outlives this call; only transient allocations
     * are released on an error. CPU still owns its backend and weight context. */
#define RLLM_FAIL(...) do { if (cbuf) buf_free(cbuf); ctx_free(cctx); \
                            if (!use_device) { ctx_free(wctx); bfree(backend); } \
                            Rf_error(__VA_ARGS__); } while (0)
#define RLLM_CHECK(t)  do { if (!(t)) RLLM_FAIL("graph construction failed (%s)", #t); } while (0)

    /* -- inputs: token ids and positions ------------------------------------ */
    int64_t neS[1] = { S };
    struct ggml_tensor *tok = new_tensor(cctx, GGML_TYPE_I32, 1, neS, NULL);
    struct ggml_tensor *pos = new_tensor(cctx, GGML_TYPE_I32, 1, neS, NULL);
    RLLM_CHECK(tok && pos);
    {
        int32_t *tp = use_device
            ? (int32_t *)R_alloc((size_t)S, sizeof(*tp))
            : (int32_t *)tdata(tok);
        int32_t *pp = use_device
            ? (int32_t *)R_alloc((size_t)S, sizeof(*pp))
            : (int32_t *)tdata(pos);
        const int *ids = INTEGER(tokens_sexp);
        for (int i = 0; i < S; i++) {
            if (ids[i] < 0 || ids[i] >= n_vocab) {
                RLLM_FAIL("token id %d out of range [0, %d)", ids[i], n_vocab);
            }
            tp[i] = ids[i];
            pp[i] = n_past + i;
        }
        rllm_add_upload(uploads, &n_uploads, max_uploads, tok, tp,
                        (size_t)S * sizeof(*tp));
        rllm_add_upload(uploads, &n_uploads, max_uploads, pos, pp,
                        (size_t)S * sizeof(*pp));
    }
    struct ggml_tensor *mpos = NULL;
    if (has_mrope) {
        int64_t ne_mpos[1] = { 4LL * S };
        mpos = new_tensor(cctx, GGML_TYPE_I32, 1, ne_mpos, NULL);
        RLLM_CHECK(mpos);
        int32_t *pp = use_device
            ? (int32_t *)R_alloc((size_t)4 * S, sizeof(*pp))
            : (int32_t *)tdata(mpos);
        for (int axis = 0; axis < 3; axis++) {
            for (int i = 0; i < S; i++) pp[axis * S + i] = n_past + i;
        }
        for (int i = 0; i < S; i++) pp[3 * S + i] = 0;
        rllm_add_upload(uploads, &n_uploads, max_uploads, mpos, pp,
                        (size_t)4 * S * sizeof(*pp));
    }

    /* -- weights ------------------------------------------------------------ */
    struct ggml_tensor *tok_embd = rllm_named_weight(
        wctx, tensors, rllm_string(input_spec, "weight"), new_tensor,
        storage_data, NULL, &n_uploads, max_uploads, cuda_ctx);
    struct ggml_tensor *out_norm = rllm_named_weight(
        wctx, tensors, rllm_string(output_spec, "norm"), new_tensor,
        storage_data, NULL, &n_uploads, max_uploads, cuda_ctx);
    struct ggml_tensor *output_w = NULL;
    if (!strcmp(output_op, "projection")) {
        output_w = rllm_named_weight(
            wctx, tensors, rllm_string(output_spec, "weight"), new_tensor,
            storage_data, NULL, &n_uploads, max_uploads, cuda_ctx);
    }

    /* -- the graph ----------------------------------------------------------- */
    /* Created before the layer loop: with a cache, the per-layer cpy nodes
     * that append K/V must be expanded into the graph ahead of the attention
     * nodes that read the cache views (ggml executes in insertion order). */
    struct ggml_cgraph *gf = new_graph(cctx, graph_sz);
    RLLM_CHECK(gf);
    const size_t fsz = sizeof(float);

    struct ggml_tensor *inpL = get_rows(cctx, tok_embd, tok);   /* [n_embd, S] */
    RLLM_CHECK(inpL);
    const double input_scale = rllm_number(input_spec, "scale");
    if (input_scale != 1.0) inpL = scale(cctx, inpL, input_scale);
    RLLM_CHECK(inpL);

#define RLLM_WEIGHT(spec, field) rllm_named_weight( \
    wctx, tensors, rllm_string((spec), (field)), new_tensor, storage_data, \
    NULL, &n_uploads, max_uploads, cuda_ctx)

    for (int il = 0; il < n_layer; il++) {
        SEXP layer = VECTOR_ELT(layers, il);
        SEXP op = rllm_list_elt(layer, "operator");
        SEXP ffn = rllm_list_elt(layer, "feed_forward");
        const char *op_name = rllm_string(op, "op");
        const char *ffn_name = rllm_string(ffn, "op");

        struct ggml_tensor *operator_norm = rllm_named_weight(
            wctx, tensors, rllm_string(layer, "operator_norm"), new_tensor,
            storage_data, NULL, &n_uploads, max_uploads, cuda_ctx);
        struct ggml_tensor *cur = mul(
            cctx, rms_norm(cctx, inpL, rms_eps), operator_norm);
        RLLM_CHECK(cur);

        struct ggml_tensor *operator_out = NULL;
        if (!strcmp(op_name, "attention") ||
            !strcmp(op_name, "gated_attention")) {
            const int gated_attention = !strcmp(op_name, "gated_attention");
            const int layer_n_head = rllm_integer(op, "n_head");
            const int layer_n_head_kv = rllm_integer(op, "n_head_kv");
            const int head_dim = rllm_integer(op, "head_dim");
            const int attention_width = head_dim * layer_n_head;
            const int n_embd_gqa = head_dim * layer_n_head_kv;
            SEXP rope_spec = rllm_list_elt(op, "rope");
            const int rope_dims = rllm_integer(rope_spec, "dims");
            const int rope_mode = gated_attention
                ? GGML_ROPE_TYPE_MROPE : rllm_integer(rope_spec, "mode");
            const double rope_base = rllm_number(rope_spec, "base");
            SEXP scale_spec = rllm_list_elt(op, "scale");
            const char *scale_at = rllm_string(scale_spec, "at");
            const double attention_scale = rllm_number(scale_spec, "value");
            SEXP mask_spec = rllm_list_elt(op, "mask");
            const char *mask_type = rllm_string(mask_spec, "type");
            if (layer_n_head < 1 || layer_n_head_kv < 1 || head_dim < 1 ||
                (!gated_attention && attention_width != n_embd) ||
                layer_n_head % layer_n_head_kv != 0 || rope_dims > head_dim) {
                RLLM_FAIL("layer %d has inconsistent attention dimensions", il);
            }
            if (use_cache && strcmp(mask_type, "causal")) {
                RLLM_FAIL("layer %d uses %s attention, which cannot use incremental state",
                          il, mask_type);
            }

            struct ggml_tensor *wk = RLLM_WEIGHT(op, "key");
            struct ggml_tensor *wv = RLLM_WEIGHT(op, "value");
            struct ggml_tensor *wo = RLLM_WEIGHT(op, "output");
            struct ggml_tensor *Q = NULL;
            struct ggml_tensor *attention_gate = NULL;
            if (gated_attention) {
                const char *layout = rllm_string(op, "query_gate_layout");
                if (strcmp(layout, "head_interleaved")) {
                    RLLM_FAIL("layer %d has unknown query-gate layout '%s'",
                              il, layout);
                }
                struct ggml_tensor *wqg = RLLM_WEIGHT(op, "query_gate");
                struct ggml_tensor *qg = mul_mat(cctx, wqg, cur);
                RLLM_CHECK(qg);
                Q = view_3d(cctx, qg, head_dim, layer_n_head, S,
                            (size_t)2 * head_dim * fsz,
                            (size_t)2 * attention_width * fsz, 0);
                attention_gate = view_3d(
                    cctx, qg, head_dim, layer_n_head, S,
                    (size_t)2 * head_dim * fsz,
                    (size_t)2 * attention_width * fsz,
                    (size_t)head_dim * fsz);
                attention_gate = reshape_2d(
                    cctx, cont(cctx, attention_gate), attention_width, S);
            } else {
                struct ggml_tensor *wq = RLLM_WEIGHT(op, "query");
                Q = reshape_3d(
                    cctx, mul_mat(cctx, wq, cur), head_dim, layer_n_head, S);
            }
            struct ggml_tensor *K = mul_mat(cctx, wk, cur);
            struct ggml_tensor *V = mul_mat(cctx, wv, cur);
            RLLM_CHECK(Q && K && V && (!gated_attention || attention_gate));

            K = reshape_3d(cctx, K, head_dim, layer_n_head_kv, S);
            const char *qnorm_name = rllm_optional_string(op, "query_norm");
            const char *knorm_name = rllm_optional_string(op, "key_norm");
            if (qnorm_name) {
                struct ggml_tensor *qnorm = rllm_named_weight(
                    wctx, tensors, qnorm_name, new_tensor, storage_data,
                    NULL, &n_uploads, max_uploads, cuda_ctx);
                Q = mul(cctx, rms_norm(cctx, Q, rms_eps), qnorm);
            }
            if (knorm_name) {
                struct ggml_tensor *knorm = rllm_named_weight(
                    wctx, tensors, knorm_name, new_tensor, storage_data,
                    NULL, &n_uploads, max_uploads, cuda_ctx);
                K = mul(cctx, rms_norm(cctx, K, rms_eps), knorm);
            }
            if (gated_attention) {
                SEXP section_spec = rllm_list_elt(rope_spec, "sections");
                if (TYPEOF(section_spec) != INTSXP ||
                    XLENGTH(section_spec) != GGML_MROPE_SECTIONS) {
                    RLLM_FAIL("layer %d MRoPE sections must contain four integers",
                              il);
                }
                int sections[GGML_MROPE_SECTIONS];
                for (int i = 0; i < GGML_MROPE_SECTIONS; i++) {
                    sections[i] = INTEGER(section_spec)[i];
                    if (sections[i] < 0) {
                        RLLM_FAIL("layer %d MRoPE sections must be non-negative",
                                  il);
                    }
                }
                Q = rope_multi(cctx, Q, mpos, rope_dims, sections,
                               rope_mode, rope_base);
                K = rope_multi(cctx, K, mpos, rope_dims, sections,
                               rope_mode, rope_base);
            } else {
                Q = rope(cctx, Q, pos, rope_dims, rope_mode, rope_base);
                K = rope(cctx, K, pos, rope_dims, rope_mode, rope_base);
            }
            if (!strcmp(scale_at, "query")) {
                Q = scale(cctx, Q, attention_scale);
            } else if (strcmp(scale_at, "logits")) {
                RLLM_FAIL("layer %d has unknown attention scale position '%s'",
                          il, scale_at);
            }
            RLLM_CHECK(Q && K);

            struct ggml_tensor *Qp = permute(cctx, Q, 0, 2, 1, 3);
            struct ggml_tensor *Kall = NULL;
            struct ggml_tensor *Vall = NULL;
            RLLM_CHECK(Qp);
            if (use_cache) {
                int64_t ne_slab[1] = { (int64_t)n_ctx * n_embd_gqa };
                void *kraw = RAW(VECTOR_ELT(kcache, il));
                void *vraw = RAW(VECTOR_ELT(vcache, il));
                struct ggml_context *cache_ctx = use_device ? cctx : wctx;
                struct ggml_tensor *kc = new_tensor(
                    cache_ctx, GGML_TYPE_F32, 1, ne_slab,
                    use_device ? NULL : kraw);
                struct ggml_tensor *vc = new_tensor(
                    cache_ctx, GGML_TYPE_F32, 1, ne_slab,
                    use_device ? NULL : vraw);
                RLLM_CHECK(kc && vc);
                if (use_device) {
                    const size_t prefix_elems =
                        (size_t)n_past * (size_t)n_embd_gqa;
                    if (prefix_elems > 0) {
                        float *vprefix = (float *)R_alloc(prefix_elems, fsz);
                        const float *vhost = (const float *)vraw;
                        for (int64_t p = 0; p < n_past; p++) {
                            for (int64_t d = 0; d < n_embd_gqa; d++) {
                                vprefix[p * n_embd_gqa + d] =
                                    vhost[p + d * n_ctx];
                            }
                        }
                        rllm_add_upload(uploads, &n_uploads, max_uploads, kc,
                                        kraw, prefix_elems * fsz);
                        rllm_add_upload(uploads, &n_uploads, max_uploads, vc,
                                        vprefix, prefix_elems * fsz);
                    }
                }

                struct ggml_tensor *k_dst = view_1d(
                    cctx, kc, (int64_t)S * n_embd_gqa,
                    (size_t)n_past * n_embd_gqa * fsz);
                struct ggml_tensor *v_dst = use_device
                    ? view_1d(cctx, vc, (int64_t)S * n_embd_gqa,
                              (size_t)n_past * n_embd_gqa * fsz)
                    : view_2d(cctx, vc, S, n_embd_gqa,
                              (size_t)n_ctx * fsz, (size_t)n_past * fsz);
                RLLM_CHECK(k_dst && v_dst);
                expand(gf, cpy(cctx, K, k_dst));
                expand(gf, cpy(cctx,
                    use_device ? V : transpose(
                        cctx, reshape_2d(cctx, V, n_embd_gqa, S)), v_dst));

                if (use_device) {
                    const size_t block_elems =
                        (size_t)S * (size_t)n_embd_gqa;
                    const size_t block_offset =
                        (size_t)n_past * (size_t)n_embd_gqa * fsz;
                    float *vblock = (float *)R_alloc(block_elems, fsz);
                    downloads[n_downloads++] = (struct rllm_download){
                        kc, (float *)kraw + (size_t)n_past * n_embd_gqa,
                        block_elems * fsz, block_offset, NULL, 0, 0, 0, 0
                    };
                    downloads[n_downloads++] = (struct rllm_download){
                        vc, vblock, block_elems * fsz, block_offset,
                        (float *)vraw, n_ctx, n_past, S, n_embd_gqa
                    };
                }

                Kall = view_3d(cctx, kc, head_dim, n_kv, layer_n_head_kv,
                               (size_t)n_embd_gqa * fsz,
                               (size_t)head_dim * fsz, 0);
                if (use_device) {
                    struct ggml_tensor *vseq = view_3d(
                        cctx, vc, head_dim, layer_n_head_kv, n_kv,
                        (size_t)head_dim * fsz,
                        (size_t)n_embd_gqa * fsz, 0);
                    Vall = cont(cctx, permute(cctx, vseq, 1, 2, 0, 3));
                } else {
                    Vall = view_3d(cctx, vc, n_kv, head_dim, layer_n_head_kv,
                                   (size_t)n_ctx * fsz,
                                   (size_t)n_ctx * head_dim * fsz, 0);
                }
                RLLM_CHECK(Kall && Vall);
            } else {
                Kall = permute(cctx, K, 0, 2, 1, 3);
                Vall = cont(cctx, permute(
                    cctx, reshape_3d(cctx, V, head_dim, layer_n_head_kv, S),
                    1, 2, 0, 3));
                RLLM_CHECK(Kall && Vall);
            }

            struct ggml_tensor *KQ = mul_mat(cctx, Kall, Qp);
            int window = 0;
            if (!strcmp(mask_type, "symmetric_window")) {
                window = rllm_integer(mask_spec, "window");
                if (window < 2) {
                    RLLM_FAIL("layer %d has invalid symmetric attention window", il);
                }
            } else if (strcmp(mask_type, "causal") &&
                       strcmp(mask_type, "bidirectional")) {
                RLLM_FAIL("layer %d has unknown attention mask '%s'", il,
                          mask_type);
            }
            int64_t ne_mask[2] = { n_kv, S };
            struct ggml_tensor *mask = new_tensor(
                cctx, GGML_TYPE_F32, 2, ne_mask, NULL);
            RLLM_CHECK(mask);
            float *mp = use_device
                ? (float *)R_alloc((size_t)n_kv * S, sizeof(*mp))
                : (float *)tdata(mask);
            const int half_window = window / 2;
            for (int q = 0; q < S; q++) {
                const int64_t qpos = (int64_t)n_past + q;
                for (int64_t k = 0; k < n_kv; k++) {
                    int masked = 0;
                    if (!strcmp(mask_type, "causal")) {
                        masked = k > qpos;
                    } else if (!strcmp(mask_type, "symmetric_window")) {
                        const int64_t d = qpos - k;
                        masked = d < -half_window || d > half_window;
                    }
                    mp[k + (int64_t)q * n_kv] = masked ? -INFINITY : 0.0f;
                }
            }
            rllm_add_upload(uploads, &n_uploads, max_uploads, mask, mp,
                            (size_t)n_kv * S * sizeof(*mp));
            KQ = soft_max_ext(cctx, KQ, mask,
                !strcmp(scale_at, "logits") ? attention_scale : 1.0, 0.0);
            RLLM_CHECK(KQ);
            struct ggml_tensor *KQV = mul_mat(cctx, Vall, KQ);
            operator_out = reshape_2d(
                cctx, cont(cctx, permute(cctx, KQV, 0, 2, 1, 3)),
                attention_width, S);
            if (gated_attention) {
                const char *activation = rllm_string(op, "gate_activation");
                if (strcmp(activation, "sigmoid")) {
                    RLLM_FAIL("layer %d has unknown attention gate '%s'",
                              il, activation);
                }
                operator_out = mul(
                    cctx, operator_out, sigmoid(cctx, attention_gate));
            }
            operator_out = mul_mat(cctx, wo, operator_out);
            RLLM_CHECK(operator_out);
        } else if (!strcmp(op_name, "gated_delta_net")) {
            const int key_heads = rllm_integer(op, "key_heads");
            const int value_heads = rllm_integer(op, "value_heads");
            const int key_dim = rllm_integer(op, "key_head_dim");
            const int value_dim = rllm_integer(op, "value_head_dim");
            const int conv_width = rllm_integer(op, "convolution_width");
            const int conv_kernel = rllm_integer(op, "convolution_kernel");
            const int key_width = key_dim * key_heads;
            const int inner_dim = value_dim * value_heads;
            if (key_dim != value_dim || value_heads % key_heads != 0 ||
                conv_width != 2 * key_width + inner_dim || conv_kernel < 2) {
                RLLM_FAIL("layer %d has inconsistent gated-delta dimensions", il);
            }

            SEXP activations = rllm_list_elt(op, "activations");
            SEXP qk_norm = rllm_list_elt(op, "qk_norm");
            if (strcmp(rllm_string(activations, "convolution"), "silu") ||
                strcmp(rllm_string(activations, "beta"), "sigmoid") ||
                strcmp(rllm_string(activations, "time_step"), "softplus") ||
                strcmp(rllm_string(activations, "output_gate"), "silu") ||
                strcmp(rllm_string(qk_norm, "kind"), "l2")) {
                RLLM_FAIL("layer %d uses unsupported gated-delta semantics", il);
            }
            const double qk_eps = rllm_number(qk_norm, "eps");

            struct ggml_tensor *wqkv = RLLM_WEIGHT(op, "qkv");
            struct ggml_tensor *wz = RLLM_WEIGHT(op, "output_gate");
            struct ggml_tensor *conv_kernel_w = RLLM_WEIGHT(op, "convolution");
            struct ggml_tensor *dt = RLLM_WEIGHT(op, "time_bias");
            struct ggml_tensor *decay = RLLM_WEIGHT(op, "decay");
            struct ggml_tensor *wbeta = RLLM_WEIGHT(op, "beta");
            struct ggml_tensor *walpha = RLLM_WEIGHT(op, "alpha");
            struct ggml_tensor *state_norm = RLLM_WEIGHT(op, "norm");
            struct ggml_tensor *wout = RLLM_WEIGHT(op, "output");

            struct ggml_tensor *qkv = mul_mat(cctx, wqkv, cur);
            struct ggml_tensor *z = mul_mat(cctx, wz, cur);
            struct ggml_tensor *beta = sigmoid(
                cctx, mul_mat(cctx, wbeta, cur));
            beta = reshape_4d(cctx, beta, 1, value_heads, S, 1);
            struct ggml_tensor *alpha = add(
                cctx, mul_mat(cctx, walpha, cur), dt);
            struct ggml_tensor *gate = mul(
                cctx, softplus(cctx, alpha), decay);
            gate = reshape_4d(cctx, gate, 1, value_heads, S, 1);
            RLLM_CHECK(qkv && z && beta && gate);

            SEXP state_spec = rllm_list_elt(layer, "state");
            SEXP matrix_spec = rllm_list_elt(state_spec, "matrix");
            SEXP conv_spec = rllm_list_elt(state_spec, "convolution");
            const int state_rows = rllm_named_integer(matrix_spec, "row");
            const int state_columns =
                rllm_named_integer(matrix_spec, "column");
            const int state_heads = rllm_named_integer(matrix_spec, "head");
            const int state_conv_width =
                rllm_named_integer(conv_spec, "width");
            const int history = rllm_named_integer(conv_spec, "history");
            if (state_rows != value_dim || state_columns != value_dim ||
                state_heads != value_heads || state_conv_width != conv_width ||
                history != conv_kernel - 1) {
                RLLM_FAIL("layer %d gated-delta state disagrees with its operator",
                          il);
            }

            const size_t conv_bytes =
                (size_t)history * conv_width * fsz;
            const size_t recurrent_bytes =
                (size_t)value_dim * value_dim * value_heads * fsz;
            void *conv_raw = use_cache
                ? (void *)RAW(VECTOR_ELT(convcache, il)) : NULL;
            void *recurrent_raw = use_cache
                ? (void *)RAW(VECTOR_ELT(recurrent_cache, il)) : NULL;
            struct ggml_context *state_ctx = use_device ? cctx :
                (use_cache ? wctx : cctx);
            int64_t ne_conv_state[3] = { history, conv_width, 1 };
            int64_t ne_recurrent_state[4] = {
                value_dim, value_dim, value_heads, 1
            };
            struct ggml_tensor *conv_state = new_tensor(
                state_ctx, GGML_TYPE_F32, 3, ne_conv_state,
                use_device ? NULL : conv_raw);
            struct ggml_tensor *recurrent_state = new_tensor(
                state_ctx, GGML_TYPE_F32, 4, ne_recurrent_state,
                use_device ? NULL : recurrent_raw);
            RLLM_CHECK(conv_state && recurrent_state);
            if (use_device) {
                if (!conv_raw) {
                    conv_raw = R_alloc(conv_bytes, 1);
                    memset(conv_raw, 0, conv_bytes);
                    recurrent_raw = R_alloc(recurrent_bytes, 1);
                    memset(recurrent_raw, 0, recurrent_bytes);
                }
                rllm_add_upload(uploads, &n_uploads, max_uploads,
                                conv_state, conv_raw, conv_bytes);
                rllm_add_upload(uploads, &n_uploads, max_uploads,
                                recurrent_state, recurrent_raw,
                                recurrent_bytes);
            } else if (!use_cache) {
                memset(tdata(conv_state), 0, conv_bytes);
                memset(tdata(recurrent_state), 0, recurrent_bytes);
            }

            struct ggml_tensor *qkv_seq = reshape_3d(
                cctx, qkv, conv_width, S, 1);
            struct ggml_tensor *conv_input = concat(
                cctx, conv_state, transpose(cctx, qkv_seq), 0);
            RLLM_CHECK(qkv_seq && conv_input);
            if (use_cache) {
                struct ggml_tensor *next_conv = view_3d(
                    cctx, conv_input, history, conv_width, 1,
                    conv_input->nb[1], conv_input->nb[2],
                    (size_t)(conv_input->ne[0] - history) * fsz);
                RLLM_CHECK(next_conv);
                expand(gf, cpy(cctx, next_conv, conv_state));
            }
            struct ggml_tensor *mixed = silu(
                cctx, ssm_conv(cctx, conv_input, conv_kernel_w));
            RLLM_CHECK(mixed);

            struct ggml_tensor *q = view_3d(
                cctx, mixed, key_dim, key_heads, S,
                (size_t)key_dim * fsz, (size_t)conv_width * fsz, 0);
            struct ggml_tensor *k = view_3d(
                cctx, mixed, key_dim, key_heads, S,
                (size_t)key_dim * fsz, (size_t)conv_width * fsz,
                (size_t)key_width * fsz);
            struct ggml_tensor *v = view_3d(
                cctx, mixed, value_dim, value_heads, S,
                (size_t)value_dim * fsz, (size_t)conv_width * fsz,
                (size_t)2 * key_width * fsz);
            q = l2_norm(cctx, q, qk_eps);
            k = l2_norm(cctx, k, qk_eps);
            RLLM_CHECK(q && k && v);

            struct ggml_tensor *delta = gated_delta(
                cctx, q, k, v, gate, beta, recurrent_state, 1);
            RLLM_CHECK(delta);
            struct ggml_tensor *delta_out = view_4d(
                cctx, delta, value_dim, value_heads, S, 1,
                (size_t)value_dim * fsz,
                (size_t)value_dim * value_heads * fsz,
                (size_t)value_dim * value_heads * S * fsz, 0);
            struct ggml_tensor *next_recurrent = view_4d(
                cctx, delta, value_dim, value_dim, value_heads, 1,
                (size_t)value_dim * fsz,
                (size_t)value_dim * value_dim * fsz,
                (size_t)value_dim * value_dim * value_heads * fsz,
                (size_t)value_dim * value_heads * S * fsz);
            RLLM_CHECK(delta_out && next_recurrent);
            if (use_cache) {
                expand(gf, cpy(cctx, next_recurrent, recurrent_state));
                if (use_device) {
                    downloads[n_downloads++] = (struct rllm_download){
                        conv_state, conv_raw, conv_bytes, 0,
                        NULL, 0, 0, 0, 0
                    };
                    downloads[n_downloads++] = (struct rllm_download){
                        recurrent_state, recurrent_raw, recurrent_bytes, 0,
                        NULL, 0, 0, 0, 0
                    };
                }
            }

            z = reshape_3d(cctx, z, value_dim, value_heads, S);
            operator_out = mul(
                cctx,
                mul(cctx, rms_norm(cctx, delta_out, rms_eps), state_norm),
                silu(cctx, z));
            operator_out = reshape_2d(cctx, operator_out, inner_dim, S);
            operator_out = mul_mat(cctx, wout, operator_out);
            RLLM_CHECK(operator_out);
        } else {
            const int width = rllm_integer(op, "width");
            const int l_cache = rllm_integer(op, "l_cache");
            const int history = l_cache - 1;
            if (width != n_embd || history < 1) {
                RLLM_FAIL("layer %d has inconsistent short-convolution dimensions", il);
            }
            struct ggml_tensor *w_in = RLLM_WEIGHT(op, "input");
            struct ggml_tensor *kernel = RLLM_WEIGHT(op, "kernel");
            struct ggml_tensor *w_out = RLLM_WEIGHT(op, "output");
            struct ggml_tensor *bcx = mul_mat(cctx, w_in, cur);
            RLLM_CHECK(bcx);
            struct ggml_tensor *b = view_3d(
                cctx, bcx, n_embd, S, 1, bcx->nb[1], bcx->nb[2], 0);
            struct ggml_tensor *c = view_3d(
                cctx, bcx, n_embd, S, 1, bcx->nb[1], bcx->nb[2],
                (size_t)n_embd * fsz);
            struct ggml_tensor *x = view_3d(
                cctx, bcx, n_embd, S, 1, bcx->nb[1], bcx->nb[2],
                (size_t)2 * n_embd * fsz);
            struct ggml_tensor *bx = transpose(cctx, mul(cctx, b, x));
            RLLM_CHECK(b && c && x && bx);

            int64_t ne_state[2] = { history, n_embd };
            struct ggml_tensor *state;
            if (use_cache) {
                state = new_tensor(wctx, GGML_TYPE_F32, 2, ne_state,
                                   RAW(VECTOR_ELT(convcache, il)));
            } else {
                state = new_tensor(cctx, GGML_TYPE_F32, 2, ne_state, NULL);
                RLLM_CHECK(state);
                memset(tdata(state), 0,
                       (size_t)history * n_embd * sizeof(float));
            }
            RLLM_CHECK(state);
            struct ggml_tensor *state3 = reshape_3d(
                cctx, state, history, n_embd, 1);
            bx = concat(cctx, state3, bx, 0);
            RLLM_CHECK(bx);
            if (use_cache) {
                struct ggml_tensor *next_state = view_3d(
                    cctx, bx, history, n_embd, 1, bx->nb[1], bx->nb[2],
                    (size_t)(bx->ne[0] - history) * fsz);
                RLLM_CHECK(next_state);
                expand(gf, cpy(cctx, next_state, state));
            }
            struct ggml_tensor *conv_out = ssm_conv(cctx, bx, kernel);
            operator_out = mul_mat(cctx, w_out, mul(cctx, c, conv_out));
            operator_out = reshape_2d(cctx, operator_out, n_embd, S);
            RLLM_CHECK(operator_out);
        }

        const char *operator_post_name =
            rllm_optional_string(layer, "operator_post_norm");
        if (operator_post_name) {
            struct ggml_tensor *operator_post = rllm_named_weight(
                wctx, tensors, operator_post_name, new_tensor, storage_data,
                NULL, &n_uploads, max_uploads, cuda_ctx);
            operator_out = mul(
                cctx, rms_norm(cctx, operator_out, rms_eps), operator_post);
            RLLM_CHECK(operator_out);
        }
        inpL = add(cctx, inpL, operator_out);
        RLLM_CHECK(inpL);

        struct ggml_tensor *ffn_norm = rllm_named_weight(
            wctx, tensors, rllm_string(layer, "ffn_norm"), new_tensor,
            storage_data, NULL, &n_uploads, max_uploads, cuda_ctx);
        cur = mul(cctx, rms_norm(cctx, inpL, rms_eps), ffn_norm);
        RLLM_CHECK(cur);
        struct ggml_tensor *ffn_out = NULL;
        if (!strcmp(ffn_name, "swiglu") || !strcmp(ffn_name, "geglu")) {
            struct ggml_tensor *w_gate = RLLM_WEIGHT(ffn, "gate");
            struct ggml_tensor *w_up = RLLM_WEIGHT(ffn, "up");
            struct ggml_tensor *w_down = RLLM_WEIGHT(ffn, "down");
            struct ggml_tensor *gate = mul_mat(cctx, w_gate, cur);
            struct ggml_tensor *up = mul_mat(cctx, w_up, cur);
            RLLM_CHECK(gate && up);
            struct ggml_tensor *act = !strcmp(ffn_name, "swiglu")
                ? mul(cctx, silu(cctx, gate), up)
                : geglu(cctx, gate, up);
            RLLM_CHECK(act);
            ffn_out = mul_mat(cctx, w_down, act);
        } else {
            const int n_expert = rllm_integer(ffn, "experts");
            const int n_selected = rllm_integer(ffn, "selected");
            const char *routing = rllm_string(ffn, "routing");
            const int normalize_selected =
                rllm_boolean(ffn, "normalize_selected");
            const double expert_scale = rllm_number(ffn, "scale");
            if (n_expert < 1 || n_selected < 1 || n_selected > n_expert) {
                RLLM_FAIL("layer %d has invalid expert counts", il);
            }
            if (strcmp(routing, "sigmoid")) {
                RLLM_FAIL("layer %d has unknown expert routing '%s'", il,
                          routing);
            }
            struct ggml_tensor *router = RLLM_WEIGHT(ffn, "router");
            struct ggml_tensor *bias = RLLM_WEIGHT(ffn, "selection_bias");
            struct ggml_tensor *w_gate = RLLM_WEIGHT(ffn, "gate");
            struct ggml_tensor *w_up = RLLM_WEIGHT(ffn, "up");
            struct ggml_tensor *w_down = RLLM_WEIGHT(ffn, "down");

            struct ggml_tensor *probs = sigmoid(cctx, mul_mat(cctx, router, cur));
            struct ggml_tensor *selected = top_k(
                cctx, add(cctx, probs, bias), n_selected);
            struct ggml_tensor *probs3 = reshape_3d(
                cctx, probs, 1, n_expert, S);
            struct ggml_tensor *weights = get_rows(cctx, probs3, selected);
            weights = reshape_2d(cctx, weights, n_selected, S);
            if (normalize_selected) {
                struct ggml_tensor *denom = clamp(
                    cctx, sum_rows(cctx, weights), 6.103515625e-5, INFINITY);
                weights = divide(cctx, weights, denom);
            }
            weights = reshape_3d(cctx, weights, 1, n_selected, S);
            RLLM_CHECK(probs && selected && weights);
            expand(gf, weights);

            struct ggml_tensor *cur3 = reshape_3d(cctx, cur, n_embd, 1, S);
            struct ggml_tensor *gate = mul_mat_id(cctx, w_gate, cur3, selected);
            struct ggml_tensor *up = mul_mat_id(cctx, w_up, cur3, selected);
            struct ggml_tensor *act = mul(cctx, silu(cctx, gate), up);
            struct ggml_tensor *experts = mul_mat_id(
                cctx, w_down, act, selected);
            experts = mul(cctx, experts, weights);
            RLLM_CHECK(gate && up && act && experts);

            ffn_out = view_2d(cctx, experts, n_embd, S,
                              experts->nb[2], 0);
            RLLM_CHECK(ffn_out);
            for (int ie = 1; ie < n_selected; ++ie) {
                struct ggml_tensor *one = view_2d(
                    cctx, experts, n_embd, S, experts->nb[2],
                    (size_t)ie * experts->nb[1]);
                ffn_out = add(cctx, ffn_out, one);
                RLLM_CHECK(ffn_out);
            }
            if (expert_scale != 1.0) {
                ffn_out = scale(cctx, ffn_out, expert_scale);
                RLLM_CHECK(ffn_out);
            }
        }
        RLLM_CHECK(ffn_out);
        const char *ffn_post_name =
            rllm_optional_string(layer, "feed_forward_post_norm");
        if (ffn_post_name) {
            struct ggml_tensor *ffn_post = rllm_named_weight(
                wctx, tensors, ffn_post_name, new_tensor, storage_data,
                NULL, &n_uploads, max_uploads, cuda_ctx);
            ffn_out = mul(cctx, rms_norm(cctx, ffn_out, rms_eps), ffn_post);
            RLLM_CHECK(ffn_out);
        }
        inpL = add(cctx, inpL, ffn_out);
        RLLM_CHECK(inpL);
    }

    struct ggml_tensor *cur = mul(cctx, rms_norm(cctx, inpL, rms_eps), out_norm);
    struct ggml_tensor *result = NULL;
    if (!strcmp(output_op, "projection")) {
        result = mul_mat(cctx, output_w, cur);  /* [n_vocab, S] */
    } else {
        const char *pooling = rllm_string(output_spec, "pooling");
        if (!strcmp(pooling, "mean")) {
            int64_t ne_mean[2] = { S, 1 };
            struct ggml_tensor *mean = new_tensor(
                cctx, GGML_TYPE_F32, 2, ne_mean, NULL);
            RLLM_CHECK(mean);
            float *meanp = use_device
                ? (float *)R_alloc((size_t)S, sizeof(*meanp))
                : (float *)tdata(mean);
            for (int i = 0; i < S; i++) meanp[i] = 1.0f / (float)S;
            rllm_add_upload(uploads, &n_uploads, max_uploads, mean, meanp,
                            (size_t)S * sizeof(*meanp));
            result = mul_mat(cctx, cont(cctx, transpose(cctx, cur)), mean);
        } else {
            result = cur;
        }

        const char *projection_1 =
            rllm_optional_string(output_spec, "projection_1");
        const char *projection_2 =
            rllm_optional_string(output_spec, "projection_2");
        if ((projection_1 == NULL) != (projection_2 == NULL)) {
            RLLM_FAIL("embedding output must declare both projections or neither");
        }
        if (projection_1) {
            struct ggml_tensor *dense_2 = rllm_named_weight(
                wctx, tensors, projection_1, new_tensor, storage_data,
                NULL, &n_uploads, max_uploads, cuda_ctx);
            struct ggml_tensor *dense_3 = rllm_named_weight(
                wctx, tensors, projection_2, new_tensor, storage_data,
                NULL, &n_uploads, max_uploads, cuda_ctx);
            result = mul_mat(cctx, dense_2, result);
            result = mul_mat(cctx, dense_3, result);
        }
    }
    RLLM_CHECK(result);

    expand(gf, result);

    if (use_device) {
        /* The model weights are already resident. Allocate only this pass's
         * activations, inputs, cache mirror and result, then upload mutable
         * inputs through the backend-neutral transfer interface. */
        cbuf = alloc_tensors(cctx, backend);
        if (!cbuf) RLLM_FAIL("CUDA compute-buffer allocation failed");
        for (int i = 0; i < n_uploads; ++i) {
            tensor_set(uploads[i].tensor, uploads[i].data, 0, uploads[i].bytes);
        }
    }

    if (compute(backend, gf) != 0 /* GGML_STATUS_SUCCESS */) {
        RLLM_FAIL("graph compute failed");
    }

    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, output_rows, output_cols));
    {
        R_xlen_t n = (R_xlen_t) output_rows * output_cols;
        const float *lp;
        float *device_logits = NULL;
        if (use_device) {
            device_logits = (float *)R_alloc((size_t)n, sizeof(*device_logits));
            tensor_get(result, device_logits, 0,
                       (size_t)n * sizeof(*device_logits));
            lp = device_logits;
            for (int i = 0; i < n_downloads; ++i) {
                tensor_get(downloads[i].tensor, downloads[i].data,
                           downloads[i].tensor_offset,
                           downloads[i].bytes);
                if (downloads[i].scatter) {
                    const float *src = (const float *)downloads[i].data;
                    float *dst = downloads[i].scatter;
                    for (int64_t p = 0; p < downloads[i].rows; p++) {
                        for (int64_t d = 0; d < downloads[i].cols; d++) {
                            dst[downloads[i].scatter_offset + p +
                                d * downloads[i].stride] =
                                src[p * downloads[i].cols + d];
                        }
                    }
                }
            }
        } else {
            lp = (const float *)tdata(result);
        }
        double *op = REAL(out);
        for (R_xlen_t i = 0; i < n; i++) op[i] = (double) lp[i];
    }

    if (cbuf) buf_free(cbuf);
    ctx_free(cctx);
    if (!use_device) {
        ctx_free(wctx);
        bfree(backend);
    }

    UNPROTECT(1);
    return out;

#undef RLLM_CHECK
#undef RLLM_FAIL
#undef RLLM_WEIGHT
}

static struct ggml_tensor *rllm_f32_source(struct ggml_tensor **values,
                                           SEXP nodes, int before, SEXP node,
                                           int input)
{
    SEXP refs = rllm_list_elt(node, "inputs");
    if (TYPEOF(refs) != STRSXP || input < 0 || input >= XLENGTH(refs)) {
        Rf_error("native F32 node has an invalid input reference");
    }
    const char *id = CHAR(STRING_ELT(refs, input));
    for (int i = 0; i < before; ++i) {
        SEXP candidate = VECTOR_ELT(nodes, i);
        if (!strcmp(rllm_string(candidate, "id"), id)) {
            if (!values[i]) Rf_error("native F32 node refers to unavailable input");
            return values[i];
        }
    }
    Rf_error("native F32 node refers to a later or unknown input");
    return NULL;
}

static int rllm_f32_same_shape(const struct ggml_tensor *a,
                               const struct ggml_tensor *b)
{
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (a->ne[i] != b->ne[i]) return 0;
    }
    return 1;
}

static const char *rllm_f32_parameter(SEXP attributes, const char *field)
{
    SEXP parameter = rllm_list_elt(attributes, field);
    if (TYPEOF(parameter) != VECSXP) {
        Rf_error("native F32 operator field '%s' must be a parameter", field);
    }
    return rllm_string(parameter, "name");
}

static const float *rllm_f32_parameter_data(SEXP tensors, const char *name,
                                             int expected, size_t *bytes,
                                             Rfmalloc_storage_data_fun storage_data)
{
    SEXP binding = rllm_list_elt(tensors, name);
    SEXP type = rllm_list_elt(binding, "type");
    SEXP payload = rllm_list_elt(binding, "payload");
    const void *data = NULL;
    size_t extent = 0;
    if (TYPEOF(type) != STRSXP || XLENGTH(type) != 1 ||
        strcmp(CHAR(STRING_ELT(type, 0)), "f32") ||
        expected < 1 || (size_t) expected > SIZE_MAX / sizeof(float) ||
        storage_data(payload, &data, &extent, NULL) != 0 ||
        extent < (size_t) expected * sizeof(float)) {
        Rf_error("native F32 parameter '%s' has an invalid payload", name);
    }
    if (bytes) *bytes = extent;
    return (const float *) data;
}

/* Persistent fixed-shape F32 execution. The protected owner retains CPU
 * mapped payload spans, and additionally the CUDA weight context when used.
 * A CPU execution context owns its backend; a CUDA execution context borrows
 * the model-owned backend. */
struct rllm_f32_context {
    int backend_code;
    int own_backend;
    int channels;
    int sequence_length;
    int batch;
    int output_dim[3];
    R_xlen_t input_count;
    R_xlen_t output_count;
    ggml_backend_t backend;
    struct ggml_context *wctx;
    struct ggml_context *cctx;
    ggml_backend_buffer_t cbuf;
    struct ggml_cgraph *graph;
    struct ggml_tensor *input;
    struct ggml_tensor *output;
    Rggml_conv_1d_f32_plan **conv_plans;
    int n_conv_plans;
    Rggml_conv_1d_f32_plan_destroy_fun conv_plan_destroy;
    float *input_host;
    float *output_host;
    char *output_name;
    Rggml_context_free_fun context_free;
    Rggml_backend_buffer_free_fun buffer_free;
    Rggml_backend_free_fun backend_free;
};

struct rllm_f32_upload {
    struct ggml_tensor *tensor;
    const void *data;
    size_t bytes;
    void *owned;
};

static SEXP rllm_f32_context_tag(void)
{
    return Rf_install("Rllm_f32_execution_context");
}

static void rllm_f32_context_destroy(struct rllm_f32_context *ctx)
{
    if (!ctx) return;
    free(ctx->input_host);
    free(ctx->output_host);
    free(ctx->output_name);
    if (ctx->conv_plans && ctx->conv_plan_destroy) {
        for (int i = 0; i < ctx->n_conv_plans; ++i) {
            ctx->conv_plan_destroy(ctx->conv_plans[i]);
        }
    }
    free(ctx->conv_plans);
    if (ctx->cbuf && ctx->buffer_free) ctx->buffer_free(ctx->cbuf);
    if (ctx->cctx && ctx->context_free) ctx->context_free(ctx->cctx);
    if (ctx->wctx && ctx->context_free) ctx->context_free(ctx->wctx);
    if (ctx->own_backend && ctx->backend && ctx->backend_free) {
        ctx->backend_free(ctx->backend);
    }
    free(ctx);
}

static void rllm_f32_context_finalizer(SEXP ext)
{
    struct rllm_f32_context *ctx =
        (struct rllm_f32_context *) R_ExternalPtrAddr(ext);
    if (!ctx) return;
    rllm_f32_context_destroy(ctx);
    R_ClearExternalPtr(ext);
}

static int rllm_f32_mul_size(size_t a, size_t b, size_t *out)
{
    if (a && b > SIZE_MAX / a) return -1;
    *out = a * b;
    return 0;
}

static int rllm_f32_product3(int a, int b, int c, R_xlen_t *out,
                              size_t *bytes)
{
    size_t n;
    if (a < 1 || b < 1 || c < 1 ||
        rllm_f32_mul_size((size_t) a, (size_t) b, &n) ||
        rllm_f32_mul_size(n, (size_t) c, &n) ||
        n > (size_t) R_XLEN_T_MAX ||
        rllm_f32_mul_size(n, sizeof(float), bytes)) return -1;
    *out = (R_xlen_t) n;
    return 0;
}

static void rllm_f32_upload_add(struct rllm_f32_upload *uploads, int *n,
                                 int max, struct ggml_tensor *tensor,
                                 const void *data, size_t bytes, void *owned)
{
    if (*n >= max || !tensor || !data || !bytes) {
        Rf_error("native F32 fixed-data upload table overflow");
    }
    uploads[*n].tensor = tensor;
    uploads[*n].data = data;
    uploads[*n].bytes = bytes;
    uploads[*n].owned = owned;
    ++*n;
}

static void rllm_f32_upload_release(struct rllm_f32_upload *uploads, int n)
{
    if (!uploads) return;
    for (int i = 0; i < n; ++i) free(uploads[i].owned);
    free(uploads);
}

static void rllm_f32_validate_dataflow(SEXP nodes)
{
    int n_nodes = (int) XLENGTH(nodes);
    for (int i = 0; i < n_nodes; ++i) {
        SEXP node = VECTOR_ELT(nodes, i);
        const char *op = rllm_string(node, "op");
        if (i == 0) continue;
        if (strcmp(op, "conv1d") && strcmp(op, "batch_norm") &&
            strcmp(op, "leaky_relu") && strcmp(op, "add") &&
            strcmp(op, "crop1d") && strcmp(op, "softmax")) {
            Rf_error("native F32 execution does not implement operator '%s'", op);
        }
        SEXP refs = rllm_list_elt(node, "inputs");
        if (TYPEOF(refs) != STRSXP || XLENGTH(refs) < 1) {
            Rf_error("native F32 node has invalid input references");
        }
        for (R_xlen_t j = 0; j < XLENGTH(refs); ++j) {
            if (STRING_ELT(refs, j) == NA_STRING) {
                Rf_error("native F32 node has an invalid input reference");
            }
            const char *id = CHAR(STRING_ELT(refs, j));
            int found = 0;
            for (int k = 0; k < i; ++k) {
                if (!strcmp(rllm_string(VECTOR_ELT(nodes, k), "id"), id)) {
                    found = 1;
                    break;
                }
            }
            if (!found) Rf_error("native F32 node refers to a later or unknown input");
        }
    }
}

static struct rllm_f32_context *rllm_f32_context_get(
    SEXP ext, SEXP execution, SEXP backend_context, int backend_code,
    int channels, int sequence_length, int batch)
{
    if (TYPEOF(ext) != EXTPTRSXP ||
        R_ExternalPtrTag(ext) != rllm_f32_context_tag()) return NULL;
    SEXP owner = R_ExternalPtrProtected(ext);
    if (TYPEOF(owner) != VECSXP || XLENGTH(owner) != (backend_code == 3 ? 2 : 1) ||
        VECTOR_ELT(owner, 0) != execution ||
        (backend_code == 3 && VECTOR_ELT(owner, 1) != backend_context)) return NULL;
    struct rllm_f32_context *ctx =
        (struct rllm_f32_context *) R_ExternalPtrAddr(ext);
    if (!ctx || ctx->backend_code != backend_code || ctx->channels != channels ||
        ctx->sequence_length != sequence_length || ctx->batch != batch ||
        !ctx->backend || !ctx->cctx || !ctx->cbuf || !ctx->graph ||
        !ctx->input || !ctx->output || !ctx->input_host || !ctx->output_host) {
        return NULL;
    }
    return ctx;
}

static void rllm_f32_copy_input(float *dst, SEXP sequence, int channels,
                                 int sequence_length, int batch)
{
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int s = 0; s < sequence_length; ++s) {
                R_xlen_t source = c + (R_xlen_t) s * channels +
                    (R_xlen_t) b * channels * sequence_length;
                R_xlen_t target = s + (R_xlen_t) c * sequence_length +
                    (R_xlen_t) b * sequence_length * channels;
                dst[target] = TYPEOF(sequence) == REALSXP
                    ? (float) REAL(sequence)[source] : (float) INTEGER(sequence)[source];
            }
        }
    }
}

static SEXP rllm_f32_result(struct rllm_f32_context *ctx)
{
    SEXP dim = PROTECT(Rf_allocVector(INTSXP, 3));
    for (int i = 0; i < 3; ++i) INTEGER(dim)[i] = ctx->output_dim[i];
    SEXP output = PROTECT(Rf_allocArray(REALSXP, dim));
    for (R_xlen_t i = 0; i < ctx->output_count; ++i) {
        REAL(output)[i] = ctx->output_host[i];
    }
    SEXP named = PROTECT(Rf_allocVector(VECSXP, 1));
    SET_VECTOR_ELT(named, 0, output);
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(names, 0, Rf_mkChar(ctx->output_name));
    Rf_setAttrib(named, R_NamesSymbol, names);
    UNPROTECT(4);
    return named;
}

static SEXP rllm_f32_context_build(SEXP execution, int channels,
                                   int sequence_length, int batch,
                                   int backend_code, SEXP backend_context)
{
    SEXP program = rllm_list_elt(execution, "program");
    SEXP tensors = rllm_list_elt(execution, "bindings");
    SEXP nodes = rllm_list_elt(program, "nodes");
    SEXP outputs = rllm_list_elt(program, "outputs");
    const int use_device = backend_code == 3;
    if (TYPEOF(tensors) != VECSXP || TYPEOF(nodes) != VECSXP ||
        XLENGTH(nodes) < 2 || XLENGTH(nodes) > 256 ||
        TYPEOF(outputs) != VECSXP || XLENGTH(outputs) != 1) {
        Rf_error("invalid native F32 program bindings");
    }
    int n_nodes = (int) XLENGTH(nodes);
    SEXP first = VECTOR_ELT(nodes, 0);
    if (strcmp(rllm_string(first, "op"), "input") ||
        strcmp(rllm_string(rllm_list_elt(first, "attributes"), "name"), "sequence") ||
        strcmp(rllm_string(first, "dtype"), "f32")) {
        Rf_error("native F32 programs must begin with input 'sequence' of type f32");
    }
    rllm_f32_validate_dataflow(nodes);
    R_xlen_t input_count;
    size_t input_bytes;
    if (rllm_f32_product3(channels, sequence_length, batch, &input_count, &input_bytes)) {
        Rf_error("native F32 input extent overflows allocation bounds");
    }

    Rggml_context_create_fun ctx_create = Rggml_context_create_ptr();
    Rggml_context_free_fun ctx_free = Rggml_context_free_ptr();
    Rggml_new_tensor_fun new_tensor = Rggml_new_tensor_ptr();
    Rggml_tensor_overhead_fun t_over = Rggml_tensor_overhead_ptr();
    Rggml_graph_overhead_fun g_over = Rggml_graph_overhead_ptr();
    Rggml_new_graph_fun new_graph = Rggml_new_graph_ptr();
    Rggml_build_forward_expand_fun expand = Rggml_build_forward_expand_ptr();
    Rggml_backend_cpu_init_fun cpu_init = Rggml_backend_cpu_init_ptr();
    Rggml_backend_free_fun bfree = Rggml_backend_free_ptr();
    Rggml_backend_alloc_ctx_tensors_fun alloc_tensors =
        Rggml_backend_alloc_ctx_tensors_ptr();
    Rggml_backend_buffer_free_fun buf_free = Rggml_backend_buffer_free_ptr();
    Rggml_backend_tensor_set_fun tensor_set = Rggml_backend_tensor_set_ptr();
    Rggml_add_fun add = Rggml_add_ptr();
    Rggml_mul_fun mul = Rggml_mul_ptr();
    Rggml_conv_1d_fun conv1d = Rggml_conv_1d_ptr();
    Rggml_conv_1d_f32_plan_create_fun conv_plan_create =
        Rggml_conv_1d_f32_plan_create_ptr();
    Rggml_conv_1d_f32_plan_apply_fun conv_plan_apply =
        Rggml_conv_1d_f32_plan_apply_ptr();
    Rggml_conv_1d_f32_plan_destroy_fun conv_plan_destroy =
        Rggml_conv_1d_f32_plan_destroy_ptr();
    Rggml_leaky_relu_fun leaky_relu = Rggml_leaky_relu_ptr();
    Rggml_soft_max_fun soft_max = Rggml_soft_max_ptr();
    Rggml_permute_fun permute = Rggml_permute_ptr();
    Rggml_cont_fun cont = Rggml_cont_ptr();
    Rggml_view_3d_fun view_3d = Rggml_view_3d_ptr();
    Rfmalloc_storage_data_fun storage_data = Rfmalloc_storage_data_ptr();

    struct rllm_f32_context *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) Rf_error("native F32 execution context allocation failed");
    ctx->backend_code = backend_code;
    ctx->own_backend = !use_device;
    ctx->channels = channels;
    ctx->sequence_length = sequence_length;
    ctx->batch = batch;
    ctx->input_count = input_count;
    ctx->context_free = ctx_free;
    ctx->buffer_free = buf_free;
    ctx->backend_free = bfree;
    ctx->conv_plan_destroy = conv_plan_destroy;
    struct rllm_cuda_context *cuda_ctx = use_device
        ? rllm_cuda_context_get(backend_context, tensors) : NULL;
    ctx->backend = use_device ? cuda_ctx->backend : cpu_init();
    if (!ctx->backend) {
        rllm_f32_context_destroy(ctx);
        Rf_error(use_device ? "CUDA backend unavailable" : "CPU backend unavailable");
    }
    size_t graph_size = (size_t) n_nodes * 16 + 64;
    size_t metadata = 0, part, wmetadata, graph_overhead = g_over(graph_size);
    if (rllm_f32_mul_size(graph_size + 64, t_over(), &metadata) ||
        rllm_f32_mul_size((size_t) XLENGTH(tensors) + 16, t_over(), &wmetadata) ||
        metadata > SIZE_MAX - wmetadata || graph_overhead > SIZE_MAX - 4096 ||
        (metadata += wmetadata) > SIZE_MAX - (graph_overhead + 4096) ||
        wmetadata > SIZE_MAX - 4096) {
        rllm_f32_context_destroy(ctx);
        Rf_error("native F32 graph metadata allocation overflows size_t");
    }
    metadata += graph_overhead + 4096;
    ctx->wctx = use_device ? NULL : ctx_create(wmetadata + 4096, 1);
    ctx->cctx = ctx_create(metadata, 1);
    if (!ctx->cctx || (!use_device && !ctx->wctx)) {
        rllm_f32_context_destroy(ctx);
        Rf_error("native F32 fixed graph context creation failed");
    }
    ctx->input_host = malloc(input_bytes);
    if (!ctx->input_host) {
        rllm_f32_context_destroy(ctx);
        Rf_error("native F32 input staging allocation failed");
    }
    const int max_uploads = 2 * n_nodes + 8;
    struct rllm_f32_upload *uploads = calloc((size_t) max_uploads, sizeof(*uploads));
    struct ggml_tensor **values = calloc((size_t) n_nodes, sizeof(*values));
    if (!use_device) ctx->conv_plans = calloc((size_t) n_nodes, sizeof(*ctx->conv_plans));
    if (!uploads || !values || (!use_device && !ctx->conv_plans)) {
        free(values); rllm_f32_upload_release(uploads, max_uploads);
        rllm_f32_context_destroy(ctx);
        Rf_error("native F32 graph table allocation failed");
    }
    int n_uploads = 0;
#define RLLM_FX_FAIL(...) do { rllm_f32_upload_release(uploads, n_uploads); free(values); rllm_f32_context_destroy(ctx); Rf_error(__VA_ARGS__); } while (0)
#define RLLM_FX_CHECK(x) do { if (!(x)) RLLM_FX_FAIL("native F32 graph construction failed (%s)", #x); } while (0)

    int64_t input_ne[3] = { sequence_length, channels, batch };
    ctx->input = new_tensor(ctx->cctx, GGML_TYPE_F32, 3, input_ne, NULL);
    RLLM_FX_CHECK(ctx->input);
    values[0] = ctx->input;
    for (int i = 1; i < n_nodes; ++i) {
        SEXP node = VECTOR_ELT(nodes, i);
        SEXP attributes = rllm_list_elt(node, "attributes");
        const char *op = rllm_string(node, "op");
        if (!strcmp(op, "conv1d")) {
            struct ggml_tensor *x = rllm_f32_source(values, nodes, i, node, 0);
            const char *weight_name = rllm_f32_parameter(attributes, "weight");
            const char *bias_name = rllm_f32_parameter(attributes, "bias");
            int dilation = rllm_integer(attributes, "dilation");
            int padding = rllm_integer(attributes, "padding");
            int stride = rllm_integer(attributes, "stride");
            struct ggml_tensor *weight = rllm_named_weight(
                use_device ? cuda_ctx->wctx : ctx->wctx, tensors, weight_name,
                new_tensor, storage_data, NULL, NULL, 0, cuda_ctx);
            if (!weight || weight->ne[1] != x->ne[1] || dilation < 1 ||
                padding < 0 || stride < 1 || weight->ne[2] > INT_MAX) {
                RLLM_FX_FAIL("native F32 convolution dimensions are inconsistent");
            }
            int out_channels = (int) weight->ne[2];
            size_t bias_bytes;
            const float *bias = rllm_f32_parameter_data(tensors, bias_name,
                out_channels, &bias_bytes, storage_data);
            if (!use_device) {
                /* The CPU primitive copies and repacks the borrowed [K, IC, OC]
                 * mapping once. The graph then holds only the opaque plan; no
                 * R object or R API crosses a GGML worker callback. */
                size_t kernel_bytes;
                const float *kernel = rllm_f32_parameter_data(tensors, weight_name,
                    1, &kernel_bytes, storage_data);
                Rggml_conv_1d_f32_plan *plan = conv_plan_create(kernel,
                    kernel_bytes, bias, bias_bytes, weight->ne[0], weight->ne[1],
                    weight->ne[2], stride, padding, dilation, weight->ne[2]);
                if (plan) {
                    if (ctx->n_conv_plans >= n_nodes) {
                        conv_plan_destroy(plan);
                        RLLM_FX_FAIL("native F32 packed convolution plan table overflow");
                    }
                    ctx->conv_plans[ctx->n_conv_plans++] = plan;
                    values[i] = conv_plan_apply(ctx->cctx, plan, x);
                    RLLM_FX_CHECK(values[i]);
                    continue;
                }
            }
            /* CUDA custom callbacks are not CUDA operations. CUDA, and a CPU
             * shape that cannot admit a direct plan, keep the official F32
             * im2col/mul_mat composition as the independent fallback. */
            values[i] = conv1d(ctx->cctx, weight, x, stride, padding, dilation);
            RLLM_FX_CHECK(values[i]);
            int64_t affine_ne[3] = { 1, out_channels, 1 };
            struct ggml_tensor *bias_tensor = new_tensor(
                ctx->cctx, GGML_TYPE_F32, 3, affine_ne, NULL);
            RLLM_FX_CHECK(bias_tensor);
            rllm_f32_upload_add(uploads, &n_uploads, max_uploads, bias_tensor,
                                 bias, (size_t) out_channels * sizeof(*bias), NULL);
            values[i] = add(ctx->cctx, values[i], bias_tensor);
            RLLM_FX_CHECK(values[i]);
        } else if (!strcmp(op, "batch_norm")) {
            struct ggml_tensor *x = rllm_f32_source(values, nodes, i, node, 0);
            if (x->ne[1] < 1 || x->ne[1] > INT_MAX) {
                RLLM_FX_FAIL("native F32 batch normalization dimensions are invalid");
            }
            int width = (int) x->ne[1];
            double eps = rllm_number(attributes, "eps");
            const float *weight = rllm_f32_parameter_data(tensors,
                rllm_f32_parameter(attributes, "weight"), width, NULL, storage_data);
            const float *bias = rllm_f32_parameter_data(tensors,
                rllm_f32_parameter(attributes, "bias"), width, NULL, storage_data);
            const float *mean = rllm_f32_parameter_data(tensors,
                rllm_f32_parameter(attributes, "running_mean"), width, NULL, storage_data);
            const float *variance = rllm_f32_parameter_data(tensors,
                rllm_f32_parameter(attributes, "running_var"), width, NULL, storage_data);
            size_t affine_bytes;
            if (!R_FINITE(eps) || eps <= 0 ||
                rllm_f32_mul_size((size_t) width, sizeof(float), &affine_bytes)) {
                RLLM_FX_FAIL("native F32 batch normalization is invalid");
            }
            float *scale = malloc(affine_bytes), *shift = malloc(affine_bytes);
            if (!scale || !shift) { free(scale); free(shift); RLLM_FX_FAIL("native F32 normalization allocation failed"); }
            for (int j = 0; j < width; ++j) {
                if (!R_FINITE(variance[j]) || variance[j] < 0) {
                    free(scale); free(shift);
                    RLLM_FX_FAIL("native F32 batch normalization variance is invalid");
                }
                scale[j] = weight[j] / sqrtf(variance[j] + (float) eps);
                shift[j] = bias[j] - mean[j] * scale[j];
            }
            int64_t affine_ne[3] = { 1, width, 1 };
            struct ggml_tensor *scale_tensor = new_tensor(ctx->cctx, GGML_TYPE_F32, 3, affine_ne, NULL);
            struct ggml_tensor *shift_tensor = new_tensor(ctx->cctx, GGML_TYPE_F32, 3, affine_ne, NULL);
            if (!scale_tensor || !shift_tensor) { free(scale); free(shift); RLLM_FX_FAIL("native F32 graph construction failed (normalization tensors)"); }
            rllm_f32_upload_add(uploads, &n_uploads, max_uploads, scale_tensor, scale, affine_bytes, scale);
            rllm_f32_upload_add(uploads, &n_uploads, max_uploads, shift_tensor, shift, affine_bytes, shift);
            values[i] = add(ctx->cctx, mul(ctx->cctx, x, scale_tensor), shift_tensor);
            RLLM_FX_CHECK(values[i]);
        } else if (!strcmp(op, "leaky_relu")) {
            values[i] = leaky_relu(ctx->cctx,
                rllm_f32_source(values, nodes, i, node, 0), rllm_number(attributes, "slope"));
            RLLM_FX_CHECK(values[i]);
        } else if (!strcmp(op, "add")) {
            SEXP refs = rllm_list_elt(node, "inputs");
            if (TYPEOF(refs) != STRSXP || XLENGTH(refs) < 2) RLLM_FX_FAIL("native F32 add needs at least two inputs");
            values[i] = rllm_f32_source(values, nodes, i, node, 0);
            for (int j = 1; j < XLENGTH(refs); ++j) {
                struct ggml_tensor *right = rllm_f32_source(values, nodes, i, node, j);
                if (!rllm_f32_same_shape(values[i], right)) RLLM_FX_FAIL("native F32 add inputs have different dimensions");
                values[i] = add(ctx->cctx, values[i], right);
                RLLM_FX_CHECK(values[i]);
            }
        } else if (!strcmp(op, "crop1d")) {
            struct ggml_tensor *x = rllm_f32_source(values, nodes, i, node, 0);
            int left = rllm_integer(attributes, "left"), right = rllm_integer(attributes, "right");
            if (left < 0 || right < 0 || (int64_t) left + right >= x->ne[0]) RLLM_FX_FAIL("native F32 crop extents are invalid");
            values[i] = view_3d(ctx->cctx, x, x->ne[0] - left - right,
                x->ne[1], x->ne[2], x->nb[1], x->nb[2], (size_t) left * sizeof(float));
            RLLM_FX_CHECK(values[i]);
        } else if (!strcmp(op, "softmax")) {
            if (strcmp(rllm_string(attributes, "axis"), "feature")) RLLM_FX_FAIL("native F32 softmax needs feature axis");
            struct ggml_tensor *x = rllm_f32_source(values, nodes, i, node, 0);
            values[i] = soft_max(ctx->cctx, cont(ctx->cctx, permute(ctx->cctx, x, 1, 0, 2, 3)));
            RLLM_FX_CHECK(values[i]);
        } else {
            RLLM_FX_FAIL("native F32 execution does not implement operator '%s'", op);
        }
    }
    SEXP out_ref = VECTOR_ELT(outputs, 0);
    if (TYPEOF(out_ref) != STRSXP || XLENGTH(out_ref) != 1 || STRING_ELT(out_ref, 0) == NA_STRING) RLLM_FX_FAIL("native F32 program output id is invalid");
    const char *out_id = CHAR(STRING_ELT(out_ref, 0));
    SEXP output_names = Rf_getAttrib(outputs, R_NamesSymbol);
    if (TYPEOF(output_names) != STRSXP || XLENGTH(output_names) != 1 ||
        STRING_ELT(output_names, 0) == NA_STRING) RLLM_FX_FAIL("native F32 program output name is invalid");
    size_t output_name_bytes = strlen(CHAR(STRING_ELT(output_names, 0))) + 1;
    ctx->output_name = malloc(output_name_bytes);
    if (!ctx->output_name) RLLM_FX_FAIL("native F32 output name allocation failed");
    memcpy(ctx->output_name, CHAR(STRING_ELT(output_names, 0)), output_name_bytes);
    for (int i = 0; i < n_nodes; ++i) {
        if (!strcmp(rllm_string(VECTOR_ELT(nodes, i), "id"), out_id)) ctx->output = values[i];
    }
    if (!ctx->output || ctx->output->ne[0] > INT_MAX || ctx->output->ne[1] > INT_MAX || ctx->output->ne[2] > INT_MAX ||
        rllm_f32_product3((int) ctx->output->ne[0], (int) ctx->output->ne[1], (int) ctx->output->ne[2], &ctx->output_count, &part)) {
        RLLM_FX_FAIL("native F32 program output extent is invalid");
    }
    for (int i = 0; i < 3; ++i) ctx->output_dim[i] = (int) ctx->output->ne[i];
    ctx->output_host = malloc(part);
    if (!ctx->output_host) RLLM_FX_FAIL("native F32 output staging allocation failed");
    ctx->graph = new_graph(ctx->cctx, graph_size);
    RLLM_FX_CHECK(ctx->graph);
    expand(ctx->graph, ctx->output);
    ctx->cbuf = alloc_tensors(ctx->cctx, ctx->backend);
    if (!ctx->cbuf) RLLM_FX_FAIL("native F32 fixed backend allocation failed");
    for (int i = 0; i < n_uploads; ++i) {
        tensor_set(uploads[i].tensor, uploads[i].data, 0, uploads[i].bytes);
    }
    rllm_f32_upload_release(uploads, n_uploads);
    free(values);
    SEXP owner = PROTECT(Rf_allocVector(VECSXP, use_device ? 2 : 1));
    SET_VECTOR_ELT(owner, 0, execution);
    if (use_device) SET_VECTOR_ELT(owner, 1, backend_context);
    SEXP ext = PROTECT(R_MakeExternalPtr(ctx, rllm_f32_context_tag(), owner));
    R_RegisterCFinalizerEx(ext, rllm_f32_context_finalizer, FALSE);
    UNPROTECT(2);
    return ext;
#undef RLLM_FX_CHECK
#undef RLLM_FX_FAIL
}

SEXP RC_rllm_f32_program_forward(SEXP execution, SEXP inputs, SEXP threads_sexp,
                                 SEXP backend_sexp, SEXP backend_context,
                                 SEXP context_sexp)
{
    if (TYPEOF(execution) != VECSXP || TYPEOF(inputs) != VECSXP) {
        Rf_error("native F32 execution needs program bindings and named inputs");
    }
    SEXP sequence = rllm_list_elt(inputs, "sequence");
    SEXP dim = Rf_getAttrib(sequence, R_DimSymbol);
    int threads = Rf_asInteger(threads_sexp), backend_code = Rf_asInteger(backend_sexp);
    if ((TYPEOF(sequence) != REALSXP && TYPEOF(sequence) != INTSXP) ||
        TYPEOF(dim) != INTSXP || XLENGTH(dim) != 3 || threads == NA_INTEGER || threads < 1 ||
        (backend_code != 0 && backend_code != 3)) {
        Rf_error("invalid native F32 input, thread count, or backend");
    }
    int channels = INTEGER(dim)[0], sequence_length = INTEGER(dim)[1], batch = INTEGER(dim)[2];
    R_xlen_t input_count;
    size_t ignored;
    if (rllm_f32_product3(channels, sequence_length, batch, &input_count, &ignored) ||
        XLENGTH(sequence) != input_count) {
        Rf_error("native F32 input extent is invalid");
    }
    struct rllm_f32_context *ctx = rllm_f32_context_get(context_sexp, execution,
        backend_context, backend_code, channels, sequence_length, batch);
    SEXP ext = context_sexp;
    if (!ctx) {
        ext = PROTECT(rllm_f32_context_build(execution, channels, sequence_length,
                                             batch, backend_code, backend_context));
        ctx = (struct rllm_f32_context *) R_ExternalPtrAddr(ext);
    } else {
        PROTECT(ext);
    }
    if (backend_code == 0) Rggml_backend_cpu_set_n_threads_ptr()(ctx->backend, threads);
    rllm_f32_copy_input(ctx->input_host, sequence, channels, sequence_length, batch);
    Rggml_backend_tensor_set_ptr()(ctx->input, ctx->input_host, 0,
                                   (size_t) ctx->input_count * sizeof(float));
    if (Rggml_backend_graph_compute_ptr()(ctx->backend, ctx->graph) != 0) {
        UNPROTECT(1);
        Rf_error("native F32 graph compute failed");
    }
    Rggml_backend_tensor_get_ptr()(ctx->output, ctx->output_host, 0,
                                   (size_t) ctx->output_count * sizeof(float));
    SEXP result = PROTECT(rllm_f32_result(ctx));
    SEXP answer = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(answer, 0, result);
    SET_VECTOR_ELT(answer, 1, ext);
    UNPROTECT(3);
    return answer;
}
