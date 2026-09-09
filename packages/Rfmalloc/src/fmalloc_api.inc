extern "C" {

static bool is_open_runtime_xptr(SEXP runtime_xptr)
{
    return TYPEOF(runtime_xptr) == EXTPTRSXP &&
           R_ExternalPtrTag(runtime_xptr) == fmalloc_runtime_tag &&
           R_ExternalPtrAddr(runtime_xptr) != nullptr;
}

static bool same_runtime_xptr(SEXP lhs, SEXP rhs)
{
    if (!is_open_runtime_xptr(lhs) || !is_open_runtime_xptr(rhs)) {
        return false;
    }
    return lhs == rhs || R_ExternalPtrAddr(lhs) == R_ExternalPtrAddr(rhs);
}

static void clear_default_runtime_xptr(void)
{
    if (default_runtime_xptr != R_NilValue) {
        R_ReleaseObject(default_runtime_xptr);
        default_runtime_xptr = R_NilValue;
    }
}

static void preserve_default_runtime_xptr(SEXP runtime_xptr)
{
    if (Rf_isNull(runtime_xptr)) {
        clear_default_runtime_xptr();
        return;
    }

    fm_runtime *runtime = runtime_from_xptr(runtime_xptr);
    if (!runtime || !runtime->info) {
        Rf_error("fmalloc runtime is closed");
        return;
    }

    if (default_runtime_xptr == runtime_xptr) {
        return;
    }
    clear_default_runtime_xptr();
    default_runtime_xptr = runtime_xptr;
    R_PreserveObject(default_runtime_xptr);
}

SEXP open_fmalloc_impl(SEXP filepath_sexp, SEXP size_gb_sexp, SEXP mode_sexp)
{
    if (TYPEOF(filepath_sexp) != STRSXP || LENGTH(filepath_sexp) != 1) {
        Rf_error("filepath must be a single character string");
        return R_NilValue;
    }

    const char *filepath = CHAR(STRING_ELT(filepath_sexp, 0));
    if (!filepath || strlen(filepath) == 0) {
        Rf_error("filepath cannot be empty");
        return R_NilValue;
    }

    size_t requested_size = parse_requested_size(size_gb_sexp);
    fm_runtime_mode mode = parse_runtime_mode(mode_sexp);
    bool init_flag = false;
    fm_runtime *runtime = open_runtime_native(filepath, requested_size, mode, false, &init_flag);
    SEXP runtime_xptr = PROTECT(make_runtime_xptr(runtime));

    SEXP cls = PROTECT(Rf_mkString("fmalloc_runtime"));
    SEXP initialized_attr = PROTECT(Rf_ScalarLogical(init_flag));
    SEXP mode_attr = PROTECT(Rf_mkString(mode == FM_MODE_PERSISTENT ? "persistent" : "scratch"));
    std::string uuid = uuid_string(runtime->file_uuid_hi, runtime->file_uuid_lo);
    SEXP uuid_attr = PROTECT(Rf_mkString(uuid.c_str()));

    Rf_setAttrib(runtime_xptr, R_ClassSymbol, cls);
    Rf_setAttrib(runtime_xptr, Rf_install("initialized"), initialized_attr);
    Rf_setAttrib(runtime_xptr, Rf_install("mode"), mode_attr);
    Rf_setAttrib(runtime_xptr, Rf_install("uuid"), uuid_attr);

    UNPROTECT(5);
    return runtime_xptr;
}

// Backward-compatible C entry point. R code wraps this into logical-returning
// init_fmalloc(); new code should call open_fmalloc().
SEXP init_fmalloc_impl(SEXP filepath_sexp, SEXP size_gb_sexp, SEXP mode_sexp)
{
    SEXP runtime_xptr = PROTECT(open_fmalloc_impl(filepath_sexp, size_gb_sexp, mode_sexp));

    preserve_default_runtime_xptr(runtime_xptr);
    SEXP initialized = Rf_getAttrib(runtime_xptr, Rf_install("initialized"));
    int result = Rf_asLogical(initialized);
    UNPROTECT(1);
    return Rf_ScalarLogical(result);
}

static R_xlen_t parse_xlen_arg(SEXP length_sexp, const char *arg_name)
{
    if (TYPEOF(length_sexp) == INTSXP && LENGTH(length_sexp) == 1) {
        int value = INTEGER(length_sexp)[0];
        if (value == NA_INTEGER || value < 0) {
            Rf_error("%s must be a non-negative integer value", arg_name);
            return 0;
        }
        return static_cast<R_xlen_t>(value);
    }

    if (TYPEOF(length_sexp) == REALSXP && LENGTH(length_sexp) == 1) {
        double value = REAL(length_sexp)[0];
        if (!R_FINITE(value) || value < 0) {
            Rf_error("%s must be a finite non-negative value", arg_name);
            return 0;
        }
        if (std::floor(value) != value) {
            Rf_error("%s must be an exact integer value", arg_name);
            return 0;
        }
        if (value > (double)std::numeric_limits<R_xlen_t>::max()) {
            Rf_error("%s exceeds supported length", arg_name);
            return 0;
        }
        return static_cast<R_xlen_t>(value);
    }

    Rf_error("%s must be an integer-like scalar", arg_name);
    return 0;
}

SEXP create_fmalloc_vector_impl(SEXP runtime_xptr, SEXP template_vec, SEXP length_sexp,
                                     SEXP zero_initialize_sexp)
{

    fm_runtime *runtime = runtime_from_xptr(runtime_xptr);

    if (!template_vec) {
        Rf_error("Invalid arguments to create_fmalloc_vector");
        return R_NilValue;
    }

    bool zero_initialize = true;
    if (TYPEOF(zero_initialize_sexp) != LGLSXP || LENGTH(zero_initialize_sexp) != 1) {
        Rf_error("zero_initialize must be a single logical value");
        return R_NilValue;
    }
    int zero_initialize_i = Rf_asLogical(zero_initialize_sexp);
    if (zero_initialize_i == NA_LOGICAL) {
        Rf_error("zero_initialize must be TRUE or FALSE");
        return R_NilValue;
    }
    zero_initialize = zero_initialize_i == 1;

    R_xlen_t length = parse_xlen_arg(length_sexp, "length");

    SEXPTYPE vec_type = (SEXPTYPE)TYPEOF(template_vec);
    if (!is_supported_vector_type(vec_type)) {
        Rf_error("Unsupported vector type: %d", vec_type);
        return R_NilValue;
    }

    if (length >= 1000) {
        if (rfmalloc_verbose_output()) Rprintf("Creating fmalloc ALTREP vector: type=%s, length=%ld\n", type_label(vec_type), (long)length);
    }

    fm_vector *vec = allocate_fm_vector(runtime, vec_type, (R_xlen_t)length, true, zero_initialize);
    SEXP result = PROTECT(fmalloc_new_altrep(vec));

    if (length >= 1000) {
        if (rfmalloc_verbose_output()) Rprintf("Successfully created fmalloc ALTREP vector\n");
    }

    UNPROTECT(1);
    return result;
}

SEXP list_fmalloc_allocations_impl(SEXP runtime_xptr)
{
    fm_runtime *runtime = runtime_from_xptr(runtime_xptr);
    std::vector<catalog_snapshot> records = snapshot_catalog(runtime);
    return make_catalog_data_frame(records);
}

SEXP fmalloc_is_fmalloc_vector_impl(SEXP vector_x)
{
    fm_vector *vec = maybe_vector_from_altrep(vector_x);
    return Rf_ScalarLogical(vec != nullptr);
}

SEXP fmalloc_is_runtime_impl(SEXP runtime_xptr)
{
    return Rf_ScalarLogical(is_open_runtime_xptr(runtime_xptr));
}

SEXP fmalloc_vector_type_impl(SEXP vector_x)
{
    fm_vector *vec = maybe_vector_from_altrep(vector_x);
    if (!vec) {
        Rf_error("x is not an fmalloc ALTREP vector");
        return R_NilValue;
    }
    return Rf_ScalarInteger((int)vec->type);
}

SEXP fmalloc_vector_length_impl(SEXP vector_x)
{
    fm_vector *vec = maybe_vector_from_altrep(vector_x);
    if (!vec) {
        Rf_error("x is not an fmalloc ALTREP vector");
        return R_NilValue;
    }
    return Rf_ScalarReal((double)vec->len);
}

SEXP fmalloc_vector_payload_ptr_impl(SEXP vector_x)
{
    fm_vector *vec = maybe_vector_from_altrep(vector_x);
    if (!vec) {
        Rf_error("x is not an fmalloc ALTREP vector");
        return R_NilValue;
    }
    if (!vec->data) {
        return R_NilValue;
    }

    SEXP ptr = R_MakeExternalPtr(vec->data, Rf_install("Rfmalloc.payload"), vector_x);
    R_SetExternalPtrProtected(ptr, vector_x);
    return ptr;
}

SEXP fmalloc_vector_info_impl(SEXP vector_x)
{
    fm_vector *vec = maybe_vector_from_altrep(vector_x);
    if (!vec) {
        Rf_error("x is not an fmalloc ALTREP vector");
        return R_NilValue;
    }

    constexpr int n_info = 15;
    SEXP info = PROTECT(Rf_allocVector(VECSXP, n_info));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, n_info));
    const char *name_values[] = {
        "type", "sexptype", "length", "payload_nbytes", "payload_offset",
        "catalog_offset", "generation", "runtime_uuid", "runtime_mode",
        "runtime_filepath", "runtime_open", "parent_refs", "dataptr_exposed",
        "maybe_dirty", "recoverable"
    };
    for (int i = 0; i < n_info; i++) {
        SET_STRING_ELT(names, i, Rf_mkChar(name_values[i]));
    }
    Rf_setAttrib(info, R_NamesSymbol, names);

    fm_runtime *runtime = vec->runtime;
    bool runtime_open = runtime && runtime->info;

    SET_VECTOR_ELT(info, 0, Rf_mkString(type_label(vec->type)));
    SET_VECTOR_ELT(info, 1, Rf_ScalarInteger((int)vec->type));
    SET_VECTOR_ELT(info, 2, Rf_ScalarReal((double)vec->len));
    SET_VECTOR_ELT(info, 3, Rf_ScalarReal((double)vec->bytes));

    if (runtime_open && vec->data) {
        SET_VECTOR_ELT(info, 4, Rf_ScalarReal((double)pointer_offset(runtime, vec->data)));
    } else {
        SET_VECTOR_ELT(info, 4, Rf_ScalarReal(NA_REAL));
    }
    if (vec->catalog_offset != 0) {
        SET_VECTOR_ELT(info, 5, Rf_ScalarReal((double)vec->catalog_offset));
    } else {
        SET_VECTOR_ELT(info, 5, Rf_ScalarReal(NA_REAL));
    }
    if (vec->generation != 0) {
        SET_VECTOR_ELT(info, 6, Rf_ScalarReal((double)vec->generation));
    } else {
        SET_VECTOR_ELT(info, 6, Rf_ScalarReal(NA_REAL));
    }

    if (runtime) {
        std::string uuid = uuid_string(runtime->file_uuid_hi, runtime->file_uuid_lo);
        SET_VECTOR_ELT(info, 7, Rf_mkString(uuid.c_str()));
        SET_VECTOR_ELT(info, 8, Rf_mkString(runtime->mode == FM_MODE_PERSISTENT ? "persistent" : "scratch"));
        SET_VECTOR_ELT(info, 9, Rf_mkString(runtime->filepath.c_str()));
    } else {
        SET_VECTOR_ELT(info, 7, Rf_ScalarString(NA_STRING));
        SET_VECTOR_ELT(info, 8, Rf_ScalarString(NA_STRING));
        SET_VECTOR_ELT(info, 9, Rf_ScalarString(NA_STRING));
    }

    bool recoverable = false;
    if (runtime_open && vec->catalog_offset != 0) {
        rfm_catalog_record *record = catalog_record_from_offset(runtime, vec->catalog_offset);
        recoverable = record && record->state == RFM_CATALOG_STATE_COMMITTED &&
            (record->flags & RFM_CATALOG_FLAG_RECOVERABLE) != 0;
    }

    SET_VECTOR_ELT(info, 10, Rf_ScalarLogical(runtime_open));
    SET_VECTOR_ELT(info, 11, Rf_ScalarReal((double)vec->parent_refs));
    SET_VECTOR_ELT(info, 12, Rf_ScalarLogical(vec->dataptr_exposed));
    SET_VECTOR_ELT(info, 13, Rf_ScalarLogical(vec->maybe_dirty));
    SET_VECTOR_ELT(info, 14, Rf_ScalarLogical(recoverable));

    UNPROTECT(2);
    return info;
}

SEXP fmalloc_runtime_of_vector_impl(SEXP vector_x)
{
    fm_vector *vec = maybe_vector_from_altrep(vector_x);
    if (!vec) {
        Rf_error("x is not an fmalloc ALTREP vector");
        return R_NilValue;
    }

    if (!vec->runtime) {
        Rf_error("x has an invalid or closed fmalloc runtime");
        return R_NilValue;
    }

    return make_runtime_xptr(vec->runtime, true, false);
}

SEXP fmalloc_set_class_in_place_impl(SEXP x, SEXP class_vec)
{
    if (TYPEOF(class_vec) != STRSXP) {
        Rf_error("class must be a character vector");
        return R_NilValue;
    }

    Rf_setAttrib(x, R_ClassSymbol, class_vec);
    return x;
}

SEXP fmalloc_runtime_info_impl(SEXP runtime_xptr)
{
    fm_runtime *runtime = runtime_from_xptr(runtime_xptr);

    SEXP info = PROTECT(Rf_allocVector(VECSXP, 11));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 11));
    const char *name_values[] = {
        "filepath", "uuid", "mode", "mapped_bytes", "close_requested",
        "live_vectors", "external_refs", "catalog_head_offset", "catalog_count",
        "catalog_epoch", "runtime_open"
    };
    for (int i = 0; i < 11; i++) {
        SET_STRING_ELT(names, i, Rf_mkChar(name_values[i]));
    }
    Rf_setAttrib(info, R_NamesSymbol, names);

    SET_VECTOR_ELT(info, 0, Rf_mkString(runtime->filepath.c_str()));
    std::string uuid = uuid_string(runtime->file_uuid_hi, runtime->file_uuid_lo);
    SET_VECTOR_ELT(info, 1, Rf_mkString(uuid.c_str()));
    SET_VECTOR_ELT(info, 2, Rf_mkString(runtime->mode == FM_MODE_PERSISTENT ? "persistent" : "scratch"));

    if (runtime->info) {
        SET_VECTOR_ELT(info, 3, Rf_ScalarReal((double)runtime->info->len));
        SET_VECTOR_ELT(info, 4, Rf_ScalarLogical(runtime->close_requested));
        SET_VECTOR_ELT(info, 5, Rf_ScalarReal((double)runtime->live_vectors));
        SET_VECTOR_ELT(info, 6, Rf_ScalarReal((double)runtime->external_refs));
        rfm_app_root *root = app_root(runtime->info);
        SET_VECTOR_ELT(info, 7, Rf_ScalarReal((double)root->catalog_head_offset));
        SET_VECTOR_ELT(info, 8, Rf_ScalarReal((double)root->catalog_count));
        SET_VECTOR_ELT(info, 9, Rf_ScalarReal((double)root->catalog_epoch));
    } else {
        SET_VECTOR_ELT(info, 3, Rf_ScalarReal(NA_REAL));
        SET_VECTOR_ELT(info, 4, Rf_ScalarLogical(runtime->close_requested));
        SET_VECTOR_ELT(info, 5, Rf_ScalarReal(NA_REAL));
        SET_VECTOR_ELT(info, 6, Rf_ScalarReal(NA_REAL));
        SET_VECTOR_ELT(info, 7, Rf_ScalarReal(NA_REAL));
        SET_VECTOR_ELT(info, 8, Rf_ScalarReal(NA_REAL));
        SET_VECTOR_ELT(info, 9, Rf_ScalarReal(NA_REAL));
    }

    SET_VECTOR_ELT(info, 10, Rf_ScalarLogical(runtime->info != nullptr));
    UNPROTECT(2);
    return info;
}

SEXP fmalloc_set_dims_in_place_impl(SEXP vector_x, SEXP dim_sexp, SEXP dimnames_sexp)
{
    if (!ALTREP(vector_x)) {
        Rf_error("x must be an ALTREP object");
        return R_NilValue;
    }

    if (dimnames_sexp != R_NilValue && TYPEOF(dimnames_sexp) != VECSXP) {
        Rf_error("dimnames must be a list or NULL");
        return R_NilValue;
    }

    SEXP vector_xptr = R_altrep_data1(vector_x);
    if (TYPEOF(vector_xptr) != EXTPTRSXP || R_ExternalPtrTag(vector_xptr) != fmalloc_vector_tag) {
        Rf_error("x is not an fmalloc ALTREP vector");
        return R_NilValue;
    }

    fm_vector *vec = static_cast<fm_vector *>(R_ExternalPtrAddr(vector_xptr));
    if (!vec || !vec->runtime) {
        Rf_error("x is a closed fmalloc vector");
        return R_NilValue;
    }

    if (TYPEOF(dim_sexp) != INTSXP || LENGTH(dim_sexp) == 0) {
        Rf_error("dim must be a non-empty integer vector");
        return R_NilValue;
    }

    Rf_setAttrib(vector_x, R_DimSymbol, dim_sexp);
    if (dimnames_sexp != R_NilValue) {
        Rf_setAttrib(vector_x, R_DimNamesSymbol, dimnames_sexp);
    }

    return vector_x;
}

SEXP destroy_fmalloc_vector_impl(SEXP vector_x, SEXP unsafe_sexp)
{
    if (!ALTREP(vector_x)) {
        Rf_error("destroy_fmalloc_vector expects an fmalloc ALTREP vector");
        return R_NilValue;
    }

    if (TYPEOF(unsafe_sexp) != LGLSXP || LENGTH(unsafe_sexp) != 1) {
        Rf_error("unsafe must be a single logical value");
        return R_NilValue;
    }
    int unsafe_i = Rf_asLogical(unsafe_sexp);
    if (unsafe_i == NA_LOGICAL) {
        Rf_error("unsafe must be TRUE or FALSE");
        return R_NilValue;
    }

    SEXP vector_xptr = R_altrep_data1(vector_x);
    if (TYPEOF(vector_xptr) != EXTPTRSXP || R_ExternalPtrTag(vector_xptr) != fmalloc_vector_tag) {
        Rf_error("x is not an fmalloc ALTREP vector");
        return R_NilValue;
    }

    fm_vector *vec = static_cast<fm_vector *>(R_ExternalPtrAddr(vector_xptr));
    if (!vec) {
        return Rf_ScalarLogical(FALSE);
    }

    bool unsafe = unsafe_i == 1;
    bool destroyed = destroy_fm_vector(vec, unsafe, true);
    R_ClearExternalPtr(vector_xptr);
    R_SetExternalPtrProtected(vector_xptr, R_NilValue);
    return Rf_ScalarLogical(destroyed);
}

SEXP cleanup_fmalloc_impl(SEXP runtime_xptr)
{
    if (Rf_isNull(runtime_xptr)) {
        if (default_runtime_xptr == R_NilValue) {
            if (rfmalloc_verbose_output()) Rprintf("cleanup_fmalloc called but fmalloc not initialized\n");
            return R_NilValue;
        }
        runtime_xptr = default_runtime_xptr;
        default_runtime_xptr = R_NilValue;
        R_ReleaseObject(runtime_xptr);
    } else if (same_runtime_xptr(runtime_xptr, default_runtime_xptr)) {
        clear_default_runtime_xptr();
    }

    if (TYPEOF(runtime_xptr) != EXTPTRSXP || R_ExternalPtrTag(runtime_xptr) != fmalloc_runtime_tag) {
        Rf_error("invalid fmalloc runtime");
        return R_NilValue;
    }

    if (rfmalloc_verbose_output()) Rprintf("Cleaning up fmalloc...\n");
    request_runtime_close(runtime_xptr, true);
    if (rfmalloc_verbose_output()) Rprintf("fmalloc cleaned up\n");
    return R_NilValue;
}

/*
 * Decode a flat, block-aligned element range of a typed fmalloc tensor into a
 * caller-owned double buffer, using the tensor's own registered codec. This is
 * the streaming read primitive an out-of-core consumer
 * (e.g. RfmallocStatgen's RSVD) needs: decode one column range of variants at a
 * time instead of materializing the whole matrix. Because standardization is a
 * property of the tensor - a standardized 'bed'/'dosage' payload decodes
 * centred/scaled, mean-imputed - no standardize flag is needed here; pass a
 * standardized tensor to stream standardized values.
 *
 * 'tensor' is an fmalloc tensor (the ALTREP payload carrying 'rfm_dtype' and
 * 'dim' attributes, as fmalloc_tensor() builds it). The range is
 * [elem_offset, elem_offset + n_elems) in column-major (dim[0]-fastest) element
 * order; the codec requires it to be block-aligned (whole variant columns for
 * 'bed'/'dosage'), and 'out' must hold at least n_elems doubles. Returns 0 on
 * success and non-zero on any error (not a tensor, unknown codec, out-of-range
 * request, or a codec-level failure). It never calls Rf_error, so a C++ caller
 * can turn a failure into its own control flow with no longjmp through its stack.
 */
int Rfmalloc_tensor_decode(SEXP tensor, R_xlen_t elem_offset, R_xlen_t n_elems,
                           double *out)
{
    if (!out || elem_offset < 0 || n_elems < 0) {
        return -1;
    }
    const void *payload;
    size_t payload_bytes;
    if (Rfmalloc_storage_data(tensor, &payload, &payload_bytes, nullptr) != 0) {
        return -1;
    }
    SEXP dtype = Rf_getAttrib(tensor, Rf_install("rfm_dtype"));
    if (TYPEOF(dtype) != STRSXP || XLENGTH(dtype) != 1) {
        return -1;
    }
    const rfm_tensor_codec *codec = tensor_codec_lookup(CHAR(STRING_ELT(dtype, 0)));
    if (!codec || !codec->decode) {
        return -1;
    }
    // Bound the request against the tensor's declared element count. Tensor
    // dims live in the 'rfm_dims' attribute (dim() is an S3 method over it),
    // not R's DimSymbol.
    SEXP dims = Rf_getAttrib(tensor, Rf_install("rfm_dims"));
    if (TYPEOF(dims) != INTSXP || XLENGTH(dims) < 1) {
        return -1;
    }
    R_xlen_t total = 1;
    for (R_xlen_t i = 0; i < XLENGTH(dims); i++) {
        R_xlen_t d = (R_xlen_t)INTEGER(dims)[i];
        if (d < 0 || (d != 0 && total > R_XLEN_T_MAX / d)) {
            return -1;
        }
        total *= d;
    }
    if (elem_offset > total || n_elems > total - elem_offset) {
        return -1;
    }
    if (n_elems == 0) {
        return 0;
    }
    const uint64_t n_blocks = (uint64_t)(total / codec->items_per_block) +
        (total % codec->items_per_block != 0);
    if (n_blocks > SIZE_MAX / codec->bytes_per_block ||
        (size_t)n_blocks * codec->bytes_per_block > payload_bytes) {
        return -1;
    }
    return codec->decode(payload, elem_offset, n_elems, out);
}

// R-level wrapper, so Rfmalloc's own tests can exercise Rfmalloc_tensor_decode
// without a second package. Offset/length come in as doubles (they can exceed
// int range); returns the decoded range as a plain numeric vector.
SEXP rfm_tensor_decode_range_impl(SEXP tensor, SEXP offset_sexp, SEXP n_sexp)
{
    const double offset_d = Rf_asReal(offset_sexp);
    const double n_d = Rf_asReal(n_sexp);
    if (!R_FINITE(offset_d) || offset_d < 0 ||
        offset_d != std::floor(offset_d) || offset_d > (double)R_XLEN_T_MAX) {
        Rf_error("offset must be a supported non-negative whole number");
    }
    if (!R_FINITE(n_d) || n_d < 0 || n_d != std::floor(n_d) ||
        n_d > (double)R_XLEN_T_MAX) {
        Rf_error("n must be a supported non-negative whole number");
    }
    const R_xlen_t elem_offset = (R_xlen_t)offset_d;
    const R_xlen_t n_elems = (R_xlen_t)n_d;
    SEXP out = PROTECT(Rf_allocVector(REALSXP, n_elems));
    int rc = Rfmalloc_tensor_decode(tensor, elem_offset, n_elems, REAL(out));
    if (rc != 0) {
        UNPROTECT(1);
        Rf_error("Rfmalloc_tensor_decode failed (code %d)", rc);
    }
    UNPROTECT(1);
    return out;
}

SEXP Rfmalloc_default_runtime(void)
{
    if (default_runtime_xptr != R_NilValue && !is_open_runtime_xptr(default_runtime_xptr)) {
        clear_default_runtime_xptr();
    }
    return default_runtime_xptr;
}

SEXP fmalloc_default_runtime_impl(void)
{
    return Rfmalloc_default_runtime();
}

SEXP set_default_fmalloc_runtime_impl(SEXP runtime_xptr)
{
    if (Rf_isNull(runtime_xptr)) {
        clear_default_runtime_xptr();
        return R_NilValue;
    }

    preserve_default_runtime_xptr(runtime_xptr);
    return Rfmalloc_default_runtime();
}

SEXP Rfmalloc_open_runtime(const char *filepath, double size_gb, const char *mode)
{
    if (!filepath || filepath[0] == '\0') {
        Rf_error("filepath must be a non-empty C string");
        return R_NilValue;
    }
    if (!mode || mode[0] == '\0') {
        mode = "persistent";
    }

    int nprotect = 0;
    SEXP filepath_sexp = PROTECT(Rf_mkString(filepath));
    nprotect++;
    SEXP mode_sexp = PROTECT(Rf_mkString(mode));
    nprotect++;
    SEXP size_gb_sexp = R_NilValue;
    if (std::isfinite(size_gb) && size_gb > 0) {
        size_gb_sexp = PROTECT(Rf_ScalarReal(size_gb));
        nprotect++;
    }

    SEXP ans = PROTECT(open_fmalloc_impl(filepath_sexp, size_gb_sexp, mode_sexp));
    nprotect++;
    UNPROTECT(nprotect);
    return ans;
}

SEXP Rfmalloc_create_vector_ex(SEXP runtime_xptr, int sexptype, R_xlen_t length,
                               int zero_initialize)
{
    if (length < 0) {
        Rf_error("length must be non-negative");
        return R_NilValue;
    }

    SEXPTYPE type = (SEXPTYPE)sexptype;
    if (!is_supported_vector_type(type)) {
        Rf_error("Unsupported vector type: %d", type);
        return R_NilValue;
    }

    fm_runtime *runtime = runtime_from_xptr(runtime_xptr);
    fm_vector *vec = allocate_fm_vector(runtime, type, length, true, zero_initialize != 0);
    SEXP ans = PROTECT(fmalloc_new_altrep(vec));
    UNPROTECT(1);
    return ans;
}

SEXP Rfmalloc_create_vector(SEXP runtime_xptr, int sexptype, R_xlen_t length)
{
    return Rfmalloc_create_vector_ex(runtime_xptr, sexptype, length, 1);
}

void Rfmalloc_cleanup_runtime(SEXP runtime_xptr)
{
    if (Rf_isNull(runtime_xptr)) {
        return;
    }
    if (TYPEOF(runtime_xptr) != EXTPTRSXP || R_ExternalPtrTag(runtime_xptr) != fmalloc_runtime_tag) {
        Rf_error("invalid fmalloc runtime");
        return;
    }
    request_runtime_close(runtime_xptr, true);
}

SEXP Rfmalloc_list_allocations(SEXP runtime_xptr)
{
    fm_runtime *runtime = runtime_from_xptr(runtime_xptr);
    std::vector<catalog_snapshot> records = snapshot_catalog(runtime);
    return make_catalog_data_frame(records);
}

SEXP Rfmalloc_set_default_runtime(SEXP runtime_xptr)
{
    return set_default_fmalloc_runtime_impl(runtime_xptr);
}

SEXP Rfmalloc_is_runtime(SEXP runtime_xptr)
{
    return fmalloc_is_runtime_impl(runtime_xptr);
}

SEXP Rfmalloc_runtime_of_vector(SEXP vector_x)
{
    return fmalloc_runtime_of_vector_impl(vector_x);
}

SEXP Rfmalloc_runtime_info(SEXP runtime_xptr)
{
    return fmalloc_runtime_info_impl(runtime_xptr);
}

SEXP Rfmalloc_vector_info(SEXP vector_x)
{
    return fmalloc_vector_info_impl(vector_x);
}

SEXP Rfmalloc_destroy_vector(SEXP vector_x, int unsafe)
{
    SEXP unsafe_sexp = PROTECT(Rf_ScalarLogical(unsafe != 0));
    SEXP ans = PROTECT(destroy_fmalloc_vector_impl(vector_x, unsafe_sexp));
    UNPROTECT(2);
    return ans;
}

static void register_c_callables(void)
{
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_default_runtime", (DL_FUNC)Rfmalloc_default_runtime);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_open_runtime", (DL_FUNC)Rfmalloc_open_runtime);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_create_vector", (DL_FUNC)Rfmalloc_create_vector);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_cleanup_runtime", (DL_FUNC)Rfmalloc_cleanup_runtime);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_set_default_runtime", (DL_FUNC)Rfmalloc_set_default_runtime);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_list_allocations", (DL_FUNC)Rfmalloc_list_allocations);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_create_vector_ex", (DL_FUNC)Rfmalloc_create_vector_ex);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_is_runtime", (DL_FUNC)Rfmalloc_is_runtime);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_is_fmalloc_vector", (DL_FUNC)fmalloc_is_fmalloc_vector_impl);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_runtime_of_vector", (DL_FUNC)Rfmalloc_runtime_of_vector);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_runtime_info", (DL_FUNC)Rfmalloc_runtime_info);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_vector_type", (DL_FUNC)fmalloc_vector_type_impl);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_vector_length", (DL_FUNC)fmalloc_vector_length_impl);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_vector_payload_ptr", (DL_FUNC)fmalloc_vector_payload_ptr_impl);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_vector_info", (DL_FUNC)Rfmalloc_vector_info);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_destroy_vector", (DL_FUNC)Rfmalloc_destroy_vector);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_register_tensor_codec", (DL_FUNC)Rfmalloc_register_tensor_codec);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_register_matmul_backend", (DL_FUNC)Rfmalloc_register_matmul_backend);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_register_matmul_backend_ex", (DL_FUNC)Rfmalloc_register_matmul_backend_ex);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_tensor_decode", (DL_FUNC)Rfmalloc_tensor_decode);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_buffer_open", (DL_FUNC)Rfmalloc_buffer_open);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_buffer_write", (DL_FUNC)Rfmalloc_buffer_write);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_buffer_finish", (DL_FUNC)Rfmalloc_buffer_finish);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_buffer_abort", (DL_FUNC)Rfmalloc_buffer_abort);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_haplotypes_data", (DL_FUNC)Rfmalloc_haplotypes_data);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_storage_view", (DL_FUNC)Rfmalloc_storage_view);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_storage_data", (DL_FUNC)Rfmalloc_storage_data);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_storage_advise", (DL_FUNC)Rfmalloc_storage_advise);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_ld_ncol", (DL_FUNC)Rfmalloc_ld_ncol);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_ld_bits", (DL_FUNC)Rfmalloc_ld_bits);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_ld_pair", (DL_FUNC)Rfmalloc_ld_pair);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_ld_col", (DL_FUNC)Rfmalloc_ld_col);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_ld_col_raw", (DL_FUNC)Rfmalloc_ld_col_raw);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_ld_build", (DL_FUNC)Rfmalloc_ld_build);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_ld_alloc", (DL_FUNC)Rfmalloc_ld_alloc);
    R_RegisterCCallable("Rfmalloc", "Rfmalloc_ld_write", (DL_FUNC)Rfmalloc_ld_write);
}

static const R_CallMethodDef CallEntries[] = {
    {"open_fmalloc_impl", (DL_FUNC)&open_fmalloc_impl, 3},
    {"init_fmalloc_impl", (DL_FUNC)&init_fmalloc_impl, 3},
    {"fmalloc_default_runtime_impl", (DL_FUNC)&fmalloc_default_runtime_impl, 0},
    {"set_default_fmalloc_runtime_impl", (DL_FUNC)&set_default_fmalloc_runtime_impl, 1},
    {"create_fmalloc_vector_impl", (DL_FUNC)&create_fmalloc_vector_impl, 4},
    {"list_fmalloc_allocations_impl", (DL_FUNC)&list_fmalloc_allocations_impl, 1},
    {"fmalloc_is_runtime_impl", (DL_FUNC)&fmalloc_is_runtime_impl, 1},
    {"fmalloc_is_fmalloc_vector_impl", (DL_FUNC)&fmalloc_is_fmalloc_vector_impl, 1},
    {"fmalloc_vector_type_impl", (DL_FUNC)&fmalloc_vector_type_impl, 1},
    {"fmalloc_vector_length_impl", (DL_FUNC)&fmalloc_vector_length_impl, 1},
    {"fmalloc_vector_payload_ptr_impl", (DL_FUNC)&fmalloc_vector_payload_ptr_impl, 1},
    {"fmalloc_vector_info_impl", (DL_FUNC)&fmalloc_vector_info_impl, 1},
    {"fmalloc_runtime_of_vector_impl", (DL_FUNC)&fmalloc_runtime_of_vector_impl, 1},
    {"fmalloc_set_class_in_place_impl", (DL_FUNC)&fmalloc_set_class_in_place_impl, 2},
    {"fmalloc_runtime_info_impl", (DL_FUNC)&fmalloc_runtime_info_impl, 1},
    {"fmalloc_set_dims_in_place_impl", (DL_FUNC)&fmalloc_set_dims_in_place_impl, 3},
    {"destroy_fmalloc_vector_impl", (DL_FUNC)&destroy_fmalloc_vector_impl, 2},
    {"cleanup_fmalloc_impl", (DL_FUNC)&cleanup_fmalloc_impl, 1},
    {"rfm_ops_dispatch", (DL_FUNC)&rfm_ops_dispatch, 4},
    {"rfm_matrix_ops_dispatch", (DL_FUNC)&rfm_matrix_ops_dispatch, 3},
    {"rfm_can_handle_ops_pair", (DL_FUNC)&rfm_can_handle_ops_pair, 3},
    {"rfm_tensor_matmul_impl", (DL_FUNC)&rfm_tensor_matmul_impl, 7},
    {"rfm_tensor_materialize_impl", (DL_FUNC)&rfm_tensor_materialize_impl, 3},
    {"rfm_tensor_decode_range_impl", (DL_FUNC)&rfm_tensor_decode_range_impl, 3},
    {"rfm_tensor_codec_info_impl", (DL_FUNC)&rfm_tensor_codec_info_impl, 1},
    {"rfm_tensor_codec_list_impl", (DL_FUNC)&rfm_tensor_codec_list_impl, 0},
    {"rfm_tensor_bed_encode_impl", (DL_FUNC)&rfm_tensor_bed_encode_impl, 2},
    {"rfm_tensor_bed_standardize_impl", (DL_FUNC)&rfm_tensor_bed_standardize_impl, 3},
    {"rfm_tensor_dosage_encode_impl", (DL_FUNC)&rfm_tensor_dosage_encode_impl, 2},
    {"rfm_tensor_dosage_standardize_impl", (DL_FUNC)&rfm_tensor_dosage_standardize_impl, 3},
    {"rfm_hap_encode_impl", (DL_FUNC)&rfm_hap_encode_impl, 2},
    {"rfm_hap_materialize_impl", (DL_FUNC)&rfm_hap_materialize_impl, 2},
    {"rfm_hap_view_info_impl", (DL_FUNC)&rfm_hap_view_info_impl, 1},
    {"rfm_ld_encode_triplets_impl", (DL_FUNC)&rfm_ld_encode_triplets_impl, 7},
    {"rfm_ld_info_impl", (DL_FUNC)&rfm_ld_info_impl, 1},
    {"rfm_ld_ncol_impl", (DL_FUNC)&rfm_ld_ncol_impl, 1},
    {"rfm_ld_pair_impl", (DL_FUNC)&rfm_ld_pair_impl, 3},
    {"rfm_ld_col_impl", (DL_FUNC)&rfm_ld_col_impl, 2},
    {"rfm_matmul_ooc_impl", (DL_FUNC)&rfm_matmul_ooc_impl, 3},
    {"rfm_crossprod_ooc_impl", (DL_FUNC)&rfm_crossprod_ooc_impl, 3},
    {"rfm_tcrossprod_ooc_impl", (DL_FUNC)&rfm_tcrossprod_ooc_impl, 2},
    {"rfm_vector_advise_impl", (DL_FUNC)&rfm_vector_advise_impl, 2},
    {"rfm_sync_impl", (DL_FUNC)&rfm_sync_impl, 2},
    {"rfm_phys_ram_bytes_impl", (DL_FUNC)&rfm_phys_ram_bytes_impl, 0},
    {"rfm_raw_fill_pattern_impl", (DL_FUNC)&rfm_raw_fill_pattern_impl, 2},
    {"rfm_set_in_place_impl", (DL_FUNC)&rfm_set_in_place_impl, 3},
    {"rfm_fill_in_place_impl", (DL_FUNC)&rfm_fill_in_place_impl, 2},
    {"rfm_inplace_op_impl", (DL_FUNC)&rfm_inplace_op_impl, 3},
    {"rfm_tensor_alp_encode_impl", (DL_FUNC)&rfm_tensor_alp_encode_impl, 2},
    {"rfm_tensor_sparse_encode_impl", (DL_FUNC)&rfm_tensor_sparse_encode_impl, 2},
    {"rfm_tensor_payload_nbytes_impl", (DL_FUNC)&rfm_tensor_payload_nbytes_impl, 1},
    {"rfm_storage_advise_impl", (DL_FUNC)&rfm_storage_advise_impl, 4},
    {"rfm_set_matmul_backend_impl", (DL_FUNC)&rfm_set_matmul_backend_impl, 1},
    {"rfm_matmul_backend_impl", (DL_FUNC)&rfm_matmul_backend_impl, 0},
    {"rfm_matmul_backends_impl", (DL_FUNC)&rfm_matmul_backends_impl, 0},
    {"rfm_register_test_backend_impl", (DL_FUNC)&rfm_register_test_backend_impl, 0},
    {"rfm_register_test_typed_backend_impl", (DL_FUNC)&rfm_register_test_typed_backend_impl, 0},
    {nullptr, nullptr, 0}
};

void R_init_Rfmalloc(DllInfo *dll)
{
    fmalloc_runtime_tag = Rf_install("Rfmalloc.runtime");
    fmalloc_vector_tag = Rf_install("Rfmalloc.vector");
    fmalloc_storage_span_tag = Rf_install("Rfmalloc.storage_span");
    fmalloc_storage_runtime_symbol = Rf_install("rfm_runtime");
    register_fmalloc_altrep_classes(dll);
    tensor_register_builtin_codecs();
    tensor_register_alp_codec();
    tensor_register_sparse_codec();
    tensor_register_bed_codec();
    tensor_register_dosage_codec();
    R_registerRoutines(dll, nullptr, CallEntries, nullptr, nullptr);
    register_c_callables();
    R_useDynamicSymbols(dll, FALSE);
}

void R_unload_Rfmalloc(DllInfo *dll)
{
    (void)dll;
    clear_default_runtime_xptr();
}

} // extern "C"
