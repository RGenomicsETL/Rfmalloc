//==============================================================================
// Banded symmetric LD (correlation) matrix as an fmalloc raw payload: a
// self-indexing, mmap-backed store of quantized correlations.
//
// This is a SIBLING interface to the matmul tensor codec ABI (like
// fmalloc_hap.inc), not a codec. An LD matrix is not consumed by a GEMM that
// decodes it to a dense p x p double matrix - it is read one column's
// neighbours at a time by a Gibbs sampler (LDpred2) or a ridge solve
// (LDpred2-inf), so it is a typed accessor with its own encode/read pair,
// never registered with Rfmalloc_register_tensor_codec and never part of "%*%".
//
// Structure exploited: the variants are position-sorted, so each column j's
// in-window neighbours form a CONTIGUOUS index range [lo_j, hi_j]. We store the
// FULL symmetric band per column (r[j,k] and r[k,j] both), the diagonal is
// r = 1, and NO explicit neighbour indices are needed - the row index of the
// t-th stored value of column j is simply lo_j + t. A per-column offset table
// (a cumulative element count + the band start lo_j) gives O(1) random access
// to any column's neighbour run and a trivial, cache-friendly banded matvec.
//
// Quantization: r in [-1, 1] -> integer q = round(r * S) clamped to [-S, S],
// decoded as r ~ q / S, with S = 127 for bits = 8 (int8, resolution ~1/127)
// or S = 32767 for bits = 16 (int16, resolution ~3e-5). bits, n_variants and
// the build window live in the header.
//
// Payload layout (little-endian, 8-byte aligned):
//
//   [0..8)    magic 'F','M','L','D',0,0,0, kind   (kind 1: only kind so far)
//   [8..16)   n_variants (m)                       uint64
//   [16..24)  nnz  (total stored band entries)     uint64
//   [24..28)  bits (8 or 16)                        uint32
//   [28..32)  window (informational, 0 if unknown)  uint32
//   [32..40)  reserved (0)                          uint64
//   [40 .. 40+8*(m+1))            cum[0..m]  uint64  cumulative element counts
//   [.. +8*m)                     lo[0..m-1] int64   per-column band start row
//   [values ..]                   nnz * (bits/8) bytes, int8 or int16, column-major
//
// Column j owns values[cum[j] .. cum[j+1]), covering row indices
// lo[j] .. lo[j] + (cum[j+1]-cum[j]) - 1. len_j = cum[j+1] - cum[j].
//
// Density: for a band of half-width w (len_j ~ 2w+1) at int8, ~2w+1 bytes per
// variant of values plus 16 bytes of offset table - vs 8*m bytes/variant for a
// dense double row. A megabase LD block at 8 bits is a small fraction of the
// dense p x p double matrix it represents, and it is mmap/OOC-capable: the
// quantized values live in the fmalloc payload and are never expanded to a
// dense p x p.
//==============================================================================

#define RFM_LD_HDR_BYTES 40
static const uint8_t rfm_ld_magic7[7] = {'F', 'M', 'L', 'D', 0, 0, 0};
#define RFM_LD_RAW 1

struct rfm_ld_header {
    uint8_t magic[8];
    uint64_t n_variants;
    uint64_t nnz;
    uint32_t bits;
    uint32_t window;
    uint64_t reserved;
};

static inline double rfm_ld_scale(int bits)
{
    return (bits == 16) ? 32767.0 : 127.0;
}

// r in [-1, 1] -> clamped integer code. Diagonal r = 1 quantizes to exactly S,
// which decodes back to 1.0 (S / S = 1).
static inline long rfm_ld_quant(double r, int bits)
{
    const long lim = (bits == 16) ? 32767L : 127L;
    if (ISNAN(r)) {
        return 0;
    }
    long q = lround(r * (double)lim);
    if (q > lim) q = lim;
    if (q < -lim) q = -lim;
    return q;
}

// A validated view over an ld store payload. All pointers reference the mmap'd
// fmalloc payload directly (zero copy); they are valid while the store lives.
struct rfm_ld_view {
    const uint8_t *base;
    R_xlen_t m;        // n_variants
    R_xlen_t nnz;
    int bits;
    uint32_t window;
    const uint64_t *cum;   // length m + 1
    const int64_t *lo;     // length m
    const void *values;    // nnz * (bits/8) bytes
};

// Parse and bounds-check an ld store. Returns 0 on success, non-zero otherwise;
// never calls Rf_error (safe from a C-callable used across packages).
static int rfm_ld_open(SEXP store, rfm_ld_view *out)
{
    fm_vector *vec = maybe_vector_from_altrep(store);
    if (!vec || vec->type != RAWSXP) {
        return -1;
    }
    if ((size_t)vec->len < RFM_LD_HDR_BYTES) {
        return -1;
    }
    const uint8_t *base = static_cast<const uint8_t *>(vector_data_or_dummy(vec));
    rfm_ld_header hdr;
    memcpy(&hdr, base, sizeof(hdr));
    if (memcmp(hdr.magic, rfm_ld_magic7, sizeof(rfm_ld_magic7)) != 0) {
        return -1;
    }
    if (hdr.magic[7] != RFM_LD_RAW) {
        return -1;
    }
    if (hdr.bits != 8 && hdr.bits != 16) {
        return -1;
    }
    const R_xlen_t m = (R_xlen_t)hdr.n_variants;
    const R_xlen_t nnz = (R_xlen_t)hdr.nnz;
    if (m < 0 || nnz < 0) {
        return -1;
    }
    const size_t bpv = hdr.bits / 8;
    const size_t off_cum = RFM_LD_HDR_BYTES;
    const size_t off_lo = off_cum + (size_t)(m + 1) * 8;
    const size_t off_val = off_lo + (size_t)m * 8;
    const size_t need = off_val + (size_t)nnz * bpv;
    if ((size_t)vec->len < need) {
        return -1;
    }
    out->base = base;
    out->m = m;
    out->nnz = nnz;
    out->bits = (int)hdr.bits;
    out->window = hdr.window;
    out->cum = reinterpret_cast<const uint64_t *>(base + off_cum);
    out->lo = reinterpret_cast<const int64_t *>(base + off_lo);
    out->values = base + off_val;
    return 0;
}

static inline double rfm_ld_value_at(const rfm_ld_view *v, R_xlen_t k)
{
    if (v->bits == 16) {
        return (double)reinterpret_cast<const int16_t *>(v->values)[k] / 32767.0;
    }
    return (double)reinterpret_cast<const int8_t *>(v->values)[k] / 127.0;
}

//==============================================================================
// Internal builder: assemble a payload from per-column bands. lo[j], len[j] and
// the column-major concatenated correlations rvals (length sum(len)) fully
// describe the band; the diagonal is expected to be present in rvals as 1.0 but
// nothing here enforces it (the R/C callers do). Returns an fmalloc ALTREP raw
// payload (the caller PROTECTs it).
//==============================================================================
// Allocate a payload whose band geometry is final but whose correlations are
// still zero, so a producer can fill it one column at a time and never hold the
// whole band in memory. rfm_ld_build_payload() is the all-at-once form of the
// same thing.
static SEXP rfm_ld_alloc_payload(fm_runtime *runtime, R_xlen_t m, int bits,
                                 uint32_t window, const R_xlen_t *lo,
                                 const R_xlen_t *len)
{
    if (bits != 8 && bits != 16) {
        Rf_error("ld bits must be 8 or 16");
    }
    R_xlen_t nnz = 0;
    for (R_xlen_t j = 0; j < m; j++) {
        if (len[j] < 0) {
            Rf_error("ld column length must be non-negative");
        }
        nnz += len[j];
    }
    const size_t bpv = (size_t)bits / 8;
    const size_t off_cum = RFM_LD_HDR_BYTES;
    const size_t off_lo = off_cum + (size_t)(m + 1) * 8;
    const size_t off_val = off_lo + (size_t)m * 8;
    const size_t total = off_val + (size_t)nnz * bpv;

    fm_vector *vec = allocate_fm_vector(runtime, RAWSXP, (R_xlen_t)total, true, false);
    SEXP ans = PROTECT(fmalloc_new_altrep(vec));
    uint8_t *p = static_cast<uint8_t *>(vector_data_or_dummy(vec));

    rfm_ld_header hdr;
    memcpy(hdr.magic, rfm_ld_magic7, sizeof(rfm_ld_magic7));
    hdr.magic[7] = RFM_LD_RAW;
    hdr.n_variants = (uint64_t)m;
    hdr.nnz = (uint64_t)nnz;
    hdr.bits = (uint32_t)bits;
    hdr.window = window;
    hdr.reserved = 0;
    memcpy(p, &hdr, sizeof(hdr));

    uint64_t *cum = reinterpret_cast<uint64_t *>(p + off_cum);
    int64_t *lo_out = reinterpret_cast<int64_t *>(p + off_lo);
    uint64_t acc = 0;
    for (R_xlen_t j = 0; j < m; j++) {
        cum[j] = acc;
        lo_out[j] = (int64_t)lo[j];
        acc += (uint64_t)len[j];
    }
    cum[m] = acc;
    memset(p + off_val, 0, (size_t)nnz * bpv);

    vec->maybe_dirty = true;
    UNPROTECT(1);
    return ans;
}

// Quantize `count` correlations into column j's band, starting `first` entries
// into that band. Returns 0, or -1 when the store or the range is invalid.
static int rfm_ld_write_column(SEXP store, R_xlen_t j, R_xlen_t first,
                               R_xlen_t count, const double *values)
{
    rfm_ld_view view;
    if (rfm_ld_open(store, &view) != 0) return -1;
    if (j < 0 || j >= view.m) return -1;
    if (first < 0 || count < 0 || !values) return -1;
    const uint64_t begin = view.cum[j];
    const R_xlen_t len = (R_xlen_t)(view.cum[j + 1] - begin);
    if (first + count > len) return -1;
    if (view.bits == 16) {
        int16_t *q = static_cast<int16_t *>(const_cast<void *>(view.values));
        for (R_xlen_t t = 0; t < count; t++) {
            q[begin + first + t] = (int16_t)rfm_ld_quant(values[t], 16);
        }
    } else {
        int8_t *q = static_cast<int8_t *>(const_cast<void *>(view.values));
        for (R_xlen_t t = 0; t < count; t++) {
            q[begin + first + t] = (int8_t)rfm_ld_quant(values[t], 8);
        }
    }
    {
        fm_vector *vec = maybe_vector_from_altrep(store);
        if (vec) vec->maybe_dirty = true;
    }
    return 0;
}

static SEXP rfm_ld_build_payload(fm_runtime *runtime, R_xlen_t m, int bits,
                                 uint32_t window, const R_xlen_t *lo,
                                 const R_xlen_t *len, const double *rvals)
{
    if (bits != 8 && bits != 16) {
        Rf_error("ld bits must be 8 or 16");
    }
    R_xlen_t nnz = 0;
    for (R_xlen_t j = 0; j < m; j++) {
        if (len[j] < 0) {
            Rf_error("ld column length must be non-negative");
        }
        nnz += len[j];
    }
    const size_t bpv = (size_t)bits / 8;
    const size_t off_cum = RFM_LD_HDR_BYTES;
    const size_t off_lo = off_cum + (size_t)(m + 1) * 8;
    const size_t off_val = off_lo + (size_t)m * 8;
    const size_t total = off_val + (size_t)nnz * bpv;

    fm_vector *vec = allocate_fm_vector(runtime, RAWSXP, (R_xlen_t)total, true, false);
    SEXP ans = PROTECT(fmalloc_new_altrep(vec));
    uint8_t *p = static_cast<uint8_t *>(vector_data_or_dummy(vec));

    rfm_ld_header hdr;
    memcpy(hdr.magic, rfm_ld_magic7, sizeof(rfm_ld_magic7));
    hdr.magic[7] = RFM_LD_RAW;
    hdr.n_variants = (uint64_t)m;
    hdr.nnz = (uint64_t)nnz;
    hdr.bits = (uint32_t)bits;
    hdr.window = window;
    hdr.reserved = 0;
    memcpy(p, &hdr, sizeof(hdr));

    uint64_t *cum = reinterpret_cast<uint64_t *>(p + off_cum);
    int64_t *lo_out = reinterpret_cast<int64_t *>(p + off_lo);
    uint64_t acc = 0;
    for (R_xlen_t j = 0; j < m; j++) {
        cum[j] = acc;
        lo_out[j] = (int64_t)lo[j];
        acc += (uint64_t)len[j];
    }
    cum[m] = acc;

    void *values = p + off_val;
    R_xlen_t k = 0;
    if (bits == 16) {
        int16_t *q = static_cast<int16_t *>(values);
        for (R_xlen_t j = 0; j < m; j++) {
            for (R_xlen_t t = 0; t < len[j]; t++, k++) {
                q[k] = (int16_t)rfm_ld_quant(rvals[k], 16);
            }
        }
    } else {
        int8_t *q = static_cast<int8_t *>(values);
        for (R_xlen_t j = 0; j < m; j++) {
            for (R_xlen_t t = 0; t < len[j]; t++, k++) {
                q[k] = (int8_t)rfm_ld_quant(rvals[k], 8);
            }
        }
    }

    vec->maybe_dirty = true;
    UNPROTECT(1);
    return ans;
}

//==============================================================================
// C-callable read API (registered with R_RegisterCCallable, declared in
// inst/include/Rfmalloc.h). These let a consumer package (RfmallocStatgen's
// snp_cor / LDpred2) read an ld store's per-column neighbour runs without
// knowing the private byte layout, and build one from computed bands.
//==============================================================================

// Number of variants (columns == rows) of the ld store, or -1 on a bad store.
extern "C" R_xlen_t Rfmalloc_ld_ncol(SEXP store)
{
    rfm_ld_view v;
    if (rfm_ld_open(store, &v) != 0) {
        return -1;
    }
    return v.m;
}

// Quantization width in bits (8 or 16), or -1 on a bad store.
extern "C" int Rfmalloc_ld_bits(SEXP store)
{
    rfm_ld_view v;
    if (rfm_ld_open(store, &v) != 0) {
        return -1;
    }
    return v.bits;
}

// r[i, j] (0-based indices). Returns 0.0 for an out-of-band or out-of-range
// pair. Because the full symmetric band is stored, column j alone answers.
extern "C" double Rfmalloc_ld_pair(SEXP store, R_xlen_t i, R_xlen_t j)
{
    rfm_ld_view v;
    if (rfm_ld_open(store, &v) != 0) {
        return 0.0;
    }
    if (j < 0 || j >= v.m || i < 0 || i >= v.m) {
        return 0.0;
    }
    const R_xlen_t lo = (R_xlen_t)v.lo[j];
    const R_xlen_t len = (R_xlen_t)(v.cum[j + 1] - v.cum[j]);
    if (i < lo || i >= lo + len) {
        return 0.0;
    }
    return rfm_ld_value_at(&v, (R_xlen_t)v.cum[j] + (i - lo));
}

// Zero-copy access to column j's neighbour run. On success sets *lo (band start
// row), *len (band length), *values (pointer to the raw int8/int16 codes for
// this column, i.e. &code[cum[j]]) and returns 0. Interpret a code c as
// c / (bits == 16 ? 32767 : 127); get bits from Rfmalloc_ld_bits(). Returns
// non-zero on a bad store or out-of-range column.
extern "C" int Rfmalloc_ld_col_raw(SEXP store, R_xlen_t j, R_xlen_t *lo,
                                   R_xlen_t *len, const void **values)
{
    rfm_ld_view v;
    if (rfm_ld_open(store, &v) != 0) {
        return -1;
    }
    if (j < 0 || j >= v.m) {
        return -1;
    }
    const R_xlen_t l = (R_xlen_t)v.lo[j];
    const R_xlen_t n = (R_xlen_t)(v.cum[j + 1] - v.cum[j]);
    if (lo) *lo = l;
    if (len) *len = n;
    if (values) {
        const size_t bpv = (size_t)v.bits / 8;
        *values = static_cast<const uint8_t *>(v.values) + (size_t)v.cum[j] * bpv;
    }
    return 0;
}

// Decode column j's neighbour run into out[0..len) as doubles, returning the
// band start row in *lo and the length in *len. out must hold at least len
// doubles (query the length first with *len via Rfmalloc_ld_col_raw, or size it
// to Rfmalloc_ld_ncol()). Returns 0 on success.
extern "C" int Rfmalloc_ld_col(SEXP store, R_xlen_t j, R_xlen_t *lo,
                               R_xlen_t *len, double *out)
{
    rfm_ld_view v;
    if (rfm_ld_open(store, &v) != 0) {
        return -1;
    }
    if (j < 0 || j >= v.m) {
        return -1;
    }
    const R_xlen_t l = (R_xlen_t)v.lo[j];
    const R_xlen_t base = (R_xlen_t)v.cum[j];
    const R_xlen_t n = (R_xlen_t)(v.cum[j + 1] - v.cum[j]);
    if (lo) *lo = l;
    if (len) *len = n;
    if (out) {
        for (R_xlen_t t = 0; t < n; t++) {
            out[t] = rfm_ld_value_at(&v, base + t);
        }
    }
    return 0;
}

// Build an ld store from computed per-column bands (used by snp_cor). runtime is
// an fmalloc runtime handle SEXP; lo[j]/len[j] are 0-based band descriptors and
// rvals is the column-major concatenation of the band correlations (length
// sum(len)). Returns the ld payload as an fmalloc ALTREP raw SEXP; the caller
// must PROTECT it immediately.
extern "C" SEXP Rfmalloc_ld_build(SEXP runtime, R_xlen_t n_variants, int bits,
                                  int window, const R_xlen_t *lo,
                                  const R_xlen_t *len, const double *rvals)
{
    fm_runtime *rt = runtime_from_xptr(runtime);
    if (!rt || !rt->info) {
        Rf_error("fmalloc runtime is closed");
    }
    if (n_variants < 0) {
        Rf_error("n_variants must be non-negative");
    }
    uint32_t win = window < 0 ? 0u : (uint32_t)window;
    return rfm_ld_build_payload(rt, n_variants, bits, win, lo, len, rvals);
}

// Allocate an ld store whose band geometry is final and whose correlations are
// all zero, for a producer that fills it one column at a time.
extern "C" SEXP Rfmalloc_ld_alloc(SEXP runtime, R_xlen_t n_variants, int bits,
                                  int window, const R_xlen_t *lo,
                                  const R_xlen_t *len)
{
    fm_runtime *rt = runtime_from_xptr(runtime);
    if (!rt || !rt->info) {
        Rf_error("fmalloc runtime is closed");
    }
    if (n_variants < 0) {
        Rf_error("n_variants must be non-negative");
    }
    uint32_t win = window < 0 ? 0u : (uint32_t)window;
    return rfm_ld_alloc_payload(rt, n_variants, bits, win, lo, len);
}

// Quantize and store `count` correlations into column j's band, starting
// `first` entries into it. Returns 0, or -1 for an invalid store or range.
extern "C" int Rfmalloc_ld_write(SEXP store, R_xlen_t j, R_xlen_t first,
                                 R_xlen_t count, const double *values)
{
    return rfm_ld_write_column(store, j, first, count, values);
}

//==============================================================================
// R-facing .Call entry points.
//==============================================================================

// Encode a banded correlation matrix given as COO triplets (i, j, x), all
// 1-based, into an ld payload. Assembles a contiguous band per column: for
// column j the band spans [min index seen .. max index seen] (always including
// j), missing interior entries are 0, and the diagonal is forced to 1.
extern "C" SEXP rfm_ld_encode_triplets_impl(SEXP i_sexp, SEXP j_sexp, SEXP x_sexp,
                                            SEXP n_sexp, SEXP bits_sexp,
                                            SEXP window_sexp, SEXP runtime_xptr)
{
    fm_runtime *runtime = runtime_from_xptr(runtime_xptr);
    if (!runtime || !runtime->info) {
        Rf_error("fmalloc runtime is closed");
    }
    if (TYPEOF(i_sexp) != INTSXP || TYPEOF(j_sexp) != INTSXP) {
        Rf_error("i and j must be integer vectors of 1-based indices");
    }
    if (TYPEOF(x_sexp) != REALSXP) {
        Rf_error("x must be a numeric vector of correlations");
    }
    const R_xlen_t nnz_in = XLENGTH(i_sexp);
    if (XLENGTH(j_sexp) != nnz_in || XLENGTH(x_sexp) != nnz_in) {
        Rf_error("i, j and x must have the same length");
    }
    const R_xlen_t m = (R_xlen_t)Rf_asInteger(n_sexp);
    const int bits = Rf_asInteger(bits_sexp);
    if (m <= 0) {
        Rf_error("n_variants must be a positive integer");
    }
    if (bits != 8 && bits != 16) {
        Rf_error("bits must be 8 or 16");
    }
    const int window = (window_sexp == R_NilValue) ? 0 : Rf_asInteger(window_sexp);

    const int *ii = INTEGER(i_sexp);
    const int *jj = INTEGER(j_sexp);
    const double *xx = REAL(x_sexp);

    // Pass 1: per-column band extent [lo_j, hi_j], always covering the diagonal.
    std::vector<R_xlen_t> lo((size_t)m), hi((size_t)m);
    for (R_xlen_t j = 0; j < m; j++) {
        lo[(size_t)j] = j;
        hi[(size_t)j] = j;
    }
    for (R_xlen_t t = 0; t < nnz_in; t++) {
        const R_xlen_t ir = (R_xlen_t)ii[t] - 1;
        const R_xlen_t jr = (R_xlen_t)jj[t] - 1;
        if (ir < 0 || ir >= m || jr < 0 || jr >= m) {
            Rf_error("triplet index out of range [1, n_variants]");
        }
        if (ir < lo[(size_t)jr]) lo[(size_t)jr] = ir;
        if (ir > hi[(size_t)jr]) hi[(size_t)jr] = ir;
    }

    std::vector<R_xlen_t> len((size_t)m);
    R_xlen_t nnz = 0;
    for (R_xlen_t j = 0; j < m; j++) {
        len[(size_t)j] = hi[(size_t)j] - lo[(size_t)j] + 1;
        nnz += len[(size_t)j];
    }

    // Pass 2: fill the concatenated band values (0 default, diagonal 1, then
    // scatter the triplets).
    std::vector<double> rvals((size_t)nnz, 0.0);
    std::vector<R_xlen_t> base((size_t)m);
    R_xlen_t acc = 0;
    for (R_xlen_t j = 0; j < m; j++) {
        base[(size_t)j] = acc;
        rvals[(size_t)(acc + (j - lo[(size_t)j]))] = 1.0; // diagonal
        acc += len[(size_t)j];
    }
    for (R_xlen_t t = 0; t < nnz_in; t++) {
        const R_xlen_t ir = (R_xlen_t)ii[t] - 1;
        const R_xlen_t jr = (R_xlen_t)jj[t] - 1;
        rvals[(size_t)(base[(size_t)jr] + (ir - lo[(size_t)jr]))] = xx[t];
    }

    return rfm_ld_build_payload(runtime, m, bits, (uint32_t)(window < 0 ? 0 : window),
                                lo.data(), len.data(), rvals.data());
}

// list(n_variants, nnz, bits, window) describing an ld store.
extern "C" SEXP rfm_ld_info_impl(SEXP store)
{
    rfm_ld_view v;
    if (rfm_ld_open(store, &v) != 0) {
        Rf_error("not a valid fmalloc ld store");
    }
    SEXP ans = PROTECT(Rf_allocVector(VECSXP, 4));
    SET_VECTOR_ELT(ans, 0, Rf_ScalarReal((double)v.m));
    SET_VECTOR_ELT(ans, 1, Rf_ScalarReal((double)v.nnz));
    SET_VECTOR_ELT(ans, 2, Rf_ScalarInteger(v.bits));
    SET_VECTOR_ELT(ans, 3, Rf_ScalarInteger((int)v.window));
    SEXP nms = PROTECT(Rf_allocVector(STRSXP, 4));
    SET_STRING_ELT(nms, 0, Rf_mkChar("n_variants"));
    SET_STRING_ELT(nms, 1, Rf_mkChar("nnz"));
    SET_STRING_ELT(nms, 2, Rf_mkChar("bits"));
    SET_STRING_ELT(nms, 3, Rf_mkChar("window"));
    Rf_setAttrib(ans, R_NamesSymbol, nms);
    UNPROTECT(2);
    return ans;
}

// Number of variants (columns) of an ld store.
extern "C" SEXP rfm_ld_ncol_impl(SEXP store)
{
    R_xlen_t m = Rfmalloc_ld_ncol(store);
    if (m < 0) {
        Rf_error("not a valid fmalloc ld store");
    }
    return Rf_ScalarReal((double)m);
}

// r[i, j] for 1-based i, j.
extern "C" SEXP rfm_ld_pair_impl(SEXP store, SEXP i_sexp, SEXP j_sexp)
{
    rfm_ld_view v;
    if (rfm_ld_open(store, &v) != 0) {
        Rf_error("not a valid fmalloc ld store");
    }
    const R_xlen_t i = (R_xlen_t)Rf_asInteger(i_sexp) - 1;
    const R_xlen_t j = (R_xlen_t)Rf_asInteger(j_sexp) - 1;
    if (j < 0 || j >= v.m || i < 0 || i >= v.m) {
        Rf_error("i and j must be in [1, n_variants]");
    }
    return Rf_ScalarReal(Rfmalloc_ld_pair(store, i, j));
}

// Column j's neighbour run as list(lo, hi, x): lo/hi are 1-based inclusive row
// bounds of the band, x the decoded correlations for rows lo..hi.
extern "C" SEXP rfm_ld_col_impl(SEXP store, SEXP j_sexp)
{
    rfm_ld_view v;
    if (rfm_ld_open(store, &v) != 0) {
        Rf_error("not a valid fmalloc ld store");
    }
    const R_xlen_t j = (R_xlen_t)Rf_asInteger(j_sexp) - 1;
    if (j < 0 || j >= v.m) {
        Rf_error("j must be in [1, n_variants]");
    }
    const R_xlen_t lo = (R_xlen_t)v.lo[j];
    const R_xlen_t base = (R_xlen_t)v.cum[j];
    const R_xlen_t len = (R_xlen_t)(v.cum[j + 1] - v.cum[j]);

    SEXP x = PROTECT(Rf_allocVector(REALSXP, len));
    double *xp = REAL(x);
    for (R_xlen_t t = 0; t < len; t++) {
        xp[t] = rfm_ld_value_at(&v, base + t);
    }
    SEXP ans = PROTECT(Rf_allocVector(VECSXP, 3));
    SET_VECTOR_ELT(ans, 0, Rf_ScalarReal((double)(lo + 1)));
    SET_VECTOR_ELT(ans, 1, Rf_ScalarReal((double)(lo + len)));
    SET_VECTOR_ELT(ans, 2, x);
    SEXP nms = PROTECT(Rf_allocVector(STRSXP, 3));
    SET_STRING_ELT(nms, 0, Rf_mkChar("lo"));
    SET_STRING_ELT(nms, 1, Rf_mkChar("hi"));
    SET_STRING_ELT(nms, 2, Rf_mkChar("x"));
    Rf_setAttrib(ans, R_NamesSymbol, nms);
    UNPROTECT(3);
    return ans;
}
