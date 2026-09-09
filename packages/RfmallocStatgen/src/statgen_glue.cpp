/*
 * R interface for RfmallocStatgen's native code. Uses R's C API directly (no
 * Rcpp) and includes only pcaone_backend.h - never the PCAone headers - so R's
 * macro remaps and PCAone's C++ never share a translation unit. GPL-3.
 */
#define R_NO_REMAP
#define R_NO_REMAP_RMATH
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Random.h>  // GetRNGstate/PutRNGstate, unif_rand, norm_rand
#include <Rmath.h>         // Rf_rbeta (Beta deviate for LDpred2-auto)
#include <Rfmalloc.h>  // Rfmalloc_tensor_decode + fmalloc_ld accessors

#include <cstddef>
#include <cstring>
#include <cmath>
#include <exception>
#include <vector>

#include "pcaone_backend.h"

// Build the list(d, u, v) result shared by the in-core and out-of-core paths.
static SEXP build_result(const rfmstatgen::RsvdResult& r) {
  SEXP d = PROTECT(Rf_allocVector(REALSXP, r.k));
  if (r.k > 0) std::memcpy(REAL(d), r.d.data(), sizeof(double) * (size_t)r.k);
  SEXP u = PROTECT(Rf_allocMatrix(REALSXP, r.n, r.k));
  if (r.k > 0) std::memcpy(REAL(u), r.u.data(), sizeof(double) * (size_t)r.n * (size_t)r.k);
  SEXP v = PROTECT(Rf_allocMatrix(REALSXP, r.m, r.k));
  if (r.k > 0) std::memcpy(REAL(v), r.v.data(), sizeof(double) * (size_t)r.m * (size_t)r.k);

  SEXP out = PROTECT(Rf_allocVector(VECSXP, 3));
  SET_VECTOR_ELT(out, 0, d);
  SET_VECTOR_ELT(out, 1, u);
  SET_VECTOR_ELT(out, 2, v);
  SEXP nms = PROTECT(Rf_allocVector(STRSXP, 3));
  SET_STRING_ELT(nms, 0, Rf_mkChar("d"));
  SET_STRING_ELT(nms, 1, Rf_mkChar("u"));
  SET_STRING_ELT(nms, 2, Rf_mkChar("v"));
  Rf_setAttrib(out, R_NamesSymbol, nms);

  UNPROTECT(5);
  return out;
}

extern "C" SEXP C_statgen_pca_incore(SEXP Xs, SEXP ks, SEXP ps, SEXP ss,
                                     SEXP methods, SEXP tols, SEXP seeds) {
  if (!Rf_isMatrix(Xs) || TYPEOF(Xs) != REALSXP) {
    Rf_error("X must be a numeric matrix");
  }
  const int n = Rf_nrows(Xs);
  const int m = Rf_ncols(Xs);
  const int k = Rf_asInteger(ks);
  const int p = Rf_asInteger(ps);
  const int s = Rf_asInteger(ss);
  const int method = Rf_asInteger(methods);
  const double tol = Rf_asReal(tols);
  const int seed = Rf_asInteger(seeds);

  rfmstatgen::RsvdResult r;
  try {
    r = rfmstatgen::rsvd_incore(REAL(Xs), n, m, k, p, s, method, tol, seed);
  } catch (const std::exception& e) {
    Rf_error("statgen_pca: %s", e.what());
  } catch (...) {
    Rf_error("statgen_pca: unknown C++ exception in RSVD backend");
  }
  return build_result(r);
}

// Adapter matching rfmstatgen::decode_range_fn: forward one variant-column
// range of the tensor to Rfmalloc's registered codec decode (C-callable). Kept
// on the R side so the PCAone backend stays free of R headers.
static int glue_decode_tensor(void* tensor, long long elem_offset,
                              long long n_elems, double* out) {
  return Rfmalloc_tensor_decode((SEXP)tensor, (R_xlen_t)elem_offset,
                                (R_xlen_t)n_elems, out);
}

extern "C" SEXP C_statgen_pca_ooc(SEXP tensor, SEXP ns, SEXP ms, SEXP ks,
                                  SEXP ps, SEXP ss, SEXP tols, SEXP seeds,
                                  SEXP block_sizes) {
  const int n = Rf_asInteger(ns);
  const int m = Rf_asInteger(ms);
  const int k = Rf_asInteger(ks);
  const int p = Rf_asInteger(ps);
  const int s = Rf_asInteger(ss);
  const double tol = Rf_asReal(tols);
  const int seed = Rf_asInteger(seeds);
  const int block_size = Rf_asInteger(block_sizes);

  rfmstatgen::RsvdResult r;
  try {
    r = rfmstatgen::rsvd_ooc((void*)tensor, glue_decode_tensor, n, m, k, p, s,
                             tol, seed, block_size);
  } catch (const std::exception& e) {
    Rf_error("statgen_pca (out-of-core): %s", e.what());
  } catch (...) {
    Rf_error("statgen_pca (out-of-core): unknown C++ exception in RSVD backend");
  }
  return build_result(r);
}

// Decode a bounded variant-column panel into an ordinary numeric matrix. The
// caller owns the panel lifetime; the complete genotype tensor is never
// exposed as a double matrix. This adapter lets the R implementation use BLAS
// on each panel while the C-callable codec remains the single decode path.
extern "C" SEXP C_statgen_decode_panel(SEXP tensor, SEXP ns, SEXP starts,
                                       SEXP counts) {
  const int n = Rf_asInteger(ns);
  const int start = Rf_asInteger(starts);
  const int count = Rf_asInteger(counts);
  if (n < 1 || start < 0 || count < 1) {
    Rf_error("invalid genotype panel dimensions");
  }

  const R_xlen_t elem_offset = (R_xlen_t)start * (R_xlen_t)n;
  const R_xlen_t n_elems = (R_xlen_t)count * (R_xlen_t)n;
  if (n_elems / n != count) {
    Rf_error("genotype panel is too large");
  }

  SEXP out = PROTECT(Rf_allocMatrix(REALSXP, n, count));
  if (Rfmalloc_tensor_decode(tensor, elem_offset, n_elems, REAL(out)) != 0) {
    UNPROTECT(1);
    Rf_error("failed to decode genotype columns [%d, %d)", start,
             start + count);
  }
  UNPROTECT(1);
  return out;
}

// ---------------------------------------------------------------------------
// statgen_snp_cor: stream a genotype tensor one variant-column at a time via
// Rfmalloc_tensor_decode, compute the windowed Pearson correlation between
// nearby variants (mean-impute + centre + unit-L2-normalize each column, then
// r = dot of the two unit columns), and pack the banded result into an
// Rfmalloc "ld" store (Rfmalloc_ld_build). The dense p x p is never formed and
// only a sliding window of unit columns (bounded by the window width, not m) is
// held resident. Storage-agnostic result: the caller reads it with
// ld_col()/ld_pair() or hands it to LDpred2.
// ---------------------------------------------------------------------------
extern "C" SEXP C_statgen_snp_cor(SEXP tensor, SEXP ns, SEXP ms, SEXP sizes,
                                  SEXP thr_r2s, SEXP infos_pos, SEXP bitss) {
  const R_xlen_t n = (R_xlen_t)Rf_asInteger(ns);
  const R_xlen_t m = (R_xlen_t)Rf_asInteger(ms);
  const double thr_r2 = Rf_asReal(thr_r2s);
  const int bits = Rf_asInteger(bitss);
  if (n <= 0 || m <= 0) Rf_error("genotype tensor must be non-empty");
  if (bits != 8 && bits != 16) Rf_error("bits must be 8 or 16");

  const bool has_pos = (infos_pos != R_NilValue);
  const double* pos = has_pos ? REAL(infos_pos) : nullptr;
  // bigsnpr semantics: with infos.pos the window is 'size' kb -> size * 1000 in
  // position units; without it, 'size' is a half-width in number of variants.
  const double W = has_pos ? (Rf_asReal(sizes) * 1000.0) : 0.0;
  const R_xlen_t size_idx = has_pos ? 0 : (R_xlen_t)Rf_asInteger(sizes);
  if (has_pos && (R_xlen_t)Rf_xlength(infos_pos) != m)
    Rf_error("infos_pos must have length equal to the number of variants");

  // Per-column contiguous band [lo_j, hi_j], 0-based. The store owns the
  // cumulative offsets; only the geometry is needed here.
  std::vector<R_xlen_t> lo((size_t)m), hi((size_t)m), len((size_t)m);
  {
    R_xlen_t a = 0, b = 0;  // two pointers for the position-based window
    for (R_xlen_t j = 0; j < m; j++) {
      R_xlen_t loj, hij;
      if (has_pos) {
        if (a > j) a = j;
        while (a < j && pos[a] < pos[j] - W) a++;
        if (b < j) b = j;
        while (b + 1 < m && pos[b + 1] <= pos[j] + W) b++;
        loj = a;
        hij = b;
      } else {
        loj = (j > size_idx) ? (j - size_idx) : 0;
        hij = (j + size_idx < m - 1) ? (j + size_idx) : (m - 1);
      }
      lo[(size_t)j] = loj;
      hi[(size_t)j] = hij;
      len[(size_t)j] = hij - loj + 1;
    }
  }
  // The band is written into the store one column at a time, so what stays
  // resident is the columns still in flight, not the whole band: column j is
  // complete once j' reaches hi[j], and hi is non-decreasing.
  SEXP runtime = Rfmalloc_runtime_of_vector(tensor);
  const int window = has_pos ? (int)W : (int)size_idx;
  SEXP store = PROTECT(Rfmalloc_ld_alloc(runtime, m, bits, window, lo.data(),
                                         len.data()));
  std::vector<std::vector<double> > band((size_t)m);   // in-flight bands
  std::vector<std::vector<double> > cols((size_t)m);   // sliding unit columns
  std::vector<double> buf((size_t)n);
  R_xlen_t cleared = 0;  // columns < cleared have been evicted
  R_xlen_t written = 0;  // columns < written have been stored

  for (R_xlen_t j = 0; j < m; j++) {
    band[(size_t)j].assign((size_t)len[(size_t)j], 0.0);
    // Evict columns that no future j (>= this one) can neighbour: lo is
    // non-decreasing, so anything below lo[j] is done.
    for (; cleared < lo[(size_t)j]; cleared++) {
      std::vector<double>().swap(cols[(size_t)cleared]);
    }

    if (Rfmalloc_tensor_decode(tensor, j * n, n, buf.data()) != 0) {
      Rf_error("statgen_snp_cor: failed to decode variant column %lld",
               (long long)j);
    }

    // mean over observed, then impute-to-mean + centre + unit-L2 normalize.
    double s = 0.0;
    R_xlen_t cnt = 0;
    for (R_xlen_t i = 0; i < n; i++) {
      const double v = buf[(size_t)i];
      if (!ISNAN(v)) { s += v; cnt++; }
    }
    const double mean = (cnt > 0) ? (s / (double)cnt) : 0.0;
    std::vector<double> u((size_t)n);
    double ss = 0.0;
    for (R_xlen_t i = 0; i < n; i++) {
      const double v = buf[(size_t)i];
      const double c = ISNAN(v) ? 0.0 : (v - mean);  // missing -> mean -> 0
      u[(size_t)i] = c;
      ss += c * c;
    }
    const double norm = std::sqrt(ss);
    if (norm > 0.0) {
      for (R_xlen_t i = 0; i < n; i++) u[(size_t)i] /= norm;
    }  // else monomorphic: unit column stays 0, so all its off-diagonals are 0
    cols[(size_t)j].swap(u);

    // diagonal r = 1
    band[(size_t)j][(size_t)(j - lo[(size_t)j])] = 1.0;

    // correlations with the already-decoded in-window neighbours below j; store
    // both (j,k) and (k,j) since the full symmetric band is kept.
    const double* uj = cols[(size_t)j].data();
    for (R_xlen_t k = lo[(size_t)j]; k < j; k++) {
      const std::vector<double>& ck = cols[(size_t)k];
      if (ck.empty()) continue;  // defensive; k >= lo[j] means not yet evicted
      const double* uk = ck.data();
      double r = 0.0;
      for (R_xlen_t i = 0; i < n; i++) r += uj[(size_t)i] * uk[(size_t)i];
      if (thr_r2 > 0.0 && r * r < thr_r2) r = 0.0;
      band[(size_t)k][(size_t)(j - lo[(size_t)k])] = r;  // (row j, col k)
      band[(size_t)j][(size_t)(k - lo[(size_t)j])] = r;  // (row k, col j)
    }

    // Flush every column no later step can still touch.
    for (; written <= j && hi[(size_t)written] <= j; written++) {
      const std::vector<double>& b = band[(size_t)written];
      if (Rfmalloc_ld_write(store, written, 0, (R_xlen_t)b.size(), b.data()) != 0) {
        UNPROTECT(1);
        Rf_error("statgen_snp_cor: failed to store ld column %lld",
                 (long long)written);
      }
      std::vector<double>().swap(band[(size_t)written]);
    }

    if ((j & 0x3FF) == 0) R_CheckUserInterrupt();
  }

  for (; written < m; written++) {
    const std::vector<double>& b = band[(size_t)written];
    if (Rfmalloc_ld_write(store, written, 0, (R_xlen_t)b.size(), b.data()) != 0) {
      UNPROTECT(1);
      Rf_error("statgen_snp_cor: failed to store ld column %lld",
               (long long)written);
    }
    std::vector<double>().swap(band[(size_t)written]);
  }

  UNPROTECT(1);
  return store;
}

// ---------------------------------------------------------------------------
// LDpred2 (Prive, Arbel & Vilhjalmsson 2020, Bioinformatics), reimplemented
// clean-room over the banded fmalloc_ld store. All three variants read the LD
// matrix one column's contiguous neighbour run at a time via the fmalloc_ld
// C-callables (Rfmalloc_ld_col_raw): never a dense p x p, never the genotypes.
// ---------------------------------------------------------------------------

namespace {

// A read-once, zero-copy view of an ld store: per-column band start (lo), band
// length (len) and a pointer to the raw int8/int16 codes (into the mmap'd
// payload). Decode a code c as c / (bits == 16 ? 32767 : 127).
struct LdView {
  R_xlen_t m = 0;
  int bits = 8;
  std::vector<R_xlen_t> lo, len;
  std::vector<const void*> vals;
};

void ld_load(SEXP corr, LdView* v) {
  const R_xlen_t m = Rfmalloc_ld_ncol(corr);
  if (m < 0) Rf_error("corr must be an fmalloc_ld store");
  v->m = m;
  v->bits = Rfmalloc_ld_bits(corr);
  v->lo.resize((size_t)m);
  v->len.resize((size_t)m);
  v->vals.resize((size_t)m);
  for (R_xlen_t j = 0; j < m; j++) {
    if (Rfmalloc_ld_col_raw(corr, j, &v->lo[(size_t)j], &v->len[(size_t)j],
                            &v->vals[(size_t)j]) != 0) {
      Rf_error("failed to read ld column %lld", (long long)j);
    }
  }
}

// out += C[:, j] * a  (the full symmetric band of column j, diagonal included).
inline void ld_axpy_col(const LdView& v, R_xlen_t j, double a, double* out) {
  const R_xlen_t l = v.lo[(size_t)j];
  const R_xlen_t n = v.len[(size_t)j];
  if (v.bits == 16) {
    const int16_t* p = static_cast<const int16_t*>(v.vals[(size_t)j]);
    for (R_xlen_t t = 0; t < n; t++) out[l + t] += (double)p[t] / 32767.0 * a;
  } else {
    const int8_t* p = static_cast<const int8_t*>(v.vals[(size_t)j]);
    for (R_xlen_t t = 0; t < n; t++) out[l + t] += (double)p[t] / 127.0 * a;
  }
}

// out = C x  (banded matrix-vector product).
void ld_matvec(const LdView& v, const double* x, double* out) {
  std::fill(out, out + v.m, 0.0);
  for (R_xlen_t j = 0; j < v.m; j++) {
    const double xj = x[(size_t)j];
    if (xj != 0.0) ld_axpy_col(v, j, xj, out);
  }
}

// Mean LD score = mean over columns of sum_k C[k, j]^2 (diagonal 1 included).
double ld_mean_ldscore(const LdView& v) {
  if (v.m == 0) return 1.0;
  double total = 0.0;
  for (R_xlen_t j = 0; j < v.m; j++) {
    const R_xlen_t n = v.len[(size_t)j];
    double s = 0.0;
    if (v.bits == 16) {
      const int16_t* p = static_cast<const int16_t*>(v.vals[(size_t)j]);
      for (R_xlen_t t = 0; t < n; t++) { double r = (double)p[t] / 32767.0; s += r * r; }
    } else {
      const int8_t* p = static_cast<const int8_t*>(v.vals[(size_t)j]);
      for (R_xlen_t t = 0; t < n; t++) { double r = (double)p[t] / 127.0; s += r * r; }
    }
    total += s;
  }
  return total / (double)v.m;
}

inline double vdot(const double* a, const double* b, R_xlen_t n) {
  double s = 0.0;
  for (R_xlen_t i = 0; i < n; i++) s += a[(size_t)i] * b[(size_t)i];
  return s;
}

}  // namespace

// snp_ldpred2_inf: solve the ridge-like system (C + diag(d)) x = beta_hat by
// conjugate gradient over the banded C, where d[i] = m / (h2 * n_eff[i]). C is
// SPD after the ridge shift, so CG converges to the unique solution; the caller
// rescales x by 'scale' to the allele scale.
extern "C" SEXP C_statgen_ldpred2_inf(SEXP corr, SEXP beta_hats, SEXP ds,
                                      SEXP tols, SEXP maxits) {
  LdView v;
  ld_load(corr, &v);
  const R_xlen_t m = v.m;
  if ((R_xlen_t)Rf_xlength(beta_hats) != m || (R_xlen_t)Rf_xlength(ds) != m) {
    Rf_error("beta_hat and add_to_diag must have length ncol(corr)");
  }
  const double* b = REAL(beta_hats);
  const double* d = REAL(ds);
  const double tol = Rf_asReal(tols);
  const int maxit = Rf_asInteger(maxits);

  std::vector<double> x((size_t)m, 0.0), r(b, b + m), p(b, b + m);
  std::vector<double> Ap((size_t)m, 0.0);

  double bnorm = std::sqrt(vdot(b, b, m));
  SEXP ans = PROTECT(Rf_allocVector(REALSXP, m));
  if (bnorm == 0.0) {
    std::memset(REAL(ans), 0, sizeof(double) * (size_t)m);
    UNPROTECT(1);
    return ans;
  }

  double rs = vdot(r.data(), r.data(), m);
  for (int it = 0; it < maxit; it++) {
    ld_matvec(v, p.data(), Ap.data());               // Ap = C p
    for (R_xlen_t i = 0; i < m; i++) Ap[(size_t)i] += d[(size_t)i] * p[(size_t)i];  // + diag(d) p
    const double pAp = vdot(p.data(), Ap.data(), m);
    if (!(pAp > 0.0)) break;                          // guard against non-SPD
    const double alpha = rs / pAp;
    for (R_xlen_t i = 0; i < m; i++) {
      x[(size_t)i] += alpha * p[(size_t)i];
      r[(size_t)i] -= alpha * Ap[(size_t)i];
    }
    const double rs_new = vdot(r.data(), r.data(), m);
    if (std::sqrt(rs_new) <= tol * bnorm) break;
    const double bcg = rs_new / rs;
    for (R_xlen_t i = 0; i < m; i++) p[(size_t)i] = r[(size_t)i] + bcg * p[(size_t)i];
    rs = rs_new;
  }

  std::memcpy(REAL(ans), x.data(), sizeof(double) * (size_t)m);
  UNPROTECT(1);
  return ans;
}

// snp_ldpred2_grid: one LDpred2-grid Gibbs model (fixed p, h2). Spike-and-slab
// prior: causal effects ~ N(0, h2/(m*p)). Returns the Rao-Blackwellized
// posterior-mean betas (unscaled), or a vector of NA on strong divergence.
extern "C" SEXP C_statgen_ldpred2_grid(SEXP corr, SEXP beta_hats, SEXP n_effs,
                                       SEXP h2s, SEXP ps, SEXP sparses,
                                       SEXP burn_ins, SEXP num_iters) {
  LdView v;
  ld_load(corr, &v);
  const R_xlen_t m = v.m;
  if ((R_xlen_t)Rf_xlength(beta_hats) != m || (R_xlen_t)Rf_xlength(n_effs) != m) {
    Rf_error("beta_hat and n_eff must have length ncol(corr)");
  }
  const double* beta_hat = REAL(beta_hats);
  const double* N = REAL(n_effs);
  const double h2 = Rf_asReal(h2s);
  const double p = Rf_asReal(ps);
  const bool sparse = (Rf_asLogical(sparses) == TRUE);
  const int burn_in = Rf_asInteger(burn_ins);
  const int num_iter = Rf_asInteger(num_iters);

  std::vector<double> curr_beta((size_t)m, 0.0), avg_beta((size_t)m, 0.0);
  std::vector<double> dotprods((size_t)m, 0.0);  // = C %*% curr_beta

  const double h2_per_var = h2 / ((double)m * p);
  const double inv_odd_p = (1.0 - p) / p;
  double gap0 = 2.0 * vdot(beta_hat, beta_hat, m);

  SEXP ans = PROTECT(Rf_allocVector(REALSXP, m));

  GetRNGstate();
  bool diverged = false;
  for (int k = -burn_in; k < num_iter && !diverged; k++) {
    double gap = 0.0;
    for (R_xlen_t j = 0; j < m; j++) {
      const double res_beta_hat_j =
          beta_hat[(size_t)j] - (dotprods[(size_t)j] - curr_beta[(size_t)j]);
      const double C1 = h2_per_var * N[(size_t)j];
      const double C2 = 1.0 / (1.0 + 1.0 / C1);
      const double C3 = C2 * res_beta_hat_j;
      const double C4 = C2 / N[(size_t)j];
      const double post_p_j =
          1.0 / (1.0 + inv_odd_p * std::sqrt(1.0 + C1) * std::exp(-C3 * C3 / C4 / 2.0));

      double diff = -curr_beta[(size_t)j];
      if (sparse && (post_p_j < p)) {
        curr_beta[(size_t)j] = 0.0;
      } else {
        if (post_p_j > unif_rand()) {
          curr_beta[(size_t)j] = C3 + std::sqrt(C4) * norm_rand();
          diff += curr_beta[(size_t)j];
          gap += curr_beta[(size_t)j] * curr_beta[(size_t)j];
        } else {
          curr_beta[(size_t)j] = 0.0;
        }
        if (k >= 0) avg_beta[(size_t)j] += C3 * post_p_j;
      }
      if (diff != 0.0) ld_axpy_col(v, j, diff, dotprods.data());
    }
    if (gap > gap0) diverged = true;
    if ((k & 0x3F) == 0) R_CheckUserInterrupt();
  }
  PutRNGstate();

  double* out = REAL(ans);
  if (diverged) {
    for (R_xlen_t i = 0; i < m; i++) out[(size_t)i] = NA_REAL;
  } else {
    for (R_xlen_t i = 0; i < m; i++) out[(size_t)i] = avg_beta[(size_t)i] / num_iter;
  }
  UNPROTECT(1);
  return ans;
}

// snp_ldpred2_auto: LDpred2-auto Gibbs sampler that estimates p and h2 from the
// data (the alpha = -1, sigma2 = h2/(m*p) variant, i.e. use_MLE = FALSE of
// bigsnpr's auto). Returns list(beta_est [unscaled], postp_est, path_p, path_h2,
// p_est, h2_est). beta_est is NA on strong divergence.
extern "C" SEXP C_statgen_ldpred2_auto(SEXP corr, SEXP beta_hats, SEXP n_effs,
                                       SEXP p_inits, SEXP h2_inits, SEXP burn_ins,
                                       SEXP num_iters, SEXP p_bounds) {
  LdView v;
  ld_load(corr, &v);
  const R_xlen_t m = v.m;
  if ((R_xlen_t)Rf_xlength(beta_hats) != m || (R_xlen_t)Rf_xlength(n_effs) != m) {
    Rf_error("beta_hat and n_eff must have length ncol(corr)");
  }
  const double* beta_hat = REAL(beta_hats);
  const double* N = REAL(n_effs);
  const int burn_in = Rf_asInteger(burn_ins);
  const int num_iter = Rf_asInteger(num_iters);
  const double p_lo = REAL(p_bounds)[0], p_hi = REAL(p_bounds)[1];
  const double MIN_H2 = 1e-3;

  const double mean_ld = ld_mean_ldscore(v);

  std::vector<double> curr_beta((size_t)m, 0.0), dotprods((size_t)m, 0.0);
  std::vector<double> avg_beta((size_t)m, 0.0), avg_postp((size_t)m, 0.0);

  const int num_iter_tot = burn_in + num_iter;
  SEXP path_p = PROTECT(Rf_allocVector(REALSXP, num_iter_tot));
  SEXP path_h2 = PROTECT(Rf_allocVector(REALSXP, num_iter_tot));
  double* pp = REAL(path_p);
  double* ph = REAL(path_h2);

  double h2 = std::max(Rf_asReal(h2_inits), MIN_H2);
  double p = std::min(std::max(p_lo, Rf_asReal(p_inits)), p_hi);
  double sigma2 = h2 / ((double)m * p);
  double cur_h2_est = 0.0;  // running curr_beta' C curr_beta, updated on change
  const double gap0 = 2.0 * vdot(beta_hat, beta_hat, m);

  GetRNGstate();
  bool diverged = false;
  for (int k = 0; k < num_iter_tot; k++) {
    const double inv_odd_p = (1.0 - p) / p;
    double gap = 0.0;
    R_xlen_t nb_causal = 0;

    for (R_xlen_t j = 0; j < m; j++) {
      const double dotprod = dotprods[(size_t)j];
      const double res_beta_hat_j = beta_hat[(size_t)j] - (dotprod - curr_beta[(size_t)j]);
      const double C1 = sigma2 * N[(size_t)j];
      const double C2 = 1.0 / (1.0 + 1.0 / C1);
      const double C3 = C2 * res_beta_hat_j;
      const double C4 = C2 / N[(size_t)j];
      const double postp =
          1.0 / (1.0 + inv_odd_p * std::sqrt(1.0 + C1) * std::exp(-C3 * C3 / C4 / 2.0));

      const double prev_beta = curr_beta[(size_t)j];
      if (k >= burn_in) {
        avg_postp[(size_t)j] += postp;
        avg_beta[(size_t)j] += C3 * postp;
      }

      double diff = -prev_beta;
      if (postp > unif_rand()) {
        const double samp = C3 + std::sqrt(C4) * norm_rand();
        curr_beta[(size_t)j] = samp;
        diff += samp;
        nb_causal++;
        gap += samp * samp;
      } else {
        curr_beta[(size_t)j] = 0.0;
      }
      if (diff != 0.0) {
        cur_h2_est += diff * (2.0 * dotprod + diff);
        ld_axpy_col(v, j, diff, dotprods.data());
      }
    }

    if (gap > gap0) { diverged = true; break; }

    p = Rf_rbeta(1.0 + (double)nb_causal / mean_ld,
                 1.0 + (double)(m - nb_causal) / mean_ld);
    p = std::min(std::max(p_lo, p), p_hi);
    h2 = std::max(cur_h2_est, MIN_H2);
    sigma2 = h2 / ((double)m * p);
    pp[k] = p;
    ph[k] = h2;
    if ((k & 0x3F) == 0) R_CheckUserInterrupt();
  }
  PutRNGstate();

  SEXP beta_est = PROTECT(Rf_allocVector(REALSXP, m));
  SEXP postp_est = PROTECT(Rf_allocVector(REALSXP, m));
  double* be = REAL(beta_est);
  double* pe = REAL(postp_est);
  double p_est = NA_REAL, h2_est = NA_REAL;
  if (diverged) {
    for (R_xlen_t i = 0; i < m; i++) { be[(size_t)i] = NA_REAL; pe[(size_t)i] = NA_REAL; }
  } else {
    for (R_xlen_t i = 0; i < m; i++) {
      be[(size_t)i] = avg_beta[(size_t)i] / num_iter;
      pe[(size_t)i] = avg_postp[(size_t)i] / num_iter;
    }
    double sp = 0.0, sh = 0.0;
    for (int k = burn_in; k < num_iter_tot; k++) { sp += pp[k]; sh += ph[k]; }
    p_est = sp / num_iter;
    h2_est = sh / num_iter;
  }

  SEXP out = PROTECT(Rf_allocVector(VECSXP, 6));
  SET_VECTOR_ELT(out, 0, beta_est);
  SET_VECTOR_ELT(out, 1, postp_est);
  SET_VECTOR_ELT(out, 2, path_p);
  SET_VECTOR_ELT(out, 3, path_h2);
  SET_VECTOR_ELT(out, 4, Rf_ScalarReal(p_est));
  SET_VECTOR_ELT(out, 5, Rf_ScalarReal(h2_est));
  SEXP nms = PROTECT(Rf_allocVector(STRSXP, 6));
  SET_STRING_ELT(nms, 0, Rf_mkChar("beta_est"));
  SET_STRING_ELT(nms, 1, Rf_mkChar("postp_est"));
  SET_STRING_ELT(nms, 2, Rf_mkChar("path_p_est"));
  SET_STRING_ELT(nms, 3, Rf_mkChar("path_h2_est"));
  SET_STRING_ELT(nms, 4, Rf_mkChar("p_est"));
  SET_STRING_ELT(nms, 5, Rf_mkChar("h2_est"));
  Rf_setAttrib(out, R_NamesSymbol, nms);
  UNPROTECT(6);  // path_p, path_h2, beta_est, postp_est, out, nms
  return out;
}

static const R_CallMethodDef CallEntries[] = {
    {"C_statgen_pca_incore", (DL_FUNC)&C_statgen_pca_incore, 7},
    {"C_statgen_pca_ooc", (DL_FUNC)&C_statgen_pca_ooc, 9},
    {"C_statgen_decode_panel", (DL_FUNC)&C_statgen_decode_panel, 4},
    {"C_statgen_snp_cor", (DL_FUNC)&C_statgen_snp_cor, 7},
    {"C_statgen_ldpred2_inf", (DL_FUNC)&C_statgen_ldpred2_inf, 5},
    {"C_statgen_ldpred2_grid", (DL_FUNC)&C_statgen_ldpred2_grid, 8},
    {"C_statgen_ldpred2_auto", (DL_FUNC)&C_statgen_ldpred2_auto, 8},
    {NULL, NULL, 0}};

extern "C" void R_init_RfmallocStatgen(DllInfo* dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
