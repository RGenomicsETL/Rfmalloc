# RfmallocStatgen 0.1.0 (unreleased)

- `statgen_snp_cor()` writes the banded LD into its store one column at a time
  instead of assembling the whole band in double precision first. Only the
  columns still in flight are resident, which is bounded by the window rather
  than by the number of variants: building a 100,000-variant band at window
  +/-500 moved peak memory by 711 MB before and by nothing now, at the same
  speed and with identical output.

- Named each sibling package explicitly in `Remotes`, so dependency installers
  distinguish monorepo subdirectories which share one repository commit.

- The pinned PCAone logger now initializes its screen state deterministically
  and sends optional output through R's console instead of `std::cout` or
  `std::cerr`. The adaptation is part of the byte-checked vendoring recipe.

- `RfmallocStatgen` is the statistical-genetics layer over Rfmalloc genotype,
  haplotype, and LD stores. Its current surface covers streamed regression and
  PCA, banded LD, LDpred2, and matrix-form colocalisation; `ROADMAP.md` tracks
  the remaining research directions.
- `statgen_gwas_lin()`: univariate linear-regression GWAS over a `bed` or
  `dosage` fmalloc genotype tensor, the fused whole-genome form (residualize
  once on the covariate design, then reduce per variant), not a loop of
  per-variant `lm()` calls. Missing genotype calls are mean-imputed per
  variant. `inst/tinytest/test_gwas_lin.R` checks `beta`/`se`/`t`/`p` for
  every variant against `lm(y ~ covar + g_j)` to floating-point tolerance,
  and cross-checks against `bigstatsr::big_univLinReg()` when `bigstatsr` is
  installed.
- `statgen_pca()`: fast principal-component analysis by randomized SVD. The
  RSVD workhorse is pinned from GPL-3 'PCAone', with deterministic dependency
  trimming and R-console adaptation
  (`src/pcaone/`, `RsvdOpData::computeUSV`), not reimplemented, behind the
  pinned drift-guard recipe `tools/vendor-pcaone/vendorpcaone.R` (CI job
  `vendored-pcaone-matches-recipe`). This makes the package compiled for the
  first time (`LinkingTo: RcppEigen`, C++17). Stage 1 drives the vendored
  core over an in-memory matrix; `inst/tinytest/test_pca.R` matches
  `base::svd()`, `stats::prcomp()`, and `pcaone::pcaone()` to ~1e-6 (in
  practice machine precision on a decaying spectrum).
- `statgen_pca()` now also takes a 2-D `bed`/`dosage` fmalloc genotype tensor
  and decomposes it **out of core**: the tensor is streamed one variant-column
  block at a time through Rfmalloc's fused-standardize decode (the new
  `Rfmalloc_tensor_decode` C-callable, API v7) and the dense samples x variants
  matrix is never materialized. To get a standardized PCA, standardize the
  tensor first (`fmalloc_bed_standardize()` / `fmalloc_dosage_standardize()`) -
  standardization rides the decode. `inst/tinytest/test_pca_ooc.R` checks that
  the out-of-core result equals the in-core path on the same standardized
  tensor to floating point, independent of the streaming block size, for both
  `bed` and `dosage`.
- `statgen_snp_cor()`: windowed LD (correlation) matrix from a `bed`/`dosage`
  fmalloc genotype tensor, packed into a banded `Rfmalloc::fmalloc_ld` store.
  The genotype tensor is streamed one variant column at a time through
  `Rfmalloc_tensor_decode` (the dense samples x variants matrix is never
  materialized, only a sliding window of unit columns bounded by the window
  width); each column is mean-imputed, centred and unit-L2-normalized, and the
  correlation of two nearby variants is the dot product of their unit columns -
  the fused form `bigsnpr::snp_cor()` uses. The window is `size` variants each
  side by index, or `size` kb of physical distance when `infos_pos` is given
  (bigsnpr's convention). `inst/tinytest/test_snp_cor.R` checks the banded
  result equals `stats::cor()` on the decoded matrix within the int8/int16
  quantization tolerance (max |diff| ~3.9e-3 at int8), the position-based
  window and `thr_r2` behaviour, and - guarded by `requireNamespace` - matches
  `bigsnpr::snp_cor()` on the same complete-data matrix.
- `statgen_ldpred2_inf()`, `statgen_ldpred2()` and `statgen_ldpred2_auto()`:
  LDpred2 (Prive, Arbel & Vilhjalmsson 2020), reimplemented clean-room over the
  banded `Rfmalloc::fmalloc_ld` LD store. All three read the LD matrix one
  column's neighbour run at a time (the `fmalloc_ld` C-callables), never the
  genotypes or a dense p x p. `statgen_ldpred2_inf()` is the deterministic
  infinitesimal model: it solves the ridge-like system `(C + m/(h2 N) I) b =
  beta_hat` by conjugate gradient over the band. `statgen_ldpred2()` is the
  LDpred2-grid Gibbs sampler (spike-and-slab prior, causal effects
  `~ N(0, h2/(m p))`), returning the Rao-Blackwellized posterior mean.
  `statgen_ldpred2_auto()` estimates `p` and `h2` from the data each iteration
  (the `alpha = -1`, `sigma2 = h2/(m p)` variant). `inst/tinytest/test_ldpred2.R`
  matches `snp_ldpred2_inf` to solver tolerance (max |diff| ~4e-12 against both
  a dense `solve()` and, guarded by `requireNamespace`, `bigsnpr`), and matches
  `snp_ldpred2_grid` / `snp_ldpred2_auto(use_MLE = FALSE)` within Monte-Carlo
  error (posterior-mean correlation > 0.99, and recovers the simulated p / h2),
  plus deterministic seeded self-consistency checks that run without bigsnpr.
- `statgen_coloc_bf()`: all-pairs Bayesian colocalisation of two traits'
  signals, coloc's `coloc.bf_bf` recast as a single matrix multiply. Stacking
  each trait's per-SNP log approximate Bayes factors as a matrix of signals,
  the shared-causal (H4) coupling for every signal pair is one stabilized
  log-sum-exp GEMM (`c = outer(m1,m2,"+") + log(E1 %*% t(E2))`); the coloc
  posteriors PP0..PP4 follow elementwise. The hot kernel therefore rides any
  backend 'Rggml' offers via `backend = "cpu"` / `"blas"` / `"vulkan"` -
  including a Vulkan GPU - and larger-than-memory signal sets via the
  out-of-core panel-GEMM; this is the same idea as gpu-coloc (Torch) without
  its CUDA/Metal-only lock-in. `trim = TRUE` returns the per-pair posterior
  overlap and flags low-overlap H4 (the analogue of `trim_by_posterior`).
  `inst/tinytest/test_coloc_bf.R` proves the matrix form is identical to the
  consecutive per-pair form to machine precision, that the single-precision
  backend agrees with the double-precision CPU path, and that it recovers
  colocalisation vs distinct causals.
