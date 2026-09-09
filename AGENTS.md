# AGENTS for the Rfmalloc monorepo

One repo, one story: **out-of-core array computation over
pluggable storage codecs and compute backends**. fmalloc (mmap storage), ALP
(lossless float compression), GGUF quantized blocks, and GGML kernels are each
*one bet*; the packages/<pkg> layout exists so a change that spans bets lands
as one commit and is validated as one unit.

## Coding standards
- write C as a BSD kernel programmer rather than a Java programmer that failed upwards
- write R as a r-lib programmer rather than a Python programmer that failed upwards
- license is GPL (>= 2) everywhere; vendored MIT/BSD components are documented
  in each package's `inst/COPYRIGHTS` (never relicense, never vendor without
  updating COPYRIGHTS)

## Layout
- `packages/Rfmalloc` - substrate: file-backed ALTREP + fmalloc runtime, codec
  registry + matmul backend registry (C API, see `inst/include/Rfmalloc.h`),
  panel matmul, OOC eviction, ALP/sparse codecs
- `packages/Rgguf` - GGUF reader/writer and quantized-codec registration layer;
  parsing and writing use the official GGUF implementation compiled once by
  Rggml, so Rgguf carries no second parser
- `packages/Rggml` - vendored GGML static lib + C-callables; BLAS backend via
  cblas→Fortran shim; opt-in Vulkan (`--with-vulkan`) and CUDA
  (`--with-cuda`) backends; on x86,
  runtime-SIMD kernels staged in `tools/simd` by configure (RsimdDispatch
  pattern: ISA flags never appear in R's recorded flags); on aarch64, GGML's
  own NEON kernels instead (mandatory baseline, no dispatch needed). The
  vendored tree is **generated** by `tools/vendor-ggml/vendorggml.R` from
  pinned sources: never hand-edit `inst/ggml`, edit a patch and re-run
  `vendorggml.R vendor`. CI enforces this (`vendored-ggml-matches-recipe`).
- `packages/Rllm` - composition: registers Rggml as Rfmalloc's typed
  (codec-aware) matmul backend; quantize/dequantize R surface; the LLM
  graph-builder lands here

## Cross-package rules (the reason this is a monorepo)
- The C-callable APIs are contracts: update every consumer in the same commit.
  While every consumer is in this monorepo, do not add compatibility shims or
  bump API versions merely to record an internal extension.
- Correctness against references, not intuition: quantized decode paths must
  stay bit-identical to GGML's `to_float` (`Rggml_dequantize`). The pins are
  `packages/Rgguf/inst/tinytest/test_gguf_codec_ggml_ref.R` (fixture) and
  `packages/Rllm/inst/tinytest/test_codec_consistency.R` (live). If you touch
  a codec, run both.
- Any backend may decline any product (return non-zero); Rfmalloc's decode+BLAS
  fallback must always produce correct results. Never make correctness depend
  on which backend is selected.

## Build gotcha

R's default rules do not track header dependencies, and `R CMD INSTALL` without
`--preclean` will happily link objects older than the headers they were
compiled against, then report success. `make install` at the repo root uses
`--preclean` for exactly this reason; a per-package `R CMD INSTALL` after a
header-only change does not, and a cross-package header (`inst/include/*.h`) is
invisible to every consumer's make. Rfmalloc declares its `.inc` dependencies
explicitly because all of its implementation lives there. Any Makevars that
adds a rule of its own must start with `all: $(SHLIB)`, or that rule becomes
make's default goal and the shared library is never linked.

## Test workflow
- Per package: `Rscript -e "tinytest::test_package('<pkg>')"` (install the
  package and its local siblings first: `R CMD INSTALL packages/<sibling>`
  in dependency order Rfmalloc, Rggml, Rgguf, Rllm, Rpgen; Rpgen needs only
  Rfmalloc).
- Focused: `Rscript -e "tinytest::run_test_file('packages/<pkg>/inst/tinytest/<file>.R')"`.
- Before finishing: build every package you touched with `R CMD build`, then run
  `R CMD check --no-manual` on the resulting tarball; CI additionally runs the
  full-stack integration job. Checking the source directory skips the build
  step that derives standard metadata from `Authors@R`.

## Conventions
- `NAMESPACE` and `man/*.Rd` are roxygen2-generated; edit roxygen comments,
  then `Rscript -e "roxygen2::roxygenise('packages/<pkg>')"`.
- `DESCRIPTION` uses `Authors@R` as the sole author source. Do not duplicate it
  with hand-written `Author` or `Maintainer` fields; `R CMD build` derives them.
- `NEWS.md` per package, newest entries on top; one bullet per user-visible
  change.
- Tests are tinytest, in `packages/<pkg>/inst/tinytest/`.
- READMEs: `README.Rmd` is the source where present; re-render to `README.md`.
- CI is `.github/workflows/R-CMD-check.yaml` (per-package matrix + integration
  job). Keep it green on every push. All six packages target Windows (Rtools);
  all six are checked on aarch64 (`ubuntu-24.04-arm`, macOS Apple Silicon).
- Prose is written without em dashes, anywhere: READMEs, DESCRIPTION, NEWS,
  commit messages. Use a colon in a title, ` - ` mid-sentence.

## GPU test harness (the rig)

Real GPU numbers come from **`ssh rig`**, a gaming-rig WSL box reachable over a
reverse tunnel (port 2222) from the dev server. It is the harness for anything
GPU: CI only ever exercises the Vulkan backend on Mesa lavapipe (a software
driver, correctness not speed), so a genuine GPU-vs-CPU measurement has to run
here.

- **Hardware:** NVIDIA RTX 5050 Laptop, 8 GB VRAM (Blackwell, compute 12.x),
  CUDA 13.2 at `/usr/local/cuda-13.2`.
- **`nvidia-smi` is at `/usr/lib/wsl/lib/nvidia-smi`**, not on the non-login
  `ssh rig` PATH; call it by full path or a bare `nvidia-smi` may report
  "not reachable" when the GPU is in fact fine.
- **Vulkan is a dead end on the rig:** WSL ships no Dozen (`dzn`) driver, so the
  RTX 5050 is not a Vulkan device there. **CUDA is the only GPU path on the
  rig.** On SmolLM2 `Q4_K_M`, upstream `llama-bench` measures about 156 tok/s
  without CUDA graphs and 460 tok/s with them. Rllm's current complete
  generation path measures about 70 tok/s after its one-time weight upload.
- **Memory:** two caps, do not confuse them. WSL system RAM is ~8 GB (default,
  no `.wslconfig`); raise it by writing `[wsl2] memory=...` to
  `C:\Users\<user>\.wslconfig` on the Windows host and running `wsl --shutdown`
  (which drops the tunnel until WSL restarts) - that only helps the nvcc build.
  GPU inference is capped by the 8 GB VRAM, which is fixed hardware and already
  ample for the models we test.
- Editing `.wslconfig` and `wsl --shutdown` are Windows-host actions the agent
  cannot do over `ssh rig`; hand them to the user.
- The executable setup and recovery procedure is in
  `docs/rig-tunnel-setup.Rmd`.
