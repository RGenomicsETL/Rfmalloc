library(tinytest)
library(Rllm)

# Opt-in numerical execution of Kuanhao-Chao/OpenSpliceAI commit 394109a,
# using its MANE 80 nt rs10 checkpoint (SHA-256
# 894594d18be6d3e339d76b9698accf4071bd2f0f32f40c52a2c1b848d1cef9f4).
# tools/convert_openspliceai.R writes the checkpoint through Rgguf. The
# reference probabilities below come from the upstream PyTorch SpliceAI class
# in eval mode over the complete 16-position output.
path <- Sys.getenv("RLLM_OPENSPLICEAI_GGUF", "")
if (!nzchar(path) || !file.exists(path)) {
    exit_file(
        "set RLLM_OPENSPLICEAI_GGUF to a converted OpenSpliceAI GGUF"
    )
}

backing <- tempfile(fileext = ".bin")
runtime <- Rfmalloc::open_fmalloc(
    backing, mode = "scratch", size_gb = 0.05
)
program <- rllm_program(path)
parameters <- Rgguf::gguf_import(
    path, tensors = names(program$parameters), runtime = runtime,
    as = "numeric"
)

sequence <- strsplit(
    paste0(
        "GAATCAGCAANTNTAGGCGGTCTGAATTGGTAGTGGCATGTCCTAGCAGTTTAGCTN",
        "GGCTTTGATCTCAGCAATCATTATTCTCTAGTCGTCCTC"
    ),
    "", fixed = TRUE
)[[1L]]
input <- array(0, dim = c(4L, length(sequence), 1L))
channel <- match(sequence, c("A", "C", "G", "T"))
for (at in which(!is.na(channel))) input[channel[[at]], at, 1L] <- 1

result <- rllm_execute(
    program, list(sequence = input), parameters = parameters
)$probabilities
reference <- matrix(c(
    1, 2.43823095e-09, 9.01716257e-10,
    1, 1.77531467e-09, 2.07266222e-08,
    1, 1.17008570e-09, 3.85215054e-10,
    1, 6.51066046e-09, 1.85356785e-09,
    1, 1.72259196e-09, 4.48484938e-09,
    1, 4.28378305e-10, 5.61280224e-11,
    0.999999762, 2.96964345e-07, 5.34654312e-08,
    0.999999881, 7.11570025e-10, 7.87054404e-08,
    0.999999881, 5.67734437e-09, 7.19059656e-08,
    0.999999642, 3.19586405e-07, 4.73196096e-08,
    1, 1.01111119e-10, 1.53772522e-10,
    1, 2.14516543e-10, 1.35844391e-10,
    1, 8.32501845e-10, 1.43491219e-10,
    1, 3.61505575e-10, 1.07716426e-10,
    0.999999881, 1.59909945e-07, 2.55486365e-09,
    1, 1.28549893e-10, 6.48963938e-11
), nrow = 3L)

expect_equal(length(program$parameters), 56L)
expect_equal(dim(result), c(3L, 16L, 1L))
expect_true(max(abs(result[, , 1L] - reference)) < 5e-7)
expect_true(max(abs(colSums(result[, , 1L]) - 1)) < 1e-12)
expect_error(
    Rllm:::.rllm_lower_program(
        program, list(architecture = "openspliceai")
    ),
    "native program input must be i32 tokens"
)

Rfmalloc::cleanup_fmalloc(runtime)
unlink(backing)
message("real OpenSpliceAI numerical program test completed: ", basename(path))
