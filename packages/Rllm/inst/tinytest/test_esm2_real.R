library(tinytest)
library(Rllm)

# Opt-in numerical execution of the official facebookresearch/esm ESM-2 8M
# checkpoint after tools/convert_esm2.R writes its unmodified F32 tensors to
# GGUF. The reference values come from fair-esm 2.0.0 with return_contacts and
# per-head attention enabled. The hermetic DSL test pins the topology without
# downloading a checkpoint.
path <- Sys.getenv("RLLM_ESM2_GGUF", "")
if (!nzchar(path) || !file.exists(path)) {
    exit_file("set RLLM_ESM2_GGUF to the converted ESM-2 8M GGUF")
}

backing <- tempfile(fileext = ".bin")
rt <- Rfmalloc::open_fmalloc(
    backing, mode = "scratch", size_gb = 0.2
)
program <- rllm_program(path)
parameters <- Rgguf::gguf_import(
    path, tensors = names(program$parameters), runtime = rt, as = "numeric"
)

tokens <- matrix(c(
    0L, 20L, 15L, 11L, 12L, 12L, 5L, 4L, 8L, 19L, 12L, 18L, 23L,
    4L, 7L, 18L, 5L, 13L, 19L, 15L, 13L, 13L, 13L, 13L, 15L, 2L
), ncol = 1L)
result <- rllm_execute(
    program,
    list(tokens = tokens, padding_mask = tokens == 1L),
    parameters = parameters
)

expect_equal(length(program$parameters), 106L)
expect_equal(dim(result$logits), c(33L, 26L, 1L))
expect_equal(dim(result$representation), c(320L, 26L, 1L))
expect_equal(dim(result$contacts), c(24L, 24L, 1L))

logits <- c(
    17.5220528, -6.2047787, -5.4338098, -6.1862698,
    0.0630226, 0.1262691, -0.2850048, -0.1282938,
    -6.5009398, -13.7416019, -4.4126458, -13.7363644,
    0.6236355, -0.6300545, -1.2744410, 0.0828393,
    -9.5050888, -16.5618076, -9.5078564, -16.5771999,
    0.8044817, -0.2942731, -1.2586002, -0.3000295,
    -9.3472033, -18.4214497, -10.3899117, -18.4395256,
    1.2850344, -0.3843019, -1.6422515, 0.1173328
)
expect_true(max(abs(result$logits[1:8, 1:4, 1L] - logits)) < 0.003)

representation <- c(
    0.0658966, 0.6288949, 0.2564336, 0.3306337,
    -0.2405722, -0.2959967, -0.9148865, 0.1469315,
    0.1004627, 0.6873642, -0.1262198, 0.4953677,
    -0.3534117, 0.0268067, -0.3312865, -0.4936540,
    -0.0099271, 0.0087663, 0.5402792, -0.1740655,
    -0.1638612, -0.0370012, 0.0329820, -0.6371211,
    -0.1626883, 0.3562712, 0.2352548, 0.0686832,
    -0.0761094, -0.3031365, -0.2237106, -0.1370436
)
expect_true(max(abs(
    result$representation[1:8, 1:4, 1L] - representation
)) < 2e-4)

attention <- matrix(c(
    0.1672455, 0.0253357, 0.0435705,
    0.2419294, 0.0140144, 0.0383336,
    0.2385300, 0.0101669, 0.0322712
), nrow = 3L, byrow = TRUE)
expect_true(max(abs(
    result$attention.1[1L, 1:3, 1:3, 1L] - attention
)) < 3e-5)

contacts <- matrix(c(
    5.0786516e-15, 4.1398871e-17, 1.0076439e-5, 0.5222887,
    4.1398871e-17, 0.0958156, 3.1535507e-15, 1.4301166e-6,
    1.0076420e-5, 3.1535388e-15, 0.0060667, 5.8051149e-15,
    0.5222883, 1.4301166e-6, 5.8051149e-15, 0.0064068
), nrow = 4L, byrow = TRUE)
expect_true(max(abs(result$contacts[1:4, 1:4, 1L] - contacts)) < 3e-4)

expect_error(
    rllm_gguf_model(path, runtime = rt),
    "native GGML lowering.*2-input grammar"
)

Rfmalloc::cleanup_fmalloc(rt)
unlink(backing)
message("real ESM-2 numerical program test completed: ", basename(path))
