library(tinytest)
library(Rllm)

program <- Rllm:::.rllm_openspliceai_program(
    n_block = 4L, n_embd = 4L, n_input = 4L, n_output = 3L,
    context_length = 16L,
    kernel_size = rep.int(3L, 4L), dilation = rep.int(1L, 4L),
    batch_norm_eps = 1e-5, leaky_relu_slope = 0.1
)
expect_equal(program$name, "openspliceai")
expect_equal(length(program$parameters), 56L)
expect_equal(
    sort(unique(vapply(program$nodes, `[[`, character(1), "op"))),
    sort(c(
        "input", "conv1d", "batch_norm", "leaky_relu", "add",
        "crop1d", "softmax"
    ))
)

set.seed(271L)
parameters <- lapply(program$parameters, function(parameter) {
    values <- rnorm(prod(parameter$shape), sd = 0.15)
    if (endsWith(parameter$name, ".running_var")) {
        values <- runif(length(values), 0.5, 1.5)
    } else if (grepl("batchnorm[12]\\.weight$", parameter$name)) {
        values <- 1 + values
    }
    if (length(parameter$shape) > 1L) dim(values) <- parameter$shape
    values
})
sequence <- array(rnorm(4L * 24L * 2L), dim = c(4L, 24L, 2L))
result <- rllm_execute(
    program, list(sequence = sequence), parameters = parameters
)
expect_equal(dim(result$probabilities), c(3L, 8L, 2L))
expect_true(all(is.finite(result$probabilities)))
expect_equal(
    apply(result$probabilities, c(2L, 3L), sum),
    matrix(1, nrow = 8L, ncol = 2L), tolerance = 1e-12
)

changed <- sequence
changed[1L, 12L, 1L] <- changed[1L, 12L, 1L] + 1
changed_result <- rllm_execute(
    program, list(sequence = changed), parameters = parameters
)
expect_false(isTRUE(all.equal(
    changed_result$probabilities, result$probabilities
)))

metadata <- list(
    general.architecture = "openspliceai",
    openspliceai.block_count = 4L,
    openspliceai.embedding_length = 4L,
    openspliceai.input_channel_count = 4L,
    openspliceai.output_channel_count = 3L,
    openspliceai.context_length = 16L,
    openspliceai.convolution.kernel_size = rep.int(3L, 4L),
    openspliceai.convolution.dilation = rep.int(1L, 4L),
    openspliceai.batch_norm_epsilon = 1e-5,
    openspliceai.leaky_relu_slope = 0.1
)
path <- tempfile(fileext = ".gguf")
Rgguf::gguf_write_tensors(path, parameters, metadata)
adapted <- rllm_program(path)
expect_equal(names(adapted$parameters), names(program$parameters))
directory <- Rgguf::gguf_tensors(path)
bad_schedule <- metadata
bad_schedule$openspliceai.convolution.kernel_size[[1L]] <- 3.5
expect_error(
    Rllm:::.rllm_program_openspliceai(
        bad_schedule, directory, rope_mode = NULL
    ),
    "convolution schedule is invalid"
)
bad_context <- metadata
bad_context$openspliceai.context_length <- 18L
expect_error(
    Rllm:::.rllm_program_openspliceai(
        bad_context, directory, rope_mode = NULL
    ),
    "context length disagrees"
)
expect_equal(
    rllm_execute(
        adapted, list(sequence = sequence), parameters = parameters
    )$probabilities,
    result$probabilities,
    tolerance = 1e-12
)
native_backing <- tempfile(fileext = ".bin")
native_runtime <- Rfmalloc::open_fmalloc(native_backing, mode = "scratch", size_gb = 0.05)
native_model <- Rllm:::.rllm_f32_model(path, runtime = native_runtime)
native <- Rllm:::.rllm_f32_forward(
    native_model, list(sequence = sequence), threads = 1L
)$probabilities
expect_equal(dim(native), c(3L, 8L, 2L))
expect_equal(native, result$probabilities, tolerance = 2e-4)
context_key <- "f32:cpu:4x24x2"
context_two <- native_model$.contexts[[context_key]]
native_repeat <- Rllm:::.rllm_f32_forward(
    native_model, list(sequence = sequence), threads = 1L
)$probabilities
expect_equal(native_repeat, native, tolerance = 0)
expect_true(identical(native_model$.contexts[[context_key]], context_two))
native_one <- Rllm:::.rllm_f32_forward(
    native_model, list(sequence = sequence[, , 1L, drop = FALSE]), threads = 1L
)$probabilities
expect_equal(native_one, native[, , 1L, drop = FALSE], tolerance = 2e-4)
expect_true(exists("f32:cpu:4x24x1", native_model$.contexts, inherits = FALSE))
expect_false(identical(
    native_model$.contexts[["f32:cpu:4x24x1"]], context_two
))
expect_equal(
    apply(native, c(2L, 3L), sum), matrix(1, nrow = 8L, ncol = 2L),
    tolerance = 2e-6
)
expect_error(
    Rllm:::.rllm_f32_forward(native_model, list(sequence = sequence), threads = 0L),
    "threads must be one positive integer"
)
expect_error(
    Rllm:::.rllm_f32_forward(native_model, list(other = sequence)),
    "exactly one input named 'sequence'"
)
# Without a device the CUDA request must fail by name; with one it must agree
# with the CPU path, whose convolutions absorb operators the device graph keeps.
if (isTRUE(Rggml::rggml_has_cuda())) {
    device <- Rllm:::.rllm_f32_forward(
        native_model, list(sequence = sequence), backend = "cuda"
    )$probabilities
    expect_equal(dim(device), dim(native))
    # cuBLAS uses TF32 tensor cores for F32 products by default on Ampere and
    # later, so the device tolerance is wider than F32 accumulation error.
    expect_equal(device, native, tolerance = 5e-3)
} else {
    expect_error(
        Rllm:::.rllm_f32_forward(native_model, list(sequence = sequence),
                                 backend = "cuda"),
        "CUDA backend unavailable"
    )
}
bad_model <- native_model
add_at <- which(vapply(
    bad_model$execution$program$nodes, `[[`, character(1), "op"
) == "add")[[1L]]
bad_model$execution$program$nodes[[add_at]]$inputs[[1L]] <- "unknown"
expect_error(
    Rllm:::.rllm_f32_forward(bad_model, list(sequence = sequence)),
    "later or unknown input"
)
# A parameter reference that only a helper can reject fails deep in
# construction, after the backend, both graph contexts, the staging buffer and
# earlier packed convolution plans exist. The error unwinds past any local
# cleanup, so the context has to be owned by its external pointer from the
# start; nothing built so far may survive the failure or disturb the next call.
late_model <- native_model
conv_at <- which(vapply(
    late_model$execution$program$nodes, `[[`, character(1), "op"
) == "conv1d")
late_model$execution$program$nodes[[
    conv_at[[length(conv_at)]]
]]$attributes$weight <- "not a parameter"
for (i in 1:3) {
    expect_error(
        Rllm:::.rllm_f32_forward(late_model, list(sequence = sequence)),
        "must be a parameter"
    )
}
gc()
expect_equal(
    Rllm:::.rllm_f32_forward(native_model, list(sequence = sequence),
                             threads = 1L)$probabilities,
    native, tolerance = 0
)
expect_error(
    Rllm:::.rllm_lower_program(
        adapted, list(architecture = "openspliceai")
    ),
    "native program input must be i32 tokens"
)
# A collected model owns the external pointer and must release its CPU backend,
# graph buffer, contexts, and staging buffers without touching the next model.
local({
    transient <- Rllm:::.rllm_f32_model(path, runtime = native_runtime)
    Rllm:::.rllm_f32_forward(transient, list(sequence = sequence), threads = 1L)
})
gc()
expect_equal(
    Rllm:::.rllm_f32_forward(native_model, list(sequence = sequence), threads = 1L)$probabilities,
    native, tolerance = 0
)
rm(native_model, context_two)
gc()
Rfmalloc::cleanup_fmalloc(native_runtime)
unlink(c(path, native_backing))

message("OpenSpliceAI program and dense execution tests completed")
