# Internal fixed-shape F32 lowering for the bound-program semantic oracle.
# It intentionally remains unexported until an evidence-based CPU execution
# path is competitive with the upstream implementation.
.rllm_f32_model <- function(path, runtime = NULL) {
    if (is.null(runtime)) runtime <- Rfmalloc::fmalloc_default_runtime()
    ctx <- Rgguf::gguf_open(path)
    definition <- .rllm_program_from_gguf(
        Rgguf::gguf_metadata(ctx), Rgguf::gguf_tensors(ctx)
    )
    program <- definition$program
    .rllm_f32_program_validate(program)
    needed <- names(program$parameters)
    bindings <- vector("list", length(needed))
    names(bindings) <- needed
    for (name in needed) {
        payload <- Rgguf::gguf_tensor(ctx, name, runtime = runtime, as = "view")
        type <- Rfmalloc::fmalloc_tensor_dtype(payload)
        if (!identical(type, "f32")) {
            stop("native F32 program parameter '", name,
                 "' has type '", type, "', not f32")
        }
        bindings[[name]] <- list(
            payload = payload, type = type,
            dims = as.integer(program$parameters[[name]]$shape)
        )
    }
    structure(list(
        execution = list(program = program, bindings = bindings),
        .contexts = new.env(parent = emptyenv())
    ), class = "rllm_f32_model")
}

.rllm_f32_forward <- function(model, inputs, threads = 1L,
                              backend = c("cpu", "cuda")) {
    if (!inherits(model, "rllm_f32_model")) {
        stop("model must come from .rllm_f32_model()")
    }
    inputs <- .rllm_execute_named(inputs, "inputs")
    if (!identical(names(inputs), "sequence")) {
        stop("native F32 execution requires exactly one input named 'sequence'")
    }
    sequence <- inputs$sequence
    if (!is.numeric(sequence) || is.object(sequence) ||
        !identical(length(dim(sequence)), 3L) || any(dim(sequence) < 1L) ||
        any(!is.finite(sequence))) {
        stop("sequence must be a finite numeric array with channel, sequence, and batch axes")
    }
    threads <- .rllm_count(threads, "threads")
    backend <- match.arg(backend)
    backend_code <- c(cpu = 0L, cuda = 3L)[[backend]]
    key <- paste0("f32:", backend, ":", paste(dim(sequence), collapse = "x"))
    context <- model$.contexts[[key]]
    backend_context <- NULL
    if (backend == "cuda") {
        backend_context <- model$.contexts$.cuda_weights
        if (is.null(backend_context)) {
            backend_context <- .Call(
                "RC_rllm_cuda_model_context", model$execution$bindings,
                PACKAGE = "Rllm"
            )
            model$.contexts$.cuda_weights <- backend_context
        }
    }
    answer <- .Call(
        "RC_rllm_f32_program_forward", model$execution, inputs,
        threads, backend_code, backend_context, context, PACKAGE = "Rllm"
    )
    model$.contexts[[key]] <- answer[[2L]]
    answer[[1L]]
}

.rllm_f32_program_validate <- function(program) {
    if (!inherits(program, "rllm_program")) stop("program must be an rllm_program")
    inputs <- Filter(function(node) identical(node$op, "input"), program$nodes)
    if (length(inputs) != 1L || !identical(inputs[[1L]]$attributes$name, "sequence") ||
        !identical(inputs[[1L]]$dtype, "f32") || length(inputs[[1L]]$shape) != 3L) {
        stop("native F32 execution requires one F32 rank-three input named 'sequence'")
    }
    allowed <- c("input", "conv1d", "batch_norm", "leaky_relu", "add", "crop1d", "softmax")
    unsupported <- setdiff(vapply(program$nodes, `[[`, character(1), "op"), allowed)
    if (length(unsupported)) {
        stop("native F32 execution does not implement operator '", unsupported[[1L]], "'")
    }
    if (length(program$outputs) != 1L) {
        stop("native F32 execution requires exactly one declared output")
    }
    invisible(program)
}
