#' Load a GGUF model as a bound semantic program
#'
#' Normalizes the model-family metadata into a typed [rllm_program()], validates
#' every tensor role and shape, then binds each required weight directly to the
#' original GGUF mapping. Quantized and floating-point payloads keep their
#' on-disk encoding; no second weight store is created. The same program is the
#' inspectable architecture and the source consumed by the native GGML
#' lowering.
#'
#' Architecture definitions are data ASTs rather than native model-family
#' branches. Native GGML lowering covers llama, Qwen3.5, LFM2MoE and
#' EmbeddingGemma. ESM-2 and OpenSpliceAI are numerically executable through
#' [rllm_execute()], but their multi-input or floating-point sequence-input
#' programs are not accepted by this native model loader. Models with tied
#' embeddings reuse `token_embd.weight` as the output
#' projection.
#'
#' @param path Path to a GGUF file.
#' @param runtime [Rfmalloc::open_fmalloc()] runtime attached to the borrowed
#'   tensor views, or `NULL` to use the default established by
#'   [Rfmalloc::init_fmalloc()]. It supplies the allocation context for
#'   operations which produce fmalloc results; the weight bytes remain in the
#'   GGUF mapping.
#' @param rope_mode Optional RoPE override: `0` for normal/interleaved or `2`
#'   for NEOX. The architecture program supplies the default.
#'
#' @return An object of class `rllm_model` containing its hyperparameters,
#'   borrowed weight payloads, tokenizer metadata, and model-owned backend
#'   contexts created lazily by [rllm_forward()].
#' @seealso [rllm_forward()]
#' @export
rllm_gguf_model <- function(path, runtime = NULL, rope_mode = NULL) {
    ctx <- Rgguf::gguf_open(path)
    md <- Rgguf::gguf_metadata(ctx)
    tt <- Rgguf::gguf_tensors(ctx)
    definition <- .rllm_program_from_gguf(md, tt, rope_mode)
    program <- definition$program
    needed <- names(program$parameters)

    tensors <- vector("list", length(needed))
    names(tensors) <- needed
    for (nm in needed) {
        nt <- Rgguf::gguf_tensor(ctx, nm, runtime = runtime, as = "view")
        tensors[[nm]] <- list(
            payload = nt,
            type = Rfmalloc::fmalloc_tensor_dtype(nt),
            dims = program$parameters[[nm]]$shape
        )
    }

    execution <- .rllm_bind_program(program, definition$symbols, tensors)

    hparams <- execution$lowering$symbols
    if (length(unique(hparams$n_head_kv)) == 1L) {
        hparams$n_head_kv <- hparams$n_head_kv[[1L]]
    }
    attention <- which(vapply(execution$lowering$layers, function(layer) {
        identical(layer$operator$op, "attention")
    }, logical(1)))[1L]
    resolved_rope_mode <- if (is.na(attention)) 0L else
        execution$lowering$layers[[attention]]$operator$rope$mode

    structure(list(
        execution = execution,
        hparams = hparams,
        rope_mode = resolved_rope_mode,
        arch = execution$program$name,
        # Tokenizer material for the ids <-> bytes edge codecs (rllm_encode/
        # rllm_decode); NULL when the file carries none (synthetic models).
        tok_model = md[["tokenizer.ggml.model"]],
        vocab = md[["tokenizer.ggml.tokens"]],
        merges = md[["tokenizer.ggml.merges"]],
        bos_id = md[["tokenizer.ggml.bos_token_id"]],
        eos_id = md[["tokenizer.ggml.eos_token_id"]],
        tokenizer_pre = md[["tokenizer.ggml.pre"]],
        chat_template = md[["tokenizer.chat_template"]],
        .contexts = new.env(parent = emptyenv())
    ), class = "rllm_model")
}

#' @export
print.rllm_model <- function(x, ...) {
    hp <- x$hparams
    bindings <- x$execution$bindings
    types <- vapply(bindings, function(t) t$type, character(1))
    operators <- vapply(x$execution$lowering$layers, function(layer) {
        layer$operator$op
    }, character(1))
    cat(sprintf(
        "<rllm_model %s: %d layers, n_embd %d, %d heads, n_ff %d, vocab %d; %d tensors (%s); %s>\n",
        x$arch, hp$n_layer, hp$n_embd, hp$n_head, hp$n_ff,
        hp$n_vocab, length(bindings),
        paste(names(sort(table(types), decreasing = TRUE)), collapse = "/"),
        .rllm_plan_counts(operators)
    ))
    invisible(x)
}

#' Create program-shaped state for incremental decoding
#'
#' Allocates the persistent state declared by the bound program. Attention
#' layers receive key/value slabs, short-convolution layers receive causal
#' history,
#' and gated-delta layers receive both convolution history and recurrent
#' matrices. State is plain R memory by default or \pkg{Rfmalloc}-backed when
#' `runtime` is supplied.
#'
#' The returned object is an environment, so [rllm_forward()] can advance its
#' `n_past` by reference.
#'
#' @param model An `rllm_model` from [rllm_gguf_model()].
#' @param n_ctx Maximum number of positions the cache can hold.
#' @param runtime Optional [Rfmalloc::open_fmalloc()] runtime for the slabs.
#'
#' @return An environment of class `rllm_kv_cache` with per-layer `k`, `v`,
#'   `conv` and `recurrent` state, plus `n_ctx` and `n_past`.
#' @seealso [rllm_forward()], [rllm_generate()]
#' @export
rllm_kv_cache <- function(model, n_ctx = 512L, runtime = NULL) {
    if (!inherits(model, "rllm_model")) {
        stop("model must be an rllm_model from rllm_gguf_model()")
    }
    if (!identical(model$execution$lowering$output$op, "projection")) {
        stop("this model produces embeddings and has no incremental decode state")
    }
    n_ctx <- as.integer(n_ctx)
    if (length(n_ctx) != 1L || is.na(n_ctx) || n_ctx < 1L) {
        stop("n_ctx must be positive")
    }

    new_slab <- function(bytes) {
        if (bytes == 0) return(raw())
        if (is.null(runtime)) {
            raw(bytes)
        } else {
            Rfmalloc::create_fmalloc_vector("raw", length = bytes,
                                            runtime = runtime,
                                            zero_initialize = TRUE)
        }
    }
    states <- lapply(model$execution$lowering$layers, `[[`, "state")
    k_bytes <- vapply(states, function(state) {
        if (state$op == "kv") as.double(n_ctx) * state$width * 4 else 0
    }, numeric(1))
    conv_bytes <- vapply(states, function(state) {
        if (state$op == "conv") {
            as.double(state$width) * state$history * 4
        } else if (state$op == "gated_delta") {
            prod(as.double(state$convolution)) * 4
        } else {
            0
        }
    }, numeric(1))
    recurrent_bytes <- vapply(states, function(state) {
        if (state$op == "gated_delta") {
            prod(as.double(state$matrix)) * 4
        } else {
            0
        }
    }, numeric(1))
    state_bytes <- c(k_bytes, conv_bytes, recurrent_bytes)
    if (any(!is.finite(state_bytes)) ||
        any(state_bytes > .Machine$integer.max)) {
        stop("requested model state is too large for an R raw vector")
    }
    cache <- new.env(parent = emptyenv())
    cache$k <- lapply(k_bytes, new_slab)
    cache$v <- lapply(k_bytes, new_slab)
    cache$conv <- lapply(conv_bytes, new_slab)
    cache$recurrent <- lapply(recurrent_bytes, new_slab)
    cache$program <- model$execution$program
    cache$states <- states
    cache$n_ctx <- n_ctx
    cache$n_past <- 0L
    class(cache) <- "rllm_kv_cache"
    cache
}

#' @export
print.rllm_kv_cache <- function(x, ...) {
    state <- c(x$k, x$conv, x$recurrent)
    state <- state[vapply(state, length, numeric(1)) > 0]
    storage <- if (length(state) && Rfmalloc::is_fmalloc_vector(state[[1L]])) {
        "fmalloc"
    } else {
        "R"
    }
    kinds <- vapply(x$states, `[[`, character(1), "op")
    cat(sprintf("<rllm_kv_cache: %d/%d positions, %s, %s state>\n",
                x$n_past, x$n_ctx, .rllm_plan_counts(kinds), storage))
    invisible(x)
}

#' Lower a bound semantic program and return its logits
#'
#' Lowers the model's inspectable [rllm_program()] to a GGML graph over its
#' memory-mapped weights and computes it on a chosen backend. The operator
#' vocabulary includes causal and gated attention, gated-delta recurrence,
#' short convolution, dense gated products and sparse routed experts.
#' Quantized weights are contracted natively in their encoded form - they are
#' never decoded to double. The CPU backend borrows the mapped bytes directly.
#' On its first use, the CUDA backend creates a model-owned context and uploads
#' the codec-native weights once. Later passes reuse those resident weights;
#' mutable inputs, cache slabs and logits move through Rggml's transfer API.
#'
#' Without a `cache`, the graph evaluates the complete token batch from zero
#' state. With a [rllm_kv_cache()], the pass advances every state declared by
#' the program, including key/value, convolution and gated-delta state, then
#' advances `cache$n_past`. This is the incremental-decoding path: prefill once
#' with the prompt, then feed one token at a time.
#'
#' @param model An `rllm_model` from [rllm_gguf_model()].
#' @param tokens Integer vector of 0-based token ids (as in the GGUF vocab).
#' @param cache Optional [rllm_kv_cache()] for incremental decoding.
#' @param backend Compute backend. `"cpu"` is the zero-copy mmap path;
#'   `"cuda"` requires Rggml installed with `--with-cuda` and a visible NVIDIA
#'   device.
#'
#' @return A numeric matrix of logits, dim `c(n_vocab, length(tokens))`:
#'   column `i` scores the token following position `i`.
#' @export
rllm_forward <- function(model, tokens, cache = NULL,
                         backend = c("cpu", "cuda")) {
    if (!inherits(model, "rllm_model")) {
        stop("model must be an rllm_model from rllm_gguf_model()")
    }
    if (!identical(model$execution$lowering$output$op, "projection")) {
        stop("this model produces embeddings; use rllm_embed()")
    }
    tokens <- as.integer(tokens)
    if (length(tokens) < 1L || anyNA(tokens)) {
        stop("tokens must be a non-empty vector of 0-based token ids")
    }
    backend <- match.arg(backend)
    backend_code <- c(cpu = 0L, cuda = 3L)[[backend]]
    backend_context <- NULL
    if (backend == "cuda") {
        if (!is.environment(model$.contexts)) {
            stop("model has no backend-context environment")
        }
        backend_context <- model$.contexts$cuda
        if (is.null(backend_context)) {
            backend_context <- .Call(
                "RC_rllm_cuda_model_context", model$execution$bindings,
                PACKAGE = "Rllm"
            )
            model$.contexts$cuda <- backend_context
        }
    }
    if (is.null(cache)) {
        return(.Call("RC_rllm_program_forward", model$execution, tokens,
                     NULL, NULL, NULL, NULL, 0L, 0L, backend_code,
                     backend_context, PACKAGE = "Rllm"))
    }
    if (!inherits(cache, "rllm_kv_cache")) {
        stop("cache must be an rllm_kv_cache from rllm_kv_cache()")
    }
    if (!identical(cache$program, model$execution$program)) {
        stop("cache belongs to a different model program")
    }
    out <- .Call("RC_rllm_program_forward", model$execution, tokens,
                 cache$k, cache$v, cache$conv, cache$recurrent,
                 cache$n_ctx, cache$n_past, backend_code, backend_context,
                 PACKAGE = "Rllm")
    cache$n_past <- cache$n_past + length(tokens)
    out
}

#' Compute one pooled sequence embedding
#'
#' Lowers an embedding model's semantic program over a complete token sequence.
#' Bidirectional and symmetric-window attention are evaluated without a decode
#' cache, then the program's pooling and projection pipeline produces one vector.
#' Quantized weights remain encoded in the mapped GGUF file.
#'
#' Tokenization is deliberately separate from model execution. Supply the
#' model's 0-based token ids, including any BOS or EOS tokens required by its
#' tokenizer.
#'
#' @param model An embedding `rllm_model` from [rllm_gguf_model()].
#' @param tokens Integer vector of 0-based token ids.
#' @param normalize Whether to return a unit-length vector.
#' @param backend Compute backend. `"cpu"` uses mapped weights directly;
#'   `"cuda"` requires a CUDA-enabled Rggml installation.
#'
#' @return A numeric vector whose length is the program's output dimension.
#' @export
rllm_embed <- function(model, tokens, normalize = TRUE,
                       backend = c("cpu", "cuda")) {
    if (!inherits(model, "rllm_model")) {
        stop("model must be an rllm_model from rllm_gguf_model()")
    }
    if (!identical(model$execution$lowering$output$op, "embedding")) {
        stop("this model produces token logits; use rllm_forward()")
    }
    tokens <- as.integer(tokens)
    if (length(tokens) < 1L || anyNA(tokens)) {
        stop("tokens must be a non-empty vector of 0-based token ids")
    }
    if (!is.logical(normalize) || length(normalize) != 1L || is.na(normalize)) {
        stop("normalize must be TRUE or FALSE")
    }
    backend <- match.arg(backend)
    backend_code <- c(cpu = 0L, cuda = 3L)[[backend]]
    backend_context <- NULL
    if (backend == "cuda") {
        if (!is.environment(model$.contexts)) {
            stop("model has no backend-context environment")
        }
        backend_context <- model$.contexts$cuda
        if (is.null(backend_context)) {
            backend_context <- .Call(
                "RC_rllm_cuda_model_context", model$execution$bindings,
                PACKAGE = "Rllm"
            )
            model$.contexts$cuda <- backend_context
        }
    }
    out <- .Call(
        "RC_rllm_program_forward", model$execution, tokens,
        NULL, NULL, NULL, NULL, 0L, 0L, backend_code, backend_context,
        PACKAGE = "Rllm"
    )[, 1L]
    if (normalize) {
        magnitude <- sqrt(sum(out * out))
        if (!is.finite(magnitude) || magnitude == 0) {
            stop("model produced an embedding with no finite magnitude")
        }
        out <- out / magnitude
    }
    out
}
