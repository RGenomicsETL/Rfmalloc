.rllm_architecture_scalar <- function(key, architecture, name,
                                      positive = TRUE,
                                      default = .rllm_missing) {
    value <- if (positive) {
        .rllm_positive(key(name, default), paste0(architecture, ".", name))
    } else {
        .rllm_integer(key(name, default), paste0(architecture, ".", name))
    }
    if (length(value) != 1L) {
        stop("GGUF hyperparameter '", paste0(architecture, ".", name),
             "' must have length one")
    }
    value
}

.rllm_embedding_shape <- function(directory, architecture, n_embd,
                                  name = "token_embd.weight") {
    at <- match(name, directory$name)
    if (is.na(at)) stop("GGUF file has no '", name, "' tensor")
    shape <- as.integer(directory$dims[[at]])
    if (length(shape) != 2L || shape[[1L]] != n_embd) {
        stop(name, " is inconsistent with ", architecture,
             ".embedding_length")
    }
    shape
}

.rllm_architecture_parameter <- function(name, role, shape) {
    rllm_parameter(name, as.integer(shape), role = role)
}

.rllm_program_entry <- function(embedding, scale) {
    n_embd <- embedding$shape[[1L]]
    tokens <- rllm_input("tokens", c(sequence = "n_token"), "i32")
    rllm_op(
        tokens, "embedding", weight = embedding, scale = scale,
        output_shape = c(
            feature = as.character(n_embd), sequence = "n_token"
        ),
        output_dtype = "f32"
    )
}

.rllm_program_semantic_op <- function(x, specification, state = NULL) {
    op <- specification$op
    specification$op <- NULL
    if (!is.null(state)) specification$state <- state
    do.call(rllm_op, c(list(x = x, op = op), specification))
}

.rllm_program_block <- function(x, index, operator_norm, operator, state,
                                ffn_norm, feed_forward, eps,
                                operator_post_norm = NULL,
                                feed_forward_post_norm = NULL) {
    block <- rllm_module(paste0("block.", index), function(x) {
        x <- rllm_residual(x, function(branch) {
            branch <- rllm_norm(branch, operator_norm, eps = eps)
            branch <- .rllm_program_semantic_op(branch, operator, state)
            if (!is.null(operator_post_norm)) {
                branch <- rllm_norm(branch, operator_post_norm, eps = eps)
            }
            branch
        })
        rllm_residual(x, function(branch) {
            branch <- rllm_norm(branch, ffn_norm, eps = eps)
            branch <- .rllm_program_semantic_op(branch, feed_forward)
            if (!is.null(feed_forward_post_norm)) {
                branch <- rllm_norm(
                    branch, feed_forward_post_norm, eps = eps
                )
            }
            branch
        })
    })
    block(x)
}

.rllm_program_from_gguf <- function(metadata, directory, rope_mode = NULL) {
    .rllm_validate_directory(directory)
    architecture <- metadata[["general.architecture"]]
    if (!is.character(architecture) || length(architecture) != 1L ||
        is.na(architecture) || !nzchar(architecture)) {
        stop("GGUF file has no 'general.architecture' metadata key")
    }
    definition <- switch(
        architecture,
        llama = .rllm_program_llama(metadata, directory, rope_mode),
        qwen35 = .rllm_program_qwen35(metadata, directory, rope_mode),
        lfm2moe = .rllm_program_lfm2moe(metadata, directory, rope_mode),
        esm2 = .rllm_program_esm2(metadata, directory, rope_mode),
        openspliceai = .rllm_program_openspliceai(
            metadata, directory, rope_mode
        ),
        `gemma-embedding` = .rllm_program_gemma_embedding(
            metadata, directory, rope_mode
        ),
        stop("unsupported GGUF architecture '", architecture,
             "': no architecture program is registered")
    )
    definition$program <- .rllm_validate_program(
        definition$program, directory
    )
    definition
}

.rllm_program_llama <- function(metadata, directory, rope_mode) {
    arch <- "llama"
    key <- function(name, default = .rllm_missing) {
        .rllm_metadata(metadata, arch, name, default)
    }
    n_layer <- .rllm_positive(key("block_count"), "llama.block_count")
    n_embd <- .rllm_positive(
        key("embedding_length"), "llama.embedding_length"
    )
    n_head <- .rllm_positive(
        key("attention.head_count"), "llama.attention.head_count"
    )
    n_head_kv <- .rllm_positive(
        key("attention.head_count_kv", n_head),
        "llama.attention.head_count_kv"
    )
    n_ff <- .rllm_positive(
        key("feed_forward_length"), "llama.feed_forward_length"
    )
    if (length(n_layer) != 1L || length(n_embd) != 1L ||
        length(n_head) != 1L || length(n_ff) != 1L) {
        stop("llama scalar hyperparameters must have length one")
    }
    if (length(n_head_kv) == 1L) {
        n_head_kv <- rep.int(n_head_kv, n_layer)
    }
    if (length(n_head_kv) != n_layer || any(n_head_kv < 1L)) {
        stop("llama attention.head_count_kv must have length one or block_count")
    }
    if (n_embd %% n_head != 0L || any(n_head %% n_head_kv != 0L)) {
        stop("llama embedding/head dimensions are inconsistent")
    }
    head_dim <- n_embd %/% n_head
    rms_eps <- as.numeric(key("attention.layer_norm_rms_epsilon", 1e-5))
    rope_base <- as.numeric(key("rope.freq_base", 10000))
    rope_dims <- .rllm_positive(
        key("rope.dimension_count", head_dim), "llama.rope.dimension_count"
    )
    if (length(rms_eps) != 1L || !is.finite(rms_eps) || rms_eps <= 0 ||
        length(rope_base) != 1L || !is.finite(rope_base) || rope_base <= 0 ||
        length(rope_dims) != 1L || rope_dims > head_dim) {
        stop("invalid llama normalization or RoPE parameters")
    }
    if (is.null(rope_mode)) rope_mode <- 0L
    rope_mode <- as.integer(rope_mode)
    if (length(rope_mode) != 1L || is.na(rope_mode) ||
        !rope_mode %in% c(0L, 2L)) {
        stop("rope_mode must be 0 (normal) or 2 (NEOX)")
    }

    embedding_shape <- .rllm_embedding_shape(directory, arch, n_embd)
    n_vocab <- embedding_shape[[2L]]
    embedding <- .rllm_architecture_parameter(
        "token_embd.weight", "token_embedding", embedding_shape
    )
    output_norm <- .rllm_architecture_parameter(
        "output_norm.weight", "output_norm", n_embd
    )
    output <- if ("output.weight" %in% directory$name) {
        .rllm_architecture_parameter(
            "output.weight", "output_projection", c(n_embd, n_vocab)
        )
    } else {
        embedding
    }

    x <- .rllm_program_entry(embedding, 1)
    for (index in seq_len(n_layer)) {
        il <- index - 1L
        prefix <- paste0("blk.", il, ".")
        parameter <- function(suffix, role, shape) {
            .rllm_architecture_parameter(
                paste0(prefix, suffix), paste0("layer.", il, ".", role),
                shape
            )
        }
        kv_dim <- head_dim * n_head_kv[[index]]
        operator_norm <- parameter("attn_norm.weight", "operator_norm", n_embd)
        query <- parameter("attn_q.weight", "attention.query",
                           c(n_embd, n_embd))
        key_weight <- parameter("attn_k.weight", "attention.key",
                                c(n_embd, kv_dim))
        value <- parameter("attn_v.weight", "attention.value",
                           c(n_embd, kv_dim))
        attention_output <- parameter(
            "attn_output.weight", "attention.output", c(n_embd, n_embd)
        )
        ffn_norm <- parameter("ffn_norm.weight", "ffn_norm", n_embd)
        gate <- parameter("ffn_gate.weight", "ffn.gate", c(n_embd, n_ff))
        up <- parameter("ffn_up.weight", "ffn.up", c(n_embd, n_ff))
        down <- parameter("ffn_down.weight", "ffn.down", c(n_ff, n_embd))

        x <- .rllm_program_block(
            x, il, operator_norm,
            list(
                op = "attention", query = query, key = key_weight,
                value = value, output = attention_output,
                query_norm = NULL, key_norm = NULL,
                n_head = n_head, n_head_kv = n_head_kv[[index]],
                head_dim = head_dim,
                rope = list(mode = rope_mode, dims = rope_dims,
                            base = rope_base),
                scale = list(at = "logits", value = 1 / sqrt(head_dim)),
                mask = list(type = "causal", window = 0L)
            ),
            list(op = "kv", width = kv_dim), ffn_norm,
            list(op = "swiglu", gate = gate, up = up, down = down,
                 width = n_ff),
            rms_eps
        )
    }
    x <- rllm_norm(x, output_norm, eps = rms_eps)
    x <- rllm_linear(x, output)

    list(
        program = rllm_program(x, arch),
        symbols = list(
            n_layer = n_layer, n_embd = n_embd, n_head = n_head,
            n_head_kv = n_head_kv, n_ff = n_ff, n_vocab = n_vocab,
            rms_eps = rms_eps, rope_base = rope_base, rope_dims = rope_dims
        )
    )
}

.rllm_program_qwen35 <- function(metadata, directory, rope_mode) {
    arch <- "qwen35"
    key <- function(name, default = .rllm_missing) {
        .rllm_metadata(metadata, arch, name, default)
    }
    scalar <- function(name, default = .rllm_missing) {
        .rllm_architecture_scalar(key, arch, name, default = default)
    }

    n_layer <- scalar("block_count")
    n_embd <- scalar("embedding_length")
    n_head <- scalar("attention.head_count")
    n_head_kv <- scalar("attention.head_count_kv")
    n_ff <- scalar("feed_forward_length")
    head_dim <- scalar("attention.key_length")
    value_dim <- scalar("attention.value_length")
    conv_kernel <- scalar("ssm.conv_kernel")
    state_dim <- scalar("ssm.state_size")
    group_count <- scalar("ssm.group_count")
    value_heads <- scalar("ssm.time_step_rank")
    inner_dim <- scalar("ssm.inner_size")
    full_interval <- scalar("full_attention_interval", 4L)
    rope_dims <- scalar("rope.dimension_count", head_dim)
    rope_sections <- .rllm_integer(
        key("rope.dimension_sections"), "qwen35.rope.dimension_sections"
    )
    rms_eps <- as.numeric(key("attention.layer_norm_rms_epsilon", 1e-6))
    rope_base <- as.numeric(key("rope.freq_base", 1e7))

    if (n_head %% n_head_kv != 0L || head_dim != value_dim ||
        inner_dim != n_head * head_dim || inner_dim %% value_heads != 0L ||
        inner_dim %/% value_heads != state_dim ||
        value_heads %% group_count != 0L) {
        stop("qwen35 attention and recurrent-state dimensions are inconsistent")
    }
    if (conv_kernel < 2L || length(rope_sections) != 4L ||
        sum(rope_sections) * 2L != rope_dims ||
        length(rms_eps) != 1L || !is.finite(rms_eps) || rms_eps <= 0 ||
        length(rope_base) != 1L || !is.finite(rope_base) || rope_base <= 0) {
        stop("invalid qwen35 convolution, normalization, or MRoPE parameters")
    }
    if (!is.null(rope_mode)) {
        stop("qwen35 uses metadata-defined MRoPE; rope_mode cannot be overridden")
    }

    recurrent <- key("attention.recurrent_layers", NULL)
    if (is.null(recurrent)) {
        recurrent <- seq_len(n_layer) %% full_interval != 0L
    } else {
        if (!(is.logical(recurrent) || is.numeric(recurrent)) ||
            length(recurrent) != n_layer || anyNA(recurrent) ||
            any(!recurrent %in% c(FALSE, TRUE, 0, 1))) {
            stop(paste0(
                "qwen35.attention.recurrent_layers must contain one flag ",
                "per block"
            ))
        }
        recurrent <- as.logical(recurrent)
    }

    embedding_shape <- .rllm_embedding_shape(directory, arch, n_embd)
    n_vocab <- embedding_shape[[2L]]
    embedding <- .rllm_architecture_parameter(
        "token_embd.weight", "token_embedding", embedding_shape
    )
    output_norm <- .rllm_architecture_parameter(
        "output_norm.weight", "output_norm", n_embd
    )
    output <- if ("output.weight" %in% directory$name) {
        .rllm_architecture_parameter(
            "output.weight", "output_projection", c(n_embd, n_vocab)
        )
    } else {
        embedding
    }

    kv_dim <- head_dim * n_head_kv
    key_width <- state_dim * group_count
    conv_width <- 2L * key_width + inner_dim
    x <- .rllm_program_entry(embedding, 1)
    for (index in seq_len(n_layer)) {
        il <- index - 1L
        prefix <- paste0("blk.", il, ".")
        parameter <- function(suffix, role, shape) {
            .rllm_architecture_parameter(
                paste0(prefix, suffix), paste0("layer.", il, ".", role),
                shape
            )
        }
        operator_norm <- parameter(
            "attn_norm.weight", "operator_norm", n_embd
        )
        ffn_norm <- parameter(
            "post_attention_norm.weight", "ffn_norm", n_embd
        )
        gate <- parameter("ffn_gate.weight", "ffn.gate", c(n_embd, n_ff))
        up <- parameter("ffn_up.weight", "ffn.up", c(n_embd, n_ff))
        down <- parameter("ffn_down.weight", "ffn.down", c(n_ff, n_embd))

        if (recurrent[[index]]) {
            qkv <- parameter(
                "attn_qkv.weight", "gated_delta.qkv",
                c(n_embd, conv_width)
            )
            output_gate <- parameter(
                "attn_gate.weight", "gated_delta.output_gate",
                c(n_embd, inner_dim)
            )
            convolution <- parameter(
                "ssm_conv1d.weight", "gated_delta.convolution",
                c(conv_kernel, conv_width)
            )
            time_bias <- parameter(
                "ssm_dt.bias", "gated_delta.time_bias", value_heads
            )
            decay <- parameter("ssm_a", "gated_delta.decay", value_heads)
            beta <- parameter(
                "ssm_beta.weight", "gated_delta.beta",
                c(n_embd, value_heads)
            )
            alpha <- parameter(
                "ssm_alpha.weight", "gated_delta.alpha",
                c(n_embd, value_heads)
            )
            delta_norm <- parameter(
                "ssm_norm.weight", "gated_delta.output_norm", state_dim
            )
            delta_output <- parameter(
                "ssm_out.weight", "gated_delta.output",
                c(inner_dim, n_embd)
            )
            operator <- list(
                op = "gated_delta_net", qkv = qkv,
                output_gate = output_gate, convolution = convolution,
                time_bias = time_bias, decay = decay, beta = beta,
                alpha = alpha, norm = delta_norm, output = delta_output,
                key_heads = group_count, value_heads = value_heads,
                key_head_dim = state_dim, value_head_dim = state_dim,
                convolution_width = conv_width,
                convolution_kernel = conv_kernel,
                qk_norm = list(kind = "l2", eps = rms_eps),
                activations = list(
                    convolution = "silu", beta = "sigmoid",
                    time_step = "softplus", output_gate = "silu"
                )
            )
            state <- list(
                op = "gated_delta",
                matrix = c(row = state_dim, column = state_dim,
                           head = value_heads),
                convolution = c(width = conv_width,
                                history = conv_kernel - 1L)
            )
        } else {
            query_gate <- parameter(
                "attn_q.weight", "attention.query_gate",
                c(n_embd, 2L * inner_dim)
            )
            key_weight <- parameter(
                "attn_k.weight", "attention.key", c(n_embd, kv_dim)
            )
            value <- parameter(
                "attn_v.weight", "attention.value", c(n_embd, kv_dim)
            )
            attention_output <- parameter(
                "attn_output.weight", "attention.output",
                c(inner_dim, n_embd)
            )
            query_norm <- parameter(
                "attn_q_norm.weight", "attention.query_norm", head_dim
            )
            key_norm <- parameter(
                "attn_k_norm.weight", "attention.key_norm", head_dim
            )
            operator <- list(
                op = "gated_attention", query_gate = query_gate,
                key = key_weight, value = value, output = attention_output,
                query_norm = query_norm, key_norm = key_norm,
                query_gate_layout = "head_interleaved",
                gate_activation = "sigmoid", n_head = n_head,
                n_head_kv = n_head_kv, head_dim = head_dim,
                rope = list(mode = "mrope", dims = rope_dims,
                            sections = rope_sections, base = rope_base),
                scale = list(at = "logits", value = 1 / sqrt(head_dim)),
                mask = list(type = "causal", window = 0L)
            )
            state <- list(op = "kv", width = kv_dim)
        }

        x <- .rllm_program_block(
            x, il, operator_norm, operator, state, ffn_norm,
            list(op = "swiglu", gate = gate, up = up, down = down,
                 width = n_ff),
            rms_eps
        )
    }
    x <- rllm_norm(x, output_norm, eps = rms_eps)
    x <- rllm_linear(x, output)

    list(
        program = rllm_program(x, arch),
        symbols = list(
            n_layer = n_layer, n_embd = n_embd, n_head = n_head,
            n_head_kv = n_head_kv, n_ff = n_ff, n_vocab = n_vocab,
            head_dim = head_dim, rms_eps = rms_eps, rope_base = rope_base,
            rope_dims = rope_dims, rope_sections = rope_sections,
            recurrent = recurrent, full_attention_interval = full_interval,
            ssm_state_size = state_dim, ssm_group_count = group_count,
            ssm_value_heads = value_heads, ssm_inner_size = inner_dim,
            ssm_conv_kernel = conv_kernel
        )
    )
}

.rllm_program_lfm2moe <- function(metadata, directory, rope_mode) {
    arch <- "lfm2moe"
    key <- function(name, default = .rllm_missing) {
        .rllm_metadata(metadata, arch, name, default)
    }
    scalar <- function(name, positive = TRUE, default = .rllm_missing) {
        .rllm_architecture_scalar(
            key, arch, name, positive = positive, default = default
        )
    }

    n_layer <- scalar("block_count")
    n_embd <- scalar("embedding_length")
    n_head <- scalar("attention.head_count")
    n_ff <- scalar("feed_forward_length")
    n_head_kv <- .rllm_integer(
        key("attention.head_count_kv"),
        "lfm2moe.attention.head_count_kv"
    )
    if (length(n_head_kv) != n_layer) {
        stop("lfm2moe.attention.head_count_kv must have block_count entries")
    }
    if (n_embd %% n_head != 0L ||
        any(n_head_kv > 0L & n_head %% n_head_kv != 0L)) {
        stop("lfm2moe embedding/head dimensions are inconsistent")
    }
    head_dim <- n_embd %/% n_head
    n_expert <- scalar("expert_count")
    n_expert_used <- scalar("expert_used_count")
    n_ff_expert <- scalar("expert_feed_forward_length")
    n_dense <- scalar("leading_dense_block_count", positive = FALSE,
                      default = 0)
    l_cache <- scalar("shortconv.l_cache")
    gating <- scalar("expert_gating_func", positive = FALSE)
    if (n_expert_used > n_expert || n_dense > n_layer || l_cache < 2L) {
        stop(paste0(
            "invalid lfm2moe expert, dense-layer, or short-convolution ",
            "parameters"
        ))
    }
    if (gating != 2L) {
        stop("lfm2moe expert_gating_func ", gating,
             " is unsupported; the program vocabulary expects sigmoid routing")
    }
    rms_eps <- as.numeric(key("attention.layer_norm_rms_epsilon", 1e-5))
    rope_base <- as.numeric(key("rope.freq_base", 5e6))
    if (length(rms_eps) != 1L || !is.finite(rms_eps) || rms_eps <= 0 ||
        length(rope_base) != 1L || !is.finite(rope_base) || rope_base <= 0) {
        stop("invalid lfm2moe normalization or RoPE parameters")
    }
    if (is.null(rope_mode)) rope_mode <- 2L
    rope_mode <- as.integer(rope_mode)
    if (length(rope_mode) != 1L || is.na(rope_mode) || rope_mode != 2L) {
        stop("lfm2moe uses NEOX RoPE; rope_mode must be 2")
    }

    embedding_shape <- .rllm_embedding_shape(directory, arch, n_embd)
    n_vocab <- embedding_shape[[2L]]
    embedding <- .rllm_architecture_parameter(
        "token_embd.weight", "token_embedding", embedding_shape
    )
    output_norm <- .rllm_architecture_parameter(
        "token_embd_norm.weight", "output_norm", n_embd
    )
    output <- if ("output.weight" %in% directory$name) {
        .rllm_architecture_parameter(
            "output.weight", "output_projection", c(n_embd, n_vocab)
        )
    } else {
        embedding
    }

    x <- .rllm_program_entry(embedding, 1)
    for (index in seq_len(n_layer)) {
        il <- index - 1L
        prefix <- paste0("blk.", il, ".")
        parameter <- function(suffix, role, shape) {
            .rllm_architecture_parameter(
                paste0(prefix, suffix), paste0("layer.", il, ".", role),
                shape
            )
        }
        operator_norm <- parameter(
            "attn_norm.weight", "operator_norm", n_embd
        )
        ffn_norm <- parameter("ffn_norm.weight", "ffn_norm", n_embd)

        if (n_head_kv[[index]] == 0L) {
            input <- parameter(
                "shortconv.in_proj.weight", "shortconv.input",
                c(n_embd, 3L * n_embd)
            )
            kernel <- parameter(
                "shortconv.conv.weight", "shortconv.kernel",
                c(l_cache, n_embd)
            )
            shortconv_output <- parameter(
                "shortconv.out_proj.weight", "shortconv.output",
                c(n_embd, n_embd)
            )
            operator <- list(
                op = "shortconv", input = input, kernel = kernel,
                output = shortconv_output, width = n_embd,
                l_cache = l_cache
            )
            state <- list(
                op = "conv", width = n_embd, history = l_cache - 1L
            )
        } else {
            kv_dim <- head_dim * n_head_kv[[index]]
            query <- parameter(
                "attn_q.weight", "attention.query", c(n_embd, n_embd)
            )
            key_weight <- parameter(
                "attn_k.weight", "attention.key", c(n_embd, kv_dim)
            )
            value <- parameter(
                "attn_v.weight", "attention.value", c(n_embd, kv_dim)
            )
            attention_output <- parameter(
                "attn_output.weight", "attention.output",
                c(n_embd, n_embd)
            )
            query_norm <- parameter(
                "attn_q_norm.weight", "attention.query_norm", head_dim
            )
            key_norm <- parameter(
                "attn_k_norm.weight", "attention.key_norm", head_dim
            )
            operator <- list(
                op = "attention", query = query, key = key_weight,
                value = value, output = attention_output,
                query_norm = query_norm, key_norm = key_norm,
                n_head = n_head, n_head_kv = n_head_kv[[index]],
                head_dim = head_dim,
                rope = list(mode = rope_mode, dims = head_dim,
                            base = rope_base),
                scale = list(at = "logits", value = 1 / sqrt(head_dim)),
                mask = list(type = "causal", window = 0L)
            )
            state <- list(op = "kv", width = kv_dim)
        }

        if (il < n_dense) {
            gate <- parameter(
                "ffn_gate.weight", "ffn.gate", c(n_embd, n_ff)
            )
            up <- parameter("ffn_up.weight", "ffn.up", c(n_embd, n_ff))
            down <- parameter(
                "ffn_down.weight", "ffn.down", c(n_ff, n_embd)
            )
            feed_forward <- list(
                op = "swiglu", gate = gate, up = up, down = down,
                width = n_ff
            )
        } else {
            router <- parameter(
                "ffn_gate_inp.weight", "moe.router",
                c(n_embd, n_expert)
            )
            selection_bias <- parameter(
                "exp_probs_b.bias", "moe.selection_bias", n_expert
            )
            gate <- parameter(
                "ffn_gate_exps.weight", "moe.gate_experts",
                c(n_embd, n_ff_expert, n_expert)
            )
            up <- parameter(
                "ffn_up_exps.weight", "moe.up_experts",
                c(n_embd, n_ff_expert, n_expert)
            )
            down <- parameter(
                "ffn_down_exps.weight", "moe.down_experts",
                c(n_ff_expert, n_embd, n_expert)
            )
            feed_forward <- list(
                op = "moe_swiglu", router = router,
                selection_bias = selection_bias, gate = gate, up = up,
                down = down, routing = "sigmoid", experts = n_expert,
                selected = n_expert_used, width = n_ff_expert,
                normalize_selected = TRUE, scale = 1
            )
        }

        x <- .rllm_program_block(
            x, il, operator_norm, operator, state, ffn_norm,
            feed_forward, rms_eps
        )
    }
    x <- rllm_norm(x, output_norm, eps = rms_eps)
    x <- rllm_linear(x, output)

    list(
        program = rllm_program(x, arch),
        symbols = list(
            n_layer = n_layer, n_embd = n_embd, n_head = n_head,
            n_head_kv = n_head_kv, n_ff = n_ff, n_vocab = n_vocab,
            rms_eps = rms_eps, rope_base = rope_base, rope_dims = head_dim,
            n_expert = n_expert, n_expert_used = n_expert_used,
            n_ff_expert = n_ff_expert, n_dense = n_dense,
            shortconv_l_cache = l_cache
        )
    )
}

.rllm_program_gemma_embedding <- function(metadata, directory, rope_mode) {
    arch <- "gemma-embedding"
    key <- function(name, default = .rllm_missing) {
        .rllm_metadata(metadata, arch, name, default)
    }
    scalar <- function(name, positive = TRUE, default = .rllm_missing) {
        .rllm_architecture_scalar(
            key, arch, name, positive = positive, default = default
        )
    }

    n_layer <- scalar("block_count")
    n_embd <- scalar("embedding_length")
    n_head <- scalar("attention.head_count")
    n_head_kv <- scalar("attention.head_count_kv")
    n_ff <- scalar("feed_forward_length")
    head_dim <- scalar("attention.key_length", default = n_embd %/% n_head)
    value_dim <- scalar("attention.value_length", default = head_dim)
    if (n_embd %% n_head != 0L || head_dim * n_head != n_embd ||
        value_dim != head_dim || n_head %% n_head_kv != 0L) {
        stop("gemma-embedding attention dimensions are inconsistent")
    }
    rms_eps <- as.numeric(key("attention.layer_norm_rms_epsilon", 1e-6))
    rope_base <- as.numeric(key("rope.freq_base", 1e6))
    rope_base_swa <- as.numeric(key("rope.freq_base_swa", 10000))
    sliding_window <- scalar("attention.sliding_window")
    sliding_period <- scalar("attention.sliding_window_pattern", default = 6)
    pooling <- scalar("pooling_type", positive = FALSE)
    if (length(rms_eps) != 1L || !is.finite(rms_eps) || rms_eps <= 0 ||
        length(rope_base) != 1L || !is.finite(rope_base) || rope_base <= 0 ||
        length(rope_base_swa) != 1L || !is.finite(rope_base_swa) ||
        rope_base_swa <= 0 || sliding_window < 2L || pooling != 1L) {
        stop(paste0(
            "invalid gemma-embedding normalization, attention, or pooling ",
            "parameters"
        ))
    }
    if (is.null(rope_mode)) rope_mode <- 2L
    rope_mode <- as.integer(rope_mode)
    if (length(rope_mode) != 1L || is.na(rope_mode) || rope_mode != 2L) {
        stop("gemma-embedding uses NEOX RoPE; rope_mode must be 2")
    }

    embedding_shape <- .rllm_embedding_shape(directory, arch, n_embd)
    n_vocab <- embedding_shape[[2L]]
    embedding <- .rllm_architecture_parameter(
        "token_embd.weight", "token_embedding", embedding_shape
    )
    output_norm <- .rllm_architecture_parameter(
        "output_norm.weight", "output_norm", n_embd
    )

    dense_2_in <- scalar("dense_2_feat_in", positive = FALSE, default = 0)
    dense_2_out <- scalar("dense_2_feat_out", positive = FALSE, default = 0)
    dense_3_in <- scalar("dense_3_feat_in", positive = FALSE, default = 0)
    dense_3_out <- scalar("dense_3_feat_out", positive = FALSE, default = 0)
    has_dense <- all(c("dense_2.weight", "dense_3.weight") %in% directory$name)
    if (has_dense) {
        if (dense_2_in != n_embd || dense_2_out < 1L ||
            dense_3_in != dense_2_out || dense_3_out != n_embd) {
            stop("gemma-embedding dense projection metadata are inconsistent")
        }
        projection_1 <- .rllm_architecture_parameter(
            "dense_2.weight", "embedding_projection.1",
            c(dense_2_in, dense_2_out)
        )
        projection_2 <- .rllm_architecture_parameter(
            "dense_3.weight", "embedding_projection.2",
            c(dense_3_in, dense_3_out)
        )
        output_dim <- dense_3_out
    } else {
        if (any(c(dense_2_in, dense_2_out, dense_3_in, dense_3_out) != 0L)) {
            stop(paste0(
                "gemma-embedding declares dense projections without both ",
                "tensors"
            ))
        }
        projection_1 <- NULL
        projection_2 <- NULL
        output_dim <- n_embd
    }

    x <- .rllm_program_entry(embedding, sqrt(n_embd))
    for (index in seq_len(n_layer)) {
        il <- index - 1L
        prefix <- paste0("blk.", il, ".")
        parameter <- function(suffix, role, shape) {
            .rllm_architecture_parameter(
                paste0(prefix, suffix), paste0("layer.", il, ".", role),
                shape
            )
        }
        operator_norm <- parameter(
            "attn_norm.weight", "operator_norm", n_embd
        )
        query <- parameter(
            "attn_q.weight", "attention.query", c(n_embd, n_embd)
        )
        key_weight <- parameter(
            "attn_k.weight", "attention.key",
            c(n_embd, head_dim * n_head_kv)
        )
        value <- parameter(
            "attn_v.weight", "attention.value",
            c(n_embd, value_dim * n_head_kv)
        )
        attention_output <- parameter(
            "attn_output.weight", "attention.output", c(n_embd, n_embd)
        )
        query_norm <- parameter(
            "attn_q_norm.weight", "attention.query_norm", head_dim
        )
        key_norm <- parameter(
            "attn_k_norm.weight", "attention.key_norm", head_dim
        )
        operator_post_norm <- parameter(
            "post_attention_norm.weight", "operator_post_norm", n_embd
        )
        ffn_norm <- parameter("ffn_norm.weight", "ffn_norm", n_embd)
        gate <- parameter("ffn_gate.weight", "ffn.gate", c(n_embd, n_ff))
        up <- parameter("ffn_up.weight", "ffn.up", c(n_embd, n_ff))
        down <- parameter("ffn_down.weight", "ffn.down", c(n_ff, n_embd))
        feed_forward_post_norm <- parameter(
            "post_ffw_norm.weight", "ffn_post_norm", n_embd
        )

        is_swa <- il %% sliding_period < sliding_period - 1L
        operator <- list(
            op = "attention", query = query, key = key_weight,
            value = value, output = attention_output,
            query_norm = query_norm, key_norm = key_norm,
            n_head = n_head, n_head_kv = n_head_kv,
            head_dim = head_dim,
            rope = list(
                mode = rope_mode, dims = head_dim,
                base = if (is_swa) rope_base_swa else rope_base
            ),
            scale = list(at = "query", value = 1 / sqrt(head_dim)),
            mask = list(
                type = if (is_swa) "symmetric_window" else "bidirectional",
                window = if (is_swa) sliding_window else 0L
            )
        )
        feed_forward <- list(
            op = "geglu", gate = gate, up = up, down = down,
            width = n_ff
        )
        x <- .rllm_program_block(
            x, il, operator_norm, operator, list(op = "none"),
            ffn_norm, feed_forward, rms_eps,
            operator_post_norm = operator_post_norm,
            feed_forward_post_norm = feed_forward_post_norm
        )
    }
    x <- rllm_norm(x, output_norm, eps = rms_eps)
    x <- rllm_pool(x, "mean")
    if (!is.null(projection_1)) {
        x <- rllm_linear(x, projection_1)
        x <- rllm_linear(x, projection_2)
    }

    list(
        program = rllm_program(x, arch),
        symbols = list(
            n_layer = n_layer, n_embd = n_embd, n_head = n_head,
            n_head_kv = n_head_kv, n_ff = n_ff, n_vocab = n_vocab,
            rms_eps = rms_eps, rope_base = rope_base, rope_dims = head_dim,
            sliding_window = sliding_window,
            sliding_period = sliding_period, output_dim = output_dim
        )
    )
}
