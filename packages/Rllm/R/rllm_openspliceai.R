.rllm_openspliceai_program <- function(
    n_block, n_embd, n_input, n_output, context_length,
    kernel_size, dilation, batch_norm_eps, leaky_relu_slope
) {
    sequence_shape <- c(
        feature = n_input, sequence = "n_sequence", batch = "batch"
    )
    input <- rllm_input("sequence", sequence_shape, "f32")
    parameter <- function(name, shape, role) {
        .rllm_architecture_parameter(name, role, shape)
    }
    convolution <- function(x, prefix, input_width, output_width, kernel,
                            rate, role) {
        weight <- parameter(
            paste0(prefix, ".weight"),
            c(kernel, input_width, output_width),
            paste0(role, ".weight")
        )
        bias <- parameter(
            paste0(prefix, ".bias"), output_width,
            paste0(role, ".bias")
        )
        value <- .rllm_one_value(x)
        shape <- value$shape
        shape[[1L]] <- output_width
        rllm_op(
            value, "conv1d", weight = weight, bias = bias,
            dilation = rate, padding = rate * (kernel - 1L) %/% 2L,
            stride = 1L, output_shape = shape
        )
    }
    batch_norm <- function(x, prefix, role) {
        component <- function(name) {
            parameter(
                paste0(prefix, ".", name), n_embd,
                paste0(role, ".", name)
            )
        }
        rllm_op(
            x, "batch_norm",
            weight = component("weight"),
            bias = component("bias"),
            running_mean = component("running_mean"),
            running_var = component("running_var"),
            eps = batch_norm_eps
        )
    }
    activation <- function(x) {
        rllm_op(x, "leaky_relu", slope = leaky_relu_slope)
    }

    hidden <- convolution(
        input, "initial_conv", n_input, n_embd, 1L, 1L,
        "input_projection"
    )
    skip <- convolution(
        hidden, "initial_skip.conv", n_embd, n_embd, 1L, 1L,
        "initial_skip"
    )

    module_index <- 0L
    for (block_index in seq_len(n_block)) {
        prefix <- paste0("residual_units.", module_index, ".")
        block <- rllm_module(
            paste0("residual.", block_index - 1L),
            function(x) {
                x |>
                    batch_norm(
                        paste0(prefix, "batchnorm1"),
                        paste0("block.", block_index - 1L, ".norm1")
                    ) |>
                    activation() |>
                    convolution(
                        paste0(prefix, "conv1"), n_embd, n_embd,
                        kernel_size[[block_index]],
                        dilation[[block_index]],
                        paste0("block.", block_index - 1L, ".conv1")
                    ) |>
                    batch_norm(
                        paste0(prefix, "batchnorm2"),
                        paste0("block.", block_index - 1L, ".norm2")
                    ) |>
                    activation() |>
                    convolution(
                        paste0(prefix, "conv2"), n_embd, n_embd,
                        kernel_size[[block_index]],
                        dilation[[block_index]],
                        paste0("block.", block_index - 1L, ".conv2")
                    )
            }
        )
        hidden <- rllm_residual(hidden, block)
        module_index <- module_index + 1L

        if (block_index %% 4L == 0L) {
            update <- convolution(
                hidden,
                paste0("residual_units.", module_index, ".conv"),
                n_embd, n_embd, 1L, 1L,
                paste0("skip.", block_index %/% 4L)
            )
            skip <- rllm_op(list(skip = skip, update = update), "add")
            module_index <- module_index + 1L
        }
    }

    cropped <- rllm_op(
        skip, "crop1d",
        left = context_length %/% 2L,
        right = context_length %/% 2L,
        output_shape = c(
            feature = n_embd, sequence = "n_output", batch = "batch"
        )
    )
    logits <- convolution(
        cropped, "final_conv", n_embd, n_output, 1L, 1L,
        "output_projection"
    )
    probabilities <- rllm_op(logits, "softmax", axis = "feature")
    rllm_program(list(probabilities = probabilities), "openspliceai")
}

.rllm_program_openspliceai <- function(metadata, directory, rope_mode) {
    arch <- "openspliceai"
    key <- function(name, default = .rllm_missing) {
        .rllm_metadata(metadata, arch, name, default)
    }
    scalar <- function(name, positive = TRUE, default = .rllm_missing) {
        .rllm_architecture_scalar(
            key, arch, name, positive = positive, default = default
        )
    }
    if (!is.null(rope_mode)) stop("openspliceai does not use rotary positions")

    n_block <- scalar("block_count")
    n_embd <- scalar("embedding_length")
    n_input <- scalar("input_channel_count", default = 4L)
    n_output <- scalar("output_channel_count", default = 3L)
    context_length <- scalar("context_length")
    kernel_size <- as.numeric(key("convolution.kernel_size"))
    dilation <- as.numeric(key("convolution.dilation"))
    batch_norm_eps <- as.numeric(key("batch_norm_epsilon", 1e-5))
    leaky_relu_slope <- as.numeric(key("leaky_relu_slope", 0.1))

    if (length(kernel_size) != n_block || length(dilation) != n_block ||
        anyNA(kernel_size) || anyNA(dilation) ||
        any(!is.finite(kernel_size)) || any(!is.finite(dilation)) ||
        any(kernel_size < 1) || any(dilation < 1) ||
        any(kernel_size != floor(kernel_size)) ||
        any(dilation != floor(dilation)) ||
        any(kernel_size > .Machine$integer.max) ||
        any(dilation > .Machine$integer.max) ||
        any(kernel_size %% 2 != 1)) {
        stop("openspliceai convolution schedule is invalid")
    }
    kernel_size <- as.integer(kernel_size)
    dilation <- as.integer(dilation)
    derived_context <- 2 * sum(as.double(dilation) * (kernel_size - 1L))
    if (context_length %% 2L != 0L || context_length != derived_context) {
        stop("openspliceai context length disagrees with its convolution schedule")
    }
    if (length(batch_norm_eps) != 1L || !is.finite(batch_norm_eps) ||
        batch_norm_eps <= 0 || length(leaky_relu_slope) != 1L ||
        !is.finite(leaky_relu_slope) || leaky_relu_slope < 0) {
        stop("openspliceai normalization or activation metadata is invalid")
    }

    program <- .rllm_openspliceai_program(
        n_block, n_embd, n_input, n_output, context_length,
        kernel_size, dilation, batch_norm_eps, leaky_relu_slope
    )
    list(
        program = program,
        symbols = list(
            n_block = n_block, n_embd = n_embd,
            n_input = n_input, n_output = n_output,
            context_length = context_length,
            kernel_size = kernel_size, dilation = dilation,
            batch_norm_eps = batch_norm_eps,
            leaky_relu_slope = leaky_relu_slope
        )
    )
}

.rllm_execute_conv1d <- function(inputs, attributes, parameters, context) {
    x <- inputs[[1L]]
    dimensions <- dim(x)
    if (is.null(dimensions) || !length(dimensions) %in% c(2L, 3L) ||
        any(dimensions < 1L)) {
        stop("conv1d input must have feature, sequence, and optional batch axes")
    }
    had_batch <- length(dimensions) == 3L
    if (!had_batch) {
        dimensions <- c(dimensions, 1L)
        dim(x) <- dimensions
    }
    weight <- .rllm_execute_parameter(
        attributes$weight, parameters, "conv1d weight"
    )
    weight_dim <- dim(weight)
    if (is.null(weight_dim) || length(weight_dim) != 3L ||
        weight_dim[[2L]] != dimensions[[1L]]) {
        stop("conv1d weight does not match the input feature dimension")
    }
    kernel <- weight_dim[[1L]]
    output_width <- weight_dim[[3L]]
    dilation <- .rllm_execute_count(
        attributes$dilation, list(), "conv1d dilation"
    )
    stride <- .rllm_execute_count(
        attributes$stride, list(), "conv1d stride"
    )
    padding <- .rllm_execute_count(
        attributes$padding, list(), "conv1d padding", zero = TRUE
    )
    extent <- as.double(dimensions[[2L]]) + 2 * padding -
        as.double(dilation) * (kernel - 1L)
    if (!is.finite(extent) || extent < 1) {
        stop("conv1d kernel exceeds the padded input")
    }
    output_length <- floor((extent - 1) / stride) + 1
    if (output_length > .Machine$integer.max) {
        stop("conv1d output sequence is too long")
    }
    output_length <- as.integer(output_length)
    out <- array(0, dim = c(output_width, output_length, dimensions[[3L]]))

    for (batch in seq_len(dimensions[[3L]])) {
        for (at in seq_len(kernel)) {
            source <- (seq_len(output_length) - 1) * as.double(stride) -
                padding + (at - 1) * as.double(dilation) + 1
            valid <- which(source >= 1L & source <= dimensions[[2L]])
            if (!length(valid)) next
            kernel_weight <- weight[at, , , drop = FALSE]
            dim(kernel_weight) <- c(dimensions[[1L]], output_width)
            input_slice <- x[, source[valid], batch, drop = FALSE]
            dim(input_slice) <- c(dimensions[[1L]], length(valid))
            current <- out[, valid, batch, drop = FALSE]
            dim(current) <- c(output_width, length(valid))
            out[, valid, batch] <- current +
                crossprod(kernel_weight, input_slice)
        }
    }
    if (!is.null(attributes$bias)) {
        bias <- .rllm_execute_parameter(
            attributes$bias, parameters, "conv1d bias"
        )
        if (length(bias) != output_width) {
            stop("conv1d bias does not match the output feature dimension")
        }
        out <- sweep(out, 1L, bias, `+`)
    }
    if (!had_batch) {
        matrix(out[, , 1L], nrow = output_width, ncol = output_length)
    } else {
        out
    }
}

.rllm_execute_batch_norm <- function(inputs, attributes, parameters, context) {
    x <- .rllm_execute_feature_matrix(inputs[[1L]], "batch norm input")
    component <- function(name) {
        value <- .rllm_execute_parameter(
            attributes[[name]], parameters, paste0("batch norm ", name)
        )
        if (length(value) != nrow(x$value)) {
            stop("batch norm ", name, " has the wrong feature dimension")
        }
        value
    }
    weight <- component("weight")
    bias <- component("bias")
    running_mean <- component("running_mean")
    running_var <- component("running_var")
    eps <- attributes$eps
    if (!is.numeric(eps) || length(eps) != 1L || !is.finite(eps) || eps <= 0 ||
        any(!is.finite(running_var)) || any(running_var < 0)) {
        stop("batch norm variance or epsilon is invalid")
    }
    out <- sweep(x$value, 1L, running_mean, `-`)
    out <- sweep(out, 1L, sqrt(running_var + eps), `/`)
    out <- sweep(out, 1L, weight, `*`)
    out <- sweep(out, 1L, bias, `+`)
    .rllm_execute_restore(out, x$dimensions)
}

.rllm_execute_leaky_relu <- function(inputs, attributes, parameters, context) {
    slope <- attributes$slope
    if (!is.numeric(slope) || length(slope) != 1L || !is.finite(slope) ||
        slope < 0) {
        stop("leaky ReLU slope must be one non-negative finite number")
    }
    x <- inputs[[1L]]
    ifelse(x >= 0, x, slope * x)
}

.rllm_execute_crop1d <- function(inputs, attributes, parameters, context) {
    x <- inputs[[1L]]
    dimensions <- dim(x)
    if (is.null(dimensions) || length(dimensions) < 2L) {
        stop("crop1d input must have a sequence axis")
    }
    left <- .rllm_execute_count(
        attributes$left, list(), "crop1d left extent", zero = TRUE
    )
    right <- .rllm_execute_count(
        attributes$right, list(), "crop1d right extent", zero = TRUE
    )
    if (left + right >= dimensions[[2L]]) {
        stop("crop1d extents remove the complete sequence")
    }
    at <- seq.int(left + 1L, dimensions[[2L]] - right)
    index <- rep(list(TRUE), length(dimensions))
    index[[2L]] <- at
    do.call(`[`, c(list(x), index, list(drop = FALSE)))
}

.rllm_execute_softmax <- function(inputs, attributes, parameters, context) {
    if (!identical(attributes$axis, "feature")) {
        stop("the dense softmax reference implements the feature axis only")
    }
    x <- .rllm_execute_feature_matrix(inputs[[1L]], "softmax input")
    maxima <- apply(x$value, 2L, max)
    shifted <- sweep(x$value, 2L, maxima, `-`)
    out <- exp(shifted)
    out <- sweep(out, 2L, colSums(out), `/`)
    .rllm_execute_restore(out, x$dimensions)
}
