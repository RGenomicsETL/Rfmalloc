#' Build a typed model program with ordinary R functions and pipes
#'
#' These functions form the R-facing architecture language. A program is
#' traced from ordinary R calls, including functions returned by
#' `rllm_module()`, residual branches and structured loops. Calling
#' `rllm_program()` freezes the trace as data only: nodes, tensor parameters,
#' shapes, module paths and named outputs. The resulting object contains no R
#' closures, environments or backend pointers.
#'
#' This is an architecture language, not a second tensor runtime. Operators
#' describe semantics; a backend may lower the operators it implements and
#' must reject the rest explicitly.
#'
#' @param name A non-empty input, parameter, module, loop or program name.
#' @param shape An atomic vector describing axes. Dimensions may be positive
#'   numbers or symbolic character strings.
#' @param dtype Storage or value type.
#' @param role Optional semantic role for a parameter.
#' @param forward An ordinary R function whose first argument is a traced
#'   value. Its body may use base R's `|>` pipe.
#' @param x A traced `rllm_value`, a named list of traced values for a
#'   multi-input operator, a model plan, a loaded model, or a GGUF path.
#' @param op A non-empty semantic operator name.
#' @param ... Named, data-only operator attributes, or arguments passed to a
#'   module's `forward` function.
#' @param output_shape Optional result shape. The input shape is retained when
#'   omitted.
#' @param output_dtype Optional result type. The input type is retained when
#'   omitted.
#' @param outputs Optional named output specifications for an operator with
#'   multiple results. Each entry is either a shape or a list with `shape` and
#'   optional `dtype`. This cannot be combined with `output_shape` or
#'   `output_dtype`.
#' @param .dtype One dtype for all inputs, or a named character vector with one
#'   dtype per input.
#' @param .name Program-builder name for a multiple-input trace.
#'
#' @return `rllm_input()` and single-result operators return a traced
#'   `rllm_value`. `rllm_inputs()` and multiple-result operators return named
#'   lists of traced values.
#'   `rllm_parameter()` returns a data-only tensor reference,
#'   `rllm_module()` returns a callable R function, and `rllm_program()`
#'   returns a data-only `rllm_program`.
#' @export
rllm_input <- function(name, shape, dtype = "f32") {
    builder <- .rllm_new_builder(name)
    .rllm_add_input(builder, name, shape, dtype)
}

#' @rdname rllm_input
#' @export
rllm_inputs <- function(..., .dtype = "f32", .name = NULL) {
    shapes <- list(...)
    .rllm_named(shapes, "inputs")
    if (!length(shapes)) stop("inputs must not be empty")
    if (!all(vapply(shapes, is.atomic, logical(1)))) {
        stop("each input must be described by an atomic shape")
    }
    if (!is.character(.dtype) || anyNA(.dtype) || any(!nzchar(.dtype))) {
        stop(".dtype must contain non-empty strings")
    }
    if (length(.dtype) == 1L) {
        .dtype <- rep(.dtype, length(shapes))
        names(.dtype) <- names(shapes)
    } else {
        if (is.null(names(.dtype)) || anyNA(names(.dtype)) ||
            anyDuplicated(names(.dtype)) ||
            !setequal(names(.dtype), names(shapes))) {
            stop("named .dtype entries must match the inputs")
        }
        .dtype <- .dtype[names(shapes)]
    }
    if (is.null(.name)) .name <- names(shapes)[[1L]]
    .rllm_name(.name, "program name")
    builder <- .rllm_new_builder(.name)
    out <- Map(function(name, shape, dtype) {
        .rllm_add_input(builder, name, shape, dtype)
    }, names(shapes), shapes, as.list(.dtype))
    names(out) <- names(shapes)
    out
}

#' @rdname rllm_input
#' @export
rllm_parameter <- function(name, shape, dtype = NULL, role = NULL) {
    .rllm_name(name, "parameter name")
    shape <- .rllm_program_shape(shape)
    if (!is.null(dtype)) .rllm_name(dtype, "parameter dtype")
    if (!is.null(role)) .rllm_name(role, "parameter role")
    structure(list(name = name, shape = shape, dtype = dtype, role = role),
              class = c("rllm_parameter", "list"))
}

#' @rdname rllm_input
#' @export
rllm_module <- function(name, forward) {
    .rllm_name(name, "module name")
    if (!is.function(forward)) stop("forward must be an R function")
    module <- function(x, ...) {
        values <- .rllm_values(x)
        builder <- values[[1L]]$.builder
        .rllm_assert_builder(x, builder)
        old <- builder$module
        builder$module <- c(old, name)
        on.exit(builder$module <- old, add = TRUE)
        out <- forward(x, ...)
        .rllm_assert_builder(out, builder)
        out
    }
    attr(module, "name") <- name
    class(module) <- c("rllm_module", "function")
    module
}

#' @rdname rllm_input
#' @export
rllm_op <- function(x, op, ..., output_shape = NULL,
                    output_dtype = NULL, outputs = NULL) {
    .rllm_name(op, "operator name")
    values <- .rllm_values(x)
    value <- values[[1L]]
    .rllm_assert_builder(x, value$.builder)
    attrs <- list(...)
    .rllm_named(attrs, "operator attributes")
    if (!is.null(outputs)) {
        if (!is.null(output_shape) || !is.null(output_dtype)) {
            stop("outputs cannot be combined with output_shape or output_dtype")
        }
        specs <- .rllm_output_specs(outputs, value$dtype)
        tuple <- .rllm_append(value$.builder, op, values, attrs, NULL, "tuple")
        out <- Map(function(spec, name, index) {
            .rllm_append(
                value$.builder, "op_result", list(tuple),
                list(result = name, index = index), spec$shape, spec$dtype
            )
        }, specs, names(specs), seq_along(specs) - 1L)
        names(out) <- names(specs)
        return(out)
    }
    if (is.null(output_shape)) output_shape <- value$shape
    if (is.null(output_dtype)) output_dtype <- value$dtype
    .rllm_append(value$.builder, op, values, attrs,
                  output_shape, output_dtype)
}

#' Typed linear, normalization, attention and pooling operators
#'
#' These are convenient typed constructors over [rllm_op()]. They keep the
#' surface close to an R module's `forward` method while recording explicit
#' parameter identities and semantic attributes in the frozen program.
#'
#' @param x A traced `rllm_value`.
#' @param weight,bias Tensor references from [rllm_parameter()]. `bias` is
#'   optional.
#' @param kind Operator kind.
#' @param eps Normalization epsilon.
#' @param query,key,value,output Projection parameters for attention.
#' @param heads,kv_heads Query and key/value head counts.
#' @param query_norm,key_norm Optional per-head normalization parameters.
#' @param rope,mask,scale Data-only attention specifications.
#' @param state Data-only persistent-state specification. The default declares
#'   no state.
#'
#' @return A traced `rllm_value`.
#' @export
rllm_linear <- function(x, weight, bias = NULL) {
    value <- .rllm_one_value(x)
    .rllm_expect_parameter(weight, "weight", 2L)
    if (!is.null(bias)) .rllm_expect_parameter(bias, "bias", 1L)
    if (length(value$shape) < 1L ||
        !.rllm_same_dimension(value$shape[[1L]], weight$shape[[1L]])) {
        stop("linear input and weight dimensions do not agree")
    }
    if (!is.null(bias) &&
        !.rllm_same_dimension(bias$shape[[1L]], weight$shape[[2L]])) {
        stop("linear bias and output dimensions do not agree")
    }
    shape <- value$shape
    shape[[1L]] <- weight$shape[[2L]]
    rllm_op(value, "linear", weight = weight, bias = bias,
            output_shape = shape)
}

#' @rdname rllm_linear
#' @export
rllm_norm <- function(x, weight, kind = c("rms", "layer"), eps = 1e-5,
                      bias = NULL) {
    value <- .rllm_one_value(x)
    kind <- match.arg(kind)
    .rllm_expect_parameter(weight, "weight", 1L)
    if (!is.null(bias)) .rllm_expect_parameter(bias, "bias", 1L)
    if (length(value$shape) < 1L ||
        !.rllm_same_dimension(value$shape[[1L]], weight$shape[[1L]]) ||
        (!is.null(bias) &&
         !.rllm_same_dimension(weight$shape[[1L]], bias$shape[[1L]]))) {
        stop("normalization parameters do not match the feature dimension")
    }
    if (!is.numeric(eps) || length(eps) != 1L || !is.finite(eps) || eps <= 0) {
        stop("eps must be one positive finite number")
    }
    rllm_op(value, paste0(kind, "_norm"), weight = weight, bias = bias,
            eps = as.numeric(eps))
}

#' @rdname rllm_linear
#' @export
rllm_attention <- function(x, query, key, value, output, heads,
                           kv_heads = heads, query_norm = NULL,
                           key_norm = NULL, rope = NULL,
                           mask = list(type = "causal"), scale = NULL,
                           state = list(op = "none")) {
    input <- .rllm_one_value(x)
    for (item in list(query = query, key = key, value = value,
                      output = output)) {
        .rllm_expect_parameter(item, "attention projection", 2L)
    }
    for (item in list(query_norm = query_norm, key_norm = key_norm)) {
        if (!is.null(item)) .rllm_expect_parameter(item, "head norm", 1L)
    }
    heads <- .rllm_count(heads, "heads")
    kv_heads <- .rllm_count(kv_heads, "kv_heads")
    if (heads %% kv_heads != 0L) stop("heads must be divisible by kv_heads")
    if (length(input$shape) < 1L ||
        !.rllm_same_dimension(input$shape[[1L]], query$shape[[1L]]) ||
        !.rllm_same_dimension(input$shape[[1L]], key$shape[[1L]]) ||
        !.rllm_same_dimension(input$shape[[1L]], value$shape[[1L]]) ||
        !.rllm_same_dimension(input$shape[[1L]], output$shape[[2L]]) ||
        !.rllm_same_dimension(query$shape[[2L]], output$shape[[1L]]) ||
        !.rllm_same_dimension(key$shape[[2L]], value$shape[[2L]])) {
        stop("attention projections do not preserve the feature dimension")
    }
    q_width <- suppressWarnings(as.numeric(query$shape[[2L]]))
    kv_width <- suppressWarnings(as.numeric(key$shape[[2L]]))
    if (!is.na(q_width) && !is.na(kv_width) &&
        (q_width %% heads != 0 || kv_width %% kv_heads != 0 ||
         q_width / heads != kv_width / kv_heads)) {
        stop("attention projections and head counts have inconsistent widths")
    }
    head_width <- if (!is.na(q_width)) q_width / heads else NULL
    if (!is.null(head_width) &&
        ((!is.null(query_norm) &&
          !.rllm_same_dimension(query_norm$shape[[1L]], head_width)) ||
         (!is.null(key_norm) &&
          !.rllm_same_dimension(key_norm$shape[[1L]], head_width)))) {
        stop("attention head normalization has the wrong width")
    }
    if (!is.list(state) || !is.character(state$op) ||
        length(state$op) != 1L || is.na(state$op) || !nzchar(state$op)) {
        stop("state must contain one non-empty operator name")
    }
    rllm_op(input, "attention", query = query, key = key, value = value,
            output = output, n_head = heads, n_head_kv = kv_heads,
            head_dim = head_width,
            query_norm = query_norm, key_norm = key_norm,
            rope = rope, mask = mask, scale = scale, state = state)
}

#' @rdname rllm_linear
#' @export
rllm_pool <- function(x, kind = c("mean", "cls", "none")) {
    value <- .rllm_one_value(x)
    kind <- match.arg(kind)
    shape <- if (kind == "none") value$shape else value$shape[[1L]]
    axes <- names(value$shape)
    names(shape) <- if (kind == "none") axes else if (length(axes)) axes[[1L]]
    rllm_op(value, "pool", kind = kind, output_shape = shape)
}

#' Trace a residual branch
#'
#' `branch` is evaluated immediately with `x`. Its nodes are recorded, then an
#' explicit add node joins the original value and the branch result.
#'
#' @param x A traced `rllm_value`.
#' @param branch A function of one traced value.
#'
#' @return A traced `rllm_value`.
#' @export
rllm_residual <- function(x, branch) {
    value <- .rllm_one_value(x)
    if (!is.function(branch)) stop("branch must be an R function")
    out <- .rllm_one_value(branch(value))
    if (!identical(out$.builder, value$.builder)) {
        stop("residual branch belongs to a different program")
    }
    if (!.rllm_same_shape(value$shape, out$shape)) {
        stop("residual branch must preserve shape")
    }
    .rllm_append(value$.builder, "add", list(value, out), list(),
                  value$shape, value$dtype)
}

#' Name an intermediate program output
#'
#' @param x A traced `rllm_value`.
#' @param name A unique output name.
#'
#' @return `x`, invisibly annotated in its builder so a pipe may continue.
#' @export
rllm_tap <- function(x, name) {
    value <- .rllm_one_value(x)
    .rllm_name(name, "output name")
    if (!is.null(value$.builder$outputs[[name]])) {
        stop("program output '", name, "' already exists")
    }
    value$.builder$outputs[[name]] <- value$id
    value
}

#' Trace a structured, shared recurrence
#'
#' Unlike an R `for` loop, `rllm_loop()` does not unroll its body. The body is
#' traced once into a nested data-only program and retained as a loop node.
#' A single value supports pipe syntax. A named list supports multiple carried
#' values, including nested recurrences with shared parameters.
#'
#' @param x A traced value or named list of values from one program.
#' @param times A positive integer or one symbolic count.
#' @param body A function returning the same value structure as `x`.
#' @param name Loop name.
#'
#' @return A traced value, or a named list of traced values when `x` is a list.
#' @export
rllm_loop <- function(x, times, body, name = "loop") {
    .rllm_name(name, "loop name")
    if (!(is.numeric(times) && length(times) == 1L && is.finite(times) &&
          times >= 1 && times == floor(times)) &&
        !(is.character(times) && length(times) == 1L && !is.na(times) &&
          nzchar(times))) {
        stop("times must be a positive integer or one symbolic name")
    }
    if (!is.function(body)) stop("body must be an R function")
    many <- is.list(x) && !inherits(x, "rllm_value")
    state <- if (many) x else list(state = .rllm_one_value(x))
    if (many && (is.null(names(state)) || anyNA(names(state)) ||
                 any(!nzchar(names(state))) || anyDuplicated(names(state)))) {
        stop("multiple loop values must have unique names")
    }
    values <- .rllm_values(state)
    builder <- values[[1L]]$.builder
    .rllm_assert_builder(state, builder)

    child <- .rllm_new_builder(name)
    placeholders <- Map(function(value, carry) {
        .rllm_add_input(child, carry, value$shape, value$dtype)
    }, values, names(state))
    names(placeholders) <- names(state)
    traced <- if (many) body(placeholders) else body(placeholders[[1L]])
    result <- if (many) traced else list(state = .rllm_one_value(traced))
    if (many && (!is.list(result) || !identical(names(result), names(state)))) {
        stop("loop body must return the same named carries")
    }
    .rllm_assert_builder(result, child)
    body_program <- .rllm_snapshot(result, name)

    tuple <- .rllm_append(builder, "loop", values,
                          list(name = name, times = times,
                               carries = names(state), body = body_program),
                          if (many) NULL else result[[1L]]$shape,
                          if (many) "tuple" else result[[1L]]$dtype)
    if (!many) return(tuple)
    out <- Map(function(value, carry, index) {
        .rllm_append(builder, "loop_result", list(tuple),
                      list(carry = carry, index = index),
                      value$shape, value$dtype)
    }, result, names(result), seq_along(result) - 1L)
    names(out) <- names(result)
    out
}

#' @rdname rllm_input
#' @export
rllm_program <- function(x, name = NULL) {
    if (inherits(x, "rllm_model")) return(x$execution$program)
    if (inherits(x, "rllm_plan")) return(x$program)
    if (is.character(x) && length(x) == 1L && !is.na(x)) {
        return(.rllm_program_from_gguf(
            Rgguf::gguf_metadata(x), Rgguf::gguf_tensors(x)
        )$program)
    }
    .rllm_snapshot(x, name)
}

#' @export
print.rllm_program <- function(x, ...) {
    operators <- vapply(x$nodes, `[[`, character(1), "op")
    cat(sprintf("<rllm_program %s: %d nodes, %d parameters, %d outputs; %s>\n",
                x$name, length(x$nodes), length(x$parameters),
                length(x$outputs), .rllm_plan_counts(operators)))
    invisible(x)
}

#' Execute a data-only architecture program
#'
#' `rllm_execute()` interprets the dataflow and structured loops in an
#' [rllm_program()]. Semantic operators are ordinary R functions supplied in
#' `operators`; a small dense reference vocabulary is provided for arithmetic
#' shared by the architecture probes. This is the dense reference interpreter
#' for the same program that [rllm_forward()] binds to mapped weights and
#' lowers natively through GGML.
#'
#' Each operator function is called with four named arguments: `inputs`, the
#' list of values arriving at the node; `attributes`, the node's data-only
#' attributes; `parameters`, the named parameter values; and `context`, a
#' list containing the current node and program, root and local inputs,
#' symbolic counts, and the enclosing loop iterations. Operator functions
#' control the representation of their results, so a lowering may preserve a
#' mapped or device-backed value rather than materializing it in R.
#'
#' The built-in dense reference vocabulary covers arithmetic, normalization,
#' pooling, activations and recurrence helpers. It also covers ESM token
#' dropout, key-padded rotary attention with attention-map results, tied
#' projection and contact regression, plus the dilated convolution, inference
#' batch normalization, cropping and softmax used by OpenSpliceAI. Entries in
#' `operators` replace built-ins with the same name.
#'
#' @param program A data-only `rllm_program`.
#' @param inputs A named list of input values. Extra entries remain available
#'   to operator functions through `context$inputs`.
#' @param parameters A named list containing a value for every declared
#'   program parameter.
#' @param counts A named list or atomic vector resolving symbolic loop counts
#'   and other symbolic integer attributes.
#' @param operators A named list of semantic operator functions.
#'
#' @return A named list containing every program output and tap.
#' @export
rllm_execute <- function(program, inputs, parameters = list(), counts = list(),
                         operators = list()) {
    if (!inherits(program, "rllm_program")) {
        stop("program must come from rllm_program()")
    }
    inputs <- .rllm_execute_named(inputs, "inputs")
    parameters <- .rllm_execute_named(parameters, "parameters")
    counts <- .rllm_execute_counts(counts)
    operators <- .rllm_execute_named(operators, "operators")
    if (length(operators) &&
        !all(vapply(operators, is.function, logical(1)))) {
        stop("operators must contain functions")
    }

    input_names <- vapply(Filter(function(node) node$op == "input",
                                 program$nodes),
                          function(node) node$attributes$name, character(1))
    missing_inputs <- setdiff(input_names, names(inputs))
    if (length(missing_inputs)) {
        stop("missing program inputs: ", paste(missing_inputs, collapse = ", "))
    }
    missing_parameters <- setdiff(names(program$parameters), names(parameters))
    if (length(missing_parameters)) {
        stop("missing program parameters: ",
             paste(missing_parameters, collapse = ", "))
    }

    reference <- .rllm_reference_operators()
    reference[names(operators)] <- operators
    .rllm_execute_program(program, inputs, parameters, counts, reference,
                          root_inputs = inputs, loops = list())
}

.rllm_execute_program <- function(program, inputs, parameters, counts,
                                  operators, root_inputs, loops) {
    values <- new.env(hash = TRUE, parent = emptyenv())

    for (node in program$nodes) {
        node_inputs <- lapply(unname(node$inputs), function(id) {
            if (!exists(id, values, inherits = FALSE)) {
                stop("program node '", node$id,
                     "' refers to unavailable input '", id, "'")
            }
            get(id, values, inherits = FALSE)
        })
        if (length(node$inputs) && !is.null(names(node$inputs))) {
            names(node_inputs) <- names(node$inputs)
        }

        value <- switch(node$op,
            input = {
                name <- node$attributes$name
                if (!(name %in% names(inputs))) {
                    stop("missing loop carry or program input '", name, "'")
                }
                inputs[[name]]
            },
            loop = .rllm_execute_loop(
                node, node_inputs, parameters, counts, operators,
                root_inputs, loops
            ),
            loop_result = {
                tuple <- node_inputs[[1L]]
                carry <- node$attributes$carry
                if (!is.list(tuple) || !(carry %in% names(tuple))) {
                    stop("loop did not return carry '", carry, "'")
                }
                tuple[[carry]]
            },
            op_result = {
                tuple <- node_inputs[[1L]]
                result <- node$attributes$result
                if (!is.list(tuple) || !(result %in% names(tuple))) {
                    stop("operator did not return result '", result, "'")
                }
                tuple[[result]]
            },
            {
                operator <- operators[[node$op]]
                if (is.null(operator)) {
                    location <- if (length(node$module)) {
                        paste0(" in module '",
                               paste(node$module, collapse = "/"), "'")
                    } else {
                        ""
                    }
                    stop("no lowering registered for operator '", node$op,
                         "'", location)
                }
                operator(
                    inputs = node_inputs,
                    attributes = node$attributes,
                    parameters = parameters,
                    context = list(
                        node = node,
                        program = program,
                        inputs = root_inputs,
                        locals = inputs,
                        counts = counts,
                        loops = loops
                    )
                )
            }
        )
        assign(node$id, value, values)
    }

    out <- lapply(unname(program$outputs), function(id) {
        if (!exists(id, values, inherits = FALSE)) {
            stop("program output refers to unavailable node '", id, "'")
        }
        get(id, values, inherits = FALSE)
    })
    names(out) <- names(program$outputs)
    out
}

.rllm_execute_loop <- function(node, inputs, parameters, counts, operators,
                               root_inputs, loops) {
    carries <- node$attributes$carries
    names(inputs) <- carries
    times <- .rllm_execute_count(node$attributes$times, counts,
                                 paste0("loop '", node$attributes$name, "'"))
    state <- inputs
    for (iteration in seq_len(times)) {
        current_loop <- list(iteration)
        names(current_loop) <- node$attributes$name
        loop_context <- c(loops, current_loop)
        result <- .rllm_execute_program(
            node$attributes$body, state, parameters, counts, operators,
            root_inputs, loop_context
        )
        missing <- setdiff(carries, names(result))
        if (length(missing)) {
            stop("loop '", node$attributes$name,
                 "' body did not return carries: ",
                 paste(missing, collapse = ", "))
        }
        state <- result[carries]
    }
    if (length(carries) == 1L) state[[1L]] else state
}

.rllm_execute_named <- function(x, what) {
    if (is.null(x)) x <- list()
    if (!is.list(x)) stop(what, " must be a named list")
    .rllm_named(x, what)
    x
}

.rllm_execute_counts <- function(x) {
    if (is.null(x)) return(list())
    if (is.atomic(x)) x <- as.list(x)
    x <- .rllm_execute_named(x, "counts")
    for (name in names(x)) {
        .rllm_execute_count(x[[name]], list(), paste0("count '", name, "'"),
                            symbolic = FALSE, zero = TRUE)
    }
    x
}

.rllm_execute_count <- function(x, counts, what, symbolic = TRUE,
                                zero = FALSE) {
    if (is.character(x) && symbolic) {
        if (!(x %in% names(counts))) stop(what, " needs count '", x, "'")
        x <- counts[[x]]
    }
    lower <- if (zero) 0 else 1
    if (!is.numeric(x) || length(x) != 1L || is.na(x) || !is.finite(x) ||
        x < lower || x != floor(x) || x > .Machine$integer.max) {
        stop(what, " must resolve to one ",
             if (zero) "non-negative" else "positive", " integer")
    }
    as.integer(x)
}

.rllm_execute_parameter <- function(reference, parameters, what) {
    if (!inherits(reference, "rllm_parameter")) {
        stop(what, " attribute is not an rllm_parameter")
    }
    if (!(reference$name %in% names(parameters))) {
        stop("missing program parameter '", reference$name, "'")
    }
    parameters[[reference$name]]
}

.rllm_execute_feature_matrix <- function(x, what) {
    dimensions <- dim(x)
    if (is.null(dimensions)) {
        return(list(value = matrix(x, ncol = 1L), dimensions = NULL))
    }
    if (!length(dimensions) || any(dimensions < 1L)) {
        stop(what, " must have a non-empty feature dimension")
    }
    list(value = matrix(x, nrow = dimensions[[1L]]),
         dimensions = dimensions)
}

.rllm_execute_restore <- function(x, dimensions) {
    if (is.null(dimensions)) return(drop(x))
    dim(x) <- c(nrow(x), dimensions[-1L])
    x
}

.rllm_execute_linear <- function(inputs, attributes, parameters, context) {
    x <- .rllm_execute_feature_matrix(inputs[[1L]], "linear input")
    weight <- .rllm_execute_parameter(attributes$weight, parameters,
                                       "linear weight")
    if (!is.matrix(weight) || nrow(weight) != nrow(x$value)) {
        stop("linear weight does not match the input feature dimension")
    }
    out <- crossprod(weight, x$value)
    if (!is.null(attributes$bias)) {
        bias <- .rllm_execute_parameter(attributes$bias, parameters,
                                        "linear bias")
        if (length(bias) != nrow(out)) {
            stop("linear bias does not match the output feature dimension")
        }
        out <- sweep(out, 1L, bias, `+`)
    }
    dimensions <- x$dimensions
    if (!is.null(dimensions)) dimensions[[1L]] <- nrow(out)
    .rllm_execute_restore(out, dimensions)
}

.rllm_execute_norm <- function(inputs, attributes, parameters, layer = FALSE) {
    x <- .rllm_execute_feature_matrix(inputs[[1L]], "normalization input")
    eps <- attributes$eps
    if (is.null(eps)) eps <- 1e-5
    if (layer) {
        centre <- colMeans(x$value)
        centred <- sweep(x$value, 2L, centre, `-`)
        scale <- sqrt(colMeans(centred^2) + eps)
        out <- sweep(centred, 2L, scale, `/`)
    } else {
        scale <- sqrt(colMeans(x$value^2) + eps)
        out <- sweep(x$value, 2L, scale, `/`)
    }
    if (!is.null(attributes$weight)) {
        weight <- .rllm_execute_parameter(attributes$weight, parameters,
                                           "normalization weight")
        if (length(weight) != nrow(out)) {
            stop("normalization weight has the wrong feature dimension")
        }
        out <- sweep(out, 1L, weight, `*`)
    }
    if (!is.null(attributes$bias)) {
        bias <- .rllm_execute_parameter(attributes$bias, parameters,
                                        "normalization bias")
        if (length(bias) != nrow(out)) {
            stop("normalization bias has the wrong feature dimension")
        }
        out <- sweep(out, 1L, bias, `+`)
    }
    .rllm_execute_restore(out, x$dimensions)
}

.rllm_execute_pool <- function(inputs, attributes, parameters, context) {
    x <- inputs[[1L]]
    kind <- attributes$kind
    if (identical(kind, "none")) return(x)
    value <- .rllm_execute_feature_matrix(x, "pool input")$value
    if (identical(kind, "cls")) return(value[, 1L])
    if (identical(kind, "mean")) return(rowMeans(value))
    stop("unknown pool kind '", kind, "'")
}

.rllm_execute_broadcast <- function(inputs, attributes, parameters, context) {
    initial <- .rllm_execute_parameter(attributes$value, parameters,
                                        "initial value")
    target <- inputs[[1L]]
    dimensions <- dim(target)
    if (is.null(dimensions)) {
        if (length(initial) != length(target)) {
            stop("initial value does not match the target feature dimension")
        }
        return(initial)
    }
    if (length(initial) != dimensions[[1L]]) {
        stop("initial value does not match the target feature dimension")
    }
    array(rep(initial, prod(dimensions[-1L])), dim = dimensions)
}

.rllm_execute_swiglu <- function(inputs, attributes, parameters, context) {
    x <- .rllm_execute_feature_matrix(inputs[[1L]], "swiglu input")
    gate_up <- .rllm_execute_parameter(attributes$gate_up, parameters,
                                        "swiglu gate_up")
    down <- .rllm_execute_parameter(attributes$down, parameters,
                                     "swiglu down")
    if (!is.matrix(gate_up) || nrow(gate_up) != nrow(x$value) ||
        ncol(gate_up) %% 2L != 0L) {
        stop("swiglu gate_up has incompatible dimensions")
    }
    projected <- crossprod(gate_up, x$value)
    width <- nrow(projected) %/% 2L
    gate <- projected[seq_len(width), , drop = FALSE]
    up <- projected[width + seq_len(width), , drop = FALSE]
    hidden <- (gate * stats::plogis(gate)) * up
    if (!is.matrix(down) || nrow(down) != width) {
        stop("swiglu down has incompatible dimensions")
    }
    out <- crossprod(down, hidden)
    dimensions <- x$dimensions
    if (!is.null(dimensions)) dimensions[[1L]] <- nrow(out)
    .rllm_execute_restore(out, dimensions)
}

.rllm_reference_operators <- function() {
    unary <- function(fun) {
        force(fun)
        function(inputs, attributes, parameters, context) fun(inputs[[1L]])
    }
    list(
        add = function(inputs, attributes, parameters, context) {
            if (!length(inputs)) stop("add needs at least one input")
            Reduce(`+`, inputs)
        },
        linear = .rllm_execute_linear,
        rms_norm = function(inputs, attributes, parameters, context) {
            .rllm_execute_norm(inputs, attributes, parameters)
        },
        layer_norm = function(inputs, attributes, parameters, context) {
            .rllm_execute_norm(inputs, attributes, parameters, layer = TRUE)
        },
        esm_token_embedding = .rllm_execute_esm_embedding,
        attention = .rllm_execute_esm_attention,
        tied_projection = .rllm_execute_tied_projection,
        esm_contact_head = .rllm_execute_esm_contact,
        conv1d = .rllm_execute_conv1d,
        batch_norm = .rllm_execute_batch_norm,
        leaky_relu = .rllm_execute_leaky_relu,
        crop1d = .rllm_execute_crop1d,
        softmax = .rllm_execute_softmax,
        pool = .rllm_execute_pool,
        gelu = unary(function(x) x * stats::pnorm(x)),
        silu = unary(function(x) x * stats::plogis(x)),
        sigmoid = unary(stats::plogis),
        tanh = unary(tanh),
        broadcast_initial = .rllm_execute_broadcast,
        select_first_token = function(inputs, attributes, parameters, context) {
            .rllm_execute_feature_matrix(inputs[[1L]],
                                          "token selection input")$value[, 1L]
        },
        drop_puzzle_prefix = function(inputs, attributes, parameters, context) {
            count <- .rllm_execute_count(
                attributes$count, context$counts, "puzzle prefix",
                zero = TRUE
            )
            value <- .rllm_execute_feature_matrix(
                inputs[[1L]], "prefix removal input"
            )$value
            if (count >= ncol(value)) return(value[, 0L, drop = FALSE])
            value[, seq.int(count + 1L, ncol(value)), drop = FALSE]
        },
        swiglu = .rllm_execute_swiglu
    )
}

.rllm_new_builder <- function(name) {
    builder <- new.env(parent = emptyenv())
    builder$name <- name
    builder$nodes <- list()
    builder$parameters <- list()
    builder$outputs <- list()
    builder$module <- character()
    builder
}

.rllm_add_input <- function(builder, name, shape, dtype) {
    .rllm_name(name, "input name")
    .rllm_name(dtype, "input dtype")
    .rllm_append(builder, "input", list(), list(name = name), shape, dtype)
}

.rllm_append <- function(builder, op, inputs, attrs, shape, dtype) {
    shape <- .rllm_program_shape(shape, allow_null = TRUE)
    .rllm_name(dtype, "value dtype")
    .rllm_assert_data(attrs, "operator attributes")
    .rllm_register_parameters(builder, attrs)
    id <- paste0("n", length(builder$nodes) + 1L)
    node <- list(
        id = id,
        op = op,
        inputs = vapply(inputs, `[[`, character(1), "id"),
        shape = shape,
        dtype = dtype,
        attributes = attrs,
        module = builder$module
    )
    builder$nodes[[length(builder$nodes) + 1L]] <- node
    structure(list(.builder = builder, id = id, shape = shape, dtype = dtype),
              class = "rllm_value")
}

.rllm_snapshot <- function(x, name = NULL) {
    values <- .rllm_values(x)
    builder <- values[[1L]]$.builder
    .rllm_assert_builder(x, builder)
    outputs <- builder$outputs
    value_names <- if (inherits(x, "rllm_value")) NULL else names(x)
    if (is.null(value_names) || anyNA(value_names) ||
        any(!nzchar(value_names))) {
        value_names <- if (length(values) == 1L) "output" else
            paste0("output_", seq_along(values))
    }
    for (i in seq_along(values)) {
        output_name <- value_names[[i]]
        existing <- outputs[[output_name]]
        if (is.null(existing)) {
            outputs[[output_name]] <- values[[i]]$id
        } else if (!identical(existing, values[[i]]$id)) {
            stop("final output name '", output_name,
                 "' already names a different tap")
        }
    }
    if (is.null(name)) name <- builder$name
    .rllm_name(name, "program name")
    structure(list(
        name = name,
        nodes = builder$nodes,
        parameters = builder$parameters,
        outputs = outputs
    ), class = c("rllm_program", "list"))
}

.rllm_values <- function(x) {
    if (inherits(x, "rllm_value")) return(list(x))
    if (!is.list(x) || !length(x) ||
        !all(vapply(x, inherits, logical(1), "rllm_value"))) {
        stop("expected a traced rllm_value or a non-empty list of them")
    }
    x
}

.rllm_one_value <- function(x) {
    values <- .rllm_values(x)
    if (length(values) != 1L) stop("operator expects one traced value")
    values[[1L]]
}

.rllm_assert_builder <- function(x, builder) {
    values <- .rllm_values(x)
    if (any(!vapply(values, function(value) identical(value$.builder, builder),
                    logical(1)))) {
        stop("traced values belong to different programs")
    }
    invisible(x)
}

.rllm_name <- function(x, what) {
    if (!is.character(x) || length(x) != 1L || is.na(x) || !nzchar(x)) {
        stop(what, " must be one non-empty string")
    }
    invisible(x)
}

.rllm_named <- function(x, what) {
    if (length(x) && (is.null(names(x)) || anyNA(names(x)) ||
                      any(!nzchar(names(x))) || anyDuplicated(names(x)))) {
        stop(what, " must have unique non-empty names")
    }
    invisible(x)
}

.rllm_program_shape <- function(shape, allow_null = FALSE) {
    if (is.null(shape) && allow_null) return(NULL)
    if (!is.atomic(shape) || !length(shape) ||
        !(is.numeric(shape) || is.character(shape))) {
        stop("shape must be a non-empty numeric or character vector")
    }
    if (is.numeric(shape) &&
        (anyNA(shape) || any(!is.finite(shape)) || any(shape < 1) ||
         any(shape != floor(shape)))) {
        stop("numeric shape dimensions must be positive integers")
    }
    if (is.character(shape) && (anyNA(shape) || any(!nzchar(shape)))) {
        stop("symbolic shape dimensions must be non-empty")
    }
    shape
}

.rllm_output_specs <- function(outputs, default_dtype) {
    if (!is.list(outputs) || !length(outputs)) {
        stop("outputs must be a non-empty named list")
    }
    .rllm_named(outputs, "outputs")
    lapply(outputs, function(spec) {
        if (is.atomic(spec)) {
            shape <- spec
            dtype <- default_dtype
        } else {
            if (!is.list(spec) || is.null(spec$shape)) {
                stop("each output must provide a shape")
            }
            unknown <- setdiff(names(spec), c("shape", "dtype"))
            if (length(unknown)) {
                stop("unknown output fields: ", paste(unknown, collapse = ", "))
            }
            shape <- spec$shape
            dtype <- spec$dtype
            if (is.null(dtype)) dtype <- default_dtype
        }
        .rllm_name(dtype, "output dtype")
        list(shape = .rllm_program_shape(shape), dtype = dtype)
    })
}

.rllm_same_dimension <- function(x, y) identical(as.character(x), as.character(y))

.rllm_same_shape <- function(x, y) {
    length(x) == length(y) &&
        all(unlist(Map(.rllm_same_dimension, as.list(x), as.list(y)),
                   use.names = FALSE))
}

.rllm_count <- function(x, what) {
    if (!is.numeric(x) || length(x) != 1L || is.na(x) || !is.finite(x) ||
        x < 1 || x != floor(x) || x > .Machine$integer.max) {
        stop(what, " must be one positive integer")
    }
    as.integer(x)
}

.rllm_expect_parameter <- function(x, what, rank = NULL) {
    if (!inherits(x, "rllm_parameter")) {
        stop(what, " must come from rllm_parameter()")
    }
    if (!is.null(rank) && length(x$shape) != rank) {
        stop(what, " must have rank ", rank)
    }
    invisible(x)
}

.rllm_assert_data <- function(x, what) {
    bad <- typeof(x) %in% c(
        "closure", "environment", "externalptr", "weakref", "promise",
        "language", "symbol"
    )
    if (bad) stop(what, " must contain data only")
    if (is.list(x)) {
        for (item in x) .rllm_assert_data(item, what)
    }
    attrs <- attributes(x)
    if (!is.null(attrs)) {
        for (item in attrs) .rllm_assert_data(item, what)
    }
    invisible(x)
}

.rllm_register_parameters <- function(builder, x) {
    if (inherits(x, "rllm_parameter")) {
        old <- builder$parameters[[x$name]]
        if (!is.null(old) && !identical(unclass(old), unclass(x))) {
            stop("parameter '", x$name, "' has conflicting declarations")
        }
        builder$parameters[[x$name]] <- x
        return(invisible(NULL))
    }
    if (is.list(x)) {
        for (item in x) .rllm_register_parameters(builder, item)
    }
    invisible(NULL)
}
