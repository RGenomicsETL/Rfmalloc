# SPDX-License-Identifier: GPL-2.0-or-later

read_f32_safetensors <- function(path, keys = NULL) {
    framework <- "rllm_base"
    frameworks <- safetensors::safetensors_frameworks
    old <- frameworks[[framework]]
    frameworks[[framework]] <- list(
        packages = character(),
        constructor = function(bytes, meta) {
            if (!identical(meta$dtype, "F32")) {
                stop("unsupported safetensors dtype ", meta$dtype)
            }
            shape <- as.numeric(meta$shape)
            count <- prod(shape)
            if (!length(shape) || anyNA(shape) || any(!is.finite(shape)) ||
                any(shape < 1) || any(shape != floor(shape)) ||
                !is.finite(count) || count > .Machine$integer.max ||
                length(bytes) != 4 * count) {
                stop("invalid F32 safetensors extent")
            }
            connection <- rawConnection(bytes, open = "rb")
            on.exit(close(connection), add = TRUE)
            value <- readBin(
                connection, what = numeric(), n = as.integer(count),
                size = 4L, endian = "little"
            )
            if (length(shape) > 1L) dim(value) <- rev(as.integer(shape))
            value
        }
    )
    on.exit(frameworks[[framework]] <- old, add = TRUE)

    safe <- safetensors::safetensors$new(path, framework = framework)
    available <- safe$keys()
    if (is.null(keys)) {
        keys <- available
    } else if (is.function(keys)) {
        keep <- vapply(available, keys, logical(1))
        keys <- available[keep]
    } else {
        if (!is.character(keys) || anyNA(keys) || any(!keys %in% available)) {
            stop("requested safetensors keys are invalid")
        }
    }
    tensors <- lapply(keys, safe$get_tensor)
    names(tensors) <- keys
    tensors
}
