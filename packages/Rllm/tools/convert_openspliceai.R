#!/usr/bin/env Rscript
# SPDX-License-Identifier: GPL-2.0-or-later

usage <- function() {
    stop(
        "usage: Rscript tools/convert_openspliceai.R OUTPUT CHECKPOINT ",
        "FLANKING_SIZE [PYTHON]\n",
        "CHECKPOINT is an OpenSpliceAI PyTorch state dictionary and ",
        "FLANKING_SIZE is one of 80, 400, 2000, or 10000.",
        call. = FALSE
    )
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3L || length(args) > 4L) usage()
output <- args[[1L]]
checkpoint <- args[[2L]]
flanking_size <- suppressWarnings(as.integer(args[[3L]]))
python <- if (length(args) == 4L) args[[4L]] else "python3"
if (!file.exists(checkpoint)) stop("checkpoint does not exist: ", checkpoint)
if (length(flanking_size) != 1L || is.na(flanking_size)) usage()

schedule <- switch(
    as.character(flanking_size),
    `80` = list(kernel = rep.int(11L, 4L), dilation = rep.int(1L, 4L)),
    `400` = list(
        kernel = rep.int(11L, 8L),
        dilation = rep(c(1L, 4L), each = 4L)
    ),
    `2000` = list(
        kernel = rep(c(11L, 21L), c(8L, 4L)),
        dilation = rep(c(1L, 4L, 10L), each = 4L)
    ),
    `10000` = list(
        kernel = rep(c(11L, 21L, 41L), c(8L, 4L, 4L)),
        dilation = rep(c(1L, 4L, 10L, 25L), each = 4L)
    ),
    usage()
)
if (2L * sum(schedule$dilation * (schedule$kernel - 1L)) !=
    flanking_size) {
    stop("internal OpenSpliceAI schedule does not match its flanking size")
}
if (!requireNamespace("Rgguf", quietly = TRUE)) {
    stop("install the monorepo Rgguf package before converting", call. = FALSE)
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
    stop("safetensors is required to read the exported weights", call. = FALSE)
}

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1L) stop("cannot locate the converter script")
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg)))
source(file.path(script_dir, "read_f32_safetensors.R"), local = TRUE)
exporter <- file.path(script_dir, "export_openspliceai.py")

intermediate <- tempfile(fileext = ".safetensors")
on.exit(unlink(intermediate), add = TRUE)
status <- system2(
    python, shQuote(c(exporter, checkpoint, intermediate))
)
if (!identical(status, 0L)) stop("OpenSpliceAI checkpoint export failed")
tensors <- read_f32_safetensors(intermediate)

expected_names <- c(
    "initial_conv.weight", "initial_conv.bias",
    "initial_skip.conv.weight", "initial_skip.conv.bias"
)
module_index <- 0L
for (block in seq_along(schedule$kernel)) {
    prefix <- paste0("residual_units.", module_index, ".")
    expected_names <- c(
        expected_names,
        paste0(prefix, "batchnorm1.", c(
            "weight", "bias", "running_mean", "running_var"
        )),
        paste0(prefix, "batchnorm2.", c(
            "weight", "bias", "running_mean", "running_var"
        )),
        paste0(prefix, "conv1.", c("weight", "bias")),
        paste0(prefix, "conv2.", c("weight", "bias"))
    )
    module_index <- module_index + 1L
    if (block %% 4L == 0L) {
        expected_names <- c(
            expected_names,
            paste0("residual_units.", module_index,
                   ".conv.", c("weight", "bias"))
        )
        module_index <- module_index + 1L
    }
}
expected_names <- c(expected_names, "final_conv.weight", "final_conv.bias")
missing <- setdiff(expected_names, names(tensors))
extra <- setdiff(names(tensors), expected_names)
if (length(missing) || length(extra)) {
    stop(
        "OpenSpliceAI state dictionary does not match the selected schedule; ",
        "missing: ", paste(missing, collapse = ", "),
        "; extra: ", paste(extra, collapse = ", ")
    )
}
tensors <- tensors[expected_names]

metadata <- list(
    general.architecture = "openspliceai",
    openspliceai.block_count = length(schedule$kernel),
    openspliceai.embedding_length = 32L,
    openspliceai.input_channel_count = 4L,
    openspliceai.output_channel_count = 3L,
    openspliceai.context_length = flanking_size,
    openspliceai.convolution.kernel_size = schedule$kernel,
    openspliceai.convolution.dilation = schedule$dilation,
    openspliceai.batch_norm_epsilon = 1e-5,
    openspliceai.leaky_relu_slope = 0.1
)

directory <- dirname(output)
if (!dir.exists(directory)) dir.create(directory, recursive = TRUE)
Rgguf::gguf_write_tensors(output, tensors, metadata)
message(
    "wrote ", length(tensors), " OpenSpliceAI tensors through Rgguf to ",
    normalizePath(output)
)
