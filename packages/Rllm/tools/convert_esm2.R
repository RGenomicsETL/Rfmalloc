#!/usr/bin/env Rscript
# SPDX-License-Identifier: GPL-2.0-or-later

usage <- function() {
    stop(
        "usage: Rscript tools/convert_esm2.R OUTPUT [MODEL_SOURCE]\n",
        "MODEL_SOURCE is a directory or URL containing model.safetensors, ",
        "config.json, and vocab.txt. It defaults to the official Facebook ",
        "ESM-2 8M repository.",
        call. = FALSE
    )
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L || length(args) > 2L) usage()

output <- args[[1L]]
model_source <- if (length(args) == 2L) args[[2L]] else
    "https://huggingface.co/facebook/esm2_t6_8M_UR50D/resolve/main"

if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite is required to read safetensors metadata", call. = FALSE)
}
if (!requireNamespace("Rgguf", quietly = TRUE)) {
    stop("install the monorepo Rgguf package before converting", call. = FALSE)
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
    stop("safetensors is required to read the source weights", call. = FALSE)
}

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1L) stop("cannot locate the converter script")
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg)))
source(file.path(script_dir, "read_f32_safetensors.R"), local = TRUE)

temporary <- character()
on.exit(unlink(temporary), add = TRUE)

source_file <- function(name) {
    if (dir.exists(model_source)) {
        path <- file.path(model_source, name)
        if (!file.exists(path)) stop("missing source file: ", path)
        return(path)
    }
    if (!grepl("^https?://", model_source)) {
        stop("MODEL_SOURCE must be a directory or HTTP(S) URL")
    }
    path <- tempfile(paste0("esm2-", name, "-"))
    temporary <<- c(temporary, path)
    status <- utils::download.file(
        paste0(sub("/$", "", model_source), "/", name), path,
        mode = "wb", quiet = FALSE
    )
    if (!identical(status, 0L)) stop("failed to download ", name)
    path
}

config_path <- source_file("config.json")
vocab_path <- source_file("vocab.txt")
tensor_path <- source_file("model.safetensors")

config <- jsonlite::fromJSON(config_path)
vocab <- readLines(vocab_path, warn = FALSE, encoding = "UTF-8")
if (!identical(config$model_type, "esm") ||
    !identical(config$position_embedding_type, "rotary") ||
    !isTRUE(config$token_dropout)) {
    stop("the source is not a rotary ESM-2 masked-language model")
}
if (length(vocab) != config$vocab_size) {
    stop("vocab.txt disagrees with config.json")
}

token_id <- function(token) {
    index <- match(token, vocab)
    if (is.na(index)) stop("vocabulary has no token ", token)
    index - 1L
}
mask_id <- token_id("<mask>")
padding_id <- token_id("<pad>")
bos_id <- token_id("<cls>")
eos_id <- token_id("<eos>")
if (mask_id != config$mask_token_id || padding_id != config$pad_token_id) {
    stop("vocabulary token ids disagree with config.json")
}

skip_tensor <- function(name) {
    name %in% c(
        "esm.embeddings.position_ids",
        "esm.embeddings.position_embeddings.weight"
    ) || endsWith(name, ".rotary_embeddings.inv_freq")
}
tensors <- read_f32_safetensors(tensor_path, function(name) {
    !skip_tensor(name)
})

fair_esm_name <- function(name) {
    exact <- c(
        "esm.embeddings.word_embeddings.weight" = "embed_tokens.weight",
        "esm.encoder.emb_layer_norm_after.weight" =
            "emb_layer_norm_after.weight",
        "esm.encoder.emb_layer_norm_after.bias" =
            "emb_layer_norm_after.bias",
        "esm.contact_head.regression.weight" =
            "contact_head.regression.weight",
        "esm.contact_head.regression.bias" =
            "contact_head.regression.bias"
    )
    if (name %in% names(exact)) return(unname(exact[[name]]))
    replacements <- list(
        c("^esm\\.encoder\\.layer\\.([0-9]+)\\.attention\\.self\\.query\\.",
          "layers.\\1.self_attn.q_proj."),
        c("^esm\\.encoder\\.layer\\.([0-9]+)\\.attention\\.self\\.key\\.",
          "layers.\\1.self_attn.k_proj."),
        c("^esm\\.encoder\\.layer\\.([0-9]+)\\.attention\\.self\\.value\\.",
          "layers.\\1.self_attn.v_proj."),
        c("^esm\\.encoder\\.layer\\.([0-9]+)\\.attention\\.output\\.dense\\.",
          "layers.\\1.self_attn.out_proj."),
        c("^esm\\.encoder\\.layer\\.([0-9]+)\\.attention\\.LayerNorm\\.",
          "layers.\\1.self_attn_layer_norm."),
        c("^esm\\.encoder\\.layer\\.([0-9]+)\\.intermediate\\.dense\\.",
          "layers.\\1.fc1."),
        c("^esm\\.encoder\\.layer\\.([0-9]+)\\.output\\.dense\\.",
          "layers.\\1.fc2."),
        c("^esm\\.encoder\\.layer\\.([0-9]+)\\.LayerNorm\\.",
          "layers.\\1.final_layer_norm.")
    )
    for (replacement in replacements) {
        mapped <- sub(replacement[[1L]], replacement[[2L]], name)
        if (!identical(mapped, name)) return(mapped)
    }
    if (startsWith(name, "lm_head.")) return(name)
    stop("no fair-esm tensor mapping for ", name)
}

tensor_names <- vapply(names(tensors), fair_esm_name, character(1))
expected_count <- 16L * config$num_hidden_layers + 10L
if (length(tensor_names) != expected_count || anyDuplicated(tensor_names)) {
    stop("ESM-2 tensor mapping is incomplete or ambiguous")
}
names(tensors) <- tensor_names

metadata <- list(
    general.architecture = "esm2",
    esm2.block_count = config$num_hidden_layers,
    esm2.embedding_length = config$hidden_size,
    esm2.feed_forward_length = config$intermediate_size,
    esm2.attention.head_count = config$num_attention_heads,
    esm2.attention.layer_norm_epsilon = config$layer_norm_eps,
    esm2.mask_token_id = mask_id,
    esm2.padding_token_id = padding_id,
    esm2.bos_token_id = bos_id,
    esm2.eos_token_id = eos_id,
    esm2.token_dropout.training_ratio = 0.15 * 0.8,
    tokenizer.ggml.model = "esm",
    tokenizer.ggml.tokens = vocab,
    tokenizer.ggml.bos_token_id = bos_id,
    tokenizer.ggml.eos_token_id = eos_id,
    tokenizer.ggml.padding_token_id = padding_id,
    tokenizer.ggml.mask_token_id = mask_id
)

directory <- dirname(output)
if (!dir.exists(directory)) dir.create(directory, recursive = TRUE)
Rgguf::gguf_write_tensors(output, tensors, metadata)
message(
    "wrote ", length(tensors), " official ESM-2 tensors through Rgguf to ",
    normalizePath(output)
)
