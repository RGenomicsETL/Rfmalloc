/*
 * rggml_cpu_ops.h - project-owned parallel CPU custom operations.
 *
 * Copyright (C) 2026 Sounkou Mahamane Toure
 *
 * This file is part of Rggml and is GPL-2-or-later.
 */
#ifndef RGGML_CPU_OPS_H
#define RGGML_CPU_OPS_H

#include <ggml.h>

/* Leaky ReLU over a contiguous F32 tensor, split across every GGML worker.
 * Returns NULL when the input cannot use it, so the caller keeps the official
 * operator; a device graph must always keep the official operator. */
struct ggml_tensor *Rggml_leaky_relu_cpu(struct ggml_context *ctx,
                                          struct ggml_tensor *a, double slope);

#endif
