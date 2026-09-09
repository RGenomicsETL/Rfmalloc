/*
 * rggml_cpu_ops.c - project-owned parallel CPU custom operations.
 *
 * Upstream GGML computes leaky ReLU on thread zero alone, so in a dataflow
 * that applies it once per residual branch every other worker waits at the
 * next barrier. These variants are CPU-only by construction: a custom callback
 * is not a device operation, so device graphs keep the official operator.
 *
 * Copyright (C) 2026 Sounkou Mahamane Toure
 *
 * This file is part of Rggml and is GPL-2-or-later.
 */
#include "rggml_cpu_ops.h"

#include <stdint.h>
#include <string.h>

/* The slope travels as its own F32 bit pattern inside the custom operation's
 * user-data pointer, so the node owns no allocation and outlives nothing. */
static float
rggml_cpu_unary_slope(const void *userdata)
{
    const uint32_t bits = (uint32_t) (uintptr_t) userdata;
    float slope;

    memcpy(&slope, &bits, sizeof(slope));
    return slope;
}

static void
rggml_leaky_relu_cpu_callback(struct ggml_tensor *dst, int ith, int nth,
                              void *userdata)
{
    const struct ggml_tensor *src = dst->src[0];
    const float slope = rggml_cpu_unary_slope(userdata);
    const int64_t n = ggml_nelements(dst);
    const int64_t per_thread = (n + nth - 1) / nth;
    const int64_t first = (int64_t) ith * per_thread;
    const int64_t last = first + per_thread < n ? first + per_thread : n;
    const float *x = (const float *) src->data;
    float *y = (float *) dst->data;

    for (int64_t i = first; i < last; ++i) {
        y[i] = x[i] > 0.0f ? x[i] : x[i] * slope;
    }
}

struct ggml_tensor *
Rggml_leaky_relu_cpu(struct ggml_context *ctx, struct ggml_tensor *a,
                     double slope)
{
    struct ggml_tensor *args[1];
    const float value = (float) slope;
    uint32_t bits;

    if (!ctx || !a || a->type != GGML_TYPE_F32 || !ggml_is_contiguous(a)) {
        return NULL;
    }
    memcpy(&bits, &value, sizeof(bits));
    args[0] = a;
    return ggml_custom_4d(ctx, GGML_TYPE_F32, a->ne[0], a->ne[1], a->ne[2],
                          a->ne[3], args, 1, rggml_leaky_relu_cpu_callback,
                          GGML_N_TASKS_MAX, (void *) (uintptr_t) bits);
}
