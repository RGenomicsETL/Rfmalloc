/*
 * rggml_conv1d.c - persistent packed F32 direct one-dimensional convolution.
 *
 * Copyright (C) 2026 Sounkou Mahamane Toure
 *
 * This file is part of Rggml and is GPL-2-or-later.
 */
#include "rggml_conv1d.h"

#include <limits.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#define RGGML_CONV1D_F32_MAX_OC 65536

/* Shared claim state for one plan. GGML barriers every node, so exactly the
 * threads of one invocation are inside the callback at a time; the thread that
 * completes last restores both counters for the next invocation. That keeps
 * the scheme correct when the thread count changes between calls. */
struct rggml_conv1d_sync {
    _Atomic int_fast64_t next;
    _Atomic int_fast64_t done;
};

static int
rggml_conv1d_mul(size_t a, size_t b, size_t *out)
{
    if (a && b > SIZE_MAX / a) return -1;
    *out = a * b;
    return 0;
}

static int
rggml_conv1d_output_length(int64_t input, const struct Rggml_conv_1d_f32_plan *plan,
                            int64_t *output)
{
    int64_t twice_padding, effective_kernel, numerator;

    if (input < 1 || plan->padding > (INT64_MAX - input) / 2 ||
        plan->kernel - 1 > (INT64_MAX - 1) / plan->dilation) return -1;
    twice_padding = plan->padding * 2;
    effective_kernel = (plan->kernel - 1) * plan->dilation + 1;
    if (input > INT64_MAX - twice_padding ||
        input + twice_padding < effective_kernel) return -1;
    numerator = input + twice_padding - effective_kernel;
    *output = numerator / plan->stride + 1;
    return *output < 1 ? -1 : 0;
}

struct Rggml_conv_1d_f32_plan *
Rggml_conv_1d_f32_plan_create(const float *kernel, size_t kernel_bytes,
    const float *bias, size_t bias_bytes, int64_t kernel_size,
    int64_t input_channels, int64_t output_channels, int64_t stride,
    int64_t padding, int64_t dilation, int64_t max_output_channels)
{
    size_t weights, weight_bytes, bias_bytes_required;
    struct Rggml_conv_1d_f32_plan *plan;
    int64_t positions, rows;

    if (!kernel || kernel_size < 1 || input_channels < 1 ||
        output_channels < 1 || stride < 1 || padding < 0 || dilation < 1 ||
        kernel_size - 1 > (INT64_MAX - 1) / dilation ||
        max_output_channels < output_channels ||
        max_output_channels > RGGML_CONV1D_F32_MAX_OC ||
        kernel_size > INT_MAX || input_channels > INT_MAX ||
        output_channels > INT_MAX) return NULL;
    if (rggml_conv1d_mul((size_t) kernel_size, (size_t) input_channels, &weights) ||
        rggml_conv1d_mul(weights, (size_t) output_channels, &weights) ||
        rggml_conv1d_mul(weights, sizeof(float), &weight_bytes) ||
        rggml_conv1d_mul((size_t) output_channels, sizeof(float),
                         &bias_bytes_required) ||
        kernel_bytes < weight_bytes || (bias && bias_bytes < bias_bytes_required)) {
        return NULL;
    }

    plan = calloc(1, sizeof(*plan));
    if (!plan) return NULL;
    plan->packed = malloc(weight_bytes);
    plan->sync = calloc(1, sizeof(*plan->sync));
    if (!plan->packed || !plan->sync) {
        free(plan->sync);
        free(plan->packed);
        free(plan);
        return NULL;
    }
    if (bias) {
        plan->bias = malloc(bias_bytes_required);
        if (!plan->bias) {
            free(plan->sync);
            free(plan->packed);
            free(plan);
            return NULL;
        }
        memcpy(plan->bias, bias, bias_bytes_required);
    }

    plan->kernel = kernel_size;
    plan->input_channels = input_channels;
    plan->output_channels = output_channels;
    plan->stride = stride;
    plan->padding = padding;
    plan->dilation = dilation;

    /* A work item gathers `tile_rows` consecutive (tap, input channel) rows
     * for `tile_positions` output positions. Both are chosen so one tile fits
     * the bounded stack buffer, and a deep kernel is accumulated in passes. */
    positions = RGGML_CONV1D_TILE_POSITIONS;
    if (positions > RGGML_CONV1D_TILE_MAX_FLOATS) positions = RGGML_CONV1D_TILE_MAX_FLOATS;
    rows = RGGML_CONV1D_TILE_MAX_FLOATS / positions;
    if (rows > kernel_size * input_channels) rows = kernel_size * input_channels;
    if (rows < 1) rows = 1;
    plan->tile_positions = positions;
    plan->tile_rows = rows;

    /* GGUF/GGML source layout is [K, IC, OC], K contiguous. The packed layout
     * makes one (tap, input channel) row address a contiguous OC block. */
    for (int64_t tap = 0; tap < kernel_size; ++tap) {
        for (int64_t channel = 0; channel < input_channels; ++channel) {
            float *to = plan->packed +
                ((size_t) tap * input_channels + (size_t) channel) * output_channels;
            for (int64_t output = 0; output < output_channels; ++output) {
                to[output] = kernel[tap + kernel_size *
                    (channel + input_channels * output)];
            }
        }
    }
    return plan;
}

void
Rggml_conv_1d_f32_plan_destroy(struct Rggml_conv_1d_f32_plan *plan)
{
    if (!plan) return;
    free(plan->sync);
    free(plan->bias);
    free(plan->packed);
    free(plan);
}

void
rggml_conv1d_f32_tile_scalar(float *dst, int64_t dst_stride, const float *tile,
    const float *weights, int64_t rows, int64_t positions,
    int64_t output_channels)
{
    for (int64_t row = 0; row < rows; ++row) {
        const float *x = tile + row * positions;
        const float *w = weights + row * output_channels;
        for (int64_t oc = 0; oc < output_channels; ++oc) {
            float *out = dst + oc * dst_stride;
            const float weight = w[oc];
            for (int64_t p = 0; p < positions; ++p) out[p] += weight * x[p];
        }
    }
}

/* Gather rows [row0, row0 + rows) of the (tap, input channel) space for one
 * batch element and `positions` output positions starting at `first`. Taps
 * that fall outside the padded input contribute zero. */
static void
rggml_conv1d_f32_gather(float *tile, const struct ggml_tensor *input,
    const struct Rggml_conv_1d_f32_plan *plan, int64_t batch, int64_t first,
    int64_t positions, int64_t row0, int64_t rows)
{
    const int64_t length = input->ne[0];
    const int64_t channels = plan->input_channels;
    const char *base = (const char *) input->data + (size_t) batch * input->nb[2];
    const int contiguous = plan->stride == 1 && input->nb[0] == sizeof(float);

    for (int64_t row = 0; row < rows; ++row) {
        const int64_t tap = (row0 + row) / channels;
        const int64_t channel = (row0 + row) % channels;
        const char *source = base + (size_t) channel * input->nb[1];
        float *out = tile + row * positions;
        const int64_t start = first * plan->stride + tap * plan->dilation -
            plan->padding;

        if (contiguous && start >= 0 && start + positions <= length) {
            memcpy(out, source + (size_t) start * sizeof(float),
                   (size_t) positions * sizeof(float));
            continue;
        }
        for (int64_t p = 0; p < positions; ++p) {
            const int64_t at = start + p * plan->stride;
            out[p] = (at >= 0 && at < length)
                ? *(const float *) (source + (size_t) at * input->nb[0])
                : 0.0f;
        }
    }
}

static void
rggml_conv1d_f32_item(struct ggml_tensor *dst, const struct ggml_tensor *input,
    const struct Rggml_conv_1d_f32_plan *plan, int64_t item, int64_t blocks)
{
    float tile[RGGML_CONV1D_TILE_MAX_FLOATS];
    const int64_t nout = dst->ne[0];
    const int64_t channels = dst->ne[1];
    const int64_t batch = item / blocks;
    const int64_t first = (item - batch * blocks) * plan->tile_positions;
    const int64_t positions = plan->tile_positions < nout - first
        ? plan->tile_positions : nout - first;
    const int64_t stride = (int64_t) (dst->nb[1] / sizeof(float));
    const int64_t rows = plan->kernel * plan->input_channels;
    float *out = (float *) ((char *) dst->data + (size_t) batch * dst->nb[2]) + first;

    for (int64_t oc = 0; oc < channels; ++oc) {
        float *row = out + oc * stride;
        const float value = plan->bias ? plan->bias[oc] : 0.0f;
        for (int64_t p = 0; p < positions; ++p) row[p] = value;
    }
    for (int64_t row0 = 0; row0 < rows; row0 += plan->tile_rows) {
        const int64_t take = plan->tile_rows < rows - row0
            ? plan->tile_rows : rows - row0;
        rggml_conv1d_f32_gather(tile, input, plan, batch, first, positions,
                                row0, take);
        if (plan->scalar_only) {
            rggml_conv1d_f32_tile_scalar(out, stride, tile,
                plan->packed + row0 * channels, take, positions, channels);
        } else {
            rggml_conv1d_f32_tile(out, stride, tile,
                plan->packed + row0 * channels, take, positions, channels);
        }
    }
}

static void
rggml_conv1d_f32_callback(struct ggml_tensor *dst, int ith, int nth,
                          void *userdata)
{
    const struct Rggml_conv_1d_f32_plan *plan = userdata;
    const struct ggml_tensor *input = dst->src[0];
    const int64_t blocks = (dst->ne[0] + plan->tile_positions - 1) /
        plan->tile_positions;
    const int64_t total = blocks * dst->ne[2];

    if (nth < 2 || !plan->sync) {
        for (int64_t item = ith; item < total; item += nth) {
            rggml_conv1d_f32_item(dst, input, plan, item, blocks);
        }
        return;
    }
    /* Performance cores retire this kernel about twice as fast as efficiency
     * cores, so a static split would leave every node waiting on the slowest
     * thread. Items are claimed instead, and the last thread out restores the
     * counters behind GGML's own end-of-node barrier. */
    for (;;) {
        const int_fast64_t item = atomic_fetch_add_explicit(&plan->sync->next, 1,
                                                            memory_order_relaxed);
        if (item >= total) break;
        rggml_conv1d_f32_item(dst, input, plan, (int64_t) item, blocks);
    }
    if (atomic_fetch_add_explicit(&plan->sync->done, 1, memory_order_acq_rel) + 1 == nth) {
        atomic_store_explicit(&plan->sync->next, 0, memory_order_relaxed);
        atomic_store_explicit(&plan->sync->done, 0, memory_order_release);
    }
}

struct ggml_tensor *
Rggml_conv_1d_f32_plan_apply(struct ggml_context *ctx,
    const struct Rggml_conv_1d_f32_plan *plan, struct ggml_tensor *input)
{
    int64_t output_length;
    struct ggml_tensor *args[1];

    if (!ctx || !plan || !input || input->type != GGML_TYPE_F32 ||
        input->ne[3] != 1 || input->ne[0] < 1 ||
        input->ne[1] != plan->input_channels || input->ne[2] < 1 ||
        input->nb[0] != sizeof(float) || input->nb[1] % sizeof(float) ||
        input->nb[2] % sizeof(float) ||
        rggml_conv1d_output_length(input->ne[0], plan, &output_length) ||
        output_length > INT64_MAX / input->ne[2] ||
        output_length * input->ne[2] > INT64_MAX / plan->output_channels ||
        (uint64_t) output_length > SIZE_MAX / (uint64_t) input->ne[2] ||
        (uint64_t) output_length * (uint64_t) input->ne[2] >
            SIZE_MAX / (uint64_t) plan->output_channels ||
        (uint64_t) output_length * (uint64_t) input->ne[2] *
            (uint64_t) plan->output_channels > SIZE_MAX / sizeof(float)) {
        return NULL;
    }
    args[0] = input;
    /* The kernel writes the AST layout directly, so no permutation or
     * materializing copy follows the convolution. */
    return ggml_custom_4d(ctx, GGML_TYPE_F32, output_length,
        plan->output_channels, input->ne[2], 1, args, 1,
        rggml_conv1d_f32_callback, GGML_N_TASKS_MAX, (void *) plan);
}
