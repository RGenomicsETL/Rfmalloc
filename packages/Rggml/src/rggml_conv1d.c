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
    if (input_channels > RGGML_CONV1D_MAX_ROWS) {
        free(plan->sync);
        free(plan->packed);
        free(plan->bias);
        free(plan);
        return NULL;
    }
    /* Per-channel float budget for one pass. Positions shrink first, then the
     * number of taps a pass covers; both gather forms must fit it. */
    {
        const int64_t budget = RGGML_CONV1D_TILE_MAX_FLOATS / input_channels;
        int64_t general;

        if (positions > budget) positions = budget;
        rows = (budget - positions) / dilation + 1;
        general = budget / positions;
        if (rows > general) rows = general;
        if (rows > RGGML_CONV1D_MAX_ROWS / input_channels) {
            rows = RGGML_CONV1D_MAX_ROWS / input_channels;
        }
        if (rows > kernel_size) rows = kernel_size;
        if (rows < 1) rows = 1;
    }
    plan->tile_positions = positions;
    plan->tile_taps = rows;

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
    free(plan->input_shift);
    free(plan->input_scale);
    free(plan->sync);
    free(plan->bias);
    free(plan->packed);
    free(plan);
}

/* Fold a per-input-channel affine, a leaky rectification, or both into the
 * gather. Either array may be NULL. The plan copies what it is given, so the
 * caller keeps ownership of its own storage. Returns non-zero and changes
 * nothing when the arguments do not describe this plan's input. */
int
Rggml_conv_1d_f32_plan_fuse_input(struct Rggml_conv_1d_f32_plan *plan,
    const float *scale, const float *shift, size_t channel_bytes,
    int leaky, double slope)
{
    size_t required;
    float *scale_copy = NULL, *shift_copy = NULL;

    if (!plan || plan->input_scale || plan->input_shift || plan->leaky) return -1;
    if (leaky && !(slope >= -1e30 && slope <= 1e30)) return -1;
    if (scale || shift) {
        if (rggml_conv1d_mul((size_t) plan->input_channels, sizeof(float), &required) ||
            channel_bytes < required) return -1;
        if (scale) {
            scale_copy = malloc(required);
            if (!scale_copy) return -1;
            memcpy(scale_copy, scale, required);
        }
        if (shift) {
            shift_copy = malloc(required);
            if (!shift_copy) {
                free(scale_copy);
                return -1;
            }
            memcpy(shift_copy, shift, required);
        }
    }
    plan->input_scale = scale_copy;
    plan->input_shift = shift_copy;
    plan->leaky = leaky ? 1 : 0;
    plan->leaky_slope = (float) slope;
    return 0;
}

void
rggml_conv1d_f32_activate_scalar(float *dst, const float *src, int64_t n,
    float scale, float shift, int leaky, float slope)
{
    if (leaky) {
        for (int64_t i = 0; i < n; ++i) {
            const float v = src[i] * scale + shift;
            dst[i] = v > 0.0f ? v : v * slope;
        }
        return;
    }
    for (int64_t i = 0; i < n; ++i) dst[i] = src[i] * scale + shift;
}

void
rggml_conv1d_f32_tile_scalar(float *dst, int64_t dst_stride,
    const float *const *rows_at, const float *weights, int64_t rows,
    int64_t positions, int64_t output_channels)
{
    for (int64_t row = 0; row < rows; ++row) {
        const float *x = rows_at[row];
        const float *w = weights + row * output_channels;
        for (int64_t oc = 0; oc < output_channels; ++oc) {
            float *out = dst + oc * dst_stride;
            const float weight = w[oc];
            for (int64_t p = 0; p < positions; ++p) out[p] += weight * x[p];
        }
    }
}

/* Gather one channel's input window for a tap range, zero outside the padded
 * input, and apply the folded activation once over it. A unit-stride
 * convolution then reads every tap of that channel as an overlapping slice. */
static void
rggml_conv1d_f32_window(float *window, const struct ggml_tensor *input,
    const struct Rggml_conv_1d_f32_plan *plan, int64_t batch, int64_t channel,
    int64_t start, int64_t span)
{
    const int64_t length = input->ne[0];
    const char *source = (const char *) input->data +
        (size_t) batch * input->nb[2] + (size_t) channel * input->nb[1];
    const int64_t from = start < 0 ? 0 : start;
    const int64_t to = start + span < length ? start + span : length;

    for (int64_t i = 0; i < span; ++i) window[i] = 0.0f;
    if (from >= to) return;
    memcpy(window + (from - start), source + (size_t) from * sizeof(float),
           (size_t) (to - from) * sizeof(float));
    /* Only what the input actually holds is activated. Padding is zero in the
     * convolution's own terms, which is not what a folded affine would make of
     * a zero, so the padded positions stay untouched. */
    if (plan->input_scale || plan->input_shift || plan->leaky) {
        rggml_conv1d_f32_activate(window + (from - start),
            window + (from - start), to - from,
            plan->input_scale ? plan->input_scale[channel] : 1.0f,
            plan->input_shift ? plan->input_shift[channel] : 0.0f,
            plan->leaky, plan->leaky_slope);
    }
}

/* Gather one (tap, input channel) row directly. This is the general form: it
 * serves any stride, and there the taps of one channel are not one window. */
static void
rggml_conv1d_f32_row(float *out, const struct ggml_tensor *input,
    const struct Rggml_conv_1d_f32_plan *plan, int64_t batch, int64_t channel,
    int64_t start, int64_t positions)
{
    const int64_t length = input->ne[0];
    const char *source = (const char *) input->data +
        (size_t) batch * input->nb[2] + (size_t) channel * input->nb[1];

    int64_t low = positions, high = 0;

    for (int64_t p = 0; p < positions; ++p) {
        const int64_t at = start + p * plan->stride;
        if (at >= 0 && at < length) {
            out[p] = *(const float *) (source + (size_t) at * input->nb[0]);
            if (p < low) low = p;
            high = p + 1;
        } else {
            out[p] = 0.0f;
        }
    }
    /* The positions the input covers are one run, because the stride is
     * positive. Padding outside it stays zero rather than being activated. */
    if (low < high && (plan->input_scale || plan->input_shift || plan->leaky)) {
        rggml_conv1d_f32_activate(out + low, out + low, high - low,
            plan->input_scale ? plan->input_scale[channel] : 1.0f,
            plan->input_shift ? plan->input_shift[channel] : 0.0f,
            plan->leaky, plan->leaky_slope);
    }
}

static void
rggml_conv1d_f32_item(struct ggml_tensor *dst, const struct ggml_tensor *input,
    const struct Rggml_conv_1d_f32_plan *plan, int64_t item, int64_t blocks)
{
    float scratch[RGGML_CONV1D_TILE_MAX_FLOATS];
    const float *rows_at[RGGML_CONV1D_MAX_ROWS];
    const int64_t nout = dst->ne[0];
    const int64_t channels = dst->ne[1];
    const int64_t in_channels = plan->input_channels;
    const int64_t batch = item / blocks;
    const int64_t first = (item - batch * blocks) * plan->tile_positions;
    const int64_t positions = plan->tile_positions < nout - first
        ? plan->tile_positions : nout - first;
    const int64_t stride = (int64_t) (dst->nb[1] / sizeof(float));
    const int64_t taps_per_pass = plan->tile_taps;
    const int64_t windowed = plan->stride == 1 && input->nb[0] == sizeof(float);
    const int64_t span = windowed
        ? positions + (taps_per_pass - 1) * plan->dilation : positions;
    float *out = (float *) ((char *) dst->data + (size_t) batch * dst->nb[2]) + first;

    for (int64_t oc = 0; oc < channels; ++oc) {
        float *row = out + oc * stride;
        const float value = plan->bias ? plan->bias[oc] : 0.0f;
        for (int64_t p = 0; p < positions; ++p) row[p] = value;
    }
    for (int64_t tap0 = 0; tap0 < plan->kernel; tap0 += taps_per_pass) {
        const int64_t taps = taps_per_pass < plan->kernel - tap0
            ? taps_per_pass : plan->kernel - tap0;
        const int64_t base = first * plan->stride + tap0 * plan->dilation -
            plan->padding;
        int64_t rows = 0;

        for (int64_t channel = 0; channel < in_channels; ++channel) {
            if (windowed) {
                rggml_conv1d_f32_window(scratch + channel * span, input, plan,
                                        batch, channel, base,
                                        positions + (taps - 1) * plan->dilation);
            }
        }
        for (int64_t tap = 0; tap < taps; ++tap) {
            for (int64_t channel = 0; channel < in_channels; ++channel) {
                float *at = scratch + (windowed
                    ? channel * span + tap * plan->dilation
                    : rows * positions);
                if (!windowed) {
                    rggml_conv1d_f32_row(at, input, plan, batch, channel,
                        base + tap * plan->dilation, positions);
                }
                rows_at[rows++] = at;
            }
        }
        if (plan->scalar_only) {
            rggml_conv1d_f32_tile_scalar(out, stride, rows_at,
                plan->packed + tap0 * in_channels * channels, rows, positions,
                channels);
        } else {
            rggml_conv1d_f32_tile(out, stride, rows_at,
                plan->packed + tap0 * in_channels * channels, rows, positions,
                channels);
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
