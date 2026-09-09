/*
 * rggml_conv1d.c - persistent packed F32 direct one-dimensional convolution.
 *
 * Copyright (C) 2026 Sounkou Mahamane Toure
 *
 * This file is part of Rggml and is GPL-2-or-later.
 */
#include "rggml_conv1d.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define RGGML_CONV1D_F32_MAX_OC 65536

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
    if (!plan->packed) {
        free(plan);
        return NULL;
    }
    if (bias) {
        plan->bias = malloc(bias_bytes_required);
        if (!plan->bias) {
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

    /* GGUF/GGML source layout is [K, IC, OC], K contiguous. The packed
     * layout makes every one input scalar update a contiguous OC block. */
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
    free(plan->bias);
    free(plan->packed);
    free(plan);
}

void
rggml_conv1d_f32_accumulate_scalar(float *dst, const struct ggml_tensor *input,
    const struct Rggml_conv_1d_f32_plan *plan, int64_t batch,
    int64_t output_position)
{
    const int64_t input_length = input->ne[0];
    const char *base = (const char *) input->data;
    const size_t output_channels = (size_t) plan->output_channels;

    for (int64_t output = 0; output < plan->output_channels; ++output) {
        dst[output] = plan->bias ? plan->bias[output] : 0.0f;
    }
    for (int64_t tap = 0; tap < plan->kernel; ++tap) {
        const int64_t position = output_position * plan->stride +
            tap * plan->dilation - plan->padding;
        if (position < 0 || position >= input_length) continue;
        for (int64_t channel = 0; channel < plan->input_channels; ++channel) {
            const float value = *(const float *) (base +
                (size_t) position * input->nb[0] +
                (size_t) channel * input->nb[1] + (size_t) batch * input->nb[2]);
            const float *weight = plan->packed +
                ((size_t) tap * plan->input_channels + (size_t) channel) * output_channels;
            for (int64_t output = 0; output < plan->output_channels; ++output) {
                dst[output] += value * weight[output];
            }
        }
    }
}

static void
rggml_conv1d_f32_callback(struct ggml_tensor *dst, int ith, int nth,
                          void *userdata)
{
    const struct Rggml_conv_1d_f32_plan *plan = userdata;
    const struct ggml_tensor *input = dst->src[0];
    const int64_t output_length = dst->ne[1];
    const int64_t rows = dst->ne[1] * dst->ne[2];
    float *data = (float *) dst->data;

    for (int64_t row = ith; row < rows; row += nth) {
        const int64_t batch = row / output_length;
        const int64_t position = row - batch * output_length;
        rggml_conv1d_f32_accumulate(data + (size_t) row * plan->output_channels,
                              input, plan, batch, position);
    }
}

struct ggml_tensor *
Rggml_conv_1d_f32_plan_apply(struct ggml_context *ctx,
    const struct Rggml_conv_1d_f32_plan *plan, struct ggml_tensor *input)
{
    int64_t output_length;
    struct ggml_tensor *args[1];
    struct ggml_tensor *packed_output;

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
    packed_output = ggml_custom_4d(ctx, GGML_TYPE_F32, plan->output_channels,
        output_length, input->ne[2], 1, args, 1, rggml_conv1d_f32_callback,
        GGML_N_TASKS_MAX, (void *) plan);
    if (!packed_output) return NULL;
    return ggml_cont(ctx, ggml_permute(ctx, packed_output, 1, 0, 2, 3));
}
