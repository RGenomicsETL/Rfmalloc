/*
 * rggml_conv1d.h - project-owned persistent F32 direct conv1d internals.
 *
 * Copyright (C) 2026 Sounkou Mahamane Toure
 *
 * This file is part of Rggml and is GPL-2-or-later.
 */
#ifndef RGGML_CONV1D_H
#define RGGML_CONV1D_H

#include <stddef.h>
#include <stdint.h>

#include <ggml.h>

/* The plan owns packed [K * IC, OC] F32 weights and an optional OC bias.
 * It never owns a ggml context, tensor, backend, or R object. */
struct Rggml_conv_1d_f32_plan {
    int64_t kernel;
    int64_t input_channels;
    int64_t output_channels;
    int64_t stride;
    int64_t padding;
    int64_t dilation;
    float *packed;
    float *bias;
    int scalar_only;
};

struct Rggml_conv_1d_f32_plan *Rggml_conv_1d_f32_plan_create(
    const float *kernel, size_t kernel_bytes, const float *bias,
    size_t bias_bytes, int64_t kernel_size, int64_t input_channels,
    int64_t output_channels, int64_t stride, int64_t padding,
    int64_t dilation, int64_t max_output_channels);
void Rggml_conv_1d_f32_plan_destroy(struct Rggml_conv_1d_f32_plan *plan);
struct ggml_tensor *Rggml_conv_1d_f32_plan_apply(
    struct ggml_context *ctx, const struct Rggml_conv_1d_f32_plan *plan,
    struct ggml_tensor *input);

/* The staged dispatcher calls this portable row implementation when AVX2/FMA
 * was not staged or the complete runtime feature predicate is false. */
void rggml_conv1d_f32_accumulate_scalar(float *dst,
    const struct ggml_tensor *input,
    const struct Rggml_conv_1d_f32_plan *plan, int64_t batch,
    int64_t output_position);
void rggml_conv1d_f32_accumulate(float *dst,
    const struct ggml_tensor *input,
    const struct Rggml_conv_1d_f32_plan *plan, int64_t batch,
    int64_t output_position);

#endif
