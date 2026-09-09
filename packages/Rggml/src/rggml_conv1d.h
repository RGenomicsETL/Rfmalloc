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

/* One work item covers this many output positions of one batch element. The
 * gathered tile is [taps * input_channels][positions] floats and is built on
 * the worker's stack, so its float count is bounded. A kernel too deep for one
 * tile is accumulated in tap chunks. */
#define RGGML_CONV1D_TILE_POSITIONS 24
#define RGGML_CONV1D_TILE_MAX_FLOATS 12288
#define RGGML_CONV1D_MAX_ROWS 4096

struct rggml_conv1d_sync;

/* The plan owns packed [K * IC][OC] F32 weights and an optional OC bias.
 * It never owns a ggml context, tensor, backend, or R object. Its sync block
 * is mutable through a const plan on purpose: it carries only the shared work
 * counter that the GGML workers claim output tiles from. */
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
    int64_t tile_positions;      /* output positions per work item */
    int64_t tile_taps;           /* taps gathered per pass, <= kernel */
    /* Optional per-input-channel affine and leaky rectification applied while
     * the tile is gathered. Folding the caller's normalization and activation
     * in here removes their graph nodes, their barriers and two whole passes
     * over the activation, and computes the same F32 values in the same
     * order. */
    float *input_scale;
    float *input_shift;
    float leaky_slope;
    int leaky;
    struct rggml_conv1d_sync *sync;
};

struct Rggml_conv_1d_f32_plan *Rggml_conv_1d_f32_plan_create(
    const float *kernel, size_t kernel_bytes, const float *bias,
    size_t bias_bytes, int64_t kernel_size, int64_t input_channels,
    int64_t output_channels, int64_t stride, int64_t padding,
    int64_t dilation, int64_t max_output_channels);
void Rggml_conv_1d_f32_plan_destroy(struct Rggml_conv_1d_f32_plan *plan);
int Rggml_conv_1d_f32_plan_fuse_input(struct Rggml_conv_1d_f32_plan *plan,
    const float *scale, const float *shift, size_t channel_bytes,
    int leaky, double slope);
struct ggml_tensor *Rggml_conv_1d_f32_plan_apply(
    struct ggml_context *ctx, const struct Rggml_conv_1d_f32_plan *plan,
    struct ggml_tensor *input);

/* Accumulate one gathered tile into an output block already initialized with
 * the bias. `dst` addresses output channel zero of the block; consecutive
 * output channels are `dst_stride` floats apart and positions are contiguous.
 * `rows_at` holds one pointer to `positions` contiguous inputs per row and
 * `weights` is [rows][output_channels], where one row is one (tap, input
 * channel) pair. Rows are addressed indirectly because a unit-stride
 * convolution reads them as overlapping windows of one gathered channel,
 * which is gathered and activated once instead of once per tap. The portable implementation is the
 * oracle; the dispatcher picks a staged one only when the runtime allows it. */
/* Apply a folded channel affine and leaky rectification to one gathered row.
 * The portable form is the oracle; the dispatcher stages a vector form because
 * this runs once per tap and per input element that a tile touches. */
void rggml_conv1d_f32_activate_scalar(float *dst, const float *src, int64_t n,
    float scale, float shift, int leaky, float slope);
void rggml_conv1d_f32_activate(float *dst, const float *src, int64_t n,
    float scale, float shift, int leaky, float slope);
void rggml_conv1d_f32_tile_scalar(float *dst, int64_t dst_stride,
    const float *const *rows_at, const float *weights, int64_t rows,
    int64_t positions, int64_t output_channels);
void rggml_conv1d_f32_tile(float *dst, int64_t dst_stride,
    const float *const *rows_at, const float *weights, int64_t rows,
    int64_t positions, int64_t output_channels);

#endif
