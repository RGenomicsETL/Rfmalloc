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
    int64_t tile_rows;           /* (tap, channel) rows gathered per pass */
    struct rggml_conv1d_sync *sync;
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

/* Accumulate one gathered tile into an output block already initialized with
 * the bias. `dst` addresses output channel zero of the block; consecutive
 * output channels are `dst_stride` floats apart and positions are contiguous.
 * `tile` is [rows][positions] and `weights` is [rows][output_channels], where
 * one row is one (tap, input channel) pair. The portable implementation is the
 * oracle; the dispatcher picks a staged one only when the runtime allows it. */
void rggml_conv1d_f32_tile_scalar(float *dst, int64_t dst_stride,
    const float *tile, const float *weights, int64_t rows, int64_t positions,
    int64_t output_channels);
void rggml_conv1d_f32_tile(float *dst, int64_t dst_stride,
    const float *tile, const float *weights, int64_t rows, int64_t positions,
    int64_t output_channels);

#endif
