/*
 * Runtime dispatch for project-owned packed F32 conv1d tiles.
 *
 * Copyright (C) 2026 Sounkou Mahamane Toure
 *
 * This file is part of Rggml and is GPL-2-or-later.
 */
#include "cpu_features.h"
#include "../../src/rggml_conv1d.h"

#ifdef RGGML_HAVE_X86_AVX2
extern void rggml_conv1d_f32_tile_avx2(float *dst, int64_t dst_stride,
    const float *tile, const float *weights, int64_t rows, int64_t positions,
    int64_t output_channels);
static int rggml_conv1d_use_avx2;
#endif

void
rggml_conv1d_simd_dispatch_init(void)
{
#ifdef RGGML_HAVE_X86_AVX2
    rggml_conv1d_use_avx2 = sd_cpu_has_upstream_avx2();
#endif
}

void
rggml_conv1d_f32_tile(float *dst, int64_t dst_stride, const float *tile,
    const float *weights, int64_t rows, int64_t positions,
    int64_t output_channels)
{
#ifdef RGGML_HAVE_X86_AVX2
    if (rggml_conv1d_use_avx2) {
        rggml_conv1d_f32_tile_avx2(dst, dst_stride, tile, weights, rows,
                                   positions, output_channels);
        return;
    }
#endif
    rggml_conv1d_f32_tile_scalar(dst, dst_stride, tile, weights, rows,
                                 positions, output_channels);
}
