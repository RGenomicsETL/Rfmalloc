/*
 * AVX2/FMA implementation for project-owned packed F32 conv1d tiles.
 *
 * Copyright (C) 2026 Sounkou Mahamane Toure
 *
 * This file is part of Rggml and is GPL-2-or-later.
 */
#include <immintrin.h>

#include "../../src/rggml_conv1d.h"

/* The hot shape: four output channels by three position vectors keeps twelve
 * accumulators, three input vectors and one broadcast live, which is exactly
 * the sixteen YMM registers. Each gathered input vector then feeds four fused
 * multiply-adds instead of one, so the loop stops being load-bound. */
static inline void
rggml_conv1d_avx2_block24x4(float *dst, int64_t dst_stride,
    const float *const *rows_at, const float *weights, int64_t rows,
    int64_t output_channels, int64_t p0, int64_t oc0)
{
    float *out0 = dst + oc0 * dst_stride + p0;
    float *out1 = out0 + dst_stride;
    float *out2 = out1 + dst_stride;
    float *out3 = out2 + dst_stride;
    __m256 a00 = _mm256_loadu_ps(out0), a01 = _mm256_loadu_ps(out0 + 8), a02 = _mm256_loadu_ps(out0 + 16);
    __m256 a10 = _mm256_loadu_ps(out1), a11 = _mm256_loadu_ps(out1 + 8), a12 = _mm256_loadu_ps(out1 + 16);
    __m256 a20 = _mm256_loadu_ps(out2), a21 = _mm256_loadu_ps(out2 + 8), a22 = _mm256_loadu_ps(out2 + 16);
    __m256 a30 = _mm256_loadu_ps(out3), a31 = _mm256_loadu_ps(out3 + 8), a32 = _mm256_loadu_ps(out3 + 16);

    for (int64_t row = 0; row < rows; ++row) {
        const float *x = rows_at[row] + p0;
        const float *w = weights + row * output_channels + oc0;
        const __m256 x0 = _mm256_loadu_ps(x);
        const __m256 x1 = _mm256_loadu_ps(x + 8);
        const __m256 x2 = _mm256_loadu_ps(x + 16);
        __m256 wb = _mm256_broadcast_ss(w);
        a00 = _mm256_fmadd_ps(x0, wb, a00);
        a01 = _mm256_fmadd_ps(x1, wb, a01);
        a02 = _mm256_fmadd_ps(x2, wb, a02);
        wb = _mm256_broadcast_ss(w + 1);
        a10 = _mm256_fmadd_ps(x0, wb, a10);
        a11 = _mm256_fmadd_ps(x1, wb, a11);
        a12 = _mm256_fmadd_ps(x2, wb, a12);
        wb = _mm256_broadcast_ss(w + 2);
        a20 = _mm256_fmadd_ps(x0, wb, a20);
        a21 = _mm256_fmadd_ps(x1, wb, a21);
        a22 = _mm256_fmadd_ps(x2, wb, a22);
        wb = _mm256_broadcast_ss(w + 3);
        a30 = _mm256_fmadd_ps(x0, wb, a30);
        a31 = _mm256_fmadd_ps(x1, wb, a31);
        a32 = _mm256_fmadd_ps(x2, wb, a32);
    }
    _mm256_storeu_ps(out0, a00); _mm256_storeu_ps(out0 + 8, a01); _mm256_storeu_ps(out0 + 16, a02);
    _mm256_storeu_ps(out1, a10); _mm256_storeu_ps(out1 + 8, a11); _mm256_storeu_ps(out1 + 16, a12);
    _mm256_storeu_ps(out2, a20); _mm256_storeu_ps(out2 + 8, a21); _mm256_storeu_ps(out2 + 16, a22);
    _mm256_storeu_ps(out3, a30); _mm256_storeu_ps(out3 + 8, a31); _mm256_storeu_ps(out3 + 16, a32);
}

/* One eight-position vector against up to four output channels: the remainder
 * of a position block, and every block of a short output. */
static inline void
rggml_conv1d_avx2_block8(float *dst, int64_t dst_stride,
    const float *const *rows_at, const float *weights, int64_t rows,
    int64_t output_channels, int64_t p0, int64_t oc0, int64_t channels)
{
    float *out[4];
    __m256 acc[4];

    for (int64_t j = 0; j < channels; ++j) {
        out[j] = dst + (oc0 + j) * dst_stride + p0;
        acc[j] = _mm256_loadu_ps(out[j]);
    }
    for (int64_t row = 0; row < rows; ++row) {
        const __m256 x = _mm256_loadu_ps(rows_at[row] + p0);
        const float *w = weights + row * output_channels + oc0;
        for (int64_t j = 0; j < channels; ++j) {
            acc[j] = _mm256_fmadd_ps(x, _mm256_broadcast_ss(w + j), acc[j]);
        }
    }
    for (int64_t j = 0; j < channels; ++j) _mm256_storeu_ps(out[j], acc[j]);
}

void
rggml_conv1d_f32_tile_avx2(float *dst, int64_t dst_stride,
    const float *const *rows_at, const float *weights, int64_t rows,
    int64_t positions, int64_t output_channels)
{
    const int64_t vector_end = positions & ~INT64_C(7);

    for (int64_t oc0 = 0; oc0 < output_channels; oc0 += 4) {
        const int64_t channels = output_channels - oc0 < 4
            ? output_channels - oc0 : 4;
        int64_t p0 = 0;

        if (channels == 4) {
            for (; p0 + 24 <= vector_end; p0 += 24) {
                rggml_conv1d_avx2_block24x4(dst, dst_stride, rows_at, weights,
                                            rows, output_channels, p0, oc0);
            }
        }
        for (; p0 + 8 <= vector_end; p0 += 8) {
            rggml_conv1d_avx2_block8(dst, dst_stride, rows_at, weights, rows,
                                     output_channels, p0, oc0, channels);
        }
        if (p0 < positions) {
            /* Fewer than eight positions are left: the portable form is both
             * correct and short enough not to matter. */
            for (int64_t row = 0; row < rows; ++row) {
                const float *x = rows_at[row];
                const float *w = weights + row * output_channels + oc0;
                for (int64_t j = 0; j < channels; ++j) {
                    float *out = dst + (oc0 + j) * dst_stride;
                    const float weight = w[j];
                    for (int64_t p = p0; p < positions; ++p) out[p] += weight * x[p];
                }
            }
        }
    }
}

void
rggml_conv1d_f32_activate_avx2(float *dst, const float *src, int64_t n,
    float scale, float shift, int leaky, float slope)
{
    const __m256 vscale = _mm256_set1_ps(scale);
    const __m256 vshift = _mm256_set1_ps(shift);
    const __m256 vslope = _mm256_set1_ps(slope);
    const __m256 zero = _mm256_setzero_ps();
    int64_t i = 0;

    if (leaky) {
        for (; i + 8 <= n; i += 8) {
            const __m256 v = _mm256_fmadd_ps(_mm256_loadu_ps(src + i), vscale, vshift);
            const __m256 mask = _mm256_cmp_ps(v, zero, _CMP_GT_OQ);
            _mm256_storeu_ps(dst + i,
                _mm256_blendv_ps(_mm256_mul_ps(v, vslope), v, mask));
        }
        for (; i < n; ++i) {
            const float v = src[i] * scale + shift;
            dst[i] = v > 0.0f ? v : v * slope;
        }
        return;
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i,
            _mm256_fmadd_ps(_mm256_loadu_ps(src + i), vscale, vshift));
    }
    for (; i < n; ++i) dst[i] = src[i] * scale + shift;
}
