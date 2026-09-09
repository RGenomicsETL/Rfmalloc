/*
 * AVX2/FMA implementation for project-owned packed F32 conv1d rows.
 *
 * Copyright (C) 2026 Sounkou Mahamane Toure
 *
 * This file is part of Rggml and is GPL-2-or-later.
 */
#include <immintrin.h>

#include "../../src/rggml_conv1d.h"

void
rggml_conv1d_f32_accumulate_avx2(float *dst, const struct ggml_tensor *input,
    const struct Rggml_conv_1d_f32_plan *plan, int64_t batch,
    int64_t output_position)
{
    const int64_t input_length = input->ne[0];
    const char *base = (const char *) input->data;
    const size_t output_channels = (size_t) plan->output_channels;
    const int64_t vector_end = plan->output_channels & ~INT64_C(7);

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
            const __m256 input_vector = _mm256_set1_ps(value);
            int64_t output = 0;
            for (; output < vector_end; output += 8) {
                const __m256 previous = _mm256_loadu_ps(dst + output);
                const __m256 weights = _mm256_loadu_ps(weight + output);
                _mm256_storeu_ps(dst + output,
                    _mm256_fmadd_ps(input_vector, weights, previous));
            }
            for (; output < plan->output_channels; ++output) {
                dst[output] += value * weight[output];
            }
        }
    }
}
