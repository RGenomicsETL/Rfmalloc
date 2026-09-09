/*
 * Runtime dispatch for project-owned packed F32 conv1d rows.
 *
 * Copyright (C) 2026 Sounkou Mahamane Toure
 *
 * This file is part of Rggml and is GPL-2-or-later.
 */
#include "cpu_features.h"
#include "../../src/rggml_conv1d.h"

#ifdef RGGML_HAVE_X86_AVX2
extern void rggml_conv1d_f32_accumulate_avx2(float *dst,
    const struct ggml_tensor *input, const struct Rggml_conv_1d_f32_plan *plan,
    int64_t batch, int64_t output_position);
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
rggml_conv1d_f32_accumulate(float *dst, const struct ggml_tensor *input,
    const struct Rggml_conv_1d_f32_plan *plan, int64_t batch,
    int64_t output_position)
{
#ifdef RGGML_HAVE_X86_AVX2
    if (!plan->scalar_only && rggml_conv1d_use_avx2) {
        rggml_conv1d_f32_accumulate_avx2(dst, input, plan, batch, output_position);
        return;
    }
#endif
    rggml_conv1d_f32_accumulate_scalar(dst, input, plan, batch, output_position);
}
