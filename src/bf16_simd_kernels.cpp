// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "bf16_simd_kernels.h"

#include <cstdlib>
#include <cmath>

#if defined(__AVX512BF16__) && defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace diskann
{

static inline float bf16_dot_scalar(const bfloat16 *a, const bfloat16 *b, uint32_t length)
{
    float dot = 0.0f;
#ifndef _WINDOWS
#pragma omp simd reduction(+ : dot) aligned(a, b : 8)
#endif
    for (int32_t i = 0; i < (int32_t)length; i++)
    {
        dot += a[i].to_float() * b[i].to_float();
    }
    return dot;
}

#if defined(__AVX512BF16__) && defined(__AVX512F__)

bool avx512bf16_kernels_compiled()
{
    return true;
}

// AVX-512 BF16 dot: each _mm512_dpbf16_ps consumes 32 bf16 elements and accumulates
// into 16 fp32 lanes (pairwise dot). We reduce the accumulator at the end.
float bf16_dot_f32_accum(const bfloat16 *a, const bfloat16 *b, uint32_t length)
{
    constexpr uint32_t kStep = 32;

    __m512 acc = _mm512_setzero_ps();
    uint32_t i = 0;

    for (; i + (kStep - 1) < length; i += kStep)
    {
        // Load 32 bf16 values (64 bytes) for each vector.
        const __m512i va_i = _mm512_loadu_si512((const void *)(a + i));
        const __m512i vb_i = _mm512_loadu_si512((const void *)(b + i));

        // Reinterpret as bf16 vectors.
        const __m512bh va = (__m512bh)va_i;
        const __m512bh vb = (__m512bh)vb_i;

        acc = _mm512_dpbf16_ps(acc, va, vb);
    }

    alignas(64) float lanes[16];
    _mm512_store_ps(lanes, acc);

    float dot = 0.0f;
    for (int lane = 0; lane < 16; ++lane)
        dot += lanes[lane];

    // Remainder.
    if (i < length)
    {
        dot += bf16_dot_scalar(a + i, b + i, length - i);
    }

    return dot;
}

#else

bool avx512bf16_kernels_compiled()
{
    return false;
}

float bf16_dot_f32_accum(const bfloat16 *a, const bfloat16 *b, uint32_t length)
{
    return bf16_dot_scalar(a, b, length);
}

#endif

} // namespace diskann
