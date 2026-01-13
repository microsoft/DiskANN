#pragma once

#include <cstdint>

#include "bfloat16.h"

namespace diskann
{
// Returns true if this build produced AVX-512 BF16 kernels (i.e. the compiler supported
// AVX-512 BF16 intrinsics for the relevant translation unit).
bool avx512bf16_kernels_compiled();

// Dot product of bf16 vectors with f32 accumulation.
// If AVX-512 BF16 kernels are not compiled in, this falls back to a scalar implementation.
float bf16_dot_f32_accum(const bfloat16 *a, const bfloat16 *b, uint32_t length);

} // namespace diskann
