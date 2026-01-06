#pragma once

#include <cstdint>

#include "bfloat16.h"

namespace diskann
{
// Returns true if this build produced AMX BF16 kernels (i.e. the compiler supported
// AMX BF16 intrinsics for the relevant translation unit).
bool amxbf16_kernels_compiled();

// Returns true if the current CPU + OS context is capable of executing AMX BF16 instructions.
// This performs feature detection (CPUID + XCR0) and checks / requests Linux permissions when needed.
bool amxbf16_runtime_available();

// Dot product of bf16 vectors with f32 accumulation using AMX BF16.
// If AMX BF16 is not available at runtime, this falls back to a scalar implementation.
float bf16_dot_f32_accum_amx(const bfloat16 *a, const bfloat16 *b, uint32_t length);

// Batch dot products: computes out[i] = dot(base[i], query) for i in [0, n_vecs).
// base is a row-major matrix of shape [n_vecs x dim].
// If AMX BF16 is not available at runtime, this falls back to a scalar implementation.
void bf16_dot_f32_accum_amx_batch(const bfloat16 *base, const bfloat16 *query, uint32_t n_vecs, uint32_t dim,
								 float *out);

// Matrix of dot products: out[i * n_queries + j] = dot(base[i], queries[j]).
// base is row-major [n_base x dim], queries is row-major [n_queries x dim].
// If AMX BF16 is not available at runtime, this falls back to a scalar implementation.
void bf16_dot_f32_accum_amx_matmul(const bfloat16 *base, const bfloat16 *queries, uint32_t n_base, uint32_t n_queries,
                                  uint32_t dim, float *out);

} // namespace diskann
