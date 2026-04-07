#pragma once

#ifdef _WINDOWS
#include <immintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#include <cstdint>

namespace diskann
{
static inline __m256 _mm256_mul_epi8(__m256i X)
{
    __m256i zero = _mm256_setzero_si256();

    __m256i sign_x = _mm256_cmpgt_epi8(zero, X);

    __m256i xlo = _mm256_unpacklo_epi8(X, sign_x);
    __m256i xhi = _mm256_unpackhi_epi8(X, sign_x);

    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(xlo, xlo), _mm256_madd_epi16(xhi, xhi)));
}

static inline __m128 _mm_mulhi_epi8(__m128i X)
{
    __m128i zero = _mm_setzero_si128();
    __m128i sign_x = _mm_cmplt_epi8(X, zero);
    __m128i xhi = _mm_unpackhi_epi8(X, sign_x);

    return _mm_cvtepi32_ps(_mm_add_epi32(_mm_setzero_si128(), _mm_madd_epi16(xhi, xhi)));
}

static inline __m128 _mm_mulhi_epi8_shift32(__m128i X)
{
    __m128i zero = _mm_setzero_si128();
    X = _mm_srli_epi64(X, 32);
    __m128i sign_x = _mm_cmplt_epi8(X, zero);
    __m128i xhi = _mm_unpackhi_epi8(X, sign_x);

    return _mm_cvtepi32_ps(_mm_add_epi32(_mm_setzero_si128(), _mm_madd_epi16(xhi, xhi)));
}
static inline __m128 _mm_mul_epi8(__m128i X, __m128i Y)
{
    __m128i zero = _mm_setzero_si128();

    __m128i sign_x = _mm_cmplt_epi8(X, zero);
    __m128i sign_y = _mm_cmplt_epi8(Y, zero);

    __m128i xlo = _mm_unpacklo_epi8(X, sign_x);
    __m128i xhi = _mm_unpackhi_epi8(X, sign_x);
    __m128i ylo = _mm_unpacklo_epi8(Y, sign_y);
    __m128i yhi = _mm_unpackhi_epi8(Y, sign_y);

    return _mm_cvtepi32_ps(_mm_add_epi32(_mm_madd_epi16(xlo, ylo), _mm_madd_epi16(xhi, yhi)));
}
static inline __m128 _mm_mul_epi8(__m128i X)
{
    __m128i zero = _mm_setzero_si128();
    __m128i sign_x = _mm_cmplt_epi8(X, zero);
    __m128i xlo = _mm_unpacklo_epi8(X, sign_x);
    __m128i xhi = _mm_unpackhi_epi8(X, sign_x);

    return _mm_cvtepi32_ps(_mm_add_epi32(_mm_madd_epi16(xlo, xlo), _mm_madd_epi16(xhi, xhi)));
}

static inline __m128 _mm_mul32_pi8(__m128i X, __m128i Y)
{
    __m128i xlo = _mm_cvtepi8_epi16(X), ylo = _mm_cvtepi8_epi16(Y);
    return _mm_cvtepi32_ps(_mm_unpacklo_epi32(_mm_madd_epi16(xlo, ylo), _mm_setzero_si128()));
}

static inline __m256 _mm256_mul_epi8(__m256i X, __m256i Y)
{
    __m256i zero = _mm256_setzero_si256();

    __m256i sign_x = _mm256_cmpgt_epi8(zero, X);
    __m256i sign_y = _mm256_cmpgt_epi8(zero, Y);

    __m256i xlo = _mm256_unpacklo_epi8(X, sign_x);
    __m256i xhi = _mm256_unpackhi_epi8(X, sign_x);
    __m256i ylo = _mm256_unpacklo_epi8(Y, sign_y);
    __m256i yhi = _mm256_unpackhi_epi8(Y, sign_y);

    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(xlo, ylo), _mm256_madd_epi16(xhi, yhi)));
}

static inline __m256 _mm256_mul32_pi8(__m128i X, __m128i Y)
{
    __m256i xlo = _mm256_cvtepi8_epi16(X), ylo = _mm256_cvtepi8_epi16(Y);
    return _mm256_blend_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(xlo, ylo)), _mm256_setzero_ps(), 252);
}

static inline float _mm256_reduce_add_ps(__m256 x)
{
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}

// Horizontal sum for __m256i (32-bit integers)
static inline int32_t _mm256_reduce_add_epi32(__m256i x)
{
    __m128i lo = _mm256_castsi256_si128(x);
    __m128i hi = _mm256_extracti128_si256(x, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    __m128i sum64 = _mm_add_epi32(sum128, hi64);
    __m128i hi32 = _mm_shuffle_epi32(sum64, 0x1);
    __m128i sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);
}

// AVX2: Compute squared L2 distance for 32 uint8 values
// Returns sum of (a[i] - b[i])^2 as float
static inline __m256 _mm256_sqrdiff_epu8(__m256i a, __m256i b)
{
    // For uint8, we need zero-extension (not sign-extension)
    __m256i zero = _mm256_setzero_si256();

    // Unpack to 16-bit: interleave with zeros
    __m256i a_lo = _mm256_unpacklo_epi8(a, zero);
    __m256i a_hi = _mm256_unpackhi_epi8(a, zero);
    __m256i b_lo = _mm256_unpacklo_epi8(b, zero);
    __m256i b_hi = _mm256_unpackhi_epi8(b, zero);

    // Compute signed 16-bit differences
    __m256i diff_lo = _mm256_sub_epi16(a_lo, b_lo);
    __m256i diff_hi = _mm256_sub_epi16(a_hi, b_hi);

    // Use madd to compute sum of squares: madd(x, x) = x[0]^2 + x[1]^2, ...
    // Each pair of 16-bit values becomes one 32-bit value
    __m256i sq_lo = _mm256_madd_epi16(diff_lo, diff_lo);
    __m256i sq_hi = _mm256_madd_epi16(diff_hi, diff_hi);

    // Sum the 32-bit results and convert to float
    return _mm256_cvtepi32_ps(_mm256_add_epi32(sq_lo, sq_hi));
}

// AVX2: Compute dot product for 32 uint8 values, returns partial sum as __m256i (32-bit lanes)
static inline __m256i _mm256_dp_epu8(__m256i a, __m256i b)
{
    __m256i zero = _mm256_setzero_si256();

    // Zero-extend uint8 to uint16
    __m256i a_lo = _mm256_unpacklo_epi8(a, zero);
    __m256i a_hi = _mm256_unpackhi_epi8(a, zero);
    __m256i b_lo = _mm256_unpacklo_epi8(b, zero);
    __m256i b_hi = _mm256_unpackhi_epi8(b, zero);

    // madd: multiply pairs of 16-bit values and add adjacent pairs to 32-bit
    __m256i prod_lo = _mm256_madd_epi16(a_lo, b_lo);
    __m256i prod_hi = _mm256_madd_epi16(a_hi, b_hi);

    return _mm256_add_epi32(prod_lo, prod_hi);
}

// AVX2: Compute sum of squares for 32 uint8 values, returns partial sum as __m256i (32-bit lanes)
static inline __m256i _mm256_sqr_epu8(__m256i a)
{
    __m256i zero = _mm256_setzero_si256();

    __m256i a_lo = _mm256_unpacklo_epi8(a, zero);
    __m256i a_hi = _mm256_unpackhi_epi8(a, zero);

    __m256i sq_lo = _mm256_madd_epi16(a_lo, a_lo);
    __m256i sq_hi = _mm256_madd_epi16(a_hi, a_hi);

    return _mm256_add_epi32(sq_lo, sq_hi);
}

// SSE: Compute squared L2 distance for 16 uint8 values
static inline __m128 _mm_sqrdiff_epu8(__m128i a, __m128i b)
{
    __m128i zero = _mm_setzero_si128();

    __m128i a_lo = _mm_unpacklo_epi8(a, zero);
    __m128i a_hi = _mm_unpackhi_epi8(a, zero);
    __m128i b_lo = _mm_unpacklo_epi8(b, zero);
    __m128i b_hi = _mm_unpackhi_epi8(b, zero);

    __m128i diff_lo = _mm_sub_epi16(a_lo, b_lo);
    __m128i diff_hi = _mm_sub_epi16(a_hi, b_hi);

    __m128i sq_lo = _mm_madd_epi16(diff_lo, diff_lo);
    __m128i sq_hi = _mm_madd_epi16(diff_hi, diff_hi);

    return _mm_cvtepi32_ps(_mm_add_epi32(sq_lo, sq_hi));
}

} // namespace diskann
