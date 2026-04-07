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

// Horizontal sum for 8x int32 (AVX2)
static inline int32_t _mm256_reduce_add_epi32(__m256i x)
{
    // Add high 128 bits to low 128 bits
    __m128i x128 = _mm_add_epi32(_mm256_extracti128_si256(x, 1), _mm256_castsi256_si128(x));
    // Horizontal add: (a0+a2, a1+a3, a0+a2, a1+a3)
    __m128i x64 = _mm_add_epi32(x128, _mm_shuffle_epi32(x128, _MM_SHUFFLE(1, 0, 3, 2)));
    // Final add: a0+a1+a2+a3
    __m128i x32 = _mm_add_epi32(x64, _mm_shuffle_epi32(x64, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(x32);
}

// AVX2 squared L2 distance for uint8 vectors
// Computes sum((a[i] - b[i])^2) using SIMD
// Strategy: Process 32 uint8 elements per iteration
//   1. Load 32 uint8 from a and b
//   2. Compute absolute difference using _mm256_sad_epu8 trick or manual subtraction
//   3. For L2 we need (a-b)^2, so we use: unpack to int16, subtract, square, accumulate
static inline float avx2_l2_distance_uint8(const uint8_t *a, const uint8_t *b, uint32_t size)
{
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();

    // Process 32 bytes at a time
    while (size >= 32)
    {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b));

        // Split into low and high 16 bytes for each 128-bit lane
        // Unpack to 16-bit: this gives us signed 16-bit values from unsigned 8-bit
        __m256i va_lo = _mm256_unpacklo_epi8(va, _mm256_setzero_si256());
        __m256i va_hi = _mm256_unpackhi_epi8(va, _mm256_setzero_si256());
        __m256i vb_lo = _mm256_unpacklo_epi8(vb, _mm256_setzero_si256());
        __m256i vb_hi = _mm256_unpackhi_epi8(vb, _mm256_setzero_si256());

        // Compute difference (a - b) as signed 16-bit
        __m256i diff_lo = _mm256_sub_epi16(va_lo, vb_lo);
        __m256i diff_hi = _mm256_sub_epi16(va_hi, vb_hi);

        // Square and accumulate: madd computes a[0]*b[0] + a[1]*b[1] for adjacent pairs
        // madd_epi16(diff, diff) = diff[0]^2 + diff[1]^2, diff[2]^2 + diff[3]^2, ...
        sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(diff_lo, diff_lo));
        sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(diff_hi, diff_hi));

        a += 32;
        b += 32;
        size -= 32;
    }

    // Process remaining 16 bytes
    if (size >= 16)
    {
        __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a));
        __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b));

        __m256i va_wide = _mm256_cvtepu8_epi16(va);
        __m256i vb_wide = _mm256_cvtepu8_epi16(vb);

        __m256i diff = _mm256_sub_epi16(va_wide, vb_wide);
        sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(diff, diff));

        a += 16;
        b += 16;
        size -= 16;
    }

    // Combine accumulators
    __m256i total = _mm256_add_epi32(sum1, sum2);
    int32_t result = _mm256_reduce_add_epi32(total);

    // Handle remaining elements (< 16) with scalar code
    while (size > 0)
    {
        int32_t diff = static_cast<int32_t>(*a) - static_cast<int32_t>(*b);
        result += diff * diff;
        a++;
        b++;
        size--;
    }

    return static_cast<float>(result);
}

} // namespace diskann
