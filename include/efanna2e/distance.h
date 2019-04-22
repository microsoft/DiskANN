//
// Created by 付聪 on 2017/6/21.
//

#pragma once

#include <efanna2e/util.h>
#ifdef __NSG_WINDOWS__
#include <intrin.h>
#else
#include <immintrin.h>
#endif
#include <iostream>

namespace {
  static inline float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 =
        _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
  }
}  // namespace

namespace NSG {
  enum Metric { L2 = 0, INNER_PRODUCT = 1, FAST_L2 = 2, PQ = 3 };
  class Distance {
   public:
    virtual float compare(const float *a, const float *b,
                          unsigned length) const = 0;
    virtual ~Distance() {
    }
  };

  class DistanceL2 : public Distance {
   public:
    float compare(const float *a, const float *b, unsigned size) const {
      float result = 0;
#ifdef USE_AVX2
      // assume size is divisible by 8
      _u64   niters = size / 8;
      __m256 sum = _mm256_setzero_ps();
      for (_u64 j = 0; j < niters; j++) {
        // scope is a[8j:8j+7], b[8j:8j+7]
        // load a_vec
        __m256 a_vec = _mm256_load_ps(a + 8 * j);
        // load b_vec
        __m256 b_vec = _mm256_load_ps(b + 8 * j);
        // a_vec - b_vec
        __m256 tmp_vec = _mm256_sub_ps(a_vec, b_vec);
        /*
    // (a_vec - b_vec)**2
        __m256 tmp_vec2 = _mm256_mul_ps(tmp_vec, tmp_vec);
    // accumulate sum
        sum = _mm256_add_ps(sum, tmp_vec2);
    */
        // sum = (tmp_vec**2) + sum
        sum = _mm256_fmadd_ps(tmp_vec, tmp_vec, sum);
      }

      // horizontal add sum
      result = _mm256_reduce_add_ps(sum);
#else
      // #pragma omp for simd reduction(+ : result)
      for (unsigned i = 0; i < size; i++) {
        float temp = (a[i] - b[i]);
        result += temp * temp;
      }
#endif
      return result;
    }
  };

  class DistanceInnerProduct : public Distance {
   public:
    float compare(const float *a, const float *b, unsigned size) const {
      float result = 0;
#ifdef __GNUC__
#ifdef __AVX__
#define AVX_DOT(addr1, addr2, dest, tmp1, tmp2) \
  tmp1 = _mm256_loadu_ps(addr1);                \
  tmp2 = _mm256_loadu_ps(addr2);                \
  tmp1 = _mm256_mul_ps(tmp1, tmp2);             \
  dest = _mm256_add_ps(dest, tmp1);

      __m256       sum;
      __m256       l0, l1;
      __m256       r0, r1;
      unsigned     D = (size + 7) & ~7U;
      unsigned     DR = D % 16;
      unsigned     DD = D - DR;
      const float *l = a;
      const float *r = b;
      const float *e_l = l + DD;
      const float *e_r = r + DD;
      float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};

      sum = _mm256_loadu_ps(unpack);
      if (DR) {
        AVX_DOT(e_l, e_r, sum, l0, r0);
      }

      for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
        AVX_DOT(l, r, sum, l0, r0);
        AVX_DOT(l + 8, r + 8, sum, l1, r1);
      }
      _mm256_storeu_ps(unpack, sum);
      result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] +
               unpack[5] + unpack[6] + unpack[7];

#else
#ifdef __SSE2__
#define SSE_DOT(addr1, addr2, dest, tmp1, tmp2) \
  tmp1 = _mm128_loadu_ps(addr1);                \
  tmp2 = _mm128_loadu_ps(addr2);                \
  tmp1 = _mm128_mul_ps(tmp1, tmp2);             \
  dest = _mm128_add_ps(dest, tmp1);
      __m128       sum;
      __m128       l0, l1, l2, l3;
      __m128       r0, r1, r2, r3;
      unsigned     D = (size + 3) & ~3U;
      unsigned     DR = D % 16;
      unsigned     DD = D - DR;
      const float *l = a;
      const float *r = b;
      const float *e_l = l + DD;
      const float *e_r = r + DD;
      float        unpack[4] __attribute__((aligned(16))) = {0, 0, 0, 0};

      sum = _mm_load_ps(unpack);
      switch (DR) {
        case 12:
          SSE_DOT(e_l + 8, e_r + 8, sum, l2, r2);
        case 8:
          SSE_DOT(e_l + 4, e_r + 4, sum, l1, r1);
        case 4:
          SSE_DOT(e_l, e_r, sum, l0, r0);
        default:
          break;
      }
      for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
        SSE_DOT(l, r, sum, l0, r0);
        SSE_DOT(l + 4, r + 4, sum, l1, r1);
        SSE_DOT(l + 8, r + 8, sum, l2, r2);
        SSE_DOT(l + 12, r + 12, sum, l3, r3);
      }
      _mm_storeu_ps(unpack, sum);
      result += unpack[0] + unpack[1] + unpack[2] + unpack[3];
#else

      float        dot0, dot1, dot2, dot3;
      const float *last = a + size;
      const float *unroll_group = last - 3;

      /* Process 4 items with each loop for efficiency. */
      while (a < unroll_group) {
        dot0 = a[0] * b[0];
        dot1 = a[1] * b[1];
        dot2 = a[2] * b[2];
        dot3 = a[3] * b[3];
        result += dot0 + dot1 + dot2 + dot3;
        a += 4;
        b += 4;
      }
      /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
      while (a < last) {
        result += *a++ * *b++;
      }
#endif
#endif
#endif
      return result;
    }
  };
  class DistanceFastL2 : public DistanceInnerProduct {
   public:
    float norm(const float *a, unsigned size) const {
      float result = 0;
#ifdef __GNUC__
#ifdef __AVX__
#define AVX_L2NORM(addr, dest, tmp) \
  tmp = _mm256_loadu_ps(addr);      \
  tmp = _mm256_mul_ps(tmp, tmp);    \
  dest = _mm256_add_ps(dest, tmp);

      __m256       sum;
      __m256       l0, l1;
      unsigned     D = (size + 7) & ~7U;
      unsigned     DR = D % 16;
      unsigned     DD = D - DR;
      const float *l = a;
      const float *e_l = l + DD;
      float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};

      sum = _mm256_loadu_ps(unpack);
      if (DR) {
        AVX_L2NORM(e_l, sum, l0);
      }
      for (unsigned i = 0; i < DD; i += 16, l += 16) {
        AVX_L2NORM(l, sum, l0);
        AVX_L2NORM(l + 8, sum, l1);
      }
      _mm256_storeu_ps(unpack, sum);
      result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] +
               unpack[5] + unpack[6] + unpack[7];
#else
#ifdef __SSE2__
#define SSE_L2NORM(addr, dest, tmp) \
  tmp = _mm128_loadu_ps(addr);      \
  tmp = _mm128_mul_ps(tmp, tmp);    \
  dest = _mm128_add_ps(dest, tmp);

      __m128       sum;
      __m128       l0, l1, l2, l3;
      unsigned     D = (size + 3) & ~3U;
      unsigned     DR = D % 16;
      unsigned     DD = D - DR;
      const float *l = a;
      const float *e_l = l + DD;
      float        unpack[4] __attribute__((aligned(16))) = {0, 0, 0, 0};

      sum = _mm_load_ps(unpack);
      switch (DR) {
        case 12:
          SSE_L2NORM(e_l + 8, sum, l2);
        case 8:
          SSE_L2NORM(e_l + 4, sum, l1);
        case 4:
          SSE_L2NORM(e_l, sum, l0);
        default:
          break;
      }
      for (unsigned i = 0; i < DD; i += 16, l += 16) {
        SSE_L2NORM(l, sum, l0);
        SSE_L2NORM(l + 4, sum, l1);
        SSE_L2NORM(l + 8, sum, l2);
        SSE_L2NORM(l + 12, sum, l3);
      }
      _mm_storeu_ps(unpack, sum);
      result += unpack[0] + unpack[1] + unpack[2] + unpack[3];
#else
      float        dot0, dot1, dot2, dot3;
      const float *last = a + size;
      const float *unroll_group = last - 3;

      /* Process 4 items with each loop for efficiency. */
      while (a < unroll_group) {
        dot0 = a[0] * a[0];
        dot1 = a[1] * a[1];
        dot2 = a[2] * a[2];
        dot3 = a[3] * a[3];
        result += dot0 + dot1 + dot2 + dot3;
        a += 4;
      }
      /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
      while (a < last) {
        result += (*a) * (*a);
        a++;
      }
#endif
#endif
#endif
      return result;
    }
    using DistanceInnerProduct::compare;
    float compare(const float *a, const float *b, float norm,
                  unsigned size) const {  // not implement
      float result = -2 * DistanceInnerProduct::compare(a, b, size);
      result += norm;
      return result;
    }
  };
}  // namespace NSG
