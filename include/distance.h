#pragma once

#include <utils.h>
#ifdef _WINDOWS
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#include <cosine_similarity.h>
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

namespace diskann {
  enum Metric { L2 = 0, INNER_PRODUCT = 1, FAST_L2 = 2, PQ = 3 };
  template<typename T>
  class Distance {
   public:
    virtual float compare(const T *a, const T *b, unsigned length) const = 0;
    virtual ~Distance() {
    }
  };

  template<typename T>
  class DistanceCosine : public Distance<T> {
    float compare(const T *a, const T *b, unsigned length) const {
      return diskann::compute_cosine_similarity<T>(a, b, length);
    }
  };

  class DistanceL2Int8 : public Distance<int8_t> {
   public:
    float compare(const int8_t *a, const int8_t *b, unsigned size) const {
      int32_t result = 0;
#ifndef _WINDOWS
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
#endif
      for (_s32 i = 0; i < (_s32) size; i++) {
        result += ((int32_t)((int16_t) a[i] - (int16_t) b[i])) *
                  ((int32_t)((int16_t) a[i] - (int16_t) b[i]));
      }
      return (float) result;
    }
  };

  class DistanceL2UInt8 : public Distance<uint8_t> {
   public:
    float compare(const uint8_t *a, const uint8_t *b, unsigned size) const {
      uint32_t result = 0;
#ifndef _WINDOWS
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
#endif
      for (_s32 i = 0; i < (_s32) size; i++) {
        result += ((int32_t)((int16_t) a[i] - (int16_t) b[i])) *
                  ((int32_t)((int16_t) a[i] - (int16_t) b[i]));
      }
      return (float) result;
    }
  };

  class DistanceL2 : public Distance<float> {
   public:
#ifndef _WINDOWS
    float compare(const float *a, const float *b, unsigned size) const
        __attribute__((hot)) {
      a = (const float *) __builtin_assume_aligned(a, 32);
      b = (const float *) __builtin_assume_aligned(b, 32);
#else
    float compare(const float *a, const float *b, unsigned size) const {
#endif

      float result = 0;
#ifdef USE_AVX2
      // assume size is divisible by 8
      _u16   niters = size / 8;
      __m256 sum = _mm256_setzero_ps();
      for (_u16 j = 0; j < niters; j++) {
        // scope is a[8j:8j+7], b[8j:8j+7]
        // load a_vec
        if (j < (niters - 1)) {
          _mm_prefetch((char *) (a + 8 * (j + 1)), _MM_HINT_T0);
          _mm_prefetch((char *) (b + 8 * (j + 1)), _MM_HINT_T0);
        }
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
#ifndef _WINDOWS
#pragma omp simd reduction(+ : result) aligned(a, b : 32)
#endif
      for (_s32 i = 0; i < (_s32) size; i++) {
        result += (a[i] - b[i]) * (a[i] - b[i]);
      }
#endif
      return result;
    }
  };

}  // namespace diskann
