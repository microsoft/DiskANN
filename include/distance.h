#pragma once

#include <utils.h>
#ifdef _WINDOWS
#include <immintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#include <cosine_similarity.h>
#include <iostream>

namespace {
  static inline __m128 _mm_mulhi_epi8(__m128i X) {
    __m128i zero = _mm_setzero_si128();
    __m128i sign_x = _mm_cmplt_epi8(X, zero);
    __m128i xhi = _mm_unpackhi_epi8(X, sign_x);

    return _mm_cvtepi32_ps(
        _mm_add_epi32(_mm_setzero_si128(), _mm_madd_epi16(xhi, xhi)));
  }

  static inline __m128 _mm_mulhi_epi8_shift32(__m128i X) {
    __m128i zero = _mm_setzero_si128();
    X = _mm_srli_epi64(X, 32);
    __m128i sign_x = _mm_cmplt_epi8(X, zero);
    __m128i xhi = _mm_unpackhi_epi8(X, sign_x);

    return _mm_cvtepi32_ps(
        _mm_add_epi32(_mm_setzero_si128(), _mm_madd_epi16(xhi, xhi)));
  }
  static inline __m128 _mm_mul_epi8(__m128i X, __m128i Y) {
    __m128i zero = _mm_setzero_si128();

    __m128i sign_x = _mm_cmplt_epi8(X, zero);
    __m128i sign_y = _mm_cmplt_epi8(Y, zero);

    __m128i xlo = _mm_unpacklo_epi8(X, sign_x);
    __m128i xhi = _mm_unpackhi_epi8(X, sign_x);
    __m128i ylo = _mm_unpacklo_epi8(Y, sign_y);
    __m128i yhi = _mm_unpackhi_epi8(Y, sign_y);

    return _mm_cvtepi32_ps(
        _mm_add_epi32(_mm_madd_epi16(xlo, ylo), _mm_madd_epi16(xhi, yhi)));
  }
  static inline __m128 _mm_mul_epi8(__m128i X) {
    __m128i zero = _mm_setzero_si128();
    __m128i sign_x = _mm_cmplt_epi8(X, zero);
    __m128i xlo = _mm_unpacklo_epi8(X, sign_x);
    __m128i xhi = _mm_unpackhi_epi8(X, sign_x);

    return _mm_cvtepi32_ps(
        _mm_add_epi32(_mm_madd_epi16(xlo, xlo), _mm_madd_epi16(xhi, xhi)));
  }

  static inline __m128 _mm_mul32_pi8(__m128i X, __m128i Y) {
    __m128i xlo = _mm_cvtepi8_epi16(X), ylo = _mm_cvtepi8_epi16(Y);
    return _mm_cvtepi32_ps(
        _mm_unpacklo_epi32(_mm_madd_epi16(xlo, ylo), _mm_setzero_si128()));
  }

  static inline __m256 _mm256_mul_epi8(__m256i X, __m256i Y) {
    __m256i zero = _mm256_setzero_si256();

    __m256i sign_x = _mm256_cmpgt_epi8(zero, X);
    __m256i sign_y = _mm256_cmpgt_epi8(zero, Y);

    __m256i xlo = _mm256_unpacklo_epi8(X, sign_x);
    __m256i xhi = _mm256_unpackhi_epi8(X, sign_x);
    __m256i ylo = _mm256_unpacklo_epi8(Y, sign_y);
    __m256i yhi = _mm256_unpackhi_epi8(Y, sign_y);

    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(xlo, ylo),
                                               _mm256_madd_epi16(xhi, yhi)));
  }

  static inline __m256 _mm256_mul32_pi8(__m128i X, __m128i Y) {
    __m256i xlo = _mm256_cvtepi8_epi16(X), ylo = _mm256_cvtepi8_epi16(Y);
    return _mm256_blend_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(xlo, ylo)),
                           _mm256_setzero_ps(), 252);
  }

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
  //  enum Metric { L2 = 0, INNER_PRODUCT = 1, FAST_L2 = 2, PQ = 3 };
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

#ifdef _WINDOWS
#ifdef USE_AVX2
      __m256 r = _mm256_setzero_ps();
      char * pX = (char *) a, *pY = (char *) b;
      while (size >= 32) {
        __m256i r1 = _mm256_subs_epi8(_mm256_loadu_si256((__m256i *) pX),
                                      _mm256_loadu_si256((__m256i *) pY));
        r = _mm256_add_ps(r, _mm256_mul_epi8(r1, r1));
        pX += 32;
        pY += 32;
        size -= 32;
      }
      while (size > 0) {
        __m128i r2 = _mm_subs_epi8(_mm_loadu_si128((__m128i *) pX),
                                   _mm_loadu_si128((__m128i *) pY));
        r = _mm256_add_ps(r, _mm256_mul32_pi8(r2, r2));
        pX += 4;
        pY += 4;
        size -= 4;
      }
      r = _mm256_hadd_ps(_mm256_hadd_ps(r, r), r);
      return r.m256_f32[0] + r.m256_f32[4];
#else
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
      for (_s32 i = 0; i < (_s32) size; i++) {
        result += ((int32_t)((int16_t) a[i] - (int16_t) b[i])) *
                  ((int32_t)((int16_t) a[i] - (int16_t) b[i]));
      }
      return (float) result;
#endif
#else
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
      for (_s32 i = 0; i < (_s32) size; i++) {
        result += ((int32_t)((int16_t) a[i] - (int16_t) b[i])) *
                  ((int32_t)((int16_t) a[i] - (int16_t) b[i]));
      }
      return (float) result;
#endif
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

  // Gopal. Slow implementations of the distance functions to get diskann to
  // work in v14 machines that do not have AVX2 support. Performance here is not
  // a concern, so we are using the simplest possible implementation.
  template<typename T>
  class SlowDistanceL2Int : public Distance<T> {
    virtual float compare(const T *a, const T *b, unsigned length) const {
      uint32_t result = 0;
      for (_u32 i = 0; i < length; i++) {
        result += ((int32_t)((int16_t) a[i] - (int16_t) b[i])) *
                  ((int32_t)((int16_t) a[i] - (int16_t) b[i]));
      }
      return (float) result;
    }
  };

  class SlowDistanceL2Float : public Distance<float> {
    virtual float compare(const float *a, const float *b,
                          unsigned length) const {
      float result = 0.0f;
      for (_u32 i = 0; i < length; i++) {
        result += (a[i] - b[i]) * (a[i] - b[i]);
      }
      return result;
    }
  };

  class AVXDistanceL2Int8 : public Distance<int8_t> {
   public:
    virtual float compare(const int8_t *a, const int8_t *b,
                          unsigned int length) const {
#ifndef _WINDOWS
      std::cout << "AVX only supported in Windows build.";
      return 0;
    }
#else
      __m128  r = _mm_setzero_ps();
      __m128i r1;
      while (length >= 16) {
        r1 = _mm_subs_epi8(_mm_load_si128((__m128i *) a),
                           _mm_load_si128((__m128i *) b));
        r = _mm_add_ps(r, _mm_mul_epi8(r1));
        a += 16;
        b += 16;
        length -= 16;
      }
      r = _mm_hadd_ps(_mm_hadd_ps(r, r), r);
      float res = r.m128_f32[0];

      if (length >= 8) {
        __m128  r2 = _mm_setzero_ps();
        __m128i r3 = _mm_subs_epi8(_mm_load_si128((__m128i *) (a - 8)),
                                   _mm_load_si128((__m128i *) (b - 8)));
        r2 = _mm_add_ps(r2, _mm_mulhi_epi8(r3));
        a += 8;
        b += 8;
        length -= 8;
        r2 = _mm_hadd_ps(_mm_hadd_ps(r2, r2), r2);
        res += r2.m128_f32[0];
      }

      if (length >= 4) {
        __m128  r2 = _mm_setzero_ps();
        __m128i r3 = _mm_subs_epi8(_mm_load_si128((__m128i *) (a - 12)),
                                   _mm_load_si128((__m128i *) (b - 12)));
        r2 = _mm_add_ps(r2, _mm_mulhi_epi8_shift32(r3));
        res += r2.m128_f32[0] + r2.m128_f32[1];
      }

      return res;
    }
#endif
  };

  class AVXDistanceL2Float : public Distance<float> {
   public:
    virtual float compare(const float *a, const float *b,
                          unsigned int length) const {
#ifndef _WINDOWS
      std::cout << "AVX only supported in Windows build.";
      return 0;
    }
#else
      __m128 diff, v1, v2;
      __m128 sum = _mm_set1_ps(0);

      while (length >= 4) {
        v1 = _mm_loadu_ps(a);
        a += 4;
        v2 = _mm_loadu_ps(b);
        b += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        length -= 4;
      }

      return sum.m128_f32[0] + sum.m128_f32[1] + sum.m128_f32[2] +
             sum.m128_f32[3];
    }
#endif
  };
}  // namespace diskann
