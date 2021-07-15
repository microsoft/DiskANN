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

  //  Slow implementations of the distance functions for machines without AVX2
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
      int32_t result = 0;
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
      for (_s32 i = 0; i < (_s32) length; i++) {
        result += ((int32_t)((int16_t) a[i] - (int16_t) b[i])) *
                  ((int32_t)((int16_t) a[i] - (int16_t) b[i]));
      }
      return (float) result;
    }
#else
      __m128 r = _mm_setzero_ps();
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
        __m128 r2 = _mm_setzero_ps();
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
        __m128 r2 = _mm_setzero_ps();
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
      float result = 0;
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
      for (_s32 i = 0; i < (_s32) length; i++) {
        result += (a[i] - b[i]) * (a[i] - b[i]);
      }
      return result;
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

  template<typename T>
  class DistanceInnerProduct : public Distance<T> {
   public:
    float inner_product(const T *a, const T *b, unsigned size) const {
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
      const float *l = (float *) a;
      const float *r = (float *) b;
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
    float compare(const T *a, const T *b, unsigned size)
        const {  // since we use normally minimization objective for distance
                 // comparisons, we are returning 1/x.
      float result = inner_product(a, b, size);
      //      if (result < 0)
      //      return std::numeric_limits<float>::max();
      //      else
      return -result;
    }
  };

  template<typename T>
  class DistanceFastL2
      : public DistanceInnerProduct<T> {  // currently defined only for float.
                                          // templated for future use.
   public:
    float norm(const T *a, unsigned size) const {
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
      const float *l = (float *) a;
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
    using DistanceInnerProduct<T>::compare;
    float compare(const T *a, const T *b, float norm,
                  unsigned size) const {  // not implement
      float result = -2 * DistanceInnerProduct<T>::inner_product(a, b, size);
      result += norm;
      return result;
    }
  };
}  // namespace diskann
