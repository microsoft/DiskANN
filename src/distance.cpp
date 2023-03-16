// TODO
// CHECK COSINE ON LINUX

#ifdef _WINDOWS
#include <immintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#include "simd_utils.h"
#include <cosine_similarity.h>
#include <iostream>

#include "distance.h"
#include "utils.h"
#include "logger.h"
#include "ann_exception.h"

namespace diskann {

  // This function was taken from: https://github.com/microsoft/SPTAG/blob/main/AnnService/src/Core/Common/DistanceUtils.cpp
  inline __m128 _mm_sqdf_epu8(__m128i X, __m128i Y) {
    __m128i zero = _mm_setzero_si128();

    __m128i xlo = _mm_unpacklo_epi8(X, zero);
    __m128i xhi = _mm_unpackhi_epi8(X, zero);
    __m128i ylo = _mm_unpacklo_epi8(Y, zero);
    __m128i yhi = _mm_unpackhi_epi8(Y, zero);

    __m128i dlo = _mm_sub_epi16(xlo, ylo);
    __m128i dhi = _mm_sub_epi16(xhi, yhi);

    return _mm_cvtepi32_ps(
        _mm_add_epi32(_mm_madd_epi16(dlo, dlo), _mm_madd_epi16(dhi, dhi)));
  }

  // This function was taken from:
  // https://github.com/microsoft/SPTAG/blob/main/AnnService/src/Core/Common/DistanceUtils.cpp
  inline __m128 _mm_sqdf_epi8(__m128i X, __m128i Y) {
    __m128i zero = _mm_setzero_si128();

    __m128i sign_x = _mm_cmplt_epi8(X, zero);
    __m128i sign_y = _mm_cmplt_epi8(Y, zero);

    __m128i xlo = _mm_unpacklo_epi8(X, sign_x);
    __m128i xhi = _mm_unpackhi_epi8(X, sign_x);
    __m128i ylo = _mm_unpacklo_epi8(Y, sign_y);
    __m128i yhi = _mm_unpackhi_epi8(Y, sign_y);

    __m128i dlo = _mm_sub_epi16(xlo, ylo);
    __m128i dhi = _mm_sub_epi16(xhi, yhi);

    return _mm_cvtepi32_ps(
        _mm_add_epi32(_mm_madd_epi16(dlo, dlo), _mm_madd_epi16(dhi, dhi)));
  }

  // This function was taken from:
  // https://github.com/microsoft/SPTAG/blob/main/AnnService/src/Core/Common/DistanceUtils.cpp
  inline __m256 _mm256_sqdf_epi8(__m256i X, __m256i Y) {
    __m256i zero = _mm256_setzero_si256();

    __m256i sign_x = _mm256_cmpgt_epi8(zero, X);
    __m256i sign_y = _mm256_cmpgt_epi8(zero, Y);

    __m256i xlo = _mm256_unpacklo_epi8(X, sign_x);
    __m256i xhi = _mm256_unpackhi_epi8(X, sign_x);
    __m256i ylo = _mm256_unpacklo_epi8(Y, sign_y);
    __m256i yhi = _mm256_unpackhi_epi8(Y, sign_y);

    __m256i dlo = _mm256_sub_epi16(xlo, ylo);
    __m256i dhi = _mm256_sub_epi16(xhi, yhi);

    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(dlo, dlo),
                                               _mm256_madd_epi16(dhi, dhi)));
  }

  // This was taken from:
  // https://github.com/microsoft/SPTAG/blob/main/AnnService/src/Core/Common/DistanceUtils.cpp
#define REPEAT(type, ctype, delta, load, exec, acc, result) \
  {                                                         \
    type c1 = load((ctype *) (a));                         \
    type c2 = load((ctype *) (b));                         \
    a += delta;                                            \
    b += delta;                                            \
    result = acc(result, exec(c1, c2));                     \
  }

  //
  // Cosine distance functions.
  //

  float DistanceCosineInt8::compare(
      const int8_t *a, const int8_t *b, uint32_t length, float break_distance) const {
#ifdef _WINDOWS
    return diskann::CosineSimilarity2<int8_t>(a, b, length);
#else
    int magA = 0, magB = 0, scalarProduct = 0;
    for (uint32_t i = 0; i < length; i++) {
      magA += ((int32_t) a[i]) * ((int32_t) a[i]);
      magB += ((int32_t) b[i]) * ((int32_t) b[i]);
      scalarProduct += ((int32_t) a[i]) * ((int32_t) b[i]);
    }
    // similarity == 1-cosine distance
    return 1.0f - (float) (scalarProduct / (sqrt(magA) * sqrt(magB)));
#endif
  }

  float DistanceCosineFloat::compare(const float *a, const float *b,
                                     uint32_t length,
                                     float    break_distance) const {
#ifdef _WINDOWS
    return diskann::CosineSimilarity2<float>(a, b, length);
#else
    float magA = 0, magB = 0, scalarProduct = 0;
    for (uint32_t i = 0; i < length; i++) {
      magA += (a[i]) * (a[i]);
      magB += (b[i]) * (b[i]);
      scalarProduct += (a[i]) * (b[i]);
    }
    // similarity == 1-cosine distance
    return 1.0f - (scalarProduct / (sqrt(magA) * sqrt(magB)));
#endif
  }

  float SlowDistanceCosineUInt8::compare(const uint8_t *a, const uint8_t *b,
                                         uint32_t length,
                                         float    break_distance) const {
    int magA = 0, magB = 0, scalarProduct = 0;
    for (uint32_t i = 0; i < length; i++) {
      magA += ((uint32_t) a[i]) * ((uint32_t) a[i]);
      magB += ((uint32_t) b[i]) * ((uint32_t) b[i]);
      scalarProduct += ((uint32_t) a[i]) * ((uint32_t) b[i]);
    }
    // similarity == 1-cosine distance
    return 1.0f - (float) (scalarProduct / (sqrt(magA) * sqrt(magB)));
  }

  //
  // L2 distance functions.
  //

  float DistanceL2Int8::compare(const int8_t *a, const int8_t *b, 
                                uint32_t size, float break_distance) const {
    int32_t result = 0;

#ifdef _WINDOWS
#ifdef USE_AVX2

    const int8_t *pEnd32 = a + ((size >> 5) << 5);
    const int8_t *pEnd1 = a + size;

    __m256        diff256 = _mm256_setzero_ps();
    const int8_t *testdist = size / 2 + a;
    float         diff = 0;

    while (a < pEnd32) {
      REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_sqdf_epi8,
             _mm256_add_ps, diff256)

      if (a >= testdist) {
        __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256),
                                    _mm256_extractf128_ps(diff256, 1));
        diff = diff128.m128_f32[0] + diff128.m128_f32[1] + 
            diff128.m128_f32[2] + diff128.m128_f32[3];

        if (diff > break_distance)
          return diff;

        testdist += (pEnd32 - a) / 2;
      }
    }

    while (a < pEnd1) {
      float c1 = ((float) (*a++) - (float) (*b++));
      diff += c1 * c1;

      if (diff > break_distance)
        return diff;
    }
    return diff;

#else
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
    for (_s32 i = 0; i < (_s32) size; i++) {
      result += ((int32_t) ((int16_t) a[i] - (int16_t) b[i])) *
                ((int32_t) ((int16_t) a[i] - (int16_t) b[i]));
    }
    return (float) result;
#endif
#else
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
    for (int32_t i = 0; i < (int32_t) size; i++) {
      result += ((int32_t) ((int16_t) a[i] - (int16_t) b[i])) *
                ((int32_t) ((int16_t) a[i] - (int16_t) b[i]));
    }
    return (float) result;
#endif
  }

  // This function implementation was adapted from:
  // https://github.com/microsoft/SPTAG/blob/main/AnnService/src/Core/Common/DistanceUtils.cpp

  float DistanceL2UInt8::compare(const uint8_t *a, const uint8_t *b,
                                 uint32_t size, float break_distance) const {
#ifdef _WINDOWS
    const uint8_t *end16 = a + ((size >> 4) << 4);
    const uint8_t *end1 = a + size;
 
     __m128 diff128 = _mm_setzero_ps();
    const uint8_t *testdist = size / 2 + a;
     float diff = 0;
 
     while (a < end16) {
       REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epu8,
       _mm_add_ps, diff128)

       if (a >= testdist) {
         diff = diff128.m128_f32[0] + diff128.m128_f32[1] +
                diff128.m128_f32[2] + diff128.m128_f32[3];
         if (diff > break_distance)
           return diff;

         testdist += (end16 - a) / 2;
       }
     }
 
     while (a < end1) {
       float c1 = ((float) (*a++) - (float) (*b++));
         diff += c1 * c1;
 
         if (diff > break_distance)
           return diff;
     }
     return diff;
#else
    uint32_t result = 0;
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
    for (int32_t i = 0; i < (int32_t) size; i++) {
      result += ((int32_t) ((int16_t) a[i] - (int16_t) b[i])) *
                ((int32_t) ((int16_t) a[i] - (int16_t) b[i]));
    }
    return (float) result;
#endif
  }

#ifndef _WINDOWS
  float DistanceL2Float::compare(const float *a, const float *b, uint32_t size,
                                 float break_distance) const {
    a = (const float *) __builtin_assume_aligned(a, 32);
    b = (const float *) __builtin_assume_aligned(b, 32);
#else
  float DistanceL2Float::compare(const float *a, const float *b, uint32_t size,
                                 float break_distance) const {
#endif

    float result = 0;
#ifdef USE_AVX2
    // assume size is divisible by 8
    uint16_t niters = (uint16_t) (size / 8);
    //test for break_distance at the halfway point
    uint16_t testidx = niters / 2;
    __m256   sum = _mm256_setzero_ps();
    for (uint16_t j = 0; j < niters; j++) {
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

      sum = _mm256_fmadd_ps(tmp_vec, tmp_vec, sum);

      if (j >= testidx) {
        // horizontal add sum
        result = _mm256_reduce_add_ps(sum);
        if (result > break_distance)
           break;
        //test for break_distance at the next halfway point
        testidx += (niters - j) / 2;
      }
    }

    //complete remaining dimensions
    for (uint32_t i = niters * 8; i < size; i++) {
      result += (a[i] - b[i]) * (a[i] - b[i]);
    }

#else
#ifndef _WINDOWS
#pragma omp simd reduction(+ : result) aligned(a, b : 32)
#endif
    for (int32_t i = 0; i < (int32_t) size; i++) {
      result += (a[i] - b[i]) * (a[i] - b[i]);
    }
#endif
    return result;
  }

  float SlowDistanceL2Float::compare(const float *a, const float *b,
                                     uint32_t length,
                                     float    break_distance) const {
    float result = 0.0f;
    for (uint32_t i = 0; i < length; i++) {
      result += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return result;
  }

#ifdef _WINDOWS

  // This function implementation was adapted from:
  // https://github.com/microsoft/SPTAG/blob/main/AnnService/src/Core/Common/DistanceUtils.cpp

  float AVXDistanceL2Int8::compare(const int8_t *a, const int8_t *b,
                                   uint32_t length,
                                   float    break_distance) const {
    const int8_t *pEnd16 = a + ((length >> 4) << 4);
    const int8_t *pEnd1 = a + length;

    __m128 diff128 = _mm_setzero_ps();
    const int8_t *testdist = length / 2 + a;
    float   diff = 0;

    while (a < pEnd16) {
      REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epi8, _mm_add_ps,
             diff128)

      if (a >= testdist) {
        diff = diff128.m128_f32[0] + diff128.m128_f32[1] + diff128.m128_f32[2] +
               diff128.m128_f32[3];
        if (diff > break_distance)
           return diff;

        testdist += (pEnd16 - a) / 2;
      }
    }

    while (a < pEnd1) {
      float c1 = ((float) (*a++) - (float) (*b++));
      diff += c1 * c1;

      if (diff > break_distance)
        return diff;
    }
    return diff;
  }

  float AVXDistanceL2Float::compare(const float *a, const float *b,
                                    uint32_t length,
                                    float    break_distance) const {
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
#else
  float AVXDistanceL2Int8::compare(const int8_t *, const int8_t *, uint32_t,
                                   float break_distance) const {
    return 0;
  }
  float AVXDistanceL2Float::compare(const float *, const float *,
                                    uint32_t, float) const {
    return 0;
  }
#endif

  template<typename T>
  float DistanceInnerProduct<T>::inner_product(const T *a, const T *b,
                                               unsigned size) const {
    if (!std::is_floating_point<T>::value) {
      diskann::cerr << "ERROR: Inner Product only defined for float currently."
                    << std::endl;
      throw diskann::ANNException(
          "ERROR: Inner Product only defined for float currently.", -1,
          __FUNCSIG__, __FILE__, __LINE__);
    }

    float result = 0;

#ifdef __GNUC__
#ifdef USE_AVX2
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

  template<typename T>
  float DistanceFastL2<T>::compare(const T *a, const T *b, float norm,
                                   unsigned size, float break_distance) const {
    float result = -2 * DistanceInnerProduct<T>::inner_product(a, b, size);
    result += norm;
    return result;
  }

  template<typename T>
  float DistanceFastL2<T>::norm(const T *a, unsigned size) const {
    if (!std::is_floating_point<T>::value) {
      diskann::cerr << "ERROR: FastL2 only defined for float currently."
                    << std::endl;
      throw diskann::ANNException(
          "ERROR: FastL2 only defined for float currently.", -1, __FUNCSIG__,
          __FILE__, __LINE__);
    }
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

  float AVXDistanceInnerProductFloat::compare(const float *a, const float *b,
                                              uint32_t size,
                                              float    break_distance) const {
    float result = 0.0f;
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
#ifndef _WINDOWS
    float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
#else
    __declspec(align(32)) float unpack[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#endif

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

    return -result;
  }

  // Get the right distance function for the given metric.
  template<>
  diskann::Distance<float> *get_distance_function(diskann::Metric m) {
    if (m == diskann::Metric::L2) {
      if (Avx2SupportedCPU) {
        diskann::cout << "L2: Using AVX2 distance computation DistanceL2Float"
                      << std::endl;
        return new diskann::DistanceL2Float();
      } else if (AvxSupportedCPU) {
        diskann::cout
            << "L2: AVX2 not supported. Using AVX distance computation"
            << std::endl;
        return new diskann::AVXDistanceL2Float();
      } else {
        diskann::cout << "L2: Older CPU. Using slow distance computation"
                      << std::endl;
        return new diskann::SlowDistanceL2Float();
      }
    } else if (m == diskann::Metric::COSINE) {
      diskann::cout << "Cosine: Using either AVX or AVX2 implementation"
                    << std::endl;
      return new diskann::DistanceCosineFloat();
    } else if (m == diskann::Metric::INNER_PRODUCT) {
      diskann::cout << "Inner product: Using AVX2 implementation "
                       "AVXDistanceInnerProductFloat"
                    << std::endl;
      return new diskann::AVXDistanceInnerProductFloat();
    } else if (m == diskann::Metric::FAST_L2) {
      diskann::cout << "Fast_L2: Using AVX2 implementation with norm "
                       "memoization DistanceFastL2<float>"
                    << std::endl;
      return new diskann::DistanceFastL2<float>();
    } else {
      std::stringstream stream;
      stream << "Only L2, cosine, and inner product supported for floating "
                "point vectors as of now."
             << std::endl;
      diskann::cerr << stream.str() << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }

  template<>
  diskann::Distance<int8_t> *get_distance_function(diskann::Metric m) {
    if (m == diskann::Metric::L2) {
      if (Avx2SupportedCPU) {
        diskann::cout << "Using AVX2 distance computation DistanceL2Int8."
                      << std::endl;
        return new diskann::DistanceL2Int8();
      } else if (AvxSupportedCPU) {
        diskann::cout << "AVX2 not supported. Using AVX distance computation"
                      << std::endl;
        return new diskann::AVXDistanceL2Int8();
      } else {
        diskann::cout << "Older CPU. Using slow distance computation "
                         "SlowDistanceL2Int<int8_t>."
                      << std::endl;
        return new diskann::SlowDistanceL2Int<int8_t>();
      }
    } else if (m == diskann::Metric::COSINE) {
      diskann::cout << "Using either AVX or AVX2 for Cosine similarity "
                       "DistanceCosineInt8."
                    << std::endl;
      return new diskann::DistanceCosineInt8();
    } else {
      std::stringstream stream;
      stream << "Only L2 and cosine supported for signed byte vectors."
             << std::endl;
      diskann::cerr << stream.str() << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }

  template<>
  diskann::Distance<uint8_t> *get_distance_function(diskann::Metric m) {
    if (m == diskann::Metric::L2) {
#ifdef _WINDOWS
      diskann::cout
          << "WARNING: AVX/AVX2 distance function not defined for Uint8. Using "
             "slow version. "
             "Contact gopalsr@microsoft.com if you need AVX/AVX2 support."
          << std::endl;
#endif
      return new diskann::DistanceL2UInt8();
    } else if (m == diskann::Metric::COSINE) {
      diskann::cout
          << "AVX/AVX2 distance function not defined for Uint8. Using "
             "slow version SlowDistanceCosineUint8() "
             "Contact gopalsr@microsoft.com if you need AVX/AVX2 support."
          << std::endl;
      return new diskann::SlowDistanceCosineUInt8();
    } else {
      std::stringstream stream;
      stream << "Only L2 and cosine supported for unsigned byte vectors."
             << std::endl;
      diskann::cerr << stream.str() << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
  }

  template DISKANN_DLLEXPORT class DistanceInnerProduct<float>;
  template DISKANN_DLLEXPORT class DistanceInnerProduct<int8_t>;
  template DISKANN_DLLEXPORT class DistanceInnerProduct<uint8_t>;

  template DISKANN_DLLEXPORT class DistanceFastL2<float>;
  template DISKANN_DLLEXPORT class DistanceFastL2<int8_t>;
  template DISKANN_DLLEXPORT class DistanceFastL2<uint8_t>;

}  // namespace diskann
