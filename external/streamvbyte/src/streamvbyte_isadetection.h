/* From
https://github.com/endorno/pytorch/blob/master/torch/lib/TH/generic/simd/simd.h
Highly modified.
Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou,
Iain Melvin, Jason Weston) Copyright (c) 2006      Idiap Research Institute
(Samy Bengio) Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert,
Samy Bengio, Johnny Mariethoz)
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories
America and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef STREAMVBYTE_ISADETECTION_H
#define STREAMVBYTE_ISADETECTION_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
#define STREAMVBYTE_IS_ARM64 1
#endif // defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
#if defined(_MSC_VER)
/* Microsoft C/C++-compatible compiler */
#include <intrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
/* GCC-compatible compiler, targeting x86/x86-64 */
#include <x86intrin.h>
#elif defined(__GNUC__) && defined(STREAMVBYTE_IS_ARM64)
/* GCC-compatible compiler, targeting ARM with NEON */
#include <arm_neon.h>
#elif defined(__GNUC__) && defined(__IWMMXT__)
/* GCC-compatible compiler, targeting ARM with WMMX */
#include <mmintrin.h>
#elif (defined(__GNUC__) || defined(__xlC__)) &&                               \
    (defined(__VEC__) || defined(__ALTIVEC__))
/* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
#include <altivec.h>
#elif defined(__GNUC__) && defined(__SPE__)
/* GCC-compatible compiler, targeting PowerPC with SPE */
#include <spe.h>
#endif

#if defined(HAVE_GCC_GET_CPUID) && defined(USE_GCC_GET_CPUID)
#include <cpuid.h>
#endif // defined(_MSC_VER)


enum streamvbyte_instruction_set {
  streamvbyte_DEFAULT = 0x0,
  streamvbyte_NEON = 0x1,
  streamvbyte_SSSE3 = 0x2,
  streamvbyte_AVX2 = 0x4,
  streamvbyte_SSE42 = 0x8,
  streamvbyte_PCLMULQDQ = 0x10,
  streamvbyte_BMI1 = 0x20,
  streamvbyte_BMI2 = 0x40,
  streamvbyte_ALTIVEC = 0x80,
  streamvbyte_SSE41 = 0x100,
  streamvbyte_UNINITIALIZED = 0x8000
};

#if defined(__PPC64__)

static inline uint32_t dynamic_streamvbyte_detect_supported_architectures(void) {
  return streamvbyte_ALTIVEC;
}

#elif defined(__arm__) || defined(__aarch64__) // incl. armel, armhf, arm64

#if defined(__ARM_NEON)

static inline uint32_t dynamic_streamvbyte_detect_supported_architectures(void) {
  return streamvbyte_NEON;
}

#else // ARM without NEON

static inline uint32_t dynamic_streamvbyte_detect_supported_architectures(void) {
  return streamvbyte_DEFAULT;
}

#endif

#elif defined(__x86_64__) || defined(_M_AMD64) // x64




static inline void cpuid(uint32_t *eax, uint32_t *ebx, uint32_t *ecx,
                         uint32_t *edx) {

#if defined(_MSC_VER)
  int cpu_info[4];
  __cpuid(cpu_info, (int)*eax);
  *eax = (uint32_t)cpu_info[0];
  *ebx = (uint32_t)cpu_info[1];
  *ecx = (uint32_t)cpu_info[2];
  *edx = (uint32_t)cpu_info[3];
#elif defined(HAVE_GCC_GET_CPUID) && defined(USE_GCC_GET_CPUID)
  uint32_t level = *eax;
  __get_cpuid(level, eax, ebx, ecx, edx);
#else
  uint32_t a = *eax, b, c = *ecx, d;
  __asm__("cpuid\n\t" : "+a"(a), "=b"(b), "+c"(c), "=d"(d));
  *eax = a;
  *ebx = b;
  *ecx = c;
  *edx = d;
#endif
}

static inline uint32_t dynamic_streamvbyte_detect_supported_architectures(void) {
  uint32_t eax, ebx, ecx, edx;
  uint32_t host_isa = 0x0;
  // Can be found on Intel ISA Reference for CPUID
  static uint32_t cpuid_ssse3_bit = 1 << 1;      ///< @private Bit 1 of EBX for EAX=0x7
  static uint32_t cpuid_avx2_bit = 1 << 5;      ///< @private Bit 5 of EBX for EAX=0x7
  static uint32_t cpuid_bmi1_bit = 1 << 3;      ///< @private bit 3 of EBX for EAX=0x7
  static uint32_t cpuid_bmi2_bit = 1 << 8;      ///< @private bit 8 of EBX for EAX=0x7
  static uint32_t cpuid_sse41_bit = 1 << 19;    ///< @private bit 20 of ECX for EAX=0x1
  static uint32_t cpuid_sse42_bit = 1 << 20;    ///< @private bit 20 of ECX for EAX=0x1
  static uint32_t cpuid_pclmulqdq_bit = 1 << 1; ///< @private bit  1 of ECX for EAX=0x1
  // ECX for EAX=0x7
  eax = 0x7;
  ecx = 0x0;
  cpuid(&eax, &ebx, &ecx, &edx);
  if (ebx & cpuid_avx2_bit) {
    host_isa |= streamvbyte_AVX2;
  }
  if (ebx & cpuid_bmi1_bit) {
    host_isa |= streamvbyte_BMI1;
  }

  if (ebx & cpuid_bmi2_bit) {
    host_isa |= streamvbyte_BMI2;
  }

  // EBX for EAX=0x1
  eax = 0x1;
  cpuid(&eax, &ebx, &ecx, &edx);
  if (ecx & cpuid_ssse3_bit) {
    host_isa |= streamvbyte_SSSE3;
  }
  if (ecx & cpuid_sse42_bit) {
    host_isa |= streamvbyte_SSE42;
  }
  if (ecx & cpuid_sse41_bit) {
    host_isa |= streamvbyte_SSE41;
  }
  if (ecx & cpuid_pclmulqdq_bit) {
    host_isa |= streamvbyte_PCLMULQDQ;
  }

  return host_isa;
}
#else // fallback


static inline uint32_t dynamic_streamvbyte_detect_supported_architectures(void) {
  return streamvbyte_DEFAULT;
}


#endif // end SIMD extension detection code


#if defined(__x86_64__) || defined(_M_AMD64) // x64
#define STREAMVBYTE_X64
#if defined(__cplusplus)
#include <atomic>
static inline uint32_t streamvbyte_detect_supported_architectures(void) {
    static std::atomic<int> buffer{streamvbyte_UNINITIALIZED};
    if(buffer == streamvbyte_UNINITIALIZED) {
      buffer = dynamic_streamvbyte_detect_supported_architectures();
    }
    return buffer;
}
#elif defined(_MSC_VER) && !defined(__clang__)
// Visual Studio does not support C11 atomics.
static inline uint32_t streamvbyte_detect_supported_architectures(void) {
    static int buffer = streamvbyte_UNINITIALIZED;
    if(buffer == streamvbyte_UNINITIALIZED) {
      buffer = dynamic_streamvbyte_detect_supported_architectures();
    }
    return buffer;
}
#else // defined(__cplusplus) and defined(_MSC_VER) && !defined(__clang__)
#if __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
#include <stdatomic.h>
#endif

static inline uint32_t streamvbyte_detect_supported_architectures(void) {
#if __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
    static _Atomic uint32_t buffer = streamvbyte_UNINITIALIZED;
#else
    static int buffer = streamvbyte_UNINITIALIZED;
#endif

#if __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
    uint32_t result = atomic_load_explicit(&buffer, memory_order_acquire);
    if(result == streamvbyte_UNINITIALIZED) {
      result = dynamic_streamvbyte_detect_supported_architectures();
      atomic_store_explicit(&buffer, result, memory_order_release);
    }
    return result;
#else
    if (buffer == streamvbyte_UNINITIALIZED) {
      buffer = dynamic_streamvbyte_detect_supported_architectures();
    }
    return buffer;
#endif
}
#endif // defined(_MSC_VER) && !defined(__clang__)


#if defined(__SSE4_1__)
static inline bool streamvbyte_sse41(void) {
  return true;
}
#else
static inline bool streamvbyte_sse41(void) {
  return  (streamvbyte_detect_supported_architectures() & streamvbyte_SSE41) == streamvbyte_SSE41;
}
#endif


#else // defined(__x86_64__) || defined(_M_AMD64) // x64

static inline bool streamvbyte_sse41(void) {
  return false;
}

static inline uint32_t streamvbyte_detect_supported_architectures(void) {
    // no runtime dispatch
    return dynamic_streamvbyte_detect_supported_architectures();
}
#endif

#ifdef STREAMVBYTE_IS_ARM64
#define STREAMVBYTE_ARM
#endif // STREAMVBYTE_IS_ARM64

#ifdef STREAMVBYTE_X64
// this is almost standard?
#undef STRINGIFY_IMPLEMENTATION_
#undef STRINGIFY
#define STRINGIFY_IMPLEMENTATION_(a) #a
#define STRINGIFY(a) STRINGIFY_IMPLEMENTATION_(a)

#ifdef __clang__
// clang does not have GCC push pop
// warning: clang attribute push can't be used within a namespace in clang up
// til 8.0 so STREAMVBYTE_TARGET_REGION and STREAMVBYTE_UNTARGET_REGION must be *outside* of a
// namespace.
#define STREAMVBYTE_TARGET_REGION(T)                                                       \
  _Pragma(STRINGIFY(                                                           \
      clang attribute push(__attribute__((target(T))), apply_to = function)))
#define STREAMVBYTE_UNTARGET_REGION _Pragma("clang attribute pop")
#elif defined(__GNUC__)
// GCC is easier
#define STREAMVBYTE_TARGET_REGION(T)                                                       \
  _Pragma("GCC push_options") _Pragma(STRINGIFY(GCC target(T)))
#define STREAMVBYTE_UNTARGET_REGION _Pragma("GCC pop_options")
#endif // clang then gcc


// Default target region macros don't do anything.
#ifndef STREAMVBYTE_TARGET_REGION
#define STREAMVBYTE_TARGET_REGION(T)
#define STREAMVBYTE_UNTARGET_REGION
#endif

#define STREAMVBYTE_TARGET_SSE41 STREAMVBYTE_TARGET_REGION("sse4.1")

#ifdef __sse41___
#undef STREAMVBYTE_TARGET_SSE41
#define STREAMVBYTE_TARGET_SSE41
#endif

#if defined(__clang__) || defined(__GNUC__)
#define STREAMVBYTE_ASSUME_ALIGNED(P, A) __builtin_assume_aligned((P), (A))
#else
#define STREAMVBYTE_ASSUME_ALIGNED(P, A)
#endif

#endif // STREAMVBYTE_IS_X64

#endif // STREAMVBYTE_ISADETECTION_H
