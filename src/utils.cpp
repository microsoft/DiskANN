// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"

#include <stdio.h>

#ifdef _WINDOWS
#include <intrin.h>

// Taken from:
// https://insufficientlycomplicated.wordpress.com/2011/11/07/detecting-intel-advanced-vector-extensions-avx-in-visual-studio/
bool cpuHasAvxSupport() {
  bool avxSupported = false;

  // Checking for AVX requires 3 things:
  // 1) CPUID indicates that the OS uses XSAVE and XRSTORE
  //     instructions (allowing saving YMM registers on context
  //     switch)
  // 2) CPUID indicates support for AVX
  // 3) XGETBV indicates the AVX registers will be saved and
  //     restored on context switch
  //
  // Note that XGETBV is only available on 686 or later CPUs, so
  // the instruction needs to be conditionally run.
  int cpuInfo[4];
  __cpuid(cpuInfo, 1);

  bool osUsesXSAVE_XRSTORE = cpuInfo[2] & (1 << 27) || false;
  bool cpuAVXSuport = cpuInfo[2] & (1 << 28) || false;

  if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
    // Check if the OS will save the YMM registers
    unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    avxSupported = (xcrFeatureMask & 0x6) || false;
  }

  return avxSupported;
}

bool cpuHasAvx2Support() {
  int cpuInfo[4];
  __cpuid(cpuInfo, 0);
  int n = cpuInfo[0];
  if (n >= 7) {
    __cpuidex(cpuInfo, 7, 0);
    static int avx2Mask = 0x20;
    return (cpuInfo[1] & avx2Mask) > 0;
  }
  return false;
}
#endif

#ifndef _WINDOWS
bool AvxSupportedCPU = false;
bool Avx2SupportedCPU = true;
#else
bool AvxSupportedCPU = cpuHasAvxSupport();
bool Avx2SupportedCPU = cpuHasAvx2Support();
#endif
