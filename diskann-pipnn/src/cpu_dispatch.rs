/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Runtime CPU feature dispatch for PiPNN SIMD kernels.
//!
//! Single source of truth for the SIMD tier the current process should use.
//! The detected tier is cached behind a `OnceLock`, so subsequent calls
//! amount to a relaxed atomic load plus a branch — cheap enough to hoist
//! into any outer loop, but the convention in this crate is to call
//! `tier()` ONCE at the outermost practical level and `match` on the
//! returned enum.

use std::sync::OnceLock;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum SimdTier {
    Avx512,
    Avx2,
    Scalar,
}

static TIER: OnceLock<SimdTier> = OnceLock::new();

#[inline]
pub(crate) fn tier() -> SimdTier {
    *TIER.get_or_init(|| {
        let detected = detect_tier();
        // First-call log lets perf benchmarks confirm which SIMD path actually
        // ran on the test machine. Subsequent calls hit the OnceLock fast path
        // and do not log.
        #[cfg(target_arch = "x86_64")]
        tracing::info!(
            tier = ?detected,
            avx512f = std::is_x86_feature_detected!("avx512f"),
            avx2 = std::is_x86_feature_detected!("avx2"),
            fma = std::is_x86_feature_detected!("fma"),
            f16c = std::is_x86_feature_detected!("f16c"),
            "pipnn cpu_dispatch: SIMD tier selected"
        );
        #[cfg(not(target_arch = "x86_64"))]
        tracing::info!(tier = ?detected, "pipnn cpu_dispatch: SIMD tier selected (non-x86_64)");
        detected
    })
}

fn detect_tier() -> SimdTier {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return SimdTier::Avx512;
        }
        if std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("f16c")
        {
            return SimdTier::Avx2;
        }
    }
    SimdTier::Scalar
}
