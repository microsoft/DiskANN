/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Cached CPU capabilities used by PiPNN's private SIMD kernels.
//!
//! Callers ask for the vector width required by a kernel family instead of
//! matching ISA names themselves. This keeps feature detection in one place
//! while allowing each family to request only the features it actually uses.

use std::sync::OnceLock;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum VectorWidth {
    Wide,
    Narrow,
    Scalar,
}

#[derive(Copy, Clone, Debug)]
struct CpuFeatures {
    avx512f: bool,
    avx512bw: bool,
    avx2: bool,
    fma: bool,
    f16c: bool,
}

static FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

fn features() -> CpuFeatures {
    *FEATURES.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        let detected = CpuFeatures {
            avx512f: std::is_x86_feature_detected!("avx512f"),
            avx512bw: std::is_x86_feature_detected!("avx512bw"),
            avx2: std::is_x86_feature_detected!("avx2"),
            fma: std::is_x86_feature_detected!("fma"),
            f16c: std::is_x86_feature_detected!("f16c"),
        };

        #[cfg(not(target_arch = "x86_64"))]
        let detected = CpuFeatures {
            avx512f: false,
            avx512bw: false,
            avx2: false,
            fma: false,
            f16c: false,
        };

        tracing::info!(?detected, "PiPNN CPU features detected");
        detected
    })
}

/// Width for kernels that use only packed `f32` arithmetic.
pub(crate) fn f32_width() -> VectorWidth {
    let features = features();
    if features.avx512f {
        VectorWidth::Wide
    } else if features.avx2 {
        VectorWidth::Narrow
    } else {
        VectorWidth::Scalar
    }
}

/// Width for packed `f32` kernels that require fused multiply-add.
pub(crate) fn fma_width() -> VectorWidth {
    let features = features();
    if features.avx512f {
        VectorWidth::Wide
    } else if features.avx2 && features.fma {
        VectorWidth::Narrow
    } else {
        VectorWidth::Scalar
    }
}

/// Width for half-to-f32 conversion kernels.
pub(crate) fn half_width() -> VectorWidth {
    let features = features();
    if features.avx512f {
        VectorWidth::Wide
    } else if features.avx2 && features.f16c {
        VectorWidth::Narrow
    } else {
        VectorWidth::Scalar
    }
}

/// Width for packed 16-bit integer kernels.
pub(crate) fn u16_width() -> VectorWidth {
    let features = features();
    if features.avx512f && features.avx512bw {
        VectorWidth::Wide
    } else if features.avx2 {
        VectorWidth::Narrow
    } else {
        VectorWidth::Scalar
    }
}
