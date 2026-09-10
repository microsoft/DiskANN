// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! L1d / L2 cache size probe used by the multi-vector tile planner.
//! Detected once and memoized; returns [`CacheInfo::FALLBACK`] when no
//! per-platform probe applies.

use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
mod cpuid;
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
mod linux;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod macos;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CacheInfo {
    /// L1 **d**ata cache size in bytes. L1i (instruction cache) is not read —
    /// tile budgets only constrain data residency.
    pub l1d_bytes: usize,
    /// L2 cache size in bytes.
    pub l2_bytes: usize,
}

impl CacheInfo {
    /// Used when no per-platform probe applies.
    pub(super) const FALLBACK: Self = Self {
        l1d_bytes: 32 * 1024,
        l2_bytes: 256 * 1024,
    };
}

pub(super) fn cache_info() -> CacheInfo {
    static CACHED: OnceLock<CacheInfo> = OnceLock::new();
    *CACHED.get_or_init(detect_uncached)
}

fn detect_uncached() -> CacheInfo {
    #[cfg(target_arch = "x86_64")]
    let detected = cpuid::detect();

    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    let detected = linux::detect();

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    let detected = macos::detect();

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_os = "linux"),
        all(target_arch = "aarch64", target_os = "macos"),
    )))]
    let detected: Option<CacheInfo> = None;

    detected.unwrap_or(CacheInfo::FALLBACK)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_info_returns_plausible_values() {
        let info = cache_info();

        // Either we detected real values or we fell back. In both cases the
        // values must be within plausible bounds for any CPU we care about:
        // 4 KB to 1 MB for L1d, 64 KB to 128 MB for L2.
        assert!(
            (4 * 1024..=1024 * 1024).contains(&info.l1d_bytes),
            "L1d out of plausible range: {} bytes",
            info.l1d_bytes
        );
        assert!(
            (64 * 1024..=128 * 1024 * 1024).contains(&info.l2_bytes),
            "L2 out of plausible range: {} bytes",
            info.l2_bytes
        );
    }

    #[test]
    fn cache_info_is_memoized() {
        let first = cache_info();
        let second = cache_info();
        assert_eq!(first, second);
    }
}
