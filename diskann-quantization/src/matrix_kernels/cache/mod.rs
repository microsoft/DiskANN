/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! L1d / L2 cache size probe used by the matrix-kernel drivers.
//! Success and failure are memoized. Returns `None` when the probe is unavailable
//! or cannot determine both cache sizes.

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

pub(super) fn cache_info() -> Option<CacheInfo> {
    static CACHED: OnceLock<Option<CacheInfo>> = OnceLock::new();
    *CACHED.get_or_init(detect_uncached)
}

fn detect_uncached() -> Option<CacheInfo> {
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

    detected
}
