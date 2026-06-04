// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! aarch64 macOS sysctl cache probe. Reads `hw.perflevel0.*`
//! and divides the cluster L2 by `cpusperl2` for a per-core budget.

use core::ffi::{CStr, c_void};

use super::CacheInfo;

pub(super) fn detect() -> Option<CacheInfo> {
    // L1d is private per core — no normalization.
    let l1d = sysctl_uint(c"hw.perflevel0.l1dcachesize")?;
    let l2 = perflevel0_l2_per_core()?;

    Some(CacheInfo {
        l1d_bytes: l1d as usize,
        l2_bytes: l2 as usize,
    })
}

/// It reports the full per-cluster L2 via `hw.perflevel0.l2cachesize`;
/// divide by `hw.perflevel0.cpusperl2` for the per-core share.
fn perflevel0_l2_per_core() -> Option<u64> {
    let total = sysctl_uint(c"hw.perflevel0.l2cachesize")?;
    let cpus = sysctl_uint(c"hw.perflevel0.cpusperl2")?;
    (cpus > 0).then(|| total / cpus)
}

/// Reads an integer sysctl reported as either 4- or 8-byte: cache sizes are
/// 64-bit, but topology counts like `cpusperl2` are 32-bit.
fn sysctl_uint(name: &CStr) -> Option<u64> {
    let mut buf = [0u8; 8];
    let mut size = buf.len();
    // SAFETY: `name` is a valid NUL-terminated C string; `buf` / `size` are
    // valid and writable; the new* parameters are null because we are only
    // reading.
    let ret = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            buf.as_mut_ptr() as *mut c_void,
            &mut size,
            core::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 {
        return None;
    }
    match size {
        4 => Some(u32::from_ne_bytes(buf[..4].try_into().ok()?) as u64),
        8 => Some(u64::from_ne_bytes(buf)),
        _ => None,
    }
}
