/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Hardware L1d/L2 cache detection for matrix-kernel budgets.
//!
//! Prefer OS cache enumeration, selecting the smallest capacity at each level to
//! avoid oversized tile budgets across heterogeneous cores.
//! L1d and L2 may come from different core types.
//!
//! Fall back to x86 CPUID, which samples the executing CPU.
//!
//! Each source must provide both levels. If all sources fail, return `None`;
//! the budget layer supplies fallback estimates.
//!
//! Adapted from the cache detection in
//! [gemm-common 0.18.2](https://github.com/sarah-quinones/gemm/blob/a9d8ed36e51ba039ffe8cb2512925202612e19fd/gemm-common/src/cache.rs).

use std::{num::NonZeroUsize, sync::OnceLock};

/// Capacity and sharing information for a hardware cache instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheLevel {
    /// Total capacity in bytes.
    pub bytes: NonZeroUsize,
    /// Number of sharing logical CPUs, if known.
    pub logical_sharers: Option<NonZeroUsize>,
}

/// Detected L1d and L2 caches, selected independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HardwareCache {
    /// Diagnostic source label; its format may change.
    pub source: &'static str,
    /// L1 data cache.
    pub l1d: CacheLevel,
    /// L2 cache.
    pub l2: CacheLevel,
}

impl HardwareCache {
    /// Detect once per process, caching both success and failure.
    ///
    /// Returns `None` if either level is unavailable or when running under Miri.
    pub fn detect() -> Option<Self> {
        static CACHED: OnceLock<Option<HardwareCache>> = OnceLock::new();
        *CACHED.get_or_init(detect_uncached)
    }
}

fn detect_uncached() -> Option<HardwareCache> {
    if cfg!(miri) {
        return None;
    }

    #[cfg(target_os = "linux")]
    if let Some(info) = linux::cache_info() {
        return Some(info);
    }

    #[cfg(target_vendor = "apple")]
    if let Some(info) = apple::cache_info() {
        return Some(info);
    }

    #[cfg(target_os = "windows")]
    if let Some(info) = windows::cache_info() {
        return Some(info);
    }

    #[cfg(any(
        all(target_arch = "x86", not(target_env = "sgx"), target_feature = "sse"),
        all(target_arch = "x86_64", not(target_env = "sgx")),
    ))]
    if let Some(info) = cpuid::cache_info() {
        return Some(info);
    }

    None
}

impl CacheLevel {
    fn new(bytes: usize, logical_sharers: Option<NonZeroUsize>) -> Option<Self> {
        Some(Self {
            bytes: NonZeroUsize::new(bytes)?,
            logical_sharers,
        })
    }
}

// Keep the cache with the smallest capacity. For equal capacities, prefer more sharers.
// Prefer a known sharer count over an unknown count. Keep both fields from the same cache.
fn update_min(current: &mut Option<CacheLevel>, candidate: CacheLevel) {
    if current.is_none_or(|old| {
        candidate.bytes < old.bytes
            || (candidate.bytes == old.bytes && candidate.logical_sharers > old.logical_sharers)
    }) {
        *current = Some(candidate);
    }
}

#[cfg(test)]
mod tests {
    use super::{CacheLevel, update_min};
    use std::num::NonZeroUsize;

    #[test]
    fn selects_smallest_capacity_then_most_sharers() {
        let candidates = [
            CacheLevel::new(64 * 1024, NonZeroUsize::new(8)).unwrap(),
            CacheLevel::new(32 * 1024, NonZeroUsize::new(2)).unwrap(),
            CacheLevel::new(48 * 1024, NonZeroUsize::new(4)).unwrap(),
            CacheLevel::new(32 * 1024, NonZeroUsize::new(4)).unwrap(),
            CacheLevel::new(32 * 1024, None).unwrap(),
        ];
        for reverse in [false, true] {
            let mut selected = None;
            for i in 0..candidates.len() {
                let i = if reverse { candidates.len() - 1 - i } else { i };
                update_min(&mut selected, candidates[i]);
            }
            let selected = selected.unwrap();
            assert_eq!(selected.bytes.get(), 32 * 1024);
            assert_eq!(selected.logical_sharers, NonZeroUsize::new(4));
        }
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use std::{fs, num::NonZeroUsize};

    use super::{CacheLevel, HardwareCache, update_min};

    pub(super) fn cache_info() -> Option<HardwareCache> {
        from_sysfs().or_else(from_lscpu)
    }

    // /sys/devices/system/cpu/cpuX/cache/indexY/
    //     |-- level            "1", "2", ...
    //     |-- type             "Data", "Instruction", or "Unified"
    //     |-- size             e.g. "32K"
    //     `-- shared_cpu_list  e.g. "0-1"
    //
    // Y is an entry index, not a cache level. Shared caches may appear under multiple CPUs.
    fn from_sysfs() -> Option<HardwareCache> {
        let mut caches = [None; 2];

        for cpu_x in fs::read_dir("/sys/devices/system/cpu").ok()? {
            let cpu_x = cpu_x.ok()?.path();
            let Some(cpu_x_name) = cpu_x.file_name().and_then(|f| f.to_str()) else {
                continue;
            };
            if !cpu_x_name.starts_with("cpu") {
                continue;
            }
            let cache = cpu_x.join("cache");
            if !cache.is_dir() {
                continue;
            }
            for index_y in fs::read_dir(cache).ok()? {
                let index_y = index_y.ok()?.path();
                if !index_y.is_dir() {
                    continue;
                }
                let Some(index_y_name) = index_y.file_name().and_then(|f| f.to_str()) else {
                    continue;
                };
                if !index_y_name.starts_with("index") {
                    continue;
                }

                let cache_type = fs::read_to_string(index_y.join("type")).ok()?;
                if !matches!(cache_type.trim(), "Data" | "Unified") {
                    continue;
                }
                let level = fs::read_to_string(index_y.join("level")).ok()?;
                let selected = match level.trim().parse::<usize>() {
                    Ok(1) => &mut caches[0],
                    Ok(2) => &mut caches[1],
                    _ => continue,
                };
                let size = fs::read_to_string(index_y.join("size")).ok()?;
                let Some(bytes) = parse_size(size.trim()) else {
                    continue;
                };
                let logical_sharers = fs::read_to_string(index_y.join("shared_cpu_list"))
                    .ok()
                    .and_then(|contents| shared_cpu_count(contents.trim()))
                    .and_then(NonZeroUsize::new);
                if let Some(cache) = CacheLevel::new(bytes, logical_sharers) {
                    update_min(selected, cache);
                }
            }
        }

        Some(HardwareCache {
            source: "linux-sysfs",
            l1d: caches[0]?,
            l2: caches[1]?,
        })
    }

    fn from_lscpu() -> Option<HardwareCache> {
        let output = std::process::Command::new("lscpu")
            .arg("-B")
            .arg("-C=type,level,one-size")
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let stdout = std::str::from_utf8(&output.stdout).ok()?;
        let mut l1d = None;
        let mut l2 = None;
        for line in stdout.lines().skip(1) {
            let mut fields = line.split_whitespace();
            if !matches!(fields.next(), Some("Data" | "Unified")) {
                continue;
            }
            let cache = match fields.next()? {
                "1" => &mut l1d,
                "2" => &mut l2,
                _ => continue,
            };
            let candidate = CacheLevel::new(fields.next()?.parse().ok()?, None)?;
            update_min(cache, candidate);
        }
        Some(HardwareCache {
            source: "lscpu",
            l1d: l1d?,
            l2: l2?,
        })
    }

    fn parse_size(contents: &str) -> Option<usize> {
        let (contents, multiplier) = if contents.ends_with('G') {
            (contents.trim_end_matches('G'), 1024 * 1024 * 1024)
        } else if contents.ends_with('M') {
            (contents.trim_end_matches('M'), 1024 * 1024)
        } else if contents.ends_with('K') {
            (contents.trim_end_matches('K'), 1024)
        } else {
            (contents, 1)
        };
        contents.parse::<usize>().ok()?.checked_mul(multiplier)
    }

    fn shared_cpu_count(contents: &str) -> Option<usize> {
        let mut shared_count = 0usize;
        for item in contents.split(',') {
            let count = if let Some((start, end)) = item.split_once('-') {
                let start = start.parse::<usize>().ok()?;
                let end = end.parse::<usize>().ok()?;
                end.checked_sub(start)?.checked_add(1)?
            } else {
                item.parse::<usize>().ok()?;
                1
            };
            shared_count = shared_count.checked_add(count)?;
        }
        Some(shared_count)
    }

    #[cfg(test)]
    mod tests {
        use super::{cache_info, from_lscpu, parse_size, shared_cpu_count};

        #[test]
        fn detects_cache_from_linux() {
            let info = cache_info().expect("Linux should report both L1 data and L2 caches");
            println!("{info:#?}");
        }

        #[test]
        fn detects_cache_from_lscpu() {
            let info = from_lscpu().expect("lscpu should report both L1 data and L2 caches");
            println!("{info:#?}");
        }

        #[test]
        fn parse_size_preserves_sysfs_syntax() {
            for (input, expected) in [
                ("0", 0),
                ("4096", 4096),
                ("32K", 32 * 1024),
                ("8M", 8 * 1024 * 1024),
                ("1G", 1024 * 1024 * 1024),
                ("32KK", 32 * 1024),
            ] {
                assert_eq!(parse_size(input), Some(expected), "{input}");
            }
            for input in ["", "K", "bad", "32KB", "32KiB", "32 K", " 32K", "32K\n"] {
                assert_eq!(parse_size(input), None, "{input}");
            }
            assert_eq!(parse_size(&usize::MAX.to_string()), Some(usize::MAX));
            for suffix in ['K', 'M', 'G'] {
                assert_eq!(parse_size(&format!("{}{suffix}", usize::MAX)), None);
            }
        }

        #[test]
        fn shared_cpu_count_preserves_range_validation() {
            for (input, expected) in [("0", 1), ("0-0", 1), ("0-3,8,10-11", 7), ("0-1,1", 3)] {
                assert_eq!(shared_cpu_count(input), Some(expected), "{input}");
            }
            for input in ["", "x", "4-1", "0-", "-1", "0-1-2", "0,,1", "0, 1"] {
                assert_eq!(shared_cpu_count(input), None, "{input}");
            }
            assert_eq!(shared_cpu_count(&format!("0-{}", usize::MAX)), None);
            assert_eq!(
                shared_cpu_count(&format!("0-{}", usize::MAX - 1)),
                Some(usize::MAX),
            );
            assert_eq!(shared_cpu_count(&format!("0-{},0", usize::MAX - 1)), None);
        }
    }
}

#[cfg(target_vendor = "apple")]
mod apple {
    //! Query `hw.perflevelN.*` for each core type, following Apple's
    //! [Determining system capabilities](https://developer.apple.com/documentation/kernel/1387446-sysctlbyname/determining_system_capabilities).

    use core::ffi::{CStr, c_void};
    use std::{ffi::CString, num::NonZeroUsize};

    use super::{CacheLevel, HardwareCache, update_min};

    pub(super) fn cache_info() -> Option<HardwareCache> {
        let mut l1d = None;
        let mut l2 = None;
        for level in 0..sysctl_uint(c"hw.nperflevels")? {
            let read = |field| {
                let name = CString::new(format!("hw.perflevel{level}.{field}")).ok()?;
                usize::try_from(sysctl_uint(&name)?).ok()
            };

            let candidate = CacheLevel::new(read("l1dcachesize")?, None)?;
            update_min(&mut l1d, candidate);
            let l2_sharers = read("cpusperl2").and_then(NonZeroUsize::new);
            let candidate = CacheLevel::new(read("l2cachesize")?, l2_sharers)?;
            update_min(&mut l2, candidate);
        }
        Some(HardwareCache {
            source: "apple-sysctl",
            l1d: l1d?,
            l2: l2?,
        })
    }

    // Cache sizes are 64-bit; topology counts such as `cpusperl2` are 32-bit.
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

    #[cfg(test)]
    mod tests {
        use super::cache_info;

        #[test]
        fn detects_cache_from_apple() {
            let info = cache_info().expect("Apple sysctl should report both L1 data and L2 caches");
            println!("{info:#?}");
        }
    }
}

#[cfg(target_os = "windows")]
mod windows {
    //! Read cache capacities and sharing masks with
    //! [`GetLogicalProcessorInformationEx`](https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getlogicalprocessorinformationex).

    use std::{num::NonZeroUsize, ptr, slice};

    use windows_sys::Win32::{
        Foundation::{ERROR_INSUFFICIENT_BUFFER, GetLastError, WIN32_ERROR},
        System::SystemInformation::{
            CacheData, CacheUnified, GetLogicalProcessorInformationEx, RelationCache,
            SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX as ProcessorInfo,
        },
    };

    use super::{CacheLevel, HardwareCache, update_min};
    use crate::alloc::{AlignedAllocator, Poly};

    pub(super) fn cache_info() -> Option<HardwareCache> {
        let mut length = 0;
        if read_cache_records(ptr::null_mut(), &mut length) != Err(ERROR_INSUFFICIENT_BUFFER) {
            return None;
        }

        loop {
            let allocated = length;
            if allocated == 0 {
                return None;
            }
            let mut buffer = Poly::broadcast(0u8, allocated as usize, AlignedAllocator::A8).ok()?;
            match read_cache_records(buffer.as_mut_ptr().cast(), &mut length) {
                Ok(()) => {
                    // SAFETY: Windows wrote complete records into the aligned buffer.
                    return unsafe { from_records(buffer.as_ptr().cast(), length as usize) };
                }
                // Retry if topology changes require a larger buffer.
                Err(ERROR_INSUFFICIENT_BUFFER) if length > allocated => {}
                Err(_) => return None,
            }
        }
    }

    fn read_cache_records(buffer: *mut ProcessorInfo, length: &mut u32) -> Result<(), WIN32_ERROR> {
        // SAFETY: The caller guarantees buffer validity; `length` is a valid reference.
        // Read the error immediately on failure.
        unsafe {
            if GetLogicalProcessorInformationEx(RelationCache, buffer, length) != 0 {
                Ok(())
            } else {
                Err(GetLastError())
            }
        }
    }

    /// # Safety
    /// The first `length` bytes must form complete, aligned Windows processor records.
    unsafe fn from_records(raw_buf: *const ProcessorInfo, length: usize) -> Option<HardwareCache> {
        let mut caches = [None; 2];
        let mut i = 0;
        // SAFETY: The caller guarantees valid records, including their sizes and group masks.
        unsafe {
            while i < length {
                let p = raw_buf.byte_add(i);
                i += (*p).Size as usize;
                if (*p).Relationship != RelationCache {
                    continue;
                }
                let cache = &(*p).Anonymous.Cache;
                if (cache.Type != CacheData && cache.Type != CacheUnified)
                    || !matches!(cache.Level, 1 | 2)
                {
                    continue;
                }
                // Older Windows uses a single GroupMask and leaves GroupCount reserved/zero.
                let count = usize::from(cache.GroupCount).max(1);
                let groups = slice::from_raw_parts(
                    ptr::addr_of!((*p).Anonymous.Cache.Anonymous.GroupMask),
                    count,
                );
                // Unhandled corner case: 32-bit processes under WOW64 on systems with more than
                // 64 logical processors may receive affinity masks folded from 64 to 32 bits.
                // We count the returned masks as-is, so sharing counts may be inaccurate.
                let logical_sharers = NonZeroUsize::new(
                    groups
                        .iter()
                        .map(|group| group.Mask.count_ones() as usize)
                        .sum(),
                );
                if let Some(level) = CacheLevel::new(cache.CacheSize as usize, logical_sharers) {
                    update_min(&mut caches[usize::from(cache.Level) - 1], level);
                }
            }
        }
        Some(HardwareCache {
            source: "windows",
            l1d: caches[0]?,
            l2: caches[1]?,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::cache_info;

        #[test]
        fn detects_cache_from_windows() {
            let info = cache_info().expect("Windows should report both L1 data and L2 caches");
            println!("{info:#?}");
        }
    }
}

#[cfg(any(
    all(target_arch = "x86", not(target_env = "sgx"), target_feature = "sse"),
    all(target_arch = "x86_64", not(target_env = "sgx"))
))]
mod cpuid {
    use super::{CacheLevel, HardwareCache};
    use raw_cpuid::{Associativity, CacheType, CpuId};

    pub(super) fn cache_info() -> Option<HardwareCache> {
        let cpuid = CpuId::new();
        match cpuid.get_vendor_info()?.as_str() {
            "GenuineIntel" => {
                let mut l1d = None;
                let mut l2 = None;
                for cache in cpuid.get_cache_parameters()? {
                    if !matches!(cache.cache_type(), CacheType::Data | CacheType::Unified) {
                        continue;
                    }
                    let selected = match cache.level() {
                        1 => &mut l1d,
                        2 => &mut l2,
                        _ => continue,
                    };
                    let bytes = cache.associativity()
                        * cache.physical_line_partitions()
                        * cache.coherency_line_size()
                        * cache.sets();
                    *selected = CacheLevel::new(bytes, None);
                }
                Some(HardwareCache {
                    source: "cpuid",
                    l1d: l1d?,
                    l2: l2?,
                })
            }
            "AuthenticAMD" => {
                let l1 = cpuid.get_l1_cache_and_tlb_info()?;
                let l2 = cpuid.get_l2_l3_cache_and_tlb_info()?;
                if matches!(
                    l1.dcache_associativity(),
                    Associativity::Unknown | Associativity::Disabled
                ) || matches!(
                    l2.l2cache_associativity(),
                    Associativity::Unknown | Associativity::Disabled
                ) {
                    return None;
                }
                Some(HardwareCache {
                    source: "cpuid",
                    l1d: CacheLevel::new(usize::from(l1.dcache_size()) * 1024, None)?,
                    l2: CacheLevel::new(usize::from(l2.l2cache_size()) * 1024, None)?,
                })
            }
            _ => None,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::cache_info;

        #[test]
        fn detects_cache_from_cpuid() {
            // CPUID may not report cache sizes, for example when running under SDE.
            let info = cache_info();
            println!("{info:#?}");
            if let Some(info) = info {
                assert_eq!(info.l1d.logical_sharers, None);
                assert_eq!(info.l2.logical_sharers, None);
            }
        }
    }
}
