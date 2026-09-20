/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! L1d / L2 cache size probe used by the matrix-kernel drivers.
//! Success and failure are memoized. Returns `None` when the probe is unavailable
//! or cannot determine both cache sizes.
//!
//! Adapted from the cache detection in
//! [gemm-common 0.18.2](https://github.com/sarah-quinones/gemm/blob/a9d8ed36e51ba039ffe8cb2512925202612e19fd/gemm-common/src/cache.rs).
//!
//! Follows the upstream order: Linux sysfs/lscpu, Apple sysctl, then x86 CPUID.
//! Only L1d/L2 capacities are retained, including upstream's per-sharing-CPU
//! normalization. Incomplete probes continue to the next source instead of
//! filling missing levels with defaults. Budget conversion and fallback
//! estimates stay outside the probe.

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CacheInfo {
    /// L1 **d**ata cache size in bytes.
    pub l1d_bytes: usize,
    /// L2 capacity in bytes.
    pub l2_bytes: usize,
}

impl CacheInfo {
    fn new(l1d_bytes: usize, l2_bytes: usize) -> Option<Self> {
        (l1d_bytes > 0 && l2_bytes > 0).then_some(Self {
            l1d_bytes,
            l2_bytes,
        })
    }
}

pub(super) fn cache_info() -> Option<CacheInfo> {
    static CACHED: OnceLock<Option<CacheInfo>> = OnceLock::new();
    *CACHED.get_or_init(detect_uncached)
}

fn detect_uncached() -> Option<CacheInfo> {
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

    #[cfg(any(
        all(target_arch = "x86", not(target_env = "sgx"), target_feature = "sse"),
        all(target_arch = "x86_64", not(target_env = "sgx")),
    ))]
    if let Some(info) = cpuid::cache_info() {
        return Some(info);
    }

    None
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(not(miri))]
    fn detects_current_cache() {
        let info = super::cache_info()
            .expect("expected L1d and L2 cache detection to succeed on this host");
        println!(
            "Detected cache sizes (bytes): L1d={}, L2={}",
            info.l1d_bytes, info.l2_bytes,
        );
        assert!(info.l1d_bytes > 0, "{info:?}");
        assert!(info.l2_bytes > 0, "{info:?}");
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use std::fs;

    use super::CacheInfo;

    pub(super) fn cache_info() -> Option<CacheInfo> {
        from_sysfs().or_else(from_lscpu)
    }

    // /sys/devices/system/cpu/
    // `-- cpuX/                       X is a logical CPU ID.
    //     `-- cache/
    //         `-- indexY/             Y is an entry index, not the cache level.
    //             |-- level          e.g. "1" or "2"
    //             |-- type           "Data", "Instruction", or "Unified"
    //             |-- size           e.g. "32K" (total size of this cache)
    //             |-- shared_cpu_list  e.g. "0-1" (logical CPUs sharing it)
    //             `-- coherency_line_size  e.g. "64" (bytes)
    //
    // Each cpuX can expose several indexY entries; shared caches can appear
    // under multiple CPUs. Read level/type rather than inferring them from Y.
    fn from_sysfs() -> Option<CacheInfo> {
        let mut sizes = [0; 2];
        let mut line_sizes = [64; 2];

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
            'index: for index_y in fs::read_dir(cache).ok()? {
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

                let mut cache_bytes = 0usize;
                let mut cache_line_bytes = 64;
                let mut level: usize = 0;
                let mut shared_count: usize = 0;

                for entry in fs::read_dir(index_y).ok()? {
                    let entry = entry.ok()?.path();
                    let Some(name) = entry.file_name() else {
                        continue;
                    };
                    let contents = fs::read_to_string(&entry).ok()?;
                    let contents = contents.trim();
                    if name == "type" && !matches!(contents, "Data" | "Unified") {
                        continue 'index;
                    }
                    if name == "shared_cpu_list" {
                        let Some(contents) = shared_cpu_count(contents) else {
                            continue 'index;
                        };
                        shared_count = contents;
                    }
                    if name == "level" {
                        let Ok(contents) = contents.parse::<usize>() else {
                            continue 'index;
                        };
                        level = contents;
                    }
                    if name == "coherency_line_size" {
                        let Ok(contents) = contents.parse::<usize>() else {
                            continue 'index;
                        };
                        cache_line_bytes = contents;
                    }
                    if name == "size" {
                        let Some(contents) = parse_size(contents) else {
                            continue 'index;
                        };
                        cache_bytes = contents;
                    }
                }

                // Preserve upstream's preference for entries with larger cache lines.
                if level > 0 && level <= 2 && cache_line_bytes >= line_sizes[level - 1] {
                    let Some(size) = cache_bytes.checked_div(shared_count) else {
                        continue;
                    };
                    if size == 0 {
                        continue;
                    }
                    sizes[level - 1] = size;
                    line_sizes[level - 1] = cache_line_bytes;
                }
            }
        }

        CacheInfo::new(sizes[0], sizes[1])
    }

    fn from_lscpu() -> Option<CacheInfo> {
        let output = std::process::Command::new("lscpu")
            .arg("-B")
            .arg("-C=type,level,one-size")
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let stdout = std::str::from_utf8(&output.stdout).ok()?;
        let mut l1d = 0;
        let mut l2 = 0;
        for line in stdout.lines().skip(1) {
            let mut fields = line.split_whitespace();
            if !matches!(fields.next(), Some("Data" | "Unified")) {
                continue;
            }
            let size = match fields.next()? {
                "1" => &mut l1d,
                "2" => &mut l2,
                _ => continue,
            };
            *size = fields.next()?.parse().ok()?;
        }
        CacheInfo::new(l1d, l2)
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
        use super::{parse_size, shared_cpu_count};

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
    use core::ffi::{CStr, c_void};

    use super::CacheInfo;

    pub(super) fn cache_info() -> Option<CacheInfo> {
        let l1d = sysctl_uint(c"hw.perflevel0.l1dcachesize")?;
        let total = sysctl_uint(c"hw.perflevel0.l2cachesize")?;
        let cpus = sysctl_uint(c"hw.perflevel0.cpusperl2")?;
        CacheInfo::new(
            usize::try_from(l1d).ok()?,
            usize::try_from(total.checked_div(cpus)?).ok()?,
        )
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
}

#[cfg(any(
    all(target_arch = "x86", not(target_env = "sgx"), target_feature = "sse"),
    all(target_arch = "x86_64", not(target_env = "sgx"))
))]
mod cpuid {
    use raw_cpuid::{Associativity, CacheType, CpuId};

    use super::CacheInfo;

    pub(super) fn cache_info() -> Option<CacheInfo> {
        let cpuid = CpuId::new();
        match cpuid.get_vendor_info()?.as_str() {
            "GenuineIntel" => {
                let params = cpuid.get_cache_parameters()?;
                let mut l1d = 0;
                let mut l2 = 0;

                for cache in params {
                    if !matches!(cache.cache_type(), CacheType::Data | CacheType::Unified) {
                        continue;
                    }
                    let size = match cache.level() {
                        1 => &mut l1d,
                        2 => &mut l2,
                        _ => continue,
                    };
                    *size = cache.associativity()
                        * cache.physical_line_partitions()
                        * cache.coherency_line_size()
                        * cache.sets();
                }

                CacheInfo::new(l1d, l2)
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
                CacheInfo::new(
                    usize::from(l1.dcache_size()) * 1024,
                    usize::from(l2.l2cache_size()) * 1024,
                )
            }
            _ => None,
        }
    }
}
