// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! aarch64 Linux cache probe via sysfs (`/sys/devices/system/cpu/cpu0/cache/`).
//! Returns `None` when the cache sysfs entries are absent, as in some
//! stripped-down containers.

use std::path::Path;

use super::CacheInfo;

const SYSFS_CACHE_DIR: &str = "/sys/devices/system/cpu/cpu0/cache";

pub(super) fn detect() -> Option<CacheInfo> {
    let mut l1d = None;
    let mut l2 = None;

    // Each `index*` subdirectory describes one cache (level + type + size).
    for entry in std::fs::read_dir(SYSFS_CACHE_DIR).ok()?.flatten() {
        let dir = entry.path();

        let Some(level) = read_trim(dir.join("level")).and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        let Some(cache_type) = read_trim(dir.join("type")) else {
            continue;
        };
        let Some(size) = read_trim(dir.join("size")).and_then(|s| parse_size(&s)) else {
            continue;
        };

        match (level, cache_type.as_str()) {
            (1, "Data") if l1d.is_none() => l1d = Some(size),
            // L2 is Unified on real hardware; accept Data defensively.
            (2, "Unified" | "Data") if l2.is_none() => l2 = Some(size),
            _ => {}
        }
    }

    Some(CacheInfo {
        l1d_bytes: l1d?,
        l2_bytes: l2?,
    })
}

fn read_trim(path: impl AsRef<Path>) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
}

fn parse_size(s: &str) -> Option<usize> {
    let s = s.trim();
    let split = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let (num, suffix) = s.split_at(split);
    let n: usize = num.parse().ok()?;
    match suffix.trim() {
        "" => Some(n),
        "K" | "KB" | "KiB" => Some(n * 1024),
        "M" | "MB" | "MiB" => Some(n * 1024 * 1024),
        "G" | "GB" | "GiB" => Some(n * 1024 * 1024 * 1024),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_size_handles_common_formats() {
        assert_eq!(parse_size("32K"), Some(32 * 1024));
        assert_eq!(parse_size("1024K"), Some(1024 * 1024));
        assert_eq!(parse_size("8M"), Some(8 * 1024 * 1024));
        assert_eq!(parse_size("1G"), Some(1024 * 1024 * 1024));
        assert_eq!(parse_size("4096"), Some(4096));
        assert_eq!(parse_size(" 32K\n"), Some(32 * 1024));
        assert_eq!(parse_size("garbage"), None);
        assert_eq!(parse_size(""), None);
    }
}
