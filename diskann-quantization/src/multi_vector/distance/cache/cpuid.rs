// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! x86_64 CPUID cache probe via deterministic cache parameters — `raw-cpuid`'s
//! cache-parameter enumeration (CPUID `0x4` on Intel, `0x8000001D` on AMD).
//! Returns `None` when those parameters are unavailable; we don't fall back to
//! AMD's legacy `0x80000005/06` leaves.

use raw_cpuid::{CacheType, CpuId};

use super::CacheInfo;

pub(super) fn detect() -> Option<CacheInfo> {
    let cpuid = CpuId::new();
    let params = cpuid.get_cache_parameters()?;

    let mut l1d = None;
    let mut l2 = None;

    for cache in params {
        let level = cache.level();
        let ty = cache.cache_type();
        let size = cache_size_bytes(&cache);

        match (level, ty) {
            (1, CacheType::Data) if l1d.is_none() => l1d = Some(size),
            // L2 is usually Unified; accept Data as a defensive fallback.
            (2, CacheType::Unified | CacheType::Data) if l2.is_none() => l2 = Some(size),
            _ => {}
        }

        if l1d.is_some() && l2.is_some() {
            break;
        }
    }

    Some(CacheInfo {
        l1d_bytes: l1d?,
        l2_bytes: l2?,
    })
}

fn cache_size_bytes(cache: &raw_cpuid::CacheParameter) -> usize {
    cache.associativity()
        * cache.physical_line_partitions()
        * cache.coherency_line_size()
        * cache.sets()
}
