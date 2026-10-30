/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # GEMM-lite Matrix Kernels
//!
//! This module is a work in progress. There are many pieces that are still needed:
//!
//! * GEMM-like kernels beyond "maxsim".
//! * Quantization support.
//! * Blocking along the contraction dimension "k".
//! * Comprehensive performance tuning.
//!
//! For consistency with GEMM terminology, the following conventions are used.
//!
//! A matrix-kernel has the general form
//!
//! ```text
//! C[M x N] = A[M x K] . B[K x N]
//! ```
//! with internal dimensions as follows:
//!
//! * `M`: The number of rows of `A` and `C`.
//! * `N`: The number of columns of `B` and `C`.
//! * `K`: Contraction dimension. Columns of `A` and rows of `B`.
//! * `MR`: Packing parameter for `A`. This is the number of rows processed in a micro-kernel.
//! * `NR`: Packing parameter for `B`. This is the number of columns processed in a micro-kernel.

use std::num::NonZeroUsize;

// Kernels
pub(crate) mod maxsim;

// Arguments
pub(crate) mod blocks;

// private
mod bounds;
mod cache;
mod driver;
mod num;
mod ptr;
mod util;

// re-export
pub(crate) use driver::Drive;
pub(crate) use num::DimK;

#[cfg(feature = "cache-diagnostics")]
pub use cache::{CacheLevel, HardwareCache};

/// Cache budgets for matrix-kernel blocking.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Cache {
    l1: NonZeroUsize,
    l2: NonZeroUsize,
}

impl Cache {
    const FALLBACK: Self = Self::from_cache_sizes(48_000, 1_250_000);

    const fn new(l1: NonZeroUsize, l2: NonZeroUsize) -> Self {
        Self { l1, l2 }
    }

    /// Return the L1 data-cache budget in bytes.
    fn l1(&self) -> NonZeroUsize {
        self.l1
    }

    /// Return the L2 cache budget in bytes.
    fn l2(&self) -> NonZeroUsize {
        self.l2
    }

    /// Detect cache budgets, using [`Self::FALLBACK`] if probing fails.
    ///
    /// Use 75% of L1d and 50% of L2 for the kernel working set.
    pub(crate) fn detect() -> Self {
        cache::HardwareCache::detect().map_or(Self::FALLBACK, |info| {
            Self::from_cache_sizes(info.l1d.bytes.get(), info.l2.bytes.get())
        })
    }

    /// Derive budgets from hardware cache capacities.
    const fn from_cache_sizes(l1d_bytes: usize, l2_bytes: usize) -> Self {
        let l1 = num::value_or_one(l1d_bytes * 3 / 4);
        let l2 = num::value_or_one(l2_bytes / 2);
        Self::new(l1, l2)
    }
}

//------------//
// Test utils //
//------------//

#[cfg(test)]
mod test_util;

#[cfg(test)]
mod tests {
    use super::Cache;

    #[test]
    fn cache_budgets_scale_detected_sizes() {
        for (l1d_bytes, l2_bytes, l1_budget, l2_budget) in [
            (32 * 1024, 256 * 1024, 24 * 1024, 128 * 1024),
            (48_000, 1_250_000, 36_000, 625_000),
            (128 * 1024, 4 * 1024 * 1024, 96 * 1024, 2 * 1024 * 1024),
            (7, 7, 5, 3),
            (0, 0, 1, 1),
            (1, 1, 1, 1),
        ] {
            let cache = Cache::from_cache_sizes(l1d_bytes, l2_bytes);
            assert_eq!(cache.l1().get(), l1_budget);
            assert_eq!(cache.l2().get(), l2_budget);
        }
    }
}
