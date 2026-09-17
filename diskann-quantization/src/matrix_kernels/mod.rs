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

/// Cache budgets for matrix-kernel blocking.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Cache {
    l1: NonZeroUsize,
    l2: NonZeroUsize,
}

impl Cache {
    fn new(l1: NonZeroUsize, l2: NonZeroUsize) -> Self {
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

    /// Use memoized CPU cache sizes, or 32 KiB L1d / 256 KiB L2 if detection fails.
    /// Reserve 75% of L1d and 50% of L2 for the kernel working set.
    pub(crate) fn detect() -> Self {
        Self::from_info(cache::cache_info())
    }

    fn from_info(info: cache::CacheInfo) -> Self {
        // Subtract the reserved quarter to avoid overflowing when scaling L1d.
        let l1 = num::value_or_one(info.l1d_bytes - info.l1d_bytes.div_ceil(4));
        let l2 = num::value_or_one(info.l2_bytes / 2);
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
    use super::{Cache, cache::CacheInfo};

    #[test]
    fn cache_budgets_scale_detected_sizes() {
        for (l1d_bytes, l2_bytes, l1_budget, l2_budget) in [
            (32 * 1024, 256 * 1024, 24 * 1024, 128 * 1024),
            (48_000, 1_250_000, 36_000, 625_000),
            (128 * 1024, 4 * 1024 * 1024, 96 * 1024, 2 * 1024 * 1024),
            (7, 7, 5, 3),
            (0, 0, 1, 1),
            (1, 1, 1, 1),
            (
                usize::MAX,
                usize::MAX,
                (usize::MAX / 4) * 3 + 2,
                usize::MAX / 2,
            ),
        ] {
            let cache = Cache::from_info(CacheInfo {
                l1d_bytes,
                l2_bytes,
            });
            assert_eq!(cache.l1().get(), l1_budget);
            assert_eq!(cache.l2().get(), l2_budget);
        }
    }

    #[test]
    fn detect_uses_probed_sizes() {
        let info = super::cache::cache_info();
        let cache = Cache::detect();
        assert_eq!(cache.l1().get(), info.l1d_bytes * 3 / 4);
        assert_eq!(cache.l2().get(), info.l2_bytes / 2);
    }
}
