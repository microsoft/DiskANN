/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # GEMM-lite Matrix Kernels
//!
//! This module is a work in progress. There are many pieces that are still needed:
//!
//! * Proper run time cache-size detection.
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
//! * `N`: The number of oclumns of `B` and `C`.
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
mod driver;
mod num;
mod ptr;
mod util;

// re-export
pub(crate) use driver::Drive;
pub(crate) use num::DimK;

/// Placeholder model for CPU caches.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Cache {
    l1: NonZeroUsize,
    l2: NonZeroUsize,
}

impl Cache {
    fn new(l1: NonZeroUsize, l2: NonZeroUsize) -> Self {
        Self { l1, l2 }
    }

    /// Return the L1 cache size in bytes.
    fn l1(&self) -> NonZeroUsize {
        self.l1
    }

    /// Return the L2 cache size in bytes.
    fn l2(&self) -> NonZeroUsize {
        self.l2
    }

    /// Try to detect the cache of the CPU.
    ///
    /// # Note
    ///
    /// This currently doesn't work.
    pub(crate) fn detect() -> Self {
        const L1: NonZeroUsize = NonZeroUsize::new((3 * 48_000) / 4).unwrap();
        const L2: NonZeroUsize = NonZeroUsize::new(1_250_000 / 2).unwrap();
        Self::new(L1, L2)
    }
}

//------------//
// Test utils //
//------------//

#[cfg(test)]
mod test_util;
