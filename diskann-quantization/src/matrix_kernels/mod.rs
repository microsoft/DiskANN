/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

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
pub(crate) use ptr::Slice;

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
}

impl Default for Cache {
    fn default() -> Self {
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
