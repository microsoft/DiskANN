/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

pub(crate) mod blocks;
pub(crate) mod bounds;
pub(crate) mod kernel;
pub(crate) mod num;
pub(crate) mod ptr;
pub(crate) mod util;

#[cfg(test)]
mod test_util;

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
