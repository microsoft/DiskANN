/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

mod blocks;
mod bounds;
mod kernel;
mod num;
mod ptr;
mod util;

pub mod tile;

#[derive(Debug, Clone, Copy)]
struct Cache {
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
