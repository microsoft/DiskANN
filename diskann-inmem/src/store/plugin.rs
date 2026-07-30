/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt::Debug;

pub(crate) trait Plugin: Debug + 'static {
    type Slot<'a>: Slot;

    unsafe fn acquire(&self, i: u32) -> Self::Slot<'_>;
    fn reclaim(&self, i: u32);
    fn retire(&self, i: u32);
}

pub(crate) trait Slot: Debug {
    fn publish(self);
    fn freeze(self);
    fn abort(self);
}
