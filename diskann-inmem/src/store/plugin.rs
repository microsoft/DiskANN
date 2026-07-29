/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{mem::ManuallyDrop, fmt::Debug, sync::atomic::Ordering};

use crate::{
    buffer::{Buffer, RawSlice},
    epoch,
    num::Bytes,
    tag::AtomicTag,
};

pub(crate) trait Plugin: 'static {
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

#[derive(Debug)]
pub(crate) struct ManagedSlot<T>
where
    T: Slot,
{
    slot: ManuallyDrop<T>
}

impl<T> ManagedSlot<T>
where
    T: Slot,
{
    fn new(slot: T) -> Self {
        Self {
            slot: ManuallyDrop::new(slot),
        }
    }

    unsafe fn publish(self) {
        let mut me = ManuallyDrop::new(self);
        unsafe { ManuallyDrop::take(&mut me.slot).publish() }
    }

    unsafe fn freeze(self) {
        let mut me = ManuallyDrop::new(self);
        unsafe { ManuallyDrop::take(&mut me.slot).freeze() }
    }
}

impl<T> Drop for ManagedSlot<T>
where
    T: Slot,
{
    fn drop(&mut self) {
        unsafe { ManuallyDrop::take(&mut self.slot).abort() }
    }
}

