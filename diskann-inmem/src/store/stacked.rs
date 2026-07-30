/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::plugin::{self, Plugin};

/// A [`super::Plugin`] for cascading multiple plugins together.
#[derive(Debug)]
pub(crate) struct Stacked<T, U> {
    first: T,
    rest: U,
}

impl<T, U> Plugin for Stacked<T, U>
where
    T: Plugin,
    U: Plugin,
{
    type Slot<'a> = Slot<T::Slot<'a>, U::Slot<'a>>;

    unsafe fn acquire(&self, i: u32) -> Self::Slot<'_> {
        Slot::new(unsafe { self.first.acquire(i) }, unsafe { self.rest.acquire(i) })
    }

    fn reclaim(&self, i: u32) {
        self.first.reclaim(i);
        self.rest.reclaim(i);
    }

    fn retire(&self, i: u32) {
        self.first.retire(i);
        self.rest.retire(i);
    }
}

#[derive(Debug)]
pub(crate) struct Slot<T, U> {
    first: T,
    rest: U,
}

impl<T, U> Slot<T, U> {
    fn new(first: T, rest: U) -> Self {
        Self { first, rest }
    }

    pub(crate) fn first(&self) -> &T {
        &self.first
    }

    pub(crate) fn rest(&self) -> &U {
        &self.rest
    }
}

impl<T, U> plugin::Slot for Slot<T, U>
where
    T: plugin::Slot,
    U: plugin::Slot,
{
    fn publish(self) {
        self.first.publish();
        self.rest.publish();
    }
    fn freeze(self) {
        self.first.freeze();
        self.rest.freeze();
    }
    fn abort(self) {
        self.first.abort();
        self.rest.abort();
    }
}
