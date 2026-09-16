/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use thiserror::Error;

use crate::{
    epoch,
    num::IdLimit,
    store::{Lifecycle, slots},
    tag,
};

#[derive(Debug)]
pub(crate) struct Config<H, T> {
    head: H,
    tail: T,
}

impl<H, T> Config<H, T> {
    pub(crate) fn new(head: H, tail: T) -> Self {
        Self { head, tail }
    }
}

impl<H, T> slots::SlotsConfig for Config<H, T>
where
    H: slots::SlotsConfig,
    T: slots::SlotsConfig,
{
    type Slots = Cons<H::Slots, T::Slots>;
    type Error = ConsError<H::Error, T::Error>;

    unsafe fn build(
        self,
        handle: epoch::RegistryHandle,
        tags: &tag::Authoritative,
    ) -> Result<Self::Slots, Self::Error> {
        let head = unsafe { self.head.build(handle.clone(), tags) }.map_err(ConsError::Head)?;
        let tail = unsafe { self.tail.build(handle, tags) }.map_err(ConsError::Tail)?;
        Ok(Cons::new(head, tail))
    }
}

#[derive(Debug, Error)]
pub(crate) enum ConsError<H, T> {
    #[error("couldn't construct head slots")]
    Head(#[source] H),
    #[error("couldn't construct tail slots")]
    Tail(#[source] T),
}

#[derive(Debug)]
pub(crate) struct Cons<H, T> {
    head: H,
    tail: T,
}

impl<H, T> Cons<H, T> {
    fn new(head: H, tail: T) -> Self {
        Self { head, tail }
    }

    pub(crate) fn head(&self) -> &H {
        &self.head
    }

    pub(crate) fn tail(&self) -> &T {
        &self.tail
    }
}

impl<H, T> slots::Slots for Cons<H, T>
where
    H: slots::Slots,
    T: slots::Slots,
{
    type Exclusive<'a> = Exclusive<H::Exclusive<'a>, T::Exclusive<'a>>;

    fn id_limit(&self) -> IdLimit {
        self.head.id_limit()
    }

    unsafe fn acquire(&self, i: u32, _: Lifecycle) -> Self::Exclusive<'_> {
        Exclusive::new(unsafe { self.head.acquire(i, Lifecycle::new()) }, unsafe {
            self.tail.acquire(i, Lifecycle::new())
        })
    }

    unsafe fn retire(&self, i: u32, _: Lifecycle) {
        unsafe {
            self.head.retire(i, Lifecycle::new());
            self.tail.retire(i, Lifecycle::new());
        }
    }

    unsafe fn reclaim(&self, i: u32, _: Lifecycle) {
        unsafe {
            self.head.reclaim(i, Lifecycle::new());
            self.tail.reclaim(i, Lifecycle::new());
        }
    }
}

#[derive(Debug)]
pub(crate) struct Exclusive<H, T> {
    head: H,
    tail: T,
}

impl<H, T> Exclusive<H, T> {
    fn new(head: H, tail: T) -> Self {
        Self { head, tail }
    }

    pub(crate) fn head(&mut self) -> &mut H {
        &mut self.head
    }

    pub(crate) fn tail(&mut self) -> &mut T {
        &mut self.tail
    }
}

impl<H, T> slots::Exclusive for Exclusive<H, T>
where
    H: slots::Exclusive,
    T: slots::Exclusive,
{
    fn publish(self, _: Lifecycle) {
        self.head.publish(Lifecycle::new());
        self.tail.publish(Lifecycle::new());
    }

    fn freeze(self, _: Lifecycle) {
        self.head.freeze(Lifecycle::new());
        self.tail.freeze(Lifecycle::new());
    }

    fn abort(self, _: Lifecycle) {
        self.head.abort(Lifecycle::new());
        self.tail.abort(Lifecycle::new());
    }
}
