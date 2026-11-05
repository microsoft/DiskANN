/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! An optional [`slots::Slots`].
//!
//! This allows run-time configuration of optional stores.

use crate::{
    epoch,
    num::IdLimit,
    store::{Lifecycle, slots},
    tag,
};

impl<T> slots::SlotsConfig for Option<T>
where
    T: slots::SlotsConfig,
{
    type Slots = Optional<T::Slots>;
    type Error = T::Error;

    unsafe fn build(
        self,
        handle: epoch::RegistryHandle,
        tags: &tag::Authoritative,
    ) -> Result<Self::Slots, Self::Error> {
        let slots = self
            .map(|config| {
                // SAFETY: Inherited from caller.
                unsafe { config.build(handle, tags) }
            })
            .transpose()?;

        Ok(Optional {
            slots,
            id_limit: tags.id_limit(),
        })
    }
}

/// An optional [`slots::Slots`]. When disabled, the inner `T` is never constructed.
#[derive(Debug)]
pub(crate) struct Optional<T> {
    slots: Option<T>,
    id_limit: IdLimit,
}

impl<T> Optional<T> {
    /// Return the inner slots.
    ///
    /// Returns `None` if disabled.
    pub(crate) fn slots(&self) -> Option<&T> {
        self.slots.as_ref()
    }
}

impl<T> slots::Slots for Optional<T>
where
    T: slots::Slots,
{
    type Exclusive<'a> = Option<T::Exclusive<'a>>;

    fn id_limit(&self) -> IdLimit {
        match &self.slots {
            Some(slots) => slots.id_limit(),
            None => self.id_limit,
        }
    }

    unsafe fn acquire(&self, i: u32, _: Lifecycle) -> Option<T::Exclusive<'_>> {
        debug_assert!(self.id_limit.is_in_bounds(i));

        self.slots.as_ref().map(|slots| {
            // SAFETY: Inherited from caller.
            unsafe { slots.acquire(i, Lifecycle::new()) }
        })
    }

    unsafe fn retire(&self, i: u32, _: Lifecycle) {
        if let Some(slots) = &self.slots {
            // SAFETY: Inherited from caller.
            unsafe { slots.retire(i, Lifecycle::new()) }
        }
    }

    unsafe fn reclaim(&self, i: u32, _: Lifecycle) {
        if let Some(slots) = &self.slots {
            // SAFETY: Inherited from caller.
            unsafe { slots.reclaim(i, Lifecycle::new()) }
        }
    }
}

impl<T> slots::Exclusive for Option<T>
where
    T: slots::Exclusive,
{
    fn publish(self, _: Lifecycle) {
        if let Some(exclusive) = self {
            exclusive.publish(Lifecycle::new());
        }
    }

    fn freeze(self, _: Lifecycle) {
        if let Some(exclusive) = self {
            exclusive.freeze(Lifecycle::new());
        }
    }

    fn abort(self, _: Lifecycle) {
        if let Some(exclusive) = self {
            exclusive.abort(Lifecycle::new());
        }
    }
}
