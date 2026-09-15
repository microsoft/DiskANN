/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{
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

    unsafe fn build(self, tags: &tag::Authoritative) -> Result<Self::Slots, Self::Error> {
        let slots = self.map(|config| unsafe { config.build(tags) }).transpose()?;
        Ok(Optional {
            slots,
            id_limit: tags.id_limit(),
        })
    }
}

#[derive(Debug)]
pub(crate) struct Optional<T> {
    slots: Option<T>,
    id_limit: IdLimit,
}

impl<T> Optional<T> {
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
        match &self.slots {
            Some(slots) => Some(unsafe { slots.acquire(i, Lifecycle::new()) }),
            None => None,
        }
    }

    unsafe fn retire(&self, i: u32, _: Lifecycle) {
        if let Some(slots) = &self.slots {
            unsafe { slots.retire(i, Lifecycle::new()) }
        }
    }

    unsafe fn reclaim(&self, i: u32, _: Lifecycle) {
        if let Some(slots) = &self.slots {
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
