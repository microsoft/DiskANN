/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A [`slots::Slots`] that combines two other [`slots::Slots`].

use thiserror::Error;

use crate::{
    epoch,
    num::IdLimit,
    store::{Lifecycle, slots},
    tag,
};

/// A [`slots::SlotsConfig`] for [`Cons`].
#[derive(Debug)]
pub(crate) struct Config<F, S> {
    first: F,
    second: S,
}

impl<F, S> Config<F, S> {
    /// Create a new [`Config`] containing the `first` and `second` configs.
    pub(crate) fn new(first: F, second: S) -> Self {
        Self { first, second }
    }
}

impl<F, S> slots::SlotsConfig for Config<F, S>
where
    F: slots::SlotsConfig,
    S: slots::SlotsConfig,
{
    type Slots = Cons<F::Slots, S::Slots>;
    type Error = ConsError<F::Error, S::Error>;

    unsafe fn build(
        self,
        handle: epoch::RegistryHandle,
        tags: &tag::Authoritative,
    ) -> Result<Self::Slots, Self::Error> {
        // SAFETY: Inherited from caller.
        let first = unsafe { self.first.build(handle.clone(), tags) }.map_err(ConsError::First)?;
        // SAFETY: Inherited from caller.
        let second = unsafe { self.second.build(handle, tags) }.map_err(ConsError::Second)?;

        let first_limit = slots::Slots::id_limit(&first);
        let second_limit = slots::Slots::id_limit(&second);
        if first_limit != second_limit {
            return Err(ConsError::MismatchLimits {
                first: first_limit,
                second: second_limit,
            });
        }

        Ok(Cons::new(first, second))
    }
}

#[derive(Debug, Error)]
pub(crate) enum ConsError<F, S> {
    #[error("couldn't construct first slots")]
    First(#[source] F),
    #[error("couldn't construct second slots")]
    Second(#[source] S),
    #[error(
        "id-limit for first ({}) not equal to the id-limit for second ({})",
        first,
        second
    )]
    MismatchLimits { first: IdLimit, second: IdLimit },
}

/// A [`slots::Slots`] that combines two other [`slots::Slots`].
///
/// Lifecycle operations will first be applied to `first`, then to `second`.
#[derive(Debug)]
pub(crate) struct Cons<F, S> {
    first: F,
    second: S,
}

impl<F, S> Cons<F, S> {
    fn new(first: F, second: S) -> Self {
        Self { first, second }
    }

    /// Return the first entry in `self`.
    pub(crate) fn first(&self) -> &F {
        &self.first
    }

    /// Return the second entry in `self`.
    pub(crate) fn second(&self) -> &S {
        &self.second
    }
}

impl<F, S> slots::Slots for Cons<F, S>
where
    F: slots::Slots,
    S: slots::Slots,
{
    type Exclusive<'a> = Exclusive<F::Exclusive<'a>, S::Exclusive<'a>>;

    fn id_limit(&self) -> IdLimit {
        self.first.id_limit()
    }

    unsafe fn acquire(&self, i: u32, _: Lifecycle) -> Self::Exclusive<'_> {
        // SAFETY: Inherited from caller.
        unsafe {
            Exclusive::new(
                self.first.acquire(i, Lifecycle::new()),
                self.second.acquire(i, Lifecycle::new()),
            )
        }
    }

    unsafe fn retire(&self, i: u32, _: Lifecycle) {
        // SAFETY: Inherited from caller.
        unsafe {
            self.first.retire(i, Lifecycle::new());
            self.second.retire(i, Lifecycle::new());
        }
    }

    unsafe fn reclaim(&self, i: u32, _: Lifecycle) {
        // SAFETY: Inherited from caller.
        unsafe {
            self.first.reclaim(i, Lifecycle::new());
            self.second.reclaim(i, Lifecycle::new());
        }
    }
}

/// A [`slots::Exclusive`] for [`Cons`].
///
/// The exclusive for [`Cons::first`] is available via [`Exclusive::first`]. Similarly,
/// [`Cons::second`]'s is available via [`Exclusive::second`].
#[derive(Debug)]
pub(crate) struct Exclusive<F, S> {
    first: F,
    second: S,
}

impl<F, S> Exclusive<F, S> {
    fn new(first: F, second: S) -> Self {
        Self { first, second }
    }

    /// Return the [`slots::Exclusive`] for the first entry in `self`.
    pub(crate) fn first(&mut self) -> &mut F {
        &mut self.first
    }

    /// Return the [`slots::Exclusive`] for the second entry in `self`.
    pub(crate) fn second(&mut self) -> &mut S {
        &mut self.second
    }
}

impl<F, S> slots::Exclusive for Exclusive<F, S>
where
    F: slots::Exclusive,
    S: slots::Exclusive,
{
    fn publish(self, _: Lifecycle) {
        self.first.publish(Lifecycle::new());
        self.second.publish(Lifecycle::new());
    }

    fn freeze(self, _: Lifecycle) {
        self.first.freeze(Lifecycle::new());
        self.second.freeze(Lifecycle::new());
    }

    fn abort(self, _: Lifecycle) {
        self.first.abort(Lifecycle::new());
        self.second.abort(Lifecycle::new());
    }
}
