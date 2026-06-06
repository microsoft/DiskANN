/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug)]
#[repr(transparent)]
pub struct Tag(AtomicU64);

impl Tag {
    pub const fn new(generation: Generation) -> Self {
        Self(AtomicU64::new(generation.value()))
    }

    pub fn as_ref(&self) -> Ref<'_> {
        Ref::new(&self.0)
    }

    pub fn as_mut(&self) -> Mut<'_> {
        Mut::new(&self.0)
    }

    pub unsafe fn from_ptr<'a>(ptr: *mut Tag) -> &'a Self {
        unsafe { &*ptr }
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Generation(u64);

impl Generation {
    pub const MAX: Self = Self::new(u64::MAX);

    // Reserved generations.
    //
    // These all have small values, with `0` marking the "available" state.
    // In this way, zeroed allocations for tags naturally begin in the "available" state and
    // don't require additional initialization.
    //
    // If you add states - make sure to increment the `RESERVED` marker!
    pub(crate) const AVAILABLE: Self = Self::new(0);
    pub(crate) const OWNED: Self = Self::new(1);
    pub(crate) const FROZEN: Self = Self::new(2);
    const RESERVED: Self = Self::FROZEN;

    #[must_use = "this function has no side-effects"]
    pub(crate) fn is_reserved(self) -> bool {
        self <= Self::RESERVED
    }

    #[inline]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[inline]
    pub const fn value(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Ref<'a>(&'a AtomicU64);

impl<'a> Ref<'a> {
    #[inline]
    pub(crate) fn new(slot: &'a AtomicU64) -> Self {
        Self(slot)
    }

    #[inline]
    fn inner(&self) -> &'a AtomicU64 {
        self.0
    }

    #[inline]
    pub fn get(&self, ordering: Ordering) -> Generation {
        Generation::new(self.0.load(ordering))
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Mut<'a>(Ref<'a>);

impl<'a> Mut<'a> {
    #[inline]
    pub(crate) fn new(slot: &'a AtomicU64) -> Self {
        Self(Ref::new(slot))
    }

    #[inline]
    pub fn try_set(
        &self,
        current: Generation,
        new: Generation,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Generation, Generation> {
        self.inner()
            .compare_exchange(current.value(), new.value(), success, failure)
            .map(Generation::new)
            .map_err(Generation::new)
    }

    #[inline]
    pub fn set(&self, generation: Generation, ordering: Ordering) {
        self.inner().store(generation.value(), ordering)
    }
}

impl<'a> std::ops::Deref for Mut<'a> {
    type Target = Ref<'a>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
