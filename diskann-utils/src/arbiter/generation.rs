/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Generation(u64);

impl Generation {
    const MASK: u64 = 0x3fff_ffff_ffff_ffff;

    pub const fn new(value: u64, metadata: Metadata) -> Option<Self> {
        if value > Self::MASK {
            None
        } else {
            Some(Self::from_raw((metadata as u64) << 62 | value))
        }
    }

    /// Return the max possible generation
    pub const fn max(metadata: Metadata) -> Self {
        Self::from_raw((metadata as u64) << 62 | Self::MASK)
    }

    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u64 {
        self.0 & Self::MASK
    }

    pub const fn raw(self) -> u64 {
        self.0
    }

    pub const fn metadata(self) -> Metadata {
        match (self.0 & !Self::MASK) >> 62 {
            0 => Metadata::Zero,
            1 => Metadata::One,
            2 => Metadata::Two,
            _ => Metadata::Three,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Metadata {
    Zero = 0,
    One = 1,
    Two = 2,
    Three = 3,
}

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Ref<'a>(&'a AtomicU64);

impl<'a> Ref<'a> {
    pub(crate) fn new(slot: &'a AtomicU64) -> Self {
        Self(slot)
    }

    fn inner(&self) -> &'a AtomicU64 {
        self.0
    }

    pub fn get(&self, ordering: Ordering) -> Generation {
        Generation::from_raw(self.0.load(ordering))
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Mut<'a>(Ref<'a>);

impl<'a> Mut<'a> {
    pub(crate) fn new(slot: &'a AtomicU64) -> Self {
        Self(Ref::new(slot))
    }

    pub fn try_set(
        &self,
        current: Generation,
        new: Generation,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Generation, Generation> {
        self.inner()
            .compare_exchange(current.raw(), new.raw(), success, failure)
            .map(Generation::from_raw)
            .map_err(Generation::from_raw)
    }

    pub fn set(&self, generation: Generation, ordering: Ordering) {
        self.inner().store(generation.raw(), ordering)
    }
}

impl<'a> std::ops::Deref for Mut<'a> {
    type Target = Ref<'a>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
