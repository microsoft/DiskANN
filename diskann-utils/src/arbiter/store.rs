/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::atomic::{AtomicU64, Ordering};

use super::{epoch, generation, Generation, Buffer, buffer, Freelist};

#[derive(Debug)]
pub struct Store {
    buffer: Buffer,
    tags: Vec<AtomicU64>,
    freelist: Freelist,
    registry: epoch::Registry,
}

impl Store {
    pub fn tag(&self, i: usize) -> Option<generation::Ref<'_>> {
        self.tags.get(i).map(generation::Ref::new)
    }

    pub fn reader(&self) -> Reader<'_> {
        Reader {
            store: self,
            epoch: self.registry.register(),
        }
    }

    pub fn write(&self, i: usize) -> Option<Write<'_>> {
        let tag = self.tag_mut(i)?;
        match tag.try_set(
            Generation::max(State::Available.into()),
            Generation::max(State::Owned.into()),
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => Some(Write {
                tag,
                generation: Generation::new(self.registry.generation(), State::Used.into())?,
                data: self.buffer.get(i).unwrap(),
            }),
            Err(_) => None,
        }
    }

    /// Creating a `Mut` is impossible for user code. Exposing this functionality would
    /// allow user code to break all safety invariantes this data structure relies on.
    fn tag_mut(&self, i: usize) -> Option<generation::Mut<'_>> {
        self.tags.get(i).map(generation::Mut::new)
    }

}

#[derive(Debug, Clone, Copy)]
pub enum State {
    Used = 0,
    /// Another thread is doing something critical with this slot. Please leave it alone.
    Owned = 1,
    /// Another thread is doing something critical with this slot. Please leave it alone.
    SoftDeleted = 2,
    /// The associated slot is guaranteed to have no active readers.
    Available = 3,
}

impl From<State> for generation::Metadata {
    fn from(state: State) -> Self {
        match state {
            State::Used => Self::Zero,
            State::Owned => Self::One,
            State::SoftDeleted => Self::Two,
            State::Available => Self::Three,
        }
    }
}

impl From<generation::Metadata> for State {
    fn from(metadata: generation::Metadata) -> Self {
        match metadata {
            generation::Metadata::Zero => Self::Used,
            generation::Metadata::One => Self::Owned,
            generation::Metadata::Two => Self::SoftDeleted,
            generation::Metadata::Three => Self::Available,
        }
    }
}

pub struct Reader<'a> {
    store: &'a Store,
    epoch: epoch::Guard<'a>,
}

impl<'a> Reader<'a> {
    /// Attempt to read the value at index `i`. This can fail for any of the
    /// following reasons:
    ///
    /// 1. Index `i` is out-of-bounds.
    /// 2. The read cannot be guaranteed to be race-free.
    fn read(&self, i: usize) -> Option<&[u8]> {
        // Only access if allowed by the generation tag.
        let generation = self.store.tag(i)?.get(Ordering::Acquire);
        if generation.value() <= self.epoch.generation() {
            Some(unsafe { self.store.buffer.get(i)?.as_slice() })
        } else {
            None
        }
    }
}

pub struct Write<'a> {
    tag: generation::Mut<'a>,
    generation: Generation,
    data: buffer::Slice<'a>,
}

impl<'a> Write<'a> {
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { self.data.as_mut_slice() }
    }
}

impl Drop for Write<'_> {
    fn drop(&mut self) {
        self.tag.set(self.generation, Ordering::Release)
    }
}

