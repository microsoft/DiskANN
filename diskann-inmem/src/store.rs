/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    iter::repeat_n,
    num::NonZeroU32,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
};

use diskann_utils::arbiter::{self, epoch, generation, Buffer, Freelist, Generation, Slice};

#[derive(Debug)]
pub struct Primary {
    // The invasive store where concurrency tags are stored inline with the data.
    //
    // These tags are mirrored from `tags` - with the latter being used for secondary scans
    // offering slightly better locality.
    buffer: Buffer,
    tags: Vec<generation::Tag>,
    freelist: Freelist,
    registry: epoch::Registry,
    drain: Mutex<Vec<(u32, u64)>>,
}

const SENTINEL: Generation = Generation::max(State::Available.metadata());

const SPLIT: usize = std::mem::size_of::<generation::Tag>();

impl Primary {
    pub fn new(len: usize, bytes: usize) -> Self {
        Self {
            buffer: Buffer::new(bytes + SPLIT, len),
            tags: repeat_n(SENTINEL, len)
                .map(|v| generation::Tag::new(v))
                .collect(),
            freelist: Freelist::new(len.try_into().unwrap(), NonZeroU32::new(1024).unwrap()),
            registry: epoch::Registry::new(),
            drain: Mutex::new(Vec::new()),
        }
    }

    #[inline]
    fn tag(&self, i: usize) -> Option<generation::Ref<'_>> {
        self.tags.get(i).map(|v| v.as_ref())
    }

    pub fn drain(&self) -> usize {
        let mut drain = self.drain.lock().unwrap();
        let waiter = self.registry.waiting();
        let before = drain.len();
        drain.retain(|(i, generation)| {
            if waiter > *generation {
                self.freelist.push(*i);
                false
            } else {
                true
            }
        });
        before - drain.len()
    }

    pub fn reader(&self) -> Reader<'_> {
        Reader {
            buffer: &self.buffer,
            epoch: self.registry.register(),
        }
    }

    pub(crate) fn write(&self, i: usize) -> Option<Write<'_>> {
        let tag = self.tag_mut(i)?;
        match tag.try_set(
            Generation::max(State::Available.into()),
            Generation::max(State::Owned.into()),
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                let (mirror, data) = unsafe { self.data(i) };
                let write = Write {
                    tag,
                    mirror,
                    generation: Generation::new(self.registry.generation(), State::Used.into())?,
                    data,
                };
                Some(write)
            }
            Err(_) => None,
        }
    }

    pub(crate) fn delete(&self, i: usize) -> bool {
        let tag = self.tag_mut(i).unwrap();
        let current = tag.get(Ordering::Relaxed);

        // We can only delete data if we transition from `Used` -> `SoftDeleted`.
        if current.metadata() != (State::Used).metadata() {
            return false;
        }

        let want = Generation::max(State::SoftDeleted.into());

        // Even if we make this change, we can't access any data until we wait for the
        // epoch to be bumped. As such, relaxed semantics are fine.
        match tag.try_set(current, want, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(current) => {
                // Set the metadata in the mirror as well.
                let (mirror, _) = unsafe { self.data(i) };
                mirror.set(want, Ordering::Relaxed);
                let wait_for = self.registry.advance();
                self.drain.lock().unwrap().push((i.try_into().unwrap(), wait_for));
                true
            }
            Err(_) => false,
        }
    }

    unsafe fn data(&self, i: usize) -> (generation::Mut<'_>, Slice<'_>) {
        let (mirror, data) = unsafe { self.buffer.get_unchecked(i) }.split(SPLIT);
        (
            unsafe { generation::Tag::from_ptr(mirror.as_ptr().as_ptr().cast()) }.as_mut(),
            data,
        )
    }

    /// Creating a `Mut` is impossible for user code. Exposing this functionality would
    /// allow user code to break all safety invariantes this data structure relies on.
    fn tag_mut(&self, i: usize) -> Option<generation::Mut<'_>> {
        self.tags.get(i).map(|v| v.as_mut())
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

impl State {
    const fn metadata(self) -> generation::Metadata {
        match self {
            State::Used => generation::Metadata::Zero,
            State::Owned => generation::Metadata::One,
            State::SoftDeleted => generation::Metadata::Two,
            State::Available => generation::Metadata::Three,
        }
    }
}

impl From<State> for generation::Metadata {
    fn from(state: State) -> Self {
        state.metadata()
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

#[derive(Debug)]
pub struct Reader<'a> {
    buffer: &'a Buffer,
    epoch: epoch::Guard<'a>,
}

impl<'a> Reader<'a> {
    /// Attempt to read the value at index `i`. This can fail for any of the
    /// following reasons:
    ///
    /// 1. Index `i` is out-of-bounds.
    /// 2. The read cannot be guaranteed to be race-free.
    #[inline]
    pub fn read(&self, i: usize) -> Option<&[u8]> {
        let (generation, rest) = match self.buffer.get(i) {
            Some(slice) => slice.split(SPLIT),
            None => return None,
        };

        // NOTE: Must be `Acquire` to correctly synchronize with writes.
        let generation = unsafe { generation::Tag::from_ptr(generation.as_ptr().as_ptr().cast()) }
            .as_ref()
            .get(Ordering::Acquire);

        if generation.value() <= self.epoch.generation() {
            // SAFETY: tags and buffer always have the same length, and we
            // verified i < tags.len() above.
            Some(unsafe { rest.as_slice() })
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct Write<'a> {
    tag: generation::Mut<'a>,
    mirror: generation::Mut<'a>,
    generation: Generation,
    data: Slice<'a>,
}

impl<'a> Write<'a> {
    pub fn raw_slice(&mut self) -> Slice<'_> {
        self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { self.data.as_mut_slice() }
    }
}

impl Drop for Write<'_> {
    fn drop(&mut self) {
        self.mirror.set(self.generation, Ordering::Release);
        self.tag.set(self.generation, Ordering::Release);
    }
}
