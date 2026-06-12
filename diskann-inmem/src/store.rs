/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    iter::repeat_n,
    num::NonZeroU32,
    sync::{atomic::Ordering, Mutex},
};

use crate::{
    arbiter::{epoch, generation, Buffer, Freelist, Generation, RawSlice},
    num::{Align, Bytes},
    Neighbors,
};

#[derive(Debug)]
pub struct Primary {
    // The invasive store where concurrency tags are stored inline with the data.
    //
    // These tags are mirrored from `tags` - with the latter being used for secondary scans
    // offering slightly better locality.
    buffer: Buffer,
    unpadded: Bytes,
    tags: Vec<generation::Tag>,
    freelist: Freelist,
    registry: epoch::Registry,
    neighbors: Neighbors,
    drain: Mutex<Vec<(u32, Generation)>>,
}

const SPLIT: Bytes = Bytes::size_of::<generation::Tag>();

impl Primary {
    pub fn new(entries: usize, bytes: Bytes, max_neighbors: usize) -> Self {
        let unpadded = bytes.checked_add(SPLIT).unwrap();
        let padded_bytes = unpadded.checked_next_multiple_of(Bytes::CACHELINE).unwrap();

        Self {
            buffer: Buffer::new(entries, padded_bytes, Align::_128).unwrap(),
            unpadded,
            tags: repeat_n(Generation::AVAILABLE, entries)
                .map(|v| generation::Tag::new(v))
                .collect(),
            freelist: Freelist::new(entries.try_into().unwrap(), NonZeroU32::new(1024).unwrap()),
            registry: epoch::Registry::new(),
            neighbors: Neighbors::new(entries, max_neighbors),
            drain: Mutex::new(Vec::new()),
        }
    }

    #[inline]
    fn tag(&self, i: usize) -> Option<generation::Ref<'_>> {
        self.tags.get(i).map(|v| v.as_ref())
    }

    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    pub fn drain(&self) -> usize {
        let mut drain = self.drain.lock().unwrap();
        let waiter = self.registry.waiting();
        let before = drain.len();
        drain.retain(|(i, generation)| {
            if waiter < *generation {
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
            unpadded: self.unpadded,
            neighbors: &self.neighbors,
            epoch: self.registry.register(),
        }
    }

    pub(crate) fn write(&self, i: usize) -> Option<Write<'_>> {
        let tag = self.tag_mut(i)?;
        match tag.try_set(
            Generation::AVAILABLE,
            Generation::OWNED,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                let (mirror, data) = unsafe { self.data(i) };
                let write = Write {
                    tag,
                    mirror,
                    generation: self.registry.generation(),
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

        // We can only perform a deletion if the generation is not in a reserved state.
        if current.is_reserved() {
            return false;
        }

        let owned = Generation::OWNED;

        // Even if we make this change, we can't access any data until we wait for the
        // epoch to be bumped. As such, relaxed semantics are fine.
        match tag.try_set(current, owned, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(current) => {
                // Set the metadata in the mirror as well.
                let (mirror, _) = unsafe { self.data(i) };
                mirror.set(owned, Ordering::Relaxed);
                let wait_for = self.registry.advance();
                self.drain
                    .lock()
                    .unwrap()
                    .push((i.try_into().unwrap(), wait_for));
                true
            }
            Err(_) => false,
        }
    }

    unsafe fn data(&self, i: usize) -> (generation::Mut<'_>, RawSlice<'_>) {
        let (mirror, data) = unsafe { self.buffer.get_unchecked(i) }
            .truncate(self.unpadded)
            .split(SPLIT);
        (
            unsafe { generation::Tag::from_ptr(mirror.as_mut_ptr().cast()) }.as_mut(),
            data,
        )
    }

    /// Creating a `Mut` is impossible for user code. Exposing this functionality would
    /// allow user code to break all safety invariantes this data structure relies on.
    fn tag_mut(&self, i: usize) -> Option<generation::Mut<'_>> {
        self.tags.get(i).map(|v| v.as_mut())
    }
}

#[derive(Debug)]
pub struct Reader<'a> {
    buffer: &'a Buffer,
    unpadded: Bytes,
    neighbors: &'a Neighbors,
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
        if self.is_in_bounds(i) {
            unsafe { self.read_in_bounds(i) }
        } else {
            None
        }
    }

    /// Return `true` if the index `i` is in-bounds.
    #[inline]
    #[must_use = "this function has no side-effects"]
    pub fn is_in_bounds(&self, i: usize) -> bool {
        i < self.buffer.len()
    }

    #[inline]
    pub(crate) unsafe fn read_in_bounds(&self, i: usize) -> Option<&[u8]> {
        debug_assert!(self.is_in_bounds(i));

        let (generation, rest) = unsafe {
            self.buffer
                .get_unchecked(i)
                .truncate_unchecked(self.unpadded)
                .split_unchecked(SPLIT)
        };

        // NOTE: Must be `Acquire` to correctly synchronize with writes.
        let generation = unsafe { generation::Tag::from_ptr(generation.as_mut_ptr().cast()) }
            .as_ref()
            .get(Ordering::Acquire);

        if generation >= self.epoch.generation() {
            // SAFETY: tags and buffer always have the same length, and we
            // verified i < tags.len() above.
            Some(unsafe { rest.as_slice() })
        } else {
            None
        }
    }

    /// Return the raw data slice for index `i` without any race guarantees.
    ///
    /// # Safety
    ///
    /// The index `i` must be in-bounds.
    #[inline]
    pub(crate) unsafe fn read_raw_unchecked(&self, i: usize) -> RawSlice<'_> {
        unsafe { self.buffer.get_unchecked(i) }.truncate(self.unpadded)
    }

    /// Return the number of bytes for each entry.
    pub(crate) fn bytes(&self) -> Bytes {
        self.unpadded
    }

    // TODO: We may want to lock `Neighbors` in some way to enable exclusive access during
    // operations like snapshots.
    pub(crate) fn neighbors(&self) -> &Neighbors {
        &self.neighbors
    }
}

#[derive(Debug)]
pub struct Write<'a> {
    tag: generation::Mut<'a>,
    mirror: generation::Mut<'a>,
    generation: Generation,
    data: RawSlice<'a>,
}

impl<'a> Write<'a> {
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
