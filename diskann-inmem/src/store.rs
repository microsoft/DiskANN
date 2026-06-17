/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{iter::repeat_n, num::NonZeroU32, sync::atomic::Ordering};

use diskann::utils::IntoUsize;
use diskann_utils::views::MatrixView;

use crate::{
    neighbors::Neighbors,
    buffer::{Buffer, RawSlice},
    num::{Align, Bytes},
    sync::{AtomicTag, Freelist, Tag, Registry, epoch, freelist},
};

#[derive(Debug)]
pub struct Primary {
    // The invasive store where concurrency tags are stored inline with the data.
    //
    // These tags are mirrored from `tags` - with the latter being used for secondary scans
    // offering slightly better locality.
    buffer: Buffer,
    unpadded: Bytes,

    // The number of unfrozen points. This is guaranteed to be less than `buffer`.
    unfrozen: usize,
    tags: Vec<AtomicTag>,
    freelist: Freelist,
    registry: Registry,
    neighbors: Neighbors,
}

const SPLIT: Bytes = Bytes::size_of::<AtomicTag>();
const RETRY_LIMIT: usize = 20;

impl Primary {
    pub fn new(
        entries: usize,
        bytes: Bytes,
        max_neighbors: usize,
        init: MatrixView<'_, u8>,
    ) -> Self {
        assert_eq!(init.ncols(), bytes.value());
        assert_ne!(init.nrows(), 0);

        let unpadded = bytes.checked_add(SPLIT).unwrap();
        let padded_bytes = unpadded.checked_next_multiple_of(Bytes::CACHELINE).unwrap();

        let total: usize = entries.checked_add(init.nrows()).unwrap();

        let this = Self {
            buffer: Buffer::new(total, padded_bytes, Align::_128).unwrap(),
            unpadded,
            unfrozen: entries,
            tags: repeat_n(Tag::AVAILABLE, total)
                .map(|v| AtomicTag::new(v))
                .collect(),

            // NOTE: The `Freelist` is initialized to `entries` and not `total` because
            // we do not want it to release frozen IDs.
            freelist: Freelist::new(entries.try_into().unwrap(), NonZeroU32::new(1024).unwrap()),
            registry: Registry::new(),
            neighbors: Neighbors::new(total.try_into().unwrap(), max_neighbors.try_into().unwrap()).unwrap(),
        };

        // Populate frozen points.
        for (i, data) in init.row_iter().enumerate() {
            let mut slot = this.slot((entries + i).try_into().unwrap()).unwrap();
            slot.as_mut_slice().copy_from_slice(data);
            slot.freeze();
        }

        this
    }

    /// Return the range of slots containing frozen items in `self`.
    pub fn frozen(&self) -> std::ops::Range<u32> {
        (self.unfrozen as u32)..(self.buffer.len() as u32)
    }

    /// Return the number of unfrozen slots managed by `self`.
    pub fn capacity(&self) -> usize {
        self.buffer.len() - self.unfrozen
    }

    /// Attempt to reclaim retired slots.
    ///
    /// If successful, returns the number of slots reclaimed.
    pub fn try_drain(&self) -> Option<usize> {
        fn release(tag: &AtomicTag, kind: &'static str) {
            // Relaxed ordering is sufficient as all readers/writers are synchronized on
            // the central generation.
            if let Err(got) = tag.compare_exchange(
                Tag::RETIRING,
                Tag::AVAILABLE,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                panic!(
                    "CONCURRENCY VIOLATION: {} - expected {} - got {}",
                    kind,
                    Tag::AVAILABLE,
                    got,
                );
            }
        }

        let drain = self.registry.try_advance()?;
        let items = drain.len();
        for i in drain {
            // We release the mirror before the main tag. The other direction would
            // prematurely advertise availability.
            let (mirror, _) = unsafe { self.data_unchecked(i.into_usize()) };
            release(mirror, "mirror");
            release(&self.tags[i.into_usize()], "tag");
            self.freelist.push(i);
        }
        Some(items)
    }

    /// Return a [`Reader`] into the store.
    ///
    /// # Errors
    ///
    /// Returns [`epoch::Unavailable`] if there are too many active readers.
    pub fn reader(&self) -> Result<Reader<'_>, epoch::Unavailable> {
        Ok(Reader {
            buffer: &self.buffer,
            unpadded: self.unpadded,
            neighbors: &self.neighbors,
            _epoch: self.registry.guard()?,
        })
    }

    /// Attempt to acquire new slot for writing.
    pub fn acquire(&self) -> Option<Slot<'_>> {
        for _ in 0..RETRY_LIMIT {
            match self.freelist.pop() {
                freelist::Id::Found(id) => {
                    if let Some(slot) = self.slot(id) {
                        return Some(slot);
                    }
                }
                freelist::Id::Scan => match self.scan_acquire() {
                    Some(slot) => return Some(slot),
                    None => {
                        self.try_drain();
                    }
                },
            }
        }
        None
    }

    fn scan_acquire(&self) -> Option<Slot<'_>> {
        // This is potentially quite slow - but stop if we've scanned the entire range
        // without finding anything.
        let mut remaining = self.unfrozen;
        let mut chunks_since_freelist_check = 0;
        let mut acquired: Option<Slot<'_>> = None;

        while remaining != 0 {
            let chunk = self.freelist.scan();
            remaining = remaining.saturating_sub(chunk.len());

            for slot in chunk {
                let tag = self.tags.get(slot.into_usize()).unwrap();

                // If this slot is available and we haven't claimed a slot yet, try to
                // claim it. Otherwise, continue with the scan to partially repopulate the
                // freelist for other threads.
                if tag.load(Ordering::Relaxed) == Tag::AVAILABLE {
                    if acquired.is_none() {
                        acquired = unsafe { self.try_acquire(tag, slot) };
                    } else {
                        self.freelist.push(slot);
                    }
                }
            }

            if acquired.is_some() {
                return acquired;
            }

            chunks_since_freelist_check += 1;
            if chunks_since_freelist_check == 4 {
                if let Some(id) = self.freelist.pop_recycled()
                    && let Some(slot) = self.slot(id)
                {
                    return Some(slot);
                }
                chunks_since_freelist_check = 0;
            }
        }
        None
    }

    fn slot(&self, i: u32) -> Option<Slot<'_>> {
        let tag = &self.tags.get(i.into_usize()).unwrap();
        unsafe { self.try_acquire(tag, i) }
    }

    unsafe fn try_acquire<'a>(&'a self, tag: &'a AtomicTag, slot: u32) -> Option<Slot<'a>> {
        match tag.compare_exchange(
            Tag::AVAILABLE,
            Tag::OWNED,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                let (mirror, data) = unsafe { self.data_unchecked(slot.into_usize()) };
                Some(Slot {
                    tag,
                    mirror,
                    data,
                    slot,
                })
            }
            Err(_) => None,
        }
    }

    pub(crate) fn delete(&self, i: usize) -> bool {
        let guard = self.registry.guard().unwrap();
        let tag = self.tags.get(i).unwrap();
        let current = tag.load(Ordering::Relaxed);

        // We can only perform a deletion if the generation is not in a reserved state.
        if current.is_reserved() {
            return false;
        }

        let retiring = Tag::RETIRING;

        // Even if we make this change, we can't access any data until we wait for the
        // epoch to be bumped. As such, relaxed semantics are fine.
        match tag.compare_exchange(current, retiring, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => {
                // Set the metadata in the mirror as well.
                let (mirror, _) = unsafe { self.data_unchecked(i) };
                mirror.store(retiring, Ordering::Relaxed);
                guard.retire(i as u32);
                true
            }
            Err(_) => false,
        }
    }

    unsafe fn data_unchecked(&self, i: usize) -> (&AtomicTag, RawSlice<'_>) {
        let (mirror, data) = unsafe { self.buffer.get_unchecked(i) }
            .truncate(self.unpadded)
            .split(SPLIT);
        (
            unsafe { AtomicTag::from_ptr(mirror.as_mut_ptr().cast()) },
            data,
        )
    }
}

#[derive(Debug)]
pub struct Reader<'a> {
    buffer: &'a Buffer,
    unpadded: Bytes,
    neighbors: &'a Neighbors,
    // It's important that we hold onto this, even if we don't use it.
    _epoch: epoch::Guard<'a>,
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

    pub(crate) fn can_read(&self, i: usize) -> Option<bool> {
        if !self.is_in_bounds(i) {
            return None;
        }

        let tag_ptr = unsafe { self.buffer.get_unchecked(i).truncate_unchecked(SPLIT) };
        let can_read = unsafe { AtomicTag::from_ptr(tag_ptr.as_mut_ptr().cast()) }
            .load(Ordering::Acquire)
            .can_read();

        Some(can_read)
    }

    #[inline]
    pub(crate) unsafe fn read_in_bounds(&self, i: usize) -> Option<&[u8]> {
        debug_assert!(self.is_in_bounds(i));

        let (tag_ptr, rest) = unsafe {
            self.buffer
                .get_unchecked(i)
                .truncate_unchecked(self.unpadded)
                .split_unchecked(SPLIT)
        };

        // NOTE: Must be `Acquire` to correctly synchronize with writes.
        let can_read = unsafe { AtomicTag::from_ptr(tag_ptr.as_mut_ptr().cast()) }
            .load(Ordering::Acquire)
            .can_read();

        if can_read {
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
pub struct Slot<'a> {
    tag: &'a AtomicTag,
    mirror: &'a AtomicTag,
    data: RawSlice<'a>,
    slot: u32,
}

impl<'a> Slot<'a> {
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { self.data.as_mut_slice() }
    }

    /// Return the slot associated with this write.
    pub fn slot(&self) -> u32 {
        self.slot
    }

    fn freeze(self) {
        let me = std::mem::ManuallyDrop::new(self);
        me.mirror.store(Tag::FROZEN, Ordering::Release);
        me.tag.store(Tag::FROZEN, Ordering::Release);
    }
}

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.mirror.store(Tag::PUBLISHED, Ordering::Release);
        self.tag.store(Tag::PUBLISHED, Ordering::Release);
    }
}
