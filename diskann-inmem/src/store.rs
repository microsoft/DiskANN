/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{iter::repeat_n, num::NonZeroU32, sync::atomic::Ordering};

use diskann::utils::IntoUsize;
use diskann_utils::views::MatrixView;

use crate::{
    arbiter::{epoch, freelist, generation, Buffer, Freelist, Generation, RawSlice},
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

    // The number of unfrozen points. This is guaranteed to be less than `buffer`.
    unfrozen: usize,
    tags: Vec<generation::Tag>,
    freelist: Freelist,
    registry: epoch::Registry,
    neighbors: Neighbors,
}

const SPLIT: Bytes = Bytes::size_of::<generation::Tag>();
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

        let total = entries.checked_add(init.nrows()).unwrap();

        let this = Self {
            buffer: Buffer::new(total, padded_bytes, Align::_128).unwrap(),
            unpadded,
            unfrozen: entries,
            tags: repeat_n(Generation::AVAILABLE, total)
                .map(|v| generation::Tag::new(v))
                .collect(),

            // NOTE: The `Freelist` is initialized to `entries` and not `total` because
            // we do not want it to release frozen IDs.
            freelist: Freelist::new(entries.try_into().unwrap(), NonZeroU32::new(1024).unwrap()),
            registry: epoch::Registry::new(),
            neighbors: Neighbors::new(entries, max_neighbors),
        };

        // Populate frozen points.
        for (i, data) in init.row_iter().enumerate() {
            let mut slot = this.slot((entries + i).try_into().unwrap()).unwrap();
            slot.as_mut_slice().copy_from_slice(data);
            slot.freeze();
        }

        this
    }

    pub fn capacity(&self) -> usize {
        self.buffer.len() - self.unfrozen
    }

    pub fn try_drain(&self) -> Option<usize> {
        fn release(tag: generation::Mut<'_>, kind: &'static str) {
            // Relaxed ordering is sufficient as all readers/writers are synchronized on
            // the central generation.
            if let Err(got) = tag.try_set(
                Generation::OWNED,
                Generation::AVAILABLE,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                panic!(
                    "CONCURRENCY VIOLATION: {} - expected {} - got {}",
                    kind,
                    Generation::AVAILABLE,
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
            release(self.tags[i.into_usize()].as_mut(), "tag");
            self.freelist.push(i);
        }
        Some(items)
    }

    pub fn reader(&self) -> Result<Reader<'_>, epoch::Unavailable> {
        Ok(Reader {
            buffer: &self.buffer,
            unpadded: self.unpadded,
            neighbors: &self.neighbors,
            epoch: self.registry.register()?,
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
                freelist::Id::Scan => unimplemented!("fallback scan not implemented"),
            }
        }
        None
    }

    fn slot(&self, i: u32) -> Option<Slot<'_>> {
        let tag = self.tag_mut(i.into_usize()).unwrap();
        match tag.try_set(
            Generation::AVAILABLE,
            Generation::OWNED,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                let (mirror, data) = unsafe { self.data_unchecked(i.into_usize()) };
                Some(Slot {
                    tag,
                    mirror,
                    generation: self.registry.generation(),
                    data,
                    slot: i,
                })
            }
            Err(_) => None,
        }
    }

    pub(crate) fn delete(&self, i: usize) -> bool {
        let guard = self.registry.register().unwrap();
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
            Ok(_) => {
                // Set the metadata in the mirror as well.
                let (mirror, _) = unsafe { self.data_unchecked(i) };
                mirror.set(owned, Ordering::Relaxed);
                guard.retire(i as u32);
                true
            }
            Err(_) => false,
        }
    }

    unsafe fn data_unchecked(&self, i: usize) -> (generation::Mut<'_>, RawSlice<'_>) {
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
pub struct Slot<'a> {
    tag: generation::Mut<'a>,
    mirror: generation::Mut<'a>,
    generation: Generation,
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
        me.mirror.set(Generation::FROZEN, Ordering::Release);
        me.tag.set(Generation::FROZEN, Ordering::Release);
    }
}

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.mirror.set(self.generation, Ordering::Release);
        self.tag.set(self.generation, Ordering::Release);
    }
}
