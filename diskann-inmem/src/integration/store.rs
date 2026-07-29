/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#![expect(
    clippy::expect_used,
    reason = "integration test tools are not production code"
)]

use std::num::{NonZeroU32, NonZeroUsize};

use diskann_utils::views::Matrix;

use crate::{num::Bytes, store};

#[derive(Debug)]
pub struct Config {
    pub capacity: usize,
    pub entry_bytes: usize,
    pub epoch_guard_slots: usize,
    pub freelist_recycle_capacity: usize,
}

#[derive(Debug)]
pub struct Store {
    store: store::Store,
}

impl Store {
    /// Construct a store with `config.capacity` writable slots, each holding
    /// `config.entry_bytes` bytes.
    ///
    /// A single zeroed frozen point is created internally to satisfy the underlying
    /// store's requirement of at least one frozen entry; it occupies the highest slot
    /// index and is always readable.
    ///
    /// # Panics
    ///
    /// Panics if the underlying store could not be constructed (e.g. `config.capacity` plus
    /// the frozen point exceeds `u32::MAX`) or if other configuration parameters such as
    /// the number of epoch guard slots are invalid (e.g. zero).
    pub fn new(config: Config) -> Self {
        let mut store_config =
            store::Config::new(config.capacity, Bytes::new(config.entry_bytes), 0);

        store_config
            .epoch_guard_slots(
                NonZeroUsize::new(config.epoch_guard_slots)
                    .expect("`epoch_guard_slots` must be non-zero"),
            )
            .freelist_recycle_capacity(
                NonZeroU32::new(
                    config
                        .freelist_recycle_capacity
                        .try_into()
                        .expect("`freelist_recycle_capacity` must fit within 32-bits"),
                )
                .expect("`freelist_recycle_capacity` must be non-zero"),
            );

        let data = Matrix::new(0u8, 1, config.entry_bytes);
        let store =
            store::Store::new(store_config, data.as_view()).expect("failed to construct store");
        Self { store }
    }

    /// Return the total number of slots, including the frozen point.
    pub fn slots(&self) -> usize {
        self.store.frozen().end as usize
    }

    /// Return the range of writable (non-frozen) slot indices.
    pub fn writable(&self) -> std::ops::Range<usize> {
        0..(self.store.frozen().start as usize)
    }

    /// Attempt to reclaim retired slots, returning the number reclaimed if any.
    pub fn reclaim(&self) -> Option<usize> {
        self.store.try_drain()
    }

    pub fn acquire(&self) -> Option<Writer<'_>> {
        self.store.acquire().map(Writer::new)
    }

    #[must_use = "result indicates success or failure"]
    pub fn retire(&self, i: usize) -> bool {
        self.store.retire(i).is_ok()
    }

    pub fn reader(&self) -> Option<Reader<'_>> {
        match self.store.reader() {
            Ok(reader) => Some(Reader::new(reader)),
            Err(crate::epoch::Unavailable) => None,
        }
    }
}

pub struct Reader<'a> {
    reader: store::Reader<'a>,
}

impl<'a> Reader<'a> {
    fn new(reader: store::Reader<'a>) -> Self {
        Self { reader }
    }

    pub fn read(&self, i: usize) -> Option<&[u8]> {
        self.reader.read(i)
    }
}

pub struct Writer<'a> {
    slot: store::Slot<'a>,
}

impl<'a> Writer<'a> {
    fn new(slot: store::Slot<'a>) -> Self {
        Self { slot }
    }

    pub fn publish(self) {
        self.slot.publish();
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.slot.as_mut_slice()
    }
}
