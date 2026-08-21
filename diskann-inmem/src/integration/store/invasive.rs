/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#![expect(
    clippy::expect_used,
    reason = "integration test tools are not production code"
)]

use std::num::{NonZeroU32, NonZeroUsize};

use crate::{
    num::{Bytes, Capacity, MaxDegree},
    store,
};

use super::boilerplate;

#[derive(Debug)]
pub struct Config {
    pub capacity: usize,
    pub entry_bytes: usize,
    pub epoch_guard_slots: usize,
    pub freelist_recycle_capacity: usize,
}

boilerplate!(
    store::invasive::Invasive => Store,
    for<'a> store::invasive::Reader<'a> => Reader,
    for<'a> store::invasive::Slot<'a> => Writer,
);

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
        let store_layout = store::Layout::new(Capacity::new(config.capacity), MaxDegree::new(0), 1);

        let store_config = store::Config::__exhaustive(
            NonZeroUsize::new(config.epoch_guard_slots)
                .expect("`epoch_guard_slots` must be non-zero"),
            NonZeroU32::new(
                config
                    .freelist_recycle_capacity
                    .try_into()
                    .expect("`freelist_recycle_capacity` must fit within 32-bits"),
            )
            .expect("`freelist_recycle_capacity` must be non-zero"),
        );

        let plugin_config = store::invasive::Invasive::config(Bytes::new(config.entry_bytes));
        let store = store::Store::new(store_layout, store_config, plugin_config)
            .expect("failed to construct store");

        Self { store }
    }
}

impl<'a> Reader<'a> {
    pub fn read(&self, i: usize) -> Option<&[u8]> {
        self.reader.read(i)
    }
}

impl<'a> Writer<'a> {
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.slot.data().as_mut_slice()
    }
}
