/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "integration test tools are not production code"
)]

use std::num::{NonZeroU32, NonZeroUsize};

use crate::{
    num::{Capacity, MaxDegree},
    store::{self, checked},
};

use super::boilerplate;

#[derive(Debug)]
pub struct Config {
    pub capacity: usize,
    pub epoch_guard_slots: usize,
    pub freelist_recycle_capacity: usize,
}

boilerplate!(
    checked::Checked => Store,
    for<'a> checked::Reader<'a> => Reader,
    for<'a> checked::Slot<'a> => Writer,
);

impl Store {
    /// Construct a store with `config.capacity` writable slots and no frozen points.
    ///
    /// # Panics
    ///
    /// Panics if the underlying store could not be constructed (e.g. `config.capacity`
    /// exceeds `u32::MAX`) or if other configuration parameters such as the number of
    /// epoch guard slots are invalid (e.g. zero).
    pub fn new(config: Config) -> Self {
        let store_layout = store::Layout::new(Capacity::new(config.capacity), MaxDegree::new(0), 0);

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

        let plugin_config = checked::Checked::config();
        let store = store::Store::new(store_layout, store_config, plugin_config)
            .expect("failed to construct store");

        Self { store }
    }
}

#[derive(Debug)]
pub struct Value<'a> {
    value: checked::Value<'a>,
}

impl<'a> Value<'a> {
    fn new(value: checked::Value<'a>) -> Self {
        Self { value }
    }

    pub fn get(&self) -> u64 {
        self.value.get()
    }
}

impl Reader<'_> {
    pub fn read(&self, i: usize) -> Option<Value<'_>> {
        self.reader.read(i.try_into().unwrap()).map(Value::new)
    }
}

impl<'a> Writer<'a> {
    pub fn set(&mut self, v: u64) {
        self.slot.data().set(v)
    }
}
