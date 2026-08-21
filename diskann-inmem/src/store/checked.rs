/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    assert_matches,
    sync::atomic::{AtomicBool, Ordering},
};

use diskann::utils::IntoUsize;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::{epoch, num::IdLimit, store::Store};

use super::{Lifecycle, plugin};

#[derive(Debug, Default)]
enum State {
    #[default]
    Available,
    Readable {
        value: u64,
    },
    Frozen {
        value: u64,
    },
}

#[derive(Debug, Default)]
struct Entry {
    readable: AtomicBool,
    state: RwLock<State>,
}

impl Entry {
    #[must_use]
    fn is_readable(&self) -> bool {
        self.readable.load(Ordering::Acquire)
    }

    fn try_read(&self) -> Option<ReadEntry<'_>> {
        if self.is_readable() {
            Some(self.expect_read())
        } else {
            None
        }
    }

    fn expect_read(&self) -> ReadEntry<'_> {
        // NOTE: we *DO NOT* check for `entry.is_readable()` because there is a race where
        // the slot is retired after checking the readable state but before this function
        // is called. We still expect to acquire the `RwLockReadGuard` in this situation.

        let Some(guard) = self.state.try_read() else {
            panic!("concurrency violation when acquiring read guard");
        };

        ReadEntry {
            readable: &self.readable,
            guard,
        }
    }

    fn expect_write(&self) -> WriteEntry<'_> {
        assert!(
            !self.is_readable(),
            "concurrency violation - entry should not be readable"
        );

        // Correct usage of the concurrency protocol means that this `try_write` failing
        // is a bug.
        let Some(guard) = self.state.try_write() else {
            panic!("concurrency violation when acquiring write guard");
        };

        WriteEntry {
            readable: &self.readable,
            guard,
        }
    }
}

#[derive(Debug)]
struct ReadEntry<'a> {
    readable: &'a AtomicBool,
    guard: RwLockReadGuard<'a, State>,
}

impl ReadEntry<'_> {
    fn retire(self) {
        assert_matches!(*self.guard, State::Readable { .. });

        // TODO: Document the slightly weird order.
        drop(self.guard);
        self.readable.store(false, Ordering::Release);
    }

    fn state(&self) -> &State {
        &self.guard
    }
}

#[derive(Debug)]
struct WriteEntry<'a> {
    readable: &'a AtomicBool,
    guard: RwLockWriteGuard<'a, State>,
}

impl WriteEntry<'_> {
    fn publish(mut self, value: u64) {
        let old = self.replace(State::Readable { value });
        assert_matches!(old, State::Available);

        drop(self.guard);
        self.readable.store(true, Ordering::Release);
    }

    fn freeze(mut self, value: u64) {
        let old = self.replace(State::Frozen { value });
        assert_matches!(old, State::Available);

        drop(self.guard);
        self.readable.store(true, Ordering::Release);
    }

    fn reclaim(mut self) {
        let old = self.replace(State::Available);
        assert_matches!(old, State::Readable { .. });
    }

    /// Replace the proctected state with `state`, returning the old state.
    fn replace(&mut self, mut state: State) -> State {
        std::mem::swap(&mut *self.guard, &mut state);
        state
    }

    fn state(&self) -> &State {
        &self.guard
    }
}

#[derive(Debug)]
pub(crate) struct Config(());

impl Config {
    pub(crate) fn new() -> Self {
        Self(())
    }
}

impl plugin::PluginConfig for Config {
    type Plugin = Checked;

    fn build(self, id_limit: IdLimit) -> diskann::ANNResult<Checked> {
        Ok(Checked::new(id_limit))
    }
}

#[derive(Debug)]
pub(crate) struct Checked {
    entries: Vec<Entry>,
}

impl Checked {
    pub(crate) fn config() -> Config {
        Config::new()
    }

    pub(crate) fn new(id_limit: IdLimit) -> Self {
        Self {
            entries: std::iter::repeat_with(|| Entry::default())
                .take(id_limit.as_usize())
                .collect(),
        }
    }

    pub(crate) fn id_limit(&self) -> IdLimit {
        IdLimit::new(self.entries.len().try_into().unwrap())
    }

    pub(crate) fn reader(store: &Store<Self>) -> Result<Reader<'_>, epoch::Unavailable> {
        store.guard(|this, guard: epoch::Guard<'_>| Reader {
            parent: this,
            _guard: guard,
        })
    }
}

#[derive(Debug)]
pub(crate) struct Value<'a> {
    value: u64,
    _entry: ReadEntry<'a>,
}

impl Value<'_> {
    pub(crate) fn get(&self) -> u64 {
        self.value
    }
}

#[derive(Debug)]
pub(crate) struct Reader<'a> {
    parent: &'a Checked,
    _guard: epoch::Guard<'a>,
}

impl Reader<'_> {
    pub(crate) fn read(&self, i: u32) -> Option<Value<'_>> {
        if let Some(entry) = self.parent.entries.get(i.into_usize())?.try_read() {
            let value = match entry.state() {
                State::Frozen { value } | State::Readable { value } => value,
                State::Available => panic!("concurrency violation"),
            };

            Some(Value {
                value: *value,
                _entry: entry,
            })
        } else {
            None
        }
    }
}

impl plugin::Plugin for Checked {
    type Slot<'a> = Slot<'a>;

    fn id_limit(&self) -> IdLimit {
        <Checked>::id_limit(self)
    }

    unsafe fn acquire(&self, i: u32, _: Lifecycle) -> Self::Slot<'_> {
        Slot::new(self.entries[i.into_usize()].expect_write())
    }

    unsafe fn retire(&self, i: u32, _: Lifecycle) {
        self.entries[i.into_usize()].expect_read().retire();
    }

    unsafe fn reclaim(&self, i: u32, _: Lifecycle) {
        self.entries[i.into_usize()].expect_write().reclaim();
    }
}

#[derive(Debug)]
pub(crate) struct Slot<'a> {
    entry: WriteEntry<'a>,
    value: Option<u64>,
}

impl<'a> Slot<'a> {
    fn new(entry: WriteEntry<'a>) -> Self {
        Self {
            entry,
            value: None,
        }
    }

    pub(crate) fn set(&mut self, value: u64) {
        self.value = Some(value)
    }
}

impl plugin::Slot for Slot<'_> {
    fn publish(self, _: Lifecycle) {
        let value = self.value.expect("`value` was not set");
        self.entry.publish(value);
    }

    fn freeze(self, _: Lifecycle) {
        let value = self.value.expect("`value` was not set");
        self.entry.freeze(value);
    }

    fn abort(self, _: Lifecycle) {
        assert_matches!(self.entry.state(), State::Available);
    }
}
