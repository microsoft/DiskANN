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

use crate::{epoch, num::IdLimit};

use super::{Lifecycle, plugin};

#[derive(Debug, Default)]
enum State {
    #[default]
    Available,
    Readable {
        value: u64,
        retired: AtomicBool,
    },
    Frozen {
        value: u64,
    },
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
    states: Vec<RwLock<State>>,
}

impl Checked {
    pub(crate) fn config() -> Config {
        Config::new()
    }

    pub(crate) fn new(id_limit: IdLimit) -> Self {
        Self {
            states: std::iter::repeat_with(|| RwLock::new(State::default()))
                .take(id_limit.as_usize())
                .collect(),
        }
    }

    pub(crate) fn id_limit(&self) -> IdLimit {
        IdLimit::new(self.states.len().try_into().unwrap())
    }

    fn expect_write(&self, i: u32) -> RwLockWriteGuard<'_, State> {
        let i = i.into_usize();

        // Note: this will panic if `i` is out-of-bounds.
        let entry = &self.states[i];

        // Correct usage of the concurrency protocol means that this `try_write` failing
        // is a bug.
        let Some(guard) = entry.try_write() else {
            panic!("concurrency violation when acquiring write guard");
        };

        guard
    }

    fn expect_read(&self, i: u32) -> RwLockReadGuard<'_, State> {
        let i = i.into_usize();

        // Note: this will panic if `i` is out-of-bounds.
        let entry = &self.states[i];

        // Correct usage of the concurrency protocol means that this `try_read` failing
        // is a bug.
        let Some(guard) = entry.try_read() else {
            panic!("concurrency violation when acquiring read guard");
        };

        guard
    }

    pub(crate) fn reader<'a>(&'a self, guard: epoch::Guard<'a>) -> Reader<'a> {
        Reader {
            parent: self,
            _guard: guard,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Value<'a> {
    value: u64,
    _guard: RwLockReadGuard<'a, State>,
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
        // This is kind of messy. The overall summary is this:
        //
        // 1. `i` has to be inbounds.
        // 2. We have to be able to read the slot (if a write guard is active, then we
        //    clearly should not be reading).
        // 3a. If the state is frozen, then we can read it.
        // 3b. If the state is readable and not retired, we can read it.
        //
        //     What happens if we transition to retired just after reading?
        //
        //     Fortunately, the EBR guard will keep the slot from being reclaimed until
        //     the current `Reader` goes out-of-scope. Holding onto the
        //     `RwLockReadGuard` allows us to detect bugs in the EBR protocol as
        //     `Checked::expect_write` will fail on reclamation if a returned `Value`
        //     is still active.
        if let Some(state) = self.parent.states.get(i.into_usize())
            && let Some(guard) = state.try_read()
        {
            let value = match &*guard {
                State::Frozen { value } => *value,
                State::Readable { value, retired } => {
                    if retired.load(Ordering::Relaxed) {
                        return None;
                    }
                    *value
                }
                _ => return None,
            };
            Some(Value {
                value,
                _guard: guard,
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
        let guard = self.expect_write(i);
        assert_matches!(*guard, State::Available, "slot is in an invalid state");
        Slot::new(guard)
    }

    unsafe fn retire(&self, i: u32, _: Lifecycle) {
        let guard = self.expect_read(i);
        match &*guard {
            State::Available => panic!("invalid \"Available\" state"),
            State::Readable { retired, .. } => {
                let old = retired.swap(true, Ordering::Relaxed);
                if old {
                    panic!("slot {i} was retired multiple times");
                }
            }
            State::Frozen { .. } => panic!("tried to retire frozen point {i}"),
        }
    }

    unsafe fn reclaim(&self, i: u32, _: Lifecycle) {
        let mut guard = self.expect_write(i);
        match &*guard {
            State::Available => panic!("invalid \"Available\" state"),
            State::Readable { retired, .. } => {
                assert!(
                    retired.load(Ordering::Relaxed),
                    "tried to reclaim {i} before it has been retired!",
                );
            }
            State::Frozen { .. } => panic!("tried to reclaim frozen point {i}"),
        }

        *guard = State::Available;
    }
}

#[derive(Debug)]
pub(crate) struct Slot<'a> {
    guard: RwLockWriteGuard<'a, State>,
    value: Option<u64>,
}

impl<'a> Slot<'a> {
    fn new(guard: RwLockWriteGuard<'a, State>) -> Self {
        Self { guard, value: None }
    }

    pub(crate) fn set(&mut self, value: u64) {
        self.value = Some(value)
    }
}

impl plugin::Slot for Slot<'_> {
    fn publish(mut self, _: Lifecycle) {
        let value = self.value.expect("`value` was not set");
        *self.guard = State::Readable {
            value,
            retired: AtomicBool::new(false),
        };
    }

    fn freeze(mut self, _: Lifecycle) {
        let value = self.value.expect("`value` was not set");
        *self.guard = State::Frozen { value };
    }

    fn abort(mut self, _: Lifecycle) {
        *self.guard = State::Available;
    }
}
