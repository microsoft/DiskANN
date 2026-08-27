/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # Pedantically testing the EBR protocol
//!
//! The goal here is to detect violations of the state machine outlined in [`slots`].
//! This is accomplished by using a [`RwLock`] to protect internal [`State`] with guards
//! **only** acquired via [`RwLock::try_read`] and [`RwLock::try_write`]. A correct EBR
//! protocol should ensure that:
//!
//! 1. States where exclusive access is needed have no readers or other concurrent exclusive
//!    accesses.
//!
//! 2. States where read access is allowed have no attempts at exclusive access.
//!
//! The "try" interfaces provide us with these checks:
//!
//! 1. [`RwLock::try_write`] will fail if there is any concurrent reader or writer.
//! 2. [`RwLock::try_read`] will fail if there is a writer.
//!
//! Importantly, we *only* use these non-blocking APIs.
//!
//! Reads from the store return a [`Value`], which contains a [`RwLockReadGuard`] for the
//! corresponding slot. Many such guards can coexist. Providing long-lived guards like this
//! improves our chances of catching a bug in the EBR scheme where exclusive access is
//! attempted too early. [`Reader::read`] constrains each [`Value`] to the borrow of the
//! [`Reader`]. Since [`Reader`]s own an [`epoch::Guard`], [`Value`]s are guaranteed to be
//! dropped before their protecting [`epoch::Guard`] is dropped.
//!
//! ## Lifecycle Details
//!
//! Lifecycle transitions are carefully designed to allow readers of the slots to avoid any
//! accesses to the authoritative [`Store`] for read-only operations. The [`Checked`] test
//! code follows this pattern, but this does introduce a subtle detail that is worth
//! highlighting. [`Reader::read`] needs to be able to check a slot for readability
//! **without** trying to acquire a [`RwLockReadGuard`] for that slot. Doing so even briefly
//! will cause a [`RwLock::try_write`] on an otherwise correct [`slots::Slots`] state
//! transition to fail.
//!
//! To circumvent this, an additional [`AtomicBool`] is bundled with [`Entry`] to broadcast
//! readability. This must be checked first on an optimistic read before attempting to
//! acquire the [`RwLockReadGuard`]. This boolean flag is toggled on the following transitions:
//!
//! * "slot" -> "published"/"frozen": Publish as readable. Note that this transition goes
//!   from an "exclusive" owned (with a [`RwLockWriteGuard`]) to a shared readable state.
//!
//!   As such, it's necessary to release the guard before toggling the "readable" flag to
//!   ensure that if [`Reader::read`] observes the "published" state it is guaranteed to
//!   succeed in [`RwLockReadGuard`] acquisition.
//!
//! * "published" -> "retiring": This switches to non-readable. Note that there is a race
//!   condition where:
//!
//!   - (Thread A) [`Reader::read`] observes "readable" and decides to acquire the read guard.
//!   - (Thread B) Toggles the "readable" flag.
//!   - (Thread A) Finishes acquiring the read guard, even though the "readable" state is no
//!     longer broadcasted.
//!
//!   This is **exactly** what the EBR protocol makes safe. These races are perfectly fine
//!   because the protected [`State::Published`] value remains valid for existing readers
//!   while the lifecycle state is "retiring". The EBR protocol ensures that a slot does
//!   **not** transition away from "retiring" until thread `A` (and all other concurrent
//!   threads that could have observed the slot while it was published) have dropped their
//!   read guards, and thus any data accessed while under the guard, since these are
//!   outlived by a proper [`epoch::Guard`].

#![expect(
    clippy::panic,
    clippy::expect_used,
    reason = "integration-test code is not production code"
)]

use std::{
    assert_matches,
    sync::atomic::{AtomicBool, Ordering},
};

use diskann::utils::IntoUsize;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::{epoch, num::IdLimit, store::Store};

use super::{Lifecycle, slots};

/// The state of a slot.
#[derive(Debug, Default)]
enum State {
    #[default]
    Available,
    Published {
        value: u64,
    },
    Frozen {
        value: u64,
    },
}

/// A slot entry. See the [module level docs](self) for a discussion on the contents of this
/// struct. The table below describes how the combination of fields in this struct maps to
/// the lifecycle states.
/// ```text
/// +-----------------+----------+-----------+------------------------+
/// | Lifecycle state | readable | State     | Lock Expectation       |
/// +=================+==========+===========+========================+
/// |       Available |  false   | Available | unlocked               |
/// +-----------------+----------+-----------+------------------------+
/// |            Slot |  false   | Available | write-locked           |
/// +-----------------+----------+-----------+------------------------+
/// |       Published |  true    | Published | shared reads allowed   |
/// +-----------------+----------+-----------+------------------------+
/// |        Retiring |  false   | Published | existing reads allowed |
/// +-----------------+----------+-----------+------------------------+
/// |          Frozen |  true    | Frozen    | shared reads allowed   |
/// +-----------------+----------+-----------+------------------------+
/// ```
#[derive(Debug, Default)]
struct Entry {
    readable: AtomicBool,
    state: RwLock<State>,
}

impl Entry {
    /// Return whether or not this [`Entry`] is broadcasted as readable.
    #[must_use]
    fn is_readable(&self) -> bool {
        self.readable.load(Ordering::Acquire)
    }

    /// Attempt to acquire a [`ReadEntry`], failing if the [`Entry`] is not marked as
    /// readable without attempting to acquire any locks.
    fn try_read(&self) -> Option<ReadEntry<'_>> {
        if self.is_readable() {
            // We are relying on EBR to avoid problems with the TOCTOU/ABA race between
            // checking the `bool` and attempting to acquire the read guard.
            //
            // EBR prevents the slot from being reclaimed and reused during the race interval.
            //
            // See the module level docs.
            Some(self.expect_read())
        } else {
            None
        }
    }

    /// Acquire a [`ReadEntry`].
    ///
    /// # Panics
    ///
    /// Panics if the [`RwLockReadGuard`] cannot be immediately acquired.
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

    /// Acquire a [`WriteEntry`].
    ///
    /// # Panics
    ///
    /// Panics if the [`RwLockWriteGuard`] cannot be immediately acquired.
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

/// A readable version of [`Entry`].
#[derive(Debug)]
struct ReadEntry<'a> {
    readable: &'a AtomicBool,
    guard: RwLockReadGuard<'a, State>,
}

impl ReadEntry<'_> {
    /// Mark this slot as "retired".
    fn retire(self) {
        assert_matches!(
            *self.guard,
            State::Published { .. },
            "\"retire\" should transition out of the \"published\" state",
        );

        // The ordering of dropping the guard vs clearing the "readable" flag doesn't
        // really matter because the EBR protocol ensures that final reclamation is
        // sufficiently deferred.
        let old = self.readable.swap(false, Ordering::Release);
        assert!(
            old,
            "\"retire\" should transition out of the \"published\" state"
        );
    }

    fn state(&self) -> &State {
        &self.guard
    }
}

/// A writable version of [`Entry`].
#[derive(Debug)]
struct WriteEntry<'a> {
    readable: &'a AtomicBool,
    guard: RwLockWriteGuard<'a, State>,
}

impl WriteEntry<'_> {
    /// Transition this slot to "published".
    fn publish(mut self, value: u64) {
        let old = self.replace(State::Published { value });
        assert_matches!(
            old,
            State::Available,
            "\"publish\" must transition out of the \"available\" state",
        );

        // The ordering here matters. `Reader` must be guaranteed to acquire the read guard
        // if it observes `readable = true`. We must drop the guard before broadcasting.
        drop(self.guard);
        self.readable.store(true, Ordering::Release);
    }

    /// Transition this slot to "frozen".
    fn freeze(mut self, value: u64) {
        let old = self.replace(State::Frozen { value });
        assert_matches!(
            old,
            State::Available,
            "\"freeze\" must transition out of the \"available\" state",
        );

        // See `Self::publish` for ordering details.
        drop(self.guard);
        self.readable.store(true, Ordering::Release);
    }

    /// Transition this slot to "available".
    fn reclaim(mut self) {
        let old = self.replace(State::Available);

        // Note: The combination of `!readable` and `State::Published` implies "retiring".
        // We check that here.
        assert!(
            !self.readable.load(Ordering::Relaxed),
            "\"reclaim\" must transition out of \"retired\"",
        );

        assert_matches!(
            old,
            State::Published { .. },
            "\"reclaim\" must transition out of \"retired\"",
        );
    }

    /// Replace the protected state with `state`, returning the old state.
    fn replace(&mut self, mut state: State) -> State {
        std::mem::swap(&mut *self.guard, &mut state);
        state
    }

    fn state(&self) -> &State {
        &self.guard
    }
}

/// A [`slots::SlotsConfig`] for [`Checked`].
#[derive(Debug)]
pub(crate) struct Config(());

impl Config {
    pub(crate) fn new() -> Self {
        Self(())
    }
}

impl slots::SlotsConfig for Config {
    type Slots = Checked;
    type Error = diskann::error::Infallible;

    fn build(self, id_limit: IdLimit) -> Result<Checked, diskann::error::Infallible> {
        Ok(Checked::new(id_limit))
    }
}

/// A correctness checking [`slots::Slots`]. See the [module level docs](self) for details.
#[derive(Debug)]
pub(crate) struct Checked {
    entries: Vec<Entry>,
}

impl Checked {
    /// Create a new [`Checked`] with `id_limit` slots.
    pub(crate) fn new(id_limit: IdLimit) -> Self {
        Self {
            entries: std::iter::repeat_with(Entry::default)
                .take(id_limit.as_usize())
                .collect(),
        }
    }

    /// Return the [`slots::SlotsConfig`] for [`Self`].
    pub(crate) fn config() -> Config {
        Config::new()
    }

    /// Return the [`IdLimit`] for this store.
    pub(crate) fn id_limit(&self) -> IdLimit {
        IdLimit::new(self.entries.len() as u32)
    }

    /// Return an epoch-protected [`Reader`] into [`Self`].
    pub(crate) fn reader(store: &Store<Self>) -> Result<Reader<'_>, epoch::Unavailable> {
        store.guard(|this, guard: epoch::Guard<'_>| Reader {
            parent: this,
            _guard: guard,
        })
    }
}

/// A valid, readable entry of [`Checked`].
#[derive(Debug)]
pub(crate) struct Value<'a> {
    value: u64,
    _entry: ReadEntry<'a>,
}

impl Value<'_> {
    /// Return the payload for this [`Value`].
    pub(crate) fn get(&self) -> u64 {
        self.value
    }
}

/// A reader for [`Checked`].
#[derive(Debug)]
pub(crate) struct Reader<'a> {
    parent: &'a Checked,
    _guard: epoch::Guard<'a>,
}

impl Reader<'_> {
    /// Attempt to read the value at slot `i`.
    ///
    /// Fails if `i` is out-of-bounds, or the slot is not in a readable state.
    pub(crate) fn read(&self, i: u32) -> Option<Value<'_>> {
        if let Some(entry) = self.parent.entries.get(i.into_usize())?.try_read() {
            let value = match entry.state() {
                State::Frozen { value } | State::Published { value } => value,
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

impl slots::Slots for Checked {
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

/// A writable [`slots::Slot`] for [`Checked`].
#[derive(Debug)]
pub(crate) struct Slot<'a> {
    entry: WriteEntry<'a>,
    value: Option<u64>,
}

impl<'a> Slot<'a> {
    fn new(entry: WriteEntry<'a>) -> Self {
        Self { entry, value: None }
    }

    /// Write `value` into the slot on a successful state transition.
    pub(crate) fn set(&mut self, value: u64) {
        self.value = Some(value)
    }
}

impl slots::Slot for Slot<'_> {
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
