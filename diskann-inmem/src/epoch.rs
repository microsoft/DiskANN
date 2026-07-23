/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The core logic for the epoch-based reclamation algorithm.
//!
//! ## What Problem is Being Solved?
//!
//! Epoch-based reclamation (EBR) can be used to safely implement read-heavy algorithms with
//! a moderate level of concurrent writes. In this context, we want readers to be able to ask
//! the question: "can I safely read some data" in a way that generates only read traffic to
//! the CPU caches.
//!
//! The crux is that after the safety check, a reader can hold a reference to the associated
//! data for an arbitrary period of time. Any actor trying to *write* to that data needs
//! to figure out when it is safe to do so.
//!
//! EBR solves this problem by separating when data is "retired" versus "reclaimed".
//! Retirement involves disabling the safety check. When an item is retired, concurrent
//! readers will fail the safety check and no longer try to read the associated data.
//! However, we still need to wait until we can prove that readers who passed the safety
//! check before retirement are no longer accessing the data. At this point, the data can be
//! "reclaimed" and written to safely.
//!
//! We can prove this by using a monotonically increasing epoch: if an item was "retired"
//! at epoch `N` its associated data could be in use by any reader belonging to any epoch
//! `N` or lower. Therefore, it is only safe to "reclaim" when all readers belong to epoch
//! `N+1` or higher.
//!
//! One consequence of this design is that misbehaving (e.g. long-lived) readers can delay
//! reclamation indefinitely. As such, this system must be used with care and in situations
//! where there is enough slack in the system to accommodate the lifetime of any readers.
//!
//! ## Primitives
//!
//! Actors call [`Registry::guard`] to receive a [`Guard`]. This guard protects items
//! at its creation epoch. Any items pushed to [`Guard::retire`] will be buffered until the
//! [`Registry`] can prove that all [`Guard`]s (correctly using the data structure) that
//! could have observed the retired item have been destroyed.
//!
//! Items can be reclaimed via [`Registry::try_advance`]. If successful, a [`Drain`] of
//! such items will be returned for processing.
//!
//! Note that retired payloads are fixed to `u32` ids (typically interpreted by the caller
//! as indices into some external storage); this is not a general-purpose deferred-drop EBR
//! system.

use std::{
    num::NonZeroUsize,
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
};

use crossbeam_queue::SegQueue;
use diskann::utils::IntoUsize;
use parking_lot::{Mutex, MutexGuard};

const DEFAULT_GUARD_SLOTS: NonZeroUsize = NonZeroUsize::new(256).unwrap();

/// A registry of epoch-based [`Guard`]s. See the [module-level docs](self).
#[derive(Debug)]
pub(crate) struct Registry {
    // A record of the active guards.
    //
    // * 0 = "available".
    // * Anything less = "guarded".
    guards: Box<[AtomicU64]>,

    // A hint for the next available registration slot.
    hint: AtomicUsize,

    // The current epoch. This begins at 1 (to not be conflated with the 0 state in `guards`)
    // and increments over time.
    //
    // NOTE: This can **only** be mutated in `try_advance`.
    //
    // Additionally, the logic in the module ensures that there are at most two active epochs
    // at any given time. It is only safe to advance an epoch if *all* readers belong to the
    // current `epoch`.
    epoch: AtomicU64,

    // We can only retire a single generation at a time.
    //
    // This guard avoids situations where two threads concurrently advance the epoch and
    // hand out overlapping `Drain`s referring to the same retiring queue.
    drain: Mutex<()>,

    // We use four queues for storing retiring items. The rationale is documented below.
    //
    // ```
    //
    //                                1. Safe to drain
    //                           +--------------------------
    //  Items retired at N-1 can |    2. Epoch N-1
    //  be observed by guards at |  +-----------------------
    //  N. If we transition to   |  | 3. Epoch N
    //  N+1, guards at N can be  +--------------------------
    //  active still. This it is    | 4. Epoch N+1
    //  not safe to reclaim items   +-----------------------
    //  from this queue until all     5. Epoch N+2 (reuse #1 queue)
    //  guards are at least N+1.
    // ```
    //
    // We cycle among the queues in a round-robin manner.
    retiring: Box<[SegQueue<u32>; 4]>,
}

// Return the queue index for the `epoch`.
fn queue(epoch: u64) -> usize {
    epoch.into_usize() % 4
}

fn last_queue(epoch: u64) -> usize {
    queue(epoch.wrapping_sub(2))
}

impl Registry {
    /// Return the default number of guard slots.
    pub(crate) const fn default_guard_slots() -> NonZeroUsize {
        DEFAULT_GUARD_SLOTS
    }

    /// Construct a new [`Registry`] with the default number of guard slots (256).
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self::with_capacity(DEFAULT_GUARD_SLOTS)
    }

    /// Construct a new [`Registry`] with `capacity` guard slots.
    ///
    /// This is the number of [`Guard`]s that can be registered concurrently.
    pub(crate) fn with_capacity(capacity: NonZeroUsize) -> Self {
        Self {
            guards: std::iter::repeat_with(|| AtomicU64::new(0))
                .take(capacity.get())
                .collect(),
            hint: AtomicUsize::new(0),
            epoch: AtomicU64::new(1),
            retiring: Box::new(core::array::from_fn(|_| SegQueue::new())),
            drain: Mutex::new(()),
        }
    }

    /// Return the current epoch.
    ///
    /// This has [`Ordering::Acquire`] semantics.
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Register the caller with `self`.
    ///
    /// Any items retired while [`Guard`] is held will be protected.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of currently active guards exceeds the number of
    /// internal guard slots and thus a new guard cannot be made.
    pub(crate) fn guard(&self) -> Result<Guard<'_>, Unavailable> {
        self.guard_inner(NoDelay)
    }

    #[inline]
    fn guard_inner<T>(&self, mut delay: T) -> Result<Guard<'_>, Unavailable>
    where
        T: GuardDelay,
    {
        // GUARD CHECK
        let mut epoch = self.epoch();
        let hint = self.hint.fetch_add(1, Ordering::Relaxed);
        delay.post_guard_check();
        let nguards = self.guards.len();
        for i in 0..nguards {
            let slot = hint.wrapping_add(i) % nguards;

            let guard_slot = &self.guards[slot];
            delay.pre_cas();
            if guard_slot
                .compare_exchange(0, epoch, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                delay.post_cas();
                let mut reset = false;
                loop {
                    // GUARD FENCE: This fence is paired with "WAITING FENCE".
                    //
                    // See that comment for details.
                    delay.pre_fence();
                    std::sync::atomic::fence(Ordering::SeqCst);
                    delay.post_fence();

                    // GUARD RECHECK
                    let current = self.epoch();
                    if current == epoch {
                        break;
                    }

                    reset = true;
                    epoch = current;
                }

                if reset {
                    guard_slot.store(epoch, Ordering::Relaxed);
                }

                return Ok(Guard {
                    slot: guard_slot,
                    retire: &self.retiring[queue(epoch)],
                    #[cfg(test)]
                    epoch,
                    #[cfg(test)]
                    slot_index: slot,
                });
            }
        }

        Err(Unavailable)
    }

    fn can_advance<T>(&self, delay: &mut T) -> (bool, u64)
    where
        T: CanAdvanceDelay,
    {
        // WAITING FENCE: This is a very important part for the correctness of the algorithm.
        //
        // What we're protecting against is a scenario where "registering" thread A reads an
        // epoch, then "waiting" thread B does a scan, thinks everything is safe, and then
        // thread A finishes its CAS for its registration.
        //
        // This is prevented by the sequentially consistent fences. Consider the following.
        //
        // 1. Thread A invokes "GUARD FENCE" after a successful CAS, and then checks the
        //    generation at "GUARD RECHECK".
        //
        // 2. Thread B now enters the this block of code, executes "WAITING FENCE", then
        //    reads the epoch tags for all guards.
        //
        // With the total order induced by the sequentially consistency, either thread A's
        // fence executes first, or thread B's executes first.
        //
        // * If thread A's fence executes first, then thread B will see the CAS and the set
        //   value is guaranteed to be less-than or equal to "WAITING CHECK" because:
        //
        //   1. The epoch is monotonically increasing.
        //   2. Writes to the epoch are also sequentially consistent.
        //
        // * If Thread B's fence executes first, then thread A's "GUARD RECHECK" will
        //   observe at least the result of "WAITING CHECK" and update itself on the retry.
        //
        //   It's possible that thread B observes the CAS to "GUARD CHECK", but since
        //   thread A will monotonically increase it before exiting, the value thread B
        //   observes is conservative and not incorrect.
        delay.pre_fence();
        std::sync::atomic::fence(Ordering::SeqCst);
        delay.post_fence();

        // WAITING CHECK
        let current = self.epoch();
        let mut min = current;

        for s in self.guards.iter() {
            let guarded = s.load(Ordering::Relaxed);
            if guarded != 0 {
                min = min.min(guarded);
            }
        }

        // This synchronizes with all the guard's `Release`s.
        std::sync::atomic::fence(Ordering::Acquire);
        (min == current, min)
    }

    /// Try to advance the current epoch.
    ///
    /// If successful, returns a [`Drain`]. All items in the drain can be reclaimed.
    ///
    /// Returns `None` if the epoch cannot yet be advanced (some [`Guard`] still belongs to
    /// a prior epoch) or if another [`Drain`] is currently active.
    ///
    /// # Panics
    ///
    /// Panics if the epoch counter is about to overflow `u64::MAX`. In practice this is
    /// effectively unreachable.
    pub(crate) fn try_advance(&self) -> Option<Drain<'_>> {
        self.try_advance_inner(NoDelay)
    }

    #[expect(
        clippy::panic,
        reason = "the panic is exceedingly unlikely to happen and if it does, we can't continue"
    )]
    fn try_advance_inner<T>(&self, mut delay: T) -> Option<Drain<'_>>
    where
        T: TryAdvanceDelay,
    {
        // We first try to acquire the `drain` lock.
        //
        // It can only fail if someone else is holding the drain lock, which means we can't
        // proceed anyways.
        //
        // This can help save an expensive slot scan.
        let drain = self.drain.try_lock()?;

        let (can_advance, current) = self.can_advance(&mut delay);

        // Don't wrap around!
        if current == u64::MAX {
            panic!(
                "we've managed to go through nearly `u64::MAX` ids - this is unlikely in a real program"
            );
        }

        // All waiters belong to the current epoch. Therefore, it is safe to release the old
        // array queue
        if can_advance {
            // We are safe to use a `fetch_add` here because `drain` is ensuring exclusivity
            // of the access.
            //
            // However, this still needs to be `SeqCst` so that this properly synchronizes
            // with "GUARD FENCE" and "WAITING FENCE".
            let _previous = self.epoch.fetch_add(1, Ordering::SeqCst);
            debug_assert_eq!(_previous, current, "concurrency violation");

            let queue = &self.retiring[last_queue(current)];
            Some(Drain {
                queue,
                _drain: drain,
            })
        } else {
            // Previous generation has not completely retired.
            None
        }
    }

    #[cfg(test)]
    fn assert_no_workers(&self) {
        for s in self.guards.iter() {
            assert_eq!(s.load(Ordering::Relaxed), 0);
        }
    }

    #[cfg(test)]
    fn waiting(&self) -> u64 {
        self.can_advance(&mut NoDelay).1
    }
}

/// A handle registering the caller as a reader at a particular epoch.
///
/// While this guard is held, the [`Registry`] will not advance past the guard's epoch, and
/// any items retired through *any* guard at that epoch (or earlier) will not be reclaimed.
///
/// Obtained via [`Registry::guard`].
#[derive(Debug)]
pub(crate) struct Guard<'a> {
    slot: &'a AtomicU64,
    retire: &'a SegQueue<u32>,

    #[cfg(test)]
    pub(super) epoch: u64,

    #[cfg(test)]
    slot_index: usize,
}

impl Guard<'_> {
    /// Retire the id `i` at this guard's epoch.
    ///
    /// `i` is a caller-defined id (typically an index into external storage). It will be
    /// returned from a future [`Drain`] once the registry has advanced far enough that no
    /// reader could observe it.
    #[inline]
    pub(crate) fn retire(&self, i: u32) {
        self.retire.push(i)
    }
}

impl Drop for Guard<'_> {
    fn drop(&mut self) {
        self.slot.store(0, Ordering::Release);
    }
}

/// An iterator over ids that are safe to reclaim, returned from [`Registry::try_advance`].
///
/// While this drain is alive, no other thread can advance the [`Registry`]'s epoch. Drop
/// it promptly after processing.
#[derive(Debug)]
pub(crate) struct Drain<'a> {
    queue: &'a SegQueue<u32>,
    _drain: MutexGuard<'a, ()>,
}

impl Drain<'_> {
    /// Pop the next id ready for reclamation, or `None` if the drain is empty.
    #[must_use = "reclaimed ids must be reclaimed"]
    pub(crate) fn pop(&self) -> Option<u32> {
        self.queue.pop()
    }

    /// Return the number of ids remaining in this drain.
    pub(crate) fn len(&self) -> usize {
        self.queue.len()
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Iterator for Drain<'_> {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        self.pop()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

// NOTE: This relies on `Drain` holding the `drain` guard. In this state, we are guaranteed
// that no-one is writing into the queue, which would otherwise invalidate the exact-size
// iterator guarantee.
impl ExactSizeIterator for Drain<'_> {}

/// Returned by [`Registry::guard`] when all guard slots are occupied.
#[derive(Debug)]
#[non_exhaustive]
pub(crate) struct Unavailable;

impl std::fmt::Display for Unavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("all available registry guard slots are occupied")
    }
}

impl std::error::Error for Unavailable {}

crate::opaque!(Unavailable);

// Delays
//
// To help test standard race scenarios without advanced tooling, we use optional delays
// that our tests can introduce to ensure threads are in various intermediate points.
//
// This does not necessarily test that the memory orderings are correct, but at least
// is a smoke test that various (known) races are handled properly.

#[derive(Debug)]
struct NoDelay;

trait GuardDelay {
    fn post_guard_check(&mut self) {}
    fn pre_cas(&mut self) {}
    fn post_cas(&mut self) {}
    fn pre_fence(&mut self) {}
    fn post_fence(&mut self) {}
}

impl GuardDelay for NoDelay {}

trait CanAdvanceDelay {
    fn pre_fence(&mut self) {}
    fn post_fence(&mut self) {}
}

impl CanAdvanceDelay for NoDelay {}

trait TryAdvanceDelay: CanAdvanceDelay {}

impl TryAdvanceDelay for NoDelay {}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test::Sequencer;

    // This test ensures that two threads racing on `hint` will correctly resolve themselves
    // when claiming a slot.
    #[test]
    fn test_cas_race() {
        let seq = Sequencer::new();

        let mut thread_a_loop_count = 0;
        let mut thread_b_loop_count = 0;
        let delay = TestGuardDelay::default()
            .post_guard_check(|| seq.wait_for(0))
            .with_post_fence(|| thread_a_loop_count += 1);

        let registry = Registry::with_capacity(NonZeroUsize::new(2).unwrap());
        std::thread::scope(|s| {
            // Thread A
            s.spawn(|| {
                let g = registry.guard_inner(delay).unwrap();
                assert_eq!(g.slot_index, 1);
                seq.wait_for(1);
            });

            // Thread B
            s.spawn(|| {
                // wait for Thread A to reach the delay point.
                seq.until_waiting_for(0);
                {
                    let delay =
                        TestGuardDelay::default().with_post_fence(|| thread_b_loop_count += 1);
                    let g = registry.guard_inner(delay).unwrap();
                    assert_eq!(g.slot_index, 1);
                }
                let g = registry.guard_inner(NoDelay).unwrap();
                assert_eq!(g.slot_index, 0);
                seq.advance_past(0);
                seq.advance_past(1);
            });
        });

        assert_eq!(thread_a_loop_count, 1);
        assert_eq!(thread_b_loop_count, 1);

        registry.assert_no_workers();
    }

    #[test]
    fn test_register_wait() {
        // This tests the case where a thread enters registration, reads a generation, then
        // sleeps for several generation advances. It ensures that the thread recovers properly.
        let seq = Sequencer::new();

        let mut loop_count = 0;
        let delay = TestGuardDelay::default()
            .post_guard_check(|| seq.wait_for(0))
            .with_post_cas(|| seq.wait_for(1))
            .with_pre_fence(|| loop_count += 1);

        let registry = Registry::with_capacity(NonZeroUsize::new(2).unwrap());

        std::thread::scope(|s| {
            let handle = s.spawn(|| {
                let guard = registry.guard_inner(delay).unwrap();

                // Since we hit the CAS loop - this serves as a sanity check that we have
                // the correct drain buffer.
                guard.retire(10);
                guard.retire(1);
                guard.retire(2);
                guard.retire(3);
                guard
            });

            // Wait for the spawned thread to reach the critical section.
            seq.until_waiting_for(0);

            assert_eq!(registry.waiting(), 1);
            {
                let drain = registry.try_advance().unwrap();
                assert!(drain.is_empty());
                assert_eq!(registry.epoch(), 2);
            }

            {
                let drain = registry.try_advance().unwrap();
                assert!(drain.is_empty());
                assert_eq!(registry.epoch(), 3);
            }

            // We allow the registering thread to make it past the CAS.
            //
            // We pause it again because we want to verify that it registers an old generation.
            seq.advance_past(0);
            seq.until_waiting_for(1);
            let (can_advance, waiter) = registry.can_advance(&mut NoDelay);
            assert!(!can_advance);
            assert_eq!(
                waiter, 1,
                "waiting thread registers an older generation before observing the change"
            );
            seq.advance_past(1);

            let expected = 3;

            // The generation should be the last set one - even though this thread was
            // parked during the transition.
            let r = handle.join().unwrap();
            assert_eq!(r.epoch, expected);
            assert_eq!(registry.waiting(), expected);
        });

        assert_eq!(
            loop_count, 2,
            "the registering thread should have looped to update its generation"
        );

        registry.assert_no_workers();

        // Verify that we reclaim the ID flushed by the registering thread.
        //
        // This requires three epoch advancements.
        {
            let drain = registry.try_advance().unwrap();
            assert!(drain.is_empty());
        }

        {
            let drain = registry.try_advance().unwrap();
            assert!(drain.is_empty());
        }

        {
            let drain = registry.try_advance().unwrap();
            let ids: Vec<_> = drain.collect();
            assert_eq!(ids, &[10, 1, 2, 3]);
        }
    }

    // Verifies that filling every slot causes `register` to return `Unavailable`, and that
    // dropping an existing guard frees up its slot for a subsequent registration.
    #[test]
    fn test_slot_exhaustion() {
        let registry = Registry::with_capacity(NonZeroUsize::new(2).unwrap());

        let g0 = registry.guard().unwrap();
        let g1 = registry.guard().unwrap();

        // All guard slots are now occupied. The next registration must fail.
        assert!(matches!(registry.guard(), Err(Unavailable)));
        assert!(matches!(registry.guard(), Err(Unavailable)));

        // Dropping a guard releases its slot.
        let freed_slot = g0.slot_index;
        drop(g0);

        let g2 = registry.guard().unwrap();
        assert_eq!(
            g2.slot_index, freed_slot,
            "newly freed slot should be reclaimed"
        );

        // Registry is full again.
        assert!(matches!(registry.guard(), Err(Unavailable)));

        drop(g1);
        drop(g2);

        registry.assert_no_workers();
    }

    #[test]
    fn test_slot_wrap_around() {
        let registry = Registry::with_capacity(NonZeroUsize::new(4).unwrap());

        let (g2, g3) = {
            let _g0 = registry.guard().unwrap();
            let _g1 = registry.guard().unwrap();

            let g2 = registry.guard().unwrap();
            let g3 = registry.guard().unwrap();
            (g2, g3)
        };

        assert_eq!(g2.slot_index, 2);
        assert_eq!(g3.slot_index, 3);

        let f = || {
            // Keep wrapping and hitting the first two guard slots.
            for _ in 0..10 {
                let g0 = registry.guard().unwrap();
                let g1 = registry.guard().unwrap();

                let s0 = g0.slot_index;
                let s1 = g1.slot_index;

                // Due to how the hint works, the slots could be acquired in either order.
                if s0 < s1 {
                    assert_eq!((s0, s1), (0, 1));
                } else {
                    assert_eq!((s0, s1), (1, 0));
                };

                assert!(matches!(registry.guard(), Err(Unavailable)));
            }
        };

        // Run with the default hint.
        f();

        // Set the hint to `usize::MAX`.
        registry.hint.store(usize::MAX - 10, Ordering::Relaxed);

        // Run tests again to ensure we can properly handle wrap-around.
        f();

        drop((g2, g3));
        registry.assert_no_workers();
    }

    // Verifies that `try_advance` short-circuits to `None` when another thread already holds
    // the `drain` mutex, even if `can_advance` would otherwise succeed. This guards the
    // early `try_lock` that avoids a redundant slot scan.
    #[test]
    fn test_concurrent_try_advance() {
        let registry = Registry::with_capacity(NonZeroUsize::new(2).unwrap());

        // No outstanding registrations, so `can_advance` would succeed for any caller.
        let drain = registry
            .try_advance()
            .expect("first try_advance must succeed");
        let gen_after_first = registry.epoch();
        assert_eq!(gen_after_first, 2);

        // While the first `Drain` is alive (holding the drain mutex), a concurrent
        // `try_advance` must return `None` without advancing the generation.
        std::thread::scope(|s| {
            s.spawn(|| {
                assert!(
                    registry.try_advance().is_none(),
                    "try_advance must fail while another holds the drain mutex"
                );
                assert_eq!(
                    registry.epoch(),
                    gen_after_first,
                    "generation must not advance when drain is contended"
                );
            });
        });

        // Releasing the drain unblocks subsequent advances.
        drop(drain);

        let _drain2 = registry
            .try_advance()
            .expect("try_advance must succeed once drain is released");
        assert_eq!(registry.epoch(), 3);
    }

    // Verifies the 3-queue rotation invariant: items retired at generation `G` are drained
    // on the second `try_advance` after `G`. The first advance returns the queue from
    // `(G - 1) % 3` (one cycle older), so it must NOT contain items from `G`.
    #[test]
    fn test_drain_rotation() {
        let registry = Registry::with_capacity(NonZeroUsize::new(1).unwrap());

        // Helper: register, retire one item, drop. Returns the generation we retired at.
        let retire_at = |id: u32| {
            let g = registry.guard().unwrap();
            let epoch = g.epoch;
            g.retire(id);
            epoch
        };

        // Retire 100 at generation A (= 1).
        let gen_a = retire_at(100);
        assert_eq!(gen_a, 1);

        // 1st advance after A: must NOT drain item 100.
        {
            let drain = registry.try_advance().unwrap();
            assert!(
                drain.is_empty(),
                "100 must not drain on 1st advance after A"
            );
        }

        // Retire 200 at generation B (= A - 1).
        let gen_b = retire_at(200);
        assert_eq!(gen_b, gen_a + 1);

        // 2nd advance after A: must NOT drain item 100.
        {
            let drain = registry.try_advance().unwrap();
            assert!(
                drain.is_empty(),
                "100 must not drain on 2nd advance after A"
            );
        }

        // Retire 300 at generation C.
        let _gen_c = retire_at(300);

        // 3rd advance after A (1st after B): drains A's queue → [100].
        {
            let drained: Vec<_> = registry.try_advance().unwrap().collect();
            assert_eq!(drained, &[100]);
        }

        // 3rd advance after B: drains B's queue → [200].
        {
            let drained: Vec<_> = registry.try_advance().unwrap().collect();
            assert_eq!(drained, &[200]);
        }

        // 3rd advance after C: drains C's queue → [300].
        {
            let drained: Vec<_> = registry.try_advance().unwrap().collect();
            assert_eq!(drained, &[300]);
        }

        // Rotation has cycled back to where A's queue used to live — must be empty,
        // proving the queue slot was drained cleanly and is reusable.
        {
            let drain = registry.try_advance().unwrap();
            assert!(
                drain.is_empty(),
                "rotation should leave queues empty after one cycle"
            );
        }

        registry.assert_no_workers();
    }

    //-------------//
    // Test Delays //
    //-------------//

    macro_rules! tester {
        ($struct:ident, $trait:ident, $($with:ident => $f:ident),* $(,)?) => {
            #[derive(Default)]
            struct $struct<'a> {
                $($f: Option<Box<dyn FnMut() + Send + 'a>>,)*
            }

            impl<'a> $struct<'a> {
                $(
                    fn $with<F>(mut self, f: F) -> Self
                    where
                        F: FnMut() + Send + 'a
                    {
                        self.$f = Some(Box::new(f));
                        self
                    }
                )*
            }

            impl $trait for $struct<'_> {
                $(
                    fn $f(&mut self) {
                        if let Some(f) = self.$f.as_mut() {
                            f()
                        }
                    }
                )*
            }
        }
    }

    tester! {
        TestGuardDelay,
        GuardDelay,
        post_guard_check => post_guard_check,
        with_post_cas => post_cas,
        with_pre_fence => pre_fence,
        with_post_fence => post_fence,
    }
}
