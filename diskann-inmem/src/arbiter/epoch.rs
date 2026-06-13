/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_queue::SegQueue;
use diskann::utils::IntoUsize;
use parking_lot::{Mutex, MutexGuard};

use crate::arbiter::{
    Generation,
    generation::{Mut, Tag},
};

const CAPACITY: usize = 256;

#[derive(Debug)]
pub struct Registry {
    /// A record of the active generations.
    ///
    /// * Generation::MAX = "available".
    /// * Anything less = "registered".
    slots: Box<[Tag]>,

    // The current epoch. This begins as `Generation::MAX.sub(1)` and decrements over time.
    //
    // NOTE: This can only be mutated in `try_advance`.
    generation: Tag,

    // A hint for the next available registration slot.
    hint: AtomicUsize,

    // We use three queues for storing slots.
    //
    // 1. Belongs to the current generation and is getting filled.
    // 2. Ready for the next generation that will be populated on the next `try_advance`.
    //    Note that after a `try_advance` call, both 1 and 2 can be added to.
    // 3. The queue returned from `try_advance` to be drained. Items drained are safe to
    //    reclaim.
    retiring: [SegQueue<u32>; 3],

    // We can only retire a single generation at a time.
    // This guard avoids situations.
    drain: Mutex<()>,
}

// Return the queue index for the `generation`.
fn queue(generation: Generation) -> usize {
    generation.value().into_usize() % 3
}

fn last_queue(generation: Generation) -> usize {
    queue(Generation::new(generation.value().wrapping_add(1)))
}

impl Registry {
    pub fn new() -> Self {
        Self::with_capacity(CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: (0..capacity).map(|_| Tag::new(Generation::MAX)).collect(),
            generation: Tag::new(Generation::MAX.sub(1)),
            hint: AtomicUsize::new(0),
            retiring: core::array::from_fn(|_| SegQueue::new()),
            drain: Mutex::new(()),
        }
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Return the current generation.
    ///
    /// This has [`Ordering::Acquire`] semantics.
    pub fn generation(&self) -> Generation {
        self.generation.as_ref().get(Ordering::Acquire)
    }

    /// Register the caller with the registry.
    ///
    /// On success, the returned [`Guard`] will protect items tagged with
    /// [`Guard::generation`] and higher.
    pub fn register(&self) -> Result<Guard<'_>, Unavailable> {
        self.register_inner(NoDelay)
    }

    #[inline]
    fn register_inner<T>(&self, mut delay: T) -> Result<Guard<'_>, Unavailable>
    where
        T: RegisterDelay,
    {
        // REGISTER CHECK
        let mut generation = self.generation();
        let hint = self.hint.fetch_add(1, Ordering::Relaxed);
        delay.post_register_check();
        let nslots = self.slots.len();
        for i in 0..nslots {
            let slot = (hint + i) % nslots;

            let m = self.slots[slot].as_mut();
            delay.pre_cas();
            if let Ok(_) = m.try_set(
                Generation::MAX,
                generation,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                delay.post_cas();
                let mut reset = false;
                loop {
                    // REGISTER FENCE: This fence is paired with "WAITING FENCE".
                    //
                    // See that comment for details.
                    delay.pre_fence();
                    std::sync::atomic::fence(Ordering::SeqCst);
                    delay.post_fence();

                    // REGISTER RECHECK
                    let current = self.generation();
                    if current == generation {
                        break;
                    }

                    reset = true;
                    generation = current;
                }

                if reset {
                    m.set(generation, Ordering::Relaxed);
                }

                return Ok(Guard {
                    slot: m,
                    retire: &self.retiring[queue(generation)],
                    generation,
                    #[cfg(test)]
                    slot_index: slot,
                });
            }
        }

        Err(Unavailable)
    }

    /// Return the oldest generation that is currently being protected.
    ///
    /// This uses a fast method that may be overly conservative.
    ///
    /// This is a synchronizing operation with [`Ordering::Acquire`] semantics.
    pub fn can_advance(&self) -> bool {
        self.can_advance_inner(&mut NoDelay).0
    }

    fn can_advance_inner<T>(&self, delay: &mut T) -> (bool, Generation)
    where
        T: CanAdvanceDelay,
    {
        // WAITING FENCE: This is a very important part for the correctness of the algorithm.
        //
        // What we're protecting against is a scenario where "registering" thread A reads a
        // generation, then "waiting" thread B does a scan, thinks everything is safe, and
        // then thread A finishes its CAS for its registration.
        //
        // This is prevented by the fence. Consider the following.
        //
        // 1. Thread A invokes "REGISTER FENCE" after a successful CAS, and then checks the
        //    generation at "REGISTER RECHECK".
        //
        // 2. Thread B now enters the this block of code, executes "WAITING FENCE", then
        //    reads the generation tags for all slots.
        //
        // With the total order induced by the sequentially consistent fence, either thread
        // A's fence executes first, or thread B's executes first.
        //
        // * If thread A's fence executes first, then thread B will see the CAS and the set
        //   value is guaranteed to be greater-than or equal to "WAITING CHECK" since the
        //   generation check since is monotonically decreasing and thread A's
        //   "REGISTER CHECK" is forced to happen before.
        //
        // * If Thread B's fence executes first, then thread A's "REGISTER RECHECK" will
        //   observe at least the result of "WAITING CHECK" and update itself on the retry.
        //
        //   It's possible that thread B observes the CAS to "REGISTER CHECK", but since
        //   thread A will monotonically decrease it before exiting, the value thread B
        //   observes is conservative and not incorrect.
        delay.pre_fence();
        std::sync::atomic::fence(Ordering::SeqCst);
        delay.post_fence();

        // WAITING CHECK
        let current = self.generation();
        let mut max = current;

        for s in self.slots.iter() {
            let generation = s.as_ref().get(Ordering::Relaxed);
            if generation != Generation::MAX {
                max = max.max(generation);
            }
        }

        // This synchronizes with all the guard's `Release`s.
        std::sync::atomic::fence(Ordering::Acquire);
        (max == current, max)
    }

    pub fn try_advance(&self) -> Option<Drain<'_>> {
        self.try_advance_inner(NoDelay)
    }

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
        //
        // We intentionally ignore lock-poison since we expect the guarded queue to be
        // robust with respect to panics.
        let drain = self.drain.try_lock()?;

        let (can_advance, current) = self.can_advance_inner(&mut delay);

        // All waiters belong to the current generation. Therefore, it is safe to release
        // the old array queue
        if can_advance {
            // We are safe to use a `fetch_sub` here because `drain` is ensuring exclusivity
            // of the access.
            //
            // However, this still needs to be `SeqCst` so that this properly synchronizes
            // with "REGISTER FENCE" and "WAITER FENCE".
            let _previous = self.generation.as_mut().fetch_decrement(Ordering::SeqCst);
            debug_assert_eq!(_previous, current, "concurrency violation");

            let queue = &self.retiring[last_queue(current)];
            Some(Drain { queue, drain })
        } else {
            // Previous generation has not completely retired.
            None
        }
    }

    #[cfg(test)]
    fn assert_no_workers(&self) {
        for s in self.slots.iter() {
            assert_eq!(s.as_ref().get(Ordering::Relaxed), Generation::MAX);
        }
    }

    #[cfg(test)]
    fn snapshot(&self) -> Vec<Generation> {
        self.slots
            .iter()
            .map(|s| s.as_ref().get(Ordering::Relaxed))
            .collect()
    }

    #[cfg(test)]
    fn waiting(&self) -> Generation {
        self.can_advance_inner(&mut NoDelay).1
    }
}

#[derive(Debug)]
pub struct Guard<'a> {
    slot: Mut<'a>,
    retire: &'a SegQueue<u32>,
    generation: Generation,

    #[cfg(test)]
    slot_index: usize,
}

impl Guard<'_> {
    /// Return the generation associated with the [`Guard`]'s creation.
    #[inline]
    pub fn generation(&self) -> Generation {
        self.generation
    }

    /// Retire the slot `i` at the current generation.
    #[inline]
    pub fn retire(&self, i: u32) {
        self.retire.push(i)
    }

    /// Retire all items in `itr`.
    pub fn retire_all<I>(&self, itr: I)
    where
        I: IntoIterator<Item = u32>,
    {
        for i in itr {
            self.retire(i)
        }
    }
}

impl Drop for Guard<'_> {
    fn drop(&mut self) {
        self.slot.set(Generation::MAX, Ordering::Release);
    }
}

#[derive(Debug)]
pub struct Drain<'a> {
    queue: &'a SegQueue<u32>,
    drain: MutexGuard<'a, ()>,
}

impl Drain<'_> {
    #[must_use = "reclaimed ids must be reclaimed"]
    pub fn pop(&self) -> Option<u32> {
        self.queue.pop()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
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

#[derive(Debug)]
#[non_exhaustive]
pub struct Unavailable;

impl std::fmt::Display for Unavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("all available registry slots are occupied")
    }
}

impl std::error::Error for Unavailable {}

impl From<Unavailable> for diskann::ANNError {
    #[track_caller]
    fn from(unavailable: Unavailable) -> Self {
        diskann::ANNError::opaque(unavailable)
    }
}

// Delays
//
// To help test standard race scenarios without advanced tooling, we use optional delays
// that our tests can introduce to ensure threads are in various intermediate points.
//
// This does not necessarily test that the memory orderings are correct, but at least
// is a smoke test that various (known) races are handled properly.

#[derive(Debug)]
struct NoDelay;

trait RegisterDelay {
    fn post_register_check(&mut self) {}
    fn pre_cas(&mut self) {}
    fn post_cas(&mut self) {}
    fn pre_fence(&mut self) {}
    fn post_fence(&mut self) {}
}

impl RegisterDelay for NoDelay {}

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

    use std::{sync::Arc, thread};

    use parking_lot::Condvar;

    #[derive(Clone)]
    struct Sequencer(Arc<SequencerInner>);

    struct SequencerInner {
        state: Mutex<State>,
        condvar: Condvar,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum State {
        Empty,
        Parked(usize),
        Released(usize),
    }

    impl Sequencer {
        fn new() -> Self {
            Self(Arc::new(SequencerInner {
                state: Mutex::new(State::Empty),
                condvar: Condvar::new(),
            }))
        }

        fn wait_for(&self, stage: usize) {
            let mut state = self.0.state.lock();
            if stage == 0 {
                assert_eq!(*state, State::Empty)
            } else {
                assert_eq!(*state, State::Released(stage - 1))
            }

            *state = State::Parked(stage);
            self.0.condvar.notify_all();
            self.0
                .condvar
                .wait_while(&mut state, move |s| *s != State::Released(stage));
        }

        fn advance_past(&self, stage: usize) {
            let mut state = self.0.state.lock();
            self.0
                .condvar
                .wait_while(&mut state, move |s| Self::check_release(*s, stage));
            *state = State::Released(stage);
            self.0.condvar.notify_all();
        }

        fn until_waiting_for(&self, stage: usize) {
            let mut state = self.0.state.lock();
            if *state != State::Parked(stage) {
                self.0
                    .condvar
                    .wait_while(&mut state, move |s| Self::check_release(*s, stage))
            }
        }

        fn check_release(current: State, stage: usize) -> bool {
            match current {
                State::Empty => {
                    assert_eq!(stage, 0);
                    true
                }
                State::Released(s) => {
                    if s + 1 != stage {
                        panic!("observed {:?} while releasing stage {}", current, stage);
                    }
                    true
                }
                State::Parked(s) => {
                    if s != stage {
                        panic!("observed {:?} while releasing stage {}", current, stage)
                    }
                    false
                }
            }
        }
    }

    // This test ensures that two threads racing on `hint` will correctly resolve themselves
    // when claiming a slot.
    #[test]
    fn test_cas_race() {
        let seq = Sequencer::new();

        let delay = TestRegisterDelay::default().with_post_register_check(|| seq.wait_for(0));

        let registry = Registry::with_capacity(2);
        assert_eq!(registry.capacity(), 2);

        thread::scope(|s| {
            // Thread A
            s.spawn(|| {
                let g = registry.register_inner(delay).unwrap();
                assert_eq!(g.slot_index, 1);
                seq.wait_for(1);
            });

            // Thread B
            s.spawn(|| {
                // wait for Thread A to reach the delay point.
                seq.until_waiting_for(0);
                {
                    let g = registry.register_inner(NoDelay).unwrap();
                    assert_eq!(g.slot_index, 1);
                }
                let g = registry.register_inner(NoDelay).unwrap();
                assert_eq!(g.slot_index, 0);
                seq.advance_past(0);
                seq.advance_past(1);
            });
        });

        registry.assert_no_workers();
    }

    #[test]
    fn test_register_wait() {
        // This tests the case where a thread enters registration, reads a generation, then
        // sleeps for several generation advances. It ensures that the thread recovers properly.
        let seq = Sequencer::new();

        let mut loop_count = 0;
        let delay = TestRegisterDelay::default()
            .with_post_register_check(|| seq.wait_for(0))
            .with_post_cas(|| seq.wait_for(1))
            .with_pre_fence(|| loop_count += 1);

        let registry = Registry::with_capacity(2);

        thread::scope(|s| {
            let handle = s.spawn(|| {
                let guard = registry.register_inner(delay).unwrap();

                // Since we hit the CAS loop - this serves as a sanity check that we have
                // the correct drain buffer.
                guard.retire(10);
                guard.retire_all([1, 2, 3]);
                guard
            });

            // Wait for the spawned thread to reach the critical section.
            seq.until_waiting_for(0);

            assert_eq!(registry.waiting(), Generation::MAX.sub(1));
            {
                let drain = registry.try_advance().unwrap();
                assert!(drain.is_empty());
                assert_eq!(registry.generation(), Generation::MAX.sub(2));
            }

            {
                let drain = registry.try_advance().unwrap();
                assert!(drain.is_empty());
                assert_eq!(registry.generation(), Generation::MAX.sub(3));
            }

            // We allow the registering thread to make it past the CAS.
            //
            // We pause it again because we want to verify that it registers an old generation.
            seq.advance_past(0);
            seq.until_waiting_for(1);
            let (can_advance, waiter) = registry.can_advance_inner(&mut NoDelay);
            assert!(!can_advance);
            assert_eq!(
                waiter,
                Generation::MAX.sub(1),
                "waiting thread registers an older generation before observing the change"
            );
            seq.advance_past(1);

            let expected = Generation::MAX.sub(3);

            // The generation should be the last set one - even though this thread was
            // parked during the transition.
            let r = handle.join().unwrap();
            assert_eq!(r.generation(), expected);
            assert_eq!(registry.waiting(), expected);
        });

        assert_eq!(
            loop_count, 2,
            "the registering thread should have looped to update its generation"
        );

        registry.assert_no_workers();

        // Verify that we reclaim the ID flushed by the registering thread.
        //
        // This requires two epoch advancements.
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
        TestRegisterDelay,
        RegisterDelay,
        with_post_register_check => post_register_check,
        with_pre_cas => pre_cas,
        with_post_cas => post_cas,
        with_pre_fence => pre_fence,
        with_post_fence => post_fence,
    }

    // #[derive(Default)]
    // struct TestRegisterDelay<'a> {
    //     post_register_check: Option<&'a mut dyn FnMut()>,
    //     pre_cas: Option<&'a mut dyn FnMut()>,
    //     pre_fence: Option<&'a mut dyn FnMut()>,
    //     post_fence: Option<&'a mut dyn FnMut()>,
    // }

    // macro_rules! builder {
    //     ($f:ident, $field:ident) => {
    //         fn $f(mut self, f: &'a mut dyn FnMut()) -> Self {
    //             self.$field = Some(f);
    //             self
    //         }
    //     }
    // }

    // macro_rules! forward {
    //     ($f:ident) => {
    //         fn $f(&mut self) {
    //             if let Some(f) = self.$f.as_mut() {
    //                 f()
    //             }
    //         }
    //     }
    // }

    // impl<'a> TestRegisterDelay<'a> {
    //     builder!(with_post_register_check, post_register_check);
    //     builder!(with_pre_cas, pre_cas);
    //     builder!(with_pre_fence, pre_fence);
    //     builder!(with_post_fence, post_fence);
    // }

    // impl RegisterDelay for TestRegisterDelay<'_> {
    //     forward!(post_register_check);
    //     forward!(pre_cas);
    //     forward!(pre_fence);
    //     forward!(post_fence);
    // }

    // struct CanAdvanceDelay;

    // impl CanAdvanceDelay for TestWaitingDelay {}

    // struct TestTryAdvanceDelay;

    // impl TryAdvanceDelay for TestTryAdvanceDelay {}
}
