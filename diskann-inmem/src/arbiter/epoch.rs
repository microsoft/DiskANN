/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex, TryLockError,
};

use crossbeam_queue::SegQueue;
use diskann::utils::IntoUsize;

use crate::arbiter::{
    generation::{Mut, Tag},
    Generation,
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

    fn register_inner<T>(&self, _: T) -> Result<Guard<'_>, Unavailable>
    where
        T: RegisterDelay,
    {
        // REGISTER CHECK
        let mut generation = self.generation();
        let hint = self.hint.fetch_add(1, Ordering::Relaxed);

        let nslots = self.slots.len();
        for i in 0..nslots {
            let slot = (hint + i) % nslots;

            let m = self.slots[slot].as_mut();
            if let Ok(_) = m.try_set(
                Generation::MAX,
                generation,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                let mut reset = false;
                loop {
                    // REGISTER FENCE: This fence is paired with "WAITING FENCE".
                    //
                    // See that comment for details.
                    std::sync::atomic::fence(Ordering::SeqCst);

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
    pub fn waiting(&self) -> Generation {
        self.waiting_inner(NoDelay)
    }

    fn waiting_inner<T>(&self, _: T) -> Generation
    where
        T: WaitingDelay,
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
        std::sync::atomic::fence(Ordering::SeqCst);

        // WAITING CHECK
        let mut max = self.generation();

        for s in self.slots.iter() {
            let generation = s.as_ref().get(Ordering::Relaxed);
            if generation != Generation::MAX {
                max = max.max(generation);
            }
        }

        // This synchronizes with all the guard's `Release`s.
        std::sync::atomic::fence(Ordering::Acquire);
        max
    }

    pub fn try_advance(&self) -> Option<Drain<'_>> {
        self.try_advance_inner(NoDelay)
    }

    fn try_advance_inner<T>(&self, _: T) -> Option<Drain<'_>>
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
        let drain = match self.drain.try_lock() {
            Ok(drain) => drain,
            Err(TryLockError::Poisoned(drain)) => drain.into_inner(),
            Err(TryLockError::WouldBlock) => return None,
        };

        let waiting = self.waiting();
        let current = self.generation.as_ref().get(Ordering::Relaxed);

        // All waiters belong to the current generation. Therefore, it is safe to release
        // the old array queue
        if waiting == current {
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
}

#[derive(Debug)]
pub struct Guard<'a> {
    slot: Mut<'a>,
    retire: &'a SegQueue<u32>,
    generation: Generation,
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
    drain: std::sync::MutexGuard<'a, ()>,
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

trait RegisterDelay {}

impl RegisterDelay for NoDelay {}

trait WaitingDelay {}

impl WaitingDelay for NoDelay {}

trait TryAdvanceDelay {}

impl TryAdvanceDelay for NoDelay {}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
}
