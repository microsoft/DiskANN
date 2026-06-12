/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};

use crate::arbiter::Generation;

const CAPACITY: usize = 256;

#[derive(Debug)]
pub struct Registry {
    /// A record of the active generations.
    ///
    /// * 0 = "available".
    /// * non-zero: generation is active.
    slots: Box<[AtomicU64]>,
    generation: AtomicU64,
    barrier: Mutex<Hint>,
}

impl Registry {
    pub fn new() -> Self {
        Self::with_capacity(CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: (0..capacity).map(|_| AtomicU64::new(0)).collect(),
            generation: AtomicU64::new(u64::MAX),
            barrier: Mutex::new(Hint(0)),
        }
    }

    pub fn generation(&self) -> Generation {
        Generation::new(self.generation.load(Ordering::Acquire))
    }

    pub fn register(&self) -> Guard<'_> {
        let mut barrier = self.barrier.lock().unwrap();

        // No synchronization happens on the global generation tag.
        let generation = self.generation.load(Ordering::Acquire);
        let hint = barrier.increment();

        let nslots = self.slots.len();
        for i in 0..nslots {
            let slot = (hint + i) % nslots;
            if let Ok(_) = self.slots[slot].compare_exchange(
                0,
                generation,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                return Guard {
                    registry: self,
                    slot,
                    generation: Generation::new(generation),
                };
            }
        }

        panic!("Let's turn this into a proper error.");
    }

    pub fn advance(&self) -> Generation {
        // TODO: What to do on the unlikely event of a wrap-around?
        Generation::new(self.generation.fetch_sub(1, Ordering::AcqRel))
    }

    fn wait_for(&self, generation: Generation) {
        let generation = generation.value();
        let wait_list = {
            let _barrier = self.barrier.lock().unwrap();
            let mut wait_list = Vec::new();
            for (i, s) in self.slots.iter().enumerate() {
                let g = s.load(Ordering::Relaxed);
                if g != 0 && g >= generation {
                    wait_list.push(i);
                }
            }

            wait_list
        };

        for slot in wait_list {
            let s = &self.slots[slot];
            loop {
                let g = s.load(Ordering::Relaxed);
                if g == 0 || g < generation {
                    break;
                }
                std::hint::spin_loop();
            }
        }

        // This barrier synchronizes with all the relaxed loads on the slots, which are
        // set with `Release` semantics.
        std::sync::atomic::fence(Ordering::Acquire);
    }

    /// Return the oldest generation that is currently being protected.
    ///
    /// Generations decrement from `Generation::MAX`
    ///
    /// This is a syncronizing operation.
    pub fn waiting(&self) -> Generation {
        let _barrier = self.barrier.lock().unwrap();
        let mut highest = 0;
        for s in self.slots.iter() {
            let g = s.load(Ordering::Relaxed);
            highest = highest.max(g);
        }

        // `acquires` with respect to all previous relaxed loads.
        std::sync::atomic::fence(Ordering::Acquire);

        Generation::new(highest)
    }
}

#[derive(Debug)]
pub struct Guard<'a> {
    registry: &'a Registry,
    slot: usize,
    generation: Generation,
}

impl Guard<'_> {
    /// Return the generation associated with the [`Guard`]'s creation.
    #[inline]
    pub fn generation(&self) -> Generation {
        self.generation
    }
}

impl Drop for Guard<'_> {
    fn drop(&mut self) {
        self.registry.slots[self.slot].store(0, Ordering::Release)
    }
}

#[derive(Debug)]
struct Hint(usize);

impl Hint {
    fn increment(&mut self) -> usize {
        let x = self.0;
        self.0 += 1;
        x
    }
}
