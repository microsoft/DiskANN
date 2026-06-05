/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::{
    atomic::{AtomicU32, AtomicU64, Ordering},
    Mutex,
};

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
            generation: AtomicU64::new(1),
            barrier: Mutex::new(Hint(0)),
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
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
                    generation,
                };
            }
        }

        panic!("Let's turn this into a proper error.");
    }

    pub fn advance(&self) -> u64 {
        // TODO: What to do on the unlikely event of a wrap-around?
        self.generation.fetch_add(1, Ordering::AcqRel)
    }

    fn wait_for(&self, generation: u64) {
        let wait_list = {
            let _barrier = self.barrier.lock().unwrap();
            let mut wait_list = Vec::new();
            for (i, s) in self.slots.iter().enumerate() {
                let g = s.load(Ordering::Relaxed);
                if g != 0 && g <= generation {
                    wait_list.push(i);
                }
            }

            wait_list
        };

        for slot in wait_list {
            let s = &self.slots[slot];
            loop {
                let g = s.load(Ordering::Relaxed);
                if g == 0 || g > generation {
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
    /// This is a syncronizing operation.
    pub fn waiting(&self) -> u64 {
        let _barrier = self.barrier.lock().unwrap();
        let mut lowest = u64::MAX;
        for s in self.slots.iter() {
            let g = s.load(Ordering::Relaxed);
            lowest = lowest.min(g);
        }

        // `acquires` with respect to all previous relaxed loads.
        std::sync::atomic::fence(Ordering::Acquire);

        lowest
    }
}

#[derive(Debug)]
pub struct Guard<'a> {
    registry: &'a Registry,
    slot: usize,
    generation: u64,
}

impl Guard<'_> {
    /// Return the generation associated with the [`Guard`]'s creation.
    #[inline]
    pub fn generation(&self) -> u64 {
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

