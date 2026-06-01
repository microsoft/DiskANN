/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    num::NonZeroU32,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering},
        Mutex,
    },
};

#[derive(Debug)]
pub struct Freelist {
    recycled: Mutex<Vec<u32>>,
    capacity: NonZeroU32,
    have_recycled: AtomicBool,

    /// The highest ID the freelist manages. This is used when in "append" to determine the
    /// maximum ID we can return this way.
    max: u32,
    /// The number of "unallocated" IDs remaining.
    unallocated: AtomicU32,
}

#[derive(Debug, Clone, Copy)]
pub enum Id {
    Found(u32),
    Scan,
}

impl Freelist {
    pub fn new(max: u32, capacity: NonZeroU32) -> Self {
        Self {
            recycled: Mutex::new(Vec::with_capacity(capacity.get() as usize)),
            capacity,
            have_recycled: AtomicBool::new(false),
            max,
            unallocated: AtomicU32::new(max),
        }
    }

    pub fn pop(&self) -> Id {
        // Small performance optimization - avoid locking the mutex if looks like that won't
        // succeed anyways.
        if self.have_recycled.load(Ordering::Relaxed) {
            let mut recycled = self.recycled.lock().unwrap();
            if let Some(id) = recycled.pop() {
                return Id::Found(id);
            }
            self.have_recycled.store(false, Ordering::Relaxed);
        }

        // Missed in the recycled buffer. Try pulling from the high-water mark.
        let mut unallocated = self.unallocated.load(Ordering::Relaxed);
        while unallocated != 0 {
            match self.unallocated.compare_exchange(
                unallocated,
                unallocated - 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(unallocated) => return Id::Found(self.max - unallocated),
                Err(actual) => {
                    unallocated = actual;
                }
            }
        }

        // Missed in the recycle bin and from unallocated IDs. Time to indicate a scan.
        Id::Scan
    }

    /// Attempt to push `id` into the recycled list. Return `true` if `id` was
    /// inserted. If `false` is returned, it is likely because the internal recycle
    /// buffer is full.
    pub fn push(&self, id: u32) -> bool {
        let mut recycled = self.recycled.lock().unwrap();
        if recycled.len() < self.capacity() {
            recycled.push(id);
            self.have_recycled.store(true, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Append items from `itr` into the recycled buffer. Return the number of items
    /// actually added.
    pub fn append<I>(&self, itr: I) -> usize
    where
        I: IntoIterator<Item = u32>,
    {
        let mut recycled = self.recycled.lock().unwrap();
        let available = self.capacity() - recycled.len();
        let mut count = 0;
        itr.into_iter().take(available).for_each(|id| {
            count += 1;
            recycled.push(id);
        });

        if count > 0 {
            self.have_recycled.store(true, Ordering::Relaxed);
        }

        count
    }

    //----------//
    // Internal //
    //----------//

    fn capacity(&self) -> usize {
        self.capacity.get() as usize
    }
}
