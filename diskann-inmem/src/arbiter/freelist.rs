/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    num::NonZeroU32,
    sync::atomic::{AtomicU32, AtomicUsize, Ordering},
};

use crossbeam_queue::ArrayQueue;
use diskann::utils::IntoUsize;

#[derive(Debug)]
pub struct Freelist {
    recycled: ArrayQueue<u32>,

    /// An (approximate) number of recycled IDs that exist outside the freelist.
    orphaned: AtomicUsize,

    /// The highest ID the freelist manages. This is used when in "append" to determine the
    /// maximum ID we can return this way.
    max: u32,

    /// The number of "unallocated" IDs remaining.
    current: AtomicU32,
}

#[derive(Debug, Clone, Copy)]
pub enum Id {
    Found(u32),
    Scan,
}

impl Freelist {
    pub fn new(max: u32, capacity: NonZeroU32) -> Self {
        Self {
            recycled: ArrayQueue::new(capacity.get().into_usize()),
            orphaned: AtomicUsize::new(0),
            max,
            current: AtomicU32::new(0),
        }
    }

    pub fn pop(&self) -> Id {
        if let Some(id) = self.recycled.pop() {
            return Id::Found(id);
        }

        // Missed in the recycled buffer. Try pulling from the high-water mark.
        let mut current = self.current.load(Ordering::Relaxed);
        while current != self.max {
            match self.current.compare_exchange(
                current,
                current + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(current) => return Id::Found(current),
                Err(actual) => {
                    current = actual;
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
        match self.recycled.push(id) {
            Ok(()) => true,
            Err(_) => {
                self.orphaned.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// Append items from `itr` into the recycled buffer. Return the number of items
    /// actually added.
    pub fn append<I>(&self, itr: I) -> usize
    where
        I: IntoIterator<Item = u32>,
    {
        let mut itr = itr.into_iter();
        let mut count = 0;
        while let Some(id) = itr.next() {
            if let Err(_) = self.recycled.push(id) {
                let (lower, _) = itr.size_hint();

                // Add 1 to "put back" the last ID.
                self.orphaned
                    .fetch_add(lower.saturating_add(1), Ordering::Relaxed);
                break;
            } else {
                count += 1;
            }
        }

        count
    }

    //----------//
    // Internal //
    //----------//

    fn capacity(&self) -> usize {
        self.recycled.capacity()
    }
}
