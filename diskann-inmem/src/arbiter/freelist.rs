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

const SCAN_SIZE: u32 = 16;

#[derive(Debug)]
pub struct Freelist {
    recycled: ArrayQueue<u32>,

    /// The highest ID the freelist manages. This is used when in "append" to determine the
    /// maximum ID we can return this way.
    max: u32,

    /// The number of "unallocated" IDs remaining.
    current: AtomicU32,

    scan_cursor: AtomicU32,
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
            max,
            current: AtomicU32::new(0),
            scan_cursor: AtomicU32::new(0),
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

    pub fn pop_recycled(&self) -> Option<u32> {
        self.recycled.pop()
    }

    pub fn scan(&self) -> Scan {
        let current = self.scan_cursor.fetch_add(SCAN_SIZE, Ordering::Relaxed) % self.max;
        Scan {
            current,
            max: self.max,
            len: SCAN_SIZE.into_usize()
        }
    }

    /// Attempt to push `id` into the recycled list. Return `true` if `id` was
    /// inserted. If `false` is returned, it is likely because the internal recycle
    /// buffer is full.
    pub fn push(&self, id: u32) -> bool {
        match self.recycled.push(id) {
            Ok(()) => true,
            Err(_) => false,
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

#[derive(Debug)]
pub struct Scan {
    current: u32,
    max: u32,
    len: usize,
}

impl Iterator for Scan {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.len == 0 {
            None
        } else {
            let mut i = self.current;
            self.current += 1;
            self.len -= 1;
            if self.current == self.max {
                self.current -= self.max;
            }
            Some(i)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl ExactSizeIterator for Scan {}
