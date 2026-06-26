/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Finding Unused IDs
//!
//! When working with slots into an index, finding an available slot efficiently can be
//! challenging. This module provides a [`Freelist`] to make this more efficient.
//!
//! IDs are retrieved in the following order of precedence:
//!
//! ## Recycles
//!
//! Previously reclaimed slots can be recycled and are the preferred way of finding slots.
//! Reclaimed slots IDs live inside an atomic queue and as such, the size of this queue is
//! bounded to conserve memory.
//!
//! ## Minted
//! If no slots live in the recycled queue, new slots can be "minted" up to the configured
//! maximum. This simply tracks the maximum slot ID that has been yielded so far and returns
//! the next one.
//!
//! This path only works during the initial filling of the managed slots and exists to
//! provide a fast-path for static index builds. Once the maximum slot has been yielded,
//! minting no longer applies.
//!
//! ## Scanning
//!
//! If a slot cannot be found via recycling or via minting, a scan is requested. Scans
//! typically involve searching over an authoritative source of slot usage to find and
//! claim an unused slot.
//!
//! The [`Freelist`] assists with scans in several ways:
//!
//! 1. [`Freelist::scan`]: Receive a range of managed IDs to scan. Multiple threads
//!    can call this method to receive disjoint ranges to process.
//!
//! 2. [`Freelist::push`]: Available slots can be placed into the
//!    freelist for recycling.
//!
//! 3. [`Freelist::pop_recycled`]: Attempt to retrieve a slot ID directly from the recycled
//!    buffer.
//!
//! Together, these tools can be used to build a cooperative scan. A thread scans a block of
//! IDs returned by [`Freelist::scan`]. If a slot is claimed this way, the thread can continue
//! scanning the rest of the block, pushing any available slot IDs to the freelist.
//!
//! Other threads that are unsuccessfully scanning can periodically check
//! [`Freelist::pop_recycled`] to benefit from the work done by another more successful thread.
//!
//! # Non-Authoritative
//!
//! Note that the [`Freelist`] does not attempt to be authoritative on the list of slots IDs
//! that are used and unused. Its job is mainly to improve performance.
//!
//! An authoritative collection of [`AtomicTag`](super::AtomicTag)s must be used to correctly
//! manage slots.

use std::{
    num::NonZeroU32,
    sync::atomic::{AtomicU32, Ordering},
};

use crossbeam_queue::ArrayQueue;
use diskann::utils::IntoUsize;

// NOTE: We want the scan size to be relatively big. Each tag occupied just a single byte,
// so a scan needs to be at least 64 to ensure a thread is working with just a single cache
// line.
const SCAN_SIZE: u32 = 256;

/// A tool for quickly finding unused slots in an index.
///
/// See [freelist](self) for details.
#[derive(Debug)]
pub(crate) struct Freelist {
    // Bounded fast queue of retired slots.
    recycled: ArrayQueue<u32>,

    // The highest ID the freelist manages. IDs `>= max` are rejected by `push`/`append`
    // and the minting path will not yield them.
    max: u32,

    // The next `unminted` Id. This becomes unused once this reaches `max`.
    next: AtomicU32,

    // The current bucket for scanning.
    scan_bucket: AtomicU32,
}

impl Freelist {
    /// Construct a new [`Freelist`] that manages `max` ids.
    ///
    /// The internal fast recycled list will hold up to `recycled` items.
    ///
    /// The memory occupied by this struct is `O(recycled)`.
    pub(crate) fn new(max: u32, recycled: NonZeroU32) -> Self {
        Self {
            recycled: ArrayQueue::new(recycled.get().into_usize()),
            max,
            next: AtomicU32::new(0),
            scan_bucket: AtomicU32::new(0),
        }
    }

    /// Try to retrieve an id.
    ///
    /// If successful, return [`Id::Found`]. Otherwise, returns [`Id::Scan`].
    pub(crate) fn pop(&self) -> Id {
        if let Some(id) = self.recycled.pop() {
            return Id::Found(id);
        }

        // Missed in the recycled buffer. Try pulling from the high-water mark.
        let mut next = self.next.load(Ordering::Relaxed);
        while next < self.max {
            match self
                .next
                .compare_exchange(next, next + 1, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(next) => return Id::Found(next),
                Err(actual) => {
                    next = actual;
                }
            }
        }

        // Missed in the recycle bin and from unallocated IDs. Time to indicate a scan.
        Id::Scan
    }

    /// Attempt to retrieve an ID directly from the recycled list.
    ///
    /// This may be used during scans to retrieve IDs found by other threads.
    pub(crate) fn pop_recycled(&self) -> Option<u32> {
        self.recycled.pop()
    }

    /// Return a new [`Scan`] containing a range of IDs to check.
    ///
    /// This is managed such that multiple threads calling this function will receive
    /// disjoint ranges to scan.
    pub(crate) fn scan(&self) -> Scan {
        if self.max == 0 {
            return Scan { start: 0, stop: 0 };
        }

        let num_buckets = self.max.div_ceil(SCAN_SIZE);

        // It's possible that if `scan_bucket` wraps, we do a bit of redundant scanning.
        //
        // This is fine as this should happen rarely.
        let bucket = self.scan_bucket.fetch_add(1, Ordering::Relaxed) % num_buckets;

        let start = bucket * SCAN_SIZE;
        let stop = match start.checked_add(SCAN_SIZE) {
            Some(stop) => stop.min(self.max),
            None => self.max,
        };

        Scan { start, stop }
    }

    /// Attempt to push `id` into the recycled list. Return `true` if `id` was inserted.
    ///
    /// If `false` is returned, it is likely because the internal recycle buffer is full.
    ///
    /// IDs at or above [`Self::max`] are discarded.
    pub(crate) fn push(&self, id: u32) -> bool {
        if id < self.max {
            self.recycled.push(id).is_ok()
        } else {
            false
        }
    }
}

/// The result of [`Freelist::pop`].
#[derive(Debug, Clone, Copy)]
#[must_use]
pub(crate) enum Id {
    /// An ID was found directly in the [`Freelist`].
    Found(u32),
    /// No ID was found in the [`Freelist`] and an exhaustive scan is recommended.
    Scan,
}

#[cfg(test)]
impl Id {
    fn unwrap(self) -> u32 {
        match self {
            Self::Found(i) => i,
            Self::Scan => panic!("expected Id::Found, got Id::Scan"),
        }
    }

    fn is_scan(self) -> bool {
        matches!(self, Self::Scan)
    }
}

/// An [`ExactSizeIterator`] over IDs to scan. Returned by [`Freelist::scan`].
#[derive(Debug)]
pub(crate) struct Scan {
    start: u32,
    stop: u32,
}

impl Scan {
    #[cfg(test)]
    fn as_range(&self) -> std::ops::Range<u32> {
        self.start..self.stop
    }
}

impl Iterator for Scan {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.start >= self.stop {
            None
        } else {
            let i = self.start;
            self.start += 1;
            Some(i)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = (self.stop - self.start).into_usize();
        (len, Some(len))
    }
}

impl ExactSizeIterator for Scan {}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::{collections::HashSet, sync::Barrier, thread};

    fn freelist(max: u32, recycled: u32) -> Freelist {
        Freelist::new(max, NonZeroU32::new(recycled).unwrap())
    }

    //---------//
    // Minting //
    //---------//

    #[test]
    fn pop_mints_sequentially_until_exhausted() {
        let fl = freelist(4, 8);

        let mut got = Vec::new();
        for _ in 0..4 {
            got.push(fl.pop().unwrap());
        }
        assert_eq!(got, vec![0, 1, 2, 3]);
        assert!(fl.pop().is_scan());
        assert!(fl.pop().is_scan());
    }

    #[test]
    fn pop_returns_scan_when_max_zero() {
        let fl = freelist(0, 1);
        assert!(fl.pop().is_scan());
    }

    #[test]
    fn recycled_ids_take_precedence_over_minting() {
        let fl = freelist(4, 8);
        // Seed the recycled queue.
        assert!(fl.push(2));
        // First pop must come from the recycled queue, not mint 0.
        assert_eq!(fl.pop().unwrap(), 2);
        // Subsequent pops mint from 0.
        assert_eq!(fl.pop().unwrap(), 0);
    }

    //------//
    // Push //
    //------//

    #[test]
    fn push_rejects_ids_at_or_above_max() {
        let fl = freelist(4, 8);
        assert!(!fl.push(4));
        assert!(!fl.push(u32::MAX));
        assert!(fl.push(3));

        assert_eq!(fl.pop_recycled().unwrap(), 3);
    }

    #[test]
    fn push_returns_false_when_recycled_full() {
        let fl = freelist(16, 2);
        assert!(fl.push(2));
        assert!(fl.push(3));
        assert!(!fl.push(5));

        // Drained from recycled queue.
        assert_eq!(fl.pop().unwrap(), 2);
        assert_eq!(fl.pop().unwrap(), 3);
    }

    #[test]
    fn pop_recycled_empty_returns_none() {
        let fl = freelist(4, 4);
        assert!(fl.pop_recycled().is_none());
    }

    #[test]
    fn pop_recycled_does_not_mint() {
        let fl = freelist(4, 4);
        // No pushes, no recycled entries — `pop_recycled` must not fall through to minting.
        assert!(fl.pop_recycled().is_none());
        // The minting counter should be untouched.
        assert_eq!(fl.pop().unwrap(), 0);
    }

    //------//
    // Scan //
    //------//

    fn as_vec<I>(itr: I) -> Vec<I::Item>
    where
        I: Iterator,
    {
        itr.collect()
    }

    #[test]
    fn scan_on_empty_freelist_yields_nothing() {
        let fl = freelist(0, 1);
        let mut scan = fl.scan();
        assert_eq!(scan.len(), 0);
        assert!(scan.next().is_none());
    }

    #[test]
    fn scan_covers_full_range_in_one_pass() {
        // Choose `max` to force a partial last bucket.
        let max = 2 * SCAN_SIZE + 50;
        let fl = freelist(max, 4);

        // First Round

        let scan = fl.scan();
        assert_eq!(scan.as_range(), 0..SCAN_SIZE);
        assert_eq!(scan.len(), SCAN_SIZE.into_usize());
        assert_eq!(as_vec(scan), as_vec(0..SCAN_SIZE));

        let scan = fl.scan();
        assert_eq!(scan.as_range(), SCAN_SIZE..2 * SCAN_SIZE);
        assert_eq!(scan.len(), SCAN_SIZE.into_usize());
        assert_eq!(as_vec(scan), as_vec(SCAN_SIZE..2 * SCAN_SIZE));

        let scan = fl.scan();
        assert_eq!(scan.as_range(), 2 * SCAN_SIZE..(2 * SCAN_SIZE + 50));
        assert_eq!(scan.len(), 50);
        assert_eq!(as_vec(scan), as_vec((2 * SCAN_SIZE)..(2 * SCAN_SIZE + 50)));

        // Check Wrapping

        let scan = fl.scan();
        assert_eq!(scan.as_range(), 0..SCAN_SIZE);
        assert_eq!(scan.len(), SCAN_SIZE.into_usize());
        assert_eq!(as_vec(scan), as_vec(0..SCAN_SIZE));
    }

    //-------------//
    // Concurrency //
    //-------------//

    #[test]
    fn concurrent_pop_yields_unique_ids() {
        let max = 4096u32;
        let fl = Freelist::new(max, NonZeroU32::new(8).unwrap());
        let nthreads = 8;
        let barrier = Barrier::new(nthreads);

        let results: Vec<Vec<u32>> = thread::scope(|s| {
            let handles: Vec<_> = (0..nthreads)
                .map(|_| {
                    s.spawn(|| {
                        let mut out = Vec::new();
                        barrier.wait();
                        while let Id::Found(id) = fl.pop() {
                            out.push(id);
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut all: Vec<u32> = results.into_iter().flatten().collect();
        all.sort();
        let expected: Vec<u32> = (0..max).collect();
        assert_eq!(all, expected, "all ids in [0, max) minted exactly once");
    }

    #[test]
    fn concurrent_scan_partitions_one_pass() {
        let max = SCAN_SIZE * 4;
        let fl = Freelist::new(max, NonZeroU32::new(4).unwrap());
        let num_buckets = max.div_ceil(SCAN_SIZE) as usize;
        let nthreads = num_buckets;
        let barrier = Barrier::new(nthreads);

        let ids: Vec<u32> = thread::scope(|s| {
            let handles: Vec<_> = (0..nthreads)
                .map(|_| {
                    s.spawn(|| {
                        barrier.wait();
                        fl.scan().collect::<Vec<u32>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        });

        let unique: HashSet<u32> = ids.iter().copied().collect();
        assert_eq!(
            unique.len(),
            ids.len(),
            "no id appeared twice across threads"
        );
        assert_eq!(
            unique.len() as u32,
            max,
            "scans covered every id in [0, max)"
        );
    }
}
