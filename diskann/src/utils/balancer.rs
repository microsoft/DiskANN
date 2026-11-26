/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    marker::PhantomData,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::utils::{IntoUsize, async_tools};

//----------//
// Balander //
//----------//

/// A utility for dynamically processing elements in a slice `T` across multiple threads.
///
/// This works by using a local `AtomicUsize` to track which indices inside the slice have
/// been claimed and is useful when the work to be performed on each `T` is not suitable
/// for static partitioning.
///
/// This can occur when the work is highly variable.
///
/// ```
/// use diskann::utils::balancer::Balancer;
///
/// let items = [10, 20];
/// let balancer = Balancer::new(items);
///
/// let item = balancer.next().unwrap();
/// assert_eq!(item.current(), &10, "the first item");
/// assert_eq!(item.position(), 0, "the first item");
/// assert_eq!(item.all(), &[10, 20]);
///
/// let item = balancer.next().unwrap();
/// assert_eq!(item.current(), &20, "the second item");
/// assert_eq!(item.position(), 1, "the second item");
/// assert_eq!(item.all(), &[10, 20]);
///
/// assert!(balancer.next().is_none());
/// ```
#[derive(Debug)]
pub struct Balancer<T, C> {
    items: C,
    current: AtomicU64,
    _type: PhantomData<T>,
}

impl<T, C> Balancer<T, C>
where
    C: AsRef<[T]>,
{
    /// Construct a new [`Balancer`] over `items`.
    ///
    /// This method does not allocate. To share the [`Balancer`] across multiple threads,
    /// embed it in a [`std::sync::Arc`].
    pub fn new(items: C) -> Self {
        Self {
            items,
            current: AtomicU64::new(0),
            _type: PhantomData,
        }
    }

    /// Retrieve the next unclamied item and return a [`BatchItem`] containing said
    /// item and its position in the underlying slice.
    ///
    /// If all items have been claimed, return `None`.
    ///
    /// This function yields all items in the underlying slice exactly once.
    pub fn next(&self) -> Option<BatchItem<'_, T>> {
        // `AtomicU64` is used instead of `AtomicUsize` to guarantee that a 64-bit counter
        // is used to increase the improbability of overflowing the counter.
        //
        // Use `IntoUsize` to ensure that we're on a 64-bit system.
        let i = self.current.fetch_add(1, Ordering::Relaxed);
        let items = self.items.as_ref();

        // There is technically a race here where enough concurrent calls to `next` in
        // between the retrieval of `i` and "putting it back" that the counter wraps around.
        //
        // The odds of that happening, however, are exceedingly rare.
        match items.get(i.into_usize()) {
            Some(current) => Some(BatchItem {
                batch: items,
                current,
                position: i.into_usize(),
            }),
            None => {
                // This branch will never be hit in a realistic program.
                if i == u64::MAX {
                    std::process::abort();
                }

                self.current.fetch_sub(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Return the number of work items.
    pub fn len(&self) -> usize {
        self.items.as_ref().len()
    }

    /// Return `true` only if the underlying batch is empty.
    pub fn is_empty(&self) -> bool {
        self.items.as_ref().is_empty()
    }
}

//----------------//
// Batch Position //
//----------------//

/// An item in a batch - returned by [`Balancer::next`].
#[derive(Debug)]
pub struct BatchItem<'a, T> {
    // The entire batch.
    batch: &'a [T],
    // The current item.
    current: &'a T,
    // Must be less than `batch.len()`.
    position: usize,
}

impl<'a, T> BatchItem<'a, T> {
    /// Return the current item in the batch.
    pub fn current(&self) -> &T {
        self.current
    }

    /// Return all items in the batch.
    pub fn all(&self) -> &[T] {
        self.batch
    }

    /// Return the logical position of this item in its bach.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Return the number of items in the batch.
    pub fn len(&self) -> usize {
        self.batch.len()
    }

    /// Return an [`async_tools::Around`] iterator around the current position.
    pub(crate) fn around(&self, count: usize) -> async_tools::Around<'_, T> {
        async_tools::around(self.batch, self.position, count)
    }
}

impl<T> Clone for BatchItem<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for BatchItem<'_, T> {}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_batch_stays_exhausted() {
        let balancer = Balancer::<u32, _>::new([]);
        for _ in 0..10 {
            assert!(balancer.next().is_none());
        }
    }

    #[test]
    fn claims_positions_in_order() {
        let items = [10, 20, 10];
        let balancer = Balancer::new(&items);

        for (position, expected) in items.iter().enumerate() {
            let item = balancer.next().unwrap();
            assert_eq!(item.position(), position);
            assert!(std::ptr::eq(item.current(), expected));
            assert!(std::ptr::eq(item.all(), items.as_slice()));
        }

        for _ in 0..10 {
            assert!(balancer.next().is_none());
        }
    }

    #[test]
    fn concurrent_workers_claim_each_position_once() {
        const WORKERS: usize = 8;
        let items: Vec<_> = (0..1024).collect();
        let balancer = Balancer::new(&items);
        let barrier = std::sync::Barrier::new(WORKERS);

        let mut claimed = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..WORKERS)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        let mut claimed = Vec::new();
                        while let Some(item) = balancer.next() {
                            assert_eq!(*item.current(), item.position());
                            claimed.push(item.position());
                            std::thread::yield_now();
                        }
                        for _ in 0..10 {
                            assert!(balancer.next().is_none());
                        }
                        claimed
                    })
                })
                .collect();

            handles
                .into_iter()
                .flat_map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        claimed.sort_unstable();
        assert_eq!(claimed, items);
    }
}
