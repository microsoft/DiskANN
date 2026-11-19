/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{marker::PhantomData, sync::atomic::{AtomicUsize, Ordering}};

use crate::utils::async_tools;

//----------------//
// Batch Position //
//----------------//

#[derive(Debug)]
pub struct BatchPosition<'a, T> {
    batch: &'a [T],
    // Must be less than `batch.len()`.
    position: usize,
}

impl<'a, T> BatchPosition<'a, T> {
    pub fn current(&self) -> &T {
        &self.batch[self.position]
    }

    pub(crate) fn around(&self, count: usize) -> async_tools::Around<'_, T> {
        async_tools::around(self.batch, self.position, count)
    }

    pub fn all(&self) -> &[T] {
        self.batch
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn len(&self) -> usize {
        self.batch.len()
    }
}

impl<T> Clone for BatchPosition<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for BatchPosition<'_, T> {}

//----------//
// Balander //
//----------//

/// A utility for dynamically processing elements in a slice `T` across multiple threads.
///
/// This works by using a local `AtomicUsize` to track which indices inside the slice have
/// been processed and is useful when the work to be performed on each `T` is not suitable
/// for static partitioning.
///
/// This can occur when the work is highly variable.
#[derive(Debug)]
pub struct Balancer<T, C> {
    items: C,
    current: AtomicUsize,
    _type: PhantomData<T>,
}

impl<T, C> Balancer<T, C>
where
    C: AsRef<[T]>,
{
    pub fn new(items: C) -> Self {
        Self {
            items,
            current: AtomicUsize::new(0),
            _type: PhantomData,
        }
    }

    /// Retrieve the next unclamied item and return a tuple `(item, index)` containing said
    /// item and its position in the underlying slice.
    ///
    /// If all items have been claimed, return `None`.
    ///
    /// This function yields all items in the underlying slice exactly once.
    pub fn next(&self) -> Option<BatchPosition<'_, T>> {
        let i = self.current.fetch_add(1, Ordering::Relaxed);
        let items = self.items.as_ref();

        // There is technically a race here where enough concurrent calls to `next` in
        // between the retrieval of `i` and "putting it back" that the counter wraps around.
        //
        // The odds of that happening, however, are exceedingly rare.
        match items.get(i) {
            Some(v) => Some(BatchPosition {
                batch: items,
                position: i,
            }),
            None => {
                // This branch will never be hit in a realistic program.
                if i == usize::MAX {
                    std::process::abort();
                }

                self.current.fetch_sub(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Return the number of work items.
    pub(crate) fn len(&self) -> usize {
        self.items.as_ref().len()
    }
}
