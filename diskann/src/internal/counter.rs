/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::atomic::{AtomicUsize, Ordering};

/// An atomic event counter.
#[derive(Debug)]
pub(crate) struct Counter(AtomicUsize);

impl Counter {
    /// Construct a new counter with a value of 0.
    pub(crate) const fn new() -> Self {
        Self(AtomicUsize::new(0))
    }

    /// Return the current value of the counter.
    ///
    /// This function cannot be used for synchronization.
    pub(crate) fn value(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }

    /// Increment the counter by 1.
    ///
    /// This function is atomic but non-synchronizing.
    pub(crate) fn increment(&self) {
        self.increment_by(1)
    }

    /// Increment the counter by the provided value.
    ///
    /// This function is atomic but non-synchronizing.
    pub(crate) fn increment_by(&self, value: usize) {
        self.0.fetch_add(value, Ordering::Relaxed);
    }

    /// Return a local counter to record events that will perform a bulk update
    /// on `self` when dropped.
    ///
    /// This helps reduce contention on the shared atomic.
    pub(crate) fn local(&self) -> LocalCounter<'_> {
        LocalCounter::new(self)
    }
}

/// A scoped vection of `Counter` that allows events to be recorded in an exclusive variable
/// and updated in bulk.
///
/// This can save contention on a shared counter.
#[derive(Debug)]
pub(crate) struct LocalCounter<'a> {
    parent: &'a Counter,
    value: usize,
}

impl<'a> LocalCounter<'a> {
    fn new(parent: &'a Counter) -> Self {
        Self { parent, value: 0 }
    }

    /// Increment the local counter by 1.
    pub(crate) fn increment(&mut self) {
        self.increment_by(1)
    }

    /// Increment the local counter by the provided value.
    pub(crate) fn increment_by(&mut self, value: usize) {
        self.value += value
    }
}

// Write back the results to the parent counter.
impl Drop for LocalCounter<'_> {
    fn drop(&mut self) {
        self.parent.increment_by(self.value)
    }
}
