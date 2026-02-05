/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{ops::Range, sync::Arc};

use diskann::utils::IntoUsize;
use diskann_benchmark_core::{
    recall::Rows,
    streaming::{self, executors},
};
use diskann_benchmark_runner::{timed, utils::MicroSeconds};
use diskann_utils::views::{Matrix, MatrixView};

use crate::utils::streaming::TagSlotManager;

/// A layer trait for inter-operating with the [`Managed`] index.
///
/// The [`Managed`] layer is responsible for mapping external IDs (tags) to internal slots
/// used by the in-mem index. It closely mirrorws [`streaming::Stream`], but is slightly
/// simplified and delegates the responsibility of requesting maintenance to [`Managed`].
///
/// This trait is purposely kept dyn-compatible to help with compile times somewhat.
pub(crate) trait ManagedStream<T> {
    /// See: [`steraming::Stream::Output`].
    type Output;

    /// See: [`streaming::Stream::search`].
    fn search(
        &self,
        queries: Arc<Matrix<T>>,
        groundtruth: &dyn Rows<u32>,
    ) -> anyhow::Result<Self::Output>;

    /// See: [`streaming::Stream::insert`].
    fn insert(&self, data: MatrixView<'_, T>, slots: &[u32]) -> anyhow::Result<Self::Output>;

    /// See: [`streaming::Stream::replace`].
    fn replace(&self, data: MatrixView<'_, T>, slots: &[u32]) -> anyhow::Result<Self::Output>;

    /// See: [`streaming::Stream::delete`].
    fn delete(&self, slots: &[u32]) -> anyhow::Result<Self::Output>;

    /// See: [`streaming::Stream::maintain`].
    fn maintain(&self) -> anyhow::Result<Self::Output>;
}

pub(crate) struct Managed<T, O> {
    // Temporary book-keeping for ID management.
    book_keeping: TagSlotManager,

    /// Buffer for storing ground truth IDs that have been translated to internal slot IDs.
    translated: Vec<Vec<u32>>,

    /// Perform maintenance when the nubmer of deleted entries is greather than `threshold`
    /// times the number of active slots.
    threshold: f32,

    /// The underlying managed stream.
    stream: Box<dyn ManagedStream<T, Output = O>>,
}

impl<T, O> Managed<T, O> {
    /// Construct a new [`Managed`] layer capable of managing up to `max` points.
    ///
    /// When the number of deleted elements exceeds `threshold` times the number of active
    /// elements, consolidation will be triggered.
    pub(crate) fn new(
        max: usize,
        threshold: f32,
        stream: impl ManagedStream<T, Output = O> + 'static,
    ) -> Self {
        Self {
            book_keeping: TagSlotManager::new(max),
            translated: Vec::new(),
            threshold,
            stream: Box::new(stream),
        }
    }
}

impl<T, O> streaming::Stream<executors::bigann::DataArgs<T, u32>> for Managed<T, O>
where
    T: 'static,
    O: 'static,
{
    type Output = Stats<O>;

    fn search(
        &mut self,
        (queries, groundtruth): (Arc<Matrix<T>>, &dyn Rows<u32>),
    ) -> anyhow::Result<Self::Output> {
        // Translate the groundtruth to the appropriate internal IDs.
        let (overhead, _): (_, ()) = timed! {
            self.translated.resize(groundtruth.nrows(), Vec::new());
            for (i, translated) in self.translated.iter_mut().enumerate() {
                translated.clear();
                for tag in groundtruth.row(i) {
                    if let Some(slot_id) = self.book_keeping.tag_to_slot.get(&tag.into_usize()) {
                        translated.push(*slot_id)
                    } else {
                        anyhow::bail!("Tag {} not found in tag-to-slot mapping", tag);
                    }
                }
            };
        };

        self.stream
            .search(queries, &self.translated)
            .map(|r| Stats::new(overhead, r))
    }

    fn insert(
        &mut self,
        (data, tags): (MatrixView<'_, T>, Range<usize>),
    ) -> anyhow::Result<Self::Output> {
        let (overhead_get, slots) = timed!(self.book_keeping.get_n_empty_slots(tags.len())?);
        let output = self.stream.insert(data, &slots)?;
        let (overhead_assign, _) = timed!(self.book_keeping.assign_slots_to_tags(tags, slots)?);

        Ok(Stats::new(overhead_get + overhead_assign, output))
    }

    fn replace(
        &mut self,
        (data, tags): (MatrixView<'_, T>, Range<usize>),
    ) -> anyhow::Result<Self::Output> {
        let (overhead, slots) = timed!(self.book_keeping.find_slots_by_tags(tags)?);
        self.stream
            .replace(data, &slots)
            .map(|r| Stats::new(overhead, r))
    }

    fn delete(&mut self, tags: Range<usize>) -> anyhow::Result<Self::Output> {
        let (overhead_slots, slots) = timed!(self.book_keeping.find_slots_by_tags(tags.clone())?);
        let output = self.stream.delete(&slots)?;
        let (overhead_mark, _) = timed!(self.book_keeping.mark_tags_deleted(tags)?);
        Ok(Stats::new(overhead_slots + overhead_mark, output))
    }

    fn maintain(&mut self, _: ()) -> anyhow::Result<Self::Output> {
        let output = self.stream.maintain()?;
        let (overhead, _) = timed!(self.book_keeping.consolidate());
        Ok(Stats::new(overhead, output))
    }

    fn needs_maintenance(&mut self) -> bool {
        let num_active = self.book_keeping.num_active();
        let threshold = (num_active as f32 * self.threshold) as usize;
        self.book_keeping.num_deleted() > threshold
    }
}

/// A [`Stream::Output`] wrapper for [`Managed`] that includes the time spent doing
/// operations in the [`TagSlotManager`].
#[derive(Debug, serde::Serialize)]
pub(crate) struct Stats<T> {
    /// Time spent in the manager layer.
    pub(crate) manager_overhead: MicroSeconds,

    /// The inner output from the underlying stream.
    #[serde(flatten)]
    pub(crate) inner: T,
}

impl<T> Stats<T> {
    fn new(manager_overhead: MicroSeconds, inner: T) -> Self {
        Self {
            manager_overhead,
            inner,
        }
    }

    /// Returns a reference to the inner output.
    pub(crate) fn inner(&self) -> &T {
        &self.inner
    }
}

impl<T> std::fmt::Display for Stats<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Manager Overhead: {}s",
            self.manager_overhead.as_seconds()
        )?;
        self.inner.fmt(f)
    }
}
