/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt::Display;

use crate::neighbor::Neighbor;

/// An logger provided to various search tasks
pub trait SearchRecord<T>: Send + Sync + 'static
where
    T: Default + Eq,
{
    /// Provides a customization point for logging done during search.
    ///
    /// # Parameters
    /// - `neighbor`: The neighbor node being recorded.
    /// - `hops`: The total number of hops taken to reach this neighbor.
    /// - `cmps`: The total number of comparisons performed to reach this neighbor.
    ///
    /// # Default Implementation
    /// The default implementation of this method is a noop, as in most contexts logging is not required.
    ///
    /// # Type Parameters
    /// - `T`: The data type associated with the neighbor.
    fn record(&mut self, _neighbor: Neighbor<T>, _hops: u32, _cmps: u32) {
        // Default no-op implementation
    }
}

//////////////////////
// NoopSearchRecord //
//////////////////////

/// A empty struct implementing `SearchRecord`.
///
/// Used for situations where a search record is not needed.
/// This is most common for production code where logging is not required, outside of index building.
#[derive(Default)]
pub struct NoopSearchRecord;

impl Display for NoopSearchRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "noop search record")
    }
}

impl NoopSearchRecord {
    pub fn new() -> Self {
        NoopSearchRecord
    }
}

impl<T> SearchRecord<T> for NoopSearchRecord where T: Default + Eq {}

#[derive(Default)]
pub struct VisitedSearchRecord<T>
where
    T: Default + Eq + Clone + Send + Sync + 'static,
{
    pub visited: Vec<Neighbor<T>>,
}

impl<T> std::fmt::Display for VisitedSearchRecord<T>
where
    T: Default + Eq + Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "visited search record")
    }
}

impl<T> VisitedSearchRecord<T>
where
    T: Default + Eq + Clone + Send + Sync + 'static,
{
    pub fn new(initial_reservation: usize) -> Self {
        Self {
            visited: Vec::with_capacity(initial_reservation),
        }
    }

    pub fn push(&mut self, neighbor: Neighbor<T>) {
        self.visited.push(neighbor);
    }
}

impl<T> SearchRecord<T> for VisitedSearchRecord<T>
where
    T: Default + Eq + Clone + Send + Sync + 'static,
{
    fn record(&mut self, neighbor: Neighbor<T>, _hops: u32, _cmps: u32) {
        self.push(neighbor);
    }
}

pub struct RecallSearchRecord<T>
where
    T: Default + Eq + Clone + Send + Sync + 'static,
{
    groundtruth: Vec<T>,
    running_recall: usize,

    pub hops: Vec<u32>,
    pub recall: Vec<usize>,
}

impl<T> std::fmt::Display for RecallSearchRecord<T>
where
    T: Default + Eq + Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "recall search record")
    }
}

impl<T> RecallSearchRecord<T>
where
    T: Default + Eq + Clone + Send + Sync + 'static,
{
    pub fn new(initial_reservation: usize, groundtruth: Vec<T>) -> Self {
        Self {
            groundtruth,
            running_recall: 0,
            hops: Vec::with_capacity(initial_reservation),
            recall: Vec::with_capacity(initial_reservation),
        }
    }

    pub fn push(&mut self, neighbor: Neighbor<T>, hops: u32) {
        self.hops.push(hops);
        if self.groundtruth.contains(&neighbor.id) {
            self.running_recall += 1;
        }
        self.recall.push(self.running_recall);
    }
}

impl<T> SearchRecord<T> for RecallSearchRecord<T>
where
    T: Default + Eq + Clone + Send + Sync + 'static,
{
    fn record(&mut self, neighbor: Neighbor<T>, hops: u32, _cmps: u32) {
        self.push(neighbor, hops);
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    ////////////////////
    // DefaultContext //
    ////////////////////

    #[test]
    fn test_default_record() {
        let record = NoopSearchRecord;

        // Check that the implementation of `Display` is correct.
        assert_eq!(record.to_string(), "noop search record");

        assert_eq!(
            std::mem::size_of::<NoopSearchRecord>(),
            0,
            "expected NoopSearchRecord to be an empty class"
        );
    }

    #[test]
    fn test_default_search_record() {
        let mut record = NoopSearchRecord::new();
        record.record(Neighbor::new(1, 2.0), 2, 3);
    }

    /////////////////////////
    // VisitedSearchRecord //
    /////////////////////////

    #[test]
    fn test_visited_search_record() {
        let record: VisitedSearchRecord<u32> = VisitedSearchRecord::new(1);

        // Check that the implementation of `Display` is correct.
        assert_eq!(record.to_string(), "visited search record");
    }

    #[test]
    fn test_visited_search_record_logging() {
        let mut record = VisitedSearchRecord::new(1);
        record.push(Neighbor::new(4, 5.0));
        record.record(Neighbor::new(1, 2.0), 2, 3);

        assert_eq!(
            record.visited.len(),
            2,
            "Expected two neighbors to be logged"
        );
    }

    ////////////////////////
    // RecallSearchRecord //
    ////////////////////////

    #[test]
    fn test_recall_search_record() {
        let record: RecallSearchRecord<u32> = RecallSearchRecord::new(1, vec![1, 2, 3]);

        // Check that the implementation of `Display` is correct.
        assert_eq!(record.to_string(), "recall search record");
    }

    #[test]
    fn test_recall_search_record_logging() {
        let mut record = RecallSearchRecord::new(1, vec![1, 2, 3]);
        record.record(Neighbor::new(4, 5.0), 1, 4);
        record.record(Neighbor::new(1, 2.0), 2, 3);

        assert_eq!(record.hops.len(), 2, "Expected two hop entries");
        assert_eq!(record.recall.len(), 2, "Expected two recall entries");

        assert_eq!(record.recall[0], 0, "Expected first recall to be 0");
        assert_eq!(record.recall[1], 1, "Expected second recall to be 1");
    }
}
