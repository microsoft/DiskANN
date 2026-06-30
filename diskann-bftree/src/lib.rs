/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! BfTree-based data provider for DiskANN async indexes.
//!
//! This crate provides a [`BfTree`](bf_tree::BfTree)-backed implementation of the DiskANN
//! [`DataProvider`](diskann::provider::DataProvider) trait, enabling indexes that can
//! transparently spill to disk for datasets larger than available memory.

pub mod neighbors;
pub mod provider;
pub mod quant;
pub mod vectors;

// Accessors
pub use provider::{
    AsVectorDtype, BfTreePaths, BfTreeProvider, BfTreeProviderParameters, CreateQuantProvider,
    FullAccessor, GraphParams, Hidden, QuantAccessor, StartPoint, VectorDtype,
};

pub use bf_tree::Config;

use diskann::{
    error::{RankedError, TransientError},
    ANNError,
};

#[derive(Debug, Clone, Copy)]
pub struct NoStore;

/// Wrapper around [`bf_tree::ConfigError`] that implements [`std::error::Error`].
#[derive(Debug, Clone)]
pub struct ConfigError(pub bf_tree::ConfigError);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BfTree configuration error: {:?}", self.0)
    }
}

impl std::error::Error for ConfigError {}

impl From<ConfigError> for ANNError {
    #[track_caller]
    #[inline(never)]
    fn from(error: ConfigError) -> ANNError {
        ANNError::new(diskann::ANNErrorKind::IndexError, error)
    }
}

////////////
// Errors //
////////////
#[derive(Debug)]
pub enum VectorError {
    /// the vector has been explicitly deleted
    Deleted,
    /// the key was not found
    NotFound,
}

#[derive(Debug)]
pub struct VectorUnavailable {
    pub id: usize,
    pub err: VectorError,
}

impl std::fmt::Display for VectorUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.err {
            VectorError::Deleted => write!(f, "vector {} was deleted", self.id),
            VectorError::NotFound => write!(f, "vector {} not found", self.id),
        }
    }
}

impl TransientError<ANNError> for VectorUnavailable {
    fn acknowledge<D>(self, _why: D)
    where
        D: std::fmt::Display,
    {
        // no-op: we are expecting transient deletion errors during traversal
    }

    fn escalate<D>(self, why: D) -> ANNError
    where
        D: std::fmt::Display,
    {
        ANNError::log_index_error(format!("{self}, escalated: {why}"))
    }
}

pub type AccessError = RankedError<VectorUnavailable, ANNError>;

/// Metrics recorded by [`DefaultContext`](diskann::provider::DefaultContext).
#[derive(Debug, Clone)]
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
pub struct ContextMetrics {
    pub spawns: usize,
    pub clones: usize,
}

/// An atomic call counter used for test instrumentation.
///
/// Under `#[cfg(test)]`, this is a real atomic counter. In production builds,
/// all methods are no-ops that the compiler can eliminate entirely.
#[cfg(test)]
pub(crate) struct TestCallCount {
    count: std::sync::atomic::AtomicUsize,
}

#[cfg(test)]
impl TestCallCount {
    pub fn new() -> Self {
        Self {
            count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn enabled() -> bool {
        true
    }

    pub fn increment(&self) {
        self.count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn get(&self) -> usize {
        self.count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(not(test))]
#[allow(dead_code)]
pub(crate) struct TestCallCount {}

#[cfg(not(test))]
#[allow(dead_code)]
impl TestCallCount {
    pub fn new() -> Self {
        Self {}
    }

    pub fn enabled() -> bool {
        false
    }

    pub fn increment(&self) {}

    pub fn get(&self) -> usize {
        0
    }
}

impl Default for TestCallCount {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn vector_unavailable_display_variants() {
        let deleted = VectorUnavailable {
            id: 7,
            err: VectorError::Deleted,
        };
        assert_eq!(deleted.to_string(), "vector 7 was deleted");

        let not_found = VectorUnavailable {
            id: 9,
            err: VectorError::NotFound,
        };
        assert_eq!(not_found.to_string(), "vector 9 not found");
    }

    #[test]
    fn vector_unavailable_acknowledge_is_noop() {
        let transient = VectorUnavailable {
            id: 1,
            err: VectorError::Deleted,
        };
        // Acknowledging a transient deletion swallows it without producing an error.
        transient.acknowledge("expected during traversal");
    }

    #[test]
    fn vector_unavailable_escalate_produces_error() {
        let transient = VectorUnavailable {
            id: 3,
            err: VectorError::NotFound,
        };
        let escalated: ANNError = transient.escalate("lookup failed");
        let message = escalated.to_string();
        assert!(message.contains("vector 3 not found"), "got: {message}");
        assert!(message.contains("lookup failed"), "got: {message}");
    }
}
