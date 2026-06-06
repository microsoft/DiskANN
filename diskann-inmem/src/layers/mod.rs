/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{error::StandardError, utils::VectorRepr, ANNResult};
use diskann_vector::{distance::Metric, DistanceFunction};

mod full;

pub trait Distance: Send + Sync + std::fmt::Debug {
    fn evaluate(&self, x: &[u8], y: &[u8]) -> ANNResult<f32>;
}

impl DistanceFunction<&[u8], &[u8]> for &dyn Distance {
    fn evaluate_similarity(&self, x: &[u8], y: &[u8]) -> f32 {
        self.evaluate(x, y).unwrap()
    }
}

pub trait AsDistance {
    fn as_distance(&self) -> &dyn Distance;
}

pub trait QueryDistance: Send + Sync + std::fmt::Debug {
    fn evaluate(&self, x: &[u8]) -> ANNResult<f32>;
}

pub trait AsQueryDistance<'a, T> {
    fn as_query_distance(&'a self, query: T) -> ANNResult<Box<dyn QueryDistance + 'a>>;
}

pub trait Set<T> {
    /// Return the number of bytes needed by this layer representation.
    ///
    /// To be well-behaved, this function must be idempotent.
    fn bytes(&self) -> usize;

    /// Write into the stored representation.
    fn into_bytes<'a>(&self, element: T, bytes: &'a mut [u8]) -> ANNResult<()>;
}

// Meta traits for `Search` and `Insert` compatibility.
pub trait Layer: Send + Sync + 'static {}
pub trait Search<'a, T>: Layer + AsQueryDistance<'a, T> {}
pub trait Insert<'a, T>: Search<'a, T> + Set<T> + AsDistance {}
