/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{error::StandardError, utils::VectorRepr, ANNResult};
use diskann_vector::{distance::Metric, DistanceFunction};

pub(crate) mod full;
pub use full::Full;

pub trait Distance: Send + Sync + std::fmt::Debug {
    fn evaluate(&self, x: &[u8], y: &[u8]) -> ANNResult<f32>;
}

pub trait AsDistance: Send + Sync + std::fmt::Debug {
    fn as_distance(&self) -> &dyn Distance;
}

impl DistanceFunction<&[u8], &[u8]> for &dyn Distance {
    fn evaluate_similarity(&self, x: &[u8], y: &[u8]) -> f32 {
        self.evaluate(x, y).unwrap()
    }
}

pub trait QueryDistance: Send + Sync + std::fmt::Debug {
    fn evaluate(&self, x: &[u8]) -> ANNResult<f32>;
}

pub trait Layer: Send + Sync + 'static {
    /// Return the number of bytes needed by this layer representation.
    ///
    /// To be well-behaved, this function must be idempotent.
    fn bytes(&self) -> usize;
}

pub trait Set<T>: Layer {
    /// Write into the stored representation.
    fn into_bytes<'a>(&self, element: T, bytes: &'a mut [u8]) -> ANNResult<()>;
}

// Meta traits for `Search` and `Insert` compatibility.
pub trait Search<'a, T>: Send + Sync + 'static {
    fn query_distance(&'a self, query: T) -> ANNResult<Box<dyn QueryDistance + 'a>>;
}

pub trait Insert<'a, T>: Search<'a, T> + Set<T> + AsDistance {}

