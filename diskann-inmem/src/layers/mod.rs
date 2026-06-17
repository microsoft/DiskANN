/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;
use diskann_vector::DistanceFunction;

use crate::num::Bytes;

pub(crate) mod full;
pub use full::Full;

pub trait AddLifetime: Send + Sync + 'static {
    type Of<'a>: Send + Sync;
}

#[derive(Debug)]
pub struct Slice<T>(std::marker::PhantomData<T>);

impl<T> Slice<T> {
    pub fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<T> Clone for Slice<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Slice<T> {}

impl<T> Default for Slice<T> {
    fn default() -> Self {
        Self::new()
    }
}

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
    fn bytes(&self) -> Bytes;
}

pub trait Set<T>: Layer {
    /// Write into the stored representation.
    fn into_bytes(&self, element: T, bytes: &mut [u8]) -> ANNResult<()>;
}

// Meta traits for `Search` and `Insert` compatibility.
pub trait Search: Send + Sync + 'static {
    type Query<'a>;

    fn query_distance<'a>(
        &'a self,
        query: Self::Query<'a>,
    ) -> ANNResult<Box<dyn QueryDistance + 'a>>;
}

pub trait Insert: Search + for<'a> Set<Self::Query<'a>> + AsDistance {
    fn insert_distance<'a>(
        &'a self,
        query: Self::Query<'a>,
    ) -> ANNResult<Box<dyn QueryDistance + 'a>> {
        self.query_distance(query)
    }
}
