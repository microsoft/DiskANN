/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::error::StandardError;

pub trait Distance {
    type Error: StandardError;
    fn evaluate(&mut self, x: &[u8], y: &[u8]) -> Result<f32, Self::Error>;
}

pub trait AsDistance {
    type Distance: Distance;

    fn as_distance(&self) -> Self::Distance;
}

pub trait QueryDistance {
    type Error: StandardError;
    fn evaluate(&mut self, x: &[u8]) -> Result<f32, Self::Error>;
}

pub trait AsQueryDistance<T> {
    type QueryDistance: QueryDistance;
    fn as_query_distance(&self, query: T) -> Self::QueryDistance;
}

pub trait Set<T> {
    /// Return the number of bytes needed by this layer representation.
    ///
    /// To be well-behaved, this function must be idempotent.
    fn bytes(&self) -> usize;

    /// Write into the stored representation.
    fn into_bytes<'a>(&self, element: T, bytes: &'a mut [u8]);
}

