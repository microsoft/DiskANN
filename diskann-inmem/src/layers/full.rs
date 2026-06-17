/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{ANNError, ANNResult};
use diskann_vector::{
    UnalignedSlice,
    distance::{self, DistanceProvider, Metric},
};
use thiserror::Error;

use crate::{layers, num::Bytes};

/// Full-precision data layer.
#[derive(Debug)]
pub struct Full<T>
where
    T: 'static,
{
    distance: Distance<T>,
    _type: std::marker::PhantomData<T>,
}

impl<T> Full<T>
where
    T: 'static,
{
    pub fn new(dim: usize, metric: Metric) -> Self
    where
        T: DistanceProvider<T>,
    {
        let distance = Distance {
            f: T::distance_comparer(metric, Some(dim)),
            dim,
        };

        Self {
            distance,
            _type: std::marker::PhantomData,
        }
    }

    pub fn dim(&self) -> usize {
        self.distance.dim
    }

    pub fn bytes(&self) -> Bytes {
        Bytes::new(self.dim() * std::mem::size_of::<T>())
    }
}

impl<T> layers::Layer for Full<T>
where
    T: bytemuck::Pod + Send + Sync,
{
    fn bytes(&self) -> Bytes {
        <Full<T>>::bytes(self)
    }
}

impl<T> layers::Set<&[T]> for Full<T>
where
    T: bytemuck::Pod + Send + Sync,
{
    fn into_bytes(&self, v: &[T], bytes: &mut [u8]) -> ANNResult<()> {
        assert_eq!(self.dim(), v.len());
        bytes.copy_from_slice(bytemuck::must_cast_slice::<T, u8>(v));
        Ok(())
    }
}

impl<T> layers::Search for Full<T>
where
    T: std::fmt::Debug + Send + Sync + 'static,
{
    type Query<'a> = &'a [T];

    fn query_distance<'a>(&'a self, query: &'a [T]) -> ANNResult<Box<dyn layers::QueryDistance + 'a>> {
        Ok(Box::new(QueryDistance::new(self.distance, query)))
    }
}

impl<T> layers::AsDistance for Full<T>
where
    T: std::fmt::Debug + Send + Sync + 'static,
{
    fn as_distance(&self) -> &dyn layers::Distance {
        &self.distance
    }
}

impl<T> layers::Insert for Full<T> where
    T: bytemuck::Pod + std::fmt::Debug + Send + Sync
{
}

//////////////
// Distance //
//////////////

#[derive(Debug)]
struct Distance<T>
where
    T: 'static,
{
    f: distance::Distance<T, T>,
    dim: usize,
}

impl<T> Clone for Distance<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Distance<T> {}

impl<T> Distance<T>
where
    T: 'static,
{
    #[cold]
    #[inline(never)]
    fn error(&self, x: &[u8], y: &[u8]) -> ANNResult<f32> {
        let error = DistanceError {
            expected: self.bytes(),
            xlen: x.len(),
            ylen: y.len(),
        };

        Err(ANNError::opaque(error))
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn bytes(&self) -> usize {
        self.dim * std::mem::size_of::<T>()
    }
}

impl<T> layers::Distance for Distance<T>
where
    T: std::fmt::Debug + 'static,
{
    fn evaluate(&self, x: &[u8], y: &[u8]) -> ANNResult<f32> {
        let bytes = self.bytes();
        if x.len() != bytes || y.len() != bytes {
            self.error(x, y)
        } else {
            Ok(self.f.call_unaligned(
                unsafe { UnalignedSlice::new(x.as_ptr().cast::<T>(), self.dim) },
                unsafe { UnalignedSlice::new(y.as_ptr().cast::<T>(), self.dim) },
            ))
        }
    }
}

#[derive(Debug, Error)]
#[error(
    "expected slices of lenght {} - instead got {} and {}",
    self.expected,
    self.xlen,
    self.ylen
)]
struct DistanceError {
    expected: usize,
    xlen: usize,
    ylen: usize,
}

///////////////////
// QueryDistance //
///////////////////

#[derive(Debug)]
struct QueryDistance<'a, T>
where
    T: 'static,
{
    distance: Distance<T>,
    query: &'a [T],
}

impl<'a, T> QueryDistance<'a, T>
where
    T: 'static,
{
    fn new(distance: Distance<T>, query: &'a [T]) -> Self {
        if query.len() != distance.dim() {
            panic!("oops");
        }

        Self { distance, query }
    }

    #[cold]
    #[inline(never)]
    fn error(&self, x: &[u8]) -> ANNResult<f32> {
        let error = QueryDistanceError {
            expected: self.distance.bytes(),
            xlen: x.len(),
        };

        Err(ANNError::opaque(error))
    }
}

impl<T> layers::QueryDistance for QueryDistance<'_, T>
where
    T: std::fmt::Debug + Sync + 'static,
{
    fn evaluate(&self, x: &[u8]) -> ANNResult<f32> {
        if x.len() != self.distance.bytes() {
            self.error(x)
        } else {
            Ok(self.distance.f.call_unaligned(
                unsafe { UnalignedSlice::new(self.query.as_ptr().cast::<T>(), self.distance.dim) },
                unsafe { UnalignedSlice::new(x.as_ptr().cast::<T>(), self.distance.dim) },
            ))
        }
    }
}

#[derive(Debug, Error)]
#[error(
    "expected slice of lenght {} - instead got {}",
    self.expected,
    self.xlen,
)]
struct QueryDistanceError {
    expected: usize,
    xlen: usize,
}
