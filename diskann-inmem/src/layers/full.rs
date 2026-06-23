/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt::Debug, marker::PhantomData};

use diskann::{ANNError, ANNResult};
use diskann_vector::{
    UnalignedSlice,
    conversion::SliceCast,
    distance::{self, DistanceProvider, InnerProduct, Metric, Specialize, SquaredL2},
};
use diskann_wide::{
    ARCH,
    arch::{Current, FTarget2},
};
use half::f16;
use thiserror::Error;

use crate::{layers, num::Bytes};

/// Full-precision data layer.
#[derive(Debug)]
pub struct Full<T>
where
    T: 'static,
{
    distance: Distance<T>,
    metric: Metric,
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
            metric,
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

impl<T> layers::AsDistance for Full<T>
where
    T: Debug + Send + Sync + 'static,
{
    fn as_distance(&self) -> &dyn layers::Distance {
        &self.distance
    }
}

impl<T> layers::Insert for Full<T>
where
    T: bytemuck::Pod + Debug + Send + Sync + 'static,
    Self: for<'a> layers::Search<Query<'a> = &'a [T]>,
{
}

//////////////
// Distance //
//////////////

#[derive(Debug)]
struct Distance<T, U = T>
where
    T: 'static,
    U: 'static,
{
    f: distance::Distance<T, U>,
    dim: usize,
}

impl<T, U> Clone for Distance<T, U> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, U> Copy for Distance<T, U> {}

impl<T, U> Distance<T, U>
where
    T: 'static,
    U: 'static,
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
    T: Debug + 'static,
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

// A baby [`std::borrow::Cow`].
#[derive(Debug)]
enum Calf<'a, T> {
    Borrowed(&'a [T]),
    Owned(Box<[T]>),
}

impl<T> std::ops::Deref for Calf<'_, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(slice) => slice,
            Self::Owned(boxed) => boxed,
        }
    }
}

impl<'a, T> From<&'a [T]> for Calf<'a, T> {
    fn from(slice: &'a [T]) -> Self {
        Self::Borrowed(slice)
    }
}

impl<T> From<Box<[T]>> for Calf<'_, T> {
    fn from(boxed: Box<[T]>) -> Self {
        Self::Owned(boxed)
    }
}

#[derive(Debug)]
struct QueryDistance<'a, T, U>
where
    T: 'static,
    U: 'static,
{
    distance: Distance<T, U>,
    query: Calf<'a, T>,
}

impl<'a, T, U> QueryDistance<'a, T, U>
where
    T: 'static,
    U: 'static,
{
    fn new(distance: Distance<T, U>, query: Calf<'a, T>) -> Self {
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

impl<T, U> layers::QueryDistance for QueryDistance<'_, T, U>
where
    T: Debug + Sync + Send + 'static,
    U: Debug + Sync + Send + 'static,
{
    fn evaluate(&self, x: &[u8]) -> ANNResult<f32> {
        if x.len() != self.distance.bytes() {
            self.error(x)
        } else {
            Ok(self.distance.f.call_unaligned(
                unsafe { UnalignedSlice::new(self.query.as_ptr().cast::<T>(), self.distance.dim) },
                unsafe { UnalignedSlice::new(x.as_ptr().cast::<U>(), self.distance.dim) },
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

macro_rules! specialize {
    ($me:ident, $query:ident, $visitor:ident, ($T:ty, $U:ty), $(($var:ident, $N:literal, $f:ty)),* $(,)?) => {
        match ($me.metric, $me.dim()) {
            $(
                (Metric::$var, $N) => {
                    let wrapped = Wrap::<Specialize<$N, $f>, $T, $U>::new($query);
                    return Ok(unsafe {
                        $visitor.visit_sized::<{ $N * std::mem::size_of::<$U>() }, _>(wrapped)
                    })
                },
            )*
            _ => {},
        }
    }
}

impl layers::Search for Full<f32> {
    type Query<'a> = &'a [f32];

    fn query_distance<'a, V>(&'a self, query: &'a [f32], visitor: V) -> ANNResult<V::Output>
    where
        V: layers::QueryVisitor<'a>,
    {
        let query = Calf::Borrowed(query);

        specialize!(
            self,
            query,
            visitor,
            (f32, f32),
            (L2, 100, SquaredL2),
            (InnerProduct, 768, InnerProduct),
        );

        // Fallback
        Ok(visitor.visit(QueryDistance::new(self.distance, query)))
    }
}

impl layers::Search for Full<f16> {
    type Query<'a> = &'a [f16];

    fn query_distance<'a, V>(&'a self, query: &'a [f16], visitor: V) -> ANNResult<V::Output>
    where
        V: layers::QueryVisitor<'a>,
    {
        let mut as_f32: Box<[f32]> = std::iter::repeat_n(0.0, self.dim()).collect();
        diskann_wide::arch::dispatch2(SliceCast::new(), &mut *as_f32, query);
        let query = Calf::Owned(as_f32);

        specialize!(
            self,
            query,
            visitor,
            (f32, f16),
            (L2, 100, SquaredL2),
            (InnerProduct, 768, InnerProduct),
        );

        // Fallback
        let distance = Distance {
            f: <f32 as DistanceProvider<f16>>::distance_comparer(self.metric, Some(self.dim())),
            dim: self.dim(),
        };

        Ok(visitor.visit(QueryDistance::new(distance, query)))
    }
}

impl layers::Search for Full<u8> {
    type Query<'a> = &'a [u8];

    fn query_distance<'a, V>(&'a self, query: &'a [u8], visitor: V) -> ANNResult<V::Output>
    where
        V: layers::QueryVisitor<'a>,
    {
        let query = Calf::Borrowed(query);

        specialize!(self, query, visitor, (u8, u8), (L2, 128, SquaredL2));

        // Fallback
        Ok(visitor.visit(QueryDistance::new(self.distance, query)))
    }
}

impl layers::Search for Full<i8> {
    type Query<'a> = &'a [i8];

    fn query_distance<'a, V>(&'a self, query: &'a [i8], visitor: V) -> ANNResult<V::Output>
    where
        V: layers::QueryVisitor<'a>,
    {
        let query = Calf::Borrowed(query);
        Ok(visitor.visit(QueryDistance::new(self.distance, query)))
    }
}

#[derive(Debug)]
struct Wrap<'a, I, T, U> {
    query: Calf<'a, T>,
    ps: PhantomData<(I, U)>,
}

impl<'a, I, T, U> Wrap<'a, I, T, U> {
    fn new(query: Calf<'a, T>) -> Self {
        Self {
            query,
            ps: PhantomData,
        }
    }
}

impl<I, T, U> layers::QueryDistance for Wrap<'_, I, T, U>
where
    I: for<'a> FTarget2<Current, f32, UnalignedSlice<'a, T>, UnalignedSlice<'a, U>>
        + Send
        + Sync
        + Debug,
    T: Send + Sync + 'static + Debug,
    U: Send + Sync + 'static + Debug,
{
    #[inline(always)]
    fn evaluate(&self, x: &[u8]) -> ANNResult<f32> {
        // TODO: This is not fully valid - we need to check.
        let x = unsafe { UnalignedSlice::new(x.as_ptr().cast::<U>(), self.query.len()) };
        Ok(I::run(ARCH, (*self.query).into(), x))
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
}
