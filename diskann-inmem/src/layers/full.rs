/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt::Debug, marker::PhantomData};

use diskann::{ANNError, ANNResult};
use diskann_vector::{
    UnalignedSlice,
    conversion::SliceCast,
    distance::{
        self, Cosine, CosineNormalized, DistanceProvider, InnerProduct, Metric, Specialize,
        SquaredL2,
    },
};
use diskann_wide::{
    ARCH,
    arch::{Current, FTarget2},
};
use half::f16;
use thiserror::Error;

use crate::{Hidden, layers, num::Bytes};

/// A useful trait bound for types compatible with [`Full`].
///
/// This encompases *everything* required for `Full: layers::Insert` and can be used as
/// a single bound.
pub trait FullPrecision: bytemuck::Pod + std::fmt::Debug + Send + Sync {
    #[doc(hidden)]
    fn __new(_: Hidden, dim: usize, metric: Metric) -> Full<Self>;

    #[doc(hidden)]
    fn __query_distance<'a, V>(
        _: Hidden,
        full: &'a Full<Self>,
        query: &'a [Self],
        visitor: V,
    ) -> ANNResult<V::Output>
    where
        V: layers::QueryVisitor<'a>;
}

/// Full-precision data layer.
#[derive(Debug)]
pub struct Full<T>
where
    T: 'static,
{
    distance: Distance<T>,
    metric: Metric,
}

impl<T> Full<T>
where
    T: 'static,
{
    /// Create a new full-precision layer for data with the given `dim` and `metric`.
    pub fn new(dim: usize, metric: Metric) -> Self
    where
        T: FullPrecision,
    {
        T::__new(Hidden::new(), dim, metric)
    }

    fn from_distance_provider(dim: usize, metric: Metric) -> Self
    where
        T: DistanceProvider<T>,
    {
        let distance = Distance {
            f: T::distance_comparer(metric, Some(dim)),
            dim,
        };

        Self { distance, metric }
    }

    /// Return the logical dimension of the data handled by this [`layers::Layer`].
    pub fn dim(&self) -> usize {
        self.distance.dim
    }

    /// Return the number of bytes of the data handles by this [`layers::Layer`].
    pub fn bytes(&self) -> Bytes {
        Bytes::new(self.dim() * std::mem::size_of::<T>())
    }

    fn check_dim(&self, dim: usize) -> Result<(), QueryDistanceError> {
        if self.dim() != dim {
            Err(QueryDistanceError {
                expected: self.dim(),
                xlen: dim,
            })
        } else {
            Ok(())
        }
    }
}

impl<T> layers::Layer for Full<T>
where
    T: FullPrecision,
{
    fn bytes(&self) -> Bytes {
        <Full<T>>::bytes(self)
    }
}

impl<T> layers::Set<&[T]> for Full<T>
where
    T: FullPrecision,
{
    fn set(&self, v: &[T], bytes: &mut [u8]) -> ANNResult<()> {
        if v.len() != self.dim() {
            Err(ANNError::from(SetError::Dim {
                got: v.len(),
                expected: self.dim(),
            }))
        } else if bytes.len() != self.bytes().value() {
            Err(ANNError::from(SetError::Bytes {
                got: bytes.len(),
                expected: self.bytes().value(),
            }))
        } else {
            bytes.copy_from_slice(bytemuck::must_cast_slice::<T, u8>(v));
            Ok(())
        }
    }
}

#[derive(Debug, Error)]
enum SetError {
    #[error(
        "data of dimension {} does not match full precision layer's dimension {}",
        got,
        expected
    )]
    Dim { got: usize, expected: usize },
    #[error(
        "raw byte slice of length {} does not match expected length {}",
        got,
        expected
    )]
    Bytes { got: usize, expected: usize },
}

crate::opaque!(SetError);

impl<T> layers::AsDistance for Full<T>
where
    T: FullPrecision,
{
    fn as_distance(&self) -> &dyn layers::Distance {
        &self.distance
    }
}

impl<T> layers::Search for Full<T>
where
    T: FullPrecision,
{
    type Query<'a> = &'a [T];

    fn query_distance<'a, V>(&'a self, query: &'a [T], visitor: V) -> ANNResult<V::Output>
    where
        V: layers::QueryVisitor<'a>,
    {
        T::__query_distance(Hidden::new(), self, query, visitor)
    }
}

impl<T> layers::Insert for Full<T> where T: FullPrecision {}

//////////////
// Distance //
//////////////

#[derive(Debug)]
#[doc(hidden)]
pub struct Distance<T, U = T>
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
        self.dim() * std::mem::size_of::<U>()
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
            // SAFETY: We've checked that both `x` and `y` are valid for
            // `size_of::<T>() * self.dim` bytes.
            let ux = unsafe { UnalignedSlice::new(x.as_ptr().cast::<T>(), self.dim) };

            // SAFETY: Same as above
            let uy = unsafe { UnalignedSlice::new(y.as_ptr().cast::<T>(), self.dim) };
            Ok(self.f.call_unaligned(ux, uy))
        }
    }
}

#[derive(Debug, Error)]
#[error(
    "expected slices of length {} - instead got {} and {}",
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

/// A fused query distance based on [`PureDistanceFunction`] to enable inlining of the final
/// distance function (`D`).
///
/// The type of the embedded query (`T`) is distinct from the expected data-set (`U`) to
/// allow `f16` queries to be pre-converted to `f32`, saving on-the-fly conversion that
/// would otherwise be needed.
#[derive(Debug)]
struct QueryDistance<'a, T, U, D> {
    query: Calf<'a, T>,
    // The type of the data in the original dataset.
    _data: PhantomData<U>,
    // The type of the `PureDistanceFunction` used for the implementation.
    _distance: PhantomData<D>,
}

impl<'a, T, U, D> QueryDistance<'a, T, U, D> {
    fn new(query: Calf<'a, T>) -> Self {
        Self {
            query,
            _data: PhantomData,
            _distance: PhantomData,
        }
    }

    fn bytes(&self) -> usize {
        std::mem::size_of::<U>() * self.query.len()
    }

    #[inline(never)]
    fn error(&self, len: usize) -> ANNResult<f32> {
        let error = QueryDistanceError {
            expected: self.bytes(),
            xlen: len,
        };

        Err(ANNError::opaque(error))
    }
}

impl<T, U, D> layers::QueryDistance for QueryDistance<'_, T, U, D>
where
    T: Send + Sync + 'static + Debug,
    U: Send + Sync + 'static + Debug,
    D: for<'a> FTarget2<Current, f32, UnalignedSlice<'a, T>, UnalignedSlice<'a, U>>
        + Send
        + Sync
        + Debug,
{
    #[inline(always)]
    fn evaluate(&self, x: &[u8]) -> ANNResult<f32> {
        if x.len() != self.bytes() {
            self.error(x.len())
        } else {
            // SAFETY: We've validated that `x` has the correct length.
            let x = unsafe { UnalignedSlice::new(x.as_ptr().cast::<U>(), self.query.len()) };
            Ok(D::run(ARCH, (*self.query).into(), x))
        }
    }
}

#[derive(Debug, Error)]
#[error(
    "expected slice of length {} - instead got {}",
    self.expected,
    self.xlen,
)]
struct QueryDistanceError {
    expected: usize,
    xlen: usize,
}

crate::opaque!(QueryDistanceError);

macro_rules! mint {
    ($query:ident, $visitor:ident, $T:ty => { $N:literal, $f:ident }) => {{
        mint!($query, $visitor, { $T, $T } => { $N, $f })
    }};
    ($query:ident, $visitor:ident, { $T:ty, $U:ty } => { $N:literal, $f:ident }) => {{
        let inner = QueryDistance::<$T, $U, Specialize<$N, $f>>::new($query);
        $visitor.visit_sized::<{ $N * std::mem::size_of::<$U>() }, _>(inner)
    }};
    ($query:ident, $visitor:ident, $T:ty => $f:ident) => {{
        mint!($query, $visitor, { $T, $T } => $f)
    }};
    ($query:ident, $visitor:ident, { $T:ty, $U:ty } => $f:ident) => {{
        let inner = QueryDistance::<$T, $U, $f>::new($query);
        $visitor.visit(inner)
    }};
}

impl FullPrecision for f32 {
    fn __new(_: Hidden, dim: usize, metric: Metric) -> Full<f32> {
        Full::from_distance_provider(dim, metric)
    }

    fn __query_distance<'a, V>(
        _: Hidden,
        full: &'a Full<f32>,
        query: &'a [f32],
        visitor: V,
    ) -> ANNResult<V::Output>
    where
        V: layers::QueryVisitor<'a>,
    {
        full.check_dim(query.len())?;

        let query = Calf::Borrowed(query);

        let output = match full.metric {
            Metric::L2 => {
                if full.dim() == 100 {
                    mint!(query, visitor, f32 => { 100, SquaredL2 })
                } else {
                    mint!(query, visitor, f32 => SquaredL2)
                }
            }
            Metric::InnerProduct => {
                mint!(query, visitor, f32 => InnerProduct)
            }
            Metric::Cosine => mint!(query, visitor, f32 => Cosine),
            Metric::CosineNormalized => mint!(query, visitor, f32 => CosineNormalized),
        };

        Ok(output)
    }
}

impl FullPrecision for f16 {
    fn __new(_: Hidden, dim: usize, metric: Metric) -> Full<f16> {
        Full::from_distance_provider(dim, metric)
    }

    fn __query_distance<'a, V>(
        _: Hidden,
        full: &'a Full<f16>,
        query: &'a [f16],
        visitor: V,
    ) -> ANNResult<V::Output>
    where
        V: layers::QueryVisitor<'a>,
    {
        full.check_dim(query.len())?;

        let mut as_f32: Box<[f32]> = std::iter::repeat_n(0.0, full.dim()).collect();
        diskann_wide::arch::dispatch2(SliceCast::new(), &mut *as_f32, query);
        let query = Calf::Owned(as_f32);

        let output = match full.metric {
            Metric::L2 => {
                if full.dim() == 100 {
                    mint!(query, visitor, { f32, f16 } => { 100, SquaredL2 })
                } else {
                    mint!(query, visitor, { f32, f16 } => SquaredL2)
                }
            }
            Metric::InnerProduct => mint!(query, visitor, { f32, f16 } => InnerProduct),
            Metric::Cosine => mint!(query, visitor, { f32, f16 } => Cosine),
            Metric::CosineNormalized => mint!(query, visitor, { f32, f16 } => CosineNormalized),
        };

        Ok(output)
    }
}

impl FullPrecision for u8 {
    fn __new(_: Hidden, dim: usize, metric: Metric) -> Full<u8> {
        Full::from_distance_provider(dim, metric)
    }

    fn __query_distance<'a, V>(
        _: Hidden,
        full: &'a Full<u8>,
        query: &'a [u8],
        visitor: V,
    ) -> ANNResult<V::Output>
    where
        V: layers::QueryVisitor<'a>,
    {
        full.check_dim(query.len())?;

        let query = Calf::Borrowed(query);

        let output = match full.metric {
            Metric::L2 => {
                if full.dim() == 128 {
                    mint!(query, visitor, u8 => { 128, SquaredL2 })
                } else {
                    mint!(query, visitor, u8 => SquaredL2)
                }
            }
            Metric::InnerProduct => mint!(query, visitor, u8 => InnerProduct),
            Metric::Cosine => mint!(query, visitor, u8 => Cosine),
            Metric::CosineNormalized => mint!(query, visitor, u8 => Cosine),
        };

        Ok(output)
    }
}

impl FullPrecision for i8 {
    fn __new(_: Hidden, dim: usize, metric: Metric) -> Full<i8> {
        Full::from_distance_provider(dim, metric)
    }

    fn __query_distance<'a, V>(
        _: Hidden,
        full: &'a Full<i8>,
        query: &'a [i8],
        visitor: V,
    ) -> ANNResult<V::Output>
    where
        V: layers::QueryVisitor<'a>,
    {
        full.check_dim(query.len())?;

        let query = Calf::Borrowed(query);

        let output = match full.metric {
            Metric::L2 => mint!(query, visitor, i8 => SquaredL2),
            Metric::InnerProduct => mint!(query, visitor, i8 => InnerProduct),
            Metric::Cosine => mint!(query, visitor, i8 => Cosine),
            Metric::CosineNormalized => mint!(query, visitor, i8 => Cosine),
        };

        Ok(output)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
#[cfg(not(miri))]
mod tests {
    use std::fmt::Display;

    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;
    // Bring the inherent-call traits into method scope. The `Distance` / `QueryDistance`
    // traits are not imported: their methods are reached through `&dyn _` trait objects,
    // which does not require the trait to be in scope.
    use crate::layers::{AsDistance as _, QueryVisitor, Search as _, Set as _};

    /// Generate random elements of a layer's data type from a seeded RNG.
    trait Sample: bytemuck::Pod {
        fn sample<R: Rng>(rng: &mut R) -> Self;
    }

    impl Sample for f32 {
        fn sample<R: Rng>(rng: &mut R) -> Self {
            rng.random_range(-1.0f32..1.0f32)
        }
    }

    impl Sample for f16 {
        fn sample<R: Rng>(rng: &mut R) -> Self {
            f16::from_f32(rng.random_range(-1.0f32..1.0f32))
        }
    }

    impl Sample for u8 {
        fn sample<R: Rng>(rng: &mut R) -> Self {
            rng.random()
        }
    }

    impl Sample for i8 {
        fn sample<R: Rng>(rng: &mut R) -> Self {
            rng.random()
        }
    }

    fn gen_vec<T: Sample, R: Rng>(rng: &mut R, dim: usize) -> Vec<T> {
        (0..dim).map(|_| T::sample(rng)).collect()
    }

    /// A [`QueryVisitor`] that simply boxes the minted kernel so the test can probe it
    /// directly. Exercises both `visit` (dynamic) and `visit_sized` (specialized) paths.
    struct Collect;

    impl<'a> QueryVisitor<'a> for Collect {
        type Output = Box<dyn layers::QueryDistance + 'a>;

        fn visit<Q>(self, distance: Q) -> Self::Output
        where
            Q: layers::QueryDistance + 'a,
        {
            Box::new(distance)
        }
    }

    /// Compare two distances allowing for floating-point reassociation between the
    /// specialized / converted kernels and the dynamic reference.
    fn approx_eq(got: f32, want: f32) -> bool {
        (got - want).abs() <= 1e-3 + 1e-4 * want.abs()
    }

    /// Exercise every `Full<T>` API across dimensions `1..=max_dim`.
    ///
    /// For each dimension we check that `bytes`/`set` agree, that `distance` and
    /// `query_distance` are consistent with `DistanceProvider`, and that all of these
    /// reject byte slices that are too long or too short.
    fn test_impl<T>(max_dim: usize, ctx: &dyn Display)
    where
        T: FullPrecision + Sample + DistanceProvider<T>,
    {
        let mut rng = StdRng::seed_from_u64(0x0D15_0ACE ^ max_dim as u64);
        let metrics = [
            Metric::L2,
            Metric::InnerProduct,
            Metric::Cosine,
            Metric::CosineNormalized,
        ];

        for dim in 1..=max_dim {
            let a = gen_vec::<T, _>(&mut rng, dim);
            let b = gen_vec::<T, _>(&mut rng, dim);

            // `bytes` and `set` agree: the encoded buffer equals the raw cast bytes.
            let layer = Full::<T>::new(dim, Metric::L2);
            assert_eq!(
                layer.bytes().value(),
                dim * std::mem::size_of::<T>(),
                "{ctx}: dim {dim}: unexpected byte length",
            );

            let mut a_bytes = vec![0u8; layer.bytes().value()];
            layer.set(&a, &mut a_bytes).unwrap();
            assert_eq!(
                a_bytes.as_slice(),
                bytemuck::cast_slice::<T, u8>(&a),
                "{ctx}: dim {dim}: set mismatch",
            );

            let mut b_bytes = vec![0u8; layer.bytes().value()];
            layer.set(&b, &mut b_bytes).unwrap();

            for metric in metrics {
                let full = Full::<T>::new(dim, metric);

                // Reference value straight from `DistanceProvider`.
                let reference =
                    <T as DistanceProvider<T>>::distance_comparer(metric, Some(dim)).call(&a, &b);

                // `distance` is built from the same comparer, so it must match exactly.
                let distance = full.as_distance();
                let via_distance = distance.evaluate(&a_bytes, &b_bytes).unwrap();
                assert_eq!(
                    via_distance, reference,
                    "{ctx}: dim {dim}, metric {metric:?}: distance != DistanceProvider",
                );

                // `query_distance` computes the same geometry. Specialized and f16-converted
                // kernels may reassociate the summation, so compare approximately.
                let query = full.query_distance(a.as_slice(), Collect).unwrap();
                let via_query = query.evaluate(&b_bytes).unwrap();
                assert!(
                    approx_eq(via_query, via_distance),
                    "{ctx}: dim {dim}, metric {metric:?}: query {via_query} != distance {via_distance}",
                );

                // Every distance API rejects byte slices that are too long or too short.
                let short = &a_bytes[..a_bytes.len() - 1];
                let mut long = a_bytes.clone();
                long.push(0);

                assert!(distance.evaluate(short, &b_bytes).is_err());
                assert!(distance.evaluate(&long, &b_bytes).is_err());
                assert!(distance.evaluate(&a_bytes, short).is_err());
                assert!(distance.evaluate(&a_bytes, &long).is_err());

                assert!(query.evaluate(short).is_err());
                assert!(query.evaluate(&long).is_err());
            }

            // `set` rejects mis-sized element and buffer slices.
            let mut buf = vec![0u8; layer.bytes().value()];
            let too_many = gen_vec::<T, _>(&mut rng, dim + 1);
            assert!(
                layer.set(&too_many, &mut buf).is_err(),
                "{ctx}: dim {dim}: set accepted an over-long element slice",
            );

            assert!(
                layer.query_distance(&too_many, Collect).is_err(),
                "{ctx}: dim {dim}: incorrect query lengths should be rejected"
            );

            let mut short_buf = vec![0u8; layer.bytes().value().saturating_sub(1)];
            assert!(
                layer.set(&a, &mut short_buf).is_err(),
                "{ctx}: dim {dim}: set accepted an under-sized buffer",
            );

            let too_few = gen_vec::<T, _>(&mut rng, dim - 1);
            assert!(
                layer.query_distance(&too_few, Collect).is_err(),
                "{ctx}: dim {dim}: incorrect query lengths should be rejected"
            );
        }
    }

    // `max_dim` must exceed the largest specialized dimension for each type so the
    // const-generic (`visit_sized`) paths are covered alongside the dynamic ones.
    #[test]
    fn full_f32() {
        test_impl::<f32>(256, &"f32");
    }

    #[test]
    fn full_f16() {
        test_impl::<f16>(256, &"f16");
    }

    #[test]
    fn full_u8() {
        test_impl::<u8>(160, &"u8");
    }

    #[test]
    fn full_i8() {
        test_impl::<i8>(160, &"i8");
    }
}
