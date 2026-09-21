/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # Full-Precision
//!
//! A concurrent data store for [`crate::Provider`] enabling full-precision searches and
//! inserts for collections consisting of `f32`, `f16`, `u8`, or `i8` data types.
//!
//! The [`FullPrecision`] generic bound can be used to constrain these data types.

mod internal_docs {
    //! Internally, the [`super::repr::Search`] and [`super::repr::Insert`] traits
    //! are implemented via [`super::FullPrecisionImpl`], which creates:
    //!
    //! * [`super::ExpandBeam`]: For index search.
    //! * [`super::Prune`]: For index construction.
    //!
    //! These two structs are modular with respect to their exact distance function and
    //! prefetcher. Since [`super::repr::ExpandBeam`] and [`super::repr::Prune`] are
    //! used as trait objects, this allows the implementation structs in this module to be
    //! highly specialized, including:
    //!
    //! * Inlining of distance functions.
    //! * Specializing distance functions on dimension.
    //! * Specializing prefetches on dimension.
    //! * Dispatching to different micro-architecture levels.
    //! * Specialized query preprocessing.
    //!
    //! Picking the best combination of all of these requires extensive experimentation.
    //! The choices made here are mainly heuristic defaults, meant to try to balance
    //! performance with compile time.
    //!
    //! Feel free to experiment and create optimized implementations for workloads that need it.
}

use std::{fmt::Debug, marker::PhantomData, num::NonZeroUsize};

use diskann::{ANNError, ANNResult, utils::IntoUsize};
use diskann_utils::views::Matrix;
use diskann_vector::{
    UnalignedSlice,
    conversion::SliceCast,
    distance::{
        Cosine, CosineNormalized, DistanceProvider, InnerProduct, Metric, Specialize, SquaredL2,
    },
};
use diskann_wide::{
    ARCH,
    arch::{Current, FTarget2},
};
use half::f16;
use thiserror::Error;

use crate::{
    counters::LocalCounters,
    epoch,
    num::{Bytes, Capacity, IdLimit, MaxDegree},
    prefetch::{self, Prefetch},
    repr::{self, internal::Calf},
    store::{
        self, Store,
        intrusive::{self, Intrusive},
    },
    tag::AtomicTag,
};

/// Construct an [`UnalignedSlice`] over `bytes`.
///
/// In release builds, this method truncates `bytes.len()` to a multiple of `size_of::<T>()`.
///
/// Debug builds assert that the length is in fact a multiple.
fn unaligned_from_bytes<T>(bytes: &[u8]) -> UnalignedSlice<'_, T>
where
    T: bytemuck::Pod,
{
    debug_assert!(bytes.len().is_multiple_of(std::mem::size_of::<T>()));

    // SAFETY: The slice `bytes` attests that the memory spanned by
    // `[ptr, ptr.add(size_of::<T>() * (len / size_of::<T>())))` is valid.
    //
    // Since `T: Pod`, all bit patterns are valid, so unaligned loads yield well-defined values.
    unsafe {
        UnalignedSlice::new(
            bytes.as_ptr().cast::<T>(),
            bytes.len() / std::mem::size_of::<T>(),
        )
    }
}

/// A useful trait bound for types compatible with [`Full`].
///
/// This encompasses *everything* required for `Full: repr::Insert` and can be used as
/// a single bound.
pub trait FullPrecision: bytemuck::Pod + std::fmt::Debug + Send + Sync {
    #[doc(hidden)]
    fn __search_accessor<'a>(
        representation: &'a Full<Self>,
        query: &'a [Self],
        provider: &'a (dyn std::any::Any + Send + Sync),
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::SearchAccessor<'a>>;

    #[doc(hidden)]
    fn __prune_accessor<'a>(
        representation: &'a Full<Self>,
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::PruneAccessor<'a>>;
}

/// A configuration struct for [`Full`].
#[derive(Debug, Clone)]
pub struct Config<T> {
    layout: store::Layout,
    metric: Metric,
    start_points: Matrix<T>,
    store: store::Config,
    lookahead: Option<NonZeroUsize>,
}

const DEFAULT_LOOKAHEAD: NonZeroUsize = NonZeroUsize::new(12).unwrap();

impl<T> Config<T> {
    /// Create a new [`Config`] for a [`Full`].
    ///
    /// The resulting store will hold `capacity` writable items and `start_points.nrows()`
    /// frozen points at internal IDs `[capacity, capacity + start_points.nrows())`. The
    /// dimensionality of the full-precision data will be inferred from
    /// `start_points.ncols()`.
    ///
    /// The associated graph will be bounded with `max_degree` and `metric` will be used to
    /// compute distances among the stored points.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of start points exceeds `u32::MAX` or the number of
    /// bytes required for each point exceeds `usize::MAX`.
    pub fn new(
        capacity: Capacity,
        max_degree: MaxDegree,
        metric: Metric,
        start_points: Matrix<T>,
    ) -> Result<Self, ConfigError> {
        let num_start_points: u32 = match start_points.nrows().try_into() {
            Ok(points) => points,
            Err(_) => return Err(ConfigError::TooManyStartPoints(start_points.nrows())),
        };

        // Check that we won't overflow when computing the number of bytes required for each
        // data point. This can happen if `start_points` has 0 rows but a large number of
        // columns.
        if start_points
            .ncols()
            .checked_mul(std::mem::size_of::<T>())
            .is_none()
        {
            return Err(ConfigError::DimTooLarge(start_points.ncols()));
        }

        Ok(Self {
            layout: store::Layout::new(capacity, max_degree, num_start_points),
            metric,
            start_points,
            store: store::Config::default(),
            lookahead: Some(DEFAULT_LOOKAHEAD),
        })
    }

    /// Override the [`store::Config`] for tailoring concurrency details.
    pub fn store(mut self, config: store::Config) -> Self {
        self.store = config;
        self
    }

    /// Set the prefetch lookahead.
    ///
    /// This controls how many iterations ahead in
    /// [`diskann::graph::glue::SearchAccessor::expand_beam`] data is prefetched into the CPU
    /// cache. Passing `None` disables prefetching.
    pub fn prefetch(mut self, lookahead: Option<NonZeroUsize>) -> Self {
        self.lookahead = lookahead;
        self
    }

    /// Return the vector dimension of this configuration and the resulting [`Full`].
    pub fn dim(&self) -> usize {
        self.start_points.ncols()
    }

    /// Construct a [`Full`] from the [`Config`].
    pub fn build(self) -> ANNResult<Full<T>>
    where
        T: FullPrecision,
    {
        Full::new(self)
    }
}

/// Errors that can arise when constructing [`Config`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ConfigError {
    #[error("{} start points exceed `u32::MAX`", 0)]
    TooManyStartPoints(usize),
    #[error(
        "the number of bytes to hold {}-dimensional data exceeds `usize::MAX`",
        0
    )]
    DimTooLarge(usize),
}

diskann::convert_error!(ConfigError);

impl<T> repr::RepresentationConfig for Config<T>
where
    T: FullPrecision,
{
    type Representation = Full<T>;

    fn build(self) -> ANNResult<Full<T>> {
        <Config<T>>::build(self)
    }
}

/// Internal helper for implementing [`FullPrecision`].
trait FullPrecisionImpl: bytemuck::Pod + std::fmt::Debug + Send + Sync {
    fn make_expand_beam<'a>(
        full: &'a Full<Self>,
        query: &'a [Self],
    ) -> ANNResult<Box<dyn repr::ExpandBeam + 'a>>;

    #[doc(hidden)]
    fn make_prune<'a>(full: &'a Full<Self>) -> ANNResult<Box<dyn repr::Prune + 'a>>;
}

/// Full-precision data representation.
#[derive(Debug)]
pub struct Full<T>
where
    T: 'static,
{
    store: Store<Intrusive>,
    metric: Metric,
    lookahead: Option<NonZeroUsize>,
    _type: PhantomData<T>,
}

impl<T> Full<T>
where
    T: 'static,
{
    /// Initialize a [`Config`] for this representation.
    ///
    /// See also: [`Config::new`].
    ///
    /// # Errors
    ///
    /// Returns the errors described by [`Config::new`].
    pub fn config(
        capacity: Capacity,
        max_degree: MaxDegree,
        metric: Metric,
        start_points: Matrix<T>,
    ) -> Result<Config<T>, ConfigError> {
        Config::new(capacity, max_degree, metric, start_points)
    }

    /// Create a new full-precision representation from `config`.
    ///
    /// See: [`Config::build`].
    fn new(config: Config<T>) -> ANNResult<Self>
    where
        T: FullPrecision,
    {
        let Config {
            layout,
            metric,
            start_points,
            store,
            lookahead,
        } = config;

        let bytes = Bytes::new(start_points.ncols() * std::mem::size_of::<T>());
        let intrusive = Intrusive::config(bytes);
        let store = Store::new(layout, store, intrusive)?;

        // Initialize start points.
        for (i, row) in std::iter::zip(store.frozen(), start_points.row_iter()) {
            #[expect(
                clippy::expect_used,
                reason = "failing this is an internal, unrecoverable bug"
            )]
            let mut slot = store
                .slot(i)
                .expect("internal store should leave frozen-points available for writing");
            slot.data()
                .as_mut_slice()
                .copy_from_slice(bytemuck::must_cast_slice::<T, u8>(row));

            slot.freeze();
        }

        Ok(Self {
            store,
            metric,
            lookahead,
            _type: PhantomData,
        })
    }

    /// Return the logical dimension of the data handled by this [`repr::Representation`].
    pub fn dim(&self) -> usize {
        self.bytes().value() / std::mem::size_of::<T>()
    }

    /// Return the number of payload bytes in each stored vector.
    pub fn bytes(&self) -> Bytes {
        self.store.slots().bytes()
    }

    #[cfg(test)]
    fn bytes_plus_tag(&self) -> Bytes {
        self.store.slots().bytes_plus_tag()
    }

    /// Return the [`Metric`] for this representation.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    fn check_dim(&self, dim: usize) -> Result<(), ExpandBeamError> {
        if self.dim() != dim {
            Err(ExpandBeamError {
                expected: self.dim(),
                xlen: dim,
            })
        } else {
            Ok(())
        }
    }

    fn reader(&self) -> Result<intrusive::Reader<'_>, epoch::Unavailable> {
        self.store.guard(|intrusive, guard| intrusive.reader(guard))
    }
}

impl<T> Full<T>
where
    T: FullPrecision,
{
    pub(crate) fn get(&self, i: u32) -> ANNResult<Box<[T]>> {
        let reader = self.reader()?;
        let data = match reader.read(i.into_usize()) {
            Some(data) => data,
            None => {
                return Err(ANNError::message("item could not be read"));
            }
        };

        let mut buf: Box<[_]> = std::iter::repeat_n(T::zeroed(), self.dim()).collect();
        bytemuck::must_cast_slice_mut::<T, u8>(&mut buf).copy_from_slice(data);
        Ok(buf)
    }
}

repr::internal::macros::representation!(
    { T } Full<T> where T: FullPrecision
);

repr::internal::macros::set_guard!(
    /// A [`repr::Guard`] for [`Full`].
    for<'a> intrusive::Exclusive<'a>
);

impl<T> repr::Set<&[T]> for Full<T>
where
    T: FullPrecision,
{
    type Guard<'a> = Guard<'a>;

    fn set(&self, v: &[T]) -> ANNResult<Guard<'_>> {
        self.check_dim(v.len())?;

        let mut slot = self
            .store
            .acquire()
            .ok_or_else(|| ANNError::message("could not allocate a new slot"))?;

        slot.data()
            .as_mut_slice()
            .copy_from_slice(bytemuck::must_cast_slice::<T, u8>(v));

        Ok(Guard::new(slot))
    }
}

impl<T> repr::Search for Full<T>
where
    T: FullPrecision,
{
    type Query<'a> = &'a [T];

    fn search_accessor<'a>(
        &'a self,
        query: Self::Query<'a>,
        provider: &'a (dyn std::any::Any + Send + Sync),
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::SearchAccessor<'a>> {
        T::__search_accessor(self, query, provider, counters)
    }
}

impl<T> repr::Insert for Full<T>
where
    T: FullPrecision,
{
    fn prune_accessor<'a>(
        &'a self,
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::PruneAccessor<'a>> {
        T::__prune_accessor(self, counters)
    }
}

//----------//
// Distance //
//----------//

trait Distance<T, U>: std::fmt::Debug + Send + Sync + 'static {
    fn eval(&self, x: UnalignedSlice<'_, T>, y: UnalignedSlice<'_, U>) -> f32;
}

#[derive(Debug)]
struct Pure<T, U, D>(PhantomData<(T, U, D)>);

impl<T, U, D> Pure<T, U, D> {
    const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T, U, D> Distance<T, U> for Pure<T, U, D>
where
    D: for<'any> FTarget2<Current, f32, UnalignedSlice<'any, T>, UnalignedSlice<'any, U>>
        + std::fmt::Debug
        + Send
        + Sync
        + 'static,
    T: std::fmt::Debug + Send + Sync + 'static,
    U: std::fmt::Debug + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, x: UnalignedSlice<'_, T>, y: UnalignedSlice<'_, U>) -> f32 {
        D::run(ARCH, x, y)
    }
}

impl<T, U, D> repr::internal::RawDistance for Pure<T, U, D>
where
    Self: Distance<T, U>,
    T: bytemuck::Pod + std::fmt::Debug + Send + Sync,
    U: bytemuck::Pod + std::fmt::Debug + Send + Sync,
{
    /// Not techncially true since we panic on length miematches.
    type Error = diskann::error::Infallible;

    #[inline(always)]
    fn eval(&self, x: &[u8], y: &[u8]) -> Result<f32, Self::Error> {
        Ok(Distance::eval(
            self,
            unaligned_from_bytes(x),
            unaligned_from_bytes(y),
        ))
    }
}

impl<T, U> Distance<T, U> for diskann_vector::distance::Distance<T, U>
where
    T: std::fmt::Debug + 'static,
    U: std::fmt::Debug + 'static,
{
    #[inline(always)]
    fn eval(&self, x: UnalignedSlice<'_, T>, y: UnalignedSlice<'_, U>) -> f32 {
        self.call_unaligned(x, y)
    }
}

impl<T, U> repr::internal::RawDistance for diskann_vector::distance::Distance<T, U>
where
    T: bytemuck::Pod + std::fmt::Debug + 'static,
    U: bytemuck::Pod + std::fmt::Debug + 'static,
{
    /// Not techncially true since we panic on length miematches.
    type Error = diskann::error::Infallible;

    #[inline(always)]
    fn eval(&self, x: &[u8], y: &[u8]) -> Result<f32, Self::Error> {
        Ok(Distance::eval(
            self,
            unaligned_from_bytes(x),
            unaligned_from_bytes(y),
        ))
    }
}

#[derive(Debug)]
pub(super) struct QueryDistance<'a, T, U, D> {
    query: Calf<'a, [T]>,
    distance: D,
    _cast: PhantomData<U>,
}

impl<'a, T, U, D> QueryDistance<'a, T, U, D> {
    pub(super) fn new(query: Calf<'a, [T]>, distance: D) -> Self {
        Self {
            query,
            distance,
            _cast: PhantomData,
        }
    }
}

impl<T, U, D> repr::internal::RawQueryDistance for QueryDistance<'_, T, U, D>
where
    D: Distance<T, U>,
    T: std::fmt::Debug + Send + Sync + 'static,
    U: bytemuck::Pod + std::fmt::Debug + Send + Sync,
{
    /// Not techncially true since we panic on length miematches.
    type Error = diskann::error::Infallible;

    #[inline(always)]
    fn eval(&self, x: &[u8]) -> Result<f32, Self::Error> {
        Ok(self
            .distance
            .eval(UnalignedSlice::from(&*self.query), unaligned_from_bytes(x)))
    }
}

//----------------------//
// Expand Beam (Search) //
//----------------------//

/// A temporary precursor for [`ExpandBeam`] to simplify macros.
#[derive(Debug)]
struct IntoExpandBeam<'a, T, U> {
    query: Calf<'a, [T]>,
    reader: store::intrusive::Reader<'a>,
    lookahead: Option<NonZeroUsize>,
    _data: PhantomData<U>,
}

impl<'a, T, U> IntoExpandBeam<'a, T, U> {
    /// Construct a new [`IntoExpandBeam`], validating the query dimension and acquiring a
    /// reader for `full`.
    fn new(full: &'a Full<U>, query: Calf<'a, [T]>) -> ANNResult<Self> {
        full.check_dim(query.len())?;
        let reader = full.reader()?;
        let lookahead = full.lookahead;
        Ok(Self {
            query,
            reader,
            lookahead,
            _data: PhantomData,
        })
    }

    fn into_expand_beam<D, P>(
        self,
        distance: D,
        prefetch: P,
    ) -> repr::internal::intrusive::ExpandBeam<'a, QueryDistance<'a, T, U, D>, P>
    where
        D: Distance<T, U>,
        P: Prefetch,
    {
        repr::internal::intrusive::ExpandBeam::new(
            self.reader,
            QueryDistance::new(self.query, distance),
            prefetch,
            self.lookahead,
        )
    }
}

#[derive(Debug, Error)]
#[error(
    "expected slice of length {} - instead got {}",
    self.expected,
    self.xlen,
)]
struct ExpandBeamError {
    expected: usize,
    xlen: usize,
}

diskann::convert_error!(ExpandBeamError);

/////////////////
// Dispatching //
/////////////////

const fn compute_bytes<T>(dim: usize) -> usize {
    dim * std::mem::size_of::<T>() + (AtomicTag::SIZE).value()
}

macro_rules! expand_beam {
    ($into:ident, { $T:ty, $N:literal, $f:ident }) => {{
        $into
            .into_expand_beam(
                Pure::<_, _, Specialize<$N, $f>>::new(),
                prefetch::Unrolled::<{ compute_bytes::<$T>($N) }>::new(),
            )
            .boxed()
    }};
    ($into:ident, $f:ident) => {{
        $into
            .into_expand_beam(Pure::<_, _, $f>::new(), prefetch::Loop::new())
            .boxed()
    }};
}

macro_rules! prune {
    ($self:ty, $reader:ident, $f:ident) => {{ repr::internal::intrusive::Prune::new($reader, Pure::<$self, $self, $f>::new()).boxed() }};
    ($self:ty, $reader:ident, { $N:literal, $f:ident }) => {{
        repr::internal::intrusive::Prune::new(
            $reader,
            Pure::<$self, $self, Specialize<$N, $f>>::new(),
        )
        .boxed()
    }};
}

impl FullPrecisionImpl for f32 {
    fn make_expand_beam<'a>(
        full: &'a Full<f32>,
        query: &'a [f32],
    ) -> ANNResult<Box<dyn repr::ExpandBeam + 'a>> {
        let into = IntoExpandBeam::new(full, Calf::Borrowed(query))?;

        let output: Box<dyn repr::ExpandBeam> = match full.metric {
            Metric::L2 => {
                if full.dim() == 100 {
                    expand_beam!(into, { f32, 100, SquaredL2 })
                } else {
                    expand_beam!(into, SquaredL2)
                }
            }
            Metric::InnerProduct => expand_beam!(into, InnerProduct),
            Metric::Cosine => expand_beam!(into, Cosine),
            Metric::CosineNormalized => expand_beam!(into, CosineNormalized),
        };

        Ok(output)
    }

    fn make_prune<'a>(full: &'a Full<Self>) -> ANNResult<Box<dyn repr::Prune + 'a>> {
        let reader = full.reader()?;

        let output: Box<dyn repr::Prune> = match full.metric {
            Metric::L2 => prune!(Self, reader, SquaredL2),
            Metric::InnerProduct => prune!(Self, reader, InnerProduct),
            Metric::Cosine => prune!(Self, reader, Cosine),
            Metric::CosineNormalized => prune!(Self, reader, CosineNormalized),
        };

        Ok(output)
    }
}

impl FullPrecisionImpl for f16 {
    fn make_expand_beam<'a>(
        full: &'a Full<f16>,
        query: &'a [f16],
    ) -> ANNResult<Box<dyn repr::ExpandBeam + 'a>> {
        let mut as_f32: Box<[f32]> = std::iter::repeat_n(0.0, full.dim()).collect();
        diskann_wide::arch::dispatch2(SliceCast::new(), &mut *as_f32, query);
        let query = Calf::Owned(as_f32);

        let into = IntoExpandBeam::new(full, query)?;

        let output: Box<dyn repr::ExpandBeam> = match full.metric {
            Metric::L2 => {
                if full.dim() == 100 {
                    expand_beam!(into, { f16, 100, SquaredL2 })
                } else {
                    expand_beam!(into, SquaredL2)
                }
            }
            Metric::InnerProduct => expand_beam!(into, InnerProduct),
            Metric::Cosine => expand_beam!(into, Cosine),
            Metric::CosineNormalized => expand_beam!(into, CosineNormalized),
        };

        Ok(output)
    }

    fn make_prune<'a>(full: &'a Full<Self>) -> ANNResult<Box<dyn repr::Prune + 'a>> {
        let reader = full.reader()?;

        let output: Box<dyn repr::Prune> = match full.metric {
            Metric::L2 => prune!(Self, reader, SquaredL2),
            Metric::InnerProduct => prune!(Self, reader, InnerProduct),
            Metric::Cosine => prune!(Self, reader, Cosine),
            Metric::CosineNormalized => prune!(Self, reader, CosineNormalized),
        };

        Ok(output)
    }
}

impl FullPrecisionImpl for u8 {
    fn make_expand_beam<'a>(
        full: &'a Full<u8>,
        query: &'a [u8],
    ) -> ANNResult<Box<dyn repr::ExpandBeam + 'a>> {
        let into = IntoExpandBeam::new(full, Calf::Borrowed(query))?;

        let output: Box<dyn repr::ExpandBeam> = match full.metric {
            Metric::L2 => {
                if full.dim() == 128 {
                    expand_beam!(into, { u8, 128, SquaredL2 })
                } else {
                    expand_beam!(into, SquaredL2)
                }
            }
            Metric::InnerProduct => expand_beam!(into, InnerProduct),
            Metric::Cosine | Metric::CosineNormalized => expand_beam!(into, Cosine),
        };

        Ok(output)
    }

    fn make_prune<'a>(full: &'a Full<Self>) -> ANNResult<Box<dyn repr::Prune + 'a>> {
        let reader = full.reader()?;

        let output: Box<dyn repr::Prune> = match full.metric {
            Metric::L2 => prune!(Self, reader, SquaredL2),
            Metric::InnerProduct => prune!(Self, reader, InnerProduct),
            Metric::Cosine => prune!(Self, reader, Cosine),
            Metric::CosineNormalized => prune!(Self, reader, CosineNormalized),
        };

        Ok(output)
    }
}

impl FullPrecisionImpl for i8 {
    fn make_expand_beam<'a>(
        full: &'a Full<i8>,
        query: &'a [i8],
    ) -> ANNResult<Box<dyn repr::ExpandBeam + 'a>> {
        let into = IntoExpandBeam::new(full, Calf::Borrowed(query))?;

        let distance =
            <Self as DistanceProvider<Self>>::distance_comparer(full.metric(), Some(full.dim()));

        let output: Box<dyn repr::ExpandBeam + 'a> = into
            .into_expand_beam(distance, prefetch::Loop::new())
            .boxed();

        Ok(output)
    }

    fn make_prune<'a>(full: &'a Full<Self>) -> ANNResult<Box<dyn repr::Prune + 'a>> {
        let reader = full.reader()?;

        let distance =
            <Self as DistanceProvider<Self>>::distance_comparer(full.metric(), Some(full.dim()));

        let output: Box<dyn repr::Prune> =
            repr::internal::intrusive::Prune::new(reader, distance).boxed();

        Ok(output)
    }
}

/// We use a macro to stamp out implementations of [`FullPrecision`] instead of using a
/// blanket implementation from [`FullPrecisionImpl`] to make implementations more
/// discoverable through the generated rust-doc.
macro_rules! impl_full_precision {
    ($T:ty) => {
        impl FullPrecision for $T {
            fn __search_accessor<'a>(
                representation: &'a Full<Self>,
                query: &'a [Self],
                provider: &'a (dyn std::any::Any + Send + Sync),
                counters: LocalCounters<'a>,
            ) -> ANNResult<crate::provider::SearchAccessor<'a>> {
                let expand_beam = <$T>::make_expand_beam(representation, query)?;
                Ok(crate::provider::SearchAccessor::new(
                    representation.store.neighbors(),
                    expand_beam,
                    None,
                    provider,
                    representation.store.frozen(),
                    counters,
                ))
            }

            fn __prune_accessor<'a>(
                representation: &'a Full<Self>,
                counters: LocalCounters<'a>,
            ) -> ANNResult<crate::provider::PruneAccessor<'a>> {
                let prune = <$T>::make_prune(representation)?;
                Ok(crate::provider::PruneAccessor::new(
                    prune,
                    representation.store.neighbors(),
                    counters,
                ))
            }
        }
    };
    ($($Ts:ty),* $(,)?) => {
        $(impl_full_precision!($Ts);)*
    }
}

impl_full_precision!(f32, f16, u8, i8);

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::fmt::Display;

    use diskann::neighbor::Neighbor;
    use diskann_utils::lazy_format;
    use hashbrown::{HashMap, HashSet};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    /// Generate random elements of a representation's data type from a seeded RNG.
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
            diskann_wide::cast_f32_to_f16(rng.random_range(-1.0f32..1.0f32))
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

    fn gen_vec<T: Sample>(dim: usize, rng: &mut impl Rng) -> Vec<T> {
        (0..dim).map(|_| T::sample(rng)).collect()
    }

    /// Compare two distances allowing for floating-point reassociation between the
    /// specialized / converted kernels and the dynamic reference.
    #[must_use]
    fn approx_eq(got: f32, want: f32) -> bool {
        (got - want).abs() <= 1e-3 + 1e-4 * want.abs()
    }

    /// A simple test `Full` containing 1-dimensional `f32` values.
    ///
    /// This is used in dedicated `ExpandBeam` and `Prune` tests in a miri-friendly way.
    ///
    /// Two start points are included, initialized to `capacity` and `capacity + 1`.
    fn test_full(capacity: Capacity) -> (Full<f32>, HashMap<u32, f32>) {
        let start_points = [capacity.value() as f32, (capacity.value() + 1) as f32];

        let full = <_ as repr::RepresentationConfig>::build(
            Full::<f32>::config(
                capacity,
                MaxDegree::new(0),
                Metric::L2,
                Matrix::column_vector(Box::new(start_points)),
            )
            .unwrap(),
        )
        .unwrap();

        assert_eq!(full.dim(), 1, "start points only have one dimension");
        assert_eq!(full.bytes(), Bytes::size_of::<f32>());
        assert_eq!(
            full.bytes_plus_tag(),
            Bytes::size_of::<f32>()
                .checked_add(Bytes::size_of::<AtomicTag>())
                .unwrap()
        );
        assert_eq!(full.metric(), Metric::L2);
        assert_eq!(
            <_ as repr::Representation>::id_limit(&full),
            IdLimit::new(capacity.value() as u32 + 2)
        );
        assert_eq!(<_ as repr::Representation>::capacity(&full), capacity);

        let points: HashMap<u32, f32> = {
            let reader = full.reader().unwrap();
            assert_eq!(
                reader.read(capacity.value()).unwrap(),
                bytemuck::bytes_of(&start_points[0])
            );
            assert_eq!(
                reader.read(capacity.value() + 1).unwrap(),
                bytemuck::bytes_of(&start_points[1])
            );

            [
                (capacity.value() as u32, start_points[0]),
                ((capacity.value() + 1) as u32, start_points[1]),
            ]
            .into_iter()
            .collect()
        };

        (full, points)
    }

    impl Distance<f32, f32> for repr::test::TestDistance {
        fn eval(&self, x: UnalignedSlice<'_, f32>, y: UnalignedSlice<'_, f32>) -> f32 {
            assert_eq!(x.len(), 1);
            assert_eq!(y.len(), 1);

            // SAFETY: `UnalignedSlice`s must point to valid data, and we've checked that
            // the length of each slice is exactly 1. Therefore, the pointer read is safe.
            unsafe { x.as_ptr().read_unaligned() + y.as_ptr().read_unaligned() }
        }
    }

    /// A Miri-friendly test for [`ExpandBeam`].
    ///
    /// This test covers the following:
    ///
    /// 1. Prefetches are in-bounds for all lookaheads.
    /// 2. [`ExpandBeam`] doesn't lie about its [`IdLimit`].
    /// 3. [`ExpandBeam`]'s methods are internally consistent with each other and
    ///    consistent with the parent [`Full`] for item readability.
    /// 4. [`ExpandBeam::expand_beam`] preserves input order and visits every item in the
    ///    input list.
    #[test]
    fn test_expand_beam() {
        let capacity = Capacity::new(20);
        let id_limit = IdLimit::new(22);

        let (mut full, mut points) = test_full(capacity);

        assert_eq!(<_ as repr::Representation>::capacity(&full), capacity);
        assert_eq!(<_ as repr::Representation>::id_limit(&full), id_limit);

        let mut available: HashSet<u32> = (0..capacity.value()).map(|i| i as u32).collect();

        // Insert the values 0 to 10.
        for i in 0u32..10 {
            let guard = <_ as repr::Set<&[f32]>>::set(&full, &[i as f32]).unwrap();

            let id = <_ as repr::Guard>::id(&guard);

            assert!(
                available.remove(&id),
                "insertion should return available slots",
            );

            assert!(
                points.insert(id, i as f32).is_none(),
                "insertion should not repeat",
            );

            <_ as repr::Guard>::publish(guard);
        }

        // Lookaheads to try.
        let lookaheads: &[Option<NonZeroUsize>] = &[
            None,
            NonZeroUsize::new(1),
            NonZeroUsize::new(2),
            NonZeroUsize::new(5),
            NonZeroUsize::new(10),
            NonZeroUsize::new(100),
        ];

        // This is the main loop for testing `ExpandBeam`.
        //
        // We do several things.
        //
        // 1. We insert two additional IDs but hold their guards without publishing.
        //    This tests that items remain unreadable until they are published.
        //
        // 2. We publish two new points and immediately retire them.
        //    This tests that we correctly make these points unreadable.
        for lookahead in lookaheads {
            full.lookahead = *lookahead;

            let g0 = <_ as repr::Set<&[f32]>>::set(&full, &[1000.0]).unwrap();
            let g1 = <_ as repr::Set<&[f32]>>::set(&full, &[2000.0]).unwrap();
            let g2 = <_ as repr::Set<&[f32]>>::set(&full, &[3000.0]).unwrap();
            let g3 = <_ as repr::Set<&[f32]>>::set(&full, &[4000.0]).unwrap();

            {
                let g0_id = <_ as repr::Guard>::id(&g0);
                <_ as repr::Guard>::publish(g0);
                <_ as repr::Representation>::retire(&full, g0_id).unwrap();
            }

            {
                let g1_id = <_ as repr::Guard>::id(&g1);
                <_ as repr::Guard>::publish(g1);
                <_ as repr::Representation>::retire(&full, g1_id).unwrap();
            }

            let query = -1.0f32;

            let into =
                IntoExpandBeam::new(&full, Calf::Borrowed(std::slice::from_ref(&query))).unwrap();

            let expand = into.into_expand_beam(repr::test::TestDistance, prefetch::Loop::new());

            assert_eq!(<_ as repr::ExpandBeam>::id_limit(&expand), id_limit);

            let mut buf = Vec::<Neighbor<u32>>::new();
            let mut list = Vec::<u32>::new();

            // Use triangular indexing from `0..id_limit` with `points` serving as the
            // groundtruth.
            //
            // Note that we purposely make `list` extra long with redundant indices to help
            // catch indexing bugs inside `ExpandBeam`.
            for i in 0..=id_limit.value() {
                list.clear();
                list.extend((0..i).rev());
                list.extend(0..i);

                buf.resize(list.len(), Default::default());

                let read = repr::safe_expand_beam(&expand, &list, &mut buf).unwrap();

                let expected: Vec<(u32, f32)> = list
                    .iter()
                    .copied()
                    .filter_map(|id| match points.get(&id) {
                        Some(point) => {
                            let expected = point + query;

                            assert!(
                                <_ as repr::Representation>::is_readable(&full, id).unwrap(),
                                "point should be readable"
                            );

                            assert_eq!(
                                <_ as repr::ExpandBeam>::evaluate(&expand, id).unwrap(),
                                Some(expected),
                                "readable points should return valid distances",
                            );

                            Some((id, expected))
                        }
                        None => {
                            assert!(
                                !<_ as repr::Representation>::is_readable(&full, id).unwrap(),
                                "points not yielded by ExpandBeam should be unreadable"
                            );

                            assert!(
                                <_ as repr::ExpandBeam>::evaluate(&expand, id)
                                    .unwrap()
                                    .is_none(),
                                "unreadable points should return `None` for their distance",
                            );

                            None
                        }
                    })
                    .collect();

                for i in 0..read {
                    assert_eq!(*buf[i].id(), expected[i].0, "i = {i}");
                    assert_eq!(*buf[i].distance(), expected[i].1, "i = {i}");
                }
            }

            assert!(
                <_ as repr::ExpandBeam>::evaluate(&expand, id_limit.value()).is_err(),
                "`ExpandBeam::evaluate` should catch out-of-bounds errors",
            );

            // Ensure we hold onto `g2` and `g3` for the duration of the above check.
            drop(g2);
            drop(g3);
        }
    }

    fn test_prune_inner(
        points: &HashMap<u32, f32>,
        prune: &mut repr::internal::intrusive::Prune<'_, repr::test::TestDistance>,
        ids: &[u32],
    ) {
        let mut items: HashMap<u32, Option<repr::PruneKey>> =
            ids.iter().map(|id| (*id, None)).collect();

        let processed = <_ as repr::Prune>::prepare(prune, items.iter_mut()).unwrap();
        assert_eq!(processed, items.values().filter(|i| i.is_some()).count());

        // Ensure that `prepare` agrees with `points`.
        for (k, v) in items.iter() {
            match v {
                Some(_) => assert!(points.contains_key(k)),
                None => assert!(!points.contains_key(k)),
            }
        }

        fn filter((k, v): (&u32, &Option<repr::PruneKey>)) -> Option<(u32, repr::PruneKey)> {
            v.map(|v| (*k, v))
        }

        // Ensure that distances agree.
        for (k0, v0) in items.iter().filter_map(filter) {
            for (k1, v1) in items.iter().filter_map(filter) {
                // Manually implement `TestDistance`.
                let expected = points[&k0] + points[&k1];
                let got = <_ as repr::Prune>::evaluate(prune, v0, v1);
                assert_eq!(expected, got);
            }
        }
    }

    /// A Miri-friendly test for `Prune`.
    #[test]
    fn test_prune() {
        let capacity = Capacity::new(20);
        let id_limit = IdLimit::new(22);

        let (full, mut points) = test_full(capacity);

        assert_eq!(<_ as repr::Representation>::capacity(&full), capacity);
        assert_eq!(<_ as repr::Representation>::id_limit(&full), id_limit);

        let mut available: HashSet<u32> = (0..capacity.value()).map(|i| i as u32).collect();

        // Insert the values 0 to 10.
        for i in 0u32..10 {
            let guard = <_ as repr::Set<&[f32]>>::set(&full, &[i as f32]).unwrap();

            let id = <_ as repr::Guard>::id(&guard);

            assert!(
                available.remove(&id),
                "insertion should return available slots",
            );

            assert!(
                points.insert(id, i as f32).is_none(),
                "insertion should not repeat",
            );

            <_ as repr::Guard>::publish(guard);
        }

        // We do several things.
        //
        // 1. We insert two additional IDs but hold their guards without publishing.
        //    This tests that items remain unreadable until they are published.
        //
        // 2. We publish two new points and immediately retire them.
        //    This tests that we correctly make these points unreadable.
        let g0 = <_ as repr::Set<&[f32]>>::set(&full, &[1000.0]).unwrap();
        let g1 = <_ as repr::Set<&[f32]>>::set(&full, &[2000.0]).unwrap();
        let g2 = <_ as repr::Set<&[f32]>>::set(&full, &[3000.0]).unwrap();
        let g3 = <_ as repr::Set<&[f32]>>::set(&full, &[4000.0]).unwrap();

        {
            let g0_id = <_ as repr::Guard>::id(&g0);
            <_ as repr::Guard>::publish(g0);
            <_ as repr::Representation>::retire(&full, g0_id).unwrap();
        }

        {
            let g1_id = <_ as repr::Guard>::id(&g1);
            <_ as repr::Guard>::publish(g1);
            <_ as repr::Representation>::retire(&full, g1_id).unwrap();
        }

        let mut prune =
            repr::internal::intrusive::Prune::new(full.reader().unwrap(), repr::test::TestDistance);

        // Note that we emit reads above the `IdLimit`, which we expect to be silently
        // rejected.
        for i in 0..=(id_limit.value() + 5) {
            let mut ids: Vec<u32> = (0..i).collect();
            test_prune_inner(&points, &mut prune, &ids);

            ids.reverse();
            test_prune_inner(&points, &mut prune, &ids);
        }

        // Drop the guards - verifying that they are held in-limbo during the test.
        drop(g2);
        drop(g3);
    }

    //----------------------//
    // Specialization Tests //
    //----------------------//

    // These test make sure that the mapping for metrics and specializations are routed
    // correctly. They do not exhaustively test the `ExpandBeam` kernls as these are left
    // to tests that are more Miri friendly.
    fn test_dispatch<T>(dim: usize, metric: Metric, seed: u64, ctx: &dyn Display)
    where
        T: FullPrecision + FullPrecisionImpl + Sample + DistanceProvider<T>,
    {
        let mut rng = StdRng::seed_from_u64(seed);

        let start_point = gen_vec::<T>(dim, &mut rng);
        let query = gen_vec::<T>(dim, &mut rng);

        let full = <_ as repr::RepresentationConfig>::build(
            Full::<T>::config(
                Capacity::new(1),
                MaxDegree::new(0),
                metric,
                Matrix::<T>::row_vector(start_point.clone().into()),
            )
            .unwrap(),
        )
        .unwrap();

        let start_id: u32 = 1;

        let internal_query = {
            let guard = <_ as repr::Set<&[T]>>::set(&full, &query).unwrap();
            let id = <_ as repr::Guard>::id(&guard);
            <_ as repr::Guard>::publish(guard);
            id
        };

        let distance = <T as DistanceProvider<T>>::distance_comparer(metric, None);
        let expected = distance.call(&start_point, &query);

        // Expand Beam - both `evaluate` and `expand_beam` share the same distance computer,
        // so we can just test `evaluate`.
        {
            let expand_beam = <T as FullPrecisionImpl>::make_expand_beam(&full, &query).unwrap();
            let got = expand_beam.evaluate(start_id).unwrap().unwrap();
            assert!(
                approx_eq(expected, got),
                "{ctx} - expected {expected}, got {got}"
            );
        }

        // Prune
        {
            let mut prune = <T as FullPrecisionImpl>::make_prune(&full).unwrap();
            let mut points: HashMap<u32, Option<repr::PruneKey>> =
                [(internal_query, None), (start_id, None)]
                    .into_iter()
                    .collect();
            prune.prepare(points.iter_mut()).unwrap();
            let got = prune.evaluate(points[&internal_query].unwrap(), points[&start_id].unwrap());
            assert!(
                approx_eq(expected, got),
                "{ctx} - expected {expected}, got {got}"
            );
        }
    }

    fn metrics() -> [Metric; 4] {
        [
            Metric::L2,
            Metric::InnerProduct,
            Metric::Cosine,
            Metric::CosineNormalized,
        ]
    }

    #[test]
    fn test_f32_dynamic() {
        let dim = 10;
        for m in metrics() {
            test_dispatch::<f32>(dim, m, 0x917a80fc68f66e04, &lazy_format!("dynamic-{m}-f32"));
        }
    }

    // Test the specialized dispatches.
    #[test]
    fn test_f32_specialized() {
        test_dispatch::<f32>(
            100,
            Metric::L2,
            0x917a80fc68f66e04,
            &lazy_format!("dynamic-l2-f32-100"),
        );
    }

    #[test]
    fn test_f16_dynamic() {
        let dim = 10;
        for m in metrics() {
            test_dispatch::<f16>(dim, m, 0x917a80fc68f66e04, &lazy_format!("dynamic-{m}-f16"));
        }
    }

    // Test the specialized dispatches.
    #[test]
    fn test_f16_specialized() {
        test_dispatch::<f16>(
            100,
            Metric::L2,
            0x917a80fc68f66e04,
            &lazy_format!("dynamic-l2-f16-100"),
        );
    }

    #[test]
    fn test_u8_dynamic() {
        let dim = 10;
        for m in [Metric::L2, Metric::InnerProduct, Metric::Cosine] {
            test_dispatch::<u8>(dim, m, 0x917a80fc68f66e04, &lazy_format!("dynamic-{m}-u8"));
        }
    }

    #[test]
    fn test_u8_specialized() {
        test_dispatch::<u8>(
            128,
            Metric::L2,
            0x917a80fc68f66e04,
            &lazy_format!("dynamic-l2-u8-100"),
        );
    }

    #[test]
    fn test_i8_dynamic() {
        let dim = 10;
        for m in [Metric::L2, Metric::InnerProduct, Metric::Cosine] {
            test_dispatch::<i8>(dim, m, 0x917a80fc68f66e04, &lazy_format!("dynamic-{m}-i8"));
        }
    }
}
