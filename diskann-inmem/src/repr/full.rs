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
    //! * [`crate::repr::internal::packed::ExpandBeam`]: For index search.
    //! * [`crate::repr::internal::packed::Prune`]: For index construction.
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
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::{
    UnalignedSlice,
    conversion::SliceCast,
    distance::{
        Cosine, CosineNormalized, Distance as VDistance, InnerProduct, Metric, Specialize,
        SquaredL2,
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
    num::{Bytes, Capacity, IdLimit, MaxDegree, Range},
    prefetch::{self, Prefetch},
    repr::{self, internal::Calf},
    store::{
        self, Store,
        packed::{self, Packed},
    },
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
    capacity: Capacity,
    metric: Metric,
    start_points: rowmajor::Owned<T>,
    start_point_ids: Range,
    store: store::Config,
    lookahead: Option<NonZeroUsize>,
}

const DEFAULT_LOOKAHEAD: NonZeroUsize = NonZeroUsize::new(12).unwrap();

impl<T> Config<T> {
    /// Create a new [`Config`] for a [`Full`].
    ///
    /// The resulting store will hold `capacity` writable items and `start_points.nrows()`
    /// frozen points. The dimensionality of the full-precision data will be inferred from
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
        start_points: rowmajor::Owned<T>,
    ) -> Result<Self, ConfigError> {
        // Check that we won't overflow when computing the number of bytes required for each
        // data point. This can happen if `start_points` has 0 rows but a large number of
        // columns.
        if start_points
            .ncols()
            .checked_mul(std::mem::size_of::<T>())
            .is_none()
        {
            return Err(ConfigError::dim_too_large(start_points.ncols()));
        }

        let num_start_points = start_points.nrows();
        let too_many_points = || ConfigError::too_many_points(capacity, num_start_points);
        let start_point_ids =
            Range::start_plus_usize(0, num_start_points).ok_or_else(too_many_points)?;

        let id_limit = capacity
            .value()
            .checked_add(num_start_points)
            .and_then(|id_limit| u32::try_from(id_limit).map(IdLimit::new).ok())
            .ok_or_else(too_many_points)?;

        Ok(Self {
            layout: store::Layout::new(id_limit, max_degree),
            capacity,
            metric,
            start_points,
            start_point_ids,
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
#[error(transparent)]
pub struct ConfigError {
    inner: ConfigErrorInner,
}

impl ConfigError {
    fn dim_too_large(dim: usize) -> Self {
        Self {
            inner: ConfigErrorInner::DimTooLarge(dim),
        }
    }

    fn too_many_points(capacity: Capacity, num_start_points: usize) -> Self {
        Self {
            inner: ConfigErrorInner::TooManyPoints {
                capacity,
                num_start_points,
            },
        }
    }
}

diskann::convert_error!(ConfigError);

#[derive(Debug, Error)]
enum ConfigErrorInner {
    #[error("the number of bytes to hold {0}-dimensional data exceeds `usize::MAX`")]
    DimTooLarge(usize),
    #[error(
        "requested number of points ({} + {} start points) exceeds u32::MAX",
        capacity,
        num_start_points
    )]
    TooManyPoints {
        capacity: Capacity,
        num_start_points: usize,
    },
}

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
    store: Store<Packed>,
    start_point_ids: Range,
    capacity: Capacity,
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
        start_points: rowmajor::Owned<T>,
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
            capacity,
            metric,
            start_points,
            start_point_ids,
            store,
            lookahead,
        } = config;

        let bytes = Bytes::new(start_points.ncols() * std::mem::size_of::<T>());
        let packed = Packed::config(bytes);
        let store = Store::new(layout, store, packed)?;

        // Initialize start points.
        for (i, row) in std::iter::zip(start_point_ids, start_points.rows()) {
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
            start_point_ids,
            capacity,
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

    fn reader(&self) -> Result<packed::Reader<'_>, epoch::Unavailable> {
        self.store
            .read_guard(|packed, guard| packed.reader(guard))
    }
}

impl<T> Full<T>
where
    T: FullPrecision,
{
    pub(crate) fn get(&self, i: u32) -> ANNResult<Box<[T]>> {
        let reader = self.reader()?;
        let data = match reader.read(i) {
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
    for<'a> packed::Exclusive<'a>
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

/// A temporary precursor for [`repr::internal::packed::ExpandBeam`] to simplify macros.
#[derive(Debug)]
struct IntoExpandBeam<'a, T, U> {
    query: Calf<'a, [T]>,
    reader: store::packed::Reader<'a>,
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
    ) -> repr::internal::packed::ExpandBeam<'a, QueryDistance<'a, T, U, D>, P>
    where
        D: Distance<T, U>,
        P: Prefetch,
    {
        repr::internal::packed::ExpandBeam::new(
            self.reader,
            QueryDistance::new(self.query, distance),
            prefetch,
            self.lookahead,
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
    dim * std::mem::size_of::<T>()
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
    ($self:ty, $reader:ident, $f:ident) => {{ repr::internal::packed::Prune::new($reader, Pure::<$self, $self, $f>::new()).boxed() }};
    ($self:ty, $reader:ident, { $N:literal, $f:ident }) => {{
        repr::internal::packed::Prune::new($reader, Pure::<$self, $self, Specialize<$N, $f>>::new())
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

        let distance = VDistance::<Self, Self>::new(full.metric(), Some(full.dim()));

        let output: Box<dyn repr::ExpandBeam + 'a> = into
            .into_expand_beam(distance, prefetch::Loop::new())
            .boxed();

        Ok(output)
    }

    fn make_prune<'a>(full: &'a Full<Self>) -> ANNResult<Box<dyn repr::Prune + 'a>> {
        let reader = full.reader()?;

        let distance = VDistance::<Self, Self>::new(full.metric(), Some(full.dim()));

        let output: Box<dyn repr::Prune> =
            repr::internal::packed::Prune::new(reader, distance).boxed();

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
                    representation.start_point_ids,
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

    use diskann::graph::test::synthetic::Grid;
    use diskann_utils::{assert_contains, lazy_format, views::rowmajor::MatrixMut};
    use diskann_vector::distance::DistanceProvider;
    use hashbrown::HashMap;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use crate::{
        counters::Counters,
        num::{LogicalId, SlotId},
        repr::test::Reference,
    };

    /// See the description for [`make_test_repr`].
    const TEST_LIMIT: IdLimit = IdLimit::new(11);

    /// Use the canonical grid layout, but center the data around the origin.
    ///
    /// This allows cosine distances to return reasonable results as the data is distributed
    /// around the origin.
    ///
    /// To keep computation mostly tractable, we only use a 2D grid with 9 points. The
    /// coordinates are as follows:
    ///
    /// 0:  [-1, -1]
    /// 1:  [-1,  0]
    /// 2:  [-1, +1]
    ///
    /// 3:  [ 0, -1]
    /// 4:  [ 0,  0]
    /// 5:  [ 0, +1]
    ///
    /// 6:  [+1, -1]
    /// 7:  [+1,  0]
    /// 8:  [+1, +1]
    ///
    /// We put two start points at `[-2, -2]` and `[+2, +2]`.
    ///
    /// Overall, there are a total of 11 points.
    fn make_test_repr(metric: Metric) -> (Full<f32>, Reference) {
        let grid = Grid::Two;
        let mut data = grid.data(3);
        let offset = 1.0;
        data.as_mut_slice().iter_mut().for_each(|v| *v -= offset);

        let mut start_points = rowmajor::Owned::from_element(2, data.ncols(), 0.0);
        start_points.row_mut(0).fill(-2.0);
        start_points.row_mut(1).fill(2.0);

        let full = <_ as repr::RepresentationConfig>::build(
            Full::<f32>::config(
                Capacity::new(data.nrows()),
                MaxDegree::new(0),
                metric,
                start_points.clone(),
            )
            .unwrap(),
        )
        .unwrap();

        let bytes = std::mem::size_of_val(data.row(0));
        assert_eq!(full.dim(), data.ncols());
        assert_eq!(full.bytes().value(), bytes);
        assert_eq!(full.metric(), metric);
        assert_eq!(repr::Representation::id_limit(&full), TEST_LIMIT,);
        assert_eq!(
            repr::Representation::capacity(&full),
            Capacity::new(data.nrows())
        );

        let mut reference = Reference::new(grid.dim().into());

        {
            let reader = full.reader().unwrap();
            for (i, (slot, start)) in
                std::iter::zip(full.start_point_ids, start_points.rows()).enumerate()
            {
                assert_eq!(
                    reader.read(slot).unwrap(),
                    bytemuck::must_cast_slice::<f32, u8>(start),
                );
                assert_eq!(repr::Representation::is_readable(&full, slot), Some(true));
                reference.insert(LogicalId(usize::MAX - i), SlotId(slot), start);
            }
        }

        for (i, row) in data.rows().enumerate() {
            let guard = repr::Set::set(&full, row).unwrap();
            let slot = repr::Guard::id(&guard);
            reference.insert(LogicalId(i), SlotId(slot), row);
            repr::Guard::publish(guard);
        }

        (full, reference)
    }

    /// These tests focus on accessor and distance-computer wiring. Exhaustive backend
    /// behavior is covered by [`repr::internal::packed`].
    ///
    /// We use:
    ///
    /// * `ExpandBeam::evaluate` and `ExpandBeam::expand_beam` to verify that the correct
    ///   distance computer is created.
    ///
    /// * `Prune` to validate that the correct distance computer is made.
    fn test_distances(
        full: &Full<f32>,
        reference: &mut Reference,
        metric: Metric,
        ctx: &dyn std::fmt::Display,
    ) {
        assert_eq!(repr::Representation::id_limit(full), TEST_LIMIT, "{ctx}");

        let query = [10.0, -10.0];

        // Select a few data points from the dataset.
        let i0 = LogicalId(2);
        let s0 = reference.slot_id_for(i0);

        let i1 = LogicalId(5);
        let s1 = reference.slot_id_for(i1);

        let i2 = LogicalId(usize::MAX);
        let s2 = reference.slot_id_for(i2);

        // Delete values.
        let i3_deleted = LogicalId(8);
        let s3_deleted = reference.slot_id_for(i3_deleted);

        repr::Representation::retire(full, s3_deleted.value()).unwrap();
        reference.delete(i3_deleted);

        let i4_deleted = LogicalId(4);
        let s4_deleted = reference.slot_id_for(i4_deleted);

        repr::Representation::retire(full, s4_deleted.value()).unwrap();
        reference.delete(i4_deleted);

        // Extract values.
        let v0 = &reference[i0];
        let v1 = &reference[i1];
        let v2 = &reference[i2];

        let f = VDistance::<f32, f32>::new(metric, None);

        // Expected distances between the query and the selected dataset vectors.
        let query_distances = HashMap::from_iter([
            (s0, f.call(&query, v0)),
            (s1, f.call(&query, v1)),
            (s2, f.call(&query, v2)),
        ]);

        // Search/Insert
        {
            let counters = Counters::new();
            let sa = repr::Search::search_accessor(full, query.as_slice(), &(), counters.local())
                .unwrap();

            repr::test::test_expand_beam(
                sa.get_expand_beam(),
                TEST_LIMIT,
                query_distances,
                &[s0, s1, s3_deleted, s2, s4_deleted],
                ctx,
            );
        }

        // Prune
        {
            let v00 = f.call(v0, v0);
            let v01 = f.call(v0, v1);
            let v02 = f.call(v0, v2);

            let v10 = v01;
            let v11 = f.call(v1, v1);
            let v12 = f.call(v1, v2);

            let v20 = v02;
            let v21 = v12;
            let v22 = f.call(v2, v2);

            let distances = HashMap::from_iter([
                ((s0, s0), v00),
                ((s0, s1), v01),
                ((s0, s2), v02),
                ((s1, s0), v10),
                ((s1, s1), v11),
                ((s1, s2), v12),
                ((s2, s0), v20),
                ((s2, s1), v21),
                ((s2, s2), v22),
            ]);

            let counters = Counters::new();
            let mut pa = repr::Insert::prune_accessor(full, counters.local()).unwrap();

            repr::test::test_prune(
                pa.get_prune(),
                distances,
                &[s0, s1, s3_deleted, s2, s4_deleted],
                ctx,
            );
        }
    }

    /// The happy-path entry point for all supported metrics.
    #[test]
    fn test_full() {
        let metrics = [
            Metric::L2,
            Metric::InnerProduct,
            Metric::Cosine,
            Metric::CosineNormalized,
        ];
        for metric in metrics {
            let (full, mut reference) = make_test_repr(metric);
            test_distances(
                &full,
                &mut reference,
                metric,
                &format_args!("metric = {:?}", metric),
            );
        }
    }

    /// Test that:
    ///
    /// 1. Slot exhaustion is handled properly.
    /// 2. Data is not visible until publishing.
    /// 3. Dropping a [`Guard`] avoids publishing.
    /// 4. We can properly reuse slots.
    #[test]
    fn test_guard_and_reuse() {
        let (full, mut reference) = make_test_repr(Metric::L2);

        let err = repr::Set::set(&full, &[-10.0, -10.0])
            .unwrap_err()
            .to_string();
        assert_contains!(err, "could not allocate a new slot");

        let id = LogicalId(2);
        let slot = reference.slot_id_for(id);
        assert_eq!(
            repr::Representation::is_readable(&full, slot.value()),
            Some(true),
        );

        // Create a reader.
        //
        // This reader observes the retirement of `id/slot`. Dropping it should allow the
        // system to progress so we can reuse the slot.
        {
            let reader = full.reader().unwrap();
            assert_eq!(
                reader.read(slot.value()).unwrap(),
                bytemuck::bytes_of(&[-1.0f32, 1.0])
            );

            repr::Representation::retire(&full, slot.value()).unwrap();
            reference.delete(id);

            assert!(reader.read(slot.value()).is_none());
            assert_eq!(
                repr::Representation::is_readable(&full, slot.value()),
                Some(false),
            );

            let err = repr::Set::set(&full, &[-2.0, 1.0]).unwrap_err().to_string();
            assert_contains!(err, "could not allocate a new slot");

            // `reader` gets dropped.
        }

        // Values are not readable until published.
        //
        // Dropping a `guard` does not publish.
        {
            let guard = repr::Set::set(&full, &[-10.0, -10.0]).unwrap();
            assert_eq!(
                repr::Guard::id(&guard),
                slot.value(),
                "expected the same slot to be reused because of the tight capacity"
            );

            let reader = full.reader().unwrap();
            assert!(reader.read(slot.value()).is_none());
            assert_eq!(
                repr::Representation::is_readable(&full, slot.value()),
                Some(false),
            );

            // Dropping means the slot is reclaimed.
        }
        assert_eq!(
            repr::Representation::is_readable(&full, slot.value()),
            Some(false),
        );

        {
            let reader = full.reader().unwrap();
            let guard = repr::Set::set(&full, &[-10.0, -10.0]).unwrap();
            assert_eq!(
                repr::Guard::id(&guard),
                slot.value(),
                "expected the same slot to be reused because of the tight capacity"
            );
            assert_eq!(
                repr::Representation::is_readable(&full, slot.value()),
                Some(false),
            );

            repr::Guard::publish(guard);
            assert_eq!(
                repr::Representation::is_readable(&full, slot.value()),
                Some(true),
            );

            assert_eq!(
                reader.read(slot.value()).unwrap(),
                bytemuck::bytes_of(&[-10.0f32, -10.0]),
            );
        }
    }

    #[test]
    fn test_set_dim_mismatch() {
        let (full, _) = make_test_repr(Metric::L2);

        for dim in [1, 3] {
            let err = repr::Set::set(&full, vec![1.0; dim].as_slice()).unwrap_err();
            assert_contains!(
                err.to_string(),
                format!("expected slice of length 2 - instead got {dim}"),
            );
        }
    }

    #[test]
    fn test_search_dim_mismatch() {
        let (full, _) = make_test_repr(Metric::L2);
        let counters = Counters::new();

        for dim in [1, 3] {
            let query = vec![1.0; dim];
            let err = repr::Search::search_accessor(&full, query.as_slice(), &(), counters.local())
                .unwrap_err();
            assert_contains!(
                err.to_string(),
                format!("expected slice of length 2 - instead got {dim}"),
            );
        }
    }

    #[test]
    fn test_config_dim_too_large() {
        let dim = usize::MAX / std::mem::size_of::<f32>() + 1;
        let start_points = rowmajor::Owned::from_element(0, dim, 0.0f32);
        let err = Full::config(
            Capacity::new(1),
            MaxDegree::new(0),
            Metric::L2,
            start_points,
        )
        .unwrap_err();

        assert_contains!(
            err.to_string(),
            format!("the number of bytes to hold {dim}-dimensional data exceeds `usize::MAX`"),
        );
    }

    #[test]
    fn test_config_too_many_points() {
        for (capacity, num_start_points) in [
            (Capacity::new(u32::MAX as usize), 1),
            (Capacity::new(usize::MAX), 1),
            (Capacity::new(0), u32::MAX as usize + 1),
        ] {
            let start_points = rowmajor::Owned::from_element(num_start_points, 0, 0.0f32);
            let err =
                Full::config(capacity, MaxDegree::new(0), Metric::L2, start_points).unwrap_err();

            assert_contains!(
                err.to_string(),
                format!(
                    "requested number of points ({capacity} + {num_start_points} start points) exceeds u32::MAX"
                ),
            );
        }
    }

    //----------------------//
    // Specialization Tests //
    //----------------------//

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

    // These tests check that metric and specialization dispatches are routed correctly.
    // They do not exhaustively test `ExpandBeam` mechanics, which are covered by the
    // backend tests.
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
                rowmajor::Owned::<T>::row_vector(start_point.clone().into()),
            )
            .unwrap(),
        )
        .unwrap();

        // We know the internal start IDs begin at 0. This is not a stability guarantee.
        let start_id: u32 = 0;
        let internal_query = {
            let guard = <_ as repr::Set<&[T]>>::set(&full, &query).unwrap();
            let id = <_ as repr::Guard>::id(&guard);
            <_ as repr::Guard>::publish(guard);
            id
        };

        let distance = VDistance::<T, T>::new(metric, None);
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
            let prune = <T as FullPrecisionImpl>::make_prune(&full).unwrap();
            let query = prune.get(internal_query).unwrap();
            let start = prune.get(start_id).unwrap();
            let got = prune.evaluate(query, start);
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
