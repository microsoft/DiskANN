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
    //! Internally, the [`super::layers::Search`] and [`super::layers::Insert`] traits
    //! are implemented via [`super::FullPrecisionImpl`], which creates:
    //!
    //! * [`super::ExpandBeam`]: For index search.
    //! * [`super::Prune`]: For index construction.
    //!
    //! These two structs are modular with respect to their exact distance function and
    //! prefetcher. Since [`super::layers::ExpandBeam`] and [`super::layers::Prune`] are
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
    epoch, layers,
    num::{Bytes, Capacity, IdLimit, MaxDegree},
    prefetch::{self, Prefetch},
    store::{
        self, Store,
        invasive::{self, Invasive},
    },
    tag::AtomicTag,
};

/// A useful trait bound for types compatible with [`Full`].
///
/// This encompasses *everything* required for `Full: layers::Insert` and can be used as
/// a single bound.
pub trait FullPrecision: bytemuck::Pod + std::fmt::Debug + Send + Sync {
    #[doc(hidden)]
    fn __search_accessor<'a>(
        layer: &'a Full<Self>,
        query: &'a [Self],
        provider: &'a (dyn std::any::Any + Send + Sync),
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::SearchAccessor<'a>>;

    #[doc(hidden)]
    fn __prune_accessor<'a>(
        layer: &'a Full<Self>,
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

impl<T> layers::LayerConfig for Config<T>
where
    T: FullPrecision,
{
    type Layer = Full<T>;

    fn build(self) -> ANNResult<Full<T>> {
        Full::new(self)
    }
}

/// Internal helper for implementing [`FullPrecision`].
trait FullPrecisionImpl: bytemuck::Pod + std::fmt::Debug + Send + Sync {
    fn make_expand_beam<'a>(
        full: &'a Full<Self>,
        query: &'a [Self],
    ) -> ANNResult<Box<dyn layers::ExpandBeam + 'a>>;

    #[doc(hidden)]
    fn make_prune<'a>(full: &'a Full<Self>) -> ANNResult<Box<dyn layers::Prune + 'a>>;
}

/// Full-precision data layer.
#[derive(Debug)]
pub struct Full<T>
where
    T: 'static,
{
    store: Store<Invasive>,
    metric: Metric,
    lookahead: Option<NonZeroUsize>,
    _type: PhantomData<T>,
}

impl<T> Full<T>
where
    T: 'static,
{
    /// Initialize a [`Config`] for this layer.
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

    /// Create a new full-precision layer from `config`.
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
        let invasive = Invasive::config(bytes);
        let store = Store::new(layout, store, invasive)?;

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

    /// Return the logical dimension of the data handled by this [`layers::Layer`].
    pub fn dim(&self) -> usize {
        self.bytes().value() / std::mem::size_of::<T>()
    }

    /// Return the number of payload bytes in each stored vector.
    pub fn bytes(&self) -> Bytes {
        self.store.plugin().bytes()
    }

    #[cfg(test)]
    fn bytes_plus_tag(&self) -> Bytes {
        self.store.plugin().bytes_plus_tag()
    }

    /// Return the [`Metric`] for this layer.
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

    fn reader(&self) -> Result<invasive::Reader<'_>, epoch::Unavailable> {
        Invasive::reader(&self.store)
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

impl<T> layers::Layer for Full<T>
where
    T: FullPrecision,
{
    fn max_degree(&self) -> MaxDegree {
        self.store.neighbors().max_degree()
    }

    fn retire(&self, i: u32) -> ANNResult<()> {
        Ok(self.store.retire(i.into_usize())?)
    }

    fn is_readable(&self, i: u32) -> Option<bool> {
        self.store.can_read_approximate(i.into_usize())
    }

    fn id_limit(&self) -> IdLimit {
        self.store.id_limit()
    }

    fn capacity(&self) -> Capacity {
        self.store.capacity()
    }
}

impl<T> layers::Set<&[T]> for Full<T>
where
    T: FullPrecision,
{
    type Guard<'a> = Guard<'a>;

    fn set(&self, v: &[T]) -> ANNResult<Guard<'_>> {
        if v.len() != self.dim() {
            return Err(ANNError::from(SetError {
                got: v.len(),
                expected: self.dim(),
            }));
        }

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

/// A [`layers::Guard`] for [`Full`].
#[derive(Debug)]
pub struct Guard<'a> {
    slot: store::Slot<'a, invasive::Slot<'a>>,
}

impl<'a> Guard<'a> {
    fn new(slot: store::Slot<'a, invasive::Slot<'a>>) -> Self {
        Self { slot }
    }
}

impl layers::Guard for Guard<'_> {
    fn publish(self) {
        self.slot.publish();
    }
    fn id(&self) -> u32 {
        self.slot.slot()
    }
}

#[derive(Debug, Error)]
#[error(
    "data of dimension {} does not match full precision layer's dimension {}",
    self.got,
    self.expected
)]
struct SetError {
    got: usize,
    expected: usize,
}

diskann::convert_error!(SetError);

impl<T> layers::Search for Full<T>
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

impl<T> layers::Insert for Full<T>
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

//----------------------//
// Expand Beam (Search) //
//----------------------//

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

/// A temporary precursor for [`ExpandBeam`] to simplify macros.
#[derive(Debug)]
struct IntoExpandBeam<'a, T, U> {
    query: Calf<'a, T>,
    reader: store::invasive::Reader<'a>,
    lookahead: Option<NonZeroUsize>,
    _data: PhantomData<U>,
}

impl<'a, T, U> IntoExpandBeam<'a, T, U> {
    /// Construct a new [`IntoExpandBeam`], validating the query dimension and acquiring a
    /// reader for `full`.
    fn new(full: &'a Full<U>, query: Calf<'a, T>) -> ANNResult<Self> {
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
}

trait Distance<T, U>: std::fmt::Debug + Send + Sync + 'static {
    fn eval(&self, x: UnalignedSlice<'_, T>, y: UnalignedSlice<'_, U>) -> f32;
}

#[derive(Debug)]
struct Pure<D>(PhantomData<D>);

impl<D> Pure<D> {
    const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T, U, D> Distance<T, U> for Pure<D>
where
    D: for<'any> FTarget2<Current, f32, UnalignedSlice<'any, T>, UnalignedSlice<'any, U>>
        + std::fmt::Debug
        + Send
        + Sync
        + 'static,
{
    #[inline(always)]
    fn eval(&self, x: UnalignedSlice<'_, T>, y: UnalignedSlice<'_, U>) -> f32 {
        D::run(ARCH, x, y)
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

/// A fused query distance based on [`diskann_vector::PureDistanceFunction`] to enable
/// inlining of the final distance function (`D`).
///
/// The type of the embedded query (`T`) is distinct from the expected data-set (`U`) to
/// allow `f16` queries to be pre-converted to `f32`, saving on-the-fly conversion that
/// would otherwise be needed.
#[derive(Debug)]
struct ExpandBeam<'a, P, T, U, D> {
    // The original query.
    query: Calf<'a, T>,
    // A reader into a layer's store.
    reader: store::invasive::Reader<'a>,
    // The prefetch lookahead.
    lookahead: Option<NonZeroUsize>,
    // The type of the data prefetcher.
    prefetch: prefetch::Checked<P>,
    // The type of the distance used for the arguments
    distance: D,
    // The type of the data in the original dataset.
    _data: PhantomData<U>,
}

impl<'a, P, T, U, D> ExpandBeam<'a, P, T, U, D> {
    fn new(into: IntoExpandBeam<'a, T, U>, prefetch: P, distance: D) -> Self
    where
        P: Prefetch,
    {
        let IntoExpandBeam {
            query,
            reader,
            lookahead,
            _data,
        } = into;

        // TAG: PREFETCH-CHECK
        #[expect(
            clippy::expect_used,
            reason = "internal APIs should only provide valid prefetchers"
        )]
        let prefetch = prefetch::Checked::new(prefetch, reader.bytes_plus_tag())
            .expect("internal APIs should only provide valid prefetchers");

        Self {
            query,
            reader,
            lookahead,
            prefetch,
            distance,
            _data,
        }
    }

    fn bytes(&self) -> usize {
        std::mem::size_of::<U>() * self.query.len()
    }

    fn boxed(self) -> Box<Self> {
        Box::new(self)
    }

    /// Compute the distance between the embedded query and `x`.
    ///
    /// # Safety
    ///
    /// `x.len()` must be exactly `self.bytes()` bytes long and contain
    /// `self.query.len()` valid values of `U`.
    #[inline(always)]
    unsafe fn run_unchecked(&self, x: &[u8]) -> f32
    where
        D: Distance<T, U>,
    {
        debug_assert_eq!(x.len(), self.bytes());

        // SAFETY: We've validated that `x` has the correct length.
        let x = unsafe { UnalignedSlice::new(x.as_ptr().cast::<U>(), self.query.len()) };
        self.distance.eval((*self.query).into(), x)
    }
}

// SAFETY: Our implementation of `layers::ExpandBeam::id_limit` is consistent with our
// `layers::ExpandBeam::expand_beam` implementation. They are both dependent on
// `invasive::Reader`'s internal bounds.
unsafe impl<P, T, U, D> layers::ExpandBeam for ExpandBeam<'_, P, T, U, D>
where
    P: Prefetch,
    T: Send + Sync + 'static + Debug,
    U: Send + Sync + 'static + Debug,
    D: Distance<T, U>,
{
    fn evaluate(&self, i: u32) -> ANNResult<Option<f32>> {
        if !self.reader.is_in_bounds(i.into_usize()) {
            Err(ANNError::new(OutOfBounds(i)))
        } else {
            // SAFETY: We have checked that `i` is in-bounds.
            match unsafe { self.reader.read_in_bounds(i.into_usize()) } {
                Some(data) => {
                    // SAFETY: Since we just read `data` from `self.reader`, we know it's
                    // exactly `self.bytes()` long.
                    let distance = unsafe { self.run_unchecked(data) };
                    Ok(Some(distance))
                }
                None => Ok(None),
            }
        }
    }

    fn id_limit(&self) -> IdLimit {
        self.reader.id_limit()
    }

    unsafe fn expand_beam(&self, list: &[u32], buffer: &mut [(u32, f32)]) -> ANNResult<usize> {
        debug_assert!(buffer.len() >= list.len());

        let len = list.len();
        let lookahead = self.lookahead.map(|l| l.get()).unwrap_or(0).min(len);

        for j in list.iter().take(lookahead) {
            // SAFETY: The in-bounds constraint is assured by the caller, both for `j` as well
            // as the validity of the prefetch bounds.
            //
            // We validated `self.prefetch` with `self.reader.bytes_with_tag()` upon construction.
            //
            // We do not materialize the `RawSlice` as a reference.
            unsafe {
                let raw = self.reader.read_raw_unchecked(j.into_usize());
                self.prefetch.prefetch(raw.as_ptr(), raw.len());
            }
        }

        // Disable prefetching if the lookahead is 0.
        let mut j = if lookahead == 0 { len } else { lookahead };
        let mut processed = 0;
        for &i in list.iter() {
            if j != len {
                // SAFETY: The in-bounds constraint is assured by the caller, both for `j` as
                // well as the validity of the prefetch bounds.
                //
                // We validated `self.prefetch` with `self.reader.bytes_with_tag()` upon
                // construction.
                //
                // We do not materialize the `RawSlice` as a reference.
                unsafe {
                    let raw = self
                        .reader
                        .read_raw_unchecked(list.get_unchecked(j).into_usize());
                    self.prefetch.prefetch(raw.as_ptr(), raw.len());
                }
                j += 1;
            }

            // SAFETY: Caller asserts that `i` is in-bounds.
            if let Some(data) = unsafe { self.reader.read_in_bounds(i.into_usize()) } {
                // SAFETY: We just read `data` from `self.reader`, so it has a length of
                // exactly `self.bytes()`.
                let distance = unsafe { self.run_unchecked(data) };

                // SAFETY: Inherited from caller.
                *unsafe { buffer.get_unchecked_mut(processed) } = (i, distance);
                processed += 1;
            }
        }

        Ok(processed)
    }
}

#[derive(Debug, Error)]
#[error("index {} is out-of-bounds", self.0)]
struct OutOfBounds(u32);

diskann::convert_error!(OutOfBounds);

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

//-------//
// Prune //
//-------//

#[derive(Debug)]
struct Prune<'a, T, D> {
    // Buffered data to prune over.
    buffer: Vec<UnalignedSlice<'a, T>>,
    // A reader into a layer's store.
    reader: store::invasive::Reader<'a>,
    // The distance implementation used for pruning.
    distance: D,
}

impl<'a, T, D> Prune<'a, T, D> {
    fn new(reader: store::invasive::Reader<'a>, distance: D) -> Self {
        // This should be ensured at construction time
        debug_assert!(
            reader
                .bytes()
                .value()
                .is_multiple_of(std::mem::size_of::<T>()),
            "internal invariant violated",
        );

        Self {
            buffer: Vec::new(),
            reader,
            distance,
        }
    }

    fn boxed(self) -> Box<Self> {
        Box::new(self)
    }
}

impl<T, D> layers::Prune for Prune<'_, T, D>
where
    T: Debug + Send + Sync + 'static,
    D: Distance<T, T>,
{
    fn prepare(
        &mut self,
        items: hashbrown::hash_map::IterMut<'_, u32, Option<layers::PruneKey>>,
    ) -> ANNResult<usize> {
        let mut counter = layers::PruneKey::counter();
        self.buffer.clear();
        self.buffer.reserve(items.len());

        for (id, key) in items {
            if let Some(v) = self.reader.read(id.into_usize()) {
                // SAFETY: We have checked that it is safe to read this data vector and
                // `self.reader` is preventing any mutation for `self`'s lifetime.
                //
                // Further, we know the raw slice has a length exactly `self.reader.bytes()`,
                // so the formed `UnalignedSlice` is within a single allocated object.
                let unaligned = unsafe {
                    UnalignedSlice::new(
                        v.as_ptr().cast::<T>(),
                        self.reader.bytes().value() / std::mem::size_of::<T>(),
                    )
                };

                self.buffer.push(unaligned);

                *key = Some(counter);

                // Potential overflow issue - but it's exceedingly unlikely that
                // someone will provide a prune list exceeding `u16::MAX`.
                //
                // In addition, `diskann` limits this bound as well.
                counter = counter.increment()?;
            }
        }

        Ok(counter.index())
    }

    fn evaluate(&self, a: layers::PruneKey, b: layers::PruneKey) -> f32 {
        self.distance
            .eval(self.buffer[a.index()], self.buffer[b.index()])
    }
}

/////////////////
// Dispatching //
/////////////////

const fn compute_bytes<T>(dim: usize) -> usize {
    dim * std::mem::size_of::<T>() + (AtomicTag::SIZE).value()
}

macro_rules! expand_beam {
    ($into:ident, { $T:ty, $N:literal, $f:ident }) => {{
        Box::new(ExpandBeam::<_, _, $T, _>::new(
            $into,
            prefetch::Unrolled::<{ compute_bytes::<$T>($N) }>::new(),
            Pure::<Specialize<$N, $f>>::new(),
        ))
    }};
    ($into:ident, $f:ident) => {{
        Box::new(ExpandBeam::new(
            $into,
            prefetch::Loop::new(),
            Pure::<$f>::new(),
        ))
    }};
}

macro_rules! prune {
    ($self:ty, $reader:ident, $f:ident) => {{
        Prune::<$self, _>::new($reader, Pure::<$f>::new()).boxed()
    }};
    ($self:ty, $reader:ident, { $N:literal, $f:ident }) => {{
        Prune::<$self, _>::new($reader, Pure::<Specialize<$N, $f>>::new()).boxed()
    }};
}

impl FullPrecisionImpl for f32 {
    fn make_expand_beam<'a>(
        full: &'a Full<f32>,
        query: &'a [f32],
    ) -> ANNResult<Box<dyn layers::ExpandBeam + 'a>> {
        let into = IntoExpandBeam::new(full, Calf::Borrowed(query))?;

        let output: Box<dyn layers::ExpandBeam> = match full.metric {
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

    fn make_prune<'a>(full: &'a Full<Self>) -> ANNResult<Box<dyn layers::Prune + 'a>> {
        let reader = full.reader()?;

        let output: Box<dyn layers::Prune> = match full.metric {
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
    ) -> ANNResult<Box<dyn layers::ExpandBeam + 'a>> {
        let mut as_f32: Box<[f32]> = std::iter::repeat_n(0.0, full.dim()).collect();
        diskann_wide::arch::dispatch2(SliceCast::new(), &mut *as_f32, query);
        let query = Calf::Owned(as_f32);

        let into = IntoExpandBeam::new(full, query)?;

        let output: Box<dyn layers::ExpandBeam> = match full.metric {
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

    fn make_prune<'a>(full: &'a Full<Self>) -> ANNResult<Box<dyn layers::Prune + 'a>> {
        let reader = full.reader()?;

        let output: Box<dyn layers::Prune> = match full.metric {
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
    ) -> ANNResult<Box<dyn layers::ExpandBeam + 'a>> {
        let into = IntoExpandBeam::new(full, Calf::Borrowed(query))?;

        let output: Box<dyn layers::ExpandBeam> = match full.metric {
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

    fn make_prune<'a>(full: &'a Full<Self>) -> ANNResult<Box<dyn layers::Prune + 'a>> {
        let reader = full.reader()?;

        let output: Box<dyn layers::Prune> = match full.metric {
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
    ) -> ANNResult<Box<dyn layers::ExpandBeam + 'a>> {
        let into = IntoExpandBeam::new(full, Calf::Borrowed(query))?;

        let distance =
            <Self as DistanceProvider<Self>>::distance_comparer(full.metric(), Some(full.dim()));

        let output: Box<dyn layers::ExpandBeam + 'a> =
            ExpandBeam::new(into, prefetch::Loop::new(), distance).boxed();

        Ok(output)
    }

    fn make_prune<'a>(full: &'a Full<Self>) -> ANNResult<Box<dyn layers::Prune + 'a>> {
        let reader = full.reader()?;

        let distance =
            <Self as DistanceProvider<Self>>::distance_comparer(full.metric(), Some(full.dim()));

        let output: Box<dyn layers::Prune> = Prune::<Self, _>::new(reader, distance).boxed();
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
                layer: &'a Full<Self>,
                query: &'a [Self],
                provider: &'a (dyn std::any::Any + Send + Sync),
                counters: LocalCounters<'a>,
            ) -> ANNResult<crate::provider::SearchAccessor<'a>> {
                let expand_beam = <$T>::make_expand_beam(layer, query)?;
                Ok(crate::provider::SearchAccessor::new(
                    layer.store.neighbors(),
                    expand_beam,
                    provider,
                    layer.store.frozen(),
                    counters,
                ))
            }

            fn __prune_accessor<'a>(
                layer: &'a Full<Self>,
                counters: LocalCounters<'a>,
            ) -> ANNResult<crate::provider::PruneAccessor<'a>> {
                let prune = <$T>::make_prune(layer)?;
                Ok(crate::provider::PruneAccessor::new(
                    prune,
                    layer.store.neighbors(),
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

    use diskann_utils::lazy_format;
    use hashbrown::{HashMap, HashSet};
    use rand::{Rng, SeedableRng, rngs::StdRng};

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

        let full = <_ as layers::LayerConfig>::build(
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
            <_ as layers::Layer>::id_limit(&full),
            IdLimit::new(capacity.value() as u32 + 2)
        );
        assert_eq!(<_ as layers::Layer>::capacity(&full), capacity);

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

    #[derive(Debug)]
    struct TestDistance;

    impl Distance<f32, f32> for TestDistance {
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

        assert_eq!(<_ as layers::Layer>::capacity(&full), capacity);
        assert_eq!(<_ as layers::Layer>::id_limit(&full), id_limit);

        let mut available: HashSet<u32> = (0..capacity.value()).map(|i| i as u32).collect();

        // Insert the values 0 to 10.
        for i in 0u32..10 {
            let guard = <_ as layers::Set<&[f32]>>::set(&full, &[i as f32]).unwrap();

            let id = <_ as layers::Guard>::id(&guard);

            assert!(
                available.remove(&id),
                "insertion should return available slots",
            );

            assert!(
                points.insert(id, i as f32).is_none(),
                "insertion should not repeat",
            );

            <_ as layers::Guard>::publish(guard);
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

            let g0 = <_ as layers::Set<&[f32]>>::set(&full, &[1000.0]).unwrap();
            let g1 = <_ as layers::Set<&[f32]>>::set(&full, &[2000.0]).unwrap();
            let g2 = <_ as layers::Set<&[f32]>>::set(&full, &[3000.0]).unwrap();
            let g3 = <_ as layers::Set<&[f32]>>::set(&full, &[4000.0]).unwrap();

            {
                let g0_id = <_ as layers::Guard>::id(&g0);
                <_ as layers::Guard>::publish(g0);
                <_ as layers::Layer>::retire(&full, g0_id).unwrap();
            }

            {
                let g1_id = <_ as layers::Guard>::id(&g1);
                <_ as layers::Guard>::publish(g1);
                <_ as layers::Layer>::retire(&full, g1_id).unwrap();
            }

            let query = -1.0f32;

            let into =
                IntoExpandBeam::new(&full, Calf::Borrowed(std::slice::from_ref(&query))).unwrap();

            let expand = ExpandBeam::new(into, prefetch::Loop::new(), TestDistance);

            assert_eq!(<_ as layers::ExpandBeam>::id_limit(&expand), id_limit);

            let mut buf = Vec::<(u32, f32)>::new();
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

                // SAFETY: By construction, all entries in `list` are within `id_limit`
                // (verified against this `ExpandBeam` instance.
                //
                // Also by construction `buf` is at least as long as `list`.
                let read =
                    unsafe { <_ as layers::ExpandBeam>::expand_beam(&expand, &list, &mut buf) }
                        .unwrap();

                let expected: Vec<(u32, f32)> = list
                    .iter()
                    .copied()
                    .filter_map(|id| match points.get(&id) {
                        Some(point) => {
                            let expected = point + query;

                            assert!(
                                <_ as layers::Layer>::is_readable(&full, id).unwrap(),
                                "point should be readable"
                            );

                            assert_eq!(
                                <_ as layers::ExpandBeam>::evaluate(&expand, id).unwrap(),
                                Some(expected),
                                "readable points should return valid distances",
                            );

                            Some((id, expected))
                        }
                        None => {
                            assert!(
                                !<_ as layers::Layer>::is_readable(&full, id).unwrap(),
                                "points not yielded by ExpandBeam should be unreadable"
                            );

                            assert!(
                                <_ as layers::ExpandBeam>::evaluate(&expand, id)
                                    .unwrap()
                                    .is_none(),
                                "unreadable points should return `None` for their distance",
                            );

                            None
                        }
                    })
                    .collect();

                assert_eq!(&buf[..read], &*expected);
            }

            assert!(
                <_ as layers::ExpandBeam>::evaluate(&expand, id_limit.value()).is_err(),
                "`ExpandBeam::evaluate` should catch out-of-bounds errors",
            );

            // Ensure we hold onto `g2` and `g3` for the duration of the above check.
            drop(g2);
            drop(g3);
        }
    }

    fn test_prune_inner(
        points: &HashMap<u32, f32>,
        prune: &mut Prune<f32, TestDistance>,
        ids: &[u32],
    ) {
        let mut items: HashMap<u32, Option<layers::PruneKey>> =
            ids.iter().map(|id| (*id, None)).collect();

        let processed = <_ as layers::Prune>::prepare(prune, items.iter_mut()).unwrap();
        assert_eq!(processed, items.values().filter(|i| i.is_some()).count());

        // Ensure that `prepare` agrees with `points`.
        for (k, v) in items.iter() {
            match v {
                Some(_) => assert!(points.contains_key(k)),
                None => assert!(!points.contains_key(k)),
            }
        }

        fn filter((k, v): (&u32, &Option<layers::PruneKey>)) -> Option<(u32, layers::PruneKey)> {
            v.map(|v| (*k, v))
        }

        // Ensure that distances agree.
        for (k0, v0) in items.iter().filter_map(filter) {
            for (k1, v1) in items.iter().filter_map(filter) {
                // Manually implement `TestDistance`.
                let expected = points[&k0] + points[&k1];
                let got = <_ as layers::Prune>::evaluate(prune, v0, v1);
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

        assert_eq!(<_ as layers::Layer>::capacity(&full), capacity);
        assert_eq!(<_ as layers::Layer>::id_limit(&full), id_limit);

        let mut available: HashSet<u32> = (0..capacity.value()).map(|i| i as u32).collect();

        // Insert the values 0 to 10.
        for i in 0u32..10 {
            let guard = <_ as layers::Set<&[f32]>>::set(&full, &[i as f32]).unwrap();

            let id = <_ as layers::Guard>::id(&guard);

            assert!(
                available.remove(&id),
                "insertion should return available slots",
            );

            assert!(
                points.insert(id, i as f32).is_none(),
                "insertion should not repeat",
            );

            <_ as layers::Guard>::publish(guard);
        }

        // We do several things.
        //
        // 1. We insert two additional IDs but hold their guards without publishing.
        //    This tests that items remain unreadable until they are published.
        //
        // 2. We publish two new points and immediately retire them.
        //    This tests that we correctly make these points unreadable.
        let g0 = <_ as layers::Set<&[f32]>>::set(&full, &[1000.0]).unwrap();
        let g1 = <_ as layers::Set<&[f32]>>::set(&full, &[2000.0]).unwrap();
        let g2 = <_ as layers::Set<&[f32]>>::set(&full, &[3000.0]).unwrap();
        let g3 = <_ as layers::Set<&[f32]>>::set(&full, &[4000.0]).unwrap();

        {
            let g0_id = <_ as layers::Guard>::id(&g0);
            <_ as layers::Guard>::publish(g0);
            <_ as layers::Layer>::retire(&full, g0_id).unwrap();
        }

        {
            let g1_id = <_ as layers::Guard>::id(&g1);
            <_ as layers::Guard>::publish(g1);
            <_ as layers::Layer>::retire(&full, g1_id).unwrap();
        }

        let mut prune = Prune::new(full.reader().unwrap(), TestDistance);

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

        let full = <_ as layers::LayerConfig>::build(
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
            let guard = <_ as layers::Set<&[T]>>::set(&full, &query).unwrap();
            let id = <_ as layers::Guard>::id(&guard);
            <_ as layers::Guard>::publish(guard);
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
            let mut points: HashMap<u32, Option<layers::PruneKey>> =
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
