/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt::Debug, num::NonZeroUsize, marker::PhantomData};

use diskann::{ANNError, ANNResult, utils::IntoUsize};
use diskann_utils::views::Matrix;
use diskann_vector::{
    UnalignedSlice,
    conversion::SliceCast,
    distance::{Cosine, CosineNormalized, InnerProduct, Metric, Specialize, SquaredL2},
};
use diskann_wide::{
    ARCH,
    arch::{Current, FTarget2},
};
use half::f16;
use thiserror::Error;

use crate::{
    arch::Prefetch,
    counters::LocalCounters,
    epoch, layers,
    num::{Bytes, Capacity, IdLimit, MaxDegree},
    store::{
        self, Store,
        invasive::{self, Invasive},
    },
    tag::AtomicTag,
};

#[derive(Debug, Clone)]
pub struct Config<T> {
    layout: store::Layout,
    metric: Metric,
    start_points: Matrix<T>,
    store: store::Config,
    lookahead: Option<NonZeroUsize>,
}

const DEFAULT_LOOKAHEAD: NonZeroUsize = NonZeroUsize::new(8).unwrap();

impl<T> Config<T> {
    pub fn new(
        capacity: Capacity,
        max_degree: MaxDegree,
        metric: Metric,
        start_points: Matrix<T>,
    ) -> Self {
        Self {
            layout: store::Layout::new(
                capacity,
                max_degree,
                start_points.nrows().try_into().unwrap(),
            ),
            metric,
            start_points,
            store: store::Config::default(),
            lookahead: Some(DEFAULT_LOOKAHEAD),
        }
    }

    pub fn store(mut self, config: store::Config) -> Self {
        self.store = config;
        self
    }

    pub fn prefetch(mut self, lookahead: Option<NonZeroUsize>) -> Self {
        self.lookahead = lookahead;
        self
    }

    /// Return the vector dimension of this configuration and the resulting [`Full`].
    pub fn dim(&self) -> usize {
        self.start_points.ncols()
    }
}

impl<T> layers::LayerConfig for Config<T>
where
    T: FullPrecision,
{
    type Layer = Full<T>;

    fn build(self) -> ANNResult<Full<T>> {
        Full::new(self)
    }
}

trait FullPrecisionImpl: bytemuck::Pod + std::fmt::Debug + Send + Sync {
    fn make_expand_beam<'a>(
        full: &'a Full<Self>,
        query: &'a [Self],
    ) -> ANNResult<Box<dyn layers::ExpandBeam + 'a>>;

    #[doc(hidden)]
    fn make_prune<'a>(full: &'a Full<Self>) -> ANNResult<Box<dyn layers::Prune + 'a>>;
}

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
    pub fn config(
        capacity: Capacity,
        max_degree: MaxDegree,
        metric: Metric,
        start_points: Matrix<T>,
    ) -> Config<T> {
        Config::new(capacity, max_degree, metric, start_points)
    }

    /// Create a new full-precision layer for data with the given `dim` and `metric`.
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
            let mut slot = store.slot(i).unwrap();
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

    /// Return the number of bytes of the data handles by this [`layers::Layer`].
    pub fn bytes(&self) -> Bytes {
        self.store.plugin().bytes()
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

    fn reader(&self) -> Result<invasive::Reader<'_>, epoch::Unavailable> {
        Ok(Invasive::reader(&self.store)?)
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

///////////
// Prune //
///////////

#[derive(Debug)]
struct Prune<'a, T, D> {
    // Buffered data to prune over.
    buffer: Vec<UnalignedSlice<'a, T>>,
    // A reader into a layer's store.
    reader: store::invasive::Reader<'a>,
    // Type type of the `PureDistanceFunction` used for the implementation.
    _distance: PhantomData<D>,
}

impl<'a, T, D> Prune<'a, T, D> {
    fn new(reader: store::invasive::Reader<'a>) -> Self {
        // This should be ensured at construction time
        debug_assert!(
            reader
                .bytes()
                .value()
                .is_multiple_of(std::mem::size_of::<T>()),
            "internal inveriant violated",
        );

        Self {
            buffer: Vec::new(),
            reader,
            _distance: PhantomData,
        }
    }
}

impl<T, D> layers::Prune for Prune<'_, T, D>
where
    T: Send + Sync + 'static + Debug,
    D: for<'any> FTarget2<Current, f32, UnalignedSlice<'any, T>, UnalignedSlice<'any, T>>
        + Send
        + Sync
        + Debug,
{
    fn prepare(
        &mut self,
        items: hashbrown::hash_map::IterMut<'_, u32, Option<layers::PruneKey>>,
    ) -> ANNResult<layers::PruneKey> {
        let mut counter = layers::PruneKey::counter();
        self.buffer.clear();
        self.buffer.reserve(items.len());

        for (id, key) in items {
            if let Some(v) = self.reader.read(id.into_usize()) {
                self.buffer.push(unsafe {
                    UnalignedSlice::new(
                        v.as_ptr().cast::<T>(),
                        self.reader.bytes().value() / std::mem::size_of::<T>(),
                    )
                });

                *key = Some(counter);

                // Potential overflow issue - but it's exceedingly unlikely that
                // someone will provide a prune list exceeding `u16::MAX`.
                //
                // In addition, `diskann` limits this bound as well.
                counter = counter.inc()?;
            }
        }

        Ok(counter)
    }

    fn evaluate(&self, a: layers::PruneKey, b: layers::PruneKey) -> f32 {
        D::run(ARCH, self.buffer[a.index()], self.buffer[b.index()])
    }
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

/// A temporary precursor for [`QueryDistance`] to simplify macros.
#[derive(Debug)]
struct IntoQueryDistance<'a, T, U> {
    query: Calf<'a, T>,
    reader: store::invasive::Reader<'a>,
    lookahead: Option<NonZeroUsize>,
    _data: PhantomData<U>,
}

impl<'a, T, U> IntoQueryDistance<'a, T, U> {
    /// Construct a new [`IntoQueryDistance`] - verifying that
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

    fn bytes_plus_tag(&self) -> Bytes {
        self.reader.bytes_plus_tag()
    }
}

trait Distance<T, U>: std::fmt::Debug + Send + Sync + 'static {
    fn eval(&self, x: UnalignedSlice<'_, T>, u: UnalignedSlice<'_, U>) -> f32;
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
        + std::fmt::Debug + Send + Sync + 'static,
{
    #[inline(always)]
    fn eval(&self, x: UnalignedSlice<'_, T>, y: UnalignedSlice<'_, U>) -> f32 {
        D::run(ARCH, x, y)
    }
}

#[derive(Debug)]
struct PureNoInline<D>(PhantomData<D>);

impl<D> PureNoInline<D> {
    const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T, U, D> Distance<T, U> for PureNoInline<D>
where
    D: for<'any> FTarget2<Current, f32, UnalignedSlice<'any, T>, UnalignedSlice<'any, U>>
    + std::fmt::Debug + Send + Sync + 'static,
{
    #[inline(never)]
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
struct QueryDistance<'a, P, T, U, D> {
    // The original query.
    query: Calf<'a, T>,
    // A reader into a layer's store.
    reader: store::invasive::Reader<'a>,
    // THe prefetch look-ahead.
    lookahead: Option<NonZeroUsize>,
    // The type of the data prefetcher.
    prefetch: P,
    // The type of the distance used for the arguments
    distance: D,
    // The type of the data in the original dataset.
    _data: PhantomData<U>,
}

impl<'a, P, T, U, D> QueryDistance<'a, P, T, U, D> {
    fn new(into: IntoQueryDistance<'a, T, U>, prefetch: P, distance: D) -> Self
    where
        P: Prefetch,
    {
        let IntoQueryDistance {
            query,
            reader,
            lookahead,
            _data,
        } = into;

        assert_eq!(
            prefetch.bytes(),
            reader.bytes_plus_tag(),
            "invalid prefetcher"
        );

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

unsafe impl<P, T, U, D> layers::ExpandBeam for QueryDistance<'_, P, T, U, D>
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
            match unsafe { self.reader.read_in_bounds(i.into_usize()) } {
                Some(data) => Ok(Some(unsafe { self.run_unchecked(data) })),
                None => Ok(None),
            }
        }
    }

    fn id_limit(&self) -> IdLimit {
        self.reader.id_limit()
    }

    unsafe fn expand_beam(&self, list: &[u32], buffer: &mut [(u32, f32)]) -> ANNResult<usize> {
        let len = list.len();
        // let lookahead = self.lookahead.map(|l| l.get()).unwrap_or(0).min(len);
        let lookahead = 8.min(len);

        for j in 0..lookahead {
            // SAFETY: The in-bounds constraint is assured by the caller, both for `j` as well
            // as the validity of the prefetch bounds.
            //
            // We do not materialize the `RawSlice` as a reference.
            unsafe {
                self.prefetch.prefetch(
                    self.reader
                        .read_raw_unchecked(list.get_unchecked(j).into_usize())
                        .as_ptr()
                        .cast(),
                )
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
                // We do not materialize the `RawSlice` as a reference.
                unsafe {
                    self.prefetch.prefetch(
                        self.reader
                            .read_raw_unchecked(list.get_unchecked(j).into_usize())
                            .as_ptr()
                            .cast(),
                    )
                }
                j += 1;
            }

            // SAFETY: Caller asserts that `i` is in-bounds.
            if let Some(data) = unsafe { self.reader.read_in_bounds(i.into_usize()) } {
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
struct QueryDistanceError {
    expected: usize,
    xlen: usize,
}

diskann::convert_error!(QueryDistanceError);

const fn compute_bytes<T>(dim: usize) -> usize {
    dim * std::mem::size_of::<T>() + (AtomicTag::SIZE).value()
}

macro_rules! mint {
    ($into:ident, $T:ty => { $N:literal, $f:ident }) => {{
        mint!($into, { $T, $T } => { $N, $f })
    }};
    ($into:ident, { $T:ty, $U:ty } => { $N:literal, $f:ident }) => {{
        Box::new(QueryDistance::<_, $T, $U, _>::new(
            $into,
            $crate::arch::Unrolled::<{ compute_bytes::<$U>($N) }>::new(),
            Pure::<Specialize<$N, $f>>::new(),
        ))
    }};
    ($into:ident, $T:ty => $f:ident) => {{
        mint!($into, { $T, $T } => $f)
    }};
    ($into:ident, { $T:ty, $U:ty } => $f:ident) => {{
        let bytes = $into.bytes_plus_tag();
        Box::new(QueryDistance::<_, $T, $U, _>::new(
            $into,
            $crate::arch::Loop::new(bytes),
            Pure::<$f>::new(),
        ))
    }};
}

impl FullPrecisionImpl for f32 {
    fn make_expand_beam<'a>(
        full: &'a Full<f32>,
        query: &'a [f32],
    ) -> ANNResult<Box<dyn layers::ExpandBeam + 'a>> {
        let into = IntoQueryDistance::new(full, Calf::Borrowed(query))?;
        let output: Box<dyn layers::ExpandBeam> = match full.metric {
            Metric::L2 => {
                // if full.dim() == 100 {
                //     mint!(into, f32 => { 100, SquaredL2 })
                // } else {
                    mint!(into, f32 => SquaredL2)
                // }
            }
            Metric::InnerProduct => mint!(into, f32 => InnerProduct),
            Metric::Cosine => mint!(into, f32 => Cosine),
            Metric::CosineNormalized => mint!(into, f32 => CosineNormalized),
        };

        Ok(output)
    }

    fn make_prune<'a>(full: &'a Full<Self>) -> ANNResult<Box<dyn layers::Prune + 'a>> {
        let reader = full.reader()?;

        let output: Box<dyn layers::Prune> = match full.metric {
            Metric::L2 => Box::new(Prune::<f32, SquaredL2>::new(reader)),
            Metric::InnerProduct => Box::new(Prune::<f32, InnerProduct>::new(reader)),
            Metric::Cosine => Box::new(Prune::<f32, Cosine>::new(reader)),
            Metric::CosineNormalized => Box::new(Prune::<f32, CosineNormalized>::new(reader)),
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

        let into = IntoQueryDistance::new(full, query)?;

        let output: Box<dyn layers::ExpandBeam> = match full.metric {
            Metric::L2 => {
                // if full.dim() == 100 {
                //     mint!(into, { f32, f16 } => { 100, SquaredL2 })
                // } else {
                    mint!(into, { f32, f16 } => SquaredL2)
                // }
            }
            Metric::InnerProduct => mint!(into, { f32, f16 } => InnerProduct),
            Metric::Cosine => mint!(into, { f32, f16 } => Cosine),
            Metric::CosineNormalized => mint!(into, { f32, f16 } => CosineNormalized),
        };

        Ok(output)
    }

    fn make_prune<'a>(full: &'a Full<Self>) -> ANNResult<Box<dyn layers::Prune + 'a>> {
        let reader = full.reader()?;

        let output: Box<dyn layers::Prune> = match full.metric {
            Metric::L2 => Box::new(Prune::<f16, SquaredL2>::new(reader)),
            Metric::InnerProduct => Box::new(Prune::<f16, InnerProduct>::new(reader)),
            Metric::Cosine => Box::new(Prune::<f16, Cosine>::new(reader)),
            Metric::CosineNormalized => Box::new(Prune::<f16, CosineNormalized>::new(reader)),
        };

        Ok(output)
    }
}

impl FullPrecisionImpl for u8 {
    fn make_expand_beam<'a>(
        full: &'a Full<u8>,
        query: &'a [u8],
    ) -> ANNResult<Box<dyn layers::ExpandBeam + 'a>> {
        let into = IntoQueryDistance::new(full, Calf::Borrowed(query))?;

        let output: Box<dyn layers::ExpandBeam> = match full.metric {
            Metric::L2 => {
                // if full.dim() == 128 {
                //     mint!(into, u8 => { 128, SquaredL2 })
                // } else {
                    mint!(into, u8 => SquaredL2)
                // }
            }
            Metric::InnerProduct => mint!(into, u8 => InnerProduct),
            Metric::Cosine => mint!(into, u8 => Cosine),
            Metric::CosineNormalized => mint!(into, u8 => Cosine),
        };

        Ok(output)
    }

    fn make_prune<'a>(full: &'a Full<Self>) -> ANNResult<Box<dyn layers::Prune + 'a>> {
        let reader = full.reader()?;

        let output: Box<dyn layers::Prune> = match full.metric {
            Metric::L2 => Box::new(Prune::<u8, SquaredL2>::new(reader)),
            Metric::InnerProduct => Box::new(Prune::<u8, InnerProduct>::new(reader)),
            Metric::Cosine => Box::new(Prune::<u8, Cosine>::new(reader)),
            Metric::CosineNormalized => Box::new(Prune::<u8, CosineNormalized>::new(reader)),
        };

        Ok(output)
    }
}

impl FullPrecisionImpl for i8 {
    fn make_expand_beam<'a>(
        full: &'a Full<i8>,
        query: &'a [i8],
    ) -> ANNResult<Box<dyn layers::ExpandBeam + 'a>> {
        let into = IntoQueryDistance::new(full, Calf::Borrowed(query))?;

        let output: Box<dyn layers::ExpandBeam + 'a> = match full.metric {
            Metric::L2 => mint!(into, i8 => SquaredL2),
            Metric::InnerProduct => mint!(into, i8 => InnerProduct),
            Metric::Cosine => mint!(into, i8 => Cosine),
            Metric::CosineNormalized => mint!(into, i8 => Cosine),
        };

        Ok(output)
    }

    fn make_prune<'a>(full: &'a Full<Self>) -> ANNResult<Box<dyn layers::Prune + 'a>> {
        let reader = full.reader()?;

        let output: Box<dyn layers::Prune> = match full.metric {
            Metric::L2 => Box::new(Prune::<i8, SquaredL2>::new(reader)),
            Metric::InnerProduct => Box::new(Prune::<i8, InnerProduct>::new(reader)),
            Metric::Cosine => Box::new(Prune::<i8, Cosine>::new(reader)),
            Metric::CosineNormalized => Box::new(Prune::<i8, CosineNormalized>::new(reader)),
        };

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

// #[cfg(test)]
// #[cfg(not(miri))]
// mod tests {
//     use std::fmt::Display;
//
//     use rand::{Rng, SeedableRng, rngs::StdRng};
//
//     use super::*;
//     // Bring the inherent-call traits into method scope. The `Distance` / `QueryDistance`
//     // traits are not imported: their methods are reached through `&dyn _` trait objects,
//     // which does not require the trait to be in scope.
//     use crate::layers::{AsDistance as _, QueryVisitor, Search as _, Set as _};
//
//     /// Generate random elements of a layer's data type from a seeded RNG.
//     trait Sample: bytemuck::Pod {
//         fn sample<R: Rng>(rng: &mut R) -> Self;
//     }
//
//     impl Sample for f32 {
//         fn sample<R: Rng>(rng: &mut R) -> Self {
//             rng.random_range(-1.0f32..1.0f32)
//         }
//     }
//
//     impl Sample for f16 {
//         fn sample<R: Rng>(rng: &mut R) -> Self {
//             f16::from_f32(rng.random_range(-1.0f32..1.0f32))
//         }
//     }
//
//     impl Sample for u8 {
//         fn sample<R: Rng>(rng: &mut R) -> Self {
//             rng.random()
//         }
//     }
//
//     impl Sample for i8 {
//         fn sample<R: Rng>(rng: &mut R) -> Self {
//             rng.random()
//         }
//     }
//
//     fn gen_vec<T: Sample, R: Rng>(rng: &mut R, dim: usize) -> Vec<T> {
//         (0..dim).map(|_| T::sample(rng)).collect()
//     }
//
//     /// A [`QueryVisitor`] that simply boxes the minted kernel so the test can probe it
//     /// directly. Exercises both `visit` (dynamic) and `visit_sized` (specialized) paths.
//     struct Collect;
//
//     impl<'a> QueryVisitor<'a> for Collect {
//         type Output = Box<dyn layers::QueryDistance + 'a>;
//
//         fn visit<Q>(self, distance: Q) -> Self::Output
//         where
//             Q: layers::QueryDistance + 'a,
//         {
//             Box::new(distance)
//         }
//     }
//
//     /// Compare two distances allowing for floating-point reassociation between the
//     /// specialized / converted kernels and the dynamic reference.
//     fn approx_eq(got: f32, want: f32) -> bool {
//         (got - want).abs() <= 1e-3 + 1e-4 * want.abs()
//     }
//
//     /// Exercise every `Full<T>` API across dimensions `1..=max_dim`.
//     ///
//     /// For each dimension we check that `bytes`/`set` agree, that `distance` and
//     /// `query_distance` are consistent with `DistanceProvider`, and that all of these
//     /// reject byte slices that are too long or too short.
//     fn test_impl<T>(max_dim: usize, ctx: &dyn Display)
//     where
//         T: FullPrecision + Sample + DistanceProvider<T>,
//     {
//         let mut rng = StdRng::seed_from_u64(0x0D15_0ACE ^ max_dim as u64);
//         let metrics = [
//             Metric::L2,
//             Metric::InnerProduct,
//             Metric::Cosine,
//             Metric::CosineNormalized,
//         ];
//
//         for dim in 1..=max_dim {
//             let a = gen_vec::<T, _>(&mut rng, dim);
//             let b = gen_vec::<T, _>(&mut rng, dim);
//
//             // `bytes` and `set` agree: the encoded buffer equals the raw cast bytes.
//             let layer = Full::<T>::new(dim, Metric::L2);
//             assert_eq!(
//                 layer.bytes().value(),
//                 dim * std::mem::size_of::<T>(),
//                 "{ctx}: dim {dim}: unexpected byte length",
//             );
//
//             let mut a_bytes = vec![0u8; layer.bytes().value()];
//             layer.set(&a, &mut a_bytes).unwrap();
//             assert_eq!(
//                 a_bytes.as_slice(),
//                 bytemuck::cast_slice::<T, u8>(&a),
//                 "{ctx}: dim {dim}: set mismatch",
//             );
//
//             let mut b_bytes = vec![0u8; layer.bytes().value()];
//             layer.set(&b, &mut b_bytes).unwrap();
//
//             for metric in metrics {
//                 let full = Full::<T>::new(dim, metric);
//
//                 // Reference value straight from `DistanceProvider`.
//                 let reference =
//                     <T as DistanceProvider<T>>::distance_comparer(metric, Some(dim)).call(&a, &b);
//
//                 // `distance` is built from the same comparer, so it must match exactly.
//                 let distance = full.as_distance();
//                 let via_distance = distance.evaluate(&a_bytes, &b_bytes).unwrap();
//                 assert_eq!(
//                     via_distance, reference,
//                     "{ctx}: dim {dim}, metric {metric:?}: distance != DistanceProvider",
//                 );
//
//                 // `query_distance` computes the same geometry. Specialized and f16-converted
//                 // kernels may reassociate the summation, so compare approximately.
//                 let query = full.query_distance(a.as_slice(), Collect).unwrap();
//                 let via_query = query.evaluate(&b_bytes).unwrap();
//                 assert!(
//                     approx_eq(via_query, via_distance),
//                     "{ctx}: dim {dim}, metric {metric:?}: query {via_query} != distance {via_distance}",
//                 );
//
//                 // Every distance API rejects byte slices that are too long or too short.
//                 let short = &a_bytes[..a_bytes.len() - 1];
//                 let mut long = a_bytes.clone();
//                 long.push(0);
//
//                 assert!(distance.evaluate(short, &b_bytes).is_err());
//                 assert!(distance.evaluate(&long, &b_bytes).is_err());
//                 assert!(distance.evaluate(&a_bytes, short).is_err());
//                 assert!(distance.evaluate(&a_bytes, &long).is_err());
//
//                 assert!(query.evaluate(short).is_err());
//                 assert!(query.evaluate(&long).is_err());
//             }
//
//             // `set` rejects mis-sized element and buffer slices.
//             let mut buf = vec![0u8; layer.bytes().value()];
//             let too_many = gen_vec::<T, _>(&mut rng, dim + 1);
//             assert!(
//                 layer.set(&too_many, &mut buf).is_err(),
//                 "{ctx}: dim {dim}: set accepted an over-long element slice",
//             );
//
//             assert!(
//                 layer.query_distance(&too_many, Collect).is_err(),
//                 "{ctx}: dim {dim}: incorrect query lengths should be rejected"
//             );
//
//             let mut short_buf = vec![0u8; layer.bytes().value().saturating_sub(1)];
//             assert!(
//                 layer.set(&a, &mut short_buf).is_err(),
//                 "{ctx}: dim {dim}: set accepted an under-sized buffer",
//             );
//
//             let too_few = gen_vec::<T, _>(&mut rng, dim - 1);
//             assert!(
//                 layer.query_distance(&too_few, Collect).is_err(),
//                 "{ctx}: dim {dim}: incorrect query lengths should be rejected"
//             );
//         }
//     }
//
//     // `max_dim` must exceed the largest specialized dimension for each type so the
//     // const-generic (`visit_sized`) paths are covered alongside the dynamic ones.
//     #[test]
//     fn full_f32() {
//         test_impl::<f32>(256, &"f32");
//     }
//
//     #[test]
//     fn full_f16() {
//         test_impl::<f16>(256, &"f16");
//     }
//
//     #[test]
//     fn full_u8() {
//         test_impl::<u8>(160, &"u8");
//     }
//
//     #[test]
//     fn full_i8() {
//         test_impl::<i8>(160, &"i8");
//     }
// }
