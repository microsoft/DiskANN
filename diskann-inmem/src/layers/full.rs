/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt::Debug, marker::PhantomData};

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
    counters::LocalCounters,
    layers,
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
}

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
        }
    }

    pub fn store(mut self, config: store::Config) -> Self {
        self.store = config;
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

impl<T> FullPrecision for T
where
    T: FullPrecisionImpl,
{
    fn __search_accessor<'a>(
        layer: &'a Full<Self>,
        query: &'a [Self],
        provider: &'a (dyn std::any::Any + Send + Sync),
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::SearchAccessor<'a>> {
        let expand_beam = T::make_expand_beam(layer, query)?;
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
        let prune = T::make_prune(layer)?;
        Ok(crate::provider::PruneAccessor::new(
            prune,
            layer.store.neighbors(),
            counters,
        ))
    }
}

/// Full-precision data layer.
#[derive(Debug)]
pub struct Full<T>
where
    T: 'static,
{
    dim: usize,
    metric: Metric,
    store: Store<Invasive>,
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
            dim: start_points.ncols(),
            metric,
            store,
            _type: PhantomData,
        })
    }

    /// Return the logical dimension of the data handled by this [`layers::Layer`].
    pub fn dim(&self) -> usize {
        self.dim
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

    fn reader(&self) -> ANNResult<invasive::Reader<'_>> {
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
        assert!(
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

/// A fused query distance based on [`diskann_vector::PureDistanceFunction`] to enable
/// inlining of the final distance function (`D`).
///
/// The type of the embedded query (`T`) is distinct from the expected data-set (`U`) to
/// allow `f16` queries to be pre-converted to `f32`, saving on-the-fly conversion that
/// would otherwise be needed.
#[derive(Debug)]
struct QueryDistance<'a, const PREFETCH: usize, T, U, D> {
    // The original query.
    query: Calf<'a, T>,
    // A reader into a layer's store.
    reader: store::invasive::Reader<'a>,
    // The type of the data in the original dataset.
    _data: PhantomData<U>,
    // The type of the `PureDistanceFunction` used for the implementation.
    _distance: PhantomData<D>,
}

impl<'a, const PREFETCH: usize, T, U, D> QueryDistance<'a, PREFETCH, T, U, D> {
    fn new(query: Calf<'a, T>, reader: store::invasive::Reader<'a>) -> Self {
        // TODO: Check PREFETCH and `query` with the reader's size.
        Self {
            query,
            reader,
            _data: PhantomData,
            _distance: PhantomData,
        }
    }

    fn bytes(&self) -> usize {
        std::mem::size_of::<U>() * self.query.len()
    }

    fn error(&self, len: usize) -> ANNResult<f32> {
        let error = QueryDistanceError {
            expected: self.bytes(),
            xlen: len,
        };

        Err(ANNError::new(error))
    }

    // TODO: Since we control the reader - we can avoid the length check.
    #[inline(always)]
    fn run(&self, x: &[u8]) -> ANNResult<f32>
    where
        D: for<'any> FTarget2<Current, f32, UnalignedSlice<'any, T>, UnalignedSlice<'any, U>>,
    {
        if x.len() != self.bytes() {
            self.error(x.len())
        } else {
            // SAFETY: We've validated that `x` has the correct length.
            let x = unsafe { UnalignedSlice::new(x.as_ptr().cast::<U>(), self.query.len()) };
            Ok(D::run(ARCH, (*self.query).into(), x))
        }
    }
}

// TEMPORARY DEFINITIONS
const LOOKAHEAD: usize = 8;
const BYTES: usize = 0;

unsafe impl<const PREFETCH: usize, T, U, D> layers::ExpandBeam
    for QueryDistance<'_, PREFETCH, T, U, D>
where
    T: Send + Sync + 'static + Debug,
    U: Send + Sync + 'static + Debug,
    D: for<'a> FTarget2<Current, f32, UnalignedSlice<'a, T>, UnalignedSlice<'a, U>>
        + Send
        + Sync
        + Debug,
{
    fn evaluate(&self, i: u32) -> ANNResult<Option<f32>> {
        if !self.reader.is_in_bounds(i.into_usize()) {
            return Err(ANNError::new(OutOfBounds(i)));
        } else {
            match unsafe { self.reader.read_in_bounds(i.into_usize()) } {
                Some(data) => Ok(Some(self.run(data)?)),
                None => Ok(None),
            }
        }
    }

    fn id_limit(&self) -> IdLimit {
        self.reader.id_limit()
    }

    unsafe fn expand_beam(&self, list: &[u32], buffer: &mut [(u32, f32)]) -> ANNResult<usize> {
        let len = list.len();
        let lookahead = LOOKAHEAD.min(len);

        let bytes = if PREFETCH == 0 {
            self.reader.bytes().value()
        } else {
            PREFETCH * std::mem::size_of::<T>() + (AtomicTag::SIZE).value()
        };

        for j in 0..lookahead {
            // SAFETY: The in-bounds constraint is assured by the caller, both for `j` as well
            // as the validity of the prefetch bounds.
            unsafe {
                crate::arch::prefetch(
                    self.reader
                        .read_raw_unchecked(list.get_unchecked(j).into_usize())
                        .as_ptr()
                        .cast(),
                    bytes,
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
                unsafe {
                    crate::arch::prefetch(
                        self.reader
                            .read_raw_unchecked(list.get_unchecked(j).into_usize())
                            .as_ptr()
                            .cast(),
                        bytes,
                    )
                }
                j += 1;
            }

            // SAFETY: Caller asserts that `i` is in-bounds.
            if let Some(data) = unsafe { self.reader.read_in_bounds(i.into_usize()) } {
                // SAFETY: Inherited from caller.
                *unsafe { buffer.get_unchecked_mut(processed) } = (i, self.run(data)?);
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

macro_rules! mint {
    ($query:ident, $reader:ident, $T:ty => { $N:literal, $f:ident }) => {{
        mint!($query, $reader, { $T, $T } => { $N, $f })
    }};
    ($query:ident, $reader:ident, { $T:ty, $U:ty } => { $N:literal, $f:ident }) => {{
        Box::new(QueryDistance::<$N, $T, $U, Specialize<$N, $f>>::new($query, $reader))
    }};
    ($query:ident, $reader:ident, $T:ty => $f:ident) => {{
        mint!($query, $reader, { $T, $T } => $f)
    }};
    ($query:ident, $reader:ident, { $T:ty, $U:ty } => $f:ident) => {{
        Box::new(QueryDistance::<0, $T, $U, $f>::new($query, $reader))
    }};
}

impl FullPrecisionImpl for f32 {
    fn make_expand_beam<'a>(
        full: &'a Full<f32>,
        query: &'a [f32],
    ) -> ANNResult<Box<dyn layers::ExpandBeam + 'a>> {
        full.check_dim(query.len())?;
        let reader = full.reader()?;
        let query = Calf::Borrowed(query);

        let output: Box<dyn layers::ExpandBeam> = match full.metric {
            Metric::L2 => {
                if full.dim() == 100 {
                    mint!(query, reader, f32 => { 100, SquaredL2 })
                } else {
                    mint!(query, reader, f32 => SquaredL2)
                }
            }
            Metric::InnerProduct => {
                mint!(query, reader, f32 => InnerProduct)
            }
            Metric::Cosine => mint!(query, reader, f32 => Cosine),
            Metric::CosineNormalized => mint!(query, reader, f32 => CosineNormalized),
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
        full.check_dim(query.len())?;
        let reader = full.reader()?;

        let mut as_f32: Box<[f32]> = std::iter::repeat_n(0.0, full.dim()).collect();
        diskann_wide::arch::dispatch2(SliceCast::new(), &mut *as_f32, query);
        let query = Calf::Owned(as_f32);

        let output: Box<dyn layers::ExpandBeam> = match full.metric {
            Metric::L2 => {
                if full.dim() == 100 {
                    mint!(query, reader, { f32, f16 } => { 100, SquaredL2 })
                } else {
                    mint!(query, reader, { f32, f16 } => SquaredL2)
                }
            }
            Metric::InnerProduct => mint!(query, reader, { f32, f16 } => InnerProduct),
            Metric::Cosine => mint!(query, reader, { f32, f16 } => Cosine),
            Metric::CosineNormalized => mint!(query, reader, { f32, f16 } => CosineNormalized),
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
        full.check_dim(query.len())?;
        let reader = full.reader()?;

        let query = Calf::Borrowed(query);

        let output: Box<dyn layers::ExpandBeam> = match full.metric {
            Metric::L2 => {
                if full.dim() == 128 {
                    mint!(query, reader, u8 => { 128, SquaredL2 })
                } else {
                    mint!(query, reader, u8 => SquaredL2)
                }
            }
            Metric::InnerProduct => mint!(query, reader, u8 => InnerProduct),
            Metric::Cosine => mint!(query, reader, u8 => Cosine),
            Metric::CosineNormalized => mint!(query, reader, u8 => Cosine),
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
        full.check_dim(query.len())?;
        let reader = full.reader()?;

        let query = Calf::Borrowed(query);

        let output: Box<dyn layers::ExpandBeam + 'a> = match full.metric {
            Metric::L2 => mint!(query, reader, i8 => SquaredL2),
            Metric::InnerProduct => mint!(query, reader, i8 => InnerProduct),
            Metric::Cosine => mint!(query, reader, i8 => Cosine),
            Metric::CosineNormalized => mint!(query, reader, i8 => Cosine),
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
