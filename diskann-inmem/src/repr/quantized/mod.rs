/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A generic quantized data store.

use std::num::NonZeroUsize;

use diskann::{ANNError, ANNResult, error::ErrorContext, utils::IntoUsize};
use diskann_utils::{
    lazy_format,
    views::rowmajor::{self, Matrix},
};
use thiserror::Error;

use crate::{
    counters::LocalCounters,
    num::{Bytes, Capacity, IdLimit, MaxDegree, Range},
    repr,
    store::{
        self, Store,
        cons::{self, Cons},
        intrusive::{self, Intrusive},
        optional::Optional,
        simple::{self, Simple},
    },
};

// Built-in quantizers
pub mod product;
pub mod scalar;
pub mod spherical;

/// Plug-in for a generalized quantizer compatible with [`Quantized`].
///
/// This is applicable for quantizers that have a single, non-split, flat representation for
/// quantized vectors with a fixed number of bytes.
///
/// Three different distance types are allowed:
///
/// * [`Self::query_distance`]: The distance computation used for general queries over an
///   index. Implementations may assume that a query is fixed for the duration of this search
///   and the distances used will not participate in any stage of index construction.
///
///   Queries may come from a different distributions than the standard data corpus.
///
/// * [`Self::insert_distance`]: Distance computations used for the search phase of an
///   insertion. This is like [`Self::query_distance`] but with some important differences:
///
///   * The item being inserted can be assumed to belong to the corpus of data upon which
///     the quantizer is trained.
///
///   * Distance yielded from this computer may be mixed with those returned by
///     [`Self::distance`] for index construction and should be compatible.
///
/// * [`Self::distance`]: General distance computations between quantized items.
///   Unlike the [`QueryDistance`] style methods, the left and right-hand sides of the
///   arguments to the returned [`Distance`] can both change.
///
///   Distance values returned from this method should be compatible with those yielded
///   by [`Self::insert_distance`].
///
/// Distance functors provided by [`Self::distance`], [`Self::insert_distance`], and
/// [`Self::query_distance`] will be inlined into a higher-level kernel, allowing
/// implementations to specialize distances.
pub trait Quantizer: std::fmt::Debug + Send + Sync + 'static {
    /// The number of bytes required for each quantized vector.
    ///
    /// This function should be idempotent.
    fn bytes(&self) -> Bytes;

    /// The full-precision dimension.
    ///
    /// This function should be idempotent.
    fn dim(&self) -> usize;

    /// Quantize full-precision data into the byte buffer.
    ///
    /// Implementations may assume that
    ///
    /// * `quantized.len() == self.bytes().value()`.
    /// * `data.len() == self.dim()`.
    fn quantize(&self, quantized: &mut [u8], data: &[f32]) -> ANNResult<()>;

    /// Return the distance computer.
    ///
    /// Slices of length [`Self::bytes`] will be provided to [`Distance::eval`].
    fn distance<'a, V>(&'a self, visitor: V) -> V::Output
    where
        V: DistanceVisitor<'a>;

    /// Return the query distance computer for `query`.
    ///
    /// Implementations may assume that `query.len() == self.dim()`.
    fn query_distance<'a, V>(&'a self, query: &'a [f32], visitor: V) -> ANNResult<V::Output>
    where
        V: QueryDistanceVisitor<'a>;

    /// Return the distance used during the search phase of insert. Implementations may use
    /// this to provide distance computations that are more compatible with [`Self::distance`].
    ///
    /// Implementations may assume that `query.len() == self.dim()`.
    ///
    /// This defaults to using [`Self::query_distance`].
    fn insert_distance<'a, V>(&'a self, query: &'a [f32], visitor: V) -> ANNResult<V::Output>
    where
        V: QueryDistanceVisitor<'a>,
    {
        self.query_distance(query, visitor)
    }

    /// Construct the [`Backend`] from `self`.
    fn into_backend(self) -> Backend
    where
        Self: Sized,
    {
        Backend {
            inner: inner::Quantizer::new(self),
        }
    }
}

/// Return the distance between two raw byte slices.
///
/// Implementations may not assume any particular alignment of `x` or `y`.
pub trait Distance: std::fmt::Debug + Send + Sync {
    fn eval(&self, x: &[u8], y: &[u8]) -> ANNResult<f32>;
}

/// Bring your own type-erasure for [`Distance`].
pub trait DistanceVisitor<'a> {
    type Output;
    fn visit<D>(self, distance: D) -> Self::Output
    where
        D: Distance + 'a;
}

/// Return the distance between a prepared query and a raw byte slice.
///
/// Implementations may not assume any particular alignment of `x`.
pub trait QueryDistance: std::fmt::Debug + Send + Sync {
    fn eval(&self, x: &[u8]) -> ANNResult<f32>;
}

/// Bring your own type-erasure for [`QueryDistance`].
pub trait QueryDistanceVisitor<'a> {
    type Output;
    fn visit<D>(self, distance: D) -> Self::Output
    where
        D: QueryDistance + 'a;
}

/// The opaque backend quantizer for [`Quantized`].
///
/// This is left opaque to allow for future optimization and customization and is
/// constructible from all types that implement [`Quantizer`] either through
/// [`Quantizer::into_backend`] or the blanket implementation of [`IntoBackend`].
///
/// Since this type is [`IntoBackend`] for itself, it can be used as a type-erased container
/// for multiple [`Quantizer`] implementations, which can simplify quantizer preparation.
pub struct Backend {
    inner: inner::Quantizer,
}

impl Backend {
    /// Construct a new [`Backend`] using [`IntoBackend`].
    pub fn new<T>(quantizer: T) -> Self
    where
        T: IntoBackend,
    {
        quantizer.into_backend()
    }

    fn into_inner(self) -> inner::Quantizer {
        self.inner
    }
}

/// Custom conversion to [`Backend`].
pub trait IntoBackend {
    fn into_backend(self) -> Backend;
}

impl IntoBackend for Backend {
    fn into_backend(self) -> Backend {
        self
    }
}

impl<Q> IntoBackend for Q
where
    Q: Quantizer,
{
    fn into_backend(self) -> Backend {
        Quantizer::into_backend(self)
    }
}

/// Configuration for [`Quantized`].
#[derive(Debug)]
pub struct Config {
    /// The underlying quantizer for the compressed store.
    quantizer: inner::Quantizer,
    capacity: Capacity,
    start_points: rowmajor::Owned<f32>,
    start_point_ids: Range,
    layout: store::Layout,
    store: store::Config,
    lookahead: Option<NonZeroUsize>,
    rerank: Rerank,
}

impl Config {
    /// Create a new [`Config`]. Parameters will be used as described below:
    ///
    /// * `quantizer`: The [`Quantizer`] that will be used to quantize data and compute
    ///   distances.
    ///
    /// * `capacity`: The number of points to allocate space for.
    ///
    /// * `max_degree`: The maximum degree of the internal graph.
    ///
    /// * `start_points`: The points to use as frozen start points in the index.
    ///
    /// * `rerank`: Whether reranking is enabled and, if so, the representation of the
    ///   higher-precision vectors.
    ///
    /// # Errors
    ///
    /// Returns an error under the following conditions:
    ///
    /// * `start_points.ncols() != quantizer.dim()`: The dimensionality of the start points
    ///   must agree with the quantizer.
    ///
    /// * `start_points.nrows() == 0`: Currently, empty start points are not supported.
    ///
    /// * The number of start points exceeds `u32::MAX`.
    pub fn new<T>(
        quantizer: T,
        capacity: Capacity,
        max_degree: MaxDegree,
        start_points: rowmajor::Owned<f32>,
        rerank: Rerank,
    ) -> Result<Self, ConfigError>
    where
        T: IntoBackend,
    {
        Self::new_inner(
            quantizer.into_backend().into_inner(),
            capacity,
            max_degree,
            start_points,
            rerank,
        )
    }

    fn new_inner(
        quantizer: inner::Quantizer,
        capacity: Capacity,
        max_degree: MaxDegree,
        start_points: rowmajor::Owned<f32>,
        rerank: Rerank,
    ) -> Result<Self, ConfigError> {
        let dim = quantizer.dim();
        if dim != start_points.ncols() {
            return Err(ConfigError::dim_mismatch(dim, start_points.ncols()));
        }

        let num_start_points = start_points.nrows();
        if num_start_points == 0 {
            return Err(ConfigError::empty_start_points());
        }

        let too_many_points = || ConfigError::too_many_points(capacity, num_start_points);
        let start_point_ids =
            Range::start_plus_usize(0, num_start_points).ok_or_else(too_many_points)?;

        let id_limit = capacity
            .value()
            .checked_add(num_start_points)
            .and_then(|id_limit| u32::try_from(id_limit).map(IdLimit::new).ok())
            .ok_or_else(too_many_points)?;

        Ok(Self {
            quantizer,
            capacity,
            start_points,
            start_point_ids,
            layout: store::Layout::new(id_limit, max_degree),
            store: store::Config::default(),
            lookahead: Some(DEFAULT_LOOKAHEAD),
            rerank,
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

    /// Build the [`Quantized`] from `self`.
    pub fn build(self) -> ANNResult<Quantized> {
        Quantized::new(self)
    }
}

impl repr::RepresentationConfig for Config {
    type Representation = Quantized;

    fn build(self) -> ANNResult<Quantized> {
        <Config>::build(self)
    }
}

/// Errors that can occur during the construction of [`Config`].
#[derive(Debug, Error)]
#[error(transparent)]
pub struct ConfigError {
    inner: ConfigErrorInner,
}

diskann::convert_error!(ConfigError);

impl ConfigError {
    fn dim_mismatch(quantizer: usize, start_points: usize) -> Self {
        Self {
            inner: ConfigErrorInner::DimMismatch {
                quantizer,
                start_points,
            },
        }
    }

    fn empty_start_points() -> Self {
        Self {
            inner: ConfigErrorInner::EmptyStartPoints,
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

#[derive(Debug, Error)]
enum ConfigErrorInner {
    #[error(
        "quantizer configured for dimension {} but given start points have dimension {}",
        quantizer,
        start_points
    )]
    DimMismatch {
        quantizer: usize,
        start_points: usize,
    },
    #[error("at least one start point must be provided")]
    EmptyStartPoints,
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

/// Choose how data is reranked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rerank {
    /// No reranking will be performed and no space for higher-precision vectors will be
    /// allocated.
    None,

    /// Use 16-bit floating-point numbers to store the higher-precision representation.
    /// These will be used automatically during search to rerank candidates.
    F16 {
        /// The distance metric used during the reranking phase.
        metric: Metric,
    },
}

/// Distance metric to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    SquaredL2,
    InnerProduct,
    Cosine,
}

impl Metric {
    fn as_internal_metric(&self) -> repr::internal::quantization::Metric {
        use repr::internal::quantization::Metric as IMetric;
        match self {
            Self::SquaredL2 => IMetric::SquaredL2,
            Self::InnerProduct => IMetric::InnerProduct,
            Self::Cosine => IMetric::Cosine,
        }
    }
}

impl From<diskann_vector::distance::Metric> for Metric {
    fn from(m: diskann_vector::distance::Metric) -> Self {
        use diskann_vector::distance::Metric as VMetric;
        match m {
            VMetric::L2 => Self::SquaredL2,
            VMetric::InnerProduct => Self::InnerProduct,
            VMetric::Cosine | VMetric::CosineNormalized => Self::Cosine,
        }
    }
}

const DEFAULT_LOOKAHEAD: NonZeroUsize = NonZeroUsize::new(16).unwrap();

///////////////
// Quantized //
///////////////

/// A generic quantized data store.
#[derive(Debug)]
pub struct Quantized {
    store: Store<Cons<Intrusive, Optional<Simple>>>,
    capacity: Capacity,
    start_point_ids: Range,
    quantizer: inner::Quantizer,
    lookahead: Option<NonZeroUsize>,
    reranker: repr::internal::quantization::Reranker,
}

impl Quantized {
    /// Initialize a [`Config`] for this representation.
    ///
    /// See also: [`Config::new`].
    ///
    /// # Errors
    ///
    /// Returns the errors described by [`Config::new`].
    pub fn config<T>(
        quantizer: T,
        capacity: Capacity,
        max_degree: MaxDegree,
        start_points: rowmajor::Owned<f32>,
        rerank: Rerank,
    ) -> Result<Config, ConfigError>
    where
        T: IntoBackend,
    {
        Config::new(quantizer, capacity, max_degree, start_points, rerank)
    }

    fn new(config: Config) -> ANNResult<Self> {
        use repr::internal::quantization::{Rerank as IRerank, Reranker};

        let Config {
            quantizer,
            capacity,
            start_points,
            start_point_ids,
            layout,
            store,
            lookahead,
            rerank,
        } = config;

        let (reranker, rerank_config) = match rerank {
            Rerank::None => Reranker::disabled(),
            Rerank::F16 { metric } => Reranker::new_with_config(
                IRerank::F16,
                metric.as_internal_metric(),
                quantizer.dim(),
            )?,
        };

        let slots = cons::Config::new(Intrusive::config(quantizer.bytes()), rerank_config);

        let store = Store::new(layout, store, slots)?;
        let this = Self {
            store,
            capacity,
            start_point_ids,
            quantizer,
            lookahead,
            reranker,
        };

        // Initialize start points.
        let num_start_points = start_points.nrows();
        for (i, (slot_index, row)) in
            std::iter::zip(start_point_ids, start_points.rows()).enumerate()
        {
            #[expect(
                clippy::expect_used,
                reason = "failing this is an internal, unrecoverable bug"
            )]
            let mut slot = this
                .store
                .slot(slot_index)
                .expect("internal store should leave frozen points available for writing");

            this.set(row, slot.data()).with_context(|| {
                lazy_format!(move, "on start point {} of {}", i + 1, num_start_points)
            })?;

            slot.freeze();
        }

        Ok(this)
    }

    /// Return the dimension of the data held within `self`.
    pub fn dim(&self) -> usize {
        self.quantizer.dim()
    }

    /// * Attempt to compress `v` into the [`cons::Exclusive::first`] position.
    /// * If [`cons::Exclusive::second`] is occupied, use `self.reranker` to store data
    ///   into that slot.
    fn set(
        &self,
        v: &[f32],
        slot: &mut cons::Exclusive<intrusive::Exclusive<'_>, Option<simple::Exclusive<'_>>>,
    ) -> ANNResult<()> {
        self.quantizer.quantize(slot.first().as_mut_slice(), v)?;
        self.reranker.store(v, slot.second());
        Ok(())
    }

    fn create_accessor<'a>(
        &'a self,
        query: &'a [f32],
        provider: &'a (dyn std::any::Any + Send + Sync),
        counters: LocalCounters<'a>,
        args: AccessorArgs,
    ) -> ANNResult<crate::provider::SearchAccessor<'a>> {
        let AccessorArgs { insert } = args;

        // NOTE: `ProcessInto` panics on a query dimension mismatch.
        //
        // To avoid that panic, validate the query dimension early.
        let dim = self.dim();
        let query_dim = query.len();
        if query_dim != dim {
            return Err(ANNError::message(lazy_format!(
                move,
                "query dim {} does not match quantizer dim {}",
                query_dim,
                dim
            )));
        }

        let (expand_beam, post_process) = self.store.guard(|cons, guard| -> ANNResult<_> {
            // This will result in some wasted work if `expand_beam` errors, but that should
            // be a relatively rare occurrence.
            let rerank_if_enabled = !insert;
            let post_process = rerank_if_enabled
                .then(|| {
                    self.reranker
                        .post_process(query, &guard, cons.second(), &counters)
                })
                .flatten();

            let kind = if insert {
                inner::ExpandBeamKind::Insert
            } else {
                inner::ExpandBeamKind::Query
            };

            let expand_beam = self.quantizer.expand_beam(
                query,
                cons.first().reader(guard),
                self.lookahead,
                kind,
            )?;

            Ok((expand_beam, post_process))
        })??;

        Ok(crate::provider::SearchAccessor::new(
            self.store.neighbors(),
            expand_beam,
            post_process,
            provider,
            self.start_point_ids,
            counters,
        ))
    }
}

#[derive(Debug)]
struct AccessorArgs {
    insert: bool,
}

repr::internal::macros::representation!(Quantized);

repr::internal::macros::set_guard!(
    /// A [`repr::Guard`] for [`Quantized`].
    for<'a> cons::Exclusive<intrusive::Exclusive<'a>, Option<simple::Exclusive<'a>>>
);

impl repr::Set<&[f32]> for Quantized {
    type Guard<'a> = Guard<'a>;

    fn set(&self, v: &[f32]) -> ANNResult<Guard<'_>> {
        // Easy check to reject invalid vectors before acquiring an epoch guard.
        let vlen = v.len();
        let dim = self.dim();
        if vlen != dim {
            return Err(ANNError::message(lazy_format!(
                move,
                "vector dim {} does not match quantizer dim {}",
                vlen,
                dim
            )));
        }

        let mut slot = self
            .store
            .acquire()
            .ok_or_else(|| ANNError::message("could not allocate a new slot"))?;

        self.set(v, slot.data())?;
        Ok(Guard::new(slot))
    }
}

impl repr::Search for Quantized {
    type Query<'a> = &'a [f32];

    fn search_accessor<'a>(
        &'a self,
        query: &'a [f32],
        provider: &'a (dyn std::any::Any + Send + Sync),
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::SearchAccessor<'a>> {
        self.create_accessor(query, provider, counters, AccessorArgs { insert: false })
    }
}

impl repr::Insert for Quantized {
    fn insert_search_accessor<'a>(
        &'a self,
        query: Self::Query<'a>,
        provider: &'a (dyn std::any::Any + Send + Sync),
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::SearchAccessor<'a>> {
        self.create_accessor(query, provider, counters, AccessorArgs { insert: true })
    }

    fn prune_accessor<'a>(
        &'a self,
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::PruneAccessor<'a>> {
        let reader = self
            .store
            .guard(|slots, guard| slots.first().reader(guard))?;

        let prune = self.quantizer.prune(reader);

        Ok(crate::provider::PruneAccessor::new(
            prune,
            self.store.neighbors(),
            counters,
        ))
    }
}

//-------//
// inner //
//-------//

mod inner {
    use std::num::NonZeroUsize;

    use diskann::{ANNError, ANNResult};

    use crate::{num::Bytes, prefetch, repr, store::intrusive};

    #[derive(Debug)]
    pub(super) enum ExpandBeamKind {
        Insert,
        Query,
    }

    #[derive(Debug)]
    pub(super) struct Quantizer {
        quantizer: Box<dyn ErasedQuantizer>,

        // These values are hoisted out of the trait object to provide access to them without
        // introducing a jump through a vtable.
        bytes: Bytes,
        dim: usize,
    }

    impl Quantizer {
        pub(super) fn new<T>(quantizer: T) -> Self
        where
            T: super::Quantizer,
        {
            let bytes = quantizer.bytes();
            let dim = quantizer.dim();
            Self {
                quantizer: Box::new(Super(quantizer)),
                bytes,
                dim,
            }
        }

        pub(super) fn bytes(&self) -> Bytes {
            self.bytes
        }

        pub(super) fn dim(&self) -> usize {
            self.dim
        }

        pub(super) fn quantize(&self, quantized: &mut [u8], data: &[f32]) -> ANNResult<()> {
            self.quantizer.quantize(quantized, data)
        }

        pub(super) fn expand_beam<'a>(
            &'a self,
            query: &'a [f32],
            reader: intrusive::Reader<'a>,
            lookahead: Option<NonZeroUsize>,
            kind: ExpandBeamKind,
        ) -> ANNResult<Box<dyn repr::ExpandBeam + 'a>> {
            self.quantizer.expand_beam(query, reader, lookahead, kind)
        }

        pub(super) fn prune<'a>(
            &'a self,
            reader: intrusive::Reader<'a>,
        ) -> Box<dyn repr::Prune + 'a> {
            self.quantizer.prune(reader)
        }
    }

    /// A type-erasure compatible version of [`super::Quantizer`].
    trait ErasedQuantizer: std::fmt::Debug + Send + Sync + 'static {
        fn quantize(&self, quantized: &mut [u8], data: &[f32]) -> ANNResult<()>;

        fn expand_beam<'a>(
            &'a self,
            query: &'a [f32],
            reader: intrusive::Reader<'a>,
            lookahead: Option<NonZeroUsize>,
            kind: ExpandBeamKind,
        ) -> ANNResult<Box<dyn repr::ExpandBeam + 'a>>;

        fn prune<'a>(&'a self, reader: intrusive::Reader<'a>) -> Box<dyn repr::Prune + 'a>;
    }

    #[derive(Debug)]
    struct Distance<T>(T);

    impl<T> repr::internal::RawDistance for Distance<T>
    where
        T: super::Distance,
    {
        type Error = ANNError;

        fn eval(&self, x: &[u8], y: &[u8]) -> ANNResult<f32> {
            self.0.eval(x, y)
        }
    }

    impl<T> repr::internal::RawQueryDistance for Distance<T>
    where
        T: super::QueryDistance,
    {
        type Error = ANNError;

        fn eval(&self, x: &[u8]) -> ANNResult<f32> {
            self.0.eval(x)
        }
    }

    #[derive(Debug)]
    struct DistanceVisitor<'a> {
        reader: intrusive::Reader<'a>,
    }

    impl<'a> super::DistanceVisitor<'a> for DistanceVisitor<'a> {
        type Output = Box<dyn repr::Prune + 'a>;

        fn visit<D>(self, distance: D) -> Self::Output
        where
            D: super::Distance + 'a,
        {
            repr::internal::intrusive::Prune::new(self.reader, Distance(distance)).boxed()
        }
    }

    #[derive(Debug)]
    struct QueryDistanceVisitor<'a> {
        reader: intrusive::Reader<'a>,
        lookahead: Option<NonZeroUsize>,
    }

    impl<'a> super::QueryDistanceVisitor<'a> for QueryDistanceVisitor<'a> {
        type Output = Box<dyn repr::ExpandBeam + 'a>;

        fn visit<D>(self, distance: D) -> Self::Output
        where
            D: super::QueryDistance + 'a,
        {
            repr::internal::intrusive::ExpandBeam::new(
                self.reader,
                Distance(distance),
                prefetch::Loop::new(),
                self.lookahead,
            )
            .boxed()
        }
    }

    /// A new-type wrapper around [`super::Quantized`] to bridge into [`ErasedQuantizer`].
    #[derive(Debug)]
    struct Super<Q>(Q);

    impl<Q> ErasedQuantizer for Super<Q>
    where
        Q: super::Quantizer,
    {
        fn quantize(&self, quantized: &mut [u8], data: &[f32]) -> ANNResult<()> {
            self.0.quantize(quantized, data)
        }
        fn expand_beam<'a>(
            &'a self,
            query: &'a [f32],
            reader: intrusive::Reader<'a>,
            lookahead: Option<NonZeroUsize>,
            kind: ExpandBeamKind,
        ) -> ANNResult<Box<dyn repr::ExpandBeam + 'a>> {
            let visitor = QueryDistanceVisitor { reader, lookahead };
            match kind {
                ExpandBeamKind::Insert => self.0.insert_distance(query, visitor),
                ExpandBeamKind::Query => self.0.query_distance(query, visitor),
            }
        }
        fn prune<'a>(&'a self, reader: intrusive::Reader<'a>) -> Box<dyn repr::Prune + 'a> {
            self.0.distance(DistanceVisitor { reader })
        }
    }
}

///////////////////////
// Shared Test Utils //
///////////////////////

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Debug)]
    pub(super) struct Boxit;

    impl<'a> DistanceVisitor<'a> for Boxit {
        type Output = Box<dyn Distance + 'a>;
        fn visit<D>(self, distance: D) -> Self::Output
        where
            D: Distance + 'a,
        {
            Box::new(distance)
        }
    }

    impl<'a> QueryDistanceVisitor<'a> for Boxit {
        type Output = Box<dyn QueryDistance + 'a>;
        fn visit<D>(self, distance: D) -> Self::Output
        where
            D: QueryDistance + 'a,
        {
            Box::new(distance)
        }
    }

    /// A dyn-compatible wrapper for [`super::Quantizer`].
    pub(super) trait DynQuantizer {
        fn dim(&self) -> usize;
        fn bytes(&self) -> Bytes;
        fn quantize(&self, quantized: &mut [u8], data: &[f32]) -> ANNResult<()>;
        fn distance<'a>(&'a self) -> Box<dyn Distance + 'a>;
        fn query_distance<'a>(&'a self, query: &'a [f32])
        -> ANNResult<Box<dyn QueryDistance + 'a>>;
        fn insert_distance<'a>(
            &'a self,
            query: &'a [f32],
        ) -> ANNResult<Box<dyn QueryDistance + 'a>>;
    }

    impl<Q> DynQuantizer for Q
    where
        Q: Quantizer,
    {
        fn dim(&self) -> usize {
            Quantizer::dim(self)
        }
        fn bytes(&self) -> Bytes {
            Quantizer::bytes(self)
        }
        fn quantize(&self, quantized: &mut [u8], data: &[f32]) -> ANNResult<()> {
            Quantizer::quantize(self, quantized, data)
        }
        fn distance<'a>(&'a self) -> Box<dyn Distance + 'a> {
            Quantizer::distance(self, Boxit)
        }
        fn query_distance<'a>(
            &'a self,
            query: &'a [f32],
        ) -> ANNResult<Box<dyn QueryDistance + 'a>> {
            Quantizer::query_distance(self, query, Boxit)
        }
        fn insert_distance<'a>(
            &'a self,
            query: &'a [f32],
        ) -> ANNResult<Box<dyn QueryDistance + 'a>> {
            Quantizer::insert_distance(self, query, Boxit)
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::assert_matches;

    use diskann::graph::test::synthetic::Grid;
    use diskann_utils::{assert_contains, views::rowmajor::MatrixMut};
    use diskann_vector::distance::{Distance as VDistance, Metric as VMetric};
    use hashbrown::HashMap;

    use crate::{
        counters::Counters,
        num::{LogicalId, SlotId},
        repr::test::{Reference, test_expand_beam, test_prune},
    };

    /////////////////////
    // Quantized Tests //
    /////////////////////

    /// A simple test quantizer that tries its best to convert `f32` to `i8`.
    ///
    /// To test that [`Quantizer::query_distance`] and [`Quantizer::insert_distance`] are
    /// called correctly, we assume that queries are all shifted by `[5.0, 5.0]` and need
    /// to be recentered around zero.
    ///
    /// In other words, a query may come in as `[5.0, 5.0]` but for purposes of all distance
    /// computations, it should be treated as `[0.0, 0.0]`.
    #[derive(Debug, Clone, Copy)]
    struct Round {
        dim: usize,
        distance: VDistance<i8, i8>,
    }

    impl Round {
        const QUERY_OFFSET: f32 = 5.0;

        fn new(dim: usize, metric: Metric) -> Self {
            let vmetric = match metric {
                Metric::SquaredL2 => VMetric::L2,
                Metric::InnerProduct => VMetric::InnerProduct,
                Metric::Cosine => VMetric::Cosine,
            };

            Self {
                dim,
                distance: VDistance::new(vmetric, None),
            }
        }
    }

    impl Quantizer for Round {
        fn bytes(&self) -> Bytes {
            Bytes::new(self.dim)
        }
        fn dim(&self) -> usize {
            self.dim
        }
        fn quantize(&self, quantized: &mut [u8], data: &[f32]) -> ANNResult<()> {
            assert_eq!(quantized.len(), self.dim);
            assert_eq!(data.len(), self.dim);

            for d in data.iter() {
                if !d.is_finite() {
                    return Err(ANNError::message("a value of infinity or NaN was observed"));
                }
            }

            std::iter::zip(quantized.iter_mut(), data.iter())
                .for_each(|(q, d)| *q = ((*d) as i8) as u8);
            Ok(())
        }
        fn distance<'a, V>(&'a self, visitor: V) -> V::Output
        where
            V: DistanceVisitor<'a>,
        {
            visitor.visit(RoundDistance(*self))
        }
        fn query_distance<'a, V>(&'a self, query: &'a [f32], visitor: V) -> ANNResult<V::Output>
        where
            V: QueryDistanceVisitor<'a>,
        {
            let distance = RoundQueryDistance {
                round: *self,
                query: query
                    .iter()
                    .map(|v| (*v - Self::QUERY_OFFSET) as i8)
                    .collect(),
            };
            Ok(visitor.visit(distance))
        }
        fn insert_distance<'a, V>(&'a self, query: &'a [f32], visitor: V) -> ANNResult<V::Output>
        where
            V: QueryDistanceVisitor<'a>,
        {
            let distance = RoundQueryDistance {
                round: *self,
                query: query.iter().map(|v| (*v) as i8).collect(),
            };
            Ok(visitor.visit(distance))
        }
    }

    #[derive(Debug)]
    struct RoundDistance(Round);

    impl Distance for RoundDistance {
        fn eval(&self, x: &[u8], y: &[u8]) -> ANNResult<f32> {
            assert_eq!(x.len(), self.0.dim);
            assert_eq!(y.len(), self.0.dim);
            Ok(self.0.distance.call(
                bytemuck::must_cast_slice::<u8, i8>(x),
                bytemuck::must_cast_slice::<u8, i8>(y),
            ))
        }
    }

    #[derive(Debug)]
    struct RoundQueryDistance {
        round: Round,
        query: Vec<i8>,
    }

    impl QueryDistance for RoundQueryDistance {
        fn eval(&self, x: &[u8]) -> ANNResult<f32> {
            assert_eq!(x.len(), self.round.dim);
            Ok(self
                .round
                .distance
                .call(&self.query, bytemuck::must_cast_slice::<u8, i8>(x)))
        }
    }

    /// See the description in [`make_test_repr`].
    const TEST_LIMIT: IdLimit = IdLimit::new(11);

    // Use the canonical grid layout, but center the data around the origin.
    //
    // This allows cosine distances to return reasonable results as the data is distributed
    // around the origin.
    //
    // To keep computation mostly tractable, we only use a 2D grid with 9 points. The
    // coordinates are as follows:
    //
    // 0:  [-1, -1]
    // 1:  [-1,  0]
    // 2:  [-1, +1]
    //
    // 3:  [ 0, -1]
    // 4:  [ 0,  0]
    // 5:  [ 0, +1]
    //
    // 6:  [+1, -1]
    // 7:  [+1,  0]
    // 8:  [+1, +1]
    //
    // We put two start points at `[-2, -2]` and `[+2, +2]`.
    fn make_test_repr(metric: Metric, rerank: Rerank, fill: bool) -> (Quantized, Reference) {
        let grid = Grid::Two;
        let mut data = grid.data(3);
        let offset = 1.0;
        data.as_mut_slice().iter_mut().for_each(|v| *v -= offset);

        let mut start_points = rowmajor::Owned::from_element(2, data.ncols(), 0.0);
        start_points.row_mut(0).fill(-2.0);
        start_points.row_mut(1).fill(2.0);

        let config = Quantized::config(
            Round::new(data.ncols(), metric),
            Capacity::new(data.nrows()),
            MaxDegree::new(0),
            start_points.clone(),
            rerank,
        )
        .unwrap();

        let quantized = config.build().unwrap();

        assert_eq!(repr::Representation::id_limit(&quantized), TEST_LIMIT);
        assert_eq!(
            repr::Representation::capacity(&quantized).value(),
            data.nrows()
        );
        assert_eq!(quantized.dim(), usize::from(grid.dim()));

        let mut reference = Reference::new(grid.dim().into());

        if fill {
            for (i, row) in data.rows().enumerate() {
                let guard = repr::Set::set(&quantized, row).unwrap();
                let id = repr::Guard::id(&guard);

                reference.insert(LogicalId(i), SlotId(id), row);
                repr::Guard::publish(guard);
            }
        }

        // Insert frozen points into the reference.
        //
        // We give frozen points logical IDs at the end of the ID space.
        for (slot, point) in std::iter::zip(quantized.start_point_ids, start_points.rows()) {
            reference.insert(
                LogicalId(slot.into_usize() + data.nrows()),
                SlotId(slot),
                point,
            );
        }

        (quantized, reference)
    }

    /// Here, we don't test the whole `expand_beam` loop. That would be a waste of time and
    /// is already tested by other code.
    ///
    /// Instead we use:
    ///
    /// * `ExpandBeam::evaluate` to verify that the correct type of distance computer is
    ///   created.
    ///
    /// * `Prune` to validate that the correct distance computer is made.
    ///
    /// * If reranking exists, that reranking works as expected.
    fn test_distances(
        quantized: &Quantized,
        reference: &mut Reference,
        metric: Metric,
        rerank: Rerank,
        ctx: &dyn std::fmt::Display,
    ) {
        assert_eq!(
            repr::Representation::id_limit(quantized),
            TEST_LIMIT,
            "{ctx}"
        );

        let insert_vector = [10.0, -10.0];
        let query_vector = insert_vector.map(|q| q + Round::QUERY_OFFSET);

        // Generate a couple of data points in the dataset.
        let i0 = LogicalId(2);
        let s0 = reference.slot_id_for(i0);

        let i1 = LogicalId(5);
        let s1 = reference.slot_id_for(i1);

        let i2 = LogicalId(10);
        let s2 = reference.slot_id_for(i2);

        // Delete values.
        let i3_deleted = LogicalId(8);
        let s3_deleted = reference.slot_id_for(i3_deleted);

        repr::Representation::retire(quantized, s3_deleted.value()).unwrap();
        reference.delete(i3_deleted);

        let i4_deleted = LogicalId(4);
        let s4_deleted = reference.slot_id_for(i4_deleted);

        repr::Representation::retire(quantized, s4_deleted.value()).unwrap();
        reference.delete(i4_deleted);

        // Extract values.
        let v0 = &reference[i0];
        let v1 = &reference[i1];
        let v2 = &reference[i2];

        // The Round quantizer is exact for these integral values.
        let f = VDistance::<f32, f32>::new(metric.as_internal_metric().as_vector_metric(), None);

        // True distances between the insert vector and the dataset vectors.
        let insert_distances = HashMap::from_iter([
            (s0, f.call(&insert_vector, v0)),
            (s1, f.call(&insert_vector, v1)),
            (s2, f.call(&insert_vector, v2)),
        ]);

        // Distances between the un-shifted query vector and the dataset vectors.
        let query_distances = insert_distances.clone();

        // THe true distances between the shifted query vector and the dataset vectors.
        let true_query_distances = HashMap::from_iter([
            (s0, f.call(&query_vector, v0)),
            (s1, f.call(&query_vector, v1)),
            (s2, f.call(&query_vector, v2)),
        ]);

        // Insert
        {
            let counters = Counters::new();
            let mut sa = repr::Insert::insert_search_accessor(
                quantized,
                insert_vector.as_slice(),
                &(),
                counters.local(),
            )
            .unwrap();

            assert!(
                sa.get_post_process().is_none(),
                "insert accessors should not build a post-processor -- {}",
                ctx,
            );

            test_expand_beam(
                sa.get_expand_beam(),
                TEST_LIMIT,
                insert_distances,
                &[s0, s1, s3_deleted, s2, s4_deleted],
                ctx,
            );
        }

        // Search
        {
            let counters = Counters::new();
            let mut sa = repr::Search::search_accessor(
                quantized,
                query_vector.as_slice(),
                &(),
                counters.local(),
            )
            .unwrap();

            test_expand_beam(
                sa.get_expand_beam(),
                TEST_LIMIT,
                query_distances,
                &[s0, s1, s3_deleted, s2, s4_deleted],
                ctx,
            );

            if rerank == Rerank::None {
                assert!(
                    sa.get_post_process().is_none(),
                    "search accessors should not build a post-processor with reranking disabled -- {}",
                    ctx,
                );
            } else {
                let post_process = match sa.get_post_process() {
                    Some(post_process) => post_process,
                    None => panic!("expected a post processor -- {}", ctx),
                };

                repr::internal::quantization::rerank::test_rerank(
                    post_process,
                    true_query_distances,
                    &[s0, s1, s3_deleted, s2, s4_deleted],
                    ctx,
                );
            }
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
            let mut pa = repr::Insert::prune_accessor(quantized, counters.local()).unwrap();

            test_prune(
                pa.get_prune(),
                distances,
                &[s0, s1, s3_deleted, s2, s4_deleted],
                ctx,
            );
        }
    }

    /// The happy-path entry point.
    ///
    /// Note that this method does a grid of the supported parameters. Adding new values
    /// to these parameters has a multiplicative effect on runtime.
    ///
    /// This mainly matters for Miri tests. Currently, the Miri test for this function takes
    /// about 10 seconds. If it starts to take much longer, this test should be split into
    /// multiple entry points for parallelism.
    #[test]
    fn test_quantized() {
        #[derive(Debug, Clone, Copy)]
        enum RerankKind {
            None,
            F16,
        }

        let metrics = [Metric::SquaredL2, Metric::InnerProduct, Metric::Cosine];
        let rerank_kinds = [RerankKind::None, RerankKind::F16];

        for metric in metrics {
            for rerank_kind in rerank_kinds {
                let rerank = match rerank_kind {
                    RerankKind::None => Rerank::None,
                    RerankKind::F16 => Rerank::F16 { metric },
                };

                let (quantized, mut reference) = make_test_repr(metric, rerank, true);

                test_distances(
                    &quantized,
                    &mut reference,
                    metric,
                    rerank,
                    &format_args!("metric = {:?}, rerank = {:?}", metric, rerank_kind),
                );
            }
        }
    }

    #[test]
    fn test_metric() {
        use diskann_vector::distance::Metric as VMetric;

        assert_matches!(Metric::from(VMetric::L2), Metric::SquaredL2);
        assert_matches!(Metric::from(VMetric::InnerProduct), Metric::InnerProduct);
        assert_matches!(Metric::from(VMetric::Cosine), Metric::Cosine);
        assert_matches!(Metric::from(VMetric::CosineNormalized), Metric::Cosine);
    }

    //-------------//
    // Error Paths //
    //-------------//

    #[test]
    fn test_config_dim_mismatch() {
        let dim = 5;
        let round = Round::new(dim, Metric::SquaredL2);

        let start_points = rowmajor::Owned::from_element(1, dim + 1, 0.0f32); // Wrong number of columns
        let err = Quantized::config(
            round,
            Capacity::new(10),
            MaxDegree::new(0),
            start_points,
            Rerank::None,
        )
        .unwrap_err();

        let msg = err.to_string();
        assert_contains!(
            msg,
            "quantizer configured for dimension 5 but given start points have dimension 6"
        );
    }

    #[test]
    fn test_empty_start_points() {
        let dim = 5;
        let round = Round::new(dim, Metric::SquaredL2);

        let start_points = rowmajor::Owned::from_element(0, dim, 0.0f32); // Empty
        let err = Quantized::config(
            round,
            Capacity::new(10),
            MaxDegree::new(0),
            start_points,
            Rerank::None,
        )
        .unwrap_err();

        let msg = err.to_string();
        assert_contains!(msg, "at least one start point must be provided");
    }

    #[test]
    fn test_too_many_start_points() {
        let dim = 0;
        let round = Round::new(0, Metric::SquaredL2);

        // Too big.
        let start_points = rowmajor::Owned::from_element((u32::MAX as usize) + 1, dim, 0.0f32);
        let err = Quantized::config(
            round,
            Capacity::new(10),
            MaxDegree::new(0),
            start_points,
            Rerank::None,
        )
        .unwrap_err();

        let msg = err.to_string();
        assert_contains!(msg, "number of points");
        assert_contains!(msg, "start points");
        assert_contains!(msg, "exceeds u32::MAX");
    }

    #[test]
    fn test_build_error_uncompressible_query() {
        let dim = 5;
        let product = Round::new(5, Metric::SquaredL2);

        let start_points = rowmajor::Owned::from_element(1, dim, f32::INFINITY);
        let config = Quantized::config(
            product,
            Capacity::new(10),
            MaxDegree::new(0),
            start_points,
            Rerank::None,
        )
        .unwrap();

        let err = repr::RepresentationConfig::build(config).unwrap_err();
        let msg = err.to_string();
        assert_contains!(
            msg,
            "a value of infinity or NaN was observed",
            "we tried to compress a start point with infinites in it - this should error"
        );

        assert_contains!(
            msg,
            "1 of 1",
            "error message should contain which query errored",
        );
    }

    #[test]
    fn test_set_capacity_exhaustion() {
        let (quantized, _) = make_test_repr(Metric::SquaredL2, Rerank::None, true);

        let err = repr::Set::set(&quantized, &[1.0, 2.0]).unwrap_err();
        assert_contains!(err.to_string(), "could not allocate a new slot",);
    }

    #[test]
    fn test_set_dim_mismatch() {
        let (quantized, _) = make_test_repr(Metric::SquaredL2, Rerank::None, false);

        let err = repr::Set::set(&quantized, &[1.0, 2.0, 3.0]).unwrap_err();
        assert_contains!(
            err.to_string(),
            "vector dim 3 does not match quantizer dim 2",
        );
    }

    #[test]
    fn test_search_dim_mismatch() {
        let (quantized, _) = make_test_repr(Metric::SquaredL2, Rerank::None, false);

        let counters = Counters::new();
        let err =
            repr::Search::search_accessor(&quantized, &[1.0], &(), counters.local()).unwrap_err();

        assert_contains!(
            err.to_string(),
            "query dim 1 does not match quantizer dim 2"
        );
    }

    #[test]
    fn test_insert_slot_recovery() {
        let (quantized, _) = make_test_repr(
            Metric::SquaredL2,
            Rerank::F16 {
                metric: Metric::SquaredL2,
            },
            true,
        );

        // Free up one slot.
        let slot = quantized.start_point_ids.end();
        repr::Representation::retire(&quantized, slot).unwrap();

        // Insert something that is incompressible.
        let err = repr::Set::set(&quantized, &[f32::INFINITY, f32::INFINITY]).unwrap_err();
        assert_contains!(err.to_string(), "infinity");

        // If we insert again, this should succeed.
        let guard = repr::Set::set(&quantized, &[1.0, 2.0]).unwrap();
        assert_eq!(repr::Guard::id(&guard), slot);
    }
}
