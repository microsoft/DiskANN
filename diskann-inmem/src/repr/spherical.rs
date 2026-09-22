/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A quantized store for RabitQ style compressed vectors.

use std::num::NonZeroUsize;

use diskann::{ANNError, ANNResult, error::ErrorContext, utils::IntoUsize};
use diskann_quantization::{
    alloc::{GlobalAllocator, Poly, ScopedAllocator},
    spherical::{SupportedMetric, iface},
};
use diskann_utils::{lazy_format, views::Matrix};
use diskann_vector::distance::{Distance, DistanceProvider};
use half::f16;
use thiserror::Error;

use crate::{
    counters::LocalCounters,
    epoch,
    num::{Bytes, Capacity, IdLimit, MaxDegree},
    prefetch,
    repr::{self, internal::Calf},
    store::{
        self, Store,
        cons::{self, Cons},
        intrusive::{self, Intrusive},
        optional::Optional,
        simple::{self, Simple},
    },
};

/// The configuration for a [`Spherical`] representation.
#[derive(Debug)]
pub struct Config {
    /// The underlying quantizer for the compressed store.
    quantizer: Poly<dyn iface::Quantizer>,
    /// The start points. These must have dimensions equal to `quantizer.full_dim()`.
    start_points: Matrix<f32>,
    layout: store::Layout,
    store: store::Config,
    lookahead: Option<NonZeroUsize>,
    rerank: Rerank,
}

const DEFAULT_LOOKAHEAD: NonZeroUsize = NonZeroUsize::new(16).unwrap();

impl Config {
    /// Create a new [`Config`]. Parameters will be used as described below:
    ///
    /// * `quantizer`: The [`iface::Quantizer`] used to compress vectors and compute
    ///   distances among compressed vectors.
    ///
    /// * `capacity`: The number points allocate space fore.
    ///
    /// * `max_degree`: The maximum degree of the internal graph.
    ///
    /// * `start_points`: The points to use as frozen start points in the index.
    ///
    /// * `rerank`: Whether or not reranking is enabled and if so, the representation of the
    ///   higher precision vectors.
    ///
    /// # Errors
    ///
    /// Errors under the following conditions:
    ///
    /// * `start_points.ncols() != quantizer.full_dim()`: The dimensionality of the start
    ///   points must agree with the quantizer.
    ///
    /// * `start_points.nrows() == 0`: Currently, empty start-points are not supported.
    ///
    /// * The number of start points exceeds `u32::MAX`.
    pub fn new(
        quantizer: Poly<dyn iface::Quantizer>,
        capacity: Capacity,
        max_degree: MaxDegree,
        start_points: Matrix<f32>,
        rerank: Rerank,
    ) -> Result<Self, ConfigError> {
        let quantizer_dim = quantizer.full_dim();
        if quantizer_dim != start_points.ncols() {
            return Err(ConfigError::dim_mismatch(
                quantizer_dim,
                start_points.ncols(),
            ));
        }

        if start_points.nrows() == 0 {
            return Err(ConfigError::empty_start_points());
        }

        let num_start_points: u32 = start_points
            .nrows()
            .try_into()
            .map_err(|_| ConfigError::too_many_start_points(start_points.nrows()))?;

        Ok(Self {
            quantizer,
            start_points,
            layout: store::Layout::new(capacity, max_degree, num_start_points),
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

    /// Build the [`Spherical`] from `self`.
    pub fn build(self) -> ANNResult<Spherical> {
        Spherical::new(self)
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

    fn too_many_start_points(num_start_points: usize) -> Self {
        Self {
            inner: ConfigErrorInner::TooManyStartPoints { num_start_points },
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
    #[error("{} start points exceeds u32::MAX", num_start_points)]
    TooManyStartPoints { num_start_points: usize },
}

impl repr::RepresentationConfig for Config {
    type Representation = Spherical;

    fn build(self) -> ANNResult<Spherical> {
        <Config>::build(self)
    }
}

/// Choose how data is going to be reranked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rerank {
    /// No reranking will be performed and no space for higher precision vectors will be
    /// allocated in [`Spherical`].
    None,

    /// Use 16-bit floating point numbers to store the higher-precision representation.
    /// These will be used automatically during search to rerank candidates.
    F16,
}

/// Internal representation of [`Rerank`].
///
/// This is used for computing distances among the raw values in the auxiliary store.
#[derive(Debug)]
enum Reranker {
    None,
    F16(Distance<f32, f16>),
}

fn convert_metric(metric: SupportedMetric) -> diskann_vector::distance::Metric {
    use diskann_vector::distance::Metric;

    match metric {
        SupportedMetric::SquaredL2 => Metric::L2,
        SupportedMetric::InnerProduct => Metric::InnerProduct,
        SupportedMetric::Cosine => Metric::Cosine,
    }
}

impl Reranker {
    /// Construct a new [`Reranker`] and a [`store::slots::SlotsConfig`]  for the auxiliary
    /// store.
    fn new_with_config(
        rerank: Rerank,
        metric: SupportedMetric,
        dim: usize,
    ) -> (Self, Option<simple::Config>) {
        let this = match rerank {
            Rerank::None => Self::None,
            Rerank::F16 => {
                let distance = <f32 as DistanceProvider<f16>>::distance_comparer(
                    convert_metric(metric),
                    Some(dim),
                );

                Self::F16(distance)
            }
        };

        let config = match &this {
            Self::None => None,
            Self::F16(_) => Some(Simple::config(this.bytes_for(dim))),
        };

        (this, config)
    }

    #[expect(
        clippy::expect_used,
        reason = "the arithmetic should not overflow for the feasible `dim` values"
    )]
    fn bytes_for(&self, dim: usize) -> Bytes {
        match self {
            Self::None => Bytes::new(0),
            Self::F16(_) => Bytes::new(
                dim.checked_mul(2)
                    .expect("f16 is smaller than the f32 in the quantizer"),
            ),
        }
    }

    /// Create a [`repr::PostProcess`].
    ///
    /// This assumes that `simple` has the same dimensions as `self`'s contained distance
    /// computation and that `guard` belongs to `simple`.
    ///
    /// # Pre-conditions
    ///
    /// This requires that `slots` is the [`store::slots::Slots`] created from the
    /// configuration returned in [`Self::new_with_config`].
    fn post_process<'a>(
        &'a self,
        query: &'a [f32],
        guard: &epoch::Guard<'a>,
        slots: &'a Optional<store::simple::Simple>,
        counters: &LocalCounters<'a>,
    ) -> Option<Box<dyn repr::PostProcess + 'a>> {
        match (self, slots.slots()) {
            (Self::None, None) => None,
            (Self::F16(distance), Some(simple)) => {
                let distance = repr::full::QueryDistance::new(Calf::Borrowed(query), *distance);
                let reader = simple.reader(guard.share());
                let post_process =
                    repr::internal::simple::Reranker::new(reader, distance, counters.fork());
                Some(Box::new(post_process))
            }
            _ => unreachable!("invalid combination of arguments"),
        }
    }

    /// Store the vector `v` into the raw buffer `buf`.
    ///
    /// # Pre-conditions
    ///
    /// `buf` must be consistent with the configuration returned from [`Self::new_with_config`],
    /// and may only be `None` if that configuration was `None`.
    ///
    /// If it is `Some`, this function may panic if its length is not consistent with the
    /// original configuration.
    fn store(&self, v: &[f32], buf: &mut Option<simple::Exclusive<'_>>) {
        match (self, buf) {
            (Self::None, None) => {}
            (Self::F16(_), Some(exclusive)) => {
                use diskann_vector::conversion::CastFromSlice;
                bytemuck::cast_slice_mut::<u8, f16>(exclusive.as_mut_slice()).cast_from_slice(v);
            }
            _ => unreachable!("invalid combination of arguments"),
        }
    }
}

/// Spherically quantized data representation.
#[derive(Debug)]
pub struct Spherical {
    store: Store<Cons<Intrusive, Optional<Simple>>>,
    quantizer: Poly<dyn iface::Quantizer>,
    // These values come directly from `quantizer`, but are hoisted out to avoid a
    // trait-object function call when accessing.
    full_dim: usize,
    lookahead: Option<NonZeroUsize>,
    reranker: Reranker,
}

impl Spherical {
    /// Initialize a [`Config`] for this representation.
    ///
    /// See also: [`Config::new`].
    ///
    /// # Errors
    ///
    /// Returns the errors described by [`Config::new`].
    pub fn config(
        quantizer: Poly<dyn iface::Quantizer>,
        capacity: Capacity,
        max_degree: MaxDegree,
        start_points: Matrix<f32>,
        rerank: Rerank,
    ) -> Result<Config, ConfigError> {
        Config::new(quantizer, capacity, max_degree, start_points, rerank)
    }

    /// Create a new full-precision representation from `config`.
    ///
    /// See: [`Config::build`].
    fn new(config: Config) -> ANNResult<Self> {
        let Config {
            quantizer,
            start_points,
            layout,
            store,
            lookahead,
            rerank,
        } = config;

        let full_dim = quantizer.full_dim();
        let (reranker, rerank_config) =
            Reranker::new_with_config(rerank, quantizer.metric(), full_dim);

        let slots = cons::Config::new(
            Intrusive::config(Bytes::new(quantizer.bytes())),
            rerank_config,
        );

        let store = Store::new(layout, store, slots)?;

        let this = Self {
            store,
            quantizer,
            full_dim,
            lookahead,
            reranker,
        };

        // Initialize start points.
        let num_start_points = start_points.nrows();
        for (i, row) in std::iter::zip(this.store.frozen(), start_points.row_iter()) {
            #[expect(
                clippy::expect_used,
                reason = "failing this is an internal, unrecoverable bug"
            )]
            let mut slot = this
                .store
                .slot(i)
                .expect("internal store should leave frozen-points available for writing");

            this.set(row, slot.data()).with_context(|| {
                lazy_format!(move, "on start point {} of {}", i + 1, num_start_points)
            })?;

            slot.freeze();
        }

        Ok(this)
    }

    /// Return the dimension of the data held within `self`.
    pub fn dim(&self) -> usize {
        self.full_dim
    }

    /// * Attempt to compress `v` into the [`cons::Exclusive::first`] position.
    /// * If [`cons::Exclusive::second`] is occupied, use `self.reranker` to store data
    ///   into that slot.
    fn set(
        &self,
        v: &[f32],
        slot: &mut cons::Exclusive<intrusive::Exclusive<'_>, Option<simple::Exclusive<'_>>>,
    ) -> ANNResult<()> {
        self.quantizer
            .compress(
                v,
                iface::OpaqueMut::new(slot.first().as_mut_slice()),
                ScopedAllocator::global(),
            )
            .map_err(ANNError::new)?;

        self.reranker.store(v, slot.second());

        Ok(())
    }

    /// Shared entry point for creating [`crate::provider::SearchAccessor`].
    ///
    /// See: [`AccessorArgs`].
    fn create_accessor<'a>(
        &'a self,
        query: &'a [f32],
        provider: &'a (dyn std::any::Any + Send + Sync),
        counters: LocalCounters<'a>,
        args: AccessorArgs,
    ) -> ANNResult<crate::provider::SearchAccessor<'a>> {
        let AccessorArgs {
            layout,
            allow_rescale,
            rerank_if_enabled,
        } = args;

        // Create the query computer.
        //
        // We do this first because this is one of the most likely things to fail since it
        // operates on largely untrusted data. If it does fail, we save the work of acquiring
        // epoch guards etc.
        let query_computer = self
            .quantizer
            .fused_query_computer(
                query,
                layout,
                allow_rescale,
                GlobalAllocator,
                ScopedAllocator::global(),
            )
            .map_err(ANNError::new)?;

        // Computer is good - time to make `ExpandBeam` and `PostProcess` (if requested).
        let (expand_beam, post_process) = self.store.guard(|cons, guard| {
            // TODO: Tailor prefetching to the number of cachelines.
            //
            // Inlining distance functions will require work in `diskann-quantization`, so
            // we can at least optimize prefetching.
            let expand_beam = repr::internal::intrusive::ExpandBeam::new(
                cons.first().reader(guard),
                query_computer,
                prefetch::Loop::new(),
                self.lookahead,
            );

            let post_process = rerank_if_enabled
                .then(|| {
                    self.reranker
                        .post_process(query, expand_beam.guard(), cons.second(), &counters)
                })
                .flatten();

            (expand_beam, post_process)
        })?;

        Ok(crate::provider::SearchAccessor::new(
            self.store.neighbors(),
            expand_beam.boxed(),
            post_process,
            provider,
            self.store.frozen(),
            counters,
        ))
    }

    #[cfg(test)]
    fn quantizer(&self) -> &dyn iface::Quantizer {
        &*self.quantizer
    }
}

/// Arguments to [`Spherical::create_accessor`].
#[derive(Debug)]
struct AccessorArgs {
    /// The [`iface::QueryLayout`] to use when compressing the query.
    layout: iface::QueryLayout,

    /// Whether or not the query vector can be rescaled.
    ///
    /// This only applies to query search. For searches used as part of insertion, rescaling
    /// should be disabled to ensure distance are compatible with distances computed during
    /// prune.
    allow_rescale: bool,

    /// Create a [`repr::PostProcess`] for reranking if enabled in the parent struct.
    ///
    /// Since insert does not use a reranking step, this can be left as `false` to save
    /// some processing and allocations.
    rerank_if_enabled: bool,
}

repr::internal::macros::representation!(Spherical);

repr::internal::macros::set_guard!(
    /// A [`repr::Guard`] for [`Spherical`].
    for<'a> cons::Exclusive<intrusive::Exclusive<'a>, Option<simple::Exclusive<'a>>>
);

impl repr::Set<&[f32]> for Spherical {
    type Guard<'a> = Guard<'a>;

    fn set(&self, v: &[f32]) -> ANNResult<Guard<'_>> {
        // Easy check to reject invalid vectors before acquiring an epoch guard.
        if v.len() != self.full_dim {
            let vlen = v.len();
            let full_dim = self.full_dim;

            return Err(ANNError::message(lazy_format!(
                move,
                "vector dim {} does not match quantizer dim {}",
                vlen,
                full_dim
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

impl repr::Search for Spherical {
    type Query<'a> = &'a [f32];

    fn search_accessor<'a>(
        &'a self,
        query: &'a [f32],
        provider: &'a (dyn std::any::Any + Send + Sync),
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::SearchAccessor<'a>> {
        self.create_accessor(
            query,
            provider,
            counters,
            AccessorArgs {
                layout: iface::QueryLayout::FullPrecision,
                allow_rescale: true,
                rerank_if_enabled: true,
            },
        )
    }
}

impl repr::Insert for Spherical {
    fn insert_search_accessor<'a>(
        &'a self,
        query: Self::Query<'a>,
        provider: &'a (dyn std::any::Any + Send + Sync),
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::SearchAccessor<'a>> {
        self.create_accessor(
            query,
            provider,
            counters,
            AccessorArgs {
                layout: iface::QueryLayout::SameAsData,
                allow_rescale: false,
                rerank_if_enabled: false,
            },
        )
    }

    fn prune_accessor<'a>(
        &'a self,
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::PruneAccessor<'a>> {
        let distance = self.quantizer.distance_computer_ref();
        let reader = self
            .store
            .guard(|slots, guard| slots.first().reader(guard))?;

        let prune = repr::internal::intrusive::Prune::new(reader, distance);
        Ok(crate::provider::PruneAccessor::new(
            prune.boxed(),
            self.store.neighbors(),
            counters,
        ))
    }
}

//----------------//
// Trait Adaptors //
//----------------//

impl<A> repr::internal::RawQueryDistance for iface::QueryComputer<A>
where
    A: diskann_quantization::alloc::Allocator + std::fmt::Debug + Send + Sync,
{
    type Error = ANNError;

    fn eval(&self, x: &[u8]) -> Result<f32, Self::Error> {
        use diskann_vector::PreprocessedDistanceFunction;
        self.evaluate_similarity(iface::Opaque::new(x))
            .map_err(ANNError::new)
    }
}

impl repr::internal::RawDistance for &dyn iface::DynDistanceComputer {
    type Error = ANNError;

    fn eval(&self, x: &[u8], y: &[u8]) -> Result<f32, Self::Error> {
        self.evaluate(iface::Opaque::new(x), iface::Opaque::new(y))
            .map_err(ANNError::new)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann::{graph::test::synthetic::Grid, neighbor::Neighbor};
    use diskann_utils::{assert_contains, views::MatrixView};
    use hashbrown::HashMap;

    use crate::{
        counters::Counters,
        num::{LogicalId, SlotId},
        repr::test::Reference,
    };

    #[derive(Debug, Clone, Copy)]
    enum Bits {
        One,
        Two,
        Four,
    }

    fn train_quantizer(
        data: MatrixView<'_, f32>,
        metric: SupportedMetric,
        bits: Bits,
    ) -> Poly<dyn iface::Quantizer> {
        use diskann_quantization::{algorithms::transforms, spherical};
        use rand::{SeedableRng, rngs::StdRng};

        let q = spherical::SphericalQuantizer::train(
            data.as_view(),
            transforms::TransformKind::Null,
            metric,
            spherical::PreScale::None,
            &mut StdRng::seed_from_u64(0x019823745),
            GlobalAllocator,
        )
        .unwrap();

        match bits {
            Bits::One => q.as_quantizer::<1>().unwrap(),
            Bits::Two => q.as_quantizer::<2>().unwrap(),
            Bits::Four => q.as_quantizer::<4>().unwrap(),
        }
    }

    /// See the description in [`make_test_repr`].
    const TEST_LIMIT: IdLimit = IdLimit::new(11);

    // For the spherical quantizer tests, we use the canonical grid layout, but center the
    // data around the origin.
    //
    // This allows cosine distances to return reasonable results as the data is distributed
    // around the origin.
    //
    // To keep computation mostly tractable, we only use a 2d grid with 9 points. So the
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
    fn make_test_repr(
        metric: SupportedMetric,
        bits: Bits,
        rerank: Rerank,
        fill: bool,
    ) -> (Spherical, Reference) {
        let grid = Grid::Two;
        let mut data = grid.data(3);
        let offset = 1.5;
        data.as_mut_slice().iter_mut().for_each(|v| *v -= offset);

        let quantizer = train_quantizer(data.as_view(), metric, bits);

        let mut start_points = Matrix::new(0.0, 2, data.ncols());
        start_points.row_mut(0).fill(-2.0);
        start_points.row_mut(1).fill(2.0);

        let config = Spherical::config(
            quantizer,
            Capacity::new(data.nrows()),
            MaxDegree::new(0),
            start_points.clone(),
            rerank,
        )
        .unwrap();

        let spherical = config.build().unwrap();

        assert_eq!(repr::Representation::id_limit(&spherical), TEST_LIMIT);
        assert_eq!(spherical.dim(), grid.dim().into());

        let mut reference = Reference::new(grid.dim().into());

        if fill {
            for (i, row) in data.row_iter().enumerate() {
                let guard = repr::Set::set(&spherical, row).unwrap();
                let id = repr::Guard::id(&guard);

                reference.insert(LogicalId(i), SlotId(id), row);
                repr::Guard::publish(guard);
            }
        }

        // Insert frozen points.
        for (slot, point) in spherical.store.frozen().zip(start_points.row_iter()) {
            reference.insert(LogicalId(slot.into_usize()), SlotId(slot), point);
        }

        (spherical, reference)
    }

    /// Performs the following set of tests:
    ///
    /// * [`ExpandBeam::evaluate`]: For each id in `ids` - attempt to evaluate the distance
    ///   through [`ExpandBeam::evaluate`]. If the id is present in `distances`, assert that
    ///   the value in `distances` agrees with the result of the `EpandBeam method.
    ///
    ///   Otherwise, assert that `ExpandBeam` returns `None`.
    ///
    /// * [`ExpandBeam::expand_beam`]: Provid all `ids` to `expand_beam`. Verify that ids not
    ///   present in `distances` get removed and all remaining ids are present and have a
    ///   distance value equal to the corresponding entry in `distances`.
    fn test_expand_beam(
        accessor: &dyn repr::ExpandBeam,
        distances: HashMap<SlotId, f32>,
        ids: &[SlotId],
        ctx: &dyn std::fmt::Display,
    ) {
        assert_eq!(accessor.id_limit(), TEST_LIMIT, "{ctx}");

        for slot_id in ids {
            if let Some(distance) = distances.get(slot_id) {
                assert_eq!(
                    accessor.evaluate(slot_id.value()).unwrap(),
                    Some(*distance),
                    "failed on slot id {} -- {}",
                    slot_id,
                    ctx,
                );
            } else {
                assert!(
                    accessor.evaluate(slot_id.value()).unwrap().is_none(),
                    "failed on slot id {} -- {}",
                    slot_id,
                    ctx
                );
            }
        }

        // Test via `expand_beam`.
        let list: Vec<u32> = ids.iter().map(|slot_id| slot_id.value()).collect();
        let mut buffer = vec![Neighbor::default(); list.len()];
        let len = repr::safe_expand_beam(accessor, &list, &mut buffer).unwrap();

        let expected: Vec<Neighbor<u32>> = ids
            .iter()
            .filter_map(|slot_id| {
                distances
                    .get(slot_id)
                    .map(|distance| Neighbor::new(slot_id.value(), *distance))
            })
            .collect();

        assert_eq!(
            expected.len(),
            len,
            "`expand_beam` returned the incorrect number of items -- {}",
            ctx,
        );

        for (i, (got, expected)) in std::iter::zip(buffer.iter(), expected.iter()).enumerate() {
            assert_eq!(
                got.id(),
                expected.id(),
                "failed on entry {} of {} -- {}",
                i,
                len,
                ctx
            );
            assert_eq!(
                got.distance(),
                expected.distance(),
                "failed on entry {} of {} -- {}",
                i,
                len,
                ctx,
            );
        }
    }

    /// Test that the [`repr::Prune`] computes distances according to the ground truth in
    /// `distances`.
    ///
    /// This assumes that `distances` contains all valid (i.e., between undeleted) entries
    /// in `ids` - including self distances.
    ///
    /// For example, if `ids` contains `[0, 1, 2, 3(deleted)]`, then `distances` should contain
    /// the keys:
    ///
    /// (0, 0), (0, 1), (0, 2)
    /// (1, 0), (1, 1), (1, 2)
    /// (2, 0), (2, 1), (2, 2)
    fn test_prune(
        accessor: &mut dyn repr::Prune,
        distances: HashMap<(SlotId, SlotId), f32>,
        ids: &[SlotId],
        ctx: &dyn std::fmt::Display,
    ) {
        let num_present_ids = ids
            .iter()
            .filter(|&&slot_id| distances.contains_key(&(slot_id, slot_id)))
            .count();

        let mut items: HashMap<u32, Option<repr::PruneKey>> =
            ids.iter().map(|slot_id| (slot_id.value(), None)).collect();

        let count = accessor.prepare(items.iter_mut()).unwrap();
        assert_eq!(count, num_present_ids, "{ctx}");

        let mut visited = 0;
        for slot_id0 in ids.iter() {
            if let Some(key0) = items[&slot_id0.value()] {
                for slot_id1 in ids.iter() {
                    if let Some(key1) = items[&slot_id1.value()] {
                        let d = accessor.evaluate(key0, key1);
                        let expected = distances[&(*slot_id0, *slot_id1)];
                        assert_eq!(
                            d, expected,
                            "failed for {} x {} -- {}",
                            slot_id0, slot_id1, ctx
                        );

                        visited += 1;
                    }
                }
            }
        }

        assert_eq!(
            visited,
            distances.len(),
            "not all distances were visited -- {}",
            ctx
        );
    }

    /// Test that [`repr::PostProcess`] reranks correctly.
    ///
    /// Pass all `ids` to [`repr::PostProcess::post_process`]. Verify that all ids not
    /// present in `distances` have been removed and the remaining ids are present, sorted,
    /// and have distance values matching those in `distances`.
    fn test_rerank(
        post_process: &mut dyn repr::PostProcess,
        distances: HashMap<SlotId, f32>,
        ids: &[SlotId],
        ctx: &dyn std::fmt::Display,
    ) {
        let mut buffer: Vec<_> = ids
            .iter()
            .map(|slot_id| Neighbor::new(slot_id.value(), 0.0))
            .collect();

        post_process.post_process(&mut buffer).unwrap();
        let mut previous = f32::NEG_INFINITY;
        assert_eq!(buffer.len(), distances.len(), "{ctx}");
        for neighbor in buffer.iter() {
            let current = *neighbor.distance();
            assert_eq!(current, distances[&SlotId(*neighbor.id())], "{ctx}",);

            assert!(
                current >= previous,
                "distances is not monotonically increasing, previous = {}, current = {} -- {}",
                previous,
                current,
                ctx,
            );

            previous = current;
        }

        assert!(
            previous > f32::NEG_INFINITY,
            "previous = {} -- {}",
            previous,
            ctx
        );
    }

    /// Here - we don't test the whole `expand_beam` loop. That would be a waste of time and
    /// is already tested by other code.
    ///
    /// Instead we use:
    ///
    /// * `ExpandBeam::evaluate` to verify that the correct type of distance computer is
    ///   created.
    ///
    /// * `Prune` to validate that the correct distance computer is made.
    ///
    /// * If reranking exists, that rereanking works as expected.
    fn test_distances(
        spherical: &Spherical,
        reference: &mut Reference,
        metric: SupportedMetric,
        rerank: Rerank,
        ctx: &dyn std::fmt::Display,
    ) {
        use repr::internal::{RawDistance, RawQueryDistance};

        assert_eq!(
            repr::Representation::id_limit(spherical),
            TEST_LIMIT,
            "{ctx}"
        );

        let query = [10.0, -10.0];

        // Generate a couple of data points in the dataset.
        let i0 = LogicalId(2);
        let s0 = reference.slot_id_for(i0);

        let i1 = LogicalId(5);
        let s1 = reference.slot_id_for(i1);

        let i2 = LogicalId(10);
        let s2 = reference.slot_id_for(i2);

        // Perform Deletes //
        let i3_deleted = LogicalId(8);
        let s3_deleted = reference.slot_id_for(i3_deleted);

        repr::Representation::retire(spherical, s3_deleted.value()).unwrap();
        reference.delete(i3_deleted);

        let i4_deleted = LogicalId(4);
        let s4_deleted = reference.slot_id_for(i4_deleted);

        repr::Representation::retire(spherical, s4_deleted.value()).unwrap();
        reference.delete(i4_deleted);

        // Extract Values.
        let v0 = &reference[i0];
        let v1 = &reference[i1];
        let v2 = &reference[i2];

        // Compress the dataset vectors.
        let quantizer = spherical.quantizer();
        let bytes = quantizer.bytes();
        let mut b0 = vec![0u8; bytes];
        let mut b1 = vec![0u8; bytes];
        let mut b2 = vec![0u8; bytes];

        fn _compress(quantizer: &dyn iface::Quantizer, v: &[f32], b: &mut [u8]) {
            let alloc = ScopedAllocator::global();
            quantizer
                .compress(v, iface::OpaqueMut::new(b), alloc)
                .unwrap();

            // Make sure that compression actually did something.
            assert!(!b.iter().all(|i| *i == 0));
        }

        _compress(quantizer, v0, &mut b0);
        _compress(quantizer, v1, &mut b1);
        _compress(quantizer, v2, &mut b2);

        // Insert
        {
            let computer = quantizer
                .fused_query_computer(
                    &query,
                    iface::QueryLayout::SameAsData,
                    false,
                    GlobalAllocator,
                    ScopedAllocator::global(),
                )
                .unwrap();

            let distances = HashMap::from_iter([
                (s0, RawQueryDistance::eval(&computer, &b0).unwrap()),
                (s1, RawQueryDistance::eval(&computer, &b1).unwrap()),
                (s2, RawQueryDistance::eval(&computer, &b2).unwrap()),
            ]);

            let counters = Counters::new();
            let mut sa = repr::Insert::insert_search_accessor(
                spherical,
                query.as_slice(),
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
                distances,
                &[s0, s1, s3_deleted, s2, s4_deleted],
                ctx,
            );
        }

        // Search
        {
            let computer = quantizer
                .fused_query_computer(
                    &query,
                    iface::QueryLayout::FullPrecision,
                    true,
                    GlobalAllocator,
                    ScopedAllocator::global(),
                )
                .unwrap();

            let distances = HashMap::from_iter([
                (s0, RawQueryDistance::eval(&computer, &b0).unwrap()),
                (s1, RawQueryDistance::eval(&computer, &b1).unwrap()),
                (s2, RawQueryDistance::eval(&computer, &b2).unwrap()),
            ]);

            let counters = Counters::new();
            let mut sa =
                repr::Search::search_accessor(spherical, query.as_slice(), &(), counters.local())
                    .unwrap();

            test_expand_beam(
                sa.get_expand_beam(),
                distances,
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

                let f =
                    <f32 as DistanceProvider<f32>>::distance_comparer(convert_metric(metric), None);

                let distances = HashMap::from_iter([
                    (s0, f.call(&query, v0)),
                    (s1, f.call(&query, v1)),
                    (s2, f.call(&query, v2)),
                ]);

                test_rerank(
                    post_process,
                    distances,
                    &[s0, s1, s3_deleted, s2, s4_deleted],
                    ctx,
                );
            }
        }

        // Prune
        {
            let computer = quantizer.distance_computer_ref();
            let b00 = RawDistance::eval(&computer, &b0, &b0).unwrap();
            let b01 = RawDistance::eval(&computer, &b0, &b1).unwrap();
            let b02 = RawDistance::eval(&computer, &b0, &b2).unwrap();

            let b10 = b01;
            let b11 = RawDistance::eval(&computer, &b1, &b1).unwrap();
            let b12 = RawDistance::eval(&computer, &b1, &b2).unwrap();

            let b20 = b02;
            let b21 = b12;
            let b22 = RawDistance::eval(&computer, &b2, &b2).unwrap();

            let distances = HashMap::from_iter([
                ((s0, s0), b00),
                ((s0, s1), b01),
                ((s0, s2), b02),
                ((s1, s0), b10),
                ((s1, s1), b11),
                ((s1, s2), b12),
                ((s2, s0), b20),
                ((s2, s1), b21),
                ((s2, s2), b22),
            ]);

            let counters = Counters::new();
            let mut pa = repr::Insert::prune_accessor(spherical, counters.local()).unwrap();

            test_prune(
                pa.get_prune(),
                distances,
                &[s0, s1, s3_deleted, s2, s4_deleted],
                ctx,
            );
        }
    }

    /// The happy-patch entry point.
    ///
    /// Note that this method does a grid of the supported parameters. Adding new values
    /// to these parameters has a multiplicative effect on runtime.
    ///
    /// This mainly matters for Miri tests. Currently, the Miri test for this function takes
    /// about 30 seconds. If it starts to take much longer, this test should be split into
    /// multiple entry points for parallelism.
    #[test]
    fn test_spherical() {
        let metrics = [
            SupportedMetric::SquaredL2,
            SupportedMetric::InnerProduct,
            SupportedMetric::Cosine,
        ];

        let bits = [Bits::One, Bits::Two, Bits::Four];
        let rerank = [Rerank::None, Rerank::F16];

        for metric in metrics {
            for bits in bits {
                for rerank in rerank {
                    let (spherical, mut reference) = make_test_repr(metric, bits, rerank, true);

                    test_distances(
                        &spherical,
                        &mut reference,
                        metric,
                        rerank,
                        &format_args!(
                            "metric = {:?}, bits = {:?}, rerank = {:?}",
                            metric, bits, rerank
                        ),
                    );
                }
            }
        }
    }

    //-------------//
    // Error Paths //
    //-------------//

    #[test]
    fn test_config_dim_mismatch() {
        let data = Matrix::new(1.0f32, 2, 5);
        let quantizer = train_quantizer(data.as_view(), SupportedMetric::SquaredL2, Bits::One);

        let start_points = Matrix::new(0.0f32, 1, 6); // Wrong number of columns
        let err = Spherical::config(
            quantizer,
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
        let data = Matrix::new(1.0f32, 2, 5);
        let quantizer = train_quantizer(data.as_view(), SupportedMetric::SquaredL2, Bits::One);

        let start_points = Matrix::new(0.0f32, 0, 5); // Empty
        let err = Spherical::config(
            quantizer,
            Capacity::new(10),
            MaxDegree::new(0),
            start_points,
            Rerank::None,
        )
        .unwrap_err();

        let msg = err.to_string();
        assert_contains!(msg, "at least one start point must be provided",);
    }

    #[test]
    fn test_build_error_uncompressible_query() {
        let data = Matrix::new(1.0f32, 2, 5);
        let quantizer = train_quantizer(data.as_view(), SupportedMetric::SquaredL2, Bits::One);

        let start_points = Matrix::new(f32::INFINITY, 1, 5); // Wrong number of columns
        let config = Spherical::config(
            quantizer,
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
            "query compression",
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
        let (spherical, _) =
            make_test_repr(SupportedMetric::SquaredL2, Bits::One, Rerank::None, true);

        let err = repr::Set::set(&spherical, &[1.0, 2.0]).unwrap_err();
        assert_contains!(err.to_string(), "could not allocate a new slot",);
    }

    #[test]
    fn test_set_dim_mismatch() {
        let (spherical, _) =
            make_test_repr(SupportedMetric::SquaredL2, Bits::One, Rerank::None, false);

        let err = repr::Set::set(&spherical, &[1.0, 2.0, 3.0]).unwrap_err();
        assert_contains!(
            err.to_string(),
            "vector dim 3 does not match quantizer dim 2",
        );
    }

    #[test]
    fn test_insert_slot_recovery() {
        let (spherical, _) =
            make_test_repr(SupportedMetric::SquaredL2, Bits::One, Rerank::F16, true);

        // Free up one slot.
        repr::Representation::retire(&spherical, 0).unwrap();

        // Insert something that is incompressible.
        let err = repr::Set::set(&spherical, &[f32::INFINITY, f32::INFINITY]).unwrap_err();
        assert_contains!(err.to_string(), "query compression");

        // If we insert again, this should succeed.
        let guard = repr::Set::set(&spherical, &[1.0, 2.0]).unwrap();
        assert_eq!(repr::Guard::id(&guard), 0);
    }
}
