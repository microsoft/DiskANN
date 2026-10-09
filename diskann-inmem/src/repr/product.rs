/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann::{ANNError, ANNResult, error::ErrorContext, utils::IntoUsize};
use diskann_quantization::{distances as quant_distances, product::tables};
use diskann_utils::{
    lazy_format,
    object_pool::{self, ObjectPool},
    views::rowmajor::{self, Matrix, MatrixMut},
};
use thiserror::Error;

use crate::{
    counters::LocalCounters,
    num::{Bytes, Capacity, IdLimit, MaxDegree},
    prefetch, repr,
    store::{
        self, Store,
        cons::{self, Cons},
        intrusive::{self, Intrusive},
        optional::Optional,
        simple::{self, Simple},
    },
};

/// The configuration for a [`Product`] quantized representation.
#[derive(Debug)]
pub struct Config {
    table: tables::BasicTable,
    start_points: rowmajor::Owned<f32>,
    metric: repr::internal::quantization::Metric,
    layout: store::Layout,
    store: store::Config,
    lookahead: Option<NonZeroUsize>,
    rerank: repr::internal::quantization::Rerank,
    // Used to buffer distance lookup tables.
    thread_hint: Option<NonZeroUsize>,
}

const DEFAULT_LOOKAHEAD: NonZeroUsize = NonZeroUsize::new(16).unwrap();

impl Config {
    /// Create a new [`Config`]. Parameters will be used as described below:
    ///
    /// * `table`: The [`tables::BasicTable`] that contains the PQ pivots and chunking strategy.
    ///
    /// * `metric`: The [`Metric`] to use for computing distances.
    ///
    /// * `capacity`: The number points to allocate space for.
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
    /// * `start_points.ncols() != table.dim()`: The dimensionality of the start
    ///   points must agree with the quantizer.
    ///
    /// * `start_points.nrows() == 0`: Currently, empty start points are not supported.
    ///
    /// * The number of start points exceeds `u32::MAX`.
    pub fn new(
        table: tables::BasicTable,
        metric: Metric,
        capacity: Capacity,
        max_degree: MaxDegree,
        start_points: rowmajor::Owned<f32>,
        rerank: Rerank,
    ) -> Result<Self, ConfigError> {
        let dim = table.dim();
        if dim != start_points.ncols() {
            return Err(ConfigError::dim_mismatch(dim, start_points.ncols()));
        }

        if start_points.nrows() == 0 {
            return Err(ConfigError::empty_start_points());
        }

        let num_start_points: u32 = start_points
            .nrows()
            .try_into()
            .map_err(|_| ConfigError::too_many_start_points(start_points.nrows()))?;

        Ok(Self {
            table,
            start_points,
            metric: metric.as_internal_metric(),
            layout: store::Layout::new(capacity, max_degree, num_start_points),
            store: store::Config::default(),
            lookahead: Some(DEFAULT_LOOKAHEAD),
            rerank: rerank.as_internal_rerank(),
            thread_hint: None,
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

    /// Provide a hint at the number of threads that will be working concurrently.
    ///
    /// The [`Product`] representation will hold up to `hint` allocated distance tables
    /// internally. Running more than `hint` concurrent searches will allocate.
    ///
    /// If `hint == None`, no buffering will be used and each search or insert will allocate
    /// a new distance table.
    pub fn thread_hint(mut self, hint: Option<NonZeroUsize>) -> Self {
        self.thread_hint = hint;
        self
    }

    /// Build the [`Product`] from `self`.
    pub fn build(self) -> ANNResult<Product> {
        Product::new(self)
    }
}

impl repr::RepresentationConfig for Config {
    type Representation = Product;

    fn build(self) -> ANNResult<Product> {
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

/// Choose how data is going to be reranked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rerank {
    /// No reranking will be performed and no space for higher precision vectors will be
    /// allocated.
    None,

    /// Use 16-bit floating point numbers to store the higher precision representation.
    /// These will be used automatically during search to rerank candidates.
    F16,
}

impl Rerank {
    fn as_internal_rerank(&self) -> repr::internal::quantization::Rerank {
        use repr::internal::quantization::Rerank as IRerank;
        match self {
            Self::None => IRerank::None,
            Self::F16 => IRerank::F16,
        }
    }
}

/// Product quantized data representation.
#[derive(Debug)]
pub struct Product {
    store: Store<Cons<Intrusive, Optional<Simple>>>,
    /// The PQ table representation used for compression and creation of query computers.
    transposed: tables::TransposedTable,
    /// The PQ table representation used for pruning.
    padded: tables::PaddedTable,
    metric: repr::internal::quantization::Metric,
    lookahead: Option<NonZeroUsize>,
    reranker: repr::internal::quantization::Reranker,

    /// An object pool for distance lookup tables.
    ///
    /// These involve non-trivial allocations, so it can be beneficial to hold onto them.
    distance_tables: ObjectPool<DistanceTable>,
}

impl Product {
    /// Initialize a [`Config`] for this representation.
    ///
    /// See also: [`Config::new`].
    ///
    /// # Errors
    ///
    /// Returns the errors described by [`Config::new`].
    pub fn config(
        table: tables::BasicTable,
        metric: Metric,
        capacity: Capacity,
        max_degree: MaxDegree,
        start_points: rowmajor::Owned<f32>,
        rerank: Rerank,
    ) -> Result<Config, ConfigError> {
        Config::new(table, metric, capacity, max_degree, start_points, rerank)
    }

    fn new(config: Config) -> ANNResult<Self> {
        let Config {
            table,
            start_points,
            metric,
            layout,
            store,
            lookahead,
            rerank,
            thread_hint,
        } = config;

        let dim = table.dim();
        let (reranker, rerank_config) =
            repr::internal::quantization::Reranker::new_with_config(rerank, metric, dim);

        let transposed = tables::TransposedTable::from_parts(
            table.view_pivots(),
            table.view_offsets().to_owned(),
        )
        .map_err(ANNError::new)
        .context("this is a broken internal invariant - please report")?;

        let padded = tables::PaddedTable::from_basic(table.as_view());

        let slots = cons::Config::new(
            Intrusive::config(Bytes::new(transposed.nchunks())),
            rerank_config,
        );

        let store = Store::new(layout, store, slots)?;

        let this = Self {
            store,
            transposed,
            padded,
            metric,
            lookahead,
            reranker,
            distance_tables: ObjectPool::with_capacity(Some(
                thread_hint.map(|v| v.get()).unwrap_or(0),
            )),
        };

        // Initialize start points.
        let num_start_points = start_points.nrows();
        for (i, (slot_index, row)) in
            std::iter::zip(this.store.frozen(), start_points.rows()).enumerate()
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
        self.transposed.dim()
    }

    fn nchunks(&self) -> usize {
        self.transposed.nchunks()
    }

    fn ncenters(&self) -> usize {
        self.transposed.ncenters()
    }

    fn metric(&self) -> repr::internal::quantization::Metric {
        self.metric
    }

    /// * Attempt to compress `v` into the [`cons::Exclusive::first`] position.
    /// * If [`cons::Exclusive::second`] is occupied, use `self.reranker` to store data
    ///   into that slot.
    fn set(
        &self,
        v: &[f32],
        slot: &mut cons::Exclusive<intrusive::Exclusive<'_>, Option<simple::Exclusive<'_>>>,
    ) -> ANNResult<()> {
        use diskann_quantization::CompressInto;

        self.transposed
            .compress_into(v, slot.first().as_mut_slice())
            .map_err(ANNError::new)?;

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
        use diskann_vector::{Norm, norm::FastL2Norm};

        // NOTE: `ProcessInto` panics on query dim mismatch.
        //
        // To avoid hitting that - validate the query dim early.
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

        let AccessorArgs { rerank_if_enabled } = args;

        // Create the query computer.
        //
        // We do this first because this is one of the most likely things to fail since it
        // operates on largely untrusted data. If it does fail, we save the work of acquiring
        // epoch guards etc.
        let query_computer = {
            let distance_table_args = DistanceTableArgs {
                nchunks: self.nchunks(),
                ncenters: self.ncenters(),
                metric: self.metric(),
            };

            let mut distance_table = self.distance_tables.get_ref(distance_table_args);

            match &mut *distance_table {
                DistanceTable::SquaredL2(table) => {
                    self.transposed
                        .process_into::<quant_distances::SquaredL2, _>(query, table.as_view_mut());
                }
                DistanceTable::InnerProduct(table) => {
                    self.transposed
                        .process_into::<quant_distances::InnerProduct, _>(
                            query,
                            table.as_view_mut(),
                        );
                }
                DistanceTable::Cosine { table, query_norm } => {
                    self.transposed
                        .process_into::<quant_distances::Cosine, _>(query, table.as_view_mut());
                    *query_norm = (FastL2Norm).evaluate(query);
                }
            }

            QueryComputer(distance_table)
        };

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
}

#[derive(Debug)]
struct AccessorArgs {
    rerank_if_enabled: bool,
}

repr::internal::macros::representation!(Product);

repr::internal::macros::set_guard!(
    /// A [`repr::Guard`] for [`Product`].
    for<'a> cons::Exclusive<intrusive::Exclusive<'a>, Option<simple::Exclusive<'a>>>
);

impl repr::Set<&[f32]> for Product {
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

impl repr::Search for Product {
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
                rerank_if_enabled: true,
            },
        )
    }
}

impl repr::Insert for Product {
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
                rerank_if_enabled: false,
            },
        )
    }

    fn prune_accessor<'a>(
        &'a self,
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::PruneAccessor<'a>> {
        let distance = Distance::new(&self.padded, self.metric());
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
// Query Distance //
//----------------//

/// A populated distance table for query distances.
#[derive(Debug)]
enum DistanceTable {
    SquaredL2(rowmajor::Owned<f32>),
    InnerProduct(rowmajor::Owned<f32>),
    Cosine {
        table: rowmajor::Owned<tables::lookup::DotAndNorm>,
        query_norm: f32,
    },
}

#[derive(Debug, Clone, Copy)]
struct DistanceTableArgs {
    nchunks: usize,
    ncenters: usize,
    metric: repr::internal::quantization::Metric,
}

impl object_pool::AsPooled<DistanceTableArgs> for DistanceTable {
    fn create(args: DistanceTableArgs) -> Self {
        use repr::internal::quantization::Metric as IMetric;

        let DistanceTableArgs {
            nchunks,
            ncenters,
            metric,
        } = args;

        match metric {
            IMetric::SquaredL2 => {
                DistanceTable::SquaredL2(rowmajor::Owned::from_element(nchunks, ncenters, 0.0))
            }

            IMetric::InnerProduct => {
                DistanceTable::InnerProduct(rowmajor::Owned::from_element(nchunks, ncenters, 0.0))
            }

            IMetric::Cosine => DistanceTable::Cosine {
                table: rowmajor::Owned::from_element(
                    nchunks,
                    ncenters,
                    tables::lookup::DotAndNorm::default(),
                ),
                query_norm: 0.0,
            },
        }
    }

    fn modify(&mut self, args: DistanceTableArgs) {
        use repr::internal::quantization::Metric as IMetric;

        let DistanceTableArgs {
            nchunks,
            ncenters,
            metric,
        } = args;

        let sizes_agree =
            |nrows: usize, ncols: usize| -> bool { nrows == nchunks && ncols == ncenters };

        let good = match (&self, metric) {
            (Self::SquaredL2(table), IMetric::SquaredL2) => {
                sizes_agree(table.nrows(), table.ncols())
            }
            (Self::InnerProduct(table), IMetric::InnerProduct) => {
                sizes_agree(table.nrows(), table.ncols())
            }
            (Self::Cosine { table, .. }, IMetric::Cosine) => {
                sizes_agree(table.nrows(), table.ncols())
            }
            _ => false,
        };

        if !good {
            *self = Self::create(args);
        }
    }
}

#[derive(Debug)]
struct QueryComputer<'a>(object_pool::PooledRef<'a, DistanceTable>);

impl repr::internal::RawQueryDistance for QueryComputer<'_> {
    type Error = ANNError;

    fn eval(&self, x: &[u8]) -> Result<f32, Self::Error> {
        let distance = match &*self.0 {
            DistanceTable::SquaredL2(table) | DistanceTable::InnerProduct(table) => {
                tables::lookup::lookup_single(tables::lookup::Sum, table.as_view(), x)
                    .map_err(ANNError::new)?
            }
            DistanceTable::Cosine { table, query_norm } => {
                let sum = tables::lookup::lookup_single(tables::lookup::Sum, table.as_view(), x)
                    .map_err(ANNError::new)?;

                sum.finish_cosine(*query_norm).into_inner()
            }
        };

        Ok(distance)
    }
}

//----------//
// Distance //
//----------//

#[derive(Debug)]
struct Distance<'a> {
    table: &'a tables::PaddedTable,
    vtable: tables::padded::VTable,
}

impl<'a> Distance<'a> {
    fn new(table: &'a tables::PaddedTable, metric: repr::internal::quantization::Metric) -> Self {
        Self {
            table,
            vtable: table.vtable(metric.into()),
        }
    }
}

impl repr::internal::RawDistance for Distance<'_> {
    type Error = ANNError;

    fn eval(&self, x: &[u8], y: &[u8]) -> Result<f32, Self::Error> {
        self.vtable
            .self_distance(self.table, x, y)
            .map_err(ANNError::new)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann::graph::test::synthetic::Grid;
    use diskann_utils::{assert_contains, views::rowmajor::MatrixMut};
    use diskann_vector::distance::DistanceProvider;
    use hashbrown::HashMap;

    use crate::{
        counters::Counters,
        num::{LogicalId, SlotId},
        repr::test::{Reference, test_expand_beam, test_prune},
    };

    fn train_quantizer(
        data: rowmajor::Ref<'_, f32>,
        chunks: usize,
        centers: usize,
    ) -> tables::BasicTable {
        use diskann_quantization::{
            Parallelism,
            cancel::DontCancel,
            product::{self, train::TrainQuantizer},
            random,
            views::ChunkOffsets,
        };

        let trainer = product::train::LightPQTrainingParameters::new(centers, 2);
        trainer
            .train(
                data,
                ChunkOffsets::partition(
                    NonZeroUsize::new(data.ncols()).unwrap(),
                    NonZeroUsize::new(chunks).unwrap(),
                )
                .unwrap()
                .as_view(),
                Parallelism::Sequential,
                &random::StdRngBuilder::new(0),
                &DontCancel,
            )
            .unwrap()
    }

    /// See the description in [`make_test_repr`].
    const TEST_LIMIT: IdLimit = IdLimit::new(11);

    // Use the canonical grid layout, but center the data around the origin.
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
    fn make_test_repr(metric: Metric, rerank: Rerank, fill: bool) -> (Product, Reference) {
        let grid = Grid::Two;
        let mut data = grid.data(3);
        let offset = 1.5;
        data.as_mut_slice().iter_mut().for_each(|v| *v -= offset);

        let mut start_points = rowmajor::Owned::from_element(2, data.ncols(), 0.0);
        start_points.row_mut(0).fill(-2.0);
        start_points.row_mut(1).fill(2.0);

        // Train with 2 chunks and 5 centers. This should be sufficient to exactly represent
        // all the points in the grid.
        //
        // To ensure the results are exact, we append the start points to the training data.
        let table = {
            let train_data =
                rowmajor::Owned::from_fn(data.nrows() + start_points.nrows(), data.ncols(), |rc| {
                    if let Some(start_point_row) = rc.row.checked_sub(data.nrows()) {
                        *start_points.element(start_point_row, rc.col)
                    } else {
                        *data.element(rc.row, rc.col)
                    }
                });

            train_quantizer(train_data.as_view(), 2, 5)
        };

        let config = Product::config(
            table,
            metric,
            Capacity::new(data.nrows()),
            MaxDegree::new(0),
            start_points.clone(),
            rerank,
        )
        .unwrap()
        .thread_hint(NonZeroUsize::new(1));

        let product = config.build().unwrap();

        assert_eq!(repr::Representation::id_limit(&product), TEST_LIMIT);
        assert_eq!(product.dim(), grid.dim().into());

        let mut reference = Reference::new(grid.dim().into());

        if fill {
            for (i, row) in data.rows().enumerate() {
                let guard = repr::Set::set(&product, row).unwrap();
                let id = repr::Guard::id(&guard);

                reference.insert(LogicalId(i), SlotId(id), row);
                repr::Guard::publish(guard);
            }
        }

        // Insert frozen points.
        for (slot, point) in product.store.frozen().zip(start_points.rows()) {
            reference.insert(LogicalId(slot.into_usize()), SlotId(slot), point);
        }

        (product, reference)
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
    /// * If reranking exists, that reranking works as expected.
    fn test_distances(
        product: &Product,
        reference: &mut Reference,
        metric: Metric,
        rerank: Rerank,
        ctx: &dyn std::fmt::Display,
    ) {
        assert_eq!(repr::Representation::id_limit(product), TEST_LIMIT, "{ctx}");

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

        repr::Representation::retire(product, s3_deleted.value()).unwrap();
        reference.delete(i3_deleted);

        let i4_deleted = LogicalId(4);
        let s4_deleted = reference.slot_id_for(i4_deleted);

        repr::Representation::retire(product, s4_deleted.value()).unwrap();
        reference.delete(i4_deleted);

        // Extract Values.
        let v0 = &reference[i0];
        let v1 = &reference[i1];
        let v2 = &reference[i2];

        // For computing distances, we rely on the PQ representation being exact.
        let f = <f32 as DistanceProvider<f32>>::distance_comparer(
            metric.as_internal_metric().as_vector_metric(),
            None,
        );

        let distances = HashMap::from_iter([
            (s0, f.call(&query, v0)),
            (s1, f.call(&query, v1)),
            (s2, f.call(&query, v2)),
        ]);

        // Insert
        {
            let counters = Counters::new();
            let mut sa = repr::Insert::insert_search_accessor(
                product,
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
                TEST_LIMIT,
                distances.clone(),
                &[s0, s1, s3_deleted, s2, s4_deleted],
                ctx,
            );
        }

        // Search
        {
            let counters = Counters::new();
            let mut sa =
                repr::Search::search_accessor(product, query.as_slice(), &(), counters.local())
                    .unwrap();

            test_expand_beam(
                sa.get_expand_beam(),
                TEST_LIMIT,
                distances.clone(),
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
                    distances,
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
            let mut pa = repr::Insert::prune_accessor(product, counters.local()).unwrap();

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
    /// about 30 seconds. If it starts to take much longer, this test should be split into
    /// multiple entry points for parallelism.
    #[test]
    fn test_product() {
        let metrics = [Metric::SquaredL2, Metric::InnerProduct, Metric::Cosine];

        let rerank = [Rerank::None, Rerank::F16];

        for metric in metrics {
            for rerank in rerank {
                let (spherical, mut reference) = make_test_repr(metric, rerank, true);

                test_distances(
                    &spherical,
                    &mut reference,
                    metric,
                    rerank,
                    &format_args!("metric = {:?}, rerank = {:?}", metric, rerank),
                );
            }
        }
    }

    //-------------//
    // Error Paths //
    //-------------//

    #[test]
    fn test_config_dim_mismatch() {
        let data = rowmajor::Owned::from_element(2, 5, 1.0f32);
        let quantizer = train_quantizer(data.as_view(), 2, 2);

        let start_points = rowmajor::Owned::from_element(1, 6, 0.0f32); // Wrong number of columns
        let err = Product::config(
            quantizer,
            Metric::SquaredL2,
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
        let data = rowmajor::Owned::from_element(2, 5, 1.0f32);
        let quantizer = train_quantizer(data.as_view(), 2, 2);

        let start_points = rowmajor::Owned::from_element(0, 5, 0.0f32); // Empty
        let err = Product::config(
            quantizer,
            Metric::SquaredL2,
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
        let data = rowmajor::Owned::from_element(2, 5, 1.0f32);
        let quantizer = train_quantizer(data.as_view(), 2, 2);

        let start_points = rowmajor::Owned::from_element(1, 5, f32::INFINITY); // Wrong number of columns
        let config = Product::config(
            quantizer,
            Metric::SquaredL2,
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
        let (product, _) = make_test_repr(Metric::SquaredL2, Rerank::None, true);

        let err = repr::Set::set(&product, &[1.0, 2.0]).unwrap_err();
        assert_contains!(err.to_string(), "could not allocate a new slot",);
    }

    #[test]
    fn test_set_dim_mismatch() {
        let (product, _) = make_test_repr(Metric::SquaredL2, Rerank::None, false);

        let err = repr::Set::set(&product, &[1.0, 2.0, 3.0]).unwrap_err();
        assert_contains!(
            err.to_string(),
            "vector dim 3 does not match quantizer dim 2",
        );
    }

    #[test]
    fn test_search_dim_mismatch() {
        let (product, _) = make_test_repr(Metric::SquaredL2, Rerank::None, false);

        let counters = Counters::new();
        let err =
            repr::Search::search_accessor(&product, &[1.0], &(), counters.local()).unwrap_err();

        assert_contains!(
            err.to_string(),
            "query dim 1 does not match quantizer dim 2"
        );
    }

    #[test]
    fn test_insert_slot_recovery() {
        let (product, _) = make_test_repr(Metric::SquaredL2, Rerank::F16, true);

        // Free up one slot.
        repr::Representation::retire(&product, 0).unwrap();

        // Insert something that is incompressible.
        let err = repr::Set::set(&product, &[f32::INFINITY, f32::INFINITY]).unwrap_err();
        assert_contains!(err.to_string(), "infinity");

        // If we insert again, this should succeed.
        let guard = repr::Set::set(&product, &[1.0, 2.0]).unwrap();
        assert_eq!(repr::Guard::id(&guard), 0);
    }

    //-------------//
    // Object Pool //
    //-------------//

    #[test]
    fn test_distance_table_as_pooled() {
        use object_pool::AsPooled;
        use repr::internal::quantization::Metric as IMetric;

        let args = DistanceTableArgs {
            nchunks: 10,
            ncenters: 4,
            metric: IMetric::SquaredL2,
        };
        let mut table = DistanceTable::create(args);

        let ptr = if let DistanceTable::SquaredL2(ref table) = table {
            assert_eq!(table.nrows(), 10);
            assert_eq!(table.ncols(), 4);
            table.as_ptr()
        } else {
            panic!("Unexpected table: {:?}", table);
        };

        // Modify should leave the allocation untouched if it matches.
        table.modify(args);

        if let DistanceTable::SquaredL2(ref table) = table {
            assert_eq!(table.nrows(), 10);
            assert_eq!(table.ncols(), 4);
            assert_eq!(table.as_ptr(), ptr);
        } else {
            panic!("Unexpected table: {:?}", table);
        };

        // Modify works when changing sizes.
        let args = DistanceTableArgs {
            nchunks: 9,
            ncenters: 5,
            metric: IMetric::SquaredL2,
        };
        table.modify(args);
        if let DistanceTable::SquaredL2(ref table) = table {
            assert_eq!(table.nrows(), 9);
            assert_eq!(table.ncols(), 5);
        } else {
            panic!("Unexpected table: {:?}", table);
        };

        // Modify changes table type.
        let args = DistanceTableArgs {
            nchunks: 10,
            ncenters: 6,
            metric: IMetric::Cosine,
        };
        table.modify(args);
        if let DistanceTable::Cosine { ref table, .. } = table {
            assert_eq!(table.nrows(), 10);
            assert_eq!(table.ncols(), 6);
        } else {
            panic!("Unexpected table: {:?}", table);
        };

        let args = DistanceTableArgs {
            nchunks: 10,
            ncenters: 6,
            metric: IMetric::InnerProduct,
        };
        table.modify(args);
        if let DistanceTable::InnerProduct(ref table) = table {
            assert_eq!(table.nrows(), 10);
            assert_eq!(table.ncols(), 6);
        } else {
            panic!("Unexpected table: {:?}", table);
        };
    }
}
