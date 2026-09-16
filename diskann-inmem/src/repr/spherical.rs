/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann::{ANNError, ANNResult, utils::IntoUsize};
use diskann_quantization::{
    alloc::{Allocator, GlobalAllocator, Poly, ScopedAllocator},
    spherical::{SupportedMetric, iface},
};
use diskann_utils::views::Matrix;
use diskann_vector::distance::{Distance, DistanceProvider};
use half::f16;

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

///////////
// Stuff //
///////////

/// Choose how data is going to be reranked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rerank {
    None,
    Float16,
}

impl Rerank {
    fn bytes(&self, dim: usize) -> Option<Bytes> {
        match self {
            Self::None => Some(Bytes::new(0)),
            Self::Float16 => dim.checked_mul(2).map(Bytes::new),
        }
    }

    fn config(self, dim: usize) -> Option<simple::Config> {
        match self {
            Self::None => None,
            Self::Float16 => Some(Simple::config(self.bytes(dim).unwrap())),
        }
    }
}

pub struct Config {
    quantizer: Poly<dyn iface::Quantizer>,
    start_points: Matrix<f32>,
    layout: store::Layout,
    store: store::Config,
    lookahead: Option<NonZeroUsize>,
    rerank: Rerank,
}

const DEFAULT_LOOKAHEAD: NonZeroUsize = NonZeroUsize::new(16).unwrap();

impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "todo")
    }
}

impl Config {
    pub fn new(
        quantizer: Poly<dyn iface::Quantizer>,
        capacity: Capacity,
        max_degree: MaxDegree,
        start_points: Matrix<f32>,
        rerank: Rerank,
    ) -> Self {
        assert_eq!(start_points.ncols(), quantizer.full_dim());

        let num_start_points: u32 = start_points.nrows().try_into().unwrap();

        Self {
            quantizer,
            start_points,
            layout: store::Layout::new(capacity, max_degree, num_start_points),
            store: store::Config::default(),
            lookahead: Some(DEFAULT_LOOKAHEAD),
            rerank,
        }
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

    // /// Return the vector dimension of this configuration and the resulting [`Full`].
    // pub fn dim(&self) -> usize {
    //     self.start_points.ncols()
    // }

    pub fn build(self) -> ANNResult<Spherical> {
        Ok(Spherical::new(self))
    }
}

impl repr::RepresentationConfig for Config {
    type Representation = Spherical;

    fn build(self) -> ANNResult<Spherical> {
        <Config>::build(self)
    }
}

#[derive(Debug)]
enum Reranker {
    None,
    Float16(Distance<f32, f16>),
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
    fn new(rerank: Rerank, metric: SupportedMetric, dim: usize) -> Self {
        match rerank {
            Rerank::None => Self::None,
            Rerank::Float16 => {
                Self::Float16(f32::distance_comparer(convert_metric(metric), Some(dim)))
            }
        }
    }

    fn post_process<'a>(
        &'a self,
        query: &'a [f32],
        guard: &epoch::Guard<'a>,
        simple: &'a store::simple::Simple,
    ) -> Option<Box<dyn repr::PostProcess + '_>> {
        match self {
            Self::None => None,
            Self::Float16(distance) => {
                let distance = repr::full::QueryDistance::new(Calf::Borrowed(query), *distance);
                let reader = simple.reader(guard.share());
                let post_process = repr::internal::simple::Reranker::new(reader, distance);
                Some(Box::new(post_process))
            }
        }
    }

    fn store(&self, v: &[f32], buf: &mut [u8]) {
        use diskann_vector::conversion::CastFromSlice;
        bytemuck::cast_slice_mut::<u8, f16>(buf).cast_from_slice(v);
    }
}

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
    fn new(config: Config) -> Self {
        let Config {
            quantizer,
            start_points,
            layout,
            store,
            lookahead,
            rerank,
        } = config;

        let full_dim = quantizer.full_dim();

        let slots = cons::Config::new(
            Intrusive::config(Bytes::new(quantizer.bytes())),
            rerank.config(full_dim),
        );

        let store = Store::new(layout, store, slots).unwrap();

        let reranker = Reranker::new(rerank, quantizer.metric(), full_dim);

        let this = Self {
            store,
            quantizer,
            full_dim,
            lookahead,
            reranker,
        };

        // Initialize start points.
        for (i, row) in std::iter::zip(this.store.frozen(), start_points.row_iter()) {
            #[expect(
                clippy::expect_used,
                reason = "failing this is an internal, unrecoverable bug"
            )]
            let mut slot = this
                .store
                .slot(i)
                .expect("internal store should leave frozen-points available for writing");

            this.set(row, slot.data());
            slot.freeze();
        }

        this
    }

    pub fn config(
        quantizer: Poly<dyn iface::Quantizer>,
        capacity: Capacity,
        max_degree: MaxDegree,
        start_points: Matrix<f32>,
        rerank: Rerank,
    ) -> Config {
        Config::new(quantizer, capacity, max_degree, start_points, rerank)
    }

    fn set(
        &self,
        v: &[f32],
        slot: &mut cons::Exclusive<intrusive::Exclusive<'_>, Option<simple::Exclusive<'_>>>,
    ) {
        self.quantizer
            .compress(
                v,
                iface::OpaqueMut::new(slot.first().as_mut_slice()),
                ScopedAllocator::global(),
            )
            .map_err(ANNError::new)
            .unwrap();

        if let Some(tail) = slot.second() {
            self.reranker.store(v, tail.as_mut_slice())
        }
    }

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
                    cons.second().slots().and_then(|simple| {
                        self.reranker
                            .post_process(query, expand_beam.guard(), simple)
                    })
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
    layout: iface::QueryLayout,
    allow_rescale: bool,
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
        if v.len() != self.full_dim {
            panic!("nope");
        }

        let mut slot = self
            .store
            .acquire()
            .ok_or_else(|| ANNError::message("could not allocate a new slot"))?;

        self.set(v, slot.data());
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
        let distance = DebugWrapper(self.quantizer.distance_computer_ref());
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
    A: Allocator + Send + Sync,
{
    type Error = ANNError;

    fn eval(&self, x: &[u8]) -> Result<f32, Self::Error> {
        use diskann_vector::PreprocessedDistanceFunction;
        self.evaluate_similarity(iface::Opaque::new(x))
            .map_err(ANNError::new)
    }
}

struct DebugWrapper<'a>(&'a dyn iface::DynDistanceComputer);

impl std::fmt::Debug for DebugWrapper<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "spherical::iface::DistanceComputer {{ layout: {:?} }}",
            self.0.layout()
        )
    }
}

impl repr::internal::RawDistance for DebugWrapper<'_> {
    type Error = ANNError;

    fn eval(&self, x: &[u8], y: &[u8]) -> Result<f32, Self::Error> {
        self.0
            .evaluate(iface::Opaque::new(x), iface::Opaque::new(y))
            .map_err(ANNError::new)
    }
}
