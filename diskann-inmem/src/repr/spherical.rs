/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann::{ANNError, ANNResult, error::ErrorContext, utils::IntoUsize};
use diskann_quantization::{
    alloc::{Allocator, GlobalAllocator, Poly, ScopedAllocator},
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

///////////
// Stuff //
///////////

/// Choose how data is going to be reranked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rerank {
    None,
    F16,
}

impl Rerank {
    #[expect(
        clippy::expect_used,
        reason = "the arithmetic should not overflow for the feasible `dim` values"
    )]
    fn bytes(&self, dim: usize) -> Bytes {
        match self {
            Self::None => Bytes::new(0),
            Self::F16 => Bytes::new(
                dim.checked_mul(2)
                    .expect("f16 is smaller than the f32 in the quantizer"),
            ),
        }
    }

    fn config(self, dim: usize) -> Option<simple::Config> {
        match self {
            Self::None => None,
            Self::F16 => Some(Simple::config(self.bytes(dim))),
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
    ) -> Result<Self, ConfigError> {
        let quantizer_dim = quantizer.full_dim();
        if quantizer_dim != start_points.ncols() {
            return Err(ConfigError::dim_mismatch(
                quantizer_dim,
                start_points.ncols(),
            ));
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

    pub fn build(self) -> ANNResult<Spherical> {
        Spherical::new(self)
    }
}

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
        start_points,
        quantizer
    )]
    DimMismatch {
        quantizer: usize,
        start_points: usize,
    },
    #[error("{} start points exceeds u32::MAX", num_start_points)]
    TooManyStartPoints { num_start_points: usize },
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
    fn new(rerank: Rerank, metric: SupportedMetric, dim: usize) -> Self {
        match rerank {
            Rerank::None => Self::None,
            Rerank::F16 => Self::F16(f32::distance_comparer(convert_metric(metric), Some(dim))),
        }
    }

    fn post_process<'a>(
        &'a self,
        query: &'a [f32],
        guard: &epoch::Guard<'a>,
        simple: &'a store::simple::Simple,
        counters: &LocalCounters<'a>,
    ) -> Option<Box<dyn repr::PostProcess + 'a>> {
        match self {
            Self::None => None,
            Self::F16(distance) => {
                let distance = repr::full::QueryDistance::new(Calf::Borrowed(query), *distance);
                let reader = simple.reader(guard.share());
                let post_process =
                    repr::internal::simple::Reranker::new(reader, distance, counters.fork());
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

        let slots = cons::Config::new(
            Intrusive::config(Bytes::new(quantizer.bytes())),
            rerank.config(full_dim),
        );

        let store = Store::new(layout, store, slots)?;

        let reranker = Reranker::new(rerank, quantizer.metric(), full_dim);

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

    pub fn config(
        quantizer: Poly<dyn iface::Quantizer>,
        capacity: Capacity,
        max_degree: MaxDegree,
        start_points: Matrix<f32>,
        rerank: Rerank,
    ) -> Result<Config, ConfigError> {
        Config::new(quantizer, capacity, max_degree, start_points, rerank)
    }

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

        if let Some(second) = slot.second() {
            self.reranker.store(v, second.as_mut_slice());
        }

        Ok(())
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
                            .post_process(query, expand_beam.guard(), simple, &counters)
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
    A: Allocator + std::fmt::Debug + Send + Sync,
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

    use diskann::graph::test::synthetic::Grid;

    // For the spherical quantizer tests, we use the canonical grid layout, but center the
    // data around the origin.
    //
    // This allows cosine distances to return reasonable results as the data is distributed
    // around the origin.
    //
    // To keep computation mostly tractable, we only use a 2d grid with 16 points. So the
    // coordinates are as follows:
    //
    // 0:  [-1.5, -1.5]
    // 1:  [-1.5, -0.5]
    // 2:  [-1.5, +0.5]
    // 3:  [-1.5, +1.5]
    //
    // 4:  [-0.5, -1.5]
    // 5:  [-0.5, -0.5]
    // 6:  [-0.5, +0.5]
    // 7:  [-0.5, +1.5]
    //
    // 8:  [+0.5, -1.5]
    // 9:  [+0.5, -0.5]
    // 10: [+0.5, +0.5]
    // 11: [+0.5, +1.5]
    //
    // 12: [+1.5, -1.5]
    // 13: [+1.5, -0.5]
    // 14: [+1.5, +0.5]
    // 15: [+1.5, +1.5]
    //
    // We put two start points at [`-2.0, -2.0`] and `[+2.0, +2.0]`.

    #[derive(Debug)]
    enum Bits {
        One,
        Two,
        Four,
    }

    fn test_repr(metric: SupportedMetric, bits: Bits, rerank: Rerank) -> Spherical {
        use diskann_quantization::{algorithms::transforms, spherical};
        use rand::{SeedableRng, rngs::StdRng};

        let grid = Grid::Two;
        let mut data = grid.data(4);
        let offset = 1.5;
        data.as_mut_slice().iter_mut().for_each(|v| *v -= offset);

        let quantizer = {
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
                Bits::One => q.as_quantizer::<1>(),
                Bits::Two => q.as_quantizer::<2>(),
                Bits::Four => q.as_quantizer::<4>(),
            }
        };

        let mut start_points = Matrix::new(0.0, 2, data.ncols());
        start_points.row(0).fill(-2.0);
        start_points.row(1).fill(2.0);

        let config = Spherical::config(

        )
    }
}
