/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann::{ANNError, ANNResult, utils::IntoUsize};
use diskann_quantization::{
    alloc::{Allocator, GlobalAllocator, Poly, ScopedAllocator},
    spherical::iface,
};
use diskann_utils::views::Matrix;

use crate::{
    counters::LocalCounters,
    num::{Bytes, Capacity, IdLimit, MaxDegree},
    prefetch, repr,
    store::{
        self, Store,
        intrusive::{self, Intrusive},
    },
};

///////////
// Stuff //
///////////

pub struct Config {
    quantizer: Poly<dyn iface::Quantizer>,
    start_points: Matrix<f32>,
    layout: store::Layout,
    store: store::Config,
    lookahead: Option<NonZeroUsize>,
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
    ) -> Self {
        assert_eq!(start_points.ncols(), quantizer.full_dim());

        let num_start_points: u32 = start_points.nrows().try_into().unwrap();

        Self {
            quantizer,
            start_points,
            layout: store::Layout::new(capacity, max_degree, num_start_points),
            store: store::Config::default(),
            lookahead: Some(DEFAULT_LOOKAHEAD),
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

pub struct Spherical {
    store: Store<Intrusive>,
    quantizer: Poly<dyn iface::Quantizer>,
    // These values come directly from `quantizer`, but are hoisted out to avoid a
    // trait-object function call when accessing.
    full_dim: usize,
    lookahead: Option<NonZeroUsize>,
}

impl Spherical {
    fn new(config: Config) -> Self {
        let Config {
            quantizer,
            start_points,
            layout,
            store,
            lookahead,
        } = config;

        let intrusive = Intrusive::config(Bytes::new(quantizer.bytes()));
        let store = Store::new(layout, store, intrusive).unwrap();

        // Initialize start points.
        for (i, row) in std::iter::zip(store.frozen(), start_points.row_iter()) {
            #[expect(
                clippy::expect_used,
                reason = "failing this is an internal, unrecoverable bug"
            )]
            let mut slot = store
                .slot(i)
                .expect("internal store should leave frozen-points available for writing");

            quantizer
                .compress(
                    row,
                    iface::OpaqueMut::new(slot.data().as_mut_slice()),
                    ScopedAllocator::global(),
                )
                .unwrap();

            slot.freeze();
        }

        let full_dim = quantizer.full_dim();

        Self {
            store,
            quantizer,
            full_dim,
            lookahead,
        }
    }

    pub fn config(
        quantizer: Poly<dyn iface::Quantizer>,
        capacity: Capacity,
        max_degree: MaxDegree,
        start_points: Matrix<f32>,
    ) -> Config {
        Config::new(quantizer, capacity, max_degree, start_points)
    }
}

impl repr::Representation for Spherical {
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

        self.quantizer
            .compress(
                v,
                iface::OpaqueMut::new(slot.data().as_mut_slice()),
                ScopedAllocator::global(),
            )
            .map_err(ANNError::new)?;

        Ok(Guard::new(slot))
    }
}

#[derive(Debug)]
pub struct Guard<'a> {
    slot: store::Exclusive<'a, intrusive::Exclusive<'a>>,
}

impl<'a> Guard<'a> {
    fn new(slot: store::Exclusive<'a, intrusive::Exclusive<'a>>) -> Self {
        Self { slot }
    }
}

impl repr::Guard for Guard<'_> {
    fn publish(self) {
        self.slot.publish();
    }
    fn id(&self) -> u32 {
        self.slot.slot()
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
        let query = self
            .quantizer
            .fused_query_computer(
                query,
                iface::QueryLayout::FullPrecision,
                true,
                GlobalAllocator,
                ScopedAllocator::global(),
            )
            .map_err(ANNError::new)?;

        let reader = Intrusive::reader(&self.store)?;

        let expand_beam = repr::internal::intrusive::ExpandBeam::new(
            reader,
            query,
            prefetch::Loop::new(),
            self.lookahead,
        )
        .boxed();

        Ok(crate::provider::SearchAccessor::new(
            self.store.neighbors(),
            expand_beam,
            provider,
            self.store.frozen(),
            counters,
        ))
    }
}

impl repr::Insert for Spherical {
    fn insert_search_accessor<'a>(
        &'a self,
        query: Self::Query<'a>,
        provider: &'a (dyn std::any::Any + Send + Sync),
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::SearchAccessor<'a>> {
        let query = self
            .quantizer
            .fused_query_computer(
                query,
                iface::QueryLayout::SameAsData,
                false,
                GlobalAllocator,
                ScopedAllocator::global(),
            )
            .map_err(ANNError::new)?;

        let reader = Intrusive::reader(&self.store)?;
        let expand_beam = repr::internal::intrusive::ExpandBeam::new(
            reader,
            query,
            prefetch::Loop::new(),
            self.lookahead,
        )
        .boxed();

        Ok(crate::provider::SearchAccessor::new(
            self.store.neighbors(),
            expand_beam,
            provider,
            self.store.frozen(),
            counters,
        ))
    }

    fn prune_accessor<'a>(
        &'a self,
        counters: LocalCounters<'a>,
    ) -> ANNResult<crate::provider::PruneAccessor<'a>> {
        let distance = DebugWrapper(self.quantizer.distance_computer_ref());
        let reader = Intrusive::reader(&self.store)?;
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
