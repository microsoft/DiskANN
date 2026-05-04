/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    future::Future,
    sync::{Arc, Mutex},
};

use diskann::{
    ANNError, ANNResult, default_post_processor,
    error::{RankedError, ToRanked, TransientError},
    graph::{
        glue::{
            CopyIds, DefaultPostProcessor, ExpandBeam, InsertStrategy, MultiInsertStrategy,
            PruneStrategy, SearchExt, SearchStrategy,
        },
        workingset::map,
    },
    neighbor::Neighbor,
    provider::{
        Accessor, BuildDistanceComputer, BuildQueryComputer, DefaultContext, DelegateNeighbor,
        HasId,
    },
    utils::IntoUsize,
};
use diskann_utils::views::Matrix;

use super::{DefaultProvider, DefaultQuant};
use crate::model::graph::provider::async_::{
    SimpleNeighborProviderAsync, TableDeleteProviderAsync, inmem::FullPrecisionStore,
};

/// A full-precision accessor that spuriously fails for non-start points with a controllable
/// frequency.
///
/// This is meant to test the non-critical error handling of index operations.
#[derive(Debug, Clone, Copy)]
pub struct Flaky {
    fail_every: usize,
}

impl Flaky {
    pub(crate) fn new(fail_every: usize) -> Self {
        Self { fail_every }
    }
}

#[derive(Debug)]
pub struct TestError {
    is_transient: bool,
    handled: bool,
}

impl TestError {
    fn transient() -> Self {
        Self {
            is_transient: true,
            handled: false,
        }
    }
}

impl std::fmt::Display for TestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl TransientError<TestError> for TestError {
    fn acknowledge<D>(mut self, _why: D)
    where
        D: std::fmt::Display,
    {
        self.handled = true;
    }

    fn escalate<D>(mut self, _why: D) -> Self
    where
        D: std::fmt::Display,
    {
        assert!(self.is_transient);
        self.handled = true;
        self.is_transient = false;
        self
    }
}

impl Drop for TestError {
    fn drop(&mut self) {
        if self.is_transient {
            assert!(self.handled, "dropping an unhandled transient error!");
        }
    }
}

impl From<TestError> for ANNError {
    fn from(value: TestError) -> Self {
        assert!(
            !value.is_transient,
            "transient errors should not be converted!"
        );
        ANNError::log_async_error(value)
    }
}

impl ToRanked for TestError {
    type Transient = Self;
    type Error = Self;

    fn to_ranked(self) -> RankedError<Self, Self> {
        if self.is_transient {
            RankedError::Transient(self)
        } else {
            RankedError::Error(self)
        }
    }

    fn from_transient(transient: Self) -> Self {
        assert!(transient.is_transient);
        transient
    }

    fn from_error(error: Self) -> Self {
        assert!(!error.is_transient);
        error
    }
}

type Tda = TableDeleteProviderAsync;
type TestProvider = DefaultProvider<FullPrecisionStore<f32>, DefaultQuant, Tda>;

pub struct FlakyAccessor<'a> {
    provider: &'a TestProvider,
    fail_every: usize,
    get_count: usize,
}

type FullAccessor<'a> = super::FullAccessor<'a, f32, DefaultQuant, Tda, DefaultContext>;

impl SearchExt for FlakyAccessor<'_> {
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<u32>>> {
        std::future::ready(self.provider.starting_points())
    }
}

impl<'a> FlakyAccessor<'a> {
    fn new(provider: &'a TestProvider, fail_every: usize, get_count: usize) -> Self {
        assert_ne!(get_count, 0);
        Self {
            provider,
            get_count,
            fail_every,
        }
    }

    fn as_full(&self) -> FullAccessor<'a> {
        FullAccessor::new(self.provider)
    }
}

impl HasId for FlakyAccessor<'_> {
    type Id = u32;
}

impl Accessor for FlakyAccessor<'_> {
    /// This accessor returns raw slices. There *is* a chance of racing when the fast
    /// providers are used. We just have to live with it.
    type Element<'a>
        = &'a [f32]
    where
        Self: 'a;

    type ElementRef<'a> = &'a [f32];

    /// Choose to panic on an out-of-bounds access rather than propagate an error.
    type GetError = TestError;

    /// Return the full-precision vector stored at index `i`.
    ///
    /// This function always completes synchronously.
    #[inline(always)]
    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        // Do not fail when retrieving start points.
        //
        // NOTE: `is_not_start_point` takes a neighbor, but only looks at the `ID` portion,
        // so we can use a dummy neighbor.
        if self.provider.is_not_start_point()(&Neighbor::new(id, 0.0)) {
            self.get_count -= 1;
            if self.get_count == 0 {
                self.get_count = self.fail_every;
                return std::future::ready(Err(TestError::transient()));
            }
        }

        // SAFETY: We've decided to live with UB (undefined behavior) that can result from
        // potentially mixing unsynchronized reads and writes on the underlying memory.
        std::future::ready(Ok(unsafe {
            self.provider.base_vectors.get_vector_sync(id.into_usize())
        }))
    }
}

impl<'a> BuildDistanceComputer for FlakyAccessor<'a> {
    type DistanceComputerError = <FullAccessor<'a> as BuildDistanceComputer>::DistanceComputerError;
    type DistanceComputer = <FullAccessor<'a> as BuildDistanceComputer>::DistanceComputer;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        self.as_full().build_distance_computer()
    }
}

impl<'a, 'b> BuildQueryComputer<&'a [f32]> for FlakyAccessor<'b> {
    type QueryComputerError =
        <FullAccessor<'b> as BuildQueryComputer<&'a [f32]>>::QueryComputerError;
    type QueryComputer = <FullAccessor<'b> as BuildQueryComputer<&'a [f32]>>::QueryComputer;

    fn build_query_computer(
        &self,
        from: &'a [f32],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        self.as_full().build_query_computer(from)
    }
}

impl ExpandBeam<&[f32]> for FlakyAccessor<'_> {}

impl<'a> DelegateNeighbor<'a> for FlakyAccessor<'_> {
    type Delegate = &'a SimpleNeighborProviderAsync<u32>;
    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.provider.neighbors()
    }
}

impl<'x> SearchStrategy<TestProvider, &'x [f32]> for Flaky {
    type QueryComputer = <FullAccessor<'static> as BuildQueryComputer<&'x [f32]>>::QueryComputer;
    type SearchAccessor<'a> = FlakyAccessor<'a>;
    type SearchAccessorError = ANNError;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a TestProvider,
        _context: &'a DefaultContext,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(FlakyAccessor::new(
            provider,
            self.fail_every,
            self.fail_every,
        ))
    }
}

impl DefaultPostProcessor<TestProvider, &[f32]> for Flaky {
    default_post_processor!(CopyIds);
}

const STATIC_PRUNE_THRESHOLD: usize = 5;
/// We need to tune the flakiness of the `Prune` accessor so that occasionally, the first
/// item retrieved is a failure.
static START_COUNT: Mutex<usize> = Mutex::new(STATIC_PRUNE_THRESHOLD);

type WorkingSet = map::Map<u32, Box<[f32]>, map::Ref<[f32]>>;

impl PruneStrategy<TestProvider> for Flaky {
    type DistanceComputer<'a> = <FullAccessor<'a> as BuildDistanceComputer>::DistanceComputer;
    type PruneAccessor<'a> = FlakyAccessor<'a>;
    type PruneAccessorError = diskann::error::Infallible;
    type WorkingSet = WorkingSet;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a TestProvider,
        _context: &'a DefaultContext,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        let mut guard = START_COUNT.lock().unwrap();
        let start = *guard;
        *guard -= 1;
        if *guard == 0 {
            *guard = STATIC_PRUNE_THRESHOLD;
        }

        Ok(FlakyAccessor::new(provider, STATIC_PRUNE_THRESHOLD, start))
    }

    fn create_working_set(&self, capacity: usize) -> Self::WorkingSet {
        map::Builder::new(map::Capacity::Default).build(capacity)
    }
}

impl InsertStrategy<TestProvider, &[f32]> for Flaky {
    type PruneStrategy = Self;

    fn prune_strategy(&self) -> Self {
        *self
    }
}

impl MultiInsertStrategy<TestProvider, Matrix<f32>> for Flaky {
    type Seed = map::Builder<u32, map::Ref<[f32]>>;
    type WorkingSet = WorkingSet;
    type FinishError = diskann::error::Infallible;
    type InsertStrategy = Self;

    fn insert_strategy(&self) -> Self::InsertStrategy {
        *self
    }

    fn finish<Itr>(
        &self,
        _provider: &TestProvider,
        _ctx: &DefaultContext,
        batch: &Arc<Matrix<f32>>,
        ids: Itr,
    ) -> impl std::future::Future<Output = Result<Self::Seed, Self::FinishError>> + Send
    where
        Itr: ExactSizeIterator<Item = u32> + Send,
    {
        std::future::ready(Ok(map::Builder::new(map::Capacity::Default)
            .with_overlay(map::Overlay::from_batch(batch.clone(), ids))))
    }
}
