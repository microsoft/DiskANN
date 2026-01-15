/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{future::Future, sync::Mutex};

use diskann::{
    ANNError, ANNResult,
    error::{RankedError, ToRanked, TransientError},
    graph::glue::{
        AsElement, CopyIds, ExpandBeam, FillSet, InsertStrategy, PruneStrategy, SearchExt,
        SearchStrategy,
    },
    neighbor::Neighbor,
    provider::{
        Accessor, BuildDistanceComputer, BuildQueryComputer, DefaultContext, DelegateNeighbor,
        HasId,
    },
    utils::IntoUsize,
};

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

impl<'a> Accessor for FlakyAccessor<'a> {
    type Extended = &'a [f32];

    /// This accessor returns raw slices. There *is* a chance of racing when the fast
    /// providers are used. We just have to live with it.
    type Element<'b>
        = &'a [f32]
    where
        Self: 'b;

    type ElementRef<'b> = &'b [f32];

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

impl<'a> BuildQueryComputer<[f32]> for FlakyAccessor<'a> {
    type QueryComputerError = <FullAccessor<'a> as BuildQueryComputer<[f32]>>::QueryComputerError;
    type QueryComputer = <FullAccessor<'a> as BuildQueryComputer<[f32]>>::QueryComputer;

    fn build_query_computer(
        &self,
        from: &[f32],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        self.as_full().build_query_computer(from)
    }
}

impl ExpandBeam<[f32]> for FlakyAccessor<'_> {}

impl<'a> DelegateNeighbor<'a> for FlakyAccessor<'_> {
    type Delegate = &'a SimpleNeighborProviderAsync<u32>;
    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.provider.neighbors()
    }
}

impl SearchStrategy<TestProvider, [f32]> for Flaky {
    type QueryComputer = <FullAccessor<'static> as BuildQueryComputer<[f32]>>::QueryComputer;
    type SearchAccessor<'a> = FlakyAccessor<'a>;
    type SearchAccessorError = ANNError;
    type PostProcessor = CopyIds;

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

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

impl FillSet for FlakyAccessor<'_> {}

const STATIC_PRUNE_THRESHOLD: usize = 5;
/// We need to tune the flakiness of the `Prune` accessor so that occasionally, the first
/// item retrieved is a failure.
static START_COUNT: Mutex<usize> = Mutex::new(STATIC_PRUNE_THRESHOLD);

impl PruneStrategy<TestProvider> for Flaky {
    type DistanceComputer = <FullAccessor<'static> as BuildDistanceComputer>::DistanceComputer;
    type PruneAccessor<'a> = FlakyAccessor<'a>;
    type PruneAccessorError = diskann::error::Infallible;

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
}

impl InsertStrategy<TestProvider, [f32]> for Flaky {
    type PruneStrategy = Self;

    fn prune_strategy(&self) -> Self {
        *self
    }
}

impl<'a> AsElement<&'a [f32]> for FlakyAccessor<'a> {
    type Error = TestError;
    fn as_element(
        &mut self,
        vector: &'a [f32],
        _id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'a>, Self::Error>> + Send {
        std::future::ready(Ok(vector))
    }
}

// Directed test to fail vector retrieval during consolidation.
pub(crate) struct SuperFlaky;

impl PruneStrategy<TestProvider> for SuperFlaky {
    type DistanceComputer = <FullAccessor<'static> as BuildDistanceComputer>::DistanceComputer;
    type PruneAccessor<'a> = FlakyAccessor<'a>;
    type PruneAccessorError = diskann::error::Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a TestProvider,
        _context: &'a DefaultContext,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(FlakyAccessor::new(provider, 1, 1))
    }
}
