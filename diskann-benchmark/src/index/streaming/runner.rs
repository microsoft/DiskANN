/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{borrow::Cow, marker::PhantomData, num::NonZeroUsize, sync::Arc};

use diskann::{
    graph::{
        glue::{DefaultSearchStrategy, InplaceDeleteStrategy, InsertStrategy},
        DiskANNIndex, InplaceDeleteMethod,
    },
    provider::{DataProvider, Delete, SetElement},
    utils::ONE,
};
use diskann_benchmark_core::{
    build::{self, graph::SingleInsert, ids::Slice, Parallelism},
    recall::Rows,
    search::{self, graph::{KNN, Strategy}},
    streaming::graph::InplaceDelete,
};
use diskann_utils::{
    future::AsyncFriendly,
    views::{Matrix, MatrixView},
};
use tokio::runtime::Runtime;

use crate::{
    index::{
        build::{BuildKind, BuildStats},
        search::knn,
        streaming::{stats::{GenericStats, StreamStats}, ManagedStream},
    },
    inputs::graph_index::TopkSearchPhase,
};

pub(crate) trait Maintainer<DP: DataProvider> {
    fn maintain(
        &self,
        index: &Arc<DiskANNIndex<DP>>,
        runtime: &Runtime,
        ntasks: NonZeroUsize,
    ) -> anyhow::Result<StreamStats>;
}

pub(crate) struct StreamRunner<DP, T, S, M>
where
    DP: DataProvider,
{
    index: Arc<DiskANNIndex<DP>>,
    strategy: S,
    search: TopkSearchPhase,
    runtime: Runtime,
    ntasks: NonZeroUsize,
    inplace_delete_num_to_replace: usize,
    inplace_delete_method: InplaceDeleteMethod,
    maintainer: M,
    _marker: PhantomData<fn() -> T>,
}

impl<DP, T, S, M> StreamRunner<DP, T, S, M>
where
    DP: DataProvider,
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        index: Arc<DiskANNIndex<DP>>,
        strategy: S,
        search: TopkSearchPhase,
        runtime: Runtime,
        ntasks: NonZeroUsize,
        inplace_delete_num_to_replace: usize,
        inplace_delete_method: InplaceDeleteMethod,
        maintainer: M,
    ) -> Self {
        Self {
            index,
            strategy,
            search,
            runtime,
            ntasks,
            inplace_delete_num_to_replace,
            inplace_delete_method,
            maintainer,
            _marker: PhantomData,
        }
    }
}

impl<DP, T, S, M> StreamRunner<DP, T, S, M>
where
    DP: DataProvider<Context: Default, ExternalId = u32> + for<'a> SetElement<&'a [T]>,
    S: for<'a> InsertStrategy<DP, &'a [T]> + Clone + AsyncFriendly,
    T: AsyncFriendly + Clone,
{
    fn build(&self, data: MatrixView<'_, T>, slots: &[u32]) -> anyhow::Result<BuildStats> {
        let runner = SingleInsert::new(
            self.index.clone(),
            Arc::new(data.to_owned()),
            self.strategy.clone(),
            Slice::new(slots.into()),
        );

        let results = build::build(
            runner,
            Parallelism::fixed(Some(ONE), self.ntasks),
            &self.runtime,
        )?;

        BuildStats::new(BuildKind::SingleInsert, results)
    }
}

impl<DP, T, S, M> ManagedStream<T> for StreamRunner<DP, T, S, M>
where
    DP: DataProvider<Context: Default, ExternalId = u32, InternalId = u32>
        + for<'a> SetElement<&'a [T]>
        + Delete,
    DP::ExternalId: search::Id,
    S: for<'a> InsertStrategy<DP, &'a [T]>
        + for<'a> DefaultSearchStrategy<DP, &'a [T], DP::ExternalId>
        + InplaceDeleteStrategy<DP>
        + Clone
        + AsyncFriendly,
    T: AsyncFriendly + Clone,
    M: Maintainer<DP>,
{
    type Output = StreamStats;

    fn search(
        &self,
        queries: Arc<Matrix<T>>,
        groundtruth: &dyn Rows<u32>,
    ) -> anyhow::Result<Self::Output> {
        let knn = KNN::new(
            self.index.clone(),
            queries,
            Strategy::broadcast(self.strategy.clone()),
        )?;

        let steps = knn::SearchSteps::new(
            self.search.reps,
            &self.search.num_threads,
            &self.search.runs,
        );
        let results = knn::run(&knn, groundtruth, steps)?;
        Ok(StreamStats::Search(results))
    }

    fn insert(
        &self,
        data: MatrixView<'_, T>,
        slots: &[u32],
    ) -> anyhow::Result<Self::Output> {
        Ok(StreamStats::Insert(self.build(data, slots)?))
    }

    fn replace(
        &self,
        data: MatrixView<'_, T>,
        slots: &[u32],
    ) -> anyhow::Result<Self::Output> {
        Ok(StreamStats::Replace(self.build(data, slots)?))
    }

    fn delete(&self, slots: &[u32]) -> anyhow::Result<Self::Output> {
        let runner = InplaceDelete::new(
            self.index.clone(),
            self.strategy.clone(),
            self.inplace_delete_num_to_replace,
            self.inplace_delete_method,
            Slice::new(slots.into()),
        );

        let results = build::build(
            runner,
            Parallelism::fixed(Some(ONE), self.ntasks),
            &self.runtime,
        )?;

        Ok(StreamStats::Delete(GenericStats::new(
            Cow::Borrowed("Delete"),
            results,
        )?))
    }

    fn maintain(&self) -> anyhow::Result<Self::Output> {
        self.maintainer.maintain(&self.index, &self.runtime, self.ntasks)
    }
}

/////////////////////
// BfTree Maintain //
/////////////////////

/// BfTree maintenance: bf-tree uses hard deletes, no deferred cleanup needed.
#[cfg(feature = "bftree")]
pub(crate) struct BfTreeMaintainer;

#[cfg(feature = "bftree")]
impl<DP> Maintainer<DP> for BfTreeMaintainer
where
    DP: DataProvider,
{
    fn maintain(
        &self,
        _index: &Arc<DiskANNIndex<DP>>,
        _runtime: &Runtime,
        _ntasks: NonZeroUsize,
    ) -> anyhow::Result<StreamStats> {
        Ok(StreamStats::Maintain(Vec::new()))
    }
}
