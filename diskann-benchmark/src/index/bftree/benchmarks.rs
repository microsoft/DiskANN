/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::Arc;

use diskann::{
    graph::{glue, DiskANNIndex},
    provider::{self, DataProvider, DefaultContext},
    utils::VectorRepr,
};
use diskann_benchmark_core::{self as benchmark_core, recall::GroundTruthMode};
use diskann_benchmark_runner::output::Output;
use diskann_utils::{
    future::AsyncFriendly,
    views::rowmajor::{self, Matrix},
};

use super::{
    build::BuildStats,
    result::AggregatedSearchResults,
    search::{self, plugins},
};
use crate::{
    inputs::graph_index::{IndexBuild, SearchPhase},
    utils::datafiles,
};

/// Associate a type (usually a [`diskann::provider::DataProvider`]) with a full-precision
/// element type. This is used in implementations of [`plugins::Plugin`] to derive the
/// correct query types to load.
pub(crate) trait QueryType {
    type Element: VectorRepr;
}

// Simplify reasoning about this rather hefty type.
type Index<DP> = Arc<DiskANNIndex<DP>>;

pub(crate) fn run_build<T, BF, CF, B, DP>(
    input: &IndexBuild,
    build_strategy: B,
    data: Option<Arc<rowmajor::Owned<T>>>,
    output: &mut dyn Output,
    create: CF,
    build: BF,
) -> anyhow::Result<(Index<DP>, BuildStats)>
where
    DP: DataProvider<Context = DefaultContext, InternalId = u32, ExternalId = u32>
        + for<'a> provider::SetElement<&'a [T]>,
    CF: FnOnce(rowmajor::Ref<T>) -> anyhow::Result<Arc<DiskANNIndex<DP>>>,
    T: diskann::graph::SampleableForStart + std::fmt::Debug + Copy + AsyncFriendly + bytemuck::Pod,
    B: for<'a> glue::SearchStrategy<'a, DP, &'a [T]> + Clone + Send + Sync,
    BF: FnOnce(
        Index<DP>,
        B,
        Arc<rowmajor::Owned<T>>,
        &IndexBuild,
        &mut dyn Output,
    ) -> anyhow::Result<BuildStats>,
{
    let data = match data {
        Some(data) => data,
        None => Arc::new(datafiles::load_dataset(datafiles::BinFile(input.data()))?),
    };

    let index = create(data.as_view())?;
    let build_stats = build(index.clone(), build_strategy.clone(), data, input, output)?;

    Ok((index, build_stats))
}

/// A new-type wrapper for [`glue::SearchStrategy`].
///
/// This exists so we can implement [`search::Plugin`] for a raw generic `DP` without
/// forming a blanket implementation for all `DP`/parameter `P` pairs.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Strategy<S>(S);

impl<S> Strategy<S> {
    pub(crate) fn new(strategy: S) -> Self {
        Self(strategy)
    }

    pub(crate) fn inner(&self) -> S
    where
        S: Clone,
    {
        self.0.clone()
    }
}

impl<DP, S> search::Plugin<DP, SearchPhase, Strategy<S>> for plugins::Topk
where
    DP: DataProvider<Context: Default, InternalId = u32, ExternalId = u32> + QueryType,
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [DP::Element],
            SearchAccessor: glue::SearchAccessor,
        > + Clone
        + AsyncFriendly,
{
    fn is_match(&self, phase: &SearchPhase) -> bool {
        plugins::Topk::is_match(phase)
    }

    fn kind(&self) -> &'static str {
        plugins::Topk::as_str()
    }

    fn run(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        phase: &SearchPhase,
        strategy: &Strategy<S>,
    ) -> anyhow::Result<AggregatedSearchResults> {
        let topk = phase.as_topk()?;

        let queries: Arc<rowmajor::Owned<DP::Element>> =
            Arc::new(datafiles::load_dataset(datafiles::BinFile(&topk.queries))?);

        // compute the maximum value of k used in any search
        let max_k = topk.max_k();

        let groundtruth =
            datafiles::load_groundtruth(datafiles::BinFile(&topk.groundtruth), Some(max_k))?;

        let knn = benchmark_core::search::graph::KNN::new(
            index.clone(),
            queries,
            benchmark_core::search::graph::Strategy::broadcast(strategy.inner()),
        )?;

        let steps = search::knn::SearchSteps::new(
            topk.reps,
            &topk.num_threads,
            &topk.runs,
            GroundTruthMode::Fixed,
        );

        let results = search::knn::run(&knn, &groundtruth, steps)?;
        Ok(AggregatedSearchResults::Topk(results))
    }
}
