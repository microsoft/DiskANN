/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{io::Write, sync::Arc};

use diskann::{
    graph::{DiskANNIndex, SampleableForStart},
    utils::VectorRepr,
};
use diskann_benchmark_runner::{
    benchmark::{FailureScore, MatchScore},
    output::Output,
    utils::datatype::AsDataType,
    Benchmark, Checkpoint,
};
use diskann_bftree::{BfTreeProvider, NoStore};
use diskann_providers::{
    model::graph::provider::async_::common::FullPrecision,
    storage::{FileStorageProvider, SaveWith},
};
use diskann_utils::sampling::WithApproximateNorm;

use crate::{
    backend::index::{
        benchmarks::{run_build, QueryType, Strategy},
        build::single_or_multi_insert,
        result::BuildResult,
        search::plugins::{Plugin, Plugins},
    },
    inputs::{bftree::BfTreeFullPrecisionBuild, graph_index::SearchPhase},
    utils::{self, tokio},
};

type BfTreeFPProvider<T> = BfTreeProvider<T, NoStore>;

pub(super) struct BfTreeFullPrecision<T>
where
    T: VectorRepr,
{
    plugins: Plugins<BfTreeFPProvider<T>, SearchPhase, Strategy<FullPrecision>>,
}

impl<T> BfTreeFullPrecision<T>
where
    T: VectorRepr,
{
    pub(super) fn new() -> Self {
        Self {
            plugins: Plugins::new(),
        }
    }

    pub(super) fn search<P>(mut self, plugin: P) -> Self
    where
        P: Plugin<BfTreeFPProvider<T>, SearchPhase, Strategy<FullPrecision>> + 'static,
    {
        self.plugins.register(plugin);
        self
    }
}

impl<T> QueryType for BfTreeFPProvider<T>
where
    T: VectorRepr,
{
    type Element = T;
}

impl<T> Benchmark for BfTreeFullPrecision<T>
where
    T: VectorRepr + AsDataType + SampleableForStart + WithApproximateNorm + 'static,
{
    type Input = BfTreeFullPrecisionBuild;
    type Output = BuildResult;

    fn try_match(&self, input: &Self::Input) -> Result<MatchScore, FailureScore> {
        let score = utils::match_data_type::<T>(input.data_type());
        if self.plugins.is_match(input.search_phase()) {
            score
        } else {
            match score {
                Ok(_) => Err(FailureScore(0)),
                Err(s) => Err(s),
            }
        }
    }

    fn description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        input: Option<&Self::Input>,
    ) -> std::fmt::Result {
        match input {
            Some(arg) => {
                let desc = T::describe(arg.build().data_type());
                if !desc.is_match() {
                    writeln!(f, "Data/Query Type: {}", desc)?;
                }
                if !self.plugins.is_match(arg.search_phase()) {
                    writeln!(
                        f,
                        "Unsupported search phase: \"{}\" - expected one of {}",
                        arg.search_phase().kind(),
                        self.plugins.format_kinds(),
                    )?;
                }
                Ok(())
            }
            None => {
                writeln!(f, "Data/Query Type: {}", T::DATA_TYPE)?;
                writeln!(f, "Search Kinds: {}", self.plugins.format_kinds())
            }
        }
    }

    fn run(
        &self,
        input: &Self::Input,
        checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        writeln!(output, "{}", input)?;

        let (index, build_stats) = run_build(
            input.build(),
            FullPrecision,
            None,
            output,
            |data| {
                let config = input.try_as_config()?.build()?;
                let params = input.bftree_parameters(data.nrows(), data.ncols());
                let start_points = input.build().start_point_strategy().compute(data)?;
                let provider = BfTreeProvider::new(params, start_points.as_view(), NoStore)?;
                Ok(Arc::new(DiskANNIndex::new(config, provider, None)))
            },
            single_or_multi_insert,
        )?;

        checkpoint.checkpoint(&build_stats)?;

        // save the index if requested
        if let Some(save_path) = input.build().save_path() {
            tokio::block_on(
                index
                    .provider()
                    .save_with(&FileStorageProvider, &save_path.to_string()),
            )?;
        }

        let search_results =
            self.plugins
                .run(index, input.search_phase(), &Strategy::new(FullPrecision))?;

        let result = BuildResult::new(Some(build_stats), search_results);
        writeln!(output, "\n\n{}", result)?;
        Ok(result)
    }
}
