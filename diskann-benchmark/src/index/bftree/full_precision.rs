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
    benchmark::{MatchContext, Score},
    output::Output,
    utils::datatype::AsDataType,
    Benchmark, Checkpoint,
};
use diskann_bftree::{BfTreeProvider, NoStore};
use diskann_providers::{
    model::graph::provider::async_::common::FullPrecision,
    storage::{FileStorageProvider, SaveWith},
};

use crate::{
    index::{
        benchmarks::{run_build, QueryType, Strategy},
        build::single_or_multi_insert,
        result::BuildResult,
        search::plugins::{Plugin, Plugins},
    },
    inputs::{
        bftree::{BfTreeBuild, QuantConfig},
        graph_index::SearchPhase,
    },
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
    T: VectorRepr + AsDataType + SampleableForStart + 'static,
{
    type Input = BfTreeBuild;
    type Output = BuildResult;

    fn try_match(&self, input: &Self::Input, context: &MatchContext) -> Score {
        let mut score = context.success(0);
        if !matches!(input.quantization(), QuantConfig::None) {
            score.fail(1, &"Full-precision index does not support quantization");
        }
        utils::match_data_type::<T>(&mut score, input.data_type());
        if !self.plugins.is_match(input.search_phase()) {
            score.fail(
                1,
                &format_args!(
                    "Unsupported search phase: \"{}\" - expected one of {}",
                    input.search_phase().kind(),
                    self.plugins.format_kinds(),
                ),
            );
        }

        score
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Data/Query Type: {}", T::DATA_TYPE)?;
        writeln!(f, "Search Kinds: {}", self.plugins.format_kinds())
    }

    fn run(
        &self,
        input: &BfTreeBuild,
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
                let params = input.bftree_parameters(data.nrows(), data.ncols())?;
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
