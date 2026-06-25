/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Write;

use serde::{Deserialize, Serialize};

use diskann::utils::VectorRepr;
use diskann_benchmark_runner::{
    benchmark::{FailureScore, MatchScore},
    output::Output,
    utils::datatype::AsDataType,
    Benchmark, Checkpoint, Registry,
};
use half::f16;

use crate::{
    backend::graph_ivf::{
        build::{build_graph_ivf, GraphIvfBuildStats},
        search::{search_graph_ivf, GraphIvfSearchStats},
    },
    inputs::graph_ivf::{GraphIvfLoad, GraphIvfOperation, GraphIvfSource},
};

/// Graph-IVF benchmark, parameterized over the stored element type `T`.
struct GraphIvf<T> {
    _vector_type: std::marker::PhantomData<T>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct GraphIvfStats {
    pub(super) build: Option<GraphIvfBuildStats>,
    pub(super) search: GraphIvfSearchStats,
}

impl<T> GraphIvf<T>
where
    T: VectorRepr,
{
    fn new() -> Self {
        Self {
            _vector_type: std::marker::PhantomData,
        }
    }
}

impl<T> Benchmark for GraphIvf<T>
where
    T: VectorRepr + AsDataType,
{
    type Input = GraphIvfOperation;
    type Output = GraphIvfStats;

    fn try_match(&self, input: &GraphIvfOperation) -> Result<MatchScore, FailureScore> {
        let data_type = match &input.source {
            GraphIvfSource::Load(load) => load.data_type,
            GraphIvfSource::Build(build) => build.data_type,
        };
        crate::utils::match_data_type::<T>(data_type)
    }

    fn description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        input: Option<&GraphIvfOperation>,
    ) -> std::fmt::Result {
        match input {
            Some(arg) => {
                let desc = match &arg.source {
                    GraphIvfSource::Load(load) => T::describe(load.data_type),
                    GraphIvfSource::Build(build) => T::describe(build.data_type),
                };
                write!(f, "{}", desc)
            }
            None => write!(f, "{}", T::DATA_TYPE),
        }
    }

    fn run(
        &self,
        input: &GraphIvfOperation,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<GraphIvfStats> {
        writeln!(output, "{}", input.source)?;

        let (build_stats, index_load) = match &input.source {
            GraphIvfSource::Load(load) => (None, (*load).clone()),
            GraphIvfSource::Build(build) => {
                let stats = build_graph_ivf::<T>(build)?;
                (
                    Some(stats),
                    GraphIvfLoad {
                        data_type: build.data_type,
                        load_path: build.save_path.clone(),
                    },
                )
            }
        };
        if let Some(build_stats) = &build_stats {
            writeln!(output, "{}", build_stats)?;
        }

        writeln!(output, "{}", input.search_phase)?;
        let search_stats = search_graph_ivf::<T>(&index_load, &input.search_phase)?;
        writeln!(output, "{}", search_stats)?;

        Ok(GraphIvfStats {
            build: build_stats,
            search: search_stats,
        })
    }
}

////////////////////////////
// Benchmark Registration //
////////////////////////////

pub(super) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register("graph-ivf-f32", GraphIvf::<f32>::new())?;
    registry.register("graph-ivf-f16", GraphIvf::<f16>::new())?;
    registry.register("graph-ivf-u8", GraphIvf::<u8>::new())?;
    registry.register("graph-ivf-i8", GraphIvf::<i8>::new())?;
    Ok(())
}
