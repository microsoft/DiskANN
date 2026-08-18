/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt, io::Write};

use serde::{Deserialize, Serialize};

use diskann::utils::VectorRepr;
use diskann_benchmark_runner::{
    benchmark::{MatchContext, Score},
    output::Output,
    Benchmark, Checkpoint, Registry,
};
use diskann_providers::common::MinMax8;
use half::f16;

use crate::{
    graph_ivf::{
        build::{build_graph_ivf, GraphIvfBuildStats},
        element::GraphIvfElement,
        online::{build_graph_ivf_online, GraphIvfOnlineBuildStats},
        search::{search_graph_ivf, GraphIvfSearchStats},
        streaming::{build_graph_ivf_runbook, GraphIvfRunbookStats},
    },
    inputs::graph_ivf::{GraphIvfLoad, GraphIvfOperation, GraphIvfSource},
};

/// Graph-IVF benchmark, parameterized over the stored element type `T`.
struct GraphIvf<T> {
    _vector_type: std::marker::PhantomData<T>,
}

/// Build statistics, tagged by how the index was constructed.
///
/// The builders share no parameters and report disjoint telemetry, so a single
/// flattened struct would be mostly-null whichever ran; the tag keeps the output
/// self-describing for downstream analysis.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "build_kind")]
pub(super) enum GraphIvfBuildOutcome {
    Static(GraphIvfBuildStats),
    Online(GraphIvfOnlineBuildStats),
    OnlineRunbook(GraphIvfRunbookStats),
}

impl fmt::Display for GraphIvfBuildOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Static(stats) => stats.fmt(f),
            Self::Online(stats) => stats.fmt(f),
            Self::OnlineRunbook(stats) => stats.fmt(f),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct GraphIvfStats {
    pub(super) build: Option<GraphIvfBuildOutcome>,
    pub(super) search: Option<GraphIvfSearchStats>,
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
    T: GraphIvfElement,
{
    type Input = GraphIvfOperation;
    type Output = GraphIvfStats;

    fn try_match(&self, input: &GraphIvfOperation, context: &MatchContext) -> Score {
        if input.source.data_type() == T::DATA_TYPE {
            context.success(0)
        } else {
            context.fail(
                crate::utils::DATA_TYPE_MISMATCH,
                &format_args!(
                    "Expected data type {:?}, instead got {:?}",
                    T::DATA_TYPE,
                    input.source.data_type()
                ),
            )
        }
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", T::DATA_TYPE)
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
            GraphIvfSource::Static(build) => {
                let stats = build_graph_ivf::<T>(build)?;
                (
                    Some(GraphIvfBuildOutcome::Static(stats)),
                    GraphIvfLoad {
                        data_type: build.data_type,
                        load_path: build.save_path.clone(),
                        centroid_search: build.routing.mode(),
                    },
                )
            }
            GraphIvfSource::Online(online) => {
                let stats = build_graph_ivf_online::<T>(online)?;
                (
                    Some(GraphIvfBuildOutcome::Online(stats)),
                    GraphIvfLoad {
                        data_type: online.data_type,
                        load_path: online.save_path.clone(),
                        centroid_search: online.routing.mode(),
                    },
                )
            }
            GraphIvfSource::OnlineRunbook(runbook) => {
                let stats = build_graph_ivf_runbook::<T>(runbook)?;
                (
                    Some(GraphIvfBuildOutcome::OnlineRunbook(stats)),
                    GraphIvfLoad {
                        data_type: runbook.build.data_type,
                        load_path: runbook.build.save_path.clone(),
                        centroid_search: runbook.build.routing.mode(),
                    },
                )
            }
        };
        if let Some(build_stats) = &build_stats {
            writeln!(output, "{}", build_stats)?;
        }

        let search_stats = match &input.search_phase {
            Some(search_phase) => {
                writeln!(output, "{search_phase}")?;
                let stats = search_graph_ivf::<T>(&index_load, search_phase)?;
                writeln!(output, "{stats}")?;
                Some(stats)
            }
            None => None,
        };

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
    registry.register("graph-ivf-minmax8", GraphIvf::<MinMax8>::new())?;
    Ok(())
}
