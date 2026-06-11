/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Benchmark backend for the `diskann-inmem` (inmem2) provider.
//!
//! This wires up the inmem2 `Provider<Full<f32>>` to the standard build and search
//! infrastructure in `diskann-benchmark-core`, giving us parallel insertion via
//! [`SingleInsert`] and KNN search with recall/latency reporting via [`KNN`].

use std::{io::Write, num::NonZeroUsize, sync::Arc};

use diskann::graph::{self, DiskANNIndex};
use diskann_benchmark_core::{
    self as benchmark_core, build as build_core, recall::GroundTruthMode, search as core_search,
};
use diskann_benchmark_runner::{
    benchmark::{FailureScore, MatchScore},
    files::InputFile,
    output::Output,
    Benchmark, Checker, Checkpoint, Input, Registry,
};
use diskann_inmem::{layers::Full, Provider, Strategy};
use diskann_utils::views::Matrix;
use diskann_vector::distance::Metric;
use serde::{Deserialize, Serialize};

use crate::{backend::index::build::ProgressMeter, utils::datafiles};

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register("inmem2-f32", Inmem2)?;
    Ok(())
}

///////////
// Input //
///////////

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Inmem2Build {
    data: InputFile,
    queries: InputFile,
    groundtruth: InputFile,

    max_degree: usize,
    l_build: usize,
    alpha: f32,

    search_n: usize,
    search_l: Vec<usize>,
    recall_k: usize,

    num_threads: usize,
    reps: NonZeroUsize,
}

impl Input for Inmem2Build {
    type Raw = Inmem2Build;

    fn tag() -> &'static str {
        "inmem2"
    }

    fn from_raw(mut raw: Self::Raw, checker: &mut Checker) -> anyhow::Result<Self> {
        raw.data.resolve(checker)?;
        raw.queries.resolve(checker)?;
        raw.groundtruth.resolve(checker)?;
        Ok(raw)
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    fn example() -> Self {
        Self {
            data: InputFile::new("path/to/base.bin"),
            queries: InputFile::new("path/to/query.bin"),
            groundtruth: InputFile::new("path/to/groundtruth.bin"),
            max_degree: 64,
            l_build: 100,
            alpha: 1.2,
            search_n: 10,
            search_l: vec![10, 20, 50, 100],
            recall_k: 10,
            num_threads: 4,
            reps: NonZeroUsize::new(3).unwrap(),
        }
    }
}

impl std::fmt::Display for Inmem2Build {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "inmem2 f32 benchmark")?;
        writeln!(f, "  max_degree: {}", self.max_degree)?;
        writeln!(f, "  l_build: {}", self.l_build)?;
        writeln!(f, "  alpha: {}", self.alpha)?;
        writeln!(f, "  search_n: {}", self.search_n)?;
        writeln!(f, "  search_l: {:?}", self.search_l)?;
        writeln!(f, "  recall_k: {}", self.recall_k)?;
        writeln!(f, "  num_threads: {}", self.num_threads)?;
        writeln!(f, "  reps: {}", self.reps)
    }
}

///////////////
// Benchmark //
///////////////

#[derive(Debug)]
struct Inmem2;

impl Benchmark for Inmem2 {
    type Input = Inmem2Build;
    type Output = ();

    fn try_match(&self, _input: &Inmem2Build) -> Result<MatchScore, FailureScore> {
        Ok(MatchScore(0))
    }

    fn description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        input: Option<&Inmem2Build>,
    ) -> std::fmt::Result {
        match input {
            Some(i) => write!(f, "{i}"),
            None => write!(f, "inmem2 f32 benchmark"),
        }
    }

    fn run(
        &self,
        input: &Inmem2Build,
        checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<()> {
        writeln!(output, "{input}")?;

        // Load data.
        let data: Arc<Matrix<f32>> =
            Arc::new(datafiles::load_dataset(datafiles::BinFile(&input.data))?);

        let dim = data.ncols();
        let num_points = data.nrows();
        writeln!(output, "Loaded {num_points} points, dim={dim}")?;

        // Compute a start point as the centroid of the first min(1000, N) points.
        let sample = num_points.min(1000);
        let mut centroid = vec![0.0f32; dim];
        for i in 0..sample {
            for (c, &v) in centroid.iter_mut().zip(data.row(i)) {
                *c += v;
            }
        }
        let inv = 1.0 / sample as f32;
        centroid.iter_mut().for_each(|c| *c *= inv);

        // Build inmem2 provider.
        let metric = Metric::L2;
        let exact_max_degree = (input.max_degree as f32 * 1.3) as usize;
        let layer = Full::<f32>::new(dim, metric);
        let start_points: [&[f32]; 1] = [&centroid];
        let provider = Provider::new(layer, num_points, start_points);

        let config = graph::config::Builder::new_with(
            input.max_degree,
            graph::config::MaxDegree::new(exact_max_degree),
            input.l_build,
            metric.into(),
            |b| {
                b.alpha(input.alpha);
            },
        )
        .build()?;

        let index = Arc::new(DiskANNIndex::new(config, provider, None));

        // Build via SingleInsert.
        let rt = benchmark_core::tokio::runtime(input.num_threads)?;
        let builder = build_core::graph::SingleInsert::new(
            index.clone(),
            data,
            Strategy,
            build_core::ids::Identity::<u32>::new(),
        );

        writeln!(
            output,
            "Building index with {} threads...",
            input.num_threads
        )?;
        let build_results = build_core::build_tracked(
            builder,
            build_core::Parallelism::dynamic(
                diskann::utils::ONE,
                NonZeroUsize::new(input.num_threads).unwrap(),
            ),
            &rt,
            Some(&ProgressMeter::new(output)),
        )?;

        let total_build_time = build_results.end_to_end_latency();
        writeln!(
            output,
            "Build complete in {:.2}s",
            total_build_time.as_seconds()
        )?;
        checkpoint.checkpoint(&total_build_time)?;

        // Search.
        let queries: Arc<Matrix<f32>> =
            Arc::new(datafiles::load_dataset(datafiles::BinFile(&input.queries))?);
        let max_k = input.recall_k;
        let groundtruth =
            datafiles::load_groundtruth(datafiles::BinFile(&input.groundtruth), Some(max_k))?;

        writeln!(output, "Loaded {} queries", queries.nrows())?;

        let knn = benchmark_core::search::graph::KNN::new(
            index,
            queries,
            benchmark_core::search::graph::Strategy::broadcast(Strategy),
        )?;

        let num_threads = NonZeroUsize::new(input.num_threads).unwrap();

        for &search_l in &input.search_l {
            let params = graph::search::Knn::new(input.search_n, search_l, None)?;

            let setup = core_search::Setup {
                threads: num_threads,
                tasks: num_threads,
                reps: input.reps,
            };

            let run = core_search::Run::new(params, setup);

            let summaries = core_search::search_all(
                knn.clone(),
                std::iter::once(run),
                benchmark_core::search::graph::knn::Aggregator::new(
                    &groundtruth,
                    input.recall_k,
                    input.search_n,
                    GroundTruthMode::Fixed,
                ),
            )?;

            for summary in &summaries {
                let qps: Vec<f64> = summary
                    .end_to_end_latencies
                    .iter()
                    .map(|lat| summary.recall.num_queries as f64 / lat.as_seconds())
                    .collect();
                let max_qps = qps.iter().cloned().fold(0.0f64, f64::max);
                let mean_qps = qps.iter().sum::<f64>() / qps.len() as f64;

                writeln!(
                    output,
                    "  L={:<4} recall={:.4}  QPS={:.0} (max {:.0})  cmps={:.1}  hops={:.1}",
                    search_l,
                    summary.recall.average,
                    mean_qps,
                    max_qps,
                    summary.mean_cmps,
                    summary.mean_hops,
                )?;
            }
        }

        Ok(())
    }
}
