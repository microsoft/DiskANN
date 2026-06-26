/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Benchmark backend for the `diskann-inmem` (inmem2) provider.
//!
//! This wires up the inmem2 `Provider<Full<f32>>` to the standard build and search
//! infrastructure in `diskann-benchmark-core`, giving us parallel insertion via
//! [`SingleInsert`] and KNN search with recall/latency reporting via [`KNN`].

use std::{io::Write, num::NonZeroUsize, ops::Range, sync::Arc};

use diskann::{
    graph::{self, DiskANNIndex, StartPointStrategy},
    provider::{self as ann_provider},
};
use diskann_benchmark_core::{
    self as benchmark_core, build as build_core,
    recall::{self, GroundTruthMode},
    search as core_search,
    streaming::{self, executors::bigann, Executor},
};
use diskann_benchmark_runner::{
    benchmark::{FailureScore, MatchScore},
    files::InputFile,
    output::Output,
    utils::datatype::DataType,
    Benchmark, Checker, Checkpoint, Input, Registry,
};
use diskann_inmem::{layers::Full, Provider, Strategy};
use diskann_utils::views::{Matrix, MatrixView};
use diskann_vector::distance::Metric;
use serde::{Deserialize, Serialize};

use crate::{
    index::build::ProgressMeter,
    inputs::graph_index::DynamicRunbookParams,
    utils::{datafiles, SimilarityMeasure},
};

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    // registry.register("inmem2-f32", Inmem2)?;
    registry.register("inmem2-f32-stream", Inmem2Stream)?;
    Ok(())
}

///////////
// Input //
///////////

mod dto {
    use super::*;

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct KnnSweep {
        pub(super) search_n: usize,
        pub(super) search_l: Vec<usize>,
        pub(super) recall_k: usize,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct KnnSearch {
        pub(super) queries: InputFile,
        pub(super) groundtruth: InputFile,
        pub(super) reps: NonZeroUsize,
        pub(super) num_threads: Vec<NonZeroUsize>,
        pub(super) runs: Vec<KnnSweep>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct Data {
        pub(super) data_type: DataType,
        pub(super) data: InputFile,
        pub(super) distance: SimilarityMeasure,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct BuildParams {
        pub(super) pruned_degree: usize,
        pub(super) max_degree: usize,
        pub(super) l_build: usize,
        pub(super) alpha: f32,
        pub(super) num_threads: NonZeroUsize,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct StaticBuild {
        pub(super) data: Data,
        pub(super) build: BuildParams,
        pub(super) search: KnnSearch,
    }
}

#[derive(Debug)]
struct KnnInstance {
    knn: graph::search::Knn,
    recall_k: usize,
}

#[derive(Debug)]
struct KnnSearch {
    queries: InputFile,
    groundtruth: InputFile,
    reps: NonZeroUsize,
    num_threads: Vec<NonZeroUsize>,
    runs: Vec<KnnInstance>,
}

impl KnnSearch {
    fn from_raw(raw: dto::KnnSearch, checker: Option<&mut Checker>) -> anyhow::Result<Self> {
        let dto::KnnSearch {
            mut queries,
            mut groundtruth,
            reps,
            num_threads,
            runs,
        } = raw;

        if let Some(checker) = checker {
            queries.resolve(checker)?;
            groundtruth.resolve(checker)?;
        }

        let runs = runs
            .into_iter()
            .flat_map(|sweep| {
                sweep
                    .search_l
                    .into_iter()
                    .map(move |search_l| -> anyhow::Result<_> {
                        let knn = graph::search::Knn::new_default(sweep.search_n, search_l)?;
                        Ok(KnnInstance {
                            knn,
                            recall_k: sweep.recall_k,
                        })
                    })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(Self {
            queries,
            groundtruth,
            reps,
            num_threads,
            runs,
        })
    }

    fn maximum_recall_k(&self) -> usize {
        self.runs.iter().map(|r| r.recall_k).max().unwrap_or(0)
    }
}

#[derive(Debug)]
struct Data {
    data_type: DataType,
    data: InputFile,
    distance: Metric,
}

impl Data {
    fn from_raw(raw: dto::Data, checker: Option<&mut Checker>) -> anyhow::Result<Self> {
        let dto::Data {
            data_type,
            mut data,
            distance,
        } = raw;

        if let Some(checker) = checker {
            data.resolve(checker)?;
        }

        Ok(Self {
            data_type,
            data,
            distance: distance.into(),
        })
    }
}

#[derive(Debug)]
struct BuildParams {
    config: graph::Config,
    num_threads: NonZeroUsize,
}

impl BuildParams {
    fn from_raw(raw: dto::BuildParams, metric: Metric) -> anyhow::Result<Self> {
        let dto::BuildParams {
            pruned_degree,
            max_degree,
            l_build,
            alpha,
            num_threads,
        } = raw;

        let config = graph::config::Builder::new_with(
            pruned_degree,
            graph::config::MaxDegree::new(max_degree),
            l_build,
            metric.into(),
            |b| {
                b.alpha(alpha);
            },
        )
        .build()?;

        Ok(Self {
            config,
            num_threads,
        })
    }
}

#[derive(Debug)]
struct StaticBuild {
    data: Data,
    build: BuildParams,
    search: KnnSearch,
}

impl StaticBuild {
    fn from_raw(raw: dto::StaticBuild, mut checker: Option<&mut Checker>) -> anyhow::Result<Self> {
        let dto::StaticBuild {
            data,
            build,
            search,
        } = raw;

        let data = Data::from_raw(data, checker.as_deref_mut())?;
        let build = BuildParams::from_raw(build, data.distance)?;
        let search = KnnSearch::from_raw(search, checker)?;

        Ok(Self {
            data,
            build,
            search,
        })
    }
}

impl Input for StaticBuild {
    type Raw = dto::StaticBuild;

    fn tag() -> &'static str {
        "inmem2"
    }

    fn from_raw(raw: Self::Raw, checker: &mut Checker) -> anyhow::Result<Self> {
        Self::from_raw(raw, Some(checker))
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(())?)
    }

    fn example() -> Self::Raw {
        const FOUR: NonZeroUsize = NonZeroUsize::new(4).unwrap();
        const THREE: NonZeroUsize = NonZeroUsize::new(3).unwrap();

        dto::StaticBuild {
            data: dto::Data {
                data_type: DataType::Float32,
                data: InputFile::new("path/to/data"),
                distance: SimilarityMeasure::SquaredL2,
            },
            build: dto::BuildParams {
                pruned_degree: 28,
                max_degree: 32,
                l_build: 100,
                alpha: 1.2,
                num_threads: FOUR,
            },
            search: dto::KnnSearch {
                queries: InputFile::new("path/to/queries"),
                groundtruth: InputFile::new("path/to/groundtruth"),
                reps: THREE,
                num_threads: vec![FOUR],
                runs: vec![dto::KnnSweep {
                    search_n: 10,
                    search_l: vec![10, 20, 30, 40, 50],
                    recall_k: 10,
                }],
            },
        }
    }
}

// impl std::fmt::Display for Inmem2Build {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         writeln!(f, "inmem2 f32 benchmark")?;
//         writeln!(f, "  max_degree: {}", self.max_degree)?;
//         writeln!(f, "  l_build: {}", self.l_build)?;
//         writeln!(f, "  alpha: {}", self.alpha)?;
//         writeln!(f, "  search_n: {}", self.search_n)?;
//         writeln!(f, "  search_l: {:?}", self.search_l)?;
//         writeln!(f, "  recall_k: {}", self.recall_k)?;
//         writeln!(f, "  num_threads: {}", self.num_threads)?;
//         writeln!(f, "  reps: {}", self.reps)
//     }
// }

///////////////
// Benchmark //
///////////////

#[derive(Debug)]
struct Build<T>(std::marker::PhantomData<T>);

impl<T> Build<T> {
    fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

// impl<T> Benchmark for Build<T>
// where
//     T: diskann_inmem::layers::FullPrecision
//         + diskann::graph::SampleableForStart,
// {
//     type Input = StaticBuild;
//     type Output = ();
//
//     fn try_match(&self, input: &StaticBuild) -> Result<MatchScore, FailureScore> {
//         Ok(MatchScore(0))
//     }
//
//     fn description(
//         &self,
//         f: &mut std::fmt::Formatter<'_>,
//         input: Option<&StaticBuild>,
//     ) -> std::fmt::Result {
//         Ok(())
//         // match input {
//         //     Some(i) => write!(f, "{i}"),
//         //     None => write!(f, "inmem2 f32 benchmark"),
//         // }
//     }
//
//     fn run(
//         &self,
//         input: &StaticBuild,
//         checkpoint: Checkpoint<'_>,
//         mut output: &mut dyn Output,
//     ) -> anyhow::Result<()> {
//         // writeln!(output, "{input}")?;
//
//         // Load data.
//         let data: Arc<Matrix<T>> = Arc::new(datafiles::load_dataset(datafiles::BinFile(
//             &input.data.data,
//         ))?);
//
//         let dim = data.ncols();
//         let num_points = data.nrows();
//         writeln!(output, "Loaded {num_points} points, dim={dim}")?;
//
//         // Compute the medoid of the dataset as the single start point.
//         let start = StartPointStrategy::Medoid.compute(data.as_view())?;
//         let layer = Full::<T>::new(dim, input.data.distance);
//         let config =
//             diskann_inmem::provider::Config::new(num_points, input.build.config.max_degree().get());
//         let provider = Provider::<_, u32>::new(layer, config, start.row_iter())?;
//
//         let index = Arc::new(DiskANNIndex::new(
//             input.build.config.clone(),
//             provider,
//             None,
//         ));
//
//         // Build via SingleInsert.
//         let rt = benchmark_core::tokio::runtime(input.build.num_threads.get())?;
//         let builder = build_core::graph::SingleInsert::new(
//             index.clone(),
//             data,
//             Strategy,
//             build_core::ids::Identity::<u32>::new(),
//         );
//
//         // writeln!(
//         //     output,
//         //     "Building index with {} threads...",
//         //     input.num_threads
//         // )?;
//         let build_results = build_core::build_tracked(
//             builder,
//             build_core::Parallelism::dynamic(
//                 diskann::utils::ONE,
//                 input.build.num_threads,
//             ),
//             &rt,
//             Some(&ProgressMeter::new(output)),
//         )?;
//
//         let total_build_time = build_results.end_to_end_latency();
//         writeln!(
//             output,
//             "Build complete in {:.2}s",
//             total_build_time.as_seconds()
//         )?;
//         checkpoint.checkpoint(&total_build_time)?;
//
//         // Search.
//         let queries: Arc<Matrix<T>> =
//             Arc::new(datafiles::load_dataset(datafiles::BinFile(&input.search.queries))?);
//         let max_k = input.search.maximum_recall_k();
//         let groundtruth =
//             datafiles::load_groundtruth(datafiles::BinFile(&input.search.groundtruth), Some(max_k))?;
//
//         writeln!(output, "Loaded {} queries", queries.nrows())?;
//
//         let knn = benchmark_core::search::graph::KNN::new(
//             index,
//             queries,
//             benchmark_core::search::graph::Strategy::broadcast(Strategy),
//         )?;
//
//         // let num_threads = NonZeroUsize::new(input.num_threads).unwrap();
//         let mut results = Vec::new();
//         for num_threads in input.search.num_threads.iter() {
//             let params = graph::search::Knn::new(input.search_n, search_l, None)?;
//
//             let setup = core_search::Setup {
//                 threads: num_threads,
//                 tasks: num_threads,
//                 reps: input.reps,
//             };
//
//             let run = core_search::Run::new(params, setup);
//
//             let summaries = core_search::search_all(
//                 knn.clone(),
//                 std::iter::once(run),
//                 benchmark_core::search::graph::knn::Aggregator::new(
//                     &groundtruth,
//                     input.recall_k,
//                     input.search_n,
//                     GroundTruthMode::Fixed,
//                 ),
//             )?;
//
//             for summary in &summaries {
//                 let qps: Vec<f64> = summary
//                     .end_to_end_latencies
//                     .iter()
//                     .map(|lat| summary.recall.num_queries as f64 / lat.as_seconds())
//                     .collect();
//                 let max_qps = qps.iter().cloned().fold(0.0f64, f64::max);
//                 let mean_qps = qps.iter().sum::<f64>() / qps.len() as f64;
//
//                 writeln!(
//                     output,
//                     "  L={:<4} recall={:.4}  QPS={:.0} (max {:.0})  cmps={:.1}  hops={:.1}",
//                     search_l,
//                     summary.recall.average,
//                     mean_qps,
//                     max_qps,
//                     summary.mean_cmps,
//                     summary.mean_hops,
//                 )?;
//             }
//         }
//
//         Ok(())
//     }
// }

///////////////
// Streaming //
///////////////

/// Input for the streaming inmem2 benchmark.
///
/// Drives the inmem2 provider through a BigANN-style runbook. Because the inmem2
/// provider already does external↔internal id translation, no `Managed`/
/// `TagSlotManager` adapter is needed — the runbook talks to the provider directly.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Inmem2StreamInput {
    /// Full dataset that the runbook indexes into.
    data: InputFile,
    /// Query set used for every search stage.
    queries: InputFile,

    /// Runbook parameters (path, dataset name, gt directory, ...).
    runbook_params: DynamicRunbookParams,

    max_degree: usize,
    l_build: usize,
    alpha: f32,

    search_n: usize,
    search_l: Vec<usize>,
    recall_k: usize,

    num_threads: usize,
    reps: NonZeroUsize,
}

impl Input for Inmem2StreamInput {
    type Raw = Inmem2StreamInput;

    fn tag() -> &'static str {
        "inmem2-stream"
    }

    fn from_raw(mut raw: Self::Raw, checker: &mut Checker) -> anyhow::Result<Self> {
        raw.data.resolve(checker)?;
        raw.queries.resolve(checker)?;
        raw.runbook_params.validate(checker)?;
        Ok(raw)
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    fn example() -> Self {
        Self {
            data: InputFile::new("path/to/base.bin"),
            queries: InputFile::new("path/to/query.bin"),
            runbook_params: <DynamicRunbookParams as crate::inputs::Example>::example(),
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

impl std::fmt::Display for Inmem2StreamInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "inmem2 f32 streaming benchmark")?;
        writeln!(
            f,
            "  runbook: {}",
            self.runbook_params.runbook_path.display()
        )?;
        writeln!(f, "  dataset: {}", self.runbook_params.dataset_name)?;
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

#[derive(Debug)]
struct Inmem2Stream;

impl Benchmark for Inmem2Stream {
    type Input = Inmem2StreamInput;
    type Output = Vec<StreamOutput>;

    fn try_match(&self, _input: &Inmem2StreamInput) -> Result<MatchScore, FailureScore> {
        Ok(MatchScore(0))
    }

    fn description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        input: Option<&Inmem2StreamInput>,
    ) -> std::fmt::Result {
        match input {
            Some(i) => write!(f, "{i}"),
            None => write!(f, "inmem2 f32 streaming benchmark"),
        }
    }

    fn run(
        &self,
        input: &Inmem2StreamInput,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        writeln!(output, "{input}")?;

        // Load the runbook so we know the eventual capacity.
        let gt_dir = input
            .runbook_params
            .resolved_gt_directory
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("groundtruth directory not resolved"))?;

        let runbook = bigann::RunBook::load(
            &input.runbook_params.runbook_path,
            &input.runbook_params.dataset_name,
            &mut bigann::ScanDirectory::new(gt_dir)?,
        )?;
        let max_points = runbook.max_points();

        // Load the dataset (consumed by `WithData`) and queries.
        let dataset: Matrix<f32> = datafiles::load_dataset(datafiles::BinFile(&input.data))?;
        let queries: Arc<Matrix<f32>> =
            Arc::new(datafiles::load_dataset(datafiles::BinFile(&input.queries))?);
        let dim = dataset.ncols();

        writeln!(
            output,
            "Loaded dataset: {} points, dim={}",
            dataset.nrows(),
            dim
        )?;
        writeln!(output, "Loaded queries: {}", queries.nrows())?;
        writeln!(output, "Runbook max_points: {max_points}")?;

        // Compute the medoid of the dataset as the single start point.
        let start = StartPointStrategy::Medoid.compute(dataset.as_view())?;
        let metric = Metric::L2;
        let exact_max_degree = (input.max_degree as f32 * 1.3) as usize;
        let layer = Full::<f32>::new(dim, metric);

        let config = diskann_inmem::provider::Config::new(max_points, exact_max_degree);
        let provider = Provider::new(layer, config, start.row_iter())?;

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

        let num_threads = NonZeroUsize::new(input.num_threads.max(1)).unwrap();
        let runtime = benchmark_core::tokio::runtime(num_threads.get())?;

        // Build the inner stream and wrap it with `WithData`.
        let stream = Stream {
            index: index.clone(),
            runtime,
            ntasks: num_threads,
            search_n: input.search_n,
            search_l: input.search_l.clone(),
            recall_k: input.recall_k,
            reps: input.reps,
        };

        let max_k = input.recall_k;
        let queries_for_search = queries.clone();
        let mut layered = bigann::WithData::new(stream, dataset, queries_for_search, move |path| {
            Ok(Box::new(datafiles::load_groundtruth(
                datafiles::BinFile(path),
                Some(max_k),
            )?))
        });

        // Drive the runbook.
        let mut runbook = runbook;
        let mut results = Vec::new();
        let stages = runbook.len();
        let mut stage_idx = 1usize;

        runbook.run_with(&mut layered, |o: StreamOutput| -> anyhow::Result<()> {
            let banner = format!("Stage {} of {}: {}", stage_idx, stages, o.kind());
            write!(output, "{}", crate::utils::SmallBanner(&banner))?;
            writeln!(output, "{o}")?;
            stage_idx += 1;
            results.push(o);
            Ok(())
        })?;

        write!(
            output,
            "{}",
            crate::utils::SmallBanner("End of Run Summary")
        )?;
        let total_inserts: usize = results.iter().filter_map(|r| r.insert_count()).sum();
        let total_deletes: usize = results.iter().filter_map(|r| r.delete_count()).sum();
        let n_searches = results
            .iter()
            .filter(|r| matches!(r, StreamOutput::Search { .. }))
            .count();
        writeln!(
            output,
            "stages={stages} inserts={total_inserts} deletes={total_deletes} searches={n_searches}",
        )?;

        Ok(results)
    }
}

/////////////////
// Stream impl //
/////////////////

/// Inner streaming index over `inmem2`.
///
/// Implements `streaming::Stream<bigann::DataArgs<f32, u32>>` so it can be wrapped
/// by `bigann::WithData` and driven by `bigann::RunBook`. Replace and maintain are
/// not supported in v1; deletes are eager so no consolidation pass is needed.
struct Stream {
    index: Arc<DiskANNIndex<Provider<Full<f32>>>>,
    runtime: tokio::runtime::Runtime,
    ntasks: NonZeroUsize,
    search_n: usize,
    search_l: Vec<usize>,
    recall_k: usize,
    reps: NonZeroUsize,
}

#[derive(Debug, Serialize)]
pub(crate) enum StreamOutput {
    Insert { count: usize, latency_s: f64 },
    Delete { count: usize, latency_s: f64 },
    Search(Vec<SearchPoint>),
}

#[derive(Debug, Serialize)]
pub(crate) struct SearchPoint {
    pub search_l: usize,
    pub recall: f64,
    pub mean_qps: f64,
    pub max_qps: f64,
}

impl StreamOutput {
    fn kind(&self) -> &'static str {
        match self {
            Self::Insert { .. } => "insert",
            Self::Delete { .. } => "delete",
            Self::Search(_) => "search",
        }
    }

    fn insert_count(&self) -> Option<usize> {
        match self {
            Self::Insert { count, .. } => Some(*count),
            _ => None,
        }
    }

    fn delete_count(&self) -> Option<usize> {
        match self {
            Self::Delete { count, .. } => Some(*count),
            _ => None,
        }
    }
}

impl std::fmt::Display for StreamOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Insert { count, latency_s } => {
                writeln!(f, "  inserted {count} points in {latency_s:.3}s")
            }
            Self::Delete { count, latency_s } => {
                writeln!(f, "  deleted {count} points in {latency_s:.3}s")
            }
            Self::Search(points) => {
                for p in points {
                    writeln!(
                        f,
                        "  L={:<4} recall={:.4}  QPS={:.0} (max {:.0})",
                        p.search_l, p.recall, p.mean_qps, p.max_qps,
                    )?;
                }
                Ok(())
            }
        }
    }
}

impl streaming::Stream<bigann::DataArgs<f32, u32>> for Stream {
    type Output = StreamOutput;

    fn search(
        &mut self,
        (queries, groundtruth): (Arc<Matrix<f32>>, &dyn recall::Rows<u32>),
    ) -> anyhow::Result<Self::Output> {
        let knn = benchmark_core::search::graph::KNN::new(
            self.index.clone(),
            queries,
            benchmark_core::search::graph::Strategy::broadcast(Strategy),
        )?;

        let mut points = Vec::with_capacity(self.search_l.len());
        for &search_l in &self.search_l {
            let params = graph::search::Knn::new(self.search_n, search_l, None)?;
            let setup = core_search::Setup {
                threads: self.ntasks,
                tasks: self.ntasks,
                reps: self.reps,
            };
            let run = core_search::Run::new(params, setup);

            let summaries = core_search::search_all(
                knn.clone(),
                std::iter::once(run),
                benchmark_core::search::graph::knn::Aggregator::new(
                    groundtruth,
                    self.recall_k,
                    self.search_n,
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
                let mean_qps = qps.iter().sum::<f64>() / qps.len().max(1) as f64;
                points.push(SearchPoint {
                    search_l,
                    recall: summary.recall.average,
                    mean_qps,
                    max_qps,
                });
            }
        }
        Ok(StreamOutput::Search(points))
    }

    fn insert(
        &mut self,
        (data, ids): (MatrixView<'_, f32>, Range<usize>),
    ) -> anyhow::Result<Self::Output> {
        anyhow::ensure!(
            data.nrows() == ids.len(),
            "insert: data rows ({}) != ids range ({})",
            data.nrows(),
            ids.len(),
        );

        let count = data.nrows();
        let slots: Box<[u32]> = ids
            .map(|id| u32::try_from(id))
            .collect::<Result<Box<[u32]>, _>>()?;

        let runner = build_core::graph::SingleInsert::new(
            self.index.clone(),
            Arc::new(data.to_owned()),
            Strategy,
            build_core::ids::Slice::new(slots),
        );

        let results = build_core::build(
            runner,
            build_core::Parallelism::dynamic(diskann::utils::ONE, self.ntasks),
            &self.runtime,
        )?;

        let latency_s = results.end_to_end_latency().as_seconds();
        Ok(StreamOutput::Insert { count, latency_s })
    }

    fn delete(&mut self, ids: Range<usize>) -> anyhow::Result<Self::Output> {
        let count = ids.len();
        let provider = self.index.provider();
        let ctx = diskann_inmem::Context;

        let start = std::time::Instant::now();

        let runner = streaming::graph::InplaceDelete::new(
            self.index.clone(),
            Strategy,
            3,
            diskann::graph::InplaceDeleteMethod::OneHop,
            build_core::ids::Slice::new(ids.clone().into_iter().map(|i| i as u32).collect()),
        );

        let _ = build_core::build(
            runner,
            diskann_benchmark_core::build::Parallelism::fixed(
                Some(diskann::utils::ONE),
                self.ntasks,
            ),
            &self.runtime,
        )?;

        let latency_s = start.elapsed().as_secs_f64();

        Ok(StreamOutput::Delete { count, latency_s })
    }

    fn replace(
        &mut self,
        _args: (MatrixView<'_, f32>, Range<usize>),
    ) -> anyhow::Result<Self::Output> {
        anyhow::bail!("inmem2-f32-stream: replace is not supported in v1")
    }

    fn maintain(&mut self, _: ()) -> anyhow::Result<Self::Output> {
        anyhow::bail!(
            "inmem2-f32-stream: maintain is not supported (deletes are eager, no consolidation needed)"
        )
    }

    fn needs_maintenance(&mut self) -> bool {
        false
    }
}
