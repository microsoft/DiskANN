/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fmt::{self, Display, Formatter},
    io::Write,
    num::NonZeroUsize,
    ops::Range,
    sync::Arc,
};

use diskann::graph::{self, DiskANNIndex, InplaceDeleteMethod, StartPointStrategy};
use diskann_benchmark_core::{
    self as benchmark_core, build as build_core, recall,
    recall::GroundTruthMode,
    search as core_search,
    streaming::{self, executors::bigann, Executor},
};
use diskann_benchmark_runner::{
    benchmark::{MatchContext, Score},
    files::InputFile,
    output::Output,
    utils::{
        datatype::{AsDataType, DataType},
        fmt::{Delimit, KeyValue, Quote},
    },
    Benchmark, Checker, Checkpoint, Input, Registry,
};
use diskann_inmem::{
    layers::{Full, FullPrecision},
    Provider, Strategy,
};
use diskann_utils::views::{Matrix, MatrixView};
use diskann_vector::distance::Metric;
use serde::{Deserialize, Serialize};

use crate::{
    index::{
        build::{BuildKind, BuildStats, ProgressMeter},
        result::{AggregatedSearchResults, SearchResults},
        streaming::stats::{GenericStats, StreamStats, Summary},
    },
    utils::{datafiles, SimilarityMeasure},
};

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register("inmem2-f32", Build::<f32>::new())?;
    // registry.register("inmem2-f16", Build::<f16>::new())?;
    registry.register("inmem2-f32-stream", StreamingBenchmark::<f32>::new())?;
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

    //-----------//
    // Streaming //
    //-----------//

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct StreamingKnnSearch {
        pub(super) queries: InputFile,
        pub(super) reps: NonZeroUsize,
        pub(super) num_threads: NonZeroUsize,
        pub(super) runs: Vec<KnnSweep>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct RunBook {
        pub(super) path: InputFile,
        pub(super) dataset: String,
        pub(super) groundtruth_directory: String,
        pub(super) delete_method: crate::inputs::graph_index::InplaceDeleteMethod,
        pub(super) delete_num_to_replace: usize,
    }

    //------------------//
    // Top Level Inputs //
    //------------------//

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct StaticBuild {
        pub(super) data: Data,
        pub(super) build: BuildParams,
        pub(super) search: KnnSearch,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct BigANNStreaming {
        pub(super) data: Data,
        pub(super) build: BuildParams,
        pub(super) search: StreamingKnnSearch,
        pub(super) runbook: RunBook,
    }
}

#[derive(Debug, Clone)]
struct KnnInstance {
    knn: graph::search::Knn,
    recall_k: usize,
}

impl KnnInstance {
    fn flatten(runs: &[dto::KnnSweep]) -> anyhow::Result<Vec<Self>> {
        runs.iter()
            .flat_map(|sweep| {
                let search_n = sweep.search_n;
                let recall_k = sweep.recall_k;

                sweep
                    .search_l
                    .iter()
                    .map(move |search_l| -> anyhow::Result<_> {
                        let knn = graph::search::Knn::new_default(search_n, *search_l)?;
                        Ok(KnnInstance { knn, recall_k })
                    })
            })
            .collect()
    }
}

impl Display for KnnInstance {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "knn = {}, search_l = {}, beam_width = {}",
            self.recall_k,
            self.knn.l_value(),
            self.knn.beam_width(),
        )
    }
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

        Ok(Self {
            queries,
            groundtruth,
            reps,
            num_threads,
            runs: KnnInstance::flatten(&runs)?,
        })
    }

    fn maximum_recall_k(&self) -> usize {
        self.runs.iter().map(|r| r.recall_k).max().unwrap_or(0)
    }
}

impl Display for KnnSearch {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut kv = KeyValue::new();
        kv.push("queries", &self.queries);
        kv.push("groundtruth", &self.groundtruth);
        kv.push("reps", &self.reps);

        let num_threads = Delimit::new(self.num_threads.iter(), ", ");
        kv.push("num_threads", &num_threads);

        let runs = Delimit::new(self.runs.iter(), "\n").to_string();
        kv.push("runs", &runs);
        write!(f, "{}", kv)
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

impl Display for Data {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut kv = KeyValue::new();
        kv.push("data_type", &self.data_type);
        kv.push("data", &self.data);
        kv.push("distance", &self.distance);
        write!(f, "{}", kv)
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

impl Display for BuildParams {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut kv = KeyValue::new();

        let pruned_degree = self.config.pruned_degree();
        let max_degree = self.config.max_degree();
        let alpha = self.config.alpha();
        let l_build = self.config.l_build();

        kv.push("pruned_degree", &pruned_degree);
        kv.push("max_degree", &max_degree);
        kv.push("alpha", &alpha);
        kv.push("l_build", &l_build);
        kv.push("num_threads", &self.num_threads);
        write!(f, "{}", kv)
    }
}

#[derive(Debug)]
struct StaticBuild {
    data: Data,
    build: BuildParams,
    search: KnnSearch,
    // The serialized representation of the original input.
    input: serde_json::Value,
}

impl StaticBuild {
    fn from_raw(raw: dto::StaticBuild, mut checker: Option<&mut Checker>) -> anyhow::Result<Self> {
        let input = serde_json::to_value(&raw)?;

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
            input,
        })
    }
}

impl Display for StaticBuild {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut kv = KeyValue::new();
        kv.push("data", &self.data);
        kv.push("build", &self.build);
        kv.push("search", &self.search);

        write!(f, "{}", kv)
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
        Ok(self.input.clone())
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

impl<T> Benchmark for Build<T>
where
    T: diskann_inmem::layers::FullPrecision + diskann::graph::SampleableForStart + AsDataType,
{
    type Input = StaticBuild;
    type Output = ();

    fn try_match(&self, input: &StaticBuild, context: &MatchContext) -> Score {
        let mut score = context.success(0);

        let data_type = input.data.data_type;
        if !T::is_match(data_type) {
            score.fail(
                1000,
                &format_args!(
                    "expected data-type {}, instead got {}",
                    Quote(T::DATA_TYPE),
                    Quote(data_type)
                ),
            )
        }

        score
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "full-precision static build+search with data type {}",
            Quote(T::DATA_TYPE)
        )?;

        Ok(())
    }

    fn run(
        &self,
        input: &StaticBuild,
        checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<()> {
        writeln!(output, "{input}\n")?;

        // Load data.
        let data: Arc<Matrix<T>> = Arc::new(datafiles::load_dataset(datafiles::BinFile(
            &input.data.data,
        ))?);

        let dim = data.ncols();
        let num_points = data.nrows();
        writeln!(output, "Loaded {num_points} points, dim={dim}")?;

        // Compute the medoid of the dataset as the single start point.
        let start = StartPointStrategy::Medoid.compute(data.as_view())?;
        let layer = Full::<T>::new(dim, input.data.distance);
        let config =
            diskann_inmem::provider::Config::new(num_points, input.build.config.max_degree().get());
        let provider = Provider::<_, u32>::new(layer, config, start.row_iter())?;

        let index = Arc::new(DiskANNIndex::new(
            input.build.config.clone(),
            provider,
            None,
        ));

        // Build via SingleInsert.
        let rt = benchmark_core::tokio::runtime(input.build.num_threads.get())?;
        let builder = build_core::graph::SingleInsert::new(
            index.clone(),
            data,
            Strategy,
            build_core::ids::Identity::<u32>::new(),
        );

        let build_results = build_core::build_tracked(
            builder,
            build_core::Parallelism::dynamic(diskann::utils::ONE, input.build.num_threads),
            &rt,
            Some(&ProgressMeter::new(output)),
        )?;

        let total_build_time = build_results.end_to_end_latency();
        writeln!(
            output,
            "\nBuild complete in {:.2}s",
            total_build_time.as_seconds()
        )?;
        checkpoint.checkpoint(&total_build_time)?;

        // Search.
        let queries: Arc<Matrix<T>> = Arc::new(datafiles::load_dataset(datafiles::BinFile(
            &input.search.queries,
        ))?);
        let max_k = input.search.maximum_recall_k();
        let groundtruth = datafiles::load_groundtruth(
            datafiles::BinFile(&input.search.groundtruth),
            Some(max_k),
        )?;

        writeln!(output, "Loaded {} queries\n", queries.nrows())?;

        let knn = benchmark_core::search::graph::KNN::new(
            index,
            queries,
            benchmark_core::search::graph::Strategy::broadcast(Strategy),
        )?;

        let results = _knn(
            &knn,
            &groundtruth,
            input.search.reps,
            &input.search.num_threads,
            &input.search.runs,
        )?;

        let results = AggregatedSearchResults::Topk(results);

        writeln!(output, "{}", results)?;

        Ok(())
    }
}

fn _knn(
    runner: &dyn crate::index::search::knn::Knn<u32>,
    groundtruth: &dyn benchmark_core::recall::Rows<u32>,
    reps: NonZeroUsize,
    num_threads: &[NonZeroUsize],
    instances: &[KnnInstance],
) -> anyhow::Result<Vec<SearchResults>> {
    let mut results = Vec::new();

    for num_threads in num_threads.iter() {
        for instance in instances.iter() {
            let setup = core_search::Setup {
                threads: *num_threads,
                tasks: *num_threads,
                reps,
            };

            let run = core_search::Run::new(instance.knn, setup);

            let r = runner.search_all(
                vec![run],
                groundtruth,
                instance.recall_k,
                instance.knn.k_value().get(),
                GroundTruthMode::Fixed,
            )?;

            results.extend(r);
        }
    }

    Ok(results)
}

///////////////
// Streaming //
///////////////

#[derive(Debug, Clone)]
struct StreamingKnnSearch {
    queries: InputFile,
    reps: NonZeroUsize,
    num_threads: NonZeroUsize,
    runs: Vec<KnnInstance>,
}

impl StreamingKnnSearch {
    fn from_raw(
        raw: dto::StreamingKnnSearch,
        checker: Option<&mut Checker>,
    ) -> anyhow::Result<Self> {
        let dto::StreamingKnnSearch {
            mut queries,
            reps,
            num_threads,
            runs,
        } = raw;

        if let Some(checker) = checker {
            queries.resolve(checker)?;
        }

        Ok(Self {
            queries,
            reps,
            num_threads,
            runs: KnnInstance::flatten(&runs)?,
        })
    }
}

impl Display for StreamingKnnSearch {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut kv = KeyValue::new();
        kv.push("queries", &self.queries);
        kv.push("reps", &self.reps);
        kv.push("num_threads", &self.num_threads);

        let runs = Delimit::new(self.runs.iter(), "\n");
        kv.push("runs", &runs);
        write!(f, "{}", runs)
    }
}

#[derive(Debug)]
struct RunBook {
    runbook: bigann::RunBook,
    delete_method: InplaceDeleteMethod,
    delete_num_to_replace: usize,
    // This is kept for display purposes.
    runbook_path: InputFile,
    dataset: String,
}

impl RunBook {
    fn from_raw(raw: dto::RunBook, checker: &mut Checker) -> anyhow::Result<Self> {
        let dto::RunBook {
            mut path,
            dataset,
            groundtruth_directory,
            delete_method,
            delete_num_to_replace,
        } = raw;

        path.resolve(checker)?;

        let groundtruth_directory = checker.find_input_dir(groundtruth_directory.as_ref())?;

        let runbook = bigann::RunBook::load(
            &path,
            &dataset,
            &mut bigann::ScanDirectory::new(&groundtruth_directory)?,
        )?;

        Ok(Self {
            runbook,
            delete_method: delete_method.into(),
            delete_num_to_replace,
            runbook_path: path,
            dataset,
        })
    }
}

impl Display for RunBook {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut kv = KeyValue::new();
        let path = self.runbook_path.display();
        kv.push("runbook", &path);
        kv.push("dataset", &self.dataset);

        let max_points = self.runbook.max_points();
        let max_tag = self.runbook.max_tag();
        let num_stages = self.runbook.len();

        kv.push("num_stages", &num_stages);
        kv.push("max_active_points", &max_points);
        if let Some(ref max_tag) = max_tag {
            kv.push("max_tag", max_tag);
        }

        kv.push_eager("delete_method", format_args!("{:?}", self.delete_method));
        kv.push("delete_num_to_replace", &self.delete_num_to_replace);
        write!(f, "{}", kv)
    }
}

#[derive(Debug)]
struct BigANNStreaming {
    data: Data,
    build: BuildParams,
    search: StreamingKnnSearch,
    runbook: RunBook,
    // The serialized representation of the original input.
    input: serde_json::Value,
}

impl BigANNStreaming {
    fn from_raw(raw: dto::BigANNStreaming, checker: &mut Checker) -> anyhow::Result<Self> {
        let input = serde_json::to_value(&raw)?;
        let data = Data::from_raw(raw.data, Some(checker))?;
        let build = BuildParams::from_raw(raw.build, data.distance)?;
        Ok(Self {
            data,
            build,
            search: StreamingKnnSearch::from_raw(raw.search, Some(checker))?,
            runbook: RunBook::from_raw(raw.runbook, checker)?,
            input,
        })
    }
}

impl Display for BigANNStreaming {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut kv = KeyValue::new();
        kv.push("data", &self.data);
        kv.push("build", &self.build);
        kv.push("search", &self.search);
        kv.push("runbook", &self.runbook);
        write!(f, "{}", kv)
    }
}

impl Input for BigANNStreaming {
    type Raw = dto::BigANNStreaming;

    fn tag() -> &'static str {
        "inmem2-streaming"
    }

    fn from_raw(raw: Self::Raw, checker: &mut Checker) -> anyhow::Result<Self> {
        Self::from_raw(raw, checker)
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(self.input.clone())
    }

    fn example() -> Self::Raw {
        const FOUR: NonZeroUsize = NonZeroUsize::new(4).unwrap();
        const THREE: NonZeroUsize = NonZeroUsize::new(3).unwrap();

        dto::BigANNStreaming {
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
            search: dto::StreamingKnnSearch {
                queries: InputFile::new("path/to/queries"),
                reps: THREE,
                num_threads: FOUR,
                runs: vec![dto::KnnSweep {
                    search_n: 10,
                    search_l: vec![10, 20, 30, 40, 50],
                    recall_k: 10,
                }],
            },
            runbook: dto::RunBook {
                path: InputFile::new("path/to/runbook.yaml"),
                dataset: "dataset-1M".into(),
                groundtruth_directory: "groundtruth/dir".into(),
                delete_method: crate::inputs::graph_index::InplaceDeleteMethod::TwoHopAndOneHop,
                delete_num_to_replace: 3,
            },
        }
    }
}

#[derive(Debug)]
struct StreamingBenchmark<T>(std::marker::PhantomData<T>);

impl<T> StreamingBenchmark<T> {
    fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<T> Benchmark for StreamingBenchmark<T>
where
    T: FullPrecision + AsDataType + diskann::graph::SampleableForStart,
{
    type Input = BigANNStreaming;
    type Output = Vec<StreamStats>;

    fn try_match(&self, input: &BigANNStreaming, context: &MatchContext) -> Score {
        let mut score = context.success(0);
        let data_type = input.data.data_type;
        if !T::is_match(data_type) {
            score.fail(
                1000,
                &format_args!(
                    "expected data-type {}, instead got {}",
                    Quote(T::DATA_TYPE),
                    Quote(data_type)
                ),
            );
        }

        score
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "full-precision streaming with data type {}",
            Quote(T::DATA_TYPE)
        )?;

        Ok(())
    }

    fn run(
        &self,
        input: &BigANNStreaming,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        writeln!(output, "{input}\n")?;

        // Load the runbook so we know the eventual capacity.
        let runbook = input.runbook.runbook.clone();
        let max_points = runbook.max_points();

        // Load the dataset (consumed by `WithData`) and queries.
        let dataset: Matrix<T> = datafiles::load_dataset(datafiles::BinFile(&input.data.data))?;
        let queries: Arc<Matrix<T>> = Arc::new(datafiles::load_dataset(datafiles::BinFile(
            &input.search.queries,
        ))?);
        let dim = dataset.ncols();

        // Compute the medoid of the dataset as the single start point.
        let start = StartPointStrategy::Medoid.compute(dataset.as_view())?;
        let index_config = input.build.config.clone();
        let layer = Full::<T>::new(dim, input.data.distance);

        let config =
            diskann_inmem::provider::Config::new(max_points, index_config.max_degree().get());
        let provider = Provider::<_, u32>::new(layer, config, start.row_iter())?;

        let index = Arc::new(DiskANNIndex::new(index_config, provider, None));

        let num_threads = input.build.num_threads;
        let runtime = benchmark_core::tokio::runtime(num_threads.get())?;

        let stream = Stream {
            index,
            runtime,
            search: input.search.clone(),
            ntasks: input.build.num_threads,
            delete_method: input.runbook.delete_method,
            delete_num_to_replace: input.runbook.delete_num_to_replace,
        };

        let mut layered = bigann::WithData::new(stream, dataset, queries, move |path| {
            Ok(Box::new(datafiles::load_groundtruth(
                datafiles::BinFile(path),
                None,
            )?))
        });

        // Here we go!
        let mut results = Vec::new();
        let stages = runbook.len();
        let mut i = 1;
        input.runbook.runbook.clone().run_with(
            &mut layered,
            |o: StreamStats| -> anyhow::Result<()> {
                if o.is_maintain() {
                    let message = format!("Ran maintenance before stage {}", i);
                    write!(output, "{}", crate::utils::SmallBanner(&message))?;
                } else {
                    let message = format!("Finished stage {} of {}: {}", i, stages, o.kind());
                    write!(output, "{}", crate::utils::SmallBanner(&message))?;
                    i += 1;
                }
                writeln!(output, "{}", o)?;
                results.push(o);
                Ok(())
            },
        )?;

        write!(
            output,
            "{}",
            crate::utils::SmallBanner("End of Run Summary")
        )?;

        writeln!(output, "{}", Summary::new(results.iter()))?;

        Ok(results)
    }
}

////////////
// Stream //
////////////

struct Stream<T>
where
    T: FullPrecision,
{
    index: Arc<DiskANNIndex<Provider<Full<T>>>>,
    runtime: tokio::runtime::Runtime,
    search: StreamingKnnSearch,
    ntasks: NonZeroUsize,
    delete_method: InplaceDeleteMethod,
    delete_num_to_replace: usize,
}

impl<T> Stream<T>
where
    T: FullPrecision,
{
    fn insert_(
        &mut self,
        data: MatrixView<'_, T>,
        ids: Range<usize>,
    ) -> anyhow::Result<BuildStats> {
        anyhow::ensure!(
            data.nrows() == ids.len(),
            "insert: data rows ({}) != ids range ({})",
            data.nrows(),
            ids.len(),
        );

        let runner = build_core::graph::SingleInsert::new(
            self.index.clone(),
            Arc::new(data.to_owned()),
            Strategy,
            build_core::ids::Range::<u32>::new(ids.start as u32..ids.end as u32),
        );

        let results = build_core::build(
            runner,
            build_core::Parallelism::dynamic(diskann::utils::ONE, self.ntasks),
            &self.runtime,
        )?;

        BuildStats::new(BuildKind::SingleInsert, results)
    }
}

impl<T> streaming::Stream<bigann::DataArgs<T, u32>> for Stream<T>
where
    T: FullPrecision,
{
    type Output = StreamStats;

    fn search(
        &mut self,
        (queries, groundtruth): (Arc<Matrix<T>>, &dyn recall::Rows<u32>),
    ) -> anyhow::Result<Self::Output> {
        let knn = benchmark_core::search::graph::KNN::new(
            self.index.clone(),
            queries,
            benchmark_core::search::graph::Strategy::broadcast(Strategy),
        )?;

        let r = _knn(
            &knn,
            groundtruth,
            self.search.reps,
            std::slice::from_ref(&self.search.num_threads),
            &self.search.runs,
        )?;

        Ok(StreamStats::Search(r))
    }

    fn insert(
        &mut self,
        (data, ids): (MatrixView<'_, T>, Range<usize>),
    ) -> anyhow::Result<Self::Output> {
        self.insert_(data, ids).map(StreamStats::Insert)
    }

    fn delete(&mut self, ids: Range<usize>) -> anyhow::Result<Self::Output> {
        let runner = streaming::graph::InplaceDelete::new(
            self.index.clone(),
            Strategy,
            self.delete_num_to_replace,
            self.delete_method,
            build_core::ids::Range::new(ids.start as u32..ids.end as u32),
        );

        let r = build_core::build(
            runner,
            diskann_benchmark_core::build::Parallelism::fixed(
                Some(diskann::utils::ONE),
                self.ntasks,
            ),
            &self.runtime,
        )?;

        Ok(StreamStats::Delete(GenericStats::new("delete".into(), r)?))
    }

    fn replace(
        &mut self,
        (data, ids): (MatrixView<'_, T>, Range<usize>),
    ) -> anyhow::Result<Self::Output> {
        use diskann::provider::Delete;

        // TODO: This is kind of a hack. It would be ideal to parallelize this.
        //
        // Also, this is *way* more expensive than it needs to be because each delete creates
        // and then destroys an EBR guard.
        let ctx = diskann_inmem::Context;
        for id in ids.clone() {
            self.runtime
                .block_on(self.index.provider().delete(&ctx, &(id as u32)))?;
        }

        self.insert_(data, ids).map(StreamStats::Replace)
    }

    fn maintain(&mut self, _: ()) -> anyhow::Result<Self::Output> {
        Ok(StreamStats::Maintain(vec![]))
    }

    fn needs_maintenance(&mut self) -> bool {
        false
    }
}
