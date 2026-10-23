/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fmt::{self, Display, Formatter},
    io::Write,
    sync::Arc,
};

use diskann_benchmark_core::streaming::{self, executors::bigann, Executor};
use diskann_benchmark_runner::{
    self as dbr,
    files::InputFile,
    utils::{
        datatype::{AsDataType, DataType},
        fmt::KeyValue,
    },
    Checker, Registry,
};
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::distance::Metric;

use crate::utils::SimilarityMeasure;

mod benchmark;

mod full;
mod product;
mod scalar;
mod spherical;

trait FromRaw: Sized {
    type Raw;
    fn from_raw(raw: &Self::Raw) -> anyhow::Result<Self>;
}

trait FromRawWith: Sized {
    type Raw;
    fn from_raw_with(raw: &Self::Raw, checker: Option<&mut dbr::Checker>) -> anyhow::Result<Self>;
}

fn reborrow<'a, T>(x: &'a mut Option<&mut T>) -> Option<&'a mut T>
where
    T: ?Sized,
{
    match x.as_mut() {
        Some(x) => Some(x),
        None => None,
    }
}

#[track_caller]
fn check_data_type_load<T>(data_type: DataType) -> anyhow::Result<()>
where
    T: AsDataType,
{
    if !T::is_match(data_type) {
        Err(anyhow::anyhow!(
            "attempted to load {} data as {}",
            data_type,
            T::DATA_TYPE
        ))
    } else {
        Ok(())
    }
}

macro_rules! nz {
    ($ident:ident = $value:literal) => {
        const $ident: ::std::num::NonZeroUsize = ::std::num::NonZeroUsize::new($value).unwrap();
    };
}

use nz;

type Provider<T> = diskann_inmem::Provider<T, u32>;
type Index<T> = diskann::graph::DiskANNIndex<Provider<T>>;

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    full::register_benchmarks(registry)?;
    spherical::register_benchmarks(registry)?;
    scalar::register_benchmarks(registry)?;
    product::register_benchmarks(registry)?;
    Ok(())
}

///////////////////
// Shared Inputs //
///////////////////

mod dto {
    use super::*;

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct Data {
        pub(super) data_type: DataType,
        pub(super) data: InputFile,
        pub(super) distance: SimilarityMeasure,
    }

    impl Data {
        pub(super) fn example() -> Self {
            Self {
                data_type: DataType::Float32,
                data: InputFile::new("path/to/data"),
                distance: SimilarityMeasure::SquaredL2,
            }
        }
    }

    // //-----------//
    // // Streaming //
    // //-----------//

    // #[derive(Debug, Serialize, Deserialize)]
    // pub(super) struct StreamingKnnSearch {
    //     pub(super) queries: InputFile,
    //     pub(super) reps: NonZeroUsize,
    //     pub(super) num_threads: NonZeroUsize,
    //     pub(super) runs: Vec<KnnSweep>,
    // }

    // #[derive(Debug, Serialize, Deserialize)]
    // pub(super) struct RunBook {
    //     pub(super) path: InputFile,
    //     pub(super) dataset: String,
    //     pub(super) groundtruth_directory: String,
    //     pub(super) delete_method: crate::inputs::graph_index::InplaceDeleteMethod,
    //     pub(super) delete_num_to_replace: usize,
    // }

    // //------------------//
    // // Top Level Inputs //
    // //------------------//

    // #[derive(Debug, Serialize, Deserialize)]
    // pub(super) struct StaticBuild {
    //     pub(super) data: Data,
    //     pub(super) build: BuildParams,
    //     pub(super) search: KnnSearch,
    //     pub(super) quantization: Option<Quantization>,
    // }

    // #[derive(Debug, Serialize, Deserialize)]
    // pub(super) struct BigANNStreaming {
    //     pub(super) data: Data,
    //     pub(super) build: BuildParams,
    //     pub(super) search: StreamingKnnSearch,
    //     pub(super) runbook: RunBook,
    // }
}

#[derive(Debug)]
struct Data {
    data_type: DataType,
    data: InputFile,
    distance: Metric,
}

impl Data {
    fn new(data_type: DataType, data: InputFile, distance: Metric) -> Self {
        Self {
            data_type,
            data,
            distance,
        }
    }

    fn distance(&self) -> Metric {
        self.distance
    }

    fn data_type(&self) -> DataType {
        self.data_type
    }

    fn dispatch_params(&self) -> DataDispatchParams {
        DataDispatchParams {
            data_type: self.data_type,
        }
    }

    fn load<T>(&self, output: Option<&mut dyn dbr::Output>) -> anyhow::Result<rowmajor::Owned<T>>
    where
        T: AsDataType + bytemuck::Pod,
    {
        check_data_type_load::<T>(self.data_type)?;

        let data =
            crate::utils::datafiles::load_dataset(crate::utils::datafiles::BinFile(&self.data))?;

        if let Some(mut output) = output {
            writeln!(
                output,
                "Loaded {} points of dim {} as type {} from {}\n",
                data.nrows(),
                data.ncols(),
                T::DATA_TYPE,
                self.data.display(),
            )?;
        }

        Ok(data)
    }

    fn load_arc<T>(
        &self,
        output: Option<&mut dyn dbr::Output>,
    ) -> anyhow::Result<Arc<rowmajor::Owned<T>>>
    where
        T: AsDataType + bytemuck::Pod,
    {
        self.load(output).map(Arc::new)
    }
}

impl FromRawWith for Data {
    type Raw = dto::Data;
    fn from_raw_with(raw: &dto::Data, checker: Option<&mut Checker>) -> anyhow::Result<Self> {
        let dto::Data {
            data_type,
            data,
            distance,
        } = raw;

        let mut data = data.clone();
        if let Some(checker) = checker {
            data.resolve(checker)?;
        }

        Ok(Self::new(*data_type, data, (*distance).into()))
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

#[derive(Debug, Clone, Copy)]
struct DataDispatchParams {
    data_type: DataType,
}

// ///////////////
// // Streaming //
// ///////////////
//
// #[derive(Debug, Clone)]
// struct StreamingKnnSearch {
//     queries: InputFile,
//     reps: NonZeroUsize,
//     num_threads: NonZeroUsize,
//     runs: Vec<KnnInstance>,
// }
//
// impl StreamingKnnSearch {
//     fn from_raw(
//         raw: dto::StreamingKnnSearch,
//         checker: Option<&mut Checker>,
//     ) -> anyhow::Result<Self> {
//         let dto::StreamingKnnSearch {
//             mut queries,
//             reps,
//             num_threads,
//             runs,
//         } = raw;
//
//         if let Some(checker) = checker {
//             queries.resolve(checker)?;
//         }
//
//         Ok(Self {
//             queries,
//             reps,
//             num_threads,
//             runs: KnnInstance::flatten(&runs)?,
//         })
//     }
// }
//
// impl Display for StreamingKnnSearch {
//     fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
//         let mut kv = KeyValue::new();
//         kv.push("queries", &self.queries);
//         kv.push("reps", &self.reps);
//         kv.push("num_threads", &self.num_threads);
//
//         let runs = Delimit::new(self.runs.iter(), "\n");
//         kv.push("runs", &runs);
//         write!(f, "{}", runs)
//     }
// }
//
// #[derive(Debug)]
// struct RunBook {
//     runbook: bigann::RunBook,
//     delete_method: InplaceDeleteMethod,
//     delete_num_to_replace: usize,
//     // This is kept for display purposes.
//     runbook_path: InputFile,
//     dataset: String,
// }
//
// impl RunBook {
//     fn from_raw(raw: dto::RunBook, checker: &mut Checker) -> anyhow::Result<Self> {
//         let dto::RunBook {
//             mut path,
//             dataset,
//             groundtruth_directory,
//             delete_method,
//             delete_num_to_replace,
//         } = raw;
//
//         path.resolve(checker)?;
//
//         let groundtruth_directory = checker.find_input_dir(groundtruth_directory.as_ref())?;
//
//         let runbook = bigann::RunBook::load(
//             &path,
//             &dataset,
//             &mut bigann::ScanDirectory::new(&groundtruth_directory)?,
//         )?;
//
//         Ok(Self {
//             runbook,
//             delete_method: delete_method.into(),
//             delete_num_to_replace,
//             runbook_path: path,
//             dataset,
//         })
//     }
// }
//
// impl Display for RunBook {
//     fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
//         let mut kv = KeyValue::new();
//         let path = self.runbook_path.display();
//         kv.push("runbook", &path);
//         kv.push("dataset", &self.dataset);
//
//         let max_points = self.runbook.max_points();
//         let max_tag = self.runbook.max_tag();
//         let num_stages = self.runbook.len();
//
//         kv.push("num_stages", &num_stages);
//         kv.push("max_active_points", &max_points);
//         if let Some(ref max_tag) = max_tag {
//             kv.push("max_tag", max_tag);
//         }
//
//         kv.push_eager("delete_method", format_args!("{:?}", self.delete_method));
//         kv.push("delete_num_to_replace", &self.delete_num_to_replace);
//         write!(f, "{}", kv)
//     }
// }
//
// #[derive(Debug)]
// struct BigANNStreaming {
//     data: Data,
//     build: BuildParams,
//     search: StreamingKnnSearch,
//     runbook: RunBook,
//     // The serialized representation of the original input.
//     input: serde_json::Value,
// }
//
// impl BigANNStreaming {
//     fn from_raw(raw: dto::BigANNStreaming, checker: &mut Checker) -> anyhow::Result<Self> {
//         let input = serde_json::to_value(&raw)?;
//         let data = Data::from_raw(raw.data, Some(checker))?;
//         let build = BuildParams::from_raw(raw.build, data.distance)?;
//         Ok(Self {
//             data,
//             build,
//             search: StreamingKnnSearch::from_raw(raw.search, Some(checker))?,
//             runbook: RunBook::from_raw(raw.runbook, checker)?,
//             input,
//         })
//     }
// }
//
// impl Display for BigANNStreaming {
//     fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
//         let mut kv = KeyValue::new();
//         kv.push("data", &self.data);
//         kv.push("build", &self.build);
//         kv.push("search", &self.search);
//         kv.push("runbook", &self.runbook);
//         write!(f, "{}", kv)
//     }
// }
//
// impl Input for BigANNStreaming {
//     type Raw = dto::BigANNStreaming;
//
//     fn tag() -> &'static str {
//         "inmem2-streaming"
//     }
//
//     fn from_raw(raw: Self::Raw, checker: &mut Checker) -> anyhow::Result<Self> {
//         Self::from_raw(raw, checker)
//     }
//
//     fn serialize(&self) -> anyhow::Result<serde_json::Value> {
//         Ok(self.input.clone())
//     }
//
//     fn example() -> Self::Raw {
//         const FOUR: NonZeroUsize = NonZeroUsize::new(4).unwrap();
//         const THREE: NonZeroUsize = NonZeroUsize::new(3).unwrap();
//
//         dto::BigANNStreaming {
//             data: dto::Data {
//                 data_type: DataType::Float32,
//                 data: InputFile::new("path/to/data"),
//                 distance: SimilarityMeasure::SquaredL2,
//             },
//             build: dto::BuildParams {
//                 pruned_degree: 28,
//                 max_degree: 32,
//                 l_build: 100,
//                 alpha: 1.2,
//                 num_threads: FOUR,
//             },
//             search: dto::StreamingKnnSearch {
//                 queries: InputFile::new("path/to/queries"),
//                 reps: THREE,
//                 num_threads: FOUR,
//                 runs: vec![dto::KnnSweep {
//                     search_n: 10,
//                     search_l: vec![10, 20, 30, 40, 50],
//                     recall_k: 10,
//                 }],
//             },
//             runbook: dto::RunBook {
//                 path: InputFile::new("path/to/runbook.yaml"),
//                 dataset: "dataset-1M".into(),
//                 groundtruth_directory: "groundtruth/dir".into(),
//                 delete_method: crate::inputs::graph_index::InplaceDeleteMethod::TwoHopAndOneHop,
//                 delete_num_to_replace: 3,
//             },
//         }
//     }
// }
//
// #[derive(Debug)]
// struct StreamingBenchmark<T>(std::marker::PhantomData<T>);
//
// impl<T> StreamingBenchmark<T> {
//     fn new() -> Self {
//         Self(std::marker::PhantomData)
//     }
// }
//
// impl<T> Benchmark for StreamingBenchmark<T>
// where
//     T: FullPrecision + AsDataType + diskann::graph::SampleableForStart,
// {
//     type Input = BigANNStreaming;
//     type Output = Vec<StreamStats>;
//
//     fn try_match(&self, input: &BigANNStreaming, context: &MatchContext) -> Score {
//         let mut score = context.success(0);
//         let data_type = input.data.data_type;
//         if !T::is_match(data_type) {
//             score.fail(
//                 1000,
//                 &format_args!(
//                     "expected data-type {}, instead got {}",
//                     Quote(T::DATA_TYPE),
//                     Quote(data_type)
//                 ),
//             );
//         }
//
//         score
//     }
//
//     fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(
//             f,
//             "full-precision streaming with data type {}",
//             Quote(T::DATA_TYPE)
//         )?;
//
//         Ok(())
//     }
//
//     fn run(
//         &self,
//         input: &BigANNStreaming,
//         _checkpoint: Checkpoint<'_>,
//         mut output: &mut dyn Output,
//     ) -> anyhow::Result<Self::Output> {
//         writeln!(output, "{input}\n")?;
//
//         // Load the runbook so we know the eventual capacity.
//         let runbook = input.runbook.runbook.clone();
//         let max_points = runbook.max_points();
//
//         // Load the dataset (consumed by `WithData`) and queries.
//         let dataset: rowmajor::Owned<T> =
//             datafiles::load_dataset(datafiles::BinFile(&input.data.data))?;
//         let queries: Arc<rowmajor::Owned<T>> = Arc::new(datafiles::load_dataset(
//             datafiles::BinFile(&input.search.queries),
//         )?);
//
//         // Compute the medoid of the dataset as the single start point.
//         let start = StartPointStrategy::Medoid.compute(dataset.as_view())?;
//         let index_config = input.build.config.clone();
//         let config = Full::config(
//             Capacity::new(max_points),
//             MaxDegree::new(index_config.max_degree().get()),
//             input.data.distance,
//             start,
//         )?;
//
//         let provider = Provider::<_, u32>::new(config)?;
//         let index = Arc::new(DiskANNIndex::new(index_config, provider, None));
//
//         let num_threads = input.build.num_threads;
//         let runtime = benchmark_core::tokio::runtime(num_threads.get())?;
//
//         let stream = Stream {
//             index,
//             runtime,
//             search: input.search.clone(),
//             ntasks: input.build.num_threads,
//             delete_method: input.runbook.delete_method,
//             delete_num_to_replace: input.runbook.delete_num_to_replace,
//         };
//
//         let mut layered = bigann::WithData::new(stream, dataset, queries, move |path| {
//             Ok(Box::new(datafiles::load_groundtruth(
//                 datafiles::BinFile(path),
//                 None,
//             )?))
//         });
//
//         // Here we go!
//         let mut results = Vec::new();
//         let stages = runbook.len();
//         let mut i = 1;
//         input.runbook.runbook.clone().run_with(
//             &mut layered,
//             |o: StreamStats| -> anyhow::Result<()> {
//                 if o.is_maintain() {
//                     let message = format!("Ran maintenance before stage {}", i);
//                     write!(output, "{}", crate::utils::SmallBanner(&message))?;
//                 } else {
//                     let message = format!("Finished stage {} of {}: {}", i, stages, o.kind());
//                     write!(output, "{}", crate::utils::SmallBanner(&message))?;
//                     i += 1;
//                 }
//                 writeln!(output, "{}", o)?;
//                 results.push(o);
//                 Ok(())
//             },
//         )?;
//
//         write!(
//             output,
//             "{}",
//             crate::utils::SmallBanner("End of Run Summary")
//         )?;
//
//         writeln!(output, "{}", Summary::new(results.iter()))?;
//
//         Ok(results)
//     }
// }
//
// ////////////
// // Stream //
// ////////////
//
// struct Stream<T>
// where
//     T: FullPrecision,
// {
//     index: Arc<DiskANNIndex<Provider<Full<T>>>>,
//     runtime: tokio::runtime::Runtime,
//     search: StreamingKnnSearch,
//     ntasks: NonZeroUsize,
//     delete_method: InplaceDeleteMethod,
//     delete_num_to_replace: usize,
// }
//
// impl<T> Stream<T>
// where
//     T: FullPrecision,
// {
//     fn insert_(
//         &mut self,
//         data: rowmajor::Ref<'_, T>,
//         ids: Range<usize>,
//     ) -> anyhow::Result<BuildStats> {
//         anyhow::ensure!(
//             data.nrows() == ids.len(),
//             "insert: data rows ({}) != ids range ({})",
//             data.nrows(),
//             ids.len(),
//         );
//
//         let runner = build_core::graph::SingleInsert::new(
//             self.index.clone(),
//             Arc::new(data.to_rowmajor_owned()),
//             Strategy,
//             build_core::ids::Range::<u32>::new(ids.start as u32..ids.end as u32),
//         );
//
//         let results = build_core::build(
//             runner,
//             build_core::Parallelism::dynamic(diskann::utils::ONE, self.ntasks),
//             &self.runtime,
//         )?;
//
//         BuildStats::new(BuildKind::SingleInsert, results)
//     }
// }
//
// impl<T> streaming::Stream<bigann::DataArgs<T, u32>> for Stream<T>
// where
//     T: FullPrecision,
// {
//     type Output = StreamStats;
//
//     fn search(
//         &mut self,
//         (queries, groundtruth): (Arc<rowmajor::Owned<T>>, &dyn recall::Rows<u32>),
//     ) -> anyhow::Result<Self::Output> {
//         let knn = benchmark_core::search::graph::KNN::new(
//             self.index.clone(),
//             queries,
//             benchmark_core::search::graph::Strategy::broadcast(Strategy),
//         )?;
//
//         let r = _knn(
//             &knn,
//             groundtruth,
//             self.search.reps,
//             std::slice::from_ref(&self.search.num_threads),
//             &self.search.runs,
//         )?;
//
//         Ok(StreamStats::Search(r))
//     }
//
//     fn insert(
//         &mut self,
//         (data, ids): (rowmajor::Ref<'_, T>, Range<usize>),
//     ) -> anyhow::Result<Self::Output> {
//         self.insert_(data, ids).map(StreamStats::Insert)
//     }
//
//     fn delete(&mut self, ids: Range<usize>) -> anyhow::Result<Self::Output> {
//         let runner = streaming::graph::InplaceDelete::new(
//             self.index.clone(),
//             Strategy,
//             self.delete_num_to_replace,
//             self.delete_method,
//             build_core::ids::Range::new(ids.start as u32..ids.end as u32),
//         );
//
//         let r = build_core::build(
//             runner,
//             diskann_benchmark_core::build::Parallelism::fixed(
//                 Some(diskann::utils::ONE),
//                 self.ntasks,
//             ),
//             &self.runtime,
//         )?;
//
//         Ok(StreamStats::Delete(GenericStats::new("delete".into(), r)?))
//     }
//
//     fn replace(
//         &mut self,
//         (data, ids): (rowmajor::Ref<'_, T>, Range<usize>),
//     ) -> anyhow::Result<Self::Output> {
//         use diskann::provider::Delete;
//
//         // TODO: This is kind of a hack. It would be ideal to parallelize this.
//         //
//         // Also, this is *way* more expensive than it needs to be because each delete creates
//         // and then destroys an EBR guard.
//         let ctx = inmem::Context;
//         for id in ids.clone() {
//             self.runtime
//                 .block_on(self.index.provider().delete(&ctx, &(id as u32)))?;
//         }
//
//         self.insert_(data, ids).map(StreamStats::Replace)
//     }
//
//     fn maintain(&mut self, _: ()) -> anyhow::Result<Self::Output> {
//         Ok(StreamStats::Maintain(vec![]))
//     }
//
//     fn needs_maintenance(&mut self) -> bool {
//         false
//     }
// }
