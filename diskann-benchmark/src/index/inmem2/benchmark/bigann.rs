/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    future::Future,
    io::Write,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::Arc,
};

use diskann::{utils::IntoUsize, ANNResult};
use diskann_benchmark_core::{
    self as dbc,
    streaming::{executors::bigann, Executor},
};
use diskann_benchmark_runner::{
    self as dbr,
    files::InputFile,
    utils::{
        datatype::{AsDataType, DataType},
        MicroSeconds,
    },
};
use diskann_inmem::{self as di};
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::{self as dv};

use crate::{
    index::inmem2::{self, nz, reborrow, FromRaw, FromRawWith, Provider, check_data_type_load},
    utils::SimilarityMeasure,
};

pub(in crate::index::inmem2) mod dto {
    use super::*;

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct RunBook {
        pub(super) runbook: InputFile,
        pub(super) dataset: String,
        pub(super) data: InputFile,
        pub(super) data_type: DataType,
        pub(super) distance: SimilarityMeasure,
        pub(super) queries: InputFile,
        pub(super) groundtruth: String,
    }

    impl RunBook {
        fn example() -> Self {
            Self {
                runbook: InputFile::new("path/to/runbook.yaml"),
                dataset: String::from("name-of-dataset-in-runbook"),
                groundtruth: String::from("directory/for/groundtruth/files"),
                data: InputFile::new("path/to/data"),
                data_type: DataType::Float32,
                queries: InputFile::new("path/to/queries"),
                distance: SimilarityMeasure::SquaredL2,
            }
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(in crate::index::inmem2) struct Stream {
        pub(super) build: inmem2::benchmark::build::dto::Config,
        pub(super) search: inmem2::benchmark::knn::dto::StreamingKnnSearch,
        pub(super) runbook: RunBook,
        pub(super) delete_method: crate::inputs::graph_index::InplaceDeleteMethod,
        pub(super) delete_num_to_replace: NonZeroUsize,
    }

    impl Stream {
        pub(in crate::index::inmem2) fn example() -> Self {
            nz!(TO_REPLACE = 3);

            Self {
                build: inmem2::benchmark::build::dto::Config::example(),
                search: inmem2::benchmark::knn::dto::StreamingKnnSearch::example(),
                runbook: RunBook::example(),
                delete_method: crate::inputs::graph_index::InplaceDeleteMethod::TwoHopAndOneHop,
                delete_num_to_replace: TO_REPLACE,
            }
        }
    }
}

#[derive(Debug)]
struct RunBook {
    runbook: bigann::RunBook,
    data: inmem2::Data,
    queries: InputFile,

    // This is kept for display purposes.
    runbook_path: InputFile,
    dataset: String,
}

impl FromRawWith for RunBook {
    type Raw = dto::RunBook;

    fn from_raw_with(
        raw: &dto::RunBook,
        checker: Option<&mut dbr::Checker>,
    ) -> anyhow::Result<Self> {
        let dto::RunBook {
            runbook,
            dataset,
            data,
            data_type,
            distance,
            queries,
            groundtruth,
        } = raw;

        let mut runbook_path = runbook.clone();
        let mut data = data.clone();
        let mut queries = queries.clone();

        let runbook = if let Some(checker) = checker {
            runbook_path.resolve(checker)?;
            data.resolve(checker)?;
            queries.resolve(checker)?;

            let groundtruth = checker.find_input_dir(groundtruth.as_ref())?;

            bigann::RunBook::load(
                &runbook_path,
                &dataset,
                &mut bigann::ScanDirectory::new(&groundtruth)?,
            )?
        } else {
            struct Copy<'a>(&'a Path);
            impl bigann::FindGroundtruth for Copy<'_> {
                fn find_groundtruth(&mut self, stage: usize) -> anyhow::Result<PathBuf> {
                    Ok(self.0.to_owned())
                }
            }

            bigann::RunBook::load(&runbook_path, &dataset, &mut Copy(groundtruth.as_ref()))?
        };

        Ok(Self {
            runbook,
            data: inmem2::Data::new(*data_type, data, (*distance).into()),
            queries,
            runbook_path,
            dataset: dataset.clone(),
        })
    }
}

impl std::fmt::Display for RunBook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut kv = dbr::utils::fmt::KeyValue::new();
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

        write!(f, "{}", kv)
    }
}

#[derive(Debug)]
pub(in crate::index::inmem2) struct Stream {
    build: inmem2::benchmark::build::Config,
    search: inmem2::benchmark::knn::StreamingKnnSearch,
    runbook: RunBook,
    delete_method: diskann::graph::InplaceDeleteMethod,
    delete_num_to_replace: NonZeroUsize,
}

impl Stream {
    pub(in crate::index::inmem2) fn distance(&self) -> dv::distance::Metric {
        self.runbook.data.distance()
    }

    pub(in crate::index::inmem2) fn data_type(&self) -> DataType {
        self.runbook.data.data_type()
    }

    pub(in crate::index::inmem2) fn max_degree(&self) -> usize {
        self.build.max_degree()
    }

    pub(in crate::index::inmem2) fn load_data<T>(
        &self,
        output: Option<&mut dyn dbr::Output>,
    ) -> anyhow::Result<rowmajor::Owned<T>>
    where
        T: AsDataType + bytemuck::Pod,
    {
        self.runbook.data.load(output)
    }

    fn load_queries_arc<T>(
        &self,
        output: Option<&mut dyn dbr::Output>,
    ) -> anyhow::Result<Arc<rowmajor::Owned<T>>>
    where
        T: bytemuck::Pod,
    {
        crate::utils::datafiles::load_dataset(crate::utils::datafiles::BinFile(&self.runbook.queries)).map(Arc::new)
    }

    pub(in crate::index::inmem2) fn engine<R>(
        &self,
        provider: Provider<R>,
    ) -> anyhow::Result<Engine<'_, R>>
    where
        R: di::repr::Representation,
    {
        let build = self.build.engine(provider);
        let num_threads = self.build.num_threads();
        let runtime = dbc::tokio::runtime(num_threads.get())?;

        Ok(Engine {
            build,
            runtime,
            num_threads,
            search: &self.search,
            delete_method: self.delete_method,
            delete_num_to_replace: self.delete_num_to_replace,
        })
    }

    pub(in crate::index::inmem2) fn runner<T>(
        &self,
        data: rowmajor::Owned<T>,
    ) -> anyhow::Result<Runner<T>>
    where
        T: bytemuck::Pod,
    {
        let queries = self.load_queries_arc::<T>(None)?;
        Ok(Runner {
            data,
            queries,
            runbook: self.runbook.runbook.clone(),
        })
    }
}

impl FromRawWith for Stream {
    type Raw = dto::Stream;
    fn from_raw_with(
        raw: &dto::Stream,
        mut checker: Option<&mut dbr::Checker>,
    ) -> anyhow::Result<Self> {
        let dto::Stream {
            runbook,
            build,
            search,
            delete_method,
            delete_num_to_replace,
        } = raw;

        let runbook = RunBook::from_raw_with(runbook, checker)?;
        let build = inmem2::benchmark::build::Config::from_raw(build, runbook.data.distance)?;
        let search = inmem2::benchmark::knn::StreamingKnnSearch::from_raw(search)?;

        Ok(Self {
            runbook,
            build,
            search,
            delete_method: (*delete_method).into(),
            delete_num_to_replace: *delete_num_to_replace,
        })
    }
}

#[derive(Debug)]
pub(in crate::index::inmem2) struct Runner<T> {
    data: rowmajor::Owned<T>,
    queries: Arc<rowmajor::Owned<T>>,
    runbook: bigann::RunBook,
}

impl<T> Runner<T> {
    pub(in crate::index::inmem2) fn run(
        mut self,
        engine: Engine<'_, di::repr::Full<T>>,
        mut output: &mut dyn dbr::Output,
    ) -> anyhow::Result<()>
    where
        T: di::repr::FullPrecision,
    {
        let load_groundtruth = |path: &Path| -> anyhow::Result<Box<dyn dbc::recall::Rows<u32>>> {
            let gt = crate::utils::datafiles::load_groundtruth(
                crate::utils::datafiles::BinFile(path),
                None,
            )?;

            Ok(Box::new(gt))
        };

        let mut stream = bigann::WithData::new(engine, self.data, self.queries, load_groundtruth);

        let mut results = Vec::new();
        let stages = self.runbook.len();
        let mut i = 1;
        Executor::run_with(
            &mut self.runbook,
            &mut stream,
            |o: Step| -> anyhow::Result<()> {
                if !o.is_maintain() {
                    let message = format!("Finished stage {} of {}: {}", i, stages, o.kind());
                    write!(output, "{}", crate::utils::SmallBanner(&message))?;
                    writeln!(output, "{}", o)?;
                    i += 1;
                }
                results.push(o);
                Ok(())
            },
        )?;

        Ok(())
    }
}

#[derive(Debug)]
pub(in crate::index::inmem2) struct Engine<'a, R>
where
    R: di::repr::Representation,
{
    build: inmem2::benchmark::build::Engine<R>,
    runtime: tokio::runtime::Runtime,
    num_threads: NonZeroUsize,
    search: &'a inmem2::benchmark::knn::StreamingKnnSearch,
    delete_method: diskann::graph::InplaceDeleteMethod,
    delete_num_to_replace: NonZeroUsize,
}

impl<T> dbc::streaming::Stream<bigann::DataArgs<T, u32>> for Engine<'_, di::repr::Full<T>>
where
    T: di::repr::FullPrecision,
{
    type Output = Step;

    fn search(
        &mut self,
        (queries, groundtruth): (Arc<rowmajor::Owned<T>>, &dyn dbc::recall::Rows<u32>),
    ) -> anyhow::Result<Self::Output> {
        let stats = self
            .search
            .search(self.build.index().clone(), queries, groundtruth, None)?;

        Ok(Step::Search(stats))
    }

    fn insert(
        &mut self,
        (data, ids): (rowmajor::Ref<'_, T>, std::ops::Range<usize>),
    ) -> anyhow::Result<Self::Output> {
        let stats = self.build.build_for(
            Arc::new(data.to_rowmajor_owned()),
            (ids.start as u32)..(ids.end as u32),
            None,
        )?;

        Ok(Step::Insert(stats))
    }

    fn delete(&mut self, ids: std::ops::Range<usize>) -> anyhow::Result<Self::Output> {
        let runner = dbc::streaming::graph::InplaceDelete::new(
            self.build.index().clone(),
            di::Strategy,
            self.delete_num_to_replace.get(),
            self.delete_method,
            dbc::build::ids::Range::new(ids.start as u32..ids.end as u32),
        );

        let r = dbc::build::build(
            runner,
            diskann_benchmark_core::build::Parallelism::fixed(
                Some(diskann::utils::ONE),
                self.num_threads,
            ),
            &self.runtime,
        )?;

        Ok(Step::Delete)
    }

    fn replace(
        &mut self,
        (data, ids): (rowmajor::Ref<'_, T>, std::ops::Range<usize>),
    ) -> anyhow::Result<Self::Output> {
        use diskann::provider::Delete;

        // TODO: This is kind of a hack. It would be ideal to parallelize this.
        //
        // Also, this is *way* more expensive than it needs to be because each delete creates
        // and then destroys an EBR guard.
        for id in ids.clone() {
            self.runtime.block_on(
                self.build
                    .index()
                    .provider()
                    .delete(&di::Context, &(id as u32)),
            )?;
        }

        let stats = self.build.build_for(
            Arc::new(data.to_rowmajor_owned()),
            (ids.start as u32)..(ids.end as u32),
            None,
        )?;

        Ok(Step::Replace(stats))
    }

    fn maintain(&mut self, _: ()) -> anyhow::Result<Self::Output> {
        Ok(Step::Maintain)
    }

    fn needs_maintenance(&mut self) -> bool {
        false
    }
}

///////////
// Stats //
///////////

/// Statistics reported for each step in a streaming pipeline.
#[derive(Debug, serde::Serialize)]
pub(in crate::index::inmem2) enum Step {
    Search(inmem2::benchmark::knn::Stats),
    Insert(inmem2::benchmark::build::Stats),
    Replace(inmem2::benchmark::build::Stats),
    Delete,
    Maintain,
}

impl Step {
    fn is_maintain(&self) -> bool {
        matches!(self, Self::Maintain)
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Search(_) => "search",
            Self::Insert(_) => "insert",
            Self::Replace(_) => "replace",
            Self::Delete => "delete",
            Self::Maintain => "maintain",
        }
    }
}

impl std::fmt::Display for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Search(stats) => {
                write!(f, "{}", stats)
            }
            Self::Insert(stats) => {
                write!(f, "{}", stats)
            }
            Self::Replace(stats) => {
                write!(f, "{}", stats)
            }
            Self::Delete => Ok(()),
            Self::Maintain => Ok(()),
        }
    }
}

// #[derive(Debug, serde::Serialize)]
// struct Stats(Vec<Step>);

// /////////////
// // Summary //
// /////////////
//
// /// A [`std::fmt::Display`] helper for a collection of [`StreamStats`] results.
// #[derive(Debug)]
// pub(crate) struct Summary<I>(I);
//
// impl<I> Summary<I> {
//     pub(crate) fn new(iter: I) -> Self {
//         Self(iter)
//     }
// }
//
// impl<'a, I> std::fmt::Display for Summary<I>
// where
//     I: ExactSizeIterator<Item = &'a StreamStats> + Clone + 'a,
// {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let stats = &self.0;
//
//         // Break the results into "search" and everything else.
//         //
//         // For each of the search operations, count the number of results for each search.
//         let search_stages = stats.clone().filter(|stage| stage.is_search()).count();
//
//         // Arithmetic cannot underflow because `search_stages` is strictly upper-bounded
//         // by `stats.len()`.
//         let other_rows = stats
//             .clone()
//             .filter(|s| !s.is_search())
//             .map(|s| match s {
//                 StreamStats::Maintain(phases) => phases.len(),
//                 _ => 1,
//             })
//             .sum::<usize>();
//         if other_rows != 0 {
//             let header = [
//                 "Stage",
//                 "Operation",
//                 "Mean Latency (us)",
//                 "P90 Latency",
//                 "P99 Latency",
//             ];
//
//             let mut table = Table::new(header, other_rows);
//             let mut row = 0;
//             let mut stage: usize = 0;
//             for s in stats.clone() {
//                 match s {
//                     // Search stats are formatter later.
//                     StreamStats::Search(_) => {
//                         stage += 1;
//                         continue;
//                     }
//                     StreamStats::Insert(stats) | StreamStats::Replace(stats) => {
//                         let mut r = table.row(row);
//                         r.insert(stage, 0);
//                         r.insert(stats.kind, 1);
//                         r.insert(format!("{:.1}us", stats.insert_latencies.mean), 2);
//                         r.insert(stats.insert_latencies.p90, 3);
//                         r.insert(stats.insert_latencies.p99, 4);
//
//                         row += 1;
//                         stage += 1;
//                     }
//                     StreamStats::Delete(stats) => {
//                         let mut r = table.row(row);
//                         r.insert(stage, 0);
//                         r.insert(stats.kind.to_string(), 1);
//                         r.insert(format!("{:.1}us", stats.latencies.mean), 2);
//                         r.insert(stats.latencies.p90, 3);
//                         r.insert(stats.latencies.p99, 4);
//
//                         row += 1;
//                         stage += 1;
//                     }
//                     StreamStats::Maintain(phases) => {
//                         for stats in phases {
//                             let mut r = table.row(row);
//                             r.insert(format!("{}-pre", stage), 0);
//                             r.insert(stats.kind.to_string(), 1);
//                             r.insert(format!("{:.1}us", stats.latencies.mean), 2);
//                             r.insert(stats.latencies.p90, 3);
//                             r.insert(stats.latencies.p99, 4);
//                             row += 1;
//                         }
//                     }
//                 }
//             }
//
//             table.fmt(f)?;
//             writeln!(f)?;
//         }
//
//         // Write out the search results.
//         if search_stages != 0 {
//             let mut stage = 0;
//             for s in stats.clone() {
//                 if let StreamStats::Search(stats) = s {
//                     writeln!(f, "Search Stage {}", stage)?;
//                     writeln!(f, "{}", DisplayWrapper(&**stats))?;
//                 }
//
//                 if !s.is_maintain() {
//                     stage += 1;
//                 }
//             }
//         }
//
//         Ok(())
//     }
// }
