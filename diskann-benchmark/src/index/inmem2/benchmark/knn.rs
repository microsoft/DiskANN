/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{future::Future, num::NonZeroUsize, sync::Arc};

use diskann::{utils::IntoUsize, ANNResult};
use diskann_benchmark_core::{self as dbc};
use diskann_benchmark_runner::{self as dbr, files::InputFile, utils::MicroSeconds};
use diskann_inmem::{self as di};
use diskann_utils::views::rowmajor::{self, Matrix};

use crate::index::inmem2::FromRawWith;

pub(in crate::index::inmem2) mod dto {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub(in crate::index::inmem2) struct KnnSweep {
        pub(super) search_n: usize,
        pub(super) search_l: Vec<usize>,
        pub(super) recall_k: NonZeroUsize,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(in crate::index::inmem2) struct KnnSearch {
        pub(super) queries: InputFile,
        pub(super) groundtruth: InputFile,
        pub(super) reps: NonZeroUsize,
        pub(super) num_threads: Vec<NonZeroUsize>,
        pub(super) runs: Vec<KnnSweep>,
    }

    impl KnnSearch {
        pub(in crate::index::inmem2) fn example() -> Self {
            const FIVE: NonZeroUsize = NonZeroUsize::new(5).unwrap();
            const EIGHT: NonZeroUsize = NonZeroUsize::new(5).unwrap();
            const TEN: NonZeroUsize = NonZeroUsize::new(10).unwrap();

            Self {
                queries: InputFile::new("path/to/queries"),
                groundtruth: InputFile::new("path/to/groundtruth"),
                reps: FIVE,
                num_threads: vec![EIGHT],
                runs: vec![KnnSweep {
                    search_n: 10,
                    search_l: vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    recall_k: TEN,
                }],
            }
        }
    }
}

#[derive(Debug, Clone)]
struct KnnInstance {
    knn: diskann::graph::search::Knn,
    num_neighbors: usize,
    recall_k: NonZeroUsize,
}

impl KnnInstance {
    fn flatten_from_raw(raw: &[dto::KnnSweep]) -> anyhow::Result<Vec<Self>> {
        raw.iter()
            .flat_map(|sweep| {
                let dto::KnnSweep {
                    search_n,
                    search_l,
                    recall_k,
                } = sweep;

                search_l.iter().map(move |search_l| -> anyhow::Result<_> {
                    let knn = diskann::graph::search::Knn::new(*search_l, None)?;
                    Ok(KnnInstance {
                        knn,
                        num_neighbors: *search_n,
                        recall_k: *recall_k,
                    })
                })
            })
            .collect()
    }
}

impl std::fmt::Display for KnnInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
pub(in crate::index::inmem2) struct KnnSearch {
    queries: InputFile,
    groundtruth: InputFile,
    reps: NonZeroUsize,
    num_threads: Vec<NonZeroUsize>,
    runs: Vec<KnnInstance>,
}

impl KnnSearch {
    fn maximum_recall_k(&self) -> Option<usize> {
        self.runs.iter().map(|r| r.recall_k).max().map(|v| v.get())
    }

    fn queries<T>(&self) -> anyhow::Result<rowmajor::Owned<T>>
    where
        T: bytemuck::Pod,
    {
        crate::utils::datafiles::load_dataset(crate::utils::datafiles::BinFile(&self.queries))
    }

    fn queries_arc<T>(&self) -> anyhow::Result<Arc<rowmajor::Owned<T>>>
    where
        T: bytemuck::Pod,
    {
        self.queries::<T>().map(Arc::new)
    }

    fn groundtruth(&self) -> anyhow::Result<rowmajor::Owned<u32>> {
        crate::utils::datafiles::load_groundtruth(
            crate::utils::datafiles::BinFile(&self.groundtruth),
            self.maximum_recall_k(),
        )
    }

    pub(in crate::index::inmem2) fn search<T, S>(&self, engine: Arc<S>) -> anyhow::Result<Stats>
    where
        T: bytemuck::Pod + Send + Sync + 'static,
        S: Knn<T>,
    {
        let queries = self.queries_arc::<T>()?;
        let groundtruth = self.groundtruth()?;
        let parameters: Vec<_> = self
            .num_threads
            .iter()
            .flat_map(|num_threads| {
                self.runs.iter().map(|run| {
                    dbc::search::Run::new(
                        run.clone(),
                        dbc::search::Setup {
                            threads: *num_threads,
                            tasks: *num_threads,
                            reps: self.reps,
                        },
                    )
                })
            })
            .collect();

        let searcher = Search {
            knn: engine,
            queries,
        };

        let stats = dbc::search::search_all(
            Arc::new(searcher),
            parameters,
            Aggregator::new(&groundtruth, dbc::recall::GroundTruthMode::Fixed),
        )?;

        Ok(Stats(stats))
    }
}

impl FromRawWith for KnnSearch {
    type Raw = dto::KnnSearch;

    fn from_raw_with(
        raw: &dto::KnnSearch,
        checker: Option<&mut dbr::Checker>,
    ) -> anyhow::Result<Self> {
        let dto::KnnSearch {
            queries,
            groundtruth,
            reps,
            num_threads,
            runs,
        } = raw;

        let mut queries = queries.clone();
        let mut groundtruth = groundtruth.clone();

        if let Some(checker) = checker {
            queries.resolve(checker)?;
            groundtruth.resolve(checker)?;
        }

        Ok(Self {
            queries,
            groundtruth,
            reps: *reps,
            num_threads: num_threads.clone(),
            runs: KnnInstance::flatten_from_raw(runs)?,
        })
    }
}

impl std::fmt::Display for KnnSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut kv = dbr::utils::fmt::KeyValue::new();
        kv.push("queries", &self.queries);
        kv.push("groundtruth", &self.groundtruth);
        kv.push("reps", &self.reps);

        let num_threads = dbr::utils::fmt::Delimit::new(self.num_threads.iter(), ", ");
        kv.push("num_threads", &num_threads);

        let runs = dbr::utils::fmt::Delimit::new(self.runs.iter(), "\n").to_string();
        kv.push("runs", &runs);
        write!(f, "{}", kv)
    }
}

#[derive(Debug, Clone, Copy)]
struct Counters {
    cmps: usize,
    hops: usize,
}

impl Counters {
    fn new(stats: &diskann::graph::index::SearchStats) -> Self {
        Self {
            cmps: stats.cmps.into_usize(),
            hops: stats.hops.into_usize(),
        }
    }
}

trait Knn<T>: Send + Sync + 'static {
    fn knn<B>(
        &self,
        parameters: &KnnInstance,
        query: &[T],
        buffer: &mut B,
    ) -> impl Future<Output = ANNResult<Counters>> + Send
    where
        B: diskann::graph::SearchOutputBuffer<u32> + Send;
}

struct Search<K, T> {
    knn: Arc<K>,
    queries: Arc<rowmajor::Owned<T>>,
}

impl<K, T> dbc::search::Search for Search<K, T>
where
    K: Knn<T>,
    T: Send + Sync + 'static,
{
    type Id = u32;
    type Parameters = KnnInstance;
    type Output = Counters;

    fn num_queries(&self) -> usize {
        self.queries.nrows()
    }

    fn id_count(&self, parameters: &KnnInstance) -> dbc::search::IdCount {
        dbc::search::IdCount::Fixed(parameters.recall_k)
    }

    fn search<B>(
        &self,
        parameters: &KnnInstance,
        buffer: &mut B,
        index: usize,
    ) -> impl Future<Output = ANNResult<Counters>> + Send
    where
        B: diskann::graph::SearchOutputBuffer<u32> + Send,
    {
        self.knn.knn(parameters, self.queries.row(index), buffer)
    }
}

impl<T> Knn<T> for diskann::graph::DiskANNIndex<di::Provider<di::repr::Full<T>, u32>>
where
    T: di::repr::FullPrecision,
{
    async fn knn<B>(
        &self,
        parameters: &KnnInstance,
        query: &[T],
        buffer: &mut B,
    ) -> ANNResult<Counters>
    where
        B: diskann::graph::SearchOutputBuffer<u32> + Send,
    {
        let stats = self
            .search(parameters.knn, &di::Strategy, &di::Context, query, buffer)
            .await?;

        Ok(Counters::new(&stats))
    }
}

impl Knn<f32> for diskann::graph::DiskANNIndex<di::Provider<di::repr::Quantized, u32>> {
    async fn knn<B>(
        &self,
        parameters: &KnnInstance,
        query: &[f32],
        buffer: &mut B,
    ) -> ANNResult<Counters>
    where
        B: diskann::graph::SearchOutputBuffer<u32> + Send,
    {
        let stats = self
            .search(parameters.knn, &di::Strategy, &di::Context, query, buffer)
            .await?;

        Ok(Counters::new(&stats))
    }
}

///////////////
// DBC stuff //
///////////////

#[derive(Debug)]
pub(in crate::index::inmem2) struct Stats(Vec<Point>);

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        format_stats_table(f, &self.0)
    }
}

fn format_stats_table(f: &mut dyn std::fmt::Write, stats: &[Point]) -> std::fmt::Result {
    use crate::utils::MaybeDisplay;

    if stats.is_empty() {
        return Ok(());
    }

    let header = &[
        "Ls",
        "KNN",
        "Avg cmps",
        "Avg hops",
        "QPS - mean(max)",
        "Avg Latency",
        "p99 Latency",
        "Recall",
        "Threads",
    ];

    let mut table = dbr::utils::fmt::Table::new(header, stats.len());
    stats.iter().enumerate().for_each(|(i, r)| {
        let mut row = table.row(i);

        row.insert(r.search_l, 0);
        row.insert(r.search_n, 1);
        row.insert(r.mean_cmps, 2);
        row.insert(r.mean_hops, 3);
        row.insert(
            format!(
                "{:.1} ({:.1})",
                MaybeDisplay(dbr::utils::percentiles::mean(&r.qps), "missing"),
                MaybeDisplay(dbr::utils::percentiles::max_f64(&r.qps), "missing"),
            ),
            4,
        );
        row.insert(
            format!(
                "{:.1}us ({:.1}us)",
                MaybeDisplay(dbr::utils::percentiles::mean(&r.mean_latencies), "missing"),
                MaybeDisplay(
                    dbr::utils::percentiles::max_f64(&r.mean_latencies),
                    "missing"
                ),
            ),
            5,
        );
        row.insert(
            format!(
                "{:.1}us ({:.1})",
                MaybeDisplay(dbr::utils::percentiles::mean(&r.p99_latencies), "missing"),
                MaybeDisplay(r.p99_latencies.iter().max(), "missing"),
            ),
            6,
        );
        row.insert(format!("{:3}", r.recall.average), 7);
        row.insert(r.num_tasks, 8);
    });

    write!(f, "{}", table)
}

#[derive(Debug)]
pub(in crate::index::inmem2) struct Point {
    num_tasks: usize,
    search_n: usize,
    search_l: usize,
    qps: Vec<f64>,
    search_latencies: Vec<MicroSeconds>,
    mean_latencies: Vec<f64>,
    p90_latencies: Vec<MicroSeconds>,
    p99_latencies: Vec<MicroSeconds>,
    recall: crate::utils::recall::RecallMetrics,
    mean_cmps: f64,
    mean_hops: f64,
}

struct Aggregator<'a> {
    groundtruth: &'a dyn dbc::recall::Rows<u32>,
    mode: dbc::recall::GroundTruthMode,
}

impl<'a> Aggregator<'a> {
    fn new(
        groundtruth: &'a dyn dbc::recall::Rows<u32>,
        mode: dbc::recall::GroundTruthMode,
    ) -> Self {
        Self { groundtruth, mode }
    }
}

impl dbc::search::Aggregate<KnnInstance, u32, Counters> for Aggregator<'_> {
    type Output = Point;

    fn aggregate(
        &mut self,
        run: dbc::search::Run<KnnInstance>,
        mut results: Vec<dbc::search::SearchResults<u32, Counters>>,
    ) -> anyhow::Result<Point> {
        let setup = run.setup();
        let knn: &KnnInstance = run.parameters();

        // Compute the recall using just the first result.
        let recall = match results.first() {
            Some(first) => dbc::recall::knn(
                self.groundtruth,
                None,
                first.ids().as_rows(),
                knn.recall_k.get(),
                knn.num_neighbors,
                self.mode,
            )?,
            None => anyhow::bail!("Results must be non-empty"),
        };

        let mut mean_latencies = Vec::with_capacity(results.len());
        let mut p90_latencies = Vec::with_capacity(results.len());
        let mut p99_latencies = Vec::with_capacity(results.len());

        results.iter_mut().for_each(|r| {
            match dbr::utils::percentiles::compute_percentiles(r.latencies_mut()) {
                Ok(values) => {
                    let dbr::utils::percentiles::Percentiles { mean, p90, p99, .. } = values;
                    mean_latencies.push(mean);
                    p90_latencies.push(p90);
                    p99_latencies.push(p99);
                }
                Err(_) => {
                    let zero = MicroSeconds::new(0);
                    mean_latencies.push(0.0);
                    p90_latencies.push(zero);
                    p99_latencies.push(zero);
                }
            }
        });

        let num_queries = recall.num_queries as f64;
        let qps: Vec<f64> = results
            .iter()
            .map(|r| num_queries / r.end_to_end_latency().as_seconds())
            .collect();

        let search_latencies = results.iter().map(|r| r.end_to_end_latency()).collect();

        let mean_cmps = dbc::utils::average_all(
            results
                .iter()
                .flat_map(|r| r.output().iter().map(|o| o.cmps)),
        );

        let mean_hops = dbc::utils::average_all(
            results
                .iter()
                .flat_map(|r| r.output().iter().map(|o| o.hops)),
        );

        Ok(Point {
            num_tasks: setup.tasks.get(),
            search_n: knn.num_neighbors,
            search_l: knn.knn.l_value().get(),
            qps,
            search_latencies,
            mean_latencies,
            p90_latencies,
            p99_latencies,
            recall: crate::utils::recall::RecallMetrics::from(&recall),
            mean_cmps,
            mean_hops,
        })
    }
}
