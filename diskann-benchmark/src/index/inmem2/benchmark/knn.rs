/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Utilities for setting up and running KNN style searches.
//!
//! The recommended flow is to instantiate a [`Knn`]. This provides utilities for loading
//! both groundtruth elements and queries. [`Knn::search`] is used to actually run the KNN
//! search and aggregate results in [`Stats`].
//!
//! For a slightly lower level interface where queries and groundtruth elements are provided
//! externally, [`Plan`] can be used to queue up a collection of operating points, with
//! [`Plan::search`] running the parameter sweep.
//!
//! To make an index Knn compatible, use the [`Search`] trait. This is used as a bridge
//! to [`dbc::search::Search`]. We don't use the graph methods in [`diskann_benchmark_core`]
//! directly since we're leaving open the possibility of tweaking strategies and search
//! parameters at a finer level of granularity than [`diskann_benchmark_core`] allows at
//! the time of writing.

use std::{future::Future, io::Write, num::NonZeroUsize, sync::Arc};

use diskann::{utils::IntoUsize, ANNResult};
use diskann_benchmark_core::{self as dbc};
use diskann_benchmark_runner::{self as dbr, files::InputFile, utils::MicroSeconds};
use diskann_inmem::{self as di};
use diskann_utils::views::rowmajor::{self, Matrix};

use crate::{index::inmem2::common::FromRawWith, utils::nz};

pub(in crate::index::inmem2) mod dto {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(super) struct Sweep {
        pub(super) search_n: NonZeroUsize,
        pub(super) search_l: Vec<usize>,
        pub(super) recall_k: NonZeroUsize,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(in crate::index::inmem2) struct Knn {
        pub(super) queries: InputFile,
        pub(super) groundtruth: InputFile,
        pub(super) reps: NonZeroUsize,
        pub(super) num_threads: Vec<NonZeroUsize>,
        pub(super) runs: Vec<Sweep>,
    }

    impl Knn {
        pub(in crate::index::inmem2) fn example() -> Self {
            nz!(REPS = 5);
            nz!(NUM_THREADS = 8);
            nz!(SEARCH_N = 10);
            nz!(RECALL_K = 10);

            Self {
                queries: InputFile::new("path/to/queries"),
                groundtruth: InputFile::new("path/to/groundtruth"),
                reps: REPS,
                num_threads: vec![NUM_THREADS],
                runs: vec![Sweep {
                    search_n: SEARCH_N,
                    search_l: vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    recall_k: RECALL_K,
                }],
            }
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(in crate::index::inmem2) struct StreamingKnn {
        pub(super) reps: NonZeroUsize,
        pub(super) num_threads: Vec<NonZeroUsize>,
        pub(super) runs: Vec<Sweep>,
    }

    impl StreamingKnn {
        pub(in crate::index::inmem2) fn example() -> Self {
            nz!(REPS = 5);
            nz!(NUM_THREADS = 8);
            nz!(SEARCH_N = 10);
            nz!(RECALL_K = 10);

            Self {
                reps: REPS,
                num_threads: vec![NUM_THREADS],
                runs: vec![Sweep {
                    search_n: SEARCH_N,
                    search_l: vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    recall_k: RECALL_K,
                }],
            }
        }
    }
}

/// Parameters for a single instance of KNN search and recall computation.
#[derive(Debug, Clone)]
pub(in crate::index::inmem2) struct Parameters {
    knn: diskann::graph::search::Knn,
    num_neighbors: NonZeroUsize,
    recall_k: NonZeroUsize,
}

/// A flattened and validated list of parameters for which a range of KNN searches will
/// be run.
#[derive(Debug)]
pub(in crate::index::inmem2) struct Plan {
    sweep: Vec<dbc::search::Run<Parameters>>,
    max_recall_k: NonZeroUsize,
}

impl Plan {
    /// Construct a new [`Plan`] by flattening the product of:
    ///
    /// * `num_threads`: To enable thread sweeps.
    /// * `runs`: For different `recall_k`/`search_n` regimes.
    /// * `runs.search_l`: For conducting search over multiple window sizes.
    ///
    /// Requires non-empty thread counts, runs, and search windows.
    /// Validates that each `recall_k` does not exceed its `search_n` and that the
    /// [`diskann::graph::search::Knn`] structs are accepted.
    fn new(
        reps: NonZeroUsize,
        num_threads: &[NonZeroUsize],
        runs: &[dto::Sweep],
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !num_threads.is_empty(),
            "search thread counts cannot be empty"
        );
        anyhow::ensure!(!runs.is_empty(), "search runs cannot be empty");

        let mut sweep = Vec::new();
        let mut max_recall_k = NonZeroUsize::MIN;

        for &threads in num_threads {
            for (position, run) in runs.iter().enumerate() {
                anyhow::ensure!(
                    !run.search_l.is_empty(),
                    "run {} of {} has no search windows",
                    position + 1,
                    runs.len(),
                );

                if run.recall_k > run.search_n {
                    anyhow::bail!(
                        "run {} of {} has recall_k ({}) greater than search_n ({})",
                        position + 1,
                        runs.len(),
                        run.recall_k,
                        run.search_n,
                    );
                }

                for &search_l in &run.search_l {
                    let parameters = Parameters {
                        knn: diskann::graph::search::Knn::new(search_l, None)?,
                        num_neighbors: run.search_n,
                        recall_k: run.recall_k,
                    };
                    max_recall_k = max_recall_k.max(run.recall_k);
                    sweep.push(dbc::search::Run::new(
                        parameters,
                        dbc::search::Setup {
                            threads,
                            tasks: threads,
                            reps,
                        },
                    ));
                }
            }
        }

        Ok(Self {
            sweep,
            max_recall_k,
        })
    }

    /// Flatten the parameters in `streaming` into a full plan.
    pub(in crate::index::inmem2) fn from_streaming(
        streaming: &dto::StreamingKnn,
    ) -> anyhow::Result<Self> {
        let dto::StreamingKnn {
            reps,
            num_threads,
            runs,
        } = streaming;
        Self::new(*reps, num_threads, runs)
    }

    /// Run each search point in `self` over `index` aggregating all results.
    ///
    /// Queries must be non-empty and have the same row count as `groundtruth`.
    pub(in crate::index::inmem2) fn search<T, S>(
        &self,
        index: Arc<S>,
        queries: Arc<rowmajor::Owned<T>>,
        groundtruth: &dyn dbc::recall::Rows<u32>,
        output: Option<&mut dyn dbr::Output>,
    ) -> anyhow::Result<Stats>
    where
        T: Send + Sync + 'static,
        S: Search<T>,
    {
        anyhow::ensure!(queries.nrows() != 0, "search queries cannot be empty");
        anyhow::ensure!(
            queries.nrows() == groundtruth.nrows(),
            "query/groundtruth row mismatch: queries = {}, groundtruth = {}",
            queries.nrows(),
            groundtruth.nrows(),
        );

        let searcher = Searcher { index, queries };

        let raw = dbc::search::search_all(
            Arc::new(searcher),
            self.sweep.iter().cloned(),
            Aggregator::new(groundtruth, dbc::recall::GroundTruthMode::Fixed),
        )?;

        let stats = Stats(raw);
        if let Some(mut output) = output {
            writeln!(output, "{}", stats)?;
        }

        Ok(stats)
    }
}

impl std::fmt::Display for Plan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let header = [
            "Threads",
            "Tasks",
            "Reps",
            "Search L",
            "Recall K",
            "Search N",
            "Beam Width",
        ];
        let mut table = dbr::utils::fmt::Table::new(header, self.sweep.len());

        for (i, run) in self.sweep.iter().enumerate() {
            let setup = run.setup();
            let parameters = run.parameters();
            let mut row = table.row(i);
            row.insert(setup.threads, 0);
            row.insert(setup.tasks, 1);
            row.insert(setup.reps, 2);
            row.insert(parameters.knn.l_value(), 3);
            row.insert(parameters.recall_k, 4);
            row.insert(parameters.num_neighbors, 5);
            row.insert(parameters.knn.beam_width(), 6);
        }

        write!(f, "{}", table)
    }
}

/////////
// Knn //
/////////

/// A prepared Knn search with queries and groundtruth on disk.
#[derive(Debug)]
pub(in crate::index::inmem2) struct Knn {
    queries: InputFile,
    groundtruth: InputFile,
    plan: Plan,
}

impl Knn {
    fn queries_arc<T>(&self) -> anyhow::Result<Arc<rowmajor::Owned<T>>>
    where
        T: bytemuck::Pod,
    {
        crate::utils::datafiles::load_dataset(crate::utils::datafiles::BinFile(&self.queries))
            .map(Arc::new)
    }

    fn groundtruth(&self) -> anyhow::Result<rowmajor::Owned<u32>> {
        crate::utils::datafiles::load_groundtruth(
            crate::utils::datafiles::BinFile(&self.groundtruth),
            Some(self.plan.max_recall_k.get()),
        )
    }

    /// Run a Knn search on `index`. Queries and groundtruth will be loaded from disk.
    pub(in crate::index::inmem2) fn search<T, S>(
        &self,
        index: Arc<S>,
        output: Option<&mut dyn dbr::Output>,
    ) -> anyhow::Result<Stats>
    where
        T: bytemuck::Pod + Send + Sync + 'static,
        S: Search<T>,
    {
        let queries = self.queries_arc::<T>()?;
        let groundtruth = self.groundtruth()?;
        self.plan.search(index, queries, &groundtruth, output)
    }
}

impl FromRawWith for Knn {
    type Raw = dto::Knn;

    fn from_raw_with(raw: &dto::Knn, checker: Option<&mut dbr::Checker>) -> anyhow::Result<Self> {
        let dto::Knn {
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

        let plan = Plan::new(*reps, num_threads, runs)?;

        Ok(Self {
            queries,
            groundtruth,
            plan,
        })
    }
}

impl std::fmt::Display for Knn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            queries,
            groundtruth,
            plan,
        } = self;

        let mut kv = dbr::utils::fmt::KeyValue::new();
        kv.push("queries", queries);
        kv.push("groundtruth", groundtruth);
        kv.push("plan", plan);
        write!(f, "{}", kv)
    }
}

////////////
// Search //
////////////

#[derive(Debug, Clone, Copy)]
pub(in crate::index::inmem2) struct Counters {
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

/// We use this with [`Searcher`] to implement [`dbc::search::Search`].
pub(in crate::index::inmem2) trait Search<T>:
    Send + Sync + 'static
{
    fn knn<B>(
        &self,
        parameters: &Parameters,
        query: &[T],
        buffer: &mut B,
    ) -> impl Future<Output = ANNResult<Counters>> + Send
    where
        B: diskann::graph::SearchOutputBuffer<u32> + Send;
}

/// New-type adaptor for [`dbc::search::Search`].
struct Searcher<K, T> {
    index: Arc<K>,
    queries: Arc<rowmajor::Owned<T>>,
}

impl<K, T> dbc::search::Search for Searcher<K, T>
where
    K: Search<T>,
    T: Send + Sync + 'static,
{
    type Id = u32;
    type Parameters = Parameters;
    type Output = Counters;

    fn num_queries(&self) -> usize {
        self.queries.nrows()
    }

    fn id_count(&self, parameters: &Parameters) -> dbc::search::IdCount {
        dbc::search::IdCount::Fixed(parameters.num_neighbors)
    }

    fn search<B>(
        &self,
        parameters: &Parameters,
        buffer: &mut B,
        index: usize,
    ) -> impl Future<Output = ANNResult<Counters>> + Send
    where
        B: diskann::graph::SearchOutputBuffer<u32> + Send,
    {
        self.index.knn(parameters, self.queries.row(index), buffer)
    }
}

impl<T> Search<T> for diskann::graph::DiskANNIndex<di::Provider<di::repr::Full<T>, u32>>
where
    T: di::repr::FullPrecision,
{
    async fn knn<B>(
        &self,
        parameters: &Parameters,
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

impl Search<f32> for diskann::graph::DiskANNIndex<di::Provider<di::repr::Quantized, u32>> {
    async fn knn<B>(
        &self,
        parameters: &Parameters,
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

///////////
// Stats //
///////////

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::index::inmem2) struct Stats(Vec<RunStats>);

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        format_stats_table(f, &self.0)
    }
}

fn format_stats_table(f: &mut dyn std::fmt::Write, stats: &[RunStats]) -> std::fmt::Result {
    use crate::utils::MaybeDisplay;

    if stats.is_empty() {
        return Ok(());
    }

    let header = &[
        "Ls",
        "Recall K",
        "Search N",
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

        let mut buf = Vec::<f64>::new();

        row.insert(r.search_l, 0);
        row.insert(r.recall_k, 1);
        row.insert(r.search_n, 2);
        row.insert(r.mean_cmps, 3);
        row.insert(r.mean_hops, 4);

        buf.clear();
        buf.extend(r.reps.iter().map(|r| r.qps));
        row.insert(
            format!(
                "{:.1} ({:.1})",
                MaybeDisplay(dbr::utils::percentiles::mean(&buf), "missing"),
                MaybeDisplay(dbr::utils::percentiles::max_f64(&buf), "missing"),
            ),
            5,
        );

        buf.clear();
        buf.extend(r.reps.iter().map(|r| r.mean_latency));
        row.insert(
            format!(
                "{:.1}us ({:.1}us)",
                MaybeDisplay(dbr::utils::percentiles::mean(&buf), "missing"),
                MaybeDisplay(dbr::utils::percentiles::max_f64(&buf), "missing"),
            ),
            6,
        );

        buf.clear();
        buf.extend(r.reps.iter().map(|r| r.p99_latency.as_f64()));
        row.insert(
            format!(
                "{:.1}us ({:.1})",
                MaybeDisplay(dbr::utils::percentiles::mean(&buf), "missing"),
                MaybeDisplay(dbr::utils::percentiles::max_f64(&buf), "missing"),
            ),
            7,
        );
        row.insert(format!("{:3}", r.recall.average), 8);
        row.insert(r.num_tasks, 9);
    });

    write!(f, "{}", table)
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RepetitionStats {
    qps: f64,
    end_to_end_latency: MicroSeconds,
    mean_latency: f64,
    p90_latency: MicroSeconds,
    p99_latency: MicroSeconds,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::index::inmem2) struct RunStats {
    num_tasks: usize,
    search_n: usize,
    search_l: usize,
    recall_k: usize,
    reps: Vec<RepetitionStats>,
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

impl dbc::search::Aggregate<Parameters, u32, Counters> for Aggregator<'_> {
    type Output = RunStats;

    fn aggregate(
        &mut self,
        run: dbc::search::Run<Parameters>,
        mut results: Vec<dbc::search::SearchResults<u32, Counters>>,
    ) -> anyhow::Result<RunStats> {
        let setup = run.setup();
        let knn: &Parameters = run.parameters();

        // Compute the recall using just the first result.
        let recall = match results.first() {
            Some(first) => dbc::recall::knn(
                self.groundtruth,
                None,
                first.ids().as_rows(),
                knn.recall_k.get(),
                knn.num_neighbors.get(),
                self.mode,
            )?,
            None => anyhow::bail!("Results must be non-empty"),
        };

        let num_queries = recall.num_queries as f64;
        let mut reps = Vec::with_capacity(results.len());
        results.iter_mut().try_for_each(|r| -> anyhow::Result<()> {
            let percentiles = dbr::utils::percentiles::compute_percentiles(r.latencies_mut())?;
            let end_to_end_latency = r.end_to_end_latency();
            let qps = num_queries / end_to_end_latency.as_seconds();
            let rep = RepetitionStats {
                qps,
                end_to_end_latency,
                mean_latency: percentiles.mean,
                p90_latency: percentiles.p90,
                p99_latency: percentiles.p99,
            };

            reps.push(rep);
            Ok(())
        })?;

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

        Ok(RunStats {
            num_tasks: setup.tasks.get(),
            search_n: knn.num_neighbors.get(),
            search_l: knn.knn.l_value().get(),
            recall_k: knn.recall_k.get(),
            reps,
            recall: crate::utils::recall::RecallMetrics::from(&recall),
            mean_cmps,
            mean_hops,
        })
    }
}

///////////
// tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_parses() {
        assert!(Plan::from_streaming(&dto::StreamingKnn::example()).is_ok());
        assert!(Knn::from_raw_with(&dto::Knn::example(), None).is_ok());
    }

    #[test]
    fn plan_expands_sweeps_in_thread_order() {
        nz!(REPS = 3);
        nz!(ONE = 1);
        nz!(TWO = 2);
        nz!(FOUR = 4);
        nz!(TEN = 10);

        let runs = [
            dto::Sweep {
                search_n: TEN,
                search_l: vec![20, 30],
                recall_k: TWO,
            },
            dto::Sweep {
                search_n: FOUR,
                search_l: vec![40],
                recall_k: FOUR,
            },
        ];
        let plan = Plan::new(REPS, &[ONE, FOUR], &runs).unwrap();
        let actual: Vec<_> = plan
            .sweep
            .iter()
            .map(|run| {
                let setup = run.setup();
                let parameters = run.parameters();
                (
                    setup.threads.get(),
                    setup.tasks.get(),
                    setup.reps.get(),
                    parameters.knn.l_value().get(),
                    parameters.num_neighbors.get(),
                    parameters.recall_k.get(),
                )
            })
            .collect();

        assert_eq!(
            actual,
            [
                (1, 1, 3, 20, 10, 2),
                (1, 1, 3, 30, 10, 2),
                (1, 1, 3, 40, 4, 4),
                (4, 4, 3, 20, 10, 2),
                (4, 4, 3, 30, 10, 2),
                (4, 4, 3, 40, 4, 4),
            ]
        );
        assert_eq!(plan.max_recall_k, FOUR);
    }

    #[test]
    fn plan_display() {
        nz!(ONE = 1);
        nz!(TWO = 2);
        nz!(THREE = 3);
        nz!(FOUR = 4);
        nz!(TEN = 10);

        let plan = Plan {
            sweep: vec![
                dbc::search::Run::new(
                    Parameters {
                        knn: diskann::graph::search::Knn::new(20, Some(2)).unwrap(),
                        num_neighbors: TEN,
                        recall_k: FOUR,
                    },
                    dbc::search::Setup {
                        threads: TWO,
                        tasks: FOUR,
                        reps: THREE,
                    },
                ),
                dbc::search::Run::new(
                    Parameters {
                        knn: diskann::graph::search::Knn::new(30, Some(1)).unwrap(),
                        num_neighbors: FOUR,
                        recall_k: TWO,
                    },
                    dbc::search::Setup {
                        threads: FOUR,
                        tasks: TWO,
                        reps: ONE,
                    },
                ),
            ],
            max_recall_k: FOUR,
        };
        let formatted = plan.to_string();
        let rows: Vec<Vec<_>> = formatted
            .lines()
            .filter(|line| line.contains(','))
            .map(|line| line.split(',').map(str::trim).collect())
            .collect();
        assert_eq!(
            rows,
            [
                vec![
                    "Threads",
                    "Tasks",
                    "Reps",
                    "Search L",
                    "Recall K",
                    "Search N",
                    "Beam Width",
                ],
                vec!["2", "4", "3", "20", "4", "10", "2"],
                vec!["4", "2", "1", "30", "2", "4", "1"],
            ]
        );
    }
}
