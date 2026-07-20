/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::HashMap,
    fs::{self, File},
    path::Path,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use chrono::Utc;
use diskann_benchmark_core::recall::{self, ComputeRecallError};
use diskann_benchmark_runner::utils::percentiles::{self, CannotBeEmpty, Percentiles};
use diskann_utils::views::Matrix;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use thiserror::Error;
use tokio::{
    sync::mpsc,
    task::{JoinError, JoinSet},
};

use crate::{
    Options, RunArgs,
    catalog::Catalog,
    dataset::{Dataset, DatasetError, RowBuf},
    driver::{ControllerError, Driver},
    report::{OpReport, Report, SearchReport, StepReport},
    runbook::{Operation, Runbook},
};

#[derive(Debug)]
pub struct Filter {
    includes: Vec<String>,
    excludes: Vec<String>,
}

impl Filter {
    pub fn include(&mut self, key: &str) {
        self.includes.push(key.to_string());
    }

    pub fn exclude(&mut self, key: &str) {
        self.excludes.push(key.to_string());
    }

    pub fn matches(&self, key: &str) -> bool {
        let included = if !self.includes.is_empty() {
            let mut included = false;
            for in_rule in &self.includes {
                if key.starts_with(in_rule) {
                    included = true;
                    break;
                }
            }
            included
        } else {
            true
        };

        for ex_rule in &self.excludes {
            if key.starts_with(ex_rule) {
                return false;
            }
        }

        included
    }
}

impl Default for Filter {
    fn default() -> Self {
        let includes = vec![];
        let excludes = vec![];
        Self { includes, excludes }
    }
}

pub struct Runner<D: Driver> {
    driver: Arc<D>,
}

#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("Nothing to do; empty runbook or nothing matched filter")]
    NothingToDo,
    #[error("data set missing: {0}")]
    DatasetMissing(String),
    #[error("recipe missing: {0}")]
    RecipeMissing(String),
    #[error("recipe max points ({0}) bigger than dataset size ({1})")]
    RecipeMaxPoints(usize, usize),
    #[error("bad or missing runbook name")]
    BadName,
    #[error("redis error: {0}")]
    Redis(#[from] redis::RedisError),
    #[error("dataset error: {0}")]
    Dataset(#[from] DatasetError),
    #[error("driver error: {0}")]
    Driver(Box<dyn std::error::Error + Send + Sync>),
    #[error("compute recall error: {0}")]
    ComputeRecall(#[from] ComputeRecallError),
    #[error("progress task failed: {0}")]
    ProgressTask(#[from] JoinError),
    #[error("progress was empty")]
    EmptyProgress(#[from] CannotBeEmpty),
    #[error("file i/o error: {0}")]
    FileIo(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

impl<D: Driver + Send + Sync + 'static> Runner<D> {
    pub fn new(driver: D) -> Self {
        Self {
            driver: Arc::new(driver),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn run(
        &self,
        runbook: &Runbook,
        data_manager: &Catalog,
        report_path: &Path,
        recall_k: usize,
        recall_n: usize,
        filter: Option<Filter>,
        args: &RunArgs,
        opts: &Options,
    ) -> Result<(), RunnerError> {
        println!(
            "Executing runbook: {}...",
            runbook.name().unwrap_or("unknown")
        );

        // Filter recipes
        let recipes: Vec<&String> = if let Some(filter) = filter {
            runbook.recipes().filter(|ds| filter.matches(ds)).collect()
        } else {
            runbook.recipes().collect()
        };

        if recipes.is_empty() {
            println!("error: Runbook is empty.");
            return Err(RunnerError::NothingToDo);
        }

        let mut step_reports = HashMap::new();

        // Run each recipe
        for recipe_name in recipes {
            println!("Running dataset {recipe_name}...");

            // Make sure dataset <-> recipe makes sense
            let ds = data_manager
                .dataset(recipe_name)
                .ok_or_else(|| RunnerError::DatasetMissing(recipe_name.clone()))?;

            let recipe = runbook
                .recipe(recipe_name)
                .ok_or_else(|| RunnerError::RecipeMissing(recipe_name.clone()))?;

            if recipe.max_points() > ds.vector_count() {
                return Err(RunnerError::RecipeMaxPoints(
                    recipe.max_points(),
                    ds.vector_count(),
                ));
            }

            let con = self
                .driver
                .get_connection()
                .await
                .map_err(|e| RunnerError::Driver(Box::new(e)))?;
            self.driver
                .prepare(con)
                .await
                .map_err(|e| RunnerError::Driver(Box::new(e)))?;

            let mut error = None;
            for (step, op) in recipe.steps().enumerate() {
                println!("Step {step}: {}", op.name());

                let step_report = match op {
                    Operation::Insert { start, end } => self.op_insert(ds, *start, *end).await,

                    Operation::Delete { start, end } => self.op_delete(*start, *end).await,
                    Operation::Replace {
                        tags_start,
                        tags_end,
                        ids_start,
                        ..
                    } => {
                        let count = tags_end - tags_start;
                        self.op_replace(ds, *tags_start, *ids_start, count).await
                    }
                    Operation::Search => {
                        match runbook
                            .name()
                            .ok_or(RunnerError::BadName)
                            .and_then(|name| ds.step_gt(name, step + 1).map_err(|e| e.into()))
                            .map(Arc::new)
                        {
                            Ok(step_gt) => {
                                let mut step_reports = Vec::new();
                                for _ in 0..args.search_repetitions {
                                    let step_report = self
                                        .op_search(ds, step_gt.clone(), recall_k, recall_n)
                                        .await;

                                    match step_report {
                                        Ok(sr) => {
                                            step_reports.push(sr);
                                        }
                                        Err(e) => {
                                            error = Some(e);
                                            break;
                                        }
                                    }
                                }

                                if let Some(e) = error.take() {
                                    Err(e)
                                } else {
                                    let step_report = consolidate_search_reports(&step_reports);
                                    Ok(step_report)
                                }
                            }
                            Err(e) => Err(e),
                        }
                    }
                };

                let step_report = match step_report {
                    Ok(sr) => sr,
                    Err(e) => {
                        error = Some(e);
                        break;
                    }
                };

                step_reports
                    .entry(recipe_name.clone())
                    .or_insert_with(Vec::new)
                    .push(step_report.clone());

                match step_report {
                    StepReport::Insert(op_report) => {
                        println!(
                            "  Inserted {} vectors in {:0.3}s ({:0.3} inserts/sec; utilization was {:0.3}).",
                            op_report.count,
                            op_report.wall_time_s,
                            op_report.throughput(),
                            op_report.utilization(),
                        );
                    }
                    StepReport::Search(search_report) => {
                        println!(
                            "  Searches completed with {}-recall@{} of {:0.3}.",
                            search_report.k, search_report.n, search_report.recall
                        );

                        for (i, or) in search_report.op_reports.iter().enumerate() {
                            println!(
                                "    Repetition {}: Queried {} vectors in {:0.3}s ({:0.3} qps; utilization was {:0.3}).",
                                i + 1,
                                or.count,
                                or.wall_time_s,
                                or.throughput(),
                                or.utilization(),
                            );
                        }
                    }
                    StepReport::Delete(op_report) => {
                        println!(
                            "  Deleted {} vectors in {:0.3}s ({:0.3} deletes/sec; utilization was {:0.3}).",
                            op_report.count,
                            op_report.wall_time_s,
                            op_report.throughput(),
                            op_report.utilization(),
                        );
                    }
                    StepReport::Replace(op_report) => {
                        println!(
                            "  Replaced {} vectors in {:0.3}s ({:0.3} replaces/sec; utilization was {:0.3}).",
                            op_report.count,
                            op_report.wall_time_s,
                            op_report.throughput(),
                            op_report.utilization(),
                        );
                    }
                }
            }

            let result: Result<(), RunnerError> = async {
                let con = self
                    .driver
                    .get_connection()
                    .await
                    .map_err(|e| RunnerError::Driver(Box::new(e)))?;
                self.driver
                    .finish(con)
                    .await
                    .map_err(|e| RunnerError::Driver(Box::new(e)))?;
                Ok(())
            }
            .await;

            if let Some(e) = error {
                if let Err(e) = result {
                    eprintln!("cleanup failed: {e}");
                }
                return Err(e);
            }

            result?;
        }

        let available_parallelism = thread::available_parallelism()?.get();
        let runbook_name = runbook.name().unwrap_or("unknown").to_string();
        let date = Utc::now();
        let report = Report {
            date,
            num_threads: opts.threads.unwrap_or(available_parallelism),
            quantizer: opts.quantizer,
            num_tasks: args.tasks.unwrap_or(available_parallelism),
            pipeline_size: args.pipeline_size,
            search_repetitions: args.search_repetitions,
            degree: args.degree,
            l_build: args.l_build,
            l_search: args.l_search,
            k: args.k,
            n: args.n,
            runbook: runbook_name.clone(),
            dataset: step_reports,
        };

        let report_path = report_path.join(self.driver.name()).join(format!(
            "{runbook_name}-{}.json",
            date.format("%Y%m%dT%H%M%S%.3fZ")
        ));
        fs::create_dir_all(report_path.parent().unwrap_or(&report_path))?;

        let mut f = File::create(report_path)?;
        serde_json::to_writer(&mut f, &report)?;

        Ok(())
    }

    async fn op_insert(
        &self,
        dataset: &Dataset,
        start: usize,
        end: usize,
    ) -> Result<StepReport, RunnerError> {
        let (tx, rx) = mpsc::unbounded_channel::<usize>();
        let mut tasks = JoinSet::<Result<Vec<(usize, Duration)>, D::Error>>::new();

        let progress_handle = tokio::spawn(progress_task::<D>(rx, end - start));

        let chunks = chunk_range(start, end, self.driver.parallelism());
        let num_tasks = chunks.len();
        let mut inputs = Vec::new();

        for (chunk_start, chunk_end) in chunks {
            inputs.push((
                Arc::clone(&self.driver),
                self.driver
                    .get_connection()
                    .await
                    .map_err(|e| RunnerError::Driver(Box::new(e)))?,
                dataset.metric(),
                dataset.vectors::<D::Data>(chunk_start, chunk_end - chunk_start)?,
                tx.clone(),
            ));
        }

        let wall_time_start = Instant::now();

        for (driver, con, metric, data, tx) in inputs {
            tasks.spawn(async move { driver.insert(con, metric, data, tx).await });
        }

        let mut timings = Vec::new();
        for result in tasks.join_all().await {
            timings.extend(result.map_err(|e| RunnerError::Driver(Box::new(e)))?);
        }

        let wall_time_s = Instant::now().duration_since(wall_time_start).as_secs_f64();

        drop(tx);
        progress_handle
            .await?
            .map_err(|e| RunnerError::Driver(Box::new(e)))?;

        let report = StepReport::Insert(make_op_report(timings, wall_time_s, num_tasks)?);

        Ok(report)
    }

    async fn op_delete(&self, start: usize, end: usize) -> Result<StepReport, RunnerError> {
        let (tx, rx) = mpsc::unbounded_channel::<usize>();
        let mut tasks = JoinSet::<Result<Vec<(usize, Duration)>, D::Error>>::new();

        let progress_handle = tokio::spawn(progress_task::<D>(rx, end - start));

        let chunks = chunk_range(start, end, self.driver.parallelism());
        let num_tasks = chunks.len();
        let mut inputs = Vec::new();

        for _ in 0..num_tasks {
            inputs.push((
                Arc::clone(&self.driver),
                self.driver
                    .get_connection()
                    .await
                    .map_err(|e| RunnerError::Driver(Box::new(e)))?,
                tx.clone(),
            ));
        }

        let wall_time_start = Instant::now();

        for ((chunk_start, chunk_end), (driver, con, tx)) in chunks.zip(inputs) {
            tasks.spawn(async move { driver.delete(con, chunk_start, chunk_end, tx).await });
        }

        let mut timings = Vec::new();
        for result in tasks.join_all().await {
            timings.extend(result.map_err(|e| RunnerError::Driver(Box::new(e)))?);
        }

        let wall_time_s = Instant::now().duration_since(wall_time_start).as_secs_f64();

        drop(tx);
        progress_handle
            .await?
            .map_err(|e| RunnerError::Driver(Box::new(e)))?;

        let report = StepReport::Delete(make_op_report(timings, wall_time_s, num_tasks)?);

        Ok(report)
    }

    async fn op_search(
        &self,
        dataset: &Dataset,
        step_gt: Arc<(RowBuf<u32>, RowBuf<f32>)>,
        recall_k: usize,
        recall_n: usize,
    ) -> Result<StepReport, RunnerError> {
        let (tx, rx) = mpsc::unbounded_channel::<usize>();
        let mut tasks =
            JoinSet::<Result<(Vec<(usize, Duration)>, usize, Vec<Vec<u32>>), D::Error>>::new();

        let progress_handle = tokio::spawn(progress_task::<D>(rx, dataset.query_count()));

        let chunks = chunk_range(0, dataset.query_count(), self.driver.parallelism());
        let num_tasks = chunks.len();
        let mut inputs = Vec::new();

        for (chunk_start, chunk_end) in chunks {
            inputs.push((
                Arc::clone(&self.driver),
                self.driver
                    .get_connection()
                    .await
                    .map_err(|e| RunnerError::Driver(Box::new(e)))?,
                dataset.queries::<D::Data>(chunk_start, chunk_end - chunk_start)?,
                tx.clone(),
            ));
        }

        let wall_time_start = Instant::now();

        for (driver, con, data, tx) in inputs {
            tasks.spawn(async move { driver.search(con, data, recall_n, tx).await });
        }

        let results = tasks.join_all().await;
        let wall_time_s = Instant::now().duration_since(wall_time_start).as_secs_f64();

        let mut timings = Vec::new();
        let mut query_results = Matrix::<u32>::new(u32::MAX, dataset.query_count(), recall_n);
        for result in results {
            let (batch_timings, start_idx, results) =
                result.map_err(|e| RunnerError::Driver(Box::new(e)))?;
            timings.extend(batch_timings);

            for (i, row) in results.into_iter().enumerate() {
                let n = row.len().min(recall_n);
                query_results.row_mut(start_idx + i)[..n].copy_from_slice(&row[..n]);
            }
        }

        drop(tx);
        progress_handle
            .await?
            .map_err(|e| RunnerError::Driver(Box::new(e)))?;

        let op_report = make_op_report(timings, wall_time_s, num_tasks)?;

        let recall_metrics = recall::knn(
            &step_gt.0,
            Some(step_gt.1.as_view().into()),
            &query_results,
            recall_k,
            recall_n,
            recall::GroundTruthMode::Fixed,
        )?;

        let op_reports = vec![op_report];
        let report = StepReport::Search(SearchReport {
            op_reports,
            k: recall_metrics.recall_k,
            n: recall_metrics.recall_n,
            recall: recall_metrics.average,
        });

        Ok(report)
    }

    async fn op_replace(
        &self,
        dataset: &Dataset,
        tags_start: usize,
        ids_start: usize,
        count: usize,
    ) -> Result<StepReport, RunnerError> {
        let (tx, rx) = mpsc::unbounded_channel::<usize>();
        let mut tasks = JoinSet::<Result<Vec<(usize, Duration)>, D::Error>>::new();

        let progress_handle = tokio::spawn(progress_task::<D>(rx, count));

        let chunks = chunk_range(0, count, self.driver.parallelism());
        let num_tasks = chunks.len();
        let mut inputs = Vec::new();

        for (chunk_start, chunk_end) in chunks.clone() {
            inputs.push((
                Arc::clone(&self.driver),
                self.driver
                    .get_connection()
                    .await
                    .map_err(|e| RunnerError::Driver(Box::new(e)))?,
                dataset.metric(),
                dataset.vectors::<D::Data>(chunk_start + ids_start, chunk_end - chunk_start)?,
                tx.clone(),
            ));
        }

        let wall_time_start = Instant::now();

        for ((chunk_start, chunk_end), (driver, con, metric, data, tx)) in chunks.zip(inputs) {
            tasks.spawn(async move {
                driver
                    .replace(
                        con,
                        metric,
                        chunk_start + tags_start,
                        chunk_end + tags_start,
                        data,
                        tx,
                    )
                    .await
            });
        }

        let mut timings = Vec::new();
        for result in tasks.join_all().await {
            timings.extend(result.map_err(|e| RunnerError::Driver(Box::new(e)))?);
        }

        let wall_time_s = Instant::now().duration_since(wall_time_start).as_secs_f64();

        drop(tx);
        progress_handle
            .await?
            .map_err(|e| RunnerError::Driver(Box::new(e)))?;

        let report = StepReport::Replace(make_op_report(timings, wall_time_s, num_tasks)?);

        Ok(report)
    }
}

fn make_op_report(
    results: Vec<(usize, Duration)>,
    wall_time_s: f64,
    parallelism: usize,
) -> Result<OpReport, RunnerError> {
    let count = results.iter().fold(0, |acc, (cnt, _dur)| acc + *cnt);
    let busy_time_s = results
        .iter()
        .fold(Duration::default(), |acc, (_cnt, dur)| acc + *dur)
        .as_secs_f64();

    let mut times: Vec<u64> = results
        .iter()
        .map(|(cnt, dur)| (dur.as_nanos() / *cnt as u128) as u64)
        .collect();
    let Percentiles { mean, p90, p99, .. } = percentiles::compute_percentiles(&mut times)?;

    Ok(OpReport {
        parallelism,
        count,
        wall_time_s,
        busy_time_s,
        latency_us_mean: mean / 1000.0,
        latency_us_p90: p90 as f64 / 1000.0,
        latency_us_p99: p99 as f64 / 1000.0,
    })
}

/// Consolidates individual search reports into a single one folded under op_reports.
/// Only the first recall metrics are preserved. Other StepReport variants are passed
/// through, discarding all but the first.
fn consolidate_search_reports(reports: &[StepReport]) -> StepReport {
    if reports.len() == 1 {
        return reports[0].clone();
    }

    let mut report = reports[0].clone();
    let op_reports = if let StepReport::Search(sr) = &mut report {
        &mut sr.op_reports
    } else {
        return report;
    };

    for r in &reports[1..] {
        if let StepReport::Search(sr) = r {
            op_reports.push(sr.op_reports[0].clone());
        } else {
            return report;
        }
    }

    report
}

async fn progress_task<D: Driver>(
    mut rx: mpsc::UnboundedReceiver<usize>,
    count: usize,
) -> Result<(), D::Error> {
    let progress =
        ProgressBar::with_draw_target(Some(count as u64), ProgressDrawTarget::stderr_with_hz(1));
    progress.set_style(
        ProgressStyle::with_template("{wide_bar} {pos}/{len} {elapsed}/{eta} {per_sec}")
            .map_err(|e| ControllerError(Box::new(e)))?,
    );

    while let Some(count) = rx.recv().await {
        progress.inc(count as u64);
    }

    progress.finish_and_clear();

    Ok(())
}

#[derive(Debug, Clone)]
pub struct ChunkIterator {
    size: usize,
    start: usize,
    end: usize,
}

impl ChunkIterator {
    fn new(start: usize, end: usize, count: usize) -> Self {
        let size = (end - start).div_ceil(count);
        Self { size, start, end }
    }

    fn len(&self) -> usize {
        if self.size == 0 {
            return 0;
        }

        (self.end - self.start).div_ceil(self.size)
    }
}

impl Iterator for ChunkIterator {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            return None;
        }

        let end = (self.start + self.size).min(self.end);
        let item = (self.start, end);
        self.start = end;
        Some(item)
    }
}

fn chunk_range(start: usize, end: usize, count: usize) -> ChunkIterator {
    ChunkIterator::new(start, end, count)
}
