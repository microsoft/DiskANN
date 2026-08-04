/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Computes KNN ground truth for every search stage in a BigANN-style runbook.
//!
//! The tool simulates the insert / replace / delete operations in the runbook,
//! tracking the set of active base-vector IDs at each search stage.  For each
//! search stage it computes the exact top-k nearest neighbours for each query
//! against the currently active set, and writes the result to
//! `<output-dir>/step<stage>.gt<recall_at>` -- the naming convention expected by
//! [`diskann_benchmark_core::streaming::executors::bigann::ScanDirectory`].

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::Context;
use clap::Parser;
use diskann::neighbor::{Neighbor, NeighborPriorityQueue};
use diskann::utils::VectorRepr;
use diskann_benchmark_core::streaming::executors::bigann::{
    FindGroundtruth, RunBook, ScanDirectory,
};
use diskann_benchmark_core::streaming::{self, Executor};
use diskann_providers::storage::{FileStorageProvider, StorageReadProvider};
use diskann_tools::utils::{
    init_subscriber, write_ground_truth, CMDResult, CMDToolError, DataType,
};
use diskann_utils::io::read_bin;
use diskann_utils::views::Matrix;
use diskann_vector::{distance::Metric, DistanceFunction};
use rayon::prelude::*;

trait GroundtruthDistance: Send + Sync {
    fn n_base(&self) -> usize;

    fn n_queries(&self) -> usize;

    fn distance(&self, query: usize, internal_id: usize) -> anyhow::Result<f32>;
}

struct MatrixDistance<'a, V: VectorRepr + Send + Sync> {
    dataset: &'a Matrix<V>,
    queries: &'a Matrix<V>,
    distance_fn: V::Distance,
}

impl<'a, V> GroundtruthDistance for MatrixDistance<'a, V>
where
    V: VectorRepr + Send + Sync,
{
    fn n_base(&self) -> usize {
        self.dataset.nrows()
    }

    fn n_queries(&self) -> usize {
        self.queries.nrows()
    }

    fn distance(&self, query: usize, internal_id: usize) -> anyhow::Result<f32> {
        if query >= self.n_queries() {
            return Err(anyhow::anyhow!("query index {} out of bounds", query));
        }
        if internal_id >= self.n_base() {
            return Err(anyhow::anyhow!("internal id {} out of bounds", internal_id));
        }

        let query_row = self.queries.row(query);
        let data_row = self.dataset.row(internal_id);
        Ok(self.distance_fn.evaluate_similarity(data_row, query_row))
    }
}

fn compute_groundtruth_results(
    distance: &dyn GroundtruthDistance,
    active_entries: &[(u32, usize)],
    recall_at: usize,
) -> anyhow::Result<Vec<NeighborPriorityQueue<u32>>> {
    // using the global threadpool is generally fine in diskann-tools
    #[allow(clippy::disallowed_methods)]
    let results = (0..distance.n_queries())
        .into_par_iter()
        .map(|query_id| {
            let mut pq = NeighborPriorityQueue::new(recall_at);
            let query_result: anyhow::Result<()> =
                active_entries
                    .iter()
                    .try_for_each(|&(external_id, internal_id)| {
                        let dist = distance.distance(query_id, internal_id)?;
                        pq.insert(Neighbor::new(external_id, dist));
                        Ok(())
                    });
            query_result.map(|()| pq)
        })
        .collect::<Vec<anyhow::Result<NeighborPriorityQueue<u32>>>>();

    results.into_iter().collect()
}

fn main() -> CMDResult<()> {
    init_subscriber();
    let args = Args::parse();
    match args.data_type {
        DataType::Float => run::<f32>(&args),
        DataType::Fp16 => run::<diskann_vector::Half>(&args),
        DataType::Uint8 => run::<u8>(&args),
        DataType::Int8 => run::<i8>(&args),
    }
}

fn run<V: VectorRepr + Send + Sync>(args: &Args) -> CMDResult<()> {
    let storage = FileStorageProvider;

    tracing::info!("Loading dataset from {}", args.base_file);
    let dataset =
        read_bin::<V>(&mut storage.open_reader(&args.base_file)?).map_err(|e| CMDToolError {
            details: e.to_string(),
        })?;

    tracing::info!("Loading queries from {}", args.query_file);
    let queries =
        read_bin::<V>(&mut storage.open_reader(&args.query_file)?).map_err(|e| CMDToolError {
            details: e.to_string(),
        })?;

    let recall_at = args.recall_at as usize;

    tracing::info!(
        "Dataset: {} vectors, Queries: {} vectors, dim: {}, recall@{}",
        dataset.nrows(),
        queries.nrows(),
        dataset.ncols(),
        recall_at,
    );

    let output_dir = Path::new(&args.output_dir);
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("creating output directory {}", output_dir.display()))
        .map_err(|e| CMDToolError {
            details: e.to_string(),
        })?;

    let gt_suffix = format!("gt{}", recall_at);

    // FindGroundtruth impl that always returns the expected output path whether
    // or not it exists yet -- we are about to generate the files.
    struct AllowMissing {
        dir: PathBuf,
        suffix: String,
    }
    impl FindGroundtruth for AllowMissing {
        fn find_groundtruth(&mut self, stage: usize) -> anyhow::Result<PathBuf> {
            Ok(self
                .dir
                .join(ScanDirectory::groundtruth_filename(stage, &self.suffix)))
        }
    }

    let mut finder = AllowMissing {
        dir: output_dir.to_path_buf(),
        suffix: gt_suffix,
    };

    tracing::info!(
        "Parsing runbook {} for dataset \"{}\"",
        args.runbook_file,
        args.dataset_name
    );
    let mut runbook = RunBook::load(
        Path::new(&args.runbook_file),
        &args.dataset_name,
        &mut finder,
    )
    .map_err(|e| CMDToolError {
        details: e.to_string(),
    })?;

    tracing::info!("Runbook has {} stages", runbook.len());
    let mut stream = GroundtruthStream {
        storage: &storage,
        distance: Box::new(MatrixDistance {
            dataset: &dataset,
            queries: &queries,
            distance_fn: V::distance(args.distance_function, Some(dataset.ncols())),
        }),
        recall_at,
        external_to_internal: HashMap::new(),
    };

    runbook
        .run_with(&mut stream, |_| Ok(()))
        .map_err(|e| CMDToolError {
            details: e.to_string(),
        })?;

    tracing::info!("Done.");
    Ok(())
}

struct GroundtruthStream<'a> {
    storage: &'a FileStorageProvider,
    distance: Box<dyn GroundtruthDistance + 'a>,
    recall_at: usize,
    external_to_internal: HashMap<u32, usize>,
}

impl<'a> GroundtruthStream<'a> {
    fn remove_active_external_id(&mut self, external_id: u32) {
        self.external_to_internal.remove(&external_id);
    }
}

impl<'a> streaming::Stream<diskann_benchmark_core::streaming::executors::bigann::Args>
    for GroundtruthStream<'a>
{
    type Output = ();

    fn search(
        &mut self,
        args: diskann_benchmark_core::streaming::executors::bigann::Search<'_>,
    ) -> anyhow::Result<Self::Output> {
        let timer = Instant::now();
        let n_active = self.external_to_internal.len();

        if n_active < self.recall_at {
            return Err(anyhow::anyhow!(
                "{}: active set has {} vectors, which is fewer than recall_at={} required to compute groundtruth",
                args.groundtruth.display(),
                n_active,
                self.recall_at,
            ));
        }

        tracing::info!(
            "Computing top-{} groundtruth for {} active vectors against {} queries",
            self.recall_at,
            n_active,
            self.distance.n_queries(),
        );

        let active_entries: Vec<(u32, usize)> = self
            .external_to_internal
            .iter()
            .map(|(external_id, internal_id)| (*external_id, *internal_id))
            .collect();

        let results =
            compute_groundtruth_results(self.distance.as_ref(), &active_entries, self.recall_at)?;

        write_ground_truth::<()>(
            self.storage,
            args.groundtruth.to_str().ok_or_else(|| {
                anyhow::anyhow!("Non-UTF8 groundtruth path: {}", args.groundtruth.display())
            })?,
            self.distance.n_queries(),
            self.recall_at,
            results,
            None,
        )
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        tracing::info!(
            "Groundtruth written to {} in {:?}",
            args.groundtruth.display(),
            timer.elapsed(),
        );

        Ok(())
    }

    fn insert(
        &mut self,
        args: diskann_benchmark_core::streaming::executors::bigann::Insert,
    ) -> anyhow::Result<Self::Output> {
        for internal_id in args.offsets.clone() {
            if internal_id < self.distance.n_base() {
                self.external_to_internal
                    .insert(internal_id as u32, internal_id);
            }
        }
        Ok(())
    }

    fn replace(
        &mut self,
        args: diskann_benchmark_core::streaming::executors::bigann::Replace,
    ) -> anyhow::Result<Self::Output> {
        for external_id in args.ids.clone() {
            self.remove_active_external_id(external_id as u32);
        }

        for (internal_id, external_id) in args.offsets.clone().zip(args.ids.clone()) {
            if internal_id < self.distance.n_base() {
                self.external_to_internal
                    .insert(external_id as u32, internal_id);
            }
        }

        Ok(())
    }

    fn delete(
        &mut self,
        args: diskann_benchmark_core::streaming::executors::bigann::Delete,
    ) -> anyhow::Result<Self::Output> {
        for external_id in args.ids.clone() {
            self.remove_active_external_id(external_id as u32);
        }
        Ok(())
    }

    fn maintain(&mut self, _args: ()) -> anyhow::Result<Self::Output> {
        Ok(())
    }

    fn needs_maintenance(&mut self) -> bool {
        false
    }
}

#[derive(Debug, Parser)]
struct Args {
    /// Data type of the base and query vectors.
    #[arg(long = "data-type", default_value = "float")]
    pub data_type: DataType,

    /// Distance function to use.
    #[arg(long = "dist-fn", default_value = "l2")]
    pub distance_function: Metric,

    /// File containing the full base dataset in binary format.
    #[arg(long = "base-file", short, required = true)]
    pub base_file: String,

    /// File containing the query vectors in binary format.
    #[arg(long = "query-file", short, required = true)]
    pub query_file: String,

    /// Path to the BigANN runbook YAML file.
    #[arg(long = "runbook-file", required = true)]
    pub runbook_file: String,

    /// Dataset name within the runbook YAML file.
    #[arg(long = "dataset-name", required = true)]
    pub dataset_name: String,

    /// Number of nearest neighbours to compute per query (k).
    ///
    /// Output files are named step<stage>.gt<recall_at>.
    #[arg(long = "recall-at", short = 'K', required = true)]
    pub recall_at: u32,

    /// Directory to write the groundtruth files into.
    ///
    /// Files are written as `step<stage>.gt<recall_at>`, matching the
    /// naming convention expected by `ScanDirectory`.
    #[arg(long = "gt-dir", short, required = true)]
    pub output_dir: String,
}
