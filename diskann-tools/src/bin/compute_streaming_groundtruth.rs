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
use diskann_benchmark_core::streaming::executors::bigann::{FindGroundtruth, RunBook};
use diskann_benchmark_core::streaming::{self, Executor};
use diskann_providers::storage::{FileStorageProvider, StorageReadProvider};
use diskann_tools::utils::{
    init_subscriber, write_ground_truth, CMDResult, CMDToolError, DataType,
};
use diskann_utils::io::read_bin;
use diskann_utils::views::Matrix;
use diskann_vector::{distance::Metric, DistanceFunction};
use rayon::prelude::*;

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

    let n_base = dataset.nrows();
    let n_queries = queries.nrows();
    let recall_at = args.recall_at as usize;

    tracing::info!(
        "Dataset: {} vectors, Queries: {} vectors, dim: {}, recall@{}",
        n_base,
        n_queries,
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
            Ok(self.dir.join(format!("step{}.{}", stage, self.suffix)))
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
    let distance_fn = V::distance(args.distance_function, Some(dataset.ncols()));

    let mut stream = GroundtruthStream {
        storage: &storage,
        dataset: &dataset,
        queries: &queries,
        distance_fn,
        n_base,
        n_queries,
        recall_at,
        active: vec![false; n_base],
        ext_id: vec![0u32; n_base],
        ext_to_offset: HashMap::new(),
    };

    runbook
        .run_with(&mut stream, |_| Ok(()))
        .map_err(|e| CMDToolError {
            details: e.to_string(),
        })?;

    tracing::info!("Done.");
    Ok(())
}

struct GroundtruthStream<'a, V: VectorRepr + Send + Sync> {
    storage: &'a FileStorageProvider,
    dataset: &'a Matrix<V>,
    queries: &'a Matrix<V>,
    distance_fn: V::Distance,
    n_base: usize,
    n_queries: usize,
    recall_at: usize,
    active: Vec<bool>,
    ext_id: Vec<u32>,
    ext_to_offset: HashMap<u32, usize>,
}

impl<'a, V> streaming::Stream<diskann_benchmark_core::streaming::executors::bigann::Args>
    for GroundtruthStream<'a, V>
where
    V: VectorRepr + Send + Sync,
{
    type Output = ();

    fn search(
        &mut self,
        args: diskann_benchmark_core::streaming::executors::bigann::Search<'_>,
    ) -> anyhow::Result<Self::Output> {
        let timer = Instant::now();
        let n_active = self.active.iter().filter(|&&b| b).count();

        tracing::info!(
            "Computing top-{} groundtruth for {} active vectors against {} queries",
            self.recall_at,
            n_active,
            self.n_queries,
        );

        let active_ids: Vec<usize> = self
            .active
            .iter()
            .enumerate()
            .filter_map(|(i, &on)| if on { Some(i) } else { None })
            .collect();

        // it's generally find to use the global threadpool in diskann-tools
        #[allow(clippy::disallowed_methods)]
        let results: Vec<NeighborPriorityQueue<u32>> = (0..self.n_queries)
            .into_par_iter()
            .map(|qi| {
                let query = self.queries.row(qi);
                let mut pq = NeighborPriorityQueue::new(self.recall_at);
                for &offset in &active_ids {
                    let dist = self
                        .distance_fn
                        .evaluate_similarity(self.dataset.row(offset), query);
                    pq.insert(Neighbor {
                        id: self.ext_id[offset],
                        distance: dist,
                    });
                }
                pq
            })
            .collect();

        let under_k: Vec<usize> = results
            .iter()
            .enumerate()
            .filter_map(|(qi, pq)| (pq.size() < self.recall_at).then_some(qi))
            .collect();
        if !under_k.is_empty() {
            let preview: Vec<usize> = under_k.iter().copied().take(20).collect();
            let suffix = if under_k.len() > preview.len() {
                format!(" (showing first {} query indices)", preview.len())
            } else {
                String::new()
            };
            return Err(anyhow::anyhow!(
                "{}: {} / {} queries have fewer than {} results (active set = {}). Query indices: {:?}{}",
                args.groundtruth.display(),
                under_k.len(),
                self.n_queries,
                self.recall_at,
                n_active,
                preview,
                suffix,
            ));
        }

        write_ground_truth::<()>(
            self.storage,
            args.groundtruth.to_str().ok_or_else(|| {
                anyhow::anyhow!("Non-UTF8 groundtruth path: {}", args.groundtruth.display())
            })?,
            self.n_queries,
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
        for id in args.offsets.clone() {
            if id < self.n_base {
                self.active[id] = true;
                self.ext_id[id] = id as u32;
                self.ext_to_offset.insert(id as u32, id);
            }
        }
        Ok(())
    }

    fn replace(
        &mut self,
        args: diskann_benchmark_core::streaming::executors::bigann::Replace,
    ) -> anyhow::Result<Self::Output> {
        for eid in args.ids.clone() {
            if let Some(&offset) = self.ext_to_offset.get(&(eid as u32)) {
                self.active[offset] = false;
                self.ext_to_offset.remove(&(eid as u32));
            }
        }

        for (offset, eid) in args.offsets.clone().zip(args.ids.clone()) {
            if offset < self.n_base {
                self.active[offset] = true;
                self.ext_id[offset] = eid as u32;
                self.ext_to_offset.insert(eid as u32, offset);
            }
        }

        Ok(())
    }

    fn delete(
        &mut self,
        args: diskann_benchmark_core::streaming::executors::bigann::Delete,
    ) -> anyhow::Result<Self::Output> {
        for eid in args.ids.clone() {
            if let Some(&offset) = self.ext_to_offset.get(&(eid as u32)) {
                self.active[offset] = false;
                self.ext_to_offset.remove(&(eid as u32));
            }
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
