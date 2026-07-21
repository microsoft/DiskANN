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
use diskann_benchmark_core::streaming::executors::bigann::{FindGroundtruth, RunBook, Stage};
use diskann::neighbor::{Neighbor, NeighborPriorityQueue};
use diskann::utils::VectorRepr;
use diskann_providers::storage::{FileStorageProvider, StorageReadProvider};
use diskann_tools::utils::{
    init_subscriber, write_ground_truth, CMDResult, CMDToolError, DataType,
};
use diskann_utils::io::read_bin;
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
    let runbook = RunBook::load(
        Path::new(&args.runbook_file),
        &args.dataset_name,
        &mut finder,
    )
    .map_err(|e| CMDToolError {
        details: e.to_string(),
    })?;

    tracing::info!("Runbook has {} stages", runbook.len());

    // Boolean active-vector mask indexed by base offset.
    let mut active: Vec<bool> = vec![false; n_base];
    // For each active dataset offset, the external ID that should appear in the groundtruth.
    // For plain inserts external_id == dataset_offset; for Replace they diverge.
    let mut ext_id: Vec<u32> = vec![0u32; n_base];
    // Reverse map: external_id -> dataset_offset, needed to resolve Delete/Replace removals.
    let mut ext_to_offset: HashMap<u32, usize> = HashMap::new();

    println!("Using distance function: {:?}", args.distance_function);

    let distance_fn = V::distance(args.distance_function, Some(dataset.ncols()));

    for (stage_idx, stage) in runbook.stages().iter().enumerate() {
        match stage {
            Stage::Insert {
                dataset_offsets_and_ids,
            } => {
                for id in dataset_offsets_and_ids.clone() {
                    if id < n_base {
                        active[id] = true;
                        ext_id[id] = id as u32;
                        ext_to_offset.insert(id as u32, id);
                    }
                }
                tracing::info!(
                    "Stage {}: insert {}..{} ({} active)",
                    stage_idx,
                    dataset_offsets_and_ids.start,
                    dataset_offsets_and_ids.end,
                    active.iter().filter(|&&b| b).count(),
                );
            }
            Stage::Delete { ids } => {
                for eid in ids.clone() {
                    if let Some(&offset) = ext_to_offset.get(&(eid as u32)) {
                        active[offset] = false;
                        ext_to_offset.remove(&(eid as u32));
                    }
                }
                tracing::info!(
                    "Stage {}: delete {}..{} ({} active)",
                    stage_idx,
                    ids.start,
                    ids.end,
                    active.iter().filter(|&&b| b).count(),
                );
            }
            Stage::Replace {
                dataset_offsets,
                ids,
            } => {
                // Remove old vectors by external ID.
                for eid in ids.clone() {
                    if let Some(&offset) = ext_to_offset.get(&(eid as u32)) {
                        active[offset] = false;
                        ext_to_offset.remove(&(eid as u32));
                    }
                }
                // Insert new vectors: they inherit the external IDs from `ids`.
                for (offset, eid) in dataset_offsets.clone().zip(ids.clone()) {
                    if offset < n_base {
                        active[offset] = true;
                        ext_id[offset] = eid as u32;
                        ext_to_offset.insert(eid as u32, offset);
                    }
                }
                tracing::info!(
                    "Stage {}: replace ids {}..{} with offsets {}..{} ({} active)",
                    stage_idx,
                    ids.start,
                    ids.end,
                    dataset_offsets.start,
                    dataset_offsets.end,
                    active.iter().filter(|&&b| b).count(),
                );
            }
            Stage::Search {
                groundtruth: output_path,
            } => {
                let timer = Instant::now();
                let n_active = active.iter().filter(|&&b| b).count();

                tracing::info!(
                    "Stage {}: computing top-{} groundtruth for {} active vectors against {} queries",
                    stage_idx, recall_at, n_active, n_queries,
                );

                let active_ids: Vec<usize> = active
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &on)| if on { Some(i) } else { None })
                    .collect();

                // Compute KNN in parallel over queries.
                // Use ext_id[offset] as the neighbor ID so that groundtruth IDs
                // match external IDs (which differ from dataset offsets after Replace).

                // Using the global threadpool is fine here
                #[allow(clippy::disallowed_methods)]
                let results: Vec<NeighborPriorityQueue<u32>> = (0..n_queries)
                    .into_par_iter()
                    .map(|qi| {
                        let query = queries.row(qi);
                        let mut pq = NeighborPriorityQueue::new(recall_at);
                        for &offset in &active_ids {
                            let dist = distance_fn.evaluate_similarity(dataset.row(offset), query);
                            pq.insert(Neighbor {
                                id: ext_id[offset],
                                distance: dist,
                            });
                        }
                        pq
                    })
                    .collect();

                // Warn about queries that got fewer than K results (active set smaller than K).
                let under_k: Vec<usize> = results
                    .iter()
                    .enumerate()
                    .filter_map(|(qi, pq)| {
                        if pq.size() < recall_at {
                            Some(qi)
                        } else {
                            None
                        }
                    })
                    .collect();
                if !under_k.is_empty() {
                    tracing::warn!(
                        "Stage {}: {} / {} queries have fewer than {} results (active set = {}). \
                         Query indices: {:?}",
                        stage_idx,
                        under_k.len(),
                        n_queries,
                        recall_at,
                        n_active,
                        under_k,
                    );
                }

                write_ground_truth::<()>(
                    &storage,
                    output_path.to_str().ok_or_else(|| CMDToolError {
                        details: format!("Non-UTF8 output path: {}", output_path.display()),
                    })?,
                    n_queries,
                    recall_at,
                    results,
                    None,
                )
                .map_err(|e| CMDToolError {
                    details: e.to_string(),
                })?;

                tracing::info!(
                    "Stage {}: groundtruth written to {} in {:?}",
                    stage_idx,
                    output_path.display(),
                    timer.elapsed(),
                );
            }
        }
    }

    tracing::info!("Done.");
    Ok(())
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
