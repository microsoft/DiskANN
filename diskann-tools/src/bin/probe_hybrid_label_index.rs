/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Probe persisted hybrid label queries across the full vector-ID range.

use std::{path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use clap::Parser;
use diskann_label_index::{EncodedLabelIndex, FilterExpressionType};
use serde_json::json;

#[derive(Debug, Parser)]
#[command(
    name = "probe_hybrid_label_index",
    about = "Measure exact hybrid-label query membership over every vector ID"
)]
struct Args {
    /// Persisted hybrid label index.
    #[arg(long)]
    index: PathBuf,

    /// DNF clause to probe. Repeat for multiple queries.
    #[arg(long, required = true)]
    query: Vec<String>,

    /// Number of full scans per query.
    #[arg(long, default_value_t = 3)]
    reps: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.reps > 0, "--reps must be greater than zero");
    let load_started = Instant::now();
    let index = EncodedLabelIndex::load(&args.index)
        .with_context(|| format!("loading {}", args.index.display()))?;
    let load_seconds = load_started.elapsed().as_secs_f64();

    let mut results = Vec::new();
    for clause in &args.query {
        let query = index
            .query(&[clause], FilterExpressionType::DNF)
            .with_context(|| format!("compiling query '{clause}'"))?;
        let mut durations = Vec::with_capacity(args.reps);
        let mut match_count = None;
        for _ in 0..args.reps {
            let started = Instant::now();
            let count = (0..index.num_vectors())
                .filter(|&vec_id| query.is_match(vec_id))
                .count() as u64;
            durations.push(started.elapsed().as_secs_f64());
            if let Some(expected) = match_count {
                anyhow::ensure!(
                    expected == count,
                    "query '{clause}' returned inconsistent counts"
                );
            } else {
                match_count = Some(count);
            }
        }

        let label_metadata = clause
            .split('&')
            .map(|label| {
                let metadata = index.hybrid_label_metadata(label);
                json!({
                    "label": label,
                    "representation": metadata.map(|value| format!("{:?}", value.0)),
                    "cardinality": metadata.map(|value| value.1),
                })
            })
            .collect::<Vec<_>>();
        let mean_seconds = durations.iter().sum::<f64>() / durations.len() as f64;
        results.push(json!({
            "query": clause,
            "labels": label_metadata,
            "match_count": match_count.unwrap_or(0),
            "density_percent": match_count.unwrap_or(0) as f64 / index.num_vectors() as f64 * 100.0,
            "durations_seconds": durations,
            "mean_seconds": mean_seconds,
            "million_probes_per_second": index.num_vectors() as f64 / mean_seconds / 1_000_000.0,
        }));
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "index": args.index,
            "num_vectors": index.num_vectors(),
            "num_labels": index.num_labels(),
            "load_seconds": load_seconds,
            "results": results,
        }))?
    );
    Ok(())
}
