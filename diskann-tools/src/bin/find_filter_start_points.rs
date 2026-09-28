/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! For each query, find one base-dataset point that satisfies the query's filter
//! predicate and write the result in groundtruth binary format (K = 1).
//!
//! The output file has the standard DiskANN groundtruth layout:
//!   int32  num_queries
//!   int32  k  (always 1)
//!   int32[num_queries]  base-point IDs  (one per query)
//!   float32[num_queries]  distances     (written as 0.0 – not a real distance)
//!
//! For queries where no base point satisfies the filter, `--fallback-id` is used
//! (default 0).  A summary of how many queries needed the fallback is printed at
//! the end.

use std::{
    fs::File,
    io::{BufWriter, Write},
    process,
};

use clap::Parser;
use diskann_label_filter::{read_and_parse_queries, read_baselabels};
use diskann_tools::utils::compute_bitmap::compute_query_bitmaps;

#[derive(Debug, Parser)]
#[command(
    about = "For each query find one base point satisfying its filter; \
             write results in groundtruth binary format (K=1)",
    author,
    version
)]
struct Args {
    /// JSONL file containing base labels (one document per line with a `doc_id` field)
    #[arg(long = "base-file-labels", short = 'b')]
    base_label_file: String,

    /// JSONL file containing query labels (one document per line with a `query_id`
    /// and a `filter` field)
    #[arg(long = "query-file-labels", short = 'q')]
    query_label_file: String,

    /// Output file path (groundtruth binary format, K=1)
    #[arg(long = "output-file", short = 'o')]
    output_file: String,

    /// Base-point ID to use for queries that have no matching base point.
    /// Defaults to 0.
    #[arg(long = "fallback-id", default_value_t = 0u32)]
    fallback_id: u32,
}

fn main() {
    let args = Args::parse();

    // ── Load labels ──────────────────────────────────────────────────────────
    let base_labels = match read_baselabels(&args.base_label_file) {
        Ok(l) => l,
        Err(e) => {
            eprintln!(
                "Error reading base labels from {}: {}",
                args.base_label_file, e
            );
            process::exit(1);
        }
    };

    let num_base = base_labels.len();
    if num_base == 0 {
        eprintln!("Base labels file is empty.");
        process::exit(1);
    }
    println!("Loaded {} base documents.", num_base);

    let query_labels = match read_and_parse_queries(&args.query_label_file) {
        Ok(q) => q,
        Err(e) => {
            eprintln!(
                "Error reading query labels from {}: {}",
                args.query_label_file, e
            );
            process::exit(1);
        }
    };

    let num_queries = query_labels.len();
    if num_queries == 0 {
        eprintln!("Query labels file is empty.");
        process::exit(1);
    }
    println!("Loaded {} queries.", num_queries);

    // ── Compute per-query bitmaps ─────────────────────────────────────────────
    let start = std::time::Instant::now();
    let bitmaps = match compute_query_bitmaps(base_labels, query_labels) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error computing bitmaps: {}", e);
            process::exit(1);
        }
    };
    println!("Bitmap computation took {:.3?}", start.elapsed());

    // ── Pick one matching base point per query ────────────────────────────────
    let mut start_point_ids: Vec<u32> = Vec::with_capacity(num_queries);
    let mut fallback_count: usize = 0;

    for bitmap in &bitmaps {
        // `BitSet::iter()` yields matching base-point indices in ascending order.
        if let Some(id) = bitmap.iter().next() {
            start_point_ids.push(id as u32);
        } else {
            start_point_ids.push(args.fallback_id);
            fallback_count += 1;
        }
    }

    println!(
        "Found matching start points for {}/{} queries ({} used fallback id {}).",
        num_queries - fallback_count,
        num_queries,
        fallback_count,
        args.fallback_id,
    );

    // ── Write groundtruth file (K = 1) ────────────────────────────────────────
    // Layout:
    //   [int32  num_queries] [int32  k=1]
    //   [int32 × num_queries  ids]
    //   [float32 × num_queries  distances]  (all 0.0)
    let file = match File::create(&args.output_file) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to create output file {}: {}", args.output_file, e);
            process::exit(1);
        }
    };
    let mut writer = BufWriter::new(file);

    // Header: num_queries (int32) followed by k=1 (int32)
    writer
        .write_all(&(num_queries as u32).to_le_bytes())
        .and_then(|_| writer.write_all(&1u32.to_le_bytes()))
        .unwrap_or_else(|e| {
            eprintln!("Failed to write header: {}", e);
            process::exit(1);
        });

    // IDs: one int32 per query
    for id in &start_point_ids {
        if let Err(e) = writer.write_all(&id.to_le_bytes()) {
            eprintln!("Failed to write IDs: {}", e);
            process::exit(1);
        }
    }

    // Distances: one float32 per query, all 0.0
    let zero_dist: f32 = 0.0;
    for _ in 0..num_queries {
        if let Err(e) = writer.write_all(&zero_dist.to_le_bytes()) {
            eprintln!("Failed to write distances: {}", e);
            process::exit(1);
        }
    }

    if let Err(e) = writer.flush() {
        eprintln!("Failed to flush output file: {}", e);
        process::exit(1);
    }

    println!(
        "Written {} start points to {} (groundtruth format, K=1).",
        num_queries, args.output_file
    );
}
