/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use clap::Parser;
use diskann_label_filter::{read_and_parse_queries, read_baselabels};
use diskann_tools::utils::compute_bitmap::compute_query_bitmaps;
use std::fs::File;
use std::io::Write;
use std::process;

#[derive(Debug, Parser)]
#[command(
    about = "Compute specificities for queries against base labels",
    author,
    version
)]
struct Args {
    /// File containing the base labels
    #[arg(long = "base-file-labels", short = 'b')]
    pub base_label_file: String,

    /// File containing the query labels
    #[arg(long = "query-file-labels", short = 'q')]
    pub query_label_file: String,

    /// Output file for specificities (optional)
    #[arg(long = "specificity-output-file", short = 'o')]
    pub specificity_output_file: Option<String>,
}

fn main() {
    let args = Args::parse();

    let base_labels = match read_baselabels(&args.base_label_file) {
        Ok(labels) => labels,
        Err(e) => {
            eprintln!("Error reading base labels: {}", e);
            process::exit(1);
        }
    };

    let total_base = base_labels.len() as u64;
    if total_base == 0 {
        eprintln!("Base labels are empty: cannot compute specificities.");
        process::exit(1);
    }

    let query_labels = match read_and_parse_queries(&args.query_label_file) {
        Ok(queries) => queries,
        Err(e) => {
            eprintln!("Error reading query labels: {}", e);
            process::exit(1);
        }
    };

    let start = std::time::Instant::now();
    let bitmaps = match compute_query_bitmaps(base_labels, query_labels) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error computing bitmaps: {}", e);
            process::exit(1);
        }
    };
    let elapsed = start.elapsed();
    println!("Computing bitmap took {:.3?} seconds", elapsed);

    let mut specificities: Vec<f64> = bitmaps
        .iter()
        .map(|bitmap| {
            let count = bitmap.len();
            count as f64 / total_base as f64
        })
        .collect();

    if let Some(path) = &args.specificity_output_file {
        let mut file = match File::create(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to create output file {}: {}", path, e);
                process::exit(1);
            }
        };
        for spec in &specificities {
            if let Err(e) = writeln!(file, "{:.6}", spec) {
                eprintln!("Failed to write to output file: {}", e);
                process::exit(1);
            }
        }
        println!("Specificities written to {}", path);
    }

    if !specificities.is_empty() {
        specificities.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = specificities[0];
        let max = specificities[specificities.len() - 1];
        let median = if specificities.len().is_multiple_of(2) {
            let mid = specificities.len() / 2;
            (specificities[mid - 1] + specificities[mid]) / 2.0
        } else {
            specificities[specificities.len() / 2]
        };
        let avg = specificities.iter().sum::<f64>() / specificities.len() as f64;
        println!("\nSpecificity stats:");
        println!("  average: {:.6}", avg);
        println!("  median:  {:.6}", median);
        println!("  min:     {:.6}", min);
        println!("  max:     {:.6}", max);
    }
}
