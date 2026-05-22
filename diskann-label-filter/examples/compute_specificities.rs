/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_label_filter::{compute_query_bitmaps, read_and_parse_queries, read_baselabels};
use rayon::prelude::*;
use std::env;
use std::fs::File;
use std::io::Write;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 && args.len() != 4 {
        eprintln!(
            "Usage: {} <base_label_file> <query_label_file> [specificity_output_file]",
            args[0]
        );
        process::exit(1);
    }
    let base_label_file = &args[1];
    let query_label_file = &args[2];
    let output_file = if args.len() == 4 {
        Some(&args[3])
    } else {
        None
    };

    let base_labels = match read_baselabels(base_label_file) {
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

    let query_labels = match read_and_parse_queries(query_label_file) {
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
    println!("read_labels_and_compute_bitmap_naive took {:.3?}", elapsed);

    let mut specificities: Vec<f64> = bitmaps
        .par_iter()
        .map(|bitmap| {
            let count = bitmap.len();
            let specificity = count as f64 / total_base as f64;
            specificity
        })
        .collect();

    if let Some(path) = output_file {
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
        let median = if specificities.len() % 2 == 0 {
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
