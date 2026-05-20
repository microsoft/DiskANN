/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_label_filter::{read_and_parse_queries, read_baselabels, eval_query_expr};
use rayon::prelude::*;
use std::env;
use std::process;
use std::fs::File;
use std::io::Write;

use crate::utils::compute_bitmaps::read_labels_and_compute_bitmap;

pub fn read_labels_and_compute_bitmap_naive(
    base_label_filename: &str,
    query_label_filename: &str,
) -> Result<Vec<BitSet>, anyhow::Error> {
    // Read base labels
    let base_labels = read_baselabels(base_label_filename)?;

    // Parse queries and evaluate against labels
    let parsed_queries = read_and_parse_queries(query_label_filename)?;

    // using the global threadpool is fine here
    #[allow(clippy::disallowed_methods)]
    let query_bitmaps: Vec<BitSet> = parsed_queries
        .par_iter()
        .map(|(_query_id, query_expr)| {
            let mut bitmap = BitSet::new();
            for base_label in base_labels.iter() {
                if eval_query_expr(query_expr, &base_label.label) {
                    bitmap.insert(base_label.doc_id);
                }
            }
            bitmap
        })
        .collect();

    Ok(query_bitmaps)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 && args.len() != 4 {
        eprintln!("Usage: {} <base_label_file> <query_label_file> [specificity_output_file]", args[0]);
        process::exit(1);
    }
    let base_label_file = &args[1];
    let query_label_file = &args[2];
    let output_file = if args.len() == 4 { Some(&args[3]) } else { None };

    let bitmaps = match read_labels_and_compute_bitmap_naive(base_label_file, query_label_file) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error computing bitmaps: {}", e);
            process::exit(1);
        }
    };

    let specificities: Vec<f64> = bitmaps
        .par_iter()
        .map(|bitmap| {
            let count = bitmap.len();
            let specificity = count as f64 / total_base as f64;
            specificity
        })
        .collect();

    // let mut specificities = Vec::with_capacity(results.len());
    // for (query_id, count, specificity) in &results {
    //     specificities.push(*specificity);
    //     println!("query_id {}: {} matches (specificity {:.6})", query_id, count, specificity);
    // }

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
