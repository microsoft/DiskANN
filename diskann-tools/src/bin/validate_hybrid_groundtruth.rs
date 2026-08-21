/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Validate that every variable-length ground-truth ID satisfies its hybrid filter.

use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    path::PathBuf,
};

use anyhow::{bail, Context, Result};
use clap::Parser;
use diskann_label_index::{EncodedLabelIndex, FilterExpressionType};
use serde_json::Value;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    index: PathBuf,
    #[arg(long)]
    predicates: PathBuf,
    #[arg(long)]
    groundtruth: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let index = EncodedLabelIndex::load(&args.index)?;
    let clauses = load_clauses(&args.predicates)?;
    let groundtruth = load_groundtruth(&args.groundtruth)?;
    if clauses.len() != groundtruth.len() {
        bail!(
            "{} predicates do not match {} ground-truth rows",
            clauses.len(),
            groundtruth.len()
        );
    }

    let mut mismatches = 0u64;
    let mut total = 0u64;
    let mut affected_queries = 0u64;
    for (clause, ids) in clauses.iter().zip(&groundtruth) {
        let query = index.query(&[clause], FilterExpressionType::DNF)?;
        let query_mismatches = ids.iter().filter(|&&id| !query.is_match(id)).count() as u64;
        if query_mismatches > 0 {
            affected_queries += 1;
            mismatches += query_mismatches;
        }
        total += ids.len() as u64;
    }

    println!(
        "{{\"queries\":{},\"truth_ids\":{},\"mismatches\":{},\"affected_queries\":{}}}",
        clauses.len(),
        total,
        mismatches,
        affected_queries
    );
    Ok(())
}

fn load_clauses(path: &PathBuf) -> Result<Vec<String>> {
    let reader = BufReader::new(File::open(path)?);
    reader
        .lines()
        .map(|line| {
            let value: Value = serde_json::from_str(&line?)?;
            let children = value["filter"]["$and"]
                .as_array()
                .context("predicate filter must contain $and array")?;
            let labels = children
                .iter()
                .map(|child| {
                    child
                        .as_object()
                        .and_then(|object| object.keys().next())
                        .cloned()
                        .context("predicate child must contain one label")
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(labels.join("&"))
        })
        .collect()
}

fn load_groundtruth(path: &PathBuf) -> Result<Vec<Vec<u32>>> {
    let mut reader = BufReader::new(File::open(path)?);
    let query_count = read_u32(&mut reader)? as usize;
    let total_results = read_u32(&mut reader)? as usize;
    let mut sizes = Vec::with_capacity(query_count);
    for _ in 0..query_count {
        sizes.push(read_u32(&mut reader)? as usize);
    }
    let mut ids = Vec::with_capacity(total_results);
    for _ in 0..total_results {
        ids.push(read_u32(&mut reader)?);
    }
    let mut offset = 0usize;
    let rows = sizes
        .into_iter()
        .map(|size| {
            let row = ids[offset..offset + size].to_vec();
            offset += size;
            row
        })
        .collect::<Vec<_>>();
    if offset != ids.len() {
        bail!("ground-truth sizes do not consume all IDs");
    }
    Ok(rows)
}

fn read_u32(reader: &mut impl Read) -> std::io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}
