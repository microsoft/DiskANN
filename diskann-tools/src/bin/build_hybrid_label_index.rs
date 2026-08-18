/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Build a persisted hybrid label index from provider ID and language TSV columns.

use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
    time::Instant,
};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use diskann_label_index::{encode_hybrid_label_index_postings, HybridBuildOptions};
use roaring::RoaringBitmap;
use serde_json::json;

#[derive(Debug, Parser)]
#[command(
    name = "build_hybrid_label_index",
    about = "Build a dense-head/sparse-tail label index from provider and language columns"
)]
struct Args {
    /// Input TSV containing provider ID in column 3 and language in column 6.
    #[arg(long)]
    input: PathBuf,

    /// Output hybrid label-index file.
    #[arg(long)]
    output: PathBuf,

    /// Override the dense posting cardinality threshold.
    #[arg(long)]
    dense_threshold: Option<u32>,

    /// Progress reporting interval.
    #[arg(long, default_value_t = 10_000_000)]
    progress_rows: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let file =
        File::open(&args.input).with_context(|| format!("opening {}", args.input.display()))?;
    let mut reader = BufReader::with_capacity(16 << 20, file);

    let mut labels = Vec::<String>::new();
    let mut postings = Vec::<RoaringBitmap>::new();
    let mut provider_ids = HashMap::<String, u32>::new();
    let mut language_ids = HashMap::<String, u32>::new();
    let mut line = Vec::with_capacity(1024);
    let mut num_vectors = 0u32;
    let mut missing_rows = 0u64;
    let mut bytes_read = 0u64;
    let started = Instant::now();

    loop {
        line.clear();
        let read = reader
            .read_until(b'\n', &mut line)
            .with_context(|| format!("reading row {}", u64::from(num_vectors) + 1))?;
        if read == 0 {
            break;
        }
        bytes_read += read as u64;
        while line
            .last()
            .is_some_and(|byte| matches!(byte, b'\n' | b'\r'))
        {
            line.pop();
        }

        let (provider, language) = provider_and_language(&line)
            .ok_or_else(|| anyhow!("row {} has fewer than 6 TSV columns", num_vectors))?;
        let provider = std::str::from_utf8(provider)
            .with_context(|| format!("row {num_vectors} provider ID is not UTF-8"))?;
        let language = std::str::from_utf8(language)
            .with_context(|| format!("row {num_vectors} language is not UTF-8"))?;

        if provider.is_empty() && language.is_empty() {
            missing_rows += 1;
        } else {
            if !provider.is_empty() {
                let label_id = label_id(
                    provider,
                    "provider:",
                    &mut provider_ids,
                    &mut labels,
                    &mut postings,
                )?;
                postings[label_id as usize].insert(num_vectors);
            }
            if !language.is_empty() {
                let label_id = label_id(
                    language,
                    "language:",
                    &mut language_ids,
                    &mut labels,
                    &mut postings,
                )?;
                postings[label_id as usize].insert(num_vectors);
            }
        }

        num_vectors = num_vectors.checked_add(1).ok_or_else(|| {
            anyhow!("input contains more rows than the u32 vector-ID space supports")
        })?;
        if args.progress_rows > 0 && num_vectors.is_multiple_of(args.progress_rows) {
            let elapsed = started.elapsed().as_secs_f64();
            eprintln!(
                "rows={num_vectors} labels={} throughput={:.1} MiB/s",
                labels.len(),
                bytes_read as f64 / (1 << 20) as f64 / elapsed
            );
        }
    }

    let stats = encode_hybrid_label_index_postings(
        &args.output,
        num_vectors,
        &labels,
        &postings,
        HybridBuildOptions {
            dense_threshold: args.dense_threshold,
        },
    )
    .with_context(|| format!("writing {}", args.output.display()))?;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "input": args.input,
            "output": args.output,
            "num_vectors": num_vectors,
            "provider_labels": provider_ids.len(),
            "language_labels": language_ids.len(),
            "missing_rows": missing_rows,
            "total_labels": labels.len(),
            "dense_threshold": stats.dense_threshold,
            "dense_labels": stats.dense_labels,
            "sparse_labels": stats.sparse_labels,
            "dense_bytes": stats.dense_bytes,
            "sparse_bytes": stats.sparse_bytes,
            "elapsed_seconds": started.elapsed().as_secs_f64(),
        }))?
    );

    Ok(())
}

fn provider_and_language(line: &[u8]) -> Option<(&[u8], &[u8])> {
    let mut provider = None;
    let mut language = None;
    for (column, value) in line.split(|byte| *byte == b'\t').enumerate() {
        match column {
            2 => provider = Some(value),
            5 => {
                language = Some(value);
                break;
            }
            _ => {}
        }
    }
    Some((provider?, language?))
}

fn label_id(
    value: &str,
    prefix: &str,
    ids: &mut HashMap<String, u32>,
    labels: &mut Vec<String>,
    postings: &mut Vec<RoaringBitmap>,
) -> Result<u32> {
    if let Some(&label_id) = ids.get(value) {
        return Ok(label_id);
    }

    let label_id = u32::try_from(labels.len()).context("label count exceeds u32")?;
    ids.insert(value.to_string(), label_id);
    labels.push(format!("{prefix}{value}"));
    postings.push(RoaringBitmap::new());
    Ok(label_id)
}
