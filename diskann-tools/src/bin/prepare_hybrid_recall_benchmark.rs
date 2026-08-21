/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Prepare uint8 vectors, filtered queries, predicates, and range ground truth from TSV files.

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    hash::{Hash, Hasher},
    io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write},
    path::PathBuf,
    time::Instant,
};

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use serde_json::{json, Map, Value};

#[derive(Debug, Parser)]
#[command(
    name = "prepare_hybrid_recall_benchmark",
    about = "Prepare a provider-and-language filtered ANN benchmark from TSV inputs"
)]
struct Args {
    /// Truth-only query TSV.
    #[arg(long)]
    truth_tsv: PathBuf,

    /// 100M base TSV.
    #[arg(long)]
    base_tsv: PathBuf,

    /// Output DiskANN uint8 base vectors.
    #[arg(long)]
    base_out: PathBuf,

    /// Output deduplicated DiskANN uint8 query vectors.
    #[arg(long)]
    queries_out: PathBuf,

    /// Output provider-and-language query predicates.
    #[arg(long)]
    predicates_out: PathBuf,

    /// Output variable-length filtered ground truth.
    #[arg(long)]
    groundtruth_out: PathBuf,

    /// Output query metadata TSV.
    #[arg(long)]
    metadata_out: PathBuf,

    /// Progress reporting interval for base conversion.
    #[arg(long, default_value_t = 10_000_000)]
    progress_rows: u32,
}

#[derive(Debug, Clone, Eq)]
struct QueryKey {
    text: String,
    provider: String,
    language: String,
}

impl PartialEq for QueryKey {
    fn eq(&self, other: &Self) -> bool {
        self.text == other.text
            && self.provider == other.provider
            && self.language == other.language
    }
}

impl Hash for QueryKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.text.hash(state);
        self.provider.hash(state);
        self.language.hash(state);
    }
}

#[derive(Debug)]
struct QueryData {
    key: QueryKey,
    embedding: Vec<u8>,
    truth_external_ids: Vec<u64>,
    seen_truth_ids: HashSet<u64>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let started = Instant::now();
    let (queries, mut external_to_internal) = load_truth_queries(&args.truth_tsv)?;
    eprintln!(
        "deduplicated {} truth rows into {} filtered queries over {} truth IDs",
        queries
            .iter()
            .map(|query| query.truth_external_ids.len())
            .sum::<usize>(),
        queries.len(),
        external_to_internal.len()
    );

    let dimension = write_base_vectors(
        &args.base_tsv,
        &args.base_out,
        args.progress_rows,
        &mut external_to_internal,
    )?;
    let missing = external_to_internal
        .iter()
        .filter(|(_, internal_id)| **internal_id == u32::MAX)
        .map(|(external_id, _)| *external_id)
        .take(10)
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!("truth IDs were not found in the base TSV; examples: {missing:?}");
    }

    write_queries(&args.queries_out, &queries, dimension)?;
    write_predicates(&args.predicates_out, &queries)?;
    write_groundtruth(&args.groundtruth_out, &queries, &external_to_internal)?;
    write_metadata(&args.metadata_out, &queries)?;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "base_tsv": args.base_tsv,
            "truth_tsv": args.truth_tsv,
            "base_out": args.base_out,
            "queries_out": args.queries_out,
            "predicates_out": args.predicates_out,
            "groundtruth_out": args.groundtruth_out,
            "metadata_out": args.metadata_out,
            "num_queries": queries.len(),
            "num_truth_ids": external_to_internal.len(),
            "dimension": dimension,
            "elapsed_seconds": started.elapsed().as_secs_f64(),
        }))?
    );
    Ok(())
}

fn load_truth_queries(path: &PathBuf) -> Result<(Vec<QueryData>, HashMap<u64, u32>)> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = BufReader::with_capacity(4 << 20, file);
    let mut line = Vec::with_capacity(1024);
    let mut queries = Vec::<QueryData>::new();
    let mut query_ids = HashMap::<QueryKey, usize>::new();
    let mut external_to_internal = HashMap::<u64, u32>::new();
    let mut row = 0u64;

    loop {
        line.clear();
        let read = reader.read_until(b'\n', &mut line)?;
        if read == 0 {
            break;
        }
        trim_line(&mut line);
        let fields = split_columns(&line, 7)
            .ok_or_else(|| anyhow!("truth row {row} has fewer than 7 columns"))?;
        let text = std::str::from_utf8(fields[0])?.to_string();
        let truth_external_id = std::str::from_utf8(fields[1])?
            .parse::<u64>()
            .with_context(|| format!("truth row {row} has invalid external ID"))?;
        let label = std::str::from_utf8(fields[4])?;
        let (language, provider) = parse_language_provider(label)
            .with_context(|| format!("truth row {row} has invalid label '{label}'"))?;
        let embedding = parse_embedding(fields[5])
            .with_context(|| format!("truth row {row} has invalid embedding"))?;
        let key = QueryKey {
            text,
            provider: provider.to_string(),
            language: language.to_string(),
        };

        let query_id = if let Some(&query_id) = query_ids.get(&key) {
            if queries[query_id].embedding != embedding {
                bail!("truth row {row} changes the embedding for an existing filtered query");
            }
            query_id
        } else {
            let query_id = queries.len();
            query_ids.insert(key.clone(), query_id);
            queries.push(QueryData {
                key,
                embedding,
                truth_external_ids: Vec::new(),
                seen_truth_ids: HashSet::new(),
            });
            query_id
        };

        if queries[query_id].seen_truth_ids.insert(truth_external_id) {
            queries[query_id].truth_external_ids.push(truth_external_id);
        }
        external_to_internal
            .entry(truth_external_id)
            .or_insert(u32::MAX);
        row += 1;
    }
    Ok((queries, external_to_internal))
}

fn write_base_vectors(
    input_path: &PathBuf,
    output_path: &PathBuf,
    progress_rows: u32,
    external_to_internal: &mut HashMap<u64, u32>,
) -> Result<u32> {
    let input =
        File::open(input_path).with_context(|| format!("opening {}", input_path.display()))?;
    let mut reader = BufReader::with_capacity(16 << 20, input);
    let output =
        File::create(output_path).with_context(|| format!("creating {}", output_path.display()))?;
    let mut writer = BufWriter::with_capacity(16 << 20, output);
    write_u32(&mut writer, 0)?;
    write_u32(&mut writer, 0)?;

    let mut line = Vec::with_capacity(1024);
    let mut embedding = Vec::<u8>::with_capacity(128);
    let mut row = 0u32;
    let mut dimension = None::<u32>;
    let started = Instant::now();
    let mut bytes_read = 0u64;
    loop {
        line.clear();
        let read = reader.read_until(b'\n', &mut line)?;
        if read == 0 {
            break;
        }
        bytes_read += read as u64;
        trim_line(&mut line);
        let (external_id, embedding_bytes) = first_two_columns(&line)
            .ok_or_else(|| anyhow!("base row {row} has fewer than 2 columns"))?;
        let external_id = std::str::from_utf8(external_id)?
            .parse::<u64>()
            .with_context(|| format!("base row {row} has invalid external ID"))?;
        parse_embedding_into(embedding_bytes, &mut embedding)
            .with_context(|| format!("base row {row} has invalid embedding"))?;
        let current_dimension = u32::try_from(embedding.len()).context("dimension exceeds u32")?;
        match dimension {
            None => dimension = Some(current_dimension),
            Some(expected) if expected != current_dimension => {
                bail!("base row {row} has dimension {current_dimension}; expected {expected}");
            }
            Some(_) => {}
        }
        writer.write_all(&embedding)?;

        if let Some(internal_id) = external_to_internal.get_mut(&external_id) {
            if *internal_id != u32::MAX {
                bail!("base TSV contains duplicate external ID {external_id}");
            }
            *internal_id = row;
        }
        row = row
            .checked_add(1)
            .ok_or_else(|| anyhow!("base row count exceeds u32"))?;
        if progress_rows > 0 && row.is_multiple_of(progress_rows) {
            eprintln!(
                "rows={row} throughput={:.1} MiB/s",
                bytes_read as f64 / (1 << 20) as f64 / started.elapsed().as_secs_f64()
            );
        }
    }

    let dimension = dimension.ok_or_else(|| anyhow!("base TSV contains no rows"))?;
    writer.flush()?;
    writer.seek(SeekFrom::Start(0))?;
    write_u32(&mut writer, row)?;
    write_u32(&mut writer, dimension)?;
    writer.flush()?;
    Ok(dimension)
}

fn write_queries(path: &PathBuf, queries: &[QueryData], dimension: u32) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    write_u32(
        &mut writer,
        u32::try_from(queries.len()).context("query count exceeds u32")?,
    )?;
    write_u32(&mut writer, dimension)?;
    for query in queries {
        if query.embedding.len() != dimension as usize {
            bail!("query embedding dimension does not match base vectors");
        }
        writer.write_all(&query.embedding)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_predicates(path: &PathBuf, queries: &[QueryData]) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    for (query_id, query) in queries.iter().enumerate() {
        let mut language = Map::new();
        language.insert(
            format!("language:{}", query.key.language),
            json!({"$eq": true}),
        );
        let mut provider = Map::new();
        provider.insert(
            format!("provider:{}", query.key.provider),
            json!({"$eq": true}),
        );
        let row = json!({
            "query_id": query_id,
            "filter": {
                "$and": [Value::Object(language), Value::Object(provider)]
            }
        });
        serde_json::to_writer(&mut writer, &row)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn write_groundtruth(
    path: &PathBuf,
    queries: &[QueryData],
    external_to_internal: &HashMap<u64, u32>,
) -> Result<()> {
    let total_results = queries
        .iter()
        .try_fold(0usize, |total, query| {
            total.checked_add(query.truth_external_ids.len())
        })
        .ok_or_else(|| anyhow!("groundtruth result count overflow"))?;
    let mut writer = BufWriter::new(File::create(path)?);
    write_u32(
        &mut writer,
        u32::try_from(queries.len()).context("query count exceeds u32")?,
    )?;
    write_u32(
        &mut writer,
        u32::try_from(total_results).context("groundtruth result count exceeds u32")?,
    )?;
    for query in queries {
        write_u32(
            &mut writer,
            u32::try_from(query.truth_external_ids.len())
                .context("per-query groundtruth count exceeds u32")?,
        )?;
    }
    for query in queries {
        for external_id in &query.truth_external_ids {
            let internal_id = external_to_internal[external_id];
            if internal_id == u32::MAX {
                bail!("truth external ID {external_id} was not mapped");
            }
            write_u32(&mut writer, internal_id)?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn write_metadata(path: &PathBuf, queries: &[QueryData]) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(b"query_id\tquery_text\tprovider_id\tlanguage\ttruth_count\n")?;
    for (query_id, query) in queries.iter().enumerate() {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}",
            query_id,
            query.key.text,
            query.key.provider,
            query.key.language,
            query.truth_external_ids.len()
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn split_columns(line: &[u8], minimum: usize) -> Option<Vec<&[u8]>> {
    let fields = line.split(|byte| *byte == b'\t').collect::<Vec<_>>();
    (fields.len() >= minimum).then_some(fields)
}

fn first_two_columns(line: &[u8]) -> Option<(&[u8], &[u8])> {
    let first = line.iter().position(|byte| *byte == b'\t')?;
    let remaining = &line[first + 1..];
    let second = remaining.iter().position(|byte| *byte == b'\t')?;
    Some((&line[..first], &remaining[..second]))
}

fn parse_language_provider(label: &str) -> Result<(&str, &str)> {
    let mut parts = label.split('_');
    let language = parts.next().unwrap_or_default();
    let provider = parts.next().unwrap_or_default();
    if language.is_empty() || provider.is_empty() || language == "UNV" || provider == "UNV" {
        bail!("language and provider must be concrete");
    }
    Ok((language, provider))
}

fn parse_embedding(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut embedding = Vec::with_capacity(128);
    parse_embedding_into(bytes, &mut embedding)?;
    Ok(embedding)
}

fn parse_embedding_into(bytes: &[u8], output: &mut Vec<u8>) -> Result<()> {
    output.clear();
    let mut value = 0u16;
    let mut has_digit = false;
    for &byte in bytes {
        if byte.is_ascii_digit() {
            value = value
                .checked_mul(10)
                .and_then(|current| current.checked_add(u16::from(byte - b'0')))
                .ok_or_else(|| anyhow!("embedding value overflow"))?;
            if value > u16::from(u8::MAX) {
                bail!("embedding value {value} exceeds u8");
            }
            has_digit = true;
        } else if byte.is_ascii_whitespace() {
            if has_digit {
                output.push(value as u8);
                value = 0;
                has_digit = false;
            }
        } else {
            bail!("embedding contains non-numeric byte {byte}");
        }
    }
    if has_digit {
        output.push(value as u8);
    }
    if output.is_empty() {
        bail!("embedding is empty");
    }
    Ok(())
}

fn trim_line(line: &mut Vec<u8>) {
    while line
        .last()
        .is_some_and(|byte| matches!(byte, b'\n' | b'\r'))
    {
        line.pop();
    }
}

fn write_u32(writer: &mut impl Write, value: u32) -> std::io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}
