/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! JSONL parsing and label-index construction.

use crate::{
    bloom::{append_label_rows, BloomFilterConfig, BLOOM_FORMAT},
    error::EncodedLabelIndexError,
    format::{
        validate_label, write_u32, write_u64, BITSLICE_FORMAT, LABEL_INDEX_MAGIC,
        LABEL_INDEX_VERSION, MAX_LABEL_COUNT, PREFIX_EXACT_BLOOM_INDEX_VERSION,
    },
};
use serde_json::{Map, Value};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

struct ScannedLabels {
    labels: Vec<String>,
    label_ids: HashMap<String, u32>,
    counts: Vec<u32>,
    num_vectors: u32,
}

struct BloomBuildBuffers<'a> {
    words_per_row: usize,
    hash_count: usize,
    exact_label_count: usize,
    label_rows: &'a [u32],
    bits: &'a mut [u64],
}

struct BloomWriteData<'a> {
    labels: &'a [String],
    counts: &'a [u32],
    words_per_row: usize,
    config: BloomFilterConfig,
    exact_label_count: usize,
    label_rows: &'a [u32],
    bits: &'a [u64],
}

/// Encode a JSONL label file as a versioned dense Bitslice index.
///
/// Supported JSONL rows are:
///
/// - objects with optional `doc_id`; `true` fields use the field name, other scalar values use
///   `field=value`;
/// - objects with a `labels` string array;
/// - a raw string or string array, using the non-empty line number as the document ID.
pub fn encode_label_index_jsonl(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<(), EncodedLabelIndexError> {
    let input_path = input_path.as_ref();
    let ScannedLabels {
        labels,
        label_ids,
        num_vectors,
        ..
    } = scan_label_jsonl(input_path)?;
    let words_per_label = (num_vectors as usize).div_ceil(64);
    let total_words = labels.len().checked_mul(words_per_label).ok_or_else(|| {
        EncodedLabelIndexError::Invalid("bitslice output size overflow".to_string())
    })?;
    let mut bits = Vec::new();
    bits.try_reserve_exact(total_words).map_err(|_| {
        EncodedLabelIndexError::Invalid("cannot reserve bitslice output".to_string())
    })?;
    bits.resize(total_words, 0);
    populate_bits(
        input_path,
        &label_ids,
        num_vectors,
        words_per_label,
        &mut bits,
    )?;
    write_label_index(
        output_path.as_ref(),
        num_vectors,
        &labels,
        words_per_label,
        &bits,
    )
}

/// Encode a JSONL label file as a versioned transposed Bloom index.
///
/// The accepted JSONL forms are identical to [`encode_label_index_jsonl`]. Each label hashes to
/// `config.hash_count()` distinct rows in a fixed `config.bit_count()`-row vector bitmap. The
/// highest-frequency labels can instead use exact rows. Payload and summary rows use one physical
/// row space: exact rows first, followed by Bloom rows.
pub fn encode_bloom_label_index_jsonl(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    config: BloomFilterConfig,
) -> Result<(), EncodedLabelIndexError> {
    let input_path = input_path.as_ref();
    let scanned = scan_label_jsonl(input_path)?;
    let (
        ScannedLabels {
            labels,
            label_ids,
            counts,
            num_vectors,
        },
        exact_label_count,
    ) = order_bloom_dictionary(scanned, config.exact_label_count())?;
    let words_per_row = (num_vectors as usize).div_ceil(64);
    let total_rows = exact_label_count
        .checked_add(config.bit_count() as usize)
        .ok_or_else(|| EncodedLabelIndexError::Invalid("Bloom row count overflow".to_string()))?;
    let total_words = total_rows
        .checked_mul(words_per_row)
        .ok_or_else(|| EncodedLabelIndexError::Invalid("Bloom output size overflow".to_string()))?;
    let mut bits = Vec::new();
    bits.try_reserve_exact(total_words)
        .map_err(|_| EncodedLabelIndexError::Invalid("cannot reserve Bloom output".to_string()))?;
    bits.resize(total_words, 0);

    let label_rows = build_bloom_label_rows(&labels[exact_label_count..], config)?;
    let mut buffers = BloomBuildBuffers {
        words_per_row,
        hash_count: config.hash_count() as usize,
        exact_label_count,
        label_rows: &label_rows,
        bits: &mut bits,
    };
    populate_bloom_bits(input_path, &label_ids, num_vectors, &mut buffers)?;
    write_bloom_label_index(
        output_path.as_ref(),
        num_vectors,
        BloomWriteData {
            labels: &labels,
            counts: &counts,
            words_per_row,
            config,
            exact_label_count,
            label_rows: &label_rows,
            bits: &bits,
        },
    )
}

fn order_bloom_dictionary(
    scanned: ScannedLabels,
    requested_exact: u32,
) -> Result<(ScannedLabels, usize), EncodedLabelIndexError> {
    let ScannedLabels {
        labels,
        counts,
        num_vectors,
        ..
    } = scanned;
    let requested_exact = usize::try_from(requested_exact).map_err(|_| {
        EncodedLabelIndexError::Invalid("exact Bloom label count exceeds usize".to_string())
    })?;
    let mut entries: Vec<_> = labels.into_iter().zip(counts).collect();
    entries.sort_unstable_by(|(left_label, left_count), (right_label, right_count)| {
        right_count
            .cmp(left_count)
            .then_with(|| left_label.as_bytes().cmp(right_label.as_bytes()))
    });
    let exact_label_count = requested_exact.min(entries.len());
    let mut labels = Vec::new();
    labels.try_reserve_exact(entries.len()).map_err(|_| {
        EncodedLabelIndexError::Invalid("cannot reserve sorted Bloom labels".to_string())
    })?;
    let mut counts = Vec::new();
    counts.try_reserve_exact(entries.len()).map_err(|_| {
        EncodedLabelIndexError::Invalid("cannot reserve sorted Bloom counts".to_string())
    })?;
    let mut label_ids = HashMap::new();
    label_ids.try_reserve(entries.len()).map_err(|_| {
        EncodedLabelIndexError::Invalid("cannot reserve sorted Bloom label map".to_string())
    })?;
    for (id, (label, count)) in entries.into_iter().enumerate() {
        let id = u32::try_from(id).map_err(|_| {
            EncodedLabelIndexError::Invalid("Bloom label ID exceeds u32".to_string())
        })?;
        label_ids.insert(label.clone(), id);
        labels.push(label);
        counts.push(count);
    }
    Ok((
        ScannedLabels {
            labels,
            label_ids,
            counts,
            num_vectors,
        },
        exact_label_count,
    ))
}

fn scan_label_jsonl(input_path: &Path) -> Result<ScannedLabels, EncodedLabelIndexError> {
    let reader = BufReader::new(File::open(input_path)?);
    let mut labels = Vec::<String>::new();
    let mut label_ids = HashMap::<String, u32>::new();
    let mut counts = Vec::<u32>::new();
    let mut max_doc_id = None::<u32>;
    let mut seen_doc_ids = HashSet::<u32>::new();
    let mut seen_labels = HashSet::<u32>::new();

    for document in read_documents(reader) {
        let (doc_id, document_labels) = document?;
        if !seen_doc_ids.insert(doc_id) {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "duplicate doc_id {doc_id} in label JSONL"
            )));
        }
        max_doc_id = Some(max_doc_id.map_or(doc_id, |current| current.max(doc_id)));

        seen_labels.clear();
        for label in document_labels {
            let label_id = if let Some(&label_id) = label_ids.get(&label) {
                label_id
            } else {
                if labels.len() >= MAX_LABEL_COUNT {
                    return Err(EncodedLabelIndexError::Invalid(format!(
                        "label count exceeds limit {MAX_LABEL_COUNT}"
                    )));
                }
                let label_id = u32::try_from(labels.len()).map_err(|_| {
                    EncodedLabelIndexError::Invalid("label count exceeds u32".to_string())
                })?;
                label_ids.insert(label.clone(), label_id);
                labels.push(label);
                counts.push(0);
                label_id
            };
            if seen_labels.insert(label_id) {
                let count = &mut counts[label_id as usize];
                *count = count.checked_add(1).ok_or_else(|| {
                    EncodedLabelIndexError::Invalid(format!(
                        "vector count for label ID {label_id} exceeds u32"
                    ))
                })?;
            }
        }
    }

    let num_vectors = max_doc_id
        .and_then(|doc_id| doc_id.checked_add(1))
        .ok_or_else(|| {
            EncodedLabelIndexError::Invalid("label JSONL contains no documents".to_string())
        })?;
    Ok(ScannedLabels {
        labels,
        label_ids,
        counts,
        num_vectors,
    })
}

fn populate_bits(
    input_path: &Path,
    label_ids: &HashMap<String, u32>,
    num_vectors: u32,
    words_per_label: usize,
    bits: &mut [u64],
) -> Result<(), EncodedLabelIndexError> {
    let reader = BufReader::new(File::open(input_path)?);
    let mut seen_doc_ids = HashSet::<u32>::new();

    for document in read_documents(reader) {
        let (doc_id, document_labels) = document?;
        if doc_id >= num_vectors {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "doc_id {doc_id} exceeds the vector count discovered during encoding"
            )));
        }
        if !seen_doc_ids.insert(doc_id) {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "duplicate doc_id {doc_id} in label JSONL"
            )));
        }
        for label in document_labels {
            let label_id = label_ids.get(&label).copied().ok_or_else(|| {
                EncodedLabelIndexError::Invalid(format!(
                    "label '{label}' appeared after the encoding dictionary was built"
                ))
            })? as usize;
            let word = label_id * words_per_label + doc_id as usize / 64;
            bits[word] |= 1u64 << (doc_id % 64);
        }
    }
    Ok(())
}

fn build_bloom_label_rows(
    labels: &[String],
    config: BloomFilterConfig,
) -> Result<Vec<u32>, EncodedLabelIndexError> {
    let total_rows = labels
        .len()
        .checked_mul(config.hash_count() as usize)
        .ok_or_else(|| {
            EncodedLabelIndexError::Invalid("Bloom label-row table size overflow".to_string())
        })?;
    let mut label_rows = Vec::new();
    label_rows.try_reserve_exact(total_rows).map_err(|_| {
        EncodedLabelIndexError::Invalid("cannot reserve Bloom label-row table".to_string())
    })?;
    for label in labels {
        append_label_rows(label, config, &mut label_rows);
    }
    Ok(label_rows)
}

fn populate_bloom_bits(
    input_path: &Path,
    label_ids: &HashMap<String, u32>,
    num_vectors: u32,
    buffers: &mut BloomBuildBuffers<'_>,
) -> Result<(), EncodedLabelIndexError> {
    let reader = BufReader::new(File::open(input_path)?);
    let mut seen_doc_ids = HashSet::<u32>::new();

    for document in read_documents(reader) {
        let (doc_id, document_labels) = document?;
        if doc_id >= num_vectors {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "doc_id {doc_id} exceeds the vector count discovered during encoding"
            )));
        }
        if !seen_doc_ids.insert(doc_id) {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "duplicate doc_id {doc_id} in label JSONL"
            )));
        }
        for label in document_labels {
            let label_id = label_ids.get(&label).copied().ok_or_else(|| {
                EncodedLabelIndexError::Invalid(format!(
                    "label '{label}' appeared after the encoding dictionary was built"
                ))
            })? as usize;
            if label_id < buffers.exact_label_count {
                let word = label_id * buffers.words_per_row + doc_id as usize / 64;
                buffers.bits[word] |= 1u64 << (doc_id % 64);
                continue;
            }
            let row_offset = (label_id - buffers.exact_label_count) * buffers.hash_count;
            for &row in &buffers.label_rows[row_offset..row_offset + buffers.hash_count] {
                let physical_row = buffers
                    .exact_label_count
                    .checked_add(row as usize)
                    .ok_or_else(|| {
                        EncodedLabelIndexError::Invalid(
                            "Bloom physical row index overflow".to_string(),
                        )
                    })?;
                let word = physical_row * buffers.words_per_row + doc_id as usize / 64;
                buffers.bits[word] |= 1u64 << (doc_id % 64);
            }
        }
    }
    Ok(())
}

fn read_documents(
    reader: impl BufRead,
) -> impl Iterator<Item = Result<(u32, Vec<String>), EncodedLabelIndexError>> {
    let mut record_number = 0u32;
    reader.lines().filter_map(move |line| match line {
        Ok(line) if line.trim().is_empty() => None,
        Ok(line) => {
            let default_doc_id = record_number;
            let Some(next_record_number) = record_number.checked_add(1) else {
                return Some(Err(EncodedLabelIndexError::Invalid(
                    "JSONL record count exceeds u32".to_string(),
                )));
            };
            record_number = next_record_number;
            Some(parse_document(&line, default_doc_id))
        }
        Err(error) => Some(Err(error.into())),
    })
}

fn parse_document(
    line: &str,
    default_doc_id: u32,
) -> Result<(u32, Vec<String>), EncodedLabelIndexError> {
    let value: Value = serde_json::from_str(line)?;
    match value {
        Value::String(label) => {
            validate_label(&label)?;
            Ok((default_doc_id, vec![label]))
        }
        Value::Array(labels) => Ok((
            default_doc_id,
            parse_direct_label_array(&labels, "root label array")?,
        )),
        Value::Object(fields) => parse_object_document(fields, default_doc_id),
        _ => Err(EncodedLabelIndexError::Invalid(
            "each JSONL line must be an object, string, or string array".to_string(),
        )),
    }
}

fn parse_object_document(
    fields: Map<String, Value>,
    default_doc_id: u32,
) -> Result<(u32, Vec<String>), EncodedLabelIndexError> {
    let doc_id = match fields.get("doc_id") {
        Some(value) => {
            let value = value.as_u64().ok_or_else(|| {
                EncodedLabelIndexError::Invalid("doc_id must be a non-negative integer".to_string())
            })?;
            u32::try_from(value)
                .map_err(|_| EncodedLabelIndexError::Invalid("doc_id exceeds u32".to_string()))?
        }
        None => default_doc_id,
    };

    let mut labels = Vec::new();
    for (field, value) in fields {
        if field == "doc_id" {
            continue;
        }
        if field == "labels" {
            if let Value::Array(values) = &value {
                labels.extend(parse_direct_label_array(values, "labels field")?);
                continue;
            }
        }

        let label = match value {
            Value::Bool(true) => field,
            Value::Bool(false) => format!("{field}=false"),
            Value::Number(value) => format!("{field}={value}"),
            Value::String(value) => format!("{field}={value}"),
            Value::Null => format!("{field}=null"),
            Value::Array(_) | Value::Object(_) => {
                return Err(EncodedLabelIndexError::Invalid(format!(
                    "field '{field}' must contain a scalar value"
                )));
            }
        };
        validate_label(&label)?;
        labels.push(label);
    }

    Ok((doc_id, labels))
}

fn parse_direct_label_array(
    values: &[Value],
    context: &str,
) -> Result<Vec<String>, EncodedLabelIndexError> {
    values
        .iter()
        .map(|value| {
            let label = value.as_str().ok_or_else(|| {
                EncodedLabelIndexError::Invalid(format!("{context} must contain only strings"))
            })?;
            validate_label(label)?;
            Ok(label.to_string())
        })
        .collect()
}

fn write_label_index(
    path: &Path,
    num_vectors: u32,
    labels: &[String],
    words_per_label: usize,
    bits: &[u64],
) -> Result<(), EncodedLabelIndexError> {
    if labels.len() > MAX_LABEL_COUNT {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "label count {} exceeds limit {MAX_LABEL_COUNT}",
            labels.len()
        )));
    }
    if num_vectors == 0 {
        return Err(EncodedLabelIndexError::Invalid(
            "label-index vector count cannot be zero".to_string(),
        ));
    }
    let expected_words = labels.len().checked_mul(words_per_label).ok_or_else(|| {
        EncodedLabelIndexError::Invalid("bitslice output size overflow".to_string())
    })?;
    if bits.len() != expected_words {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "bitslice payload has {} words; expected {expected_words}",
            bits.len()
        )));
    }

    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(&LABEL_INDEX_MAGIC)?;
    write_u32(&mut writer, LABEL_INDEX_VERSION)?;
    write_u32(&mut writer, BITSLICE_FORMAT)?;
    write_u64(&mut writer, u64::from(num_vectors))?;
    write_u64(&mut writer, labels.len() as u64)?;

    for label in labels {
        validate_label(label)?;
        let bytes = label.as_bytes();
        let len = u32::try_from(bytes.len()).map_err(|_| {
            EncodedLabelIndexError::Invalid(format!("label '{label}' is too long to encode"))
        })?;
        write_u32(&mut writer, len)?;
        writer.write_all(bytes)?;
    }

    write_u64(&mut writer, words_per_label as u64)?;
    for word in bits {
        write_u64(&mut writer, *word)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_bloom_label_index(
    path: &Path,
    num_vectors: u32,
    data: BloomWriteData<'_>,
) -> Result<(), EncodedLabelIndexError> {
    let BloomWriteData {
        labels,
        counts,
        words_per_row,
        config,
        exact_label_count,
        label_rows,
        bits,
    } = data;
    if labels.len() > MAX_LABEL_COUNT {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "label count {} exceeds limit {MAX_LABEL_COUNT}",
            labels.len()
        )));
    }
    if num_vectors == 0 {
        return Err(EncodedLabelIndexError::Invalid(
            "label-index vector count cannot be zero".to_string(),
        ));
    }
    if counts.len() != labels.len()
        || counts
            .iter()
            .any(|&count| count == 0 || count > num_vectors)
    {
        return Err(EncodedLabelIndexError::Invalid(
            "Bloom label counts must match the dictionary and vector count".to_string(),
        ));
    }
    if exact_label_count > labels.len() {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "exact Bloom label count {exact_label_count} exceeds label count {}",
            labels.len()
        )));
    }
    let expected_label_rows = (labels.len() - exact_label_count)
        .checked_mul(config.hash_count() as usize)
        .ok_or_else(|| {
            EncodedLabelIndexError::Invalid("Bloom label-row table size overflow".to_string())
        })?;
    if label_rows.len() != expected_label_rows {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "Bloom label-row table has {} entries; expected {expected_label_rows}",
            label_rows.len()
        )));
    }
    for rows in label_rows.chunks_exact(config.hash_count() as usize) {
        if rows.iter().any(|&row| row >= config.bit_count())
            || rows
                .iter()
                .enumerate()
                .any(|(index, row)| rows[..index].contains(row))
        {
            return Err(EncodedLabelIndexError::Invalid(
                "Bloom label rows must be distinct and within the Bloom row count".to_string(),
            ));
        }
    }
    let total_rows = exact_label_count
        .checked_add(config.bit_count() as usize)
        .ok_or_else(|| EncodedLabelIndexError::Invalid("Bloom row count overflow".to_string()))?;
    let expected_words = total_rows
        .checked_mul(words_per_row)
        .ok_or_else(|| EncodedLabelIndexError::Invalid("Bloom output size overflow".to_string()))?;
    if bits.len() != expected_words {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "Bloom payload has {} words; expected {expected_words}",
            bits.len()
        )));
    }

    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(&LABEL_INDEX_MAGIC)?;
    write_u32(&mut writer, PREFIX_EXACT_BLOOM_INDEX_VERSION)?;
    write_u32(&mut writer, BLOOM_FORMAT)?;
    write_u64(&mut writer, u64::from(num_vectors))?;
    write_u64(&mut writer, labels.len() as u64)?;

    for label in labels {
        validate_label(label)?;
        let bytes = label.as_bytes();
        let len = u32::try_from(bytes.len()).map_err(|_| {
            EncodedLabelIndexError::Invalid(format!("label '{label}' is too long to encode"))
        })?;
        write_u32(&mut writer, len)?;
        writer.write_all(bytes)?;
    }

    write_u32(&mut writer, config.bit_count())?;
    write_u32(&mut writer, config.hash_count())?;
    write_u32(
        &mut writer,
        u32::try_from(exact_label_count).map_err(|_| {
            EncodedLabelIndexError::Invalid("exact Bloom label count exceeds u32".to_string())
        })?,
    )?;
    write_u32(&mut writer, 0)?;
    write_u64(&mut writer, words_per_row as u64)?;
    for &row in label_rows {
        write_u32(&mut writer, row)?;
    }

    let summary_words_per_row = words_per_row.div_ceil(64);
    let total_summary_words = total_rows
        .checked_mul(summary_words_per_row)
        .ok_or_else(|| {
            EncodedLabelIndexError::Invalid("Bloom word-summary size overflow".to_string())
        })?;
    let mut nonzero_words = Vec::new();
    nonzero_words
        .try_reserve_exact(total_summary_words)
        .map_err(|_| {
            EncodedLabelIndexError::Invalid("cannot reserve Bloom word summaries".to_string())
        })?;
    nonzero_words.resize(total_summary_words, 0u64);

    for (index, &word) in bits.iter().enumerate() {
        write_u64(&mut writer, word)?;
        if word != 0 {
            let row = index / words_per_row;
            let word_index = index % words_per_row;
            nonzero_words[row * summary_words_per_row + word_index / 64] |=
                1u64 << (word_index % 64);
        }
    }
    write_u64(&mut writer, summary_words_per_row as u64)?;
    for word in nonzero_words {
        write_u64(&mut writer, word)?;
    }
    for &count in counts {
        write_u32(&mut writer, count)?;
    }
    writer.flush()?;
    Ok(())
}
