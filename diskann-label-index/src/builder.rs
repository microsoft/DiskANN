/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! JSONL parsing and label-index construction.

use crate::{
    bloom::{append_label_rows, BloomFilterConfig},
    error::EncodedLabelIndexError,
    format::{
        validate_label, write_u32, write_u64, LABEL_INDEX_FLAGS, LABEL_INDEX_MAGIC,
        LABEL_INDEX_VERSION, MAX_LABEL_COUNT,
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

struct BuildBuffers<'a> {
    words_per_row: usize,
    hash_count: usize,
    exact_label_count: usize,
    label_rows: &'a [u32],
    bits: &'a mut [u64],
}

struct WriteData<'a> {
    labels: &'a [String],
    counts: &'a [u32],
    words_per_row: usize,
    exact_label_count: usize,
    bloom_row_count: usize,
    hash_count: usize,
    label_rows: &'a [u32],
    bits: &'a [u64],
}

#[derive(Debug, Clone, Copy)]
struct RowMetadata {
    valid_node_count: u32,
    nonzero_word_count: u32,
}

/// Encode a JSONL label file as a unified exact/Bloom label index.
///
/// Supported JSONL rows are:
///
/// - objects with optional `doc_id`; `true` fields use the field name, other scalar values use
///   `field=value`;
/// - objects with a `labels` string array;
/// - a raw string or string array, using the non-empty line number as the document ID.
///
/// An exact-label count of zero creates a Bloom-only index. A count greater than or equal to the
/// discovered dictionary size creates an exact-only index. Intermediate values create a hybrid
/// exact-prefix/Bloom layout.
pub fn encode_unified_label_index_jsonl(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    config: BloomFilterConfig,
) -> Result<(), EncodedLabelIndexError> {
    let input_path = input_path.as_ref();
    let output_path = output_path.as_ref();
    let scanned = scan_label_jsonl(input_path)?;
    let requested_exact = config.exact_label_count();
    let (
        ScannedLabels {
            labels,
            label_ids,
            counts,
            num_vectors,
        },
        exact_label_count,
    ) = order_dictionary(scanned, requested_exact)?;
    let has_bloom_rows = exact_label_count < labels.len();
    let (bloom_row_count, hash_count, bloom_config) = if has_bloom_rows {
        (
            config.bit_count() as usize,
            config.hash_count() as usize,
            Some(config),
        )
    } else {
        (0, 0, None)
    };
    let words_per_row = (num_vectors as usize).div_ceil(64);
    let total_rows = exact_label_count
        .checked_add(bloom_row_count)
        .ok_or_else(|| EncodedLabelIndexError::Invalid("label row count overflow".to_string()))?;
    let total_words = total_rows
        .checked_mul(words_per_row)
        .ok_or_else(|| EncodedLabelIndexError::Invalid("label output size overflow".to_string()))?;
    let mut bits = Vec::new();
    bits.try_reserve_exact(total_words)
        .map_err(|_| EncodedLabelIndexError::Invalid("cannot reserve label output".to_string()))?;
    bits.resize(total_words, 0);

    let label_row_count = labels.len().checked_mul(hash_count).ok_or_else(|| {
        EncodedLabelIndexError::Invalid("label-row table size overflow".to_string())
    })?;
    let mut label_rows = Vec::new();
    label_rows.try_reserve_exact(label_row_count).map_err(|_| {
        EncodedLabelIndexError::Invalid("cannot reserve label-row table".to_string())
    })?;
    label_rows.resize(label_row_count, u32::MAX);
    if let Some(config) = bloom_config {
        let logical_rows = build_bloom_label_rows(&labels[exact_label_count..], config)?;
        for (suffix_index, rows) in logical_rows.chunks_exact(hash_count).enumerate() {
            let label_id = exact_label_count + suffix_index;
            let start = label_id * hash_count;
            for (hash_index, &logical_row) in rows.iter().enumerate() {
                label_rows[start + hash_index] = u32::try_from(exact_label_count)
                    .ok()
                    .and_then(|prefix| prefix.checked_add(logical_row))
                    .ok_or_else(|| {
                        EncodedLabelIndexError::Invalid(
                            "Bloom physical row index overflow".to_string(),
                        )
                    })?;
            }
        }
    }
    let mut buffers = BuildBuffers {
        words_per_row,
        hash_count,
        exact_label_count,
        label_rows: &label_rows,
        bits: &mut bits,
    };
    populate_bits(input_path, &label_ids, num_vectors, &mut buffers)?;
    write_label_index(
        output_path,
        num_vectors,
        WriteData {
            labels: &labels,
            counts: &counts,
            words_per_row,
            exact_label_count,
            bloom_row_count,
            hash_count,
            label_rows: &label_rows,
            bits: &bits,
        },
    )
}

fn order_dictionary(
    scanned: ScannedLabels,
    requested_exact: u32,
) -> Result<(ScannedLabels, usize), EncodedLabelIndexError> {
    let ScannedLabels {
        labels: scanned_labels,
        counts: scanned_counts,
        num_vectors,
        ..
    } = scanned;
    let requested_exact = usize::try_from(requested_exact).map_err(|_| {
        EncodedLabelIndexError::Invalid("exact label count exceeds usize".to_string())
    })?;
    let mut entries: Vec<_> = scanned_labels.into_iter().zip(scanned_counts).collect();
    entries.sort_unstable_by(|(left_label, left_count), (right_label, right_count)| {
        right_count
            .cmp(left_count)
            .then_with(|| left_label.as_bytes().cmp(right_label.as_bytes()))
    });
    let exact_label_count = requested_exact.min(entries.len());
    let mut ordered_labels = Vec::new();
    ordered_labels
        .try_reserve_exact(entries.len())
        .map_err(|_| {
            EncodedLabelIndexError::Invalid("cannot reserve ordered labels".to_string())
        })?;
    let mut ordered_counts = Vec::new();
    ordered_counts
        .try_reserve_exact(entries.len())
        .map_err(|_| {
            EncodedLabelIndexError::Invalid("cannot reserve ordered label counts".to_string())
        })?;
    let mut ordered_label_ids = HashMap::new();
    ordered_label_ids.try_reserve(entries.len()).map_err(|_| {
        EncodedLabelIndexError::Invalid("cannot reserve ordered label map".to_string())
    })?;
    for (id, (label, count)) in entries.into_iter().enumerate() {
        let id = u32::try_from(id)
            .map_err(|_| EncodedLabelIndexError::Invalid("label ID exceeds u32".to_string()))?;
        ordered_label_ids.insert(label.clone(), id);
        ordered_labels.push(label);
        ordered_counts.push(count);
    }
    Ok((
        ScannedLabels {
            labels: ordered_labels,
            label_ids: ordered_label_ids,
            counts: ordered_counts,
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

fn populate_bits(
    input_path: &Path,
    label_ids: &HashMap<String, u32>,
    num_vectors: u32,
    buffers: &mut BuildBuffers<'_>,
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
            if buffers.hash_count == 0 {
                return Err(EncodedLabelIndexError::Invalid(
                    "non-exact label has no Bloom rows".to_string(),
                ));
            }
            let row_offset = label_id * buffers.hash_count;
            for &row in &buffers.label_rows[row_offset..row_offset + buffers.hash_count] {
                let word = row as usize * buffers.words_per_row + doc_id as usize / 64;
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
    data: WriteData<'_>,
) -> Result<(), EncodedLabelIndexError> {
    let WriteData {
        labels,
        counts,
        words_per_row,
        exact_label_count,
        bloom_row_count,
        hash_count,
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
            "label counts must match the dictionary and vector count".to_string(),
        ));
    }
    if exact_label_count > labels.len() {
        return Err(EncodedLabelIndexError::Invalid(
            "exact label count exceeds label count".to_string(),
        ));
    }
    if exact_label_count < labels.len() && (bloom_row_count == 0 || hash_count == 0) {
        return Err(EncodedLabelIndexError::Invalid(
            "non-exact labels require Bloom rows and hashes".to_string(),
        ));
    }
    if exact_label_count == labels.len() && (bloom_row_count != 0 || hash_count != 0) {
        return Err(EncodedLabelIndexError::Invalid(
            "all-exact indexes cannot contain Bloom rows".to_string(),
        ));
    }
    let expected_label_rows = labels.len().checked_mul(hash_count).ok_or_else(|| {
        EncodedLabelIndexError::Invalid("label-row table size overflow".to_string())
    })?;
    if label_rows.len() != expected_label_rows {
        return Err(EncodedLabelIndexError::Invalid(
            "label-row table length does not match the dictionary".to_string(),
        ));
    }
    let total_rows = exact_label_count
        .checked_add(bloom_row_count)
        .ok_or_else(|| EncodedLabelIndexError::Invalid("label row count overflow".to_string()))?;
    for label_id in exact_label_count..labels.len() {
        let start = label_id * hash_count;
        let rows = &label_rows[start..start + hash_count];
        if rows
            .iter()
            .any(|&row| row < exact_label_count as u32 || row as usize >= total_rows)
            || rows
                .iter()
                .enumerate()
                .any(|(index, row)| rows[..index].contains(row))
        {
            return Err(EncodedLabelIndexError::Invalid(
                "Bloom label rows must be distinct physical Bloom rows".to_string(),
            ));
        }
    }
    let expected_words = total_rows.checked_mul(words_per_row).ok_or_else(|| {
        EncodedLabelIndexError::Invalid("label payload size overflow".to_string())
    })?;
    if bits.len() != expected_words {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "label payload has {} words; expected {expected_words}",
            bits.len()
        )));
    }

    let summary_words_per_row = words_per_row.div_ceil(64);
    let summary_word_count = total_rows
        .checked_mul(summary_words_per_row)
        .ok_or_else(|| {
            EncodedLabelIndexError::Invalid("label summary size overflow".to_string())
        })?;
    let mut summaries = Vec::new();
    summaries
        .try_reserve_exact(summary_word_count)
        .map_err(|_| {
            EncodedLabelIndexError::Invalid("cannot reserve label summaries".to_string())
        })?;
    summaries.resize(summary_word_count, 0u64);
    let mut row_metadata = Vec::new();
    row_metadata.try_reserve_exact(total_rows).map_err(|_| {
        EncodedLabelIndexError::Invalid("cannot reserve label row metadata".to_string())
    })?;
    for row in 0..total_rows {
        let payload = &bits[row * words_per_row..(row + 1) * words_per_row];
        let mut valid_node_count = 0u32;
        let mut nonzero_word_count = 0u32;
        for (word_index, &word) in payload.iter().enumerate() {
            valid_node_count =
                valid_node_count
                    .checked_add(word.count_ones())
                    .ok_or_else(|| {
                        EncodedLabelIndexError::Invalid(
                            "row valid-node count exceeds u32".to_string(),
                        )
                    })?;
            if word != 0 {
                nonzero_word_count += 1;
                summaries[row * summary_words_per_row + word_index / 64] |=
                    1u64 << (word_index % 64);
            }
        }
        if row < exact_label_count && valid_node_count != counts[row] {
            return Err(EncodedLabelIndexError::Invalid(
                "exact row count does not match its label count".to_string(),
            ));
        }
        row_metadata.push(RowMetadata {
            valid_node_count,
            nonzero_word_count,
        });
    }

    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(&LABEL_INDEX_MAGIC)?;
    write_u32(&mut writer, LABEL_INDEX_VERSION)?;
    write_u32(&mut writer, LABEL_INDEX_FLAGS)?;
    write_u64(&mut writer, u64::from(num_vectors))?;
    write_u64(&mut writer, labels.len() as u64)?;
    write_u32(
        &mut writer,
        u32::try_from(exact_label_count).map_err(|_| {
            EncodedLabelIndexError::Invalid("exact label count exceeds u32".to_string())
        })?,
    )?;
    write_u32(
        &mut writer,
        u32::try_from(bloom_row_count).map_err(|_| {
            EncodedLabelIndexError::Invalid("Bloom row count exceeds u32".to_string())
        })?,
    )?;
    write_u32(
        &mut writer,
        u32::try_from(hash_count).map_err(|_| {
            EncodedLabelIndexError::Invalid("Bloom hash count exceeds u32".to_string())
        })?,
    )?;
    write_u32(&mut writer, 0)?;
    write_u64(&mut writer, words_per_row as u64)?;
    write_u64(&mut writer, summary_words_per_row as u64)?;

    for (label, &count) in labels.iter().zip(counts) {
        validate_label(label)?;
        let bytes = label.as_bytes();
        let len = u32::try_from(bytes.len()).map_err(|_| {
            EncodedLabelIndexError::Invalid(format!("label '{label}' is too long to encode"))
        })?;
        write_u32(&mut writer, len)?;
        write_u32(&mut writer, count)?;
        writer.write_all(bytes)?;
    }

    for label_id in exact_label_count..labels.len() {
        let start = label_id * hash_count;
        for &row in &label_rows[start..start + hash_count] {
            write_u32(&mut writer, row)?;
        }
    }

    for metadata in row_metadata {
        write_u32(&mut writer, metadata.valid_node_count)?;
        write_u32(&mut writer, metadata.nonzero_word_count)?;
    }
    for word in summaries {
        write_u64(&mut writer, word)?;
    }
    for word in bits {
        write_u64(&mut writer, *word)?;
    }
    writer.flush()?;
    Ok(())
}
