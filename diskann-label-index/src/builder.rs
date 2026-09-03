/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! JSONL parsing and dense Bitslice label-index construction.

use crate::{
    error::EncodedLabelIndexError,
    format::{
        validate_label, write_u32, write_u64, BITSLICE_FORMAT, LABEL_INDEX_MAGIC,
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
    num_vectors: u32,
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

fn scan_label_jsonl(input_path: &Path) -> Result<ScannedLabels, EncodedLabelIndexError> {
    let reader = BufReader::new(File::open(input_path)?);
    let mut labels = Vec::<String>::new();
    let mut label_ids = HashMap::<String, u32>::new();
    let mut max_doc_id = None::<u32>;
    let mut seen_doc_ids = HashSet::<u32>::new();

    for document in read_documents(reader) {
        let (doc_id, document_labels) = document?;
        if !seen_doc_ids.insert(doc_id) {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "duplicate doc_id {doc_id} in label JSONL"
            )));
        }
        max_doc_id = Some(max_doc_id.map_or(doc_id, |current| current.max(doc_id)));

        for label in document_labels {
            if label_ids.contains_key(&label) {
                continue;
            }
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
