/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Versioned on-disk label-index encoding, loading, and query evaluation.

use diskann::graph::ext::labeled::QueryLabelProvider;
use roaring::RoaringBitmap;
use serde_json::{Map, Value};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Read, Seek, Write},
    path::Path,
};
use thiserror::Error;

const LABEL_INDEX_MAGIC: [u8; 8] = *b"DANLBL01";
const LABEL_INDEX_VERSION: u32 = 1;
const MAX_LABEL_COUNT: usize = 1_000_000;
const MAX_LABEL_LENGTH: usize = 1 << 20;
const MAX_POSTING_BYTES: usize = 512 << 20;
const MAX_DENSE_BITMAP_BYTES: usize = 256 << 20;
const MAX_BITMAP_VECTORS: u64 = (MAX_DENSE_BITMAP_BYTES as u64) * 8;

/// The persisted label-index storage format.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelIndexFormat {
    /// One contiguous dense bit slice per encoded label.
    Bitslice = 0,
    /// One serialized Roaring posting list per encoded label.
    Bitmap = 1,
}

/// The outer Boolean structure of a clause list passed to [`EncodedLabelIndex::query`].
#[allow(clippy::upper_case_acronyms)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterExpressionType {
    /// Outer OR with `&`-separated labels inside each string:
    /// `["A&B", "C&D"]` means `(A AND B) OR (C AND D)`.
    ORMajor = 0,
    /// Outer AND with `|`-separated labels inside each string:
    /// `["A|B", "C|D"]` means `(A OR B) AND (C OR D)`.
    ANDMajor = 1,
}

#[derive(Debug, Error)]
pub enum EncodedLabelIndexError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Invalid(String),
}

enum LabelStorage {
    Bitslice {
        words_per_label: usize,
        bits: Box<[u64]>,
    },
    Bitmap {
        postings: Box<[RoaringBitmap]>,
    },
}

/// An immutable encoded label index loaded from a versioned label-index file.
pub struct EncodedLabelIndex {
    labels: Box<[String]>,
    label_ids: HashMap<String, u32>,
    num_vectors: u32,
    storage: LabelStorage,
}

impl std::fmt::Debug for EncodedLabelIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodedLabelIndex")
            .field("num_labels", &self.labels.len())
            .field("num_vectors", &self.num_vectors)
            .field("format", &self.format())
            .finish()
    }
}

#[derive(Debug, Clone, Copy)]
enum PlanKind {
    OrMajor,
    AndMajor,
}

impl From<FilterExpressionType> for PlanKind {
    fn from(value: FilterExpressionType) -> Self {
        match value {
            FilterExpressionType::ORMajor => Self::OrMajor,
            FilterExpressionType::ANDMajor => Self::AndMajor,
        }
    }
}

#[derive(Debug)]
struct CompiledPlan {
    kind: PlanKind,
    clause_offsets: Box<[usize]>,
    label_ids: Box<[Option<u32>]>,
}

enum QueryStorage<'a> {
    Bitslice {
        words_per_label: usize,
        bits: &'a [u64],
        plan: CompiledPlan,
    },
    DenseBitmap {
        bits: Box<[u64]>,
    },
}

/// Query-scoped label provider backed by an [`EncodedLabelIndex`].
pub struct EncodedLabelQuery<'a> {
    num_vectors: u32,
    storage: QueryStorage<'a>,
}

impl std::fmt::Debug for EncodedLabelQuery<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodedLabelQuery")
            .field("num_vectors", &self.num_vectors)
            .finish()
    }
}

impl QueryLabelProvider<u32> for EncodedLabelQuery<'_> {
    fn is_match(&self, vec_id: u32) -> bool {
        if vec_id >= self.num_vectors {
            return false;
        }

        match &self.storage {
            QueryStorage::Bitslice {
                words_per_label,
                bits,
                plan,
            } => plan.matches_bitslice(*words_per_label, bits, vec_id),
            QueryStorage::DenseBitmap { bits } => dense_contains(bits, vec_id),
        }
    }
}

impl CompiledPlan {
    fn matches_bitslice(&self, words_per_label: usize, bits: &[u64], vec_id: u32) -> bool {
        let terminal_matches = |label_id: Option<u32>| {
            label_id.is_some_and(|label_id| {
                let label_id = label_id as usize;
                let vec_id = vec_id as usize;
                let word = bits[label_id * words_per_label + vec_id / 64];
                word & (1u64 << (vec_id % 64)) != 0
            })
        };

        match self.kind {
            PlanKind::OrMajor => self.clause_offsets.windows(2).any(|clause| {
                self.label_ids[clause[0]..clause[1]]
                    .iter()
                    .copied()
                    .all(terminal_matches)
            }),
            PlanKind::AndMajor => self.clause_offsets.windows(2).all(|clause| {
                self.label_ids[clause[0]..clause[1]]
                    .iter()
                    .copied()
                    .any(terminal_matches)
            }),
        }
    }
}

impl EncodedLabelIndex {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, EncodedLabelIndexError> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; LABEL_INDEX_MAGIC.len()];
        reader.read_exact(&mut magic)?;
        if magic != LABEL_INDEX_MAGIC {
            return Err(EncodedLabelIndexError::Invalid(
                "invalid label-index file magic".to_string(),
            ));
        }

        let version = read_u32(&mut reader)?;
        if version != LABEL_INDEX_VERSION {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "unsupported label-index version {version}"
            )));
        }

        let format = match read_u32(&mut reader)? {
            0 => LabelIndexFormat::Bitslice,
            1 => LabelIndexFormat::Bitmap,
            value => {
                return Err(EncodedLabelIndexError::Invalid(format!(
                    "unsupported label-index format {value}"
                )));
            }
        };
        let num_vectors = u32::try_from(read_u64(&mut reader)?).map_err(|_| {
            EncodedLabelIndexError::Invalid("label-index vector count exceeds u32".to_string())
        })?;
        if num_vectors == 0 {
            return Err(EncodedLabelIndexError::Invalid(
                "label-index vector count cannot be zero".to_string(),
            ));
        }
        if format == LabelIndexFormat::Bitmap && u64::from(num_vectors) > MAX_BITMAP_VECTORS {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "bitmap vector count {num_vectors} exceeds limit {MAX_BITMAP_VECTORS}"
            )));
        }

        let num_labels = usize::try_from(read_u64(&mut reader)?).map_err(|_| {
            EncodedLabelIndexError::Invalid("label-index label count exceeds usize".to_string())
        })?;
        if num_labels > MAX_LABEL_COUNT {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "label-index label count {num_labels} exceeds limit {MAX_LABEL_COUNT}"
            )));
        }
        let minimum_dictionary_bytes = num_labels.checked_mul(4).ok_or_else(|| {
            EncodedLabelIndexError::Invalid("label dictionary size overflow".to_string())
        })?;
        ensure_remaining(
            &mut reader,
            file_len,
            minimum_dictionary_bytes,
            "label dictionary",
        )?;

        let mut labels = Vec::new();
        labels.try_reserve_exact(num_labels).map_err(|_| {
            EncodedLabelIndexError::Invalid("cannot reserve label dictionary".to_string())
        })?;

        let mut label_ids = HashMap::new();
        label_ids.try_reserve(num_labels).map_err(|_| {
            EncodedLabelIndexError::Invalid("cannot reserve label lookup map".to_string())
        })?;

        for id in 0..num_labels {
            let len = usize::try_from(read_u32(&mut reader)?).map_err(|_| {
                EncodedLabelIndexError::Invalid("label length exceeds usize".to_string())
            })?;
            if len > MAX_LABEL_LENGTH {
                return Err(EncodedLabelIndexError::Invalid(format!(
                    "label length {len} exceeds limit {MAX_LABEL_LENGTH}"
                )));
            }
            ensure_remaining(&mut reader, file_len, len, "label bytes")?;
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(len).map_err(|_| {
                EncodedLabelIndexError::Invalid("cannot reserve label bytes".to_string())
            })?;
            bytes.resize(len, 0);
            reader.read_exact(&mut bytes)?;

            let label = String::from_utf8(bytes).map_err(|_| {
                EncodedLabelIndexError::Invalid("label-index contains invalid UTF-8".to_string())
            })?;
            validate_label(&label)?;

            let id = u32::try_from(id).map_err(|_| {
                EncodedLabelIndexError::Invalid("label count exceeds u32".to_string())
            })?;
            if label_ids.insert(label.clone(), id).is_some() {
                return Err(EncodedLabelIndexError::Invalid(format!(
                    "duplicate label '{label}' in label-index"
                )));
            }
            labels.push(label);
        }

        let storage = match format {
            LabelIndexFormat::Bitslice => {
                let words_per_label = usize::try_from(read_u64(&mut reader)?).map_err(|_| {
                    EncodedLabelIndexError::Invalid("bitslice row length exceeds usize".to_string())
                })?;
                let expected_words = (num_vectors as usize).div_ceil(64);
                if words_per_label != expected_words {
                    return Err(EncodedLabelIndexError::Invalid(format!(
                        "bitslice row has {words_per_label} words; expected {expected_words}"
                    )));
                }
                let total_words = num_labels.checked_mul(words_per_label).ok_or_else(|| {
                    EncodedLabelIndexError::Invalid("bitslice allocation size overflow".to_string())
                })?;
                let payload_bytes = total_words.checked_mul(8).ok_or_else(|| {
                    EncodedLabelIndexError::Invalid("bitslice byte size overflow".to_string())
                })?;
                ensure_remaining(&mut reader, file_len, payload_bytes, "bitslice payload")?;

                let mut bits = Vec::new();
                bits.try_reserve_exact(total_words).map_err(|_| {
                    EncodedLabelIndexError::Invalid("cannot reserve bitslice payload".to_string())
                })?;
                for _ in 0..total_words {
                    bits.push(read_u64(&mut reader)?);
                }
                validate_bitslice_padding(&bits, num_labels, words_per_label, num_vectors)?;

                LabelStorage::Bitslice {
                    words_per_label,
                    bits: bits.into_boxed_slice(),
                }
            }
            LabelIndexFormat::Bitmap => {
                let mut postings = Vec::new();
                postings.try_reserve_exact(num_labels).map_err(|_| {
                    EncodedLabelIndexError::Invalid("cannot reserve posting list table".to_string())
                })?;
                for _ in 0..num_labels {
                    let len = usize::try_from(read_u64(&mut reader)?).map_err(|_| {
                        EncodedLabelIndexError::Invalid(
                            "serialized posting length exceeds usize".to_string(),
                        )
                    })?;
                    if len > MAX_POSTING_BYTES {
                        return Err(EncodedLabelIndexError::Invalid(format!(
                            "serialized posting length {len} exceeds limit {MAX_POSTING_BYTES}"
                        )));
                    }
                    ensure_remaining(&mut reader, file_len, len, "serialized posting")?;

                    let mut limited = reader.by_ref().take(len as u64);
                    let posting = RoaringBitmap::deserialize_from(&mut limited)?;
                    if limited.limit() != 0 {
                        return Err(EncodedLabelIndexError::Invalid(
                            "serialized posting length does not match payload".to_string(),
                        ));
                    }
                    if posting.max().is_some_and(|max| max >= num_vectors) {
                        return Err(EncodedLabelIndexError::Invalid(
                            "serialized posting contains an out-of-range vector ID".to_string(),
                        ));
                    }
                    postings.push(posting);
                }

                LabelStorage::Bitmap {
                    postings: postings.into_boxed_slice(),
                }
            }
        };

        if reader.read(&mut [0u8; 1])? != 0 {
            return Err(EncodedLabelIndexError::Invalid(
                "label-index contains trailing bytes".to_string(),
            ));
        }

        Ok(Self {
            labels: labels.into_boxed_slice(),
            label_ids,
            num_vectors,
            storage,
        })
    }

    pub fn query<S>(
        &self,
        clauses: &[S],
        expression_type: FilterExpressionType,
    ) -> Result<EncodedLabelQuery<'_>, EncodedLabelIndexError>
    where
        S: AsRef<str>,
    {
        let plan = compile_plan(clauses, expression_type.into(), &self.label_ids)?;

        let storage = match &self.storage {
            LabelStorage::Bitslice {
                words_per_label,
                bits,
            } => QueryStorage::Bitslice {
                words_per_label: *words_per_label,
                bits,
                plan,
            },
            LabelStorage::Bitmap { postings } => QueryStorage::DenseBitmap {
                bits: materialize_bitmap(&plan, postings, self.num_vectors)?.into_boxed_slice(),
            },
        };

        Ok(EncodedLabelQuery {
            num_vectors: self.num_vectors,
            storage,
        })
    }

    fn format(&self) -> LabelIndexFormat {
        match &self.storage {
            LabelStorage::Bitslice { .. } => LabelIndexFormat::Bitslice,
            LabelStorage::Bitmap { .. } => LabelIndexFormat::Bitmap,
        }
    }
}

/// Encode a JSONL label file into a versioned Bitslice or Roaring-bitmap label index.
///
/// Supported JSONL rows are:
///
/// - objects with optional `doc_id`; `true` fields use the field name, other scalar values use
///   `field=value`;
/// - objects with a `labels` string array;
/// - a raw string or string array, using the line number as the document ID.
pub fn encode_label_index_jsonl(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    format: LabelIndexFormat,
) -> Result<(), EncodedLabelIndexError> {
    encode_jsonl(input_path, output_path, format)
}

fn compile_plan<S: AsRef<str>>(
    clauses: &[S],
    kind: PlanKind,
    label_ids: &HashMap<String, u32>,
) -> Result<CompiledPlan, EncodedLabelIndexError> {
    if clauses.is_empty() {
        return Err(EncodedLabelIndexError::Invalid(
            "filter must contain at least one clause".to_string(),
        ));
    }

    let delimiter = match kind {
        PlanKind::OrMajor => '&',
        PlanKind::AndMajor => '|',
    };
    let mut clause_offsets = vec![0usize];
    let mut encoded = Vec::new();

    for clause in clauses {
        let mut terminal_count = 0usize;
        for terminal in clause.as_ref().split(delimiter) {
            let terminal = terminal.trim();
            validate_label(terminal)?;
            encoded.push(label_ids.get(terminal).copied());
            terminal_count += 1;
        }
        if terminal_count == 0 {
            return Err(EncodedLabelIndexError::Invalid(
                "filter clauses cannot be empty".to_string(),
            ));
        }
        clause_offsets.push(encoded.len());
    }

    Ok(CompiledPlan {
        kind,
        clause_offsets: clause_offsets.into_boxed_slice(),
        label_ids: encoded.into_boxed_slice(),
    })
}

fn materialize_bitmap(
    plan: &CompiledPlan,
    postings: &[RoaringBitmap],
    num_vectors: u32,
) -> Result<Vec<u64>, EncodedLabelIndexError> {
    let result = match plan.kind {
        PlanKind::OrMajor => {
            let mut result = RoaringBitmap::new();
            for clause in plan.clause_offsets.windows(2) {
                let labels = &plan.label_ids[clause[0]..clause[1]];
                if labels.iter().any(Option::is_none) {
                    continue;
                }
                let mut labels = labels.iter().filter_map(|label| *label);
                let Some(first) = labels.next() else {
                    continue;
                };
                let mut clause_result = postings[first as usize].clone();
                for label in labels {
                    clause_result &= &postings[label as usize];
                    if clause_result.is_empty() {
                        break;
                    }
                }
                result |= clause_result;
            }
            result
        }
        PlanKind::AndMajor => {
            let mut result: Option<RoaringBitmap> = None;
            for clause in plan.clause_offsets.windows(2) {
                let mut clause_result = RoaringBitmap::new();
                for label in plan.label_ids[clause[0]..clause[1]]
                    .iter()
                    .filter_map(|label| *label)
                {
                    clause_result |= &postings[label as usize];
                }
                result = Some(match result {
                    None => clause_result,
                    Some(mut result) => {
                        result &= clause_result;
                        result
                    }
                });
                if result.as_ref().is_some_and(RoaringBitmap::is_empty) {
                    break;
                }
            }
            result.unwrap_or_default()
        }
    };

    densify(&result, num_vectors)
}

fn dense_contains(bits: &[u64], vec_id: u32) -> bool {
    let vec_id = vec_id as usize;
    bits[vec_id / 64] & (1u64 << (vec_id % 64)) != 0
}

fn densify(result: &RoaringBitmap, num_vectors: u32) -> Result<Vec<u64>, EncodedLabelIndexError> {
    let words = (num_vectors as usize).div_ceil(64);
    let bytes = words.checked_mul(8).ok_or_else(|| {
        EncodedLabelIndexError::Invalid("dense bitmap byte size overflow".to_string())
    })?;
    if bytes > MAX_DENSE_BITMAP_BYTES {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "dense query bitmap requires {bytes} bytes; limit is {MAX_DENSE_BITMAP_BYTES}"
        )));
    }
    let mut bits = Vec::new();
    bits.try_reserve_exact(words).map_err(|_| {
        EncodedLabelIndexError::Invalid("cannot reserve dense query bitmap".to_string())
    })?;
    bits.resize(words, 0);
    for vec_id in result {
        let vec_id = vec_id as usize;
        bits[vec_id / 64] |= 1u64 << (vec_id % 64);
    }
    Ok(bits)
}

fn encode_jsonl(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    format: LabelIndexFormat,
) -> Result<(), EncodedLabelIndexError> {
    let reader = BufReader::new(File::open(input_path)?);
    let mut labels = Vec::<String>::new();
    let mut label_ids = HashMap::<String, u32>::new();
    let mut postings = Vec::<RoaringBitmap>::new();
    let mut max_doc_id = None::<u32>;
    let mut seen_doc_ids = HashSet::<u32>::new();

    let mut record_number = 0u32;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let default_doc_id = record_number;
        record_number = record_number.checked_add(1).ok_or_else(|| {
            EncodedLabelIndexError::Invalid("JSONL record count exceeds u32".to_string())
        })?;
        let (doc_id, document_labels) = parse_document(&line, default_doc_id)?;
        if !seen_doc_ids.insert(doc_id) {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "duplicate doc_id {doc_id} in label JSONL"
            )));
        }
        max_doc_id = Some(max_doc_id.map_or(doc_id, |current| current.max(doc_id)));

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
                postings.push(RoaringBitmap::new());
                label_id
            };
            postings[label_id as usize].insert(doc_id);
        }
    }

    let num_vectors = max_doc_id
        .and_then(|doc_id| doc_id.checked_add(1))
        .ok_or_else(|| {
            EncodedLabelIndexError::Invalid("label JSONL contains no documents".to_string())
        })?;
    write_label_index(output_path, format, num_vectors, &labels, &postings)
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

fn validate_label(label: &str) -> Result<(), EncodedLabelIndexError> {
    if label.is_empty() {
        return Err(EncodedLabelIndexError::Invalid(
            "labels cannot be empty".to_string(),
        ));
    }
    if label.contains(['&', '|']) {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "label '{label}' contains a reserved filter delimiter"
        )));
    }
    if label.as_bytes().contains(&0) {
        return Err(EncodedLabelIndexError::Invalid(
            "labels cannot contain embedded NUL bytes".to_string(),
        ));
    }
    if label.len() > MAX_LABEL_LENGTH {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "label length {} exceeds limit {MAX_LABEL_LENGTH}",
            label.len()
        )));
    }
    if label.trim() != label {
        return Err(EncodedLabelIndexError::Invalid(
            "labels cannot have leading or trailing whitespace".to_string(),
        ));
    }
    Ok(())
}

fn validate_bitslice_padding(
    bits: &[u64],
    num_labels: usize,
    words_per_label: usize,
    num_vectors: u32,
) -> Result<(), EncodedLabelIndexError> {
    let remainder = (num_vectors as usize) % 64;
    if remainder == 0 || words_per_label == 0 {
        return Ok(());
    }

    let valid_mask = (1u64 << remainder) - 1;
    for label_id in 0..num_labels {
        let last_word = bits[(label_id + 1) * words_per_label - 1];
        if last_word & !valid_mask != 0 {
            return Err(EncodedLabelIndexError::Invalid(
                "bitslice padding contains out-of-range vector IDs".to_string(),
            ));
        }
    }
    Ok(())
}

fn write_label_index(
    path: impl AsRef<Path>,
    format: LabelIndexFormat,
    num_vectors: u32,
    labels: &[String],
    postings: &[RoaringBitmap],
) -> Result<(), EncodedLabelIndexError> {
    if labels.len() > MAX_LABEL_COUNT {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "label count {} exceeds limit {MAX_LABEL_COUNT}",
            labels.len()
        )));
    }
    if labels.len() != postings.len() {
        return Err(EncodedLabelIndexError::Invalid(
            "label dictionary and posting counts differ".to_string(),
        ));
    }
    if num_vectors == 0 {
        return Err(EncodedLabelIndexError::Invalid(
            "label-index vector count cannot be zero".to_string(),
        ));
    }
    if format == LabelIndexFormat::Bitmap && u64::from(num_vectors) > MAX_BITMAP_VECTORS {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "bitmap vector count {num_vectors} exceeds limit {MAX_BITMAP_VECTORS}"
        )));
    }

    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(&LABEL_INDEX_MAGIC)?;
    write_u32(&mut writer, LABEL_INDEX_VERSION)?;
    write_u32(&mut writer, format as u32)?;
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

    match format {
        LabelIndexFormat::Bitslice => {
            let words_per_label = (num_vectors as usize).div_ceil(64);
            write_u64(&mut writer, words_per_label as u64)?;

            let total_words = labels.len().checked_mul(words_per_label).ok_or_else(|| {
                EncodedLabelIndexError::Invalid("bitslice output size overflow".to_string())
            })?;
            let mut bits = Vec::new();
            bits.try_reserve_exact(total_words).map_err(|_| {
                EncodedLabelIndexError::Invalid("cannot reserve bitslice output".to_string())
            })?;
            bits.resize(total_words, 0);

            for (label_id, posting) in postings.iter().enumerate() {
                let row = &mut bits[label_id * words_per_label..(label_id + 1) * words_per_label];
                for vec_id in posting {
                    let vec_id = vec_id as usize;
                    row[vec_id / 64] |= 1u64 << (vec_id % 64);
                }
            }

            for word in bits {
                write_u64(&mut writer, word)?;
            }
        }
        LabelIndexFormat::Bitmap => {
            for posting in postings {
                let serialized_size = posting.serialized_size();
                if serialized_size > MAX_POSTING_BYTES {
                    return Err(EncodedLabelIndexError::Invalid(format!(
                        "serialized posting length {serialized_size} exceeds limit {MAX_POSTING_BYTES}"
                    )));
                }

                let mut bytes = Vec::new();
                bytes.try_reserve_exact(serialized_size).map_err(|_| {
                    EncodedLabelIndexError::Invalid("cannot reserve serialized posting".to_string())
                })?;
                posting.serialize_into(&mut bytes)?;
                write_u64(&mut writer, bytes.len() as u64)?;
                writer.write_all(&bytes)?;
            }
        }
    }

    writer.flush()?;
    Ok(())
}

fn write_u32(writer: &mut impl Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u64(writer: &mut impl Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn ensure_remaining(
    reader: &mut BufReader<File>,
    file_len: u64,
    required: usize,
    context: &str,
) -> Result<(), EncodedLabelIndexError> {
    let position = reader.stream_position()?;
    let remaining = file_len.checked_sub(position).ok_or_else(|| {
        EncodedLabelIndexError::Invalid("reader position exceeds file length".to_string())
    })?;
    if required as u64 > remaining {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "{context} requires {required} bytes but only {remaining} remain"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;

    fn sample_jsonl() -> &'static str {
        concat!(
            "{\"doc_id\":0,\"A\":true,\"group\":\"x\"}\n",
            "{\"doc_id\":1,\"B\":true,\"group\":\"x\"}\n",
            "{\"doc_id\":2,\"A\":true,\"B\":true,\"score\":2}\n",
            "{\"doc_id\":3,\"labels\":[\"C\",\"D\"]}\n",
        )
    }

    fn compile<'a>(
        index: &'a EncodedLabelIndex,
        clauses: &[&str],
        expression_type: FilterExpressionType,
    ) -> EncodedLabelQuery<'a> {
        index.query(clauses, expression_type).unwrap()
    }

    fn round_trip(format: LabelIndexFormat) -> EncodedLabelIndex {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let output = dir.path().join("labels.bin");
        std::fs::write(&input, sample_jsonl()).unwrap();
        encode_label_index_jsonl(&input, &output, format).unwrap();
        EncodedLabelIndex::load(output).unwrap()
    }

    #[test]
    fn bitslice_and_bitmap_match_for_or_major() {
        for format in [LabelIndexFormat::Bitslice, LabelIndexFormat::Bitmap] {
            let index = round_trip(format);
            let query = compile(&index, &["A&B", "C&D"], FilterExpressionType::ORMajor);
            assert!(!query.is_match(0));
            assert!(!query.is_match(1));
            assert!(query.is_match(2));
            assert!(query.is_match(3));
        }
    }

    #[test]
    fn bitslice_and_bitmap_match_for_and_major() {
        for format in [LabelIndexFormat::Bitslice, LabelIndexFormat::Bitmap] {
            let index = round_trip(format);
            let query = compile(
                &index,
                &["A|B", "group=x|score=2"],
                FilterExpressionType::ANDMajor,
            );
            assert!(query.is_match(0));
            assert!(query.is_match(1));
            assert!(query.is_match(2));
            assert!(!query.is_match(3));
        }
    }

    #[test]
    fn query_accepts_owned_strings() {
        let index = round_trip(LabelIndexFormat::Bitmap);
        let clauses = vec!["A&B".to_string(), "C&D".to_string()];
        let query = index
            .query(&clauses, FilterExpressionType::ORMajor)
            .unwrap();
        assert!(!query.is_match(0));
        assert!(query.is_match(2));
        assert!(query.is_match(3));
    }

    #[test]
    fn unknown_labels_are_non_matches() {
        let index = round_trip(LabelIndexFormat::Bitslice);
        let query = compile(&index, &["missing"], FilterExpressionType::ORMajor);
        for id in 0..4 {
            assert!(!query.is_match(id));
        }
    }

    #[test]
    fn raw_and_object_jsonl_forms_are_supported() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let output = dir.path().join("labels.bin");
        std::fs::write(
            &input,
            concat!(
                "\"solo\"\n",
                "[\"left\",\"right\"]\n",
                "{\"doc_id\":4,\"enabled\":true,\"group\":\"g\",\"count\":2,\"deleted\":false,\"labels\":[\"inline\"]}\n",
            ),
        )
        .unwrap();

        encode_label_index_jsonl(&input, &output, LabelIndexFormat::Bitslice).unwrap();
        let index = EncodedLabelIndex::load(output).unwrap();

        let raw = compile(&index, &["solo"], FilterExpressionType::ORMajor);
        assert!(raw.is_match(0));
        assert!(!raw.is_match(1));

        let array = compile(&index, &["left&right"], FilterExpressionType::ORMajor);
        assert!(array.is_match(1));
        assert!(!array.is_match(0));

        let object = compile(
            &index,
            &["enabled&group=g&count=2&deleted=false&inline"],
            FilterExpressionType::ORMajor,
        );
        assert!(object.is_match(4));
        assert!(!object.is_match(3));
    }

    #[test]
    fn blank_lines_do_not_shift_implicit_document_ids() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let output = dir.path().join("labels.bin");
        std::fs::write(&input, "\n\"A\"\n\n\"B\"\n").unwrap();
        encode_label_index_jsonl(&input, &output, LabelIndexFormat::Bitslice).unwrap();
        let index = EncodedLabelIndex::load(output).unwrap();

        let a = compile(&index, &["A"], FilterExpressionType::ORMajor);
        let b = compile(&index, &["B"], FilterExpressionType::ORMajor);
        assert!(a.is_match(0));
        assert!(!a.is_match(1));
        assert!(!b.is_match(0));
        assert!(b.is_match(1));
    }

    #[test]
    fn reserved_delimiters_are_rejected_in_label_files() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let output = dir.path().join("labels.bin");
        std::fs::write(&input, "{\"doc_id\":0,\"labels\":[\"A&B\"]}\n").unwrap();
        assert!(encode_label_index_jsonl(&input, &output, LabelIndexFormat::Bitslice).is_err());
    }

    #[test]
    fn embedded_nul_labels_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let output = dir.path().join("labels.bin");
        std::fs::write(&input, "{\"doc_id\":0,\"labels\":[\"A\\u0000B\"]}\n").unwrap();
        assert!(encode_label_index_jsonl(&input, &output, LabelIndexFormat::Bitslice).is_err());
    }

    #[test]
    fn labels_with_outer_whitespace_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let output = dir.path().join("labels.bin");
        std::fs::write(&input, "{\"doc_id\":0,\"labels\":[\" A\"]}\n").unwrap();
        assert!(encode_label_index_jsonl(&input, &output, LabelIndexFormat::Bitslice).is_err());
    }

    #[test]
    fn duplicate_document_ids_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let output = dir.path().join("labels.bin");
        std::fs::write(
            &input,
            "{\"doc_id\":0,\"A\":true}\n{\"doc_id\":0,\"B\":true}\n",
        )
        .unwrap();
        assert!(encode_label_index_jsonl(&input, &output, LabelIndexFormat::Bitslice).is_err());
    }

    #[test]
    fn load_rejects_invalid_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.bin");
        std::fs::write(&path, b"not-an-index").unwrap();
        assert!(EncodedLabelIndex::load(path).is_err());
    }

    #[test]
    fn load_rejects_out_of_range_posting_ids() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.bin");
        let mut posting = RoaringBitmap::new();
        posting.insert(1);
        write_label_index(
            &path,
            LabelIndexFormat::Bitmap,
            1,
            &["A".to_string()],
            &[posting],
        )
        .unwrap();
        assert!(EncodedLabelIndex::load(path).is_err());
    }

    #[test]
    fn load_rejects_excessive_label_count_before_allocation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.bin");
        let mut writer = BufWriter::new(File::create(&path).unwrap());
        writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
        write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
        write_u32(&mut writer, LabelIndexFormat::Bitslice as u32).unwrap();
        write_u64(&mut writer, 1).unwrap();
        write_u64(&mut writer, (MAX_LABEL_COUNT as u64) + 1).unwrap();
        writer.flush().unwrap();
        assert!(EncodedLabelIndex::load(path).is_err());
    }

    #[test]
    fn load_rejects_set_bits_in_bitslice_padding() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.bin");
        let mut writer = BufWriter::new(File::create(&path).unwrap());
        writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
        write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
        write_u32(&mut writer, LabelIndexFormat::Bitslice as u32).unwrap();
        write_u64(&mut writer, 1).unwrap();
        write_u64(&mut writer, 1).unwrap();
        write_u32(&mut writer, 1).unwrap();
        writer.write_all(b"A").unwrap();
        write_u64(&mut writer, 1).unwrap();
        write_u64(&mut writer, 1u64 << 1).unwrap();
        writer.flush().unwrap();
        assert!(EncodedLabelIndex::load(path).is_err());
    }

    #[test]
    fn load_rejects_oversized_bitmap_index() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.bin");
        let oversized_vectors = u32::try_from((MAX_DENSE_BITMAP_BYTES / 8) * 64 + 1).unwrap();
        let mut writer = BufWriter::new(File::create(&path).unwrap());
        writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
        write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
        write_u32(&mut writer, LabelIndexFormat::Bitmap as u32).unwrap();
        write_u64(&mut writer, u64::from(oversized_vectors)).unwrap();
        write_u64(&mut writer, 0).unwrap();
        writer.flush().unwrap();

        assert!(EncodedLabelIndex::load(path).is_err());
    }

    #[test]
    fn load_rejects_trailing_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let output = dir.path().join("labels.bin");
        std::fs::write(&input, sample_jsonl()).unwrap();
        encode_label_index_jsonl(&input, &output, LabelIndexFormat::Bitmap).unwrap();

        let mut file = OpenOptions::new().append(true).open(&output).unwrap();
        file.write_all(&[0]).unwrap();
        drop(file);

        assert!(EncodedLabelIndex::load(output).is_err());
    }

    #[test]
    fn load_rejects_zero_vector_count() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.bin");
        let mut writer = BufWriter::new(File::create(&path).unwrap());
        writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
        write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
        write_u32(&mut writer, LabelIndexFormat::Bitslice as u32).unwrap();
        write_u64(&mut writer, 0).unwrap();
        writer.flush().unwrap();
        assert!(EncodedLabelIndex::load(path).is_err());
    }
}
