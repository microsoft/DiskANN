/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Dense, versioned label-index encoding and flat DNF/CNF query evaluation for DiskANN.

use serde_json::{Map, Value};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Read, Seek, Write},
    marker::PhantomData,
    path::Path,
    sync::Arc,
};
use thiserror::Error;

const LABEL_INDEX_MAGIC: [u8; 8] = *b"DANLBL01";
const LABEL_INDEX_VERSION: u32 = 1;
const BITSLICE_FORMAT: u32 = 0;
const MAX_LABEL_COUNT: usize = 1_000_000;
const MAX_LABEL_LENGTH: usize = 1 << 20;

/// The Boolean normal form of a flat clause list passed to [`EncodedLabelIndex::query`].
#[allow(clippy::upper_case_acronyms)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterExpressionType {
    /// An outer OR of `&`-separated AND clauses.
    ///
    /// `["A&B", "C&D"]` represents `(A AND B) OR (C AND D)`.
    DNF = 0,
    /// An outer AND of `|`-separated OR clauses.
    ///
    /// `["A|B", "C|D"]` represents `(A OR B) AND (C OR D)`.
    CNF = 1,
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

/// An immutable dense label index loaded from a versioned label-index file.
pub struct EncodedLabelIndex {
    labels: Box<[String]>,
    label_ids: HashMap<String, u32>,
    num_vectors: u32,
    words_per_label: usize,
    bits: Arc<[u64]>,
}

impl std::fmt::Debug for EncodedLabelIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodedLabelIndex")
            .field("num_labels", &self.labels.len())
            .field("num_vectors", &self.num_vectors)
            .finish()
    }
}

#[derive(Debug)]
struct CompiledPlan {
    expression_type: FilterExpressionType,
    clause_offsets: Box<[usize]>,
    label_ids: Box<[Option<u32>]>,
}

struct ScannedLabels {
    labels: Vec<String>,
    label_ids: HashMap<String, u32>,
    num_vectors: u32,
}

/// Query-scoped evaluator compiled from an [`EncodedLabelIndex`].
///
/// Queries share the immutable dense index payload, so they remain usable after the source index
/// is dropped.
pub struct EncodedLabelQuery<'a> {
    num_vectors: u32,
    words_per_label: usize,
    bits: Arc<[u64]>,
    plan: CompiledPlan,
    _lifetime: PhantomData<&'a ()>,
}

impl std::fmt::Debug for EncodedLabelQuery<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodedLabelQuery")
            .field("num_vectors", &self.num_vectors)
            .field("expression_type", &self.plan.expression_type)
            .finish()
    }
}

impl EncodedLabelQuery<'_> {
    /// Return whether `vec_id` satisfies this compiled label query.
    pub fn is_match(&self, vec_id: u32) -> bool {
        if vec_id >= self.num_vectors {
            return false;
        }

        let terminal_matches = |label_id: Option<u32>| {
            label_id.is_some_and(|label_id| {
                let word =
                    self.bits[label_id as usize * self.words_per_label + vec_id as usize / 64];
                word & (1u64 << (vec_id % 64)) != 0
            })
        };

        match self.plan.expression_type {
            FilterExpressionType::DNF => self.plan.clause_offsets.windows(2).any(|clause| {
                self.plan.label_ids[clause[0]..clause[1]]
                    .iter()
                    .copied()
                    .all(terminal_matches)
            }),
            FilterExpressionType::CNF => self.plan.clause_offsets.windows(2).all(|clause| {
                self.plan.label_ids[clause[0]..clause[1]]
                    .iter()
                    .copied()
                    .any(terminal_matches)
            }),
        }
    }
}

impl EncodedLabelIndex {
    /// Load a dense label index from `path`.
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

        let format = read_u32(&mut reader)?;
        if format != BITSLICE_FORMAT {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "unsupported label-index format {format}; only dense bitslice format 0 is supported"
            )));
        }

        let num_vectors = u32::try_from(read_u64(&mut reader)?).map_err(|_| {
            EncodedLabelIndexError::Invalid("label-index vector count exceeds u32".to_string())
        })?;
        if num_vectors == 0 {
            return Err(EncodedLabelIndexError::Invalid(
                "label-index vector count cannot be zero".to_string(),
            ));
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

        if reader.read(&mut [0u8; 1])? != 0 {
            return Err(EncodedLabelIndexError::Invalid(
                "label-index contains trailing bytes".to_string(),
            ));
        }

        Ok(Self {
            labels: labels.into_boxed_slice(),
            label_ids,
            num_vectors,
            words_per_label,
            bits: Arc::from(bits),
        })
    }

    /// Return the number of vectors covered by this index.
    pub fn num_vectors(&self) -> u32 {
        self.num_vectors
    }

    /// Return the number of encoded labels.
    pub fn num_labels(&self) -> usize {
        self.labels.len()
    }

    /// Return whether this index contains an encoded label.
    pub fn contains_label(&self, label: &str) -> bool {
        self.label_ids.contains_key(label)
    }

    /// Compile a flat clause-list query using DNF or CNF semantics.
    pub fn query<S>(
        &self,
        clauses: &[S],
        expression_type: FilterExpressionType,
    ) -> Result<EncodedLabelQuery<'static>, EncodedLabelIndexError>
    where
        S: AsRef<str>,
    {
        Ok(EncodedLabelQuery {
            num_vectors: self.num_vectors,
            words_per_label: self.words_per_label,
            bits: Arc::clone(&self.bits),
            plan: compile_plan(clauses, expression_type, &self.label_ids)?,
            _lifetime: PhantomData,
        })
    }
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

fn compile_plan<S: AsRef<str>>(
    clauses: &[S],
    expression_type: FilterExpressionType,
    label_ids: &HashMap<String, u32>,
) -> Result<CompiledPlan, EncodedLabelIndexError> {
    if clauses.is_empty() {
        return Err(EncodedLabelIndexError::Invalid(
            "filter must contain at least one clause".to_string(),
        ));
    }

    let delimiter = match expression_type {
        FilterExpressionType::DNF => '&',
        FilterExpressionType::CNF => '|',
    };
    let mut clause_offsets = vec![0usize];
    let mut encoded = Vec::new();

    for clause in clauses {
        let clause = clause.as_ref();
        if clause.is_empty() {
            return Err(EncodedLabelIndexError::Invalid(
                "filter clauses cannot be empty".to_string(),
            ));
        }
        for terminal in clause.split(delimiter) {
            let terminal = terminal.trim();
            validate_label(terminal)?;
            encoded.push(label_ids.get(terminal).copied());
        }
        clause_offsets.push(encoded.len());
    }

    Ok(CompiledPlan {
        expression_type,
        clause_offsets: clause_offsets.into_boxed_slice(),
        label_ids: encoded.into_boxed_slice(),
    })
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

    fn round_trip() -> EncodedLabelIndex {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let output = dir.path().join("labels.bin");
        std::fs::write(&input, sample_jsonl()).unwrap();
        encode_label_index_jsonl(&input, &output).unwrap();
        EncodedLabelIndex::load(output).unwrap()
    }

    fn compile(
        index: &EncodedLabelIndex,
        clauses: &[&str],
        expression_type: FilterExpressionType,
    ) -> EncodedLabelQuery<'static> {
        index.query(clauses, expression_type).unwrap()
    }

    fn matching_ids(query: &EncodedLabelQuery, num_vectors: u32) -> Vec<u32> {
        (0..num_vectors)
            .filter(|&vec_id| query.is_match(vec_id))
            .collect()
    }

    fn assert_send_sync_static<T: Send + Sync + 'static>(_: &T) {}

    #[test]
    fn dense_round_trip_supports_dnf() {
        let index = round_trip();
        let query = compile(&index, &["A&B", "C&D"], FilterExpressionType::DNF);
        assert_eq!(matching_ids(&query, index.num_vectors()), vec![2, 3]);
    }

    #[test]
    fn dense_round_trip_supports_cnf() {
        let index = round_trip();
        let query = compile(
            &index,
            &["A|B", "group=x|score=2"],
            FilterExpressionType::CNF,
        );
        assert_eq!(matching_ids(&query, index.num_vectors()), vec![0, 1, 2]);
    }

    #[test]
    fn query_accepts_owned_strings() {
        let index = round_trip();
        let clauses = vec!["A&B".to_string(), "C&D".to_string()];
        let query = index.query(&clauses, FilterExpressionType::DNF).unwrap();
        assert_eq!(matching_ids(&query, index.num_vectors()), vec![2, 3]);
    }

    #[test]
    fn unknown_labels_follow_normal_form_semantics() {
        let index = round_trip();
        let dnf = compile(&index, &["missing"], FilterExpressionType::DNF);
        assert!(matching_ids(&dnf, index.num_vectors()).is_empty());

        let cnf = compile(&index, &["missing|A"], FilterExpressionType::CNF);
        assert_eq!(matching_ids(&cnf, index.num_vectors()), vec![0, 2]);
    }

    #[test]
    fn compiled_query_remains_usable_after_index_drop() {
        let query = {
            let index = round_trip();
            Arc::new(
                index
                    .query(&["A&B", "C&D"], FilterExpressionType::DNF)
                    .unwrap(),
            )
        };
        assert_send_sync_static(query.as_ref());
        assert_eq!(matching_ids(&query, 4), vec![2, 3]);
    }

    #[test]
    fn query_rejects_empty_input_and_invalid_clauses() {
        let index = round_trip();
        assert!(index.query::<&str>(&[], FilterExpressionType::DNF).is_err());
        assert!(index.query(&[""], FilterExpressionType::DNF).is_err());
        assert!(index.query(&["A&&B"], FilterExpressionType::DNF).is_err());
        assert!(index.query(&["A|B"], FilterExpressionType::DNF).is_err());
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

        encode_label_index_jsonl(&input, &output).unwrap();
        let index = EncodedLabelIndex::load(output).unwrap();

        assert_eq!(
            matching_ids(
                &compile(&index, &["solo"], FilterExpressionType::DNF),
                index.num_vectors()
            ),
            vec![0]
        );
        assert_eq!(
            matching_ids(
                &compile(&index, &["left&right"], FilterExpressionType::DNF),
                index.num_vectors()
            ),
            vec![1]
        );
        assert_eq!(
            matching_ids(
                &compile(
                    &index,
                    &["enabled&group=g&count=2&deleted=false&inline"],
                    FilterExpressionType::DNF,
                ),
                index.num_vectors()
            ),
            vec![4]
        );
    }

    #[test]
    fn blank_lines_do_not_shift_implicit_document_ids() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let output = dir.path().join("labels.bin");
        std::fs::write(&input, "\n\"A\"\n\n\"B\"\n").unwrap();
        encode_label_index_jsonl(&input, &output).unwrap();
        let index = EncodedLabelIndex::load(output).unwrap();

        assert_eq!(
            matching_ids(
                &compile(&index, &["A"], FilterExpressionType::DNF),
                index.num_vectors()
            ),
            vec![0]
        );
        assert_eq!(
            matching_ids(
                &compile(&index, &["B"], FilterExpressionType::DNF),
                index.num_vectors()
            ),
            vec![1]
        );
    }

    #[test]
    fn invalid_labels_and_duplicate_document_ids_are_rejected() {
        let cases = [
            "{\"doc_id\":0,\"labels\":[\"A&B\"]}\n",
            "{\"doc_id\":0,\"labels\":[\"A\\u0000B\"]}\n",
            "{\"doc_id\":0,\"labels\":[\" A\"]}\n",
            "{\"doc_id\":0,\"A\":true}\n{\"doc_id\":0,\"B\":true}\n",
        ];
        for contents in cases {
            let dir = tempfile::tempdir().unwrap();
            let input = dir.path().join("labels.jsonl");
            let output = dir.path().join("labels.bin");
            std::fs::write(&input, contents).unwrap();
            assert!(encode_label_index_jsonl(&input, &output).is_err());
        }
    }

    #[test]
    fn load_rejects_invalid_magic_and_unsupported_formats() {
        let dir = tempfile::tempdir().unwrap();
        let invalid_magic = dir.path().join("invalid-magic.bin");
        std::fs::write(&invalid_magic, b"not-an-index").unwrap();
        assert!(EncodedLabelIndex::load(invalid_magic).is_err());

        for format in [1, 2, 3] {
            let path = dir.path().join(format!("format-{format}.bin"));
            let mut writer = BufWriter::new(File::create(&path).unwrap());
            writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
            write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
            write_u32(&mut writer, format).unwrap();
            writer.flush().unwrap();
            let error = EncodedLabelIndex::load(path).unwrap_err();
            assert!(error.to_string().contains("only dense bitslice"));
        }
    }

    #[test]
    fn load_rejects_excessive_label_count_before_allocation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.bin");
        let mut writer = BufWriter::new(File::create(&path).unwrap());
        writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
        write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
        write_u32(&mut writer, BITSLICE_FORMAT).unwrap();
        write_u64(&mut writer, 1).unwrap();
        write_u64(&mut writer, (MAX_LABEL_COUNT as u64) + 1).unwrap();
        writer.flush().unwrap();
        assert!(EncodedLabelIndex::load(path).is_err());
    }

    #[test]
    fn load_rejects_invalid_row_length_and_padding() {
        let dir = tempfile::tempdir().unwrap();

        let invalid_length = dir.path().join("invalid-length.bin");
        let mut writer = BufWriter::new(File::create(&invalid_length).unwrap());
        writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
        write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
        write_u32(&mut writer, BITSLICE_FORMAT).unwrap();
        write_u64(&mut writer, 1).unwrap();
        write_u64(&mut writer, 0).unwrap();
        write_u64(&mut writer, 2).unwrap();
        writer.flush().unwrap();
        assert!(EncodedLabelIndex::load(invalid_length).is_err());

        let invalid_padding = dir.path().join("invalid-padding.bin");
        let mut writer = BufWriter::new(File::create(&invalid_padding).unwrap());
        writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
        write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
        write_u32(&mut writer, BITSLICE_FORMAT).unwrap();
        write_u64(&mut writer, 1).unwrap();
        write_u64(&mut writer, 1).unwrap();
        write_u32(&mut writer, 1).unwrap();
        writer.write_all(b"A").unwrap();
        write_u64(&mut writer, 1).unwrap();
        write_u64(&mut writer, 1u64 << 1).unwrap();
        writer.flush().unwrap();
        assert!(EncodedLabelIndex::load(invalid_padding).is_err());
    }

    #[test]
    fn load_rejects_trailing_bytes_and_zero_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("labels.jsonl");
        let output = dir.path().join("labels.bin");
        std::fs::write(&input, sample_jsonl()).unwrap();
        encode_label_index_jsonl(&input, &output).unwrap();
        let mut file = OpenOptions::new().append(true).open(&output).unwrap();
        file.write_all(&[0]).unwrap();
        drop(file);
        assert!(EncodedLabelIndex::load(output).is_err());

        let zero_vectors = dir.path().join("zero-vectors.bin");
        let mut writer = BufWriter::new(File::create(&zero_vectors).unwrap());
        writer.write_all(&LABEL_INDEX_MAGIC).unwrap();
        write_u32(&mut writer, LABEL_INDEX_VERSION).unwrap();
        write_u32(&mut writer, BITSLICE_FORMAT).unwrap();
        write_u64(&mut writer, 0).unwrap();
        writer.flush().unwrap();
        assert!(EncodedLabelIndex::load(zero_vectors).is_err());
    }
}
