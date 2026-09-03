/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Dense label-index loading and query serving.

use crate::{
    error::EncodedLabelIndexError,
    format::{
        ensure_remaining, read_u32, read_u64, validate_bitslice_padding, validate_label,
        BITSLICE_FORMAT, LABEL_INDEX_MAGIC, LABEL_INDEX_VERSION, MAX_LABEL_COUNT, MAX_LABEL_LENGTH,
    },
};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
    marker::PhantomData,
    path::Path,
    sync::Arc,
};

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
