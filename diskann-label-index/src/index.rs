/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Label-index loading and query serving.

use crate::{
    bloom::{append_label_rows, validate_bloom_config, BloomFilterConfig, BLOOM_FORMAT},
    error::EncodedLabelIndexError,
    format::{
        ensure_remaining, read_u32, read_u64, validate_bitslice_padding, validate_label,
        BITSLICE_FORMAT, COUNTED_BLOOM_INDEX_VERSION, LABEL_INDEX_MAGIC, LABEL_INDEX_VERSION,
        MAX_LABEL_COUNT, MAX_LABEL_LENGTH,
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

enum LabelIndexPayload {
    Bitslice {
        words_per_row: usize,
        bits: Box<[u64]>,
    },
    Bloom {
        words_per_row: usize,
        hash_count: usize,
        bits: Box<[u64]>,
        label_rows: Box<[u32]>,
    },
}

/// An immutable label index loaded from a versioned label-index file.
pub struct EncodedLabelIndex {
    labels: Box<[String]>,
    label_ids: HashMap<String, u32>,
    counts: Option<Arc<[u32]>>,
    num_vectors: u32,
    payload: Arc<LabelIndexPayload>,
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
/// Queries share the immutable index payload, so they remain usable after the source index is
/// dropped.
pub struct EncodedLabelQuery<'a> {
    num_vectors: u32,
    payload: Arc<LabelIndexPayload>,
    plan: CompiledPlan,
    counts: Option<Arc<[u32]>>,
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
    /// Return a conservative bound on exact matching vectors, if exact label counts were stored.
    pub fn match_upper_bound(&self) -> Option<u64> {
        let counts = self.counts.as_ref()?;
        let count =
            |label_id: &Option<u32>| label_id.map_or(0, |id| u64::from(counts[id as usize]));
        let bound = match self.plan.expression_type {
            FilterExpressionType::DNF => {
                self.plan
                    .clause_offsets
                    .windows(2)
                    .fold(0u64, |sum, clause| {
                        sum.saturating_add(
                            self.plan.label_ids[clause[0]..clause[1]]
                                .iter()
                                .map(count)
                                .min()
                                .unwrap_or(0),
                        )
                    })
            }
            FilterExpressionType::CNF => self
                .plan
                .clause_offsets
                .windows(2)
                .map(|clause| {
                    self.plan.label_ids[clause[0]..clause[1]]
                        .iter()
                        .map(count)
                        .fold(0u64, u64::saturating_add)
                })
                .min()
                .unwrap_or(0),
        };
        Some(bound.min(u64::from(self.num_vectors)))
    }

    /// Return whether `vec_id` satisfies this compiled label query.
    pub fn is_match(&self, vec_id: u32) -> bool {
        if vec_id >= self.num_vectors {
            return false;
        }

        let terminal_matches = |label_id: Option<u32>| {
            let Some(label_id) = label_id.map(|id| id as usize) else {
                return false;
            };
            match self.payload.as_ref() {
                LabelIndexPayload::Bitslice {
                    words_per_row,
                    bits,
                } => {
                    let word = bits[label_id * *words_per_row + vec_id as usize / 64];
                    word & (1u64 << (vec_id % 64)) != 0
                }
                LabelIndexPayload::Bloom {
                    words_per_row,
                    hash_count,
                    bits,
                    label_rows,
                } => {
                    let row_offset = label_id * *hash_count;
                    label_rows[row_offset..row_offset + *hash_count]
                        .iter()
                        .all(|&row| {
                            let word = bits[row as usize * *words_per_row + vec_id as usize / 64];
                            word & (1u64 << (vec_id % 64)) != 0
                        })
                }
            }
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

    /// Visit matching vector IDs in ascending order, including Bloom false positives.
    pub fn visit_matches(&self, mut visit: impl FnMut(u32)) {
        let words_per_row = self.words_per_row();

        #[cfg(target_arch = "x86_64")]
        if matches!(self.payload.as_ref(), LabelIndexPayload::Bloom { .. })
            && words_per_row >= 4
            && std::is_x86_feature_detected!("avx2")
        {
            // SAFETY: The AVX2 path is called only after runtime feature detection.
            unsafe { self.visit_bloom_avx2(&mut visit) };
            return;
        }

        for word_index in 0..words_per_row {
            self.visit_word(word_index, self.word_matches(word_index), &mut visit);
        }
    }

    fn visit_word(&self, word_index: usize, mut matches: u64, visit: &mut impl FnMut(u32)) {
        while matches != 0 {
            let id = word_index * 64 + matches.trailing_zeros() as usize;
            if id < self.num_vectors as usize {
                visit(id as u32);
            }
            matches &= matches - 1;
        }
    }

    fn words_per_row(&self) -> usize {
        match self.payload.as_ref() {
            LabelIndexPayload::Bitslice { words_per_row, .. }
            | LabelIndexPayload::Bloom { words_per_row, .. } => *words_per_row,
        }
    }

    fn word_matches(&self, word_index: usize) -> u64 {
        let words_per_row = self.words_per_row();
        let word_for_label = |label_id: Option<u32>| {
            let Some(label_id) = label_id.map(|id| id as usize) else {
                return 0;
            };
            match self.payload.as_ref() {
                LabelIndexPayload::Bitslice { bits, .. } => {
                    bits[label_id * words_per_row + word_index]
                }
                LabelIndexPayload::Bloom {
                    hash_count,
                    bits,
                    label_rows,
                    ..
                } => {
                    let start = label_id * *hash_count;
                    label_rows[start..start + *hash_count]
                        .iter()
                        .fold(u64::MAX, |word, &row| {
                            word & bits[row as usize * words_per_row + word_index]
                        })
                }
            }
        };

        match self.plan.expression_type {
            FilterExpressionType::DNF => {
                self.plan.clause_offsets.windows(2).fold(0, |acc, clause| {
                    acc | self.plan.label_ids[clause[0]..clause[1]]
                        .iter()
                        .copied()
                        .fold(u64::MAX, |word, label_id| word & word_for_label(label_id))
                })
            }
            FilterExpressionType::CNF => {
                self.plan
                    .clause_offsets
                    .windows(2)
                    .fold(u64::MAX, |acc, clause| {
                        acc & self.plan.label_ids[clause[0]..clause[1]]
                            .iter()
                            .copied()
                            .fold(0, |word, label_id| word | word_for_label(label_id))
                    })
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn visit_bloom_avx2(&self, visit: &mut impl FnMut(u32)) {
        use std::arch::x86_64::{
            __m256i, _mm256_and_si256, _mm256_loadu_si256, _mm256_or_si256, _mm256_set1_epi64x,
            _mm256_setzero_si256, _mm256_storeu_si256, _mm256_testz_si256,
        };

        let LabelIndexPayload::Bloom {
            words_per_row,
            hash_count,
            bits,
            label_rows,
        } = self.payload.as_ref()
        else {
            unreachable!("AVX2 visitor is only called for Bloom indexes")
        };
        let simd_end = words_per_row / 4 * 4;
        for word_index in (0..simd_end).step_by(4) {
            let word_for_label = |label_id: Option<u32>| {
                let Some(label_id) = label_id.map(|id| id as usize) else {
                    return _mm256_setzero_si256();
                };
                let start = label_id * *hash_count;
                label_rows[start..start + *hash_count].iter().fold(
                    _mm256_set1_epi64x(-1),
                    |word, &row| {
                        // SAFETY: Every loaded row has `words_per_row` words and this block
                        // starts at most four words before the end; unaligned loads are valid.
                        let ptr = unsafe {
                            bits.as_ptr()
                                .add(row as usize * *words_per_row + word_index)
                                .cast::<__m256i>()
                        };
                        // SAFETY: The four words at `ptr` are in bounds; the intrinsic
                        // accepts the `u64` storage alignment.
                        _mm256_and_si256(word, unsafe { _mm256_loadu_si256(ptr) })
                    },
                )
            };
            let matches = match self.plan.expression_type {
                FilterExpressionType::DNF => self.plan.clause_offsets.windows(2).fold(
                    _mm256_setzero_si256(),
                    |outer, clause| {
                        let inner = self.plan.label_ids[clause[0]..clause[1]]
                            .iter()
                            .copied()
                            .fold(_mm256_set1_epi64x(-1), |word, id| {
                                _mm256_and_si256(word, word_for_label(id))
                            });
                        _mm256_or_si256(outer, inner)
                    },
                ),
                FilterExpressionType::CNF => self.plan.clause_offsets.windows(2).fold(
                    _mm256_set1_epi64x(-1),
                    |outer, clause| {
                        let inner = self.plan.label_ids[clause[0]..clause[1]]
                            .iter()
                            .copied()
                            .fold(_mm256_setzero_si256(), |word, id| {
                                _mm256_or_si256(word, word_for_label(id))
                            });
                        _mm256_and_si256(outer, inner)
                    },
                ),
            };
            if _mm256_testz_si256(matches, matches) != 0 {
                continue;
            }
            let mut lanes = [0u64; 4];
            // SAFETY: `lanes` has space for the four u64 words in one AVX2 register.
            unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), matches) };
            for (offset, word) in lanes.into_iter().enumerate() {
                self.visit_word(word_index + offset, word, visit);
            }
        }
        for word_index in simd_end..*words_per_row {
            self.visit_word(word_index, self.word_matches(word_index), visit);
        }
    }

    /// Count matching vectors; Bloom indexes count candidates, including false positives.
    pub fn count_matches(&self) -> u64 {
        (0..self.words_per_row())
            .map(|word_index| u64::from(self.word_matches(word_index).count_ones()))
            .sum()
    }
}

impl diskann::graph::ext::labeled::QueryLabelProvider<u32> for EncodedLabelQuery<'_> {
    fn is_match(&self, id: u32) -> bool {
        EncodedLabelQuery::is_match(self, id)
    }
}

impl diskann::graph::ext::labeled::CandidateLabelProvider<u32> for EncodedLabelQuery<'_> {
    fn match_upper_bound(&self) -> Option<u64> {
        EncodedLabelQuery::match_upper_bound(self)
    }

    fn visit_candidates(&self, visit: impl FnMut(u32)) {
        self.visit_matches(visit);
    }
}

impl EncodedLabelIndex {
    /// Load a dense Bitslice or transposed Bloom label index from `path`.
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
        if !matches!(version, LABEL_INDEX_VERSION | COUNTED_BLOOM_INDEX_VERSION) {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "unsupported label-index version {version}"
            )));
        }

        let format = read_u32(&mut reader)?;
        if !matches!(format, BITSLICE_FORMAT | BLOOM_FORMAT) {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "unsupported label-index format {format}; supported formats are dense bitslice 0 and Bloom 1"
            )));
        }
        if version == COUNTED_BLOOM_INDEX_VERSION && format != BLOOM_FORMAT {
            return Err(EncodedLabelIndexError::Invalid(
                "counted label-index version only supports Bloom format".to_string(),
            ));
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

        let expected_words = (num_vectors as usize).div_ceil(64);
        let payload = match format {
            BITSLICE_FORMAT => {
                let words_per_row = usize::try_from(read_u64(&mut reader)?).map_err(|_| {
                    EncodedLabelIndexError::Invalid("bitslice row length exceeds usize".to_string())
                })?;
                if words_per_row != expected_words {
                    return Err(EncodedLabelIndexError::Invalid(format!(
                        "bitslice row has {words_per_row} words; expected {expected_words}"
                    )));
                }
                let bits =
                    read_payload(&mut reader, file_len, num_labels, words_per_row, "bitslice")?;
                validate_bitslice_padding(&bits, num_labels, words_per_row, num_vectors)?;
                LabelIndexPayload::Bitslice {
                    words_per_row,
                    bits: bits.into_boxed_slice(),
                }
            }
            BLOOM_FORMAT => {
                let bit_count = read_u32(&mut reader)?;
                let hash_count = read_u32(&mut reader)?;
                validate_bloom_config(bit_count, hash_count)?;
                let config = BloomFilterConfig::new(bit_count, hash_count)?;
                let words_per_row = usize::try_from(read_u64(&mut reader)?).map_err(|_| {
                    EncodedLabelIndexError::Invalid("Bloom row length exceeds usize".to_string())
                })?;
                if words_per_row != expected_words {
                    return Err(EncodedLabelIndexError::Invalid(format!(
                        "Bloom row has {words_per_row} words; expected {expected_words}"
                    )));
                }
                let bits = read_payload(
                    &mut reader,
                    file_len,
                    bit_count as usize,
                    words_per_row,
                    "Bloom",
                )?;
                validate_bitslice_padding(&bits, bit_count as usize, words_per_row, num_vectors)?;

                let total_label_rows =
                    num_labels.checked_mul(hash_count as usize).ok_or_else(|| {
                        EncodedLabelIndexError::Invalid(
                            "Bloom label-row table size overflow".to_string(),
                        )
                    })?;
                let mut label_rows = Vec::new();
                label_rows
                    .try_reserve_exact(total_label_rows)
                    .map_err(|_| {
                        EncodedLabelIndexError::Invalid(
                            "cannot reserve Bloom label-row table".to_string(),
                        )
                    })?;
                for label in &labels {
                    append_label_rows(label, config, &mut label_rows);
                }

                LabelIndexPayload::Bloom {
                    words_per_row,
                    hash_count: hash_count as usize,
                    bits: bits.into_boxed_slice(),
                    label_rows: label_rows.into_boxed_slice(),
                }
            }
            _ => unreachable!("label-index format was validated above"),
        };

        let counts = if version == COUNTED_BLOOM_INDEX_VERSION {
            let count_bytes = num_labels.checked_mul(4).ok_or_else(|| {
                EncodedLabelIndexError::Invalid("Bloom label count table size overflow".to_string())
            })?;
            ensure_remaining(&mut reader, file_len, count_bytes, "Bloom label counts")?;
            let mut counts = Vec::new();
            counts.try_reserve_exact(num_labels).map_err(|_| {
                EncodedLabelIndexError::Invalid("cannot reserve Bloom label counts".to_string())
            })?;
            for _ in 0..num_labels {
                let count = read_u32(&mut reader)?;
                if count == 0 || count > num_vectors {
                    return Err(EncodedLabelIndexError::Invalid(format!(
                        "Bloom label count {count} is outside 1..={num_vectors}"
                    )));
                }
                counts.push(count);
            }
            Some(Arc::from(counts))
        } else {
            None
        };

        if reader.read(&mut [0u8; 1])? != 0 {
            return Err(EncodedLabelIndexError::Invalid(
                "label-index contains trailing bytes".to_string(),
            ));
        }

        Ok(Self {
            labels: labels.into_boxed_slice(),
            label_ids,
            counts,
            num_vectors,
            payload: Arc::new(payload),
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
            payload: Arc::clone(&self.payload),
            plan: compile_plan(clauses, expression_type, &self.label_ids)?,
            counts: self.counts.clone(),
            _lifetime: PhantomData,
        })
    }
}

fn read_payload(
    reader: &mut BufReader<File>,
    file_len: u64,
    row_count: usize,
    words_per_row: usize,
    format_name: &str,
) -> Result<Vec<u64>, EncodedLabelIndexError> {
    let total_words = row_count.checked_mul(words_per_row).ok_or_else(|| {
        EncodedLabelIndexError::Invalid(format!("{format_name} allocation size overflow"))
    })?;
    let payload_bytes = total_words.checked_mul(8).ok_or_else(|| {
        EncodedLabelIndexError::Invalid(format!("{format_name} byte size overflow"))
    })?;
    ensure_remaining(
        reader,
        file_len,
        payload_bytes,
        &format!("{format_name} payload"),
    )?;

    let mut bits = Vec::new();
    bits.try_reserve_exact(total_words).map_err(|_| {
        EncodedLabelIndexError::Invalid(format!("cannot reserve {format_name} payload"))
    })?;
    for _ in 0..total_words {
        bits.push(read_u64(reader)?);
    }
    Ok(bits)
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
