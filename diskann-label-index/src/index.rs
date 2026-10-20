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
        MAX_LABEL_COUNT, MAX_LABEL_LENGTH, MIXED_BLOOM_INDEX_VERSION,
        PREFIX_EXACT_BLOOM_INDEX_VERSION, SUMMARIZED_BLOOM_INDEX_VERSION,
    },
};
use std::{
    cell::RefCell,
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
        exact_row_by_label: Box<[u32]>,
        exact_label_count: usize,
        nonzero_words: Option<BloomWordSummary>,
    },
}

struct BloomWordSummary {
    words_per_row: usize,
    bits: Box<[u64]>,
    scan_useful: bool,
}

/// Reusable allocation for summarized Bloom candidate enumeration.
#[derive(Debug)]
pub struct BloomVisitScratch {
    surviving_words: Vec<SummarizedWord>,
}

#[derive(Debug)]
struct SummarizedWord {
    summary_index: usize,
    possible_words: u64,
}

/// Query-time Bloom candidate enumeration configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BloomQueryConfig {
    summary_cache_bound_percent: f64,
    summary_scan_max_word_percent: f64,
}

impl BloomQueryConfig {
    /// Create a query configuration with a 1% to 100% summary-cache bound.
    ///
    /// A value of 1 caches masks when the conservative match bound is at most
    /// 1% of the searchable vectors. The summary-scan word limit defaults to 10%.
    pub fn new(summary_cache_bound_percent: f64) -> Result<Self, EncodedLabelIndexError> {
        validate_percent(
            "Bloom summary cache bound percent",
            summary_cache_bound_percent,
        )?;
        Ok(Self {
            summary_cache_bound_percent,
            summary_scan_max_word_percent: 10.0,
        })
    }

    /// Set the maximum surviving payload-word percentage that uses word summaries.
    ///
    /// A value of 100 always uses available summaries instead of falling back
    /// to a full payload scan.
    pub fn with_summary_scan_max_word_percent(
        mut self,
        summary_scan_max_word_percent: f64,
    ) -> Result<Self, EncodedLabelIndexError> {
        validate_percent(
            "Bloom summary scan maximum word percent",
            summary_scan_max_word_percent,
        )?;
        self.summary_scan_max_word_percent = summary_scan_max_word_percent;
        Ok(self)
    }

    /// Return the maximum conservative match-bound percentage eligible for mask caching.
    pub fn summary_cache_bound_percent(self) -> f64 {
        self.summary_cache_bound_percent
    }

    /// Return the maximum surviving payload-word percentage that uses word summaries.
    pub fn summary_scan_max_word_percent(self) -> f64 {
        self.summary_scan_max_word_percent
    }

    pub(crate) fn should_scan_summarized_words(
        self,
        possible_word_count: usize,
        words_per_row: usize,
    ) -> bool {
        (possible_word_count as f64) * 100.0
            <= (words_per_row as f64) * self.summary_scan_max_word_percent
    }
}

impl Default for BloomQueryConfig {
    fn default() -> Self {
        Self {
            summary_cache_bound_percent: 1.0,
            summary_scan_max_word_percent: 10.0,
        }
    }
}

fn validate_percent(name: &str, value: f64) -> Result<(), EncodedLabelIndexError> {
    if !value.is_finite() || !(1.0..=100.0).contains(&value) {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "{name} must be in 1..=100, got {value}"
        )));
    }
    Ok(())
}

impl Default for BloomVisitScratch {
    fn default() -> Self {
        Self {
            surviving_words: Vec::with_capacity(4096),
        }
    }
}

impl BloomVisitScratch {
    fn reset(&mut self, summary_words: usize) {
        self.surviving_words.clear();
        if self.surviving_words.capacity() > 65_536 {
            self.surviving_words.shrink_to(65_536);
        }
        let initial_capacity = summary_words.min(4096);
        if self.surviving_words.capacity() < initial_capacity {
            self.surviving_words.reserve(initial_capacity);
        }
    }
}

thread_local! {
    static BLOOM_VISIT_SCRATCH: RefCell<BloomVisitScratch> =
        RefCell::new(BloomVisitScratch::default());
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
            .field("num_exact_labels", &self.num_exact_labels())
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

#[derive(Debug, Clone, Copy)]
enum PointPlan {
    Generic,
    ExactAnd2 {
        left_word_offset: usize,
        right_word_offset: usize,
    },
}

/// Query-scoped evaluator compiled from an [`EncodedLabelIndex`].
///
/// Queries share the immutable index payload, so they remain usable after the source index is
/// dropped.
pub struct EncodedLabelQuery<'a> {
    num_vectors: u32,
    payload: Arc<LabelIndexPayload>,
    plan: CompiledPlan,
    point_plan: PointPlan,
    counts: Option<Arc<[u32]>>,
    config: BloomQueryConfig,
    _lifetime: PhantomData<&'a ()>,
}

impl std::fmt::Debug for EncodedLabelQuery<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodedLabelQuery")
            .field("num_vectors", &self.num_vectors)
            .field("expression_type", &self.plan.expression_type)
            .field(
                "summary_cache_bound_percent",
                &self.config.summary_cache_bound_percent(),
            )
            .field(
                "summary_scan_max_word_percent",
                &self.config.summary_scan_max_word_percent(),
            )
            .finish()
    }
}

impl EncodedLabelQuery<'_> {
    #[cfg(test)]
    pub(crate) fn uses_exact_and2_point_plan(&self) -> bool {
        matches!(self.point_plan, PointPlan::ExactAnd2 { .. })
    }

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
        if let PointPlan::ExactAnd2 {
            left_word_offset,
            right_word_offset,
        } = self.point_plan
        {
            let LabelIndexPayload::Bloom { bits, .. } = self.payload.as_ref() else {
                unreachable!("exact-row point plans require a Bloom payload")
            };
            let word_index = vec_id as usize / 64;
            let mask = 1u64 << (vec_id % 64);
            return bits[left_word_offset + word_index]
                & bits[right_word_offset + word_index]
                & mask
                != 0;
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
                    exact_row_by_label,
                    ..
                } => {
                    let exact_row = exact_row_by_label[label_id];
                    if exact_row != u32::MAX {
                        let word = bits[exact_row as usize * *words_per_row + vec_id as usize / 64];
                        return word & (1u64 << (vec_id % 64)) != 0;
                    }
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
        if !self.should_cache_summary_matches() {
            self.visit_matches_without_scratch(&mut visit);
            return;
        }
        BLOOM_VISIT_SCRATCH.with(|cell| {
            if let Ok(mut scratch) = cell.try_borrow_mut() {
                self.visit_matches_with_scratch(&mut scratch, &mut visit);
            } else {
                let mut scratch = BloomVisitScratch::default();
                self.visit_matches_with_scratch(&mut scratch, &mut visit);
            }
        });
    }

    /// Visit matching vector IDs using caller-owned reusable scratch memory.
    pub fn visit_matches_with_scratch(
        &self,
        scratch: &mut BloomVisitScratch,
        mut visit: impl FnMut(u32),
    ) {
        if self.should_cache_summary_matches()
            && self.visit_cached_summarized_matches(scratch, &mut visit)
        {
            return;
        }
        self.visit_matches_without_scratch(&mut visit);
    }

    fn visit_matches_without_scratch(&self, mut visit: impl FnMut(u32)) {
        if self.visit_summarized_matches_uncached(&mut visit) {
            return;
        }
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

    pub(crate) fn should_cache_summary_matches(&self) -> bool {
        self.match_upper_bound().is_some_and(|bound| {
            (bound as f64) * 100.0
                <= (self.num_vectors as f64) * self.config.summary_cache_bound_percent()
        })
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

    fn summary_word_matches(&self, word_index: usize) -> Option<u64> {
        let LabelIndexPayload::Bloom {
            hash_count,
            label_rows,
            nonzero_words: Some(summary),
            exact_row_by_label,
            ..
        } = self.payload.as_ref()
        else {
            return None;
        };
        let word_for_label = |label_id: Option<u32>| {
            let Some(label_id) = label_id.map(|id| id as usize) else {
                return 0;
            };
            let exact_row = exact_row_by_label[label_id];
            if exact_row != u32::MAX {
                return summary.bits[exact_row as usize * summary.words_per_row + word_index];
            }
            let start = label_id * *hash_count;
            label_rows[start..start + *hash_count]
                .iter()
                .fold(u64::MAX, |word, &row| {
                    word & summary.bits[row as usize * summary.words_per_row + word_index]
                })
        };

        Some(match self.plan.expression_type {
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
        })
    }

    fn collect_cached_summarized_matches(&self, scratch: &mut BloomVisitScratch) -> bool {
        let LabelIndexPayload::Bloom {
            words_per_row,
            nonzero_words:
                Some(
                    summary @ BloomWordSummary {
                        scan_useful: true, ..
                    },
                ),
            ..
        } = self.payload.as_ref()
        else {
            return false;
        };

        scratch.reset(summary.words_per_row);
        let mut possible_word_count = 0usize;
        for summary_index in 0..summary.words_per_row {
            let possible_words = self
                .summary_word_matches(summary_index)
                .expect("Bloom summary disappeared while collecting");
            possible_word_count =
                possible_word_count.saturating_add(possible_words.count_ones() as usize);
            if possible_words != 0 {
                scratch.surviving_words.push(SummarizedWord {
                    summary_index,
                    possible_words,
                });
            }
            if !self
                .config
                .should_scan_summarized_words(possible_word_count, *words_per_row)
            {
                scratch.surviving_words.clear();
                return false;
            }
        }
        true
    }

    fn visit_cached_summarized_matches(
        &self,
        scratch: &mut BloomVisitScratch,
        visit: &mut impl FnMut(u32),
    ) -> bool {
        if !self.collect_cached_summarized_matches(scratch) {
            return false;
        }
        for summarized in &scratch.surviving_words {
            self.visit_summarized_word(summarized.summary_index, summarized.possible_words, visit);
        }
        true
    }

    fn visit_summarized_matches_uncached(&self, visit: &mut impl FnMut(u32)) -> bool {
        let LabelIndexPayload::Bloom {
            words_per_row,
            nonzero_words:
                Some(
                    summary @ BloomWordSummary {
                        scan_useful: true, ..
                    },
                ),
            ..
        } = self.payload.as_ref()
        else {
            return false;
        };

        let possible_word_count: usize = (0..summary.words_per_row)
            .map(|summary_index| {
                self.summary_word_matches(summary_index)
                    .expect("Bloom summary disappeared while sizing")
                    .count_ones() as usize
            })
            .sum();
        if !self
            .config
            .should_scan_summarized_words(possible_word_count, *words_per_row)
        {
            return false;
        }
        for summary_index in 0..summary.words_per_row {
            let possible_words = self
                .summary_word_matches(summary_index)
                .expect("Bloom summary disappeared while visiting");
            self.visit_summarized_word(summary_index, possible_words, visit);
        }
        true
    }

    fn visit_summarized_word(
        &self,
        summary_index: usize,
        mut possible_words: u64,
        visit: &mut impl FnMut(u32),
    ) {
        while possible_words != 0 {
            let bit = possible_words.trailing_zeros() as usize;
            let word_index = summary_index * 64 + bit;
            if word_index < self.words_per_row() {
                self.visit_word(word_index, self.word_matches(word_index), visit);
            }
            possible_words &= possible_words - 1;
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
                    exact_row_by_label,
                    ..
                } => {
                    let exact_row = exact_row_by_label[label_id];
                    if exact_row != u32::MAX {
                        return bits[exact_row as usize * words_per_row + word_index];
                    }
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
            exact_row_by_label,
            ..
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
                let exact_row = exact_row_by_label[label_id];
                if exact_row != u32::MAX {
                    // SAFETY: Exact rows use the same row length and alignment contract as
                    // Bloom rows, and this block begins at most four words before the end.
                    let ptr = unsafe {
                        bits.as_ptr()
                            .add(exact_row as usize * *words_per_row + word_index)
                            .cast::<__m256i>()
                    };
                    // SAFETY: The four exact-row words at `ptr` are in bounds.
                    return unsafe { _mm256_loadu_si256(ptr) };
                }
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
        if !self.should_cache_summary_matches() {
            return self.count_matches_without_scratch();
        }
        BLOOM_VISIT_SCRATCH.with(|cell| {
            if let Ok(mut scratch) = cell.try_borrow_mut() {
                self.count_matches_with_scratch(&mut scratch)
            } else {
                let mut scratch = BloomVisitScratch::default();
                self.count_matches_with_scratch(&mut scratch)
            }
        })
    }

    /// Count matching vectors using caller-owned reusable scratch memory.
    pub fn count_matches_with_scratch(&self, scratch: &mut BloomVisitScratch) -> u64 {
        if self.should_cache_summary_matches() && self.collect_cached_summarized_matches(scratch) {
            return scratch
                .surviving_words
                .iter()
                .map(|summarized| {
                    self.count_summarized_word(summarized.summary_index, summarized.possible_words)
                })
                .sum();
        }
        self.count_matches_without_scratch()
    }

    fn count_matches_without_scratch(&self) -> u64 {
        if let LabelIndexPayload::Bloom {
            words_per_row,
            nonzero_words:
                Some(
                    summary @ BloomWordSummary {
                        scan_useful: true, ..
                    },
                ),
            ..
        } = self.payload.as_ref()
        {
            let possible_word_count: usize = (0..summary.words_per_row)
                .map(|summary_index| {
                    self.summary_word_matches(summary_index)
                        .expect("Bloom summary disappeared while sizing count")
                        .count_ones() as usize
                })
                .sum();
            if self
                .config
                .should_scan_summarized_words(possible_word_count, *words_per_row)
            {
                return (0..summary.words_per_row)
                    .map(|summary_index| {
                        let possible_words = self
                            .summary_word_matches(summary_index)
                            .expect("Bloom summary disappeared while counting");
                        self.count_summarized_word(summary_index, possible_words)
                    })
                    .sum();
            }
        }
        (0..self.words_per_row())
            .map(|word_index| u64::from(self.word_matches(word_index).count_ones()))
            .sum()
    }

    fn count_summarized_word(&self, summary_index: usize, mut possible_words: u64) -> u64 {
        let mut count = 0u64;
        while possible_words != 0 {
            let bit = possible_words.trailing_zeros() as usize;
            let word_index = summary_index * 64 + bit;
            if word_index < self.words_per_row() {
                count += u64::from(self.word_matches(word_index).count_ones());
            }
            possible_words &= possible_words - 1;
        }
        count
    }
}

impl diskann::graph::ext::labeled::QueryLabelProvider<u32> for EncodedLabelQuery<'_> {
    fn is_match(&self, id: u32) -> bool {
        EncodedLabelQuery::is_match(self, id)
    }

    fn cache_rejections(&self) -> bool {
        let LabelIndexPayload::Bloom {
            exact_row_by_label, ..
        } = self.payload.as_ref()
        else {
            return false;
        };
        self.plan
            .label_ids
            .iter()
            .flatten()
            .any(|&id| exact_row_by_label[id as usize] == u32::MAX)
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
    ///
    /// Bloom versions without word summaries remain supported and use the full payload scan.
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
        if !matches!(
            version,
            LABEL_INDEX_VERSION
                | COUNTED_BLOOM_INDEX_VERSION
                | SUMMARIZED_BLOOM_INDEX_VERSION
                | MIXED_BLOOM_INDEX_VERSION
                | PREFIX_EXACT_BLOOM_INDEX_VERSION
        ) {
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
        if matches!(
            version,
            COUNTED_BLOOM_INDEX_VERSION
                | SUMMARIZED_BLOOM_INDEX_VERSION
                | MIXED_BLOOM_INDEX_VERSION
                | PREFIX_EXACT_BLOOM_INDEX_VERSION
        ) && format != BLOOM_FORMAT
        {
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
        let mut payload = match format {
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
                let hash_count = hash_count as usize;
                let prefix_exact = version == PREFIX_EXACT_BLOOM_INDEX_VERSION;
                let exact_label_count = if prefix_exact {
                    let count = usize::try_from(read_u32(&mut reader)?).map_err(|_| {
                        EncodedLabelIndexError::Invalid(
                            "exact Bloom label count exceeds usize".to_string(),
                        )
                    })?;
                    if count > num_labels {
                        return Err(EncodedLabelIndexError::Invalid(format!(
                            "exact Bloom label count {count} exceeds label count {num_labels}"
                        )));
                    }
                    let flags = read_u32(&mut reader)?;
                    if flags != 0 {
                        return Err(EncodedLabelIndexError::Invalid(format!(
                            "unsupported prefix-exact Bloom flags {flags}"
                        )));
                    }
                    count
                } else {
                    0
                };
                let words_per_row = usize::try_from(read_u64(&mut reader)?).map_err(|_| {
                    EncodedLabelIndexError::Invalid("Bloom row length exceeds usize".to_string())
                })?;
                if words_per_row != expected_words {
                    return Err(EncodedLabelIndexError::Invalid(format!(
                        "Bloom row has {words_per_row} words; expected {expected_words}"
                    )));
                }
                let total_label_rows = num_labels.checked_mul(hash_count).ok_or_else(|| {
                    EncodedLabelIndexError::Invalid(
                        "Bloom label-row table size overflow".to_string(),
                    )
                })?;
                let mut label_rows = vec![u32::MAX; total_label_rows];
                let mut exact_row_by_label = vec![u32::MAX; num_labels];
                if prefix_exact {
                    for (label_id, exact_row) in exact_row_by_label
                        .iter_mut()
                        .take(exact_label_count)
                        .enumerate()
                    {
                        *exact_row = u32::try_from(label_id).map_err(|_| {
                            EncodedLabelIndexError::Invalid(
                                "exact Bloom row count exceeds u32".to_string(),
                            )
                        })?;
                    }
                    let mapping_count = (num_labels - exact_label_count)
                        .checked_mul(hash_count)
                        .ok_or_else(|| {
                            EncodedLabelIndexError::Invalid(
                                "Bloom label-row mapping size overflow".to_string(),
                            )
                        })?;
                    let mapping_bytes = mapping_count.checked_mul(4).ok_or_else(|| {
                        EncodedLabelIndexError::Invalid(
                            "Bloom label-row mapping byte size overflow".to_string(),
                        )
                    })?;
                    ensure_remaining(
                        &mut reader,
                        file_len,
                        mapping_bytes,
                        "Bloom label-row mappings",
                    )?;
                    for label_id in exact_label_count..num_labels {
                        let start = label_id * hash_count;
                        for hash_index in 0..hash_count {
                            let logical_row = read_u32(&mut reader)?;
                            let physical_row = u32::try_from(exact_label_count)
                                .ok()
                                .and_then(|prefix| prefix.checked_add(logical_row))
                                .ok_or_else(|| {
                                    EncodedLabelIndexError::Invalid(
                                        "Bloom physical row index overflow".to_string(),
                                    )
                                })?;
                            if logical_row >= bit_count
                                || label_rows[start..start + hash_index].contains(&physical_row)
                            {
                                return Err(EncodedLabelIndexError::Invalid(
                                    "Bloom label rows must be distinct and within the Bloom row count"
                                        .to_string(),
                                ));
                            }
                            label_rows[start + hash_index] = physical_row;
                        }
                    }
                } else {
                    let config = BloomFilterConfig::new(bit_count, hash_count as u32)?;
                    let mut generated_rows = Vec::new();
                    generated_rows
                        .try_reserve_exact(total_label_rows)
                        .map_err(|_| {
                            EncodedLabelIndexError::Invalid(
                                "cannot reserve Bloom label-row table".to_string(),
                            )
                        })?;
                    for label in &labels {
                        append_label_rows(label, config, &mut generated_rows);
                    }
                    label_rows.copy_from_slice(&generated_rows);
                }

                let physical_row_count = exact_label_count
                    .checked_add(bit_count as usize)
                    .ok_or_else(|| {
                        EncodedLabelIndexError::Invalid("Bloom row count overflow".to_string())
                    })?;
                let bits = read_payload(
                    &mut reader,
                    file_len,
                    physical_row_count,
                    words_per_row,
                    "Bloom",
                )?;
                validate_bitslice_padding(&bits, physical_row_count, words_per_row, num_vectors)?;

                let nonzero_words = if matches!(
                    version,
                    SUMMARIZED_BLOOM_INDEX_VERSION
                        | MIXED_BLOOM_INDEX_VERSION
                        | PREFIX_EXACT_BLOOM_INDEX_VERSION
                ) {
                    let summary_words_per_row =
                        usize::try_from(read_u64(&mut reader)?).map_err(|_| {
                            EncodedLabelIndexError::Invalid(
                                "Bloom word-summary row length exceeds usize".to_string(),
                            )
                        })?;
                    let expected_summary_words = words_per_row.div_ceil(64);
                    if summary_words_per_row != expected_summary_words {
                        return Err(EncodedLabelIndexError::Invalid(format!(
                            "Bloom word-summary row has {summary_words_per_row} words; expected {expected_summary_words}"
                        )));
                    }
                    let bits = read_payload(
                        &mut reader,
                        file_len,
                        physical_row_count,
                        summary_words_per_row,
                        "Bloom word-summary",
                    )?;
                    let summarized_words = u32::try_from(words_per_row).map_err(|_| {
                        EncodedLabelIndexError::Invalid(
                            "Bloom row word count exceeds u32".to_string(),
                        )
                    })?;
                    validate_bitslice_padding(
                        &bits,
                        physical_row_count,
                        summary_words_per_row,
                        summarized_words,
                    )?;
                    let nonzero_payload_words: usize =
                        bits.iter().map(|word| word.count_ones() as usize).sum();
                    let payload_words =
                        physical_row_count
                            .checked_mul(words_per_row)
                            .ok_or_else(|| {
                                EncodedLabelIndexError::Invalid(
                                    "Bloom payload word count overflow".to_string(),
                                )
                            })?;
                    Some(BloomWordSummary {
                        words_per_row: summary_words_per_row,
                        bits: bits.into_boxed_slice(),
                        scan_useful: prefix_exact
                            || nonzero_payload_words.saturating_mul(4) <= payload_words,
                    })
                } else {
                    None
                };

                LabelIndexPayload::Bloom {
                    words_per_row,
                    hash_count,
                    bits: bits.into_boxed_slice(),
                    label_rows: label_rows.into_boxed_slice(),
                    exact_row_by_label: exact_row_by_label.into_boxed_slice(),
                    exact_label_count,
                    nonzero_words,
                }
            }
            _ => unreachable!("label-index format was validated above"),
        };

        let counts = if matches!(
            version,
            COUNTED_BLOOM_INDEX_VERSION
                | SUMMARIZED_BLOOM_INDEX_VERSION
                | MIXED_BLOOM_INDEX_VERSION
                | PREFIX_EXACT_BLOOM_INDEX_VERSION
        ) {
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

        if version == MIXED_BLOOM_INDEX_VERSION {
            let LabelIndexPayload::Bloom {
                words_per_row,
                bits: combined_bits,
                exact_row_by_label,
                exact_label_count,
                nonzero_words,
                ..
            } = &mut payload
            else {
                unreachable!("mixed label-index version requires Bloom format")
            };
            let exact_count = usize::try_from(read_u32(&mut reader)?).map_err(|_| {
                EncodedLabelIndexError::Invalid("exact Bloom row count exceeds usize".to_string())
            })?;
            if exact_count == 0 || exact_count > num_labels {
                return Err(EncodedLabelIndexError::Invalid(format!(
                    "exact Bloom row count {exact_count} is outside 1..={num_labels}"
                )));
            }
            let exact_id_bytes = exact_count.checked_mul(4).ok_or_else(|| {
                EncodedLabelIndexError::Invalid("exact Bloom label ID size overflow".to_string())
            })?;
            ensure_remaining(
                &mut reader,
                file_len,
                exact_id_bytes,
                "exact Bloom label IDs",
            )?;
            let first_exact_row = combined_bits.len() / *words_per_row;
            for row in 0..exact_count {
                let label_id = usize::try_from(read_u32(&mut reader)?).map_err(|_| {
                    EncodedLabelIndexError::Invalid(
                        "exact Bloom label ID exceeds usize".to_string(),
                    )
                })?;
                if label_id >= num_labels || exact_row_by_label[label_id] != u32::MAX {
                    return Err(EncodedLabelIndexError::Invalid(
                        "exact Bloom label IDs must be unique dictionary entries".to_string(),
                    ));
                }
                let physical_row = first_exact_row.checked_add(row).ok_or_else(|| {
                    EncodedLabelIndexError::Invalid("exact Bloom physical row overflow".to_string())
                })?;
                exact_row_by_label[label_id] = u32::try_from(physical_row).map_err(|_| {
                    EncodedLabelIndexError::Invalid(
                        "exact Bloom physical row exceeds u32".to_string(),
                    )
                })?;
            }

            let exact_bits = read_payload(
                &mut reader,
                file_len,
                exact_count,
                *words_per_row,
                "exact Bloom",
            )?;
            validate_bitslice_padding(&exact_bits, exact_count, *words_per_row, num_vectors)?;
            let mut bits = std::mem::take(combined_bits).into_vec();
            bits.extend_from_slice(&exact_bits);
            *combined_bits = bits.into_boxed_slice();

            let summary_words_per_row = usize::try_from(read_u64(&mut reader)?).map_err(|_| {
                EncodedLabelIndexError::Invalid(
                    "exact Bloom word-summary row length exceeds usize".to_string(),
                )
            })?;
            let expected_summary_words = words_per_row.div_ceil(64);
            if summary_words_per_row != expected_summary_words {
                return Err(EncodedLabelIndexError::Invalid(format!(
                    "exact Bloom word-summary row has {summary_words_per_row} words; expected {expected_summary_words}"
                )));
            }
            let summary_bits = read_payload(
                &mut reader,
                file_len,
                exact_count,
                summary_words_per_row,
                "exact Bloom word-summary",
            )?;
            let summarized_words = u32::try_from(*words_per_row).map_err(|_| {
                EncodedLabelIndexError::Invalid("Bloom row word count exceeds u32".to_string())
            })?;
            validate_bitslice_padding(
                &summary_bits,
                exact_count,
                summary_words_per_row,
                summarized_words,
            )?;
            let summary = nonzero_words
                .as_mut()
                .expect("mixed Bloom indexes always contain Bloom summaries");
            let mut combined_summary_bits = std::mem::take(&mut summary.bits).into_vec();
            combined_summary_bits.extend_from_slice(&summary_bits);
            summary.bits = combined_summary_bits.into_boxed_slice();
            *exact_label_count = exact_count;
        }

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

    #[cfg(test)]
    pub(crate) fn has_bloom_word_summary(&self) -> bool {
        matches!(
            self.payload.as_ref(),
            LabelIndexPayload::Bloom {
                nonzero_words: Some(_),
                ..
            }
        )
    }

    /// Return the number of encoded labels.
    pub fn num_labels(&self) -> usize {
        self.labels.len()
    }

    /// Return how many Bloom dictionary labels also have exact rows.
    pub fn num_exact_labels(&self) -> usize {
        match self.payload.as_ref() {
            LabelIndexPayload::Bloom {
                exact_label_count, ..
            } => *exact_label_count,
            _ => 0,
        }
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
        self.query_with_config(clauses, expression_type, BloomQueryConfig::default())
    }

    /// Compile a query with explicit Bloom candidate-enumeration configuration.
    pub fn query_with_config<S>(
        &self,
        clauses: &[S],
        expression_type: FilterExpressionType,
        config: BloomQueryConfig,
    ) -> Result<EncodedLabelQuery<'static>, EncodedLabelIndexError>
    where
        S: AsRef<str>,
    {
        let plan = compile_plan(clauses, expression_type, &self.label_ids)?;
        let point_plan = compile_point_plan(&plan, &self.payload);
        Ok(EncodedLabelQuery {
            num_vectors: self.num_vectors,
            payload: Arc::clone(&self.payload),
            plan,
            point_plan,
            counts: self.counts.clone(),
            config,
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

fn compile_point_plan(plan: &CompiledPlan, payload: &LabelIndexPayload) -> PointPlan {
    let LabelIndexPayload::Bloom {
        words_per_row,
        exact_row_by_label,
        ..
    } = payload
    else {
        return PointPlan::Generic;
    };
    if plan.expression_type != FilterExpressionType::DNF
        || plan.clause_offsets.as_ref() != [0, 2]
        || plan.label_ids.len() != 2
    {
        return PointPlan::Generic;
    }
    let Some(left_label) = plan.label_ids[0].map(|id| id as usize) else {
        return PointPlan::Generic;
    };
    let Some(right_label) = plan.label_ids[1].map(|id| id as usize) else {
        return PointPlan::Generic;
    };
    let left_row = exact_row_by_label[left_label];
    let right_row = exact_row_by_label[right_label];
    if left_row == u32::MAX || right_row == u32::MAX {
        return PointPlan::Generic;
    }
    let Some(left_word_offset) = (left_row as usize).checked_mul(*words_per_row) else {
        return PointPlan::Generic;
    };
    let Some(right_word_offset) = (right_row as usize).checked_mul(*words_per_row) else {
        return PointPlan::Generic;
    };
    PointPlan::ExactAnd2 {
        left_word_offset,
        right_word_offset,
    }
}
