/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Label-index loading and query serving.

use crate::{
    bloom::validate_bloom_config,
    error::EncodedLabelIndexError,
    format::{
        ensure_remaining, read_u32, read_u64, validate_label, validate_row_padding,
        LABEL_INDEX_FLAGS, LABEL_INDEX_MAGIC, LABEL_INDEX_VERSION, MAX_LABEL_COUNT,
        MAX_LABEL_LENGTH,
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

struct LabelIndexPayload {
    words_per_row: usize,
    hash_count: usize,
    bits: Box<[u64]>,
    label_rows: Box<[u32]>,
    exact_row_by_label: Box<[u32]>,
    exact_label_count: usize,
    row_valid_counts: Box<[u32]>,
    nonzero_words: BloomWordSummary,
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
    point_prefetch_lookahead: usize,
}

const DEFAULT_POINT_PREFETCH_LOOKAHEAD: usize = 32;

impl BloomQueryConfig {
    /// Create a query configuration with a 1% to 100% summary-cache bound.
    ///
    /// A value of 1 caches masks when the conservative match bound is at most
    /// 1% of the searchable vectors. The summary-scan word limit defaults to 10%, and exact
    /// payload point classification prefetches 32 IDs ahead on supported architectures.
    pub fn new(summary_cache_bound_percent: f64) -> Result<Self, EncodedLabelIndexError> {
        validate_percent(
            "Bloom summary cache bound percent",
            summary_cache_bound_percent,
        )?;
        Ok(Self {
            summary_cache_bound_percent,
            summary_scan_max_word_percent: 10.0,
            point_prefetch_lookahead: DEFAULT_POINT_PREFETCH_LOOKAHEAD,
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

    /// Set the payload point-classification prefetch lookahead.
    ///
    /// Zero disables prefetching. Nonzero values are used only on supported architectures and only
    /// by exact payload evaluators.
    pub fn with_point_prefetch_lookahead(mut self, lookahead: usize) -> Self {
        self.point_prefetch_lookahead = lookahead;
        self
    }

    /// Return the maximum conservative match-bound percentage eligible for mask caching.
    pub fn summary_cache_bound_percent(self) -> f64 {
        self.summary_cache_bound_percent
    }

    /// Return the maximum surviving payload-word percentage that uses word summaries.
    pub fn summary_scan_max_word_percent(self) -> f64 {
        self.summary_scan_max_word_percent
    }

    /// Return the payload point-classification prefetch lookahead.
    pub fn point_prefetch_lookahead(self) -> usize {
        self.point_prefetch_lookahead
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
            point_prefetch_lookahead: DEFAULT_POINT_PREFETCH_LOOKAHEAD,
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
    counts: Arc<[u32]>,
    summary_nonzero_words_per_row: Arc<[u32]>,
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

#[derive(Debug)]
enum PointPlan {
    Dnf(Box<DnfPointPlan>),
    Cnf(Box<CnfPointPlan>),
}

#[derive(Debug)]
struct DnfPointPlan {
    clause_offsets: Box<[usize]>,
    payload_offsets: Box<[usize]>,
    summary_offsets: Box<[usize]>,
    use_summary: bool,
}

#[derive(Debug)]
struct CnfPointPlan {
    clause_offsets: Box<[usize]>,
    terminal_offsets: Box<[usize]>,
    payload_offsets: Box<[usize]>,
    summary_offsets: Box<[usize]>,
    use_summary: bool,
}

/// A point evaluator selected once from a compiled [`EncodedLabelQuery`].
///
/// Match this enum outside a repeated classification loop so each variant's concrete evaluator
/// can be statically dispatched without checking the query mode for every vector ID.
#[derive(Debug, Clone, Copy)]
pub enum EncodedLabelPointEvaluator<'a> {
    /// A DNF evaluator that reads payload rows directly.
    DnfPayload(DnfPayloadEvaluator<'a>),
    /// A DNF evaluator that checks summary rows before payload rows.
    DnfSummary(DnfSummaryEvaluator<'a>),
    /// A CNF evaluator that reads payload rows directly.
    CnfPayload(CnfPayloadEvaluator<'a>),
    /// A CNF evaluator that checks summary rows before payload rows.
    CnfSummary(CnfSummaryEvaluator<'a>),
}

/// A statically dispatched DNF evaluator for payload rows.
#[derive(Debug, Clone, Copy)]
pub struct DnfPayloadEvaluator<'a> {
    num_vectors: u32,
    bits: &'a [u64],
    clause_offsets: &'a [usize],
    payload_offsets: &'a [usize],
    prefetch_word_offset: Option<usize>,
    prefetch_lookahead: usize,
    cache_rejections: bool,
}

/// A statically dispatched DNF evaluator for summary-guarded payload rows.
#[derive(Debug, Clone, Copy)]
pub struct DnfSummaryEvaluator<'a> {
    num_vectors: u32,
    bits: &'a [u64],
    summary_bits: &'a [u64],
    clause_offsets: &'a [usize],
    payload_offsets: &'a [usize],
    summary_offsets: &'a [usize],
    cache_rejections: bool,
}

/// A statically dispatched CNF evaluator for payload rows.
#[derive(Debug, Clone, Copy)]
pub struct CnfPayloadEvaluator<'a> {
    num_vectors: u32,
    bits: &'a [u64],
    clause_offsets: &'a [usize],
    terminal_offsets: &'a [usize],
    payload_offsets: &'a [usize],
    prefetch_word_offset: Option<usize>,
    prefetch_lookahead: usize,
    cache_rejections: bool,
}

/// A statically dispatched CNF evaluator for summary-guarded payload rows.
#[derive(Debug, Clone, Copy)]
pub struct CnfSummaryEvaluator<'a> {
    num_vectors: u32,
    bits: &'a [u64],
    summary_bits: &'a [u64],
    clause_offsets: &'a [usize],
    terminal_offsets: &'a [usize],
    payload_offsets: &'a [usize],
    summary_offsets: &'a [usize],
    cache_rejections: bool,
}

const POINT_SUMMARY_MIN_ZERO_PERCENT: usize = 85;

/// Query-scoped evaluator compiled from an [`EncodedLabelIndex`].
///
/// Queries share the immutable index payload, so they remain usable after the source index is
/// dropped.
pub struct EncodedLabelQuery<'a> {
    num_vectors: u32,
    payload: Arc<LabelIndexPayload>,
    plan: CompiledPlan,
    point_plan: PointPlan,
    counts: Arc<[u32]>,
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
            .field(
                "point_prefetch_lookahead",
                &self.config.point_prefetch_lookahead(),
            )
            .finish()
    }
}

impl EncodedLabelQuery<'_> {
    /// Select a concrete evaluator for repeated point classification.
    ///
    /// Callers should match the returned enum once before entering a hot loop. Each contained
    /// evaluator implements [`diskann::graph::ext::labeled::QueryLabelProvider`]. The owning query
    /// remains the [`diskann::graph::ext::labeled::CandidateLabelProvider`] used for exhaustive
    /// enumeration.
    pub fn point_evaluator(&self) -> EncodedLabelPointEvaluator<'_> {
        let cache_rejections = self.should_cache_rejections();
        match &self.point_plan {
            PointPlan::Dnf(plan) if plan.use_summary => {
                EncodedLabelPointEvaluator::DnfSummary(DnfSummaryEvaluator {
                    num_vectors: self.num_vectors,
                    bits: &self.payload.bits,
                    summary_bits: &self.payload.nonzero_words.bits,
                    clause_offsets: &plan.clause_offsets,
                    payload_offsets: &plan.payload_offsets,
                    summary_offsets: &plan.summary_offsets,
                    cache_rejections,
                })
            }
            PointPlan::Dnf(plan) => EncodedLabelPointEvaluator::DnfPayload(DnfPayloadEvaluator {
                num_vectors: self.num_vectors,
                bits: &self.payload.bits,
                clause_offsets: &plan.clause_offsets,
                payload_offsets: &plan.payload_offsets,
                prefetch_word_offset: plan.payload_offsets.first().copied(),
                prefetch_lookahead: self.config.point_prefetch_lookahead(),
                cache_rejections,
            }),
            PointPlan::Cnf(plan) if plan.use_summary => {
                EncodedLabelPointEvaluator::CnfSummary(CnfSummaryEvaluator {
                    num_vectors: self.num_vectors,
                    bits: &self.payload.bits,
                    summary_bits: &self.payload.nonzero_words.bits,
                    clause_offsets: &plan.clause_offsets,
                    terminal_offsets: &plan.terminal_offsets,
                    payload_offsets: &plan.payload_offsets,
                    summary_offsets: &plan.summary_offsets,
                    cache_rejections,
                })
            }
            PointPlan::Cnf(plan) => EncodedLabelPointEvaluator::CnfPayload(CnfPayloadEvaluator {
                num_vectors: self.num_vectors,
                bits: &self.payload.bits,
                clause_offsets: &plan.clause_offsets,
                terminal_offsets: &plan.terminal_offsets,
                payload_offsets: &plan.payload_offsets,
                prefetch_word_offset: plan.payload_offsets.first().copied(),
                prefetch_lookahead: self.config.point_prefetch_lookahead(),
                cache_rejections,
            }),
        }
    }
}

impl EncodedLabelQuery<'_> {
    #[cfg(test)]
    pub(crate) fn point_plan_uses_summary(&self) -> bool {
        match &self.point_plan {
            PointPlan::Dnf(plan) => plan.use_summary,
            PointPlan::Cnf(plan) => plan.use_summary,
        }
    }

    #[cfg(test)]
    pub(crate) fn point_plan_payload_rows(&self) -> Vec<usize> {
        let offsets = match &self.point_plan {
            PointPlan::Dnf(plan) => &plan.payload_offsets,
            PointPlan::Cnf(plan) => &plan.payload_offsets,
        };
        offsets
            .iter()
            .map(|offset| offset / self.payload.words_per_row)
            .collect()
    }

    /// Return a conservative bound on exact matching vectors, if exact label counts were stored.
    pub fn match_upper_bound(&self) -> Option<u64> {
        let count =
            |label_id: &Option<u32>| label_id.map_or(0, |id| u64::from(self.counts[id as usize]));
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
    #[inline]
    pub fn is_match(&self, vec_id: u32) -> bool {
        if vec_id >= self.num_vectors {
            return false;
        }
        match &self.point_plan {
            PointPlan::Dnf(plan) if plan.use_summary => dnf_summary_matches(
                &self.payload.bits,
                &self.payload.nonzero_words.bits,
                &plan.clause_offsets,
                &plan.payload_offsets,
                &plan.summary_offsets,
                vec_id,
            ),
            PointPlan::Dnf(plan) => dnf_payload_matches(
                &self.payload.bits,
                &plan.clause_offsets,
                &plan.payload_offsets,
                vec_id,
            ),
            PointPlan::Cnf(plan) if plan.use_summary => cnf_summary_matches(
                &self.payload.bits,
                &self.payload.nonzero_words.bits,
                &plan.clause_offsets,
                &plan.terminal_offsets,
                &plan.payload_offsets,
                &plan.summary_offsets,
                vec_id,
            ),
            PointPlan::Cnf(plan) => cnf_payload_matches(
                &self.payload.bits,
                &plan.clause_offsets,
                &plan.terminal_offsets,
                &plan.payload_offsets,
                vec_id,
            ),
        }
    }

    fn should_cache_rejections(&self) -> bool {
        self.plan
            .label_ids
            .iter()
            .flatten()
            .any(|&id| self.payload.exact_row_by_label[id as usize] == u32::MAX)
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
        if words_per_row >= 4 && std::is_x86_feature_detected!("avx2") {
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
        self.payload.words_per_row
    }

    fn summary_word_matches(&self, word_index: usize) -> Option<u64> {
        let summary = &self.payload.nonzero_words;
        let word_for_label = |label_id: Option<u32>| {
            let Some(label_id) = label_id.map(|id| id as usize) else {
                return 0;
            };
            let exact_row = self.payload.exact_row_by_label[label_id];
            if exact_row != u32::MAX {
                return summary.bits[exact_row as usize * summary.words_per_row + word_index];
            }
            let start = label_id * self.payload.hash_count;
            self.payload.label_rows[start..start + self.payload.hash_count]
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
        let summary = &self.payload.nonzero_words;
        if !summary.scan_useful {
            return false;
        }

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
                .should_scan_summarized_words(possible_word_count, self.payload.words_per_row)
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
        let summary = &self.payload.nonzero_words;
        if !summary.scan_useful {
            return false;
        }

        let possible_word_count: usize = (0..summary.words_per_row)
            .map(|summary_index| {
                self.summary_word_matches(summary_index)
                    .expect("Bloom summary disappeared while sizing")
                    .count_ones() as usize
            })
            .sum();
        if !self
            .config
            .should_scan_summarized_words(possible_word_count, self.payload.words_per_row)
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
            let exact_row = self.payload.exact_row_by_label[label_id];
            if exact_row != u32::MAX {
                return self.payload.bits[exact_row as usize * words_per_row + word_index];
            }
            let start = label_id * self.payload.hash_count;
            self.payload.label_rows[start..start + self.payload.hash_count]
                .iter()
                .fold(u64::MAX, |word, &row| {
                    word & self.payload.bits[row as usize * words_per_row + word_index]
                })
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

        let words_per_row = self.payload.words_per_row;
        let hash_count = self.payload.hash_count;
        let bits = &self.payload.bits;
        let label_rows = &self.payload.label_rows;
        let exact_row_by_label = &self.payload.exact_row_by_label;
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
                            .add(exact_row as usize * words_per_row + word_index)
                            .cast::<__m256i>()
                    };
                    // SAFETY: The four exact-row words at `ptr` are in bounds.
                    return unsafe { _mm256_loadu_si256(ptr) };
                }
                let start = label_id * hash_count;
                label_rows[start..start + hash_count].iter().fold(
                    _mm256_set1_epi64x(-1),
                    |word, &row| {
                        // SAFETY: Every loaded row has `words_per_row` words and this block
                        // starts at most four words before the end; unaligned loads are valid.
                        let ptr = unsafe {
                            bits.as_ptr()
                                .add(row as usize * words_per_row + word_index)
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
        for word_index in simd_end..words_per_row {
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
        let summary = &self.payload.nonzero_words;
        if summary.scan_useful {
            let possible_word_count: usize = (0..summary.words_per_row)
                .map(|summary_index| {
                    self.summary_word_matches(summary_index)
                        .expect("Bloom summary disappeared while sizing count")
                        .count_ones() as usize
                })
                .sum();
            if self
                .config
                .should_scan_summarized_words(possible_word_count, self.payload.words_per_row)
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

#[inline]
fn prefetch_payload_word(bits: &[u64], word_offset: usize, vec_id: u32) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

        let word_index = vec_id as usize / 64;
        // SAFETY: Point plans validate row offsets during query compilation, and callers validate
        // `vec_id` against the index vector count before prefetching.
        let ptr = unsafe { bits.as_ptr().add(word_offset + word_index) };
        // SAFETY: `ptr` addresses a valid payload word. Prefetch is only a cache hint and does not
        // dereference the pointer architecturally.
        unsafe { _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0) };
    }

    #[cfg(not(target_arch = "x86_64"))]
    let _ = (bits, word_offset, vec_id);
}

#[inline]
fn dnf_payload_matches(
    bits: &[u64],
    clause_offsets: &[usize],
    payload_offsets: &[usize],
    vec_id: u32,
) -> bool {
    let word_index = vec_id as usize / 64;
    let mask = 1u64 << (vec_id % 64);
    let mut clause_index = 0usize;
    while clause_index + 1 < clause_offsets.len() {
        let mut row_index = clause_offsets[clause_index];
        let row_end = clause_offsets[clause_index + 1];
        while row_index < row_end && bits[payload_offsets[row_index] + word_index] & mask != 0 {
            row_index += 1;
        }
        if row_index == row_end {
            return true;
        }
        clause_index += 1;
    }
    false
}

#[inline]
fn dnf_summary_matches(
    bits: &[u64],
    summary_bits: &[u64],
    clause_offsets: &[usize],
    payload_offsets: &[usize],
    summary_offsets: &[usize],
    vec_id: u32,
) -> bool {
    let word_index = vec_id as usize / 64;
    let summary_word_index = word_index / 64;
    let summary_mask = 1u64 << (word_index % 64);
    let mask = 1u64 << (vec_id % 64);
    let mut clause_index = 0usize;
    while clause_index + 1 < clause_offsets.len() {
        let mut row_index = clause_offsets[clause_index];
        let row_end = clause_offsets[clause_index + 1];
        while row_index < row_end
            && summary_bits[summary_offsets[row_index] + summary_word_index] & summary_mask != 0
            && bits[payload_offsets[row_index] + word_index] & mask != 0
        {
            row_index += 1;
        }
        if row_index == row_end {
            return true;
        }
        clause_index += 1;
    }
    false
}

#[inline]
fn cnf_payload_matches(
    bits: &[u64],
    clause_offsets: &[usize],
    terminal_offsets: &[usize],
    payload_offsets: &[usize],
    vec_id: u32,
) -> bool {
    let word_index = vec_id as usize / 64;
    let mask = 1u64 << (vec_id % 64);
    let mut clause_index = 0usize;
    while clause_index + 1 < clause_offsets.len() {
        let mut terminal = clause_offsets[clause_index];
        let terminal_end = clause_offsets[clause_index + 1];
        while terminal < terminal_end {
            let mut row_index = terminal_offsets[terminal];
            let row_end = terminal_offsets[terminal + 1];
            while row_index < row_end && bits[payload_offsets[row_index] + word_index] & mask != 0 {
                row_index += 1;
            }
            if row_index == row_end {
                break;
            }
            terminal += 1;
        }
        if terminal == terminal_end {
            return false;
        }
        clause_index += 1;
    }
    true
}

#[inline]
fn cnf_summary_matches(
    bits: &[u64],
    summary_bits: &[u64],
    clause_offsets: &[usize],
    terminal_offsets: &[usize],
    payload_offsets: &[usize],
    summary_offsets: &[usize],
    vec_id: u32,
) -> bool {
    let word_index = vec_id as usize / 64;
    let summary_word_index = word_index / 64;
    let summary_mask = 1u64 << (word_index % 64);
    let mask = 1u64 << (vec_id % 64);
    let mut clause_index = 0usize;
    while clause_index + 1 < clause_offsets.len() {
        let mut terminal = clause_offsets[clause_index];
        let terminal_end = clause_offsets[clause_index + 1];
        while terminal < terminal_end {
            let mut row_index = terminal_offsets[terminal];
            let row_end = terminal_offsets[terminal + 1];
            while row_index < row_end
                && summary_bits[summary_offsets[row_index] + summary_word_index] & summary_mask != 0
                && bits[payload_offsets[row_index] + word_index] & mask != 0
            {
                row_index += 1;
            }
            if row_index == row_end {
                break;
            }
            terminal += 1;
        }
        if terminal == terminal_end {
            return false;
        }
        clause_index += 1;
    }
    true
}

impl diskann::graph::ext::labeled::QueryLabelProvider<u32> for DnfPayloadEvaluator<'_> {
    #[inline]
    fn is_match(&self, id: u32) -> bool {
        id < self.num_vectors
            && dnf_payload_matches(self.bits, self.clause_offsets, self.payload_offsets, id)
    }

    #[inline]
    fn prefetch(&self, id: u32) {
        if id < self.num_vectors {
            if let Some(offset) = self.prefetch_word_offset {
                prefetch_payload_word(self.bits, offset, id);
            }
        }
    }

    #[inline]
    fn prefetch_lookahead(&self) -> usize {
        if cfg!(target_arch = "x86_64") {
            self.prefetch_lookahead
        } else {
            0
        }
    }

    fn cache_rejections(&self) -> bool {
        self.cache_rejections
    }
}

impl diskann::graph::ext::labeled::QueryLabelProvider<u32> for DnfSummaryEvaluator<'_> {
    #[inline]
    fn is_match(&self, id: u32) -> bool {
        id < self.num_vectors
            && dnf_summary_matches(
                self.bits,
                self.summary_bits,
                self.clause_offsets,
                self.payload_offsets,
                self.summary_offsets,
                id,
            )
    }

    fn cache_rejections(&self) -> bool {
        self.cache_rejections
    }
}

impl diskann::graph::ext::labeled::QueryLabelProvider<u32> for CnfPayloadEvaluator<'_> {
    #[inline]
    fn is_match(&self, id: u32) -> bool {
        id < self.num_vectors
            && cnf_payload_matches(
                self.bits,
                self.clause_offsets,
                self.terminal_offsets,
                self.payload_offsets,
                id,
            )
    }

    #[inline]
    fn prefetch(&self, id: u32) {
        if id < self.num_vectors {
            if let Some(offset) = self.prefetch_word_offset {
                prefetch_payload_word(self.bits, offset, id);
            }
        }
    }

    #[inline]
    fn prefetch_lookahead(&self) -> usize {
        if cfg!(target_arch = "x86_64") {
            self.prefetch_lookahead
        } else {
            0
        }
    }

    fn cache_rejections(&self) -> bool {
        self.cache_rejections
    }
}

impl diskann::graph::ext::labeled::QueryLabelProvider<u32> for CnfSummaryEvaluator<'_> {
    #[inline]
    fn is_match(&self, id: u32) -> bool {
        id < self.num_vectors
            && cnf_summary_matches(
                self.bits,
                self.summary_bits,
                self.clause_offsets,
                self.terminal_offsets,
                self.payload_offsets,
                self.summary_offsets,
                id,
            )
    }

    fn cache_rejections(&self) -> bool {
        self.cache_rejections
    }
}

impl diskann::graph::ext::labeled::QueryLabelProvider<u32> for EncodedLabelQuery<'_> {
    #[inline]
    fn is_match(&self, id: u32) -> bool {
        EncodedLabelQuery::is_match(self, id)
    }

    fn cache_rejections(&self) -> bool {
        self.should_cache_rejections()
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
    /// Load a unified exact/Bloom label index from `path`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, EncodedLabelIndexError> {
        load_label_index(path.as_ref())
    }

    /// Return the number of vectors covered by this index.
    pub fn num_vectors(&self) -> u32 {
        self.num_vectors
    }

    /// Return the number of encoded labels.
    pub fn num_labels(&self) -> usize {
        self.labels.len()
    }

    /// Return how many dictionary labels use exact Bitslice rows.
    pub fn num_exact_labels(&self) -> usize {
        self.payload.exact_label_count
    }

    /// Return valid-node counts for physical rows in exact-prefix then Bloom-row order.
    pub fn physical_row_valid_counts(&self) -> &[u32] {
        &self.payload.row_valid_counts
    }

    /// Return nonzero payload-word counts for physical rows.
    pub fn physical_row_nonzero_word_counts(&self) -> &[u32] {
        &self.summary_nonzero_words_per_row
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
        let point_plan = compile_point_plan(
            &plan,
            &self.payload,
            &self.counts,
            &self.summary_nonzero_words_per_row,
        );
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

fn load_label_index(path: &Path) -> Result<EncodedLabelIndex, EncodedLabelIndexError> {
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
    let flags = read_u32(&mut reader)?;
    if flags != LABEL_INDEX_FLAGS {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "unsupported label-index flags {flags}"
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
    if num_labels == 0 || num_labels > MAX_LABEL_COUNT {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "label-index label count {num_labels} is outside 1..={MAX_LABEL_COUNT}"
        )));
    }
    let exact_label_count = usize::try_from(read_u32(&mut reader)?).map_err(|_| {
        EncodedLabelIndexError::Invalid("exact label count exceeds usize".to_string())
    })?;
    let bloom_row_count = usize::try_from(read_u32(&mut reader)?).map_err(|_| {
        EncodedLabelIndexError::Invalid("Bloom row count exceeds usize".to_string())
    })?;
    let hash_count = usize::try_from(read_u32(&mut reader)?).map_err(|_| {
        EncodedLabelIndexError::Invalid("Bloom hash count exceeds usize".to_string())
    })?;
    let reserved = read_u32(&mut reader)?;
    if reserved != 0 {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "label-index reserved field is {reserved}, expected 0"
        )));
    }
    if exact_label_count > num_labels {
        return Err(EncodedLabelIndexError::Invalid(
            "exact label count exceeds label count".to_string(),
        ));
    }
    if exact_label_count == num_labels {
        if bloom_row_count != 0 || hash_count != 0 {
            return Err(EncodedLabelIndexError::Invalid(
                "all-exact label-index cannot contain Bloom rows".to_string(),
            ));
        }
    } else {
        let bloom_rows = u32::try_from(bloom_row_count).map_err(|_| {
            EncodedLabelIndexError::Invalid("Bloom row count exceeds u32".to_string())
        })?;
        let hashes = u32::try_from(hash_count).map_err(|_| {
            EncodedLabelIndexError::Invalid("Bloom hash count exceeds u32".to_string())
        })?;
        validate_bloom_config(bloom_rows, hashes)?;
    }
    let physical_row_count = exact_label_count
        .checked_add(bloom_row_count)
        .ok_or_else(|| {
            EncodedLabelIndexError::Invalid("physical row count overflow".to_string())
        })?;
    if physical_row_count == 0 {
        return Err(EncodedLabelIndexError::Invalid(
            "label-index must contain at least one physical row".to_string(),
        ));
    }

    let words_per_row = usize::try_from(read_u64(&mut reader)?).map_err(|_| {
        EncodedLabelIndexError::Invalid("payload row length exceeds usize".to_string())
    })?;
    let expected_words_per_row = (num_vectors as usize).div_ceil(64);
    if words_per_row != expected_words_per_row {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "payload row has {words_per_row} words; expected {expected_words_per_row}"
        )));
    }
    let summary_words_per_row = usize::try_from(read_u64(&mut reader)?).map_err(|_| {
        EncodedLabelIndexError::Invalid("summary row length exceeds usize".to_string())
    })?;
    let expected_summary_words = words_per_row.div_ceil(64);
    if summary_words_per_row != expected_summary_words {
        return Err(EncodedLabelIndexError::Invalid(format!(
            "summary row has {summary_words_per_row} words; expected {expected_summary_words}"
        )));
    }

    let minimum_dictionary_bytes = num_labels.checked_mul(8).ok_or_else(|| {
        EncodedLabelIndexError::Invalid("label dictionary size overflow".to_string())
    })?;
    ensure_remaining(
        &mut reader,
        file_len,
        minimum_dictionary_bytes,
        "label dictionary",
    )?;
    let mut labels = Vec::new();
    let mut counts = Vec::new();
    let mut label_ids = HashMap::new();
    labels.try_reserve_exact(num_labels).map_err(|_| {
        EncodedLabelIndexError::Invalid("cannot reserve label dictionary".to_string())
    })?;
    counts
        .try_reserve_exact(num_labels)
        .map_err(|_| EncodedLabelIndexError::Invalid("cannot reserve label counts".to_string()))?;
    label_ids.try_reserve(num_labels).map_err(|_| {
        EncodedLabelIndexError::Invalid("cannot reserve label lookup map".to_string())
    })?;
    for id in 0..num_labels {
        let len = usize::try_from(read_u32(&mut reader)?).map_err(|_| {
            EncodedLabelIndexError::Invalid("label length exceeds usize".to_string())
        })?;
        let count = read_u32(&mut reader)?;
        if len > MAX_LABEL_LENGTH {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "label length {len} exceeds limit {MAX_LABEL_LENGTH}"
            )));
        }
        if count == 0 || count > num_vectors {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "label count {count} is outside 1..={num_vectors}"
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
        let id = u32::try_from(id)
            .map_err(|_| EncodedLabelIndexError::Invalid("label count exceeds u32".to_string()))?;
        if label_ids.insert(label.clone(), id).is_some() {
            return Err(EncodedLabelIndexError::Invalid(format!(
                "duplicate label '{label}' in label-index"
            )));
        }
        labels.push(label);
        counts.push(count);
    }

    let total_label_rows = num_labels.checked_mul(hash_count).ok_or_else(|| {
        EncodedLabelIndexError::Invalid("label-row table size overflow".to_string())
    })?;
    let mut label_rows = Vec::new();
    label_rows
        .try_reserve_exact(total_label_rows)
        .map_err(|_| {
            EncodedLabelIndexError::Invalid("cannot reserve label-row table".to_string())
        })?;
    label_rows.resize(total_label_rows, u32::MAX);
    let mut exact_row_by_label = Vec::new();
    exact_row_by_label
        .try_reserve_exact(num_labels)
        .map_err(|_| {
            EncodedLabelIndexError::Invalid("cannot reserve exact-row table".to_string())
        })?;
    exact_row_by_label.resize(num_labels, u32::MAX);
    for (label_id, exact_row) in exact_row_by_label
        .iter_mut()
        .take(exact_label_count)
        .enumerate()
    {
        *exact_row = u32::try_from(label_id).map_err(|_| {
            EncodedLabelIndexError::Invalid("exact row index exceeds u32".to_string())
        })?;
    }
    let mapping_count = (num_labels - exact_label_count)
        .checked_mul(hash_count)
        .ok_or_else(|| {
            EncodedLabelIndexError::Invalid("Bloom mapping count overflow".to_string())
        })?;
    ensure_remaining(
        &mut reader,
        file_len,
        mapping_count.checked_mul(4).ok_or_else(|| {
            EncodedLabelIndexError::Invalid("Bloom mapping byte size overflow".to_string())
        })?,
        "Bloom label-row mappings",
    )?;
    for label_id in exact_label_count..num_labels {
        let start = label_id * hash_count;
        for hash_index in 0..hash_count {
            let row = read_u32(&mut reader)?;
            if row < exact_label_count as u32
                || row as usize >= physical_row_count
                || label_rows[start..start + hash_index].contains(&row)
            {
                return Err(EncodedLabelIndexError::Invalid(
                    "Bloom label rows must be distinct physical Bloom rows".to_string(),
                ));
            }
            label_rows[start + hash_index] = row;
        }
    }

    ensure_remaining(
        &mut reader,
        file_len,
        physical_row_count.checked_mul(8).ok_or_else(|| {
            EncodedLabelIndexError::Invalid("row metadata byte size overflow".to_string())
        })?,
        "row metadata",
    )?;
    let mut row_valid_counts = Vec::new();
    let mut row_nonzero_word_counts = Vec::new();
    row_valid_counts
        .try_reserve_exact(physical_row_count)
        .map_err(|_| {
            EncodedLabelIndexError::Invalid("cannot reserve row valid counts".to_string())
        })?;
    row_nonzero_word_counts
        .try_reserve_exact(physical_row_count)
        .map_err(|_| {
            EncodedLabelIndexError::Invalid("cannot reserve row nonzero counts".to_string())
        })?;
    for _ in 0..physical_row_count {
        let valid_node_count = read_u32(&mut reader)?;
        let nonzero_word_count = read_u32(&mut reader)?;
        if valid_node_count > num_vectors || nonzero_word_count as usize > words_per_row {
            return Err(EncodedLabelIndexError::Invalid(
                "row metadata exceeds index dimensions".to_string(),
            ));
        }
        row_valid_counts.push(valid_node_count);
        row_nonzero_word_counts.push(nonzero_word_count);
    }
    if row_valid_counts[..exact_label_count] != counts[..exact_label_count] {
        return Err(EncodedLabelIndexError::Invalid(
            "exact row valid count does not match label count".to_string(),
        ));
    }

    let summary_bits = read_payload(
        &mut reader,
        file_len,
        physical_row_count,
        summary_words_per_row,
        "label summary",
    )?;
    let summarized_words = u32::try_from(words_per_row).map_err(|_| {
        EncodedLabelIndexError::Invalid("payload row word count exceeds u32".to_string())
    })?;
    validate_row_padding(
        &summary_bits,
        physical_row_count,
        summary_words_per_row,
        summarized_words,
    )?;
    for (row, summary) in summary_bits.chunks_exact(summary_words_per_row).enumerate() {
        let nonzero = summary.iter().map(|word| word.count_ones()).sum::<u32>();
        if nonzero != row_nonzero_word_counts[row] {
            return Err(EncodedLabelIndexError::Invalid(
                "summary popcount does not match row metadata".to_string(),
            ));
        }
    }

    let payload_bits = read_payload(
        &mut reader,
        file_len,
        physical_row_count,
        words_per_row,
        "label payload",
    )?;
    validate_row_padding(
        &payload_bits,
        physical_row_count,
        words_per_row,
        num_vectors,
    )?;
    if reader.read(&mut [0u8; 1])? != 0 {
        return Err(EncodedLabelIndexError::Invalid(
            "label-index contains trailing bytes".to_string(),
        ));
    }

    let total_nonzero_words = row_nonzero_word_counts
        .iter()
        .try_fold(0usize, |sum, &count| sum.checked_add(count as usize))
        .ok_or_else(|| {
            EncodedLabelIndexError::Invalid("row nonzero word count overflow".to_string())
        })?;
    let total_payload_words = physical_row_count
        .checked_mul(words_per_row)
        .ok_or_else(|| {
            EncodedLabelIndexError::Invalid("payload word count overflow".to_string())
        })?;
    let summary = BloomWordSummary {
        words_per_row: summary_words_per_row,
        bits: summary_bits.into_boxed_slice(),
        scan_useful: exact_label_count > 0
            || total_nonzero_words.saturating_mul(4) <= total_payload_words,
    };
    let payload = LabelIndexPayload {
        words_per_row,
        hash_count,
        bits: payload_bits.into_boxed_slice(),
        label_rows: label_rows.into_boxed_slice(),
        exact_row_by_label: exact_row_by_label.into_boxed_slice(),
        exact_label_count,
        row_valid_counts: row_valid_counts.into_boxed_slice(),
        nonzero_words: summary,
    };

    Ok(EncodedLabelIndex {
        labels: labels.into_boxed_slice(),
        label_ids,
        counts: Arc::from(counts),
        summary_nonzero_words_per_row: Arc::from(row_nonzero_word_counts),
        num_vectors,
        payload: Arc::new(payload),
    })
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

fn compile_point_plan(
    plan: &CompiledPlan,
    payload: &LabelIndexPayload,
    counts: &[u32],
    summary_nonzero_words_per_row: &[u32],
) -> PointPlan {
    match plan.expression_type {
        FilterExpressionType::DNF => PointPlan::Dnf(Box::new(compile_dnf_point_plan(
            plan,
            payload,
            summary_nonzero_words_per_row,
        ))),
        FilterExpressionType::CNF => PointPlan::Cnf(Box::new(compile_cnf_point_plan(
            plan,
            payload,
            counts,
            summary_nonzero_words_per_row,
        ))),
    }
}

fn label_physical_rows(payload: &LabelIndexPayload, label_id: usize, rows: &mut Vec<u32>) {
    let exact_row = payload.exact_row_by_label[label_id];
    if exact_row != u32::MAX {
        rows.push(exact_row);
        return;
    }
    let start = label_id * payload.hash_count;
    rows.extend_from_slice(&payload.label_rows[start..start + payload.hash_count]);
}

fn sort_rows_by_valid_count(payload: &LabelIndexPayload, rows: &mut Vec<u32>) {
    rows.sort_unstable_by_key(|&row| (payload.row_valid_counts[row as usize], row));
    rows.dedup();
}

fn row_uses_summary(
    payload: &LabelIndexPayload,
    summary_nonzero_words_per_row: &[u32],
    payload_offset: usize,
) -> bool {
    let row = payload_offset / payload.words_per_row;
    let nonzero = summary_nonzero_words_per_row[row] as usize;
    let zero = payload.words_per_row.saturating_sub(nonzero);
    zero.saturating_mul(100)
        >= payload
            .words_per_row
            .saturating_mul(POINT_SUMMARY_MIN_ZERO_PERCENT)
}

fn append_row_offsets(
    payload: &LabelIndexPayload,
    rows: &[u32],
    payload_offsets: &mut Vec<usize>,
    summary_offsets: &mut Vec<usize>,
) {
    payload_offsets.extend(rows.iter().map(|&row| row as usize * payload.words_per_row));
    summary_offsets.extend(
        rows.iter()
            .map(|&row| row as usize * payload.nonzero_words.words_per_row),
    );
}

fn compile_dnf_point_plan(
    plan: &CompiledPlan,
    payload: &LabelIndexPayload,
    summary_nonzero_words_per_row: &[u32],
) -> DnfPointPlan {
    let mut clause_offsets = vec![0usize];
    let mut payload_offsets = Vec::new();
    let mut summary_offsets = Vec::new();
    let mut clause_rows = Vec::new();

    for clause in plan.clause_offsets.windows(2) {
        clause_rows.clear();
        let mut valid_clause = true;
        for label_id in &plan.label_ids[clause[0]..clause[1]] {
            let Some(label_id) = label_id.map(|id| id as usize) else {
                valid_clause = false;
                break;
            };
            label_physical_rows(payload, label_id, &mut clause_rows);
        }
        if !valid_clause {
            continue;
        }
        sort_rows_by_valid_count(payload, &mut clause_rows);
        append_row_offsets(
            payload,
            &clause_rows,
            &mut payload_offsets,
            &mut summary_offsets,
        );
        clause_offsets.push(payload_offsets.len());
    }

    let use_summary = clause_offsets.len() > 1
        && clause_offsets.windows(2).all(|clause| {
            clause[0] < clause[1]
                && row_uses_summary(
                    payload,
                    summary_nonzero_words_per_row,
                    payload_offsets[clause[0]],
                )
        });
    DnfPointPlan {
        clause_offsets: clause_offsets.into_boxed_slice(),
        payload_offsets: payload_offsets.into_boxed_slice(),
        summary_offsets: summary_offsets.into_boxed_slice(),
        use_summary,
    }
}

fn compile_cnf_point_plan(
    plan: &CompiledPlan,
    payload: &LabelIndexPayload,
    counts: &[u32],
    summary_nonzero_words_per_row: &[u32],
) -> CnfPointPlan {
    let mut clause_offsets = vec![0usize];
    let mut terminal_offsets = vec![0usize];
    let mut payload_offsets = Vec::new();
    let mut summary_offsets = Vec::new();
    let mut clause_labels = Vec::new();
    let mut terminal_rows = Vec::new();

    for clause in plan.clause_offsets.windows(2) {
        clause_labels.clear();
        clause_labels.extend(
            plan.label_ids[clause[0]..clause[1]]
                .iter()
                .flatten()
                .map(|&id| id as usize),
        );
        clause_labels.sort_unstable();
        clause_labels.dedup();
        clause_labels.sort_unstable_by(|&left, &right| {
            counts[right]
                .cmp(&counts[left])
                .then_with(|| left.cmp(&right))
        });

        for &label_id in &clause_labels {
            terminal_rows.clear();
            label_physical_rows(payload, label_id, &mut terminal_rows);
            sort_rows_by_valid_count(payload, &mut terminal_rows);
            append_row_offsets(
                payload,
                &terminal_rows,
                &mut payload_offsets,
                &mut summary_offsets,
            );
            terminal_offsets.push(payload_offsets.len());
        }
        clause_offsets.push(terminal_offsets.len() - 1);
    }

    let use_summary = clause_offsets.windows(2).all(|clause| {
        clause[0] < clause[1]
            && (clause[0]..clause[1]).all(|terminal| {
                let row = terminal_offsets[terminal];
                row < terminal_offsets[terminal + 1]
                    && row_uses_summary(
                        payload,
                        summary_nonzero_words_per_row,
                        payload_offsets[row],
                    )
            })
    });
    CnfPointPlan {
        clause_offsets: clause_offsets.into_boxed_slice(),
        terminal_offsets: terminal_offsets.into_boxed_slice(),
        payload_offsets: payload_offsets.into_boxed_slice(),
        summary_offsets: summary_offsets.into_boxed_slice(),
        use_summary,
    }
}
