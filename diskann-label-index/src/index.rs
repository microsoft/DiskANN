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
    exact_point_prefetch_lookahead: usize,
}

const DEFAULT_EXACT_POINT_PREFETCH_LOOKAHEAD: usize = 32;

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
            exact_point_prefetch_lookahead: DEFAULT_EXACT_POINT_PREFETCH_LOOKAHEAD,
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

    /// Set the exact payload point-classification prefetch lookahead.
    ///
    /// Zero disables prefetching. Nonzero values are used only on supported architectures and only
    /// by exact payload evaluators.
    pub fn with_exact_point_prefetch_lookahead(mut self, lookahead: usize) -> Self {
        self.exact_point_prefetch_lookahead = lookahead;
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

    /// Return the exact payload point-classification prefetch lookahead.
    pub fn exact_point_prefetch_lookahead(self) -> usize {
        self.exact_point_prefetch_lookahead
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
            exact_point_prefetch_lookahead: DEFAULT_EXACT_POINT_PREFETCH_LOOKAHEAD,
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
    Dnf,
    Cnf,
    ExactAnd2Payload {
        first_word_offset: usize,
        second_word_offset: usize,
    },
    // Keep the four-offset summary plan out of line so PointPlan remains as compact as the
    // payload-only plan. The allocation occurs once during query compilation.
    ExactAnd2Summary(Box<ExactAnd2SummaryPlan>),
}

#[derive(Debug)]
struct ExactAnd2SummaryPlan {
    first_word_offset: usize,
    second_word_offset: usize,
    first_summary_word_offset: usize,
    second_summary_word_offset: usize,
}

#[derive(Clone, Copy)]
struct GenericEvaluator<'a> {
    num_vectors: u32,
    payload: &'a LabelIndexPayload,
    clause_offsets: &'a [usize],
    label_ids: &'a [Option<u32>],
    cache_rejections: bool,
}

impl std::fmt::Debug for GenericEvaluator<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenericEvaluator")
            .field("num_vectors", &self.num_vectors)
            .field("num_clauses", &self.clause_offsets.len().saturating_sub(1))
            .field("num_terminals", &self.label_ids.len())
            .field("cache_rejections", &self.cache_rejections)
            .finish()
    }
}

/// A point evaluator selected once from a compiled [`EncodedLabelQuery`].
///
/// Match this enum outside a repeated classification loop so each variant's concrete evaluator
/// can be statically dispatched without checking the query mode for every vector ID.
#[derive(Debug, Clone, Copy)]
pub enum EncodedLabelPointEvaluator<'a> {
    /// Two exact rows evaluated directly from the payload.
    ExactAnd2Payload(ExactAnd2PayloadEvaluator<'a>),
    /// Two exact rows guarded by compact nonzero-word summaries.
    ExactAnd2Summary(ExactAnd2SummaryEvaluator<'a>),
    /// A generic DNF evaluator.
    Dnf(DnfEvaluator<'a>),
    /// A generic CNF evaluator.
    Cnf(CnfEvaluator<'a>),
}

/// A statically dispatched evaluator for two exact payload rows.
#[derive(Debug, Clone, Copy)]
pub struct ExactAnd2PayloadEvaluator<'a> {
    num_vectors: u32,
    bits: &'a [u64],
    first_word_offset: usize,
    second_word_offset: usize,
    prefetch_lookahead: usize,
}

/// A statically dispatched evaluator for two summary-guarded exact rows.
#[derive(Debug, Clone, Copy)]
pub struct ExactAnd2SummaryEvaluator<'a> {
    num_vectors: u32,
    bits: &'a [u64],
    summary_bits: &'a [u64],
    first_word_offset: usize,
    second_word_offset: usize,
    first_summary_word_offset: usize,
    second_summary_word_offset: usize,
}

/// A statically dispatched generic DNF evaluator.
#[derive(Debug, Clone, Copy)]
pub struct DnfEvaluator<'a> {
    inner: GenericEvaluator<'a>,
}

/// A statically dispatched generic CNF evaluator.
#[derive(Debug, Clone, Copy)]
pub struct CnfEvaluator<'a> {
    inner: GenericEvaluator<'a>,
}

const EXACT_POINT_SUMMARY_MIN_ZERO_PERCENT: usize = 85;

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
                "exact_point_prefetch_lookahead",
                &self.config.exact_point_prefetch_lookahead(),
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
        match &self.point_plan {
            PointPlan::ExactAnd2Payload {
                first_word_offset,
                second_word_offset,
            } => EncodedLabelPointEvaluator::ExactAnd2Payload(ExactAnd2PayloadEvaluator {
                num_vectors: self.num_vectors,
                bits: &self.payload.bits,
                first_word_offset: *first_word_offset,
                second_word_offset: *second_word_offset,
                prefetch_lookahead: self.config.exact_point_prefetch_lookahead(),
            }),
            PointPlan::ExactAnd2Summary(plan) => {
                EncodedLabelPointEvaluator::ExactAnd2Summary(ExactAnd2SummaryEvaluator {
                    num_vectors: self.num_vectors,
                    bits: &self.payload.bits,
                    summary_bits: &self.payload.nonzero_words.bits,
                    first_word_offset: plan.first_word_offset,
                    second_word_offset: plan.second_word_offset,
                    first_summary_word_offset: plan.first_summary_word_offset,
                    second_summary_word_offset: plan.second_summary_word_offset,
                })
            }
            PointPlan::Dnf => EncodedLabelPointEvaluator::Dnf(DnfEvaluator {
                inner: self.generic_evaluator(),
            }),
            PointPlan::Cnf => EncodedLabelPointEvaluator::Cnf(CnfEvaluator {
                inner: self.generic_evaluator(),
            }),
        }
    }
}

impl EncodedLabelQuery<'_> {
    #[cfg(test)]
    pub(crate) fn uses_exact_and2_point_plan(&self) -> bool {
        matches!(
            self.point_plan,
            PointPlan::ExactAnd2Payload { .. } | PointPlan::ExactAnd2Summary(_)
        )
    }

    #[cfg(test)]
    pub(crate) fn exact_and2_rows(&self) -> Option<(usize, usize)> {
        let (first_word_offset, second_word_offset) = match &self.point_plan {
            PointPlan::ExactAnd2Payload {
                first_word_offset,
                second_word_offset,
            } => (first_word_offset, second_word_offset),
            PointPlan::ExactAnd2Summary(plan) => {
                (&plan.first_word_offset, &plan.second_word_offset)
            }
            PointPlan::Dnf | PointPlan::Cnf => return None,
        };
        Some((
            *first_word_offset / self.payload.words_per_row,
            *second_word_offset / self.payload.words_per_row,
        ))
    }

    #[cfg(test)]
    pub(crate) fn exact_and2_uses_summary(&self) -> bool {
        matches!(self.point_plan, PointPlan::ExactAnd2Summary(_))
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
            PointPlan::ExactAnd2Payload {
                first_word_offset,
                second_word_offset,
            } => exact_and2_payload_matches(
                &self.payload.bits,
                *first_word_offset,
                *second_word_offset,
                vec_id,
            ),
            PointPlan::ExactAnd2Summary(plan) => exact_and2_summary_matches(
                &self.payload.bits,
                &self.payload.nonzero_words.bits,
                plan.first_word_offset,
                plan.second_word_offset,
                plan.first_summary_word_offset,
                plan.second_summary_word_offset,
                vec_id,
            ),
            PointPlan::Dnf => self.is_match_dnf(vec_id),
            PointPlan::Cnf => self.is_match_cnf(vec_id),
        }
    }

    #[inline(never)]
    fn is_match_dnf(&self, vec_id: u32) -> bool {
        dnf_matches(
            self.payload.as_ref(),
            &self.plan.clause_offsets,
            &self.plan.label_ids,
            vec_id,
        )
    }

    #[inline(never)]
    fn is_match_cnf(&self, vec_id: u32) -> bool {
        cnf_matches(
            self.payload.as_ref(),
            &self.plan.clause_offsets,
            &self.plan.label_ids,
            vec_id,
        )
    }

    fn generic_evaluator(&self) -> GenericEvaluator<'_> {
        GenericEvaluator {
            num_vectors: self.num_vectors,
            payload: self.payload.as_ref(),
            clause_offsets: &self.plan.clause_offsets,
            label_ids: &self.plan.label_ids,
            cache_rejections: self.should_cache_rejections(),
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
fn terminal_matches(payload: &LabelIndexPayload, label_id: Option<u32>, vec_id: u32) -> bool {
    let Some(label_id) = label_id.map(|id| id as usize) else {
        return false;
    };
    let exact_row = payload.exact_row_by_label[label_id];
    if exact_row != u32::MAX {
        let word = payload.bits[exact_row as usize * payload.words_per_row + vec_id as usize / 64];
        return word & (1u64 << (vec_id % 64)) != 0;
    }
    let row_offset = label_id * payload.hash_count;
    payload.label_rows[row_offset..row_offset + payload.hash_count]
        .iter()
        .all(|&row| {
            let word = payload.bits[row as usize * payload.words_per_row + vec_id as usize / 64];
            word & (1u64 << (vec_id % 64)) != 0
        })
}

#[inline(never)]
fn dnf_matches(
    payload: &LabelIndexPayload,
    clause_offsets: &[usize],
    label_ids: &[Option<u32>],
    vec_id: u32,
) -> bool {
    let terminal_matches = |label_id| terminal_matches(payload, label_id, vec_id);
    clause_offsets.windows(2).any(|clause| {
        label_ids[clause[0]..clause[1]]
            .iter()
            .copied()
            .all(terminal_matches)
    })
}

#[inline(never)]
fn cnf_matches(
    payload: &LabelIndexPayload,
    clause_offsets: &[usize],
    label_ids: &[Option<u32>],
    vec_id: u32,
) -> bool {
    let terminal_matches = |label_id| terminal_matches(payload, label_id, vec_id);
    clause_offsets.windows(2).all(|clause| {
        label_ids[clause[0]..clause[1]]
            .iter()
            .copied()
            .any(terminal_matches)
    })
}

#[inline]
fn prefetch_exact_payload_word(bits: &[u64], first_word_offset: usize, vec_id: u32) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

        let word_index = vec_id as usize / 64;
        // SAFETY: Exact point plans validate row offsets during query compilation, and callers
        // validate `vec_id` against the index vector count before prefetching.
        let ptr = unsafe { bits.as_ptr().add(first_word_offset + word_index) };
        // SAFETY: `ptr` addresses a valid payload word. Prefetch is only a cache hint and does not
        // dereference the pointer architecturally.
        unsafe { _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0) };
    }

    #[cfg(not(target_arch = "x86_64"))]
    let _ = (bits, first_word_offset, vec_id);
}

#[inline]
fn exact_and2_payload_matches(
    bits: &[u64],
    first_word_offset: usize,
    second_word_offset: usize,
    vec_id: u32,
) -> bool {
    let word_index = vec_id as usize / 64;
    let mask = 1u64 << (vec_id % 64);
    if bits[first_word_offset + word_index] & mask == 0 {
        return false;
    }
    bits[second_word_offset + word_index] & mask != 0
}

#[inline]
fn exact_and2_summary_matches(
    bits: &[u64],
    summary_bits: &[u64],
    first_word_offset: usize,
    second_word_offset: usize,
    first_summary_word_offset: usize,
    second_summary_word_offset: usize,
    vec_id: u32,
) -> bool {
    let word_index = vec_id as usize / 64;
    let summary_word_index = word_index / 64;
    let summary_mask = 1u64 << (word_index % 64);
    if summary_bits[first_summary_word_offset + summary_word_index] & summary_mask == 0 {
        return false;
    }
    let mask = 1u64 << (vec_id % 64);
    if bits[first_word_offset + word_index] & mask == 0 {
        return false;
    }
    if summary_bits[second_summary_word_offset + summary_word_index] & summary_mask == 0 {
        return false;
    }
    bits[second_word_offset + word_index] & mask != 0
}

impl diskann::graph::ext::labeled::QueryLabelProvider<u32> for ExactAnd2PayloadEvaluator<'_> {
    #[inline]
    fn is_match(&self, id: u32) -> bool {
        id < self.num_vectors
            && exact_and2_payload_matches(
                self.bits,
                self.first_word_offset,
                self.second_word_offset,
                id,
            )
    }

    #[inline]
    fn prefetch(&self, id: u32) {
        if id < self.num_vectors {
            prefetch_exact_payload_word(self.bits, self.first_word_offset, id);
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
}

impl diskann::graph::ext::labeled::QueryLabelProvider<u32> for ExactAnd2SummaryEvaluator<'_> {
    #[inline]
    fn is_match(&self, id: u32) -> bool {
        id < self.num_vectors
            && exact_and2_summary_matches(
                self.bits,
                self.summary_bits,
                self.first_word_offset,
                self.second_word_offset,
                self.first_summary_word_offset,
                self.second_summary_word_offset,
                id,
            )
    }
}

impl diskann::graph::ext::labeled::QueryLabelProvider<u32> for DnfEvaluator<'_> {
    #[inline]
    fn is_match(&self, id: u32) -> bool {
        id < self.inner.num_vectors
            && dnf_matches(
                self.inner.payload,
                self.inner.clause_offsets,
                self.inner.label_ids,
                id,
            )
    }

    fn cache_rejections(&self) -> bool {
        self.inner.cache_rejections
    }
}

impl diskann::graph::ext::labeled::QueryLabelProvider<u32> for CnfEvaluator<'_> {
    #[inline]
    fn is_match(&self, id: u32) -> bool {
        id < self.inner.num_vectors
            && cnf_matches(
                self.inner.payload,
                self.inner.clause_offsets,
                self.inner.label_ids,
                id,
            )
    }

    fn cache_rejections(&self) -> bool {
        self.inner.cache_rejections
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

fn generic_point_plan(expression_type: FilterExpressionType) -> PointPlan {
    match expression_type {
        FilterExpressionType::DNF => PointPlan::Dnf,
        FilterExpressionType::CNF => PointPlan::Cnf,
    }
}

fn compile_point_plan(
    plan: &CompiledPlan,
    payload: &LabelIndexPayload,
    counts: &[u32],
    summary_nonzero_words_per_row: &[u32],
) -> PointPlan {
    if plan.expression_type != FilterExpressionType::DNF
        || plan.clause_offsets.as_ref() != [0, 2]
        || plan.label_ids.len() != 2
    {
        return generic_point_plan(plan.expression_type);
    }
    let Some(left_label) = plan.label_ids[0].map(|id| id as usize) else {
        return generic_point_plan(plan.expression_type);
    };
    let Some(right_label) = plan.label_ids[1].map(|id| id as usize) else {
        return generic_point_plan(plan.expression_type);
    };
    let left_row = payload.exact_row_by_label[left_label];
    let right_row = payload.exact_row_by_label[right_label];
    if left_row == u32::MAX || right_row == u32::MAX {
        return generic_point_plan(plan.expression_type);
    }
    let Some(left_word_offset) = (left_row as usize).checked_mul(payload.words_per_row) else {
        return generic_point_plan(plan.expression_type);
    };
    let Some(right_word_offset) = (right_row as usize).checked_mul(payload.words_per_row) else {
        return generic_point_plan(plan.expression_type);
    };
    let count_swap = counts[left_label] > counts[right_label];
    {
        let summary = &payload.nonzero_words;
        if let (Some(&left_nonzero), Some(&right_nonzero)) = (
            summary_nonzero_words_per_row.get(left_row as usize),
            summary_nonzero_words_per_row.get(right_row as usize),
        ) {
            let summary_swap = left_nonzero > right_nonzero;
            let first_nonzero = if summary_swap {
                right_nonzero
            } else {
                left_nonzero
            } as usize;
            let zero = payload.words_per_row.saturating_sub(first_nonzero);
            if zero.saturating_mul(100)
                >= payload
                    .words_per_row
                    .saturating_mul(EXACT_POINT_SUMMARY_MIN_ZERO_PERCENT)
            {
                let (first_row, second_row, first_word_offset, second_word_offset) = if summary_swap
                {
                    (
                        right_row as usize,
                        left_row as usize,
                        right_word_offset,
                        left_word_offset,
                    )
                } else {
                    (
                        left_row as usize,
                        right_row as usize,
                        left_word_offset,
                        right_word_offset,
                    )
                };
                let Some(first_summary_word_offset) = first_row.checked_mul(summary.words_per_row)
                else {
                    return generic_point_plan(plan.expression_type);
                };
                let Some(second_summary_word_offset) =
                    second_row.checked_mul(summary.words_per_row)
                else {
                    return generic_point_plan(plan.expression_type);
                };
                return PointPlan::ExactAnd2Summary(Box::new(ExactAnd2SummaryPlan {
                    first_word_offset,
                    second_word_offset,
                    first_summary_word_offset,
                    second_summary_word_offset,
                }));
            }
        }
    }
    let (first_word_offset, second_word_offset) = if count_swap {
        (right_word_offset, left_word_offset)
    } else {
        (left_word_offset, right_word_offset)
    };
    PointPlan::ExactAnd2Payload {
        first_word_offset,
        second_word_offset,
    }
}
