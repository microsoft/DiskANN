/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Sparse candidate enumeration and summarized row scanning.

use super::*;
use std::cell::RefCell;

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

impl EncodedLabelQuery<'_> {
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

impl diskann::graph::ext::labeled::CandidateLabelProvider<u32> for EncodedLabelQuery<'_> {
    fn match_upper_bound(&self) -> Option<u64> {
        EncodedLabelQuery::match_upper_bound(self)
    }

    fn visit_candidates(&self, visit: impl FnMut(u32)) {
        self.visit_matches(visit);
    }
}
