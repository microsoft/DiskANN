/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Dense point-query planning and evaluation.

use super::*;

impl EncodedLabelQuery<'_> {
    /// Select a concrete evaluator for repeated point classification.
    ///
    /// Callers should match the returned enum once before entering a hot loop. Each contained
    /// evaluator implements [`diskann::graph::ext::labeled::QueryLabelProvider`]. The owning query
    /// remains the [`diskann::graph::ext::labeled::CandidateLabelProvider`] used for exhaustive
    /// enumeration.
    pub fn point_evaluator(&self) -> EncodedLabelPointEvaluator<'_> {
        match &self.point_plan {
            PointPlan::Dnf(plan) if plan.use_summary => {
                EncodedLabelPointEvaluator::DnfSummary(DnfSummaryEvaluator {
                    num_vectors: self.num_vectors,
                    bits: &self.payload.bits,
                    summary_bits: &self.payload.nonzero_words.bits,
                    clause_offsets: &plan.clause_offsets,
                    payload_offsets: &plan.payload_offsets,
                    summary_offsets: &plan.summary_offsets,
                })
            }
            PointPlan::Dnf(plan) => EncodedLabelPointEvaluator::DnfPayload(DnfPayloadEvaluator {
                num_vectors: self.num_vectors,
                bits: &self.payload.bits,
                clause_offsets: &plan.clause_offsets,
                payload_offsets: &plan.payload_offsets,
                prefetch_word_offset: plan.payload_offsets.first().copied(),
                prefetch_lookahead: self.config.point_prefetch_lookahead(),
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
}

impl diskann::graph::ext::labeled::QueryLabelProvider<u32> for EncodedLabelQuery<'_> {
    #[inline]
    fn is_match(&self, id: u32) -> bool {
        EncodedLabelQuery::is_match(self, id)
    }
}

pub(super) fn compile_point_plan(
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
