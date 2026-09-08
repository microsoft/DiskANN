/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Fixed-capacity candidate rows with cached farthest distances.
//! Updates allocate no storage. NaN and positive infinity are not retained.

use diskann_utils::views::MutMatrixView;

use super::simd::{DistanceBlock, PiPNNSIMDVector};

/// An output slot with no assigned candidate.
pub(super) const UNASSIGNED: u32 = u32::MAX;

/// A local candidate index paired with its ranking distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Candidate {
    pub(super) local_idx: u32,
    pub(super) distance: f32,
}

impl Candidate {
    pub(super) const fn new(local_idx: u32, distance: f32) -> Self {
        Self {
            local_idx,
            distance,
        }
    }

    pub(super) const fn is_assigned(self) -> bool {
        self.local_idx != UNASSIGNED
    }
}

impl Default for Candidate {
    fn default() -> Self {
        Self::new(UNASSIGNED, f32::INFINITY)
    }
}

// Private dispatch tag; callers supply only the result shape.
pub(super) const RUNTIME_WIDTH: usize = 0;

/// Select the top-k implementation once from the row width.
///
/// For widths one to three, `row_mut` exposes a row with a compile-time length.
/// After inlining, the compiler can unroll the insertion loop and remove bounds checks.
/// Other widths use the same insertion loop with a runtime row length.
///
/// Each branch selects a different `TopKRows<K>` type.
/// Use a code block to keep that type through the candidate loops without a callback.
/// Like an ordinary block, `return` and `?` affect the caller.
macro_rules! with_topk_rows {
    ($rows:expr, $worst:expr, |$topks:ident| $body:block) => {{
        let rows = $rows;
        let worst = $worst;
        match rows.ncols() {
            1 => {
                let mut $topks = $crate::graph::pipnn::topk::TopKRows::<1>::new(rows, worst);
                $body
            }
            2 => {
                let mut $topks = $crate::graph::pipnn::topk::TopKRows::<2>::new(rows, worst);
                $body
            }
            3 => {
                let mut $topks = $crate::graph::pipnn::topk::TopKRows::<3>::new(rows, worst);
                $body
            }
            _ => {
                let mut $topks = $crate::graph::pipnn::topk::TopKRows::<
                    { $crate::graph::pipnn::topk::RUNTIME_WIDTH },
                >::new(rows, worst);
                $body
            }
        }
    }};
}
pub(super) use with_topk_rows;

/// Borrowed top-k results and one contiguous cached threshold per row.
///
/// Each row has the same positive capacity. Only this view mutates the rows and
/// their thresholds while borrowed. Callers supply each candidate ID at most once
/// per row; updates do not deduplicate. Equal distances need no fixed tie order.
/// The with_topk_rows macro selects the implementation from the result shape.
pub(super) struct TopKRows<'a, const K: usize> {
    rows: MutMatrixView<'a, Candidate>,
    worst: &'a mut [f32],
}

impl<'a, const K: usize> TopKRows<'a, K> {
    /// Bind and clear caller-owned result and threshold buffers.
    pub(super) fn new(rows: MutMatrixView<'a, Candidate>, worst: &'a mut [f32]) -> Self {
        debug_assert!(rows.ncols() > 0, "top-k row capacity must be positive");
        debug_assert!(
            K == RUNTIME_WIDTH || K == rows.ncols(),
            "top-k compile-time width must match its rows"
        );
        debug_assert_eq!(
            rows.nrows(),
            worst.len(),
            "top-k rows and thresholds must match"
        );
        let mut topk = Self { rows, worst };
        topk.reset();
        topk
    }

    pub(super) fn reset(&mut self) {
        self.rows.as_mut_slice().fill(Candidate::default());
        self.worst.fill(f32::INFINITY);
    }

    pub(super) fn row(&self, row_idx: usize) -> &[Candidate] {
        self.rows.row(row_idx)
    }

    /// Offer one candidate and synchronize the row's cached threshold.
    #[inline(always)]
    pub(super) fn insert(&mut self, row_idx: usize, candidate: Candidate) {
        if candidate.distance < self.worst[row_idx] {
            self.worst[row_idx] = insert_eligible(row_mut::<K>(&mut self.rows, row_idx), candidate);
        }
    }

    /// Offer the block's candidates to one row using their slice-column indexes.
    #[inline(always)]
    pub(super) fn update_one<F: PiPNNSIMDVector>(
        &mut self,
        row_idx: usize,
        block: &DistanceBlock<'_, F>,
    ) {
        match *block {
            DistanceBlock::Scalar { idx, distance } => {
                self.insert(row_idx, Candidate::new(idx as u32, distance));
            }
            DistanceBlock::Simd {
                first_idx,
                values,
                lanes,
            } => {
                let mut worst = self.worst[row_idx];
                let mut eligible = F::active_lanes(values.lt_simd(F::splat(values.arch(), worst)));
                if eligible == 0 {
                    return;
                }
                let row = row_mut::<K>(&mut self.rows, row_idx);
                while eligible != 0 {
                    let lane = eligible.trailing_zeros() as usize;
                    eligible &= eligible - 1;
                    // Earlier candidates in this same block can lower the threshold.
                    if lanes[lane] < worst {
                        worst = insert_eligible(
                            row,
                            Candidate::new((first_idx + lane) as u32, lanes[lane]),
                        );
                    }
                }
                self.worst[row_idx] = worst;
            }
        }
    }

    /// Offer one candidate to each row identified by the block's slice-column indexes.
    #[inline(always)]
    pub(super) fn update_many<F: PiPNNSIMDVector>(
        &mut self,
        candidate_idx: u32,
        block: &DistanceBlock<'_, F>,
    ) {
        match *block {
            DistanceBlock::Scalar { idx, distance } => {
                self.insert(idx, Candidate::new(candidate_idx, distance));
            }
            DistanceBlock::Simd {
                first_idx,
                values,
                lanes,
            } => {
                let thresholds = &mut self.worst[first_idx..][..F::LANES];
                // SAFETY: the slice above has one threshold for every SIMD lane.
                let worst = unsafe { F::load_simd(values.arch(), thresholds.as_ptr()) };
                let mut eligible = F::active_lanes(values.lt_simd(worst));
                while eligible != 0 {
                    let lane = eligible.trailing_zeros() as usize;
                    eligible &= eligible - 1;
                    // Each lane updates a different row once; its threshold is still current.
                    thresholds[lane] = insert_eligible(
                        row_mut::<K>(&mut self.rows, first_idx + lane),
                        Candidate::new(candidate_idx, lanes[lane]),
                    );
                }
            }
        }
    }
}

/// Preserve fixed row lengths for inlined candidate insertion.
#[inline(always)]
fn row_mut<'a, const K: usize>(
    rows: &'a mut MutMatrixView<'_, Candidate>,
    row_idx: usize,
) -> &'a mut [Candidate] {
    if K == RUNTIME_WIDTH {
        rows.row_mut(row_idx)
    } else {
        let (rows, _) = rows.as_mut_slice().as_chunks_mut::<K>();
        &mut rows[row_idx]
    }
}

/// Insert after the threshold check; keep candidates in nearest-first order.
#[inline(always)]
fn insert_eligible(row: &mut [Candidate], candidate: Candidate) -> f32 {
    let last = row.len() - 1;
    let mut slot = last;
    while slot > 0 && candidate.distance < row[slot - 1].distance {
        row[slot] = row[slot - 1];
        slot -= 1;
    }
    row[slot] = candidate;
    row[last].distance
}

#[cfg(test)]
mod tests {
    use super::*;

    mod insert_eligible_tests {
        use super::*;
        use rstest::rstest;

        #[rstest]
        #[case::singleton(&[(1, 4.0)], (2, 2.0), &[(2, 2.0)])]
        #[case::no_shift(&[(1, 1.0), (2, 2.0), (3, 4.0)], (4, 3.0), &[(1, 1.0), (2, 2.0), (4, 3.0)])]
        #[case::one_shift(&[(1, 1.0), (2, 2.0), (3, 4.0)], (4, 1.5), &[(1, 1.0), (4, 1.5), (2, 2.0)])]
        #[case::multiple_shifts(&[(1, 1.0), (2, 2.0), (3, 3.0), (4, 5.0)], (5, 0.5), &[(5, 0.5), (1, 1.0), (2, 2.0), (3, 3.0)])]
        #[case::negative_infinity(&[(1, -1.0), (2, 1.0), (UNASSIGNED, f32::INFINITY)], (3, f32::NEG_INFINITY), &[(3, f32::NEG_INFINITY), (1, -1.0), (2, 1.0)])]
        #[case::positive_zero(&[(1, 1.0)], (2, 0.0), &[(2, 0.0)])]
        #[case::negative_zero(&[(1, 1.0)], (2, -0.0), &[(2, -0.0)])]
        fn insertion_keeps_nearest_candidates_and_returns_threshold(
            #[case] initial: &[(u32, f32)],
            #[case] candidate: (u32, f32),
            #[case] expected: &[(u32, f32)],
        ) {
            // Given
            let expected_threshold = expected.last().unwrap().1.to_bits();
            let expected: Vec<_> = expected
                .iter()
                .map(|&(idx, distance)| (idx, distance.to_bits()))
                .collect();
            let mut row: Vec<_> = initial
                .iter()
                .map(|&(idx, distance)| Candidate::new(idx, distance))
                .collect();

            // When
            let threshold = insert_eligible(&mut row, Candidate::new(candidate.0, candidate.1));

            // Then
            let actual: Vec<_> = row
                .iter()
                .map(|candidate| (candidate.local_idx, candidate.distance.to_bits()))
                .collect();
            assert_eq!(actual, expected);
            assert_eq!(threshold.to_bits(), expected_threshold);
        }
    }

    #[test]
    fn reset_clears_rows_and_thresholds() {
        // Given
        let mut output = [Candidate::new(10, -1.0), Candidate::new(11, 2.0)];
        let mut worst = [-1.0, 2.0];
        let mut topk = TopKRows::<1> {
            rows: MutMatrixView::try_from(&mut output[..], 2, 1).unwrap(),
            worst: &mut worst,
        };

        // When
        topk.reset();

        // Then
        assert_eq!(topk.rows.as_slice(), [Candidate::default(); 2]);
        assert_eq!(topk.worst, [f32::INFINITY; 2]);
    }

    mod update_tests {
        use super::*;
        use crate::graph::pipnn::simd::distance_blocks;
        use rstest::rstest;

        #[test]
        fn one_row_rechecks_the_threshold_after_each_insert() {
            let mut distances = [f32::INFINITY; 16];
            distances[0] = 1.0;
            distances[1] = 9.0;
            let mut output = [Candidate::default(); 1];
            let mut worst = [f32::INFINITY];
            let mut topk = TopKRows::<RUNTIME_WIDTH>::new(
                MutMatrixView::row_vector(&mut output[..]),
                &mut worst,
            );

            for block in distance_blocks(diskann_wide::ARCH, &distances) {
                topk.update_one(0, &block);
            }

            assert_eq!(topk.row(0), [Candidate::new(0, 1.0)]);
            assert_eq!(topk.worst, [1.0]);
        }

        #[test]
        fn many_rows_use_their_own_threshold_and_distance() {
            let mut distances = [2.0; 16];
            distances[1] = 4.0;
            distances[2] = 6.0;
            let mut output = [Candidate::default(); 16];
            let mut worst = [f32::INFINITY; 16];
            let mut topk = TopKRows::<RUNTIME_WIDTH>::new(
                MutMatrixView::try_from(&mut output[..], 16, 1).unwrap(),
                &mut worst,
            );
            topk.insert(0, Candidate::new(80, 1.0));
            topk.insert(1, Candidate::new(81, 5.0));
            topk.insert(2, Candidate::new(82, 5.0));

            for block in distance_blocks(diskann_wide::ARCH, &distances) {
                topk.update_many(42, &block);
            }

            assert_eq!(topk.row(0), [Candidate::new(80, 1.0)]);
            assert_eq!(topk.row(1), [Candidate::new(42, 4.0)]);
            assert_eq!(topk.row(2), [Candidate::new(82, 5.0)]);
            assert!(
                topk.rows
                    .row_iter()
                    .skip(3)
                    .all(|row| row == [Candidate::new(42, 2.0)])
            );
            assert_eq!(topk.worst[..3], [1.0, 4.0, 5.0]);
            assert!(topk.worst[3..].iter().all(|&worst| worst == 2.0));
        }

        #[test]
        fn one_and_many_updates_have_independent_acceptance() {
            let distances = [2.0; 16];
            let mut output = [Candidate::default(); 17];
            let mut worst = [f32::INFINITY; 17];
            let mut topk = TopKRows::<RUNTIME_WIDTH>::new(
                MutMatrixView::try_from(&mut output[..], 17, 1).unwrap(),
                &mut worst,
            );
            topk.insert(16, Candidate::new(70, 1.0));

            for block in distance_blocks(diskann_wide::ARCH, &distances) {
                topk.update_one(16, &block);
                topk.update_many(16, &block);
            }

            assert_eq!(topk.row(16), [Candidate::new(70, 1.0)]);
            assert!(
                topk.rows
                    .row_iter()
                    .take(16)
                    .all(|row| row == [Candidate::new(16, 2.0)])
            );
        }

        #[rstest]
        #[case::nan(f32::NAN)]
        #[case::infinity(f32::INFINITY)]
        fn non_rankable_candidates_leave_reset_rows_unassigned(#[case] distance: f32) {
            let distances = [distance; 16];
            let mut output = [Candidate::new(70, 1.0); 16];
            let mut worst = [1.0; 16];
            let mut topk = TopKRows::<RUNTIME_WIDTH>::new(
                MutMatrixView::try_from(&mut output[..], 16, 1).unwrap(),
                &mut worst,
            );

            for block in distance_blocks(diskann_wide::ARCH, &distances) {
                topk.update_one(0, &block);
                topk.update_many(42, &block);
            }
            topk.insert(0, Candidate::new(43, distance));

            assert!(
                topk.rows
                    .as_slice()
                    .iter()
                    .all(|&candidate| candidate == Candidate::default())
            );
            assert!(topk.worst.iter().all(|&worst| worst == f32::INFINITY));
        }

        #[rstest]
        #[case::empty(0)]
        #[case::lane_minus_one(15)]
        #[case::one_lane(16)]
        #[case::lane_with_tail(17)]
        #[case::two_lanes_with_tail(33)]
        #[trace]
        fn shared_block_updates_match_runtime_scalar_insertion(
            #[case] count: usize,
            #[values(1, 2, 3, 4)] width: usize,
        ) {
            // Given
            let rows = count + 1;
            let source_idx = count;
            let distances: Vec<_> = (0..count)
                .map(|idx| {
                    let distance = idx as f32 + 0.25;
                    if idx % 2 == 0 { -distance } else { distance }
                })
                .collect();
            let mut actual_rows = vec![Candidate::default(); rows * width];
            let mut expected_rows = actual_rows.clone();
            let mut actual_worst = vec![f32::INFINITY; rows];
            let mut expected_worst = actual_worst.clone();
            let mut expected = TopKRows::<RUNTIME_WIDTH>::new(
                MutMatrixView::try_from(expected_rows.as_mut_slice(), rows, width).unwrap(),
                &mut expected_worst,
            );
            expected.insert(source_idx, Candidate::new(90, 0.0));
            expected.insert(0, Candidate::new(91, -4.0));
            for (idx, &distance) in distances.iter().enumerate() {
                expected.insert(source_idx, Candidate::new(idx as u32, distance));
                expected.insert(idx, Candidate::new(42, distance));
            }

            with_topk_rows!(
                MutMatrixView::try_from(actual_rows.as_mut_slice(), rows, width).unwrap(),
                &mut actual_worst[..],
                |actual| {
                    actual.insert(source_idx, Candidate::new(90, 0.0));
                    actual.insert(0, Candidate::new(91, -4.0));

                    // When
                    for block in distance_blocks(diskann_wide::ARCH, &distances) {
                        actual.update_one(source_idx, &block);
                        actual.update_many(42, &block);
                    }

                    // Then
                    assert_eq!(actual.rows.as_slice(), expected.rows.as_slice());
                    assert_eq!(actual.worst, expected.worst);
                }
            );
        }
    }
}
