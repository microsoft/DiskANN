/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Top-k updates from caller-selected distance slices.
//! NaN and positive infinity are not retained.

use diskann_utils::views::MutMatrixView;
use diskann_wide::{SIMDPartialOrd, SIMDVector};

use super::simd::Simd;

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

/// One borrowed result row whose length is its capacity.
pub(super) trait Ranker: AsRef<[Candidate]> + AsMut<[Candidate]> + Sized {
    /// Insert a candidate below this non-empty row's limit and return the new limit.
    fn insert(&mut self, candidate: Candidate) -> f32;

    /// Replace this row with the nearest input positions.
    ///
    /// Equal distances need no fixed tie order. Empty rows retain no candidates.
    #[inline(always)]
    fn select_topk<A: Simd>(&mut self, arch: A, distances: &[f32]) {
        arch.run2(
            #[inline(always)]
            move |distances: &[f32], ranker: &mut Self| {
                if ranker.as_ref().is_empty() {
                    return;
                }
                ranker.as_mut().fill(Candidate::default());
                distance_blocks(arch, distances).fold(
                    f32::INFINITY,
                    #[inline(always)]
                    |limit, block| block.update_topk_from_candidates(ranker, limit),
                );
            },
            distances,
            self,
        );
    }
}

impl Ranker for &mut [Candidate] {
    #[inline(always)]
    fn insert(&mut self, candidate: Candidate) -> f32 {
        insert_sorted(self, candidate)
    }
}

impl<const K: usize> Ranker for &mut [Candidate; K] {
    #[inline(always)]
    fn insert(&mut self, candidate: Candidate) -> f32 {
        insert_sorted(*self, candidate)
    }
}

/// A batch operation over one bound result row.
pub(super) trait TopKVisitor {
    fn visit<R: Ranker>(self, ranker: R);
}

/// Bind one result row and specialize its capacity once for the visitor.
///
/// Specialize capacities 1, 2, 3, 8, and 10; other capacities use the slice path.
/// Typical partition K is 10 or 3, and leaf K is 3 or 2.
#[inline]
#[expect(
    clippy::unwrap_used,
    reason = "each match arm proves the candidate slice has the array length"
)]
pub(super) fn with_topk<V: TopKVisitor>(candidates: &mut [Candidate], visitor: V) {
    // Each arm checks this same slice's length, so conversion to its array cannot fail.
    match candidates.len() {
        1 => visitor.visit::<&mut [_; 1]>(candidates.try_into().unwrap()),
        2 => visitor.visit::<&mut [_; 2]>(candidates.try_into().unwrap()),
        3 => visitor.visit::<&mut [_; 3]>(candidates.try_into().unwrap()),
        8 => visitor.visit::<&mut [_; 8]>(candidates.try_into().unwrap()),
        10 => visitor.visit::<&mut [_; 10]>(candidates.try_into().unwrap()),
        _ => visitor.visit(candidates),
    }
}

/// Access complete result rows with their static or runtime capacity.
pub(super) trait BatchRanker {
    type Ranker<'a>: Ranker
    where
        Self: 'a;

    fn ranker(&mut self, index: usize) -> Option<Self::Ranker<'_>>;
}

/// Candidate rows and their limits stay borrowed together throughout a pair scan.
pub(super) struct Batch<'a, Rows> {
    candidates: Rows,
    thresholds: &'a mut [f32],
}

impl<'a> Batch<'a, MutMatrixView<'a, Candidate>> {
    fn new(mut candidates: MutMatrixView<'a, Candidate>, thresholds: &'a mut Vec<f32>) -> Self {
        let rows = candidates.nrows();
        candidates.as_mut_slice().fill(Candidate::default());
        thresholds.truncate(rows);
        thresholds.fill(f32::INFINITY);
        thresholds.resize(rows, f32::INFINITY);
        Self {
            candidates,
            thresholds,
        }
    }
}

impl<Rows> Batch<'_, Rows> {
    fn nrows(&self) -> usize {
        self.thresholds.len()
    }
}

impl<const K: usize> BatchRanker for Batch<'_, &mut [[Candidate; K]]> {
    type Ranker<'a>
        = &'a mut [Candidate; K]
    where
        Self: 'a;

    #[inline(always)]
    fn ranker(&mut self, index: usize) -> Option<Self::Ranker<'_>> {
        self.candidates.get_mut(index)
    }
}

impl BatchRanker for Batch<'_, MutMatrixView<'_, Candidate>> {
    type Ranker<'a>
        = &'a mut [Candidate]
    where
        Self: 'a;

    #[inline(always)]
    fn ranker(&mut self, index: usize) -> Option<Self::Ranker<'_>> {
        if index < self.nrows() {
            Some(self.candidates.row_mut(index))
        } else {
            None
        }
    }
}

/// A batch operation over initialized pair-scan storage.
pub(super) trait BatchVisitor {
    fn visit<'a, Rows>(self, batch: Batch<'a, Rows>)
    where
        Batch<'a, Rows>: BatchRanker;
}

/// Initialize a result matrix and specialize its row representation once for the visitor.
#[inline]
pub(super) fn with_batch<V: BatchVisitor>(
    candidates: MutMatrixView<'_, Candidate>,
    thresholds: &mut Vec<f32>,
    visitor: V,
) {
    let Batch {
        candidates,
        thresholds,
    } = Batch::new(candidates, thresholds);
    // The matrix width selects K, so its storage contains only complete K-element rows.
    match candidates.ncols() {
        1 => visitor.visit(Batch {
            candidates: candidates.into_inner().as_chunks_mut::<1>().0,
            thresholds,
        }),
        2 => visitor.visit(Batch {
            candidates: candidates.into_inner().as_chunks_mut::<2>().0,
            thresholds,
        }),
        3 => visitor.visit(Batch {
            candidates: candidates.into_inner().as_chunks_mut::<3>().0,
            thresholds,
        }),
        8 => visitor.visit(Batch {
            candidates: candidates.into_inner().as_chunks_mut::<8>().0,
            thresholds,
        }),
        10 => visitor.visit(Batch {
            candidates: candidates.into_inner().as_chunks_mut::<10>().0,
            thresholds,
        }),
        _ => visitor.visit(Batch {
            candidates,
            thresholds,
        }),
    }
}

impl<'a, Rows> Batch<'a, Rows>
where
    Self: BatchRanker,
{
    /// Offer one point's distances to all preceding points to both endpoints' results.
    ///
    /// The source point index is `distances.len()`: each entry describes its pair
    /// with the earlier point at that position. Supply each pair once; earlier results are preserved.
    /// Equal distances need no fixed tie order.
    ///
    /// # Panics
    ///
    /// Panics if `distances.len()` is outside the batch.
    #[inline]
    #[expect(
        clippy::unwrap_used,
        reason = "the first ranker lookup rejects an invalid point before scanning"
    )]
    pub(super) fn update_dual_topk<A: Simd>(&mut self, arch: A, distances: &[f32]) {
        let point_idx = distances.len();
        // ranker rejects an out-of-bounds point before any results are changed.
        if self.ranker(point_idx).unwrap().as_ref().is_empty() {
            return;
        }
        arch.run2(
            #[inline(always)]
            move |distances: &[f32], batch: &mut Self| {
                let limit = distance_blocks(arch, distances).fold(
                    batch.thresholds[point_idx],
                    #[inline(always)]
                    |limit, block| {
                        // End the source row borrow before offering this block to earlier rows.
                        let limit = {
                            let mut nearest = batch.ranker(point_idx).unwrap();
                            block.update_topk_from_candidates(&mut nearest, limit)
                        };
                        block.update_topks_with_point(batch, point_idx as u32);
                        limit
                    },
                );
                batch.thresholds[point_idx] = limit;
            },
            distances,
            self,
        );
    }
}

/// A complete SIMD group or one tail distance. This never leaves the TopK module.
/// Both update directions share the loaded vector; scalar reads borrow the input.
#[derive(Clone, Copy)]
enum DistanceBlock<'a, A: Simd> {
    Simd {
        first_candidate: usize,
        values: A::Vector,
        distances: &'a [f32],
    },
    Scalar {
        candidate_idx: usize,
        distance: f32,
    },
}

/// Keep all distance loads and tail indexing inside TopK's architecture scope.
#[inline(always)]
fn distance_blocks<A: Simd>(
    arch: A,
    distances: &[f32],
) -> impl Iterator<Item = DistanceBlock<'_, A>> {
    let simd_end = distances.len() - distances.len() % A::Vector::LANES;
    distances[..simd_end]
        .chunks_exact(A::Vector::LANES)
        .enumerate()
        .map(move |(group, distances)| DistanceBlock::Simd {
            first_candidate: group * A::Vector::LANES,
            // SAFETY: chunks_exact supplies one distance for every SIMD lane.
            values: unsafe { A::Vector::load_simd(arch, distances.as_ptr()) },
            distances,
        })
        .chain(
            distances[simd_end..]
                .iter()
                .enumerate()
                .map(move |(tail, &distance)| DistanceBlock::Scalar {
                    candidate_idx: simd_end + tail,
                    distance,
                }),
        )
}

impl<A: Simd> DistanceBlock<'_, A> {
    /// Update one top-k result from this block's candidates.
    ///
    /// Each candidate uses its index in the full distance row as its local ID.
    /// Partition ranking uses leader IDs. Leaf ranking uses IDs of earlier points in the leaf.
    ///
    /// Compare candidates against the complete result row without clearing its slots.
    /// The distance limit is the last slot's distance, or positive infinity while slots remain empty.
    /// Each insertion can lower this limit. Return the updated limit for the next block.
    #[inline(always)]
    fn update_topk_from_candidates<R: Ranker>(self, nearest: &mut R, mut max_distance: f32) -> f32 {
        match self {
            Self::Scalar {
                candidate_idx,
                distance,
            } => {
                if distance < max_distance {
                    max_distance = nearest.insert(Candidate::new(candidate_idx as u32, distance));
                }
            }
            Self::Simd {
                first_candidate,
                values,
                distances,
            } => {
                // An unfilled result admits all rankable values. Scan them in
                // order without constructing and traversing an all-eligible mask.
                if max_distance == f32::INFINITY {
                    for (lane, &distance) in distances.iter().enumerate() {
                        if distance < max_distance {
                            max_distance = nearest
                                .insert(Candidate::new((first_candidate + lane) as u32, distance));
                        }
                    }
                } else {
                    let mut eligible = A::active_lanes(
                        values.lt_simd(A::Vector::splat(values.arch(), max_distance)),
                    );
                    while eligible != 0 {
                        let lane = eligible.trailing_zeros() as usize;
                        eligible &= eligible - 1;
                        // Earlier insertions in this block can lower the limit.
                        if distances[lane] < max_distance {
                            max_distance = nearest.insert(Candidate::new(
                                (first_candidate + lane) as u32,
                                distances[lane],
                            ));
                        }
                    }
                }
            }
        }
        max_distance
    }

    /// Offer one leaf point to the top-k results of earlier leaf points.
    ///
    /// This block stores distances from `point_idx` to those earlier points.
    /// Each candidate's local ID selects its result row in the full leaf, not its SIMD lane.
    /// If that row accepts the pair, insert `point_idx` with the pair's distance.
    ///
    /// Only the earlier rows identified by this block can change.
    /// Their candidates are retained unless a nearer point displaces them.
    ///
    /// Each row compares the pair distance against its own limit.
    /// Its decision does not depend on the current point's top-k result.
    /// Update the row's limit after an insertion.
    #[inline(always)]
    #[expect(
        clippy::unwrap_used,
        reason = "all block positions precede the checked source point"
    )]
    fn update_topks_with_point<'a, Rows>(self, batch: &mut Batch<'a, Rows>, point_idx: u32)
    where
        Batch<'a, Rows>: BatchRanker,
    {
        match self {
            Self::Scalar {
                candidate_idx,
                distance,
            } => {
                if distance < batch.thresholds[candidate_idx] {
                    // Block positions are less than the source index checked by update_dual_topk.
                    let limit = batch
                        .ranker(candidate_idx)
                        .unwrap()
                        .insert(Candidate::new(point_idx, distance));
                    batch.thresholds[candidate_idx] = limit;
                }
            }
            Self::Simd {
                first_candidate,
                values,
                distances,
            } => {
                let limits = &batch.thresholds[first_candidate..][..A::Vector::LANES];
                // SAFETY: the slice above has one distance limit per SIMD lane.
                let limit_values = unsafe { A::Vector::load_simd(values.arch(), limits.as_ptr()) };
                let mut eligible = A::active_lanes(values.lt_simd(limit_values));
                while eligible != 0 {
                    let lane = eligible.trailing_zeros() as usize;
                    eligible &= eligible - 1;
                    // Each lane updates a different result; its limit is current.
                    let index = first_candidate + lane;
                    // Every SIMD lane describes an earlier, in-bounds point.
                    let limit = batch
                        .ranker(index)
                        .unwrap()
                        .insert(Candidate::new(point_idx, distances[lane]));
                    batch.thresholds[index] = limit;
                }
            }
        }
    }
}

/// Insert an eligible candidate in nearest-first order and return the new distance limit.
/// `nearest` contains exactly one non-empty result row.
/// The caller must check the candidate against the current limit before insertion.
#[inline(always)]
fn insert_sorted(nearest: &mut [Candidate], candidate: Candidate) -> f32 {
    let last = nearest.len() - 1;
    let mut slot = last;
    while slot > 0 && candidate.distance < nearest[slot - 1].distance {
        nearest[slot] = nearest[slot - 1];
        slot -= 1;
    }
    nearest[slot] = candidate;
    nearest[last].distance
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_wide::{ARCH, arch::Current};
    use rstest::rstest;

    const EMPTY: Candidate = Candidate::new(UNASSIGNED, f32::INFINITY);
    const LANES: usize = <<Current as Simd>::Vector as SIMDVector>::LANES;

    #[rstest]
    #[case::first_batch(0, 3)]
    #[case::larger_batch(2, 5)]
    #[case::smaller_batch(5, 2)]
    #[case::same_size_batch(3, 3)]
    #[case::empty_batch(3, 0)]
    fn starting_a_batch_clears_results_and_sizes_its_limits(
        #[case] previous_rows: usize,
        #[case] rows: usize,
    ) {
        // Given: reusable storage contains assigned candidates and finite limits.
        let mut output = vec![[Candidate::new(0, -12.0), Candidate::new(1, -6.0)]; rows];
        let mut limits = vec![-6.0; previous_rows];

        // When: prepare storage for a new batch.
        Batch::new(
            MutMatrixView::try_from(output.as_flattened_mut(), rows, 2).unwrap(),
            &mut limits,
        );

        // Then: no old candidate or limit belongs to the new batch.
        assert_eq!(output, vec![[EMPTY; 2]; rows]);
        assert_eq!(limits, vec![f32::INFINITY; rows]);
    }

    #[test]
    fn batch_rankers_borrow_one_complete_row_without_touching_adjacent_rows() {
        struct UpdateMiddleRow;

        impl BatchVisitor for UpdateMiddleRow {
            fn visit<'a, Rows>(self, mut batch: Batch<'a, Rows>)
            where
                Batch<'a, Rows>: BatchRanker,
            {
                assert!(batch.ranker(3).is_none());
                batch
                    .ranker(1)
                    .unwrap()
                    .as_mut()
                    .fill(Candidate::new(42, -5.0));
            }
        }

        // Exercise array rows and runtime rows through the production dispatcher.
        for capacity in [2, 4] {
            let mut candidates = vec![Candidate::new(9, -12.0); 3 * capacity];
            let mut thresholds = vec![-12.0; 5];

            with_batch(
                MutMatrixView::try_from(candidates.as_mut_slice(), 3, capacity).unwrap(),
                &mut thresholds,
                UpdateMiddleRow,
            );

            assert_eq!(&candidates[..capacity], vec![EMPTY; capacity]);
            assert_eq!(
                &candidates[capacity..2 * capacity],
                vec![Candidate::new(42, -5.0); capacity]
            );
            assert_eq!(&candidates[2 * capacity..], vec![EMPTY; capacity]);
        }
    }

    #[test]
    #[should_panic]
    fn a_pair_scan_rejects_a_point_outside_the_batch() {
        let mut candidates = [EMPTY; 3];
        let mut limits = Vec::new();
        let output = MutMatrixView::try_from(&mut candidates[..], 3, 1).unwrap();
        let mut batch = Batch::new(output, &mut limits);

        batch.update_dual_topk(ARCH, &[3.0, 4.0, 5.0]);
    }

    #[rstest]
    #[case::empty_input(&[], [EMPTY; 3])]
    #[case::no_rankable_values(&[f32::INFINITY, f32::NAN], [EMPTY; 3])]
    #[case::fewer_candidates_than_slots(
        &[11.0, -4.0],
        [Candidate::new(1, -4.0), Candidate::new(0, 11.0), EMPTY],
    )]
    #[case::already_nearest_first(
        &[-7.0, 2.0, 8.0, 13.0],
        [Candidate::new(0, -7.0), Candidate::new(1, 2.0), Candidate::new(2, 8.0)],
    )]
    #[case::replacements_and_reordering(
        &[8.0, -7.0, 13.0, 2.0],
        [Candidate::new(1, -7.0), Candidate::new(3, 2.0), Candidate::new(0, 8.0)],
    )]
    #[case::unrankable_values_leave_gaps_in_ids(
        &[f32::NAN, 6.0, f32::INFINITY, -2.0, 1.0],
        [Candidate::new(3, -2.0), Candidate::new(4, 1.0), Candidate::new(1, 6.0)],
    )]
    fn selection_returns_nearest_input_positions_and_clears_unused_slots(
        #[case] distances: &[f32],
        #[case] expected: [Candidate; 3],
    ) {
        // Given: output still holds results from a different input.
        let mut output = [
            Candidate::new(2, -20.0),
            Candidate::new(4, -10.0),
            Candidate::new(9, -1.0),
        ];

        (&mut output).select_topk(ARCH, distances);

        assert_eq!(output, expected);
    }

    #[test]
    fn reusing_output_starts_a_selection_with_a_fresh_distance_limit() {
        let mut output = [EMPTY; 2];
        (&mut output).select_topk(ARCH, &[-20.0, -10.0]);
        assert_eq!(output, [Candidate::new(0, -20.0), Candidate::new(1, -10.0)]);

        // Every new distance exceeds the previous selection's limit.
        (&mut output).select_topk(ARCH, &[16.0, 3.0, 9.0]);

        assert_eq!(output, [Candidate::new(1, 3.0), Candidate::new(2, 9.0)]);
    }

    #[test]
    fn batch_selection_matches_a_full_sort() {
        struct SelectRow<'a> {
            distances: &'a [f32],
        }

        impl TopKVisitor for SelectRow<'_> {
            fn visit<R: Ranker>(self, mut ranker: R) {
                ranker.select_topk(ARCH, self.distances);
            }
        }

        for count in [
            0,
            1,
            LANES - 1,
            LANES,
            LANES + 1,
            2 * LANES,
            2 * LANES + 3,
            3 * LANES + 2,
        ] {
            for capacity in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, LANES + 1] {
                // Each pair offers a nearer score before a farther one. Later pairs improve
                // on earlier pairs, so a full result must keep lowering its cutoff mid-vector.
                let mut distances: Vec<_> = (0..count).map(|i| -(i as f32) - 1.0).collect();
                for pair in distances.chunks_exact_mut(2) {
                    pair.swap(0, 1);
                }

                // Sorting the whole input is independent of TopK's insertion and cutoff logic.
                let mut expected: Vec<_> = distances
                    .iter()
                    .enumerate()
                    .map(|(index, &distance)| Candidate::new(index as u32, distance))
                    .collect();
                expected.sort_by(|left, right| left.distance.total_cmp(&right.distance));
                expected.truncate(capacity);
                expected.resize(capacity, EMPTY);
                let mut output = vec![EMPTY; capacity];
                with_topk(
                    &mut output,
                    SelectRow {
                        distances: &distances,
                    },
                );

                assert_eq!(output, expected, "count={count}, capacity={capacity}");
            }
        }
    }

    #[test]
    fn selection_returns_distinct_candidates_when_distances_tie() {
        let best_index = 2 * LANES + 2;
        let mut distances = vec![8.0; best_index + 1];
        distances[best_index] = -2.0;
        let mut output = [EMPTY; 4];

        (&mut output).select_topk(ARCH, &distances);

        assert_eq!(output[0], Candidate::new(best_index as u32, -2.0));
        // Any three distinct input positions with distance 8 are valid; tie order is free.
        let ties = &output[1..];
        assert!(
            ties.iter()
                .all(|c| c.local_idx < best_index as u32 && c.distance == 8.0),
            "invalid tied candidates: {ties:?}"
        );
        let mut ids: Vec<_> = ties.iter().map(|c| c.local_idx).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 3, "a candidate must not occupy multiple slots");
    }

    #[rstest]
    #[case::negative_zero(-0.0)]
    #[case::positive_zero(0.0)]
    #[case::lowest_finite(f32::MIN)]
    #[case::highest_finite(f32::MAX)]
    #[case::negative_infinity(f32::NEG_INFINITY)]
    fn a_rankable_value_keeps_its_input_id_and_float_bits(
        #[case] distance: f32,
        #[values(0, LANES - 1, LANES, 2 * LANES + 1)] index: usize,
    ) {
        let mut distances = vec![f32::NAN; 2 * LANES + 2];
        distances[index] = distance;
        let mut output = [EMPTY; 2];

        output.as_mut_slice().select_topk(ARCH, &distances);

        assert_eq!(output[0].local_idx, index as u32);
        assert_eq!(output[0].distance.to_bits(), distance.to_bits());
        assert_eq!(output[1], EMPTY);
    }

    #[test]
    fn successive_pair_rows_accumulate_each_points_nearest_neighbors() {
        fn check<'a, Rows>(mut batch: Batch<'a, Rows>)
        where
            Batch<'a, Rows>: BatchRanker,
        {
            // Given: pairs 0-1=9, 0-2=2, and 1-2=7; point 3 has no offered pairs.
            batch.update_dual_topk(ARCH, &[9.0]);
            batch.update_dual_topk(ARCH, &[2.0, 7.0]);
            let before_last_point = [
                [Candidate::new(2, 2.0), Candidate::new(1, 9.0)],
                [Candidate::new(2, 7.0), Candidate::new(0, 9.0)],
                [Candidate::new(0, 2.0), Candidate::new(1, 7.0)],
                [EMPTY; 2],
            ];
            for (index, expected) in before_last_point.iter().enumerate() {
                assert_eq!(batch.ranker(index).unwrap().as_ref(), expected);
            }
            assert_eq!(batch.thresholds, [9.0, 9.0, 7.0, f32::INFINITY]);

            // When: point 3 is at distances 6, 1, and 4 from points 0, 1, and 2.
            batch.update_dual_topk(ARCH, &[6.0, 1.0, 4.0]);

            // Then: each endpoint keeps its own nearest two, with its farthest retained limit.
            let expected = [
                [Candidate::new(2, 2.0), Candidate::new(3, 6.0)],
                [Candidate::new(3, 1.0), Candidate::new(2, 7.0)],
                [Candidate::new(0, 2.0), Candidate::new(3, 4.0)],
                [Candidate::new(1, 1.0), Candidate::new(2, 4.0)],
            ];
            for (index, expected) in expected.iter().enumerate() {
                assert_eq!(batch.ranker(index).unwrap().as_ref(), expected);
            }
            assert_eq!(batch.thresholds, [6.0, 7.0, 4.0, 4.0]);
        }

        let mut fixed = [[EMPTY; 2]; 4];
        let mut fixed_limits = Vec::new();
        fixed_limits.resize(fixed.len(), f32::INFINITY);
        check(Batch {
            candidates: fixed.as_mut_slice(),
            thresholds: &mut fixed_limits,
        });

        let mut dynamic = [EMPTY; 8];
        let mut dynamic_limits = Vec::new();
        let rows = MutMatrixView::try_from(&mut dynamic[..], 4, 2).unwrap();
        check(Batch::new(rows, &mut dynamic_limits));
    }

    #[test]
    fn pair_scans_match_full_row_sort_across_capacities_and_lengths() {
        struct ScanPairs<'a> {
            distances: &'a [Vec<f32>],
        }

        impl BatchVisitor for ScanPairs<'_> {
            fn visit<'a, Rows>(self, mut batch: Batch<'a, Rows>)
            where
                Batch<'a, Rows>: BatchRanker,
            {
                for (point, row) in self.distances.iter().enumerate().skip(1) {
                    batch.update_dual_topk(ARCH, &row[..point]);
                }
            }
        }

        for point_count in [
            LANES,
            LANES + 1,
            LANES + 2,
            2 * LANES,
            2 * LANES + 1,
            2 * LANES + 2,
        ] {
            for capacity in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17] {
                // Given: XOR gives symmetric distances with distinct, exact scores in each row.
                // The final pair row has point_count - 1 distances, straddling vector boundaries.
                let distances: Vec<Vec<f32>> = (0..point_count)
                    .map(|point| {
                        (0..point_count)
                            .map(|other| (point ^ other) as f32)
                            .collect()
                    })
                    .collect();
                let mut output = vec![EMPTY; point_count * capacity];
                let mut limits = Vec::new();
                let mut rows =
                    MutMatrixView::try_from(output.as_mut_slice(), point_count, capacity).unwrap();

                // When: offer every non-self pair once, through the production width dispatch.
                with_batch(
                    rows.as_mut_view(),
                    &mut limits,
                    ScanPairs {
                        distances: &distances,
                    },
                );

                // Then: independently sort each complete row, excluding only the point itself.
                for (point, row) in distances.iter().enumerate() {
                    let mut expected: Vec<_> = row
                        .iter()
                        .enumerate()
                        .filter(|(other, _)| *other != point)
                        .map(|(other, &distance)| Candidate::new(other as u32, distance))
                        .collect();
                    expected.sort_by(|left, right| left.distance.total_cmp(&right.distance));
                    expected.truncate(capacity);
                    expected.resize(capacity, EMPTY);

                    assert_eq!(
                        rows.row(point),
                        expected,
                        "point={point}, point_count={point_count}, capacity={capacity}"
                    );
                    assert_eq!(
                        limits[point],
                        expected[capacity - 1].distance,
                        "limit for point={point}, point_count={point_count}, capacity={capacity}"
                    );
                }
            }
        }
    }

    #[rstest]
    fn a_source_rejection_does_not_prevent_the_target_from_accepting(
        #[values(1, LANES - 1, LANES, 2 * LANES + 1)] target: usize,
    ) {
        let source = 2 * LANES + 2;
        let mut output = vec![EMPTY; source + 1];
        let mut limits = Vec::new();
        let mut rows = MutMatrixView::try_from(output.as_mut_slice(), source + 1, 1).unwrap();
        let mut batch = Batch::new(rows.as_mut_view(), &mut limits);
        let mut distances = vec![f32::INFINITY; source];
        distances[0] = -6.0;
        distances[target] = 4.0;

        // The source fills its only slot with point 0 before reaching this target.
        batch.update_dual_topk(ARCH, &distances);

        let mut expected = vec![EMPTY; source + 1];
        expected[0] = Candidate::new(source as u32, -6.0);
        expected[target] = Candidate::new(source as u32, 4.0);
        expected[source] = Candidate::new(0, -6.0);
        let expected_limits: Vec<_> = expected.iter().map(|c| c.distance).collect();
        assert_eq!(batch.candidates.as_mut_slice(), expected, "target={target}");
        assert_eq!(batch.thresholds, expected_limits);
    }

    #[rstest]
    fn a_target_rejection_does_not_prevent_the_source_from_accepting(
        #[values(1, LANES - 1, LANES, 2 * LANES + 1)] target: usize,
    ) {
        let source = 2 * LANES + 2;
        let mut output = vec![EMPTY; source + 1];
        let mut limits = Vec::new();
        let mut rows = MutMatrixView::try_from(output.as_mut_slice(), source + 1, 1).unwrap();
        let mut batch = Batch::new(rows.as_mut_view(), &mut limits);
        let mut previous_distances = vec![f32::INFINITY; target];
        previous_distances[0] = 1.0;
        batch.update_dual_topk(ARCH, &previous_distances);
        let mut distances = vec![f32::INFINITY; source];
        distances[target] = 5.0;

        // The target already has a closer neighbor; the source still needs one.
        batch.update_dual_topk(ARCH, &distances);

        let mut expected = vec![EMPTY; source + 1];
        expected[0] = Candidate::new(target as u32, 1.0);
        expected[target] = Candidate::new(0, 1.0);
        expected[source] = Candidate::new(target as u32, 5.0);
        let expected_limits: Vec<_> = expected.iter().map(|c| c.distance).collect();
        assert_eq!(batch.candidates.as_mut_slice(), expected, "target={target}");
        assert_eq!(batch.thresholds, expected_limits);
    }

    #[rstest]
    #[case::finite(12.0)]
    #[case::negative_infinity(f32::NEG_INFINITY)]
    #[case::negative_zero(-0.0)]
    fn a_single_pair_updates_exactly_its_two_endpoints(
        #[case] distance: f32,
        #[values(LANES - 1, LANES, 2 * LANES + 1)] target: usize,
    ) {
        let source = 2 * LANES + 2;
        let mut output = vec![EMPTY; (source + 1) * 2];
        let mut limits = Vec::new();
        let mut rows = MutMatrixView::try_from(output.as_mut_slice(), source + 1, 2).unwrap();
        let mut batch = Batch::new(rows.as_mut_view(), &mut limits);
        let mut distances = vec![f32::INFINITY; source];
        distances[target] = distance;

        batch.update_dual_topk(ARCH, &distances);

        let mut expected = vec![EMPTY; (source + 1) * 2];
        expected[source * 2] = Candidate::new(target as u32, distance);
        expected[target * 2] = Candidate::new(source as u32, distance);
        assert_eq!(batch.candidates.as_mut_slice(), expected);
        assert_eq!(
            batch.candidates.as_mut_slice()[source * 2]
                .distance
                .to_bits(),
            distance.to_bits()
        );
        assert_eq!(
            batch.candidates.as_mut_slice()[target * 2]
                .distance
                .to_bits(),
            distance.to_bits()
        );
        // One neighbor leaves each two-slot result open to another candidate.
        assert_eq!(batch.thresholds, vec![f32::INFINITY; source + 1]);
    }

    #[rstest]
    #[case::no_pairs(&[])]
    #[case::nan_pairs(&[f32::NAN; 2 * LANES + 1])]
    #[case::infinite_pairs(&[f32::INFINITY; 2 * LANES + 1])]
    fn an_update_without_rankable_pairs_preserves_existing_results(#[case] distances: &[f32]) {
        let source = 2 * LANES + 1;
        let mut output = vec![EMPTY; source + 1];
        let mut limits = Vec::new();
        let mut rows = MutMatrixView::try_from(output.as_mut_slice(), source + 1, 1).unwrap();
        let mut batch = Batch::new(rows.as_mut_view(), &mut limits);
        batch.update_dual_topk(ARCH, &[3.0]);
        let previous_output = batch.candidates.as_mut_slice().to_vec();
        let previous_limits = batch.thresholds.to_vec();

        batch.update_dual_topk(ARCH, distances);

        assert_eq!(batch.candidates.as_mut_slice(), previous_output);
        assert_eq!(batch.thresholds, previous_limits);
    }

    #[test]
    fn zero_capacity_pair_scans_leave_no_neighbors_or_finite_limits() {
        fn check<'a, Rows>(mut batch: Batch<'a, Rows>)
        where
            Batch<'a, Rows>: BatchRanker,
        {
            batch.update_dual_topk(ARCH, &[4.0]);
            batch.update_dual_topk(ARCH, &[2.0, 7.0]);

            for index in 0..batch.nrows() {
                assert!(batch.ranker(index).unwrap().as_ref().is_empty());
            }
            assert_eq!(batch.thresholds, [f32::INFINITY; 3]);
        }

        let mut fixed: [[Candidate; 0]; 3] = [[]; 3];
        let mut fixed_limits = Vec::new();
        fixed_limits.resize(fixed.len(), f32::INFINITY);
        check(Batch {
            candidates: fixed.as_mut_slice(),
            thresholds: &mut fixed_limits,
        });

        let mut dynamic = [];
        let mut dynamic_limits = Vec::new();
        let rows = MutMatrixView::try_from(&mut dynamic[..], 3, 0).unwrap();
        check(Batch::new(rows, &mut dynamic_limits));
    }
}
