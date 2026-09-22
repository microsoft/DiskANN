/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Top-k updates from caller-selected distance slices.
//! NaN and positive infinity are not retained.

use diskann_utils::views::MutMatrixView;
use diskann_wide::{SIMDPartialOrd, SIMDVector};

use super::simd::PiPNNSIMDSchema;

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

/// A result capacity known at compile time.
#[derive(Clone, Copy, Debug)]
pub(super) struct Fixed<const K: usize>;

/// A result capacity selected at runtime.
#[derive(Clone, Copy, Debug)]
pub(super) struct Runtime(pub(super) usize);

/// Keep fixed capacities available to inlined insertion across architecture scopes.
pub(super) trait Width: Copy {
    fn capacity(self) -> usize;
}

impl<const K: usize> Width for Fixed<K> {
    #[inline(always)]
    fn capacity(self) -> usize {
        K
    }
}

impl Width for Runtime {
    #[inline(always)]
    fn capacity(self) -> usize {
        self.0
    }
}

/// A batch operation that accepts any TopK width specialization.
pub(super) trait TopKVisitor {
    /// Consume the operation and run it with the selected width.
    fn visit<W: Width>(self, topk: TopK<W>);
}

/// Select the width specialization and run the visitor once for the batch.
///
/// Specialize capacities 1..=10; zero and larger capacities use the runtime path.
/// Typical partition K is 10 or 3, and leaf K is 3 or 2.
#[inline]
pub(super) fn with_topk<V: TopKVisitor>(width: usize, visitor: V) {
    match width {
        1 => visitor.visit(TopK::new(Fixed::<1>)),
        2 => visitor.visit(TopK::new(Fixed::<2>)),
        3 => visitor.visit(TopK::new(Fixed::<3>)),
        4 => visitor.visit(TopK::new(Fixed::<4>)),
        5 => visitor.visit(TopK::new(Fixed::<5>)),
        6 => visitor.visit(TopK::new(Fixed::<6>)),
        7 => visitor.visit(TopK::new(Fixed::<7>)),
        8 => visitor.visit(TopK::new(Fixed::<8>)),
        9 => visitor.visit(TopK::new(Fixed::<9>)),
        10 => visitor.visit(TopK::new(Fixed::<10>)),
        _ => visitor.visit(TopK::new(Runtime(width))),
    }
}

/// Top-k selection with a fixed result capacity and caller-owned storage.
///
/// This object stores no candidates or thresholds. Each operation borrows only
/// the buffers it updates. Candidate IDs are slice positions, not dataset IDs.
/// Callers supply each candidate at most once per result; equal distances need
/// no fixed tie order. [`with_topk`] specializes the width per batch.
/// Selection and dual updates run inside their own architecture scope.
pub(super) struct TopK<W> {
    width: W,
}

impl<W: Width> TopK<W> {
    /// Configure the result capacity.
    pub(super) fn new(width: W) -> Self {
        Self { width }
    }

    /// Clear results and prepare reusable thresholds before a sequence of dual updates.
    pub(super) fn initialize(
        &self,
        mut output: MutMatrixView<'_, Candidate>,
        thresholds: &mut Vec<f32>,
    ) {
        output.as_mut_slice().fill(Candidate::default());
        thresholds.resize(output.nrows(), f32::INFINITY);
        thresholds.fill(f32::INFINITY);
    }

    /// Select the nearest candidates, replacing the previous contents of output.
    /// Slice positions become candidate IDs. The threshold is local to this call.
    #[inline(always)]
    pub(super) fn select_topk<A: PiPNNSIMDSchema>(
        &self,
        arch: A,
        distances: &[f32],
        output: &mut [Candidate],
    ) {
        arch.run2(
            #[inline(always)]
            move |distances: &[f32], output: &mut [Candidate]| {
                let k = self.width.capacity();
                if k == 0 {
                    return;
                }
                output[..k].fill(Candidate::default());
                distance_blocks(arch, distances).fold(
                    f32::INFINITY,
                    #[inline(always)]
                    |limit, block| block.update_topk_from_candidates(output, limit, self.width),
                );
            },
            distances,
            output,
        );
    }

    /// Add each pair (point_idx, j) independently to both endpoints' top-k results.
    /// Call [`Self::initialize`] once before the pair scan; subsequent updates preserve its state.
    /// Preserve existing results; all slice indexes must be less than point_idx.
    #[inline]
    pub(super) fn update_dual_topk<A: PiPNNSIMDSchema>(
        &self,
        arch: A,
        point_idx: usize,
        distances: &[f32],
        output: MutMatrixView<'_, Candidate>,
        thresholds: &mut [f32],
    ) {
        arch.run3(
            #[inline(always)]
            move |distances: &[f32],
                  mut output: MutMatrixView<'_, Candidate>,
                  thresholds: &mut [f32]| {
                let k = self.width.capacity();
                debug_assert_eq!(
                    output.ncols(),
                    k,
                    "top-k output width must match its capacity"
                );
                if k == 0 {
                    return;
                }
                debug_assert!(
                    distances.len() <= point_idx,
                    "candidate indexes must exclude the updated point"
                );
                // Keep this point's result and local limit independent of reciprocal updates.
                let (others, remaining) = output.as_mut_slice().split_at_mut(point_idx * k);
                let limit = distance_blocks(arch, distances).fold(
                    thresholds[point_idx],
                    #[inline(always)]
                    |limit, block| {
                        let limit = block.update_topk_from_candidates(remaining, limit, self.width);
                        block.update_topks_with_point(
                            others,
                            thresholds,
                            point_idx as u32,
                            self.width,
                        );
                        limit
                    },
                );
                thresholds[point_idx] = limit;
            },
            distances,
            output,
            thresholds,
        );
    }
}

/// A complete SIMD group or one tail distance. This never leaves the TopK module.
/// Both update directions share the loaded vector; scalar reads borrow the input.
#[derive(Clone, Copy)]
enum DistanceBlock<'a, A: PiPNNSIMDSchema> {
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
fn distance_blocks<A: PiPNNSIMDSchema>(
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

impl<A: PiPNNSIMDSchema> DistanceBlock<'_, A> {
    /// Update one top-k result from this block's candidates.
    ///
    /// Each candidate uses its index in the full distance row as its local ID.
    /// Partition ranking uses leader IDs. Leaf ranking uses IDs of earlier points in the leaf.
    ///
    /// Compare candidates against the existing result without clearing its slots.
    /// The first `width.capacity()` slots of `nearest` hold this result.
    /// The distance limit is the last slot's distance, or positive infinity while slots remain empty.
    /// Each insertion can lower this limit. Return the updated limit for the next block.
    #[inline(always)]
    fn update_topk_from_candidates<W: Width>(
        self,
        nearest: &mut [Candidate],
        mut max_distance: f32,
        width: W,
    ) -> f32 {
        match self {
            Self::Scalar {
                candidate_idx,
                distance,
            } => {
                if distance < max_distance {
                    max_distance = insert_sorted(
                        nearest,
                        Candidate::new(candidate_idx as u32, distance),
                        width,
                    );
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
                            max_distance = insert_sorted(
                                nearest,
                                Candidate::new((first_candidate + lane) as u32, distance),
                                width,
                            );
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
                            max_distance = insert_sorted(
                                nearest,
                                Candidate::new((first_candidate + lane) as u32, distances[lane]),
                                width,
                            );
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
    /// `candidates` starts at leaf row zero, with `width.capacity()` slots per row.
    /// `thresholds` holds one distance limit per row.
    /// The caller initializes both buffers once before the leaf scan.
    /// These updates retain earlier candidates unless a nearer point displaces them.
    ///
    /// Each row compares the pair distance against its own limit.
    /// Its decision does not depend on the current point's top-k result.
    /// Update the row's limit after an insertion.
    #[inline(always)]
    fn update_topks_with_point<W: Width>(
        self,
        candidates: &mut [Candidate],
        thresholds: &mut [f32],
        point_idx: u32,
        width: W,
    ) {
        let k = width.capacity();
        match self {
            Self::Scalar {
                candidate_idx,
                distance,
            } => {
                let limit = &mut thresholds[candidate_idx];
                if distance < *limit {
                    *limit = insert_sorted(
                        &mut candidates[candidate_idx * k..],
                        Candidate::new(point_idx, distance),
                        width,
                    );
                }
            }
            Self::Simd {
                first_candidate,
                values,
                distances,
            } => {
                let limits = &mut thresholds[first_candidate..][..A::Vector::LANES];
                // SAFETY: the slice above has one distance limit per SIMD lane.
                let limit_values = unsafe { A::Vector::load_simd(values.arch(), limits.as_ptr()) };
                let mut eligible = A::active_lanes(values.lt_simd(limit_values));
                while eligible != 0 {
                    let lane = eligible.trailing_zeros() as usize;
                    eligible &= eligible - 1;
                    // Each lane updates a different result; its limit is current.
                    limits[lane] = insert_sorted(
                        &mut candidates[(first_candidate + lane) * k..],
                        Candidate::new(point_idx, distances[lane]),
                        width,
                    );
                }
            }
        }
    }
}

/// Insert an eligible candidate in nearest-first order and return the new distance limit.
/// Only the first `width.capacity()` slots belong to this result.
/// The caller must check the candidate against the current limit before insertion.
#[inline(always)]
fn insert_sorted<W: Width>(nearest: &mut [Candidate], candidate: Candidate, width: W) -> f32 {
    let k = width.capacity();
    // Keep this slice here for performance.
    // Fixed<K> gives the slice a compile-time constant length.
    // After inlining, LLVM can unroll the loop and remove bounds checks.
    let nearest = &mut nearest[..k];
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
    const LANES: usize = <<Current as PiPNNSIMDSchema>::Vector as SIMDVector>::LANES;

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
        let topk = TopK::new(Fixed::<2>);
        let mut output = vec![[Candidate::new(0, -12.0), Candidate::new(1, -6.0)]; rows];
        let mut limits = vec![-6.0; previous_rows];

        // When: prepare storage for a new batch.
        topk.initialize(
            MutMatrixView::try_from(output.as_flattened_mut(), rows, 2).unwrap(),
            &mut limits,
        );

        // Then: no old candidate or limit belongs to the new batch.
        assert_eq!(output, vec![[EMPTY; 2]; rows]);
        assert_eq!(limits, vec![f32::INFINITY; rows]);
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
        let topk = TopK::new(Fixed::<3>);
        let mut output = [
            Candidate::new(2, -20.0),
            Candidate::new(4, -10.0),
            Candidate::new(9, -1.0),
        ];

        topk.select_topk(ARCH, distances, &mut output);

        assert_eq!(output, expected);
    }

    #[test]
    fn reusing_output_starts_a_selection_with_a_fresh_distance_limit() {
        let topk = TopK::new(Fixed::<2>);
        let mut output = [EMPTY; 2];
        topk.select_topk(ARCH, &[-20.0, -10.0], &mut output);
        assert_eq!(output, [Candidate::new(0, -20.0), Candidate::new(1, -10.0)]);

        // Every new distance exceeds the previous selection's limit.
        topk.select_topk(ARCH, &[16.0, 3.0, 9.0], &mut output);

        assert_eq!(output, [Candidate::new(1, 3.0), Candidate::new(2, 9.0)]);
    }

    #[test]
    fn batch_selection_matches_a_full_sort() {
        struct SelectRow<'a> {
            distances: &'a [f32],
            output: &'a mut [Candidate],
        }

        impl TopKVisitor for SelectRow<'_> {
            fn visit<W: Width>(self, topk: TopK<W>) {
                topk.select_topk(ARCH, self.distances, self.output);
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
                let mut capacity_reads = 0;

                with_topk(
                    {
                        capacity_reads += 1;
                        capacity
                    },
                    SelectRow {
                        distances: &distances,
                        output: &mut output,
                    },
                );

                assert_eq!(output, expected, "count={count}, capacity={capacity}");
                assert_eq!(capacity_reads, 1, "count={count}, capacity={capacity}");
            }
        }
    }

    #[test]
    fn selection_returns_distinct_candidates_when_distances_tie() {
        let best_index = 2 * LANES + 2;
        let mut distances = vec![8.0; best_index + 1];
        distances[best_index] = -2.0;
        let mut output = [EMPTY; 4];

        TopK::new(Fixed::<4>).select_topk(ARCH, &distances, &mut output);

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

        TopK::new(Runtime(2)).select_topk(ARCH, &distances, &mut output);

        assert_eq!(output[0].local_idx, index as u32);
        assert_eq!(output[0].distance.to_bits(), distance.to_bits());
        assert_eq!(output[1], EMPTY);
    }

    #[rstest]
    #[case::fixed(Fixed::<2>)]
    #[case::runtime(Runtime(2))]
    fn successive_pair_rows_accumulate_each_points_nearest_neighbors(#[case] width: impl Width) {
        let topk = TopK::new(width);
        let mut output = [[EMPTY; 2]; 4];
        let mut limits = Vec::new();
        let mut rows = MutMatrixView::try_from(output.as_flattened_mut(), 4, 2).unwrap();
        topk.initialize(rows.as_mut_view(), &mut limits);

        // Given: pairs 0-1=9, 0-2=2, and 1-2=7; point 3 has no offered pairs.
        topk.update_dual_topk(ARCH, 1, &[9.0], rows.as_mut_view(), &mut limits);
        topk.update_dual_topk(ARCH, 2, &[2.0, 7.0], rows.as_mut_view(), &mut limits);
        let before_last_point = [
            [Candidate::new(2, 2.0), Candidate::new(1, 9.0)],
            [Candidate::new(2, 7.0), Candidate::new(0, 9.0)],
            [Candidate::new(0, 2.0), Candidate::new(1, 7.0)],
            [EMPTY; 2],
        ];
        assert_eq!(rows.as_slice(), before_last_point.as_flattened());
        assert_eq!(limits, [9.0, 9.0, 7.0, f32::INFINITY]);

        // When: point 3 is at distances 6, 1, and 4 from points 0, 1, and 2.
        topk.update_dual_topk(ARCH, 3, &[6.0, 1.0, 4.0], rows.as_mut_view(), &mut limits);

        // Then: each endpoint keeps its own nearest two, with its farthest retained limit.
        let expected = [
            [Candidate::new(2, 2.0), Candidate::new(3, 6.0)],
            [Candidate::new(3, 1.0), Candidate::new(2, 7.0)],
            [Candidate::new(0, 2.0), Candidate::new(3, 4.0)],
            [Candidate::new(1, 1.0), Candidate::new(2, 4.0)],
        ];
        assert_eq!(rows.as_slice(), expected.as_flattened());
        assert_eq!(limits, [6.0, 7.0, 4.0, 4.0]);
    }

    #[test]
    fn pair_scans_match_full_row_sort_across_capacities_and_lengths() {
        struct ScanPairs<'a> {
            distances: &'a [Vec<f32>],
            output: MutMatrixView<'a, Candidate>,
            limits: &'a mut Vec<f32>,
        }

        impl TopKVisitor for ScanPairs<'_> {
            fn visit<W: Width>(mut self, topk: TopK<W>) {
                topk.initialize(self.output.as_mut_view(), self.limits);
                for (point, row) in self.distances.iter().enumerate().skip(1) {
                    topk.update_dual_topk(
                        ARCH,
                        point,
                        &row[..point],
                        self.output.as_mut_view(),
                        self.limits,
                    );
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
                with_topk(
                    capacity,
                    ScanPairs {
                        distances: &distances,
                        output: rows.as_mut_view(),
                        limits: &mut limits,
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
        let topk = TopK::new(Fixed::<1>);
        let mut output = vec![EMPTY; source + 1];
        let mut limits = Vec::new();
        let mut rows = MutMatrixView::try_from(output.as_mut_slice(), source + 1, 1).unwrap();
        topk.initialize(rows.as_mut_view(), &mut limits);
        let mut distances = vec![f32::INFINITY; source];
        distances[0] = -6.0;
        distances[target] = 4.0;

        // The source fills its only slot with point 0 before reaching this target.
        topk.update_dual_topk(ARCH, source, &distances, rows.as_mut_view(), &mut limits);

        let mut expected = vec![EMPTY; source + 1];
        expected[0] = Candidate::new(source as u32, -6.0);
        expected[target] = Candidate::new(source as u32, 4.0);
        expected[source] = Candidate::new(0, -6.0);
        let expected_limits: Vec<_> = expected.iter().map(|c| c.distance).collect();
        assert_eq!(rows.as_slice(), expected, "target={target}");
        assert_eq!(limits, expected_limits);
    }

    #[rstest]
    fn a_target_rejection_does_not_prevent_the_source_from_accepting(
        #[values(1, LANES - 1, LANES, 2 * LANES + 1)] target: usize,
    ) {
        let source = 2 * LANES + 2;
        let topk = TopK::new(Fixed::<1>);
        let mut output = vec![EMPTY; source + 1];
        let mut limits = Vec::new();
        let mut rows = MutMatrixView::try_from(output.as_mut_slice(), source + 1, 1).unwrap();
        topk.initialize(rows.as_mut_view(), &mut limits);
        let mut previous_distances = vec![f32::INFINITY; target];
        previous_distances[0] = 1.0;
        topk.update_dual_topk(
            ARCH,
            target,
            &previous_distances,
            rows.as_mut_view(),
            &mut limits,
        );
        let mut distances = vec![f32::INFINITY; source];
        distances[target] = 5.0;

        // The target already has a closer neighbor; the source still needs one.
        topk.update_dual_topk(ARCH, source, &distances, rows.as_mut_view(), &mut limits);

        let mut expected = vec![EMPTY; source + 1];
        expected[0] = Candidate::new(target as u32, 1.0);
        expected[target] = Candidate::new(0, 1.0);
        expected[source] = Candidate::new(target as u32, 5.0);
        let expected_limits: Vec<_> = expected.iter().map(|c| c.distance).collect();
        assert_eq!(rows.as_slice(), expected, "target={target}");
        assert_eq!(limits, expected_limits);
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
        let topk = TopK::new(Fixed::<2>);
        let mut output = vec![EMPTY; (source + 1) * 2];
        let mut limits = Vec::new();
        let mut rows = MutMatrixView::try_from(output.as_mut_slice(), source + 1, 2).unwrap();
        topk.initialize(rows.as_mut_view(), &mut limits);
        let mut distances = vec![f32::INFINITY; source];
        distances[target] = distance;

        topk.update_dual_topk(ARCH, source, &distances, rows.as_mut_view(), &mut limits);

        let mut expected = vec![EMPTY; (source + 1) * 2];
        expected[source * 2] = Candidate::new(target as u32, distance);
        expected[target * 2] = Candidate::new(source as u32, distance);
        assert_eq!(rows.as_slice(), expected);
        assert_eq!(rows.row(source)[0].distance.to_bits(), distance.to_bits());
        assert_eq!(rows.row(target)[0].distance.to_bits(), distance.to_bits());
        // One neighbor leaves each two-slot result open to another candidate.
        assert_eq!(limits, vec![f32::INFINITY; source + 1]);
    }

    #[rstest]
    #[case::no_pairs(&[])]
    #[case::nan_pairs(&[f32::NAN; 2 * LANES + 1])]
    #[case::infinite_pairs(&[f32::INFINITY; 2 * LANES + 1])]
    fn an_update_without_rankable_pairs_preserves_existing_results(#[case] distances: &[f32]) {
        let source = 2 * LANES + 1;
        let topk = TopK::new(Fixed::<1>);
        let mut output = vec![EMPTY; source + 1];
        let mut limits = Vec::new();
        let mut rows = MutMatrixView::try_from(output.as_mut_slice(), source + 1, 1).unwrap();
        topk.initialize(rows.as_mut_view(), &mut limits);
        topk.update_dual_topk(ARCH, 1, &[3.0], rows.as_mut_view(), &mut limits);
        let previous_output = rows.as_slice().to_vec();
        let previous_limits = limits.clone();

        topk.update_dual_topk(ARCH, source, distances, rows.as_mut_view(), &mut limits);

        assert_eq!(rows.as_slice(), previous_output);
        assert_eq!(limits, previous_limits);
    }

    #[rstest]
    #[case::fixed(Fixed::<0>)]
    #[case::runtime(Runtime(0))]
    fn zero_capacity_pair_scans_leave_no_neighbors_or_finite_limits(#[case] width: impl Width) {
        let topk = TopK::new(width);
        let mut output = [];
        let mut limits = Vec::new();
        let mut rows = MutMatrixView::try_from(&mut output[..], 3, 0).unwrap();
        topk.initialize(rows.as_mut_view(), &mut limits);

        topk.update_dual_topk(ARCH, 1, &[4.0], rows.as_mut_view(), &mut limits);
        topk.update_dual_topk(ARCH, 2, &[2.0, 7.0], rows.as_mut_view(), &mut limits);

        assert!(rows.as_slice().is_empty());
        assert_eq!(limits, [f32::INFINITY; 3]);
    }
}
