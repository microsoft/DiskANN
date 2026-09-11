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

// Fixed widths preserve row lengths for inlined insertion. Zero selects runtime width.
pub(super) const RUNTIME_WIDTH: usize = 0;

/// Select the width specialization once for a batch.
/// Specialize small capacities 1..=10; larger capacities use the runtime path.
/// Typical partition K is 10 or 3, and leaf K is 3 or 2.
/// The width is evaluated once. The body has ordinary block control flow.
macro_rules! with_topk {
    ($width:expr, |$topk:ident| $body:block) => {{
        let width = $width;
        match width {
            1 => {
                let $topk = $crate::graph::pipnn::topk::TopK::<1>::new(width);
                $body
            }
            2 => {
                let $topk = $crate::graph::pipnn::topk::TopK::<2>::new(width);
                $body
            }
            3 => {
                let $topk = $crate::graph::pipnn::topk::TopK::<3>::new(width);
                $body
            }
            4 => {
                let $topk = $crate::graph::pipnn::topk::TopK::<4>::new(width);
                $body
            }
            5 => {
                let $topk = $crate::graph::pipnn::topk::TopK::<5>::new(width);
                $body
            }
            6 => {
                let $topk = $crate::graph::pipnn::topk::TopK::<6>::new(width);
                $body
            }
            7 => {
                let $topk = $crate::graph::pipnn::topk::TopK::<7>::new(width);
                $body
            }
            8 => {
                let $topk = $crate::graph::pipnn::topk::TopK::<8>::new(width);
                $body
            }
            9 => {
                let $topk = $crate::graph::pipnn::topk::TopK::<9>::new(width);
                $body
            }
            10 => {
                let $topk = $crate::graph::pipnn::topk::TopK::<10>::new(width);
                $body
            }
            _ => {
                let $topk = $crate::graph::pipnn::topk::TopK::<
                    { $crate::graph::pipnn::topk::RUNTIME_WIDTH },
                >::new(width);
                $body
            }
        }
    }};
}
pub(super) use with_topk;

/// Top-k selection with a fixed result capacity and caller-owned storage.
///
/// This object stores no candidates or thresholds. Each operation borrows only
/// the buffers it updates. Candidate IDs are slice positions, not dataset IDs.
/// Callers supply each candidate at most once per result; equal distances need
/// no fixed tie order. The construction macro specializes the width per batch.
/// Selection and dual updates run inside their own architecture scope.
pub(super) struct TopK<const K: usize> {
    k: usize,
}

impl<const K: usize> TopK<K> {
    /// Configure the result capacity; a fixed K must match it.
    pub(super) fn new(k: usize) -> Self {
        debug_assert!(
            K == RUNTIME_WIDTH || K == k,
            "top-k width must match its capacity"
        );
        Self { k }
    }

    // Keep initialization and row addressing specialized across the architecture boundary.
    #[inline(always)]
    fn capacity(&self) -> usize {
        if K == RUNTIME_WIDTH { self.k } else { K }
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
                let output = &mut output[..self.capacity()];
                if output.is_empty() {
                    return;
                }
                output.fill(Candidate::default());
                distance_blocks(arch, distances).fold(
                    f32::INFINITY,
                    #[inline(always)]
                    |limit, block| block.update_one::<K>(output, limit),
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
                let k = self.capacity();
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
                let nearest = &mut remaining[..k];
                let limit = distance_blocks(arch, distances).fold(
                    thresholds[point_idx],
                    #[inline(always)]
                    |limit, block| {
                        let limit = block.update_one::<K>(nearest, limit);
                        block.update_many::<K>(others, thresholds, point_idx as u32, k);
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
    /// Offer the block's candidates, preserving and returning the updated limit.
    #[inline(always)]
    fn update_one<const K: usize>(self, nearest: &mut [Candidate], mut max_distance: f32) -> f32 {
        match self {
            Self::Scalar {
                candidate_idx,
                distance,
            } => {
                if distance < max_distance {
                    max_distance =
                        insert_sorted::<K>(nearest, Candidate::new(candidate_idx as u32, distance));
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
                            max_distance = insert_sorted::<K>(
                                nearest,
                                Candidate::new((first_candidate + lane) as u32, distance),
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
                            max_distance = insert_sorted::<K>(
                                nearest,
                                Candidate::new((first_candidate + lane) as u32, distances[lane]),
                            );
                        }
                    }
                }
            }
        }
        max_distance
    }

    /// Offer one point to each result set identified by this block's candidate IDs.
    #[inline(always)]
    fn update_many<const K: usize>(
        self,
        candidates: &mut [Candidate],
        thresholds: &mut [f32],
        point_idx: u32,
        k: usize,
    ) {
        match self {
            Self::Scalar {
                candidate_idx,
                distance,
            } => {
                let limit = &mut thresholds[candidate_idx];
                if distance < *limit {
                    *limit = insert_sorted::<K>(
                        &mut candidates[candidate_idx * k..][..k],
                        Candidate::new(point_idx, distance),
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
                    limits[lane] = insert_sorted::<K>(
                        &mut candidates[(first_candidate + lane) * k..][..k],
                        Candidate::new(point_idx, distances[lane]),
                    );
                }
            }
        }
    }
}

/// Insert an eligible candidate in nearest-first order and return the new distance limit.
/// The caller must check the candidate against the current limit before insertion.
/// Fixed K exposes the insertion capacity to loop unrolling; runtime K uses the slice length.
#[inline(always)]
fn insert_sorted<const K: usize>(nearest: &mut [Candidate], candidate: Candidate) -> f32 {
    let last = if K == RUNTIME_WIDTH {
        nearest.len() - 1
    } else {
        K - 1
    };
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
    use diskann_wide::arch::{self, Target1};
    use rstest::rstest;

    // These adapters only select the real CPU backend; all updates use production TopK.
    struct SelectTopK;
    impl<A: PiPNNSIMDSchema, const K: usize> Target1<A, (), (&TopK<K>, &[f32], &mut [Candidate])>
        for SelectTopK
    {
        fn run(self, arch: A, (topk, distances, output): (&TopK<K>, &[f32], &mut [Candidate])) {
            topk.select_topk(arch, distances, output);
        }
    }

    struct UpdatePair;
    impl<A: PiPNNSIMDSchema, const K: usize>
        Target1<
            A,
            (),
            (
                &TopK<K>,
                usize,
                &[f32],
                MutMatrixView<'_, Candidate>,
                &mut [f32],
            ),
        > for UpdatePair
    {
        fn run(
            self,
            arch: A,
            (topk, point, distances, output, thresholds): (
                &TopK<K>,
                usize,
                &[f32],
                MutMatrixView<'_, Candidate>,
                &mut [f32],
            ),
        ) {
            topk.update_dual_topk(arch, point, distances, output, thresholds);
        }
    }

    fn sorted_candidates(distances: &[f32], width: usize) -> Vec<Candidate> {
        let mut candidates: Vec<_> = distances
            .iter()
            .enumerate()
            .filter(|(_, distance)| **distance < f32::INFINITY)
            .map(|(id, &distance)| Candidate::new(id as u32, distance))
            .collect();
        candidates.sort_by(|left, right| left.distance.partial_cmp(&right.distance).unwrap());
        candidates.truncate(width);
        candidates.resize(width, Candidate::default());
        candidates
    }

    #[test]
    fn initialize_clears_candidates_and_resizes_thresholds() {
        // Given: candidate storage contains old results; the threshold buffer
        // still has the smaller previous leaf's length.
        let mut output = [
            Candidate::new(1, 1.0),
            Candidate::new(2, 4.0),
            Candidate::new(0, 1.0),
            Candidate::new(2, 3.0),
            Candidate::new(1, 3.0),
            Candidate::new(0, 4.0),
        ];
        let mut thresholds = vec![4.0, 3.0];
        let expected_output = [Candidate::default(); 6];
        let expected_thresholds = [f32::INFINITY; 3];
        let topk = TopK::<2>::new(2);

        // When
        topk.initialize(
            MutMatrixView::try_from(&mut output[..], 3, 2).unwrap(),
            &mut thresholds,
        );

        // Then: clear both old thresholds and the newly added slot.
        assert_eq!(output, expected_output);
        assert_eq!(thresholds, expected_thresholds);
    }

    #[test]
    fn zero_capacity_accepts_both_update_operations() {
        let mut output = [];
        let mut thresholds = [f32::INFINITY; 2];
        let topk = TopK::<RUNTIME_WIDTH>::new(0);

        arch::dispatch1_no_features(SelectTopK, (&topk, &[1.0, 2.0][..], &mut output[..]));
        arch::dispatch1_no_features(
            UpdatePair,
            (
                &topk,
                1,
                &[0.0][..],
                MutMatrixView::try_from(&mut output[..], 2, 0).unwrap(),
                &mut thresholds[..],
            ),
        );

        assert!(output.is_empty());
        assert_eq!(thresholds, [f32::INFINITY; 2]);
    }

    mod select_topk_tests {
        use super::*;

        #[rstest]
        #[case::lane_minus_one(15, 3)]
        #[case::complete_lane(16, 2)]
        #[case::lane_and_tail(17, 10)]
        #[case::two_lanes_minus_one(31, 1)]
        #[case::two_complete_lanes(32, 4)]
        #[case::two_lanes_and_tail(33, 17)]
        #[case::capacity_five(33, 5)]
        #[case::capacity_six(33, 6)]
        #[case::capacity_seven(33, 7)]
        #[case::capacity_eight(33, 8)]
        #[case::capacity_nine(33, 9)]
        fn dispatched_selection_matches_independent_sort(
            #[case] count: usize,
            #[case] width: usize,
        ) {
            // Alternating signs force insertions at different positions, not just append.
            let distances: Vec<_> = (0..count)
                .map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) })
                .collect();
            let expected = sorted_candidates(&distances, width);
            let mut output = vec![Candidate::default(); width];

            with_topk!(width, |topk| {
                arch::dispatch1_no_features(
                    SelectTopK,
                    (&topk, distances.as_slice(), output.as_mut_slice()),
                );
                assert_eq!(output, expected);
            });
        }

        #[rstest]
        #[case::unrankable(&[f32::NAN, f32::INFINITY, f32::INFINITY], [Candidate::default(); 2])]
        #[case::partly_filled(
            &[f32::NAN, f32::INFINITY, 9.0],
            [Candidate::new(2, 9.0), Candidate::default()]
        )]
        fn each_point_replaces_the_previous_points_candidates(
            #[case] distances: &[f32],
            #[case] expected: [Candidate; 2],
        ) {
            let mut output = [Candidate::default(); 2];
            let topk = TopK::<2>::new(2);
            topk.select_topk(diskann_wide::ARCH, &[3.0, 1.0, 2.0], &mut output);

            arch::dispatch1_no_features(SelectTopK, (&topk, distances, &mut output[..]));

            assert_eq!(output, expected);
        }

        #[test]
        fn finite_simd_mask_rechecks_the_threshold_after_insertion() {
            // Group one fills K=1 at 5. Both 1 and 4 pass group two's initial
            // mask, but inserting 1 must prevent the later 4 from replacing it.
            let mut distances = [f32::INFINITY; 32];
            distances[0] = 5.0;
            distances[16] = 1.0;
            distances[17] = 4.0;
            let mut output = [Candidate::default()];
            let topk = TopK::<1>::new(1);

            arch::dispatch1_no_features(SelectTopK, (&topk, &distances[..], &mut output[..]));

            assert_eq!(output, [Candidate::new(16, 1.0)]);
        }

        #[test]
        fn closer_candidates_shift_the_middle_of_a_full_result() {
            let mut output = [Candidate::default(); 3];
            let topk = TopK::<3>::new(3);

            // After retaining [0, 4, 6], distances 2 and 3 must enter between
            // existing candidates instead of replacing the last slot directly.
            arch::dispatch1_no_features(
                SelectTopK,
                (&topk, &[0.0, 4.0, 6.0, 2.0, 3.0][..], &mut output[..]),
            );

            assert_eq!(
                output,
                [
                    Candidate::new(0, 0.0),
                    Candidate::new(3, 2.0),
                    Candidate::new(4, 3.0)
                ]
            );
        }

        #[test]
        fn non_rankable_first_block_preserves_later_candidate_ids() {
            let mut distances = [f32::INFINITY; 33];
            distances[..8].fill(f32::NAN);
            distances[16] = 4.0;
            distances[17] = 2.0;
            distances[31] = 3.0;
            distances[32] = 1.0;
            let mut output = [Candidate::default(); 3];
            let topk = TopK::<3>::new(3);

            arch::dispatch1_no_features(SelectTopK, (&topk, &distances[..], &mut output[..]));

            assert_eq!(
                output,
                [
                    Candidate::new(32, 1.0),
                    Candidate::new(17, 2.0),
                    Candidate::new(31, 3.0),
                ]
            );
        }

        #[test]
        fn nearer_tail_displaces_a_tied_candidate_without_duplicate_ids() {
            let mut distances = [2.0; 17];
            distances[16] = 1.0;
            let mut output = [Candidate::default(); 3];
            let topk = TopK::<3>::new(3);

            arch::dispatch1_no_features(SelectTopK, (&topk, &distances[..], &mut output[..]));

            assert_eq!(output[0], Candidate::new(16, 1.0));
            let tied = &output[1..];
            assert!(
                tied.iter()
                    .all(|candidate| candidate.local_idx < 16 && candidate.distance == 2.0)
            );
            assert_ne!(tied[0].local_idx, tied[1].local_idx);
        }

        #[test]
        fn special_rankable_distances_keep_distinct_ids_and_original_bits() {
            let mut distances = [f32::INFINITY; 33];
            distances[0] = -0.0;
            distances[16] = 0.0;
            distances[17] = f32::MAX;
            distances[32] = f32::NEG_INFINITY;
            let mut output = [Candidate::default(); 4];
            let topk = TopK::<RUNTIME_WIDTH>::new(4);

            arch::dispatch1_no_features(SelectTopK, (&topk, &distances[..], &mut output[..]));

            assert_eq!(output[0], Candidate::new(32, f32::NEG_INFINITY));
            // Tie order is unspecified; each retained value must match its own ID.
            let mut actual: Vec<_> = output
                .iter()
                .map(|candidate| (candidate.local_idx, candidate.distance.to_bits()))
                .collect();
            actual.sort_unstable_by_key(|&(id, _)| id);
            assert_eq!(
                actual,
                [
                    (0, (-0.0_f32).to_bits()),
                    (16, 0.0_f32.to_bits()),
                    (17, f32::MAX.to_bits()),
                    (32, f32::NEG_INFINITY.to_bits())
                ]
            );
        }
    }

    mod update_dual_topk_tests {
        use super::*;

        #[test]
        fn endpoints_accept_independently_in_both_simd_groups_and_tail() {
            let mut distances = [f32::INFINITY; 33];
            distances[0] = 0.5;
            distances[1] = 4.0;
            distances[2] = 6.0;
            distances[16] = 2.0;
            distances[32] = 0.75;
            let mut output = [Candidate::default(); 34];
            let mut thresholds = [f32::INFINITY; 34];
            let topk = TopK::<1>::new(1);
            let mut rows = MutMatrixView::try_from(&mut output[..], 34, 1).unwrap();
            // Earlier leaf rows establish each destination's threshold.
            let mut previous = [f32::INFINITY; 32];
            for point in 1..33 {
                if point == 3 {
                    previous[..3].copy_from_slice(&[0.2, 5.0, 5.0]);
                }
                arch::dispatch1_no_features(
                    UpdatePair,
                    (
                        &topk,
                        point,
                        &previous[..point],
                        rows.as_mut_view(),
                        &mut thresholds[..],
                    ),
                );
                previous.fill(f32::INFINITY);
            }
            let mut expected = [Candidate::default(); 34];
            expected[0] = Candidate::new(3, 0.2);
            expected[1] = Candidate::new(33, 4.0);
            expected[2] = Candidate::new(3, 5.0);
            expected[3] = Candidate::new(0, 0.2);
            expected[16] = Candidate::new(33, 2.0);
            expected[32] = Candidate::new(33, 0.75);
            expected[33] = Candidate::new(0, 0.5);

            arch::dispatch1_no_features(
                UpdatePair,
                (
                    &topk,
                    33,
                    &distances[..],
                    rows.as_mut_view(),
                    &mut thresholds[..],
                ),
            );

            assert_eq!(rows.as_slice(), expected);
        }

        #[test]
        fn a_source_fills_across_blocks_with_few_rankable_distances() {
            let mut output = [Candidate::default(); 34 * 3];
            let mut thresholds = [f32::INFINITY; 34];
            let topk = TopK::<3>::new(3);
            let mut rows = MutMatrixView::try_from(&mut output[..], 34, 3).unwrap();
            let mut distances = [f32::INFINITY; 33];
            distances[..8].fill(f32::NAN);
            distances[15] = 4.0;
            distances[16] = 2.0;
            distances[17] = 9.0;
            distances[31] = 3.0;
            distances[32] = f32::NEG_INFINITY;

            // A leaf source starts empty. It remains partially filled after
            // group one and acquires its finite threshold within group two.
            arch::dispatch1_no_features(
                UpdatePair,
                (
                    &topk,
                    33,
                    &distances[..],
                    rows.as_mut_view(),
                    &mut thresholds[..],
                ),
            );

            assert_eq!(
                rows.row(33),
                [
                    Candidate::new(32, f32::NEG_INFINITY),
                    Candidate::new(16, 2.0),
                    Candidate::new(31, 3.0)
                ]
            );
            assert_eq!(
                rows.row(17),
                [
                    Candidate::new(33, 9.0),
                    Candidate::default(),
                    Candidate::default()
                ]
            );
        }

        #[test]
        fn later_pairs_use_the_sources_updated_threshold() {
            let mut output = [Candidate::default(); 3];
            let mut thresholds = [f32::INFINITY; 3];
            let topk = TopK::<1>::new(1);
            let mut rows = MutMatrixView::try_from(&mut output[..], 3, 1).unwrap();

            arch::dispatch1_no_features(
                UpdatePair,
                (
                    &topk,
                    1,
                    &[1.0][..],
                    rows.as_mut_view(),
                    &mut thresholds[..],
                ),
            );
            arch::dispatch1_no_features(
                UpdatePair,
                (
                    &topk,
                    2,
                    &[3.0, 2.0][..],
                    rows.as_mut_view(),
                    &mut thresholds[..],
                ),
            );

            assert_eq!(
                rows.as_slice(),
                [
                    Candidate::new(1, 1.0),
                    Candidate::new(0, 1.0),
                    Candidate::new(1, 2.0)
                ]
            );
        }

        #[test]
        fn runtime_capacity_updates_rows_filled_by_earlier_leaf_pairs() {
            let mut output = [Candidate::default(); 18 * 4];
            let mut thresholds = [f32::INFINITY; 18];
            let mut previous = [f32::INFINITY; 17];
            let mut distances = [f32::INFINITY; 17];
            distances[0] = 4.0;
            distances[2] = 3.0;
            distances[15] = 2.0;
            distances[16] = 1.0;

            let mut rows = MutMatrixView::try_from(&mut output[..], 18, 4).unwrap();
            let topk = TopK::<RUNTIME_WIDTH>::new(4);
            for point in 1..17 {
                if point == 15 {
                    previous[..4].copy_from_slice(&[-1.0, 3.0, 5.0, 7.0]);
                }
                arch::dispatch1_no_features(
                    UpdatePair,
                    (
                        &topk,
                        point,
                        &previous[..point],
                        rows.as_mut_view(),
                        &mut thresholds[..],
                    ),
                );
                previous.fill(f32::INFINITY);
            }
            arch::dispatch1_no_features(
                UpdatePair,
                (
                    &topk,
                    17,
                    &distances[..],
                    rows.as_mut_view(),
                    &mut thresholds[..],
                ),
            );

            assert_eq!(
                rows.row(17),
                [
                    Candidate::new(16, 1.0),
                    Candidate::new(15, 2.0),
                    Candidate::new(2, 3.0),
                    Candidate::new(0, 4.0)
                ]
            );
            assert_eq!(
                rows.row(15),
                [
                    Candidate::new(0, -1.0),
                    Candidate::new(17, 2.0),
                    Candidate::new(1, 3.0),
                    Candidate::new(2, 5.0)
                ]
            );
            assert_eq!(
                rows.row(16),
                [
                    Candidate::new(17, 1.0),
                    Candidate::default(),
                    Candidate::default(),
                    Candidate::default()
                ]
            );
        }

        #[rstest]
        #[case::simd(15)]
        #[case::tail(16)]
        fn negative_infinity_reaches_both_endpoints(#[case] candidate: usize) {
            let mut distances = [f32::NAN; 17];
            distances[candidate] = f32::NEG_INFINITY;
            let mut output = [Candidate::default(); 18 * 2];
            let mut thresholds = [f32::INFINITY; 18];
            let topk = TopK::<2>::new(2);
            let mut rows = MutMatrixView::try_from(&mut output[..], 18, 2).unwrap();
            let mut expected = [[Candidate::default(); 2]; 18];
            expected[17][0] = Candidate::new(candidate as u32, f32::NEG_INFINITY);
            expected[candidate][0] = Candidate::new(17, f32::NEG_INFINITY);

            arch::dispatch1_no_features(
                UpdatePair,
                (
                    &topk,
                    17,
                    &distances[..],
                    rows.as_mut_view(),
                    &mut thresholds[..],
                ),
            );

            assert_eq!(rows.as_slice(), expected.as_flattened());
        }
    }
}
