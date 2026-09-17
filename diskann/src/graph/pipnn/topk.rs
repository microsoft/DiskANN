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

/// Select the width specialization once for a batch.
/// Specialize small capacities 1..=10; larger capacities use the runtime path.
/// Typical partition K is 10 or 3, and leaf K is 3 or 2.
/// The width is evaluated once. The body has ordinary block control flow.
macro_rules! with_topk {
    ($width:expr, |$topk:ident| $body:block) => {{
        let width = $width;
        match width {
            1 => {
                let $topk =
                    $crate::graph::pipnn::topk::TopK::new($crate::graph::pipnn::topk::Fixed::<1>);
                $body
            }
            2 => {
                let $topk =
                    $crate::graph::pipnn::topk::TopK::new($crate::graph::pipnn::topk::Fixed::<2>);
                $body
            }
            3 => {
                let $topk =
                    $crate::graph::pipnn::topk::TopK::new($crate::graph::pipnn::topk::Fixed::<3>);
                $body
            }
            4 => {
                let $topk =
                    $crate::graph::pipnn::topk::TopK::new($crate::graph::pipnn::topk::Fixed::<4>);
                $body
            }
            5 => {
                let $topk =
                    $crate::graph::pipnn::topk::TopK::new($crate::graph::pipnn::topk::Fixed::<5>);
                $body
            }
            6 => {
                let $topk =
                    $crate::graph::pipnn::topk::TopK::new($crate::graph::pipnn::topk::Fixed::<6>);
                $body
            }
            7 => {
                let $topk =
                    $crate::graph::pipnn::topk::TopK::new($crate::graph::pipnn::topk::Fixed::<7>);
                $body
            }
            8 => {
                let $topk =
                    $crate::graph::pipnn::topk::TopK::new($crate::graph::pipnn::topk::Fixed::<8>);
                $body
            }
            9 => {
                let $topk =
                    $crate::graph::pipnn::topk::TopK::new($crate::graph::pipnn::topk::Fixed::<9>);
                $body
            }
            10 => {
                let $topk =
                    $crate::graph::pipnn::topk::TopK::new($crate::graph::pipnn::topk::Fixed::<10>);
                $body
            }
            _ => {
                let $topk = $crate::graph::pipnn::topk::TopK::new(
                    $crate::graph::pipnn::topk::Runtime(width),
                );
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
    use diskann_wide::arch::{self, Target1};
    use rstest::rstest;

    // These adapters only select the real CPU backend; all updates use production TopK.
    struct SelectTopK;
    impl<A: PiPNNSIMDSchema, W: Width> Target1<A, (), (&TopK<W>, &[f32], &mut [Candidate])>
        for SelectTopK
    {
        fn run(self, arch: A, (topk, distances, output): (&TopK<W>, &[f32], &mut [Candidate])) {
            topk.select_topk(arch, distances, output);
        }
    }

    struct UpdatePair;
    impl<A: PiPNNSIMDSchema, W: Width>
        Target1<
            A,
            (),
            (
                &TopK<W>,
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
                &TopK<W>,
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
    fn initialize_resets_results_and_limits() {
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
        let topk = TopK::new(Fixed::<2>);

        // When
        topk.initialize(
            MutMatrixView::try_from(&mut output[..], 3, 2).unwrap(),
            &mut thresholds,
        );

        // Then: clear both old thresholds and the newly added slot.
        assert_eq!(output, expected_output);
        assert_eq!(thresholds, expected_thresholds);
    }

    #[rstest]
    #[case::fixed(Fixed::<0>)]
    #[case::runtime(Runtime(0))]
    fn zero_capacity_preserves_results_and_limits(#[case] width: impl Width) {
        let mut output = [];
        let mut thresholds = [f32::INFINITY; 2];
        let topk = TopK::new(width);

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

    mod insert_sorted_tests {
        use super::*;

        #[rstest]
        #[case::fixed(Fixed::<3>)]
        #[case::runtime(Runtime(3))]
        fn insertion_preserves_other_rows(#[case] width: impl Width) {
            // Given: the new candidate displaces the first result's farthest neighbor.
            let first_result = [
                Candidate::new(0, 1.0),
                Candidate::new(1, 4.0),
                Candidate::new(2, 6.0),
            ];
            let next_result = [
                Candidate::new(3, 8.0),
                Candidate::new(4, 10.0),
                Candidate::new(5, 12.0),
            ];
            let candidate = Candidate::new(6, 3.0);
            let expected_first_result = [
                Candidate::new(0, 1.0),
                Candidate::new(6, 3.0),
                Candidate::new(1, 4.0),
            ];
            let expected_limit = 4.0;
            let mut results = [first_result, next_result];

            // When
            let limit = insert_sorted(results.as_flattened_mut(), candidate, width);

            // Then
            assert_eq!(results, [expected_first_result, next_result]);
            assert_eq!(limit, expected_limit);
        }
    }

    mod select_topk_tests {
        use super::*;

        #[rstest]
        #[case::before_first_simd_block(15)]
        #[case::one_simd_block(16)]
        #[case::one_simd_block_and_tail(17)]
        #[case::before_second_simd_block(31)]
        #[case::two_simd_blocks(32)]
        #[case::two_simd_blocks_and_tail(33)]
        #[case::three_simd_blocks_and_tail(49)]
        fn dispatched_selection_matches_independent_sort(
            #[case] count: usize,
            #[values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)] width: usize,
        ) {
            // Given: alternating signs force replacement and interior insertion.
            let distances: Vec<_> = (0..count)
                .map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) })
                .collect();
            let expected = sorted_candidates(&distances, width);
            let mut output = vec![Candidate::default(); width];
            let mut runtime_output = output.clone();

            // When
            with_topk!(width, |topk| {
                arch::dispatch1_no_features(
                    SelectTopK,
                    (&topk, distances.as_slice(), output.as_mut_slice()),
                );
            });
            arch::dispatch1_no_features(
                SelectTopK,
                (
                    &TopK::new(Runtime(width)),
                    distances.as_slice(),
                    runtime_output.as_mut_slice(),
                ),
            );

            // Then: both width representations must satisfy the independent oracle.
            assert_eq!(output, expected);
            assert_eq!(runtime_output, expected);
        }

        #[rstest]
        #[case::before_second_simd_block(31)]
        #[case::two_simd_blocks(32)]
        #[case::two_simd_blocks_and_tail(33)]
        #[case::three_simd_blocks_and_tail(49)]
        fn runtime_selection_matches_independent_sort(
            #[case] count: usize,
            #[values(11, 17)] width: usize,
        ) {
            // Given: K=17 fills during the second block. The third block starts
            // with a finite limit. Each input has more candidates than result slots.
            let distances: Vec<_> = (0..count)
                .map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) })
                .collect();
            let expected = sorted_candidates(&distances, width);
            let mut output = vec![Candidate::default(); width];

            // When: these capacities select the runtime branch of the batch dispatch.
            with_topk!(width, |topk| {
                arch::dispatch1_no_features(
                    SelectTopK,
                    (&topk, distances.as_slice(), output.as_mut_slice()),
                );
            });

            // Then
            assert_eq!(output, expected);
        }

        #[rstest]
        #[case::unrankable(&[f32::NAN, f32::INFINITY, f32::INFINITY], [Candidate::default(); 2])]
        #[case::partly_filled(
            &[f32::NAN, f32::INFINITY, 9.0],
            [Candidate::new(2, 9.0), Candidate::default()]
        )]
        fn selection_replaces_previous_results(
            #[case] distances: &[f32],
            #[case] expected: [Candidate; 2],
        ) {
            // Given: both slots contain candidates from the previous selection.
            let mut output = [Candidate::new(1, 1.0), Candidate::new(2, 2.0)];
            let topk = TopK::new(Fixed::<2>);

            // When: select again with the same output buffer.
            arch::dispatch1_no_features(SelectTopK, (&topk, distances, &mut output[..]));

            // Then: unused slots contain no candidates from the previous selection.
            assert_eq!(output, expected);
        }

        #[test]
        fn selection_rechecks_candidates_against_updated_limit() {
            // Group one fills K=1 at 5. Both 1 and 4 pass group two's initial
            // mask, but inserting 1 must prevent the later 4 from replacing it.
            let mut distances = [f32::INFINITY; 32];
            distances[0] = 5.0;
            distances[16] = 1.0;
            distances[17] = 4.0;
            let mut output = [Candidate::default()];
            let topk = TopK::new(Fixed::<1>);

            arch::dispatch1_no_features(SelectTopK, (&topk, &distances[..], &mut output[..]));

            assert_eq!(output, [Candidate::new(16, 1.0)]);
        }

        #[test]
        fn middle_insertions_keep_nearest_first_order() {
            let mut output = [Candidate::default(); 3];
            let topk = TopK::new(Fixed::<3>);

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
        fn non_rankable_block_preserves_input_indices() {
            // The first block contributes no candidates. Later IDs still refer
            // to positions in the full input, not positions among rankable values.
            let mut distances = [f32::INFINITY; 33];
            distances[..8].fill(f32::NAN);
            distances[16] = 4.0;
            distances[17] = 2.0;
            distances[31] = 3.0;
            distances[32] = 1.0;
            let mut output = [Candidate::default(); 3];
            let topk = TopK::new(Fixed::<3>);

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
        fn tail_candidate_replaces_a_tied_neighbor() {
            let mut distances = [2.0; 17];
            distances[16] = 1.0;
            let mut output = [Candidate::default(); 3];
            let topk = TopK::new(Fixed::<3>);

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
        fn selection_preserves_ids_and_float_bits() {
            let mut distances = [f32::INFINITY; 33];
            distances[0] = -0.0;
            distances[16] = 0.0;
            distances[17] = f32::MAX;
            distances[32] = f32::NEG_INFINITY;
            let mut output = [Candidate::default(); 4];
            let topk = TopK::new(Runtime(4));

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

        // Assign one unique score to each unordered pair. Shuffling prevents the
        // leaf scan order from also being distance order. The diagonal cannot rank.
        fn shuffled_symmetric_distances(count: usize) -> Vec<f32> {
            use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

            let mut scores: Vec<_> = (1..=count * (count - 1) / 2).map(|i| i as f32).collect();
            scores.shuffle(&mut StdRng::seed_from_u64(1287));
            let mut distances = vec![f32::INFINITY; count * count];
            for source in 1..count {
                for target in 0..source {
                    let score = scores.pop().unwrap();
                    distances[source * count + target] = score;
                    distances[target * count + source] = score;
                }
            }
            distances
        }

        #[rstest]
        #[case::one_simd_block_and_tail(18)]
        #[case::two_simd_blocks(33)]
        #[case::two_simd_blocks_and_tail(34)]
        #[case::three_simd_blocks_and_tail(50)]
        fn each_point_retains_its_nearest_neighbors(
            #[case] count: usize,
            #[values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17)] width: usize,
        ) {
            // Given: the oracle sorts complete rows, independent of the pair scan.
            // Every row has enough non-self candidates to fill its result.
            let distances = shuffled_symmetric_distances(count);
            let expected: Vec<_> = distances
                .chunks_exact(count)
                .map(|row| sorted_candidates(row, width))
                .collect();
            let mut output = vec![Candidate::default(); count * width];
            let mut thresholds = Vec::new();
            let mut rows = MutMatrixView::try_from(output.as_mut_slice(), count, width).unwrap();

            // When: each lower-triangle pair reaches both endpoints exactly once.
            with_topk!(width, |topk| {
                topk.initialize(rows.as_mut_view(), &mut thresholds);
                for source in 1..count {
                    arch::dispatch1_no_features(
                        UpdatePair,
                        (
                            &topk,
                            source,
                            &distances[source * count..source * count + source],
                            rows.as_mut_view(),
                            thresholds.as_mut_slice(),
                        ),
                    );
                }
            });

            // Then: check all source and reciprocal rows with their persisted limits.
            for (source, expected) in expected.iter().enumerate() {
                assert_eq!(
                    rows.row(source),
                    expected,
                    "source={source}, N={count}, K={width}"
                );
                assert_eq!(
                    thresholds[source],
                    expected[width - 1].distance,
                    "threshold for source={source}, N={count}, K={width}"
                );
            }
        }

        #[test]
        fn only_the_source_accepts_the_pair() {
            // Given: point 2 is the source. Point 0 is the target.
            // Points 0 and 1 already retain each other at distance 1.
            // The source has no neighbors. Distance 2 improves only its result.
            let mut output = [
                Candidate::new(1, 1.0),
                Candidate::new(0, 1.0),
                Candidate::default(),
            ];
            let mut thresholds = output.map(|candidate| candidate.distance);
            let expected = [output[0], output[1], Candidate::new(0, 2.0)];

            // When
            arch::dispatch1_no_features(
                UpdatePair,
                (
                    &TopK::new(Fixed::<1>),
                    2,
                    &[2.0, 3.0][..],
                    MutMatrixView::try_from(&mut output[..], 3, 1).unwrap(),
                    &mut thresholds[..],
                ),
            );

            // Then
            assert_eq!(output, expected);
            assert_eq!(thresholds, expected.map(|candidate| candidate.distance));
        }

        #[test]
        fn only_the_target_accepts_the_pair() {
            // Given: point 2 is the source. Point 1 is the target.
            // Points 0 and 1 retain each other at distance 3.
            // The source first selects point 0 at distance 1.
            // Its later distance 2 to the target improves only the target's result.
            let mut output = [
                Candidate::new(1, 3.0),
                Candidate::new(0, 3.0),
                Candidate::default(),
            ];
            let mut thresholds = output.map(|candidate| candidate.distance);
            let expected = [
                Candidate::new(2, 1.0),
                Candidate::new(2, 2.0),
                Candidate::new(0, 1.0),
            ];

            // When
            arch::dispatch1_no_features(
                UpdatePair,
                (
                    &TopK::new(Fixed::<1>),
                    2,
                    &[1.0, 2.0][..],
                    MutMatrixView::try_from(&mut output[..], 3, 1).unwrap(),
                    &mut thresholds[..],
                ),
            );

            // Then
            assert_eq!(output, expected);
            assert_eq!(thresholds, expected.map(|candidate| candidate.distance));
        }

        #[test]
        fn both_points_reject_the_farther_pair() {
            // Given: points 0 and 1 retain each other at distance 1.
            // Point 2 first selects point 0 at distance 0.5. The later pair
            // (2, 1) at distance 2 improves neither endpoint.
            let mut output = [
                Candidate::new(1, 1.0),
                Candidate::new(0, 1.0),
                Candidate::default(),
            ];
            let mut thresholds = output.map(|candidate| candidate.distance);
            let expected = [Candidate::new(2, 0.5), output[1], Candidate::new(0, 0.5)];

            // When
            arch::dispatch1_no_features(
                UpdatePair,
                (
                    &TopK::new(Fixed::<1>),
                    2,
                    &[0.5, 2.0][..],
                    MutMatrixView::try_from(&mut output[..], 3, 1).unwrap(),
                    &mut thresholds[..],
                ),
            );

            // Then
            assert_eq!(output, expected);
            assert_eq!(thresholds, expected.map(|candidate| candidate.distance));
        }

        #[test]
        fn later_blocks_fill_source_neighbor_slots() {
            let mut output = [Candidate::default(); 34 * 3];
            let mut thresholds = [f32::INFINITY; 34];
            let topk = TopK::new(Fixed::<3>);
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
        fn later_pairs_respect_saved_distance_limits() {
            let mut output = [Candidate::default(); 3];
            let mut thresholds = [f32::INFINITY; 3];
            let topk = TopK::new(Fixed::<1>);
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
        fn target_replaces_its_farthest_neighbor() {
            // Given: point 15 retains points 0, 1, 2, and 3. It is the last
            // lane in group one. Point 16 is the tail target and has no neighbors.
            // Point 17 enters the middle of point 15's full row.
            let mut output = [Candidate::default(); 18 * 4];
            let mut thresholds = [f32::INFINITY; 18];
            let mut previous = [f32::INFINITY; 15];
            previous[..4].copy_from_slice(&[-1.0, 3.0, 5.0, 7.0]);
            let mut distances = [f32::INFINITY; 17];
            distances[0] = 4.0;
            distances[2] = 3.0;
            distances[15] = 2.0;
            distances[16] = 1.0;
            let expected_source = [
                Candidate::new(16, distances[16]),
                Candidate::new(15, distances[15]),
                Candidate::new(2, distances[2]),
                Candidate::new(0, distances[0]),
            ];
            let expected_prior_row = [
                Candidate::new(0, previous[0]),
                Candidate::new(17, distances[15]),
                Candidate::new(1, previous[1]),
                Candidate::new(2, previous[2]),
            ];
            let expected_tail_row = [
                Candidate::new(17, distances[16]),
                Candidate::default(),
                Candidate::default(),
                Candidate::default(),
            ];

            let mut rows = MutMatrixView::try_from(&mut output[..], 18, 4).unwrap();
            let topk = TopK::new(Runtime(4));
            arch::dispatch1_no_features(
                UpdatePair,
                (
                    &topk,
                    15,
                    &previous[..],
                    rows.as_mut_view(),
                    &mut thresholds[..],
                ),
            );

            // When
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

            // Then: the source fills, the prior row evicts, and the tail stays underfilled.
            assert_eq!(rows.row(17), expected_source);
            assert_eq!(rows.row(15), expected_prior_row);
            assert_eq!(rows.row(16), expected_tail_row);
            assert_eq!(thresholds[17], expected_source[3].distance);
            assert_eq!(thresholds[15], expected_prior_row[3].distance);
            assert_eq!(thresholds[16], f32::INFINITY);
        }

        #[rstest]
        #[case::simd(15)]
        #[case::tail(16)]
        fn both_points_accept_negative_infinity(#[case] candidate: usize) {
            let mut distances = [f32::NAN; 17];
            distances[candidate] = f32::NEG_INFINITY;
            let mut output = [Candidate::default(); 18 * 2];
            let mut thresholds = [f32::INFINITY; 18];
            let topk = TopK::new(Fixed::<2>);
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
