/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Leaf-local top-k selection from packed `f32` point vectors.
//!
//! The metric fills a flattened lower-triangular distance buffer. The kernel
//! reads each strict-lower point pair once and updates both points.
//!
//! The output is an `n × k` matrix of sorted [`LeafNeighbor`] values. Each target
//! is a position in the leaf. Widths 1 through 3 use fixed insertion. Larger
//! widths use the runtime insertion loop.
//!
//! Equal distances can select either candidate. The kernel does not rank NaN.
//! An unfilled output slot contains [`LeafNeighbor::default`]. All supported
//! metrics use the same SIMD-group and single-value traversal.
//!
//! The caller supplies concrete architecture `A` and metric `M`.
//! [`LeafKernelWorkspace`] stores reusable numerical scratch.

use crate::{ANNError, ANNResult};
use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_wide::{SIMDPartialOrd, SIMDVector};

use super::{
    leaf_metric::LeafMetric,
    simd::{PiPNNSIMDSchema, PiPNNSIMDVector},
};

/// One leaf-local neighbor and its ranking distance.
///
/// A ranking distance preserves nearest-first order. It need not equal the
/// metric distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct LeafNeighbor {
    /// Target position in the leaf, not a dataset ID.
    pub(super) target: u32,
    /// Ranking distance from the source point to `target`.
    pub(super) distance: f32,
}

impl LeafNeighbor {
    /// Construct a leaf-local neighbor.
    ///
    /// The output row determines the source point.
    pub(super) const fn new(target: u32, distance: f32) -> Self {
        Self { target, distance }
    }

    /// Return true when this slot contains a rankable leaf-local target.
    pub(super) const fn is_assigned(self) -> bool {
        self.target != u32::MAX
    }
}

impl Default for LeafNeighbor {
    fn default() -> Self {
        Self::new(u32::MAX, f32::INFINITY)
    }
}

/// Reusable storage for one leaf numerical pipeline.
#[derive(Debug, Default)]
pub(super) struct LeafKernelWorkspace {
    distance_scratch: Vec<f32>,
    worst: Vec<f32>,
}

/// Invalid output width for leaf-neighbor selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(super) enum LeafKernelError {
    /// A source requests more neighbors than the leaf has other points.
    #[error("invalid leaf neighbor count {neighbors} for {points} points; maximum is {maximum}")]
    InvalidNeighborCount {
        points: usize,
        neighbors: usize,
        maximum: usize,
    },
}

/// Return the non-self neighbor count for one leaf.
///
/// `points` is the number of points in the leaf. `requested_k` is the configured
/// neighbor count. The result is `min(requested_k, points - 1)`.
///
pub(super) fn leaf_neighbor_count(points: usize, requested_k: usize) -> usize {
    requested_k.min(points.saturating_sub(1))
}

/// Compute local nearest neighbors for one packed leaf matrix.
///
/// # Errors
///
/// Returns an error for invalid linear-algebra input or output width.
/// Invalid output widths leave the output and workspace unchanged.
pub(super) fn select_leaf_neighbors<A, M>(
    arch: A,
    points: MatrixView<'_, f32>,
    output: MutMatrixView<'_, LeafNeighbor>,
    workspace: &mut LeafKernelWorkspace,
) -> ANNResult<()>
where
    A: PiPNNSIMDSchema,
    M: LeafMetric,
{
    let point_count = points.nrows();
    validate_neighbor_count(point_count, &output).map_err(ANNError::new)?;
    let distance_count = point_count * point_count;
    let LeafKernelWorkspace {
        distance_scratch,
        worst,
    } = workspace;
    if distance_scratch.len() < distance_count {
        distance_scratch.resize(distance_count, 0.0);
    }
    M::compute_distances(points, &mut distance_scratch[..distance_count])?;
    rank_leaf_distances(
        arch,
        &distance_scratch[..distance_count],
        point_count,
        output,
        worst,
    );
    Ok(())
}

/// Rank one flattened lower-triangle buffer.
///
/// `distance_flatten` contains `point_count * point_count` elements. The metric
/// initializes each strict-lower entry. The kernel does not read the upper triangle.
/// The caller validates the output width against the non-self point count.
fn rank_leaf_distances<A>(
    arch: A,
    distance_flatten: &[f32],
    point_count: usize,
    mut output: MutMatrixView<'_, LeafNeighbor>,
    worst: &mut Vec<f32>,
) where
    A: PiPNNSIMDSchema,
{
    let neighbor_count = output.ncols();
    if neighbor_count == 0 {
        return;
    }

    worst.resize(point_count, f32::INFINITY);
    output.as_mut_slice().fill(LeafNeighbor::default());
    worst.fill(f32::INFINITY);

    match neighbor_count {
        1 => scan_fixed_width::<A, 1>(
            arch,
            distance_flatten,
            point_count,
            output.as_mut_slice(),
            worst,
        ),
        2 => scan_fixed_width::<A, 2>(
            arch,
            distance_flatten,
            point_count,
            output.as_mut_slice(),
            worst,
        ),
        3 => scan_fixed_width::<A, 3>(
            arch,
            distance_flatten,
            point_count,
            output.as_mut_slice(),
            worst,
        ),
        _ => scan_runtime_width(
            arch,
            distance_flatten,
            point_count,
            output.as_mut_slice(),
            neighbor_count,
            worst,
        ),
    }
}

/// Check the output width against the number of non-self points.
fn validate_neighbor_count(
    point_count: usize,
    output: &MutMatrixView<'_, LeafNeighbor>,
) -> Result<(), LeafKernelError> {
    let maximum_neighbors = point_count.saturating_sub(1);
    let neighbor_count = output.ncols();
    if neighbor_count > maximum_neighbors {
        return Err(LeafKernelError::InvalidNeighborCount {
            points: point_count,
            neighbors: neighbor_count,
            maximum: maximum_neighbors,
        });
    }
    Ok(())
}

/// Select neighbors with a fixed output width.
fn scan_fixed_width<A, const N: usize>(
    arch: A,
    distance_flatten: &[f32],
    point_count: usize,
    output: &mut [LeafNeighbor],
    worst: &mut [f32],
) where
    A: PiPNNSIMDSchema,
    [LeafNeighbor; N]: NeighborInsert,
{
    let (rows, _) = output.as_chunks_mut::<N>();
    // Rayon outlines leaf workers. Reapply target features before the SIMD scan.
    arch.run(move || {
        scan_point_pairs(
            arch,
            distance_flatten,
            point_count,
            worst,
            |source, target, distance| {
                rows[source].insert_eligible(LeafNeighbor::new(target, distance))
            },
        );
    });
}

/// Select neighbors with a runtime output width.
fn scan_runtime_width<A>(
    arch: A,
    distance_flatten: &[f32],
    point_count: usize,
    output: &mut [LeafNeighbor],
    width: usize,
    worst: &mut [f32],
) where
    A: PiPNNSIMDSchema,
{
    // Rayon outlines leaf workers. Reapply target features before the SIMD scan.
    arch.run(move || {
        scan_point_pairs(
            arch,
            distance_flatten,
            point_count,
            worst,
            |source, target, distance| {
                let first = source * width;
                output[first..first + width].insert_eligible(LeafNeighbor::new(target, distance))
            },
        );
    });
}

/// Select neighbors from all unordered point pairs in one leaf.
///
/// The function reads the strict lower triangle once. It offers each distance to
/// both endpoint lists.
#[inline(always)]
fn scan_point_pairs<A, I>(
    arch: A,
    distance_flatten: &[f32],
    point_count: usize,
    worst: &mut [f32],
    mut insert: I,
) where
    A: PiPNNSIMDSchema,
    I: FnMut(usize, u32, f32) -> f32,
{
    let worst_ptr = worst.as_mut_ptr();

    for source in 1..point_count {
        let source_start = source * point_count;
        // SAFETY: `rank_leaf_distances` created one threshold for each point.
        let mut source_worst = unsafe { *worst_ptr.add(source) };
        let mut target = 0;
        let simd_prefix = source - source % A::Vector::LANES;

        while target < simd_prefix {
            // SAFETY: This complete SIMD group is in the strict-lower prefix.
            let distance_group = unsafe {
                A::Vector::load_simd(arch, distance_flatten.as_ptr().add(source_start + target))
            };
            let source_eligible = distance_group.lt_simd(A::Vector::splat(arch, source_worst));
            // SAFETY: The complete target group is below `source < point_count`.
            let target_worst = unsafe { A::Vector::load_simd(arch, worst_ptr.add(target)) };
            let target_eligible = distance_group.lt_simd(target_worst);
            let source_bits = A::Vector::active_lanes(source_eligible);
            let target_bits = A::Vector::active_lanes(target_eligible);

            if source_bits | target_bits != 0 {
                let distance_lanes = distance_group.to_lane_array();
                let distance_lanes = distance_lanes.as_ref();
                let mut source_bits = source_bits;
                while source_bits != 0 {
                    let lane = source_bits.trailing_zeros() as usize;
                    source_bits &= source_bits - 1;
                    let distance = distance_lanes[lane];
                    if distance < source_worst {
                        source_worst = insert(source, (target + lane) as u32, distance);
                    }
                }

                let mut target_bits = target_bits;
                while target_bits != 0 {
                    let lane = target_bits.trailing_zeros() as usize;
                    target_bits &= target_bits - 1;
                    let target_source = target + lane;
                    let new_worst = insert(target_source, source as u32, distance_lanes[lane]);
                    // SAFETY: `target_source < source < worst.len()`.
                    unsafe { *worst_ptr.add(target_source) = new_worst };
                }
            }
            target += A::Vector::LANES;
        }

        while target < source {
            // SAFETY: The target is in this source's strict-lower prefix.
            let distance = unsafe { *distance_flatten.get_unchecked(source_start + target) };
            if distance < source_worst {
                source_worst = insert(source, target as u32, distance);
            }
            // SAFETY: `target < source < worst.len()`.
            let target_worst = unsafe { *worst_ptr.add(target) };
            if distance < target_worst {
                let new_worst = insert(target, source as u32, distance);
                // SAFETY: `target < source < worst.len()`.
                unsafe { *worst_ptr.add(target) = new_worst };
            }
            target += 1;
        }
        // SAFETY: `source < worst.len()`.
        unsafe { *worst_ptr.add(source) = source_worst };
    }
}

/// Insert a neighbor that is closer than the current farthest neighbor.
///
/// The pair scan checks eligibility before insertion. The retained neighbors
/// stay in distance order. The result is their new farthest ranking distance.
trait NeighborInsert {
    fn insert_eligible(&mut self, candidate: LeafNeighbor) -> f32;
}

impl NeighborInsert for [LeafNeighbor; 1] {
    #[inline(always)]
    fn insert_eligible(&mut self, candidate: LeafNeighbor) -> f32 {
        self[0] = candidate;
        candidate.distance
    }
}

impl NeighborInsert for [LeafNeighbor; 2] {
    #[inline(always)]
    fn insert_eligible(&mut self, candidate: LeafNeighbor) -> f32 {
        let first = self[0];
        if candidate.distance < first.distance {
            self[0] = candidate;
            self[1] = first;
            first.distance
        } else {
            self[1] = candidate;
            candidate.distance
        }
    }
}

impl NeighborInsert for [LeafNeighbor; 3] {
    #[inline(always)]
    fn insert_eligible(&mut self, candidate: LeafNeighbor) -> f32 {
        let (first, second) = (self[0], self[1]);
        if candidate.distance < first.distance {
            self[0] = candidate;
            self[1] = first;
            self[2] = second;
            second.distance
        } else if candidate.distance < second.distance {
            self[1] = candidate;
            self[2] = second;
            second.distance
        } else {
            self[2] = candidate;
            candidate.distance
        }
    }
}

impl NeighborInsert for [LeafNeighbor] {
    #[inline(always)]
    fn insert_eligible(&mut self, candidate: LeafNeighbor) -> f32 {
        let last = self.len() - 1;
        let mut slot = last;
        while slot > 0 && candidate.distance < self[slot - 1].distance {
            self[slot] = self[slot - 1];
            slot -= 1;
        }
        self[slot] = candidate;
        self[last].distance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::pipnn::L2;
    use diskann_utils::views::{MatrixView, MutMatrixView};

    mod test_support {
        use std::cmp::Ordering;

        use super::*;
        use diskann_wide::arch::{self, Target1};

        struct KernelCall<'a> {
            distance_flatten: &'a [f32],
            point_count: usize,
            output: MutMatrixView<'a, LeafNeighbor>,
            worst: &'a mut Vec<f32>,
        }

        struct RankDistances;

        impl<A> Target1<A, (), KernelCall<'_>> for RankDistances
        where
            A: PiPNNSIMDSchema,
        {
            fn run(self, arch: A, call: KernelCall<'_>) {
                rank_leaf_distances(
                    arch,
                    call.distance_flatten,
                    call.point_count,
                    call.output,
                    call.worst,
                );
            }
        }

        pub(super) fn rank_distance_fixture(
            distances: &[f32],
            points: usize,
            output_width: usize,
        ) -> Vec<LeafNeighbor> {
            let mut output = vec![LeafNeighbor::default(); points * output_width];
            arch::dispatch1_no_features(
                RankDistances,
                KernelCall {
                    distance_flatten: distances,
                    point_count: points,
                    output: MutMatrixView::try_from(output.as_mut_slice(), points, output_width)
                        .unwrap(),
                    worst: &mut Vec::new(),
                },
            );
            output
        }

        pub(super) fn reference_neighbors(
            distances: &[f32],
            points: usize,
            width: usize,
        ) -> Vec<LeafNeighbor> {
            let mut output = vec![LeafNeighbor::default(); points * width];
            for source in 0..points {
                let mut candidates = Vec::with_capacity(points.saturating_sub(1));
                for target in 0..points {
                    if source == target {
                        continue;
                    }
                    let row = source.max(target);
                    let column = source.min(target);
                    let distance = distances[row * points + column];
                    if distance < f32::INFINITY {
                        candidates.push(LeafNeighbor::new(target as u32, distance));
                    }
                }
                candidates.sort_unstable_by(|left, right| {
                    left.distance
                        .partial_cmp(&right.distance)
                        .unwrap_or(Ordering::Equal)
                });
                let retained = candidates.len().min(width);
                output[source * width..source * width + retained]
                    .copy_from_slice(&candidates[..retained]);
            }
            output
        }

        /// Give each pair a unique distance, with alternating signs in each row.
        pub(super) fn lower_triangle_distances(points: usize) -> Vec<f32> {
            // NaN in unused entries detects reads of the diagonal or upper triangle.
            let mut distances = vec![f32::NAN; points * points];
            for source in 1..points {
                for target in 0..source {
                    let distance = (source * points + target + 1) as f32;
                    distances[source * points + target] =
                        if target % 2 == 0 { -distance } else { distance };
                }
            }
            distances
        }
    }

    mod insert_eligible_tests {
        use super::*;

        #[test]
        fn nearer_candidate_replaces_the_only_retained_neighbor() {
            // Given
            let retained_neighbor = LeafNeighbor::new(1, 4.0);
            let nearer_candidate = LeafNeighbor::new(2, 2.0);
            let expected_neighbors = [nearer_candidate];
            let expected_farthest = nearer_candidate.distance;
            let mut actual_neighbors = [retained_neighbor];

            // When
            let actual_farthest = actual_neighbors.insert_eligible(nearer_candidate);

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
            assert_eq!(actual_farthest, expected_farthest);
        }

        #[test]
        fn nearer_candidate_moves_to_the_front_of_two_retained_neighbors() {
            // Given
            let nearest = LeafNeighbor::new(1, 1.0);
            let farthest = LeafNeighbor::new(2, 3.0);
            let nearer_candidate = LeafNeighbor::new(3, 0.5);
            let expected_neighbors = [nearer_candidate, nearest];
            let expected_farthest = nearest.distance;
            let mut actual_neighbors = [nearest, farthest];

            // When
            let actual_farthest = actual_neighbors.insert_eligible(nearer_candidate);

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
            assert_eq!(actual_farthest, expected_farthest);
        }

        #[test]
        fn middle_distance_candidate_replaces_the_farther_of_two_neighbors() {
            // Given
            let nearest = LeafNeighbor::new(1, 1.0);
            let farthest = LeafNeighbor::new(2, 3.0);
            let eligible_candidate = LeafNeighbor::new(3, 2.0);
            let expected_neighbors = [nearest, eligible_candidate];
            let expected_farthest = eligible_candidate.distance;
            let mut actual_neighbors = [nearest, farthest];

            // When
            let actual_farthest = actual_neighbors.insert_eligible(eligible_candidate);

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
            assert_eq!(actual_farthest, expected_farthest);
        }

        #[test]
        fn nearest_candidate_moves_to_the_front_of_three_retained_neighbors() {
            // Given
            let nearest = LeafNeighbor::new(1, 1.0);
            let middle = LeafNeighbor::new(2, 2.0);
            let farthest = LeafNeighbor::new(3, 4.0);
            let nearer_candidate = LeafNeighbor::new(4, 0.5);
            let expected_neighbors = [nearer_candidate, nearest, middle];
            let expected_farthest = middle.distance;
            let mut actual_neighbors = [nearest, middle, farthest];

            // When
            let actual_farthest = actual_neighbors.insert_eligible(nearer_candidate);

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
            assert_eq!(actual_farthest, expected_farthest);
        }

        #[test]
        fn middle_candidate_is_inserted_between_three_retained_neighbors() {
            // Given
            let nearest = LeafNeighbor::new(1, 1.0);
            let middle = LeafNeighbor::new(2, 2.0);
            let farthest = LeafNeighbor::new(3, 4.0);
            let middle_candidate = LeafNeighbor::new(4, 1.5);
            let expected_neighbors = [nearest, middle_candidate, middle];
            let expected_farthest = middle.distance;
            let mut actual_neighbors = [nearest, middle, farthest];

            // When
            let actual_farthest = actual_neighbors.insert_eligible(middle_candidate);

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
            assert_eq!(actual_farthest, expected_farthest);
        }

        #[test]
        fn closer_candidate_replaces_the_farthest_of_three_neighbors() {
            // Given
            let nearest = LeafNeighbor::new(1, 1.0);
            let middle = LeafNeighbor::new(2, 2.0);
            let farthest = LeafNeighbor::new(3, 4.0);
            let eligible_candidate = LeafNeighbor::new(4, 3.0);
            let expected_neighbors = [nearest, middle, eligible_candidate];
            let expected_farthest = eligible_candidate.distance;
            let mut actual_neighbors = [nearest, middle, farthest];

            // When
            let actual_farthest = actual_neighbors.insert_eligible(eligible_candidate);

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
            assert_eq!(actual_farthest, expected_farthest);
        }

        #[test]
        fn middle_candidate_shifts_only_farther_runtime_neighbors() {
            // Given
            let first = LeafNeighbor::new(1, 1.0);
            let second = LeafNeighbor::new(2, 2.0);
            let third = LeafNeighbor::new(3, 3.0);
            let fourth = LeafNeighbor::new(4, 5.0);
            let candidate = LeafNeighbor::new(5, 2.5);
            let expected_neighbors = [first, second, candidate, third];
            let expected_farthest = third.distance;
            let mut actual_neighbors = [first, second, third, fourth];

            // When
            let actual_farthest = actual_neighbors.as_mut_slice().insert_eligible(candidate);

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
            assert_eq!(actual_farthest, expected_farthest);
        }
    }

    mod leaf_neighbor_count_tests {
        use super::leaf_neighbor_count;

        #[test]
        fn returns_zero_when_the_leaf_contains_only_the_source() {
            // Given
            let point_count = 1;
            let requested_k = 3;
            let expected_non_self_neighbor_count = 0;

            // When
            let actual_neighbor_count = leaf_neighbor_count(point_count, requested_k);

            // Then
            assert_eq!(actual_neighbor_count, expected_non_self_neighbor_count);
        }

        #[test]
        fn returns_the_non_self_point_count_when_requested_k_is_larger() {
            // Given
            let point_count = 4;
            let requested_k = 4;
            let expected_all_non_self_neighbors = point_count - 1;

            // When
            let actual_neighbor_count = leaf_neighbor_count(point_count, requested_k);

            // Then
            assert_eq!(actual_neighbor_count, expected_all_non_self_neighbors);
        }

        #[test]
        fn returns_requested_k_when_enough_non_self_points_exist() {
            // Given
            let point_count = 8;
            let requested_k = 5;
            let expected_requested_neighbor_count = requested_k;

            // When
            let actual_neighbor_count = leaf_neighbor_count(point_count, requested_k);

            // Then
            assert_eq!(actual_neighbor_count, expected_requested_neighbor_count);
        }
    }

    mod select_leaf_neighbors_tests {
        use super::*;

        #[test]
        fn invalid_neighbor_width_leaves_buffers_unchanged() {
            // Given
            let values = [0.0_f32, 1.0, 3.0];
            let point_count = values.len();
            let invalid_width = point_count;
            let points = MatrixView::try_from(&values[..], point_count, 1).unwrap();
            let expected_output = [LeafNeighbor::default(); 9];
            // A smaller prior leaf forces scratch growth if validation runs too late.
            let expected_distances = [99.0; 4];
            let expected_thresholds = [7.0; 2];
            let expected_error = LeafKernelError::InvalidNeighborCount {
                points: point_count,
                neighbors: invalid_width,
                maximum: point_count - 1,
            };
            let mut output = expected_output;
            let mut workspace = LeafKernelWorkspace {
                distance_scratch: expected_distances.to_vec(),
                worst: expected_thresholds.to_vec(),
            };

            // When
            let error = select_leaf_neighbors::<_, L2>(
                diskann_wide::ARCH,
                points,
                MutMatrixView::try_from(&mut output[..], point_count, invalid_width).unwrap(),
                &mut workspace,
            )
            .unwrap_err();

            // Then
            assert_eq!(
                error.downcast_ref::<LeafKernelError>(),
                Some(&expected_error)
            );
            assert_eq!(output, expected_output);
            assert_eq!(workspace.distance_scratch, expected_distances);
            assert_eq!(workspace.worst, expected_thresholds);
        }

        #[test]
        fn orders_neighbors_by_squared_distance_with_l2() {
            // Given
            let values = [0.0_f32, 1.0, 3.0, 10.0];
            let points = MatrixView::try_from(&values[..], 4, 1).unwrap();
            let expected_neighbors = [
                LeafNeighbor::new(1, (values[0] - values[1]).powi(2)),
                LeafNeighbor::new(2, (values[0] - values[2]).powi(2)),
                LeafNeighbor::new(0, (values[1] - values[0]).powi(2)),
                LeafNeighbor::new(2, (values[1] - values[2]).powi(2)),
                LeafNeighbor::new(1, (values[2] - values[1]).powi(2)),
                LeafNeighbor::new(0, (values[2] - values[0]).powi(2)),
                LeafNeighbor::new(2, (values[3] - values[2]).powi(2)),
                LeafNeighbor::new(1, (values[3] - values[1]).powi(2)),
            ];
            let mut actual_neighbors = [LeafNeighbor::default(); 8];

            // When
            select_leaf_neighbors::<_, L2>(
                diskann_wide::ARCH,
                points,
                MutMatrixView::try_from(&mut actual_neighbors[..], 4, 2).unwrap(),
                &mut LeafKernelWorkspace::default(),
            )
            .unwrap();

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn later_farther_candidate_cannot_replace_the_retained_neighbor() {
            // Given
            let point_values = [0.0_f32, 10.0, 1.0];
            let source = 2;
            let first_scanned_target = 0_u32;
            let later_farther_target = 1_usize;
            let expected_distance =
                (point_values[source] - point_values[first_scanned_target as usize]).powi(2);
            let later_distance =
                (point_values[source] - point_values[later_farther_target]).powi(2);
            let expected_nearest_neighbor =
                LeafNeighbor::new(first_scanned_target, expected_distance);
            assert!(later_distance > expected_distance);
            let mut actual_neighbors = [LeafNeighbor::default(); 3];

            // When
            select_leaf_neighbors::<_, L2>(
                diskann_wide::ARCH,
                MatrixView::try_from(&point_values[..], 3, 1).unwrap(),
                MutMatrixView::try_from(&mut actual_neighbors[..], 3, 1).unwrap(),
                &mut LeafKernelWorkspace::default(),
            )
            .unwrap();

            // Then
            assert_eq!(actual_neighbors[source], expected_nearest_neighbor);
        }

        #[test]
        fn reused_workspace_matches_fresh_neighbor_selection() {
            // Given
            let values = [0.0_f32, 1.0, 3.0, 10.0];
            let smaller_points = MatrixView::try_from(&values[..3], 3, 1).unwrap();
            let mut reused_workspace = LeafKernelWorkspace::default();
            let mut discarded_large_output = [LeafNeighbor::default(); 8];
            select_leaf_neighbors::<_, L2>(
                diskann_wide::ARCH,
                MatrixView::try_from(&values[..], 4, 1).unwrap(),
                MutMatrixView::try_from(&mut discarded_large_output[..], 4, 2).unwrap(),
                &mut reused_workspace,
            )
            .unwrap();
            let mut expected_neighbors_from_fresh_workspace = [LeafNeighbor::default(); 6];
            select_leaf_neighbors::<_, L2>(
                diskann_wide::ARCH,
                smaller_points,
                MutMatrixView::try_from(&mut expected_neighbors_from_fresh_workspace[..], 3, 2)
                    .unwrap(),
                &mut LeafKernelWorkspace::default(),
            )
            .unwrap();

            // When
            let mut actual_neighbors_from_reused_workspace = [LeafNeighbor::default(); 6];
            select_leaf_neighbors::<_, L2>(
                diskann_wide::ARCH,
                smaller_points,
                MutMatrixView::try_from(&mut actual_neighbors_from_reused_workspace[..], 3, 2)
                    .unwrap(),
                &mut reused_workspace,
            )
            .unwrap();

            // Then
            assert_eq!(
                actual_neighbors_from_reused_workspace,
                expected_neighbors_from_fresh_workspace
            );
        }
    }

    mod rank_leaf_distances_tests {
        use super::test_support::*;
        use super::*;
        use rstest::rstest;

        #[rstest]
        #[case::two_points_fixed_one(2, 1)]
        #[case::scalar_fixed_two(7, 2)]
        #[case::lane_minus_one_fixed_three(16, 3)]
        #[case::one_complete_lane_fixed_one(17, 1)]
        #[case::one_complete_lane_fixed_two(17, 2)]
        #[case::one_complete_lane_fixed_three(17, 3)]
        #[case::lane_plus_one_runtime_width(18, 4)]
        #[case::all_non_self_neighbors(17, 16)]
        #[case::two_lanes_minus_one_runtime_width(32, 7)]
        #[case::two_complete_lanes_runtime_width(33, 7)]
        #[case::two_lanes_plus_one_runtime_width(34, 7)]
        #[case::four_complete_lanes_runtime_width(65, 7)]
        #[case::sixteen_complete_lanes_runtime_width(257, 7)]
        #[case::maximum_leaf_size_runtime_width(512, 7)]
        #[trace]
        fn dispatched_leaf_ranking_matches_scalar_reference_across_lane_boundaries(
            #[case] point_count: usize,
            #[case] requested_k: usize,
        ) {
            // Miri covers the pointer boundaries in the smaller lane cases.
            if cfg!(miri) && point_count > 64 {
                return;
            }

            // Given: the last source row has `point_count - 1` distances.
            let distances = lower_triangle_distances(point_count);
            let expected_neighbors = reference_neighbors(&distances, point_count, requested_k);

            // When
            let actual_neighbors = rank_distance_fixture(&distances, point_count, requested_k);

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn only_the_lower_triangle_supplies_neighbor_distances() {
            // Given
            #[rustfmt::skip]
            let distances = [
                f32::NAN, f32::NAN, f32::NAN,
                3.0,      f32::NAN, f32::NAN,
                1.0,      2.0,      f32::NAN,
            ];
            let expected_neighbors = [
                LeafNeighbor::new(2, 1.0),
                LeafNeighbor::new(2, 2.0),
                LeafNeighbor::new(0, 1.0),
            ];

            // When
            let actual_neighbors = rank_distance_fixture(&distances, 3, 1);

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[rstest]
        #[case::finite_negative(-f32::EPSILON)]
        #[case::negative_infinity(f32::NEG_INFINITY)]
        fn negative_distances_remain_rankable(#[case] distance: f32) {
            // Given
            let distances = [f32::NAN, f32::NAN, distance, f32::NAN];
            let expected = [
                LeafNeighbor::new(1, distance),
                LeafNeighbor::new(0, distance),
            ];

            // When
            let actual = rank_distance_fixture(&distances, 2, 1);

            // Then
            assert_eq!(actual, expected);
        }

        #[test]
        fn tied_distances_fill_capacity_with_distinct_non_self_neighbors() {
            // Given: each point has three equally distant candidates for two slots.
            let point_count = 4;
            let width = 2;
            let distances = [1.0; 16];

            // When
            let actual = rank_distance_fixture(&distances, point_count, width);

            // Then: any two candidates are valid; no tie order is required.
            for (source, neighbors) in actual.chunks_exact(width).enumerate() {
                assert_eq!(neighbors[0].distance, 1.0);
                assert_eq!(neighbors[1].distance, 1.0);
                assert_ne!(neighbors[0].target, neighbors[1].target);
                for neighbor in neighbors {
                    assert!(neighbor.target < point_count as u32);
                    assert_ne!(neighbor.target, source as u32);
                }
            }
        }

        #[test]
        fn f32_max_distance_is_still_a_rankable_neighbor() {
            // Given
            let point_count = 4;
            let width = 3;
            let source = 3;
            let mut distances = [1.0; 16];
            distances[source * point_count] = f32::MAX;
            let expected_last_neighbor = LeafNeighbor::new(0, f32::MAX);

            // When
            let actual = rank_distance_fixture(&distances, point_count, width);

            // Then
            assert_eq!(actual[source * width + width - 1], expected_last_neighbor);
        }

        #[rstest]
        #[case::nan(f32::NAN)]
        #[case::positive_infinity(f32::INFINITY)]
        fn non_rankable_distances_leave_neighbor_slots_unassigned(
            #[case] distance: f32,
            #[values(2, 17, 18)] point_count: usize,
        ) {
            // Given
            let distances = vec![distance; point_count * point_count];
            let expected = vec![LeafNeighbor::default(); point_count];

            // When
            let actual = rank_distance_fixture(&distances, point_count, 1);

            // Then
            assert_eq!(actual, expected);
        }

        #[rstest]
        #[case::nan(f32::NAN)]
        #[case::positive_infinity(f32::INFINITY)]
        fn non_rankable_candidates_cannot_replace_finite_neighbors(
            #[case] distance: f32,
            #[values(17, 18)] point_count: usize,
        ) {
            // Given: the last candidate is in a full SIMD group or its scalar tail.
            let source = point_count - 1;
            let invalid_target = source - 1;
            let mut distances = vec![1.0; point_count * point_count];
            distances[source * point_count + invalid_target] = distance;

            // When
            let actual = rank_distance_fixture(&distances, point_count, 1);

            // Then
            assert!(actual[source].target < invalid_target as u32);
            assert_eq!(actual[source].distance, 1.0);
        }

        #[test]
        fn scalar_tail_updates_both_endpoints_after_a_rejected_simd_group() {
            // Given: the final pair is the only rankable pair in the leaf.
            let point_count = 18;
            let source = 17;
            let target = 16;
            let mut distances = vec![f32::INFINITY; point_count * point_count];
            distances[source * point_count + target] = -1.0;
            let mut expected = vec![LeafNeighbor::default(); point_count];
            expected[source] = LeafNeighbor::new(target as u32, -1.0);
            expected[target] = LeafNeighbor::new(source as u32, -1.0);

            // When
            let actual = rank_distance_fixture(&distances, point_count, 1);

            // Then
            assert_eq!(actual, expected);
        }

        #[test]
        fn singleton_leaf_has_no_neighbors() {
            // Given
            let distances = [f32::NAN];
            let expected: [LeafNeighbor; 0] = [];

            // When
            let actual = rank_distance_fixture(&distances, 1, 0);

            // Then
            assert_eq!(actual, expected);
        }
    }
}
