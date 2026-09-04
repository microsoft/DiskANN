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
//! Strict comparisons keep scan order for equal distances. They do not rank NaN.
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

/// One leaf-local neighbor and its metric distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct LeafNeighbor {
    /// Target position in the leaf, not a dataset ID.
    pub(super) target: u32,
    /// Distance from the source point to `target`.
    pub(super) distance: f32,
}

impl LeafNeighbor {
    /// Construct a leaf-local neighbor.
    ///
    /// `target` is a position in the leaf. `distance` is measured from the
    /// source point of the output row.
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

/// Validation error returned by the distance-ranking loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(super) enum LeafKernelError {
    /// A source requests more neighbors than the leaf or fixed kernel supports.
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
    let distance_count = point_count * point_count;
    let LeafKernelWorkspace {
        distance_scratch,
        worst,
    } = workspace;
    if distance_scratch.len() < distance_count {
        distance_scratch.resize(distance_count, 0.0);
    }
    M::compute_distances(arch, points, &mut distance_scratch[..distance_count])?;
    rank_leaf_distances(
        arch,
        &distance_scratch[..distance_count],
        point_count,
        output,
        worst,
    )
    .map_err(ANNError::new)
}

/// Rank one flattened lower-triangle buffer.
///
/// `distance_flatten` contains `point_count * point_count` elements. The metric
/// initializes each strict-lower entry. The kernel does not read the upper triangle.
fn rank_leaf_distances<A>(
    arch: A,
    distance_flatten: &[f32],
    point_count: usize,
    mut output: MutMatrixView<'_, LeafNeighbor>,
    worst: &mut Vec<f32>,
) -> Result<(), LeafKernelError>
where
    A: PiPNNSIMDSchema,
{
    validate_neighbor_count(point_count, &output)?;
    let neighbor_count = output.ncols();
    if neighbor_count == 0 {
        return Ok(());
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
    Ok(())
}

/// Check the safety conditions for the SIMD kernel.
///
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
    [LeafNeighbor; N]: SortedInsert<LeafNeighbor>,
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
                insert_eligible_neighbor(&mut rows[source], target, distance)
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
                insert_eligible_neighbor(&mut output[first..first + width], target, distance)
            },
        );
    });
}

/// Select neighbors from all unordered point pairs in one leaf.
///
/// The function reads the strict lower triangle once. It offers each distance to
/// both endpoint lists. SIMD groups and scalar tails preserve pair scan order.
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

/// Insert one value that the caller has already found eligible.
///
/// The caller must prove that `value` precedes the current last retained value.
/// This method intentionally does not repeat that check. A caller that violates
/// the precondition replaces a valid retained value and corrupts the top-k set.
/// The return value is the new last retained value.
trait SortedInsert<T: Copy> {
    fn insert_eligible_sorted_by(&mut self, value: T, precedes: impl Fn(T, T) -> bool) -> T;
}

impl<T: Copy> SortedInsert<T> for [T; 1] {
    #[inline(always)]
    fn insert_eligible_sorted_by(&mut self, value: T, _precedes: impl Fn(T, T) -> bool) -> T {
        self[0] = value;
        value
    }
}

impl<T: Copy> SortedInsert<T> for [T; 2] {
    #[inline(always)]
    fn insert_eligible_sorted_by(&mut self, value: T, precedes: impl Fn(T, T) -> bool) -> T {
        let first = self[0];
        if precedes(value, first) {
            self[0] = value;
            self[1] = first;
            first
        } else {
            self[1] = value;
            value
        }
    }
}

impl<T: Copy> SortedInsert<T> for [T; 3] {
    #[inline(always)]
    fn insert_eligible_sorted_by(&mut self, value: T, precedes: impl Fn(T, T) -> bool) -> T {
        let (first, second) = (self[0], self[1]);
        if precedes(value, first) {
            self[0] = value;
            self[1] = first;
            self[2] = second;
            second
        } else if precedes(value, second) {
            self[1] = value;
            self[2] = second;
            second
        } else {
            self[2] = value;
            value
        }
    }
}

impl<T: Copy> SortedInsert<T> for [T] {
    #[inline(always)]
    fn insert_eligible_sorted_by(&mut self, value: T, precedes: impl Fn(T, T) -> bool) -> T {
        let last = self.len() - 1;
        let mut slot = last;
        while slot > 0 && precedes(value, self[slot - 1]) {
            self[slot] = self[slot - 1];
            slot -= 1;
        }
        self[slot] = value;
        self[last]
    }
}

/// Insert one candidate that the caller has already found nearer than the current farthest.
///
/// This function intentionally does not reject an ineligible candidate. The pair
/// scan owns the eligibility check so it can filter SIMD lanes before insertion.
/// The return value is the new farthest retained distance.
#[inline(always)]
fn insert_eligible_neighbor<R>(neighbors: &mut R, target: u32, distance: f32) -> f32
where
    R: SortedInsert<LeafNeighbor> + ?Sized,
{
    neighbors
        .insert_eligible_sorted_by(
            LeafNeighbor::new(target, distance),
            |candidate, retained| candidate.distance < retained.distance,
        )
        .distance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::pipnn::{Cosine, CosineNormalized, InnerProduct, L2};
    use diskann_utils::views::{MatrixView, MutMatrixView};
    use diskann_vector::distance::Metric;

    mod test_support {
        use std::cmp::Ordering;

        use super::*;
        use diskann_wide::arch::{self, Target1};

        pub(super) trait TestMetric {
            const METRIC: Metric;
        }

        impl TestMetric for L2 {
            const METRIC: Metric = Metric::L2;
        }

        impl TestMetric for Cosine {
            const METRIC: Metric = Metric::Cosine;
        }

        impl TestMetric for CosineNormalized {
            const METRIC: Metric = Metric::CosineNormalized;
        }

        impl TestMetric for InnerProduct {
            const METRIC: Metric = Metric::InnerProduct;
        }

        struct KernelCall<'a> {
            distance_flatten: &'a [f32],
            point_count: usize,
            output: MutMatrixView<'a, LeafNeighbor>,
            worst: &'a mut Vec<f32>,
        }

        struct RankDistances;

        impl<A> Target1<A, Result<(), LeafKernelError>, KernelCall<'_>> for RankDistances
        where
            A: PiPNNSIMDSchema,
        {
            fn run(self, arch: A, call: KernelCall<'_>) -> Result<(), LeafKernelError> {
                rank_leaf_distances(
                    arch,
                    call.distance_flatten,
                    call.point_count,
                    call.output,
                    call.worst,
                )
            }
        }

        fn reference_distance(
            metric: Metric,
            dot: f32,
            source_diagonal: f32,
            target_diagonal: f32,
        ) -> f32 {
            match metric {
                Metric::L2 => (-2.0_f32).mul_add(dot, source_diagonal) + target_diagonal,
                Metric::CosineNormalized => -dot,
                Metric::InnerProduct => -dot,
                Metric::Cosine => {
                    let source_norm = source_diagonal.sqrt();
                    let target_norm = target_diagonal.sqrt();
                    if source_norm < f32::MIN_POSITIVE.sqrt()
                        || target_norm < f32::MIN_POSITIVE.sqrt()
                    {
                        1.0
                    } else {
                        1.0 - (dot / (source_norm * target_norm)).clamp(-1.0, 1.0)
                    }
                }
            }
        }

        fn distance_flatten(metric: Metric, dots: &[f32], points: usize) -> Vec<f32> {
            let mut distances = vec![0.0; points * points];
            for source in 0..points {
                for target in 0..=source {
                    distances[source * points + target] = reference_distance(
                        metric,
                        dots[source * points + target],
                        dots[source * points + source],
                        dots[target * points + target],
                    );
                }
            }
            distances
        }

        fn rank_with_output_width(
            metric: Metric,
            dots: &[f32],
            points: usize,
            output_width: usize,
            workspace: &mut LeafKernelWorkspace,
        ) -> Result<Vec<LeafNeighbor>, LeafKernelError> {
            let distance_flatten = distance_flatten(metric, dots, points);
            let mut output = vec![LeafNeighbor::default(); points * output_width];
            arch::dispatch1_no_features(
                RankDistances,
                KernelCall {
                    distance_flatten: &distance_flatten,
                    point_count: points,
                    output: MutMatrixView::try_from(output.as_mut_slice(), points, output_width)
                        .unwrap(),
                    worst: &mut workspace.worst,
                },
            )?;
            Ok(output)
        }

        pub(super) fn rank_distance_fixture<M: TestMetric>(
            dots: &[f32],
            points: usize,
            output_width: usize,
        ) -> Result<Vec<LeafNeighbor>, LeafKernelError> {
            rank_with_output_width(
                M::METRIC,
                dots,
                points,
                output_width,
                &mut LeafKernelWorkspace::default(),
            )
        }

        fn run_with_workspace(
            metric: Metric,
            dots: &[f32],
            points: usize,
            requested_k: usize,
            workspace: &mut LeafKernelWorkspace,
        ) -> (usize, Vec<LeafNeighbor>) {
            let leaf_k = leaf_neighbor_count(points, requested_k);
            let output = rank_with_output_width(metric, dots, points, leaf_k, workspace)
                .expect("valid leaf neighbor width");
            (leaf_k, output)
        }

        pub(super) fn run_rank_leaf_distances(
            metric: Metric,
            dots: &[f32],
            points: usize,
            requested_k: usize,
        ) -> (usize, Vec<LeafNeighbor>) {
            run_with_workspace(
                metric,
                dots,
                points,
                requested_k,
                &mut LeafKernelWorkspace::default(),
            )
        }

        pub(super) fn reference_neighbors(
            metric: Metric,
            dots: &[f32],
            points: usize,
            requested_k: usize,
        ) -> Vec<LeafNeighbor> {
            let leaf_k = requested_k.min(points.saturating_sub(1));
            let mut output = vec![LeafNeighbor::default(); points * leaf_k];
            for source in 0..points {
                let mut candidates = Vec::with_capacity(points.saturating_sub(1));
                for target in 0..points {
                    if source == target {
                        continue;
                    }
                    let (row, column) = if source > target {
                        (source, target)
                    } else {
                        (target, source)
                    };
                    let distance = reference_distance(
                        metric,
                        dots[row * points + column],
                        dots[source * points + source],
                        dots[target * points + target],
                    );
                    if distance.partial_cmp(&f32::INFINITY) == Some(Ordering::Less) {
                        candidates.push(LeafNeighbor::new(target as u32, distance));
                    }
                }
                candidates.sort_by(|left, right| left.distance.total_cmp(&right.distance));
                let retained = candidates.len().min(leaf_k);
                output[source * leaf_k..source * leaf_k + retained]
                    .copy_from_slice(&candidates[..retained]);
            }
            output
        }

        /// Build a Gram matrix from points on the line `x = 1`.
        pub(super) fn lane_boundary_gram_from_point_vectors(
            metric: Metric,
            points: usize,
        ) -> Vec<f32> {
            let denominator = (points + 1) as f32;
            let point_vectors: Vec<_> = (0..points)
                .map(|point| {
                    let vector = [1.0, (point + 1) as f32 / denominator];
                    if metric == Metric::CosineNormalized {
                        let norm = vector[0].hypot(vector[1]);
                        [vector[0] / norm, vector[1] / norm]
                    } else {
                        vector
                    }
                })
                .collect();
            let mut gram = vec![0.0; points * points];
            for source in 0..points {
                for target in 0..points {
                    gram[source * points + target] = point_vectors[source][0]
                        * point_vectors[target][0]
                        + point_vectors[source][1] * point_vectors[target][1];
                }
            }
            gram
        }

        pub(super) fn gram_with_uniform_self_dots(points: usize, self_dot: f32) -> Vec<f32> {
            let mut gram = vec![0.0; points * points];
            for point in 0..points {
                gram[point * points + point] = self_dot;
            }
            gram
        }
    }

    mod insert_eligible_neighbor_tests {
        use super::*;

        #[test]
        fn nearer_candidate_replaces_the_only_retained_neighbor() {
            // Given
            let retained_neighbor = LeafNeighbor::new(1, 4.0);
            let nearer_candidate = LeafNeighbor::new(2, 2.0);
            let expected_neighbors = [nearer_candidate];
            let mut actual_neighbors = [retained_neighbor];

            // When
            insert_eligible_neighbor(
                &mut actual_neighbors,
                nearer_candidate.target,
                nearer_candidate.distance,
            );

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn direct_call_does_not_recheck_candidate_eligibility() {
            // Given: deliberately bypass the pair scan's eligibility check.
            let nearest = LeafNeighbor::new(1, 1.0);
            let current_farthest = LeafNeighbor::new(2, 3.0);
            let ineligible_farther_candidate = LeafNeighbor::new(3, 5.0);
            let expected_unchecked_result = [nearest, ineligible_farther_candidate];
            let mut actual_neighbors = [nearest, current_farthest];

            // When
            insert_eligible_neighbor(
                &mut actual_neighbors,
                ineligible_farther_candidate.target,
                ineligible_farther_candidate.distance,
            );

            // Then
            assert_eq!(actual_neighbors, expected_unchecked_result);
        }

        #[test]
        fn nearer_candidate_moves_to_the_front_of_two_retained_neighbors() {
            // Given
            let nearest = LeafNeighbor::new(1, 1.0);
            let farthest = LeafNeighbor::new(2, 3.0);
            let nearer_candidate = LeafNeighbor::new(3, 0.5);
            let expected_neighbors = [nearer_candidate, nearest];
            let mut actual_neighbors = [nearest, farthest];

            // When
            insert_eligible_neighbor(
                &mut actual_neighbors,
                nearer_candidate.target,
                nearer_candidate.distance,
            );

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn middle_distance_candidate_replaces_the_farther_of_two_neighbors() {
            // Given
            let nearest = LeafNeighbor::new(1, 1.0);
            let farthest = LeafNeighbor::new(2, 3.0);
            let eligible_candidate = LeafNeighbor::new(3, 2.0);
            let expected_neighbors = [nearest, eligible_candidate];
            let mut actual_neighbors = [nearest, farthest];

            // When
            insert_eligible_neighbor(
                &mut actual_neighbors,
                eligible_candidate.target,
                eligible_candidate.distance,
            );

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn nearest_candidate_moves_to_the_front_of_three_retained_neighbors() {
            // Given
            let nearest = LeafNeighbor::new(1, 1.0);
            let middle = LeafNeighbor::new(2, 2.0);
            let farthest = LeafNeighbor::new(3, 4.0);
            let nearer_candidate = LeafNeighbor::new(4, 0.5);
            let expected_neighbors = [nearer_candidate, nearest, middle];
            let mut actual_neighbors = [nearest, middle, farthest];

            // When
            insert_eligible_neighbor(
                &mut actual_neighbors,
                nearer_candidate.target,
                nearer_candidate.distance,
            );

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn middle_candidate_is_inserted_between_three_retained_neighbors() {
            // Given
            let nearest = LeafNeighbor::new(1, 1.0);
            let middle = LeafNeighbor::new(2, 2.0);
            let farthest = LeafNeighbor::new(3, 4.0);
            let middle_candidate = LeafNeighbor::new(4, 1.5);
            let expected_neighbors = [nearest, middle_candidate, middle];
            let mut actual_neighbors = [nearest, middle, farthest];

            // When
            insert_eligible_neighbor(
                &mut actual_neighbors,
                middle_candidate.target,
                middle_candidate.distance,
            );

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn closer_candidate_replaces_the_farthest_of_three_neighbors() {
            // Given
            let nearest = LeafNeighbor::new(1, 1.0);
            let middle = LeafNeighbor::new(2, 2.0);
            let farthest = LeafNeighbor::new(3, 4.0);
            let eligible_candidate = LeafNeighbor::new(4, 3.0);
            let expected_neighbors = [nearest, middle, eligible_candidate];
            let mut actual_neighbors = [nearest, middle, farthest];

            // When
            insert_eligible_neighbor(
                &mut actual_neighbors,
                eligible_candidate.target,
                eligible_candidate.distance,
            );

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
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
            let mut actual_neighbors = [first, second, third, fourth];

            // When
            insert_eligible_neighbor(
                actual_neighbors.as_mut_slice(),
                candidate.target,
                candidate.distance,
            );

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn equal_distance_candidate_stays_after_the_existing_neighbor() {
            // Given
            let nearest = LeafNeighbor::new(1, 1.0);
            let existing_tie = LeafNeighbor::new(2, 2.0);
            let farthest = LeafNeighbor::new(3, 4.0);
            let tied_candidate = LeafNeighbor::new(4, 2.0);
            let expected_neighbors = [nearest, existing_tie, tied_candidate];
            let mut actual_neighbors = [nearest, existing_tie, farthest];

            // When
            insert_eligible_neighbor(
                &mut actual_neighbors,
                tied_candidate.target,
                tied_candidate.distance,
            );

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
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
        #[case::lane_minus_one_fixed_three(15, 3)]
        #[case::one_complete_lane_fixed_three(16, 3)]
        #[case::lane_plus_one_runtime_width(17, 4)]
        #[case::two_lanes_minus_one_runtime_width(31, 7)]
        #[case::two_complete_lanes_runtime_width(32, 7)]
        #[case::two_lanes_plus_one_runtime_width(33, 7)]
        #[case::four_complete_lanes_runtime_width(64, 7)]
        #[case::sixteen_complete_lanes_runtime_width(256, 7)]
        #[case::maximum_leaf_size_runtime_width(512, 7)]
        #[trace]
        fn dispatched_leaf_ranking_matches_scalar_reference_across_lane_boundaries(
            #[values(
                Metric::L2,
                Metric::Cosine,
                Metric::CosineNormalized,
                Metric::InnerProduct
            )]
            metric: Metric,
            #[case] point_count: usize,
            #[case] requested_k: usize,
        ) {
            // Miri covers the pointer boundaries in the smaller lane cases.
            if cfg!(miri) && point_count > 64 {
                return;
            }

            // Given
            let dots = lane_boundary_gram_from_point_vectors(metric, point_count);
            let expected_neighbors = reference_neighbors(metric, &dots, point_count, requested_k);

            // When
            let actual_neighbors =
                run_rank_leaf_distances(metric, &dots, point_count, requested_k).1;

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        const POINT_ZERO: [f32; 2] = [1.0, 0.0];
        const POINT_ONE: [f32; 2] = [0.0, 1.0];
        const POINT_TWO: [f32; 2] = [0.6, 0.8];

        fn point_dot(left: [f32; 2], right: [f32; 2]) -> f32 {
            left[0] * right[0] + left[1] * right[1]
        }

        fn three_unit_point_gram() -> [f32; 9] {
            let points = [POINT_ZERO, POINT_ONE, POINT_TWO];
            std::array::from_fn(|index| {
                let source = index / points.len();
                let target = index % points.len();
                point_dot(points[source], points[target])
            })
        }

        #[test]
        fn selects_the_smallest_squared_distance_neighbor_for_each_point_with_l2() {
            // Given
            let point_zero_two_distance = point_dot(POINT_ZERO, POINT_ZERO)
                + point_dot(POINT_TWO, POINT_TWO)
                - 2.0 * point_dot(POINT_ZERO, POINT_TWO);
            let point_one_two_distance = point_dot(POINT_ONE, POINT_ONE)
                + point_dot(POINT_TWO, POINT_TWO)
                - 2.0 * point_dot(POINT_ONE, POINT_TWO);
            let expected_neighbors = [
                LeafNeighbor::new(2, point_zero_two_distance),
                LeafNeighbor::new(2, point_one_two_distance),
                LeafNeighbor::new(1, point_one_two_distance),
            ];
            let gram = three_unit_point_gram();

            // When
            let actual_neighbors = rank_distance_fixture::<L2>(&gram, 3, 1).unwrap();

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn selects_the_highest_similarity_neighbor_for_each_point_with_cosine() {
            // Given
            let expected_neighbors = [
                LeafNeighbor::new(2, 1.0 - point_dot(POINT_ZERO, POINT_TWO)),
                LeafNeighbor::new(2, 1.0 - point_dot(POINT_ONE, POINT_TWO)),
                LeafNeighbor::new(1, 1.0 - point_dot(POINT_ONE, POINT_TWO)),
            ];
            let gram = three_unit_point_gram();

            // When
            let actual_neighbors = rank_distance_fixture::<Cosine>(&gram, 3, 1).unwrap();

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn selects_the_highest_dot_product_neighbor_for_each_point_with_normalized_cosine() {
            // Given
            let expected_neighbors = [
                LeafNeighbor::new(2, -point_dot(POINT_ZERO, POINT_TWO)),
                LeafNeighbor::new(2, -point_dot(POINT_ONE, POINT_TWO)),
                LeafNeighbor::new(1, -point_dot(POINT_ONE, POINT_TWO)),
            ];
            let gram = three_unit_point_gram();

            // When
            let actual_neighbors = rank_distance_fixture::<CosineNormalized>(&gram, 3, 1).unwrap();

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn selects_the_highest_dot_product_neighbor_for_each_point_with_inner_product() {
            // Given
            let expected_neighbors = [
                LeafNeighbor::new(2, -point_dot(POINT_ZERO, POINT_TWO)),
                LeafNeighbor::new(2, -point_dot(POINT_ONE, POINT_TWO)),
                LeafNeighbor::new(1, -point_dot(POINT_ONE, POINT_TWO)),
            ];
            let gram = three_unit_point_gram();

            // When
            let actual_neighbors = rank_distance_fixture::<InnerProduct>(&gram, 3, 1).unwrap();

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn equal_distances_keep_target_scan_order_with_l2() {
            // Given
            let unit_squared_norm = 1.0;
            let tied_dot_product = 0.0;
            let expected_tied_distance = 2.0 * unit_squared_norm - 2.0 * tied_dot_product;
            // This is the Gram matrix of four orthogonal unit vectors.
            #[rustfmt::skip]
            let gram = [
                unit_squared_norm, tied_dot_product,  tied_dot_product,  tied_dot_product,
                tied_dot_product,  unit_squared_norm, tied_dot_product,  tied_dot_product,
                tied_dot_product,  tied_dot_product,  unit_squared_norm, tied_dot_product,
                tied_dot_product,  tied_dot_product,  tied_dot_product,  unit_squared_norm,
            ];
            let expected_neighbors_in_scan_order = [
                LeafNeighbor::new(1, expected_tied_distance),
                LeafNeighbor::new(2, expected_tied_distance),
                LeafNeighbor::new(0, expected_tied_distance),
                LeafNeighbor::new(2, expected_tied_distance),
                LeafNeighbor::new(0, expected_tied_distance),
                LeafNeighbor::new(1, expected_tied_distance),
                LeafNeighbor::new(0, expected_tied_distance),
                LeafNeighbor::new(1, expected_tied_distance),
            ];

            // When
            let actual_neighbors = rank_distance_fixture::<L2>(&gram, 4, 2).unwrap();

            // Then
            assert_eq!(actual_neighbors, expected_neighbors_in_scan_order);
        }

        #[test]
        fn finite_negative_l2_ranking_remains_rankable() {
            // Given
            let point_count = 2;
            let nearest_neighbor_count = 1;
            let self_dot = 1.0_f32;
            let dot_roundoff = f32::EPSILON;
            let cross_dot = self_dot + dot_roundoff;
            let expected_distance = (-2.0_f32).mul_add(cross_dot, self_dot) + self_dot;
            let gram = [self_dot, cross_dot, cross_dot, self_dot];
            let expected = [
                LeafNeighbor::new(1, expected_distance),
                LeafNeighbor::new(0, expected_distance),
            ];

            // When
            let actual =
                rank_distance_fixture::<L2>(&gram, point_count, nearest_neighbor_count).unwrap();

            // Then
            assert!(expected_distance < 0.0);
            assert_eq!(actual, expected);
        }

        #[test]
        fn scalar_distance_stays_finite_when_twice_the_dot_product_overflows_with_l2() {
            // Given
            let dot_product = f32::from_bits(f32::MAX.to_bits() - 1);
            let unfused_twice_dot_product = 2.0 * dot_product;
            let expected_fused_distance = (-2.0_f32).mul_add(dot_product, f32::MAX) + f32::MAX;
            let gram = [f32::MAX, dot_product, dot_product, f32::MAX];

            // When
            let actual_neighbors = rank_distance_fixture::<L2>(&gram, 2, 1).unwrap();

            // Then
            assert!(unfused_twice_dot_product.is_infinite());
            assert!(expected_fused_distance.is_finite() && expected_fused_distance > 0.0);
            assert_eq!(
                actual_neighbors[0].distance.to_bits(),
                expected_fused_distance.to_bits()
            );
        }

        #[test]
        fn simd_distance_stays_finite_when_twice_the_dot_product_overflows_with_l2() {
            // Given
            let dot_product = f32::from_bits(f32::MAX.to_bits() - 1);
            let unfused_twice_dot_product = 2.0 * dot_product;
            let expected_fused_distance = (-2.0_f32).mul_add(dot_product, f32::MAX) + f32::MAX;
            let expected_simd_neighbor_target = 0;
            let points = 17;
            let mut gram = gram_with_uniform_self_dots(points, f32::MAX);
            gram[16 * points] = dot_product;
            gram[16] = dot_product;

            // When
            let actual_neighbors = run_rank_leaf_distances(Metric::L2, &gram, points, 1).1;

            // Then
            assert!(unfused_twice_dot_product.is_infinite());
            assert_eq!(actual_neighbors[16].target, expected_simd_neighbor_target);
            assert_eq!(
                actual_neighbors[16].distance.to_bits(),
                expected_fused_distance.to_bits()
            );
        }

        #[test]
        fn zero_norm_produces_unit_distance_with_cosine() {
            // Given
            // This is the Gram matrix of one zero vector and two orthogonal unit vectors.
            #[rustfmt::skip]
            let gram = [
                0.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ];
            let expected_zero_norm_neighbors =
                [LeafNeighbor::new(1, 1.0), LeafNeighbor::new(2, 1.0)];

            // When
            let actual_neighbors = rank_distance_fixture::<Cosine>(&gram, 3, 2).unwrap();

            // Then
            assert_eq!(&actual_neighbors[..2], &expected_zero_norm_neighbors);
        }

        #[test]
        fn zero_target_norm_remains_rankable_in_a_complete_simd_group_with_cosine() {
            // Given
            let points = 17;
            let mut gram = gram_with_uniform_self_dots(points, 1.0);
            gram[0] = 0.0;
            let expected_zero_norm_neighbor = LeafNeighbor::new(0, 1.0);

            // When
            let actual_neighbors = run_rank_leaf_distances(Metric::Cosine, &gram, points, 1).1;

            // Then
            assert_eq!(actual_neighbors[16], expected_zero_norm_neighbor);
        }

        #[test]
        fn similarity_above_one_clamps_to_zero_distance_with_cosine() {
            // Given
            // A small excess models dot-product roundoff above cosine similarity one.
            let rounded_dot_product = 1.000_001;
            let gram = [1.0, rounded_dot_product, rounded_dot_product, 1.0];
            let maximum_cosine_similarity = 1.0;
            let expected_one_minus_maximum_similarity = 1.0 - maximum_cosine_similarity;

            // When
            let actual_neighbors = rank_distance_fixture::<Cosine>(&gram, 2, 1).unwrap();

            // Then
            assert_eq!(
                actual_neighbors[0].distance,
                expected_one_minus_maximum_similarity
            );
        }

        #[test]
        fn similarity_below_negative_one_clamps_to_distance_two_with_cosine() {
            // Given
            // A small excess models dot-product roundoff below cosine similarity minus one.
            let rounded_dot_product = -1.000_001;
            let gram = [1.0, rounded_dot_product, rounded_dot_product, 1.0];
            let minimum_cosine_similarity = -1.0;
            let expected_one_minus_minimum_similarity = 1.0 - minimum_cosine_similarity;

            // When
            let actual_neighbors = rank_distance_fixture::<Cosine>(&gram, 2, 1).unwrap();

            // Then
            assert_eq!(
                actual_neighbors[0].distance,
                expected_one_minus_minimum_similarity
            );
        }

        #[test]
        fn subnormal_norm_is_treated_as_zero_with_cosine() {
            // Given
            let subnormal_self_dot = f32::MIN_POSITIVE / 2.0;
            let gram = [subnormal_self_dot, 0.0, 0.0, 1.0];
            let zero_norm_similarity = 0.0;
            let expected_one_minus_zero_similarity = 1.0 - zero_norm_similarity;

            // When
            let actual_neighbors = rank_distance_fixture::<Cosine>(&gram, 2, 1).unwrap();

            // Then
            assert_eq!(
                actual_neighbors[0].distance,
                expected_one_minus_zero_similarity
            );
        }

        #[test]
        fn f32_max_distance_is_still_a_rankable_neighbor() {
            // Given
            let points = 4;
            let expected_leaf_k = 3;
            let expected_last_neighbor = LeafNeighbor::new(0, f32::MAX);
            let mut gram = gram_with_uniform_self_dots(points, f32::MAX);
            gram[3 * points] = -f32::MAX;
            gram[3] = -f32::MAX;

            // When
            let actual_neighbors =
                rank_distance_fixture::<InnerProduct>(&gram, points, expected_leaf_k).unwrap();

            // Then
            assert_eq!(
                actual_neighbors[3 * expected_leaf_k + expected_leaf_k - 1],
                expected_last_neighbor
            );
        }

        #[test]
        fn scalar_nan_distance_leaves_the_neighbor_slot_unassigned() {
            // Given
            let gram = [1.0, f32::NAN, f32::NAN, 1.0];
            let expected_unassigned_neighbors = [LeafNeighbor::default(), LeafNeighbor::default()];

            // When
            let actual_neighbors = rank_distance_fixture::<CosineNormalized>(&gram, 2, 1).unwrap();

            // Then
            assert_eq!(actual_neighbors, expected_unassigned_neighbors);
        }

        #[test]
        fn complete_simd_group_without_eligible_distances_leaves_source_unassigned_with_l2() {
            // Given
            let point_count = 17;
            let source = 16;
            let requested_k = 1;
            let mut gram = gram_with_uniform_self_dots(point_count, 1.0);
            gram[source * point_count..source * point_count + source].fill(f32::NEG_INFINITY);
            let expected_unassigned_neighbor = LeafNeighbor::default();

            // When
            let actual_neighbors =
                rank_distance_fixture::<L2>(&gram, point_count, requested_k).unwrap();

            // Then
            assert_eq!(
                actual_neighbors[source * requested_k],
                expected_unassigned_neighbor
            );
        }

        #[rstest]
        #[case::l2(Metric::L2, 2.0)]
        #[case::cosine(Metric::Cosine, 1.0)]
        #[case::normalized_cosine(Metric::CosineNormalized, -0.0)]
        #[case::inner_product(Metric::InnerProduct, -0.0)]
        fn simd_nan_distance_cannot_replace_a_finite_neighbor(
            #[case] metric: Metric,
            #[case] expected_distance: f32,
        ) {
            // Given
            let points = 17;
            let mut gram = gram_with_uniform_self_dots(points, 1.0);
            gram[16 * points] = f32::NAN;
            gram[16] = f32::NAN;
            let expected_finite_neighbor = LeafNeighbor::new(1, expected_distance);

            // When
            let actual_neighbors = run_rank_leaf_distances(metric, &gram, points, 1).1;

            // Then
            assert_eq!(actual_neighbors[16], expected_finite_neighbor);
        }

        #[rstest]
        #[case::l2(Metric::L2, 2.0)]
        #[case::normalized_cosine(Metric::CosineNormalized, -0.0)]
        #[case::inner_product(Metric::InnerProduct, -0.0)]
        fn simd_positive_infinity_cannot_fill_a_neighbor_slot(
            #[case] metric: Metric,
            #[case] expected_distance: f32,
        ) {
            // Given: negative-infinite dot products produce positive-infinite distances here.
            let points = 17;
            let mut gram = gram_with_uniform_self_dots(points, 1.0);
            gram[16 * points] = f32::NEG_INFINITY;
            gram[16] = f32::NEG_INFINITY;
            let expected_finite_neighbor = LeafNeighbor::new(1, expected_distance);

            // When
            let actual_neighbors = run_rank_leaf_distances(metric, &gram, points, 1).1;

            // Then
            assert_eq!(actual_neighbors[16], expected_finite_neighbor);
        }

        #[test]
        fn singleton_leaf_has_no_neighbors() {
            // Given
            let singleton_point_count = 1;
            let singleton_gram = [4.0];
            let expected_zero_neighbor_width = 0;
            let expected_no_neighbors: [LeafNeighbor; 0] = [];

            // When
            let actual_neighbors = rank_distance_fixture::<Cosine>(
                &singleton_gram,
                singleton_point_count,
                expected_zero_neighbor_width,
            )
            .unwrap();

            // Then
            assert_eq!(actual_neighbors, expected_no_neighbors);
        }

        #[test]
        fn neighbor_width_equal_to_point_count_is_rejected() {
            // Given
            let point_count = 3;
            let invalid_neighbor_width = point_count;
            let maximum_non_self_width = point_count - 1;
            let gram = [0.0; 9];
            let expected_error = LeafKernelError::InvalidNeighborCount {
                points: point_count,
                neighbors: invalid_neighbor_width,
                maximum: maximum_non_self_width,
            };

            // When
            let actual_error =
                rank_distance_fixture::<L2>(&gram, point_count, invalid_neighbor_width)
                    .unwrap_err();

            // Then
            assert_eq!(actual_error, expected_error);
        }
    }
}
