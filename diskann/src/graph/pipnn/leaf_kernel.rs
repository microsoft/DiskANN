/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Leaf-local top-k selection from packed `f32` point vectors.
//!
//! The kernel computes the lower-triangular Gram matrix and metric-specific
//! norms. Its ranking loop reads each strict-lower point pair once and updates
//! both points.
//!
//! The output is an `n × k` matrix of sorted [`LeafNeighbor`] values. Each target
//! is a position in the leaf. Widths 1 through 3 use fixed insertion. Larger
//! widths use the runtime insertion loop.
//!
//! Strict comparisons keep scan order for equal distances. They do not rank NaN.
//! An unfilled output slot contains [`LeafNeighbor::default`]. All supported
//! metrics use the same SIMD-group and single-value traversal.
//!
//! The caller supplies concrete architecture `A` and metric `M`. The private
//! dot ranker receives the square matrix created by this module.
//! [`LeafKernelWorkspace`] stores reusable numerical scratch.

use crate::{ANNError, ANNResult};
use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_wide::{SIMDPartialOrd, SIMDVector};

use super::{
    kernel_metric::LeafMetric,
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
    /// `target` is a position in the leaf. `distance` is its score relative to
    /// the source of the output row.
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
    dot_scratch: Vec<f32>,
    norm_scratch: Vec<f32>,
    worst: Vec<f32>,
}

/// Validation error returned by the dot-ranking loop.
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
    let dot_count = point_count * point_count;
    let LeafKernelWorkspace {
        dot_scratch,
        norm_scratch,
        worst,
    } = workspace;
    if dot_scratch.len() < dot_count {
        dot_scratch.resize(dot_count, 0.0);
    }
    diskann_linalg::sgemm_aat_lower(
        point_count,
        points.ncols(),
        points.as_slice(),
        &mut dot_scratch[..dot_count],
    )
    .map_err(ANNError::new)?;
    let dots = MatrixView::try_from(&dot_scratch[..dot_count], point_count, point_count)
        .map_err(|error| ANNError::new(error.as_static()))?;
    M::prepare_leaf_norms(dots, norm_scratch);
    rank_leaf_dots::<A, M>(arch, dots, norm_scratch, output, worst).map_err(ANNError::new)
}

/// Rank a prepared lower-triangular Gram matrix.
fn rank_leaf_dots<A, M>(
    arch: A,
    input: MatrixView<'_, f32>,
    norms: &[f32],
    mut output: MutMatrixView<'_, LeafNeighbor>,
    worst: &mut Vec<f32>,
) -> Result<(), LeafKernelError>
where
    A: PiPNNSIMDSchema,
    M: LeafMetric,
{
    validate_neighbor_count(input, &output)?;
    let neighbor_count = output.ncols();
    if neighbor_count == 0 {
        return Ok(());
    }

    worst.resize(input.nrows(), f32::INFINITY);
    output.as_mut_slice().fill(LeafNeighbor::default());
    worst.fill(f32::INFINITY);

    match neighbor_count {
        1 => scan_fixed_width::<A, M, 1>(arch, input, norms, output.as_mut_slice(), worst),
        2 => scan_fixed_width::<A, M, 2>(arch, input, norms, output.as_mut_slice(), worst),
        3 => scan_fixed_width::<A, M, 3>(arch, input, norms, output.as_mut_slice(), worst),
        _ => scan_runtime_width::<A, M>(
            arch,
            input,
            norms,
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
    input: MatrixView<'_, f32>,
    output: &MutMatrixView<'_, LeafNeighbor>,
) -> Result<(), LeafKernelError> {
    let point_count = input.nrows();
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
fn scan_fixed_width<A, M, const N: usize>(
    arch: A,
    input: MatrixView<'_, f32>,
    norms: &[f32],
    output: &mut [LeafNeighbor],
    worst: &mut [f32],
) where
    A: PiPNNSIMDSchema,
    M: LeafMetric,
    [LeafNeighbor; N]: SortedInsert<LeafNeighbor>,
{
    let (rows, _) = output.as_chunks_mut::<N>();
    scan_point_pairs::<A, M, _>(arch, input, norms, worst, |source, target, distance| {
        insert_neighbor(&mut rows[source], target, distance)
    });
}

/// Select neighbors with a runtime output width.
fn scan_runtime_width<A, M>(
    arch: A,
    input: MatrixView<'_, f32>,
    norms: &[f32],
    output: &mut [LeafNeighbor],
    width: usize,
    worst: &mut [f32],
) where
    A: PiPNNSIMDSchema,
    M: LeafMetric,
{
    scan_point_pairs::<A, M, _>(arch, input, norms, worst, |source, target, distance| {
        let first = source * width;
        insert_neighbor(&mut output[first..first + width], target, distance)
    });
}

/// Select neighbors from all unordered point pairs in one leaf.
///
/// The function reads the strict lower triangle once. It offers each distance to
/// both endpoint lists. SIMD groups and single values preserve pair scan order.
#[inline(never)]
fn scan_point_pairs<A, M, I>(
    arch: A,
    input: MatrixView<'_, f32>,
    norms: &[f32],
    worst: &mut [f32],
    mut insert: I,
) where
    A: PiPNNSIMDSchema,
    M: LeafMetric,
    I: FnMut(usize, u32, f32) -> f32,
{
    let point_count = input.nrows();
    let dots = input.as_slice();
    let worst_ptr = worst.as_mut_ptr();

    for source in 1..point_count {
        let source_start = source * point_count;
        let source_simd = M::source_simd(arch, norms, source);
        let source_single = M::source_single(norms, source);
        // SAFETY: `rank_leaf_dots` created one threshold for each point.
        let mut source_worst = unsafe { *worst_ptr.add(source) };
        let mut target = 0;
        let simd_prefix = source - source % M::Simd::<A>::LANES;

        while target < simd_prefix {
            // SAFETY: This complete SIMD group is in the strict-lower prefix.
            let dot_products =
                unsafe { M::Simd::<A>::load_simd(arch, dots.as_ptr().add(source_start + target)) };
            let distances = M::distances_simd(arch, norms, source_simd, dot_products, target);
            let source_eligible = distances.lt_simd(M::Simd::<A>::splat(arch, source_worst));
            // SAFETY: The complete target group is below `source < point_count`.
            let target_worst = unsafe { M::Simd::<A>::load_simd(arch, worst_ptr.add(target)) };
            let target_eligible = distances.lt_simd(target_worst);
            let source_bits = M::Simd::<A>::active_lanes(source_eligible);
            let target_bits = M::Simd::<A>::active_lanes(target_eligible);

            if source_bits | target_bits != 0 {
                let values = distances.to_array();
                let values = values.as_ref();
                let mut source_bits = source_bits;
                while source_bits != 0 {
                    let lane = source_bits.trailing_zeros() as usize;
                    source_bits &= source_bits - 1;
                    let distance = values[lane];
                    if distance < source_worst {
                        source_worst = insert(source, (target + lane) as u32, distance);
                    }
                }

                let mut target_bits = target_bits;
                while target_bits != 0 {
                    let lane = target_bits.trailing_zeros() as usize;
                    target_bits &= target_bits - 1;
                    let target_source = target + lane;
                    let new_worst = insert(target_source, source as u32, values[lane]);
                    // SAFETY: `target_source < source < worst.len()`.
                    unsafe { *worst_ptr.add(target_source) = new_worst };
                }
            }
            target += M::Simd::<A>::LANES;
        }

        while target < source {
            // SAFETY: The target is in this source's strict-lower prefix.
            let dot_product = unsafe { *dots.get_unchecked(source_start + target) };
            let distance = M::distance_single(norms, source_single, dot_product, target);
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

/// Insert one value that precedes the current last retained value.
///
/// The caller checks eligibility before insertion. The returned value is the new
/// last retained value for the next eligibility check.
trait SortedInsert<T: Copy> {
    fn insert_sorted_by(&mut self, value: T, precedes: impl Fn(T, T) -> bool) -> T;
}

impl<T: Copy> SortedInsert<T> for [T; 1] {
    #[inline(always)]
    fn insert_sorted_by(&mut self, value: T, _precedes: impl Fn(T, T) -> bool) -> T {
        self[0] = value;
        value
    }
}

impl<T: Copy> SortedInsert<T> for [T; 2] {
    #[inline(always)]
    fn insert_sorted_by(&mut self, value: T, precedes: impl Fn(T, T) -> bool) -> T {
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
    fn insert_sorted_by(&mut self, value: T, precedes: impl Fn(T, T) -> bool) -> T {
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
    fn insert_sorted_by(&mut self, value: T, precedes: impl Fn(T, T) -> bool) -> T {
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

/// Insert one candidate that is nearer than the current farthest neighbor.
///
/// Return the new farthest retained distance for the next candidate check.
#[inline(always)]
fn insert_neighbor<R>(neighbors: &mut R, target: u32, distance: f32) -> f32
where
    R: SortedInsert<LeafNeighbor> + ?Sized,
{
    neighbors
        .insert_sorted_by(
            LeafNeighbor::new(target, distance),
            |candidate, retained| candidate.distance < retained.distance,
        )
        .distance
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::*;
    use crate::graph::pipnn::kernel_metric::{Cosine, CosineNormalized, InnerProduct, L2};
    use diskann_utils::views::{MatrixView, MutMatrixView};
    use diskann_vector::distance::Metric;
    use diskann_wide::arch::{self, Target1};

    struct KernelCall<'a> {
        input: MatrixView<'a, f32>,
        norms: &'a [f32],
        output: MutMatrixView<'a, LeafNeighbor>,
        workspace: &'a mut LeafKernelWorkspace,
    }

    struct DispatchMetric(Metric);

    impl<A> Target1<A, Result<(), LeafKernelError>, KernelCall<'_>> for DispatchMetric
    where
        A: PiPNNSIMDSchema,
    {
        fn run(self, arch: A, call: KernelCall<'_>) -> Result<(), LeafKernelError> {
            match self.0 {
                Metric::L2 => rank_leaf_dots::<A, L2>(
                    arch,
                    call.input,
                    call.norms,
                    call.output,
                    &mut call.workspace.worst,
                ),
                Metric::Cosine => rank_leaf_dots::<A, Cosine>(
                    arch,
                    call.input,
                    call.norms,
                    call.output,
                    &mut call.workspace.worst,
                ),
                Metric::CosineNormalized => rank_leaf_dots::<A, CosineNormalized>(
                    arch,
                    call.input,
                    call.norms,
                    call.output,
                    &mut call.workspace.worst,
                ),
                Metric::InnerProduct => rank_leaf_dots::<A, InnerProduct>(
                    arch,
                    call.input,
                    call.norms,
                    call.output,
                    &mut call.workspace.worst,
                ),
            }
        }
    }

    fn lower_gram_view(dots: &[f32], points: usize) -> MatrixView<'_, f32> {
        MatrixView::try_from(dots, points, points).unwrap()
    }

    fn metric_norms(metric: Metric, lower_gram: MatrixView<'_, f32>) -> Vec<f32> {
        fn prepare<M: LeafMetric>(lower_gram: MatrixView<'_, f32>) -> Vec<f32> {
            let mut norms = Vec::new();
            M::prepare_leaf_norms(lower_gram, &mut norms);
            norms
        }

        match metric {
            Metric::L2 => prepare::<L2>(lower_gram),
            Metric::Cosine => prepare::<Cosine>(lower_gram),
            Metric::CosineNormalized => prepare::<CosineNormalized>(lower_gram),
            Metric::InnerProduct => prepare::<InnerProduct>(lower_gram),
        }
    }

    fn rank_neighbors_with_workspace(
        metric: Metric,
        dots: &[f32],
        points: usize,
        requested_k: usize,
        workspace: &mut LeafKernelWorkspace,
    ) -> (usize, Vec<LeafNeighbor>) {
        let leaf_k = leaf_neighbor_count(points, requested_k);
        let lower_gram = lower_gram_view(dots, points);
        let norms = metric_norms(metric, lower_gram);
        let mut output = vec![LeafNeighbor::default(); points * leaf_k];
        arch::dispatch1_no_features(
            DispatchMetric(metric),
            KernelCall {
                input: lower_gram,
                norms: &norms,
                output: MutMatrixView::try_from(output.as_mut_slice(), points, leaf_k).unwrap(),
                workspace,
            },
        )
        .unwrap();
        (leaf_k, output)
    }

    fn rank_neighbors(
        metric: Metric,
        dots: &[f32],
        points: usize,
        requested_k: usize,
    ) -> (usize, Vec<LeafNeighbor>) {
        rank_neighbors_with_workspace(
            metric,
            dots,
            points,
            requested_k,
            &mut LeafKernelWorkspace::default(),
        )
    }

    fn reference_distance(
        metric: Metric,
        dot: f32,
        source_diagonal: f32,
        target_diagonal: f32,
    ) -> f32 {
        match metric {
            Metric::L2 => ((-2.0_f32).mul_add(dot, source_diagonal) + target_diagonal).max(0.0),
            Metric::CosineNormalized => 1.0 - dot,
            Metric::InnerProduct => -dot,
            Metric::Cosine => {
                let source_norm = source_diagonal.sqrt();
                let target_norm = target_diagonal.sqrt();
                if source_norm < f32::MIN_POSITIVE.sqrt() || target_norm < f32::MIN_POSITIVE.sqrt()
                {
                    1.0
                } else {
                    let similarity = dot / (source_norm * target_norm);
                    1.0 - similarity.clamp(-1.0, 1.0)
                }
            }
        }
    }

    fn reference_neighbors(
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

    /// Build a unit-diagonal lower Gram matrix for the lane-boundary sweep.
    /// Similarity decreases as the point-index separation increases.
    fn index_distance_lower_gram(points: usize) -> Vec<f32> {
        let mut dots = vec![f32::NAN; points * points];
        for source in 0..points {
            dots[source * points + source] = 1.0;
            for target in 0..source {
                let separation = (source - target) as f32;
                dots[source * points + target] = 1.0 - separation / points as f32;
            }
        }
        dots
    }

    fn square_matrix_with_constant_diagonal(points: usize, diagonal: f32) -> Vec<f32> {
        let mut values = vec![0.0; points * points];
        for point in 0..points {
            values[point * points + point] = diagonal;
        }
        values
    }

    mod insert_neighbor_tests {
        use super::*;

        #[test]
        fn one_slot_insertion_replaces_the_retained_neighbor() {
            // Given
            let retained_neighbor = LeafNeighbor::new(1, 4.0);
            let nearer_candidate = LeafNeighbor::new(2, 2.0);
            let expected_neighbors = [nearer_candidate];
            let mut actual_neighbors = [retained_neighbor];

            // When
            insert_neighbor(
                &mut actual_neighbors,
                nearer_candidate.target,
                nearer_candidate.distance,
            );

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn two_slot_insertion_places_a_nearer_candidate_first() {
            // Given
            let nearest = LeafNeighbor::new(1, 1.0);
            let farthest = LeafNeighbor::new(2, 3.0);
            let nearer_candidate = LeafNeighbor::new(3, 0.5);
            let expected_neighbors = [nearer_candidate, nearest];
            let mut actual_neighbors = [nearest, farthest];

            // When
            insert_neighbor(
                &mut actual_neighbors,
                nearer_candidate.target,
                nearer_candidate.distance,
            );

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn two_slot_insertion_places_a_middle_distance_last() {
            // Given
            let nearest = LeafNeighbor::new(1, 1.0);
            let farthest = LeafNeighbor::new(2, 3.0);
            let eligible_candidate = LeafNeighbor::new(3, 2.0);
            let expected_neighbors = [nearest, eligible_candidate];
            let mut actual_neighbors = [nearest, farthest];

            // When
            insert_neighbor(
                &mut actual_neighbors,
                eligible_candidate.target,
                eligible_candidate.distance,
            );

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn three_slot_insertion_places_the_nearest_candidate_first() {
            // Given
            let nearest = LeafNeighbor::new(1, 1.0);
            let middle = LeafNeighbor::new(2, 2.0);
            let farthest = LeafNeighbor::new(3, 4.0);
            let nearer_candidate = LeafNeighbor::new(4, 0.5);
            let expected_neighbors = [nearer_candidate, nearest, middle];
            let mut actual_neighbors = [nearest, middle, farthest];

            // When
            insert_neighbor(
                &mut actual_neighbors,
                nearer_candidate.target,
                nearer_candidate.distance,
            );

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn three_slot_insertion_places_a_middle_candidate_between_neighbors() {
            // Given
            let nearest = LeafNeighbor::new(1, 1.0);
            let middle = LeafNeighbor::new(2, 2.0);
            let farthest = LeafNeighbor::new(3, 4.0);
            let middle_candidate = LeafNeighbor::new(4, 1.5);
            let expected_neighbors = [nearest, middle_candidate, middle];
            let mut actual_neighbors = [nearest, middle, farthest];

            // When
            insert_neighbor(
                &mut actual_neighbors,
                middle_candidate.target,
                middle_candidate.distance,
            );

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn three_slot_insertion_replaces_the_farthest_neighbor() {
            // Given
            let nearest = LeafNeighbor::new(1, 1.0);
            let middle = LeafNeighbor::new(2, 2.0);
            let farthest = LeafNeighbor::new(3, 4.0);
            let eligible_candidate = LeafNeighbor::new(4, 3.0);
            let expected_neighbors = [nearest, middle, eligible_candidate];
            let mut actual_neighbors = [nearest, middle, farthest];

            // When
            insert_neighbor(
                &mut actual_neighbors,
                eligible_candidate.target,
                eligible_candidate.distance,
            );

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn runtime_width_insertion_shifts_only_the_later_neighbors() {
            // Given
            let first = LeafNeighbor::new(1, 1.0);
            let second = LeafNeighbor::new(2, 2.0);
            let third = LeafNeighbor::new(3, 3.0);
            let fourth = LeafNeighbor::new(4, 5.0);
            let candidate = LeafNeighbor::new(5, 2.5);
            let expected_neighbors = [first, second, candidate, third];
            let mut actual_neighbors = [first, second, third, fourth];

            // When
            insert_neighbor(
                actual_neighbors.as_mut_slice(),
                candidate.target,
                candidate.distance,
            );

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn sorted_insertion_preserves_existing_order_for_equal_distances() {
            // Given
            let nearest = LeafNeighbor::new(1, 1.0);
            let existing_tie = LeafNeighbor::new(2, 2.0);
            let farthest = LeafNeighbor::new(3, 4.0);
            let tied_candidate = LeafNeighbor::new(4, 2.0);
            let expected_neighbors = [nearest, existing_tie, tied_candidate];
            let mut actual_neighbors = [nearest, existing_tie, farthest];

            // When
            insert_neighbor(
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
        fn empty_leaf_cannot_retain_neighbors() {
            // Given
            let point_count = 0;
            let requested_k = 3;
            let expected_neighbor_count = 0;

            // When
            let actual_neighbor_count = leaf_neighbor_count(point_count, requested_k);

            // Then
            assert_eq!(actual_neighbor_count, expected_neighbor_count);
        }

        #[test]
        fn singleton_leaf_cannot_retain_its_source_point() {
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
        fn requested_k_above_available_neighbors_is_clamped() {
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
        fn requested_k_within_available_neighbors_is_unchanged() {
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
        fn l2_pipeline_orders_neighbors_by_squared_distance() {
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

    mod rank_leaf_dots_tests {
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
            // Given
            let dots = index_distance_lower_gram(point_count);
            let expected_neighbors = reference_neighbors(metric, &dots, point_count, requested_k);

            // When
            let actual_neighbors = rank_neighbors(metric, &dots, point_count, requested_k).1;

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        const UNIT_SQUARED_NORM: f32 = 1.0;
        const POINT_0_1_DOT: f32 = 0.0;
        const POINT_0_2_DOT: f32 = -1.0;
        const POINT_1_2_DOT: f32 = 0.5;

        #[rustfmt::skip]
    const THREE_POINT_LOWER_GRAM: [f32; 9] = [
        UNIT_SQUARED_NORM, f32::NAN,          f32::NAN,
        POINT_0_1_DOT,      UNIT_SQUARED_NORM, f32::NAN,
        POINT_0_2_DOT,      POINT_1_2_DOT,     UNIT_SQUARED_NORM,
    ];

        fn rank_three_point_fixture(metric: Metric) -> Vec<LeafNeighbor> {
            rank_neighbors(metric, &THREE_POINT_LOWER_GRAM, 3, 1).1
        }

        #[test]
        fn l2_selects_the_nearest_target_from_the_lower_gram_triangle() {
            // Given
            let expected_neighbors = [
                LeafNeighbor::new(1, 2.0 * UNIT_SQUARED_NORM - 2.0 * POINT_0_1_DOT),
                LeafNeighbor::new(2, 2.0 * UNIT_SQUARED_NORM - 2.0 * POINT_1_2_DOT),
                LeafNeighbor::new(1, 2.0 * UNIT_SQUARED_NORM - 2.0 * POINT_1_2_DOT),
            ];

            // When
            let actual_neighbors = rank_three_point_fixture(Metric::L2);

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn cosine_selects_the_nearest_target_from_the_lower_gram_triangle() {
            // Given
            let expected_neighbors = [
                LeafNeighbor::new(1, 1.0 - POINT_0_1_DOT),
                LeafNeighbor::new(2, 1.0 - POINT_1_2_DOT),
                LeafNeighbor::new(1, 1.0 - POINT_1_2_DOT),
            ];

            // When
            let actual_neighbors = rank_three_point_fixture(Metric::Cosine);

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn normalized_cosine_selects_the_largest_dot_product() {
            // Given
            let expected_neighbors = [
                LeafNeighbor::new(1, 1.0 - POINT_0_1_DOT),
                LeafNeighbor::new(2, 1.0 - POINT_1_2_DOT),
                LeafNeighbor::new(1, 1.0 - POINT_1_2_DOT),
            ];

            // When
            let actual_neighbors = rank_three_point_fixture(Metric::CosineNormalized);

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn inner_product_selects_the_largest_dot_product() {
            // Given
            let expected_neighbors = [
                LeafNeighbor::new(1, -POINT_0_1_DOT),
                LeafNeighbor::new(2, -POINT_1_2_DOT),
                LeafNeighbor::new(1, -POINT_1_2_DOT),
            ];

            // When
            let actual_neighbors = rank_three_point_fixture(Metric::InnerProduct);

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[test]
        fn equal_l2_distances_keep_target_scan_order() {
            // Given
            let unit_squared_norm = 1.0;
            let tied_dot_product = 0.0;
            let expected_tied_distance = 2.0 * unit_squared_norm - 2.0 * tied_dot_product;
            #[rustfmt::skip]
        let dots = [
            unit_squared_norm, f32::NAN,          f32::NAN,          f32::NAN,
            tied_dot_product,  unit_squared_norm, f32::NAN,          f32::NAN,
            tied_dot_product,  tied_dot_product,  unit_squared_norm, f32::NAN,
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
            let actual_neighbors = rank_neighbors(Metric::L2, &dots, 4, 2).1;

            // Then
            assert_eq!(actual_neighbors, expected_neighbors_in_scan_order);
        }

        #[test]
        fn scalar_l2_distance_stays_finite_when_twice_the_dot_product_overflows() {
            // Given
            let dot_product = f32::from_bits(f32::MAX.to_bits() - 1);
            let unfused_twice_dot_product = 2.0 * dot_product;
            let expected_fused_distance = (-2.0_f32).mul_add(dot_product, f32::MAX) + f32::MAX;
            let dots = [f32::MAX, 0.0, dot_product, f32::MAX];

            // When
            let actual_neighbors = rank_neighbors(Metric::L2, &dots, 2, 1).1;

            // Then
            assert!(unfused_twice_dot_product.is_infinite());
            assert!(expected_fused_distance.is_finite() && expected_fused_distance > 0.0);
            assert_eq!(
                actual_neighbors[0].distance.to_bits(),
                expected_fused_distance.to_bits()
            );
        }

        #[test]
        fn simd_l2_distance_stays_finite_when_twice_the_dot_product_overflows() {
            // Given
            let dot_product = f32::from_bits(f32::MAX.to_bits() - 1);
            let unfused_twice_dot_product = 2.0 * dot_product;
            let expected_fused_distance = (-2.0_f32).mul_add(dot_product, f32::MAX) + f32::MAX;
            let expected_simd_neighbor_target = 0;
            let points = 17;
            let mut dots = square_matrix_with_constant_diagonal(points, f32::MAX);
            dots[16 * points] = dot_product;

            // When
            let actual_neighbors = rank_neighbors(Metric::L2, &dots, points, 1).1;

            // Then
            assert!(unfused_twice_dot_product.is_infinite());
            assert_eq!(actual_neighbors[16].target, expected_simd_neighbor_target);
            assert_eq!(
                actual_neighbors[16].distance.to_bits(),
                expected_fused_distance.to_bits()
            );
        }

        #[test]
        fn cosine_zero_norm_produces_unit_distance() {
            // Given
            #[rustfmt::skip]
        let dots = [
            0.0, 99.0, 99.0,
            0.0,  1.0, 99.0,
            0.0,  0.0,  1.0,
        ];
            let expected_zero_norm_neighbors =
                [LeafNeighbor::new(1, 1.0), LeafNeighbor::new(2, 1.0)];

            // When
            let actual_neighbors = rank_neighbors(Metric::Cosine, &dots, 3, 2).1;

            // Then
            assert_eq!(&actual_neighbors[..2], &expected_zero_norm_neighbors);
        }

        #[test]
        fn cosine_similarity_above_one_clamps_to_zero_distance() {
            // Given
            let dots = [1.0, 0.0, 2.0, 1.0];
            let maximum_cosine_similarity = 1.0;
            let expected_one_minus_maximum_similarity = 1.0 - maximum_cosine_similarity;

            // When
            let actual_neighbors = rank_neighbors(Metric::Cosine, &dots, 2, 1).1;

            // Then
            assert_eq!(
                actual_neighbors[0].distance,
                expected_one_minus_maximum_similarity
            );
        }

        #[test]
        fn cosine_similarity_below_negative_one_clamps_to_distance_two() {
            // Given
            let dots = [1.0, 0.0, -2.0, 1.0];
            let minimum_cosine_similarity = -1.0;
            let expected_one_minus_minimum_similarity = 1.0 - minimum_cosine_similarity;

            // When
            let actual_neighbors = rank_neighbors(Metric::Cosine, &dots, 2, 1).1;

            // Then
            assert_eq!(
                actual_neighbors[0].distance,
                expected_one_minus_minimum_similarity
            );
        }

        #[test]
        fn cosine_subnormal_norm_is_treated_as_zero() {
            // Given
            let dots = [f32::MIN_POSITIVE / 2.0, 0.0, 1.0, 1.0];
            let zero_norm_similarity = 0.0;
            let expected_one_minus_zero_similarity = 1.0 - zero_norm_similarity;

            // When
            let actual_neighbors = rank_neighbors(Metric::Cosine, &dots, 2, 1).1;

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
            let mut dots = vec![0.0; points * points];
            dots[3 * points] = -f32::MAX;

            // When
            let (actual_leaf_k, actual_neighbors) =
                rank_neighbors(Metric::InnerProduct, &dots, points, expected_leaf_k);

            // Then
            assert_eq!(actual_leaf_k, expected_leaf_k);
            assert_eq!(
                actual_neighbors[3 * actual_leaf_k + actual_leaf_k - 1],
                expected_last_neighbor
            );
        }

        #[test]
        fn scalar_nan_distance_leaves_the_neighbor_slot_unassigned() {
            // Given
            let dots = [1.0, 0.0, f32::NAN, 1.0];
            let expected_unassigned_neighbors = [LeafNeighbor::default(), LeafNeighbor::default()];

            // When
            let actual_neighbors = rank_neighbors(Metric::CosineNormalized, &dots, 2, 1).1;

            // Then
            assert_eq!(actual_neighbors, expected_unassigned_neighbors);
        }

        #[test]
        fn simd_nan_distance_cannot_replace_a_finite_neighbor() {
            // Given
            let points = 17;
            let mut dots = square_matrix_with_constant_diagonal(points, 1.0);
            dots[16 * points] = f32::NAN;
            let expected_finite_neighbor = LeafNeighbor::new(1, 1.0);

            // When
            let actual_neighbors = rank_neighbors(Metric::CosineNormalized, &dots, points, 1).1;

            // Then
            assert_eq!(actual_neighbors[16], expected_finite_neighbor);
        }

        #[test]
        fn empty_leaf_has_no_neighbors() {
            // Given
            let dots = [];
            let expected_zero_neighbor_width = 0;
            let expected_no_neighbors: [LeafNeighbor; 0] = [];

            // When
            let (actual_leaf_k, actual_neighbors) = rank_neighbors(Metric::L2, &dots, 0, 2);

            // Then
            assert_eq!(actual_leaf_k, expected_zero_neighbor_width);
            assert_eq!(actual_neighbors, expected_no_neighbors);
        }

        #[test]
        fn singleton_leaf_has_no_neighbors() {
            // Given
            let dots = [4.0];
            let expected_zero_neighbor_width = 0;
            let expected_no_neighbors: [LeafNeighbor; 0] = [];

            // When
            let (actual_leaf_k, actual_neighbors) = rank_neighbors(Metric::Cosine, &dots, 1, 2);

            // Then
            assert_eq!(actual_leaf_k, expected_zero_neighbor_width);
            assert_eq!(actual_neighbors, expected_no_neighbors);
        }

        #[test]
        fn zero_requested_k_has_no_neighbors() {
            // Given
            let dots = [1.0, 0.0, 0.0, 1.0];
            let expected_zero_neighbor_width = 0;
            let expected_no_neighbors: [LeafNeighbor; 0] = [];

            // When
            let (actual_leaf_k, actual_neighbors) =
                rank_neighbors(Metric::InnerProduct, &dots, 2, 0);

            // Then
            assert_eq!(actual_leaf_k, expected_zero_neighbor_width);
            assert_eq!(actual_neighbors, expected_no_neighbors);
        }

        #[test]
        fn neighbor_width_equal_to_point_count_is_rejected() {
            // Given
            let dots = [0.0; 9];
            let input = lower_gram_view(&dots, 3);
            let norms = metric_norms(Metric::L2, input);
            let expected_error = LeafKernelError::InvalidNeighborCount {
                points: 3,
                neighbors: 3,
                maximum: 2,
            };
            let mut output = [LeafNeighbor::default(); 9];

            // When
            let actual_error = arch::dispatch1_no_features(
                DispatchMetric(Metric::L2),
                KernelCall {
                    input,
                    norms: &norms,
                    output: MutMatrixView::try_from(&mut output[..], 3, 3).unwrap(),
                    workspace: &mut LeafKernelWorkspace::default(),
                },
            )
            .unwrap_err();

            // Then
            assert_eq!(actual_error, expected_error);
        }
    }
}
