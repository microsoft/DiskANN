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
{
    let (rows, _) = output.as_chunks_mut::<N>();
    scan_point_pairs::<A, M, _>(arch, input, norms, worst, |source, target, distance| {
        insert_fixed_neighbor(&mut rows[source], target, distance)
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
        insert_runtime_neighbor(&mut output[first..first + width], target, distance)
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

/// Insert one target into a fixed-width retained neighbor set.
#[inline(always)]
fn insert_fixed_neighbor<const N: usize>(
    neighbors: &mut [LeafNeighbor; N],
    target: u32,
    distance: f32,
) -> f32 {
    let entry = LeafNeighbor::new(target, distance);
    if N == 1 {
        neighbors[0] = entry;
        return distance;
    }
    if N == 2 {
        let first = neighbors[0];
        if distance < first.distance {
            neighbors[0] = entry;
            neighbors[1] = first;
            return first.distance;
        }
        neighbors[1] = entry;
        return distance;
    }

    let (first, second) = (neighbors[0], neighbors[1]);
    if distance < first.distance {
        neighbors[0] = entry;
        neighbors[1] = first;
        neighbors[2] = second;
    } else if distance < second.distance {
        neighbors[1] = entry;
        neighbors[2] = second;
    } else {
        neighbors[2] = entry;
        return distance;
    }
    second.distance
}

/// Insert one target into a runtime-width retained neighbor set.
#[inline(always)]
fn insert_runtime_neighbor(neighbors: &mut [LeafNeighbor], target: u32, distance: f32) -> f32 {
    let last = neighbors.len() - 1;
    let mut slot = last;
    while slot > 0 && distance < neighbors[slot - 1].distance {
        neighbors[slot] = neighbors[slot - 1];
        slot -= 1;
    }
    neighbors[slot] = LeafNeighbor::new(target, distance);
    neighbors[last].distance
}

#[cfg(test)]
struct DispatchedLeafCall<'a> {
    input: MatrixView<'a, f32>,
    norms: &'a [f32],
    output: MutMatrixView<'a, LeafNeighbor>,
    workspace: &'a mut LeafKernelWorkspace,
}

#[cfg(test)]
struct DispatchLeafForTest(diskann_vector::distance::Metric);

#[cfg(test)]
impl<A> diskann_wide::arch::Target1<A, Result<(), LeafKernelError>, DispatchedLeafCall<'_>>
    for DispatchLeafForTest
where
    A: PiPNNSIMDSchema,
{
    fn run(self, arch: A, call: DispatchedLeafCall<'_>) -> Result<(), LeafKernelError> {
        use super::kernel_metric::{Cosine, CosineNormalized, InnerProduct, L2};
        use diskann_vector::distance::Metric;

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

#[cfg(test)]
fn dispatch_nearest_neighbors(
    metric: diskann_vector::distance::Metric,
    input: MatrixView<'_, f32>,
    norms: &[f32],
    output: MutMatrixView<'_, LeafNeighbor>,
    workspace: &mut LeafKernelWorkspace,
) -> Result<(), LeafKernelError> {
    diskann_wide::arch::dispatch1_no_features(
        DispatchLeafForTest(metric),
        DispatchedLeafCall {
            input,
            norms,
            output,
            workspace,
        },
    )
}

#[cfg(test)]
fn prepared_test_norms(
    metric: diskann_vector::distance::Metric,
    input: MatrixView<'_, f32>,
) -> Vec<f32> {
    use super::kernel_metric::{Cosine, CosineNormalized, InnerProduct, L2};
    use diskann_vector::distance::Metric;

    fn prepare<M: LeafMetric>(input: MatrixView<'_, f32>) -> Vec<f32> {
        let mut norms = Vec::new();
        M::prepare_leaf_norms(input, &mut norms);
        norms
    }

    match metric {
        Metric::L2 => prepare::<L2>(input),
        Metric::Cosine => prepare::<Cosine>(input),
        Metric::CosineNormalized => prepare::<CosineNormalized>(input),
        Metric::InnerProduct => prepare::<InnerProduct>(input),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_vector::distance::Metric;

    fn test_dots(metric: Metric, points: usize) -> Vec<f32> {
        let mut dots = vec![f32::NAN; points * points];
        for source in 0..points {
            dots[source * points + source] = if metric == Metric::Cosine && source == 0 {
                0.0
            } else {
                1.0 + (source % 5) as f32
            };
            for target in 0..source {
                dots[source * points + target] =
                    (((source * 17 + target * 11) % 23) as f32 - 11.0) * 0.03125;
            }
        }
        dots
    }

    fn test_input(dots: &[f32], points: usize) -> MatrixView<'_, f32> {
        MatrixView::try_from(dots, points, points).unwrap()
    }

    #[test]
    fn fixed_insertion_orders_candidates() {
        let mut output = [LeafNeighbor::default(); 3];
        let mut worst = f32::INFINITY;

        for (target, distance) in [(0, 4.0), (1, 1.0), (2, 3.0), (3, 2.0), (4, 0.5)] {
            if distance < worst {
                worst = insert_fixed_neighbor(&mut output, target, distance);
            }
        }

        assert_eq!(
            output,
            [
                LeafNeighbor::new(4, 0.5),
                LeafNeighbor::new(1, 1.0),
                LeafNeighbor::new(3, 2.0),
            ]
        );
        assert_eq!(worst, 2.0);
    }

    #[test]
    fn neighbor_count_clamps_to_non_self_neighbors() {
        assert_eq!(leaf_neighbor_count(0, 3), 0);
        assert_eq!(leaf_neighbor_count(1, 3), 0);
        assert_eq!(leaf_neighbor_count(4, 4), 3);
        assert_eq!(leaf_neighbor_count(8, 5), 5);
    }

    #[test]
    fn kernel_accepts_different_neighbor_counts() {
        let points = 7;
        let dots = test_dots(Metric::L2, points);
        let input = test_input(&dots, points);
        let norms = prepared_test_norms(Metric::L2, input);
        let mut workspace = LeafKernelWorkspace::default();

        for neighbor_count in [1, 3, 5, 2] {
            let mut output = vec![LeafNeighbor::default(); points * neighbor_count];
            dispatch_nearest_neighbors(
                Metric::L2,
                input,
                &norms,
                MutMatrixView::try_from(output.as_mut_slice(), points, neighbor_count).unwrap(),
                &mut workspace,
            )
            .unwrap();
            assert!(output.iter().all(|neighbor| neighbor.target != u32::MAX));
        }
    }

    #[test]
    fn vector_pipeline_selects_exact_l2_neighbors_and_reuses_workspace() {
        use super::super::kernel_metric::L2;

        let values = [0.0, 1.0, 3.0, 10.0];
        let points = MatrixView::try_from(&values[..], 4, 1).unwrap();
        let mut output = [LeafNeighbor::default(); 8];
        let mut workspace = LeafKernelWorkspace::default();
        select_leaf_neighbors::<_, L2>(
            diskann_wide::ARCH,
            points,
            MutMatrixView::try_from(&mut output[..], 4, 2).unwrap(),
            &mut workspace,
        )
        .unwrap();

        assert_eq!(
            output,
            [
                LeafNeighbor::new(1, 1.0),
                LeafNeighbor::new(2, 9.0),
                LeafNeighbor::new(0, 1.0),
                LeafNeighbor::new(2, 4.0),
                LeafNeighbor::new(1, 4.0),
                LeafNeighbor::new(0, 9.0),
                LeafNeighbor::new(2, 49.0),
                LeafNeighbor::new(1, 81.0),
            ]
        );

        let dot_scratch = workspace.dot_scratch.as_ptr();
        let norm_scratch = workspace.norm_scratch.as_ptr();
        let worst = workspace.worst.as_ptr();
        let mut smaller_output = [LeafNeighbor::default(); 6];
        select_leaf_neighbors::<_, L2>(
            diskann_wide::ARCH,
            MatrixView::try_from(&values[..3], 3, 1).unwrap(),
            MutMatrixView::try_from(&mut smaller_output[..], 3, 2).unwrap(),
            &mut workspace,
        )
        .unwrap();

        assert_eq!(workspace.dot_scratch.as_ptr(), dot_scratch);
        assert_eq!(workspace.norm_scratch.as_ptr(), norm_scratch);
        assert_eq!(workspace.worst.as_ptr(), worst);
    }

    #[test]
    fn workspace_can_shrink_and_grow_between_calls() {
        let mut workspace = LeafKernelWorkspace::default();
        for points in [17, 7, 17] {
            let dots = test_dots(Metric::L2, points);
            let mut output = vec![LeafNeighbor::default(); points * 2];
            let input = test_input(&dots, points);
            let norms = prepared_test_norms(Metric::L2, input);
            dispatch_nearest_neighbors(
                Metric::L2,
                input,
                &norms,
                MutMatrixView::try_from(output.as_mut_slice(), points, 2).unwrap(),
                &mut workspace,
            )
            .unwrap();
            assert!(output.iter().all(|neighbor| neighbor.target != u32::MAX));
        }
    }
}
#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "deterministic test fixture construction must abort on invalid setup"
)]
mod integration_tests {
    use std::cmp::Ordering;

    use super::{
        LeafKernelError, LeafKernelWorkspace, LeafNeighbor, dispatch_nearest_neighbors,
        leaf_neighbor_count, prepared_test_norms,
    };
    use diskann_utils::views::{MatrixView, MutMatrixView};
    use diskann_vector::distance::Metric;

    const SIMD_BOUNDARY_POINTS: [usize; 15] =
        [2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 256, 512];
    const ZERO_NORM_POSITION: usize = 0;
    const DISTINCT_NORM_POSITION: usize = 2;
    const NORM_PERIOD: usize = 5;
    const SOURCE_MIXER: usize = 17;
    const TARGET_MIXER: usize = 11;
    const MIX_MODULUS: usize = 23;
    const MIX_CENTER: f32 = 11.0;
    const DOT_FACTOR: f32 = 1.0 / 32.0;
    const TIED_TARGETS: [usize; 2] = [1, 2];

    fn differential_dots(metric: Metric, points: usize) -> Vec<f32> {
        let mut dots = vec![f32::NAN; points * points];
        for source in 0..points {
            dots[source * points + source] =
                if metric == Metric::Cosine && source == ZERO_NORM_POSITION {
                    0.0
                } else if source == DISTINCT_NORM_POSITION {
                    2.0
                } else {
                    1.0 + (source % NORM_PERIOD) as f32
                };
            for target in 0..source {
                let pair = ((source * SOURCE_MIXER + target * TARGET_MIXER) % MIX_MODULUS) as f32
                    - MIX_CENTER;
                dots[source * points + target] = if TIED_TARGETS.contains(&target) {
                    0.5
                } else {
                    pair * DOT_FACTOR
                };
            }
        }
        dots
    }

    fn test_input(dots: &[f32], points: usize) -> MatrixView<'_, f32> {
        MatrixView::try_from(dots, points, points).unwrap()
    }

    fn brute_force_reference(
        dots: &[f32],
        points: usize,
        requested_k: usize,
        metric: Metric,
    ) -> Vec<LeafNeighbor> {
        let leaf_k = requested_k.min(points.saturating_sub(1));
        let mut output = vec![LeafNeighbor::default(); points * leaf_k];
        if leaf_k == 0 {
            return output;
        }

        let norms: Vec<_> = (0..points)
            .map(|source| {
                let diagonal = dots[source * points + source];
                if metric == Metric::Cosine {
                    if diagonal < f32::MIN_POSITIVE {
                        0.0
                    } else {
                        diagonal.sqrt()
                    }
                } else {
                    diagonal
                }
            })
            .collect();

        for source in 0..points {
            let mut candidates = Vec::with_capacity(points - 1);
            for target in 0..points {
                if target == source {
                    continue;
                }
                let (lower_source, lower_target) = if source > target {
                    (source, target)
                } else {
                    (target, source)
                };
                let dot = dots[lower_source * points + lower_target];
                let clamp = |distance: f32| if distance < 0.0 { 0.0 } else { distance };
                let distance = match metric {
                    Metric::L2 => clamp((-2.0_f32).mul_add(dot, norms[source]) + norms[target]),
                    Metric::CosineNormalized => 1.0 - dot,
                    Metric::InnerProduct => -dot,
                    Metric::Cosine => {
                        let denominator = norms[source] * norms[target];
                        let similarity = if denominator == 0.0 {
                            0.0
                        } else {
                            dot / denominator
                        };
                        1.0 - (-1.0_f32).max(1.0_f32.min(similarity))
                    }
                };
                if distance.partial_cmp(&f32::INFINITY) == Some(Ordering::Less) {
                    candidates.push(LeafNeighbor::new(target as u32, distance));
                }
            }
            candidates.sort_by(|left, right| {
                left.distance
                    .partial_cmp(&right.distance)
                    .expect("NaN distances were filtered")
            });
            let count = candidates.len().min(leaf_k);
            output[source * leaf_k..source * leaf_k + count].copy_from_slice(&candidates[..count]);
        }
        output
    }

    fn run_leaf_kernel(
        dots: &[f32],
        points: usize,
        requested_k: usize,
        metric: Metric,
    ) -> (usize, Vec<LeafNeighbor>) {
        let leaf_k = leaf_neighbor_count(points, requested_k);
        let input = test_input(dots, points);
        let norms = prepared_test_norms(metric, input);
        let mut output = vec![LeafNeighbor::default(); points * leaf_k];
        dispatch_nearest_neighbors(
            metric,
            input,
            &norms,
            MutMatrixView::try_from(output.as_mut_slice(), points, leaf_k).unwrap(),
            &mut LeafKernelWorkspace::default(),
        )
        .unwrap();
        (leaf_k, output)
    }

    #[test]
    fn dispatched_kernel_matches_reference_across_simd_width_boundaries() {
        for metric in [
            Metric::L2,
            Metric::Cosine,
            Metric::CosineNormalized,
            Metric::InnerProduct,
        ] {
            for points in SIMD_BOUNDARY_POINTS {
                let dots = differential_dots(metric, points);
                for requested_k in [1, 2, 3, 4, 7] {
                    let expected = brute_force_reference(&dots, points, requested_k, metric);
                    let actual = run_leaf_kernel(&dots, points, requested_k, metric).1;
                    assert_eq!(actual, expected, "{metric:?}, n={points}, k={requested_k}");
                }
            }
        }
    }

    #[test]
    fn l2_scans_only_the_lower_triangle_and_breaks_ties_by_position() {
        #[rustfmt::skip]
        let dots = [
            0.0, 999.0, 999.0, 999.0,
            0.0,   1.0, 999.0, 999.0,
            0.0,   0.0,   1.0, 999.0,
            0.0,   1.0,   1.0,   2.0,
        ];

        assert_eq!(
            run_leaf_kernel(&dots, 4, 2, Metric::L2).1,
            [
                LeafNeighbor::new(1, 1.0),
                LeafNeighbor::new(2, 1.0),
                LeafNeighbor::new(0, 1.0),
                LeafNeighbor::new(3, 1.0),
                LeafNeighbor::new(0, 1.0),
                LeafNeighbor::new(3, 1.0),
                LeafNeighbor::new(1, 1.0),
                LeafNeighbor::new(2, 1.0),
            ]
        );
    }

    #[test]
    fn supports_every_leaf_metric() {
        #[rustfmt::skip]
        let dots = [
            1.0, 77.0, 77.0,
            0.0,  1.0, 77.0,
           -1.0,  0.5,  1.0,
        ];
        for (metric, expected) in [
            (Metric::L2, [1, 2, 1]),
            (Metric::Cosine, [1, 2, 1]),
            (Metric::CosineNormalized, [1, 2, 1]),
            (Metric::InnerProduct, [1, 2, 1]),
        ] {
            let positions: Vec<_> = run_leaf_kernel(&dots, 3, 1, metric)
                .1
                .iter()
                .map(|neighbor| neighbor.target)
                .collect();
            assert_eq!(positions, expected, "metric {metric:?}");
        }
    }

    #[test]
    fn cosine_treats_zero_norm_as_zero_similarity() {
        #[rustfmt::skip]
        let dots = [
            0.0, 11.0, 11.0,
            0.0,  1.0, 11.0,
            0.0,  0.0,  1.0,
        ];

        let output = run_leaf_kernel(&dots, 3, 2, Metric::Cosine).1;
        assert_eq!(output[0], LeafNeighbor::new(1, 1.0));
        assert_eq!(output[1], LeafNeighbor::new(2, 1.0));
    }

    #[test]
    fn l2_fma_avoids_intermediate_overflow_in_scalar_and_simd_paths() {
        let dot = f32::from_bits(f32::MAX.to_bits() - 1);
        let expected = (-2.0_f32).mul_add(dot, f32::MAX) + f32::MAX;
        assert!(expected.is_finite() && expected > 0.0);

        let scalar = [f32::MAX, 0.0, dot, f32::MAX];
        let scalar_output = run_leaf_kernel(&scalar, 2, 1, Metric::L2).1;
        assert_eq!(scalar_output[0].distance.to_bits(), expected.to_bits());

        let points = 17;
        let mut simd = vec![0.0; points * points];
        for point in 0..points {
            simd[point * points + point] = f32::MAX;
        }
        simd[16 * points] = dot;
        let simd_output = run_leaf_kernel(&simd, points, 1, Metric::L2).1;
        assert_eq!(simd_output[16].target, 0);
        assert_eq!(simd_output[16].distance.to_bits(), expected.to_bits());
    }

    #[test]
    fn cosine_clamps_simd_similarity_to_metric_range() {
        let points = 17;
        let mut dots = vec![0.0; points * points];
        for point in 0..points {
            dots[point * points + point] = 1.0;
        }
        dots[16 * points] = 1.000_001;
        dots[16 * points + 1] = -1.000_001;

        let output = run_leaf_kernel(&dots, points, 16, Metric::Cosine).1;
        let source = &output[16 * 16..17 * 16];
        assert_eq!(source[0], LeafNeighbor::new(0, 0.0));
        assert_eq!(source[15], LeafNeighbor::new(1, 2.0));
    }

    #[test]
    fn clamps_leaf_distances_and_cosine_similarity() {
        #[rustfmt::skip]
        let out_of_range = [1.0, 0.0, 2.0, 1.0];
        assert_eq!(
            run_leaf_kernel(&out_of_range, 2, 1, Metric::L2).1[0].distance,
            0.0
        );
        assert_eq!(
            run_leaf_kernel(&out_of_range, 2, 1, Metric::CosineNormalized).1[0].distance,
            -1.0
        );
        assert_eq!(
            run_leaf_kernel(&out_of_range, 2, 1, Metric::Cosine).1[0].distance,
            0.0
        );

        #[rustfmt::skip]
        let opposite = [1.0, 0.0, -2.0, 1.0];
        assert_eq!(
            run_leaf_kernel(&opposite, 2, 1, Metric::Cosine).1[0].distance,
            2.0
        );

        let subnormal = [f32::MIN_POSITIVE / 2.0, 0.0, 1.0, 1.0];
        assert_eq!(
            run_leaf_kernel(&subnormal, 2, 1, Metric::Cosine).1[0].distance,
            1.0
        );

        let minimum_normal = [f32::MIN_POSITIVE, 0.0, f32::MIN_POSITIVE.sqrt(), 1.0];
        assert_eq!(
            run_leaf_kernel(&minimum_normal, 2, 1, Metric::Cosine).1[0].distance,
            0.0
        );
    }

    #[test]
    fn finite_max_distance_fills_the_final_fixed_slot() {
        let points = 4;
        let mut dots = vec![0.0; points * points];
        dots[3 * points] = -f32::MAX;

        let (leaf_k, output) = run_leaf_kernel(&dots, points, 3, Metric::InnerProduct);
        assert_eq!(leaf_k, 3);
        assert_eq!(
            output[3 * leaf_k + leaf_k - 1],
            LeafNeighbor::new(0, f32::MAX)
        );
    }

    #[test]
    fn leaf_metrics_define_nan_candidate_behavior() {
        #[rustfmt::skip]
        let dots = [
            1.0,       0.0, 0.0,
            f32::NAN,  1.0, 0.0,
            0.5,       0.25, 1.0,
        ];

        for metric in [Metric::L2, Metric::Cosine] {
            let output = run_leaf_kernel(&dots, 3, 1, metric).1;
            assert_eq!(output[0], LeafNeighbor::new(1, 0.0), "metric {metric:?}");
            assert_eq!(output[1], LeafNeighbor::new(0, 0.0), "metric {metric:?}");
        }

        for metric in [Metric::CosineNormalized, Metric::InnerProduct] {
            let output = run_leaf_kernel(&dots, 3, 1, metric).1;
            assert_eq!(output[0].target, 2, "metric {metric:?}");
            assert_eq!(output[1].target, 2, "metric {metric:?}");
        }
    }

    #[test]
    fn clamps_k_to_available_non_self_neighbors() {
        #[rustfmt::skip]
        let dots = [
            1.0, 3.0, 3.0,
            0.0, 1.0, 3.0,
            0.0, 0.0, 1.0,
        ];
        let (leaf_k, output) = run_leaf_kernel(&dots, 3, 3, Metric::L2);

        assert_eq!(leaf_k, 2);
        for (source, neighbors) in output.chunks_exact(leaf_k).enumerate() {
            assert!(
                neighbors
                    .iter()
                    .all(|neighbor| neighbor.target as usize != source)
            );
        }
    }

    #[test]
    fn accepts_empty_singleton_and_zero_k_inputs() {
        for (dots, points, requested_k, metric) in [
            (&[][..], 0, 2, Metric::L2),
            (&[4.0][..], 1, 2, Metric::Cosine),
            (&[1.0, 0.0, 0.0, 1.0][..], 2, 0, Metric::InnerProduct),
        ] {
            assert_eq!(run_leaf_kernel(dots, points, requested_k, metric).0, 0);
        }
    }

    #[test]
    fn rejects_invalid_neighbor_counts() {
        let square = [0.0; 9];
        let square_input = test_input(&square, 3);
        let square_norms = prepared_test_norms(Metric::L2, square_input);
        let mut too_many = [LeafNeighbor::default(); 9];
        assert_eq!(
            dispatch_nearest_neighbors(
                Metric::L2,
                square_input,
                &square_norms,
                MutMatrixView::try_from(&mut too_many[..], 3, 3).unwrap(),
                &mut LeafKernelWorkspace::default(),
            ),
            Err(LeafKernelError::InvalidNeighborCount {
                points: 3,
                neighbors: 3,
                maximum: 2,
            })
        );

        let square = [0.0; 25];
        let square_input = test_input(&square, 5);
        let square_norms = prepared_test_norms(Metric::L2, square_input);
        let mut too_wide = [LeafNeighbor::default(); 25];
        assert_eq!(
            dispatch_nearest_neighbors(
                Metric::L2,
                square_input,
                &square_norms,
                MutMatrixView::try_from(&mut too_wide[..], 5, 5).unwrap(),
                &mut LeafKernelWorkspace::default(),
            ),
            Err(LeafKernelError::InvalidNeighborCount {
                points: 5,
                neighbors: 5,
                maximum: 4,
            })
        );
    }

    #[test]
    fn normalized_cosine_keeps_nan_non_rankable_in_scalar_and_simd_paths() {
        let scalar = [1.0, 0.0, f32::NAN, 1.0];
        let scalar_output = run_leaf_kernel(&scalar, 2, 1, Metric::CosineNormalized).1;
        assert_eq!(scalar_output[0], LeafNeighbor::default());

        let points = 17;
        let mut dots = vec![0.0; points * points];
        for point in 0..points {
            dots[point * points + point] = 1.0;
        }
        dots[16 * points] = f32::NAN;
        let simd_output = run_leaf_kernel(&dots, points, 1, Metric::CosineNormalized).1;
        assert_eq!(simd_output[16], LeafNeighbor::new(1, 1.0));
    }
}
