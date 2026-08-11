/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Leaf-local top-k selection from a lower-triangular Gram matrix.
//!
//! The input is an `n × n` [`MatrixView`] from `sgemm_aat_lower`. The diagonal
//! contains metric norms. The kernel reads only the strict lower triangle. It
//! evaluates each point pair once and updates both points.
//!
//! The output is an `n × k` matrix of sorted [`LeafNeighbor`] values. Each target
//! is a position in the leaf. The kernel supports `k` from zero through
//! [`MAX_LEAF_NEIGHBORS`]. Positive widths use fixed arrays.
//!
//! Strict comparisons keep scan order for equal distances. They do not rank NaN.
//! All supported metrics use the same scalar and SIMD traversal.
//!
//! The caller supplies concrete architecture `A` and metric `M`. The function
//! checks all shapes and local-ID bounds before it changes workspace or uses an
//! unchecked SIMD load. [`LeafKernelWorkspace`] stores reusable norms and
//! rejection thresholds.

use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_wide::{Architecture, Const, SIMDFloat, SIMDMask, SIMDSelect, SIMDVector};

use super::kernel_metric::LeafKernelMetric;

/// Largest leaf-local neighbor count supported by the fixed insertion kernel.
pub(super) const MAX_LEAF_NEIGHBORS: usize = 3;

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
}

impl Default for LeafNeighbor {
    fn default() -> Self {
        Self::new(u32::MAX, f32::INFINITY)
    }
}

/// Reusable temporary storage for leaf top-k selection.
#[derive(Debug, Default)]
pub(super) struct LeafKernelWorkspace {
    worst: Vec<f32>,
}

/// Validation or allocation error returned by [`nearest_neighbors`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(super) enum LeafKernelError {
    /// The point count cannot be represented in leaf-local `u32` positions.
    #[error("point count {0} exceeds the u32 position limit")]
    TooManyPoints(usize),
    /// The dot-product matrix is not square.
    #[error("leaf dot-product matrix must be square, got {rows} x {cols}")]
    NonSquareDots { rows: usize, cols: usize },
    /// The output matrix does not have one row per input point.
    #[error("invalid output row count: expected {expected}, got {actual} with {columns} columns")]
    InvalidOutputRows {
        expected: usize,
        actual: usize,
        columns: usize,
    },
    /// A source requests more neighbors than the leaf or fixed kernel supports.
    #[error("invalid leaf neighbor count {neighbors} for {points} points; maximum is {maximum}")]
    InvalidNeighborCount {
        points: usize,
        neighbors: usize,
        maximum: usize,
    },
    /// The prepared norm count does not match the point count.
    #[error("invalid leaf norm count: expected {expected}, got {actual}")]
    InvalidNormCount { expected: usize, actual: usize },
    /// Temporary storage could not be reserved.
    #[error("failed to reserve {additional} values for {buffer}")]
    Allocation {
        buffer: &'static str,
        additional: usize,
    },
    /// A source did not contain enough rankable targets to fill its output.
    #[error("source {source_index} has fewer than {neighbors} rankable leaf neighbors")]
    InsufficientRankableNeighbors {
        source_index: usize,
        neighbors: usize,
    },
}

/// Return the non-self neighbor count for one leaf.
///
/// `points` is the number of points in the leaf. `requested_k` is the configured
/// neighbor count. The result is `min(requested_k, points - 1)`. The function
/// rejects a value above [`MAX_LEAF_NEIGHBORS`].
///
/// # Errors
///
/// Returns [`LeafKernelError::TooManyPoints`] when leaf-local positions cannot
/// fit in `u32`, or [`LeafKernelError::InvalidNeighborCount`] when `requested_k`
/// exceeds [`MAX_LEAF_NEIGHBORS`].
pub(super) fn leaf_neighbor_count(
    points: usize,
    requested_k: usize,
) -> Result<usize, LeafKernelError> {
    if points > u32::MAX as usize {
        return Err(LeafKernelError::TooManyPoints(points));
    }
    if requested_k > MAX_LEAF_NEIGHBORS {
        return Err(LeafKernelError::InvalidNeighborCount {
            points,
            neighbors: requested_k,
            maximum: MAX_LEAF_NEIGHBORS,
        });
    }
    Ok(requested_k.min(points.saturating_sub(1)))
}

/// Select the nearest non-self positions for each point in a leaf.
///
/// `output` has one row for each input point. Its column count requests the
/// neighbor count. The function checks this shape and the supported count before
/// it changes output. Equal distances keep pair scan order.
///
/// # Errors
///
/// Returns [`LeafKernelError`] for an invalid shape or count. It also returns an
/// error for allocation failure or insufficient rankable neighbors.
pub(super) fn nearest_neighbors<A, M>(
    arch: A,
    input: MatrixView<'_, f32>,
    norms: &[f32],
    mut output: MutMatrixView<'_, LeafNeighbor>,
    workspace: &mut LeafKernelWorkspace,
) -> Result<(), LeafKernelError>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    M: LeafKernelMetric,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    validate(input, norms, &output)?;
    let neighbor_count = output.ncols();
    if neighbor_count == 0 {
        return Ok(());
    }

    resize(
        "worst distances",
        &mut workspace.worst,
        input.nrows(),
        f32::INFINITY,
    )?;
    output.as_mut_slice().fill(LeafNeighbor::default());
    workspace.worst.fill(f32::INFINITY);

    match (norms.is_empty(), neighbor_count) {
        (false, 1) => scan_point_pairs::<A::f32x16, M, 1, true>(
            arch,
            input,
            output.as_mut_slice(),
            norms,
            &mut workspace.worst,
        ),
        (false, 2) => scan_point_pairs::<A::f32x16, M, 2, true>(
            arch,
            input,
            output.as_mut_slice(),
            norms,
            &mut workspace.worst,
        ),
        (false, 3) => scan_point_pairs::<A::f32x16, M, 3, true>(
            arch,
            input,
            output.as_mut_slice(),
            norms,
            &mut workspace.worst,
        ),
        (true, 1) => scan_point_pairs::<A::f32x16, M, 1, false>(
            arch,
            input,
            output.as_mut_slice(),
            norms,
            &mut workspace.worst,
        ),
        (true, 2) => scan_point_pairs::<A::f32x16, M, 2, false>(
            arch,
            input,
            output.as_mut_slice(),
            norms,
            &mut workspace.worst,
        ),
        (true, 3) => scan_point_pairs::<A::f32x16, M, 3, false>(
            arch,
            input,
            output.as_mut_slice(),
            norms,
            &mut workspace.worst,
        ),
        _ => {
            return Err(LeafKernelError::InvalidNeighborCount {
                points: input.nrows(),
                neighbors: neighbor_count,
                maximum: MAX_LEAF_NEIGHBORS,
            });
        }
    }
    if let Some(source) = output
        .as_slice()
        .chunks_exact(neighbor_count)
        .position(|neighbors| neighbors[neighbor_count - 1].target == u32::MAX)
    {
        return Err(LeafKernelError::InsufficientRankableNeighbors {
            source_index: source,
            neighbors: neighbor_count,
        });
    }
    Ok(())
}

/// Check the safety conditions for the SIMD kernel.
///
/// The matrix views already prove their backing lengths. This function checks
/// that the dot matrix is square. It also checks local-ID range and output width.
/// An error occurs before the kernel changes output or workspace.
fn validate(
    input: MatrixView<'_, f32>,
    norms: &[f32],
    output: &MutMatrixView<'_, LeafNeighbor>,
) -> Result<(), LeafKernelError> {
    let point_count = input.nrows();
    let dot_columns = input.ncols();
    if point_count > u32::MAX as usize {
        return Err(LeafKernelError::TooManyPoints(point_count));
    }
    if point_count != dot_columns {
        return Err(LeafKernelError::NonSquareDots {
            rows: point_count,
            cols: dot_columns,
        });
    }
    if !norms.is_empty() && norms.len() != point_count {
        return Err(LeafKernelError::InvalidNormCount {
            expected: point_count,
            actual: norms.len(),
        });
    }
    if output.nrows() != point_count {
        return Err(LeafKernelError::InvalidOutputRows {
            expected: point_count,
            actual: output.nrows(),
            columns: output.ncols(),
        });
    }
    let maximum_neighbors = point_count.saturating_sub(1).min(MAX_LEAF_NEIGHBORS);
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

fn resize<T: Clone>(
    buffer: &'static str,
    values: &mut Vec<T>,
    len: usize,
    value: T,
) -> Result<(), LeafKernelError> {
    let additional = len.saturating_sub(values.len());
    values
        .try_reserve(additional)
        .map_err(|_| LeafKernelError::Allocation { buffer, additional })?;
    values.resize(len, value);
    Ok(())
}

/// Select neighbors from all unordered point pairs in one leaf.
///
/// The function reads the strict lower triangle once. It offers each distance to
/// both endpoint lists. SIMD groups and the scalar tail preserve pair scan order.
///
/// `input` supplies square dot products. `norms` contains prepared norm values.
#[inline(never)]
fn scan_point_pairs<F, M, const N: usize, const USES_NORMS: bool>(
    arch: F::Arch,
    input: MatrixView<'_, f32>,
    output: &mut [LeafNeighbor],
    norms: &[f32],
    worst: &mut [f32],
) where
    F: SIMDVector<Scalar = f32, ConstLanes = Const<16>> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    M: LeafKernelMetric,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let (output, _) = output.as_chunks_mut::<N>();
    let point_count = input.nrows();
    let dots = input.as_slice();
    let worst_ptr = worst.as_mut_ptr();

    // Source zero has no earlier target. Each source after zero can still add
    // itself to the neighbor list of source zero.
    for source in 1..point_count {
        let source_start = source * point_count;
        let source_norm = if USES_NORMS { norms[source] } else { 0.0 };
        let source_norms = F::splat(arch, source_norm);
        // SAFETY: `nearest_neighbors` created one threshold for each point.
        let mut source_worst = unsafe { *worst_ptr.add(source) };
        let mut target = 0;
        let full = source / F::LANES * F::LANES;

        while target < full {
            // SAFETY: the full chunk is contained in this source's strict-lower prefix.
            let pair_dots = unsafe { F::load_simd(arch, dots.as_ptr().add(source_start + target)) };
            let target_norms = if USES_NORMS {
                // SAFETY: the full target chunk is below `source < point_count`.
                unsafe { F::load_simd(arch, norms.as_ptr().add(target)) }
            } else {
                F::default(arch)
            };
            let distances = M::leaf_distance(arch, pair_dots, source_norms, target_norms);
            // Every pair may improve the current source and its earlier target.
            // Derive both masks before either endpoint mutates its threshold.
            let source_eligible = distances.lt_simd(F::splat(arch, source_worst));
            // SAFETY: the full target chunk is below `source < point_count`.
            // `nearest_neighbors` created one threshold for each point.
            let target_worst = unsafe { F::load_simd(arch, worst_ptr.add(target)) };
            let target_eligible = distances.lt_simd(target_worst);
            let source_bits = u64::from(source_eligible.bitmask().to_underlying());
            let target_bits = u64::from(target_eligible.bitmask().to_underlying());

            if source_bits | target_bits != 0 {
                let values: [f32; 16] = distances.to_array();
                let mut source_bits = source_bits;
                while source_bits != 0 {
                    let lane = source_bits.trailing_zeros() as usize;
                    source_bits &= source_bits - 1;
                    let distance = values[lane];
                    if distance < source_worst {
                        source_worst = insert_fixed_neighbor(
                            &mut output[source],
                            (target + lane) as u32,
                            distance,
                        );
                    }
                }

                let mut target_bits = target_bits;
                while target_bits != 0 {
                    let lane = target_bits.trailing_zeros() as usize;
                    target_bits &= target_bits - 1;
                    let target_source = target + lane;
                    let new_worst = insert_fixed_neighbor(
                        &mut output[target_source],
                        source as u32,
                        values[lane],
                    );
                    // SAFETY: `target_source < source < worst.len()`.
                    unsafe { *worst_ptr.add(target_source) = new_worst };
                }
            }
            target += F::LANES;
        }

        while target < source {
            // SAFETY: the scalar target remains in this source's strict-lower prefix.
            let dot = unsafe { *dots.get_unchecked(source_start + target) };
            let target_norm = if USES_NORMS {
                // SAFETY: `target < source < point_count == norms.len()`.
                unsafe { *norms.get_unchecked(target) }
            } else {
                0.0
            };
            let distance = M::leaf_distance_scalar(dot, source_norm, target_norm);
            if distance < source_worst {
                source_worst = insert_fixed_neighbor(&mut output[source], target as u32, distance);
            }
            // SAFETY: `target < source < worst.len()`.
            let target_worst = unsafe { *worst_ptr.add(target) };
            if distance < target_worst {
                let new_worst = insert_fixed_neighbor(&mut output[target], source as u32, distance);
                // SAFETY: `target < source < worst.len()`.
                unsafe { *worst_ptr.add(target) = new_worst };
            }
            target += 1;
        }
        // SAFETY: `source < worst.len()`.
        unsafe { *worst_ptr.add(source) = source_worst };
    }
}

/// Insert one target point into a source point's retained neighbor set.
///
/// `N` is the configured leaf neighbor count. The candidate is closer than the
/// current farthest neighbor. Equal distances keep pair scan order. The function
/// returns the new farthest retained distance.
#[inline(always)]
fn insert_fixed_neighbor<const N: usize>(
    neighbors: &mut [LeafNeighbor; N],
    target: u32,
    distance: f32,
) -> f32 {
    let entry = LeafNeighbor::new(target, distance);
    match N {
        1 => {
            neighbors[0] = entry;
            distance
        }
        2 => {
            let first = neighbors[0];
            if distance < first.distance {
                neighbors[0] = entry;
                neighbors[1] = first;
                first.distance
            } else {
                neighbors[1] = entry;
                distance
            }
        }
        3 => {
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
        _ => f32::INFINITY,
    }
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
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    fn run(self, arch: A, call: DispatchedLeafCall<'_>) -> Result<(), LeafKernelError> {
        use super::kernel_metric::{Cosine, CosineNormalized, InnerProduct, L2};
        use diskann_vector::distance::Metric;

        match self.0 {
            Metric::L2 => nearest_neighbors::<A, L2>(
                arch,
                call.input,
                call.norms,
                call.output,
                call.workspace,
            ),
            Metric::Cosine => nearest_neighbors::<A, Cosine>(
                arch,
                call.input,
                call.norms,
                call.output,
                call.workspace,
            ),
            Metric::CosineNormalized => nearest_neighbors::<A, CosineNormalized>(
                arch,
                call.input,
                call.norms,
                call.output,
                call.workspace,
            ),
            Metric::InnerProduct => nearest_neighbors::<A, InnerProduct>(
                arch,
                call.input,
                call.norms,
                call.output,
                call.workspace,
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
    use diskann_vector::distance::Metric;

    match metric {
        Metric::L2 => (0..input.nrows())
            .map(|point| input[(point, point)])
            .collect(),
        Metric::Cosine => (0..input.nrows())
            .map(|point| super::kernel_metric::norm_from_squared(input[(point, point)]))
            .collect(),
        Metric::CosineNormalized | Metric::InnerProduct => Vec::new(),
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
    fn neighbor_count_clamps_to_non_self_neighbors_and_rejects_large_k() {
        assert_eq!(leaf_neighbor_count(0, 3).unwrap(), 0);
        assert_eq!(leaf_neighbor_count(1, 3).unwrap(), 0);
        assert_eq!(leaf_neighbor_count(4, 3).unwrap(), 3);
        assert_eq!(
            leaf_neighbor_count(4, 4),
            Err(LeafKernelError::InvalidNeighborCount {
                points: 4,
                neighbors: 4,
                maximum: MAX_LEAF_NEIGHBORS,
            })
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            leaf_neighbor_count(u32::MAX as usize + 1, 1),
            Err(LeafKernelError::TooManyPoints(u32::MAX as usize + 1))
        );
    }

    #[test]
    fn kernel_accepts_different_neighbor_counts() {
        let points = 7;
        let dots = test_dots(Metric::L2, points);
        let input = test_input(&dots, points);
        let norms = prepared_test_norms(Metric::L2, input);
        let mut workspace = LeafKernelWorkspace::default();

        for neighbor_count in [1, 3, 2] {
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
        LeafKernelError, LeafKernelWorkspace, LeafNeighbor, MAX_LEAF_NEIGHBORS,
        dispatch_nearest_neighbors, leaf_neighbor_count, prepared_test_norms,
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
                    Metric::L2 => clamp(norms[source] + norms[target] - 2.0 * dot),
                    Metric::CosineNormalized => clamp(1.0 - dot),
                    Metric::InnerProduct => -dot,
                    Metric::Cosine => {
                        let denominator = norms[source] * norms[target];
                        let similarity = if denominator == 0.0 {
                            0.0
                        } else {
                            dot / denominator
                        };
                        clamp(1.0 - similarity)
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
        let leaf_k = leaf_neighbor_count(points, requested_k).unwrap();
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
                for requested_k in [1, 2, 3] {
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
    fn clamps_negative_distances_and_preserves_cosine_extremes() {
        #[rustfmt::skip]
        let out_of_range = [1.0, 0.0, 2.0, 1.0];
        assert_eq!(
            run_leaf_kernel(&out_of_range, 2, 1, Metric::L2).1[0].distance,
            0.0
        );
        assert_eq!(
            run_leaf_kernel(&out_of_range, 2, 1, Metric::CosineNormalized).1[0].distance,
            0.0
        );
        assert_eq!(
            run_leaf_kernel(&out_of_range, 2, 1, Metric::Cosine).1[0].distance,
            0.0
        );

        #[rustfmt::skip]
        let opposite = [1.0, 0.0, -2.0, 1.0];
        assert_eq!(
            run_leaf_kernel(&opposite, 2, 1, Metric::Cosine).1[0].distance,
            3.0
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

        let (leaf_k, output) =
            run_leaf_kernel(&dots, points, MAX_LEAF_NEIGHBORS, Metric::InnerProduct);
        assert_eq!(leaf_k, MAX_LEAF_NEIGHBORS);
        assert_eq!(
            output[3 * leaf_k + leaf_k - 1],
            LeafNeighbor::new(0, f32::MAX)
        );
    }

    #[test]
    fn every_metric_ignores_nan_pairs() {
        #[rustfmt::skip]
        let dots = [
            1.0,       0.0, 0.0,
            f32::NAN,  1.0, 0.0,
            0.5,       0.25, 1.0,
        ];

        for metric in [
            Metric::L2,
            Metric::Cosine,
            Metric::CosineNormalized,
            Metric::InnerProduct,
        ] {
            let output = run_leaf_kernel(&dots, 3, 1, metric).1;
            assert_eq!(output[0].target, 2, "metric {metric:?}");
            assert_eq!(output[1].target, 2, "metric {metric:?}");
        }
    }

    #[test]
    fn rejects_sources_with_too_few_rankable_neighbors() {
        let dots = [1.0, 0.0, f32::NAN, 1.0];
        let mut output = [LeafNeighbor::default(); 2];
        let input = test_input(&dots, 2);
        let norms = prepared_test_norms(Metric::L2, input);
        let error = dispatch_nearest_neighbors(
            Metric::L2,
            input,
            &norms,
            MutMatrixView::try_from(&mut output[..], 2, 1).unwrap(),
            &mut LeafKernelWorkspace::default(),
        )
        .unwrap_err();

        assert_eq!(
            error,
            LeafKernelError::InsufficientRankableNeighbors {
                source_index: 0,
                neighbors: 1
            }
        );
    }

    #[test]
    fn clamps_k_to_available_non_self_neighbors() {
        #[rustfmt::skip]
        let dots = [
            1.0, 3.0, 3.0,
            0.0, 1.0, 3.0,
            0.0, 0.0, 1.0,
        ];
        let (leaf_k, output) = run_leaf_kernel(&dots, 3, MAX_LEAF_NEIGHBORS, Metric::L2);

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
    fn rejects_non_square_input_and_invalid_output_dimensions() {
        let dots = [0.0; 6];
        let non_square = MatrixView::try_from(&dots[..], 2, 3).unwrap();
        let mut output = [LeafNeighbor::default(); 2];
        assert_eq!(
            dispatch_nearest_neighbors(
                Metric::L2,
                non_square,
                &[],
                MutMatrixView::try_from(&mut output[..], 2, 1).unwrap(),
                &mut LeafKernelWorkspace::default(),
            ),
            Err(LeafKernelError::NonSquareDots { rows: 2, cols: 3 })
        );

        let square = [0.0; 9];
        let square_input = test_input(&square, 3);
        let square_norms = prepared_test_norms(Metric::L2, square_input);
        let mut valid_output = [LeafNeighbor::default(); 3];
        assert_eq!(
            dispatch_nearest_neighbors(
                Metric::L2,
                square_input,
                &square_norms[..2],
                MutMatrixView::try_from(&mut valid_output[..], 3, 1).unwrap(),
                &mut LeafKernelWorkspace::default(),
            ),
            Err(LeafKernelError::InvalidNormCount {
                expected: 3,
                actual: 2,
            })
        );

        let mut wrong_rows = [LeafNeighbor::default(); 2];
        assert_eq!(
            dispatch_nearest_neighbors(
                Metric::L2,
                square_input,
                &square_norms,
                MutMatrixView::try_from(&mut wrong_rows[..], 2, 1).unwrap(),
                &mut LeafKernelWorkspace::default(),
            ),
            Err(LeafKernelError::InvalidOutputRows {
                expected: 3,
                actual: 2,
                columns: 1,
            })
        );

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
        let mut too_wide = [LeafNeighbor::default(); 20];
        assert_eq!(
            dispatch_nearest_neighbors(
                Metric::L2,
                square_input,
                &square_norms,
                MutMatrixView::try_from(&mut too_wide[..], 5, 4).unwrap(),
                &mut LeafKernelWorkspace::default(),
            ),
            Err(LeafKernelError::InvalidNeighborCount {
                points: 5,
                neighbors: 4,
                maximum: MAX_LEAF_NEIGHBORS,
            })
        );
    }

    #[test]
    fn cosine_zero_norm_masks_nan_norm_at_simd_boundaries() {
        for points in [9, 17] {
            let mut dots = vec![0.0; points * points];
            for source in 1..points {
                dots[source * points + source] = f32::NAN;
            }

            let output = run_leaf_kernel(&dots, points, 1, Metric::Cosine).1;
            for (source, neighbor) in output.iter().enumerate().skip(1) {
                assert_eq!(
                    *neighbor,
                    LeafNeighbor::new(0, 1.0),
                    "n={points}, source={source}"
                );
            }
        }
    }
}
