/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Prepared leaf-local top-k selection over a lower-triangular Gram matrix.
//!
//! Caller supplies an `n × n` [`MatrixView`] produced by `sgemm_aat_lower`.
//! Diagonal entries provide metric scales; only strict-lower pair dots are read.
//! Each pair is evaluated once and offered to both endpoint rows.
//!
//! Output is an `n × k` matrix of sorted [`LeafNeighbor`] values with leaf-local
//! targets. Supported `k` is zero through [`MAX_LEAF_NEIGHBORS`]; positive widths
//! use fixed arrays. Strict comparisons preserve encounter order for ties and
//! reject NaN. L2, cosine, normalized cosine, and inner product share the same
//! scalar/SIMD traversal.
//!
//! [`LeafKernel::new`] selects metric and runtime architecture once, storing one
//! direct function pointer reused across leaves. Every call validates square
//! shape, row count, local-ID bounds, and output width before scratch mutation or
//! unchecked SIMD loads. [`LeafKernelWorkspace`] retains per-worker norm and
//! threshold buffers.
//!
//! Work is `n(n - 1) / 2` distance evaluations with constant bounded insertion;
//! scratch is `O(n)` and output is `O(nk)`.

use std::marker::PhantomData;

use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_vector::distance::Metric;
use diskann_wide::{
    Architecture, Const, SIMDFloat, SIMDMask, SIMDSelect, SIMDVector,
    arch::{self, Dispatched1, FTarget1},
    lifetime::AddLifetime,
};

use super::kernel_metric::{KernelMetric, MetricVisitor, visit_metric};

/// Largest leaf-local neighbor count supported by the fixed insertion kernel.
pub const MAX_LEAF_NEIGHBORS: usize = 3;

/// One leaf-local neighbor and its metric distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LeafNeighbor {
    /// Target position in the leaf, not a dataset ID.
    pub target: u32,
    /// Distance from the source point to `target`.
    pub distance: f32,
}

impl LeafNeighbor {
    /// Construct a leaf-local neighbor.
    ///
    /// `target` is a position in the current leaf and `distance` is its score
    /// from the source represented by the containing output row.
    pub const fn new(target: u32, distance: f32) -> Self {
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
pub struct LeafKernelWorkspace {
    norms: Vec<f32>,
    worst: Vec<f32>,
}

impl LeafKernelWorkspace {
    /// Construct an empty workspace.
    ///
    /// This does not allocate. First use grows buffers to the leaf point count;
    /// later calls reuse capacity owned by the same worker.
    pub const fn new() -> Self {
        Self {
            norms: Vec::new(),
            worst: Vec::new(),
        }
    }
}

/// Validation or allocation error returned by [`LeafKernel::nearest_neighbors`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LeafKernelError {
    /// The point count cannot be represented in leaf-local `u32` positions.
    #[error("point count {0} exceeds the u32 position limit")]
    TooManyPoints(usize),
    /// The dot-product matrix is not square.
    #[error("leaf dot-product matrix must be square, got {rows} x {cols}")]
    NonSquareDots {
        /// Supplied row count.
        rows: usize,
        /// Supplied column count.
        cols: usize,
    },
    /// A declared output shape overflowed `usize`.
    #[error("{buffer} shape {rows} x {cols} overflows usize")]
    ShapeOverflow {
        /// Name of the buffer whose shape overflowed.
        buffer: &'static str,
        /// Declared row count.
        rows: usize,
        /// Declared column count.
        cols: usize,
    },
    /// The output matrix does not have one row per input point.
    #[error("invalid output row count: expected {expected}, got {actual} with {columns} columns")]
    InvalidOutputRows {
        /// Required row count.
        expected: usize,
        /// Supplied row count.
        actual: usize,
        /// Supplied neighbor columns.
        columns: usize,
    },
    /// A source requests more neighbors than the leaf or fixed kernel supports.
    #[error("invalid leaf neighbor count {neighbors} for {points} points; maximum is {maximum}")]
    InvalidNeighborCount {
        /// Point count in the leaf.
        points: usize,
        /// Supplied output-column count.
        neighbors: usize,
        /// Maximum non-self neighbors per point.
        maximum: usize,
    },
    /// Temporary storage could not be reserved.
    #[error("failed to reserve {additional} values for {buffer}")]
    Allocation {
        /// Name of the temporary buffer.
        buffer: &'static str,
        /// Additional element capacity requested.
        additional: usize,
    },
    /// A source did not contain enough rankable targets to fill its output.
    #[error("source {source_index} has fewer than {neighbors} rankable leaf neighbors")]
    InsufficientRankableNeighbors {
        /// Zero-based source position in the leaf.
        source_index: usize,
        /// Required number of non-self neighbors.
        neighbors: usize,
    },
}

/// Return the usable non-self neighbor count for one leaf.
///
/// `points` is the leaf point count and `requested_k` is the build-wide target.
/// Values above [`MAX_LEAF_NEIGHBORS`] are rejected. Otherwise the returned
/// width is `min(requested_k, points - 1)`, allowing empty, singleton, and small
/// leaves without a second effective-k state.
///
/// # Errors
///
/// Returns [`LeafKernelError::TooManyPoints`] when leaf-local positions cannot
/// fit in `u32`, or [`LeafKernelError::InvalidNeighborCount`] when `requested_k`
/// exceeds [`MAX_LEAF_NEIGHBORS`].
///
/// # Performance
///
/// Constant-time and allocation-free.
pub fn leaf_neighbor_count(points: usize, requested_k: usize) -> Result<usize, LeafKernelError> {
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

/// Return the required output length for [`LeafKernel::nearest_neighbors`].
///
/// `points` and `requested_k` have the same meaning as in
/// [`leaf_neighbor_count`]. The return value is `points * effective_k`.
///
/// # Errors
///
/// Returns [`LeafKernelError::TooManyPoints`] or
/// [`LeafKernelError::ShapeOverflow`] instead of wrapping the output area.
///
/// # Performance
///
/// Constant-time and allocation-free.
pub fn leaf_output_len(points: usize, requested_k: usize) -> Result<usize, LeafKernelError> {
    checked_area("output", points, leaf_neighbor_count(points, requested_k)?)
}

/// One invocation bundled for `Dispatched1`.
///
/// `AddLifetime` can attach one lifetime to this aggregate, allowing the direct
/// function-pointer interface to carry the input view, exclusive output view,
/// and exclusive scratch lease without storing any of them in `LeafKernel`.
#[derive(Debug)]
struct LeafCall<'a> {
    input: MatrixView<'a, f32>,
    output: MutMatrixView<'a, LeafNeighbor>,
    workspace: &'a mut LeafKernelWorkspace,
}

#[derive(Debug)]
struct LeafCallArg;

impl AddLifetime for LeafCallArg {
    type Of<'a> = LeafCall<'a>;
}

type LeafFn = Dispatched1<Result<(), LeafKernelError>, LeafCallArg>;

/// A leaf kernel prepared for one metric and the current CPU.
///
/// Construct this once with [`LeafKernel::new`] and share it across leaf workers.
/// Each output view carries its leaf-specific neighbor width.
///
/// The handle stores only one direct function pointer. It borrows no leaf data
/// or workspace and is therefore `Copy`, `Send`, and `Sync`.
#[derive(Clone, Copy, Debug)]
pub struct LeafKernel {
    run: LeafFn,
}

impl LeafKernel {
    /// Prepare a leaf kernel for `metric` and the current CPU.
    ///
    /// The returned handle contains one architecture/metric-specialized function
    /// pointer and can process any valid leaf size or neighbor width.
    ///
    /// # Performance
    ///
    /// Performs runtime architecture detection and one metric match once.
    /// Reusing the handle keeps both decisions out of per-leaf hot loops.
    pub fn new(metric: Metric) -> Self {
        diskann_wide::arch::dispatch1_no_features(PrepareLeaf, metric)
    }

    /// Select the nearest non-self leaf positions for every source point.
    ///
    /// `output` must have one row per input point. Its column count is the
    /// neighbor count for this leaf and must not exceed either `point_count - 1`
    /// or [`MAX_LEAF_NEIGHBORS`].
    /// Equal distances retain pair scan order.
    ///
    /// `input` supplies the square lower-triangular dot matrix. `output` is
    /// overwritten with sorted leaf-local neighbors. `workspace` is an exclusive
    /// worker-owned scratch lease whose capacity is retained after return.
    /// Successful return guarantees every source has exactly `output.ncols()`
    /// rankable, non-self neighbors.
    ///
    /// # Core flow
    ///
    /// The prepared entry validates every view before mutation, prepares scales,
    /// clears output and thresholds, scans the strict lower triangle once, then
    /// verifies the final slot of every source. Each pair updates both endpoints.
    ///
    /// # Errors
    ///
    /// Returns [`LeafKernelError`] for invalid or overflowing shapes, excessive
    /// point/neighbor counts, scratch allocation failure, or an underfilled
    /// source caused by non-rankable distances. Validation errors leave output
    /// and workspace contents unchanged.
    ///
    /// # Performance
    ///
    /// See module-level complexity. This call uses the prepared direct function
    /// pointer; it performs no runtime ISA or metric dispatch.
    pub fn nearest_neighbors(
        &self,
        input: MatrixView<'_, f32>,
        output: MutMatrixView<'_, LeafNeighbor>,
        workspace: &mut LeafKernelWorkspace,
    ) -> Result<(), LeafKernelError> {
        self.run.call(LeafCall {
            input,
            output,
            workspace,
        })
    }
}

/// First dispatch stage: choose the runtime architecture once.
///
/// The factory itself uses `dispatch1_no_features`; only the returned leaf entry
/// needs target features, so architecture-specific code remains behind the final
/// direct function pointer.
struct PrepareLeaf;

impl<A> arch::Target1<A, LeafKernel, Metric> for PrepareLeaf
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    fn run(self, arch: A, metric: Metric) -> LeafKernel {
        visit_metric(metric, BuildLeaf(arch))
    }
}

/// Metric visitor holding a concrete architecture.
///
/// `visit<M>` combines architecture `A` and concrete metric `M` into exactly
/// one `Dispatched1`. Leaf width remains call data because it varies by leaf.
struct BuildLeaf<A>(A);

impl<A> MetricVisitor for BuildLeaf<A>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    type Output = LeafKernel;

    fn visit<M: KernelMetric>(self) -> Self::Output {
        LeafKernel {
            run: self
                .0
                .dispatch1::<LeafEntry<M>, Result<(), LeafKernelError>, LeafCallArg>(),
        }
    }
}

/// Architecture/metric-specialized function-pointer destination.
///
/// This type is zero-sized. All per-leaf state, including output width, arrives
/// through `LeafCall`; validation completes before pointer-based SIMD executes.
///
/// Call order is fixed: validate without mutation, allocate/reset scratch,
/// initialize output, execute one specialized traversal, then verify fill state.
/// Keeping those phases in the dispatched destination makes every unchecked
/// load depend on one visible validation gate.
struct LeafEntry<M>(PhantomData<M>);

impl<A, M> FTarget1<A, Result<(), LeafKernelError>, LeafCall<'_>> for LeafEntry<M>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    M: KernelMetric,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    fn run(arch: A, mut call: LeafCall<'_>) -> Result<(), LeafKernelError> {
        // Validation establishes every shape and active-prefix invariant used by
        // unchecked loads below. No output or scratch mutation occurs on error.
        validate(call.input, &call.output)?;
        let neighbor_count = call.output.ncols();
        // Empty or singleton leaves request zero columns. Avoid touching scratch
        // or output so this path remains allocation-free.
        if neighbor_count == 0 {
            return Ok(());
        }

        // Norm and threshold scratch are reset for this leaf, while Vec capacity
        // remains reusable by the worker that owns the workspace.
        prepare_workspace::<M>(call.input, call.workspace)?;
        call.output.as_mut_slice().fill(LeafNeighbor::default());
        call.workspace.worst.fill(f32::INFINITY);

        // Width dispatch happens once per leaf. Common production widths become
        // fixed arrays; uncommon widths retain the same traversal through slices.
        process_neighbor_width::<A::f32x16, M>(
            arch,
            call.input,
            neighbor_count,
            call.output.as_mut_slice(),
            &call.workspace.norms,
            &mut call.workspace.worst,
        )?;
        // Sorted lists use the last slot as both worst-distance threshold and
        // underfill sentinel, so one slot check per source proves full output.
        if let Some(source) = call
            .output
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
}

/// Validate the complete safety contract before dispatched SIMD executes.
///
/// `MatrixView` and `MutMatrixView` construction guarantee exact, non-overflowing
/// backing lengths. This check establishes square dots, representable local IDs,
/// and an output width bounded by the point count and fixed kernel capacity.
/// Failure returns [`LeafKernelError`] before output or workspace mutation.
fn validate(
    input: MatrixView<'_, f32>,
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

/// Prepare metric-specific scale and threshold scratch.
///
/// L2 stores diagonal squared norms; cosine converts diagonals to norms using
/// DiskANN's zero threshold. Normalized cosine and inner product skip the norm
/// allocation entirely. `worst` is reset separately after allocation succeeds.
///
/// `input` supplies diagonal dots and `workspace` owns reusable vectors. Success
/// prepares one scale and one threshold per point when needed; allocation failure
/// is returned without entering SIMD traversal. Work is `O(n)`, with at most
/// `O(n)` retained capacity per buffer.
fn prepare_workspace<M: KernelMetric>(
    input: MatrixView<'_, f32>,
    workspace: &mut LeafKernelWorkspace,
) -> Result<(), LeafKernelError> {
    let points = input.nrows();
    if M::LEAF_SCALE.is_some() {
        resize("norms", &mut workspace.norms, points, 0.0)?;
        for (source, norm) in workspace.norms.iter_mut().enumerate() {
            *norm = M::LEAF_SCALE.transform(input[(source, source)]);
        }
    } else {
        workspace.norms.clear();
    }
    resize(
        "worst distances",
        &mut workspace.worst,
        points,
        f32::INFINITY,
    )
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

fn checked_area(buffer: &'static str, rows: usize, cols: usize) -> Result<usize, LeafKernelError> {
    rows.checked_mul(cols)
        .ok_or(LeafKernelError::ShapeOverflow { buffer, rows, cols })
}

/// Convert the validated neighbor count into fixed source storage.
///
/// This branch runs once per leaf. `as_chunks_mut` performs one safe conversion,
/// avoiding per-candidate slice-to-array checks. `output` contains
/// `point_count * neighbor_count` initialized slots; `norms` and `worst` satisfy
/// the invariants established by `prepare_workspace`.
fn process_neighbor_width<F, M>(
    arch: F::Arch,
    input: MatrixView<'_, f32>,
    neighbor_count: usize,
    output: &mut [LeafNeighbor],
    norms: &[f32],
    worst: &mut [f32],
) -> Result<(), LeafKernelError>
where
    F: SIMDVector<Scalar = f32, ConstLanes = Const<16>> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    M: KernelMetric,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    match neighbor_count {
        1 => process_fixed_width::<F, M, 1>(arch, input, output, norms, worst),
        2 => process_fixed_width::<F, M, 2>(arch, input, output, norms, worst),
        3 => process_fixed_width::<F, M, 3>(arch, input, output, norms, worst),
        _ => {
            return Err(LeafKernelError::InvalidNeighborCount {
                points: input.nrows(),
                neighbors: neighbor_count,
                maximum: MAX_LEAF_NEIGHBORS,
            });
        }
    }
    Ok(())
}

/// Reinterpret validated output as one fixed array per source, then run shared
/// pair traversal.
///
/// `N` is one, two, or three. `as_chunks_mut` performs one safe shape split per
/// leaf, keeping array conversion out of candidate insertion.
fn process_fixed_width<F, M, const N: usize>(
    arch: F::Arch,
    input: MatrixView<'_, f32>,
    output: &mut [LeafNeighbor],
    norms: &[f32],
    worst: &mut [f32],
) where
    F: SIMDVector<Scalar = f32, ConstLanes = Const<16>> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    M: KernelMetric,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let (neighbor_lists, remainder) = output.as_chunks_mut::<N>();
    debug_assert!(remainder.is_empty());
    process_pairs::<F, M, N>(arch, input, neighbor_lists, norms, worst);
}

/// Scan the strict lower triangle once and update both endpoint sources.
///
/// Invariants on entry:
///
/// - `dots` is a validated square row-major matrix;
/// - `output` has one sorted neighbor list per source point;
/// - `worst[source]` equals that source's last slot;
/// - `norms` has one value per point exactly when `M` requires scales.
///
/// Each SIMD chunk computes both endpoint eligibility masks before mutation.
/// Multiple lanes compete for the current source, so source candidates recheck
/// its live cached threshold. Every target lane belongs to a distinct earlier
/// source and can use the precomputed mask directly. Scalar tails call the
/// matching scalar metric operation to preserve established rounding semantics.
///
/// `M` is concrete before type erasure and `N` is one through three. `input`
/// supplies `n × n` dots, `output` owns `n` sorted fixed-size lists, `norms`
/// holds metric scales when required, and `worst` mirrors every list's final
/// distance. The function evaluates exactly `n(n - 1) / 2` pairs. SIMD computes
/// up to `F::LANES` distances together; accepted candidates still insert in scan
/// order to keep deterministic ties.
#[inline(never)]
fn process_pairs<F, M, const N: usize>(
    arch: F::Arch,
    input: MatrixView<'_, f32>,
    output: &mut [[LeafNeighbor; N]],
    norms: &[f32],
    worst: &mut [f32],
) where
    F: SIMDVector<Scalar = f32, ConstLanes = Const<16>> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    M: KernelMetric,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let point_count = input.nrows();
    assert_eq!(
        input.ncols(),
        point_count,
        "validated leaf dot matrix must be square"
    );
    assert_eq!(
        output.len(),
        point_count,
        "validated leaf output must have one list per point"
    );
    assert_eq!(
        worst.len(),
        point_count,
        "validated leaf thresholds must have one value per point"
    );
    let dots = input.as_slice();
    let uses_norms = M::LEAF_SCALE.is_some();
    if uses_norms {
        assert_eq!(
            norms.len(),
            point_count,
            "validated leaf norms must have one value per point"
        );
    }
    let worst_ptr = worst.as_mut_ptr();

    // `source` starts at one because source zero has no strict-lower targets;
    // later sources still offer their pair back to source zero.
    for source in 1..point_count {
        let source_start = source * point_count;
        // `uses_norms` comes from a metric associated constant. Specialization
        // removes both branch and scale memory traffic for scale-free metrics.
        let source_scale = if uses_norms {
            F::splat(arch, norms[source])
        } else {
            F::default(arch)
        };
        // SAFETY: `source < point_count == worst.len()` by the assertions above.
        let mut source_worst = unsafe { *worst_ptr.add(source) };
        let mut target = 0;

        while target + F::LANES <= source {
            // SAFETY: the full chunk is contained in this source's strict-lower prefix.
            let pair_dots = unsafe { F::load_simd(arch, dots.as_ptr().add(source_start + target)) };
            let target_scales = if uses_norms {
                // SAFETY: the full target chunk lies below `source < point_count`, and
                // the assertion above established `norms.len() == point_count`.
                unsafe { F::load_simd(arch, norms.as_ptr().add(target)) }
            } else {
                F::default(arch)
            };
            let distances = M::leaf_distance(arch, pair_dots, source_scale, target_scales);
            // Every pair may improve the current source and its earlier target.
            // Derive both masks before either endpoint mutates its threshold.
            let source_eligible = distances.lt_simd(F::splat(arch, source_worst));
            // SAFETY: the full target chunk lies below `source < point_count`, and
            // the assertion above established `worst.len() == point_count`.
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
            let (source_scale, target_scale) = if uses_norms {
                // SAFETY: `target < source < point_count == norms.len()`.
                (norms[source], unsafe { *norms.get_unchecked(target) })
            } else {
                (0.0, 0.0)
            };
            let distance = M::leaf_distance_scalar(dot, source_scale, target_scale);
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

    debug_assert_eq!(output.len(), point_count);
}

/// Insert into a fixed-width neighbor list and return its new worst distance.
///
/// Width is a compile-time constant from one through three. Strict `<`
/// comparisons preserve scan order for ties; callers already rejected NaN via
/// the eligibility comparison. Explicit shifts save about 0.5% estimated cycles
/// versus the generic bubble loop in the local Callgrind `k=3` fixture.
///
/// `neighbors` is the sorted list for one source. `target` and `distance` are a
/// candidate already known to beat its final slot. The return value is the new
/// final-slot distance. Unsupported instantiations return an underfill sentinel;
/// `process_neighbor_width` never constructs them.
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
mod tests {
    use super::*;

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
    fn output_length_clamps_to_non_self_neighbors_and_rejects_large_k() {
        assert_eq!(leaf_output_len(0, 3).unwrap(), 0);
        assert_eq!(leaf_output_len(1, 3).unwrap(), 0);
        assert_eq!(leaf_output_len(4, 3).unwrap(), 12);
        assert_eq!(
            leaf_output_len(4, 4),
            Err(LeafKernelError::InvalidNeighborCount {
                points: 4,
                neighbors: 4,
                maximum: MAX_LEAF_NEIGHBORS,
            })
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            leaf_output_len(u32::MAX as usize + 1, 1),
            Err(LeafKernelError::TooManyPoints(u32::MAX as usize + 1))
        );
    }

    #[test]
    fn matrix_area_overflow_is_rejected_before_kernel_access() {
        assert_eq!(
            checked_area("leaf dot-product matrix", usize::MAX, 2),
            Err(LeafKernelError::ShapeOverflow {
                buffer: "leaf dot-product matrix",
                rows: usize::MAX,
                cols: 2,
            })
        );
    }

    #[test]
    fn prepared_kernel_accepts_different_neighbor_counts() {
        let points = 7;
        let dots = test_dots(Metric::L2, points);
        let input = test_input(&dots, points);
        let kernel = LeafKernel::new(Metric::L2);
        let mut workspace = LeafKernelWorkspace::new();

        for neighbor_count in [1, 3, 2] {
            let mut output = vec![LeafNeighbor::default(); points * neighbor_count];
            kernel
                .nearest_neighbors(
                    input,
                    MutMatrixView::try_from(output.as_mut_slice(), points, neighbor_count).unwrap(),
                    &mut workspace,
                )
                .unwrap();
            assert!(output.iter().all(|neighbor| neighbor.target != u32::MAX));
        }
    }

    #[test]
    fn workspace_can_shrink_and_grow_between_calls() {
        let kernel = LeafKernel::new(Metric::L2);
        let mut workspace = LeafKernelWorkspace::new();
        for points in [17, 7, 17] {
            let dots = test_dots(Metric::L2, points);
            let mut output = vec![LeafNeighbor::default(); points * 2];
            kernel
                .nearest_neighbors(
                    test_input(&dots, points),
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
        LeafKernel, LeafKernelError, LeafKernelWorkspace, LeafNeighbor, MAX_LEAF_NEIGHBORS,
        leaf_neighbor_count, leaf_output_len,
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
    const DOT_SCALE: f32 = 1.0 / 32.0;
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
                    pair * DOT_SCALE
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

    fn run_kernel(
        dots: &[f32],
        points: usize,
        requested_k: usize,
        metric: Metric,
    ) -> (usize, Vec<LeafNeighbor>) {
        let leaf_k = leaf_neighbor_count(points, requested_k).unwrap();
        let mut output = vec![LeafNeighbor::default(); points * leaf_k];
        LeafKernel::new(metric)
            .nearest_neighbors(
                test_input(dots, points),
                MutMatrixView::try_from(output.as_mut_slice(), points, leaf_k).unwrap(),
                &mut LeafKernelWorkspace::new(),
            )
            .unwrap();
        (leaf_k, output)
    }

    #[test]
    fn prepared_dispatch_matches_reference_across_simd_width_boundaries() {
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
                    let actual = run_kernel(&dots, points, requested_k, metric).1;
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
            run_kernel(&dots, 4, 2, Metric::L2).1,
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
            let positions: Vec<_> = run_kernel(&dots, 3, 1, metric)
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

        let output = run_kernel(&dots, 3, 2, Metric::Cosine).1;
        assert_eq!(output[0], LeafNeighbor::new(1, 1.0));
        assert_eq!(output[1], LeafNeighbor::new(2, 1.0));
    }

    #[test]
    fn clamps_negative_distances_and_preserves_cosine_extremes() {
        #[rustfmt::skip]
        let out_of_range = [1.0, 0.0, 2.0, 1.0];
        assert_eq!(
            run_kernel(&out_of_range, 2, 1, Metric::L2).1[0].distance,
            0.0
        );
        assert_eq!(
            run_kernel(&out_of_range, 2, 1, Metric::CosineNormalized).1[0].distance,
            0.0
        );
        assert_eq!(
            run_kernel(&out_of_range, 2, 1, Metric::Cosine).1[0].distance,
            0.0
        );

        #[rustfmt::skip]
        let opposite = [1.0, 0.0, -2.0, 1.0];
        assert_eq!(
            run_kernel(&opposite, 2, 1, Metric::Cosine).1[0].distance,
            3.0
        );

        let subnormal = [f32::MIN_POSITIVE / 2.0, 0.0, 1.0, 1.0];
        assert_eq!(
            run_kernel(&subnormal, 2, 1, Metric::Cosine).1[0].distance,
            1.0
        );

        let minimum_normal = [f32::MIN_POSITIVE, 0.0, f32::MIN_POSITIVE.sqrt(), 1.0];
        assert_eq!(
            run_kernel(&minimum_normal, 2, 1, Metric::Cosine).1[0].distance,
            0.0
        );
    }

    #[test]
    fn finite_max_distance_fills_the_final_fixed_slot() {
        let points = 4;
        let mut dots = vec![0.0; points * points];
        dots[3 * points] = -f32::MAX;

        let (leaf_k, output) = run_kernel(&dots, points, MAX_LEAF_NEIGHBORS, Metric::InnerProduct);
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
            let output = run_kernel(&dots, 3, 1, metric).1;
            assert_eq!(output[0].target, 2, "metric {metric:?}");
            assert_eq!(output[1].target, 2, "metric {metric:?}");
        }
    }

    #[test]
    fn rejects_sources_with_too_few_rankable_neighbors() {
        let dots = [1.0, 0.0, f32::NAN, 1.0];
        let mut output = [LeafNeighbor::default(); 2];
        let error = LeafKernel::new(Metric::L2)
            .nearest_neighbors(
                test_input(&dots, 2),
                MutMatrixView::try_from(&mut output[..], 2, 1).unwrap(),
                &mut LeafKernelWorkspace::new(),
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
        let (leaf_k, output) = run_kernel(&dots, 3, MAX_LEAF_NEIGHBORS, Metric::L2);

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
            assert_eq!(run_kernel(dots, points, requested_k, metric).0, 0);
        }
    }

    #[test]
    fn rejects_non_square_input_and_invalid_output_dimensions() {
        let dots = [0.0; 6];
        let non_square = MatrixView::try_from(&dots[..], 2, 3).unwrap();
        let mut output = [LeafNeighbor::default(); 2];
        let kernel = LeafKernel::new(Metric::L2);
        assert_eq!(
            kernel.nearest_neighbors(
                non_square,
                MutMatrixView::try_from(&mut output[..], 2, 1).unwrap(),
                &mut LeafKernelWorkspace::new(),
            ),
            Err(LeafKernelError::NonSquareDots { rows: 2, cols: 3 })
        );

        let square = [0.0; 9];
        let mut wrong_rows = [LeafNeighbor::default(); 2];
        assert_eq!(
            kernel.nearest_neighbors(
                test_input(&square, 3),
                MutMatrixView::try_from(&mut wrong_rows[..], 2, 1).unwrap(),
                &mut LeafKernelWorkspace::new(),
            ),
            Err(LeafKernelError::InvalidOutputRows {
                expected: 3,
                actual: 2,
                columns: 1,
            })
        );

        let mut too_many = [LeafNeighbor::default(); 9];
        assert_eq!(
            kernel.nearest_neighbors(
                test_input(&square, 3),
                MutMatrixView::try_from(&mut too_many[..], 3, 3).unwrap(),
                &mut LeafKernelWorkspace::new(),
            ),
            Err(LeafKernelError::InvalidNeighborCount {
                points: 3,
                neighbors: 3,
                maximum: 2,
            })
        );

        let square = [0.0; 25];
        let mut too_wide = [LeafNeighbor::default(); 20];
        assert_eq!(
            kernel.nearest_neighbors(
                test_input(&square, 5),
                MutMatrixView::try_from(&mut too_wide[..], 5, 4).unwrap(),
                &mut LeafKernelWorkspace::new(),
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

            let output = run_kernel(&dots, points, 1, Metric::Cosine).1;
            for (source, neighbor) in output.iter().enumerate().skip(1) {
                assert_eq!(
                    *neighbor,
                    LeafNeighbor::new(0, 1.0),
                    "n={points}, source={source}"
                );
            }
        }
    }

    #[test]
    fn output_length_rejects_unrepresentable_point_count() {
        assert_eq!(
            leaf_output_len(usize::MAX, 1),
            Err(LeafKernelError::TooManyPoints(usize::MAX))
        );
    }
}
