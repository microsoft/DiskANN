/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Prepared nearest-neighbor kernels over a leaf's lower dot-product matrix.
//!
//! PiPNN partitioning produces small, overlapping groups of dataset points
//! called *leaves*. Points sharing a leaf are treated as likely neighbors. For
//! each leaf, the builder gathers its vectors into a matrix `A` (one vector per
//! row). `sgemm_aat_lower` computes the lower triangle of the Gram matrix
//! `A · Aᵀ`, so entry `(i, j)` is the dot product of leaf points `i` and `j`.
//! This module consumes that result and picks each point's `k` nearest non-self
//! points in the leaf.
//!
//! This module does not gather vectors, run GEMM, translate dataset IDs, merge
//! candidates from overlapping leaves, or prune final graph degree. Its output
//! uses leaf-local positions. The caller maps those positions back through the
//! leaf's dataset-ID array and normally offers each selected pair in both graph
//! directions before cross-leaf merge/pruning.
//!
//! Here *source* means the point whose `k`-neighbor output list is being built;
//! *target* means another point in the same leaf. Pair distance is symmetric, so
//! one strict-lower matrix entry is evaluated once and offered independently to
//! both endpoint source lists.
//!
//! `sgemm_aat_lower` writes pair `(source, target)` only when `target <= source`.
//! The kernel scans that strict lower triangle once and offers each distance to
//! both endpoint points. A [`LeafKernel`] is prepared once for the build metric
//! and runtime CPU; each output view supplies its leaf-specific neighbor count.
//! Repeated leaves call a direct `diskann-wide` function pointer without ISA or
//! metric dispatch in the loop.
//! NaN distances are not rankable, and equal distances retain pair scan order.
//!
//! ```text
//! metric + runtime architecture
//!              │
//!              v
//!    prepared Dispatched1 handle
//!              │ reused with input + output.ncols()
//!              v
//! shape validation -> scale scratch -> strict-lower scan -> sorted neighbor slots
//! ```
//!
//! Between source iterations, `workspace.worst[point]` mirrors that point's last
//! retained slot. While one source is scanned, its threshold lives in local
//! `source_worst`; thresholds for earlier target points are updated in the
//! workspace immediately. The SIMD loop snapshots both endpoint thresholds
//! before either list changes, then writes the current source threshold back
//! after its strict-lower prefix is complete.
//!
//! # Main structures
//!
//! - [`LeafKernel`] is the reusable public handle containing one prepared direct
//!   function pointer.
//! - [`LeafInput`] identifies the borrowed lower-triangular dot matrix.
//! - [`LeafKernelWorkspace`] owns norm and rejection-threshold scratch and is
//!   reused by one worker across leaves.
//! - [`LeafNeighbor`] is one output slot containing leaf-local target position
//!   plus distance.
//! - `process_neighbor_width` chooses fixed storage for widths one through three
//!   or dynamic storage for larger widths.
//! - `process_pairs` is the shared SIMD/scalar strict-lower traversal;
//!   `insert_fixed_neighbor` and `insert_dynamic_neighbor` maintain stable sorted
//!   output for both endpoints.
//!
//! # Inputs and output
//!
//! For `n` leaf points, [`LeafInput::dots`] is an `n × n` row-major matrix from
//! `sgemm_aat_lower`. Diagonal entries provide norms when the metric needs them;
//! only strict-lower entries `(source, target)` with `target < source` provide
//! pair dots. Output is an `n × k` [`LeafNeighbor`] matrix. Every output row is
//! sorted by ascending distance and stores leaf-local target positions, not
//! dataset IDs.
//!
//! Distances are reconstructed from one pair dot and, when required, diagonal
//! entries of the Gram matrix. Smaller is better:
//!
//! | Prepared metric | Leaf distance |
//! | --- | --- |
//! | squared L2 | `max(0, ‖source‖² + ‖target‖² - 2(source·target))` |
//! | cosine | `max(0, 1 - (source·target)/(‖source‖‖target‖))` |
//! | normalized cosine | `max(0, 1 - source·target)` |
//! | inner product | `-(source·target)` |
//!
//! `CosineNormalized` assumes leaf vectors were normalized before GEMM. For
//! unnormalized cosine, a norm below `sqrt(f32::MIN_POSITIVE)` gives zero
//! similarity. NaN scores never enter output because selection uses strict
//! ordered comparisons.
//!
//! # Core flow
//!
//! 1. Validate matrix areas, backing lengths, point IDs, and output width.
//! 2. Build metric scales from diagonal dots and reset per-source thresholds.
//! 3. Scan every strict-lower pair once in SIMD groups plus scalar tails.
//! 4. Offer that distance to both pair endpoints using stable top-k insertion.
//! 5. Reject any source whose final slot remains unfilled.
//!
//! # Performance
//!
//! With `k > 0`, the kernel evaluates exactly `n(n - 1) / 2` pair distances;
//! `k = 0` returns before traversal. Widths `k = 1, 2, 3` use fixed arrays and
//! straight-line insertion, giving `O(n²)` work.
//! Larger widths use `O(k)` insertion, giving `O(n²k)` worst-case work. Scratch
//! is `O(n)` (`worst`, plus norms only when required); output is `O(nk)`. No
//! allocation occurs after a worker workspace has sufficient capacity. Runtime
//! architecture and metric selection happen once in [`LeafKernel::new`].
//!
//! # Example
//!
//! ```
//! use diskann::graph::pipnn::leaf_kernel::{
//!     leaf_output_len, LeafInput, LeafKernel, LeafKernelWorkspace, LeafNeighbor,
//! };
//! use diskann_utils::views::{MatrixView, MutMatrixView};
//! use diskann_vector::distance::Metric;
//!
//! // Only the diagonal and strict lower triangle are consumed.
//! let dots = [
//!     1.0, f32::NAN, f32::NAN,
//!     0.9, 1.0,      f32::NAN,
//!     0.1, 0.2,      1.0,
//! ];
//! let input = LeafInput {
//!     dots: MatrixView::try_from(&dots[..], 3, 3).unwrap(),
//! };
//! let mut neighbors = vec![LeafNeighbor::default(); leaf_output_len(3, 1).unwrap()];
//! let output = MutMatrixView::try_from(&mut neighbors[..], 3, 1).unwrap();
//! let mut workspace = LeafKernelWorkspace::new();
//!
//! LeafKernel::new(Metric::CosineNormalized)
//!     .nearest_neighbors(input, output, &mut workspace)
//!     .unwrap();
//!
//! assert_eq!(neighbors.iter().map(|neighbor| neighbor.target).collect::<Vec<_>>(), [1, 0, 1]);
//! ```

use std::marker::PhantomData;

use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_vector::distance::Metric;
use diskann_wide::{
    Architecture, SIMDFloat, SIMDMask, SIMDSelect, SIMDVector,
    arch::{self, Dispatched1, FTarget1},
    lifetime::AddLifetime,
};

use super::kernel_metric::{KernelMetric, MetricVisitor, visit_metric};

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

/// Square lower-triangular dot-product matrix for one leaf.
#[derive(Clone, Copy, Debug)]
pub struct LeafInput<'a> {
    /// Point-by-point matrix. Only entries with `target <= source` are read.
    pub dots: MatrixView<'a, f32>,
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
    /// A view's backing slice does not match its declared shape.
    #[error("invalid {buffer} length: expected {expected}, got {actual}")]
    InvalidBufferLength {
        /// Name of the invalid buffer.
        buffer: &'static str,
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
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
    /// A source requests more non-self neighbors than the leaf contains.
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
/// The returned width is `min(requested_k, points - 1)`, allowing empty,
/// singleton, and small leaves without a second effective-k state.
///
/// # Errors
///
/// Returns [`LeafKernelError::TooManyPoints`] when leaf-local positions cannot
/// fit in `u32`.
///
/// # Performance
///
/// Constant-time and allocation-free.
pub fn leaf_neighbor_count(points: usize, requested_k: usize) -> Result<usize, LeafKernelError> {
    if points > u32::MAX as usize {
        return Err(LeafKernelError::TooManyPoints(points));
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
    input: LeafInput<'a>,
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
    /// neighbor count for this leaf and must not exceed `point_count - 1`.
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
        input: LeafInput<'_>,
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
        );
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
/// Matrix views are rechecked with `checked_mul` because the hot loop performs
/// unchecked contiguous loads. Output columns are the leaf-specific neighbor
/// width and cannot exceed the number of non-self points.
///
/// `input` and `output` are borrowed only for inspection. Success returns no
/// value; it establishes square dots, exact backing lengths, representable local
/// IDs, and valid output width. Failure returns [`LeafKernelError`] before any
/// output or workspace mutation. Runtime is constant apart from view metadata
/// checks; matrix contents are not scanned.
fn validate(
    input: LeafInput<'_>,
    output: &MutMatrixView<'_, LeafNeighbor>,
) -> Result<(), LeafKernelError> {
    let point_count = input.dots.nrows();
    let dot_columns = input.dots.ncols();
    if point_count > u32::MAX as usize {
        return Err(LeafKernelError::TooManyPoints(point_count));
    }
    if point_count != dot_columns {
        return Err(LeafKernelError::NonSquareDots {
            rows: point_count,
            cols: dot_columns,
        });
    }
    let dots_len = checked_area("leaf dot-product matrix", point_count, dot_columns)?;
    check_length(
        "leaf dot-product matrix",
        input.dots.as_slice().len(),
        dots_len,
    )?;
    let output_len = checked_area("output", output.nrows(), output.ncols())?;
    check_length("output", output.as_slice().len(), output_len)?;

    if output.nrows() != point_count {
        return Err(LeafKernelError::InvalidOutputRows {
            expected: point_count,
            actual: output.nrows(),
            columns: output.ncols(),
        });
    }
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
    input: LeafInput<'_>,
    workspace: &mut LeafKernelWorkspace,
) -> Result<(), LeafKernelError> {
    let points = input.dots.nrows();
    if M::LEAF_SCALE.is_some() {
        resize("norms", &mut workspace.norms, points, 0.0)?;
        for (source, norm) in workspace.norms.iter_mut().enumerate() {
            *norm = M::LEAF_SCALE.transform(input.dots[(source, source)]);
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

fn check_length(
    buffer: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), LeafKernelError> {
    if actual == expected {
        Ok(())
    } else {
        Err(LeafKernelError::InvalidBufferLength {
            buffer,
            expected,
            actual,
        })
    }
}

/// Convert neighbor count into fixed source storage or the dynamic fallback.
///
/// This branch runs once per leaf. Fixed conversion uses `as_chunks_mut` once,
/// avoiding per-candidate slice-to-array checks while retaining safe insertion.
///
/// `output` contains `point_count * neighbor_count` initialized slots; `norms`
/// and `worst` satisfy the invariants established by `prepare_workspace`. The
/// function writes output and thresholds in place and returns no value. Widths
/// one through three take the fixed path; all others pay one division per source
/// insertion to locate its dynamic slice.
fn process_neighbor_width<F, M>(
    arch: F::Arch,
    input: LeafInput<'_>,
    neighbor_count: usize,
    output: &mut [LeafNeighbor],
    norms: &[f32],
    worst: &mut [f32],
) where
    F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    M: KernelMetric,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    match neighbor_count {
        1 => process_fixed_width::<F, M, 1>(arch, input, output, norms, worst),
        2 => process_fixed_width::<F, M, 2>(arch, input, output, norms, worst),
        3 => process_fixed_width::<F, M, 3>(arch, input, output, norms, worst),
        dynamic_count => process_pairs::<F, M, _>(
            arch,
            input,
            DynamicNeighborStorage {
                values: output,
                neighbor_count: dynamic_count,
            },
            norms,
            worst,
        ),
    }
}

/// Reinterpret validated output as one fixed array per source, then run shared
/// pair traversal.
///
/// `N` is one, two, or three. `as_chunks_mut` performs one safe shape split per
/// leaf, keeping array conversion out of candidate insertion.
fn process_fixed_width<F, M, const N: usize>(
    arch: F::Arch,
    input: LeafInput<'_>,
    output: &mut [LeafNeighbor],
    norms: &[f32],
    worst: &mut [f32],
) where
    F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    M: KernelMetric,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let (neighbor_lists, remainder) = output.as_chunks_mut::<N>();
    debug_assert!(remainder.is_empty());
    process_pairs::<F, M, _>(
        arch,
        input,
        FixedNeighborStorage(neighbor_lists),
        norms,
        worst,
    );
}

/// Mutable neighbor-list adapter used by the shared pair traversal.
///
/// Implementations own the exclusive output borrow for the whole scan. Each
/// insertion borrows one source list briefly, so updates to the current source
/// and earlier targets cannot alias simultaneously.
trait NeighborStorage {
    /// Number of source neighbor lists owned by this adapter.
    fn source_count(&self) -> usize;

    /// Insert one source-target candidate and return that source's new threshold.
    fn insert(&mut self, source: usize, target: u32, distance: f32) -> f32;
}

struct FixedNeighborStorage<'a, const N: usize>(&'a mut [[LeafNeighbor; N]]);

impl<const N: usize> NeighborStorage for FixedNeighborStorage<'_, N> {
    #[inline(always)]
    fn source_count(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    fn insert(&mut self, source: usize, target: u32, distance: f32) -> f32 {
        insert_fixed_neighbor(&mut self.0[source], target, distance)
    }
}

struct DynamicNeighborStorage<'a> {
    values: &'a mut [LeafNeighbor],
    neighbor_count: usize,
}

impl NeighborStorage for DynamicNeighborStorage<'_> {
    #[inline(always)]
    fn source_count(&self) -> usize {
        self.values.len() / self.neighbor_count
    }

    #[inline(always)]
    fn insert(&mut self, source: usize, target: u32, distance: f32) -> f32 {
        insert_dynamic_neighbor(
            &mut self.values[source * self.neighbor_count..(source + 1) * self.neighbor_count],
            target,
            distance,
        )
    }
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
/// `M` is concrete before type erasure. `R` presents fixed neighbor arrays for
/// common counts or safe dynamic slices for the uncommon fallback.
///
/// `input` supplies `n × n` dots, `output` owns `n` sorted lists, `norms` holds
/// metric scales when required, and `worst` mirrors every list's final distance.
/// The function mutates output and thresholds in place and returns no value.
/// It evaluates exactly `n(n - 1) / 2` pairs. SIMD computes up to `F::LANES`
/// distances together; accepted candidates still insert in scan order to keep
/// deterministic ties. Fixed widths cost constant work per accepted endpoint;
/// dynamic widths cost `O(k)` per insertion.
#[inline(never)]
fn process_pairs<F, M, R>(
    arch: F::Arch,
    input: LeafInput<'_>,
    mut output: R,
    norms: &[f32],
    worst: &mut [f32],
) where
    F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    M: KernelMetric,
    R: NeighborStorage,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let point_count = input.dots.nrows();
    let dots = input.dots.as_slice();
    let uses_norms = M::LEAF_SCALE.is_some();
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
        // SAFETY: `source < point_count == worst.len()` after validation.
        let mut source_worst = unsafe { *worst_ptr.add(source) };
        let mut target = 0;

        while target + F::LANES <= source {
            // SAFETY: the full chunk is contained in this source's strict-lower prefix.
            let pair_dots = unsafe { F::load_simd(arch, dots.as_ptr().add(source_start + target)) };
            let target_scales = if uses_norms {
                // SAFETY: the full target chunk lies below `source <= norms.len()`.
                unsafe { F::load_simd(arch, norms.as_ptr().add(target)) }
            } else {
                F::default(arch)
            };
            let distances = M::leaf_distance(arch, pair_dots, source_scale, target_scales);
            // Every pair may improve the current source and its earlier target.
            // Derive both masks before either endpoint mutates its threshold.
            let source_eligible = distances.lt_simd(F::splat(arch, source_worst));
            // SAFETY: the full target chunk lies below `source`, so it is inside `worst`.
            let target_worst = unsafe { F::load_simd(arch, worst_ptr.add(target)) };
            let target_eligible = distances.lt_simd(target_worst);
            let source_bits = u64::from(source_eligible.bitmask().to_underlying());
            let target_bits = u64::from(target_eligible.bitmask().to_underlying());

            if source_bits | target_bits != 0 {
                let values = distances.to_array();
                let values = values.as_ref();
                let mut source_bits = source_bits;
                while source_bits != 0 {
                    let lane = source_bits.trailing_zeros() as usize;
                    source_bits &= source_bits - 1;
                    let distance = values[lane];
                    if distance < source_worst {
                        source_worst = output.insert(source, (target + lane) as u32, distance);
                    }
                }

                let mut target_bits = target_bits;
                while target_bits != 0 {
                    let lane = target_bits.trailing_zeros() as usize;
                    target_bits &= target_bits - 1;
                    let target_source = target + lane;
                    let new_worst = output.insert(target_source, source as u32, values[lane]);
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
                source_worst = output.insert(source, target as u32, distance);
            }
            // SAFETY: `target < source < worst.len()`.
            let target_worst = unsafe { *worst_ptr.add(target) };
            if distance < target_worst {
                let new_worst = output.insert(target, source as u32, distance);
                // SAFETY: `target < source < worst.len()`.
                unsafe { *worst_ptr.add(target) = new_worst };
            }
            target += 1;
        }
        // SAFETY: `source < worst.len()`.
        unsafe { *worst_ptr.add(source) = source_worst };
    }

    debug_assert_eq!(output.source_count(), point_count);
}

/// Insert into a fixed-width neighbor list and return its new worst distance.
///
/// Production widths one through three use straight-line shifts. Strict `<`
/// comparisons preserve scan order for ties; callers already rejected NaN via
/// the eligibility comparison.
///
/// `neighbors` is the sorted list for one source. `target` and `distance` are a
/// candidate already known to beat its final slot. The return value is the new
/// final-slot distance. Insertion is allocation-free and constant-time because
/// `N <= 3`.
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
        _ => unreachable!("fixed leaf widths are one through three"),
    }
}

/// Insert into a run-time-width neighbor list using the same stable ordering contract.
///
/// The candidate replaces the last slot, then bubbles toward the front. This
/// path is used only for neighbor counts greater than three.
///
/// `neighbors` is one non-empty sorted source list. `target` and `distance` are
/// already known to beat its final slot. The return value is the new final-slot
/// distance. Work is `O(k)` worst-case and allocation-free.
#[inline(always)]
fn insert_dynamic_neighbor(neighbors: &mut [LeafNeighbor], target: u32, distance: f32) -> f32 {
    let last = neighbors.len() - 1;
    neighbors[last] = LeafNeighbor::new(target, distance);
    let mut index = last;
    while index > 0 && neighbors[index].distance < neighbors[index - 1].distance {
        neighbors.swap(index, index - 1);
        index -= 1;
    }
    neighbors[last].distance
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

    fn test_input(dots: &[f32], points: usize) -> LeafInput<'_> {
        LeafInput {
            dots: MatrixView::try_from(dots, points, points).unwrap(),
        }
    }

    fn insert_reference(
        output: &mut [LeafNeighbor],
        worst: &mut [f32],
        neighbor_count: usize,
        source: usize,
        target: u32,
        distance: f32,
    ) {
        if distance.partial_cmp(&worst[source]) != Some(std::cmp::Ordering::Less) {
            return;
        }
        worst[source] = insert_dynamic_neighbor(
            &mut output[source * neighbor_count..(source + 1) * neighbor_count],
            target,
            distance,
        );
    }

    #[test]
    fn scalar_insertion_orders_candidates_and_rejects_nan() {
        let mut output = [LeafNeighbor::default(); 4];
        let mut worst = [f32::INFINITY];

        for (target, distance) in [(0, 4.0), (1, 1.0), (2, 3.0), (3, 2.0), (4, 0.5)] {
            insert_reference(&mut output, &mut worst, 4, 0, target, distance);
        }
        insert_reference(&mut output, &mut worst, 4, 0, 5, f32::NAN);

        assert_eq!(
            output,
            [
                LeafNeighbor::new(4, 0.5),
                LeafNeighbor::new(1, 1.0),
                LeafNeighbor::new(3, 2.0),
                LeafNeighbor::new(2, 3.0),
            ]
        );
        assert_eq!(worst, [3.0]);
    }

    #[test]
    fn output_length_clamps_to_non_self_neighbors() {
        assert_eq!(leaf_output_len(0, 3).unwrap(), 0);
        assert_eq!(leaf_output_len(1, 3).unwrap(), 0);
        assert_eq!(leaf_output_len(4, 9).unwrap(), 12);
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
        LeafInput, LeafKernel, LeafKernelError, LeafKernelWorkspace, LeafNeighbor,
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

    fn test_input(dots: &[f32], points: usize) -> LeafInput<'_> {
        LeafInput {
            dots: MatrixView::try_from(dots, points, points).unwrap(),
        }
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
                for requested_k in [1, 2, 3, 4, 5] {
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
    fn finite_max_distance_fills_the_final_simd_slot() {
        let points = 9;
        let mut dots = vec![0.0; points * points];
        dots[8 * points] = -f32::MAX;

        let (leaf_k, output) = run_kernel(&dots, points, points - 1, Metric::InnerProduct);
        assert_eq!(leaf_k, 8);
        assert_eq!(
            output[8 * leaf_k + leaf_k - 1],
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
        let (leaf_k, output) = run_kernel(&dots, 3, 99, Metric::L2);

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
        let non_square = LeafInput {
            dots: MatrixView::try_from(&dots[..], 2, 3).unwrap(),
        };
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
