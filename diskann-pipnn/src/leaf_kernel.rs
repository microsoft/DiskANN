/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Prepared nearest-neighbor kernels over a leaf's lower dot-product matrix.
//!
//! `sgemm_aat_lower` writes pair `(row, column)` only when `column <= row`.
//! The kernel scans that strict lower triangle once and offers each distance to
//! both endpoint rows. A [`LeafKernel`] is prepared once for the build metric,
//! requested neighbor count, and runtime CPU; repeated leaves call a direct
//! `diskann-wide` function pointer without ISA or metric dispatch in the loop.
//! NaN distances are not rankable, and equal distances retain pair scan order.
//!
//! ```text
//! metric + requested k + runtime architecture
//!                 │
//!                 v
//!       prepared Dispatched1 handle
//!                 │ reused for every leaf
//!                 v
//! shape validation -> scale scratch -> strict-lower scan -> sorted row slots
//! ```
//!
//! `workspace.worst[row]` always mirrors the last (worst) retained slot for that
//! row. The SIMD loop may update both endpoints of a pair, so this mirror is the
//! threshold shared by row and column candidate masks.

use std::marker::PhantomData;

use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_vector::distance::Metric;
use diskann_wide::{
    arch::{self, Dispatched1, FTarget1},
    lifetime::AddLifetime,
    Architecture, SIMDFloat, SIMDMask, SIMDSelect, SIMDVector,
};

use crate::kernel_metric::{erase_metric, EraseMetric, KernelMetric};

/// One leaf-local neighbor and its metric distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LeafNeighbor {
    /// Position in the leaf, not a dataset ID.
    pub position: u32,
    /// Distance from the row point to `position`.
    pub distance: f32,
}

impl LeafNeighbor {
    /// Construct a leaf-local neighbor.
    pub const fn new(position: u32, distance: f32) -> Self {
        Self { position, distance }
    }
}

impl Default for LeafNeighbor {
    fn default() -> Self {
        Self::new(u32::MAX, f32::INFINITY)
    }
}

/// Square lower-triangular dot-product matrix for one leaf.
#[derive(Clone, Copy, Debug)]
pub struct LeafTopK<'a> {
    /// Point-by-point matrix. Only entries with `column <= row` are read.
    pub dots: MatrixView<'a, f32>,
}

/// Reusable temporary storage for leaf top-k selection.
#[derive(Debug, Default)]
pub struct LeafTopKWorkspace {
    norms: Vec<f32>,
    worst: Vec<f32>,
}

impl LeafTopKWorkspace {
    /// Construct an empty workspace.
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
    /// The output matrix does not match the requested neighbor shape.
    #[error(
        "invalid output shape: expected {expected_rows} x {expected_cols}, got {actual_rows} x {actual_cols}"
    )]
    InvalidOutputShape {
        /// Required row count.
        expected_rows: usize,
        /// Required column count.
        expected_cols: usize,
        /// Supplied row count.
        actual_rows: usize,
        /// Supplied column count.
        actual_cols: usize,
    },
    /// Temporary storage could not be reserved.
    #[error("failed to reserve {additional} values for {buffer}")]
    Allocation {
        /// Name of the temporary buffer.
        buffer: &'static str,
        /// Additional element capacity requested.
        additional: usize,
    },
    /// A row did not contain enough rankable pair distances to fill its output.
    #[error("row {row} has fewer than {neighbors} rankable leaf neighbors")]
    InsufficientRankableNeighbors {
        /// Zero-based row position in the leaf.
        row: usize,
        /// Required number of non-self neighbors.
        neighbors: usize,
    },
}

/// Return the required output length for [`LeafKernel::nearest_neighbors`].
pub fn leaf_output_len(points: usize, k: usize) -> Result<usize, LeafKernelError> {
    if points > u32::MAX as usize {
        return Err(LeafKernelError::TooManyPoints(points));
    }
    checked_area("output", points, k.min(points.saturating_sub(1)))
}

/// One invocation bundled for `Dispatched1`.
///
/// `AddLifetime` can attach one lifetime to this aggregate, allowing the direct
/// function-pointer interface to carry the input view, exclusive output view,
/// and exclusive scratch lease without storing any of them in `LeafKernel`.
#[derive(Debug)]
struct LeafCall<'a> {
    input: LeafTopK<'a>,
    output: MutMatrixView<'a, LeafNeighbor>,
    workspace: &'a mut LeafTopKWorkspace,
    requested_k: usize,
}

#[derive(Debug)]
struct LeafCallArg;

impl AddLifetime for LeafCallArg {
    type Of<'a> = LeafCall<'a>;
}

type LeafFn = Dispatched1<Result<usize, LeafKernelError>, LeafCallArg>;

/// A leaf kernel prepared for one metric, neighbor count, and the current CPU.
///
/// Construct this once with [`LeafKernel::new`] and share it across leaf workers.
/// The handle stores only a direct function pointer and the requested `k`.
#[derive(Clone, Copy, Debug)]
pub struct LeafKernel {
    run: LeafFn,
    requested_k: usize,
}

impl LeafKernel {
    /// Prepare a leaf kernel for `metric`, `k`, and the current CPU.
    pub fn new(metric: Metric, k: usize) -> Self {
        diskann_wide::arch::dispatch1_no_features(PrepareLeaf { requested_k: k }, metric)
    }

    /// Select the nearest non-self leaf positions for every row.
    ///
    /// `output` must have `input.dots.nrows()` rows and
    /// `min(k, rows - 1)` columns. The returned value is that effective column
    /// count. Equal distances retain pair scan order.
    pub fn nearest_neighbors(
        &self,
        input: LeafTopK<'_>,
        output: MutMatrixView<'_, LeafNeighbor>,
        workspace: &mut LeafTopKWorkspace,
    ) -> Result<usize, LeafKernelError> {
        self.run.call(LeafCall {
            input,
            output,
            workspace,
            requested_k: self.requested_k,
        })
    }
}

/// Requested-width dispatch selected once while preparing the kernel.
///
/// Widths one through three receive fixed array rows. Larger widths retain one
/// dynamic implementation instead of multiplying code size by every possible k.
#[derive(Clone, Copy, Debug)]
enum KValue {
    One,
    Two,
    Three,
    Large,
}

impl KValue {
    const fn from_requested(k: usize) -> Self {
        match k {
            1 => Self::One,
            2 => Self::Two,
            3 => Self::Three,
            _ => Self::Large,
        }
    }
}

/// First dispatch stage: choose the runtime architecture once.
///
/// The factory itself uses `dispatch1_no_features`; only the returned leaf entry
/// needs target features, so architecture-specific code remains behind the final
/// direct function pointer.
struct PrepareLeaf {
    requested_k: usize,
}

impl<A> arch::Target1<A, LeafKernel, Metric> for PrepareLeaf
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    fn run(self, arch: A, metric: Metric) -> LeafKernel {
        erase_metric(
            metric,
            BuildLeaf {
                arch,
                requested_k: self.requested_k,
            },
        )
    }
}

/// BYO-type-erasure visitor holding a concrete architecture.
///
/// `erase<M>` receives a concrete metric marker, then combines `A`, `M`, and
/// the requested width before erasing the result into exactly one `Dispatched1`.
struct BuildLeaf<A> {
    arch: A,
    requested_k: usize,
}

impl<A> BuildLeaf<A>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    fn build<M: KernelMetric, S: SlotSelection>(self) -> LeafKernel {
        LeafKernel {
            run: self
                .arch
                .dispatch1::<LeafEntry<M, S>, Result<usize, LeafKernelError>, LeafCallArg>(),
            requested_k: self.requested_k,
        }
    }
}

impl<A> EraseMetric for BuildLeaf<A>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    type Output = LeafKernel;

    fn erase<M: KernelMetric>(self) -> Self::Output {
        match KValue::from_requested(self.requested_k) {
            KValue::One => self.build::<M, FixedSelection<1>>(),
            KValue::Two => self.build::<M, FixedSelection<2>>(),
            KValue::Three => self.build::<M, FixedSelection<3>>(),
            KValue::Large => self.build::<M, DynamicSelection>(),
        }
    }
}

/// Architecture/metric/width-specialized function-pointer destination.
///
/// This type is zero-sized. All per-leaf state arrives through `LeafCall`; the
/// entry validates and initializes that state before reaching pointer-based SIMD.
struct LeafEntry<M, S>(PhantomData<(M, S)>);

impl<A, M, S> FTarget1<A, Result<usize, LeafKernelError>, LeafCall<'_>> for LeafEntry<M, S>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    M: KernelMetric,
    S: SlotSelection,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    fn run(arch: A, mut call: LeafCall<'_>) -> Result<usize, LeafKernelError> {
        // Validation establishes every shape and active-prefix invariant used by
        // unchecked loads below. No output or scratch mutation occurs on error.
        let actual_k = validate(call.input, call.requested_k, &call.output)?;
        if actual_k == 0 {
            return Ok(0);
        }

        // Norm and threshold scratch are reset for this leaf, while Vec capacity
        // remains reusable by the worker that owns the workspace.
        prepare_workspace::<M>(call.input, call.workspace)?;
        call.output.as_mut_slice().fill(LeafNeighbor::default());
        call.workspace.worst.fill(f32::INFINITY);

        S::process::<A::f32x16, M>(
            arch,
            call.input,
            actual_k,
            call.output.as_mut_slice(),
            &call.workspace.norms,
            &mut call.workspace.worst,
        );
        if let Some(row) = call
            .output
            .as_slice()
            .chunks_exact(actual_k)
            .position(|neighbors| neighbors[actual_k - 1].position == u32::MAX)
        {
            return Err(LeafKernelError::InsufficientRankableNeighbors {
                row,
                neighbors: actual_k,
            });
        }
        Ok(actual_k)
    }
}

/// Validate the complete safety contract before dispatched SIMD executes.
///
/// Matrix views are rechecked with `checked_mul` because the hot loop performs
/// unchecked contiguous loads. Output columns must equal the clamped effective
/// k so fixed-row conversion cannot expose a partial row.
fn validate(
    input: LeafTopK<'_>,
    k: usize,
    output: &MutMatrixView<'_, LeafNeighbor>,
) -> Result<usize, LeafKernelError> {
    let rows = input.dots.nrows();
    let columns = input.dots.ncols();
    if rows != columns {
        return Err(LeafKernelError::NonSquareDots {
            rows,
            cols: columns,
        });
    }
    let output_len = leaf_output_len(rows, k)?;
    let dots_len = checked_area("leaf dot-product matrix", rows, columns)?;
    check_length(
        "leaf dot-product matrix",
        input.dots.as_slice().len(),
        dots_len,
    )?;

    let actual_k = k.min(rows.saturating_sub(1));
    if output.nrows() != rows || output.ncols() != actual_k {
        return Err(LeafKernelError::InvalidOutputShape {
            expected_rows: rows,
            expected_cols: actual_k,
            actual_rows: output.nrows(),
            actual_cols: output.ncols(),
        });
    }
    check_length("output", output.as_slice().len(), output_len)?;
    Ok(actual_k)
}

/// Prepare metric-specific scale and threshold scratch.
///
/// L2 stores diagonal squared norms; cosine converts diagonals to norms using
/// DiskANN's zero threshold. Normalized cosine and inner product skip the norm
/// allocation entirely. `worst` is reset separately after allocation succeeds.
fn prepare_workspace<M: KernelMetric>(
    input: LeafTopK<'_>,
    workspace: &mut LeafTopKWorkspace,
) -> Result<(), LeafKernelError> {
    let points = input.dots.nrows();
    if M::LEAF_SCALE.is_some() {
        resize("norms", &mut workspace.norms, points, 0.0)?;
        for (row, norm) in workspace.norms.iter_mut().enumerate() {
            *norm = M::LEAF_SCALE.transform(input.dots[(row, row)]);
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

/// Prepared requested-width policy.
///
/// The actual width can be smaller for singleton/tiny leaves, so each policy
/// performs one pre-loop clamp dispatch while keeping width selection out of the
/// pair scan.
trait SlotSelection: Send + Sync + 'static {
    fn process<F, M>(
        arch: F::Arch,
        input: LeafTopK<'_>,
        actual_k: usize,
        output: &mut [LeafNeighbor],
        norms: &[f32],
        worst: &mut [f32],
    ) where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
        M: KernelMetric,
        u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>;
}

struct FixedSelection<const N: usize>;
struct DynamicSelection;

impl<const N: usize> SlotSelection for FixedSelection<N> {
    fn process<F, M>(
        arch: F::Arch,
        input: LeafTopK<'_>,
        actual_k: usize,
        output: &mut [LeafNeighbor],
        norms: &[f32],
        worst: &mut [f32],
    ) where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
        M: KernelMetric,
        u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    {
        debug_assert!(actual_k <= N);
        process_selected::<F, M>(arch, input, actual_k, output, norms, worst);
    }
}

impl SlotSelection for DynamicSelection {
    fn process<F, M>(
        arch: F::Arch,
        input: LeafTopK<'_>,
        actual_k: usize,
        output: &mut [LeafNeighbor],
        norms: &[f32],
        worst: &mut [f32],
    ) where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
        M: KernelMetric,
        u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    {
        process_selected::<F, M>(arch, input, actual_k, output, norms, worst);
    }
}

/// Convert effective k into one fixed row representation or the dynamic fallback.
///
/// This branch runs once per leaf. Fixed conversion uses `as_chunks_mut` once,
/// avoiding per-candidate slice-to-array checks while retaining safe insertion.
fn process_selected<F, M>(
    arch: F::Arch,
    input: LeafTopK<'_>,
    actual_k: usize,
    output: &mut [LeafNeighbor],
    norms: &[f32],
    worst: &mut [f32],
) where
    F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    M: KernelMetric,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    match actual_k {
        1 => process_fixed::<F, M, 1>(arch, input, output, norms, worst),
        2 => process_fixed::<F, M, 2>(arch, input, output, norms, worst),
        3 => process_fixed::<F, M, 3>(arch, input, output, norms, worst),
        width => process_pairs::<F, M, _>(
            arch,
            input,
            DynamicRows {
                values: output,
                width,
            },
            norms,
            worst,
        ),
    }
}

fn process_fixed<F, M, const N: usize>(
    arch: F::Arch,
    input: LeafTopK<'_>,
    output: &mut [LeafNeighbor],
    norms: &[f32],
    worst: &mut [f32],
) where
    F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    M: KernelMetric,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let (rows, remainder) = output.as_chunks_mut::<N>();
    debug_assert!(remainder.is_empty());
    process_pairs::<F, M, _>(arch, input, FixedRows(rows), norms, worst);
}

/// Mutable row adapter used by the shared pair traversal.
///
/// Implementations own the exclusive output borrow for the whole scan. Each
/// insertion borrows one row briefly, so updates to the current row and earlier
/// endpoint rows cannot alias simultaneously.
trait NeighborRows {
    fn len(&self) -> usize;
    fn insert(&mut self, row: usize, position: u32, distance: f32) -> f32;
}

struct FixedRows<'a, const N: usize>(&'a mut [[LeafNeighbor; N]]);

impl<const N: usize> NeighborRows for FixedRows<'_, N> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    fn insert(&mut self, row: usize, position: u32, distance: f32) -> f32 {
        insert_fixed(&mut self.0[row], position, distance)
    }
}

struct DynamicRows<'a> {
    values: &'a mut [LeafNeighbor],
    width: usize,
}

impl NeighborRows for DynamicRows<'_> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.values.len() / self.width
    }

    #[inline(always)]
    fn insert(&mut self, row: usize, position: u32, distance: f32) -> f32 {
        insert_dynamic(
            &mut self.values[row * self.width..(row + 1) * self.width],
            position,
            distance,
        )
    }
}

/// Scan the strict lower triangle once and update both endpoint rows.
///
/// Invariants on entry:
///
/// - `dots` is a validated square row-major matrix;
/// - `output` has one sorted ascending-distance row per point;
/// - `worst[row]` equals that row's last slot;
/// - `norms` has one value per point exactly when `M` requires scales.
///
/// Each SIMD chunk computes both endpoint eligibility masks before mutation.
/// Multiple lanes compete for the current row, so row candidates recheck its
/// live cached threshold. Every column lane targets a distinct earlier row and
/// can use the precomputed mask directly. Scalar tails call the matching scalar
/// metric operation to preserve established rounding semantics.
///
/// `M` is concrete before type erasure. `R` presents fixed array rows for common
/// widths or safe dynamic slices for the uncommon fallback.
#[inline(never)]
fn process_pairs<F, M, R>(
    arch: F::Arch,
    input: LeafTopK<'_>,
    mut output: R,
    norms: &[f32],
    worst: &mut [f32],
) where
    F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    M: KernelMetric,
    R: NeighborRows,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let points = input.dots.nrows();
    let dots = input.dots.as_slice();
    let uses_norms = M::LEAF_SCALE.is_some();
    let worst_ptr = worst.as_mut_ptr();

    for row in 1..points {
        let row_start = row * points;
        let row_norm = if uses_norms {
            F::splat(arch, norms[row])
        } else {
            F::default(arch)
        };
        // SAFETY: `row < points == worst.len()` after validation.
        let mut row_worst = unsafe { *worst_ptr.add(row) };
        let mut column = 0;

        while column + F::LANES <= row {
            // SAFETY: the full chunk is contained in the strict lower row prefix.
            let pair_dots = unsafe { F::load_simd(arch, dots.as_ptr().add(row_start + column)) };
            let column_norms = if uses_norms {
                // SAFETY: the full chunk lies below `row <= norms.len()`.
                unsafe { F::load_simd(arch, norms.as_ptr().add(column)) }
            } else {
                F::default(arch)
            };
            let distances = M::leaf_distance(arch, pair_dots, row_norm, column_norms);
            // Every pair may improve the current row and its earlier endpoint.
            // Derive both masks from the same distance vector before either side
            // mutates its threshold.
            let row_eligible = distances.lt_simd(F::splat(arch, row_worst));
            // SAFETY: the full chunk lies below `row`, so it is inside `worst`.
            let column_worst = unsafe { F::load_simd(arch, worst_ptr.add(column)) };
            let column_eligible = distances.lt_simd(column_worst);
            let row_bits = u64::from(row_eligible.bitmask().to_underlying());
            let column_bits = u64::from(column_eligible.bitmask().to_underlying());

            if row_bits | column_bits != 0 {
                let values = distances.to_array();
                let values = values.as_ref();
                let mut row_bits = row_bits;
                while row_bits != 0 {
                    let lane = row_bits.trailing_zeros() as usize;
                    row_bits &= row_bits - 1;
                    let distance = values[lane];
                    if distance < row_worst {
                        row_worst = output.insert(row, (column + lane) as u32, distance);
                    }
                }

                let mut column_bits = column_bits;
                while column_bits != 0 {
                    let lane = column_bits.trailing_zeros() as usize;
                    column_bits &= column_bits - 1;
                    let target = column + lane;
                    let new_worst = output.insert(target, row as u32, values[lane]);
                    // SAFETY: `target < row < worst.len()`.
                    unsafe { *worst_ptr.add(target) = new_worst };
                }
            }
            column += F::LANES;
        }

        while column < row {
            // SAFETY: the scalar tail remains in the strict lower triangle.
            let dot = unsafe { *dots.get_unchecked(row_start + column) };
            let (row_norm, column_norm) = if uses_norms {
                // SAFETY: `column < row < points == norms.len()`.
                (norms[row], unsafe { *norms.get_unchecked(column) })
            } else {
                (0.0, 0.0)
            };
            let distance = M::leaf_distance_scalar(dot, row_norm, column_norm);
            if distance < row_worst {
                row_worst = output.insert(row, column as u32, distance);
            }
            // SAFETY: `column < row < worst.len()`.
            let column_worst = unsafe { *worst_ptr.add(column) };
            if distance < column_worst {
                let new_worst = output.insert(column, row as u32, distance);
                // SAFETY: `column < row < worst.len()`.
                unsafe { *worst_ptr.add(column) = new_worst };
            }
            column += 1;
        }
        // SAFETY: `row < worst.len()`.
        unsafe { *worst_ptr.add(row) = row_worst };
    }

    debug_assert_eq!(output.len(), points);
}

#[cfg(test)]
fn process_pairs_scalar<M: KernelMetric>(
    input: LeafTopK<'_>,
    k: usize,
    output: &mut [LeafNeighbor],
    norms: &[f32],
    worst: &mut [f32],
) {
    let points = input.dots.nrows();
    let uses_norms = M::LEAF_SCALE.is_some();
    for row in 1..points {
        for column in 0..row {
            let (row_norm, column_norm) = if uses_norms {
                (norms[row], norms[column])
            } else {
                (0.0, 0.0)
            };
            let distance =
                M::leaf_distance_scalar(input.dots[(row, column)], row_norm, column_norm);
            insert_scalar(output, worst, k, row, column as u32, distance);
            insert_scalar(output, worst, k, column, row as u32, distance);
        }
    }
}

#[cfg(test)]
fn insert_scalar(
    output: &mut [LeafNeighbor],
    worst: &mut [f32],
    k: usize,
    row: usize,
    position: u32,
    distance: f32,
) {
    if distance.partial_cmp(&worst[row]) != Some(std::cmp::Ordering::Less) {
        return;
    }
    worst[row] = insert_dynamic(&mut output[row * k..(row + 1) * k], position, distance);
}

/// Insert into a fixed-width row and return its new worst distance.
///
/// Production widths one through three use straight-line shifts. Strict `<`
/// comparisons preserve scan order for ties; callers already rejected NaN via
/// the eligibility comparison.
#[inline(always)]
fn insert_fixed<const N: usize>(row: &mut [LeafNeighbor; N], position: u32, distance: f32) -> f32 {
    let entry = LeafNeighbor::new(position, distance);
    match N {
        1 => {
            row[0] = entry;
            distance
        }
        2 => {
            let first = row[0];
            if distance < first.distance {
                row[0] = entry;
                row[1] = first;
                first.distance
            } else {
                row[1] = entry;
                distance
            }
        }
        3 => {
            let (first, second) = (row[0], row[1]);
            if distance < first.distance {
                row[0] = entry;
                row[1] = first;
                row[2] = second;
            } else if distance < second.distance {
                row[1] = entry;
                row[2] = second;
            } else {
                row[2] = entry;
                return distance;
            }
            second.distance
        }
        _ => unreachable!("fixed leaf widths are one through three"),
    }
}

/// Insert into a run-time-width row using the same stable ordering contract.
///
/// The candidate replaces the last slot, then bubbles toward the front. This
/// path is used only for k greater than three.
#[inline(always)]
fn insert_dynamic(row: &mut [LeafNeighbor], position: u32, distance: f32) -> f32 {
    let last = row.len() - 1;
    row[last] = LeafNeighbor::new(position, distance);
    let mut index = last;
    while index > 0 && row[index].distance < row[index - 1].distance {
        row.swap(index, index - 1);
        index -= 1;
    }
    row[last].distance
}

#[cfg(test)]
mod tests {
    use crate::kernel_metric::{Cosine, CosineNormalized, InnerProduct, KernelMetric, L2};

    use super::*;

    fn dots(metric: Metric, points: usize) -> Vec<f32> {
        let mut dots = vec![f32::NAN; points * points];
        for row in 0..points {
            dots[row * points + row] = if metric == Metric::Cosine && row == 0 {
                0.0
            } else {
                1.0 + (row % 5) as f32
            };
            for column in 0..row {
                dots[row * points + column] =
                    (((row * 17 + column * 11) % 23) as f32 - 11.0) * 0.03125;
            }
        }
        dots
    }

    fn input(dots: &[f32], points: usize) -> LeafTopK<'_> {
        LeafTopK {
            dots: MatrixView::try_from(dots, points, points).unwrap(),
        }
    }

    fn scalar<M: KernelMetric>(input: LeafTopK<'_>, k: usize, output: &mut [LeafNeighbor]) {
        let points = input.dots.nrows();
        let norms: Vec<_> = (0..points)
            .map(|row| M::LEAF_SCALE.transform(input.dots[(row, row)]))
            .collect();
        let mut worst = vec![f32::INFINITY; points];
        process_pairs_scalar::<M>(input, k, output, &norms, &mut worst);
    }

    fn scalar_for_metric(
        metric: Metric,
        input: LeafTopK<'_>,
        k: usize,
        output: &mut [LeafNeighbor],
    ) {
        match metric {
            Metric::L2 => scalar::<L2>(input, k, output),
            Metric::Cosine => scalar::<Cosine>(input, k, output),
            Metric::CosineNormalized => scalar::<CosineNormalized>(input, k, output),
            Metric::InnerProduct => scalar::<InnerProduct>(input, k, output),
        }
    }

    fn assert_scalar_reference_matches_prepared_dispatch(metric: Metric) {
        // Point count controls SIMD chunking. Cover both sides of 4-, 8-, and
        // 16-lane boundaries, then the boundary around a second 16-lane chunk.
        for points in [2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            let dots = dots(metric, points);
            let input = input(&dots, points);
            for requested_k in [1, 2, 3, 4] {
                let k = requested_k.min(points - 1);
                let kernel = LeafKernel::new(metric, requested_k);
                let mut expected = vec![LeafNeighbor::default(); points * k];
                kernel
                    .nearest_neighbors(
                        input,
                        MutMatrixView::try_from(expected.as_mut_slice(), points, k).unwrap(),
                        &mut LeafTopKWorkspace::new(),
                    )
                    .unwrap();

                let mut actual = vec![LeafNeighbor::default(); points * k];
                scalar_for_metric(metric, input, k, &mut actual);

                assert_eq!(actual, expected, "{metric:?}, n={points}, k={requested_k}");
            }
        }
    }

    #[test]
    fn l2_scalar_reference_matches_prepared_dispatch_at_lane_boundaries() {
        assert_scalar_reference_matches_prepared_dispatch(Metric::L2);
    }

    #[test]
    fn cosine_scalar_reference_matches_prepared_dispatch_at_lane_boundaries() {
        assert_scalar_reference_matches_prepared_dispatch(Metric::Cosine);
    }

    #[test]
    fn normalized_cosine_scalar_reference_matches_prepared_dispatch_at_lane_boundaries() {
        assert_scalar_reference_matches_prepared_dispatch(Metric::CosineNormalized);
    }

    #[test]
    fn inner_product_scalar_reference_matches_prepared_dispatch_at_lane_boundaries() {
        assert_scalar_reference_matches_prepared_dispatch(Metric::InnerProduct);
    }

    #[test]
    fn scalar_insertion_orders_candidates_and_rejects_nan() {
        let mut output = [LeafNeighbor::default(); 4];
        let mut worst = [f32::INFINITY];

        for (position, distance) in [(0, 4.0), (1, 1.0), (2, 3.0), (3, 2.0), (4, 0.5)] {
            insert_scalar(&mut output, &mut worst, 4, 0, position, distance);
        }
        insert_scalar(&mut output, &mut worst, 4, 0, 5, f32::NAN);

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
    fn workspace_can_shrink_and_grow_between_calls() {
        let kernel = LeafKernel::new(Metric::L2, 2);
        let mut workspace = LeafTopKWorkspace::new();
        for points in [17, 7, 17] {
            let dots = dots(Metric::L2, points);
            let mut output = vec![LeafNeighbor::default(); points * 2];
            kernel
                .nearest_neighbors(
                    input(&dots, points),
                    MutMatrixView::try_from(output.as_mut_slice(), points, 2).unwrap(),
                    &mut workspace,
                )
                .unwrap();
            assert!(output.iter().all(|neighbor| neighbor.position != u32::MAX));
        }
    }
}
