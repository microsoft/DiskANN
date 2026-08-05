/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Prepared distance and top-k kernels for partition assignment.
//!
//! PiPNN recursively turns a dataset into small, overlapping groups called
//! *leaves*. At one recursion node it samples several existing points as
//! *leaders*. Each leader represents one child group. Every point is assigned to
//! its nearest `fanout` leaders, so `fanout > 1` copies that point into multiple
//! children and creates overlap. Children larger than the configured leaf limit
//! are partitioned again.
//!
//! This module performs only the nearest-leader selection inside that stage. It
//! does not sample leaders, gather vectors, run GEMM, group point IDs, or recurse.
//! The caller gathers a stripe of points and all leaders, computes their dot
//! products as one general matrix multiplication (GEMM), and passes that matrix
//! here.
//! [`PartitionKernel::nearest_leaders`] converts dots to metric scores and writes
//! leader column positions; the caller uses those positions to form child groups.
//!
//! For example, output `[2, 5, 7]` for one point at fanout three means: add that
//! point to children represented by leader columns 2, 5, and 7. It does not mean
//! those leaders are final graph neighbors.
//!
//! The caller computes a row-major `points · leadersᵀ` tile with GEMM, then
//! passes it to a [`PartitionKernel`] prepared once for the build metric. Kernel
//! preparation selects the runtime architecture and concrete metric type once;
//! repeated stripes call a direct `diskann-wide` function pointer with no ISA or
//! metric branch in the point loop.
//!
//! L2 deliberately omits the point norm because it is constant across every
//! leader for that point. Cosine consumes squared point norms and leader norms. NaN
//! distances are not rankable, and equal distances retain leader scan order.
//!
//! ```text
//! metric + runtime architecture
//!             │
//!             v
//!   prepared Dispatched2 handle
//!             │ reused for every point stripe
//!             v
//! shape/scale validation -> SIMD chunks + scalar tail -> sorted leader IDs
//! ```
//!
//! Each point owns a fixed-capacity sorted tracker. Its last retained distance
//! is the rejection threshold, so noncompetitive SIMD chunks avoid lane extraction.
//!
//! # Main structures
//!
//! - [`PartitionKernel`] is the reusable public handle containing one prepared
//!   direct function pointer.
//! - [`PartitionInput`] bundles borrowed point-leader dots with
//!   [`PartitionScales`], whose variants make scale units explicit.
//! - `PartitionEntry<M>` is the architecture/metric-specialized destination that
//!   validates a call before entering pointer-based SIMD.
//! - `process_points` is the shared point traversal. Concrete metric scale kinds
//!   specialize unary/no-scale and binary-scale formulas without separate
//!   runtime row processors.
//! - `LeaderTracker`, `insert_leader_lanes`, and `insert_leader` maintain one
//!   fixed-capacity, stable sorted prefix per point.
//!
//! # Inputs and output
//!
//! For `p` points and `l` leaders, [`PartitionInput::dots`] is the row-major
//! `p × l` GEMM result. [`PartitionScales`] supplies exactly the scale units
//! required by the prepared metric. Output is a `p × f` matrix of leader-local
//! positions, where `f = output.ncols()` is requested fanout. Every point's
//! output is sorted by ascending score.
//!
//! Scores are derived from one point-leader dot product. Smaller is better:
//!
//! | Prepared metric | Score | Required [`PartitionScales`] |
//! | --- | --- | --- |
//! | squared L2 | `‖leader‖² - 2(point·leader)` | [`PartitionScales::L2`] |
//! | cosine | `1 - (point·leader)/(‖point‖‖leader‖)` | [`PartitionScales::Cosine`] |
//! | normalized cosine | `1 - point·leader` | [`PartitionScales::None`] |
//! | inner product | `-(point·leader)` | [`PartitionScales::None`] |
//!
//! Squared L2 omits `‖point‖²` because adding the same value to every leader
//! cannot change their order. `CosineNormalized` assumes vectors were normalized
//! before GEMM; this kernel does not verify vector norms.
//!
//! # Core flow
//!
//! 1. Validate matrix areas, backing lengths, fanout, and metric scale variant.
//! 2. Transform one point scale outside its leader loop when required.
//! 3. Score full SIMD leader groups and reject noncompetitive groups by mask.
//! 4. Score scalar-tail leaders with the metric's scalar operation order.
//! 5. Copy sorted leader IDs and reject underfilled points.
//!
//! # Performance
//!
//! With `p > 0` and `f > 0`, the kernel evaluates exactly `p * l` scores;
//! empty stripes or zero fanout return before traversal. Competitive leaders
//! bubble through at most `f <= MAX_PARTITION_FANOUT` tracker slots, giving
//! `O(plf)` worst-case work and `O(pl)` score computation. Tracker storage is a
//! fixed `O(MAX_PARTITION_FANOUT)` stack array per point; output is `O(pf)` and
//! no heap allocation occurs. Whole SIMD groups with no score below the current
//! threshold avoid lane materialization. Runtime architecture and metric selection happen
//! once in [`PartitionKernel::new`], outside stripe processing.
//!
//! # Example
//!
//! ```
//! use diskann::graph::pipnn::partition_kernel::{
//!     PartitionInput, PartitionKernel, PartitionScales,
//! };
//! use diskann_utils::views::{MatrixView, MutMatrixView};
//! use diskann_vector::distance::Metric;
//!
//! let dots = [
//!     0.8, 0.2, 0.5,
//!     0.1, 0.9, 0.3,
//! ];
//! let input = PartitionInput {
//!     dots: MatrixView::try_from(&dots[..], 2, 3).unwrap(),
//!     scales: PartitionScales::None,
//! };
//! let mut assignments = vec![u32::MAX; 2 * 2];
//! let output = MutMatrixView::try_from(&mut assignments[..], 2, 2).unwrap();
//!
//! PartitionKernel::new(Metric::CosineNormalized)
//!     .nearest_leaders(input, output)
//!     .unwrap();
//!
//! assert_eq!(assignments, [0, 2, 1, 2]);
//! ```

use std::marker::PhantomData;

use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_vector::distance::Metric;
use diskann_wide::{
    Architecture, SIMDFloat, SIMDMask, SIMDPartialOrd, SIMDSelect, SIMDVector,
    arch::{self, Dispatched2, FTarget2},
    lifetime::AddLifetime,
};

use super::kernel_metric::{KernelMetric, MetricVisitor, ScaleKind, visit_metric};

/// Maximum number of leaders retained for one point.
///
/// Supported PiPNN partition fanouts fit within 16. Keeping this as a fixed
/// stack tracker bounds per-point stack use and code size; larger requests are
/// rejected rather than silently truncated.
pub const MAX_PARTITION_FANOUT: usize = 16;

type LeaderTracker = [(u32, f32); MAX_PARTITION_FANOUT];

/// Metric-specific normalization inputs for one partition tile.
///
/// Slice lengths are checked against dot-matrix dimensions before output
/// mutation. Names encode units: cosine points arrive as squared norms because
/// they come from the point matrix diagonal, while leaders are normalized once
/// by the partition caller and arrive as norms.
#[derive(Clone, Copy, Debug)]
pub enum PartitionScales<'a> {
    /// L2 needs only squared leader norms; the point norm cannot affect ranking.
    L2 {
        /// Squared norm for every leader column.
        leader_squared_norms: &'a [f32],
    },
    /// Unnormalized cosine needs squared point norms and leader norms.
    Cosine {
        /// Squared norm for every point.
        point_squared_norms: &'a [f32],
        /// Norm for every leader column.
        leader_norms: &'a [f32],
    },
    /// Normalized cosine and inner product need no normalization inputs.
    None,
}

/// One row-major point-by-leader dot-product tile.
///
/// Matrix rows are points, columns are leaders, and [`Self::scales`] must match
/// the metric used to prepare [`PartitionKernel`]. This value only borrows input;
/// the prepared kernel stores no tile state.
#[derive(Clone, Copy, Debug)]
pub struct PartitionInput<'a> {
    /// One point per matrix row and one leader per column.
    pub dots: MatrixView<'a, f32>,
    /// Normalization inputs matching the prepared metric.
    pub scales: PartitionScales<'a>,
}

/// Validation error returned by [`PartitionKernel::nearest_leaders`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PartitionKernelError {
    /// A declared matrix shape overflowed `usize`.
    #[error("{buffer} shape {rows} x {cols} overflows usize")]
    ShapeOverflow {
        /// Name of the buffer whose shape overflowed.
        buffer: &'static str,
        /// Declared row count.
        rows: usize,
        /// Declared column count.
        cols: usize,
    },
    /// The output matrix does not match the input row count.
    #[error(
        "invalid output shape: expected {expected_rows} rows, got {actual_rows} rows and {actual_cols} columns"
    )]
    InvalidOutputShape {
        /// Required row count.
        expected_rows: usize,
        /// Supplied row count.
        actual_rows: usize,
        /// Supplied column count.
        actual_cols: usize,
    },
    /// A metric-specific scale slice has the wrong length.
    #[error("invalid {buffer} length: expected {expected}, got {actual}")]
    InvalidBufferLength {
        /// Name of the invalid scale buffer.
        buffer: &'static str,
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },
    /// Scale inputs do not match the metric used to prepare the kernel.
    #[error("partition scales do not match prepared {expected} metric")]
    InvalidScales {
        /// Expected scale layout.
        expected: &'static str,
    },
    /// The requested fanout cannot be represented by the fixed top-k tracker.
    #[error(
        "invalid fanout {fanout}: must not exceed {leader_count} leaders or kernel maximum {maximum}"
    )]
    InvalidFanout {
        /// Requested number of leaders per point.
        fanout: usize,
        /// Available leader count.
        leader_count: usize,
        /// Kernel maximum.
        maximum: usize,
    },
    /// Leader positions cannot be represented as `u32`.
    #[error("leader count {0} exceeds the u32 position limit")]
    TooManyLeaders(usize),
    /// A point did not contain enough rankable leaders to fill its output.
    #[error("point {point} has fewer than {fanout} rankable leaders")]
    InsufficientRankableLeaders {
        /// Zero-based point position in the input tile.
        point: usize,
        /// Requested number of leader positions.
        fanout: usize,
    },
}

/// Lifetime families used by the direct function-pointer interface.
///
/// Input and output receive independent call lifetimes. The prepared handle
/// stores neither view, so it remains `Copy + Send + Sync` across worker threads.
#[derive(Debug)]
struct PartitionInputArg;

impl AddLifetime for PartitionInputArg {
    type Of<'a> = PartitionInput<'a>;
}

#[derive(Debug)]
struct PartitionOutput;

impl AddLifetime for PartitionOutput {
    type Of<'a> = MutMatrixView<'a, u32>;
}

type PartitionFn =
    Dispatched2<Result<(), PartitionKernelError>, PartitionInputArg, PartitionOutput>;

/// A partition kernel prepared for one metric and the current CPU.
///
/// Construct this once with [`PartitionKernel::new`] and reuse it for every
/// point stripe. The handle is a direct function pointer and is `Copy`, `Send`,
/// and `Sync`.
///
/// It stores no matrix or output borrow, so callers may share one handle across
/// Rayon workers while each call owns independent views.
#[derive(Clone, Copy, Debug)]
pub struct PartitionKernel {
    run: PartitionFn,
}

impl PartitionKernel {
    /// Prepare a partition kernel for `metric` and the current CPU.
    ///
    /// The return value contains one architecture/metric-specialized function
    /// pointer and can process any valid stripe shape and fanout.
    ///
    /// # Performance
    ///
    /// Performs runtime architecture detection and one metric match once.
    /// Reusing the handle removes both decisions from point and leader loops.
    pub fn new(metric: Metric) -> Self {
        diskann_wide::arch::dispatch1_no_features(PreparePartition, metric)
    }

    /// Select the nearest leader positions for every input point.
    ///
    /// `output.nrows()` must equal `input.dots.nrows()`; its column count is the
    /// requested fanout. Results are ordered by ascending distance. For L2, the
    /// score omits the point norm because it cannot affect that point's ranking.
    ///
    /// `input` supplies point-leader dots and typed metric scales. `output` is
    /// overwritten with leader-local positions. Successful return guarantees
    /// exactly `output.ncols()` rankable leaders for every point.
    ///
    /// # Core flow
    ///
    /// The prepared entry validates every view and scale slice before mutation,
    /// runs one architecture/metric-specialized point traversal, then checks the
    /// final tracker slot for underfill.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionKernelError`] for overflowing or mismatched shapes,
    /// wrong scale variants or lengths, excessive fanout/leader counts, or a
    /// point with too few rankable scores. Validation errors leave output
    /// unchanged.
    ///
    /// # Performance
    ///
    /// See module-level complexity. This call follows one prepared direct
    /// function pointer and performs no runtime ISA or metric dispatch.
    pub fn nearest_leaders(
        &self,
        input: PartitionInput<'_>,
        output: MutMatrixView<'_, u32>,
    ) -> Result<(), PartitionKernelError> {
        self.run.call(input, output)
    }
}

/// First dispatch stage: select runtime architecture once.
///
/// `dispatch1_no_features` runs only this factory. The returned entry pointer is
/// generated by the selected architecture and carries its required features.
struct PreparePartition;

impl<A> arch::Target1<A, PartitionKernel, Metric> for PreparePartition
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    fn run(self, arch: A, metric: Metric) -> PartitionKernel {
        visit_metric(metric, BuildPartition(arch))
    }
}

/// BYO-type-erasure visitor holding a concrete architecture.
///
/// `visit<M>` combines architecture `A` and concrete metric `M`, then produces
/// one direct function pointer. No nested metric trait object remains at runtime.
struct BuildPartition<A>(A);

impl<A> MetricVisitor for BuildPartition<A>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    type Output = PartitionKernel;

    fn visit<M: KernelMetric>(self) -> Self::Output {
        PartitionKernel {
            run: self.0.dispatch2::<
                PartitionEntry<M>,
                Result<(), PartitionKernelError>,
                PartitionInputArg,
                PartitionOutput,
            >(),
        }
    }
}

/// Architecture/metric-specialized function-pointer destination.
///
/// The zero-sized entry receives all stripe state as arguments. Validation must
/// complete before `process_points` reaches unchecked contiguous SIMD loads.
///
/// Call order is fixed: validate without mutation, handle empty work, execute one
/// specialized traversal, then verify each point's last assignment. Keeping the
/// phases together makes every unchecked load depend on one visible gate.
struct PartitionEntry<M>(PhantomData<M>);

impl<A, M> FTarget2<A, Result<(), PartitionKernelError>, PartitionInput<'_>, MutMatrixView<'_, u32>>
    for PartitionEntry<M>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    M: KernelMetric,
{
    fn run(
        arch: A,
        input: PartitionInput<'_>,
        mut output: MutMatrixView<'_, u32>,
    ) -> Result<(), PartitionKernelError> {
        // Validation establishes matrix areas, backing lengths, scale units,
        // and fanout bounds before any output mutation or unchecked load.
        let scales = validate::<M>(input, &output)?;
        let fanout = output.ncols();
        // Zero fanout and empty stripes require no assignments. Return before
        // constructing trackers or touching output.
        if fanout == 0 || input.dots.nrows() == 0 {
            return Ok(());
        }

        // Architecture and metric are concrete here; only stripe dimensions and
        // fanout remain runtime values.
        process_points::<A::f32x16, M>(arch, input.dots, scales, fanout, output.as_mut_slice());
        // A sorted tracker can be underfilled only at its last slot. This keeps
        // post-validation linear in points rather than scanning every output ID.
        if let Some(point) = output
            .as_slice()
            .chunks_exact(fanout)
            .position(|assignments| assignments[fanout - 1] == u32::MAX)
        {
            return Err(PartitionKernelError::InsufficientRankableLeaders { point, fanout });
        }
        Ok(())
    }
}

/// Validated scale slices in the storage form required by `M`.
///
/// Empty slices are intentional for metrics that omit a scale; consumers branch
/// on associated `ScaleKind` constants that monomorphize out of hot loops.
#[derive(Clone, Copy)]
struct ScaleSlices<'a> {
    point_scales: &'a [f32],
    leader_scales: &'a [f32],
}

/// Validate the complete partition-kernel safety and metric contract.
///
/// Matrix areas are recomputed with `checked_mul` before pointer loads. The
/// `PartitionScales` variant must match concrete metric `M`, preventing plausible
/// but incorrect norm units from crossing the interface.
///
/// `input` and `output` are inspected only. Success returns borrowed scale slices
/// normalized to the storage layout expected by `M`; it establishes exact
/// backing lengths, representable leader IDs, and bounded fanout. Failure returns
/// [`PartitionKernelError`] before output mutation. Runtime is constant apart
/// from view metadata checks; matrix and scale contents are not scanned.
fn validate<'a, M: KernelMetric>(
    input: PartitionInput<'a>,
    output: &MutMatrixView<'_, u32>,
) -> Result<ScaleSlices<'a>, PartitionKernelError> {
    let point_count = input.dots.nrows();
    let leader_count = input.dots.ncols();
    let fanout = output.ncols();

    let dots_len = checked_area("dot-product tile", point_count, leader_count)?;
    check_length("dot-product tile", input.dots.as_slice().len(), dots_len)?;
    let output_len = checked_area("output", output.nrows(), fanout)?;
    check_length("output", output.as_slice().len(), output_len)?;

    if output.nrows() != point_count {
        return Err(PartitionKernelError::InvalidOutputShape {
            expected_rows: point_count,
            actual_rows: output.nrows(),
            actual_cols: output.ncols(),
        });
    }
    if leader_count > u32::MAX as usize {
        return Err(PartitionKernelError::TooManyLeaders(leader_count));
    }
    if fanout > MAX_PARTITION_FANOUT || fanout > leader_count {
        return Err(PartitionKernelError::InvalidFanout {
            fanout,
            leader_count,
            maximum: MAX_PARTITION_FANOUT,
        });
    }

    // Match the public enum against the concrete marker before erasing it to
    // slices. This prevents squared point norms from being mistaken for leader
    // norms even though both representations are `&[f32]`.
    let scales = match (M::METRIC, input.scales) {
        (
            Metric::L2,
            PartitionScales::L2 {
                leader_squared_norms,
            },
        ) => ScaleSlices {
            point_scales: &[],
            leader_scales: leader_squared_norms,
        },
        (
            Metric::Cosine,
            PartitionScales::Cosine {
                point_squared_norms,
                leader_norms,
            },
        ) => ScaleSlices {
            point_scales: point_squared_norms,
            leader_scales: leader_norms,
        },
        (Metric::CosineNormalized | Metric::InnerProduct, PartitionScales::None) => ScaleSlices {
            point_scales: &[],
            leader_scales: &[],
        },
        (Metric::L2, _) => return Err(PartitionKernelError::InvalidScales { expected: "L2" }),
        (Metric::Cosine, _) => {
            return Err(PartitionKernelError::InvalidScales { expected: "cosine" });
        }
        (Metric::CosineNormalized, _) => {
            return Err(PartitionKernelError::InvalidScales {
                expected: "normalized cosine",
            });
        }
        (Metric::InnerProduct, _) => {
            return Err(PartitionKernelError::InvalidScales {
                expected: "inner product",
            });
        }
    };

    // After variant validation, associated scale kinds define exact lengths.
    // Scale-free metrics must provide empty slices so stale data cannot be used.
    check_length(
        "point scales",
        scales.point_scales.len(),
        expected_scale_len(M::PARTITION_POINT_SCALE, point_count),
    )?;
    check_length(
        "leader scales",
        scales.leader_scales.len(),
        expected_scale_len(M::PARTITION_LEADER_SCALE, leader_count),
    )?;
    Ok(scales)
}

/// Return required scale length after metric specialization.
///
/// Associated `ScaleKind` constants make this choice compile away.
const fn expected_scale_len(kind: ScaleKind, count: usize) -> usize {
    if kind.is_some() { count } else { 0 }
}

fn checked_area(
    buffer: &'static str,
    rows: usize,
    cols: usize,
) -> Result<usize, PartitionKernelError> {
    rows.checked_mul(cols)
        .ok_or(PartitionKernelError::ShapeOverflow { buffer, rows, cols })
}

fn check_length(
    buffer: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), PartitionKernelError> {
    if actual == expected {
        Ok(())
    } else {
        Err(PartitionKernelError::InvalidBufferLength {
            buffer,
            expected,
            actual,
        })
    }
}

/// Convert each point's leader scores into sorted top-fanout IDs.
///
/// Per-point flow:
///
/// 1. transform the point scale once according to concrete metric `M`;
/// 2. process full SIMD leader groups, rejecting lanes against the last slot;
/// 3. process the remaining leaders with the scalar metric operation;
/// 4. copy the sorted tracker prefix to that point's output.
///
/// `tracker[..fanout]` remains sorted after every accepted candidate. Strict `<`
/// preserves leader scan order for ties and makes NaNs non-rankable. L2 keeps
/// historical bulk-FMA/scalar-tail rounding because changing it can alter graph
/// assignment at near ties.
///
/// `dots` supplies `p × l` scores, `scales` contains validated metric inputs,
/// `fanout` is both tracker prefix length and output width, and `output` contains
/// `p * fanout` slots. The function writes leader IDs in place and returns no
/// value. It computes `p * l` scores; each competitive score may shift `O(fanout)`
/// tracker entries. Tracker memory is fixed on the stack and no allocation occurs.
fn process_points<F, M>(
    arch: F::Arch,
    dots: MatrixView<'_, f32>,
    scales: ScaleSlices<'_>,
    fanout: usize,
    output: &mut [u32],
) where
    F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    M: KernelMetric,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let leader_count = dots.ncols();
    // Each point is independent. Reinitialize the fixed tracker here so no
    // assignment state or tie order leaks across points.
    for (point, (point_dots, point_output)) in dots
        .as_slice()
        .chunks_exact(leader_count)
        .zip(output.chunks_exact_mut(fanout))
        .enumerate()
    {
        // Transform once per point rather than once per leader. For metrics
        // without a point scale, specialization removes this branch and load.
        let point_scale = if M::PARTITION_POINT_SCALE.is_some() {
            M::PARTITION_POINT_SCALE.transform(scales.point_scales[point])
        } else {
            0.0
        };
        let point_scale_vector = F::splat(arch, point_scale);
        let mut tracker = [(u32::MAX, f32::INFINITY); MAX_PARTITION_FANOUT];
        // Split at the largest complete vector boundary. Scalar tail uses the
        // metric's explicit scalar operation order, not a padded SIMD load.
        let full = leader_count / F::LANES * F::LANES;

        for base in (0..full).step_by(F::LANES) {
            // SAFETY: `base + F::LANES <= full <= point_dots.len()`.
            let point_dots = unsafe { F::load_simd(arch, point_dots.as_ptr().add(base)) };
            let leader_scales = if M::PARTITION_LEADER_SCALE.is_some() {
                // SAFETY: validation requires one scale per leader.
                unsafe { F::load_simd(arch, scales.leader_scales.as_ptr().add(base)) }
            } else {
                F::default(arch)
            };
            insert_leader_lanes(
                M::partition_distance(arch, point_dots, point_scale_vector, leader_scales),
                base,
                &mut tracker,
                fanout,
            );
        }

        // Tail values use scalar metric functions intentionally. Padding a SIMD
        // group would risk out-of-bounds scale loads and different L2 rounding.
        for (leader, &dot) in point_dots.iter().enumerate().skip(full) {
            let leader_scale = if M::PARTITION_LEADER_SCALE.is_some() {
                M::PARTITION_LEADER_SCALE.transform(scales.leader_scales[leader])
            } else {
                0.0
            };
            insert_leader(
                &mut tracker,
                fanout,
                leader as u32,
                M::partition_distance_scalar(dot, point_scale, leader_scale),
            );
        }
        // Distances are only tracker state; child-group construction needs leader
        // column positions in deterministic nearest-first order.
        copy_leader_ids(&tracker, point_output);
    }
}

/// Offer competitive SIMD lanes to a point tracker in increasing leader order.
///
/// The broadcast threshold avoids materializing lanes when none can improve the
/// last slot. Bit iteration follows low-to-high lane order, preserving scalar tie
/// behavior across SIMD widths.
///
/// `distances` contains consecutive leaders beginning at `first_leader`;
/// `tracker[..fanout]` is the point's sorted retained prefix. The function
/// mutates that tracker and returns no value. Rejected groups cost one comparison
/// and mask test; accepted lanes each pay `O(fanout)` worst-case insertion.
fn insert_leader_lanes<F>(
    distances: F,
    first_leader: usize,
    tracker: &mut LeaderTracker,
    fanout: usize,
) where
    F: SIMDVector<Scalar = f32> + SIMDPartialOrd,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let threshold = F::splat(distances.arch(), tracker[fanout - 1].1);
    let eligible = distances.lt_simd(threshold);
    if eligible.none() {
        return;
    }

    let values = distances.to_array();
    let values = values.as_ref();
    let mut lanes = u64::from(eligible.bitmask().to_underlying());
    while lanes != 0 {
        let lane = lanes.trailing_zeros() as usize;
        lanes &= lanes - 1;
        insert_leader(tracker, fanout, (first_leader + lane) as u32, values[lane]);
    }
}

/// Insert one strictly better candidate while preserving sorted-prefix state.
///
/// The last slot is overwritten, then bubbled left. Equal and NaN distances do
/// not enter, so scan order is the deterministic tie breaker and the last slot
/// remains both rejection threshold and underfill sentinel.
///
/// `tracker[..fanout]` must already be sorted and `fanout` must be non-zero.
/// `leader` is a local column position. The function returns no value and shifts
/// at most `fanout - 1` entries without allocation.
#[inline(always)]
fn insert_leader(tracker: &mut LeaderTracker, fanout: usize, leader: u32, distance: f32) {
    let threshold = fanout - 1;
    if distance.partial_cmp(&tracker[threshold].1) != Some(std::cmp::Ordering::Less) {
        return;
    }

    tracker[threshold] = (leader, distance);
    let mut slot = threshold;
    while slot > 0 && tracker[slot].1 < tracker[slot - 1].1 {
        tracker.swap(slot, slot - 1);
        slot -= 1;
    }
}

/// Publish only leader IDs; distances stay private tracker state.
///
/// `assignments.len()` is validated fanout. Copying costs `O(fanout)` and leaves
/// tracker state available for the underfill sentinel check encoded in IDs.
fn copy_leader_ids(tracker: &LeaderTracker, assignments: &mut [u32]) {
    for (destination, &(leader, _)) in assignments.iter_mut().zip(tracker) {
        *destination = leader;
    }
}

#[cfg(test)]
mod tests {
    use super::super::kernel_metric::{Cosine, CosineNormalized, InnerProduct, KernelMetric, L2};

    use super::*;

    fn test_input<'a>(
        metric: Metric,
        dots: &'a [f32],
        point_count: usize,
        leader_count: usize,
        point_scales: &'a [f32],
        leader_scales: &'a [f32],
    ) -> PartitionInput<'a> {
        let scales = match metric {
            Metric::L2 => PartitionScales::L2 {
                leader_squared_norms: leader_scales,
            },
            Metric::Cosine => PartitionScales::Cosine {
                point_squared_norms: point_scales,
                leader_norms: leader_scales,
            },
            Metric::CosineNormalized | Metric::InnerProduct => PartitionScales::None,
        };
        PartitionInput {
            dots: MatrixView::try_from(dots, point_count, leader_count).unwrap(),
            scales,
        }
    }

    // Differential oracle for SIMD chunking, scalar tails, and tracker order.
    // It intentionally shares `M::partition_distance_scalar`; public API tests
    // independently spell out ranking formulas and full sorting behavior.
    fn scalar_traversal_reference<M: KernelMetric>(
        input: PartitionInput<'_>,
        fanout: usize,
        output: &mut [u32],
    ) {
        let scales = match input.scales {
            PartitionScales::L2 {
                leader_squared_norms,
            } => ScaleSlices {
                point_scales: &[],
                leader_scales: leader_squared_norms,
            },
            PartitionScales::Cosine {
                point_squared_norms,
                leader_norms,
            } => ScaleSlices {
                point_scales: point_squared_norms,
                leader_scales: leader_norms,
            },
            PartitionScales::None => ScaleSlices {
                point_scales: &[],
                leader_scales: &[],
            },
        };
        let leader_count = input.dots.ncols();
        for (point, (point_dots, point_output)) in input
            .dots
            .as_slice()
            .chunks_exact(leader_count)
            .zip(output.chunks_exact_mut(fanout))
            .enumerate()
        {
            let point_scale = if M::PARTITION_POINT_SCALE.is_some() {
                M::PARTITION_POINT_SCALE.transform(scales.point_scales[point])
            } else {
                0.0
            };
            let mut tracker = [(u32::MAX, f32::INFINITY); MAX_PARTITION_FANOUT];
            for (leader, &dot) in point_dots.iter().enumerate() {
                let leader_scale = if M::PARTITION_LEADER_SCALE.is_some() {
                    M::PARTITION_LEADER_SCALE.transform(scales.leader_scales[leader])
                } else {
                    0.0
                };
                insert_leader(
                    &mut tracker,
                    fanout,
                    leader as u32,
                    M::partition_distance_scalar(dot, point_scale, leader_scale),
                );
            }
            copy_leader_ids(&tracker, point_output);
        }
    }

    #[test]
    fn scalar_distance_matches_metric_contract() {
        assert_eq!(L2::partition_distance_scalar(2.0, 0.0, 9.0), 5.0);
        assert_eq!(
            CosineNormalized::partition_distance_scalar(0.25, 0.0, 0.0),
            0.75
        );
        assert_eq!(InnerProduct::partition_distance_scalar(3.0, 0.0, 0.0), -3.0);
        assert_eq!(Cosine::partition_distance_scalar(4.0, 2.0, 4.0), 0.5);
        assert_eq!(Cosine::partition_distance_scalar(4.0, 0.0, 4.0), 1.0);
        assert!(Cosine::partition_distance_scalar(1.0, f32::NAN, 1.0).is_nan());
    }

    #[test]
    fn cosine_special_norms_match_scalar_and_prepared_dispatch() {
        let leader_count = 17;
        let point_scales = [0.0, f32::MIN_POSITIVE / 2.0, f32::MIN_POSITIVE, f32::NAN];
        let dots = vec![1.0; point_scales.len() * leader_count];
        let mut leader_scales = vec![1.0; leader_count];
        leader_scales[..4].copy_from_slice(&[
            0.0,
            f32::MIN_POSITIVE.sqrt() / 2.0,
            f32::MIN_POSITIVE.sqrt(),
            f32::NAN,
        ]);
        let input = test_input(
            Metric::Cosine,
            &dots,
            point_scales.len(),
            leader_count,
            &point_scales,
            &leader_scales,
        );
        let mut expected = vec![u32::MAX; point_scales.len() * 2];
        scalar_traversal_reference::<Cosine>(input, 2, &mut expected);
        let mut actual = vec![u32::MAX; point_scales.len() * 2];
        PartitionKernel::new(Metric::Cosine)
            .nearest_leaders(
                input,
                MutMatrixView::try_from(actual.as_mut_slice(), point_scales.len(), 2).unwrap(),
            )
            .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(&actual[..4], &[0, 1, 0, 1]);
        assert_eq!(&actual[6..], &[0, 1]);
    }

    #[test]
    fn matrix_area_overflow_is_rejected_before_kernel_access() {
        assert_eq!(
            checked_area("dot-product tile", usize::MAX, 2),
            Err(PartitionKernelError::ShapeOverflow {
                buffer: "dot-product tile",
                rows: usize::MAX,
                cols: 2,
            })
        );
    }

    #[test]
    fn scalar_topk_orders_candidates_and_preserves_ties() {
        let mut tracker = [(u32::MAX, f32::INFINITY); MAX_PARTITION_FANOUT];
        for (leader, distance) in [(0, 4.0), (1, 1.0), (2, 3.0), (3, 2.0), (4, 1.0)] {
            insert_leader(&mut tracker, 4, leader, distance);
        }
        insert_leader(&mut tracker, 4, 5, f32::NAN);

        assert_eq!(tracker[..4], [(1, 1.0), (4, 1.0), (3, 2.0), (2, 3.0)]);
    }
}
