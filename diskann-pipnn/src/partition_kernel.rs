/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Distance and top-k kernel for partition assignment.
//!
//! The kernel consumes a row-major tile of point-to-leader dot products. It
//! converts those products to metric distances while retaining only leader
//! positions; partition recursion and cluster ownership stay with the caller.

use diskann_vector::distance::Metric;
#[cfg(target_arch = "x86_64")]
use diskann_wide::{SIMDFloat, SIMDMask, SIMDPartialOrd, SIMDSelect, SIMDVector};

/// Maximum number of leaders retained for one point.
pub const MAX_PARTITION_FANOUT: usize = 16;

type TopK = [(u32, f32); MAX_PARTITION_FANOUT];

/// Input tile and metric-specific normalization terms for partition top-k.
#[derive(Clone, Copy, Debug)]
pub struct PartitionTopK<'a> {
    /// Row-major `rows * leaders` point-to-leader dot products.
    pub dots: &'a [f32],
    /// Number of points represented by `dots`.
    pub rows: usize,
    /// Number of leaders represented by each row.
    pub leaders: usize,
    /// Squared point norms for cosine, otherwise empty.
    pub row_scales: &'a [f32],
    /// Leader norms for cosine, squared leader norms for L2, otherwise empty.
    pub leader_scales: &'a [f32],
    /// Distance metric used to rank leaders.
    pub metric: Metric,
}

/// Validation error returned by [`nearest_leaders`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PartitionKernelError {
    /// A declared matrix or output shape overflowed `usize`.
    #[error("{buffer} shape {rows} x {cols} overflows usize")]
    ShapeOverflow {
        /// Name of the buffer whose shape overflowed.
        buffer: &'static str,
        /// Declared row count.
        rows: usize,
        /// Declared column count.
        cols: usize,
    },
    /// A supplied slice did not match its declared shape.
    #[error("invalid {buffer} length: expected {expected}, got {actual}")]
    InvalidBufferLength {
        /// Name of the invalid buffer.
        buffer: &'static str,
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },
    /// The requested fanout cannot be represented by the fixed top-k tracker.
    #[error("invalid fanout {fanout} for {leaders} leaders; maximum is {maximum}")]
    InvalidFanout {
        /// Requested number of leaders per row.
        fanout: usize,
        /// Available leader count.
        leaders: usize,
        /// Kernel maximum.
        maximum: usize,
    },
    /// Leader positions cannot be represented as `u32`.
    #[error("leader count {0} exceeds the u32 position limit")]
    TooManyLeaders(usize),
    /// A row did not contain enough rankable distances to fill its output.
    #[error("row {row} has fewer than {fanout} rankable leader distances")]
    InsufficientRankableDistances {
        /// Zero-based row position in the input tile.
        row: usize,
        /// Requested number of leader positions.
        fanout: usize,
    },
}

/// Select the nearest `fanout` leader positions for every input row.
///
/// Results for each row are ordered by ascending distance. Equal distances do
/// not replace or move an already retained entry, so leader scan order breaks
/// ties. A zero fanout is a validated no-op.
///
/// For L2, the point's squared norm is omitted because it is constant across
/// every leader in a row and cannot change the ranking.
pub fn nearest_leaders(
    input: PartitionTopK<'_>,
    fanout: usize,
    output: &mut [u32],
) -> Result<(), PartitionKernelError> {
    validate(input, fanout, output)?;
    if fanout == 0 || input.rows == 0 {
        return Ok(());
    }

    diskann_wide::arch::dispatch(PartitionKernel {
        input,
        fanout,
        output,
    });
    if let Some(row) = output
        .chunks_exact(fanout)
        .position(|leaders| leaders.contains(&u32::MAX))
    {
        return Err(PartitionKernelError::InsufficientRankableDistances { row, fanout });
    }
    Ok(())
}

fn validate(
    input: PartitionTopK<'_>,
    fanout: usize,
    output: &[u32],
) -> Result<(), PartitionKernelError> {
    if input.leaders > u32::MAX as usize {
        return Err(PartitionKernelError::TooManyLeaders(input.leaders));
    }
    if fanout > MAX_PARTITION_FANOUT || fanout > input.leaders {
        return Err(PartitionKernelError::InvalidFanout {
            fanout,
            leaders: input.leaders,
            maximum: MAX_PARTITION_FANOUT,
        });
    }

    let expected_dots = checked_area("dot-product tile", input.rows, input.leaders)?;
    check_length("dot-product tile", input.dots.len(), expected_dots)?;
    let expected_output = checked_area("output", input.rows, fanout)?;
    check_length("output", output.len(), expected_output)?;

    let (row_scales, leader_scales) = match input.metric {
        Metric::Cosine => (input.rows, input.leaders),
        Metric::L2 => (0, input.leaders),
        Metric::CosineNormalized | Metric::InnerProduct => (0, 0),
    };
    check_length("row scales", input.row_scales.len(), row_scales)?;
    check_length("leader scales", input.leader_scales.len(), leader_scales)
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

struct PartitionKernel<'a, 'o> {
    input: PartitionTopK<'a>,
    fanout: usize,
    output: &'o mut [u32],
}

impl PartitionKernel<'_, '_> {
    fn run_scalar(self) {
        process_rows_scalar(self.input, self.fanout, self.output);
    }

    #[cfg(target_arch = "x86_64")]
    fn run_simd<F>(self, arch: F::Arch)
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
        u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    {
        process_rows_simd::<F>(arch, self.input, self.fanout, self.output);
    }
}

impl diskann_wide::arch::Target<diskann_wide::arch::Scalar, ()> for PartitionKernel<'_, '_> {
    #[inline(always)]
    fn run(self, _: diskann_wide::arch::Scalar) {
        self.run_scalar();
    }
}

#[cfg(target_arch = "x86_64")]
impl diskann_wide::arch::Target<diskann_wide::arch::x86_64::V3, ()> for PartitionKernel<'_, '_> {
    #[inline(always)]
    fn run(self, arch: diskann_wide::arch::x86_64::V3) {
        diskann_wide::alias!(F32x8 = <diskann_wide::arch::x86_64::V3>::f32x8);
        self.run_simd::<F32x8>(arch);
    }
}

#[cfg(target_arch = "x86_64")]
impl diskann_wide::arch::Target<diskann_wide::arch::x86_64::V4, ()> for PartitionKernel<'_, '_> {
    #[inline(always)]
    fn run(self, arch: diskann_wide::arch::x86_64::V4) {
        diskann_wide::alias!(F32x16 = <diskann_wide::arch::x86_64::V4>::f32x16);
        self.run_simd::<F32x16>(arch);
    }
}

#[cfg(target_arch = "aarch64")]
impl diskann_wide::arch::Target<diskann_wide::arch::aarch64::Neon, ()> for PartitionKernel<'_, '_> {
    #[inline(always)]
    fn run(self, arch: diskann_wide::arch::aarch64::Neon) {
        let _scalar = arch.retarget();
        self.run_scalar();
    }
}

fn process_rows_scalar(input: PartitionTopK<'_>, fanout: usize, output: &mut [u32]) {
    for (row_index, (dot_row, output_row)) in input
        .dots
        .chunks_exact(input.leaders)
        .zip(output.chunks_exact_mut(fanout))
        .enumerate()
    {
        let mut top = [(u32::MAX, f32::MAX); MAX_PARTITION_FANOUT];
        let row_scale = input.row_scales.get(row_index).copied().unwrap_or(0.0);
        for (leader, &dot) in dot_row.iter().enumerate() {
            let leader_scale = input.leader_scales.get(leader).copied().unwrap_or(0.0);
            insert_topk(
                &mut top,
                fanout,
                leader as u32,
                distance(input.metric, dot, row_scale, leader_scale),
            );
        }
        copy_ids(&top, output_row);
    }
}

#[cfg(target_arch = "x86_64")]
fn process_rows_simd<F>(arch: F::Arch, input: PartitionTopK<'_>, fanout: usize, output: &mut [u32])
where
    F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    for (row_index, (dot_row, output_row)) in input
        .dots
        .chunks_exact(input.leaders)
        .zip(output.chunks_exact_mut(fanout))
        .enumerate()
    {
        let mut top = [(u32::MAX, f32::MAX); MAX_PARTITION_FANOUT];
        match input.metric {
            Metric::L2 => process_binary::<F, _>(
                arch,
                dot_row,
                input.leader_scales,
                &mut top,
                fanout,
                |dot, norm| F::splat(arch, -2.0).mul_add_simd(dot, norm),
            ),
            Metric::CosineNormalized => {
                process_unary::<F, _>(arch, dot_row, &mut top, fanout, |dot| {
                    F::splat(arch, 1.0) - dot
                })
            }
            Metric::InnerProduct => process_unary::<F, _>(arch, dot_row, &mut top, fanout, |dot| {
                F::default(arch) - dot
            }),
            Metric::Cosine => process_cosine::<F>(
                arch,
                dot_row,
                input.row_scales[row_index],
                input.leader_scales,
                &mut top,
                fanout,
            ),
        }
        copy_ids(&top, output_row);
    }
}

#[cfg(target_arch = "x86_64")]
fn process_cosine<F>(
    arch: F::Arch,
    dots: &[f32],
    row_norm_squared: f32,
    leader_norms: &[f32],
    top: &mut TopK,
    fanout: usize,
) where
    F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let row_norm = F::splat(arch, row_norm_squared.sqrt());
    let one = F::splat(arch, 1.0);
    let zero = F::default(arch);
    process_binary::<F, _>(arch, dots, leader_norms, top, fanout, |dot, leader_norm| {
        let denominator = row_norm * leader_norm;
        let valid = denominator.gt_simd(zero);
        let safe_denominator = valid.select(denominator, one);
        let cosine = valid.select(dot / safe_denominator, zero);
        one - cosine
    });
}

#[cfg(target_arch = "x86_64")]
fn process_unary<F, Transform>(
    arch: F::Arch,
    dots: &[f32],
    top: &mut TopK,
    fanout: usize,
    transform: Transform,
) where
    F: SIMDVector<Scalar = f32> + SIMDFloat,
    Transform: Fn(F) -> F,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let full = dots.len() / F::LANES * F::LANES;
    for base in (0..full).step_by(F::LANES) {
        // SAFETY: `base + F::LANES <= full <= dots.len()`.
        let dots = unsafe { F::load_simd(arch, dots.as_ptr().add(base)) };
        insert_lanes(transform(dots), base, top, fanout);
    }
    for (offset, &dot) in dots[full..].iter().enumerate() {
        let mut lane = [0.0f32; 16];
        let value = transform(F::splat(arch, dot));
        // SAFETY: `lane` has capacity for every supported `F`.
        unsafe { value.store_simd(lane.as_mut_ptr()) };
        insert_topk(top, fanout, (full + offset) as u32, lane[0]);
    }
}

#[cfg(target_arch = "x86_64")]
fn process_binary<F, Transform>(
    arch: F::Arch,
    dots: &[f32],
    scales: &[f32],
    top: &mut TopK,
    fanout: usize,
    transform: Transform,
) where
    F: SIMDVector<Scalar = f32> + SIMDFloat,
    Transform: Fn(F, F) -> F,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let full = dots.len() / F::LANES * F::LANES;
    for base in (0..full).step_by(F::LANES) {
        // SAFETY: both slices contain the full SIMD chunk at `base`.
        let dots = unsafe { F::load_simd(arch, dots.as_ptr().add(base)) };
        // SAFETY: shape validation guarantees `scales.len() == dots.len()`.
        let scales = unsafe { F::load_simd(arch, scales.as_ptr().add(base)) };
        insert_lanes(transform(dots, scales), base, top, fanout);
    }
    for offset in 0..dots.len() - full {
        let mut lane = [0.0f32; 16];
        let value = transform(
            F::splat(arch, dots[full + offset]),
            F::splat(arch, scales[full + offset]),
        );
        // SAFETY: `lane` has capacity for every supported `F`.
        unsafe { value.store_simd(lane.as_mut_ptr()) };
        insert_topk(top, fanout, (full + offset) as u32, lane[0]);
    }
}

#[cfg(target_arch = "x86_64")]
fn insert_lanes<F>(distances: F, base: usize, top: &mut TopK, fanout: usize)
where
    F: SIMDVector<Scalar = f32> + SIMDPartialOrd,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let threshold = F::splat(distances.arch(), top[fanout - 1].1);
    let eligible = distances.lt_simd(threshold);
    if eligible.none() {
        return;
    }

    let mut values = [0.0f32; 16];
    // SAFETY: `values` has capacity for every f32 SIMD width DiskANN exposes.
    unsafe { distances.store_simd(values.as_mut_ptr()) };
    let mut lanes = u64::from(eligible.bitmask().to_underlying());
    while lanes != 0 {
        let lane = lanes.trailing_zeros() as usize;
        lanes &= lanes - 1;
        insert_topk(top, fanout, (base + lane) as u32, values[lane]);
    }
}

#[inline(always)]
fn distance(metric: Metric, dot: f32, row_scale: f32, leader_scale: f32) -> f32 {
    match metric {
        Metric::L2 => (-2.0f32).mul_add(dot, leader_scale),
        Metric::CosineNormalized => 1.0 - dot,
        Metric::InnerProduct => -dot,
        Metric::Cosine => {
            let denominator = row_scale.sqrt() * leader_scale;
            let cosine = if denominator > 0.0 {
                dot / denominator
            } else {
                0.0
            };
            1.0 - cosine
        }
    }
}

#[inline(always)]
fn insert_topk(top: &mut TopK, fanout: usize, leader: u32, distance: f32) {
    let threshold = fanout - 1;
    if distance.partial_cmp(&top[threshold].1) != Some(std::cmp::Ordering::Less) {
        return;
    }

    top[threshold] = (leader, distance);
    let mut position = threshold;
    while position > 0 && top[position].1 < top[position - 1].1 {
        top.swap(position, position - 1);
        position -= 1;
    }
}

fn copy_ids(top: &TopK, output: &mut [u32]) {
    for (destination, &(leader, _)) in output.iter_mut().zip(top) {
        *destination = leader;
    }
}
