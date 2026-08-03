/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Distance and top-k kernel for partition assignment.
//!
//! The caller gathers a point stripe and a leader matrix, then computes the
//! row-major `points · leadersᵀ` tile with GEMM. This module performs the second
//! half of assignment: convert each dot product to the configured metric and
//! retain only the nearest leader positions.
//!
//! L2 deliberately omits the point norm because it adds the same constant to
//! every leader in one row and cannot change their order. Cosine still needs a
//! point scale because it divides each dot product. The fixed 16-entry tracker
//! bounds stack use and matches the configuration fanout limit. SIMD chunks and
//! scalar tails feed the same insertion routine; NaNs are ignored and equal
//! distances keep the first leader encountered.

use diskann_vector::distance::Metric;
use diskann_wide::{Architecture, SIMDFloat, SIMDMask, SIMDPartialOrd, SIMDSelect, SIMDVector};

/// Maximum number of leaders retained for one point.
///
/// Supported PiPNN partition fanouts fit within 16. Keeping this as a fixed
/// stack tracker bounds per-row stack use and code size; larger requests are
/// rejected rather than silently truncated.
pub const MAX_PARTITION_FANOUT: usize = 16;

type TopK = [(u32, f32); MAX_PARTITION_FANOUT];

/// One row-major point-by-leader dot-product tile and its normalization terms.
///
/// The scale slices are deliberately metric-specific:
///
/// | metric | `row_scales` | `leader_scales` |
/// |---|---|---|
/// | [`Metric::L2`] | empty | squared leader norms |
/// | [`Metric::Cosine`] | squared point norms | leader norms |
/// | [`Metric::CosineNormalized`] / [`Metric::InnerProduct`] | empty | empty |
///
/// [`nearest_leaders`] validates every declared shape before dispatch.
#[derive(Clone, Copy, Debug)]
pub struct PartitionTopK<'a> {
    /// Row-major `rows * leaders` point-to-leader dot products.
    pub dots: &'a [f32],
    /// Number of points represented by `dots`.
    pub rows: usize,
    /// Number of leaders represented by each row.
    pub leaders: usize,
    /// Metric-specific point normalization terms described in the type table.
    pub row_scales: &'a [f32],
    /// Metric-specific leader normalization terms described in the type table.
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
    #[error(
        "invalid fanout {fanout}: must not exceed {leaders} leaders or kernel maximum {maximum}"
    )]
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
        .position(|leaders| leaders[fanout - 1] == u32::MAX)
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
    fn run_simd<F>(self, arch: F::Arch)
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
        u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    {
        process_rows_simd::<F>(arch, self.input, self.fanout, self.output);
    }
}

impl<A> diskann_wide::arch::Target<A, ()> for PartitionKernel<'_, '_>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    #[inline(always)]
    fn run(self, arch: A) {
        self.run_simd::<A::f32x16>(arch);
    }
}

#[cfg(test)]
fn process_rows_scalar(input: PartitionTopK<'_>, fanout: usize, output: &mut [u32]) {
    for (row_index, (dot_row, output_row)) in input
        .dots
        .chunks_exact(input.leaders)
        .zip(output.chunks_exact_mut(fanout))
        .enumerate()
    {
        let mut top = [(u32::MAX, f32::INFINITY); MAX_PARTITION_FANOUT];
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

fn process_rows_simd<F>(arch: F::Arch, input: PartitionTopK<'_>, fanout: usize, output: &mut [u32])
where
    F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    match input.metric {
        Metric::L2 => process_rows(input, fanout, output, |_, dot_row, top| {
            process_binary::<F, _, _>(
                arch,
                dot_row,
                input.leader_scales,
                top,
                fanout,
                |dot, norm| F::splat(arch, -2.0).mul_add_simd(dot, norm),
                |dot, norm| norm - 2.0 * dot,
            );
        }),
        Metric::CosineNormalized => process_rows(input, fanout, output, |_, dot_row, top| {
            process_unary::<F, _>(arch, dot_row, top, fanout, |dot| F::splat(arch, 1.0) - dot);
        }),
        Metric::InnerProduct => process_rows(input, fanout, output, |_, dot_row, top| {
            process_unary::<F, _>(arch, dot_row, top, fanout, |dot| F::default(arch) - dot);
        }),
        Metric::Cosine => process_rows(input, fanout, output, |row, dot_row, top| {
            process_cosine::<F>(
                arch,
                dot_row,
                input.row_scales[row],
                input.leader_scales,
                top,
                fanout,
            );
        }),
    }
}

#[inline(always)]
fn process_rows(
    input: PartitionTopK<'_>,
    fanout: usize,
    output: &mut [u32],
    mut process: impl FnMut(usize, &[f32], &mut TopK),
) {
    for (row, (dot_row, output_row)) in input
        .dots
        .chunks_exact(input.leaders)
        .zip(output.chunks_exact_mut(fanout))
        .enumerate()
    {
        let mut top = [(u32::MAX, f32::INFINITY); MAX_PARTITION_FANOUT];
        process(row, dot_row, &mut top);
        copy_ids(&top, output_row);
    }
}

#[inline(always)]
fn cosine_distance(row_norm_squared: f32, leader_norm: f32, dot: f32) -> f32 {
    let row_norm = if row_norm_squared < f32::MIN_POSITIVE {
        0.0
    } else {
        row_norm_squared.sqrt()
    };
    let leader_norm = if leader_norm < f32::MIN_POSITIVE.sqrt() {
        0.0
    } else {
        leader_norm
    };
    if row_norm == 0.0 || leader_norm == 0.0 {
        1.0
    } else {
        1.0 - dot / (row_norm * leader_norm)
    }
}

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
    let row_norm = if row_norm_squared < f32::MIN_POSITIVE {
        0.0
    } else {
        row_norm_squared.sqrt()
    };
    let row_norm = F::splat(arch, row_norm);
    let one = F::splat(arch, 1.0);
    let minimum_norm = F::splat(arch, f32::MIN_POSITIVE.sqrt());
    process_binary::<F, _, _>(
        arch,
        dots,
        leader_norms,
        top,
        fanout,
        |dot, leader_norm| {
            let row_zero = row_norm.lt_simd(minimum_norm);
            let leader_zero = leader_norm.lt_simd(minimum_norm);
            let denominator = row_norm * leader_norm;
            let safe_denominator = row_zero.select(one, leader_zero.select(one, denominator));
            let cosine = row_zero.select(
                F::default(arch),
                leader_zero.select(F::default(arch), dot / safe_denominator),
            );
            one - cosine
        },
        |dot, leader_norm| cosine_distance(row_norm_squared, leader_norm, dot),
    );
}

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
        let value = transform(F::splat(arch, dot)).to_array();
        insert_topk(top, fanout, (full + offset) as u32, value.as_ref()[0]);
    }
}

fn process_binary<F, Transform, ScalarTransform>(
    arch: F::Arch,
    dots: &[f32],
    scales: &[f32],
    top: &mut TopK,
    fanout: usize,
    transform: Transform,
    scalar_transform: ScalarTransform,
) where
    F: SIMDVector<Scalar = f32> + SIMDFloat,
    Transform: Fn(F, F) -> F,
    ScalarTransform: Fn(f32, f32) -> f32,
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
        let value = scalar_transform(dots[full + offset], scales[full + offset]);
        insert_topk(top, fanout, (full + offset) as u32, value);
    }
}

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

    let values = distances.to_array();
    let values = values.as_ref();
    let mut lanes = u64::from(eligible.bitmask().to_underlying());
    while lanes != 0 {
        let lane = lanes.trailing_zeros() as usize;
        lanes &= lanes - 1;
        insert_topk(top, fanout, (base + lane) as u32, values[lane]);
    }
}

#[inline(always)]
#[cfg(test)]
fn distance(metric: Metric, dot: f32, row_scale: f32, leader_scale: f32) -> f32 {
    match metric {
        Metric::L2 => (-2.0f32).mul_add(dot, leader_scale),
        Metric::CosineNormalized => 1.0 - dot,
        Metric::InnerProduct => -dot,
        Metric::Cosine => cosine_distance(row_scale, leader_scale, dot),
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

#[cfg(test)]
mod tests;
