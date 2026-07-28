/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Fused nearest-neighbor kernel for a leaf's lower dot-product matrix.

use diskann_vector::distance::Metric;
#[cfg(target_arch = "x86_64")]
use diskann_wide::{SIMDFloat, SIMDMask, SIMDSelect, SIMDVector};

/// Widest f32 SIMD lane count DiskANN dispatches to, used to size lane scratch.
#[cfg(target_arch = "x86_64")]
const MAX_LANES: usize = 16;

#[cfg(target_arch = "x86_64")]
const L2: u8 = 0;
#[cfg(target_arch = "x86_64")]
const COSINE_NORMALIZED: u8 = 1;
#[cfg(target_arch = "x86_64")]
const INNER_PRODUCT: u8 = 2;
#[cfg(target_arch = "x86_64")]
const COSINE: u8 = 3;

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
        Self::new(u32::MAX, f32::MAX)
    }
}

/// Lower-triangular dot products consumed by [`nearest_leaf_neighbors`].
#[derive(Clone, Copy, Debug)]
pub struct LeafTopK<'a> {
    /// Row-major `points * points` matrix. Only entries with `column <= row` are read.
    pub dots: &'a [f32],
    /// Number of points represented by the matrix.
    pub points: usize,
    /// Metric used to rank pairs.
    pub metric: Metric,
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

/// Validation or allocation error returned by [`nearest_leaf_neighbors`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LeafKernelError {
    /// The point count cannot be represented in leaf-local `u32` positions.
    #[error("point count {0} exceeds the u32 position limit")]
    TooManyPoints(usize),
    /// A declared shape overflowed `usize`.
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

/// Select the nearest non-self leaf positions for every row.
///
/// The strictly lower triangle is scanned once. Each pair updates both row
/// trackers, so the upper triangle is neither read nor materialized. The
/// returned value is `min(k, points - 1)`, and `output` contains exactly
/// `points * returned_k` entries grouped by row and ordered by ascending
/// distance. Equal distances retain pair scan order.
pub fn nearest_leaf_neighbors(
    input: LeafTopK<'_>,
    k: usize,
    output: &mut [LeafNeighbor],
    workspace: &mut LeafTopKWorkspace,
) -> Result<usize, LeafKernelError> {
    let actual_k = validate(input, k, output)?;
    if actual_k == 0 {
        return Ok(0);
    }

    resize("norms", &mut workspace.norms, input.points, 0.0)?;
    resize(
        "worst distances",
        &mut workspace.worst,
        input.points,
        f32::MAX,
    )?;
    for (row, norm) in workspace.norms.iter_mut().enumerate() {
        let squared_norm = input.dots[row * input.points + row];
        *norm = if input.metric == Metric::Cosine {
            // Match diskann-vector: a finite/subnormal squared norm below this
            // threshold is a zero vector, while NaN continues through the
            // distance calculation as non-rankable.
            if squared_norm < f32::MIN_POSITIVE {
                0.0
            } else {
                squared_norm.sqrt()
            }
        } else {
            squared_norm
        };
    }
    output.fill(LeafNeighbor::default());
    workspace.worst.fill(f32::MAX);

    diskann_wide::arch::dispatch(LeafKernel {
        input,
        k: actual_k,
        output,
        norms: &workspace.norms,
        worst: &mut workspace.worst,
    });
    if let Some(row) = output
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

fn validate(
    input: LeafTopK<'_>,
    k: usize,
    output: &[LeafNeighbor],
) -> Result<usize, LeafKernelError> {
    if input.points > u32::MAX as usize {
        return Err(LeafKernelError::TooManyPoints(input.points));
    }
    let matrix_len = checked_area("lower dot-product matrix", input.points, input.points)?;
    check_length("lower dot-product matrix", input.dots.len(), matrix_len)?;
    let actual_k = k.min(input.points.saturating_sub(1));
    let output_len = checked_area("output", input.points, actual_k)?;
    check_length("output", output.len(), output_len)?;
    Ok(actual_k)
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

struct LeafKernel<'a, 'o, 'w> {
    input: LeafTopK<'a>,
    k: usize,
    output: &'o mut [LeafNeighbor],
    norms: &'w [f32],
    worst: &'w mut [f32],
}

impl LeafKernel<'_, '_, '_> {
    fn run_scalar(self) {
        process_pairs_scalar(self.input, self.k, self.output, self.norms, self.worst);
    }

    #[cfg(target_arch = "x86_64")]
    fn run_simd<F>(self, arch: F::Arch)
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
        u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    {
        if self.k > 3 {
            process_pairs_simd_dynamic::<F>(
                arch,
                self.input,
                self.k,
                self.output,
                self.norms,
                self.worst,
            );
            return;
        }
        match self.k {
            1 => self.run_fused::<F, 1>(arch),
            2 => self.run_fused::<F, 2>(arch),
            3 => self.run_fused::<F, 3>(arch),
            _ => unreachable!("validated non-zero leaf width"),
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn run_fused<F, const SLOTS: usize>(self, arch: F::Arch)
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
        u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    {
        match self.input.metric {
            Metric::L2 => process_pairs_simd_fused::<F, L2, SLOTS>(
                arch,
                self.input,
                self.output,
                self.norms,
                self.worst,
            ),
            Metric::CosineNormalized => process_pairs_simd_fused::<F, COSINE_NORMALIZED, SLOTS>(
                arch,
                self.input,
                self.output,
                self.norms,
                self.worst,
            ),
            Metric::InnerProduct => process_pairs_simd_fused::<F, INNER_PRODUCT, SLOTS>(
                arch,
                self.input,
                self.output,
                self.norms,
                self.worst,
            ),
            Metric::Cosine => process_pairs_simd_fused::<F, COSINE, SLOTS>(
                arch,
                self.input,
                self.output,
                self.norms,
                self.worst,
            ),
        }
    }
}

impl diskann_wide::arch::Target<diskann_wide::arch::Scalar, ()> for LeafKernel<'_, '_, '_> {
    #[inline(always)]
    fn run(self, _: diskann_wide::arch::Scalar) {
        self.run_scalar();
    }
}

#[cfg(target_arch = "x86_64")]
impl diskann_wide::arch::Target<diskann_wide::arch::x86_64::V3, ()> for LeafKernel<'_, '_, '_> {
    #[inline(always)]
    fn run(self, arch: diskann_wide::arch::x86_64::V3) {
        diskann_wide::alias!(F32x8 = <diskann_wide::arch::x86_64::V3>::f32x8);
        self.run_simd::<F32x8>(arch);
    }
}

#[cfg(target_arch = "x86_64")]
impl diskann_wide::arch::Target<diskann_wide::arch::x86_64::V4, ()> for LeafKernel<'_, '_, '_> {
    #[inline(always)]
    fn run(self, arch: diskann_wide::arch::x86_64::V4) {
        diskann_wide::alias!(F32x16 = <diskann_wide::arch::x86_64::V4>::f32x16);
        self.run_simd::<F32x16>(arch);
    }
}

#[cfg(target_arch = "aarch64")]
impl diskann_wide::arch::Target<diskann_wide::arch::aarch64::Neon, ()> for LeafKernel<'_, '_, '_> {
    #[inline(always)]
    fn run(self, arch: diskann_wide::arch::aarch64::Neon) {
        let _scalar = arch.retarget();
        self.run_scalar();
    }
}

fn process_pairs_scalar(
    input: LeafTopK<'_>,
    k: usize,
    output: &mut [LeafNeighbor],
    norms: &[f32],
    worst: &mut [f32],
) {
    for row in 1..input.points {
        for column in 0..row {
            let dot = input.dots[row * input.points + column];
            let distance = pair_distance(input.metric, dot, norms[row], norms[column]);
            insert_row(output, worst, k, row, column as u32, distance);
            insert_row(output, worst, k, column, row as u32, distance);
        }
    }
}

#[cfg(target_arch = "x86_64")]
/// Fused dual-endpoint scan for row widths without a specialized arm.
///
/// Identical structure to [`process_pairs_simd_fused`], with the slot count
/// read at run time. Wider leaves are rare, so the extra indirection is
/// cheaper than instantiating an arm per width.
fn process_pairs_simd_dynamic<F>(
    arch: F::Arch,
    input: LeafTopK<'_>,
    k: usize,
    output: &mut [LeafNeighbor],
    norms: &[f32],
    worst: &mut [f32],
) where
    F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let output_ptr = output.as_mut_ptr();
    let worst_ptr = worst.as_mut_ptr();
    for row in 1..input.points {
        let row_start = row * input.points;
        let row_norm = F::splat(arch, norms[row]);
        // SAFETY: `row < input.points == worst.len()`.
        let mut row_worst = unsafe { *worst_ptr.add(row) };
        let mut column = 0;
        while column + F::LANES <= row {
            // SAFETY: the full chunk is contained in the strict lower row prefix.
            let dots = unsafe { F::load_simd(arch, input.dots.as_ptr().add(row_start + column)) };
            // SAFETY: `column + F::LANES <= row < input.points == norms.len()`.
            let column_norms = unsafe { F::load_simd(arch, norms.as_ptr().add(column)) };
            let distances = pair_distances::<F>(arch, input.metric, dots, row_norm, column_norms);
            let row_eligible = distances.lt_simd(F::splat(arch, row_worst));
            // SAFETY: the full chunk lies below `row`, so it is within `worst`.
            let column_worst = unsafe { F::load_simd(arch, worst_ptr.add(column)) };
            let column_eligible = distances.lt_simd(column_worst);
            let row_bits = u64::from(row_eligible.bitmask().to_underlying());
            let column_bits = u64::from(column_eligible.bitmask().to_underlying());
            if row_bits | column_bits != 0 {
                let mut values = [0.0f32; MAX_LANES];
                // SAFETY: the array covers every f32 SIMD width DiskANN exposes.
                unsafe { distances.store_simd(values.as_mut_ptr()) };
                let mut row_bits = row_bits;
                while row_bits != 0 {
                    let lane = row_bits.trailing_zeros() as usize;
                    row_bits &= row_bits - 1;
                    let distance = values[lane];
                    if distance < row_worst {
                        // SAFETY: `row * k + k` is inside the validated output.
                        row_worst = unsafe {
                            insert_slots(output_ptr, row * k, k, (column + lane) as u32, distance)
                        };
                    }
                }
                let mut column_bits = column_bits;
                while column_bits != 0 {
                    let lane = column_bits.trailing_zeros() as usize;
                    column_bits &= column_bits - 1;
                    let target = column + lane;
                    // SAFETY: `target < row`, so its slots are inside the output.
                    let new_worst = unsafe {
                        insert_slots(output_ptr, target * k, k, row as u32, values[lane])
                    };
                    // SAFETY: `target < row < worst.len()`.
                    unsafe { *worst_ptr.add(target) = new_worst };
                }
            }
            column += F::LANES;
        }
        while column < row {
            // SAFETY: the scalar tail remains in the strict lower triangle.
            let dot = unsafe { *input.dots.get_unchecked(row_start + column) };
            // SAFETY: `column < row < input.points == norms.len()`.
            let column_norm = unsafe { *norms.get_unchecked(column) };
            let distance = pair_distance(input.metric, dot, norms[row], column_norm);
            if distance < row_worst {
                // SAFETY: `row * k + k` is inside the validated output.
                row_worst =
                    unsafe { insert_slots(output_ptr, row * k, k, column as u32, distance) };
            }
            // SAFETY: `column < row < worst.len()`.
            let column_worst = unsafe { *worst_ptr.add(column) };
            if distance < column_worst {
                // SAFETY: `column < row`, so its slots are inside the output.
                let new_worst =
                    unsafe { insert_slots(output_ptr, column * k, k, row as u32, distance) };
                // SAFETY: `column < row < worst.len()`.
                unsafe { *worst_ptr.add(column) = new_worst };
            }
            column += 1;
        }
        // SAFETY: `row < worst.len()`.
        unsafe { *worst_ptr.add(row) = row_worst };
    }
}

/// Fused dual-endpoint scan of the strict lower triangle.
///
/// The row's current worst distance stays in a register for the whole row, and
/// each chunk derives both endpoint candidate masks before touching memory, so
/// a chunk where neither endpoint can accept costs one branch. `SLOTS` is the
/// per-row neighbor count, threaded as a const so the insert arm is selected at
/// compile time.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
fn process_pairs_simd_fused<F, const METRIC: u8, const SLOTS: usize>(
    arch: F::Arch,
    input: LeafTopK<'_>,
    output: &mut [LeafNeighbor],
    norms: &[f32],
    worst: &mut [f32],
) where
    F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let output_ptr = output.as_mut_ptr();
    let worst_ptr = worst.as_mut_ptr();
    for row in 1..input.points {
        let row_start = row * input.points;
        let row_norm = F::splat(arch, norms[row]);
        // SAFETY: `row < input.points == worst.len()`.
        let mut row_worst = unsafe { *worst_ptr.add(row) };
        let mut column = 0;
        while column + F::LANES <= row {
            // SAFETY: the full chunks are inside the validated matrix and norms.
            let dots = unsafe { F::load_simd(arch, input.dots.as_ptr().add(row_start + column)) };
            // SAFETY: `column + F::LANES <= row < input.points == norms.len()`.
            let column_norms = unsafe { F::load_simd(arch, norms.as_ptr().add(column)) };
            let distances =
                pair_distances::<F>(arch, metric::<METRIC>(), dots, row_norm, column_norms);
            let row_eligible = distances.lt_simd(F::splat(arch, row_worst));
            // SAFETY: the full chunk lies below `row`, so it is within `worst`.
            let column_worst = unsafe { F::load_simd(arch, worst_ptr.add(column)) };
            let column_eligible = distances.lt_simd(column_worst);
            // Test both candidate masks with a single reduction. Reducing each
            // mask separately costs an extra cross-lane extraction per chunk,
            // and the overwhelmingly common case is that neither end accepts.
            let row_bits = u64::from(row_eligible.bitmask().to_underlying());
            let column_bits = u64::from(column_eligible.bitmask().to_underlying());
            if row_bits | column_bits != 0 {
                let mut values = [0.0f32; MAX_LANES];
                // SAFETY: the array covers every f32 SIMD width DiskANN exposes.
                unsafe { distances.store_simd(values.as_mut_ptr()) };
                let mut row_bits = row_bits;
                while row_bits != 0 {
                    let lane = row_bits.trailing_zeros() as usize;
                    row_bits &= row_bits - 1;
                    let distance = values[lane];
                    // Earlier lanes in this chunk may already have tightened the
                    // threshold, so re-check against the live value.
                    if distance < row_worst {
                        // SAFETY: `row * SLOTS + SLOTS` is inside the validated output.
                        row_worst = unsafe {
                            insert_slots(
                                output_ptr,
                                row * SLOTS,
                                SLOTS,
                                (column + lane) as u32,
                                distance,
                            )
                        };
                    }
                }
                let mut column_bits = column_bits;
                while column_bits != 0 {
                    let lane = column_bits.trailing_zeros() as usize;
                    column_bits &= column_bits - 1;
                    let target = column + lane;
                    // SAFETY: `target < row`, so its slots are inside the output.
                    let new_worst = unsafe {
                        insert_slots(output_ptr, target * SLOTS, SLOTS, row as u32, values[lane])
                    };
                    // SAFETY: `target < row < worst.len()`.
                    unsafe { *worst_ptr.add(target) = new_worst };
                }
            }
            column += F::LANES;
        }
        while column < row {
            // SAFETY: the scalar tail remains in the strict lower triangle.
            let dot = unsafe { *input.dots.get_unchecked(row_start + column) };
            // SAFETY: `column < row < input.points == norms.len()`.
            let column_norm = unsafe { *norms.get_unchecked(column) };
            let distance = pair_distance(metric::<METRIC>(), dot, norms[row], column_norm);
            if distance < row_worst {
                // SAFETY: `row * SLOTS + SLOTS` is inside the validated output.
                row_worst = unsafe {
                    insert_slots(output_ptr, row * SLOTS, SLOTS, column as u32, distance)
                };
            }
            // SAFETY: `column < row < worst.len()`.
            let column_worst = unsafe { *worst_ptr.add(column) };
            if distance < column_worst {
                // SAFETY: `column < row`, so its slots are inside the output.
                let new_worst = unsafe {
                    insert_slots(output_ptr, column * SLOTS, SLOTS, row as u32, distance)
                };
                // SAFETY: `column < row < worst.len()`.
                unsafe { *worst_ptr.add(column) = new_worst };
            }
            column += 1;
        }
        // SAFETY: `row < worst.len()`.
        unsafe { *worst_ptr.add(row) = row_worst };
    }
}

#[cfg(target_arch = "x86_64")]
const fn metric<const METRIC: u8>() -> Metric {
    match METRIC {
        L2 => Metric::L2,
        COSINE_NORMALIZED => Metric::CosineNormalized,
        INNER_PRODUCT => Metric::InnerProduct,
        COSINE => Metric::Cosine,
        _ => unreachable!(),
    }
}

/// Insert one candidate into a row's ascending-distance slots and return the
/// row's new worst distance.
///
/// Slot counts of one, two, and three are the production leaf widths and get
/// straight-line arms. Wider rows fall back to a bubble-up over the same
/// layout, which produces identical results at a lower instruction count than
/// specializing further would justify.
///
/// # Safety
///
/// `base + slots` must be within the allocation behind `output`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn insert_slots(
    output: *mut LeafNeighbor,
    base: usize,
    slots: usize,
    position: u32,
    distance: f32,
) -> f32 {
    let entry = LeafNeighbor::new(position, distance);
    match slots {
        1 => {
            // SAFETY: the caller guarantees `base` is in bounds.
            unsafe { *output.add(base) = entry };
            distance
        }
        2 => {
            // SAFETY: the caller guarantees `base` and `base + 1` are in bounds.
            let first = unsafe { *output.add(base) };
            if distance < first.distance {
                // SAFETY: as above.
                unsafe {
                    *output.add(base) = entry;
                    *output.add(base + 1) = first;
                }
                first.distance
            } else {
                // SAFETY: as above.
                unsafe { *output.add(base + 1) = entry };
                distance
            }
        }
        3 => {
            // SAFETY: the caller guarantees `base..base + 3` is in bounds.
            let (first, second) = unsafe { (*output.add(base), *output.add(base + 1)) };
            if distance < first.distance {
                // SAFETY: as above.
                unsafe {
                    *output.add(base) = entry;
                    *output.add(base + 1) = first;
                    *output.add(base + 2) = second;
                }
            } else if distance < second.distance {
                // SAFETY: as above.
                unsafe {
                    *output.add(base + 1) = entry;
                    *output.add(base + 2) = second;
                }
            } else {
                // SAFETY: as above.
                unsafe { *output.add(base + 2) = entry };
                return distance;
            }
            second.distance
        }
        _ => {
            let last = base + slots - 1;
            // SAFETY: the caller guarantees `base..base + slots` is in bounds.
            unsafe { *output.add(last) = entry };
            let mut position = last;
            while position > base {
                // SAFETY: `base < position <= last` stays inside the row.
                let (current, previous) =
                    unsafe { (*output.add(position), *output.add(position - 1)) };
                if current.distance >= previous.distance {
                    break;
                }
                // SAFETY: as above.
                unsafe {
                    *output.add(position) = previous;
                    *output.add(position - 1) = current;
                }
                position -= 1;
            }
            // SAFETY: `last` is in bounds.
            unsafe { (*output.add(last)).distance }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn pair_distances<F>(arch: F::Arch, metric: Metric, dot: F, row_norm: F, column_norm: F) -> F
where
    F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
{
    let zero = F::default(arch);
    match metric {
        Metric::L2 => {
            let distance = row_norm + column_norm - F::splat(arch, 2.0) * dot;
            zero.max_simd(distance)
        }
        Metric::CosineNormalized => {
            let distance = F::splat(arch, 1.0) - dot;
            zero.max_simd(distance)
        }
        Metric::InnerProduct => zero - dot,
        Metric::Cosine => {
            let one = F::splat(arch, 1.0);
            let row_zero = row_norm.eq_simd(zero);
            let column_zero = column_norm.eq_simd(zero);
            let denominator = row_norm * column_norm;
            let safe_denominator = row_zero.select(one, column_zero.select(one, denominator));
            let cosine = row_zero.select(zero, column_zero.select(zero, dot / safe_denominator));
            let distance = one - cosine;
            // Comparisons with NaN are false, so this explicit lower clamp
            // preserves non-rankable NaNs while matching the existing PiPNN
            // distance formulas for finite values.
            zero.max_simd(distance)
        }
    }
}

#[inline(always)]
fn pair_distance(metric: Metric, dot: f32, row_norm: f32, column_norm: f32) -> f32 {
    match metric {
        Metric::L2 => {
            let distance = row_norm + column_norm - 2.0 * dot;
            if distance < 0.0 {
                0.0
            } else {
                distance
            }
        }
        Metric::CosineNormalized => {
            let distance = 1.0 - dot;
            if distance < 0.0 {
                0.0
            } else {
                distance
            }
        }
        Metric::InnerProduct => -dot,
        Metric::Cosine => {
            let denominator = row_norm * column_norm;
            let cosine = if row_norm != 0.0 && column_norm != 0.0 {
                dot / denominator
            } else {
                0.0
            };
            let distance = 1.0 - cosine;
            if distance < 0.0 {
                0.0
            } else {
                distance
            }
        }
    }
}

#[inline(always)]
fn insert_row(
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

    let start = row * k;
    let row_output = &mut output[start..start + k];
    row_output[k - 1] = LeafNeighbor::new(position, distance);
    let mut index = k - 1;
    while index > 0 && row_output[index].distance < row_output[index - 1].distance {
        row_output.swap(index, index - 1);
        index -= 1;
    }
    worst[row] = row_output[k - 1].distance;
}

#[cfg(test)]
mod tests;
