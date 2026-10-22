/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Two-center assignment with SIMD lanes spanning points in blocks of sixteen.
//! These kernels use a different reduction order from coordinate-wise L2. Finite
//! cancellation is observable; an invalid score is marked for caller-owned fallback.

use diskann_utils::views::rowmajor;
use diskann_wide::{
    ARCH, Architecture, SIMDMulAdd, SIMDPartialOrd, SIMDSelect, SIMDVector, arch::Target3,
};
use thiserror::Error;

use super::{common::square_norm, lloyds::distances_in_place};
use crate::multi_vector::{BlockTransposed, BlockTransposedRef, Overflow};

pub const BLOCK_SIZE: usize = 16;
/// No valid decision: at least one computed score was NaN or infinite.
pub const INVALID: u32 = u32::MAX;

#[derive(Debug, Clone, Copy)]
pub enum Metric {
    SquaredL2,
    /// Center norm minus twice the dot product; omits the shared point norm.
    DotProduct,
}

#[derive(Debug, Error)]
pub enum PackError {
    #[error("two-center packing requires nonempty rows and a positive dimension")]
    Empty,
    #[error("row {row} has dimension {actual}; expected {expected}")]
    Dimension {
        row: usize,
        expected: usize,
        actual: usize,
    },
    #[error("row iterator did not produce its declared length of {expected}")]
    RowCount { expected: usize },
    #[error(transparent)]
    Overflow(#[from] Overflow),
}

#[derive(Debug, Error)]
#[error(
    "expected two {dim}-dimensional centers and {rows} assignments, got {centers} coordinates and {outputs} assignments"
)]
pub struct ShapeError {
    dim: usize,
    rows: usize,
    centers: usize,
    outputs: usize,
}

/// Pack borrowed rows directly, without an intermediate row-major copy. Only
/// the final block is padded; padding is zero and never counted as an input row.
///
/// # Errors
/// Rejects empty shapes, inconsistent rows, and padded allocation-size overflow.
pub fn pack_rows<'a>(
    mut rows: impl ExactSizeIterator<Item = &'a [f32]>,
    dim: usize,
) -> Result<BlockTransposed<f32, BLOCK_SIZE>, PackError> {
    let count = rows.len();
    if count == 0 || dim == 0 {
        return Err(PackError::Empty);
    }
    let mut packed = BlockTransposed::try_new(count, dim)?;
    for (block, out) in packed
        .as_mut_slice()
        .chunks_exact_mut(BLOCK_SIZE * dim)
        .enumerate()
    {
        let mut group = [&[][..]; BLOCK_SIZE];
        let valid = (count - block * BLOCK_SIZE).min(BLOCK_SIZE);
        for (lane, slot) in group[..valid].iter_mut().enumerate() {
            let row = rows.next().ok_or(PackError::RowCount { expected: count })?;
            if row.len() != dim {
                return Err(PackError::Dimension {
                    row: block * BLOCK_SIZE + lane,
                    expected: dim,
                    actual: row.len(),
                });
            }
            *slot = row;
        }
        for (d, column) in out.chunks_exact_mut(BLOCK_SIZE).enumerate() {
            for (value, row) in column.iter_mut().zip(&group[..valid]) {
                *value = row[d];
            }
        }
    }
    if rows.next().is_some() {
        return Err(PackError::RowCount { expected: count });
    }
    Ok(packed)
}

fn check_shape(
    points: BlockTransposedRef<'_, f32, BLOCK_SIZE>,
    centers: &[f32],
    outputs: usize,
) -> Result<(), ShapeError> {
    if points.nrows() == 0
        || points.ncols() == 0
        || points.ncols().checked_mul(2) != Some(centers.len())
        || outputs != points.nrows()
    {
        return Err(ShapeError {
            dim: points.ncols(),
            rows: points.nrows(),
            centers: centers.len(),
            outputs,
        });
    }
    Ok(())
}

/// Write 0 or 1 for each real row, preferring 0 on equal finite scores (including
/// signed zero). A nonfinite score writes `INVALID`. No allocation or N-by-2
/// score matrix is needed. The caller defines fallback and centroid updates.
///
/// # Errors
/// Rejects inconsistent/empty shapes before accessing output or SIMD memory.
pub fn assign(
    points: BlockTransposedRef<'_, f32, BLOCK_SIZE>,
    centers: &[f32],
    metric: Metric,
    assignments: &mut [u32],
) -> Result<(), ShapeError> {
    check_shape(points, centers, assignments.len())?;
    let norms = match metric {
        Metric::SquaredL2 => [0.0; 2],
        Metric::DotProduct => [
            square_norm(&centers[..points.ncols()]),
            square_norm(&centers[points.ncols()..]),
        ],
    };
    ARCH.run3(Kernel { metric, norms }, points, centers, assignments);
    Ok(())
}

#[derive(Clone, Copy)]
struct Kernel {
    metric: Metric,
    norms: [f32; 2],
}

impl<A: Architecture> Target3<A, (), BlockTransposedRef<'_, f32, BLOCK_SIZE>, &[f32], &mut [u32]>
    for Kernel
{
    #[inline]
    fn run(
        self,
        arch: A,
        points: BlockTransposedRef<'_, f32, BLOCK_SIZE>,
        centers: &[f32],
        out: &mut [u32],
    ) {
        match self.metric {
            Metric::SquaredL2 => run::<A, false>(arch, points, centers, self.norms, out),
            Metric::DotProduct => run::<A, true>(arch, points, centers, self.norms, out),
        }
    }
}

#[inline]
fn run<A: Architecture, const DOT: bool>(
    arch: A,
    points: BlockTransposedRef<'_, f32, BLOCK_SIZE>,
    centers: &[f32],
    norms: [f32; 2],
    out: &mut [u32],
) {
    diskann_wide::alias!(F<A> = f32x8);
    diskann_wide::alias!(I<A> = i32x8);
    let dim = points.ncols();
    for (block, output) in out.chunks_mut(BLOCK_SIZE).enumerate() {
        let mut s00 = F::<A>::default(arch);
        let mut s01 = s00;
        let mut s10 = s00;
        let mut s11 = s00;
        // SAFETY: Shape checked at entry; block covers real rows or a padded final block.
        let source = unsafe { points.block_ptr_unchecked(block) };
        for d in 0..dim {
            // SAFETY: Every physical column contains sixteen accessible, initialized lanes.
            let lo = unsafe { F::<A>::load_simd(arch, source.add(BLOCK_SIZE * d)) };
            // SAFETY: The second half is within the same sixteen-lane column.
            let hi = unsafe { F::<A>::load_simd(arch, source.add(BLOCK_SIZE * d + 8)) };
            let c0 = F::<A>::splat(arch, centers[d]);
            let c1 = F::<A>::splat(arch, centers[dim + d]);
            if DOT {
                s00 = c0.mul_add_simd(lo, s00);
                s01 = c0.mul_add_simd(hi, s01);
                s10 = c1.mul_add_simd(lo, s10);
                s11 = c1.mul_add_simd(hi, s11);
            } else {
                let a = lo - c0;
                let b = hi - c0;
                let c = lo - c1;
                let e = hi - c1;
                s00 = a.mul_add_simd(a, s00);
                s01 = b.mul_add_simd(b, s01);
                s10 = c.mul_add_simd(c, s10);
                s11 = e.mul_add_simd(e, s11);
            }
        }
        if DOT {
            let n0 = F::<A>::splat(arch, norms[0]);
            let n1 = F::<A>::splat(arch, norms[1]);
            // Match the existing Lloyd norm-compensation order, without the point norm.
            s00 = n0 - s00 - s00;
            s01 = n0 - s01 - s01;
            s10 = n1 - s10 - s10;
            s11 = n1 - s11 - s11;
        }
        let select = |first: F<A>, second: F<A>| {
            let infinity = F::<A>::splat(arch, f32::INFINITY);
            let negative_infinity = F::<A>::splat(arch, f32::NEG_INFINITY);
            let mut selected = <I<A> as SIMDVector>::Mask::from(second.lt_simd(first))
                .select(I::<A>::splat(arch, 1), I::<A>::splat(arch, 0));
            for finite in [
                first.lt_simd(infinity),
                first.gt_simd(negative_infinity),
                second.lt_simd(infinity),
                second.gt_simd(negative_infinity),
            ] {
                selected = <I<A> as SIMDVector>::Mask::from(finite)
                    .select(selected, I::<A>::splat(arch, -1));
            }
            selected
        };
        let lo = output.len().min(8);
        // SAFETY: Only real output rows are written. i32/u32 have the same layout;
        // -1 writes the INVALID bit pattern without reading through the cast pointer.
        unsafe { select(s00, s10).store_simd_first(output.as_mut_ptr().cast(), lo) };
        if output.len() > 8 {
            // SAFETY: The second half exists and the masked length fits within output.
            unsafe {
                select(s01, s11)
                    .store_simd_first(output.as_mut_ptr().add(8).cast(), output.len() - 8)
            };
        }
    }
}

/// Controlled reference to the existing generic Lloyd assignment kernel. Unlike
/// `assign`, it computes point norms and a residual. Invalid output is an error;
/// callers must not use this reference for unrestricted production assignment.
pub struct LloydReference {
    points: BlockTransposed<f32, BLOCK_SIZE>,
    norms: Vec<f32>,
}

#[derive(Debug, Error)]
pub enum ReferenceError {
    #[error(transparent)]
    Shape(#[from] ShapeError),
    #[error("Lloyd reference produced a nonfinite/negative residual or invalid center ID")]
    InvalidOutput,
}

impl LloydReference {
    /// # Errors
    /// Returns the same shape and overflow errors as `pack_rows`.
    pub fn new<'a>(
        rows: impl ExactSizeIterator<Item = &'a [f32]> + Clone,
        dim: usize,
    ) -> Result<Self, PackError> {
        let points = pack_rows(rows.clone(), dim)?;
        let norms = rows.map(square_norm).collect();
        Ok(Self { points, norms })
    }

    pub fn packed_bytes(&self) -> usize {
        std::mem::size_of_val(self.points.as_slice())
    }
    pub fn scratch_bytes(&self) -> usize {
        std::mem::size_of_val(self.norms.as_slice())
    }

    /// # Errors
    /// Rejects inconsistent shapes and an invalid residual/assignment from the reference.
    pub fn assign(&self, centers: &[f32], out: &mut [u32]) -> Result<(), ReferenceError> {
        let points = self.points.as_view();
        check_shape(points, centers, out.len())?;
        let center_norms = [
            square_norm(&centers[..points.ncols()]),
            square_norm(&centers[points.ncols()..]),
        ];
        let centers =
            rowmajor::Ref::try_from_data(centers, 2, points.ncols()).map_err(|_| ShapeError {
                dim: points.ncols(),
                rows: points.nrows(),
                centers: centers.len(),
                outputs: out.len(),
            })?;
        let residual = distances_in_place(points, &self.norms, centers, &center_norms, out);
        if !residual.is_finite() || residual < 0.0 || out.iter().any(|&id| id > 1) {
            return Err(ReferenceError::InvalidOutput);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
