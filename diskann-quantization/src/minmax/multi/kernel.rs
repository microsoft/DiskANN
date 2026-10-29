/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::rowmajor;

use crate::{
    bits::{Representation, Unsigned},
    minmax::{MinMaxCompensation, MinMaxMeta},
    multi_vector::{MatRef, MaxSimError, Overflow},
};

/// An owned MinMax query with a backend selected at construction.
#[derive(Debug)]
pub struct MinMaxMaxSim<const B_BITS: usize>
where
    Unsigned: Representation<B_BITS>,
{
    pub(super) inner: Box<dyn MinMaxMaxSimKernel<B_BITS>>,
}

impl<const B_BITS: usize> MinMaxMaxSim<B_BITS>
where
    Unsigned: Representation<B_BITS>,
{
    pub fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    /// Overwrite scores, ignoring NaN inner products. Empty documents and query rows
    /// with only NaN or negative-infinite inner products produce f32::MAX.
    /// Nonempty zero-dimensional documents produce zero.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong output length or document dimension,
    /// without modifying the output.
    pub fn compute_max_sim(
        &self,
        document: MatRef<'_, MinMaxMeta<B_BITS>>,
        scores: &mut [f32],
    ) -> Result<(), MaxSimError> {
        self.inner.compute_max_sim(document, scores)
    }
}

// Closed implementation boundary: callers never instantiate the matrix backends.
pub(super) trait MinMaxMaxSimKernel<const B_BITS: usize>:
    Send + Sync + std::fmt::Debug + 'static
where
    Unsigned: Representation<B_BITS>,
{
    fn nrows(&self) -> usize;
    fn compute_max_sim(
        &self,
        document: MatRef<'_, MinMaxMeta<B_BITS>>,
        scores: &mut [f32],
    ) -> Result<(), MaxSimError>;

    #[cfg(test)]
    fn type_id(&self) -> std::any::TypeId {
        std::any::TypeId::of::<Self>()
    }
}

/// Validate shape and handle empty inputs; return whether scoring is still needed.
pub(super) fn validate<const B_BITS: usize>(
    rows: usize,
    dim: usize,
    document: MatRef<'_, MinMaxMeta<B_BITS>>,
    scores: &mut [f32],
) -> Result<bool, MaxSimError>
where
    Unsigned: Representation<B_BITS>,
{
    if scores.len() != rows {
        return Err(MaxSimError::InvalidBufferLength(scores.len(), rows));
    }
    if document.repr().intrinsic_dim() != dim {
        return Err(MaxSimError::UnequalDim(
            document.repr().intrinsic_dim(),
            dim,
        ));
    }
    if rows == 0 {
        return Ok(false);
    }
    if document.num_vectors() == 0 {
        scores.fill(f32::MAX);
        return Ok(false);
    }
    if dim == 0 {
        scores.fill(0.0);
        return Ok(false);
    }
    Ok(true)
}

/// Borrow canonical MinMax storage as a checked row-major byte view.
pub(super) fn canonical_view<const NBITS: usize>(
    matrix: MatRef<'_, MinMaxMeta<NBITS>>,
) -> Result<rowmajor::Ref<'_, u8>, Overflow>
where
    Unsigned: Representation<NBITS>,
{
    let rows = matrix.num_vectors();
    let dim = matrix.repr().intrinsic_dim();
    let overflow = || Overflow::for_type::<u8>(rows, dim);
    let stride = dim
        .checked_mul(NBITS)
        .ok_or_else(overflow)?
        .div_ceil(8)
        .checked_add(size_of::<MinMaxCompensation>())
        .ok_or_else(overflow)?;
    Overflow::check::<u8>(rows, stride)?;
    // SAFETY: MatRef supplies the canonical initialized span. Arithmetic is checked before
    // reading the pointer, including legacy shapes that could wrap in release builds.
    let bytes = unsafe { std::slice::from_raw_parts(matrix.as_raw_ptr(), rows * stride) };
    #[expect(
        clippy::expect_used,
        reason = "the checked dimensions match the slice length"
    )]
    Ok(rowmajor::Ref::try_from_data(bytes, rows, stride).expect("validated canonical rows"))
}
