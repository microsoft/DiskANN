/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{
    bits::{Representation, Unsigned},
    minmax::MinMaxMeta,
    multi_vector::{BoxErase, MatRef, MaxSimError},
};

/// A prepared MinMax query producing one negated maximum inner product per query row.
/// Query storage is owned by the kernel; documents and output remain borrowed.
pub trait MinMaxMaxSimKernel<const B_BITS: usize>: Send + Sync + std::fmt::Debug
where
    Unsigned: Representation<B_BITS>,
{
    fn nrows(&self) -> usize;

    /// Overwrite scores, ignoring NaN inner products. Empty documents and query rows
    /// with only NaN or negative-infinite inner products produce `f32::MAX`.
    /// Nonempty zero-dimensional documents produce zero.
    ///
    /// # Errors
    ///
    /// Returns [`MaxSimError`] for a wrong output length or document dimension,
    /// without modifying the output.
    fn compute_max_sim(
        &self,
        document: MatRef<'_, MinMaxMeta<B_BITS>>,
        scores: &mut [f32],
    ) -> Result<(), MaxSimError>;
}

/// Customize factory output while the kernel's concrete type is still known.
pub trait MinMaxErase<const B_BITS: usize>
where
    Unsigned: Representation<B_BITS>,
{
    type Output;
    fn erase<K: MinMaxMaxSimKernel<B_BITS> + 'static>(self, kernel: K) -> Self::Output;
}

impl<const B_BITS: usize> MinMaxErase<B_BITS> for BoxErase
where
    Unsigned: Representation<B_BITS>,
{
    type Output = Box<dyn MinMaxMaxSimKernel<B_BITS>>;

    fn erase<K: MinMaxMaxSimKernel<B_BITS> + 'static>(self, kernel: K) -> Self::Output {
        Box::new(kernel)
    }
}

pub(super) fn validate<const B_BITS: usize>(
    rows: usize,
    dim: usize,
    document: MatRef<'_, MinMaxMeta<B_BITS>>,
    scores: &[f32],
) -> Result<(), MaxSimError>
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
    Ok(())
}
