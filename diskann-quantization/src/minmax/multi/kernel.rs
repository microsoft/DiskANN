/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{
    bits::{Representation, Unsigned},
    minmax::MinMaxMeta,
    multi_vector::{BoxErase, MatRef, MaxSimError},
};

/// Prepared MinMax MaxSim computation with independent per-call scratch.
pub trait MinMaxMaxSimKernel<const QUERY_BITS: usize, const DOC_BITS: usize>:
    Send + Sync + std::fmt::Debug
where
    Unsigned: Representation<QUERY_BITS> + Representation<DOC_BITS>,
{
    /// Number of query vectors whose distances are written by this kernel.
    fn nrows(&self) -> usize;

    /// Overwrite one distance per query row. Empty documents produce f32::MAX;
    /// nonempty zero-dimensional documents produce zero. NaN distances are ignored.
    ///
    /// # Errors
    ///
    /// Shape errors leave scores unchanged.
    fn compute_max_sim(
        &self,
        doc: MatRef<'_, MinMaxMeta<DOC_BITS>>,
        scores: &mut [f32],
    ) -> Result<(), MaxSimError>;
}

/// Choose how the factory packages a concrete prepared kernel.
pub trait MinMaxErase<const QUERY_BITS: usize, const DOC_BITS: usize>
where
    Unsigned: Representation<QUERY_BITS> + Representation<DOC_BITS>,
{
    type Output;
    fn erase<K: MinMaxMaxSimKernel<QUERY_BITS, DOC_BITS> + 'static>(
        self,
        kernel: K,
    ) -> Self::Output;
}

impl<const Q: usize, const D: usize> MinMaxErase<Q, D> for BoxErase
where
    Unsigned: Representation<Q> + Representation<D>,
{
    type Output = Box<dyn MinMaxMaxSimKernel<Q, D>>;
    fn erase<K: MinMaxMaxSimKernel<Q, D> + 'static>(self, kernel: K) -> Self::Output {
        Box::new(kernel)
    }
}
