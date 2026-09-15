// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Object-safe MinMax kernel boundary and type-erasure visitor.

use crate::bits::{Representation, Unsigned};
use crate::minmax::MinMaxMeta;
use crate::multi_vector::{BoxErase, MatRef, MaxSimError};

/// Object-safe interface for MinMax-quantized MaxSim matrix kernels.
pub trait MinMaxMaxSimKernel<const QUERY_BITS: usize, const DOC_BITS: usize>:
    Send + Sync + std::fmt::Debug
where
    Unsigned: Representation<QUERY_BITS> + Representation<DOC_BITS>,
{
    /// Number of query vectors whose scores this kernel produces.
    fn nrows(&self) -> usize;

    /// Compute one MaxSim distance for every query vector.
    ///
    /// Overwrites all scores on each call. Empty documents produce [`f32::MAX`];
    /// nonempty documents with zero-dimensional vectors produce zero.
    /// NaN pairwise distances are ignored, matching [`f32::min`].
    ///
    /// # Errors
    ///
    /// * [`MaxSimError::InvalidBufferLength`] if `scores.len() != self.nrows()`.
    /// * [`MaxSimError::UnequalDim`] if the document dimension does not match the query.
    fn compute_max_sim(
        &self,
        doc: MatRef<'_, MinMaxMeta<DOC_BITS>>,
        scores: &mut [f32],
    ) -> Result<(), MaxSimError>;
}

/// Type-erasure visitor for MinMax matrix kernels.
pub trait MinMaxErase<const QUERY_BITS: usize, const DOC_BITS: usize>
where
    Unsigned: Representation<QUERY_BITS> + Representation<DOC_BITS>,
{
    /// Erased kernel type.
    type Output;

    /// Package a concrete MinMax matrix kernel.
    fn erase<K>(self, kernel: K) -> Self::Output
    where
        K: MinMaxMaxSimKernel<QUERY_BITS, DOC_BITS> + 'static;
}

impl<const QUERY_BITS: usize, const DOC_BITS: usize> MinMaxErase<QUERY_BITS, DOC_BITS> for BoxErase
where
    Unsigned: Representation<QUERY_BITS> + Representation<DOC_BITS>,
{
    type Output = Box<dyn MinMaxMaxSimKernel<QUERY_BITS, DOC_BITS>>;

    fn erase<K>(self, kernel: K) -> Self::Output
    where
        K: MinMaxMaxSimKernel<QUERY_BITS, DOC_BITS> + 'static,
    {
        Box::new(kernel)
    }
}
