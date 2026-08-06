// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Object-safe kernel boundary trait plus BYOTE visitor trait.

use crate::multi_vector::{MatRef, MaxSimError, RowMajor};

/// Object-safe interface for computing per-query MaxSim scores.
pub trait MaxSimKernel<T: Copy>: Send + Sync + std::fmt::Debug {
    /// Number of query rows whose scores this kernel produces.
    fn nrows(&self) -> usize;

    /// Compute per-query MaxSim scores into `scores`. On zero docs, fills
    /// every slot with `f32::MAX`.
    ///
    /// # Errors
    ///
    /// [`MaxSimError::InvalidBufferLength`] if `scores.len() != self.nrows()`.
    fn compute_max_sim(
        &self,
        doc: MatRef<'_, RowMajor<T>>,
        scores: &mut [f32],
    ) -> Result<(), MaxSimError>;
}

/// "Bring your own type erasure" visitor: the factory hands a concrete
/// kernel to [`Erase::erase`], which decides how to package it (e.g. as
/// `Box<dyn MaxSimKernel<T>>` via [`BoxErase`], a chamfer-only closure, a
/// batched evaluator, …).
pub trait Erase<T: Copy> {
    type Output;
    /// `K` is generic so the body sees its concrete type and the compiler
    /// can inline it.
    fn erase<K: MaxSimKernel<T> + 'static>(self, kernel: K) -> Self::Output;
}

/// Default boxing [`Erase`] impl.
#[derive(Debug, Clone, Copy)]
pub struct BoxErase;

impl<T: Copy + 'static> Erase<T> for BoxErase {
    type Output = Box<dyn MaxSimKernel<T>>;

    fn erase<K: MaxSimKernel<T> + 'static>(self, kernel: K) -> Self::Output {
        Box::new(kernel)
    }
}
