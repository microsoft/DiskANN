// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Object-safe kernel trait and BYOTE visitor.

use crate::multi_vector::{MatRef, Standard};

/// Object-safe interface for computing per-query MaxSim scores.
pub trait MaxSimKernel<T: Copy>: Send + Sync + std::fmt::Debug {
    /// Number of query rows whose scores this kernel produces.
    fn nrows(&self) -> usize;

    /// Compute per-query MaxSim scores against `doc` into `scores`.
    ///
    /// `scores` must have length `self.nrows()`. Every entry will be written;
    /// callers may rely on the full buffer being populated.
    ///
    /// # Panics
    ///
    /// Panics if `scores.len() != self.nrows()`.
    fn compute_max_sim(&self, doc: MatRef<'_, Standard<T>>, scores: &mut [f32]);
}

/// BYOTE visitor: the factory hands a concrete kernel to [`erase`](Self::erase),
/// which packages it however the caller needs. `K` is generic so the body
/// sees the concrete type and can inline.
pub trait Erase<T: Copy> {
    type Output;
    fn erase<K: MaxSimKernel<T> + 'static>(self, kernel: K) -> Self::Output;
}

/// Default [`Erase`] impl — produces `Box<dyn MaxSimKernel<T>>`.
#[derive(Debug, Clone, Copy)]
pub struct BoxErase;

impl<T: Copy + 'static> Erase<T> for BoxErase {
    type Output = Box<dyn MaxSimKernel<T>>;

    fn erase<K: MaxSimKernel<T> + 'static>(self, kernel: K) -> Self::Output {
        Box::new(kernel)
    }
}
