// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Object-safe kernel boundary trait plus BYOTE visitor trait.

use crate::multi_vector::{MatRef, Standard};

/// Object-safe interface for computing per-query MaxSim scores.
///
/// # Contract
///
/// - `scores.len() == self.nrows()` (caller's precondition).
/// - The implementation must populate **all** `nrows()` entries of `scores`.
///   Callers that derive quantities from the full score vector (e.g. sums)
///   would silently corrupt their result if any trailing entry were left
///   unwritten.
pub trait MaxSimKernel<T: Copy>: Send + Sync + std::fmt::Debug {
    /// Number of query rows whose scores this kernel produces.
    fn nrows(&self) -> usize;

    /// Compute per-query MaxSim scores against `doc` into `scores`.
    fn compute_max_sim(&self, doc: MatRef<'_, Standard<T>>, scores: &mut [f32]);
}

/// "Bring your own type erasure" visitor. The factory hands an implementation
/// to `erase`, which decides how to package / type-erase it. Lets different
/// callers produce different output shapes (e.g. `Box<dyn MaxSimKernel<T>>`,
/// a chamfer-only closure, a batched evaluator, ...) from the same factory.
///
/// See [`BoxErase`] for the default impl used by most callers.
pub trait Erase<T: Copy> {
    /// What the visitor produces.
    type Output;
    /// Visit the concrete kernel. `K` is generic so the body sees its concrete
    /// type and the compiler can inline it into the wrapper.
    fn erase<K: MaxSimKernel<T> + 'static>(self, kernel: K) -> Self::Output;
}

/// Default [`Erase`] impl: produces `Box<dyn MaxSimKernel<T>>`.
///
/// Use this when the caller just wants a heap-allocated kernel object behind
/// a vtable. For custom packaging (chamfer-only, batched, composed), write
/// your own `Erase` impl and pass it to the factory in place of `BoxErase`.
#[derive(Debug, Clone, Copy)]
pub struct BoxErase;

impl<T: Copy + 'static> Erase<T> for BoxErase {
    type Output = Box<dyn MaxSimKernel<T>>;

    fn erase<K: MaxSimKernel<T> + 'static>(self, kernel: K) -> Self::Output {
        Box::new(kernel)
    }
}
