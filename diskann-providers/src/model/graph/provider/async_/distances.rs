/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Utilities for supporting full precision, quantized, and mixed distance computations.

/// A temporary adaptor to promote the error types for fallible distance functions into
/// panics until DiskANN gets proper support for such fallible functions.
#[derive(Debug, Clone)]
pub struct UnwrapErr<T, E>(T, std::marker::PhantomData<E>);

impl<T, E> UnwrapErr<T, E> {
    pub fn new(v: T) -> Self {
        Self(v, std::marker::PhantomData)
    }

    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T, E> std::ops::Deref for UnwrapErr<T, E> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T, E> std::ops::DerefMut for UnwrapErr<T, E> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<A, B, T, E> diskann_vector::DistanceFunction<A, B, f32> for UnwrapErr<T, E>
where
    T: diskann_vector::DistanceFunction<A, B, Result<f32, E>>,
    E: std::fmt::Debug,
{
    fn evaluate_similarity(&self, a: A, b: B) -> f32 {
        // Lint: We don't quite have full support for fallible distance functions.
        #[expect(clippy::unwrap_used)]
        self.0.evaluate_similarity(a, b).unwrap()
    }
}

impl<A, T, E> diskann_vector::PreprocessedDistanceFunction<A, f32> for UnwrapErr<T, E>
where
    T: diskann_vector::PreprocessedDistanceFunction<A, Result<f32, E>>,
    E: std::fmt::Debug,
{
    fn evaluate_similarity(&self, a: A) -> f32 {
        // Lint: We don't quite have full support for fallible distance functions.
        #[expect(clippy::unwrap_used)]
        self.0.evaluate_similarity(a).unwrap()
    }
}

pub mod pq {
    //! Support for hybrid data types for full-precision and PQ compressed vectors.

    use std::sync::Arc;

    use diskann::utils::VectorRepr;
    use diskann_utils::Reborrow;
    use diskann_vector::DistanceFunction;

    use crate::model::pq::{self, FixedChunkPQTable};

    /// An element for two-level datasets that is either full-precision or quantized.
    /// This allows a pruning strategy that combines full-precision and quantized distances.
    pub enum Hybrid<F, Q> {
        Full(F),
        Quant(Q),
    }

    impl<F, Q> Hybrid<F, Q> {
        pub fn is_full(&self) -> bool {
            matches!(self, Self::Full(_))
        }
    }

    impl<'short, F, Q> Reborrow<'short> for Hybrid<F, Q>
    where
        F: Reborrow<'short>,
        Q: Reborrow<'short>,
    {
        type Target = Hybrid<F::Target, Q::Target>;

        fn reborrow(&'short self) -> Self::Target {
            match self {
                Self::Full(v) => Hybrid::Full(v.reborrow()),
                Self::Quant(v) => Hybrid::Quant(v.reborrow()),
            }
        }
    }

    /// A distance computer that operates on `Hybrid`.
    pub struct HybridComputer<T>
    where
        T: VectorRepr,
    {
        quant: pq::distance::DistanceComputer<Arc<FixedChunkPQTable>>,
        full: T::Distance,
    }

    impl<T> HybridComputer<T>
    where
        T: VectorRepr,
    {
        pub fn new(
            quant: pq::distance::DistanceComputer<Arc<FixedChunkPQTable>>,
            full: T::Distance,
        ) -> Self {
            Self { quant, full }
        }
    }

    /// The implementation of `DistanceFunction` for the hybrid computer.
    impl<T> DistanceFunction<Hybrid<&[T], &[u8]>, Hybrid<&[T], &[u8]>, f32> for HybridComputer<T>
    where
        T: VectorRepr,
    {
        #[inline(always)]
        fn evaluate_similarity(&self, x: Hybrid<&[T], &[u8]>, y: Hybrid<&[T], &[u8]>) -> f32 {
            match x {
                Hybrid::Full(x) => match y {
                    Hybrid::Full(y) => self.full.evaluate_similarity(x, y),
                    Hybrid::Quant(y) => {
                        // SAFETY: This can only panic when T = `MinMaxElement` and the underlying slice is ill-defined.
                        // we are ok with panicking in distance functions for now.
                        #[allow(clippy::unwrap_used)]
                        self.quant.evaluate_similarity(&*T::as_f32(x).unwrap(), y)
                    }
                },
                Hybrid::Quant(x) => match y {
                    Hybrid::Full(y) => {
                        // SAFETY: This can only panic when T = `MinMaxElement` and the underlying slice is ill-defined.
                        // we are ok with panicking in distance functions for now.
                        #[allow(clippy::unwrap_used)]
                        self.quant.evaluate_similarity(&*T::as_f32(y).unwrap(), x)
                    }
                    Hybrid::Quant(y) => self.quant.evaluate_similarity(x, y),
                },
            }
        }
    }
}
