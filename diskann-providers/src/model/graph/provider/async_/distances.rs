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
    //!
    //! During hybrid PQ pruning, each candidate is represented as either a full-precision
    //! vector or a PQ-compressed code. The [`Hybrid`] enum captures this duality, and the
    //! remaining types adapt it to the [`workingset`](diskann::graph::workingset) framework.

    use diskann::utils::VectorRepr;
    use diskann_utils::Reborrow;
    use diskann_vector::DistanceFunction;

    use crate::model::pq;

    /// An element that is either a full-precision vector or a PQ-compressed code.
    ///
    /// During hybrid pruning, the closest candidates receive full-precision vectors for
    /// accurate distance computation, while the remaining candidates use cheaper PQ codes.
    /// The [`HybridComputer`] dispatches to the appropriate distance function based on
    /// which variant each operand is.
    pub enum Hybrid<F, Q> {
        Full(F),
        Quant(Q),
    }

    impl<F, Q> Hybrid<F, Q> {
        pub fn is_full(&self) -> bool {
            matches!(self, Self::Full(_))
        }
    }

    // NOTE: This definition always maps slices to the full-precision type and is used
    // for zero-copy multi-insert compatibility.
    impl<'a, F, Q> From<&'a [F]> for Hybrid<&'a [F], &'a [Q]> {
        fn from(slice: &'a [F]) -> Self {
            Self::Full(slice)
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

    /// Distance computer that handles mixed full-precision and PQ-compressed operands.
    ///
    /// When both operands are full-precision, the native distance function is used. When
    /// at least one is quantized, the PQ distance table is used instead. Mixed pairs
    /// (full vs quant) convert the full-precision side to `f32` for the PQ lookup.
    pub struct HybridComputer<'a, T>
    where
        T: VectorRepr,
    {
        quant: pq::distance::DistanceComputer<'a>,
        full: T::Distance,
    }

    impl<'a, T> HybridComputer<'a, T>
    where
        T: VectorRepr,
    {
        pub fn new(quant: pq::distance::DistanceComputer<'a>, full: T::Distance) -> Self {
            Self { quant, full }
        }
    }

    /// The implementation of `DistanceFunction` for the hybrid computer.
    impl<T> DistanceFunction<Hybrid<&[T], &[u8]>, Hybrid<&[T], &[u8]>, f32> for HybridComputer<'_, T>
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

    #[cfg(test)]
    mod tests {
        use approx::assert_relative_eq;
        use diskann::utils::VectorRepr;
        use diskann_vector::{
            DistanceFunction, PureDistanceFunction,
            distance::{Metric, SquaredL2},
        };

        use super::{Hybrid, HybridComputer};
        use crate::model::pq::{FixedChunkPQTable, distance::DistanceComputer};

        #[test]
        fn hybrid_cosine_normalized_pq_pairs_use_scaled_l2() {
            let table = FixedChunkPQTable::new(
                4,
                vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0].into(),
                vec![0, 2, 4].into(),
            )
            .unwrap();
            let computer = HybridComputer::<f32>::new(
                DistanceComputer::new(&table, Metric::CosineNormalized),
                f32::distance(Metric::CosineNormalized, Some(4)),
            );
            let full = [1.0, 0.0, 0.0, 0.0];
            let code0 = [0, 1];
            let code1 = [1, 0];
            let reconstructed0 = table.inflate_vector(&code0);
            let reconstructed1 = table.inflate_vector(&code1);

            let full_quant = computer.evaluate_similarity(
                Hybrid::Full(full.as_slice()),
                Hybrid::Quant(code0.as_slice()),
            );
            let squared_l2: f32 = SquaredL2::evaluate(full.as_slice(), reconstructed0.as_slice());
            let expected_full_quant = 0.5 * squared_l2;
            assert_relative_eq!(full_quant, expected_full_quant, max_relative = 1.0e-7);

            let quant_quant = computer.evaluate_similarity(
                Hybrid::Quant(code0.as_slice()),
                Hybrid::Quant(code1.as_slice()),
            );
            let squared_l2: f32 =
                SquaredL2::evaluate(reconstructed0.as_slice(), reconstructed1.as_slice());
            let expected_quant_quant = 0.5 * squared_l2;
            assert_relative_eq!(quant_quant, expected_quant_quant, max_relative = 1.0e-7);
        }
    }
}
