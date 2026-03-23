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

    use std::sync::Arc;

    use diskann::{
        graph::workingset::{self, map},
        utils::VectorRepr,
    };
    use diskann_utils::{Reborrow, future::AsyncFriendly};
    use diskann_vector::DistanceFunction;

    use crate::model::{
        graph::provider::async_::common::Unseeded,
        pq::{self, FixedChunkPQTable},
    };

    type InnerMap<F, Q> = workingset::Map<u32, Hybrid<Vec<F>, Vec<Q>>, Projection<F, Q>>;

    /// Projects owned `Hybrid<Vec<F>, Vec<Q>>` values stored in a [`workingset::Map`] into
    /// borrowed `Hybrid<&[F], &[Q]>` views for distance computation.
    pub struct Projection<F, Q> {
        _marker: std::marker::PhantomData<(F, Q)>,
    }

    impl<F, Q> map::Projection for Projection<F, Q>
    where
        F: AsyncFriendly,
        Q: AsyncFriendly,
    {
        type Element<'a> = Hybrid<&'a [F], &'a [Q]>;
        type ElementRef<'a> = Hybrid<&'a [F], &'a [Q]>;
    }

    impl<F, Q> map::Project<Projection<F, Q>> for Hybrid<Vec<F>, Vec<Q>>
    where
        F: AsyncFriendly,
        Q: AsyncFriendly,
    {
        fn project(&self) -> Hybrid<&[F], &[Q]> {
            self.reborrow()
        }
    }

    /// Newtype around [`workingset::Map`] for hybrid PQ pruning state.
    ///
    /// This wrapper exists to avoid the blanket [`workingset::Fill`] implementation on
    /// raw `Map`, allowing the hybrid accessor to provide a custom `Fill` that selectively
    /// fetches full-precision vectors for the closest candidates and PQ codes for the rest.
    pub struct HybridMap<F, Q>(InnerMap<F, Q>)
    where
        F: AsyncFriendly,
        Q: AsyncFriendly;

    /// The [`workingset::View`] for [`HybridMap`].
    pub type View<'a, F, Q> = map::View<'a, u32, Hybrid<Vec<F>, Vec<Q>>, Projection<F, Q>>;

    impl<F, Q> HybridMap<F, Q>
    where
        F: AsyncFriendly,
        Q: AsyncFriendly,
    {
        /// Create a new `HybridMap` with the given capacity and no overlay.
        pub fn with_capacity(capacity: usize) -> Self {
            Self(map::Builder::new(map::Capacity::Default).build(capacity))
        }

        /// Create a new `HybridMap` with the given capacity and a batch overlay.
        pub fn with_capacity_and(
            capacity: usize,
            overlay: map::Overlay<u32, Projection<F, Q>>,
        ) -> Self {
            Self(
                map::Builder::new(map::Capacity::Default)
                    .with_overlay(overlay)
                    .build(capacity),
            )
        }

        /// Borrow the underlying map.
        pub fn get(&self) -> &InnerMap<F, Q> {
            &self.0
        }

        /// Mutably borrow the underlying map.
        pub fn get_mut(&mut self) -> &mut InnerMap<F, Q> {
            &mut self.0
        }
    }

    impl<F, Q> workingset::AsWorkingSet<HybridMap<F, Q>> for map::Overlay<u32, Projection<F, Q>>
    where
        F: AsyncFriendly,
        Q: AsyncFriendly,
    {
        fn as_working_set(&self, capacity: usize) -> HybridMap<F, Q> {
            HybridMap::with_capacity_and(capacity, self.clone())
        }
    }

    impl<F, Q> workingset::AsWorkingSet<HybridMap<F, Q>> for Unseeded
    where
        F: AsyncFriendly,
        Q: AsyncFriendly,
    {
        fn as_working_set(&self, capacity: usize) -> HybridMap<F, Q> {
            HybridMap::with_capacity(capacity)
        }
    }

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
