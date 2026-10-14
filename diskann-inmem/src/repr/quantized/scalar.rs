/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{ANNError, ANNResult};
use diskann_quantization::{
    CompressInto,
    bits::{Representation, Unsigned},
    distances as quant_distances,
    scalar::{
        self, CompensatedCosineNormalized as CCosineNormalized, CompensatedIP as CInnerProduct,
        CompensatedSquaredL2 as CSquaredL2,
    },
};
use thiserror::Error;

use crate::{num::Bytes, repr::quantized};

// Help bring the ridiculous names under control.
type VectorRef<'a, const N: usize> = scalar::CompensatedVectorRef<'a, N>;
type VectorMut<'a, const N: usize> = scalar::MutCompensatedVectorRef<'a, N>;
type Vector<const N: usize> = scalar::CompensatedVector<N>;

/// A scalar quantizer specialized for `N` bit computation.
#[derive(Debug)]
pub struct Scalar<const N: usize> {
    quantizer: scalar::ScalarQuantizer,
    metric: Metric,
}

impl<const N: usize> Scalar<N> {
    /// Construct a new [`Scalar`] quantizer.
    pub fn new(quantizer: scalar::ScalarQuantizer, metric: Metric) -> Self {
        Self { quantizer, metric }
    }

    // The following methods are used to implement `Quantizer` in their full generic glory.

    fn bytes(&self) -> Bytes
    where
        Unsigned: Representation<N>,
    {
        Bytes::new(Vector::<N>::canonical_bytes(self.dim()))
    }

    fn dim(&self) -> usize {
        self.quantizer.dim()
    }

    fn quantize(&self, quantized: &mut [u8], data: &[f32]) -> ANNResult<()>
    where
        Unsigned: Representation<N>,
        scalar::ScalarQuantizer: for<'a> CompressInto<&'a [f32], VectorMut<'a, N>>,
    {
        let quantized =
            VectorMut::from_canonical_back_mut(quantized, self.dim()).map_err(ANNError::new)?;
        self.quantizer
            .compress_into(data, quantized)
            .map_err(ANNError::new)?;
        Ok(())
    }

    fn distance<'a, V>(&'a self, visitor: V) -> V::Output
    where
        V: quantized::DistanceVisitor<'a>,
        Distance<N, CSquaredL2>: quantized::Distance,
        Distance<N, CInnerProduct>: quantized::Distance,
        Distance<N, CCosineNormalized>: quantized::Distance,
    {
        use diskann_quantization::AsFunctor;

        let dim = self.dim();
        match self.metric {
            Metric::SquaredL2 => visitor.visit(Distance::<N, CSquaredL2>::new(
                self.quantizer.as_functor(),
                dim,
            )),
            Metric::InnerProduct => visitor.visit(Distance::<N, CInnerProduct>::new(
                self.quantizer.as_functor(),
                dim,
            )),
            Metric::CosineNormalized => visitor.visit(Distance::<N, CCosineNormalized>::new(
                self.quantizer.as_functor(),
                dim,
            )),
        }
    }

    fn query_distance<'a, V>(&'a self, query: &'a [f32], visitor: V) -> ANNResult<V::Output>
    where
        V: quantized::QueryDistanceVisitor<'a>,
        Unsigned: Representation<N>,
        scalar::ScalarQuantizer: for<'b> CompressInto<&'b [f32], VectorMut<'b, N>>,
        QueryDistance<N, CSquaredL2>: quantized::QueryDistance,
        QueryDistance<N, CInnerProduct>: quantized::QueryDistance,
        QueryDistance<N, CCosineNormalized>: quantized::QueryDistance,
    {
        use diskann_quantization::AsFunctor;
        use diskann_utils::ReborrowMut;

        let mut q = Vector::new_boxed(self.dim());

        self.quantizer
            .compress_into(query, q.reborrow_mut())
            .map_err(ANNError::new)?;

        let visited = match self.metric {
            Metric::SquaredL2 => visitor.visit(QueryDistance::<N, CSquaredL2>::new(
                q,
                self.quantizer.as_functor(),
            )),
            Metric::InnerProduct => visitor.visit(QueryDistance::<N, CInnerProduct>::new(
                q,
                self.quantizer.as_functor(),
            )),
            Metric::CosineNormalized => visitor.visit(QueryDistance::<N, CCosineNormalized>::new(
                q,
                self.quantizer.as_functor(),
            )),
        };

        Ok(visited)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    SquaredL2,
    InnerProduct,
    CosineNormalized,
}

impl TryFrom<diskann_vector::distance::Metric> for Metric {
    type Error = CosineNotSupported;
    fn try_from(m: diskann_vector::distance::Metric) -> Result<Self, Self::Error> {
        use diskann_vector::distance::Metric as VMetric;

        match m {
            VMetric::L2 => Ok(Self::SquaredL2),
            VMetric::InnerProduct => Ok(Self::InnerProduct),
            VMetric::CosineNormalized => Ok(Self::CosineNormalized),
            VMetric::Cosine => Err(CosineNotSupported),
        }
    }
}

#[derive(Debug, Error)]
#[error("Metric::Cosine is not supported")]
#[non_exhaustive]
pub struct CosineNotSupported;

diskann::convert_error!(CosineNotSupported);

/// A macro is used to stamp out implementations instead of the crazy generic bounds that
/// we *could* use because it's (1) easier to see what exactly is implemented via rustdoc
/// and (2) gives us a bit of flexibility to changes details on the backend.
macro_rules! quantizer {
    ($N:literal) => {
        impl quantized::Quantizer for Scalar<$N> {
            fn bytes(&self) -> Bytes {
                <Scalar<$N>>::bytes(self)
            }
            fn dim(&self) -> usize {
                <Scalar<$N>>::dim(self)
            }
            fn quantize(&self, quantized: &mut [u8], data: &[f32]) -> ANNResult<()> {
                <Scalar<$N>>::quantize(self, quantized, data)
            }
            fn distance<'a, V>(&'a self, visitor: V) -> V::Output
            where
                V: quantized::DistanceVisitor<'a>,
            {
                <Scalar<$N>>::distance(self, visitor)
            }
            fn query_distance<'a, V>(&'a self, query: &'a [f32], visitor: V) -> ANNResult<V::Output>
            where
                V: quantized::QueryDistanceVisitor<'a>,
            {
                <Scalar<$N>>::query_distance(self, query, visitor)
            }
        }
    };
    ($($Ns:literal),+ $(,)?) => {
        $(quantizer!($Ns);)+
    }
}

quantizer!(1, 2, 3, 4, 5, 6, 7, 8);

//----------------//
// Query Distance //
//----------------//

#[derive(Debug)]
struct QueryDistance<const N: usize, T>
where
    Unsigned: Representation<N>,
{
    query: Vector<N>,
    inner: T,
}

impl<const N: usize, T> QueryDistance<N, T>
where
    Unsigned: Representation<N>,
{
    fn new(query: Vector<N>, inner: T) -> Self {
        Self { query, inner }
    }
}

impl<const N: usize, T> quantized::QueryDistance for QueryDistance<N, T>
where
    Unsigned: Representation<N>,
    T: for<'a> diskann_vector::DistanceFunction<
            VectorRef<'a, N>,
            VectorRef<'a, N>,
            quant_distances::Result<f32>,
        > + Send
        + Sync
        + std::fmt::Debug,
{
    #[inline]
    fn eval(&self, x: &[u8]) -> ANNResult<f32> {
        use diskann_utils::Reborrow;

        let x = VectorRef::<N>::from_canonical_back(x, self.query.len()).map_err(ANNError::new)?;
        self.inner
            .evaluate_similarity(self.query.reborrow(), x)
            .map_err(ANNError::new)
    }
}

//----------//
// Distance //
//----------//

#[derive(Debug)]
struct Distance<const N: usize, T> {
    inner: T,
    dim: usize,
}

impl<const N: usize, T> Distance<N, T> {
    fn new(inner: T, dim: usize) -> Self {
        Self { inner, dim }
    }
}

impl<const N: usize, T> quantized::Distance for Distance<N, T>
where
    Unsigned: Representation<N>,
    T: for<'a> diskann_vector::DistanceFunction<
            VectorRef<'a, N>,
            VectorRef<'a, N>,
            quant_distances::Result<f32>,
        > + Send
        + Sync
        + std::fmt::Debug,
{
    #[inline]
    fn eval(&self, x: &[u8], y: &[u8]) -> ANNResult<f32> {
        let x = VectorRef::<N>::from_canonical_back(x, self.dim).map_err(ANNError::new)?;
        let y = VectorRef::<N>::from_canonical_back(y, self.dim).map_err(ANNError::new)?;
        self.inner.evaluate_similarity(x, y).map_err(ANNError::new)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;


}
