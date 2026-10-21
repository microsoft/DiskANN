/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Support for [`diskann_quantization::scalar`].

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
use diskann_vector::{Norm, norm::FastL2Norm};
use thiserror::Error;

use crate::{
    num::Bytes,
    repr::{internal::Calf, quantized},
};

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

    /// Return the configured metric.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Return the number of bits used for each dimension.
    pub fn nbits(&self) -> usize {
        N
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
        InvalidDim::check(self.dim(), data.len())?;

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

    fn query_distance<'a, V>(
        &'a self,
        query: &'a [f32],
        visitor: V,
        kind: QueryDistanceKind,
    ) -> ANNResult<V::Output>
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

        InvalidDim::check(self.dim(), query.len())?;

        let mut q = Vector::new_boxed(self.dim());

        // TODO: `diskann-quantization` needs to expose an asymmetric distance API.
        //
        // We really shouldn't have to handle this externally.
        let query = (|| match kind {
            QueryDistanceKind::Query => {
                if matches!(self.metric, Metric::InnerProduct)
                    && let Some(mean_norm) = self.quantizer.mean_norm()
                {
                    let norm = (FastL2Norm).evaluate(query);
                    if norm != 0.0 {
                        let scale = mean_norm / norm;
                        let rescaled = query.iter().map(|q| q * scale).collect();
                        return Calf::Owned(rescaled);
                    }
                }
                Calf::Borrowed(query)
            }
            QueryDistanceKind::Insert => Calf::Borrowed(query),
        })();

        self.quantizer
            .compress_into(&*query, q.reborrow_mut())
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

#[derive(Debug, Clone, Copy)]
enum QueryDistanceKind {
    Insert,
    Query,
}

/// Metrics supported by the [`Scalar`] quantizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    SquaredL2,
    InnerProduct,
    CosineNormalized,
}

impl Metric {
    fn as_str(&self) -> &'static str {
        match self {
            Self::SquaredL2 => "squared-l2",
            Self::InnerProduct => "inner-product",
            Self::CosineNormalized => "cosine-normalized",
        }
    }
}

impl std::fmt::Display for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
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

impl From<Metric> for diskann_vector::distance::Metric {
    fn from(m: Metric) -> Self {
        match m {
            Metric::SquaredL2 => Self::L2,
            Metric::InnerProduct => Self::InnerProduct,
            Metric::CosineNormalized => Self::CosineNormalized,
        }
    }
}

/// Cosine is not supported.
#[derive(Debug, Error)]
#[error("Metric::Cosine is not supported for scalar quantization")]
#[non_exhaustive]
pub struct CosineNotSupported;

diskann::convert_error!(CosineNotSupported);

/// TODO: This works around `diskann-quantization` panicking instead of erroring.
#[derive(Debug, Error)]
#[error("scalar quantizer expected dim {} - instead got {}", expected, got)]
struct InvalidDim {
    expected: usize,
    got: usize,
}

impl InvalidDim {
    fn check(expected: usize, got: usize) -> Result<(), Self> {
        if expected != got {
            Err(Self { expected, got })
        } else {
            Ok(())
        }
    }
}

diskann::convert_error!(InvalidDim);

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
            fn query_distance<'a, V>(
                &'a self,
                query: &'a [f32],
                visitor: V,
            ) -> ANNResult<V::Output>
            where
                V: quantized::QueryDistanceVisitor<'a>,
            {
                <Scalar<$N>>::query_distance(self, query, visitor, QueryDistanceKind::Query)
            }
            fn insert_distance<'a, V>(
                &'a self,
                query: &'a [f32],
                visitor: V,
            ) -> ANNResult<V::Output>
            where
                V: quantized::QueryDistanceVisitor<'a>,
            {
                <Scalar<$N>>::query_distance(self, query, visitor, QueryDistanceKind::Insert)
            }
        }
    };
    ($($Ns:literal),+ $(,)?) => {
        $(quantizer!($Ns);)+
    };
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

    use diskann_utils::assert_contains;
    use rand::{
        SeedableRng,
        distr::{Distribution, Uniform},
        rngs::StdRng,
    };

    use crate::repr::quantized::test::DynQuantizer;

    const DIM: usize = 5;
    const TRIALS: usize = 10;
    const SCALE: f32 = 10.0;

    #[derive(Debug, Clone, Copy)]
    struct Bits(usize);

    impl Bits {
        fn value(&self, bucket: usize) -> f32 {
            assert!(bucket < self.buckets());
            self.step() * (bucket as f32) - self.shift()
        }

        fn buckets(&self) -> usize {
            2usize.pow(self.0.try_into().unwrap())
        }

        fn shift(&self) -> f32 {
            self.half_step() * ((self.buckets() - 1) as f32)
        }

        fn half_step(&self) -> f32 {
            self.step() / 2.0
        }

        fn step(&self) -> f32 {
            1.0
        }
    }

    /// For the most part, we rely on the integration tests to make sure that distances
    /// actually make sense.
    ///
    /// These tests ensure that the [`Distance`] and [`QueryDistance`] implementations agree
    /// and that errors are thrown as needed.
    fn test_scalar(
        quantizer: &dyn DynQuantizer,
        bits: usize,
        metric: Metric,
        rng: &mut StdRng,
        ntrials: usize,
        ctx: &dyn std::fmt::Display,
    ) {
        let bits = Bits(bits);
        let buckets = bits.buckets();
        let dist = Uniform::new(0usize, buckets).unwrap();

        let dim = quantizer.dim();
        let bytes = quantizer.bytes().value();

        let mut x = vec![0.0f32; dim];
        let mut bx = vec![0u8; bytes];

        let mut y = vec![0.0f32; dim];
        let mut by = vec![0u8; bytes];

        let mut x_scaled = vec![0.0f32; dim];

        for trial in 0..ntrials {
            // One the first trial, use `0` for `x` to make sure we don't try to scale a
            // zeroed vector.
            if trial == 0 {
                x.fill(0.0);
            } else {
                x.iter_mut().for_each(|x| *x = bits.value(dist.sample(rng)));
            }
            y.iter_mut().for_each(|y| *y = bits.value(dist.sample(rng)));

            quantizer.quantize(&mut bx, &x).unwrap();
            quantizer.quantize(&mut by, &y).unwrap();

            // Check distance
            let distance = quantizer.distance();
            let from_distance = distance.eval(&bx, &by).unwrap();

            let insert_distance = quantizer.insert_distance(&x).unwrap();
            let from_insert_distance = insert_distance.eval(&by).unwrap();

            assert_eq!(
                from_distance, from_insert_distance,
                "x = {:?}, bx = {:?}, y = {:?}, by = {:?}, trial = {} -- {}",
                x, bx, y, by, trial, ctx,
            );

            // Check scaling.
            //
            // This mimics the scaling logic.
            match metric {
                Metric::InnerProduct => {
                    let norm = (FastL2Norm).evaluate(&*x);
                    let norm = if norm == 0.0 { 1.0 } else { norm };
                    let scale = SCALE / norm;
                    std::iter::zip(x_scaled.iter_mut(), x.iter())
                        .for_each(|(xs, x)| *xs = (*x) * scale)
                }
                Metric::SquaredL2 | Metric::CosineNormalized => x_scaled.copy_from_slice(&x),
            };

            let insert_distance = quantizer.insert_distance(&x_scaled).unwrap();
            let from_insert_distance = insert_distance.eval(&by).unwrap();

            let query_distance = quantizer.query_distance(&x).unwrap();
            let from_query_distance = query_distance.eval(&by).unwrap();

            assert_eq!(
                from_insert_distance, from_query_distance,
                "x = {:?}, bx = {:?}, x_scaled = {:?}, y = {:?}, by = {:?}, trial = {} -- {}",
                x, bx, x_scaled, y, by, trial, ctx,
            );
        }

        // Compression Errors
        {
            // Input size
            let x = vec![0.0f32; dim - 1];
            let mut bx = vec![0u8; bytes];
            let err = quantizer.quantize(&mut bx, &x).unwrap_err().to_string();
            assert_contains!(err, "expected dim 5");

            let x = vec![0.0f32; dim + 1];
            let err = quantizer.quantize(&mut bx, &x).unwrap_err().to_string();
            assert_contains!(err, "expected dim 5");

            // Output size
            let x = vec![0.0f32; dim];
            let mut bx = vec![0u8; bytes - 1];
            assert!(quantizer.quantize(&mut bx, &x).is_err());

            let mut bx = vec![0u8; bytes + 1];
            assert!(quantizer.quantize(&mut bx, &x).is_err());

            // NaN
            let x = vec![f32::NAN; dim];
            let mut bx = vec![0u8; bytes];
            let err = quantizer.quantize(&mut bx, &x).unwrap_err().to_string();
            assert_contains!(err, "NaN");
        }

        // Distance
        {
            let d = quantizer.distance();
            let bx = vec![0u8; bytes];
            let by = vec![0u8; bytes - 1];
            let bz = vec![0u8; bytes + 1];

            assert!(d.eval(&bx, &by).is_err());
            assert!(d.eval(&bx, &bz).is_err());

            assert!(d.eval(&by, &bx).is_err());
            assert!(d.eval(&bz, &bx).is_err());

            assert!(d.eval(&bx, &bx).is_ok());
        }

        // Query Distance
        {
            let x = vec![0.0f32; dim - 1];
            let err = quantizer.query_distance(&x).unwrap_err().to_string();
            assert_contains!(err, "expected dim 5");

            let x = vec![0.0f32; dim + 1];
            let err = quantizer.query_distance(&x).unwrap_err().to_string();
            assert_contains!(err, "expected dim 5");

            let x = vec![f32::NAN; dim];
            let err = quantizer.query_distance(&x).unwrap_err().to_string();
            assert_contains!(err, "NaN");

            let x = vec![0.0f32; dim];
            let d = quantizer.query_distance(&x).unwrap();

            let b = vec![0u8; bytes - 1];
            assert!(d.eval(&b).is_err());

            let b = vec![0u8; bytes + 1];
            assert!(d.eval(&b).is_err());

            let b = vec![0u8; bytes];
            assert!(d.eval(&b).is_ok());
        }

        // Insert Distance
        {
            let x = vec![0.0f32; dim - 1];
            let err = quantizer.insert_distance(&x).unwrap_err().to_string();
            assert_contains!(err, "expected dim 5");

            let x = vec![0.0f32; dim + 1];
            let err = quantizer.insert_distance(&x).unwrap_err().to_string();
            assert_contains!(err, "expected dim 5");

            let x = vec![f32::NAN; dim];
            let err = quantizer.insert_distance(&x).unwrap_err().to_string();
            assert_contains!(err, "NaN");

            let x = vec![0.0f32; dim];
            let d = quantizer.insert_distance(&x).unwrap();

            let b = vec![0u8; bytes - 1];
            assert!(d.eval(&b).is_err());

            let b = vec![0u8; bytes + 1];
            assert!(d.eval(&b).is_err());

            let b = vec![0u8; bytes];
            assert!(d.eval(&b).is_ok());
        }
    }

    /// This quantizer is set up to encode data in the range `[-1, 1)`.
    ///
    /// As such, its shift is centered around 1.0 and it can encode data at a granularity of
    /// `1/2^n` where `n` is the number of bits.
    fn test_quantizer(bits: Bits) -> scalar::ScalarQuantizer {
        scalar::ScalarQuantizer::new(
            (bits.buckets() as f32) / 2.0,
            vec![-bits.shift(); DIM],
            Some(SCALE),
        )
    }

    fn run_test<const N: usize>()
    where
        Scalar<N>: quantized::Quantizer,
    {
        let bits = Bits(N);
        let mut rng = StdRng::seed_from_u64(0xcbd17799cff736a7);

        let quantizer = Scalar::<N>::new(test_quantizer(bits), Metric::SquaredL2);
        assert_eq!(quantizer.nbits(), N);
        assert_eq!(quantizer.metric(), Metric::SquaredL2);

        test_scalar(
            &quantizer,
            quantizer.nbits(),
            quantizer.metric(),
            &mut rng,
            TRIALS,
            &"squared-l2",
        );

        let quantizer = Scalar::<N>::new(test_quantizer(bits), Metric::InnerProduct);
        assert_eq!(quantizer.nbits(), N);
        assert_eq!(quantizer.metric(), Metric::InnerProduct);

        test_scalar(
            &quantizer,
            quantizer.nbits(),
            quantizer.metric(),
            &mut rng,
            TRIALS,
            &"inner-product",
        );

        let quantizer = Scalar::<N>::new(test_quantizer(bits), Metric::CosineNormalized);
        assert_eq!(quantizer.nbits(), N);
        assert_eq!(quantizer.metric(), Metric::CosineNormalized);

        test_scalar(
            &quantizer,
            quantizer.nbits(),
            quantizer.metric(),
            &mut rng,
            TRIALS,
            &"cosine-normalized",
        );
    }

    #[test]
    fn test_one_bit() {
        run_test::<1>();
    }

    #[test]
    fn test_two_bit() {
        run_test::<2>();
    }

    #[test]
    fn test_three_bit() {
        run_test::<3>();
    }

    #[test]
    fn test_four_bit() {
        run_test::<4>();
    }

    #[test]
    fn test_five_bit() {
        run_test::<5>();
    }

    #[test]
    fn test_six_bit() {
        run_test::<6>();
    }

    #[test]
    fn test_seven_bit() {
        run_test::<7>();
    }

    #[test]
    fn test_eight_bit() {
        run_test::<8>();
    }

    #[test]
    fn test_no_scaling() {
        let bits = Bits(8);

        let quantizer = Scalar::<8>::new(
            scalar::ScalarQuantizer::new(
                (bits.buckets() as f32) / 2.0,
                vec![-bits.shift(); DIM],
                None,
            ),
            Metric::InnerProduct,
        );

        let dyn_quantizer: &dyn DynQuantizer = &quantizer;

        let dim = dyn_quantizer.dim();
        let bytes = dyn_quantizer.bytes().value();

        let x = vec![-1.0, -0.5, -0.5, 0.5, 1.0];
        let y = vec![-1.0f32; dim];
        let mut by = vec![0u8; bytes];

        dyn_quantizer.quantize(&mut by, &y).unwrap();

        let insert_distance = dyn_quantizer.insert_distance(&x).unwrap();
        let from_insert_distance = insert_distance.eval(&by).unwrap();

        let query_distance = dyn_quantizer.query_distance(&x).unwrap();
        let from_query_distance = query_distance.eval(&by).unwrap();

        assert_eq!(from_insert_distance, from_query_distance);
    }

    //--------//
    // Metric //
    //--------//

    #[test]
    fn test_metric() {
        use diskann_vector::distance::Metric as VMetric;

        // VMetric -> Metric
        assert_eq!(Metric::try_from(VMetric::L2).unwrap(), Metric::SquaredL2);
        assert_eq!(
            Metric::try_from(VMetric::InnerProduct).unwrap(),
            Metric::InnerProduct
        );
        assert_eq!(
            Metric::try_from(VMetric::CosineNormalized).unwrap(),
            Metric::CosineNormalized
        );
        let err = Metric::try_from(VMetric::Cosine).unwrap_err();
        assert_contains!(
            err.to_string(),
            "Metric::Cosine is not supported for scalar quantization"
        );

        // Metric -> VMetric
        assert_eq!(VMetric::from(Metric::SquaredL2), VMetric::L2);
        assert_eq!(VMetric::from(Metric::InnerProduct), VMetric::InnerProduct);
        assert_eq!(
            VMetric::from(Metric::CosineNormalized),
            VMetric::CosineNormalized
        );
    }
}
