/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{ANNError, ANNResult};
use diskann_quantization::{
    alloc::{Allocator, GlobalAllocator, ScopedAllocator},
    spherical::{InputHints, iface},
};

use crate::{num::Bytes, repr::quantized};

/// A new-type wrapper around [`iface::Quantizer`] for implementing [`quantized::Quantizer`].
#[derive(Debug)]
pub struct Spherical<T> {
    inner: T,
    data_hints: InputHints,
    query_hints: InputHints,
}

impl<T> Spherical<T> {
    /// Construct a new [`Spherical`] new-type wrapper around `quantizer`.
    pub fn new(quantizer: T) -> Self {
        Self {
            inner: quantizer,
            data_hints: InputHints::new(),
            query_hints: InputHints::new().with_maybe_out_of_distribution(true),
        }
    }

    /// Configure properties of stored data and queries independently. Callers must ensure
    /// normalized hints remain valid after any conversion into the f32 input representation.
    pub fn with_input_hints(mut self, data: InputHints, query: InputHints) -> Self {
        self.data_hints = data;
        self.query_hints = query;
        self
    }

    /// Return the wrapped quantizer.
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T> std::ops::Deref for Spherical<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.inner
    }
}

impl<T> std::ops::DerefMut for Spherical<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

impl<T> quantized::Quantizer for Spherical<T>
where
    T: iface::Quantizer + 'static,
{
    fn bytes(&self) -> Bytes {
        Bytes::new(iface::Quantizer::bytes(&**self))
    }

    fn dim(&self) -> usize {
        iface::Quantizer::full_dim(&**self)
    }

    fn quantize(&self, quantized: &mut [u8], data: &[f32]) -> ANNResult<()> {
        iface::Quantizer::compress_with_hints(
            &**self,
            data,
            self.data_hints,
            iface::OpaqueMut::new(quantized),
            ScopedAllocator::global(),
        )
        .map_err(ANNError::new)
    }

    fn distance<'a, V>(&'a self, visitor: V) -> V::Output
    where
        V: quantized::DistanceVisitor<'a>,
    {
        visitor.visit(iface::Quantizer::distance_computer_ref(&**self))
    }

    fn insert_distance<'a, V>(&'a self, query: &'a [f32], visitor: V) -> ANNResult<V::Output>
    where
        V: quantized::QueryDistanceVisitor<'a>,
    {
        let query_computer = iface::Quantizer::fused_query_computer_with_hints(
            &**self,
            query,
            iface::QueryLayout::SameAsData,
            self.data_hints.with_maybe_out_of_distribution(false),
            GlobalAllocator,
            ScopedAllocator::global(),
        )
        .map_err(ANNError::new)?;

        Ok(visitor.visit(query_computer))
    }

    fn query_distance<'a, V>(&'a self, query: &'a [f32], visitor: V) -> ANNResult<V::Output>
    where
        V: quantized::QueryDistanceVisitor<'a>,
    {
        let query_computer = iface::Quantizer::fused_query_computer_with_hints(
            &**self,
            query,
            iface::QueryLayout::FullPrecision,
            self.query_hints,
            GlobalAllocator,
            ScopedAllocator::global(),
        )
        .map_err(ANNError::new)?;

        Ok(visitor.visit(query_computer))
    }
}

impl<A> quantized::QueryDistance for iface::QueryComputer<A>
where
    A: Allocator + std::fmt::Debug + Send + Sync,
{
    fn eval(&self, x: &[u8]) -> ANNResult<f32> {
        use diskann_vector::PreprocessedDistanceFunction;
        self.evaluate_similarity(iface::Opaque::new(x))
            .map_err(ANNError::new)
    }
}

impl quantized::Distance for &dyn iface::DynDistanceComputer {
    fn eval(&self, x: &[u8], y: &[u8]) -> ANNResult<f32> {
        self.evaluate(iface::Opaque::new(x), iface::Opaque::new(y))
            .map_err(ANNError::new)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann_quantization::spherical::SupportedMetric;
    use diskann_utils::{
        assert_contains,
        views::rowmajor::{self, Matrix, MatrixMut},
    };
    use diskann_vector::PreprocessedDistanceFunction;
    use rand::{
        SeedableRng,
        distr::{Distribution, Uniform},
        rngs::StdRng,
    };

    use crate::repr::quantized::test::DynQuantizer;

    const DIM: usize = 5;
    const TRIALS: usize = 10;

    #[test]
    fn normalized_data_with_independent_query_hints() {
        let data = rowmajor::Owned::from_fn(2, 4, |rc| if rc.row == rc.col { 1.0 } else { 0.0 });
        let normalized = InputHints::new().with_normalized(true);
        let make = || train_quantizer::<4>(data.as_view(), SupportedMetric::Cosine);
        let raw_query = Spherical::new(make()).with_input_hints(normalized, InputHints::new());
        let normalized_query = Spherical::new(make()).with_input_hints(normalized, normalized);
        let mut code = vec![0; raw_query.bytes().value()];
        raw_query
            .quantize(&mut code, &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        let raw = raw_query.query_distance(&[3.0, 4.0, 0.0, 0.0]).unwrap();
        let unit = normalized_query
            .query_distance(&[0.6, 0.8, 0.0, 0.0])
            .unwrap();
        let raw = raw.eval(&code).unwrap();
        assert!((raw - 0.4).abs() < 1e-4);
        assert!((raw - unit.eval(&code).unwrap()).abs() < 1e-4);
        let insert = normalized_query
            .insert_distance(&[0.6, 0.8, 0.0, 0.0])
            .unwrap();
        assert!((insert.eval(&code).unwrap() - 0.4).abs() < 0.1);
    }

    fn train_quantizer<const N: usize>(
        data: rowmajor::Ref<'_, f32>,
        metric: SupportedMetric,
    ) -> iface::Impl<N>
    where
        iface::Impl<N>: iface::Constructible + iface::Quantizer,
    {
        use diskann_quantization::{algorithms::transforms, spherical};

        spherical::SphericalQuantizer::train(
            data.as_view(),
            transforms::TransformKind::Null,
            metric,
            spherical::PreScale::None,
            &mut StdRng::seed_from_u64(0x019823745),
            GlobalAllocator,
        )
        .unwrap()
        .as_quantizer_impl::<N>()
        .unwrap()
    }

    /// Integration tests cover the correctness of the distances. Here, check that the
    /// adapter agrees with the spherical API and propagates errors.
    fn test_spherical(
        quantizer: &dyn DynQuantizer,
        reference: &dyn iface::Quantizer,
        rng: &mut StdRng,
        distribution: Uniform<f32>,
        ntrials: usize,
        ctx: &dyn std::fmt::Display,
    ) {
        assert_eq!(quantizer.dim(), reference.full_dim(), "{}", ctx);
        assert_eq!(quantizer.bytes().value(), reference.bytes(), "{}", ctx);

        let dim = quantizer.dim();
        let bytes = quantizer.bytes().value();

        let mut x = vec![0.0f32; dim];
        let mut bx = vec![0u8; bytes];

        let mut y = vec![0.0f32; dim];
        let mut by = vec![0u8; bytes];

        let mut expected = vec![0u8; bytes];

        for trial in 0..ntrials {
            // Include a zero query to exercise query scaling at zero norm.
            if trial == 0 {
                x.fill(0.0);
            } else {
                x.iter_mut().for_each(|x| *x = distribution.sample(rng));
            }
            y.iter_mut().for_each(|y| *y = distribution.sample(rng));

            quantizer.quantize(&mut bx, &x).unwrap();
            quantizer.quantize(&mut by, &y).unwrap();

            // Check compression.
            reference
                .compress(
                    &x,
                    iface::OpaqueMut::new(&mut expected),
                    ScopedAllocator::global(),
                )
                .unwrap();
            assert_eq!(bx, expected, "trial = {trial} -- {ctx}");

            // Check distance.
            let expected = reference
                .distance_computer_ref()
                .evaluate(iface::Opaque::new(&bx), iface::Opaque::new(&by))
                .unwrap();

            let distance = quantizer.distance();
            let from_distance = distance.eval(&bx, &by).unwrap();

            assert_eq!(from_distance, expected, "trial = {trial} -- {ctx}");

            for (layout, scale, actual) in [
                (
                    iface::QueryLayout::SameAsData,
                    false,
                    quantizer.insert_distance(&x).unwrap(),
                ),
                (
                    iface::QueryLayout::FullPrecision,
                    true,
                    quantizer.query_distance(&x).unwrap(),
                ),
            ] {
                let expected = reference
                    .fused_query_computer(
                        &x,
                        layout,
                        scale,
                        GlobalAllocator,
                        ScopedAllocator::global(),
                    )
                    .unwrap()
                    .evaluate_similarity(iface::Opaque::new(&by))
                    .unwrap();

                assert_eq!(
                    actual.eval(&by).unwrap(),
                    expected,
                    "trial = {trial}, layout = {layout} -- {ctx}"
                );
            }
        }

        // Compression Errors
        {
            // Input size
            let x = vec![1.0f32; dim - 1];
            let mut bx = vec![0u8; bytes];
            let err = quantizer.quantize(&mut bx, &x).unwrap_err().to_string();
            assert_contains!(err, "expected source vector to have length 5", "{}", ctx);

            let x = vec![1.0f32; dim + 1];
            let err = quantizer.quantize(&mut bx, &x).unwrap_err().to_string();
            assert_contains!(err, "expected source vector to have length 5", "{}", ctx);

            // Output size
            let x = vec![1.0f32; dim];
            let mut bx = vec![0u8; bytes - 1];
            assert!(quantizer.quantize(&mut bx, &x).is_err(), "{}", ctx);

            let mut bx = vec![0u8; bytes + 1];
            assert!(quantizer.quantize(&mut bx, &x).is_err(), "{}", ctx);

            // NaN
            let x = vec![f32::NAN; dim];
            let mut bx = vec![0u8; bytes];
            let err = quantizer.quantize(&mut bx, &x).unwrap_err().to_string();
            assert_contains!(err, "NaN");
        }

        // Distance
        {
            let d = quantizer.distance();
            for length in [bytes - 1, bytes + 1] {
                let invalid = vec![0u8; length];
                assert!(d.eval(&bx, &invalid).is_err(), "{ctx}");
                assert!(d.eval(&invalid, &bx).is_err(), "{ctx}");
            }
            assert!(d.eval(&bx, &by).is_ok(), "{ctx}");
        }

        // Query Distance
        {
            for length in [dim - 1, dim + 1] {
                let x = vec![1.0f32; length];
                assert!(quantizer.query_distance(&x).is_err(), "{ctx}");
            }

            let x = vec![f32::NAN; dim];
            let err = quantizer.query_distance(&x).unwrap_err().to_string();
            assert_contains!(err, "NaN");

            let x = vec![0.0f32; dim];
            let d = quantizer.query_distance(&x).unwrap();
            for length in [bytes - 1, bytes + 1] {
                let invalid = vec![0u8; length];
                assert!(d.eval(&invalid).is_err(), "{ctx}");
            }
            assert!(d.eval(&bx).is_ok(), "{ctx}");
        }

        // Insert Distance
        {
            for length in [dim - 1, dim + 1] {
                let x = vec![1.0f32; length];
                assert!(quantizer.insert_distance(&x).is_err(), "{ctx}");
            }

            let x = vec![f32::NAN; dim];
            let err = quantizer.insert_distance(&x).unwrap_err().to_string();
            assert_contains!(err, "NaN");

            let x = vec![0.0f32; dim];
            let d = quantizer.insert_distance(&x).unwrap();
            for length in [bytes - 1, bytes + 1] {
                let invalid = vec![0u8; length];
                assert!(d.eval(&invalid).is_err(), "{ctx}");
            }
            assert!(d.eval(&bx).is_ok(), "{ctx}");
        }
    }

    fn run_test<const N: usize>()
    where
        iface::Impl<N>: iface::Constructible + iface::Quantizer,
    {
        let mut rng = StdRng::seed_from_u64(0x019823745);
        let distribution = Uniform::new(-4.0f32, 4.0).unwrap();
        let mut data = rowmajor::Owned::from_element(TRIALS, DIM, 0.0);
        data.as_mut_slice()
            .iter_mut()
            .for_each(|v| *v = distribution.sample(&mut rng));

        for metric in [
            SupportedMetric::SquaredL2,
            SupportedMetric::InnerProduct,
            SupportedMetric::Cosine,
        ] {
            let quantizer = Spherical::new(train_quantizer::<N>(data.as_view(), metric));
            test_spherical(
                &quantizer,
                &*quantizer,
                &mut rng,
                distribution,
                TRIALS,
                &format_args!("metric = {metric:?}, bits = {N}"),
            );
        }
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
    fn test_four_bit() {
        run_test::<4>();
    }
}
