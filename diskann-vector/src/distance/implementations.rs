/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_wide::{arch::Target2, Architecture, ARCH};

/// Experimental traits for distance functions.
use super::simd;
use crate::{Half, MathematicalValue, PureDistanceFunction, SimilarityScore};

trait ToSlice {
    type Target;
    fn to_slice(&self) -> &[Self::Target];
}

impl<T> ToSlice for &[T] {
    type Target = T;
    fn to_slice(&self) -> &[T] {
        self
    }
}
impl<T, const N: usize> ToSlice for &[T; N] {
    type Target = T;
    fn to_slice(&self) -> &[T] {
        &self[..]
    }
}
impl<T, const N: usize> ToSlice for [T; N] {
    type Target = T;
    fn to_slice(&self) -> &[T] {
        &self[..]
    }
}

macro_rules! architecture_hook {
    ($functor:ty, $impl:path) => {
        impl<A, T, L, R> diskann_wide::arch::Target2<A, T, L, R> for $functor
        where
            A: Architecture,
            L: ToSlice,
            R: ToSlice,
            $impl: simd::SIMDSchema<L::Target, R::Target, A>,
            Self: PostOp<<$impl as simd::SIMDSchema<L::Target, R::Target, A>>::Return, T>,
        {
            #[inline(always)]
            fn run(self, arch: A, left: L, right: R) -> T {
                Self::post_op(simd::simd_op(
                    &$impl,
                    arch,
                    left.to_slice(),
                    right.to_slice(),
                ))
            }
        }

        impl<A, T, L, R> diskann_wide::arch::FTarget2<A, T, L, R> for $functor
        where
            A: Architecture,
            L: ToSlice,
            R: ToSlice,
            Self: diskann_wide::arch::Target2<A, T, L, R>,
        {
            #[inline(always)]
            fn run(arch: A, left: L, right: R) -> T {
                arch.run2(Self::default(), left, right)
            }
        }
    };
}

/// A utility for specializing distance computatiosn for fixed-length slices.
#[cfg(any(test, target_arch = "x86_64"))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Specialize<const N: usize, F>(std::marker::PhantomData<F>);

#[cfg(any(test, target_arch = "x86_64"))]
impl<A, T, L, R, const N: usize, F> diskann_wide::arch::FTarget2<A, T, &[L], &[R]>
    for Specialize<N, F>
where
    A: Architecture,
    F: for<'a, 'b> diskann_wide::arch::Target2<A, T, &'a [L; N], &'b [R; N]> + Default,
{
    #[inline(always)]
    fn run(arch: A, x: &[L], y: &[R]) -> T {
        if (x.len() != N) | (y.len() != N) {
            fail_length_check(x, y, N);
        }

        // SAFETY: We have checked that both arguments have the correct length.
        //
        // The alignment requirements of arrays are the alignment requirements of
        // `Left` and `Right` respectively, which is provided by the corresponding slices.
        arch.run2(
            F::default(),
            unsafe { &*(x.as_ptr() as *const [L; N]) },
            unsafe { &*(y.as_ptr() as *const [R; N]) },
        )
    }
}

// Outline the panic formatting and keep the calling convention the same as
// the top function. This keeps code generation extremely lightweight.
#[cfg(any(test, target_arch = "x86_64"))]
#[inline(never)]
#[allow(clippy::panic)]
fn fail_length_check<L, R>(x: &[L], y: &[R], len: usize) -> ! {
    let message = if x.len() != len {
        ("first", x.len())
    } else {
        ("second", y.len())
    };
    panic!(
        "expected {} argument to have length {}, instead it has length {}",
        message.0, len, message.1
    );
}

/// An internal trait to transform the result of the low-level SIMD ops into a value
/// expected by the rest of DiskANN.
///
/// Keep this trait private as it is likely to either change or be removed completely in the
/// near future once better integer implementations come online.
pub(super) trait PostOp<From, To> {
    fn post_op(x: From) -> To;
}

/// Provide explicit dynamic and sized implementations for a distance functor.
macro_rules! use_simd_implementation {
    ($functor:ty, $T:ty, $U:ty) => {
        //////////////////////
        // Similarity Score //
        //////////////////////

        // Dynamically Sized.
        impl PureDistanceFunction<&[$T], &[$U], SimilarityScore<f32>> for $functor {
            #[inline]
            fn evaluate(x: &[$T], y: &[$U]) -> SimilarityScore<f32> {
                <$functor>::default().run(ARCH, x, y)
            }
        }
        // Statically Sized
        impl<const N: usize> PureDistanceFunction<&[$T; N], &[$U; N], SimilarityScore<f32>>
            for $functor
        {
            #[inline]
            fn evaluate(x: &[$T; N], y: &[$U; N]) -> SimilarityScore<f32> {
                <$functor>::default().run(ARCH, x, y)
            }
        }

        ////////////////////////
        // Mathematical Value //
        ////////////////////////

        // Dynamically Sized.
        impl PureDistanceFunction<&[$T], &[$U], MathematicalValue<f32>> for $functor {
            #[inline]
            fn evaluate(x: &[$T], y: &[$U]) -> MathematicalValue<f32> {
                <$functor>::default().run(ARCH, x, y)
            }
        }
        // Statically Sized
        impl<const N: usize> PureDistanceFunction<&[$T; N], &[$U; N], MathematicalValue<f32>>
            for $functor
        {
            #[inline]
            fn evaluate(x: &[$T; N], y: &[$U; N]) -> MathematicalValue<f32> {
                <$functor>::default().run(ARCH, x, y)
            }
        }

        /////////
        // f32 //
        /////////

        // Dynamically Sized
        impl PureDistanceFunction<&[$T], &[$U], f32> for $functor {
            #[inline(always)]
            fn evaluate(x: &[$T], y: &[$U]) -> f32 {
                <$functor>::default().run(ARCH, x, y)
            }
        }

        // Statically Sized
        impl<const N: usize> PureDistanceFunction<&[$T; N], &[$U; N], f32> for $functor {
            #[inline]
            fn evaluate(x: &[$T; N], y: &[$U; N]) -> f32 {
                <$functor>::default().run(ARCH, x, y)
            }
        }
    };
}

///////////////
// SquaredL2 //
///////////////

/// Compute the squared L2 distance between two vectors.
#[derive(Debug, Clone, Copy, Default)]
pub struct SquaredL2 {}

impl PostOp<f32, SimilarityScore<f32>> for SquaredL2 {
    #[inline(always)]
    fn post_op(x: f32) -> SimilarityScore<f32> {
        SimilarityScore::new(x)
    }
}

impl PostOp<f32, f32> for SquaredL2 {
    #[inline(always)]
    fn post_op(x: f32) -> f32 {
        x
    }
}

impl PostOp<f32, MathematicalValue<f32>> for SquaredL2 {
    #[inline(always)]
    fn post_op(x: f32) -> MathematicalValue<f32> {
        MathematicalValue::new(x)
    }
}

architecture_hook!(SquaredL2, simd::L2);
use_simd_implementation!(SquaredL2, f32, f32);
use_simd_implementation!(SquaredL2, f32, Half);
use_simd_implementation!(SquaredL2, Half, Half);
use_simd_implementation!(SquaredL2, i8, i8);
use_simd_implementation!(SquaredL2, u8, u8);

////////////
// FullL2 //
////////////

/// Computes the full L2 distance between two vectors.
///
/// Unlike `SquaredL2`, this function-like object will perform compute the full L2 distance
/// including the trailing square root.
#[derive(Debug, Clone, Copy, Default)]
pub struct FullL2 {}

impl PostOp<f32, SimilarityScore<f32>> for FullL2 {
    #[inline(always)]
    fn post_op(x: f32) -> SimilarityScore<f32> {
        SimilarityScore::new(x.sqrt())
    }
}

impl PostOp<f32, f32> for FullL2 {
    #[inline(always)]
    fn post_op(x: f32) -> f32 {
        x.sqrt()
    }
}

impl PostOp<f32, MathematicalValue<f32>> for FullL2 {
    #[inline(always)]
    fn post_op(x: f32) -> MathematicalValue<f32> {
        MathematicalValue::new(x.sqrt())
    }
}

architecture_hook!(FullL2, simd::L2);
use_simd_implementation!(FullL2, f32, f32);
use_simd_implementation!(FullL2, f32, Half);
use_simd_implementation!(FullL2, Half, Half);
use_simd_implementation!(FullL2, i8, i8);
use_simd_implementation!(FullL2, u8, u8);

//////////////////
// InnerProduct //
//////////////////

/// Compute the inner product between two vectors.
#[derive(Debug, Clone, Copy, Default)]
pub struct InnerProduct {}

impl PostOp<f32, SimilarityScore<f32>> for InnerProduct {
    // The low-level operations compute the mathematical dot product.
    // Similarity scores used in DiskANN expect the InnerProduct to be negated.
    // This PostOp does that negation.
    #[inline(always)]
    fn post_op(x: f32) -> SimilarityScore<f32> {
        SimilarityScore::new(-x)
    }
}

impl PostOp<f32, MathematicalValue<f32>> for InnerProduct {
    #[inline(always)]
    fn post_op(x: f32) -> MathematicalValue<f32> {
        MathematicalValue::new(x)
    }
}

impl PostOp<f32, f32> for InnerProduct {
    #[inline(always)]
    fn post_op(x: f32) -> f32 {
        <Self as PostOp<f32, SimilarityScore<f32>>>::post_op(x).into_inner()
    }
}

architecture_hook!(InnerProduct, simd::IP);
use_simd_implementation!(InnerProduct, f32, f32);
use_simd_implementation!(InnerProduct, f32, Half);
use_simd_implementation!(InnerProduct, Half, Half);
use_simd_implementation!(InnerProduct, i8, i8);
use_simd_implementation!(InnerProduct, u8, u8);

////////////
// Cosine //
////////////

/// Perform the conversion `x -> 1 - x`.
///
/// Don't clamp the output - assume the output is clamped from the inner computation.
fn cosine_transformation(x: f32) -> f32 {
    1.0 - x
}

/// Compute the cosine similarity between two vectors.
#[derive(Debug, Clone, Copy, Default)]
pub struct Cosine {}

impl PostOp<f32, SimilarityScore<f32>> for Cosine {
    fn post_op(x: f32) -> SimilarityScore<f32> {
        debug_assert!(x >= -1.0);
        debug_assert!(x <= 1.0);
        SimilarityScore::new(cosine_transformation(x))
    }
}

impl PostOp<f32, MathematicalValue<f32>> for Cosine {
    fn post_op(x: f32) -> MathematicalValue<f32> {
        debug_assert!(x >= -1.0);
        debug_assert!(x <= 1.0);
        MathematicalValue::new(x)
    }
}

impl PostOp<f32, f32> for Cosine {
    fn post_op(x: f32) -> f32 {
        <Self as PostOp<f32, SimilarityScore<f32>>>::post_op(x).into_inner()
    }
}

architecture_hook!(Cosine, simd::CosineStateless);
use_simd_implementation!(Cosine, f32, f32);
use_simd_implementation!(Cosine, f32, Half);
use_simd_implementation!(Cosine, Half, Half);
use_simd_implementation!(Cosine, i8, i8);
use_simd_implementation!(Cosine, u8, u8);

//////////////////////
// CosineNormalized //
//////////////////////

/// Compute the cosine similarity between two normalized vectors.
#[derive(Debug, Clone, Copy, Default)]
pub struct CosineNormalized {}

impl PostOp<f32, SimilarityScore<f32>> for CosineNormalized {
    #[inline(always)]
    fn post_op(x: f32) -> SimilarityScore<f32> {
        // If the vectors are assumed to be normalized, then the implementation of
        // normalized cosine can be expressed in terms of an inner product inner loop.
        //
        // Don't use `clamp` at the end since the simple non-vector implementations do not
        // clamp their outputs.
        SimilarityScore::new(cosine_transformation(x))
    }
}

impl PostOp<f32, MathematicalValue<f32>> for CosineNormalized {
    #[inline(always)]
    fn post_op(x: f32) -> MathematicalValue<f32> {
        MathematicalValue::new(x)
    }
}

impl PostOp<f32, f32> for CosineNormalized {
    #[inline(always)]
    fn post_op(x: f32) -> f32 {
        <Self as PostOp<f32, SimilarityScore<f32>>>::post_op(x).into_inner()
    }
}

architecture_hook!(CosineNormalized, simd::IP);
use_simd_implementation!(CosineNormalized, f32, f32);
use_simd_implementation!(CosineNormalized, f32, Half);
use_simd_implementation!(CosineNormalized, Half, Half);

////////////
// L1Norm //
////////////

/// Compute the L1 norm of a vector.
#[derive(Debug, Clone, Copy, Default)]
pub struct L1NormFunctor {}

impl PostOp<f32, f32> for L1NormFunctor {
    #[inline(always)]
    fn post_op(x: f32) -> f32 {
        x
    }
}

architecture_hook!(L1NormFunctor, simd::L1Norm);

impl PureDistanceFunction<&[f32], &[f32], f32> for L1NormFunctor {
    #[inline]
    fn evaluate(x: &[f32], y: &[f32]) -> f32 {
        L1NormFunctor::default().run(ARCH, x, y)
    }
}

////////////
// Tests //
////////////

#[cfg(test)]
mod tests {

    use std::hash::{Hash, Hasher};

    use approx::assert_relative_eq;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::{
        distance::{
            reference::{self, ReferenceProvider},
            Metric,
        },
        test_util::{self, Normalize},
    };

    pub fn as_function_pointer<T, Left, Right, Return>(x: &[Left], y: &[Right]) -> Return
    where
        T: for<'a, 'b> PureDistanceFunction<&'a [Left], &'b [Right], Return>,
    {
        T::evaluate(x, y)
    }

    fn simd_provider(metric: Metric) -> fn(&[f32], &[f32]) -> f32 {
        match metric {
            Metric::L2 => as_function_pointer::<SquaredL2, _, _, _>,
            Metric::InnerProduct => as_function_pointer::<InnerProduct, _, _, _>,
            Metric::Cosine => as_function_pointer::<Cosine, _, _, _>,
            Metric::CosineNormalized => as_function_pointer::<CosineNormalized, _, _, _>,
        }
    }

    fn random_normal_arguments(dim: usize, lo: f32, hi: f32, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let x: Vec<f32> = (0..dim).map(|_| rng.random_range(lo..hi)).collect();
        let y: Vec<f32> = (0..dim).map(|_| rng.random_range(lo..hi)).collect();
        (x, y)
    }

    struct LeftRightPair {
        pub x: Vec<f32>,
        pub y: Vec<f32>,
    }

    fn generate_corner_cases(dim: usize) -> Vec<LeftRightPair> {
        let mut output = Vec::<LeftRightPair>::new();
        let fixed_values = [0.0, -5.0, 5.0, 10.0];

        for va in fixed_values.iter() {
            for vb in fixed_values.iter() {
                let x: Vec<f32> = vec![*va; dim];
                let y: Vec<f32> = vec![*vb; dim];
                output.push(LeftRightPair { x, y });
            }
        }
        output
    }

    fn collect_random_arguments(
        dim: usize,
        num_trials: usize,
        lo: f32,
        hi: f32,
        mut seed: u64,
    ) -> Vec<LeftRightPair> {
        (0..num_trials)
            .map(|_| {
                let (x, y) = random_normal_arguments(dim, lo, hi, seed);

                // update the seed.
                let mut hasher = std::hash::DefaultHasher::new();
                seed.hash(&mut hasher);
                seed = hasher.finish();

                LeftRightPair { x, y }
            })
            .collect()
    }

    fn test_pure_functions_impl<T>(metric: Metric, _func: T, normalize: bool)
    where
        T: for<'a, 'b> PureDistanceFunction<&'a [f32], &'b [f32], f32> + Clone,
    {
        let epsilon: f32 = 1e-4;
        let max_relative: f32 = 1e-4;

        let max_dim = 256;
        let num_trials = 10;

        let f_reference = <f32 as ReferenceProvider<f32>>::reference_implementation(metric);
        let f_simd = simd_provider(metric);

        // Inner test that loops over a vector of arguments.
        let run_tests = |argument_pairs: Vec<LeftRightPair>| {
            for LeftRightPair { mut x, mut y } in argument_pairs {
                if normalize {
                    x.normalize();
                    y.normalize();
                }

                let reference: f32 = f_reference(&x, &y).into_inner();
                let simd = f_simd(&x, &y);

                assert_relative_eq!(
                    reference,
                    simd,
                    epsilon = epsilon,
                    max_relative = max_relative
                );

                // Compute via direct call.
                let simd_direct = T::evaluate(&x, &y);
                assert_eq!(simd_direct, simd);
            }
        };

        // Corner Cases
        for dim in 0..max_dim {
            run_tests(generate_corner_cases(dim));
        }

        // Generated tests
        for dim in 0..max_dim {
            run_tests(collect_random_arguments(
                dim, num_trials, -10.0, 10.0, 0x5643,
            ));
        }
    }

    #[test]
    fn test_pure_functions() {
        println!("L2");
        test_pure_functions_impl(Metric::L2, SquaredL2 {}, false);
        println!("InnerProduct");
        test_pure_functions_impl(Metric::InnerProduct, InnerProduct {}, false);
        println!("Cosine");
        test_pure_functions_impl(Metric::Cosine, Cosine {}, false);
        println!("CosineNormalized");
        test_pure_functions_impl(Metric::CosineNormalized, CosineNormalized {}, true);
    }

    /// Test that the constant function pointer implementation returns the same result as
    /// non-sized counterpart..
    #[test]
    fn test_specialize() {
        use diskann_wide::arch::FTarget2;

        const DIM: usize = 123;
        let (x, y) = random_normal_arguments(DIM, -100.0, 100.0, 0x023457AA);

        let reference: f32 = SquaredL2::evaluate(x.as_slice(), y.as_slice());
        let evaluated: f32 =
            Specialize::<DIM, SquaredL2>::run(diskann_wide::ARCH, x.as_slice(), y.as_slice());

        // Equality should be exact.
        assert_eq!(reference, evaluated);
    }

    #[test]
    #[should_panic]
    fn test_function_pointer_const_panics_left() {
        use diskann_wide::arch::FTarget2;

        const DIM: usize = 34;
        let x = vec![0.0f32; DIM + 1];
        let y = vec![0.0f32; DIM];
        // Since `x` does not have the correct dimensions, this should panic.
        let _: f32 =
            Specialize::<DIM, SquaredL2>::run(diskann_wide::ARCH, x.as_slice(), y.as_slice());
    }

    #[test]
    #[should_panic]
    fn test_function_pointer_const_panics_right() {
        use diskann_wide::arch::FTarget2;

        const DIM: usize = 34;
        let x = vec![0.0f32; DIM];
        let y = vec![0.0f32; DIM + 1];
        // Since `y` does not have the correct dimensions, this should panic.
        let _: f32 =
            Specialize::<DIM, SquaredL2>::run(diskann_wide::ARCH, x.as_slice(), y.as_slice());
    }

    ////////////////////
    // Test Version 2 //
    ////////////////////

    trait GetInner {
        fn get_inner(self) -> f32;
    }

    impl GetInner for f32 {
        fn get_inner(self) -> f32 {
            self
        }
    }

    impl GetInner for SimilarityScore<f32> {
        fn get_inner(self) -> f32 {
            self.into_inner()
        }
    }

    impl GetInner for MathematicalValue<f32> {
        fn get_inner(self) -> f32 {
            self.into_inner()
        }
    }

    // Comparison Bounds
    #[derive(Clone, Copy)]
    struct EpsilonAndRelative {
        epsilon: f32,
        max_relative: f32,
    }

    #[allow(clippy::too_many_arguments)]
    fn run_test<L, R, To, Distribution, Callback>(
        under_test: fn(&[L], &[R]) -> To,
        reference: fn(&[L], &[R]) -> To,
        bounds: EpsilonAndRelative,
        dim: usize,
        num_trials: usize,
        distribution: Distribution,
        rng: &mut impl Rng,
        mut cb: Callback,
    ) where
        L: test_util::CornerCases,
        R: test_util::CornerCases,
        Distribution:
            test_util::GenerateRandomArguments<L> + test_util::GenerateRandomArguments<R> + Clone,
        To: GetInner + Copy,
        Callback: FnMut(To, To),
    {
        let checker =
            test_util::Checker::<L, R, To>::new(under_test, reference, |got, expected| {
                // Invoke the callback with the received numbers.
                cb(got, expected);
                assert_relative_eq!(
                    got.get_inner(),
                    expected.get_inner(),
                    epsilon = bounds.epsilon,
                    max_relative = bounds.max_relative
                );
            });

        test_util::test_distance_function(
            checker,
            distribution.clone(),
            distribution.clone(),
            dim,
            num_trials,
            rng,
        );
    }

    /// The maximum dimension tested for these tests.
    #[cfg(not(debug_assertions))]
    const MAX_DIM: usize = 256;

    #[cfg(debug_assertions)]
    const MAX_DIM: usize = 160;

    // Decrease the number of trials in debug mode to keep test run-time down.
    #[cfg(not(debug_assertions))]
    const INTEGER_TRIALS: usize = 10000;

    #[cfg(debug_assertions)]
    const INTEGER_TRIALS: usize = 100;

    ////////////////////
    // Integer Tester //
    ////////////////////

    // For integer tests - we expect exact reproducibility with the reference
    // implementations.
    fn run_integer_test<T, R>(
        under_test: fn(&[T], &[T]) -> R,
        reference: fn(&[T], &[T]) -> R,
        rng: &mut impl Rng,
    ) where
        T: test_util::CornerCases,
        R: GetInner + Copy,
        rand::distr::StandardUniform: test_util::GenerateRandomArguments<T> + Clone,
    {
        let distribution = rand::distr::StandardUniform {};
        let num_corner_cases = <T as test_util::CornerCases>::corner_cases().len();

        for dim in 0..MAX_DIM {
            let mut callcount = 0;
            let callback = |_, _| {
                callcount += 1;
            };

            run_test(
                under_test,
                reference,
                EpsilonAndRelative {
                    epsilon: 0.0,
                    max_relative: 0.0,
                },
                dim,
                INTEGER_TRIALS,
                distribution,
                rng,
                callback,
            );

            // Make sure the expected number of callbacks were made.
            assert_eq!(
                callcount,
                INTEGER_TRIALS + num_corner_cases * num_corner_cases
            );
        }
    }

    //////////////////
    // L2 - Integer //
    //////////////////

    #[test]
    fn test_l2_i8_mathematical() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x2bb701074c2b81c9);
        run_integer_test(
            as_function_pointer::<FullL2, i8, i8, MathematicalValue<f32>>,
            reference::reference_l2_i8_mathematical,
            &mut rng,
        );
    }

    #[test]
    fn test_l2_u8_mathematical() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x9284ced6d080808c);
        run_integer_test(
            as_function_pointer::<FullL2, u8, u8, MathematicalValue<f32>>,
            reference::reference_l2_u8_mathematical,
            &mut rng,
        );
    }

    #[test]
    fn test_l2_i8_similarity() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xb196fecc4def04fa);
        run_integer_test(
            as_function_pointer::<FullL2, i8, i8, SimilarityScore<f32>>,
            reference::reference_l2_i8_similarity,
            &mut rng,
        );
    }

    #[test]
    fn test_l2_u8_similarity() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x07f6463e4a654aea);
        run_integer_test(
            as_function_pointer::<FullL2, u8, u8, SimilarityScore<f32>>,
            reference::reference_l2_u8_similarity,
            &mut rng,
        );
    }

    ////////////////////////////
    // InnerProduct - Integer //
    ////////////////////////////

    #[test]
    fn test_innerproduct_i8_mathematical() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x2c1b1bddda5774be);
        run_integer_test(
            as_function_pointer::<InnerProduct, i8, i8, MathematicalValue<f32>>,
            reference::reference_innerproduct_i8_mathematical,
            &mut rng,
        );
    }

    #[test]
    fn test_innerproduct_u8_mathematical() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x757e363832d7f215);
        run_integer_test(
            as_function_pointer::<InnerProduct, u8, u8, MathematicalValue<f32>>,
            reference::reference_innerproduct_u8_mathematical,
            &mut rng,
        );
    }

    #[test]
    fn test_innerproduct_i8_similarity() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x4788ce0b991eb15a);
        run_integer_test(
            as_function_pointer::<InnerProduct, i8, i8, SimilarityScore<f32>>,
            reference::reference_innerproduct_i8_similarity,
            &mut rng,
        );
    }

    #[test]
    fn test_innerproduct_u8_similarity() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x4994adb68f814d96);
        run_integer_test(
            as_function_pointer::<InnerProduct, u8, u8, SimilarityScore<f32>>,
            reference::reference_innerproduct_u8_similarity,
            &mut rng,
        );
    }

    //////////////////////
    // Cosine - Integer //
    //////////////////////

    #[test]
    fn test_cosine_i8_mathematical() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xedef81c780491ada);
        run_integer_test(
            as_function_pointer::<Cosine, i8, i8, MathematicalValue<f32>>,
            reference::reference_cosine_i8_mathematical,
            &mut rng,
        );
    }

    #[test]
    fn test_cosine_u8_mathematical() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x107cee2adcc58b73);
        run_integer_test(
            as_function_pointer::<Cosine, u8, u8, MathematicalValue<f32>>,
            reference::reference_cosine_u8_mathematical,
            &mut rng,
        );
    }

    #[test]
    fn test_cosine_i8_similarity() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x02d95c1cc0843647);
        run_integer_test(
            as_function_pointer::<Cosine, i8, i8, SimilarityScore<f32>>,
            reference::reference_cosine_i8_similarity,
            &mut rng,
        );
    }

    #[test]
    fn test_cosine_u8_similarity() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xf5ea1974bf8d8b3b);
        run_integer_test(
            as_function_pointer::<Cosine, u8, u8, SimilarityScore<f32>>,
            reference::reference_cosine_u8_similarity,
            &mut rng,
        );
    }

    //////////////////
    // Float Tester //
    //////////////////

    // For integer tests - we expect exact reproducibility with the reference
    // implementations.
    fn run_float_test<L, R, To, Dist>(
        under_test: fn(&[L], &[R]) -> To,
        reference: fn(&[L], &[R]) -> To,
        rng: &mut impl Rng,
        distribution: Dist,
        bounds: EpsilonAndRelative,
    ) where
        L: test_util::CornerCases,
        R: test_util::CornerCases,
        To: GetInner + Copy,
        Dist: test_util::GenerateRandomArguments<L> + test_util::GenerateRandomArguments<R> + Clone,
    {
        let left_corner_cases = <L as test_util::CornerCases>::corner_cases().len();
        let right_corner_cases = <R as test_util::CornerCases>::corner_cases().len();
        for dim in 0..MAX_DIM {
            let mut callcount = 0;
            let callback = |_, _| {
                callcount += 1;
            };

            run_test(
                under_test,
                reference,
                bounds,
                dim,
                INTEGER_TRIALS,
                distribution.clone(),
                rng,
                callback,
            );

            // Make sure the expected number of callbacks were made.
            assert_eq!(
                callcount,
                INTEGER_TRIALS + left_corner_cases * right_corner_cases
            );
        }
    }

    ////////////////
    // L2 - Float //
    ////////////////

    fn expected_l2_errors() -> EpsilonAndRelative {
        EpsilonAndRelative {
            epsilon: 0.0,
            max_relative: 1.2e-6,
        }
    }

    #[test]
    fn test_l2_f32_mathematical() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x6d22d320bdf35aec);
        let distribution = rand_distr::Normal::new(0.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<FullL2, f32, f32, MathematicalValue<f32>>,
            reference::reference_l2_f32_mathematical,
            &mut rng,
            distribution,
            expected_l2_errors(),
        );
    }

    #[test]
    fn test_l2_f16_mathematical() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x755819460c190db4);
        let distribution = rand_distr::Normal::new(0.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<FullL2, Half, Half, MathematicalValue<f32>>,
            reference::reference_l2_f16_mathematical,
            &mut rng,
            distribution,
            expected_l2_errors(),
        );
    }

    #[test]
    fn test_l2_f32xf16_mathematical() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x755819460c190db4);
        let distribution = rand_distr::Normal::new(0.0, 1.0).unwrap();

        run_float_test(
            as_function_pointer::<FullL2, f32, Half, MathematicalValue<f32>>,
            reference::reference_l2_f32xf16_mathematical,
            &mut rng,
            distribution,
            expected_l2_errors(),
        );
    }

    #[test]
    fn test_l2_f32_similarity() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xbfc5f4b42b5bc0c1);
        let distribution = rand_distr::Normal::new(0.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<FullL2, f32, f32, SimilarityScore<f32>>,
            reference::reference_l2_f32_similarity,
            &mut rng,
            distribution,
            expected_l2_errors(),
        );
    }

    #[test]
    fn test_l2_f16_similarity() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x9d3809d84f54e4b6);
        let distribution = rand_distr::Normal::new(0.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<FullL2, Half, Half, SimilarityScore<f32>>,
            reference::reference_l2_f16_similarity,
            &mut rng,
            distribution,
            expected_l2_errors(),
        );
    }

    #[test]
    fn test_l2_f32xf16_similarity() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x755819460c190db4);
        let distribution = rand_distr::Normal::new(0.0, 1.0).unwrap();

        run_float_test(
            as_function_pointer::<FullL2, f32, Half, SimilarityScore<f32>>,
            reference::reference_l2_f32xf16_similarity,
            &mut rng,
            distribution,
            expected_l2_errors(),
        );
    }

    ///////////////////////////
    // InnerProduct - Floats //
    ///////////////////////////

    fn expected_innerproduct_errors() -> EpsilonAndRelative {
        EpsilonAndRelative {
            epsilon: 2.5e-5,
            max_relative: 1.6e-5,
        }
    }

    #[test]
    fn test_innerproduct_f32_mathematical() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x1ef6ac3b65869792);
        let distribution = rand_distr::Normal::new(0.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<InnerProduct, f32, f32, MathematicalValue<f32>>,
            reference::reference_innerproduct_f32_mathematical,
            &mut rng,
            distribution,
            expected_innerproduct_errors(),
        );
    }

    #[test]
    fn test_innerproduct_f16_mathematical() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x24c51e4b825b0329);
        let distribution = rand_distr::Normal::new(0.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<InnerProduct, Half, Half, MathematicalValue<f32>>,
            reference::reference_innerproduct_f16_mathematical,
            &mut rng,
            distribution,
            expected_innerproduct_errors(),
        );
    }

    #[test]
    fn test_innerproduct_f32xf16_mathematical() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x24c51e4b825b0329);
        let distribution = rand_distr::Normal::new(0.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<InnerProduct, f32, Half, MathematicalValue<f32>>,
            reference::reference_innerproduct_f32xf16_mathematical,
            &mut rng,
            distribution,
            expected_innerproduct_errors(),
        );
    }

    #[test]
    fn test_innerproduct_f32_similarity() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x40326b22a57db0d7);
        let distribution = rand_distr::Normal::new(0.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<InnerProduct, f32, f32, SimilarityScore<f32>>,
            reference::reference_innerproduct_f32_similarity,
            &mut rng,
            distribution,
            expected_innerproduct_errors(),
        );
    }

    #[test]
    fn test_innerproduct_f16_similarity() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xfb8cff47bcbc9528);
        let distribution = rand_distr::Normal::new(0.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<InnerProduct, Half, Half, SimilarityScore<f32>>,
            reference::reference_innerproduct_f16_similarity,
            &mut rng,
            distribution,
            expected_innerproduct_errors(),
        );
    }

    #[test]
    fn test_innerproduct_f32xf16_similarity() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x24c51e4b825b0329);
        let distribution = rand_distr::Normal::new(0.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<InnerProduct, f32, Half, SimilarityScore<f32>>,
            reference::reference_innerproduct_f32xf16_similarity,
            &mut rng,
            distribution,
            expected_innerproduct_errors(),
        );
    }

    /////////////////////
    // Cosine - Floats //
    /////////////////////

    fn expected_cosine_errors() -> EpsilonAndRelative {
        EpsilonAndRelative {
            epsilon: 3e-7,
            max_relative: 5e-6,
        }
    }

    #[test]
    fn test_cosine_f32_mathematical() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xca6eaac942999500);
        let distribution = rand_distr::Normal::new(0.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<Cosine, f32, f32, MathematicalValue<f32>>,
            reference::reference_cosine_f32_mathematical,
            &mut rng,
            distribution,
            expected_cosine_errors(),
        );
    }

    #[test]
    fn test_cosine_f16_mathematical() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xa736c789aa16ce86);
        let distribution = rand_distr::Normal::new(0.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<Cosine, Half, Half, MathematicalValue<f32>>,
            reference::reference_cosine_f16_mathematical,
            &mut rng,
            distribution,
            expected_cosine_errors(),
        );
    }

    #[test]
    fn test_cosine_f32xf16_mathematical() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xac550231088a0d5c);
        let distribution = rand_distr::Normal::new(0.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<Cosine, f32, Half, MathematicalValue<f32>>,
            reference::reference_cosine_f32xf16_mathematical,
            &mut rng,
            distribution,
            expected_cosine_errors(),
        );
    }

    #[test]
    fn test_cosine_f32_similarity() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x4a09ad987a6204f3);
        let distribution = rand_distr::Normal::new(0.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<Cosine, f32, f32, SimilarityScore<f32>>,
            reference::reference_cosine_f32_similarity,
            &mut rng,
            distribution,
            expected_cosine_errors(),
        );
    }

    #[test]
    fn test_cosine_f16_similarity() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x77a48d1914f850f2);
        let distribution = rand_distr::Normal::new(0.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<Cosine, Half, Half, SimilarityScore<f32>>,
            reference::reference_cosine_f16_similarity,
            &mut rng,
            distribution,
            expected_cosine_errors(),
        );
    }

    #[test]
    fn test_cosine_f32xf16_similarity() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xbd7471b815655ca1);
        let distribution = rand_distr::Normal::new(0.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<Cosine, f32, Half, SimilarityScore<f32>>,
            reference::reference_cosine_f32xf16_similarity,
            &mut rng,
            distribution,
            expected_cosine_errors(),
        );
    }

    ///////////////////////////////
    // CosineNormalized - Floats //
    ///////////////////////////////

    fn expected_cosine_normalized_errors() -> EpsilonAndRelative {
        EpsilonAndRelative {
            epsilon: 3e-7,
            max_relative: 5e-6,
        }
    }

    #[test]
    fn test_cosine_normalized_f32_mathematical() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x1fda98112747f8dd);
        let distribution = rand_distr::Normal::new(-1.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<CosineNormalized, f32, f32, MathematicalValue<f32>>,
            reference::reference_cosine_normalized_f32_mathematical,
            &mut rng,
            test_util::Normalized(distribution),
            expected_cosine_normalized_errors(),
        );
    }

    #[test]
    fn test_cosine_normalized_f16_mathematical() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5e8c5d5e19cdd840);
        let distribution = rand_distr::Normal::new(-1.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<CosineNormalized, Half, Half, MathematicalValue<f32>>,
            reference::reference_cosine_normalized_f16_mathematical,
            &mut rng,
            test_util::Normalized(distribution),
            expected_cosine_normalized_errors(),
        );
    }

    #[test]
    fn test_cosine_normalized_f32xf16_mathematical() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x3fd01e1c11c9bc45);
        let distribution = rand_distr::Normal::new(-1.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<CosineNormalized, f32, Half, MathematicalValue<f32>>,
            reference::reference_cosine_normalized_f32xf16_mathematical,
            &mut rng,
            test_util::Normalized(distribution),
            expected_cosine_normalized_errors(),
        );
    }

    #[test]
    fn test_cosine_normalized_f32_similarity() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x9446d057870e5605);
        let distribution = rand_distr::Normal::new(-1.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<CosineNormalized, f32, f32, SimilarityScore<f32>>,
            reference::reference_cosine_normalized_f32_similarity,
            &mut rng,
            test_util::Normalized(distribution),
            expected_cosine_normalized_errors(),
        );
    }

    #[test]
    fn test_cosine_normalized_f16_similarity() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x885c371801f18174);
        let distribution = rand_distr::Normal::new(-1.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<CosineNormalized, Half, Half, SimilarityScore<f32>>,
            reference::reference_cosine_normalized_f16_similarity,
            &mut rng,
            test_util::Normalized(distribution),
            expected_cosine_normalized_errors(),
        );
    }

    #[test]
    fn test_cosine_normalized_f32xf16_similarity() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x1c356c92d0522c0f);
        let distribution = rand_distr::Normal::new(-1.0, 1.0).unwrap();
        run_float_test(
            as_function_pointer::<CosineNormalized, f32, Half, SimilarityScore<f32>>,
            reference::reference_cosine_normalized_f32xf16_similarity,
            &mut rng,
            test_util::Normalized(distribution),
            expected_cosine_normalized_errors(),
        );
    }
}
