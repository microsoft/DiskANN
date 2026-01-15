/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};
use diskann_wide::{
    arch::{Dispatched2, FTarget2, Scalar},
    lifetime::Ref,
    Architecture,
};
use half::f16;

use super::{Cosine, CosineNormalized, InnerProduct, SquaredL2};
use crate::distance::Metric;

#[cfg(target_arch = "x86_64")]
use super::implementations::Specialize;

/// Return a function pointer-like [`Distance`] to compute the requested metric.
///
/// If `dimension` is provided, then the returned function may **only** be used on
/// slices with length `dimension`. Calling the returned function with a different sized
/// slice **may** panic.
///
/// If `dimension` is not provided, then the returned function will work for all sizes.
///
/// The functions returned by `distance_comparer` do not have strict alignment
/// requirements, though aligning your data *may* yield better memory performance.
///
/// # Metric Semantics
///
/// The values computed by the returned functions may be modified from the true mathematical
/// definition of the metric to ensure that values closer to `-infinity` imply more similar.
///
/// * `L2`: Computes the squared L2 distance between vectors.
/// * `InnerProduct`: Returns the **negative** inner-product.
/// * `Cosine`: Returns `1 - cosine-similarity` and will work on un-normalized vectors.
/// * `CosineNormalized`: Returns `1 - cosinesimilarity` with the hint that the provided
///   vectors have norm 1. This allows for potentially more-efficient implementations but the
///   results may be incorrect if called with unnormalized data.
///
///   When provided with integer arguments (for which normalization does not make sense), this
///   behaves as if `Cosine` was provided.
pub trait DistanceProvider<T>: Sized + 'static {
    fn distance_comparer(metric: Metric, dimension: Option<usize>) -> Distance<Self, T>;
}

/// A function pointer-like type for computing distances between `&[T]` and `&[U]`.
///
/// See: [`DistanceProvider`].
#[derive(Debug, Clone, Copy)]
pub struct Distance<T, U>
where
    T: 'static,
    U: 'static,
{
    f: Dispatched2<f32, Ref<[T]>, Ref<[U]>>,
}

impl<T, U> Distance<T, U>
where
    T: 'static,
    U: 'static,
{
    fn new(f: Dispatched2<f32, Ref<[T]>, Ref<[U]>>) -> Self {
        Self { f }
    }

    /// Compute a distances between `x` and `y`.
    ///
    /// The actual distance computed depends on the metric supplied to [`DistanceProvider`].
    ///
    /// Additionally, if a dimension were given to [`DistanceProvider`], this function may
    /// panic if provided with slices with a length not equal to this dimension.
    #[inline]
    pub fn call(&self, x: &[T], y: &[U]) -> f32 {
        self.f.call(x, y)
    }
}

impl<T, U> crate::DistanceFunction<&[T], &[U], f32> for Distance<T, U>
where
    T: 'static,
    U: 'static,
{
    fn evaluate_similarity(&self, x: &[T], y: &[U]) -> f32 {
        self.call(x, y)
    }
}

////////////////////
// Implementation //
////////////////////

// Implementation Notes
//
// Our implementation of `DistanceProvider` dispatches across:
//
// * Data Types
// * Metric
// * Dimensions
// * Runtime Micro-architecture
//
// This is a combinatorial explosing of potentially compiled kernels. To get a handle on the
// sheer number of compiled functions, we manually control the dimensional specialization on
// a case-by-case basis.
//
// This is facilitated by the `specialize!` macro, which accepts a list of dimensions and
// instantiates the necessary machinery.
//
// To explain the machiner a little, a [`Cons`] compile-time list is constructed. This type
// might look like
//
// * `Cons<Spec<100>, Cons<Spec<64>, Spec<32>>>`: To specialize dimensions 100, 64, and 32.
// * `Cons<Spec<100>, Null>`: To specialize just dimension 100.
// * `Cons<Null, Null>`: To specialize no dimensions.
//
// The `TrySpecialize` trait is then used specialize a kernel `F` for an architecture `A`
// with implementations
//
// * `Spec<N>`: Check if the requested dimension is equal to `N` and if so, return the
//   specialized method.
// * `Null`: Never specialize.
// * `Cons<Head, Tail>`: Try to specialize using `Head` returning if successful. Otherwise,
//   return the specialization of `Tail`.
//
//   This definition is what allows nested `Cons` structures to specialize multiple dimensions.
//
//   The `Cons` list also compiles a generic-dimensional fallback if none of the
//   specializations match.
//
// The overall flow is
//
// 1. Enter the `DistanceProvider` implementation.
//
// 2. First dispatch across micro-architecture using `ArgumentTypes` to hold the data types.
//    `ArgumentTypes` implements `Target2` to facilitate this dispatch.
//
// 3. The implementations of `Target2` for `ArgumentTypes` are performed by the `specialize!`
//    macro, which creates a `Cons` list of requested specializations, switches across
//    metrics and invokes `Cons:specialize` on the requested metric.

macro_rules! provider {
    ($T:ty, $U:ty) => {
        impl DistanceProvider<$U> for $T {
            fn distance_comparer(metric: Metric, dimension: Option<usize>) -> Distance<$T, $U> {
                // Use the `no-features` variant because we do not care if the target gets
                // compiled for higher micro-architecture levels.
                //
                // It's the returned kernel that matters.
                diskann_wide::arch::dispatch2_no_features(
                    ArgumentTypes::<$T, $U>::new(),
                    metric,
                    dimension,
                )
            }
        }
    };
}

provider!(f32, f32);
provider!(f16, f16);
provider!(f32, f16);
provider!(i8, i8);
provider!(u8, u8);

/////////////////////////
// Specialization List //
/////////////////////////

macro_rules! spec_list {
    ($($Ns:literal),* $(,)?) => {
        spec_list!(@value, $($Ns,)*)
    };
    (@value $(,)?) => {
        Cons::new(Null, Null)
    };
    (@value, $N0:literal $(,)?) => {
        Cons::new(Spec::<$N0>, Null)
    };
    (@value, $N0:literal, $N1:literal $(,)?) => {
        Cons::new(Spec::<$N0>, Spec::<$N1>)
    };
    (@value, $N0:literal, $N1:literal, $($Ns:literal),+ $(,)?) => {
        Cons::new(Spec::<$N0>, spec_list!(@value, $N1, $($Ns,)+))
    };
}

struct ArgumentTypes<T: 'static, U: 'static>(std::marker::PhantomData<(T, U)>);

impl<T, U> ArgumentTypes<T, U>
where
    T: 'static,
    U: 'static,
{
    fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

macro_rules! specialize {
    ($arch:ty, $T:ty, $U:ty, $($Ns:literal),* $(,)?) => {
        impl diskann_wide::arch::Target2<
            $arch,
            Distance<$T, $U>,
            Metric,
            Option<usize>,
        > for ArgumentTypes<$T, $U> {
            fn run(
                self,
                arch: $arch,
                metric: Metric,
                dim: Option<usize>,
            ) -> Distance<$T, $U> {
                let spec = spec_list!($($Ns),*);
                match metric {
                    Metric::L2 => spec.specialize(arch, SquaredL2 {}, dim),
                    Metric::Cosine => spec.specialize(arch, Cosine {}, dim),
                    Metric::CosineNormalized => spec.specialize(arch, CosineNormalized {}, dim),
                    Metric::InnerProduct => spec.specialize(arch, InnerProduct {}, dim),
                }
            }
        }
    };
    // Integer types redirect `CosineNormalized` to `Cosine`.
    (@integer, $arch:ty, $T:ty, $U:ty, $($Ns:literal),* $(,)?) => {
        impl diskann_wide::arch::Target2<
            $arch,
            Distance<$T, $U>,
            Metric,
            Option<usize>,
        > for ArgumentTypes<$T, $U> {
            fn run(
                self,
                arch: $arch,
                metric: Metric,
                dim: Option<usize>,
            ) -> Distance<$T, $U> {
                let spec = spec_list!($($Ns),*);
                match metric {
                    Metric::L2 => spec.specialize(arch, SquaredL2 {}, dim),
                    Metric::Cosine | Metric::CosineNormalized => {
                        spec.specialize(arch, Cosine {}, dim)
                    },
                    Metric::InnerProduct => spec.specialize(arch, InnerProduct {}, dim),
                }
            }
        }
    };
}

specialize!(Scalar, f32, f32,);
specialize!(Scalar, f32, f16,);
specialize!(Scalar, f16, f16,);
specialize!(@integer, Scalar, u8, u8,);
specialize!(@integer, Scalar, i8, i8,);

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::*;

    specialize!(V3, f32, f32, 768, 384, 128, 100);
    specialize!(V4, f32, f32, 768, 384, 128, 100);

    specialize!(V3, f32, f16, 768, 384, 128, 100);
    specialize!(V4, f32, f16, 768, 384, 128, 100);

    specialize!(V3, f16, f16, 768, 384, 128, 100);
    specialize!(V4, f16, f16, 768, 384, 128, 100);

    specialize!(@integer, V3, u8, u8, 128);
    specialize!(@integer, V4, u8, u8, 128);

    specialize!(@integer, V3, i8, i8, 128, 100);
    specialize!(@integer, V4, i8, i8, 128, 100);
}

/// Specialize a distance function `F` for the dimension `dim` if possible. Otherwise,
/// return `None`.
trait TrySpecialize<A, F, T, U>
where
    A: Architecture,
    T: 'static,
    U: 'static,
{
    fn try_specialize(&self, arch: A, dim: Option<usize>) -> Option<Distance<T, U>>;
}

/// Specialize a distance function for the requested dimensionality.
#[cfg(target_arch = "x86_64")]
struct Spec<const N: usize>;

#[cfg(target_arch = "x86_64")]
impl<A, F, const N: usize, T, U> TrySpecialize<A, F, T, U> for Spec<N>
where
    A: Architecture,
    Specialize<N, F>: for<'a, 'b> FTarget2<A, f32, &'a [T], &'b [U]>,
    T: 'static,
    U: 'static,
{
    fn try_specialize(&self, arch: A, dim: Option<usize>) -> Option<Distance<T, U>> {
        if let Some(d) = dim {
            if d == N {
                return Some(Distance::new(
                    // NOTE: This line here is what actually compiles the specialized kernel.
                    arch.dispatch2::<Specialize<N, F>, f32, Ref<[T]>, Ref<[U]>>(),
                ));
            }
        }
        None
    }
}

/// Don't specialize at all.
struct Null;

impl<A, F, T, U> TrySpecialize<A, F, T, U> for Null
where
    A: Architecture,
    T: 'static,
    U: 'static,
{
    fn try_specialize(&self, _arch: A, _dim: Option<usize>) -> Option<Distance<T, U>> {
        None
    }
}

/// A recursive compile-time list for building a list of specializations.
struct Cons<Head, Tail> {
    head: Head,
    tail: Tail,
}

impl<Head, Tail> Cons<Head, Tail> {
    const fn new(head: Head, tail: Tail) -> Self {
        Self { head, tail }
    }

    /// Try to specialize `F`. If no such specialization is available, return a fallback
    /// implementation.
    fn specialize<A, F, T, U>(&self, arch: A, _f: F, dim: Option<usize>) -> Distance<T, U>
    where
        A: Architecture,
        F: for<'a, 'b> FTarget2<A, f32, &'a [T], &'b [U]>,
        Head: TrySpecialize<A, F, T, U>,
        Tail: TrySpecialize<A, F, T, U>,
        T: 'static,
        U: 'static,
    {
        if let Some(f) = self.try_specialize(arch, dim) {
            f
        } else {
            Distance::new(arch.dispatch2::<F, f32, Ref<[T]>, Ref<[U]>>())
        }
    }
}

// Try `Head` and then `Tail`.
impl<A, Head, Tail, F, T, U> TrySpecialize<A, F, T, U> for Cons<Head, Tail>
where
    A: Architecture,
    Head: TrySpecialize<A, F, T, U>,
    Tail: TrySpecialize<A, F, T, U>,
    T: 'static,
    U: 'static,
{
    fn try_specialize(&self, arch: A, dim: Option<usize>) -> Option<Distance<T, U>> {
        if let Some(f) = self.head.try_specialize(arch, dim) {
            Some(f)
        } else {
            self.tail.try_specialize(arch, dim)
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_unaligned_distance_provider {
    use approx::assert_relative_eq;
    use rand::{self, SeedableRng};

    use super::*;
    use crate::{
        distance::{reference::ReferenceProvider, Metric},
        test_util, SimilarityScore,
    };

    // Comparison Bounds
    struct EpsilonAndRelative {
        epsilon: f32,
        max_relative: f32,
    }

    /// For now - these are rough bounds selected heuristically.
    /// Eventually (once we have implementations using compensated arithmetic), we should
    /// empirically derive bounds based on a combination of
    ///
    /// 1. Input Distribution
    /// 2. Distance Function
    /// 3. Dimensionality
    ///
    /// To ensure that these bounds are tight.
    fn get_float_bounds(metric: Metric) -> EpsilonAndRelative {
        match metric {
            Metric::L2 => EpsilonAndRelative {
                epsilon: 1e-5,
                max_relative: 1e-5,
            },
            Metric::InnerProduct => EpsilonAndRelative {
                epsilon: 1e-4,
                max_relative: 1e-4,
            },
            Metric::Cosine => EpsilonAndRelative {
                epsilon: 1e-4,
                max_relative: 1e-4,
            },
            Metric::CosineNormalized => EpsilonAndRelative {
                epsilon: 1e-4,
                max_relative: 1e-4,
            },
        }
    }

    fn get_int_bounds(metric: Metric) -> EpsilonAndRelative {
        match metric {
            // Allow for some error when handling the normalization at the end.
            Metric::Cosine | Metric::CosineNormalized => EpsilonAndRelative {
                epsilon: 1e-6,
                max_relative: 1e-6,
            },
            // These should be exact.
            Metric::L2 | Metric::InnerProduct => EpsilonAndRelative {
                epsilon: 0.0,
                max_relative: 0.0,
            },
        }
    }

    fn do_test<T, Distribution>(
        under_test: Distance<T, T>,
        reference: fn(&[T], &[T]) -> SimilarityScore<f32>,
        bounds: EpsilonAndRelative,
        dim: usize,
        distribution: Distribution,
    ) where
        T: test_util::CornerCases,
        Distribution: test_util::GenerateRandomArguments<T> + Clone,
    {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xef0053c);

        // Unwrap the SimilarityScore for the reference implementation.
        let converted = |a: &[T], b: &[T]| -> f32 { reference(a, b).into_inner() };

        let checker = test_util::Checker::<T, T, f32>::new(
            |a, b| under_test.call(a, b),
            converted,
            |got: f32, expected: f32| {
                assert_relative_eq!(
                    got,
                    expected,
                    epsilon = bounds.epsilon,
                    max_relative = bounds.max_relative
                );
            },
        );

        test_util::test_distance_function(
            checker,
            distribution.clone(),
            distribution.clone(),
            dim,
            10,
            &mut rng,
        );
    }

    fn all_metrics() -> [Metric; 4] {
        [
            Metric::L2,
            Metric::InnerProduct,
            Metric::Cosine,
            Metric::CosineNormalized,
        ]
    }

    /// The maximum dimension used for unaligned behavior checking with simple distances.
    const MAX_DIM: usize = 256;

    #[test]
    fn test_unaligned_f32() {
        let dist = rand_distr::Normal::new(0.0, 1.0).unwrap();
        for metric in all_metrics() {
            for dim in 0..MAX_DIM {
                println!("Metric = {:?}, dim = {}", metric, dim);
                let unaligned = <f32 as DistanceProvider<f32>>::distance_comparer(metric, None);
                let simple = <f32 as ReferenceProvider<f32>>::reference_implementation(metric);
                let bounds = get_float_bounds(metric);
                do_test(unaligned, simple, bounds, dim, dist);
            }
        }
    }

    #[test]
    fn test_unaligned_f16() {
        let dist = rand_distr::Normal::new(0.0, 1.0).unwrap();
        for metric in all_metrics() {
            for dim in 0..MAX_DIM {
                println!("Metric = {:?}, dim = {}", metric, dim);
                let unaligned = <f16 as DistanceProvider<f16>>::distance_comparer(metric, None);
                let simple = <f16 as ReferenceProvider<f16>>::reference_implementation(metric);
                let bounds = get_float_bounds(metric);
                do_test(unaligned, simple, bounds, dim, dist);
            }
        }
    }

    #[test]
    fn test_unaligned_u8() {
        let dist = rand::distr::StandardUniform {};
        for metric in all_metrics() {
            for dim in 0..MAX_DIM {
                println!("Metric = {:?}, dim = {}", metric, dim);
                let unaligned = <u8 as DistanceProvider<u8>>::distance_comparer(metric, None);
                let simple = <u8 as ReferenceProvider<u8>>::reference_implementation(metric);
                let bounds = get_int_bounds(metric);
                do_test(unaligned, simple, bounds, dim, dist);
            }
        }
    }

    #[test]
    fn test_unaligned_i8() {
        let dist = rand::distr::StandardUniform {};
        for metric in all_metrics() {
            for dim in 0..MAX_DIM {
                println!("Metric = {:?}, dim = {}", metric, dim);
                let unaligned = <i8 as DistanceProvider<i8>>::distance_comparer(metric, None);
                let simple = <i8 as ReferenceProvider<i8>>::reference_implementation(metric);

                let bounds = get_int_bounds(metric);
                do_test(unaligned, simple, bounds, dim, dist);
            }
        }
    }
}

#[cfg(test)]
mod distance_provider_f32_tests {
    use approx::assert_abs_diff_eq;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::{distance::reference, test_util::*};

    #[repr(C, align(32))]
    pub struct F32Slice112([f32; 112]);
    #[repr(C, align(32))]
    pub struct F32Slice104([f32; 104]);
    #[repr(C, align(32))]
    pub struct F32Slice128([f32; 128]);
    #[repr(C, align(32))]
    pub struct F32Slice256([f32; 256]);
    #[repr(C, align(32))]
    pub struct F32Slice4096([f32; 4096]);

    pub fn get_turing_test_data_f32_dim(dim: usize) -> (Vec<f32>, Vec<f32>) {
        let mut a_slice = vec![0.0f32; dim];
        let mut b_slice = vec![0.0f32; dim];

        let mut rng = StdRng::seed_from_u64(42);
        for i in 0..dim {
            a_slice[i] = rng.random_range(-1.0..1.0);
            b_slice[i] = rng.random_range(-1.0..1.0);
        }

        ((a_slice), (b_slice))
    }

    #[test]
    fn test_dist_l2_float_turing_104() {
        let (a_data, b_data) = get_turing_test_data_f32_dim(104);
        let (a_slice, b_slice) = (
            F32Slice104(a_data.try_into().unwrap()),
            F32Slice104(b_data.try_into().unwrap()),
        );

        let distance: f32 = compare_two_vec::<f32>(104, Metric::L2, &a_slice.0, &b_slice.0);

        assert_abs_diff_eq!(
            distance as f64,
            no_vector_compare_f32_as_f64(&a_slice.0, &b_slice.0),
            epsilon = 1e-4f64
        );
    }

    #[test]
    fn test_dist_l2_float_turing_112() {
        let (a_data, b_data) = get_turing_test_data_f32_dim(112);
        let (a_slice, b_slice) = (
            F32Slice112(a_data.try_into().unwrap()),
            F32Slice112(b_data.try_into().unwrap()),
        );

        let distance: f32 = compare_two_vec::<f32>(112, Metric::L2, &a_slice.0, &b_slice.0);

        assert_abs_diff_eq!(
            distance as f64,
            no_vector_compare_f32_as_f64(&a_slice.0, &b_slice.0),
            epsilon = 1e-4f64
        );
    }

    #[test]
    fn test_dist_l2_float_turing_128() {
        let (a_data, b_data) = get_turing_test_data_f32_dim(128);
        let (a_slice, b_slice) = (
            F32Slice128(a_data.try_into().unwrap()),
            F32Slice128(b_data.try_into().unwrap()),
        );

        let distance: f32 = compare_two_vec::<f32>(128, Metric::L2, &a_slice.0, &b_slice.0);

        assert_abs_diff_eq!(
            distance as f64,
            no_vector_compare_f32_as_f64(&a_slice.0, &b_slice.0),
            epsilon = 1e-4f64
        );
    }

    #[test]
    fn test_dist_l2_float_turing_256() {
        let (a_data, b_data) = get_turing_test_data_f32_dim(256);
        let (a_slice, b_slice) = (
            F32Slice256(a_data.try_into().unwrap()),
            F32Slice256(b_data.try_into().unwrap()),
        );

        let distance: f32 = compare_two_vec::<f32>(256, Metric::L2, &a_slice.0, &b_slice.0);

        assert_abs_diff_eq!(
            distance as f64,
            no_vector_compare_f32_as_f64(&a_slice.0, &b_slice.0),
            epsilon = 1e-3f64
        );
    }

    #[test]
    fn test_dist_l2_float_turing_4096() {
        let (a_data, b_data) = get_turing_test_data_f32_dim(4096);
        let (a_slice, b_slice) = (
            F32Slice4096(a_data.try_into().unwrap()),
            F32Slice4096(b_data.try_into().unwrap()),
        );

        let distance: f32 = compare_two_vec::<f32>(4096, Metric::L2, &a_slice.0, &b_slice.0);

        assert_abs_diff_eq!(
            distance as f64,
            no_vector_compare_f32_as_f64(&a_slice.0, &b_slice.0),
            epsilon = 1e-2f64
        );
    }

    #[test]
    fn test_dist_ip_float_turing_112() {
        let (a_data, b_data) = get_turing_test_data_f32_dim(112);
        let (a_slice, b_slice) = (
            F32Slice112(a_data.try_into().unwrap()),
            F32Slice112(b_data.try_into().unwrap()),
        );

        let distance: f32 =
            compare_two_vec::<f32>(112, Metric::InnerProduct, &a_slice.0, &b_slice.0);

        assert_abs_diff_eq!(
            distance,
            reference::reference_innerproduct_f32_similarity(&a_slice.0, &b_slice.0).into_inner(),
            epsilon = 1e-4f32
        );
    }

    #[test]
    fn distance_test() {
        #[repr(C, align(32))]
        struct Vector32ByteAligned {
            v: [f32; 512],
        }

        // two vectors are allocated in the contiguous heap memory
        let two_vec = Box::new(Vector32ByteAligned {
            v: [
                69.02492, 78.84786, 63.125072, 90.90581, 79.2592, 70.81731, 3.0829668, 33.33287,
                20.777142, 30.147898, 23.681915, 42.553043, 12.602162, 7.3808074, 19.157589,
                65.6791, 76.44677, 76.89124, 86.40756, 84.70118, 87.86142, 16.126896, 5.1277637,
                95.11038, 83.946945, 22.735607, 11.548555, 59.51482, 24.84603, 15.573776, 78.27185,
                71.13179, 38.574017, 80.0228, 13.175261, 62.887978, 15.205181, 18.89392, 96.13162,
                87.55455, 34.179806, 62.920044, 4.9305916, 54.349373, 21.731495, 14.982187,
                40.262867, 20.15214, 36.61963, 72.450806, 55.565, 95.5375, 93.73356, 95.36308,
                66.30762, 58.0397, 18.951357, 67.11702, 43.043316, 30.65622, 99.85361, 2.5889993,
                27.844774, 39.72441, 46.463238, 71.303764, 90.45308, 36.390602, 63.344395,
                26.427078, 35.99528, 82.35505, 32.529175, 23.165905, 74.73179, 9.856939, 59.38126,
                35.714924, 79.81213, 46.704124, 24.47884, 36.01743, 0.46678782, 29.528152,
                1.8980742, 24.68853, 75.58984, 98.72279, 68.62601, 11.890173, 49.49361, 55.45572,
                72.71067, 34.107483, 51.357758, 76.400635, 81.32725, 66.45081, 17.848074,
                62.398876, 94.20444, 2.10886, 17.416393, 64.88253, 29.000723, 62.434315, 53.907238,
                70.51412, 78.70744, 55.181683, 64.45116, 23.419212, 53.68544, 43.506958, 46.89598,
                35.905994, 64.51397, 91.95555, 20.322979, 74.80128, 97.548744, 58.312725, 78.81985,
                31.911612, 14.445949, 49.85094, 70.87396, 40.06766, 7.129991, 78.48008, 75.21636,
                93.623604, 95.95479, 29.571129, 22.721554, 26.73875, 52.075504, 56.783104,
                94.65493, 61.778534, 85.72401, 85.369514, 29.922367, 41.410553, 94.12884,
                80.276855, 55.604828, 54.70947, 74.07216, 44.61955, 31.38113, 68.48596, 34.56782,
                14.424729, 48.204506, 9.675444, 32.01946, 92.32695, 36.292683, 78.31955, 98.05327,
                14.343918, 46.017002, 95.90888, 82.63626, 16.873539, 3.698051, 7.8042626,
                64.194405, 96.71023, 67.93692, 21.618402, 51.92182, 22.834194, 61.56986, 19.749891,
                55.31206, 38.29552, 67.57593, 67.145836, 38.92673, 94.95708, 72.38746, 90.70901,
                69.43995, 9.394085, 31.646872, 88.20112, 9.134722, 99.98214, 5.423498, 41.51995,
                76.94409, 77.373276, 3.2966614, 9.611201, 57.231106, 30.747868, 76.10228, 91.98308,
                70.893585, 0.9067178, 43.96515, 16.321218, 27.734184, 83.271835, 88.23312,
                87.16445, 5.556643, 15.627432, 58.547127, 93.6459, 40.539192, 49.124157, 91.13276,
                57.485855, 8.827019, 4.9690843, 46.511234, 53.91469, 97.71925, 20.135271,
                23.353004, 70.92099, 93.38748, 87.520134, 51.684677, 29.89813, 9.110392, 65.809204,
                34.16554, 93.398605, 84.58669, 96.409645, 9.876037, 94.767784, 99.21523, 1.9330144,
                94.92429, 75.12728, 17.218828, 97.89164, 35.476578, 77.629456, 69.573746,
                40.200542, 42.117836, 5.861628, 75.45282, 82.73633, 0.98086596, 77.24894,
                11.248695, 61.070026, 52.692616, 80.5449, 80.76036, 29.270136, 67.60252, 48.782394,
                95.18851, 83.47162, 52.068756, 46.66002, 90.12216, 15.515327, 33.694042, 96.963036,
                73.49627, 62.805485, 44.715607, 59.98627, 3.8921833, 37.565327, 29.69184,
                39.429665, 83.46899, 44.286453, 21.54851, 56.096413, 18.169249, 5.214751,
                14.691341, 99.779335, 26.32643, 67.69903, 36.41243, 67.27333, 12.157213, 96.18984,
                2.438283, 78.14289, 0.14715195, 98.769, 53.649532, 21.615898, 39.657497, 95.45616,
                18.578386, 71.47976, 22.348118, 17.85519, 6.3717127, 62.176777, 22.033644,
                23.178005, 79.44858, 89.70233, 37.21273, 71.86182, 21.284317, 52.908623, 30.095518,
                63.64478, 77.55823, 80.04871, 15.133011, 30.439043, 70.16561, 4.4014096, 89.28944,
                26.29093, 46.827854, 11.764729, 61.887516, 47.774887, 57.19503, 59.444664,
                28.592825, 98.70386, 1.2497544, 82.28431, 46.76423, 83.746124, 53.032673, 86.53457,
                99.42168, 90.184, 92.27852, 9.059965, 71.75723, 70.45299, 10.924053, 68.329704,
                77.27232, 6.677854, 75.63629, 57.370533, 17.09031, 10.554659, 99.56178, 37.53221,
                72.311104, 75.7565, 65.2042, 36.096478, 64.69502, 38.88497, 64.33723, 84.87812,
                66.84958, 8.508932, 79.134, 83.431015, 66.72124, 61.801838, 64.30524, 37.194263,
                77.94725, 89.705185, 23.643505, 19.505919, 48.40264, 43.01083, 21.171177,
                18.717121, 10.805857, 69.66983, 77.85261, 57.323063, 3.28964, 38.758026, 5.349946,
                7.46572, 57.485138, 30.822384, 33.9411, 95.53746, 65.57723, 42.1077, 28.591347,
                11.917269, 5.031073, 31.835615, 19.34116, 85.71027, 87.4516, 1.3798475, 70.70583,
                51.988052, 45.217144, 14.308596, 54.557167, 86.18323, 79.13666, 76.866745,
                46.010685, 79.739235, 44.667603, 39.36416, 72.605896, 73.83187, 13.137412,
                6.7911267, 63.952374, 10.082436, 86.00318, 99.760376, 92.84948, 63.786434,
                3.4429908, 18.244314, 75.65299, 14.964747, 70.126366, 80.89449, 91.266655,
                96.58798, 46.439327, 38.253975, 87.31036, 21.093178, 37.19671, 58.28973, 9.75231,
                12.350321, 25.75115, 87.65073, 53.610504, 36.850048, 18.66356, 94.48941, 83.71898,
                44.49315, 44.186737, 19.360733, 84.365974, 46.76272, 44.924366, 50.279808,
                54.868866, 91.33004, 18.683397, 75.13282, 15.070831, 47.04839, 53.780903,
                26.911152, 74.65651, 57.659935, 25.604189, 37.235474, 65.39667, 53.952206,
                40.37131, 59.173275, 96.00756, 54.591274, 10.787476, 69.51549, 31.970142,
                25.408005, 55.972492, 85.01888, 97.48981, 91.006134, 28.98619, 97.151276,
                34.388496, 47.498177, 11.985874, 64.73775, 33.877014, 13.370312, 34.79146,
                86.19321, 15.019405, 94.07832, 93.50433, 60.168625, 50.95409, 38.27827, 47.458614,
                32.83715, 69.54998, 69.0361, 84.1418, 34.270298, 74.23852, 70.707466, 78.59845,
                9.651399, 24.186779, 58.255756, 53.72362, 92.46477, 97.75528, 20.257462, 30.122698,
                50.41517, 28.156603, 42.644154,
            ],
        });

        let distance: f32 = compare::<f32>(256, Metric::L2, &two_vec.v);

        assert_eq!(distance, 429141.2);
    }

    fn compare<T>(dim: usize, metric: Metric, v: &[T]) -> f32
    where
        T: DistanceProvider<T>,
    {
        let distance_comparer = T::distance_comparer(metric, Some(dim));
        distance_comparer.call(&v[..dim], &v[dim..])
    }

    pub fn compare_two_vec<T>(dim: usize, metric: Metric, v1: &[T], v2: &[T]) -> f32
    where
        T: DistanceProvider<T>,
    {
        let distance_comparer = T::distance_comparer(metric, Some(dim));
        distance_comparer.call(&v1[..dim], &v2[..dim])
    }
}

#[cfg(test)]
mod distance_provider_f16_tests {
    use approx::assert_abs_diff_eq;

    use super::{distance_provider_f32_tests::get_turing_test_data_f32_dim, *};
    use crate::{
        distance::distance_provider::distance_provider_f32_tests::compare_two_vec,
        test_util::no_vector_compare_f16_as_f64,
    };

    #[repr(C, align(32))]
    pub struct F16Slice112([f16; 112]);
    #[repr(C, align(32))]
    pub struct F16Slice104([f16; 104]);
    #[repr(C, align(32))]
    pub struct F16Slice128([f16; 128]);
    #[repr(C, align(32))]
    pub struct F16Slice256([f16; 256]);
    #[repr(C, align(32))]
    pub struct F16Slice4096([f16; 4096]);

    fn get_turing_test_data_f16_dim(dim: usize) -> (Vec<f16>, Vec<f16>) {
        let (a_slice, b_slice) = get_turing_test_data_f32_dim(dim);
        let a_data = a_slice.iter().map(|x| f16::from_f32(*x)).collect();
        let b_data = b_slice.iter().map(|x| f16::from_f32(*x)).collect();
        (a_data, b_data)
    }

    #[test]
    fn test_dist_l2_f16_turing_112() {
        // two vectors are allocated in the contiguous heap memory
        let (a_data, b_data) = get_turing_test_data_f16_dim(112);
        let (a_slice, b_slice) = (
            F16Slice112(a_data.try_into().unwrap()),
            F16Slice112(b_data.try_into().unwrap()),
        );

        let distance: f32 = compare_two_vec::<f16>(112, Metric::L2, &a_slice.0, &b_slice.0);

        // Note the variance between the full 32 bit precision and the 16 bit precision
        assert_abs_diff_eq!(
            distance as f64,
            no_vector_compare_f16_as_f64(&a_slice.0, &b_slice.0),
            epsilon = 1e-3f64
        );
    }

    #[test]
    fn test_dist_l2_f16_turing_104() {
        // two vectors are allocated in the contiguous heap memory
        let (a_data, b_data) = get_turing_test_data_f16_dim(104);
        let (a_slice, b_slice) = (
            F16Slice104(a_data.try_into().unwrap()),
            F16Slice104(b_data.try_into().unwrap()),
        );

        let distance: f32 = compare_two_vec::<f16>(104, Metric::L2, &a_slice.0, &b_slice.0);

        // Note the variance between the full 32 bit precision and the 16 bit precision
        assert_abs_diff_eq!(
            distance as f64,
            no_vector_compare_f16_as_f64(&a_slice.0, &b_slice.0),
            epsilon = 1e-3f64
        );
    }

    #[test]
    fn test_dist_l2_f16_turing_256() {
        // two vectors are allocated in the contiguous heap memory
        let (a_data, b_data) = get_turing_test_data_f16_dim(256);
        let (a_slice, b_slice) = (
            F16Slice256(a_data.try_into().unwrap()),
            F16Slice256(b_data.try_into().unwrap()),
        );

        let distance: f32 = compare_two_vec::<f16>(256, Metric::L2, &a_slice.0, &b_slice.0);

        // Note the variance between the full 32 bit precision and the 16 bit precision
        assert_abs_diff_eq!(
            distance as f64,
            no_vector_compare_f16_as_f64(&a_slice.0, &b_slice.0),
            epsilon = 1e-3f64
        );
    }

    #[test]
    fn test_dist_l2_f16_turing_128() {
        // two vectors are allocated in the contiguous heap memory
        let (a_data, b_data) = get_turing_test_data_f16_dim(128);
        let (a_slice, b_slice) = (
            F16Slice128(a_data.try_into().unwrap()),
            F16Slice128(b_data.try_into().unwrap()),
        );

        let distance: f32 = compare_two_vec::<f16>(128, Metric::L2, &a_slice.0, &b_slice.0);

        // Note the variance between the full 32 bit precision and the 16 bit precision
        assert_abs_diff_eq!(
            distance as f64,
            no_vector_compare_f16_as_f64(&a_slice.0, &b_slice.0),
            epsilon = 1e-3f64
        );
    }

    #[test]
    fn test_dist_l2_f16_turing_4096() {
        // two vectors are allocated in the contiguous heap memory
        let (a_data, b_data) = get_turing_test_data_f16_dim(4096);
        let (a_slice, b_slice) = (
            F16Slice4096(a_data.try_into().unwrap()),
            F16Slice4096(b_data.try_into().unwrap()),
        );

        let distance: f32 = compare_two_vec::<f16>(4096, Metric::L2, &a_slice.0, &b_slice.0);

        // Note the variance between the full 32 bit precision and the 16 bit precision
        assert_abs_diff_eq!(
            distance as f64,
            no_vector_compare_f16_as_f64(&a_slice.0, &b_slice.0),
            epsilon = 1e-2f64
        );
    }

    #[test]
    fn test_dist_l2_f16_produces_nan_distance_for_infinity_vectors() {
        let a_data = vec![f16::INFINITY; 384];
        let b_data = vec![f16::INFINITY; 384];

        let distance: f32 = compare_two_vec::<f16>(384, Metric::L2, &a_data, &b_data);
        assert!(distance.is_nan());
    }
}
