/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::vectors::{DataMutRef, MinMaxCompensation, MinMaxIP, MinMaxL2Squared};
use core::f32;
use diskann_utils::views::MutDenseData;

use crate::{
    algorithms::Transform,
    alloc::{GlobalAllocator, ScopedAllocator},
    bits::{Representation, Unsigned},
    minmax::{vectors::FullQueryMeta, FullQuery, MinMaxCosine, MinMaxCosineNormalized},
    num::Positive,
    scalar::{bit_scale, InputContainsNaN},
    AsFunctor, CompressInto,
};

/// Recall that from the module-level documentation, MinMaxQuantizer, quantizes X
/// into `n` bit vectors as follows  -
/// ```math
/// X' = round((X - s) * (2^n - 1) / c).clamp(0, 2^n - 1))
/// ```
/// where `s` is a shift value and `c` is a scaling parameter computed from the range of values.
///
/// For most bit widths (>1), given a positive scaling parameter `grid_scale : f32`,
/// these are computed as:
/// ```math
/// - m = (max_i X[i] + min_i X[i]) / 2.0
/// - w = max_i X[i] - min_i X[i]
///
/// - s = m - w * grid_scale
/// - c = 2 * w * grid_scale
///
/// where `grid_scale` is an input to the quantizer.
/// ```
/// For 1-bit quantization, to avoid outliers, `s` and `c` are derived differently:
/// - Values are first split into two groups: those below and above the mean.
/// - `s` is the average of values below the mean.
/// - `c` is the difference between the average of values above the mean and `s`.
///
/// See [`MinMaxCompensation`] for notation.
/// We have then that
/// ```math
/// X = X' * (c / (2^n - 1)) + s
///          --------------    -
///                 |          |
///                ax          bx
/// ```
pub struct MinMaxQuantizer {
    /// Support for different strategies of pre-transforming vectors before applying compression.
    /// See [`Transform`] for more details on supported types. The input dimension of vectors
    /// to the quantizer is derived from `transform.input_dim()`.
    transform: Transform<GlobalAllocator>,

    /// Scaling parameter used to scale the range (min, max) in order to avoid outliers.
    /// The input must be a positive value. In general, any value between [0.8, 1] does well.
    grid_scale: Positive<f32>,
}

impl MinMaxQuantizer {
    /// Instantiates a new quantizer with specific transform.
    pub fn new(transform: Transform<GlobalAllocator>, grid_scale: Positive<f32>) -> Self {
        Self {
            transform,
            grid_scale,
        }
    }

    /// Input dimension of vectors to quantizer.
    pub fn dim(&self) -> usize {
        self.transform.input_dim()
    }

    /// Output dimension of vectors after applying transform.
    ///
    /// Output storage vectors should use this dimension instead of `self.dim()` because
    /// in general, the output dim **may** be different from the input dimension.
    pub fn output_dim(&self) -> usize {
        self.transform.output_dim()
    }

    /// Outputs the minimum and maximum value of the range of values
    /// for an input vector `vec`. The function cases based on the
    /// intended number of bits `NBITS` per dimension.
    ///
    /// * `1-bit` - In order to avoid outlier values, the range
    ///   is defined by taking the values larger and smaller than
    ///   the numeric mean, and then taking the respective means of
    ///   each of these sets as the `max` and `min`.
    ///
    /// * `N-bits` - Computes the `min` and `max` of the vector values.
    ///
    /// # Returns
    ///
    /// * `(m - w * g, m + w * g)` - the lower and upper end of the range, where,
    ///   `m = (max + min) / 2.0`, `w = (max - min) / 2.0`, and `g = self.grid_scale`.
    fn get_range<const NBITS: usize>(&self, vec: &[f32]) -> (f32, f32) {
        let (min, max) = match NBITS {
            1 => {
                let (mut min, mut min_count) = (0.0f32, 0.0f32);
                let (mut max, mut max_count) = (0.0f32, 0.0f32);

                let mean = vec.iter().sum::<f32>() / (vec.len() as f32);

                vec.iter().for_each(|x| {
                    let m = f32::from((*x < mean) as u8);
                    min += m * x;
                    min_count += m;
                    max += (1.0 - m) * x;
                    max_count += 1.0 - m;
                });

                ((min / min_count).min(mean), (max / max_count).max(mean))
            }
            _ => {
                vec // Using `f32::NAN` since [`core::f32::min`] and `max` output the other value if one of them is NAN .
                    .iter()
                    .fold((f32::NAN, f32::NAN), |(cmin, cmax), &e| {
                        (cmin.min(e), cmax.max(e))
                    })
            }
        };

        let width = (max - min) / 2.0;
        let mid = min + width;

        (
            mid - width * self.grid_scale.into_inner(),
            mid + width * self.grid_scale.into_inner(),
        )
    }

    fn compress<const NBITS: usize, T>(
        &self,
        from: &[T],
        mut into: DataMutRef<'_, NBITS>,
    ) -> Result<L2Loss, InputContainsNaN>
    where
        T: Copy + Into<f32>,
        Unsigned: Representation<NBITS>,
    {
        let mut into_vec = into.vector_mut();

        assert_eq!(from.len(), self.dim());
        assert_eq!(self.output_dim(), into_vec.len());

        let domain = Unsigned::domain_const::<NBITS>();
        let domain_min = *domain.start() as f32;
        let domain_max = *domain.end() as f32;

        let mut vec = vec![f32::default(); self.output_dim()];

        // We know vec.len() == self.output_dim() and `from.len() == self.dim`
        #[allow(clippy::unwrap_used)]
        self.transform
            .transform_into(
                &mut vec,
                &from.iter().map(|&x| x.into()).collect::<Vec<f32>>(),
                ScopedAllocator::global(),
            )
            .unwrap();

        let (min, max) = self.get_range::<NBITS>(&vec);

        let inverse_scale = (max - min).max(1e-8) / bit_scale::<NBITS>(); // To avoid NaN. This is ONLY possible if the vector is all the same value.
        let mut norm_squared: f32 = 0.0;
        let mut code_sum: f32 = 0.0;
        let mut loss: f32 = 0.0;

        let mut nan_check = false;

        vec.iter().enumerate().for_each(|(i, &v)| {
            nan_check |= v.is_nan();

            let code = ((v - min) / inverse_scale)
                .clamp(domain_min, domain_max)
                .round();

            let v_r = (code * inverse_scale) + min; // reconstructed value for `v`.
            norm_squared += v_r * v_r;
            code_sum += code;
            loss += (v_r - v).powi(2);

            //SAFETY: we checked that the lengths of `from` and `into_vec` are the same.
            unsafe {
                into_vec.set_unchecked(i, code as u8);
            }
        });

        let meta = MinMaxCompensation {
            dim: self.output_dim() as u32,
            b: min,
            a: inverse_scale,
            n: inverse_scale * code_sum,
            norm_squared,
        };

        into.set_meta(meta);

        if nan_check {
            Err(InputContainsNaN)
        } else {
            Ok(match Positive::new(loss) {
                Ok(p) => L2Loss::Positive(p),
                Err(_) => L2Loss::Zero,
            })
        }
    }
}

/////////////////
// Compression //
/////////////////

/// A struct defining euclidean loss from quantization.
///
/// For an input vector `x` and its representation `x'`,
/// this is supposed to store `||x - x'||^2`.
#[derive(Clone, Copy, Debug)]
pub enum L2Loss {
    Zero,
    Positive(Positive<f32>),
}

impl L2Loss {
    /// Euclidean loss as a `f32` value
    pub fn as_f32(&self) -> f32 {
        match self {
            L2Loss::Zero => 0.0,
            L2Loss::Positive(p) => p.into_inner(),
        }
    }
}

impl<const NBITS: usize, T> CompressInto<&[T], DataMutRef<'_, NBITS>> for MinMaxQuantizer
where
    T: Copy + Into<f32>,
    Unsigned: Representation<NBITS>,
{
    type Error = InputContainsNaN;

    type Output = L2Loss;

    /// Compress the input vector `from` into a mut ref of Data `to`.
    ///
    /// This method computes and stores the compensation coefficients required for computing
    /// distances correctly.
    ///
    /// # Error
    ///
    /// Returns an error if the input contains `NaN`.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * `from.len() != self.dim()`: Vector to be compressed must have the same
    ///   dimensionality as the quantizer.
    /// * `to.vector().len() != self.output_dim()`: Compressed vector must have the same dimensionality
    ///   as the quantizer.
    fn compress_into(&self, from: &[T], to: DataMutRef<'_, NBITS>) -> Result<L2Loss, Self::Error> {
        self.compress::<NBITS, T>(from, to)
    }
}

impl<T> CompressInto<&[T], &mut FullQuery> for MinMaxQuantizer
where
    T: Copy + Into<f32>,
{
    type Error = InputContainsNaN;

    type Output = ();

    /// Compress the input vector `from` into a mutable reference for a [`FullQuery`] `to`.
    ///
    /// This method simply applies the transformation to the input without
    /// any compression.
    ///
    /// # Error
    ///
    /// Returns an error if the input contains `NaN`.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * `from.len() != self.dim()`: Vector to be compressed must have the same
    ///   dimensionality as the quantizer.
    /// * `to.len() != self.output_dim()`: Compressed vector must have the same dimensionality
    ///   as the quantizer.
    fn compress_into(&self, from: &[T], to: &mut FullQuery) -> Result<(), Self::Error> {
        assert_eq!(from.len(), self.dim());
        assert_eq!(self.output_dim(), to.len());

        // Transform the input vector and return error if it contains NaN
        let from: Vec<f32> = from.iter().map(|&x| x.into()).collect();
        if from.iter().any(|x| x.is_nan()) {
            return Err(InputContainsNaN);
        }

        // We know vec.len() == self.output_dim() and `from.len() == self.dim`
        #[allow(clippy::unwrap_used)]
        self.transform
            .transform_into(to.data.as_mut_slice(), &from, ScopedAllocator::global())
            .unwrap();

        let norm_squared = to.data.iter().map(|x| *x * *x).sum::<f32>();
        let sum = to.data.iter().sum::<f32>();

        to.meta = FullQueryMeta { norm_squared, sum };

        Ok(())
    }
}

///////////////////////
// Distance Functors //
///////////////////////

macro_rules! impl_functor {
    ($dist:ident) => {
        impl AsFunctor<$dist> for MinMaxQuantizer {
            // no need to do any work here.
            fn as_functor(&self) -> $dist {
                $dist
            }
        }
    };
}

impl_functor!(MinMaxIP);
impl_functor!(MinMaxL2Squared);
impl_functor!(MinMaxCosine);
impl_functor!(MinMaxCosineNormalized);

///////////
// Tests //
///////////
#[cfg(test)]
mod minmax_quantizer_tests {
    use std::num::NonZeroUsize;

    use diskann_utils::{Reborrow, ReborrowMut};
    use diskann_vector::{distance::SquaredL2, PureDistanceFunction};
    use rand::{
        distr::{Distribution, Uniform},
        rngs::StdRng,
        SeedableRng,
    };

    use super::*;
    use crate::{
        algorithms::transforms::NullTransform,
        minmax::vectors::{Data, DataRef},
    };

    fn reconstruct_minmax<const NBITS: usize>(v: DataRef<'_, NBITS>) -> Vec<f32>
    where
        Unsigned: Representation<NBITS>,
    {
        (0..v.len())
            .map(|i| {
                let m = v.meta();
                v.vector().get(i).unwrap() as f32 * m.a + m.b
            })
            .collect()
    }

    fn test_quantizer_encoding_random<const NBITS: usize>(
        dim: usize,
        rng: &mut StdRng,
        relative_err: f32,
        scale: f32,
    ) where
        Unsigned: Representation<NBITS>,
        MinMaxQuantizer: for<'a, 'b> CompressInto<&'a [f32], DataMutRef<'b, NBITS>, Output = L2Loss>
            + for<'a, 'b> CompressInto<&'a [f32], &'b mut FullQuery, Output = ()>,
    {
        let distribution = Uniform::new_inclusive::<f32, f32>(-1.0, 1.0).unwrap();

        let quantizer = MinMaxQuantizer::new(
            Transform::Null(NullTransform::new(NonZeroUsize::new(dim).unwrap())),
            Positive::new(scale).unwrap(),
        );

        assert_eq!(quantizer.dim(), dim);

        let vector: Vec<f32> = distribution.sample_iter(rng).take(dim).collect();

        let mut encoded = Data::new_boxed(dim);
        let loss = quantizer
            .compress_into(&*vector, encoded.reborrow_mut())
            .unwrap();

        let reconstructed = reconstruct_minmax::<NBITS>(encoded.reborrow());
        assert_eq!(reconstructed.len(), dim);

        let reconstruction_error: f32 = SquaredL2::evaluate(&*vector, &*reconstructed);
        let norm = vector.iter().map(|x| x * x).sum::<f32>();
        assert!(
                (reconstruction_error / norm) <= relative_err,
                "Expected vector : {:?} to be reconstructed within error {} but instead got : {:?}, with error {} for dim : {}",
                &vector,
                relative_err,
                &reconstructed,
                reconstruction_error / norm,
                dim,
            );

        assert!((loss.as_f32() - reconstruction_error) <= 1e-4);

        let expected_code_sum = (0..dim)
            .map(|i| encoded.vector().get(i).unwrap() as f32)
            .sum::<f32>();
        let code_sum = encoded.reborrow().meta().n / encoded.reborrow().meta().a;
        assert!(
            (code_sum - expected_code_sum).abs() <= 2e-5 * (dim as f32),
            "Encoded vector with dim : {dim} is {:?}, got error : {} for vector : {:?}",
            encoded.reborrow(),
            (code_sum - expected_code_sum).abs(),
            &vector,
        );
        let recon_norm_sq = reconstructed.iter().map(|x| x * x).sum::<f32>();
        assert!((encoded.reborrow().meta().norm_squared - recon_norm_sq).abs() <= 1e-3);

        // FullQuery
        let mut f = FullQuery::empty(dim);
        quantizer
            .compress_into(vector.as_slice(), f.reborrow_mut())
            .unwrap();

        f.data
            .iter()
            .enumerate()
            .zip(vector.iter())
            .for_each(|((i, x), y)| {
                assert!(
                    (*x - *y).abs() < 1e-10,
                    "Full Query did not compress dimension {i} with value {} correctly, got {} instead.",
                    *y,
                    *x,
                )
            });

        assert!(
            (f.meta.norm_squared - norm).abs() < 1e-10,
            "Full Query norm in meta should be {norm} but instead got {}",
            f.meta.norm_squared
        );

        let sum = vector.iter().sum::<f32>();
        assert!(
            (f.meta.sum - sum) < 1e-10,
            "Full Query norm in meta should be {sum} but instead got {}",
            f.meta.sum
        );
    }

    cfg_if::cfg_if! {
        if #[cfg(miri)] {
            // The max dim does not need to be as high for `CompensatedVectors` because they
            // defer their distance function implementation to `BitSlice`, which is more
            // heavily tested.
            const TRIALS: usize = 2;
        } else {
            const TRIALS: usize = 10;
        }
    }

    macro_rules! test_minmax_quantizer_encoding {
        ($name:ident, $dim:literal, $nbits:literal, $seed:literal, $err:expr) => {
            #[test]
            fn $name() {
                let mut rng = StdRng::seed_from_u64($seed);
                let scales = [1.0, 1.1, 0.9];
                for (s, e) in scales.iter().zip($err) {
                    for d in 10..$dim {
                        for _ in 0..TRIALS {
                            test_quantizer_encoding_random::<$nbits>(d, &mut rng, e, *s);
                        }
                    }
                }
            }
        };
    }
    test_minmax_quantizer_encoding!(
        test_minmax_encoding_1bit,
        100,
        1,
        0xa32d5658097a1c35,
        vec![0.5, 0.5, 0.5]
    );
    test_minmax_quantizer_encoding!(
        test_minmax_encoding_2bit,
        100,
        2,
        0xf60c0c8d1aadc126,
        vec![0.5, 0.5, 0.5]
    );
    test_minmax_quantizer_encoding!(
        test_minmax_encoding_4bit,
        100,
        4,
        0x09fa14c42a9d7d98,
        vec![1.0e-2, 1.0e-2, 3.0e-2]
    );
    test_minmax_quantizer_encoding!(
        test_minmax_encoding_8bit,
        100,
        8,
        0xaedf3d2a223b7b77,
        vec![2.0e-3, 2.0e-3, 7.0e-3]
    );

    macro_rules! expand_to_bitrates {
        ($name:ident, $func:ident) => {
            #[test]
            fn $name() {
                $func::<1>();
                $func::<2>();
                $func::<4>();
                $func::<8>();
            }
        };
    }

    /// Tests the edge case where min == max but both are non-zero.
    fn test_all_same_value_vector<const NBITS: usize>()
    where
        Unsigned: Representation<NBITS>,
        MinMaxQuantizer:
            for<'a, 'b> CompressInto<&'a [f32], DataMutRef<'b, NBITS>, Output = L2Loss>,
    {
        let dim = 30;
        let quantizer = MinMaxQuantizer::new(
            Transform::Null(NullTransform::new(NonZeroUsize::new(dim).unwrap())),
            Positive::new(1.0).unwrap(),
        );
        let constant_value = 42.5f32;
        let vector = vec![constant_value; dim];

        let mut encoded = Data::new_boxed(dim);
        let result = quantizer.compress_into(&vector, encoded.reborrow_mut());

        assert!(
            result.is_ok(),
            "Constant-value vector should compress successfully"
        );

        assert!(result.unwrap().as_f32().abs() <= 1e-6);

        // Reconstruction should yield the original constant value (approximately)
        let reconstructed = reconstruct_minmax(encoded.reborrow());
        for &val in &reconstructed {
            assert!(
                (val - constant_value).abs() < 1e-3,
                "Reconstructed value {} should be close to original {}. Compressed vector is {:?}",
                val,
                constant_value,
                encoded.meta(),
            );
        }
    }

    /// This tests boundary conditions in the quantization logic.
    fn test_two_distinct_values<const NBITS: usize>()
    where
        Unsigned: Representation<NBITS>,
        MinMaxQuantizer:
            for<'a, 'b> CompressInto<&'a [f32], DataMutRef<'b, NBITS>, Output = L2Loss>,
    {
        let dim = 20;
        let quantizer = MinMaxQuantizer::new(
            Transform::Null(NullTransform::new(NonZeroUsize::new(dim).unwrap())),
            Positive::new(1.0).unwrap(),
        );

        let val1 = -10.0f32;
        let val2 = 15.0f32;
        let mut vector = vec![val1; dim];
        // Make half the vector the second value
        for i in vector.iter_mut().skip(dim) {
            *i = val2;
        }

        let mut encoded = Data::new_boxed(dim);
        let result = quantizer.compress_into(&vector, encoded.reborrow_mut());

        assert!(
            result.is_ok(),
            "Two-value vector should compress successfully"
        );

        assert!(result.unwrap().as_f32().abs() <= 1e-6);

        // Verify that only two distinct codes are used
        let mut codes_used = std::collections::HashSet::new();
        for i in 0..dim {
            codes_used.insert(encoded.vector().get(i).unwrap());
        }

        // For most bit widths, we should see exactly 2 codes (min and max of domain)
        if NBITS > 1 {
            assert!(
                codes_used.len() <= 2,
                "Should use at most 2 distinct codes for 2-value input, but used: {:?}",
                codes_used
            );
        }

        // Verify reconstruction maintains the two-value structure approximately
        let reconstructed = reconstruct_minmax(encoded.reborrow());
        for ((i, val), v) in reconstructed.into_iter().enumerate().zip(&vector) {
            // Round to nearest 0.1 to account for quantization error
            assert!(
                (val - v).abs() < 1e-4,
                "Reconstructed value in dim : {i} is {val}, when it should be {v}."
            );
        }
    }

    /// Verifies that NaN values in the input cause the expected error but
    /// dimension in meta is correctly set.
    fn test_nan_input_error<const NBITS: usize>()
    where
        Unsigned: Representation<NBITS>,
        MinMaxQuantizer:
            for<'a, 'b> CompressInto<&'a [f32], DataMutRef<'b, NBITS>, Output = L2Loss>,
    {
        let dim = 100;
        let quantizer = MinMaxQuantizer::new(
            Transform::Null(NullTransform::new(NonZeroUsize::new(dim).unwrap())),
            Positive::new(1.0).unwrap(),
        );

        // Test vector with NaN in the middle.
        let mut vector_nan = vec![1.0f32; dim];
        vector_nan[33] = f32::NAN;
        let mut encoded = Data::new_boxed(dim);
        let result = quantizer.compress_into(&vector_nan, encoded.reborrow_mut());
        assert!(result.is_err(), "Vector with NaN should cause an error");

        let meta = encoded.meta();
        assert_eq!(meta.dim as usize, dim);
    }

    expand_to_bitrates!(all_same_values_vector, test_all_same_value_vector);
    expand_to_bitrates!(two_distinct_values, test_two_distinct_values);
    expand_to_bitrates!(nan_input_error, test_nan_input_error);

    /// Verifies that providing a vector with wrong dimensionality causes a panic.
    #[test]
    #[should_panic(expected = "assertion `left == right` failed\n  left: 15\n right: 10")]
    fn test_dimension_mismatch_panic()
    where
        Unsigned: Representation<8>,
        MinMaxQuantizer: for<'a, 'b> CompressInto<&'a [f32], DataMutRef<'b, 8>, Output = L2Loss>,
    {
        let expected_dim = 10;
        let quantizer = MinMaxQuantizer::new(
            Transform::Null(NullTransform::new(NonZeroUsize::new(expected_dim).unwrap())),
            Positive::new(1.0).unwrap(),
        );

        // Provide vector with wrong dimension
        let wrong_vector = vec![1.0f32; expected_dim + 5]; // Too many dimensions
        let mut encoded = Data::new_boxed(expected_dim);

        // This should panic due to assertion in compress_into
        let _ = quantizer.compress_into(&wrong_vector, encoded.reborrow_mut());
    }
}
