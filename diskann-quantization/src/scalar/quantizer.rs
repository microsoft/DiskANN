/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_vector::{MathematicalValue, PureDistanceFunction};
use thiserror::Error;

use super::{
    bit_scale, inverse_bit_scale,
    vectors::{
        CompensatedCosineNormalized, CompensatedIP, CompensatedSquaredL2, Compensation,
        MutCompensatedVectorRef,
    },
};
use crate::{
    bits::{MutBitSlice, PermutationStrategy, Representation, Unsigned},
    AsFunctor, CompressInto,
};

/// A central parameter collection for a scalar quantization schema.
///
/// # Example
///
/// An self-contained end-to-end example containing training, compression, and distance
/// computations is shown below.
///
/// ```rust
/// use diskann_quantization::{
///     AsFunctor, CompressInto,
///     distances,
///     num::Positive, bits::MutBitSlice,
///     scalar::{
///         self,
///         ScalarQuantizer,
///         train::ScalarQuantizationParameters,
///         CompensatedVector, MutCompensatedVectorRef,
///         CompensatedIP, CompensatedSquaredL2,
///     }
/// };
/// use diskann_utils::{views::Matrix, Reborrow, ReborrowMut};
/// use diskann_vector::DistanceFunction;
///
/// // A small training set consisting of two 5-dimensional vectors.
/// let mut data = Matrix::<f32>::new(0.0, 2, 5);
/// data.row_mut(0).copy_from_slice(&[-1.0, -1.0, -1.0, -1.0, -1.0]);
/// data.row_mut(1).copy_from_slice(&[1.0, 1.0, 1.0, 1.0, 1.0]);
///
/// let trainer = ScalarQuantizationParameters::new(Positive::new(1.0).unwrap());
/// let quantizer: ScalarQuantizer = trainer.train(data.as_view());
///
/// // The dimension of the quantizer is based on the dimension of the training data.
/// assert_eq!(quantizer.dim(), data.ncols());
///
/// // Compress the two input vectors.
/// // For one vector, we will use the "boxed" API. The other we will construct "manually".
///
/// // Boxed API
/// let mut c0 = CompensatedVector::<8>::new_boxed(data.ncols());
///
/// // Manual construction.
/// let mut buffer: Vec<u8> = vec![0; c0.vector().bytes()];
/// let mut compensation = scalar::Compensation(0.0);
/// let mut c1 = MutCompensatedVectorRef::new(
///     MutBitSlice::new(buffer.as_mut_slice(), data.ncols()).unwrap(),
///     &mut compensation
/// );
///
/// quantizer.compress_into(data.row(0), c0.reborrow_mut()).unwrap();
/// quantizer.compress_into(data.row(1), c1.reborrow_mut()).unwrap();
///
/// // Compute inner product.
/// let ip: CompensatedIP = quantizer.as_functor();
/// let distance: distances::Result<f32> = ip.evaluate_similarity(c0.reborrow(), c1.reborrow());
///
/// // The inner product computation to `f32` is the same as a SimilarityScore and is
/// // therefore negative of the mathematical value.
/// assert!((distance.unwrap() - 5.0).abs() < 0.00001);
///
/// // Compute squared eudlicean distance.
/// let l2: CompensatedSquaredL2 = quantizer.as_functor();
/// let distance: distances::Result<f32> = l2.evaluate_similarity(c0.reborrow(), c1.reborrow());
/// assert!((distance.unwrap() - 20.0).abs() < 0.00001);
/// ```
#[derive(Clone, Debug)]
pub struct ScalarQuantizer {
    /// The scaling parameter applied to each vector component.
    scale: f32,

    /// The amount each data point is shifted.
    ///
    /// This is computed as the dataset mean subtracted by the scaling parameter.
    /// The additional subtraction is needed to ensure we can map encodings into an unsigned
    /// integer.
    ///
    /// For datasets that have components with non-zero mean, this can greatly improve the
    /// quality of quantization by decreasing the observed dynamic range across all vector
    /// component, but this shift must be applied regardless of whether or not the mean
    /// is calculated.
    shift: Vec<f32>,

    /// The square norm of the shift.
    /// This quantity is useful when computing dot-products.
    shift_square_norm: f32,

    /// When processing queries, it may be beneficial to modify the query norm to match the
    /// dataset norm.
    ///
    /// This is only applicable when `InnerProduct` and `Cosine` are used, but serves to
    /// move the query into the dynamic range of the quantization.
    mean_norm: Option<f32>,
}

impl ScalarQuantizer {
    /// Construct a new scalar quantizer.
    pub fn new(scale: f32, shift: Vec<f32>, mean_norm: Option<f32>) -> Self {
        let shift_square_norm: MathematicalValue<f32> =
            diskann_vector::distance::InnerProduct::evaluate(&*shift, &*shift);

        Self {
            scale,
            shift,
            shift_square_norm: shift_square_norm.into_inner(),
            mean_norm,
        }
    }

    /// Return the number dimensions this ScalarQuantizer has been trained for.
    pub fn dim(&self) -> usize {
        self.shift.len()
    }

    /// Return the scaling coefficient.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Return the square norm of the dataset shift.
    pub fn shift_square_norm(&self) -> f32 {
        self.shift_square_norm
    }

    /// Return the per-dimension shift vector.
    ///
    /// This vector is meant to accomplish two goals:
    ///
    /// 1. Centers the data around the training dataset mean.
    /// 2. Offsets each dimension into a range that can be encoded in unsigned values.
    pub fn shift(&self) -> &[f32] {
        &self.shift
    }

    /// Return the average norm of vectors in the training set.
    pub fn mean_norm(&self) -> Option<f32> {
        self.mean_norm
    }

    /// Rescale the argument so it has the average norm of the training set.
    ///
    /// This can be used to help with compression queries that come from a different
    /// distribution when the norm of the query may be safely discarded for purposes of
    /// distance computations.
    ///
    /// This operation can fail is the mean norm was not computed during training.
    pub fn rescale(&self, x: &mut [f32]) -> Result<(), MeanNormMissing> {
        match self.mean_norm {
            Some(mean_norm) => {
                rescale(x, mean_norm);
                Ok(())
            }
            None => Err(MeanNormMissing),
        }
    }

    /// An private compression method used by the implementations of `CompressInto`.
    ///
    /// This function works by shifting each dimension by `self.shift`, dividing by
    /// `self.scale`, and rounding to the nearest integer.
    ///
    /// Values that exceed the dynamic range of the quantization are clamped.
    ///
    /// To help with computing compensation coefficients, `callback` is included which
    /// is given the compressed value as a floating point number.
    ///
    /// # Notes
    ///
    /// This function allows the `ScalarQuantizer` to compress to bit-widths other than the
    /// one assigned to the quantizer. Though we have to compute a correcting factor for the
    /// scale, this allows us to mix and match compression bit-widths.
    fn compress<const NBITS: usize, T, F, Perm>(
        &self,
        from: &[T],
        mut into: MutBitSlice<'_, NBITS, Unsigned, Perm>,
        mut callback: F,
    ) -> Result<(), InputContainsNaN>
    where
        T: Copy + Into<f32>,
        F: FnMut(f32, usize),
        Unsigned: Representation<NBITS>,
        Perm: PermutationStrategy<NBITS>,
    {
        let len = self.shift.len();
        assert_eq!(from.len(), len);
        assert_eq!(into.len(), len);

        let domain = Unsigned::domain_const::<NBITS>();
        let min = *domain.start() as f32;
        let max = *domain.end() as f32;
        let inverse_scale = bit_scale::<NBITS>() / (self.scale);
        let mut nan_check = false;

        std::iter::zip(from.iter(), self.shift.iter())
            .enumerate()
            .for_each(|(i, (&f, &s))| {
                // Center and scale this component.
                // Then clamp to the unsigned dynamic range representable by the quantizer.
                let f: f32 = f.into();
                nan_check |= f.is_nan();

                let code: f32 = ((f - s) * inverse_scale).clamp(min, max).round();

                // Let the callback do some work on the final code if desired.
                callback(code, i);

                // SAFETY: We've checked that `into` and `from` have the same length.
                // The iterator will ensure the `i < into.len()`.
                //
                // By construction, `code` is in the domain of this `Unsigned` so the conversion
                // to `u8` is lossless.
                unsafe { into.set_unchecked(i, code as u8) };
            });

        if nan_check {
            Err(InputContainsNaN)
        } else {
            Ok(())
        }
    }

    /// Compare two `ScalarQuantizer` instances field by field.
    /// On success, returns `Ok(())`. On failure, returns `Err(SQComparisonError)`
    /// explaining which field differs.
    pub fn compare(&self, other: &Self) -> Result<(), SQComparisonError> {
        if self.scale != other.scale {
            return Err(SQComparisonError::Scale(self.scale, other.scale));
        }

        if self.shift.len() != other.shift.len() {
            return Err(SQComparisonError::ShiftLength(
                self.shift.len(),
                other.shift.len(),
            ));
        }

        for (i, (a, b)) in self.shift.iter().zip(other.shift.iter()).enumerate() {
            if a != b {
                return Err(SQComparisonError::ShiftElement {
                    index: i,
                    a: *a,
                    b: *b,
                });
            }
        }

        if self.shift_square_norm != other.shift_square_norm {
            return Err(SQComparisonError::ShiftSquareNorm(
                self.shift_square_norm,
                other.shift_square_norm,
            ));
        }

        match (&self.mean_norm, &other.mean_norm) {
            (Some(a), Some(b)) => {
                if a != b {
                    return Err(SQComparisonError::MeanNorm(*a, *b));
                }
            }
            (None, None) => {
                // both are None, no issue
            }
            _ => {
                return Err(SQComparisonError::MeanNormPresence);
            }
        }

        Ok(())
    }
}

#[derive(Debug, Error, Clone, Copy)]
#[error("mean norm is missing from the quantizer")]
#[non_exhaustive]
pub struct MeanNormMissing;

#[derive(Debug, Error, Clone, Copy)]
#[error("input contains NaN")]
#[non_exhaustive]
pub struct InputContainsNaN;

fn rescale(x: &mut [f32], to_norm: f32) {
    let norm_square: MathematicalValue<f32> =
        diskann_vector::distance::InnerProduct::evaluate(&*x, &*x);
    let norm = norm_square.into_inner().sqrt();
    if norm == 0.0 {
        return;
    }

    let scale = to_norm / norm;
    x.iter_mut().for_each(|i| (*i) *= scale);
}

///////////////////////
// Distance Functors //
///////////////////////

impl AsFunctor<CompensatedSquaredL2> for ScalarQuantizer {
    fn as_functor(&self) -> CompensatedSquaredL2 {
        let scale = self.scale();
        CompensatedSquaredL2::new(scale * scale)
    }
}

impl AsFunctor<CompensatedIP> for ScalarQuantizer {
    fn as_functor(&self) -> CompensatedIP {
        let scale = self.scale();
        CompensatedIP::new(scale * scale, self.shift_square_norm())
    }
}

impl AsFunctor<CompensatedCosineNormalized> for ScalarQuantizer {
    fn as_functor(&self) -> CompensatedCosineNormalized {
        let scale = self.scale();
        CompensatedCosineNormalized::new(scale * scale)
    }
}

/////////////////
// Compression //
/////////////////

impl<const NBITS: usize, T, Perm> CompressInto<&[T], MutBitSlice<'_, NBITS, Unsigned, Perm>>
    for ScalarQuantizer
where
    T: Copy + Into<f32>,
    Unsigned: Representation<NBITS>,
    Perm: PermutationStrategy<NBITS>,
{
    type Error = InputContainsNaN;

    type Output = ();

    /// Compress the input vector `from` into the bitslice `into`.
    ///
    /// This method *does not* compute compensation coefficients required for fast
    /// inner product computations. If only L2 distances is desired, this method can be
    /// slightly faster.
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
    /// * `into.len() != self.dim()`: Compressed vector must have the same dimensionality
    ///   as the quantizer.
    fn compress_into(
        &self,
        from: &[T],
        into: MutBitSlice<'_, NBITS, Unsigned, Perm>,
    ) -> Result<(), Self::Error> {
        // In this case, we don't need to pass anything special for `callback` because
        // there is no extra computation needed.
        ScalarQuantizer::compress(self, from, into, |_, _| {})
    }
}

impl<const NBITS: usize, T, Perm> CompressInto<&[T], MutCompensatedVectorRef<'_, NBITS, Perm>>
    for ScalarQuantizer
where
    T: Copy + Into<f32>,
    Unsigned: Representation<NBITS>,
    Perm: PermutationStrategy<NBITS>,
{
    type Error = InputContainsNaN;

    type Output = ();

    /// Compress the input vector `from` into the bitslice `into`.
    ///
    /// This method computes and stores the compensation coefficient required for fast
    /// inner product computations.
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
    /// * `into.len() != self.dim()`: Compressed vector must have the same dimensionality
    ///   as the quantizer.
    fn compress_into(
        &self,
        from: &[T],
        mut into: MutCompensatedVectorRef<'_, NBITS, Perm>,
    ) -> Result<(), Self::Error> {
        // Compress the raw code.
        //
        // While doing so, also compute the dot prodcut between the encoded vector and
        // the shift.
        let mut dot: f32 = 0.0;
        let result = ScalarQuantizer::compress(
            self,
            from,
            into.vector_mut(),
            // Compute the dot-product between `shift` and the compressed values.
            |code: f32, index: usize| {
                dot = code.mul_add(self.shift[index], dot);
            },
        );
        into.set_meta(Compensation(
            self.scale * inverse_bit_scale::<NBITS>() * dot,
        ));
        result
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum SQComparisonError {
    #[error("Scale mismatch: {0} vs {1}")]
    Scale(f32, f32),

    #[error("Shift vector length mismatch: {0} vs {1}")]
    ShiftLength(usize, usize),

    #[error("Shift element at index {index} mismatch: {a} vs {b}")]
    ShiftElement { index: usize, a: f32, b: f32 },

    #[error("Shift square norm mismatch: {0} vs {1}")]
    ShiftSquareNorm(f32, f32),

    #[error("Mean norm mismatch: {0} vs {1}")]
    MeanNorm(f32, f32),

    #[error("Mean norm is missing in one quantizer but present in the other")]
    MeanNormPresence,
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use diskann_utils::{views, ReborrowMut};

    use rand::{
        distr::{Distribution, Uniform},
        rngs::StdRng,
        seq::SliceRandom,
        SeedableRng,
    };
    use rand_distr::Normal;

    use super::*;
    use crate::{
        bits::BoxedBitSlice,
        scalar::{inverse_bit_scale, CompensatedVector},
    };

    /// Test Rescale.
    #[test]
    fn test_rescale() {
        let dim = 32;
        let to_norm = 25.0;

        let mut rng = StdRng::seed_from_u64(0x64e956ca2eb726ee);
        let distribution = Normal::<f32>::new(0.0, 16.0).unwrap();

        let mut v: Vec<f32> = distribution.sample_iter(&mut rng).take(dim).collect();
        let norm = v.iter().map(|&i| i * i).sum::<f32>().sqrt();

        rescale(&mut v, to_norm);
        let norm_next = v.iter().map(|&i| i * i).sum::<f32>().sqrt();
        let relative_error = (norm_next - to_norm).abs() / to_norm;

        assert!(
            relative_error <= 1.0e-7,
            "vector was not renormalized, expected {}, got {}, started with {}. Relative error: {}",
            to_norm,
            norm_next,
            norm,
            relative_error,
        );

        // Ensure that zero normed vectors are handled properly.
        let mut v: Vec<f32> = vec![0.0; dim];
        rescale(&mut v, 10.0);
        assert!(v.iter().all(|&i| i == 0.0));

        // Test in the context of a quantizer.
        let mut quantizer = ScalarQuantizer::new(0.0, vec![0.0; dim], Some(to_norm));

        let mut v: Vec<f32> = distribution.sample_iter(&mut rng).take(dim).collect();
        let norm = v.iter().map(|&i| i * i).sum::<f32>().sqrt();

        quantizer.rescale(&mut v).unwrap();
        let norm_next = v.iter().map(|&i| i * i).sum::<f32>().sqrt();
        let relative_error = (norm_next - to_norm).abs() / to_norm;

        assert!(
            relative_error <= 1.0e-7,
            "vector was not renormalized, expected {}, got {}, started with {}. Relative error: {}",
            to_norm,
            norm_next,
            norm,
            relative_error,
        );

        // Ensure that zero normed vectors are handled properly.
        let mut v: Vec<f32> = vec![0.0; dim];
        quantizer.rescale(&mut v).unwrap();
        assert!(v.iter().all(|&i| i == 0.0));

        // If the `mean_norm` is `None`, ensure we get an error.
        quantizer.mean_norm = None;
        let r = quantizer.rescale(&mut v);
        assert!(matches!(r, Err(MeanNormMissing)));
    }

    /// Quantizer tests
    ///
    /// This test works as follows: we create a simple quantizer with a know shift and scale.
    ///
    /// We then provide a range of offsets relative to the shift vector: some below and
    /// enough above to hit all the codes representable by the quantizer.
    ///
    /// These offsets are applied in different orders to each dimensions.
    ///
    /// Our checks are this:
    ///
    /// * If a value is *lower* than the shift (i.e., a negative offset), its code should be 0.
    /// * If an offset is *above* `2^NBITS * shift`, its code should be `2^NBITS - 1`.
    /// * For any offset in between, the reconstructed offsets computed by `shift * code`
    ///   should have an error no more than `shift / 2.0`.
    fn test_nbit_quantizer<const NBITS: usize>(dim: usize, rng: &mut StdRng)
    where
        Unsigned: Representation<NBITS>,
        ScalarQuantizer: for<'a, 'b> CompressInto<&'a [f32], MutBitSlice<'b, NBITS, Unsigned>>
            + for<'a, 'b> CompressInto<&'a [f32], MutCompensatedVectorRef<'b, NBITS>>,
    {
        let distribution = Uniform::new_inclusive::<i64, i64>(-10, 10).unwrap();
        let shift: Vec<f32> = (0..dim).map(|_| distribution.sample(rng) as f32).collect();
        let scale: f32 = 2.0;
        let mean_norm: f32 = 1.0;

        let quantizer =
            ScalarQuantizer::new(scale * bit_scale::<NBITS>(), shift.clone(), Some(mean_norm));

        assert_eq!(quantizer.dim(), dim);
        assert_eq!(quantizer.scale(), scale * bit_scale::<NBITS>());
        assert_eq!(quantizer.shift(), shift);
        assert_eq!(quantizer.mean_norm().unwrap(), mean_norm);

        let expected_shift_norm: f32 = shift.iter().map(|&i| i * i).sum();
        assert_eq!(quantizer.shift_square_norm(), expected_shift_norm);

        // Check conversion to distance functors.
        {
            let l2: CompensatedSquaredL2 = quantizer.as_functor();
            assert_eq!(l2.scale_squared, quantizer.scale() * quantizer.scale());

            let ip: CompensatedIP = quantizer.as_functor();
            assert_eq!(ip.scale_squared, quantizer.scale() * quantizer.scale());
            assert_eq!(ip.shift_square_norm, quantizer.shift_square_norm());
        }

        // Our strategy here is to generate a range of values for each dimension that should
        // enable all encodings. The reconstruction error for encoded vectors should be
        // within `scale / 2.0` (if the values are in range).
        let sample_points: f32 = 1.25 * (2_usize.pow(NBITS as u32) as f32) + 10.0;

        let min_encodable: f32 = 0.0;
        let max_encodable: f32 = (*Unsigned::domain_const::<NBITS>().end() as f32) * scale;

        // Create a shuffled matrix of offset values for each dimension. This ensure that
        // each dimension covers the target dynamic range, but in a different order so
        // we can rule out cross-coupling of dimensions.
        let dim_offsets: views::Matrix<f32> = {
            let range_min = -min_encodable - 3.0 * scale;
            let range_max = max_encodable + 3.0 * scale;
            let mut base: Vec<f32> = Vec::new();

            let step_size = (range_max - range_min) / sample_points;
            let mut i: f32 = range_min;
            while i < range_max {
                base.push(i);
                i += step_size;
            }
            // Push one more to have one point above `range_max`.
            base.push(i);

            let mut output = views::Matrix::new(0.0, base.len(), dim);
            (0..dim).for_each(|j| {
                base.shuffle(rng);
                for (i, b) in base.iter().enumerate() {
                    output[(i, j)] = *b;
                }
            });
            output
        };
        let ntests = dim_offsets.nrows();
        assert!(ntests as f32 >= sample_points);

        // Post-run checks to ensure coverage.
        let mut seen_below_min = false;
        let mut seen_above_max = false;
        let mut seen: Vec<HashSet<i64>> = (0..dim).map(|_| HashSet::new()).collect();

        // Reuse query space across tests.
        let mut query: Vec<f32> = vec![0.0; dim];
        for test_number in 0..ntests {
            let offsets = dim_offsets.row(test_number);
            query
                .iter_mut()
                .zip(std::iter::zip(shift.iter(), offsets.iter()))
                .for_each(|(q, (c, o))| {
                    *q = *c + *o;
                });

            // Test both `UnsignedBitSlice` and `CompensatedVector`.
            let mut bitslice = BoxedBitSlice::<NBITS, _>::new_boxed(dim);
            let mut compensated = CompensatedVector::<NBITS>::new_boxed(dim);

            quantizer
                .compress_into(&*query, bitslice.reborrow_mut())
                .unwrap();
            quantizer
                .compress_into(&*query, compensated.reborrow_mut())
                .unwrap();

            // Start checking!.
            let domain = Unsigned::domain_const::<NBITS>();

            // We compute the expected compensation inline with the checking code.
            let mut computed_compensation: f32 = 0.0;
            for d in 0..dim {
                let code = bitslice.get(d).unwrap();
                computed_compensation = (code as f32).mul_add(shift[d], computed_compensation);

                // Mark this code as having been observed.
                seen[d].insert(code);

                let offset = offsets[d];
                if offset <= min_encodable {
                    assert_eq!(
                        code,
                        *domain.start(),
                        "expected values below threshold to be set to zero \
                         test_number = {}, dim = {} of {}, offset = {}, scale = {}",
                        test_number,
                        d,
                        dim,
                        offset,
                        scale,
                    );
                    seen_below_min = true;
                } else if offset >= max_encodable {
                    assert_eq!(
                        code,
                        *domain.end(),
                        "expected values below threshold to be set to max value \
                         test_number = {}, dim = {} of {}, offset = {}, scale = {}",
                        test_number,
                        d,
                        dim,
                        offset,
                        scale,
                    );
                    seen_above_max = true;
                } else {
                    // This value is encodable - make sure its reconstruction error is with
                    // our tolerance.
                    let reconstructed =
                        quantizer.scale() * (code as f32) * inverse_bit_scale::<NBITS>();
                    let error = (offset - reconstructed).abs();
                    assert!(
                        error <= scale / 2.0,
                        "failed reconstruction check: \
                         test_number = {}, dim = {} of {}, offset = {}, scale = {} \
                         code = {}, reconstructed = {}, error = {}",
                        test_number,
                        d,
                        dim,
                        offset,
                        scale,
                        code,
                        reconstructed,
                        error,
                    );
                }

                // Now that we have checked the reconstruction, ensure that the
                // `CompensatedVector` has the same code.
                assert_eq!(
                    compensated.vector().get(d).unwrap(),
                    code,
                    "compensated disagrees with bitslice"
                );
            }
            assert_eq!(scale * computed_compensation, compensated.meta().0);
        }

        // Check coverage.
        assert!(seen_below_min);
        assert!(seen_above_max);
        let num_codes = 2usize.pow(NBITS as u32);
        for (i, s) in seen.iter().enumerate() {
            assert_eq!(
                s.len(),
                num_codes,
                "dimension {} did not have full coverage",
                i
            );
        }

        // Check NaN detection.
        {
            let mut query: Vec<f32> = shift.clone();
            let mut bitslice = BoxedBitSlice::<NBITS, _>::new_boxed(query.len());
            let mut compensated = CompensatedVector::<NBITS>::new_boxed(query.len());
            for i in 0..query.len() {
                let last = query[i];
                query[i] = f32::NAN;

                let err = quantizer
                    .compress_into(&*query, bitslice.reborrow_mut())
                    .unwrap_err();
                assert_eq!(err.to_string(), "input contains NaN");

                let err = quantizer
                    .compress_into(&*query, compensated.reborrow_mut())
                    .unwrap_err();
                assert_eq!(err.to_string(), "input contains NaN");

                query[i] = last;
            }
        }
    }

    cfg_if::cfg_if! {
        if #[cfg(miri)] {
            const TEST_DIM: usize = 2;
        } else {
            const TEST_DIM: usize = 10;
        }
    }

    macro_rules! test_quantizer {
        ($name:ident, $nbits:literal, $seed:literal) => {
            #[test]
            fn $name() {
                let mut rng = StdRng::seed_from_u64($seed);
                test_nbit_quantizer::<$nbits>(TEST_DIM, &mut rng);
            }
        };
    }

    test_quantizer!(test_8bit_quantizer, 8, 0xb7b4c124102b9fb9);
    test_quantizer!(test_7bit_quantizer, 7, 0x86d19a821fe934d1);
    test_quantizer!(test_6bit_quantizer, 6, 0x0de9610f0b9be4f7);
    test_quantizer!(test_5bit_quantizer, 5, 0x605ed3e7ed775047);
    test_quantizer!(test_4bit_quantizer, 4, 0x9b66ace7090fa728);
    test_quantizer!(test_3bit_quantizer, 3, 0x0ce424ddc61ebdb0);
    test_quantizer!(test_2bit_quantizer, 2, 0x2ba8e5ef6415d4f0);
    test_quantizer!(test_1bit_quantizer, 1, 0xdcd8c10c4a407956);

    fn base_quantizer() -> ScalarQuantizer {
        ScalarQuantizer {
            scale: 2.0,
            shift: vec![1.0, -1.0, 0.5],
            shift_square_norm: 1.0_f32 * 1.0 + (-1.0_f32) * (-1.0) + 0.5_f32 * 0.5,
            mean_norm: Some(4.13),
        }
    }

    #[test]
    fn test_compare_identical_returns_ok() {
        let q1 = base_quantizer();
        let q2 = base_quantizer();
        assert!(q1.compare(&q2).is_ok());
    }

    #[test]
    fn test_compare_scale_mismatch() {
        let q1 = base_quantizer();
        let mut q2 = base_quantizer();
        q2.scale = 4.0;
        let err = q1.compare(&q2).unwrap_err();
        assert_eq!(err, SQComparisonError::Scale(2.0, 4.0));
    }

    #[test]
    fn test_compare_shift_length_mismatch() {
        let q1 = base_quantizer();
        let mut q2 = base_quantizer();
        q2.shift.push(0.0);
        let err = q1.compare(&q2).unwrap_err();
        assert_eq!(
            err,
            SQComparisonError::ShiftLength(q1.shift.len(), q2.shift.len())
        );
    }

    #[test]
    fn test_compare_shift_element_mismatch() {
        let q1 = base_quantizer();
        let mut q2 = base_quantizer();
        q2.shift[2] = 0.0;
        let err = q1.compare(&q2).unwrap_err();
        match err {
            SQComparisonError::ShiftElement { index, a, b } => {
                assert_eq!(index, 2);
                assert_eq!(a, 0.5);
                assert_eq!(b, 0.0);
            }
            _ => panic!("Expected ShiftElementMismatch variant"),
        }
    }

    #[test]
    fn test_compare_shift_square_norm_mismatch() {
        let q1 = base_quantizer();
        let mut q2 = base_quantizer();
        q2.shift_square_norm = 9.0;
        let err = q1.compare(&q2).unwrap_err();
        assert_eq!(err, SQComparisonError::ShiftSquareNorm(2.25, 9.0));
    }

    #[test]
    fn test_compare_mean_norm_value_mismatch() {
        let q1 = base_quantizer();
        let mut q2 = base_quantizer();
        q2.mean_norm = Some(1.0);
        let err = q1.compare(&q2).unwrap_err();
        assert_eq!(err, SQComparisonError::MeanNorm(4.13, 1.0));
    }

    #[test]
    fn test_compare_mean_norm_presence_mismatch_left_none() {
        let mut q1 = base_quantizer();
        let q2 = base_quantizer();
        q1.mean_norm = None;
        let err = q1.compare(&q2).unwrap_err();
        assert_eq!(err, SQComparisonError::MeanNormPresence);
    }

    #[test]
    fn test_compare_mean_norm_presence_mismatch_right_none() {
        let q1 = base_quantizer();
        let mut q2 = base_quantizer();
        q2.mean_norm = None;
        let err = q1.compare(&q2).unwrap_err();
        assert_eq!(err, SQComparisonError::MeanNormPresence);
    }
}
