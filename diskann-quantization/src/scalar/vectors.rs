/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_vector::{DistanceFunction, PureDistanceFunction};

use super::inverse_bit_scale;
use crate::{
    bits::{BitSlice, Dense, Representation, Unsigned},
    distances::{self, check_lengths, InnerProduct, SquaredL2, MV},
    meta,
};

/// A per-vector precomputed coefficient to help compute inner products.
///
/// To understand the use of the compensation coefficient, assume that we wish to compute
/// the inner product between two scalar compressed vectors where the quantization has
/// scale parameter `a` and centroid `B` (note: capital letters represent vectors, lower
/// case letters represent scalars).
///
/// The inner product between a `X = a * (X' + B)` and `Y = a * (Y' + B)` where
/// `X'` and `Y'` are the scalar encodings for `X` and `Y` respectively is:
/// ```math
/// P = <a * X' + B, a * Y' + B>
///   = a^2 * <X', Y'> + a * <X', B> + a * <Y', B> + <B, B>
///            ------    -----------   -----------   ------
///               |           |             |           |
///          Integer Dot      |        Compensation     |
///            Product        |           for Y         |
///                           |                    Constant for
///                      Compensation               all vectors
///                         for X
///
/// ```
/// In other words, the inner product can be decomposed into an integer dot-product plus
/// a bunch of other terms that compensate for the compression.
///
/// These compensation terms can be computed as the vectors are compressed. At run time,
/// we can the return vectors consisting of the quantized encodings (e.g. `X'`) and the
/// compensation `<X', B>`.
///
/// Computation of squared Euclidean distance is more straight forward:
/// ```math
/// P = sum( ((a * X' + B) - (a * Y' + B))^2 )
///   = sum( a^2 * (X' - Y')^2 )
///   = a^2 * sum( (X' - Y')^2 )
/// ```
/// This means the squared Euclidean distance is computed by scaling the squared Euclidean
/// distance computed directly on the integer codes.
///
/// # Distance Implementations
///
/// The following distance function types are implemented:
///
/// * [`CompensatedSquaredL2`]: For computing squared euclidean distances.
/// * [`CompensatedIP`]: For computing inner products.
///
/// # Examples
///
/// The `CompensatedVector` has several named variants that are commonly used:
/// * [`CompensatedVector`]: An owning, indepndently allocated `CompensatedVector`.
/// * [`MutCompensatedVectorRef`]: A mutable, reference-like type to a `CompensatedVector`.
/// * [`CompensatedVectorRef`]: A const, reference-like type to a `CompensatedVector`.
///
/// ```
/// use diskann_quantization::{
///     scalar::{
///         self,
///         CompensatedVector,
///         MutCompensatedVectorRef,
///         CompensatedVectorRef
///     },
/// };
///
/// use diskann_utils::{Reborrow, ReborrowMut};
///
/// // Create a new heap-allocated CompensatedVector for 4-bit compressions capable of
/// // holding 3 elements.
/// let mut v = CompensatedVector::<4>::new_boxed(3);
///
/// // We can inspect the underlying bitslice.
/// let bitslice = v.vector();
/// assert_eq!(bitslice.get(0).unwrap(), 0);
/// assert_eq!(bitslice.get(1).unwrap(), 0);
/// assert_eq!(v.meta().0, 0.0, "expected default compensation value");
///
/// // If we want, we can mutably borrow the bitslice and mutate its components.
/// let mut bitslice = v.vector_mut();
/// bitslice.set(0, 1).unwrap();
/// bitslice.set(1, 2).unwrap();
/// bitslice.set(2, 3).unwrap();
///
/// assert!(bitslice.set(3, 4).is_err(), "out-of-bounds access");
///
/// // Get the underlying pointer for comparision.
/// let ptr = bitslice.as_ptr();
///
/// // Vectors can be converted to a generalized reference.
/// let mut v_ref = v.reborrow_mut();
///
/// // The generalized reference preserves the underlying pointer.
/// assert_eq!(v_ref.vector().as_ptr(), ptr);
/// let mut bitslice = v_ref.vector_mut();
/// bitslice.set(0, 10).unwrap();
///
/// // Setting the underlying compensation will be visible in the original allocation.
/// v_ref.set_meta(scalar::Compensation(1.0));
///
/// // Check that the changes are visible.
/// assert_eq!(v.meta().0, 1.0);
/// assert_eq!(v.vector().get(0).unwrap(), 10);
///
/// // Finally, the immutable ref also maintains pointer compatibility.
/// let v_ref = v.reborrow();
/// assert_eq!(v_ref.vector().as_ptr(), ptr);
/// ```
///
/// ## Constructing a `MutCompensatedVectorRef` From Components
///
/// The following example shows how to assemble a `MutCompensatedVectorRef` from raw memory.
/// ```
/// use diskann_quantization::{
///     bits::{Unsigned, MutBitSlice},
///     scalar::{self, MutCompensatedVectorRef}
/// };
///
/// // Start with 2 bytes of memory. We will impose a 4-bit scalar quantization on top of
/// // these 4 bytes.
/// let mut data = vec![0u8; 2];
/// let mut compensation = scalar::Compensation(0.0);
/// {
///     // First, we need to construct a bit-slice over the data.
///     // This will check that it is sized properly for 4, 4-bit values.
///     let mut slice = MutBitSlice::<4, Unsigned>::new(data.as_mut_slice(), 4).unwrap();
///
///     // Next, we construct the `MutCompensatedVectorRef`.
///     let mut v = MutCompensatedVectorRef::new(slice, &mut compensation);
///
///     // Through `v`, we can set all the components in `slice` and the compensation.
///     v.set_meta(scalar::Compensation(1.0));
///     let mut from_v = v.vector_mut();
///     from_v.set(0, 1).unwrap();
///     from_v.set(1, 2).unwrap();
///     from_v.set(2, 3).unwrap();
///     from_v.set(3, 4).unwrap();
/// }
///
/// // Now we can check that the changes made internally are visible.
/// assert_eq!(&data, &[0x21, 0x43]);
/// assert_eq!(compensation.0, 1.0);
/// ```
#[derive(Default, Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(transparent)]
pub struct Compensation(pub f32);

/// A borrowed `ComptensatedVector`.
///
/// See: [`meta::Vector`].
pub type CompensatedVectorRef<'a, const NBITS: usize, Perm = Dense> =
    meta::VectorRef<'a, NBITS, Unsigned, Compensation, Perm>;

/// A mutably borrowed `ComptensatedVector`.
///
/// See: [`meta::Vector`].
pub type MutCompensatedVectorRef<'a, const NBITS: usize, Perm = Dense> =
    meta::VectorMut<'a, NBITS, Unsigned, Compensation, Perm>;

/// An owning `CompensatedVector`.
///
/// See: [`meta::Vector`].
pub type CompensatedVector<const NBITS: usize, Perm = Dense> =
    meta::Vector<NBITS, Unsigned, Compensation, Perm>;

////////////////////////////
// Compensated Squared L2 //
////////////////////////////

/// A `DistanceFunction` containing scaling parameters to enable distance the SquaredL2
/// distance function over `CompensatedVectors` belonging to the same quantization space.
#[derive(Debug, Clone, Copy)]
pub struct CompensatedSquaredL2 {
    pub(super) scale_squared: f32,
}

impl CompensatedSquaredL2 {
    /// Construct a new `CompensatedSquaredL2` with the given scaling factor.
    pub fn new(scale_squared: f32) -> Self {
        Self { scale_squared }
    }
}

/// Compute the squared euclidean distance between the two compensated vectors.
///
/// The value returned by this function is scaled properly, meaning that distances returned
/// by this method are compatible with full-precision distances.
///
/// # Validity
///
/// The results of this function are only meaningful if both `x`, `y`, and `Self` belong to
/// the same quantizer.
///
/// # Panics
///
/// Panics if `x.len() != y.len()`.
impl<const NBITS: usize>
    DistanceFunction<
        CompensatedVectorRef<'_, NBITS>,
        CompensatedVectorRef<'_, NBITS>,
        distances::MathematicalResult<f32>,
    > for CompensatedSquaredL2
where
    Unsigned: Representation<NBITS>,
    SquaredL2: for<'a, 'b> PureDistanceFunction<
        BitSlice<'a, NBITS, Unsigned>,
        BitSlice<'b, NBITS, Unsigned>,
        distances::MathematicalResult<u32>,
    >,
{
    fn evaluate_similarity(
        &self,
        x: CompensatedVectorRef<'_, NBITS>,
        y: CompensatedVectorRef<'_, NBITS>,
    ) -> distances::MathematicalResult<f32> {
        check_lengths!(x, y)?;
        let squared_l2: distances::MathematicalResult<u32> =
            SquaredL2::evaluate(x.vector(), y.vector());
        let squared_l2 = squared_l2?.into_inner() as f32;

        // This should constant-propagate.
        let bit_scale = inverse_bit_scale::<NBITS>() * inverse_bit_scale::<NBITS>();

        let result = bit_scale * self.scale_squared * squared_l2;
        Ok(MV::new(result))
    }
}

/// Compute the squared euclidean distance between the two compensated vectors.
///
/// The value returned by this function is scaled properly, meaning that distances returned
/// by this method are compatible with full-precision distances.
///
/// # Validity
///
/// The results of this function are only meaningful if both `x`, `y`, and `Self` belong to
/// the same quantizer.
///
/// # Panics
///
/// Panics if `x.len() != y.len()`.
impl<const NBITS: usize>
    DistanceFunction<
        CompensatedVectorRef<'_, NBITS>,
        CompensatedVectorRef<'_, NBITS>,
        distances::Result<f32>,
    > for CompensatedSquaredL2
where
    Unsigned: Representation<NBITS>,
    Self: for<'a, 'b> DistanceFunction<
        CompensatedVectorRef<'a, NBITS>,
        CompensatedVectorRef<'b, NBITS>,
        distances::MathematicalResult<f32>,
    >,
{
    fn evaluate_similarity(
        &self,
        x: CompensatedVectorRef<'_, NBITS>,
        y: CompensatedVectorRef<'_, NBITS>,
    ) -> distances::Result<f32> {
        let v: MV<f32> = self.evaluate_similarity(x, y)?;
        Ok(v.into_inner())
    }
}

////////////////////
// Compensated IP //
////////////////////

/// A `DistanceFunction` containing scaling parameters to enable distance the SquaredL2
/// distance function over `CompensatedVectors` belonging to the same quantization space.
#[derive(Debug, Clone, Copy)]
pub struct CompensatedIP {
    pub(super) scale_squared: f32,
    pub(super) shift_square_norm: f32,
}

impl CompensatedIP {
    /// Construct a new `CompensatedIP` with the given scaling factor and shift norm.
    pub fn new(scale_squared: f32, shift_square_norm: f32) -> Self {
        Self {
            scale_squared,
            shift_square_norm,
        }
    }
}

/// Compute the inner product between the two compensated vectors.
///
/// The value returned by this function is scaled properly, meaning that distances returned
/// by this method are compatible with full-precision computations.
///
/// # Validity
///
/// The results of this function are only meaningful if both `x`, `y`, and `Self` belong to
/// the same quantizer.
///
/// # Panics
///
/// Panics if `x.len() != y.len()`.
impl<const NBITS: usize>
    DistanceFunction<
        CompensatedVectorRef<'_, NBITS>,
        CompensatedVectorRef<'_, NBITS>,
        distances::MathematicalResult<f32>,
    > for CompensatedIP
where
    Unsigned: Representation<NBITS>,
    InnerProduct: for<'a, 'b> PureDistanceFunction<
        BitSlice<'a, NBITS, Unsigned>,
        BitSlice<'b, NBITS, Unsigned>,
        distances::MathematicalResult<u32>,
    >,
{
    fn evaluate_similarity(
        &self,
        x: CompensatedVectorRef<'_, NBITS>,
        y: CompensatedVectorRef<'_, NBITS>,
    ) -> distances::MathematicalResult<f32> {
        let product: MV<u32> = InnerProduct::evaluate(x.vector(), y.vector())?;

        // This should constant-propagate.
        let bit_scale = inverse_bit_scale::<NBITS>() * inverse_bit_scale::<NBITS>();

        let result = (bit_scale * self.scale_squared)
            .mul_add(product.into_inner() as f32, self.shift_square_norm)
            + (y.meta().0 + x.meta().0);
        Ok(MV::new(result))
    }
}

/// Compute the inner product between the two compensated vectors.
///
/// The value returned by this function is scaled properly, meaning that distances returned
/// by this method are compatible with full-precision computations.
///
/// # Validity
///
/// The results of this function are only meaningful if both `x`, `y`, and `Self` belong to
/// the same quantizer.
///
/// # Panics
///
/// Panics if `x.len() != y.len()`.
impl<const NBITS: usize>
    DistanceFunction<
        CompensatedVectorRef<'_, NBITS>,
        CompensatedVectorRef<'_, NBITS>,
        distances::Result<f32>,
    > for CompensatedIP
where
    Unsigned: Representation<NBITS>,
    Self: for<'a, 'b> DistanceFunction<
        CompensatedVectorRef<'a, NBITS>,
        CompensatedVectorRef<'b, NBITS>,
        distances::MathematicalResult<f32>,
    >,
{
    fn evaluate_similarity(
        &self,
        x: CompensatedVectorRef<'_, NBITS>,
        y: CompensatedVectorRef<'_, NBITS>,
    ) -> distances::Result<f32> {
        let v: MV<f32> = self.evaluate_similarity(x, y)?;
        Ok(-v.into_inner())
    }
}

/// Compensated CosineNormalized distance function.
#[derive(Debug, Clone, Copy)]
pub struct CompensatedCosineNormalized {
    pub(super) scale_squared: f32,
}

impl CompensatedCosineNormalized {
    pub fn new(scale_squared: f32) -> Self {
        Self { scale_squared }
    }
}

/// CosineNormalized
///
/// This implementation calculates the <x, y> = 1 - L2 / 2 value, which will be further used
/// to compute the CosineNormalised distance function
///
/// # Notes
///
/// s = 1 - cosine(X, Y) = 1- <X, Y> / (||X|| * ||Y||)
///
/// We can make simply assumption that ||X|| = 1 and ||Y|| = 1.
/// Then s = 1 - <X, Y>
///
/// The squared L2 distance can be computed as follows:
/// p = ||x||^2 + ||y||^2 - 2<x, y>
/// When vectors are normalized, this becomes
/// p = 2 - 2<x, y> = 2 * (1 - <x, y>)
///
/// In other words, the similarity score for the squared L2 distance in an ideal world is
/// 2 times that for cosine similarity. Therefore, squared L2 may serves as a stand-in for
/// cosine normalized as ordering is preserved.
impl<const NBITS: usize>
    DistanceFunction<
        CompensatedVectorRef<'_, NBITS>,
        CompensatedVectorRef<'_, NBITS>,
        distances::MathematicalResult<f32>,
    > for CompensatedCosineNormalized
where
    Unsigned: Representation<NBITS>,
    SquaredL2: for<'a, 'b> PureDistanceFunction<
        BitSlice<'a, NBITS, Unsigned>,
        BitSlice<'b, NBITS, Unsigned>,
        distances::MathematicalResult<u32>,
    >,
{
    fn evaluate_similarity(
        &self,
        x: CompensatedVectorRef<'_, NBITS>,
        y: CompensatedVectorRef<'_, NBITS>,
    ) -> distances::MathematicalResult<f32> {
        let squared_l2: MV<u32> = SquaredL2::evaluate(x.vector(), y.vector())?;

        // This should constant-propagate.
        let bit_scale = inverse_bit_scale::<NBITS>() * inverse_bit_scale::<NBITS>();

        let l2 = bit_scale * self.scale_squared * squared_l2.into_inner() as f32;

        let result = 1.0 - l2 / 2.0;
        Ok(MV::new(result))
    }
}

impl<const NBITS: usize>
    DistanceFunction<
        CompensatedVectorRef<'_, NBITS>,
        CompensatedVectorRef<'_, NBITS>,
        distances::Result<f32>,
    > for CompensatedCosineNormalized
where
    Unsigned: Representation<NBITS>,
    Self: for<'a, 'b> DistanceFunction<
        CompensatedVectorRef<'a, NBITS>,
        CompensatedVectorRef<'b, NBITS>,
        distances::MathematicalResult<f32>,
    >,
{
    fn evaluate_similarity(
        &self,
        x: CompensatedVectorRef<'_, NBITS>,
        y: CompensatedVectorRef<'_, NBITS>,
    ) -> distances::Result<f32> {
        let v: MV<f32> = self.evaluate_similarity(x, y)?;
        Ok(1.0 - v.into_inner())
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use diskann_utils::{Reborrow, ReborrowMut};
    use rand::{
        distr::{Distribution, Uniform},
        rngs::StdRng,
        Rng, SeedableRng,
    };

    use super::*;
    use crate::{
        bits::{Representation, Unsigned},
        scalar::bit_scale,
        test_util,
    };

    ///////////////
    // Distances //
    ///////////////

    /// This test works as follows:
    ///
    /// First, generate a random value for `a`, `X'` and `B` where:
    ///
    /// * `a`: Is the scaling parameters.
    /// * `X'`: Is the integer compressed codes for a vector.
    /// * `B`: The floating point vector representing the dataset center.
    ///
    /// Next, compute the reconstructed vector using `X = a * X' + B`.
    /// Repeat this process for another vector `Y` using the same `a` and `B`.
    ///
    /// Then, the result of a distance computation can be done on the compressed
    /// representation and on the reconstructed representation. The results should match
    /// (modulo floating-point rounding).
    ///
    /// To get a handle on floating point issues, we pick "nice" numbers for the values of
    /// `a` and each component of `B` that are either small integers, or nice binary fractions
    /// like 1/2 or 3/4.
    ///
    /// Even with nice numbers, there is still a small amount of rounding instability.
    fn test_compensated_distance<const NBITS: usize, R>(
        dim: usize,
        ntrials: usize,
        max_relative_err_l2: f32,
        max_relative_err_ip: f32,
        max_relative_err_cos: f32,
        max_absolute_error: f32,
        rng: &mut R,
    ) where
        Unsigned: Representation<NBITS>,
        R: Rng,
        CompensatedSquaredL2: for<'a, 'b> DistanceFunction<
            CompensatedVectorRef<'a, NBITS>,
            CompensatedVectorRef<'b, NBITS>,
            distances::MathematicalResult<f32>,
        >,
        CompensatedSquaredL2: for<'a, 'b> DistanceFunction<
            CompensatedVectorRef<'a, NBITS>,
            CompensatedVectorRef<'b, NBITS>,
            distances::Result<f32>,
        >,
        CompensatedIP: for<'a, 'b> DistanceFunction<
            CompensatedVectorRef<'a, NBITS>,
            CompensatedVectorRef<'b, NBITS>,
            distances::MathematicalResult<f32>,
        >,
        CompensatedIP: for<'a, 'b> DistanceFunction<
            CompensatedVectorRef<'a, NBITS>,
            CompensatedVectorRef<'b, NBITS>,
            distances::Result<f32>,
        >,
        CompensatedCosineNormalized: for<'a, 'b> DistanceFunction<
            CompensatedVectorRef<'a, NBITS>,
            CompensatedVectorRef<'b, NBITS>,
            distances::MathematicalResult<f32>,
        >,
        CompensatedCosineNormalized: for<'a, 'b> DistanceFunction<
            CompensatedVectorRef<'a, NBITS>,
            CompensatedVectorRef<'b, NBITS>,
            distances::Result<f32>,
        >,
    {
        // The distributions we use for `a` and `B` are taken from integer distributions,
        // which we then convert to `f32` and divide by a power of 2.
        //
        // This helps keep computations exact so we don't also have to worry about tracking
        // floating rounding issues.
        //
        // Here, `alpha` refers to `a` in the function docstring and `beta` refers to `B`.
        let alpha_distribution = Uniform::new_inclusive(-16, 16).unwrap();
        let beta_distribution = Uniform::new_inclusive(-32, 32).unwrap();

        // What we divide the results generated by the alpha and beta distributions.
        let alpha_divisor: f32 = 64.0;
        let beta_divisor: f32 = 128.0;

        let domain = Unsigned::domain_const::<NBITS>();
        let code_distribution = Uniform::new_inclusive(*domain.start(), *domain.end()).unwrap();

        // Preallocate buffers.
        let mut beta: Vec<f32> = vec![0.0; dim];
        let mut x_prime: Vec<u8> = vec![0; dim];
        let mut y_prime: Vec<u8> = vec![0; dim];
        let mut x_reconstructed: Vec<f32> = vec![0.0; dim];
        let mut y_reconstructed: Vec<f32> = vec![0.0; dim];

        let mut x_compensated = CompensatedVector::<NBITS>::new_boxed(dim);
        let mut y_compensated = CompensatedVector::<NBITS>::new_boxed(dim);

        // Populate a compensated vector from the codes and `beta`.
        let populate_compensation = |mut dst: MutCompensatedVectorRef<'_, NBITS>,
                                     codes: &[u8],
                                     alpha: f32,
                                     beta: &[f32]| {
            assert_eq!(dst.len(), codes.len());
            assert_eq!(dst.len(), beta.len());

            let mut compensation: f32 = 0.0;
            let mut vector = dst.vector_mut();
            for (i, (&c, &b)) in std::iter::zip(codes.iter(), beta.iter()).enumerate() {
                vector.set(i, c.into()).unwrap();

                let c: f32 = c.into();
                compensation += c * b;
            }
            dst.set_meta(Compensation(alpha * compensation / bit_scale::<NBITS>()));
        };

        for trial in 0..ntrials {
            // Generate the problem.
            let alpha = (alpha_distribution.sample(rng) as f32) / alpha_divisor;
            beta.iter_mut().for_each(|b| {
                *b = (beta_distribution.sample(rng) as f32) / beta_divisor;
            });
            x_prime
                .iter_mut()
                .for_each(|x| *x = code_distribution.sample(rng).try_into().unwrap());
            y_prime
                .iter_mut()
                .for_each(|y| *y = code_distribution.sample(rng).try_into().unwrap());

            // Generate the reconstructed vectors.
            let bit_scale = inverse_bit_scale::<NBITS>();
            x_reconstructed
                .iter_mut()
                .zip(x_prime.iter())
                .zip(beta.iter())
                .for_each(|((x, xp), b)| {
                    *x = (alpha * *xp as f32) * bit_scale + *b;
                });

            y_reconstructed
                .iter_mut()
                .zip(y_prime.iter())
                .zip(beta.iter())
                .for_each(|((y, yp), b)| {
                    *y = (alpha * *yp as f32) * bit_scale + *b;
                });

            populate_compensation(x_compensated.reborrow_mut(), &x_prime, alpha, &beta);
            populate_compensation(y_compensated.reborrow_mut(), &y_prime, alpha, &beta);

            // Squared L2
            let expected: MV<f32> =
                diskann_vector::distance::SquaredL2::evaluate(&*x_reconstructed, &*y_reconstructed);

            let distance = CompensatedSquaredL2::new(alpha * alpha);
            let got: distances::MathematicalResult<f32> =
                distance.evaluate_similarity(x_compensated.reborrow(), y_compensated.reborrow());
            let got = got.unwrap();

            let relative_err =
                test_util::compute_relative_error(got.into_inner(), expected.into_inner());
            let absolute_err =
                test_util::compute_absolute_error(got.into_inner(), expected.into_inner());

            assert!(
                relative_err <= max_relative_err_l2 || absolute_err <= max_absolute_error,
                "failed SquaredL2 for NBITS = {}, dim = {}, trial = {}. \
                 Got an error {} (rel) / {} (abs) with tolerance {}/{}. \
                 Expected {}, got {}",
                NBITS,
                dim,
                trial,
                relative_err,
                absolute_err,
                max_relative_err_l2,
                max_absolute_error,
                expected.into_inner(),
                got.into_inner(),
            );

            // f32 should match Mathematicalvalue.
            let got_f32: distances::Result<f32> =
                distance.evaluate_similarity(x_compensated.reborrow(), y_compensated.reborrow());
            let got_f32 = got_f32.unwrap();
            assert_eq!(got.into_inner(), got_f32);

            // Inner Product
            let expected: MV<f32> = diskann_vector::distance::InnerProduct::evaluate(
                &*x_reconstructed,
                &*y_reconstructed,
            );

            let distance =
                CompensatedIP::new(alpha * alpha, beta.iter().map(|&i| i * i).sum::<f32>());
            let got: distances::MathematicalResult<f32> =
                distance.evaluate_similarity(x_compensated.reborrow(), y_compensated.reborrow());
            let got = got.unwrap();

            let relative_err =
                test_util::compute_relative_error(got.into_inner(), expected.into_inner());
            let absolute_err =
                test_util::compute_absolute_error(got.into_inner(), expected.into_inner());

            assert!(
                relative_err <= max_relative_err_ip || absolute_err < max_absolute_error,
                "failed InnerProduct for NBITS = {}, dim = {}, trial = {}. \
                 Got an error {} (rel) / {} (abs) with tolerance {}/{}. \
                 Expected {}, got {}",
                NBITS,
                dim,
                trial,
                relative_err,
                absolute_err,
                max_relative_err_ip,
                max_absolute_error,
                expected.into_inner(),
                got.into_inner(),
            );

            // f32 should be the negative Mathematicalvalue.
            let got_f32: distances::Result<f32> =
                distance.evaluate_similarity(x_compensated.reborrow(), y_compensated.reborrow());
            let got_f32 = got_f32.unwrap();

            assert_eq!(-got.into_inner(), got_f32);

            // CosineNormalized:
            // expected value is cosine similarity of reconstructed vectors (no scale/shift)
            let expected: MV<f32> =
                diskann_vector::distance::SquaredL2::evaluate(&*x_reconstructed, &*y_reconstructed);
            let expected = 1.0 - expected.into_inner() / 2.0;

            let distance = CompensatedCosineNormalized::new(alpha * alpha);
            let got: distances::MathematicalResult<f32> =
                distance.evaluate_similarity(x_compensated.reborrow(), y_compensated.reborrow());
            let got = got.unwrap();

            if expected != 0.0 {
                let relative_err = test_util::compute_relative_error(got.into_inner(), expected);
                let absolute_err = test_util::compute_absolute_error(got.into_inner(), expected);
                assert!(
                    relative_err < max_relative_err_cos || absolute_err < max_absolute_error,
                    "failed CosineNormalized for NBITS = {}, dim = {}, trial = {}. \
                     Got an error {} (rel) / {} (abs) with tolerance {}/{}. \
                     Expected {}, got {}",
                    NBITS,
                    dim,
                    trial,
                    relative_err,
                    absolute_err,
                    max_relative_err_cos,
                    max_absolute_error,
                    expected,
                    got.into_inner(),
                );
            } else {
                let absolute_err = test_util::compute_absolute_error(got.into_inner(), expected);
                assert!(
                    absolute_err < max_absolute_error,
                    "failed CosineNormalized for NBITS = {}, dim = {}, trial = {}. \
                    Got an absolute error {} with tolerance {}. \
                    Expected {}, got {}",
                    NBITS,
                    dim,
                    trial,
                    absolute_err,
                    max_absolute_error,
                    expected,
                    got.into_inner(),
                );
            }

            let got_f32: distances::Result<f32> =
                distance.evaluate_similarity(x_compensated.reborrow(), y_compensated.reborrow());
            let got_f32 = got_f32.unwrap();
            assert_eq!(1.0 - got.into_inner(), got_f32);
        }
    }

    cfg_if::cfg_if! {
        if #[cfg(miri)] {
            // The max dim does not need to be as high for `CompensatedVectors` because they
            // defer their distance function implementation to `BitSlice`, which is more
            // heavily tested.
            const MAX_DIM: usize = 37;
            const TRIALS_PER_DIM: usize = 1;
        } else {
            const MAX_DIM: usize = 256;
            const TRIALS_PER_DIM: usize = 20;
        }
    }

    macro_rules! test_unsigned_compensated {
        (
            $name:ident,
            $nbits:literal,
            $relative_err_l2:literal,
            $relative_err_ip:literal,
            $relative_err_cos:literal,
            $seed:literal
        ) => {
            #[test]
            fn $name() {
                let mut rng = StdRng::seed_from_u64($seed);
                let absolute_error: f32 = 2.0e-7;
                for dim in 0..MAX_DIM {
                    test_compensated_distance::<$nbits, _>(
                        dim,
                        TRIALS_PER_DIM,
                        $relative_err_l2,
                        $relative_err_ip,
                        $relative_err_cos,
                        absolute_error,
                        &mut rng,
                    );
                }
            }
        };
    }

    test_unsigned_compensated!(
        unsigned_compensated_distances_8bit,
        8,
        4.0e-4,
        3.0e-6,
        1.0e-3,
        0xa32d5658097a1c35
    );
    test_unsigned_compensated!(
        unsigned_compensated_distances_7bit,
        7,
        5.0e-6,
        3.0e-6,
        1.0e-3,
        0x0b65ca44ec7b47d8
    );
    test_unsigned_compensated!(
        unsigned_compensated_distances_6bit,
        6,
        5.0e-6,
        3.0e-6,
        1.0e-3,
        0x471b640fba5c520b
    );
    test_unsigned_compensated!(
        unsigned_compensated_distances_5bit,
        5,
        5.0e-6,
        3.0e-6,
        1.0e-3,
        0xf60c0c8d1aadc126
    );
    test_unsigned_compensated!(
        unsigned_compensated_distances_4bit,
        4,
        3.0e-6,
        3.0e-6,
        1.0e-3,
        0xcc2b897373a143f3
    );
    test_unsigned_compensated!(
        unsigned_compensated_distances_3bit,
        3,
        3.0e-6,
        3.0e-6,
        1.0e-3,
        0xaedf3d2a223b7b77
    );
    test_unsigned_compensated!(
        unsigned_compensated_distances_2bit,
        2,
        3.0e-6,
        3.0e-6,
        1.0e-3,
        0x2b34015910b34083
    );
    test_unsigned_compensated!(
        unsigned_compensated_distances_1bit,
        1,
        0.0,
        0.0,
        0.0,
        0x09fa14c42a9d7d98
    );
}
