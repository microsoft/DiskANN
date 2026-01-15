/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Scalar quantization training, compression data, and distance comparisons.
//!
//! Scalar quantization works by converting floating point values to small n-bit integers
//! using the formula
//! ```math
//! X' = round((X - B) / a).clamp(0, 2^n - 1)
//! ```
//! where `B` is a shift vector and `a` is a scaling parameter. The training algorithm is this:
//!
//! 1. Find the mean vector M of the training data set.
//! 2. Compute the standard deviation of each dimension and find `stdmax`: the maximum
//!    standard deviation.
//! 3. The parameter `a` is then `(2 * S * stdmax) / (2^n - 1)` where `S` is configurable
//!    in [`train::ScalarQuantizationParameters`].
//! 4. `B` is finally computed as `M - S * stdmax`.
//!
//! Compression then via [`ScalarQuantizer`] simply consists of applying the above formula
//! to each vector.
//!
//! # Training
//!
//! Training is provided by the [`train::ScalarQuantizationParameters`] struct.
//! See the struct level documentation for more details.
//!
//! # Compensated Vectors
//!
//! Scalar compression involves data shifts by both the dataset mean and to shift the
//! representable range into a domain representable by an unsigned integer.
//!
//! This does not preserve inner-products at all and can result in a scaled value for
//! L2 computations.
//!
//! To deal with this, we can use [`CompensatedVector`]s, which consist of both the scalar
//! quantization integer codes and the scaled inner product of the compressed value with the
//! data set mean. This enables inner-products to be computed efficiently and correctly, at
//! the cost of some extra storage.
//!
//! See the struct-level documentation for more information as well as the documentation
//! for distance function objects:
//!
//! * [`CompensatedSquaredL2`]
//! * [`CompensatedIP`]
//!
//! # Example
//!
//! In this example, we will do the following:
//!
//! 1. Create a sample data set with normally distributed data using random offsets.
//! 2. Train a 4-bit scalar quantizer on this data.
//! 3. Compress elements of the dataset using the quantizer.
//! 4. Perform distance computations using the compressed vectors.
//!
//! ```
//! use diskann_quantization::{
//!     AsFunctor, CompressInto,
//!     distances,
//!     scalar::{self, train, CompensatedVector, CompensatedIP, CompensatedSquaredL2},
//!     num::Positive,
//! };
//! use diskann_utils::{Reborrow, ReborrowMut, views::Matrix};
//! use rand::{rngs::StdRng, SeedableRng, distr::Distribution};
//! use rand_distr::StandardNormal;
//! use diskann_vector::{PureDistanceFunction, DistanceFunction, distance};
//!
//! let dim = 20;
//! let nvectors = 100;
//! let distribution = StandardNormal;
//! let mut rng = StdRng::seed_from_u64(0xc674c06f4f8013f7);
//! // Construct a set of offsets for each dimension.
//! let offset: Vec<f32> = (0..dim).map(|_| distribution.sample(&mut rng)).collect();
//! // The output dataset.
//! let mut data = Matrix::<f32>::new(0.0, nvectors, dim);
//! for row in data.row_iter_mut() {
//!     std::iter::zip(row.iter_mut(), offset.iter()).for_each(|(r, i)| {
//!         let v: f32 = distribution.sample(&mut rng);
//!         *r = i + v;
//!     });
//! }
//!
//! // Create parameters for 4-bit compression.
//! // Here, we use 2.0 standard deviations are our cutoff.
//! let p = train::ScalarQuantizationParameters::new(Positive::new(2.0).unwrap());
//!
//! // Train a quantizer.
//! let quantizer: scalar::ScalarQuantizer = p.train(data.as_view());
//!
//! // Compress vectors 0 and 1 into their scalar quantized representation.
//! let mut v0 = CompensatedVector::<4>::new_boxed(quantizer.dim());
//! let mut v1 = CompensatedVector::<4>::new_boxed(quantizer.dim());
//! quantizer.compress_into(data.row(0), v0.reborrow_mut()).unwrap();
//! quantizer.compress_into(data.row(1), v1.reborrow_mut()).unwrap();
//!
//! // Compute inner product distances.
//! let f: CompensatedIP = quantizer.as_functor();
//! let distance_quantized: distances::Result<f32> = f.evaluate_similarity(
//!     v0.reborrow(),
//!     v1.reborrow()
//! );
//! let distance_full: f32 = distance::InnerProduct::evaluate(data.row(0), data.row(1));
//! let relative_error = (distance_quantized.unwrap() - distance_full).abs() / distance_full.abs();
//! assert!(relative_error < 0.02);
//!
//! // Compute squared l2 distances.
//! let f: CompensatedSquaredL2 = quantizer.as_functor();
//! let distance_quantized: distances::Result<f32> = f.evaluate_similarity(
//!     v0.reborrow(),
//!     v1.reborrow()
//! );
//! let distance_full: f32 = distance::SquaredL2::evaluate(data.row(0), data.row(1));
//! let relative_error = (distance_quantized.unwrap() - distance_full).abs() / distance_full.abs();
//! assert!(relative_error < 0.03);
//! ```

pub mod train;

mod quantizer;
mod vectors;

/////////////
// Exports //
/////////////

/// Return the scaling parameter `a` that should be multiplied to compressed vector codes.
pub const fn bit_scale<const NBITS: usize>() -> f32 {
    (2_usize.pow(NBITS as u32) - 1) as f32
}

pub const fn inverse_bit_scale<const NBITS: usize>() -> f32 {
    1.0 / bit_scale::<NBITS>()
}

// The central scalar quantization schema.
// Error types.
pub use quantizer::{InputContainsNaN, MeanNormMissing, ScalarQuantizer};
// Distance Functions.
pub use vectors::CompensatedIP;
// Scalar-quantized specific vector types.
pub use vectors::CompensatedVector;
pub use vectors::{
    CompensatedCosineNormalized, CompensatedSquaredL2, CompensatedVectorRef, Compensation,
    MutCompensatedVectorRef,
};
