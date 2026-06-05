/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::vectors::{DataMutRef, DataRef, MinMaxCompensation};
use crate::CompressInto;
use crate::bits::{Representation, Unsigned};
use crate::num::Positive;
use crate::scalar::bit_scale;
use thiserror::Error;

/// Recompression utilities for MinMax quantized vectors.
///
/// This struct provides functionality to further compress MinMax quantized
/// vectors from a source bitrate `N` to a target bitrate `M` for `N` > `M`.
///
/// A positive `grid_scale` is used to (optionally) tighten or widen the
/// reconstruction range that the target codes span before re-quantization.
/// `grid_scale == 1.0` preserves the source reconstruction range exactly
/// (legacy behavior). `grid_scale < 1.0` narrows the range, dedicating more
/// resolution to interior values at the cost of clipping the extremes;
/// `grid_scale > 1.0` widens the range (coarser steps, no clipping).
///
/// # Notes
/// - Currently this API only supports the following conversions: 8 -> 4, 8 -> 2, 4 -> 2
///
/// # Example
///
/// ```rust
/// use std::num::NonZeroUsize;
/// use diskann_quantization::algorithms::{Transform, transforms::NullTransform};
/// use diskann_quantization::minmax::{Data, MinMaxQuantizer, Recompressor};
/// use diskann_quantization::num::Positive;
/// use diskann_quantization::CompressInto;
/// use diskann_utils::{Reborrow, ReborrowMut};
///
/// // Create a quantizer and compress an f32 vector to 8-bit
/// let vector = vec![0.1, -0.5, 0.8, -0.2];
/// let quantizer = MinMaxQuantizer::new(
///     Transform::Null(NullTransform::new(NonZeroUsize::new(4).unwrap())),
///     Positive::new(1.0).unwrap(),
/// );
///
/// let mut encoded_8 = Data::<8>::new_boxed(4);
/// quantizer.compress_into(vector.as_slice(), encoded_8.reborrow_mut()).unwrap();
///
/// // Recompress from 8-bit to 4-bit, preserving the source range
/// let recompressor = Recompressor::new(Positive::new(1.0).unwrap());
/// let mut encoded_4 = Data::<4>::new_boxed(4);
/// recompressor.compress_into(encoded_8.reborrow(), encoded_4.reborrow_mut()).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Recompressor {
    grid_scale: Positive<f32>,
}

impl Recompressor {
    /// Construct a new `Recompressor` with the given positive `grid_scale`.
    ///
    /// `grid_scale == 1.0` reproduces the legacy behavior of preserving the
    /// source reconstruction range exactly.
    pub fn new(grid_scale: Positive<f32>) -> Self {
        Self { grid_scale }
    }

    /// Returns the configured grid scale.
    pub fn grid_scale(&self) -> Positive<f32> {
        self.grid_scale
    }
}

/// Error type for recompression operations.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum RecompressError {
    /// Source and destination vectors have different dimensions.
    #[error("dimension mismatch: source has {src} dimensions, destination has {dst}")]
    DimensionMismatch {
        /// Dimension of the source vector.
        src: usize,
        /// Dimension of the destination vector.
        dst: usize,
    },
}

/// Macro to implement `CompressInto<DataRef<'_, N>, DataMutRef<'_, M>>` for M > 1.
macro_rules! impl_recompress {
    ($n:literal -> $m:literal) => {
        impl<'a, 'b> CompressInto<DataRef<'a, $n>, DataMutRef<'b, $m>> for Recompressor
        where
            Unsigned: Representation<$n> + Representation<$m>,
        {
            type Error = RecompressError;
            type Output = ();

            fn compress_into(
                &self,
                from: DataRef<'a, $n>,
                to: DataMutRef<'b, $m>,
            ) -> Result<(), Self::Error> {
                recompress_kernel::<$n, $m>(from, to, self.grid_scale.into_inner())
            }
        }
    };
}

impl_recompress!(8 -> 4);
impl_recompress!(8 -> 2);
impl_recompress!(4 -> 2);

////////////////////////////////////
// Recompression Kernel for M > 1 //
////////////////////////////////////

/// Recompress N-bit codes to M-bit codes where M > 1.
///
/// Recall from the algorithm for minmax described in
/// [`crate::minmax::MinMaxQuantizer`], the encoding of a vector
/// `X` into `N`-bits per dimension using minmax is given by:
///
/// ```text
/// X' = round((X - b) * a).clamp(0, 2^n - 1))
/// ```
///
/// where `b = min_i X_i` and `a = max_i X_i - b / (2^N - 1)`.
///
/// The source reconstruction range is `[b, b + a * (2^N - 1)]`.
///
/// Let `mid = b + a*(2^N-1)/2` and `w = a*(2^N-1)/2`.
/// Given a positive `grid_scale` (`g`), the target
/// reconstruction range is `[mid - w*g, mid + w*g]`,
/// which yields:
///
/// ```text
///   new_b = mid - w*g = b + a*(2^N-1)*(1-g)/2
///   new_a = 2*w*g / (2^M - 1) = a*(2^N-1)*g/(2^M - 1)
/// ```
///
/// For each source code `old_code`, the reconstructed value is
/// `X_hat = old_code * a + b`, and the new code is
///
/// ```text
///   new_code = round((X_hat - new_b) / new_a).clamp(0, 2^M - 1)
///            = round((old_code - offset) * code_scale).clamp(0, 2^M - 1)
/// ```
///
/// where `offset = (2^N-1)*(1-g)/2` and `code_scale = (2^M-1) / (g * (2^N-1))`.
#[inline(always)]
fn recompress_kernel<const N: usize, const M: usize>(
    from: DataRef<'_, N>,
    mut to: DataMutRef<'_, M>,
    grid_scale: f32,
) -> Result<(), RecompressError>
where
    Unsigned: Representation<N> + Representation<M>,
{
    const { assert!(N > M, "source bit width must exceed target bits") };
    const { assert!(M > 1, "target bit width must exceed 1") };

    // Validate dimensions
    let dim = from.len();
    if dim != to.vector().len() {
        return Err(RecompressError::DimensionMismatch {
            src: dim,
            dst: to.vector().len(),
        });
    }

    let src_meta = from.meta();
    let src_a = src_meta.a;
    let src_b = src_meta.b;

    let scale_n = bit_scale::<N>();
    let scale_m = bit_scale::<M>();

    // re-center the source reconstruction range by `grid_scale`.
    let offset = scale_n * (1.0 - grid_scale) * 0.5;
    let code_scale = scale_m / (scale_n * grid_scale);

    // new_a / new_b in source-value space.
    let new_a = src_a / code_scale;
    let new_b = src_b + src_a * offset;

    // Single pass: encode and compute statistics
    let from_vec = from.vector();
    let mut to_vec = to.vector_mut();

    let mut code_sum: f32 = 0.0;
    let mut norm_squared: f32 = 0.0;

    for i in 0..dim {
        // Read source code
        // SAFETY: we checked that `dim == from.len() == src.len()`
        let old_code = unsafe { from_vec.get_unchecked(i) };
        let old_code_f = old_code as f32;

        // new code
        let new_code = ((old_code_f - offset) * code_scale)
            .round_ties_even()
            .clamp(0.0, scale_m) as u8;

        // Write destination code
        // SAFETY: we checked that `dim == from.len() == src.len()`
        unsafe { to_vec.set_unchecked(i, new_code) };

        // Accumulate statistics using the actual truncated integer code
        let new_code_f = new_code as f32;
        code_sum += new_code_f;

        // Reconstruct value for norm computation
        let v_m = new_code_f * new_a + new_b;

        norm_squared += v_m * v_m;
    }

    // Construct metadata
    to.set_meta(MinMaxCompensation {
        dim: dim as u32,
        b: new_b,
        a: new_a,
        n: new_a * code_sum,
        norm_squared,
    });

    Ok(())
}

#[cfg(test)]
mod recompress_tests {
    use std::num::NonZeroUsize;

    use diskann_utils::{Reborrow, ReborrowMut};
    use rand::{
        SeedableRng,
        distr::{Distribution, Uniform},
        rngs::StdRng,
    };

    use super::*;
    use crate::{
        algorithms::{Transform, transforms::NullTransform},
        minmax::quantizer::MinMaxQuantizer,
        minmax::vectors::Data,
        num::Positive,
    };

    /// Reconstruct a MinMax quantized vector to f32 values.
    fn reconstruct<const NBITS: usize>(v: DataRef<'_, NBITS>) -> Vec<f32>
    where
        Unsigned: Representation<NBITS>,
    {
        let meta = v.meta();
        (0..v.len())
            .map(|i| v.vector().get(i).unwrap() as f32 * meta.a + meta.b)
            .collect()
    }

    /// Test recompression from N bits to M bits with random vectors and a
    /// specific `grid_scale`. The recompressed codes are compared **bit for
    /// bit** against an inline oracle implementation derived directly from
    /// the kernel's documented formula, and the emitted metadata is checked
    /// for consistency with the actually-written codes.
    fn test_recompress_random<const N: usize, const M: usize>(
        dim: usize,
        grid_scale: f32,
        rng: &mut StdRng,
    ) where
        Unsigned: Representation<N> + Representation<M>,
        MinMaxQuantizer: for<'a, 'b> CompressInto<&'a [f32], DataMutRef<'b, N>>,
        Recompressor: for<'a, 'b> CompressInto<DataRef<'a, N>, DataMutRef<'b, M>, Output = ()>,
    {
        let distribution = Uniform::new_inclusive::<f32, f32>(-1.0, 1.0).unwrap();
        let quantizer = MinMaxQuantizer::new(
            Transform::Null(NullTransform::new(NonZeroUsize::new(dim).unwrap())),
            Positive::new(1.0).unwrap(),
        );

        let g = Positive::new(grid_scale).unwrap();

        let recompressor = Recompressor::new(g);

        assert_eq!(recompressor.grid_scale(), g);

        // Generate random vector and compress to N bits
        let vector: Vec<f32> = distribution.sample_iter(rng).take(dim).collect();
        let mut encoded_n = Data::<N>::new_boxed(dim);
        quantizer
            .compress_into(&*vector, encoded_n.reborrow_mut())
            .unwrap();

        // Recompress to M bits
        let mut encoded_m = Data::<M>::new_boxed(dim);
        recompressor
            .compress_into(encoded_n.reborrow(), encoded_m.reborrow_mut())
            .unwrap();

        // ---- Oracle: recompute the expected M-bit codes from scratch using
        // the same formula the kernel documents. This is independent of the
        // kernel implementation but uses the same source 8-bit codes.
        let scale_n = ((1u64 << N) - 1) as f32;
        let scale_m = ((1u64 << M) - 1) as f32;
        let offset = scale_n * (1.0 - grid_scale) * 0.5;
        let code_scale = scale_m / (scale_n * grid_scale);
        let expected_codes: Vec<u8> = (0..dim)
            .map(|i| {
                let c = encoded_n.vector().get(i).unwrap() as f32;
                ((c - offset) * code_scale)
                    .round_ties_even()
                    .clamp(0.0, scale_m) as u8
            })
            .collect();

        // ---- Bit-for-bit assertion against the oracle.
        for (i, &expected) in expected_codes.iter().enumerate() {
            let actual = encoded_m.vector().get(i).unwrap() as u8;
            assert_eq!(
                actual, expected,
                "code mismatch at dim={dim}, g={grid_scale}, i={i}: expected {expected}, got {actual}"
            );
        }

        // ---- Metadata consistency with the emitted codes.
        let meta_m = encoded_m.meta();
        assert_eq!(meta_m.dim as usize, dim, "Dimension should be preserved");

        let actual_code_sum: f32 = (0..dim)
            .map(|i| encoded_m.vector().get(i).unwrap() as f32)
            .sum();
        let computed_code_sum = meta_m.n / meta_m.a;
        assert!(
            (computed_code_sum - actual_code_sum).abs() < 1e-4,
            "Code sum mismatch at g={grid_scale}: expected {actual_code_sum}, got {computed_code_sum}"
        );

        let reconstructed_m = reconstruct(encoded_m.reborrow());
        let expected_norm_sq: f32 = reconstructed_m.iter().map(|x| x * x).sum();
        assert!(
            (meta_m.norm_squared - expected_norm_sq).abs() < 1e-4,
            "norm_squared mismatch at g={grid_scale}: expected {expected_norm_sq}, got {}",
            meta_m.norm_squared
        );
    }

    cfg_if::cfg_if! {
        if #[cfg(miri)] {
            const TRIALS: usize = 2;
            const MAX_DIM: usize = 20;
        } else {
            const TRIALS: usize = 10;
            const MAX_DIM: usize = 100;
        }
    }

    /// Grid of `grid_scale` values exercised by every `test_recompress_pair!`
    /// invocation. Includes `1.0` (legacy-equivalent), `< 1.0` (narrowing,
    /// exercises clamping), and `> 1.0` (widening).
    const GRID_SCALES: &[f32] = &[1.0, 0.8, 0.6, 1.2];

    macro_rules! test_recompress_pair {
        ($name:ident, $n:literal -> $m:literal, $seed:literal) => {
            #[test]
            fn $name() {
                let mut rng = StdRng::seed_from_u64($seed);
                for dim in 10..=MAX_DIM {
                    for _ in 0..TRIALS {
                        for &g in GRID_SCALES {
                            test_recompress_random::<$n, $m>(dim, g, &mut rng);
                        }
                    }
                }
            }
        };
    }

    test_recompress_pair!(recompress_8_to_4, 8 -> 4, 0xabc123def456);
    test_recompress_pair!(recompress_8_to_2, 8 -> 2, 0xdef456abc123);
    test_recompress_pair!(recompress_4_to_2, 4 -> 2, 0x456def123abc);

    #[test]
    fn test_dimension_mismatch_error() {
        // SAFETY: I'm positive that 1.0 is positive.
        let recompressor = Recompressor::new(Positive::<f32>::new(1.0).unwrap());

        let mut src = Data::<8>::new_boxed(10);
        src.set_meta(MinMaxCompensation {
            dim: 10,
            b: 0.0,
            a: 1.0,
            n: 0.0,
            norm_squared: 0.0,
        });

        let mut dst = Data::<4>::new_boxed(15); // Different dimension

        let result: Result<(), RecompressError> =
            recompressor.compress_into(src.reborrow(), dst.reborrow_mut());

        assert_eq!(
            result.unwrap_err(),
            RecompressError::DimensionMismatch { src: 10, dst: 15 }
        );
    }

    #[test]
    fn test_constant_value_vector() {
        let dim = 30;
        let quantizer = MinMaxQuantizer::new(
            Transform::Null(NullTransform::new(NonZeroUsize::new(dim).unwrap())),
            Positive::new(1.0).unwrap(),
        );

        // SAFETY: I'm positive that 1.0 is positive.
        let recompressor = Recompressor::new(Positive::<f32>::new(1.0).unwrap());

        let constant_value = 42.5f32;
        let vector = vec![constant_value; dim];

        // Compress to 8 bits
        let mut encoded_8 = Data::<8>::new_boxed(dim);
        quantizer
            .compress_into(&*vector, encoded_8.reborrow_mut())
            .unwrap();

        // Recompress to 4 bits
        let mut encoded_4 = Data::<4>::new_boxed(dim);
        recompressor
            .compress_into(encoded_8.reborrow(), encoded_4.reborrow_mut())
            .unwrap();

        // For constant value, all codes should be the same
        let first_code = encoded_4.vector().get(0).unwrap();
        for i in 1..dim {
            assert_eq!(
                encoded_4.vector().get(i).unwrap(),
                first_code,
                "All codes should be identical for constant-value vector"
            );
        }

        // Reconstruction should be close to original
        let reconstructed = reconstruct(encoded_4.reborrow());
        for &val in &reconstructed {
            assert!(
                (val - constant_value).abs() < 1.0,
                "Reconstructed value {} should be close to original {}",
                val,
                constant_value
            );
        }
    }
}
