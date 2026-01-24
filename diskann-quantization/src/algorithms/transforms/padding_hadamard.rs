/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

#[cfg(feature = "flatbuffers")]
use flatbuffers::{FlatBufferBuilder, WIPOffset};
use rand::{
    distr::{Distribution, StandardUniform},
    Rng,
};
use thiserror::Error;

#[cfg(feature = "flatbuffers")]
use super::utils::{bool_to_sign, sign_to_bool};
use super::{
    utils::{check_dims, is_sign, subsample_indices, TransformFailed},
    TargetDim,
};
#[cfg(feature = "flatbuffers")]
use crate::flatbuffers as fb;
use crate::{
    algorithms::hadamard_transform,
    alloc::{Allocator, AllocatorError, Poly, ScopedAllocator, TryClone},
    utils,
};

/// A Hadamard transform that zero pads non-power-of-two dimensions to the next power of two.
///
/// This struct performs the transformation
/// ```math
/// HDx / sqrt(n)
/// ```
/// where
///
/// * `H` is a Hadamard Matrix
/// * `D` is a diagonal matrix with diagonal entries in `{-1, +1}`.
/// * `x` is the vector to transform, zero padded to have a length that is a multiple of two.
/// * `n` is the output-dimension.
#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub struct PaddingHadamard<A>
where
    A: Allocator,
{
    /// A vector of `+/-1` used to add randomness to the Hadamard transform.
    ///
    /// This is stored as a `Vec<u32>` instead of something more representative like a
    /// `Vec<bool>` because we store the sign-bits for the `f32` representation explicitly
    /// to turn sign flipping into a cheap `xor` operation.
    ///
    /// An internal invariant is that each value is either `0` or `0x8000_0000`.
    ///
    /// # Details
    ///
    /// On `x86` - a vectorized `xor` has a latency of 1 and a clocks-per-instruction (CPI)
    /// of 0.333 where-as a `f32` multiply has a latency of 4 and a CPI 0.5.
    signs: Poly<[u32], A>,

    /// The padded-up dimension pre-rotation. This should always be a power of two and
    /// greater than `signs`.
    padded_dim: usize,

    /// Indices of sub-sampled points. This should be sorted to provide more efficient
    /// memory access. If `None`, then no subsampling is performed.
    subsample: Option<Poly<[u32], A>>,
}

impl<A> PaddingHadamard<A>
where
    A: Allocator,
{
    /// Construct a new `PaddingHadamard` that transforms input vectors of dimension `dim`.
    ///
    /// The parameter `rng` is used to randomly initialize the diagonal matrix portion of
    /// the transform.
    ///
    /// The following dimensionalities will be configured depending on the value of `target`:
    ///
    /// * [`TargetDim::Same`]
    ///   - `self.input_dim() == dim.get()`
    ///   - `self.output_dim() == dim.get()`
    /// * [`TargetDim::Natural`]
    ///   - `self.input_dim() == dim.get()`
    ///   - `self.output_dim() == dim.get().next_power_of_two()`
    /// * [`TargetDim::Override`]
    ///   - `self.input_dim() == dim.get()`
    ///   - `self.output_dim()`: The value provided by the override.
    ///
    /// Subsampling occurs if `self.output_dim()` is not a power of two and greater-than or
    /// equal to `self.input_dim()`.
    pub fn new<R>(
        dim: NonZeroUsize,
        target: TargetDim,
        rng: &mut R,
        allocator: A,
    ) -> Result<Self, AllocatorError>
    where
        R: Rng + ?Sized,
    {
        let signs = Poly::from_iter(
            (0..dim.get()).map(|_| {
                let sign: bool = StandardUniform {}.sample(rng);
                if sign {
                    0x8000_0000
                } else {
                    0
                }
            }),
            allocator.clone(),
        )?;

        let (padded_dim, target_dim) = match target {
            TargetDim::Same => (dim.get().next_power_of_two(), dim.get()),
            TargetDim::Natural => {
                let next = dim.get().next_power_of_two();
                (next, next)
            }
            TargetDim::Override(target) => {
                (target.max(dim).get().next_power_of_two(), target.get())
            }
        };

        let subsample = if padded_dim > target_dim {
            Some(subsample_indices(rng, padded_dim, target_dim, allocator)?)
        } else {
            None
        };

        Ok(Self {
            signs,
            padded_dim,
            subsample,
        })
    }

    /// Construct `Self` from constituent parts. This validates that the necessary
    /// invariants hold for the constituent parts, returning an error if they do not.
    pub fn try_from_parts(
        signs: Poly<[u32], A>,
        padded_dim: usize,
        subsample: Option<Poly<[u32], A>>,
    ) -> Result<Self, PaddingHadamardError> {
        if !signs.iter().copied().all(is_sign) {
            return Err(PaddingHadamardError::InvalidSignRepresentation);
        }

        if signs.len() > padded_dim {
            return Err(PaddingHadamardError::SignsTooLong);
        }

        if !padded_dim.is_power_of_two() {
            return Err(PaddingHadamardError::DimNotPowerOfTwo);
        }

        if let Some(ref subsample) = subsample {
            if !utils::is_strictly_monotonic(subsample.iter()) {
                return Err(PaddingHadamardError::SubsampleNotMonotonic);
            }

            if let Some(last) = subsample.last() {
                if *last as usize >= padded_dim {
                    return Err(PaddingHadamardError::LastSubsampleTooLarge);
                }
            } else {
                return Err(PaddingHadamardError::SubsampleEmpty);
            }
        }

        Ok(Self {
            signs,
            padded_dim,
            subsample,
        })
    }

    /// Return the input dimension for the transformation.
    pub fn input_dim(&self) -> usize {
        self.signs.len()
    }

    /// Return the output dimension for the transformation.
    pub fn output_dim(&self) -> usize {
        match &self.subsample {
            None => self.padded_dim,
            Some(v) => v.len(),
        }
    }

    /// Return whether or not the transform preserves norms.
    ///
    /// For this transform, norms are not preserved when the output dimensionality is not a
    /// power of two greater than or equal to the input dimensionality.
    pub fn preserves_norms(&self) -> bool {
        self.subsample.is_none()
    }

    /// An internal helper for performing the sign flipping operation.
    //A
    /// # Preconditions
    ///
    /// This function requires (but only checks in debug build) the following pre-conditions
    ///
    /// * `src.len() == self.input_dim()`.
    /// * `dst.len() == self.output_dim()`.
    fn copy_and_flip_signs(&self, dst: &mut [f32], src: &[f32]) {
        debug_assert_eq!(dst.len(), self.padded_dim);
        debug_assert_eq!(src.len(), self.input_dim());

        // Copy the sign bits.
        std::iter::zip(dst.iter_mut(), src.iter())
            .zip(self.signs.iter())
            .for_each(|((dst, src), sign)| *dst = f32::from_bits(src.to_bits() ^ sign));

        // Pad the rest to zero.
        dst.iter_mut()
            .skip(self.input_dim())
            .for_each(|dst| *dst = 0.0);
    }

    /// Perform the transformation of the `src` vector into the `dst` vector.
    ///
    /// # Errors
    ///
    /// Returns an error if
    ///
    /// * `src.len() != self.input_dim()`.
    /// * `dst.len() != self.output_dim()`.
    pub fn transform_into(
        &self,
        dst: &mut [f32],
        src: &[f32],
        allocator: ScopedAllocator<'_>,
    ) -> Result<(), TransformFailed> {
        let input_dim = self.input_dim();
        let output_dim = self.output_dim();
        check_dims(dst, src, input_dim, output_dim)?;

        // If we are not sub-sampling, then we can transform directly into the destination.
        match &self.subsample {
            None => {
                // Copy over values from `src`, applying the sign flipping.
                self.copy_and_flip_signs(dst, src);

                // Lint: We satisfy the pre-condidions for `hadamard_transform` because:
                //
                // 1. `output_dim` is a power of 2 by construction.
                // 2. We've checked that `dst.len() == output_dim`.
                #[allow(clippy::unwrap_used)]
                hadamard_transform(dst).unwrap();
            }
            Some(indices) => {
                let mut tmp = Poly::broadcast(0.0f32, self.padded_dim, allocator)?;

                self.copy_and_flip_signs(&mut tmp, src);

                // Lint: We satisfy the pre-condidions for `hadamard_transform` because:
                //
                // 1. `padded_dim` is a power of 2 by construction.
                // 2. We've checked that `tmp.len() == padded_dim`.
                #[allow(clippy::unwrap_used)]
                hadamard_transform(&mut tmp).unwrap();

                let rescale = ((tmp.len() as f32) / (indices.len() as f32)).sqrt();
                debug_assert_eq!(dst.len(), indices.len());
                std::iter::zip(dst.iter_mut(), indices.iter()).for_each(
                    |(d, i): (&mut f32, &u32)| {
                        *d = tmp[*i as usize] * rescale;
                    },
                );
            }
        }

        Ok(())
    }
}

impl<A> TryClone for PaddingHadamard<A>
where
    A: Allocator,
{
    fn try_clone(&self) -> Result<Self, AllocatorError> {
        Ok(Self {
            signs: self.signs.try_clone()?,
            padded_dim: self.padded_dim,
            subsample: self.subsample.try_clone()?,
        })
    }
}

/// Errors that may occur while constructing a [`PaddingHadamard`] from constituent parts.
#[derive(Debug, Clone, Copy, Error, PartialEq)]
#[non_exhaustive]
pub enum PaddingHadamardError {
    #[error("an invalid sign representation was discovered")]
    InvalidSignRepresentation,
    #[error("`signs` length exceeds `padded_dim`")]
    SignsTooLong,
    #[error("padded dim is not a power of two")]
    DimNotPowerOfTwo,
    #[error("subsample indices cannot be empty")]
    SubsampleEmpty,
    #[error("subsample indices is not monotonic")]
    SubsampleNotMonotonic,
    #[error("last subsample index exceeded `padded_dim`")]
    LastSubsampleTooLarge,
    #[error(transparent)]
    AllocatorError(#[from] AllocatorError),
}

#[cfg(feature = "flatbuffers")]
impl<A> PaddingHadamard<A>
where
    A: Allocator,
{
    /// Pack into a [`crate::flatbuffers::transforms::PaddingHadamard`] serialized representation.
    pub(crate) fn pack<'a, FA>(
        &self,
        buf: &mut FlatBufferBuilder<'a, FA>,
    ) -> WIPOffset<fb::transforms::PaddingHadamard<'a>>
    where
        FA: flatbuffers::Allocator + 'a,
    {
        // First, pack the sign bits as boolean values.
        let signs = buf.create_vector_from_iter(self.signs.iter().copied().map(sign_to_bool));

        // If subsample indices are present - save those as well.
        let subsample = self
            .subsample
            .as_ref()
            .map(|indices| buf.create_vector(indices));

        // Finish up.
        fb::transforms::PaddingHadamard::create(
            buf,
            &fb::transforms::PaddingHadamardArgs {
                signs: Some(signs),
                padded_dim: self.padded_dim as u32,
                subsample,
            },
        )
    }

    /// Attempt to unpack from a [`crate::flatbuffers::transforms::PaddingHadamard`]
    /// serialized representation, returning any error if encountered.
    pub(crate) fn try_unpack(
        alloc: A,
        proto: fb::transforms::PaddingHadamard<'_>,
    ) -> Result<Self, PaddingHadamardError> {
        let signs = Poly::from_iter(proto.signs().iter().map(bool_to_sign), alloc.clone())?;

        let subsample = match proto.subsample() {
            Some(subsample) => Some(Poly::from_iter(subsample.into_iter(), alloc)?),
            None => None,
        };

        Self::try_from_parts(signs, proto.padded_dim() as usize, subsample)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use diskann_utils::lazy_format;
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{
        algorithms::transforms::{test_utils, Transform, TransformKind},
        alloc::GlobalAllocator,
    };

    // Since we use a slightly non-obvious strategy for applying the +/-1 permutation, we
    // test its behavior explicitly.
    #[test]
    fn test_sign_flipping() {
        let mut rng = StdRng::seed_from_u64(0xf8ee12b1e9f33dbd);
        let dim = 14;

        let transform = PaddingHadamard::new(
            NonZeroUsize::new(dim).unwrap(),
            TargetDim::Same,
            &mut rng,
            GlobalAllocator,
        )
        .unwrap();

        assert_eq!(transform.input_dim(), dim);
        assert_eq!(transform.output_dim(), dim);

        let positive = vec![1.0f32; dim];
        let negative = vec![-1.0f32; dim];

        let mut output = vec![f32::INFINITY; 16];

        // Transform positive numbers
        transform.copy_and_flip_signs(&mut output, &positive);

        let mut unflipped = 0;
        let mut flipped = 0;
        std::iter::zip(output.iter(), transform.signs.iter())
            .enumerate()
            .for_each(|(i, (o, s))| {
                if *s == 0x8000_0000 {
                    flipped += 1;
                    assert_eq!(*o, -1.0, "expected entry {} to be flipped", i);
                } else {
                    unflipped += 1;
                    assert_eq!(*o, 1.0, "expected entry {} to be unchanged", i);
                }
            });

        // Check that we have a mixture of flipped and unflipped signs.
        assert!(unflipped > 0);
        assert!(flipped > 0);

        // Assert that everything else was zero padded.
        assert_eq!(output[14], 0.0f32);
        assert_eq!(output[15], 0.0f32);

        // Transform negative numbers
        output.fill(f32::INFINITY);
        transform.copy_and_flip_signs(&mut output, &negative);
        std::iter::zip(output.iter(), transform.signs.iter())
            .enumerate()
            .for_each(|(i, (o, s))| {
                if *s == 0x8000_0000 {
                    assert_eq!(*o, 1.0, "expected entry {} to be flipped", i);
                } else {
                    assert_eq!(*o, -1.0, "expected entry {} to be unchanged", i);
                }
            });

        // Assert that everything else was zero padded.
        assert_eq!(output[14], 0.0f32);
        assert_eq!(output[15], 0.0f32);
    }

    test_utils::delegate_transformer!(PaddingHadamard<GlobalAllocator>);

    // This tests the natural hadamard transform where the output dimension is upgraded
    // to the next power of 2.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_padding_hadamard() {
        // Inner product computations are more susceptible to floating point error.
        // Instead of using ULP here, we fall back to using absolute and relative error.
        //
        // These error bounds are for when we set the output dimenion to a power of 2 that
        // is higher than input dimension.
        let natural_errors = test_utils::ErrorSetup {
            norm: test_utils::Check::ulp(4),
            l2: test_utils::Check::ulp(4),
            ip: test_utils::Check::absrel(5.0e-6, 2e-4),
        };

        // NOTE: Subsampling introduces high variance in the norm and L2, so our error
        // bounds need to be looser.
        //
        // Subsampling results in poor preservation of inner products, so we skip it
        // altogether.
        let subsampled_errors = test_utils::ErrorSetup {
            norm: test_utils::Check::absrel(0.0, 1e-1),
            l2: test_utils::Check::absrel(0.0, 1e-1),
            ip: test_utils::Check::skip(),
        };

        let target_dim = |v| TargetDim::Override(NonZeroUsize::new(v).unwrap());

        let dim_combos = [
            // Natural
            (15, 16, true, target_dim(16), &natural_errors),
            (15, 16, true, TargetDim::Natural, &natural_errors),
            (16, 16, true, TargetDim::Same, &natural_errors),
            (16, 16, true, TargetDim::Natural, &natural_errors),
            (16, 32, true, target_dim(32), &natural_errors),
            (16, 64, true, target_dim(64), &natural_errors),
            (100, 128, true, target_dim(128), &natural_errors),
            (100, 128, true, TargetDim::Natural, &natural_errors),
            (256, 256, true, target_dim(256), &natural_errors),
            // Subsampled,
            (1000, 1000, false, TargetDim::Same, &subsampled_errors),
            (500, 1000, false, target_dim(1000), &subsampled_errors),
        ];

        let trials_per_combo = 20;
        let trials_per_dim = 100;

        let mut rng = StdRng::seed_from_u64(0x6d1699abe0626147);
        for (input, output, preserves_norms, target, errors) in dim_combos {
            let input_nz = NonZeroUsize::new(input).unwrap();
            for trial in 0..trials_per_combo {
                let ctx = lazy_format!(
                    "input dim = {}, output dim = {}, macro trial {} of {}",
                    input,
                    output,
                    trial,
                    trials_per_combo
                );

                let mut checker = |io: test_utils::IO<'_>, context: &dyn std::fmt::Display| {
                    assert_ne!(io.input0, &io.output0[..input]);
                    assert_ne!(io.input1, &io.output1[..input]);
                    test_utils::check_errors(io, context, errors);
                };

                // Clone the Rng state so the abstract transform behaves the same.
                let mut rng_clone = rng.clone();

                // Base Transformer
                {
                    let transformer = PaddingHadamard::new(
                        NonZeroUsize::new(input).unwrap(),
                        target,
                        &mut rng,
                        GlobalAllocator,
                    )
                    .unwrap();

                    assert_eq!(transformer.input_dim(), input);
                    assert_eq!(transformer.output_dim(), output);
                    assert_eq!(transformer.preserves_norms(), preserves_norms);

                    test_utils::test_transform(
                        &transformer,
                        trials_per_dim,
                        &mut checker,
                        &mut rng,
                        &ctx,
                    )
                }

                // Abstract Transformer
                {
                    let kind = TransformKind::PaddingHadamard { target_dim: target };
                    let transformer =
                        Transform::new(kind, input_nz, Some(&mut rng_clone), GlobalAllocator)
                            .unwrap();

                    assert_eq!(transformer.input_dim(), input);
                    assert_eq!(transformer.output_dim(), output);
                    assert_eq!(transformer.preserves_norms(), preserves_norms);

                    test_utils::test_transform(
                        &transformer,
                        trials_per_dim,
                        &mut checker,
                        &mut rng_clone,
                        &ctx,
                    )
                }
            }
        }
    }

    #[cfg(feature = "flatbuffers")]
    mod serialization {
        use super::*;
        use crate::{flatbuffers::to_flatbuffer, poly};

        #[test]
        fn padding_hadamard() {
            let mut rng = StdRng::seed_from_u64(0x123456789abcdef0);
            let alloc = GlobalAllocator;

            // Test various dimension combinations
            let test_cases = [
                (5, TargetDim::Same),
                (10, TargetDim::Natural),
                (16, TargetDim::Natural),
                (8, TargetDim::Override(NonZeroUsize::new(12).unwrap())),
                (15, TargetDim::Override(NonZeroUsize::new(10).unwrap())),
            ];

            for (dim, target_dim) in test_cases {
                let transform = PaddingHadamard::new(
                    NonZeroUsize::new(dim).unwrap(),
                    target_dim,
                    &mut rng,
                    alloc,
                )
                .unwrap();
                let data = to_flatbuffer(|buf| transform.pack(buf));

                let proto = flatbuffers::root::<fb::transforms::PaddingHadamard>(&data).unwrap();
                let reloaded = PaddingHadamard::try_unpack(alloc, proto).unwrap();

                assert_eq!(transform, reloaded);
            }

            let gen_err = |x: PaddingHadamard<_>| -> PaddingHadamardError {
                let data = to_flatbuffer(|buf| x.pack(buf));
                let proto = flatbuffers::root::<fb::transforms::PaddingHadamard>(&data).unwrap();
                PaddingHadamard::try_unpack(alloc, proto).unwrap_err()
            };

            // Signs too longs.
            {
                let err = gen_err(PaddingHadamard {
                    signs: poly!([0, 0, 0, 0, 0], alloc).unwrap(), // longer than `padded_dim`.
                    padded_dim: 4,
                    subsample: None,
                });

                assert_eq!(err, PaddingHadamardError::SignsTooLong);
            }

            // Dim Not a power of 2.
            {
                let err = gen_err(PaddingHadamard {
                    signs: poly!([0, 0, 0, 0, 0], alloc).unwrap(),
                    padded_dim: 5, // not a power of 2
                    subsample: None,
                });

                assert_eq!(err, PaddingHadamardError::DimNotPowerOfTwo);
            }

            // Subsample empty
            {
                let err = gen_err(PaddingHadamard {
                    signs: poly!([0, 0, 0, 0], alloc).unwrap(),
                    padded_dim: 4,
                    subsample: Some(poly!([], alloc).unwrap()), // empty
                });

                assert_eq!(err, PaddingHadamardError::SubsampleEmpty);
            }

            // Not monotonic
            {
                let err = gen_err(PaddingHadamard {
                    signs: poly!([0, 0, 0, 0], alloc).unwrap(),
                    padded_dim: 4,
                    subsample: Some(poly!([0, 2, 2], alloc).unwrap()), // not monotonic
                });
                assert_eq!(err, PaddingHadamardError::SubsampleNotMonotonic);
            }

            // Subsample too long.
            {
                let err = gen_err(PaddingHadamard {
                    signs: poly!([0, 0, 0, 0], alloc).unwrap(),
                    padded_dim: 4,
                    subsample: Some(poly!([0, 1, 2, 3, 4], alloc).unwrap()),
                });

                assert_eq!(err, PaddingHadamardError::LastSubsampleTooLarge);
            }

            // Subsample too large
            {
                let err = gen_err(PaddingHadamard {
                    signs: poly!([0, 0, 0, 0], alloc).unwrap(),
                    padded_dim: 4,
                    subsample: Some(poly!([0, 4], alloc).unwrap()),
                });

                assert_eq!(err, PaddingHadamardError::LastSubsampleTooLarge);
            }
        }
    }
}
