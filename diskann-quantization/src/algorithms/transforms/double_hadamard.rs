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

/// A Double Hadamard transform that applies the signed Hadamard Transform to a head of the
/// vector and then the tail.
///
/// This struct performs the transformation
/// ```math
/// [I 0; 0 H/sqrt(t)] · D1 · [H/sqrt(t) 0; 0 I] · zeropad(D0 · x)
/// ```
///
/// * `n` is the dimensionality of the input vector.
/// * `m` is the desired output dimensionality.
/// * `o = max(n, m)` is an intermediate dimension.
/// * `t` is the largest power of 2 less than or equal to `o`.
/// * `H` is a Hadamard Matrix of dimension `t`,
/// * `I` is the identity matrix of dimension `n - t`
/// * `D0` and `D1` are diagonal matrices with diagonal entries in `{-1, +1}` drawn
///   uniformly at random with lengths `n` and `o` respectively.
/// * `x` is the input vector of dimension `n`
/// * `[A 0; 0 B]` represents a block diagonal matrix with blocks `A` and `B`.
/// * `zeropad` indicates that the result `D0 · x` is zero-padded to the dimension `o` if
///   needed.
///
/// As indicated above, if `o` is a power of two, then only a single transformation is
/// applied. Further, if `o` exceeds `n`, then the input vector is zero padded at the end to
/// `o` dimensions.
#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub struct DoubleHadamard<A>
where
    A: Allocator,
{
    /// Vectors of `+/-1` used for to add randomness to the Hadamard transform
    /// in each step.
    ///
    /// These are stored as a slice of `u32` where each value is either `0` or `0x8000_0000`,
    /// corresponding to the sign-bit for an `f32` value, allowing sign flipping using
    /// a cheap `xor` operation.
    signs0: Poly<[u32], A>,
    signs1: Poly<[u32], A>,

    /// The target output dimension of the transformation.
    target_dim: usize,

    /// Optional array storing (in sorted order) the indices to sample if `target_dim < dim`
    subsample: Option<Poly<[u32], A>>,
}

impl<A> DoubleHadamard<A>
where
    A: Allocator,
{
    /// Construct a new `DoubleHadamard` that transforms input vectors of dimension `dim`.
    ///
    /// The parameter `rng` is used to randomly initialize the diagonal matrices portion of
    /// the transform.
    ///
    /// The following dimensionalities will be configured depending on the value of `target`:
    ///
    /// * [`TargetDim::Same`]
    ///   - `self.input_dim() == dim.get()`
    ///   - `self.output_dim() == dim.get()`
    /// * [`TargetDim::Natural`]
    ///   - `self.input_dim() == dim.get()`
    ///   - `self.output_dim() == dim.get()`
    /// * [`TargetDim::Override`]
    ///   - `self.input_dim() == dim.get()`
    ///   - `self.output_dim()`: The value provided by the override.
    ///
    /// Subsampling occurs if `self.output_dim()` is smaller than `self.input_dim()`.
    pub fn new<R>(
        dim: NonZeroUsize,
        target_dim: TargetDim,
        rng: &mut R,
        allocator: A,
    ) -> Result<Self, AllocatorError>
    where
        R: Rng + ?Sized,
    {
        let dim = dim.get();

        let target_dim = match target_dim {
            TargetDim::Override(target) => target.get(),
            TargetDim::Same => dim,
            TargetDim::Natural => dim,
        };

        // The intermediate dimension after applying the first transform.
        //
        // If `target_dim` exceeds `dim`, then we perform zero padding up to `target_dim`
        // for this stage.
        let intermediate_dim = dim.max(target_dim);

        // Generate random signs for the diagonal matrices
        let mut sample = |_: usize| {
            let sign: bool = StandardUniform {}.sample(rng);
            if sign {
                0x8000_0000
            } else {
                0
            }
        };

        // Since implicit zero padding is used for this stage, we only create space for
        // `dim` values.
        let signs0 = Poly::from_iter((0..dim).map(&mut sample), allocator.clone())?;
        let signs1 = Poly::from_iter((0..intermediate_dim).map(&mut sample), allocator.clone())?;

        let subsample = if dim > target_dim {
            Some(subsample_indices(rng, dim, target_dim, allocator)?)
        } else {
            None
        };

        Ok(Self {
            signs0,
            signs1,
            target_dim,
            subsample,
        })
    }

    pub fn try_from_parts(
        signs0: Poly<[u32], A>,
        signs1: Poly<[u32], A>,
        subsample: Option<Poly<[u32], A>>,
    ) -> Result<Self, DoubleHadamardError> {
        type E = DoubleHadamardError;
        if signs0.is_empty() {
            return Err(E::Signs0Empty);
        }
        if signs1.len() < signs0.len() {
            return Err(E::Signs1TooSmall);
        }
        if !signs0.iter().copied().all(is_sign) {
            return Err(E::Signs0Invalid);
        }
        if !signs1.iter().copied().all(is_sign) {
            return Err(E::Signs1Invalid);
        }

        // Some preliminary checks on `subsample` that must always hold if it is present.
        let target_dim = if let Some(ref subsample) = subsample {
            if !utils::is_strictly_monotonic(subsample.iter()) {
                return Err(E::SubsampleNotMonotonic);
            }

            match subsample.last() {
                Some(last) => {
                    if *last as usize >= signs1.len() {
                        // Since the entries in `subsample` are used to index an output
                        // vector of lengths `signs1`, the last element must be strictly
                        // less than this length.
                        //
                        // From the monotonicity check, we can therefore deduce that *all*
                        // entries are in-bounds.
                        return Err(E::LastSubsampleTooLarge);
                    }
                }
                None => {
                    // Subsample cannot be empty.
                    return Err(E::InvalidSubsampleLength);
                }
            }

            debug_assert!(
                subsample.len() < signs1.len(),
                "since we've verified monotonicity and the last element, this is implied"
            );

            subsample.len()
        } else {
            // With no subsampling, the target dim is the length of `signs1`.
            signs1.len()
        };

        Ok(Self {
            signs0,
            signs1,
            target_dim,
            subsample,
        })
    }

    /// Return the input dimension for the transformation.
    pub fn input_dim(&self) -> usize {
        self.signs0.len()
    }

    /// Return the output dimension for the transformation.
    pub fn output_dim(&self) -> usize {
        self.target_dim
    }

    /// Return whether or not the transform preserves norms.
    ///
    /// For this transform, norms are not preserved when the output dimensionality is less
    /// than the input dimensionality.
    pub fn preserves_norms(&self) -> bool {
        self.subsample.is_none()
    }

    fn intermediate_dim(&self) -> usize {
        self.input_dim().max(self.output_dim())
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
        check_dims(dst, src, self.input_dim(), self.output_dim())?;

        // Copy and flip signs
        let intermediate_dim = self.intermediate_dim();
        let mut tmp = Poly::broadcast(0.0f32, intermediate_dim, allocator)?;

        std::iter::zip(tmp.iter_mut(), src.iter())
            .zip(self.signs0.iter())
            .for_each(|((dst, src), sign)| *dst = f32::from_bits(src.to_bits() ^ sign));

        let split = 1usize << (usize::BITS - intermediate_dim.leading_zeros() - 1);

        // `split` is less than or equal to `tmp` and is a power to 2.
        //
        // If it is equal to the size of `tmp`, then we only run the first transform. Otherwise,
        // we perform two transforms on the head and tail of `tmp`.
        #[allow(clippy::unwrap_used)]
        hadamard_transform(&mut tmp[..split]).unwrap();

        // Apply the second transformation.
        // Since random signs are applied to the intermediate value, the second transform
        // does not undo the first.
        tmp.iter_mut()
            .zip(self.signs1.iter())
            .for_each(|(dst, sign)| *dst = f32::from_bits(dst.to_bits() ^ sign));

        #[allow(clippy::unwrap_used)]
        hadamard_transform(&mut tmp[intermediate_dim - split..]).unwrap();

        match self.subsample.as_ref() {
            None => {
                dst.copy_from_slice(&tmp);
            }
            Some(indices) => {
                let rescale = ((tmp.len() as f32) / (indices.len() as f32)).sqrt();
                debug_assert_eq!(dst.len(), indices.len());
                dst.iter_mut()
                    .zip(indices.iter())
                    .for_each(|(d, s)| *d = tmp[*s as usize] * rescale);
            }
        }

        Ok(())
    }
}

impl<A> TryClone for DoubleHadamard<A>
where
    A: Allocator,
{
    fn try_clone(&self) -> Result<Self, AllocatorError> {
        Ok(Self {
            signs0: self.signs0.try_clone()?,
            signs1: self.signs1.try_clone()?,
            target_dim: self.target_dim,
            subsample: self.subsample.try_clone()?,
        })
    }
}

#[derive(Debug, Clone, Copy, Error, PartialEq)]
#[non_exhaustive]
pub enum DoubleHadamardError {
    #[error("first signs stage cannot be empty")]
    Signs0Empty,
    #[error("first signs stage has invalid coding")]
    Signs0Invalid,

    #[error("invalid sign representation for second stage")]
    Signs1Invalid,
    #[error("second sign stage must be at least as large as the first stage")]
    Signs1TooSmall,

    #[error("subsample length must equal `target_dim`")]
    InvalidSubsampleLength,
    #[error("subsample indices is not monotonic")]
    SubsampleNotMonotonic,
    #[error("last subsample index exceeded intermediate dim")]
    LastSubsampleTooLarge,

    #[error(transparent)]
    AllocatorError(#[from] AllocatorError),
}

// Serialization
#[cfg(feature = "flatbuffers")]
impl<A> DoubleHadamard<A>
where
    A: Allocator,
{
    /// Pack into a [`crate::flatbuffers::transforms::DoubleHadamard`] serialized
    /// represntation.
    pub(crate) fn pack<'a, FA>(
        &self,
        buf: &mut FlatBufferBuilder<'a, FA>,
    ) -> WIPOffset<fb::transforms::DoubleHadamard<'a>>
    where
        FA: flatbuffers::Allocator + 'a,
    {
        // Store the sign vectors.
        let signs0 = buf.create_vector_from_iter(self.signs0.iter().copied().map(sign_to_bool));
        let signs1 = buf.create_vector_from_iter(self.signs1.iter().copied().map(sign_to_bool));

        // If subsample indices are present - save those as well.
        let subsample = self
            .subsample
            .as_ref()
            .map(|indices| buf.create_vector(indices));

        fb::transforms::DoubleHadamard::create(
            buf,
            &fb::transforms::DoubleHadamardArgs {
                signs0: Some(signs0),
                signs1: Some(signs1),
                subsample,
            },
        )
    }

    /// Attempt to unpack from a [`crate::flatbuffers::transforms::DoubleHadamard`]
    /// serialized representation, returning any error if encountered.
    pub(crate) fn try_unpack(
        alloc: A,
        proto: fb::transforms::DoubleHadamard<'_>,
    ) -> Result<Self, DoubleHadamardError> {
        let signs0 = Poly::from_iter(proto.signs0().iter().map(bool_to_sign), alloc.clone())?;
        let signs1 = Poly::from_iter(proto.signs1().iter().map(bool_to_sign), alloc.clone())?;

        let subsample = match proto.subsample() {
            Some(subsample) => Some(Poly::from_iter(subsample.into_iter(), alloc)?),
            None => None,
        };

        Self::try_from_parts(signs0, signs1, subsample)
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

    test_utils::delegate_transformer!(DoubleHadamard<GlobalAllocator>);

    #[test]
    #[cfg(not(miri))]
    fn test_double_hadamard() {
        // Inner product computations are more susceptible to floating point error.
        // Instead of using ULP here, we fall back to using absolute and relative error.
        //
        // These error bounds are for when we set the output dimenion to a power of 2 that
        // is higher than input dimension.
        let natural_errors = test_utils::ErrorSetup {
            norm: test_utils::Check::ulp(5),
            l2: test_utils::Check::ulp(5),
            ip: test_utils::Check::absrel(2.5e-5, 2e-4),
        };

        // NOTE: Subsampling introduces high variance in the norm and L2, so our error
        // bounds need to be looser.
        //
        // Subsampling results in poor preservation of inner products, so we skip it
        // altogether.
        let subsampled_errors = test_utils::ErrorSetup {
            norm: test_utils::Check::absrel(0.0, 2e-2),
            l2: test_utils::Check::absrel(0.0, 2e-2),
            ip: test_utils::Check::skip(),
        };

        let target_dim = |v| TargetDim::Override(NonZeroUsize::new(v).unwrap());
        let dim_combos = [
            // Natural
            (15, 15, true, TargetDim::Same, &natural_errors),
            (15, 15, true, TargetDim::Natural, &natural_errors),
            (16, 16, true, TargetDim::Same, &natural_errors),
            (16, 16, true, TargetDim::Natural, &natural_errors),
            (256, 256, true, TargetDim::Same, &natural_errors),
            (1000, 1000, true, TargetDim::Same, &natural_errors),
            // Larger
            (15, 16, true, target_dim(16), &natural_errors),
            (100, 128, true, target_dim(128), &natural_errors),
            (15, 32, true, target_dim(32), &natural_errors),
            (16, 64, true, target_dim(64), &natural_errors),
            // Sub-Sampling.
            (1024, 1023, false, target_dim(1023), &subsampled_errors),
            (1000, 999, false, target_dim(999), &subsampled_errors),
        ];

        let trials_per_combo = 20;
        let trials_per_dim = 100;

        let mut rng = StdRng::seed_from_u64(0x6d1699abe066147);
        for (input, output, preserves_norms, target, errors) in dim_combos {
            let input_nz = NonZeroUsize::new(input).unwrap();
            for trial in 0..trials_per_combo {
                let ctx = &lazy_format!(
                    "input dim = {}, output dim = {}, macro trial {} of {}",
                    input,
                    output,
                    trial,
                    trials_per_combo
                );

                let mut checker = |io: test_utils::IO<'_>, context: &dyn std::fmt::Display| {
                    let d = input.min(output);
                    assert_ne!(&io.input0[..d], &io.output0[..d]);
                    assert_ne!(&io.input1[..d], &io.output1[..d]);
                    test_utils::check_errors(io, context, errors);
                };

                // Clone the Rng state so the abstract transform behaves the same.
                let mut rng_clone = rng.clone();

                // Test the underlying transformer.
                {
                    let transformer = DoubleHadamard::new(
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
                        ctx,
                    )
                }

                // Abstract Transformer
                {
                    let kind = TransformKind::DoubleHadamard { target_dim: target };
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
                        ctx,
                    )
                }
            }
        }
    }

    #[cfg(feature = "flatbuffers")]
    mod serialization {
        use super::*;
        use crate::flatbuffers::to_flatbuffer;

        #[test]
        fn double_hadamard() {
            let mut rng = StdRng::seed_from_u64(0x123456789abcdef0);
            let alloc = GlobalAllocator;

            // Test various dimension combinations
            let test_cases = [
                // No sub or super sampling
                (5, TargetDim::Same),
                (8, TargetDim::Same),
                (10, TargetDim::Natural),
                (16, TargetDim::Natural),
                // Super sampling with both stages
                (8, TargetDim::Override(NonZeroUsize::new(12).unwrap())),
                (10, TargetDim::Override(NonZeroUsize::new(12).unwrap())),
                // Super sample with one stage
                (15, TargetDim::Override(NonZeroUsize::new(16).unwrap())),
                (16, TargetDim::Override(NonZeroUsize::new(16).unwrap())),
                (15, TargetDim::Override(NonZeroUsize::new(32).unwrap())),
                (16, TargetDim::Override(NonZeroUsize::new(32).unwrap())),
                // Sub sampling.
                (15, TargetDim::Override(NonZeroUsize::new(10).unwrap())),
                (16, TargetDim::Override(NonZeroUsize::new(10).unwrap())),
            ];

            for (dim, target_dim) in test_cases {
                let transform = DoubleHadamard::new(
                    NonZeroUsize::new(dim).unwrap(),
                    target_dim,
                    &mut rng,
                    alloc,
                )
                .unwrap();
                let data = to_flatbuffer(|buf| transform.pack(buf));

                let proto = flatbuffers::root::<fb::transforms::DoubleHadamard>(&data).unwrap();
                let reloaded = DoubleHadamard::try_unpack(alloc, proto).unwrap();

                assert_eq!(transform, reloaded);
            }

            let gen_err = |x: DoubleHadamard<_>| -> DoubleHadamardError {
                let data = to_flatbuffer(|buf| x.pack(buf));
                let proto = flatbuffers::root::<fb::transforms::DoubleHadamard>(&data).unwrap();
                DoubleHadamard::try_unpack(alloc, proto).unwrap_err()
            };

            type E = DoubleHadamardError;
            let error_cases = [
                // Signs1TooSmall: signs0.len() > signs1.len()
                (
                    vec![0, 0, 0, 0, 0], // 5 elements
                    vec![0, 0, 0, 0],    // 4 elements
                    4,
                    None,
                    E::Signs1TooSmall,
                ),
                // Signs0Empty
                (
                    vec![], // empty signs0
                    vec![0, 0, 0, 0],
                    4,
                    None,
                    E::Signs0Empty,
                ),
                // SubsampleNotMonotonic: subsample indices not in increasing order
                (
                    vec![0, 0, 0, 0],
                    vec![0, 0, 0, 0],
                    3,
                    Some(vec![0, 2, 1]), // not monotonic
                    E::SubsampleNotMonotonic,
                ),
                // SubsampleNotMonotonic: duplicate values
                (
                    vec![0, 0, 0, 0],
                    vec![0, 0, 0, 0],
                    3,
                    Some(vec![0, 1, 1]), // duplicate values
                    E::SubsampleNotMonotonic,
                ),
                // LastSubsampleTooLarge: exceeds intermediate dim with signs1
                (
                    vec![0, 0, 0], // 3 elements
                    vec![0, 0, 0], // 3 elements
                    2,
                    Some(vec![0, 3]), // index 3 >= intermediate_dim(3,2) = 3
                    E::LastSubsampleTooLarge,
                ),
                // LastSubsampleTooLarge: exceeds intermediate dim with signs1
                (
                    vec![0, 0, 0], // 3 elements
                    vec![0, 0, 0], // 3 elements
                    2,
                    Some(vec![]), // empty
                    E::InvalidSubsampleLength,
                ),
            ];

            let poly = |v: &Vec<u32>| Poly::from_iter(v.iter().copied(), alloc).unwrap();

            for (signs0, signs1, target_dim, subsample, expected) in error_cases.iter() {
                println!(
                    "on case ({:?}, {:?}, {}, {:?})",
                    signs0, signs1, target_dim, subsample,
                );
                let err = gen_err(DoubleHadamard {
                    signs0: poly(signs0),
                    signs1: poly(signs1),
                    target_dim: *target_dim,
                    subsample: subsample.as_ref().map(poly),
                });

                assert_eq!(
                    err, *expected,
                    "failed for case ({:?}, {:?}, {}, {:?})",
                    signs0, signs1, target_dim, subsample
                );
            }
        }
    }
}
