/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

#[cfg(feature = "flatbuffers")]
use flatbuffers::{FlatBufferBuilder, WIPOffset};
use diskann_linalg::{self, Transpose};
use rand::Rng;
#[cfg(feature = "flatbuffers")]
use thiserror::Error;

use super::{
    utils::{check_dims, TransformFailed},
    TargetDim,
};
#[cfg(feature = "flatbuffers")]
use crate::flatbuffers as fb;

//////////////////////
// Dense Transforms //
//////////////////////

/// A distance-perserving transformation from `N`-dimensions to `N`-dimensions.
///
/// This struct materializes a full `NxN` transformation matrix and mechanically applies
/// transformations via matrix multiplication.
#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub struct RandomRotation {
    /// This data structure maintains the invariant that this **must** be a square matrix.
    transform: diskann_utils::views::Matrix<f32>,
}

impl RandomRotation {
    /// Construct a new `RandomRotation` that transforms input vectors of dimension `dim`.
    ///
    /// The parameter `rng` is used to randomly sample the transformation matrix.
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
    /// Sub-sampling occurs if `self.output_dim()` is less than `self.input_dim()`.
    pub fn new<R>(dim: NonZeroUsize, target_dim: TargetDim, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
    {
        let dim = dim.get();

        // There are three cases we need to consider:
        //
        // 1. If the target dim is the **same** as `dim`, then our transformation matrix can
        //    be square.
        //
        // 2. If the target dim is **less** than `dim`, then we generate a random `dim x dim`
        //    matrix and remove the output rows that would not be included in the output,
        //    resulting in a `target_dim x dim` matrix.
        //
        // 3. If the target dim is **greater** than `dim`, then we generate a
        //    `target_dim x target_dim` random matrix and remove columns to end up with a
        //    `target_dim x dim` matrix.
        //
        //    Removing columns is equivalent to zero padding the original vector up to
        //    `target_dim` and multiplying by the full matrix.
        let (target_dim, matrix_dim) = match target_dim {
            TargetDim::Same | TargetDim::Natural => (dim, dim),
            TargetDim::Override(target) => {
                let target_dim = target.get();
                if target_dim <= dim {
                    (target_dim, dim)
                } else {
                    (target_dim, target_dim)
                }
            }
        };

        // Lint: By construction, the matrix returned from
        // `diskann_linalg::random_distance_preserving_matrix` will by `matrix_dim x matrix_dim`.
        #[allow(clippy::unwrap_used)]
        let initial = diskann_utils::views::Matrix::try_from(
            diskann_linalg::random_distance_preserving_matrix(matrix_dim, rng).into(),
            matrix_dim,
            matrix_dim,
        )
        .unwrap();

        // Restructure the matrix as needed to apply the desired sub/super sampling.
        let transform = match target_dim.cmp(&dim) {
            std::cmp::Ordering::Equal => initial,
            std::cmp::Ordering::Less => {
                let indices = rand::seq::index::sample(rng, dim, target_dim);
                let scaling = (dim as f32 / target_dim as f32).sqrt();

                let mut transform = diskann_utils::views::Matrix::new(0.0f32, target_dim, dim);
                std::iter::zip(transform.row_iter_mut(), indices.iter()).for_each(|(ro, ri)| {
                    std::iter::zip(ro.iter_mut(), initial.row(ri).iter()).for_each(|(o, i)| {
                        *o = scaling * (*i);
                    })
                });
                transform
            }
            std::cmp::Ordering::Greater => {
                let mut transform = diskann_utils::views::Matrix::new(0.0f32, target_dim, dim);
                std::iter::zip(transform.row_iter_mut(), initial.row_iter())
                    .for_each(|(o, i)| o.copy_from_slice(&i[..dim]));
                transform
            }
        };

        Self { transform }
    }

    /// Return the input dimension for the transformation.
    pub fn input_dim(&self) -> usize {
        self.transform.ncols()
    }

    /// Return the output dimension for the transformation.
    pub fn output_dim(&self) -> usize {
        self.transform.nrows()
    }

    /// Return whether or not the transform preserves norms.
    ///
    /// For this transform, norms are not preserved when the output dimensionality is less
    /// than the input dimensionality.
    pub fn preserves_norms(&self) -> bool {
        self.output_dim() >= self.input_dim()
    }

    /// Perform the transformation of the `src` vector into the `dst` vector.
    ///
    /// # Errors
    ///
    /// Returns an error if
    ///
    /// * `src.len() != self.input_dim()`.
    /// * `dst.len() != self.output_dim()`.
    pub fn transform_into(&self, dst: &mut [f32], src: &[f32]) -> Result<(), TransformFailed> {
        let input_dim = self.input_dim();
        let output_dim = self.output_dim();
        check_dims(dst, src, input_dim, output_dim)?;
        diskann_linalg::sgemm(
            Transpose::None,
            Transpose::None,
            output_dim,
            1,
            input_dim,
            1.0,
            self.transform.as_slice(),
            src,
            None,
            dst,
        );
        Ok(())
    }
}

#[cfg(feature = "flatbuffers")]
#[cfg_attr(docsrs, doc(cfg(feature = "flatbuffers")))]
#[derive(Debug, Clone, Copy, Error, PartialEq)]
#[non_exhaustive]
pub enum RandomRotationError {
    #[error("buffer size not product of rows and columns")]
    IncorrectDim,
    #[error("number of rows cannot be zero")]
    RowsZero,
    #[error("number of cols cannot be zero")]
    ColsZero,
}

// Serialization
#[cfg(feature = "flatbuffers")]
impl RandomRotation {
    /// Pack into a [`crate::flatbuffers::transforms::RandomRotation`] serialized
    /// representation.
    pub(crate) fn pack<'a, A>(
        &self,
        buf: &mut FlatBufferBuilder<'a, A>,
    ) -> WIPOffset<fb::transforms::RandomRotation<'a>>
    where
        A: flatbuffers::Allocator + 'a,
    {
        let data = buf.create_vector(self.transform.as_slice());

        fb::transforms::RandomRotation::create(
            buf,
            &fb::transforms::RandomRotationArgs {
                data: Some(data),
                nrows: self.transform.nrows() as u32,
                ncols: self.transform.ncols() as u32,
            },
        )
    }

    /// Attempt to unpack from a [`crate::flatbuffers::transforms::RandomRotation`]
    /// serialized representation, returning any error if encountered.
    pub(crate) fn try_unpack(
        proto: fb::transforms::RandomRotation<'_>,
    ) -> Result<Self, RandomRotationError> {
        let nrows = proto.nrows();
        let ncols = proto.ncols();
        if nrows == 0 {
            return Err(RandomRotationError::RowsZero);
        }
        if ncols == 0 {
            return Err(RandomRotationError::ColsZero);
        }

        let data = proto.data().into_iter().collect();
        let transform =
            diskann_utils::views::Matrix::try_from(data, nrows as usize, ncols as usize)
                .map_err(|_| RandomRotationError::IncorrectDim)?;

        Ok(Self { transform })
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
        algorithms::transforms::{test_utils, Transform, TransformFailed, TransformKind},
        alloc::GlobalAllocator,
    };

    impl test_utils::Transformer for RandomRotation {
        fn input_dim_(&self) -> usize {
            self.input_dim()
        }
        fn output_dim_(&self) -> usize {
            self.output_dim()
        }
        fn transform_into_(&self, dst: &mut [f32], src: &[f32]) -> Result<(), TransformFailed> {
            self.transform_into(dst, src)
        }
    }

    #[test]
    fn test_transform_matrix() {
        let nonsubsampled_errors = test_utils::ErrorSetup {
            norm: test_utils::Check::ulp(10),
            l2: test_utils::Check::ulp(10),
            ip: test_utils::Check::absrel(2e-5, 1e-4),
        };

        // Because we're using relatively low dimensions, subsampling yields pretty large
        // variances. We can't use higher dimensionality, though, because then the tests
        // would never complete.
        let subsampled_errors = test_utils::ErrorSetup {
            norm: test_utils::Check::absrel(0.0, 0.18),
            l2: test_utils::Check::absrel(0.0, 0.18),
            ip: test_utils::Check::skip(),
        };

        let target_dim = |v| TargetDim::Override(NonZeroUsize::new(v).unwrap());

        // Combinations of input to output dimensions.
        let dim_combos = [
            // Same dimension
            (15, 15, true, TargetDim::Same, &nonsubsampled_errors),
            (15, 15, true, TargetDim::Natural, &nonsubsampled_errors),
            (16, 16, true, TargetDim::Same, &nonsubsampled_errors),
            (100, 100, true, TargetDim::Same, &nonsubsampled_errors),
            (100, 100, true, TargetDim::Natural, &nonsubsampled_errors),
            (256, 256, true, TargetDim::Same, &nonsubsampled_errors),
            // Super Sampling
            (15, 20, true, target_dim(20), &nonsubsampled_errors),
            // Sub Sampling
            (256, 200, false, target_dim(200), &subsampled_errors),
        ];

        let trials_per_combo = 20;
        let trials_per_dim = 50;

        let mut rng = StdRng::seed_from_u64(0x30e37c10c36cc64b);
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
                    test_utils::check_errors(io, context, errors);
                };

                // Clone the Rng state so the abstract transform behaves the same.
                let mut rng_clone = rng.clone();

                // Test the underlying transformer.
                {
                    let transformer =
                        RandomRotation::new(NonZeroUsize::new(input).unwrap(), target, &mut rng);
                    assert_eq!(transformer.input_dim(), input, "{}", ctx);
                    assert_eq!(transformer.output_dim(), output, "{}", ctx);
                    assert_eq!(transformer.preserves_norms(), preserves_norms, "{}", ctx);

                    test_utils::test_transform(
                        &transformer,
                        trials_per_dim,
                        &mut checker,
                        &mut rng,
                        ctx,
                    );
                }

                // Abstract Transformer
                {
                    let kind = TransformKind::RandomRotation { target_dim: target };
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
        fn random_rotation() {
            let mut rng = StdRng::seed_from_u64(0x123456789abcdef0);

            // Test various dimension combinations
            let test_cases = [
                (5, TargetDim::Same),
                (10, TargetDim::Natural),
                (8, TargetDim::Override(NonZeroUsize::new(12).unwrap())),
                (15, TargetDim::Override(NonZeroUsize::new(10).unwrap())),
            ];

            for (dim, target_dim) in test_cases {
                let transform =
                    RandomRotation::new(NonZeroUsize::new(dim).unwrap(), target_dim, &mut rng);
                let data = to_flatbuffer(|buf| transform.pack(buf));

                let proto = flatbuffers::root::<fb::transforms::RandomRotation>(&data).unwrap();
                let reloaded = RandomRotation::try_unpack(proto).unwrap();

                assert_eq!(transform, reloaded);
            }

            // Test error cases for invalid dimensions
            {
                let data = to_flatbuffer(|buf| {
                    let data = buf.create_vector::<f32>(&[1.0, 0.0, 0.0, 1.0]); // 2x2 matrix
                    fb::transforms::RandomRotation::create(
                        buf,
                        &fb::transforms::RandomRotationArgs {
                            data: Some(data),
                            nrows: 0, // Invalid: zero rows
                            ncols: 2,
                        },
                    )
                });

                let proto = flatbuffers::root::<fb::transforms::RandomRotation>(&data).unwrap();
                let err = RandomRotation::try_unpack(proto).unwrap_err();
                assert_eq!(err, RandomRotationError::RowsZero);
            }

            {
                let data = to_flatbuffer(|buf| {
                    let data = buf.create_vector::<f32>(&[1.0, 0.0, 0.0, 1.0]);
                    fb::transforms::RandomRotation::create(
                        buf,
                        &fb::transforms::RandomRotationArgs {
                            data: Some(data), // 2x2 matrix
                            nrows: 2,
                            ncols: 0, // Invalid: zero cols
                        },
                    )
                });

                let proto = flatbuffers::root::<fb::transforms::RandomRotation>(&data).unwrap();
                let err = RandomRotation::try_unpack(proto).unwrap_err();
                assert_eq!(err, RandomRotationError::ColsZero);
            }

            {
                let data = to_flatbuffer(|buf| {
                    let data = buf.create_vector::<f32>(&[1.0, 0.0, 0.0]); // 3 elements
                    fb::transforms::RandomRotation::create(
                        buf,
                        &fb::transforms::RandomRotationArgs {
                            data: Some(data),
                            nrows: 2,
                            ncols: 2, // Should be 4 elements for 2x2 matrix
                        },
                    )
                });

                let proto = flatbuffers::root::<fb::transforms::RandomRotation>(&data).unwrap();
                let err = RandomRotation::try_unpack(proto).unwrap_err();
                assert_eq!(err, RandomRotationError::IncorrectDim);
            }
        }
    }
}
