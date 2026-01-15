/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

#[cfg(feature = "flatbuffers")]
use flatbuffers::{FlatBufferBuilder, WIPOffset};
#[cfg(feature = "flatbuffers")]
use thiserror::Error;

use super::utils::{check_dims, TransformFailed};
#[cfg(feature = "flatbuffers")]
use crate::flatbuffers as fb;

#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub struct NullTransform {
    // no transform -> needed for quantizers that mandatorily use a transform
    dim: usize,
}

impl NullTransform {
    pub fn new(dim: NonZeroUsize) -> Self {
        NullTransform { dim: dim.get() }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The null transform always preserves norms because it leaves data unmodified.
    pub const fn preserves_norms(&self) -> bool {
        true
    }

    pub fn transform_into(&self, dst: &mut [f32], src: &[f32]) -> Result<(), TransformFailed> {
        check_dims(dst, src, self.dim(), self.dim())?;
        dst.copy_from_slice(src);
        Ok(())
    }
}

// Serialization
#[cfg(feature = "flatbuffers")]
#[cfg_attr(docsrs, doc(cfg(feature = "flatbuffers")))]
#[derive(Debug, Clone, Copy, Error, PartialEq)]
#[non_exhaustive]
pub enum NullTransformError {
    #[error("dim cannot be zero")]
    DimCannotBeZero,
}

#[cfg(feature = "flatbuffers")]
impl NullTransform {
    /// Pack into a [`crate::flatbuffers::transforms::NullTransform`] serialized
    /// representation.
    pub(crate) fn pack<'a, A>(
        &self,
        buf: &mut FlatBufferBuilder<'a, A>,
    ) -> WIPOffset<fb::transforms::NullTransform<'a>>
    where
        A: flatbuffers::Allocator + 'a,
    {
        fb::transforms::NullTransform::create(
            buf,
            &fb::transforms::NullTransformArgs {
                dim: self.dim as u32,
            },
        )
    }

    /// Attempt to unpack from a [`crate::flatbuffers::transforms::NullTransform`]
    /// serialized representation, returning any error if encountered.
    pub(crate) fn try_unpack(
        proto: fb::transforms::NullTransform<'_>,
    ) -> Result<Self, NullTransformError> {
        let dim =
            NonZeroUsize::new(proto.dim() as usize).ok_or(NullTransformError::DimCannotBeZero)?;
        Ok(Self::new(dim))
    }
}

///////////
// Tests //
///////////

#[cfg(all(test, feature = "flatbuffers"))]
mod tests {
    use super::*;
    mod serialization {
        use super::*;
        use crate::flatbuffers::to_flatbuffer;

        #[test]
        fn null_transform() {
            for dim in [1, 2, 10, 20, 1536] {
                let transform = NullTransform::new(NonZeroUsize::new(dim).unwrap());
                assert!(transform.preserves_norms());

                let data = to_flatbuffer(|buf| transform.pack(buf));

                let proto = flatbuffers::root::<fb::transforms::NullTransform>(&data).unwrap();
                let reloaded = NullTransform::try_unpack(proto).unwrap();
                assert_eq!(transform, reloaded);
            }

            // Ensure that invalid dims are rejected.
            {
                let data = to_flatbuffer(|buf| {
                    fb::transforms::NullTransform::create(
                        buf,
                        &fb::transforms::NullTransformArgs::default(),
                    )
                });

                let proto = flatbuffers::root::<fb::transforms::NullTransform>(&data).unwrap();
                let err = NullTransform::try_unpack(proto).unwrap_err();
                assert_eq!(err, NullTransformError::DimCannotBeZero);
            }
        }
    }
}
