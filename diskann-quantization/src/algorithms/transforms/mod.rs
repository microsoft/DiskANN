/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// imports
use std::num::NonZeroUsize;

#[cfg(feature = "flatbuffers")]
use flatbuffers::{FlatBufferBuilder, WIPOffset};
use rand::RngCore;
use thiserror::Error;

use crate::alloc::{Allocator, AllocatorError, ScopedAllocator, TryClone};
#[cfg(feature = "flatbuffers")]
use crate::flatbuffers as fb;

// modules
mod double_hadamard;
mod null;
mod padding_hadamard;

crate::utils::features! {
    #![feature = "linalg"]
    mod random_rotation;
}

mod utils;

#[cfg(test)]
#[cfg(not(miri))]
mod test_utils;

// reexports
pub use double_hadamard::{DoubleHadamard, DoubleHadamardError};
pub use null::NullTransform;
pub use padding_hadamard::{PaddingHadamard, PaddingHadamardError};
pub use utils::TransformFailed;

crate::utils::features! {
    #![feature = "linalg"]
    pub use random_rotation::RandomRotation;
}

crate::utils::features! {
    #![all(feature = "linalg", feature = "flatbuffers")]
    pub use random_rotation::RandomRotationError;
}

crate::utils::features! {
    #![feature = "flatbuffers"]
    pub use null::NullTransformError;
}

///////////////
// Transform //
///////////////

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum TransformKind {
    /// Use a Hadamard transform
    /// ```math
    /// HDx / sqrt(n)
    /// ```
    /// where
    ///
    /// * `H` is an (implicit) [Hadamard matrix](https://en.wikipedia.org/wiki/Hadamard_matrix)
    /// * `D` is a diagonal matrix with `+/-1` on the diagonal.
    /// * `x` is the input vector.
    /// * `n` is the number of rows in `x`.
    ///
    /// Unlike [`Self::RandomRotation`], this method does not require matrix-vector
    /// multiplication and is therefore much faster for high-dimensional vectors.
    ///
    /// The Hadamard multiplication requires dimensions to be a power of two. Internally,
    /// this method will pad `x` with zeros up to the next power of two and transform the
    /// result.
    PaddingHadamard { target_dim: TargetDim },

    /// Use a Double Hadamard transform, which applies two Hadamard transformations
    /// in sequence; first to the head of the vector and then to the tail.
    ///
    /// This approach does not have any requirement on the input dimension to apply
    /// the distance preserving transformation using Hadamard multiplication,
    /// unlike [`PaddingHadamard`].
    ///
    /// Empirically, this approach seems to give better recall performance than
    /// applying [`PaddingHadamard`] and sampling down when `self.output_dim() == self.dim()`
    /// and the dimension is not a power of two.
    ///
    /// See [`DoubleHadamard`] for the implementation details.
    DoubleHadamard { target_dim: TargetDim },

    /// A naive transform that copies source into destination.
    Null,

    /// Use a full-dimensional, randomly sampled orthogonal matrix to transform vectors.
    ///
    /// Transformation involves matrix multiplication and may be slow for high-dimensional
    /// vectors.
    #[cfg(feature = "linalg")]
    #[cfg_attr(docsrs, doc(cfg(feature = "linalg")))]
    RandomRotation { target_dim: TargetDim },
}

#[derive(Debug, Clone, Error)]
pub enum NewTransformError {
    #[error("random number generator is required for {0:?}")]
    RngMissing(TransformKind),
    #[error(transparent)]
    AllocatorError(#[from] AllocatorError),
}

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Transform<A>
where
    A: Allocator,
{
    PaddingHadamard(PaddingHadamard<A>),
    DoubleHadamard(DoubleHadamard<A>),
    Null(NullTransform),

    #[cfg(feature = "linalg")]
    #[cfg_attr(docsrs, doc(cfg(feature = "linalg")))]
    RandomRotation(RandomRotation),
}

impl<A> Transform<A>
where
    A: Allocator,
{
    /// Construct a new `Transform` from a `TransformKind`, input dimension
    /// and an optional rng (if needed).
    ///
    /// Currently, `rng` should be supplied for the following transforms:
    /// - [`RandomRotation`]
    /// - [`PaddingHadamard`]
    /// - [`DoubleHadamard`]
    ///
    /// The [`NullTransform`] can be initialized without `rng`.
    pub fn new(
        transform_kind: TransformKind,
        dim: NonZeroUsize,
        rng: Option<&mut dyn RngCore>,
        allocator: A,
    ) -> Result<Self, NewTransformError> {
        match transform_kind {
            TransformKind::PaddingHadamard { target_dim } => {
                let rng = rng.ok_or(NewTransformError::RngMissing(transform_kind))?;
                Ok(Transform::PaddingHadamard(PaddingHadamard::new(
                    dim, target_dim, rng, allocator,
                )?))
            }
            TransformKind::DoubleHadamard { target_dim } => {
                let rng = rng.ok_or(NewTransformError::RngMissing(transform_kind))?;
                Ok(Transform::DoubleHadamard(DoubleHadamard::new(
                    dim, target_dim, rng, allocator,
                )?))
            }
            TransformKind::Null => Ok(Transform::Null(NullTransform::new(dim))),
            #[cfg(feature = "linalg")]
            TransformKind::RandomRotation { target_dim } => {
                let rng = rng.ok_or(NewTransformError::RngMissing(transform_kind))?;
                Ok(Transform::RandomRotation(RandomRotation::new(
                    dim, target_dim, rng,
                )))
            }
        }
    }

    pub(crate) fn input_dim(&self) -> usize {
        match self {
            Self::PaddingHadamard(t) => t.input_dim(),
            Self::DoubleHadamard(t) => t.input_dim(),
            Self::Null(t) => t.dim(),
            #[cfg(feature = "linalg")]
            Self::RandomRotation(t) => t.input_dim(),
        }
    }
    pub(crate) fn output_dim(&self) -> usize {
        match self {
            Self::PaddingHadamard(t) => t.output_dim(),
            Self::DoubleHadamard(t) => t.output_dim(),
            Self::Null(t) => t.dim(),
            #[cfg(feature = "linalg")]
            Self::RandomRotation(t) => t.output_dim(),
        }
    }

    pub(crate) fn preserves_norms(&self) -> bool {
        match self {
            Self::PaddingHadamard(t) => t.preserves_norms(),
            Self::DoubleHadamard(t) => t.preserves_norms(),
            Self::Null(t) => t.preserves_norms(),
            #[cfg(feature = "linalg")]
            Self::RandomRotation(t) => t.preserves_norms(),
        }
    }

    pub(crate) fn transform_into(
        &self,
        dst: &mut [f32],
        src: &[f32],
        allocator: ScopedAllocator<'_>,
    ) -> Result<(), TransformFailed> {
        match self {
            Self::PaddingHadamard(t) => t.transform_into(dst, src, allocator),
            Self::DoubleHadamard(t) => t.transform_into(dst, src, allocator),
            Self::Null(t) => t.transform_into(dst, src),
            #[cfg(feature = "linalg")]
            Self::RandomRotation(t) => t.transform_into(dst, src),
        }
    }
}

impl<A> TryClone for Transform<A>
where
    A: Allocator,
{
    fn try_clone(&self) -> Result<Self, AllocatorError> {
        match self {
            Self::PaddingHadamard(t) => Ok(Self::PaddingHadamard(t.try_clone()?)),
            Self::DoubleHadamard(t) => Ok(Self::DoubleHadamard(t.try_clone()?)),
            Self::Null(t) => Ok(Self::Null(t.clone())),
            #[cfg(feature = "linalg")]
            Self::RandomRotation(t) => Ok(Self::RandomRotation(t.clone())),
        }
    }
}

#[cfg(feature = "flatbuffers")]
#[cfg_attr(docsrs, doc(cfg(feature = "flatbuffers")))]
#[derive(Debug, Clone, Copy, Error, PartialEq)]
#[non_exhaustive]
pub enum TransformError {
    #[error(transparent)]
    PaddingHadamardError(#[from] PaddingHadamardError),
    #[error(transparent)]
    DoubleHadamardError(#[from] DoubleHadamardError),
    #[error(transparent)]
    NullTransformError(#[from] NullTransformError),
    #[cfg(feature = "linalg")]
    #[cfg_attr(docsrs, doc(cfg(feature = "linalg")))]
    #[error(transparent)]
    RandomRotationError(#[from] RandomRotationError),
    #[error("invalid transform kind")]
    InvalidTransformKind,
}

#[cfg(feature = "flatbuffers")]
impl<A> Transform<A>
where
    A: Allocator,
{
    /// Pack into a [`crate::flatbuffers::transforms::Transform`] serialized representation.
    pub(crate) fn pack<'a, FA>(
        &self,
        buf: &mut FlatBufferBuilder<'a, FA>,
    ) -> WIPOffset<fb::transforms::Transform<'a>>
    where
        FA: flatbuffers::Allocator + 'a,
    {
        let (kind, offset) = match self {
            Self::PaddingHadamard(t) => (
                fb::transforms::TransformKind::PaddingHadamard,
                t.pack(buf).as_union_value(),
            ),
            Self::DoubleHadamard(t) => (
                fb::transforms::TransformKind::DoubleHadamard,
                t.pack(buf).as_union_value(),
            ),
            Self::Null(t) => (
                fb::transforms::TransformKind::NullTransform,
                t.pack(buf).as_union_value(),
            ),
            #[cfg(feature = "linalg")]
            Self::RandomRotation(t) => (
                fb::transforms::TransformKind::RandomRotation,
                t.pack(buf).as_union_value(),
            ),
        };

        fb::transforms::Transform::create(
            buf,
            &fb::transforms::TransformArgs {
                transform_type: kind,
                transform: Some(offset),
            },
        )
    }

    /// Attempt to unpack from a [`crate::flatbuffers::transforms::Transform`] serialized
    /// representation, returning any error if encountered.
    pub(crate) fn try_unpack(
        alloc: A,
        proto: fb::transforms::Transform<'_>,
    ) -> Result<Self, TransformError> {
        if let Some(transform) = proto.transform_as_padding_hadamard() {
            return Ok(Self::PaddingHadamard(PaddingHadamard::try_unpack(
                alloc, transform,
            )?));
        }

        #[cfg(feature = "linalg")]
        if let Some(transform) = proto.transform_as_random_rotation() {
            return Ok(Self::RandomRotation(RandomRotation::try_unpack(transform)?));
        }

        if let Some(transform) = proto.transform_as_double_hadamard() {
            return Ok(Self::DoubleHadamard(DoubleHadamard::try_unpack(
                alloc, transform,
            )?));
        }

        if let Some(transform) = proto.transform_as_null_transform() {
            return Ok(Self::Null(NullTransform::try_unpack(transform)?));
        }

        Err(TransformError::InvalidTransformKind)
    }
}

/// Transformations possess the ability to keep dimensionality the same, increase it, or
/// decrease it.
///
/// This struct enables the caller to communicate the desired behavior upon transform
/// construction.
#[derive(Debug, Clone, Copy)]
pub enum TargetDim {
    /// Keep the output dimensionality the same as the input dimensionality.
    ///
    /// # Note
    ///
    /// When the input dimensionality is less than the "natural" dimensionality (
    /// see [`Self::Natural`], post-transformed sampling may be invoked where only a subset
    /// of the transformed vector's dimensions are retained.
    ///
    /// For low dimensional embeddings, this sampling may result in high norm variance and
    /// poor recall.
    Same,

    /// Use the "natural" dimensionality for the output.
    ///
    /// This allows transformations like [`PaddingHadamard`] to increase the dimensionality
    /// to the next power of two if needed. This will usually provide better accuracy than
    /// [`Self::Same`] but may result in a worse compression ratio.
    Natural,

    /// Set a hard value for the output dimensionality.
    ///
    /// This may result in arbitrary subsampling (see the note in [`TargetDim::Same`] or
    /// supersampling (zero padding the pretransformed vector). Use with care.
    Override(NonZeroUsize),
}

#[cfg(test)]
#[cfg(not(miri))]
test_utils::delegate_transformer!(Transform<crate::alloc::GlobalAllocator>);
