/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use rand::Rng;
use thiserror::Error;

use crate::alloc::{Allocator, AllocatorError, Poly};

#[derive(Debug, Clone, Copy, Error, PartialEq)]
pub enum TransformFailed {
    #[error("incorrect transform input vector - expected length {expected} but got {found}")]
    SourceMismatch { expected: usize, found: usize },
    #[error("incorrect transform output vector - expected length {expected} but got {found}")]
    DestinationMismatch { expected: usize, found: usize },
    #[error(transparent)]
    AllocatorError(#[from] AllocatorError),
}

pub(super) fn check_dims(
    dst: &[f32],
    src: &[f32],
    input_dim: usize,
    output_dim: usize,
) -> Result<(), TransformFailed> {
    if src.len() != input_dim {
        return Err(TransformFailed::SourceMismatch {
            expected: input_dim,
            found: src.len(),
        });
    }

    if dst.len() != output_dim {
        return Err(TransformFailed::DestinationMismatch {
            expected: output_dim,
            found: dst.len(),
        });
    }
    Ok(())
}

pub(super) fn is_sign(x: u32) -> bool {
    x == 0 || x == 0x8000_0000
}

#[cfg(feature = "flatbuffers")]
pub(super) fn sign_to_bool(x: u32) -> bool {
    x == 0x8000_0000
}

#[cfg(feature = "flatbuffers")]
pub(super) fn bool_to_sign(x: bool) -> u32 {
    if x {
        0x8000_0000
    } else {
        0
    }
}

pub(super) fn subsample_indices<R, A>(
    rng: &mut R,
    length: usize,
    amount: usize,
    allocator: A,
) -> Result<Poly<[u32], A>, AllocatorError>
where
    R: Rng + ?Sized,
    A: Allocator,
{
    let mut subsample = Poly::from_iter(
        rand::seq::index::sample(rng, length, amount)
            .into_iter()
            .map(|i| i as u32),
        allocator,
    )?;
    subsample.sort();
    Ok(subsample)
}
