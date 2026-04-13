/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_vector::contains::ContainsSimd;
use num_traits::FromPrimitive;

use crate::utils::{IntoUsize, TryIntoInteger, TypeStr, VectorIdTryFrom};

/// The data type used to assign an identity to vectors. Canonically, this is either a
/// `u32` or a `u64`. Since this type is used to stored neighbor information in graphs,
/// using `u32` can reduce the memory footprint significantly and should be preferred if
/// possible. However, DiskANN supports both types.
pub trait VectorId:
    std::cmp::Eq
    + std::cmp::PartialEq
    + std::cmp::Ord
    + std::hash::Hash
    + TypeStr
    + VectorIdTryFrom<u32>
    + VectorIdTryFrom<u64>
    + VectorIdTryFrom<usize>
    + TryIntoInteger<u32>
    + Into<u64>
    + IntoUsize
    + FromPrimitive
    + Copy
    + Sized
    + Send
    + Sync
    + std::fmt::Debug
    + std::fmt::Display
    + Default
    + ContainsSimd
    + bytemuck::Pod
    + 'static
{
}

impl<T> VectorId for T where
    T: std::cmp::Eq
        + std::cmp::PartialEq
        + std::cmp::Ord
        + std::hash::Hash
        + TypeStr
        + Into<u64>
        + IntoUsize
        + VectorIdTryFrom<u32>
        + VectorIdTryFrom<u64>
        + VectorIdTryFrom<usize>
        + TryIntoInteger<u32>
        + FromPrimitive
        + Copy
        + Sized
        + Send
        + Sync
        + std::fmt::Debug
        + std::fmt::Display
        + Default
        + ContainsSimd
        + bytemuck::Pod
        + 'static
{
}
