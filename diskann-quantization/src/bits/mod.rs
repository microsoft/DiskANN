/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Utilities for working with packed 1-bit to 8-bit integers (inclusive).

mod length;
mod packing;
mod ptr;
mod slice;

pub mod distances;

pub use length::{Dynamic, Length, Static};
pub use ptr::{AsMutPtr, AsPtr, MutSlicePtr, SlicePtr};
pub use slice::{
    Binary, BitSlice, BitSliceBase, BitTranspose, BoxedBitSlice, ConstructionError, Dense,
    EncodingError, GetError, IndexOutOfBounds, MutBitSlice, PermutationStrategy, Representation,
    SetError, Unsigned,
};
