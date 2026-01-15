/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Binary quantization compression and distance comparisions.
//!
//! Binary quantization works by compressing each dimension of a vector to just 1 bit,
//! where that bit is "1" is that value is positive and "0" if negative.
//!
//! The main quantizer is [`BinaryQuantizer`].
//!
//! Distances are computed using the [`crate::distances::Hamming`] distance function where
//! [`crate::bits::BitSlice::<1, crate::bits::Binary>`] is the compressed-representation
//! for binary vectors.

mod quantizer;

/////////////
// Exports //
/////////////

pub use quantizer::BinaryQuantizer;
