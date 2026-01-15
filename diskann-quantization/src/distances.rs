/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Zero-sized types representing distance functions.
//!
//! The types defined here are zero-sized types that represent logical distance function
//! operations.
//!
//! Some types like bit-vectors and [`crate::bits::BitSlice`]s can be efficiently
//! implemented as pure functions and therefore can use [`PureDistanceFunction`] semantics
//! for these types directly.
//!
//! Other quantization schemes such as scalar quantization (e.g.
//! [`crate::scalar::CompensatedSquaredL2`]) need auxiliary state and therefore need
//! customized, stateful distance function types.

use diskann_vector::{DistanceFunction, MathematicalValue, PureDistanceFunction};

pub(crate) type MV<T> = MathematicalValue<T>;

/// A marker type that indicates a distance computation failed because the arguments had
/// unequal lengths.
///
/// This struct intentionally is a zero-sized type to allow return paths to be as efficient
/// as possible.
#[derive(Debug, Default, Clone)]
pub struct UnequalLengths;

impl UnequalLengths {
    /// Escalate the unequal length error to a full-blown panic.
    #[allow(clippy::panic)]
    #[inline(never)]
    pub fn panic(self, xlen: usize, ylen: usize) -> ! {
        panic!(
            "vector lengths must be equal, instead got xlen = {}, ylen = {}",
            xlen, ylen
        );
    }
}

impl std::fmt::Display for UnequalLengths {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(fmt, "vector lengths must be equal")
    }
}

impl std::error::Error for UnequalLengths {}

/// Upcoming return type for distance functions to allow graceful failure instead of
/// panicking when incorrect dimensions are provided.
pub type Result<T> = std::result::Result<T, UnequalLengths>;

/// Upcoming return type for distance functions to allow graceful failure instead of
/// panicking when incorrect dimensions are provided.
pub type MathematicalResult<T> = std::result::Result<MathematicalValue<T>, UnequalLengths>;

/// Check that `x.len() == y.len()`, returning `Err(diskann_vector::distances::UnequalLengths)` if
/// the results are different.
///
/// If the results are the same, return the length.
#[macro_export]
macro_rules! check_lengths {
    ($x:ident, $y:ident) => {{
        let len = $x.len();
        if len != $y.len() {
            Err($crate::distances::UnequalLengths)
        } else {
            Ok(len)
        }
    }};
}

pub use check_lengths;

/// Compute the squared Euclidean distance between vector-like types.
#[derive(Debug, Clone, Copy)]
pub struct SquaredL2;

/// Compute the inner-product between vector-like types.
#[derive(Debug, Clone, Copy)]
pub struct InnerProduct;

/// Compute the hamming distance between bit-vectors.
#[derive(Debug, Clone, Copy)]
pub struct Hamming;

macro_rules! auto_distance_function {
    ($T:ty) => {
        impl<A, B, To> DistanceFunction<A, B, To> for $T
        where
            $T: PureDistanceFunction<A, B, To>,
        {
            fn evaluate_similarity(&self, a: A, b: B) -> To {
                <$T>::evaluate(a, b)
            }
        }
    };
}

auto_distance_function!(SquaredL2);
auto_distance_function!(InnerProduct);
auto_distance_function!(Hamming);

///////////
// Tests //
///////////

#[cfg(test)]
mod test {
    use super::*;

    fn test_error_impl<T>(x: T)
    where
        T: std::error::Error,
    {
        assert_eq!(x.to_string(), "vector lengths must be equal");
        assert!(x.source().is_none());
    }

    #[test]
    fn test_error() {
        test_error_impl(UnequalLengths);
    }

    fn test_check_length_impl(x: &[f32], y: &[f32]) -> Result<usize> {
        check_lengths!(x, y)
    }

    #[test]
    fn test_check_length() {
        let x = [0.0f32; 10];
        let y = [0.0f32; 10];

        for i in 0..10 {
            for j in 0..10 {
                match test_check_length_impl(&x[..i], &y[..j]) {
                    Ok(len) => {
                        assert_eq!(i, j, "Ok should only be returned when i == j");
                        assert_eq!(i, len, "`check_lengths` should return the final length");
                    }
                    Err(UnequalLengths) => {
                        assert_ne!(i, j, "An error should be returned for unequal lengths");
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "vector lengths must be equal, instead got xlen = 10, ylen = 20")]
    fn unequal_lenghts_panic() {
        (UnequalLengths).panic(10, 20)
    }
}
