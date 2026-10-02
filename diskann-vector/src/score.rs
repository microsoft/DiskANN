/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Typed results of low-level distance kernels.
//!
//! The `i8` squared L2 and inner product kernels compute their results in `i32`.
//! Instead of converting to `f32` inside the kernel, they return a score type that records
//! *which* quantity was computed and leaves the choice of output type to the caller. A
//! single kernel can then serve both exact integer consumers and `f32` consumers.

use crate::{MathematicalValue, SimilarityScore};

/// The result of a squared L2 kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SquaredL2Score<T>(T);

impl<T> SquaredL2Score<T> {
    #[inline(always)]
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

/// The result of an inner product kernel: the mathematical (un-negated) inner product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InnerProductScore<T>(T);

impl<T> InnerProductScore<T> {
    #[inline(always)]
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

/// Views of a kernel result as the type `T`.
///
/// There is intentionally no conversion from a floating-point score to an integer.
///
/// Integer results are exact for `i8` inputs up to dimension 33,025 (squared L2) and 131,071
/// (inner product). Beyond that, the `i32` accumulators may overflow. The `f32` views of
/// integer results round once, at the end, and are exact up to a magnitude of 2^24.
pub trait Score<T: Copy>: Copy {
    /// The value as defined mathematically.
    fn mathematical_value(self) -> MathematicalValue<T>;

    /// The value transformed so that smaller means more similar.
    fn similarity_score(self) -> SimilarityScore<T>;
}

impl Score<i32> for SquaredL2Score<i32> {
    #[inline(always)]
    fn mathematical_value(self) -> MathematicalValue<i32> {
        MathematicalValue::new(self.0)
    }
    #[inline(always)]
    fn similarity_score(self) -> SimilarityScore<i32> {
        SimilarityScore::new(self.0)
    }
}

impl Score<f32> for SquaredL2Score<i32> {
    #[inline(always)]
    fn mathematical_value(self) -> MathematicalValue<f32> {
        MathematicalValue::new(self.0 as f32)
    }
    #[inline(always)]
    fn similarity_score(self) -> SimilarityScore<f32> {
        SimilarityScore::new(self.0 as f32)
    }
}

impl Score<i32> for InnerProductScore<i32> {
    #[inline(always)]
    fn mathematical_value(self) -> MathematicalValue<i32> {
        MathematicalValue::new(self.0)
    }
    #[inline(always)]
    fn similarity_score(self) -> SimilarityScore<i32> {
        // Only `i32::MIN` wraps, and the `i8` kernels return it only outside the exact range.
        SimilarityScore::new(self.0.wrapping_neg())
    }
}

impl Score<f32> for InnerProductScore<i32> {
    #[inline(always)]
    fn mathematical_value(self) -> MathematicalValue<f32> {
        MathematicalValue::new(self.0 as f32)
    }
    #[inline(always)]
    fn similarity_score(self) -> SimilarityScore<f32> {
        SimilarityScore::new(-(self.0 as f32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squared_l2_views() {
        let score = SquaredL2Score::new(7);
        assert_eq!(Score::<i32>::mathematical_value(score).into_inner(), 7);
        assert_eq!(Score::<i32>::similarity_score(score).into_inner(), 7);
        assert_eq!(Score::<f32>::mathematical_value(score).into_inner(), 7.0);
        assert_eq!(Score::<f32>::similarity_score(score).into_inner(), 7.0);
    }

    #[test]
    fn inner_product_views() {
        let score = InnerProductScore::new(7);
        assert_eq!(Score::<i32>::mathematical_value(score).into_inner(), 7);
        assert_eq!(Score::<i32>::similarity_score(score).into_inner(), -7);
        assert_eq!(Score::<f32>::mathematical_value(score).into_inner(), 7.0);
        assert_eq!(Score::<f32>::similarity_score(score).into_inner(), -7.0);

        // The `f32` similarity negates after converting, matching the previous `f32`
        // similarity score bit-for-bit (including `-0.0`).
        let zero = Score::<f32>::similarity_score(InnerProductScore::new(0)).into_inner();
        assert_eq!(zero.to_bits(), (-0.0f32).to_bits());

        // Negation wraps instead of panicking.
        let min = Score::<i32>::similarity_score(InnerProductScore::new(i32::MIN));
        assert_eq!(min.into_inner(), i32::MIN);
    }
}
