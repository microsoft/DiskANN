// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Compile-time unroll reduction over fixed-size accumulator arrays.
//!
//! Shared by every micro-kernel family (f32, f16, future u8/i8, …): each
//! kernel keeps `UNROLL` independent SIMD accumulators in the inner loop and
//! folds them down to a single value at the end with a caller-supplied binary
//! operator (e.g. `max_simd`).
//!
//! Implementations are provided for `[T; 1..=4]`, matching the unroll factors
//! currently used by the kernels. The 4-element fold is balanced (`(a⊕b)⊕(c⊕d)`)
//! to shorten the dependency chain; 2- and 3-element folds are left-associative.

/// Compile-time unroll reduction over fixed-size arrays.
///
/// Used by the micro-kernels to reduce `UNROLL` accumulators into a single
/// value using a caller-supplied binary operator (e.g. `max_simd`).
pub(super) trait Reduce {
    type Element;
    fn reduce<F>(&self, f: &F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element;
}

impl<T: Copy> Reduce for [T; 1] {
    type Element = T;

    #[inline(always)]
    fn reduce<F>(&self, _f: &F) -> T
    where
        F: Fn(T, T) -> T,
    {
        self[0]
    }
}

impl<T: Copy> Reduce for [T; 2] {
    type Element = T;

    #[inline(always)]
    fn reduce<F>(&self, f: &F) -> T
    where
        F: Fn(T, T) -> T,
    {
        f(self[0], self[1])
    }
}

impl<T: Copy> Reduce for [T; 3] {
    type Element = T;

    #[inline(always)]
    fn reduce<F>(&self, f: &F) -> T
    where
        F: Fn(T, T) -> T,
    {
        f(f(self[0], self[1]), self[2])
    }
}

impl<T: Copy> Reduce for [T; 4] {
    type Element = T;

    #[inline(always)]
    fn reduce<F>(&self, f: &F) -> T
    where
        F: Fn(T, T) -> T,
    {
        f(f(self[0], self[1]), f(self[2], self[3]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_folds_correctly() {
        let max = |a: f32, b: f32| a.max(b);
        assert_eq!([5.0f32].reduce(&max), 5.0);
        assert_eq!([1.0f32, 3.0].reduce(&max), 3.0);
        assert_eq!([2.0f32, 1.0, 4.0].reduce(&max), 4.0);
        assert_eq!([3.0f32, 1.0, 4.0, 2.0].reduce(&max), 4.0);
    }

    /// Verify the exact fold order of each `Reduce` impl using a
    /// non-commutative operator (subtraction).
    ///
    /// - `[a; 1]`       → `a`
    /// - `[a, b; 2]`    → `a - b`
    /// - `[a, b, c; 3]` → `(a - b) - c`         (left fold)
    /// - `[a, b, c, d; 4]` → `(a - b) - (c - d)` (balanced tree)
    #[test]
    fn reduce_fold_order() {
        let sub = |a: f32, b: f32| a - b;
        // [10]                          → 10
        assert_eq!([10.0f32].reduce(&sub), 10.0);
        // [10, 3]                       → 10 - 3 = 7
        assert_eq!([10.0f32, 3.0].reduce(&sub), 7.0);
        // [10, 3, 1]                    → (10 - 3) - 1 = 6
        assert_eq!([10.0f32, 3.0, 1.0].reduce(&sub), 6.0);
        // [10, 3, 1, 2]                 → (10 - 3) - (1 - 2) = 7 - (-1) = 8
        assert_eq!([10.0f32, 3.0, 1.0, 2.0].reduce(&sub), 8.0);
    }
}
