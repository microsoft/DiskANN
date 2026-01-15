/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// The DiskANN library uses `f32` values to represent similarity scores, but distance
/// functions can return either similarity-scores (those that are potentially transformed
/// so that minimization yields higher similarity) and mathematical values (those that
/// are computed from the mathematical definition of the operation.
///
/// Since those are currently mixed in the library with the more common use being a
/// similarity-score, make the `MathematicalValue` the type to represent no transformation
/// was performed.
#[derive(Debug, Clone, Copy, PartialOrd, PartialEq)]
pub struct MathematicalValue<T>(T)
where
    T: Copy;

impl<T: Copy> MathematicalValue<T> {
    pub fn new(value: T) -> Self {
        Self(value)
    }

    #[inline(always)]
    pub fn into_inner(self) -> T {
        self.0
    }
}

/// The DiskANN library uses `f32` values to represent similarity scores, but distance
/// functions can return either similarity-scores (those that are potentially transformed
/// so that minimization yields higher similarity) and mathematical values (those that
/// are computed from the mathematical definition of the operation.
///
/// Since those are currently mixed in the library with the more common use being a
/// similarity-score, make the `MathematicalValue` the type to represent no transformation
/// was performed.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SimilarityScore<T>(T)
where
    T: Copy;

impl<T: Copy> SimilarityScore<T> {
    pub fn new(value: T) -> Self {
        Self(value)
    }

    #[inline(always)]
    pub fn into_inner(self) -> T {
        self.0
    }
}
