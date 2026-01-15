/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// An overloadable, 2-argument distance function with a parameterized return type.
///
/// Pure distance functions depend only on the values of the argument and the type of the
/// return value.
pub trait PureDistanceFunction<Left, Right, To = f32> {
    fn evaluate(x: Left, y: Right) -> To;
}

/// An overloadable, 2-argument distance function with a parameterized return type.
///
/// Unlike `PureDistanceFunction`, this takes a functor as the receiver. This allows
/// distance functors to contain auxiliary state required to do their job
/// (for example, access to some shared quantization tables).
pub trait DistanceFunction<Left, Right, To = f32> {
    /// Perform a distance computation between the left-hand and right-hand arguments.
    fn evaluate_similarity(&self, x: Left, y: Right) -> To;
}

/// A distance function where one argument is static and interned within `Self`.
///
/// The method `self.evaluate_similarity` can then be invoked for arbitrarily many values of
/// `changing`.
///
/// The main idea behind this trait is to enable distance functions where some amount of
/// preprocessing on a query can be used to accelerate distance computations.
pub trait PreprocessedDistanceFunction<Changing, To = f32> {
    fn evaluate_similarity(&self, changing: Changing) -> To;
}

/// Evaluate a norm of the argument `x` and return the result as the requested type.
///
/// Note that while this has a similar signature to `PreprocessedDistanceFunction`, the
/// semantics of this trait are different. Implementations are expected to be light-weight
/// types that implement some kind of reduction on on `x`.
pub trait Norm<T, To = f32> {
    fn evaluate(&self, x: T) -> To;
}
