/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_vector::{DistanceFunction, PureDistanceFunction};

/// A helper struct that prevents a `Fn` or `PureDistanceFunction` from getting inlined
/// into a call-site.
///
/// Helpful for removing the effects of inlining.
pub struct InlineBarrier<F> {
    _f: F,
}

impl<F> InlineBarrier<F> {
    pub fn new(f: F) -> Self {
        Self { _f: f }
    }
}

impl<F, A, B, To> DistanceFunction<A, B, To> for InlineBarrier<F>
where
    F: PureDistanceFunction<A, B, To>,
{
    #[inline(never)]
    fn evaluate_similarity(&self, a: A, b: B) -> To {
        F::evaluate(a, b)
    }
}
