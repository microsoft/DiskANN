/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::hash::{Hash, Hasher};

use rand::{rngs::StdRng, Rng, RngCore, SeedableRng};

/// Creation of random number generator in potentially parallelized applications.
pub trait RngBuilder<T> {
    type Rng: Rng + 'static;

    // Construct an Rng with the provided value for mixing randomness.
    fn build_rng(&self, mixin: T) -> Self::Rng;
}

/// A `RngBuilder` that returns a `rand::rngs::StdRng` and uses the default hasher to seed
/// the `mixin` value.
pub struct StdRngBuilder {
    hasher: std::hash::DefaultHasher,
}

impl StdRngBuilder {
    /// Construct a new `StdRngBuilder` using the given seed.
    pub fn new(seed: u64) -> Self {
        let mut hasher = std::hash::DefaultHasher::new();
        seed.hash(&mut hasher);
        Self { hasher }
    }
}

impl<T> RngBuilder<T> for StdRngBuilder
where
    T: std::hash::Hash,
{
    type Rng = StdRng;

    fn build_rng(&self, mixin: T) -> Self::Rng {
        let mut hasher = self.hasher.clone();
        mixin.hash(&mut hasher);
        StdRng::seed_from_u64(hasher.finish())
    }
}

/// An object-safe version of `RngBuilder`.
pub trait BoxedRngBuilder<T> {
    fn build_boxed_rng(&self, mixin: T) -> Box<dyn RngCore>;
}

impl<T, M> BoxedRngBuilder<M> for T
where
    T: RngBuilder<M>,
{
    fn build_boxed_rng(&self, mixin: M) -> Box<dyn RngCore> {
        Box::new(self.build_rng(mixin))
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use rand::distr::{Distribution, StandardUniform};

    use super::*;

    fn test_builder<T>(builder: T, last: Option<[u32; 2]>) -> [u32; 2]
    where
        T: RngBuilder<u64>,
    {
        let standard = StandardUniform {};

        let seed_0: u64 = 0xd60f0bc189624369;
        let seed_1: u64 = 0x7478ac104ed40abb;

        // Make sure that the random number generator returned is the same when the seed
        // is the same.
        let mut rng0 = builder.build_rng(seed_0);
        let mut rng1 = builder.build_rng(seed_0);

        let v0: u32 = standard.sample(&mut rng0);
        let v1: u32 = standard.sample(&mut rng1);
        assert_eq!(v0, v1);

        // Changing the seed should change the random number generator.
        let mut rng1 = builder.build_rng(seed_1);
        let v1: u32 = standard.sample(&mut rng1);
        assert_ne!(v0, v1);

        let v = [v0, v1];
        if let Some(last) = last {
            assert_ne!(v, last);
        }
        v
    }

    #[test]
    fn test_stdrng_builder() {
        let builder = StdRngBuilder::new(0x376226f7d2d5a16b);
        let v = test_builder(builder, None);

        // If we change the seed for the builder - the returned values should be different.
        let builder = StdRngBuilder::new(0x1f197993987ed14f);
        let _ = test_builder(builder, Some(v));
    }

    // Make sure the test actually panics if the results are the same.
    #[test]
    #[should_panic]
    fn test_stdrng_builder_test_panics() {
        let builder = StdRngBuilder::new(0x49a85a468d6865e6);
        let v = test_builder(builder, None);

        let builder = StdRngBuilder::new(0x49a85a468d6865e6);
        // Panics
        let _ = test_builder(builder, Some(v));
    }
}
