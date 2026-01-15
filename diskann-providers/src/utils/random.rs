/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use rand::{Rng, SeedableRng};

/// The default seed for tests.
/// Don't change this constant. Instead, use create_rnd_from_seed_in_tests(u64) or create_rnd_provider_from_seed_in_tests(u64)
pub const DEFAULT_SEED_FOR_TESTS: u64 = 42;

/// The random number generator used by default in DiskANN.
///
/// Users are encouraged to avoid relying too heavily on the exact type.
pub type StandardRng = rand::rngs::StdRng;

/// Creates a pseudo-random number generator from a seed.
/// This function should not be used in tests to simplify tracking references to this function in the code.
pub fn create_rnd_from_seed(seed: u64) -> StandardRng {
    rand::rngs::StdRng::seed_from_u64(seed)
}

/// Creates a pseudo-random number generator from a default seed.
/// All tests should use this function to create a random number generator to ensure reproducibility.
pub fn create_rnd_in_tests() -> StandardRng {
    create_rnd_from_seed(DEFAULT_SEED_FOR_TESTS)
}

/// Creates a pseudo-random number generator from a seed.
/// All tests should use this function to create a random number generator to ensure reproducibility.
pub fn create_rnd_from_seed_in_tests(seed: u64) -> StandardRng {
    create_rnd_from_seed(seed)
}

/// Creates a randomly seeded random number generator.
#[cfg(not(test))]
#[allow(clippy::disallowed_methods)]
pub fn create_rnd() -> StandardRng {
    rand::rngs::StdRng::from_os_rng()
}

/// Creates a pseudo-random number generator from a predefined seed to ensure reproducibility
/// of tests and benchmarks.
#[cfg(test)]
pub fn create_rnd() -> StandardRng {
    create_rnd_from_seed(DEFAULT_SEED_FOR_TESTS)
}

/// Creates a pseudo-random number generator provider from a seed.
/// This function should not be used in tests to simplify tracking references to this function in the code.
pub fn create_rnd_provider_from_seed(seed: u64) -> RandomProvider<StandardRng> {
    RandomProvider {
        seed: Some(seed),
        factory: rand::rngs::StdRng::seed_from_u64,
    }
}

/// Creates a pseudo-random number generator provider from a seed.
/// All tests should use this function to create a random number generator provider to ensure reproducibility.
pub fn create_rnd_provider_from_seed_in_tests(seed: u64) -> RandomProvider<StandardRng> {
    RandomProvider {
        seed: Some(seed),
        factory: rand::rngs::StdRng::seed_from_u64,
    }
}

/// Creates a random number generator provider.
#[cfg(not(test))]
pub fn create_rnd_provider() -> RandomProvider<StandardRng> {
    RandomProvider {
        seed: None,
        factory: |_seed| create_rnd(),
    }
}

/// Creates a pseudo-random number generator provider from a predefined seed to ensure reproducibility of tests and benchmarks.
#[cfg(test)]
pub fn create_rnd_provider() -> RandomProvider<StandardRng> {
    RandomProvider {
        seed: None,
        factory: rand::rngs::StdRng::seed_from_u64,
    }
}

/// Create a random generator based on the random seed.
pub fn create_rnd_from_optional_seed(optional_seed: Option<u64>) -> StandardRng {
    match optional_seed {
        Some(seed) => create_rnd_from_seed(seed),
        None => create_rnd(),
    }
}

/// Create a random generator provider based on the random seed.
pub fn create_rnd_provider_from_optional_seed(
    optional_seed: Option<u64>,
) -> RandomProvider<StandardRng> {
    match optional_seed {
        Some(seed) => create_rnd_provider_from_seed(seed),
        None => create_rnd_provider(),
    }
}

/// The random generator provider.
/// It is needed, when we can't pass an instance of a random generator to a function, but we need to pass a factory function to create it. For example, if the function spawns threads with a random generator assigned to each thread.
pub struct RandomProvider<T: Rng + 'static> {
    seed: Option<u64>,
    factory: fn(u64) -> T,
}

impl<T: Rng + 'static> RandomProvider<T> {
    /// Creates a new random generator.
    pub fn create_rnd(&self) -> T {
        match self.seed {
            Some(seed) => (self.factory)(seed),
            None => (self.factory)(DEFAULT_SEED_FOR_TESTS),
        }
    }

    /// Creates a new random generator with an additional seed.
    pub fn create_rnd_from_seed(&self, additional_seed: u64) -> T {
        match self.seed {
            Some(seed) => (self.factory)(seed.wrapping_add(additional_seed)),
            None => (self.factory)(additional_seed),
        }
    }
}
