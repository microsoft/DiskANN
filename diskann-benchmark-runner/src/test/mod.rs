/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::HashSet;

use crate::{registry, Features};

mod dim;
mod gated;
mod typed;

pub(crate) use typed::TypeInput;

////////////////////////
// Test Configuration //
////////////////////////

/// The set of features considered "enabled" when registering the test benchmarks.
///
/// Gated test items branch on this configuration at registration time: when their controlling
/// feature requirement is satisfied, the real benchmark (and input) is registered; otherwise a
/// gated placeholder is registered in its place. This emulates the compile-time `cfg_if!` swap a
/// downstream crate performs, but lets a single build exercise both sides of every gate.
#[derive(Debug, Default, Clone)]
pub struct TestConfig {
    features: HashSet<String>,
}

impl TestConfig {
    /// Construct a configuration with no features enabled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a configuration with the given features enabled.
    pub fn with_features<I, S>(features: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            features: features.into_iter().map(Into::into).collect(),
        }
    }

    /// Return `true` if the given feature requirement is satisfied by this configuration.
    fn enabled(&self, features: &Features) -> bool {
        features.satisfied_by(&self.features)
    }
}

//////////////////
// Registration //
//////////////////

// Full feature list:
//
// gated-feature-0
// gated-feature-1
// gated-feature-2
// gated-feature-3
pub fn register_benchmarks(
    registry: &mut registry::Registry,
    config: &TestConfig,
) -> Result<(), registry::RegistryError> {
    registry.register_regression("type-bench-f32", typed::TypeBench::<f32>::new())?;
    registry.register_regression("type-bench-i8", typed::TypeBench::<i8>::new())?;
    registry.register_regression(
        "exact-type-bench-f32-1000",
        typed::ExactTypeBench::<f32, 1000>::new(),
    )?;

    registry.register("simple-bench", dim::SimpleBench)?;
    registry.register_regression("dim-bench", dim::DimBench)?;

    // A partially-gated benchmark: its input (`TypeInput`) is always compiled, only the
    // benchmark is toggled behind `gated-feature-0`.
    let features = Features::new("gated-feature-0");
    if config.enabled(&features) {
        registry.register("gated-bench", gated::GatedBench)?;
    } else {
        registry.register_partially_gated::<TypeInput>(
            "gated-bench",
            features,
            "A gated benchmark",
        )?;
    }

    // A partially-gated benchmark behind a conjunction of features.
    let features = Features::all(["gated-feature-0", "gated-feature-1"]);
    if config.enabled(&features) {
        registry.register("another-gated-bench", gated::AnotherGatedBench)?;
    } else {
        registry.register_partially_gated::<dim::DimInput>(
            "another-gated-bench",
            features,
            "Another gated benchmark",
        )?;
    }

    // A fully-gated benchmark: both the benchmark and its `phantom-input` are toggled behind a
    // disjunction of features.
    let features = Features::any(["gated-feature-2", "gated-feature-3"]);
    if config.enabled(&features) {
        registry.register("fully-gated-bench", gated::FullyGatedBench)?;
    } else {
        registry.register_gated(
            "phantom-input",
            "fully-gated-bench",
            features,
            "A fully gated benchmark",
        )?;
    }

    // A benchmark that always registers its input, but is the only one to use it.
    let features = Features::new("gated-feature-3");
    if config.enabled(&features) {
        registry.register("something-else", gated::GatedWithIndependentInput)?;
    } else {
        // Register it twice to test pluralization in the print-out.
        registry.register_partially_gated::<gated::SampleInput>(
            "something-else",
            features.clone(),
            "A gated benchmark that always registers its input",
        )?;

        registry.register_partially_gated::<gated::SampleInput>(
            "something-else-v3",
            features,
            "A gated benchmark that always registers its input - a redundant registration.",
        )?;
    }

    Ok(())
}
