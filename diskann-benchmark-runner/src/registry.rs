/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::{hash_map::Entry, HashMap};

use thiserror::Error;

use crate::{
    benchmark::{self, Benchmark, FailureScore, MatchScore, Regression},
    input, Checkpoint, Input, Output,
};

/// A registered benchmark entry: a name paired with a type-erased benchmark.
pub(crate) struct RegisteredBenchmark {
    name: String,
    benchmark: Box<dyn benchmark::internal::Benchmark>,
}

impl std::fmt::Debug for RegisteredBenchmark {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let benchmark = Capture(&*self.benchmark, None);
        f.debug_struct("RegisteredBenchmark")
            .field("name", &self.name)
            .field("benchmark", &benchmark)
            .finish()
    }
}

impl RegisteredBenchmark {
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn benchmark(&self) -> &dyn benchmark::internal::Benchmark {
        &*self.benchmark
    }
}

/// A collection of registered inputs and benchmarks.
pub struct Registry {
    // Inputs keyed by their tag type.
    inputs: HashMap<&'static str, Box<dyn input::internal::DynInput>>,
    benchmarks: Vec<RegisteredBenchmark>,
}

impl Registry {
    /// Return a new empty registry.
    pub fn new() -> Self {
        Self {
            inputs: HashMap::new(),
            benchmarks: Vec::new(),
        }
    }

    /// Return the input with the registered `tag` if present. Otherwise, return `None`.
    ///
    /// Inputs are automatically registered as a side-effect of:
    ///
    /// * [`register`](Self::register)
    /// * [`register_regression`](Self::register_regression)
    pub fn input(&self, tag: &str) -> Option<input::Registered<'_>> {
        self._input(tag).map(input::Registered)
    }

    /// Return an iterator over all registered input tags in an unspecified order.
    pub fn input_tags(&self) -> impl ExactSizeIterator<Item = &'static str> + use<'_> {
        self.inputs.keys().copied()
    }

    /// Register a new `benchmark` with the given `name`.
    ///
    /// As a side-effect, the benchmark's [`Input`](Benchmark::Input) type is also registered.
    /// Duplicate registrations of the same tag and type are allowed; mismatched types for the
    /// same tag return an error.
    pub fn register<T>(
        &mut self,
        name: impl Into<String>,
        benchmark: T,
    ) -> Result<(), RegistryError>
    where
        T: Benchmark,
    {
        self.register_input::<T::Input>()?;

        self.benchmarks.push(RegisteredBenchmark {
            name: name.into(),
            benchmark: Box::new(benchmark::internal::Wrapper::<T, _>::new(
                benchmark,
                benchmark::internal::NoRegression,
            )),
        });
        Ok(())
    }

    /// Return an iterator over registered benchmark names and their descriptions.
    pub(crate) fn names(&self) -> impl ExactSizeIterator<Item = (&str, String)> {
        self.benchmarks.iter().map(|entry| {
            (
                entry.name.as_str(),
                Capture(&*entry.benchmark, None).to_string(),
            )
        })
    }

    /// Return `true` if `job` matches with any registered benchmark. Otherwise, return `false`.
    pub(crate) fn has_match(&self, job: &input::internal::Any) -> bool {
        self.find_best_match(job).is_some()
    }

    /// Attempt to run the best matching benchmark for `job`.
    ///
    /// Returns the results of the benchmark if successful.
    ///
    /// Errors if a suitable method could not be found or if the invoked benchmark failed.
    pub(crate) fn call(
        &self,
        job: &input::internal::Any,
        checkpoint: Checkpoint<'_>,
        output: &mut dyn Output,
    ) -> anyhow::Result<serde_json::Value> {
        match self.find_best_match(job) {
            Some(entry) => entry.benchmark.run(job, checkpoint, output),
            None => Err(anyhow::Error::msg(
                "could not find a matching benchmark for the given input",
            )),
        }
    }

    /// Attempt to debug reasons for a missed dispatch, returning at most `max_methods`
    /// reasons.
    ///
    /// Returns `Ok(())` if a match was found.
    pub(crate) fn debug(
        &self,
        job: &input::internal::Any,
        max_methods: usize,
    ) -> Result<(), Vec<Mismatch>> {
        if self.has_match(job) {
            return Ok(());
        }

        // Collect all failures with their scores, sorted by score (best near-misses first).
        let mut failures: Vec<(&RegisteredBenchmark, FailureScore)> = self
            .benchmarks
            .iter()
            .filter_map(|entry| match entry.benchmark.try_match(job) {
                Ok(_) => None,
                Err(score) => Some((entry, score)),
            })
            .collect();

        failures.sort_by_key(|(_, score)| *score);
        failures.truncate(max_methods);

        let mismatches = failures
            .into_iter()
            .map(|(entry, _)| {
                let reason = Capture(&*entry.benchmark, Some(job)).to_string();

                Mismatch {
                    method: entry.name.clone(),
                    reason,
                }
            })
            .collect();

        Err(mismatches)
    }

    /// Find the best matching benchmark for `job` by score.
    fn find_best_match(&self, job: &input::internal::Any) -> Option<&RegisteredBenchmark> {
        self.benchmarks
            .iter()
            .filter_map(|entry| {
                entry
                    .benchmark
                    .try_match(job)
                    .ok()
                    .map(|score| (entry, score))
            })
            .min_by_key(|(_, score)| *score)
            .map(|(entry, _)| entry)
    }

    fn _input(&self, tag: &str) -> Option<&dyn input::internal::DynInput> {
        self.inputs.get(tag).map(|v| &**v)
    }

    fn register_input<T>(&mut self) -> Result<(), RegistryError>
    where
        T: Input + 'static,
    {
        let tag = T::tag();
        let wrapper = crate::input::internal::Wrapper::<T>::new();
        match self.inputs.entry(tag) {
            Entry::Vacant(v) => {
                v.insert(Box::new(wrapper));
                Ok(())
            }
            Entry::Occupied(o) => {
                use input::internal::DynInput;

                if o.get().as_any().is::<crate::input::internal::Wrapper<T>>() {
                    Ok(())
                } else {
                    Err(RegistryError {
                        tag,
                        existing: o.get().type_name(),
                        new: wrapper.type_name(),
                    })
                }
            }
        }
    }

    //-------------------//
    // Regression Checks //
    //-------------------//

    /// Register a regression-checkable `benchmark` with the given `name`.
    ///
    /// As a side-effect, the benchmark's [`Input`](Benchmark::Input) type is also registered.
    /// Duplicate registrations of the same tag and type are allowed; mismatched types for the
    /// same tag return an error.
    ///
    /// Upon registration, the associated [`Regression::Tolerances`] input and the benchmark
    /// itself will be reachable via [`Check`](crate::app::Check).
    pub fn register_regression<T>(
        &mut self,
        name: impl Into<String>,
        benchmark: T,
    ) -> Result<(), RegistryError>
    where
        T: Regression,
    {
        self.register_input::<T::Input>()?;

        let registered = benchmark::internal::Wrapper::<T, _>::new(
            benchmark,
            benchmark::internal::WithRegression,
        );
        self.benchmarks.push(RegisteredBenchmark {
            name: name.into(),
            benchmark: Box::new(registered),
        });

        Ok(())
    }

    /// Return a collection of all tolerance related inputs, keyed by the input tag type
    /// of the tolerance.
    pub(crate) fn tolerances(&self) -> HashMap<&'static str, RegisteredTolerance<'_>> {
        let mut tolerances = HashMap::<&'static str, RegisteredTolerance<'_>>::new();
        for b in self.benchmarks.iter() {
            if let Some(regression) = b.benchmark.as_regression() {
                // If a tolerance input already exists - then simply add this benchmark
                // to the list of benchmarks associated with the tolerance.
                //
                // Otherwise, create a new entry.
                let t = regression.tolerance();
                let packaged = RegressionBenchmark {
                    benchmark: b,
                    regression,
                };

                match tolerances.entry(t.tag()) {
                    Entry::Occupied(occupied) => occupied.into_mut().regressions.push(packaged),
                    Entry::Vacant(vacant) => {
                        vacant.insert(RegisteredTolerance {
                            tolerance: input::Registered(t),
                            regressions: vec![packaged],
                        });
                    }
                }
            }
        }

        tolerances
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

/// Error for [`Registry::register`] or [`Registry::register_regression`].
#[derive(Debug, Error)]
#[error(
    "A different input with tag \"{}\" was already registered. Existing type: \"{}\". New type: \"{}\"",
    self.tag,
    self.existing,
    self.new,
)]
pub struct RegistryError {
    tag: &'static str,
    existing: &'static str,
    new: &'static str,
}

/// Document the reason for a method matching failure.
pub struct Mismatch {
    method: String,
    reason: String,
}

impl Mismatch {
    /// Return the name of the benchmark that we failed to match.
    pub fn method(&self) -> &str {
        &self.method
    }

    /// Return the reason why this method was not a match.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

//----------//
// Internal //
//----------//

#[derive(Debug, Clone, Copy)]
pub(crate) struct RegressionBenchmark<'a> {
    benchmark: &'a RegisteredBenchmark,
    regression: &'a dyn benchmark::internal::Regression,
}

impl RegressionBenchmark<'_> {
    pub(crate) fn name(&self) -> &str {
        self.benchmark.name()
    }

    pub(crate) fn input_tag(&self) -> &'static str {
        self.regression.input_tag()
    }

    pub(crate) fn try_match(
        &self,
        input: &input::internal::Any,
    ) -> Result<MatchScore, FailureScore> {
        self.benchmark.benchmark().try_match(input)
    }

    pub(crate) fn check(
        &self,
        tolerance: &input::internal::Any,
        input: &input::internal::Any,
        before: &serde_json::Value,
        after: &serde_json::Value,
    ) -> anyhow::Result<benchmark::internal::CheckedPassFail> {
        self.regression.check(tolerance, input, before, after)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RegisteredTolerance<'a> {
    /// The tolerance parser.
    pub(crate) tolerance: input::Registered<'a>,

    /// A single tolerance input can apply to multiple benchmarks. This field records all
    /// such benchmarks that are available in the registry that use this tolerance.
    pub(crate) regressions: Vec<RegressionBenchmark<'a>>,
}

/// Helper to capture a `Benchmark::description` call into a `String` via `Display`.
struct Capture<'a>(
    &'a dyn benchmark::internal::Benchmark,
    Option<&'a input::internal::Any>,
);

impl std::fmt::Display for Capture<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.description(f, self.1)
    }
}

impl std::fmt::Debug for Capture<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.description(f, self.1)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{input, Checker};

    macro_rules! input {
        ($T:ident, $tag:literal) => {
            #[derive(Debug)]
            struct $T;

            impl Input for $T {
                type Raw = ();
                fn tag() -> &'static str {
                    $tag
                }
                fn from_raw(_raw: Self::Raw, _checker: &mut Checker) -> anyhow::Result<$T> {
                    unimplemented!("this struct is for test only");
                }
                fn serialize(&self) -> anyhow::Result<serde_json::Value> {
                    unimplemented!("this struct is for test only");
                }
                fn example() -> Self::Raw {
                    unimplemented!("this struct is for test only");
                }
            }
        };
    }

    // For the types below, `A` and `B` have distinct tags, but `A2`'s tag conflicts with `A`.
    input!(A, "type-a");
    input!(B, "type-b");
    input!(A2, "type-a");

    #[test]
    fn test_tag_conflicts() {
        let mut registry = Registry::new();
        registry.register_input::<A>().unwrap();
        registry.register_input::<B>().unwrap();

        let mut tags: Vec<_> = registry.input_tags().collect();
        tags.sort();
        assert_eq!(tags.as_slice(), ["type-a", "type-b"]);

        {
            let a = registry._input(A::tag()).unwrap();
            assert!(a.as_any().is::<input::internal::Wrapper<A>>());

            let name = a.type_name();
            assert!(name.contains("A"), "{}", name);
        }

        {
            let b = registry._input(B::tag()).unwrap();
            assert!(b.as_any().is::<input::internal::Wrapper<B>>());

            let name = b.type_name();
            assert!(name.contains("B"), "{}", name);
        }

        let err = registry.register_input::<A2>().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("A different input with tag \"type-a\" was already registered"),
            "FAILED: {}",
            msg
        );
    }
}
