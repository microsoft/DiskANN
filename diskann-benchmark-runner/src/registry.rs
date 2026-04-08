/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::{hash_map::Entry, HashMap};

use thiserror::Error;

use crate::{
    benchmark::{self, Benchmark, Regression},
    dispatcher::{FailureScore, MatchScore},
    input, Any, Checkpoint, Input, Output,
};

/// A collection of [`crate::Input`].
pub struct Inputs {
    // Inputs keyed by their tag type.
    inputs: HashMap<&'static str, Box<dyn input::DynInput>>,
}

impl Inputs {
    /// Construct a new empty [`Inputs`] registry.
    pub fn new() -> Self {
        Self {
            inputs: HashMap::new(),
        }
    }

    /// Return the input with the registered `tag` if present. Otherwise, return `None`.
    pub fn get(&self, tag: &str) -> Option<input::Registered<'_>> {
        self.inputs.get(tag).map(|v| input::Registered(&**v))
    }

    /// Register the [`Input`] `T` in the registry.
    ///
    /// Returns an error if any other input with the same [`Input::tag()`] has been registered
    /// while leaving the underlying registry unchanged.
    pub fn register<T>(&mut self) -> anyhow::Result<()>
    where
        T: Input + 'static,
    {
        let tag = T::tag();
        match self.inputs.entry(tag) {
            Entry::Vacant(entry) => {
                entry.insert(Box::new(crate::input::Wrapper::<T>::new()));
                Ok(())
            }
            Entry::Occupied(_) => {
                #[derive(Debug, Error)]
                #[error("An input with the tag \"{}\" already exists", self.0)]
                struct AlreadyExists(&'static str);

                Err(anyhow::anyhow!(AlreadyExists(tag)))
            }
        }
    }

    /// Return an iterator over all registered input tags in an unspecified order.
    pub fn tags(&self) -> impl ExactSizeIterator<Item = &'static str> + use<'_> {
        self.inputs.keys().copied()
    }
}

impl Default for Inputs {
    fn default() -> Self {
        Self::new()
    }
}

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

/// A collection of registered benchmarks.
pub struct Benchmarks {
    benchmarks: Vec<RegisteredBenchmark>,
}

impl Benchmarks {
    /// Return a new empty registry.
    pub fn new() -> Self {
        Self {
            benchmarks: Vec::new(),
        }
    }

    /// Register a new benchmark with the given name.
    pub fn register<T>(&mut self, name: impl Into<String>)
    where
        T: Benchmark + 'static,
    {
        self.benchmarks.push(RegisteredBenchmark {
            name: name.into(),
            benchmark: Box::new(benchmark::internal::Wrapper::<T>::new()),
        });
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
    pub fn has_match(&self, job: &Any) -> bool {
        self.find_best_match(job).is_some()
    }

    /// Attempt to run the best matching benchmark for `job`.
    ///
    /// Returns the results of the benchmark if successful.
    ///
    /// Errors if a suitable method could not be found or if the invoked benchmark failed.
    pub fn call(
        &self,
        job: &Any,
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
    pub fn debug(&self, job: &Any, max_methods: usize) -> Result<(), Vec<Mismatch>> {
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
    fn find_best_match(&self, job: &Any) -> Option<&RegisteredBenchmark> {
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

    //-------------------//
    // Regression Checks //
    //-------------------//

    /// Register a regression-checkable benchmark with the associated name.
    ///
    /// Upon registration, the associated [`Regression::Tolerances`] input and the benchmark
    /// itself will be reachable via [`Check`](crate::app::Check).
    pub fn register_regression<T>(&mut self, name: impl Into<String>)
    where
        T: Regression + 'static,
    {
        let registered = benchmark::internal::Wrapper::<T, _>::new_with(
            benchmark::internal::WithRegression::<T>::new(),
        );
        self.benchmarks.push(RegisteredBenchmark {
            name: name.into(),
            benchmark: Box::new(registered),
        });
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

impl Default for Benchmarks {
    fn default() -> Self {
        Self::new()
    }
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

    pub(crate) fn try_match(&self, input: &Any) -> Result<MatchScore, FailureScore> {
        self.benchmark.benchmark().try_match(input)
    }

    pub(crate) fn check(
        &self,
        tolerance: &Any,
        input: &Any,
        before: &serde_json::Value,
        after: &serde_json::Value,
    ) -> anyhow::Result<benchmark::internal::CheckedPassFail> {
        self.regression.check(tolerance, input, before, after)
    }
}

#[derive(Debug)]
pub(crate) struct RegisteredTolerance<'a> {
    /// The tolerance parser.
    pub(crate) tolerance: input::Registered<'a>,

    /// A single tolerance input can apply to multiple benchmarks. This field records all
    /// such benchmarks that are available in the registry that use this tolerance.
    pub(crate) regressions: Vec<RegressionBenchmark<'a>>,
}

/// Helper to capture a `Benchmark::description` call into a `String` via `Display`.
struct Capture<'a>(&'a dyn benchmark::internal::Benchmark, Option<&'a Any>);

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
