/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::HashMap;

use thiserror::Error;

use crate::{
    benchmark::{self, Benchmark, DynBenchmark},
    dispatcher::FailureScore,
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
        use std::collections::hash_map::Entry;

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
struct RegisteredBenchmark {
    name: String,
    benchmark: Box<dyn DynBenchmark>,
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
            benchmark: Box::new(benchmark::Wrapper::<T>::new()),
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

/// Helper to capture a `DynBenchmark::description` call into a `String` via `Display`.
struct Capture<'a>(&'a dyn DynBenchmark, Option<&'a Any>);

impl std::fmt::Display for Capture<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.description(f, self.1)
    }
}
