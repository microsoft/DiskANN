/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::{hash_map::Entry, HashMap};

use thiserror::Error;

use crate::{
    benchmark::{self, internal::AnnotatedMatch, Benchmark, MatchContext, Regression, Score},
    input,
    internal::visibility::Visibility,
    Checkpoint, Features, Input, Output,
};

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

    pub(crate) fn inputs(&self) -> impl ExactSizeIterator<Item = input::Registered<'_>> {
        self.inputs.values().map(|v| input::Registered(&**v))
    }

    //--------------//
    // Registration //
    //--------------//

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

    pub fn register_gated(
        &mut self,
        tag: &'static str,
        name: impl Into<String>,
        features: Features,
        description: impl Into<String>,
    ) -> Result<(), RegistryError> {
        self.register_gated_input(crate::input::internal::Gated::new(tag, features.clone()))?;

        self.benchmarks.push(RegisteredBenchmark {
            name: name.into(),
            benchmark: Box::new(benchmark::internal::FullyGated::new(
                tag,
                features,
                description.into(),
            )),
        });
        Ok(())
    }

    pub fn register_partially_gated<I>(
        &mut self,
        name: impl Into<String>,
        features: Features,
        description: impl Into<String>,
    ) -> Result<(), RegistryError>
    where
        I: Input,
    {
        self.register_input::<I>()?;

        self.benchmarks.push(RegisteredBenchmark {
            name: name.into(),
            benchmark: Box::new(benchmark::internal::PartiallyGated::<I>::new(
                features,
                description.into(),
            )),
        });
        Ok(())
    }

    /// Return an iterator over registered benchmark names and their descriptions.
    pub(crate) fn benchmarks(&self) -> &[RegisteredBenchmark] {
        &self.benchmarks
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
    ) -> Result<(), Mismatches<'_>> {
        // Simple `has_match` will minimally allocate if an actual match is found, saving
        // the work done below to put together a comprehensive summary of the closest matches.
        if self.has_match(job) {
            return Ok(());
        }

        let mut gated = Vec::<(&RegisteredBenchmark, &Features)>::new();
        let mut failures = Vec::<(&RegisteredBenchmark, Score)>::new();

        for entry in self.benchmarks() {
            match entry
                .benchmark
                .try_match(job, &MatchContext::with_reasons())
            {
                AnnotatedMatch::User(score) => {
                    if score.is_success() {
                        // This should be unreachable from the earlier `has_match` check.
                        return Ok(());
                    } else {
                        failures.push((entry, score));
                    }
                }
                // Don't report anything for a wrong tag. It's not very useful.
                AnnotatedMatch::WrongTag => {}
                AnnotatedMatch::Gated(features) => gated.push((entry, features)),
            }
        }

        failures.sort_by(|(_, left), (_, right)| Score::order(left, right));
        failures.truncate(max_methods);

        let mismatches = Mismatches {
            tag: job.tag(),
            user: failures,
            gated,
        };

        Err(mismatches)
    }

    /// Find the best matching benchmark for `job` by score.
    fn find_best_match(&self, job: &input::internal::Any) -> Option<&RegisteredBenchmark> {
        find_best_match(job, self.benchmarks())
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
                } else if let Some(existing) = o
                    .get()
                    .as_any()
                    .downcast_ref::<crate::input::internal::Gated>()
                {
                    Err(RegistryError {
                        tag,
                        existing: Kind::Gated(existing.features().to_string()),
                        new: Kind::Available(wrapper.type_name()),
                    })
                } else {
                    Err(RegistryError {
                        tag,
                        existing: Kind::Available(o.get().type_name()),
                        new: Kind::Available(wrapper.type_name()),
                    })
                }
            }
        }
    }

    fn register_gated_input(
        &mut self,
        input: crate::input::internal::Gated,
    ) -> Result<(), RegistryError> {
        match self.inputs.entry(input.tag()) {
            Entry::Vacant(v) => {
                v.insert(Box::new(input));
                Ok(())
            }
            Entry::Occupied(o) => {
                if let Some(existing) = o
                    .get()
                    .as_any()
                    .downcast_ref::<crate::input::internal::Gated>()
                {
                    if existing.features() != input.features() {
                        Err(RegistryError {
                            tag: input.tag(),
                            existing: Kind::Gated(existing.features().to_string()),
                            new: Kind::Gated(input.features().to_string()),
                        })
                    } else {
                        Ok(())
                    }
                } else {
                    let type_name = o.get().type_name();
                    Err(RegistryError {
                        tag: input.tag(),
                        existing: Kind::Available(type_name),
                        new: Kind::Gated(input.features().to_string()),
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

pub(crate) trait CommonTryMatch {
    fn common_try_match(&self, input: &input::internal::Any) -> AnnotatedMatch<'_>;
}

/// Return the entry in `benchmarks` that has the best successful match to `job`.
///
/// Returns `None` if no entries in `benchmarks` matches `job`.
pub(crate) fn find_best_match<'a, T>(
    job: &input::internal::Any,
    benchmarks: &'a [T],
) -> Option<&'a T>
where
    T: CommonTryMatch,
{
    benchmarks
        .iter()
        .filter_map(|entry| {
            let score: benchmark::Score = entry.common_try_match(job).try_into_score()?;
            score
                .match_score()
                .map(|s: benchmark::SuccessScore| (entry, s))
        })
        .min_by_key(|(_, score)| *score)
        .map(|(entry, _)| entry)
}

/// A registered benchmark entry: a name paired with a type-erased benchmark.
pub(crate) struct RegisteredBenchmark {
    name: String,
    benchmark: Box<dyn benchmark::internal::Benchmark>,
}

impl std::fmt::Debug for RegisteredBenchmark {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let benchmark = Capture(&*self.benchmark);
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

    pub(crate) fn visibility(&self) -> Visibility<'_> {
        self.benchmark.visibility()
    }

    pub(crate) fn display(&self) -> Display<'_> {
        Display(self)
    }

    pub(crate) fn description(&self) -> String {
        Capture(&*self.benchmark).to_string()
    }

    /// Order available benchmarks first (alphabetically) followed by gated benchmarks
    /// (also alphabetically).
    pub(crate) fn order(this: &&Self, other: &&Self) -> std::cmp::Ordering {
        use Visibility::{Available, Gated};

        match (this.visibility(), other.visibility()) {
            (Available, Available) => this.name().cmp(other.name()),
            (Available, Gated { .. }) => std::cmp::Ordering::Less,
            (Gated { .. }, Available) => std::cmp::Ordering::Greater,
            (
                Gated {
                    features: this_features,
                },
                Gated {
                    features: other_features,
                },
            ) => this
                .name()
                .cmp(other.name())
                .then_with(|| this_features.cmp(other_features)),
        }
    }
}

impl CommonTryMatch for RegisteredBenchmark {
    fn common_try_match(&self, input: &input::internal::Any) -> AnnotatedMatch<'_> {
        self.benchmark().try_match(input, &MatchContext::new())
    }
}

pub(crate) struct Display<'a>(&'a RegisteredBenchmark);

impl std::fmt::Display for Display<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = self.0.name();
        match self.0.visibility() {
            Visibility::Available => writeln!(f, "{name}:")?,
            Visibility::Gated { features } => writeln!(f, "{name} (requires the {}):", features)?,
        };

        write!(
            f,
            "{}",
            crate::utils::fmt::Indent::new(&self.0.description(), 4)
        )
    }
}

/// Error for [`Registry::register`] or [`Registry::register_regression`].
#[derive(Debug, Error)]
#[error(
    "A different input with tag \"{}\" was already registered. Existing {}. New {}",
    self.tag,
    self.existing,
    self.new,
)]
pub struct RegistryError {
    tag: &'static str,
    existing: Kind,
    new: Kind,
}

#[derive(Debug)]
enum Kind {
    Available(&'static str),
    Gated(String),
}

impl std::fmt::Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Available(type_name) => write!(f, "type \"{}\"", type_name),
            Self::Gated(features) => write!(f, "gated input for {}", features),
        }
    }
}

#[derive(Debug)]
pub(crate) struct Mismatches<'a> {
    tag: &'static str,
    user: Vec<(&'a RegisteredBenchmark, Score)>,
    gated: Vec<(&'a RegisteredBenchmark, &'a Features)>,
}

impl std::fmt::Display for Mismatches<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Print out user mismatches
        if self.user.is_empty() {
            writeln!(
                f,
                "No active benchmarks are registered for tag \"{}\"",
                self.tag
            )?;
        } else {
            writeln!(f, "Closest matches for tag \"{}\":\n", self.tag)?;
            for (i, (entry, score)) in self.user.iter().enumerate() {
                writeln!(f, "    {}. \"{}\":", i + 1, entry.name())?;
                writeln!(
                    f,
                    "{}\n",
                    crate::utils::fmt::Indent::new(&score.reason().to_string(), 8)
                )?;
            }
        }

        // Print out gated benchmarks (if any).
        if !self.gated.is_empty() {
            writeln!(
                f,
                "\nFound {} gated {} matching this input:\n",
                self.gated.len(),
                if self.gated.len() == 1 {
                    "benchmark"
                } else {
                    "benchmarks"
                },
            )?;
            for (entry, features) in self.gated.iter() {
                writeln!(f, "    * \"{}\" (requires the {})", entry.name(), features)?;
            }
        }

        Ok(())
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

impl CommonTryMatch for RegressionBenchmark<'_> {
    fn common_try_match(&self, input: &input::internal::Any) -> AnnotatedMatch<'_> {
        self.benchmark.common_try_match(input)
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
struct Capture<'a>(&'a dyn benchmark::internal::Benchmark);

impl std::fmt::Display for Capture<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.description(f)
    }
}

impl std::fmt::Debug for Capture<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.description(f)
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

        let mut tags: Vec<_> = registry.inputs().map(|i| i.tag()).collect();
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

    #[test]
    fn test_gated_to_real_conflict() {
        let mut registry = Registry::new();
        registry
            .register_gated_input(crate::input::internal::Gated::new(
                "type-a",
                Features::new("feature"),
            ))
            .unwrap();

        registry.register_input::<B>().unwrap();

        let mut tags: Vec<_> = registry.inputs().map(|i| i.tag()).collect();
        tags.sort();
        assert_eq!(tags.as_slice(), ["type-a", "type-b"]);

        {
            let a = registry._input(A::tag()).unwrap();
            assert!(a.as_any().is::<crate::input::internal::Gated>());
        }

        let err = registry.register_input::<A>().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("A different input with tag \"type-a\" was already registered"),
            "FAILED: {}",
            msg
        );
    }

    #[test]
    fn test_real_to_gated_conflict() {
        let mut registry = Registry::new();

        registry.register_input::<A>().unwrap();
        registry.register_input::<B>().unwrap();

        let err = registry
            .register_gated_input(crate::input::internal::Gated::new(
                "type-a",
                Features::new("feature"),
            ))
            .unwrap_err();

        let msg = err.to_string();
        assert!(
            msg.contains("A different input with tag \"type-a\" was already registered"),
            "FAILED: {}",
            msg
        );
    }

    #[test]
    fn test_gated_to_gated() {
        let mut registry = Registry::new();

        registry
            .register_gated_input(crate::input::internal::Gated::new(
                "type-a",
                Features::new("feature"),
            ))
            .unwrap();

        // If we register with the same feature set, there is no error.
        registry
            .register_gated_input(crate::input::internal::Gated::new(
                "type-a",
                Features::new("feature"),
            ))
            .unwrap();

        // If we register with a different feature-set, that is an error.
        let err = registry
            .register_gated_input(crate::input::internal::Gated::new(
                "type-a",
                Features::any(["feature", "another"]),
            ))
            .unwrap_err();

        let msg = err.to_string();
        assert!(
            msg.contains("A different input with tag \"type-a\" was already registered"),
            "FAILED: {}",
            msg
        );
    }
}
