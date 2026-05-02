/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! This module contains the tolerance parsing, matching, and running logic.
//!
//! ## Loading/Parsing/Matching
//!
//! There are a whole host of things that can go wrong during this process, and we're
//! obligated to provide at least somewhat reasonable error messages at each stage.
//!
//! The logic here follows process of continually refining the state of parsed inputs,
//! tolerances, and benchmarks. This is outlined as follows:
//!
//! 1. Deserialize a raw tolerances JSON file into [`Raw`] via [`Raw::load`]. This parses
//!    the tolerance skeleton into a sequence of [`RawInner`].
//!
//!    At this stage, we've:
//!    * Verified the structure of the tolerances file.
//!
//! 2. Parse each [`RawInner`] into a [`ParsedInner`] via [`Raw::parse`].
//!    This attempts to match each [`RawInner`] with a [`registry::RegisteredTolerance`]
//!    and uses said regression to attempt to parse the raw value into a concrete type.
//!
//!    After this stage, each [`ParsedInner`] has the following invariants:
//!    * The tolerance input has been properly deserialized into a concrete struct.
//!    * It has been matched with a [`registry::RegisteredTolerance`], which contains the
//!      collection of inputs and benchmarks that compatible with the parsed tolerance.
//!    * Verified that the `input` associated with the tolerance has a proper association
//!      in the registry.
//!
//! 3. Convert each [`ParsedInner`] into a [`Check`]. This works by matching the raw input
//!    associated with each [`ParsedInner`] to an actual registered input, and then finding
//!    the registered benchmark this is the best match for the input.
//!
//!    For ergonomics, we allow an "input/tolerance" pair to match multiple positional
//!    "inputs" in the input JSON. A "tolerance input" matches with an "actual input" if its
//!    raw JSON passes [`is_subset`] of the actual input's raw JSON. At this step, we need
//!    to work on raw JSON because a parsed input will have deserialization checks run and
//!    can thus look different.
//!
//!    However, matching only succeeds if the above process is complete and unambiguous:
//!    1. Each "input/tolerance" pair gets matched with at least one "actual input".
//!    2. All "actual inputs" have exactly one "input/tolerance" pair that matches them.
//!
//!    At this step, we have the invariants:
//!    * The tolerance is parsed to a concrete type.
//!    * Its associated input has been verified to be consistent with the registry and has
//!      been unambiguously selected from the "actual inputs".
//!    * The selected "actual input" has then been successfully matched with a valid
//!      regression benchmark using the normal matching flow.
//!
//! 4. Finally, [`Checks`] gets converted into [`Jobs`]. During this process, we also verify
//!    the structure of the before/after JSON files and ensure that the number of results mostly
//!    lines up. At this stage, each [`Job`] has the invariants associated with a [`Check`]
//!    with the addition:
//!
//!    * We've been paired with raw before/after JSON that we expect to have the dynamic type
//!      of the output of the associated [`registry::RegisteredBenchmark`].
//!
//!      This gets verified during the actual check runs.
//!
//! The entry points here are:
//!
//! * [`Checks::new`]: Do everything up to step 3. This enables preflight validation checks.
//! * [`Checks::jobs`]: Perform step 4. This prepares us to run all the checks.
//!
//! ## Running Checks
//!
//! Running checks simply involves running each [`Job`] and aggregating the results.
//! Each executed job can end up in one of three states:
//!
//! * Success (yay).
//! * Graceful Failure: Everything looked right in terms of deserialization, but the actual
//!   check failed.
//! * Error: Something went wrong. This could either be because the output JSON could not be
//!   deserialized properly, or for another critical reason.
//!
//! To provide better diagnostics, we wait until all checks have run before beginning a report.
//! The report is triaged in reverse order:
//!
//! * If any check fails with an error, we report all such errors and propagate an error to the top.
//! * If any check gracefully fails, report all such failures and propagate an error to the top.
//! * Otherwise, report the diagnostics from all successes and propagate `Ok(())`.
//!
//! The entry point here is:
//!
//! * [`Jobs::run`]: Run each job, prepare the report, and return a meaningful `Result`.
//!
//! ## Testing
//!
//! Testing is largely facilitated by the crate level UX framework.

use std::{collections::HashMap, io::Write, path::Path, rc::Rc};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    benchmark::{internal::CheckedPassFail, PassFail},
    internal::load_from_disk,
    jobs, registry, result, Any, Checker,
};

////////////
// Checks //
////////////

/// See module level documentation for invariants.
///
/// A `tolerance` can be mapped to multiple inputs and is thus shared behind an [`Rc`].
struct Check<'a> {
    regression: registry::RegressionBenchmark<'a>,
    tolerance: Rc<Any>,
    input: Any,
}

/// See module level documentation for invariants.
pub(crate) struct Checks<'a> {
    checks: Vec<Check<'a>>,
}

impl<'a> Checks<'a> {
    pub(crate) fn new(
        tolerances: &Path,
        input_file: &Path,
        inputs: &registry::Inputs,
        entries: &'a HashMap<&'static str, registry::RegisteredTolerance<'a>>,
    ) -> anyhow::Result<Self> {
        // Load the raw input file.
        let partial = jobs::Partial::load(input_file)?;

        // Parse and validate the raw jobs against the registered inputs.
        //
        // This preserves the ordering of the jobs.
        let inputs = jobs::Jobs::parse(&partial, inputs)?;

        // Now that the inputs have been fully parsed and validated, we then check that we
        // can load the raw tolerance file.
        let parsed = Raw::load(tolerances)?.parse(entries)?;
        Self::match_all(parsed, partial, inputs)
    }

    pub(crate) fn jobs(self, before: &Path, after: &Path) -> anyhow::Result<Jobs<'a>> {
        let (before_path, after_path) = (before, after);

        let before = result::RawResult::load(before_path)?;
        let after = result::RawResult::load(after_path)?;

        let expected = self.checks.len();
        anyhow::ensure!(
            before.len() == expected,
            "\"before\" file \"{}\" has {} entries but expected {}",
            before_path.display(),
            before.len(),
            expected,
        );

        anyhow::ensure!(
            after.len() == expected,
            "\"after\" file \"{}\" has {} entries but expected {}",
            after_path.display(),
            after.len(),
            expected,
        );

        // At this point, `before` and `after` have been deserialized (though not parsed)
        // and we know that the lengths of everything are consistent. We can finally
        // formulate the final list of jobs.
        let jobs = std::iter::zip(self.checks, std::iter::zip(before, after))
            .map(|(check, (before, after))| {
                let Check {
                    regression,
                    tolerance,
                    input,
                } = check;
                Job {
                    regression,
                    tolerance,
                    input,
                    before,
                    after,
                }
            })
            .collect();

        Ok(Jobs { jobs })
    }

    fn match_all(
        parsed: Parsed<'a>,
        partial: jobs::Partial,
        inputs: jobs::Jobs,
    ) -> anyhow::Result<Self> {
        debug_assert_eq!(
            partial.jobs().len(),
            inputs.jobs().len(),
            "expected \"inputs\" to be the parsed representation of \"partial\""
        );

        // Map each `ParsedInner` entry to all `partial` inputs they map to.
        //
        // Each `ParsedInner` unfortunately needs to get compared with every `partial` so we can
        // detect overlapping matches and reject them.
        let mut parsed_to_input: Vec<Vec<usize>> = vec![Vec::default(); parsed.inner.len()];
        let mut input_to_parsed: Vec<Vec<usize>> = vec![Vec::default(); inputs.jobs().len()];

        parsed.inner.iter().enumerate().for_each(|(i, t)| {
            partial.jobs().iter().enumerate().for_each(|(j, raw)| {
                if raw.tag == t.input.tag && is_subset(&raw.content, &t.input.content) {
                    parsed_to_input[i].push(j);
                    input_to_parsed[j].push(i);
                }
            })
        });

        // Validate the whole matching process.
        let input_to_parsed = check_matches(parsed_to_input, input_to_parsed)?;

        // At this point:
        //
        // - `parsed` is known to contain parsed tolerances.
        // - `inputs` is known to contain parsed benchmark inputs.
        // - We've verified that all the parsed tolerances unambiguously match with a
        //   tolerance input.
        //
        // We can now package everything together!
        debug_assert_eq!(input_to_parsed.len(), inputs.jobs().len());

        let checks = std::iter::zip(inputs.into_inner(), input_to_parsed.into_iter())
            .map(|(input, index)| {
                // This index should always be inbounds.
                let inner = &parsed.inner[index];
                assert_eq!(inner.input.tag, input.tag());

                // Within the parsed tolerance, we should be able to find the best-matching
                // regression benchmark for this concrete input. This benchmark should exist,
                // but it's possible that code changes between when the results were generated
                // and now has led to the input no longer being matchable with anything.
                let regression = inner
                    .entry
                    .regressions
                    .iter()
                    .filter_map(|r| r.try_match(&input).ok().map(|score| (*r, score)))
                    .min_by_key(|(_, score)| *score)
                    .map(|(r, _)| r)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Could not match input tag \"{}\" and tolerance tag \"{}\" to \
                             a valid benchmark. This likely means file or code changes \
                             between when the input file was last used. If the normal \
                             benchmark flow succeeds, please report this issue.",
                            inner.input.tag,
                            inner.tolerance.tag(),
                        )
                    })?;

                Ok(Check {
                    regression,
                    tolerance: inner.tolerance.clone(),
                    input,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(Self { checks })
    }
}

//---------//
// Helpers //
//---------//

/// A raw unprocessed tolerance job.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RawInner {
    input: jobs::Unprocessed,
    tolerance: jobs::Unprocessed,
}

impl RawInner {
    pub(crate) fn new(input: jobs::Unprocessed, tolerance: jobs::Unprocessed) -> Self {
        Self { input, tolerance }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct Raw {
    checks: Vec<RawInner>,
}

impl Raw {
    pub(crate) fn load(path: &Path) -> anyhow::Result<Self> {
        load_from_disk(path)
    }

    fn parse<'a>(
        self,
        entries: &'a HashMap<&'static str, registry::RegisteredTolerance<'a>>,
    ) -> anyhow::Result<Parsed<'a>> {
        // Attempt to parse raw tolerances into registered tolerance inputs.
        let num_checks = self.checks.len();
        let mut checker = Checker::new(vec![], None);
        let inner = self
            .checks
            .into_iter()
            .enumerate()
            .map(|(i, unprocessed)| {
                let context = || {
                    format!(
                        "while processing tolerance input {} of {}",
                        i.wrapping_add(1),
                        num_checks,
                    )
                };

                // Does this tolerance tag matched a registered tolerance?
                let entry = entries
                    .get(&*unprocessed.tolerance.tag)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Unrecognized tolerance tag: \"{}\"",
                            unprocessed.tolerance.tag
                        )
                    })
                    .with_context(context)?;

                // Verify that the accompanying input tag is accepted by at least one
                // benchmark registered under this tolerance.
                if !entry
                    .regressions
                    .iter()
                    .any(|r| r.input_tag() == unprocessed.input.tag)
                {
                    let valid: Vec<_> = entry
                        .regressions
                        .iter()
                        .map(|pair| pair.input_tag())
                        .collect();
                    return Err(anyhow::anyhow!(
                        "input tag \"{}\" is not compatible with tolerance tag \"{}\". \
                         Valid input tags are: {:?}",
                        unprocessed.input.tag,
                        unprocessed.tolerance.tag,
                        valid,
                    ))
                    .with_context(context);
                }

                checker.set_tag(entry.tolerance.tag());
                let tolerance = entry
                    .tolerance
                    .try_deserialize(&unprocessed.tolerance.content, &mut checker)
                    .with_context(context)?;

                Ok(ParsedInner {
                    entry,
                    tolerance: Rc::new(tolerance),
                    input: unprocessed.input,
                })
            })
            .collect::<anyhow::Result<_>>()?;

        Ok(Parsed { inner })
    }

    pub(crate) fn example() -> String {
        #[expect(
            clippy::expect_used,
            reason = "we control the concrete struct and its serialization implementation"
        )]
        serde_json::to_string_pretty(&Self::default())
            .expect("built-in serialization should succeed")
    }
}

/// Invariants:
///
/// * `tolerance` is parsed to the dynamic type of the associated tolerance in `entry`.
/// * The tag in `input` exists within at least one of the regressions in `entry`.
#[derive(Debug)]
struct ParsedInner<'a> {
    entry: &'a registry::RegisteredTolerance<'a>,
    tolerance: Rc<Any>,
    input: jobs::Unprocessed,
}

#[derive(Debug)]
struct Parsed<'a> {
    inner: Vec<ParsedInner<'a>>,
}

/// Return `true` only `needle` is a structural subset of `haystack`. This is defined as:
///
/// 1. All flattened paths of `needle` are flattened paths of `haystack`.
/// 2. The values at the end of all flattened paths are equal.
///
/// When matching arrays, `needle` is matched as a potential prefix of the corresponding
/// entry in `haystack`.
#[must_use]
pub(crate) fn is_subset(mut haystack: &Value, mut needle: &Value) -> bool {
    macro_rules! false_if {
        ($expr:expr) => {
            if $expr {
                return false;
            }
        };
    }

    // Note that we use a `do-while` style loop to short-circuit situations where we
    // match/mismatch immediately, saving an allocation.
    //
    // If we exit on the first iteration, the vector stays empty and thus doesn't allocate.
    let mut stack = Vec::new();
    loop {
        match (haystack, needle) {
            (Value::Null, Value::Null) => {
                // Null always matches
            }
            (Value::Bool(h), Value::Bool(n)) => false_if!(h != n),
            (Value::Number(h), Value::Number(n)) => false_if!(h != n),
            (Value::String(h), Value::String(n)) => false_if!(h != n),
            (Value::Array(h), Value::Array(n)) => {
                // If `n` is longer, then it cannot possibly be a subset of `h`.
                // On the flip side, if `n` is shorter, then we can at least try to match
                // the prefix.
                false_if!(h.len() < n.len());
                std::iter::zip(h.iter(), n.iter()).for_each(|(h, n)| stack.push((h, n)));
            }
            (Value::Object(h), Value::Object(n)) => {
                for (k, v) in n.iter() {
                    match h.get(k) {
                        Some(h) => stack.push((h, v)),
                        None => return false,
                    }
                }
            }
            // If the two enums are not the same, then we have a fundamental mismatch.
            _ => return false,
        }

        if let Some((h, n)) = stack.pop() {
            (haystack, needle) = (h, n);
        } else {
            break;
        }
    }

    true
}

/// A single problem detected during bipartite tolerance-to-input matching.
#[derive(Debug, PartialEq)]
enum MatchProblem {
    /// Tolerance at this index matched no inputs.
    OrphanedTolerance(usize),
    /// Input at this index matched no tolerances.
    UncoveredInput(usize),
    /// Input at this index matched multiple tolerances.
    AmbiguousInput(usize, Vec<usize>),
}

/// Error returned when the bipartite matching between tolerance entries and inputs is
/// invalid.
#[derive(Debug)]
struct AmbiguousMatch(Vec<MatchProblem>);

impl std::fmt::Display for AmbiguousMatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tolerance matching failed:")?;
        for problem in &self.0 {
            match problem {
                MatchProblem::OrphanedTolerance(i) => {
                    write!(f, "\n  tolerance {} matched no inputs", i + 1)?;
                }
                MatchProblem::UncoveredInput(i) => {
                    write!(f, "\n  input {} matched no tolerances", i + 1)?;
                }
                MatchProblem::AmbiguousInput(i, tolerances) => {
                    write!(f, "\n  input {} matched tolerances ", i + 1)?;
                    for (j, &t) in tolerances.iter().enumerate() {
                        if j > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", t + 1)?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl std::error::Error for AmbiguousMatch {}

/// Validate that every entry in `parsed_to_input` has at least one match and that all
/// entries in `input_to_parsed` have exactly one match.
///
/// Return unique matches from `input_to_parsed` on success. Otherwise, return a
/// descriptive error.
fn check_matches(
    parsed_to_input: Vec<Vec<usize>>,
    input_to_parsed: Vec<Vec<usize>>,
) -> Result<Vec<usize>, AmbiguousMatch> {
    let mut problems = Vec::new();

    for (i, matches) in parsed_to_input.iter().enumerate() {
        if matches.is_empty() {
            problems.push(MatchProblem::OrphanedTolerance(i));
        }
    }

    let mut result = Vec::with_capacity(input_to_parsed.len());
    for (i, matches) in input_to_parsed.into_iter().enumerate() {
        match matches.len() {
            0 => problems.push(MatchProblem::UncoveredInput(i)),
            1 => result.push(matches[0]),
            _ => problems.push(MatchProblem::AmbiguousInput(i, matches)),
        }
    }

    if problems.is_empty() {
        Ok(result)
    } else {
        Err(AmbiguousMatch(problems))
    }
}

//////////
// Jobs //
//////////

/// A fully parsed and (hopefully) ready to run regression check.
#[derive(Debug)]
pub(crate) struct Job<'a> {
    /// The executor for the actual check we wish to run.
    regression: registry::RegressionBenchmark<'a>,

    /// The [`crate::benchmark::Regression::Tolerance`] associated with `regression`.
    tolerance: Rc<Any>,

    /// The [`crate::Benchmark::Input`] associated with `benchmark`.
    input: Any,

    /// The [`result::RawResult`] from the "before" comparison.
    ///
    /// Payload should be deserializable to [`crate::Benchmark::Output`].
    before: result::RawResult,

    /// The [`result::RawResult`] from the "after" comparison.
    ///
    /// Payload should be deserializable to [`crate::Benchmark::Output`].
    after: result::RawResult,
}

impl Job<'_> {
    /// Actually run the jobs.
    ///
    /// As long as the chain of custody throughout this module is correct, at least the
    /// `tolerance` and `input` fields should match the associated regression capable
    /// `benchmark`.
    ///
    /// The associated outputs may still fail to deserialize properly and the check could
    /// still fail. This is why the [`Jobs`] struct aggregates together all results before
    /// deciding how they should be displayed.
    fn run(&self) -> anyhow::Result<CheckedPassFail> {
        self.regression.check(
            &self.tolerance,
            &self.input,
            &self.before.results,
            &self.after.results,
        )
    }
}

#[derive(Debug)]
pub(crate) struct Jobs<'a> {
    jobs: Vec<Job<'a>>,
}

impl Jobs<'_> {
    /// Run regression checks by comparing before/after output files against the matched
    /// tolerances.
    ///
    /// The priority cascade for terminal output is:
    ///
    /// 1. If any checks produce an infrastructure error, report **all** errors and return
    ///    `Err`. Pass/fail results are suppressed so errors stay front-and-center.
    /// 2. Otherwise, if any checks fail, report **all** failures and return `Err`.
    ///    Successes are suppressed for the same reason.
    /// 3. Otherwise, all checks passed — report them and return `Ok`.
    ///
    /// The JSON output (if `output_file` is provided) is always written regardless of
    /// outcome, so downstream tooling can inspect all results.
    ///
    /// TODO: We could consider a `--verbose` flag to record all outcomes regardless of
    /// priority, but for now the hierarchy of reporting seems the most pragmatic.
    pub(crate) fn run(
        &self,
        mut output: &mut dyn crate::output::Output,
        output_file: Option<&Path>,
    ) -> anyhow::Result<()> {
        // Step 1: Run all checks, collecting results.
        let results: Vec<_> = self.jobs.iter().map(|job| job.run()).collect();

        // Step 2: Build the JSON output array (always, even on errors).
        let check_outputs: Vec<CheckOutput<'_>> = std::iter::zip(self.jobs.iter(), results.iter())
            .map(|(job, result)| -> anyhow::Result<_> {
                let tolerance = job.tolerance.serialize()?;
                let o = match result {
                    Ok(PassFail::Pass(checked)) => CheckOutput::pass(tolerance, &checked.json),
                    Ok(PassFail::Fail(checked)) => CheckOutput::fail(tolerance, &checked.json),
                    Err(err) => CheckOutput::error(tolerance, err),
                };

                Ok(o)
            })
            .collect::<anyhow::Result<_>>()?;

        // Write JSON output before the cascade so it's available even on failure.
        if let Some(path) = output_file {
            let json = serde_json::to_string_pretty(&check_outputs)?;
            std::fs::write(path, json)
                .with_context(|| format!("failed to write output to \"{}\"", path.display()))?;
        }

        // Step 3: If any errors, report all of them and bail.
        let mut has_errors = false;
        for (i, result) in results.iter().enumerate() {
            if let Err(err) = result {
                let job = &self.jobs[i];
                writeln!(
                    output,
                    "Check {} of {} ({:?}) encountered an error:\n{:?}\n",
                    i + 1,
                    self.jobs.len(),
                    job.regression.name(),
                    err,
                )?;
                has_errors = true;
            }
        }
        if has_errors {
            return Err(anyhow::anyhow!("one or more checks failed with errors"));
        }

        // Step 4: All checks completed. Report any failures.
        // (Safe to unwrap since we've handled all Err cases above.)
        let mut has_failures = false;
        for (i, result) in results.iter().enumerate() {
            #[expect(
                clippy::expect_used,
                reason = "we would have ready returned if errors were present"
            )]
            let outcome = result
                .as_ref()
                .expect("no errors should be present any more");
            if let PassFail::Fail(checked) = outcome {
                let job = &self.jobs[i];
                writeln!(
                    output,
                    "Check {} of {} ({:?}) FAILED:",
                    i + 1,
                    self.jobs.len(),
                    job.regression.name(),
                )?;
                writeln!(output, "{}", checked.display)?;
                writeln!(output)?;
                has_failures = true;
            }
        }
        if has_failures {
            return Err(anyhow::anyhow!("one or more regression checks failed"));
        }

        // Step 5: Everything passed.
        for (i, result) in results.iter().enumerate() {
            #[expect(
                clippy::expect_used,
                reason = "we would have returned if errors were present"
            )]
            let outcome = result
                .as_ref()
                .expect("no errors should be present any more");
            let PassFail::Pass(checked) = outcome else {
                unreachable!("all failures handled above");
            };
            let job = &self.jobs[i];
            writeln!(
                output,
                "Check {} of {} ({:?}) PASSED:",
                i + 1,
                self.jobs.len(),
                job.regression.name(),
            )?;
            writeln!(output, "{}", checked.display)?;
            writeln!(output)?;
        }

        Ok(())
    }
}

/// Serialized output for a single regression check, suitable for downstream tooling.
///
/// Positional index in the output array corresponds to the input/tolerance files.
#[derive(Serialize)]
struct CheckOutput<'a> {
    status: &'static str,
    tolerance: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<&'a Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl<'a> CheckOutput<'a> {
    fn pass(tolerance: Value, result: &'a Value) -> Self {
        Self {
            status: "pass",
            tolerance,
            result: Some(result),
            error: None,
        }
    }

    fn fail(tolerance: Value, result: &'a Value) -> Self {
        Self {
            status: "fail",
            tolerance,
            result: Some(result),
            error: None,
        }
    }

    fn error(tolerance: Value, err: &anyhow::Error) -> Self {
        let error = err
            .chain()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(": ");
        Self {
            status: "error",
            tolerance,
            result: None,
            error: Some(error),
        }
    }
}

///////////
// Tests //
///////////

// Note: much of the functionality in this file is related to error handling and relies on
// having a fully functional registry.
//
// To that end, the UX tests are the primary test vessel for much of parsing code.
// The unit tests here stay focused on the bits that are actually feasibly unit testable.
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Construct a vector of `serde_json::Values` with all possible variants.
    ///
    /// Aggregates `Array` and `Map` are empty.
    fn empty_values() -> Vec<Value> {
        vec![
            Value::Null,
            Value::Bool(false),
            Value::Number(serde_json::Number::from_f64(0.0).unwrap()),
            Value::String(String::new()),
            Value::Array(Vec::new()),
            Value::Object(serde_json::Map::new()),
        ]
    }

    #[test]
    fn test_is_subset() {
        // Null
        for v in empty_values() {
            if matches!(v, Value::Null) {
                assert!(is_subset(&Value::Null, &v));
            } else {
                assert!(!is_subset(&Value::Null, &v));
            }
        }

        // Bool / Number / String require exact equality and type matches.
        assert!(is_subset(&json!(true), &json!(true)));
        assert!(!is_subset(&json!(true), &json!(false)));
        assert!(!is_subset(&json!(true), &json!(0)));

        assert!(is_subset(&json!(7), &json!(7)));
        assert!(!is_subset(&json!(7), &json!(8)));
        assert!(!is_subset(&json!(7), &json!("7")));

        assert!(is_subset(&json!("abc"), &json!("abc")));
        assert!(!is_subset(&json!("abc"), &json!("def")));

        // Arrays match by prefix.
        assert!(is_subset(&json!([1, 2, 3]), &json!([])));
        assert!(is_subset(&json!([1, 2, 3]), &json!([1])));
        assert!(is_subset(&json!([1, 2, 3]), &json!([1, 2])));
        assert!(is_subset(&json!([1, 2, 3]), &json!([1, 2, 3])));
        assert!(!is_subset(&json!([1, 2]), &json!([1, 2, 3])));
        assert!(!is_subset(&json!([1, 2, 3]), &json!([1, 3])));

        // Objects match by recursive structural subset.
        assert!(is_subset(&json!({"a": 1, "b": 2}), &json!({"a": 1})));
        assert!(is_subset(&json!({"a": 1, "b": 2}), &json!({})));
        assert!(is_subset(
            &json!({"a": {"b": 1, "c": 2}, "d": 3}),
            &json!({"a": {"b": 1}}),
        ));
        assert!(!is_subset(&json!({"a": 1}), &json!({"a": 1, "b": 2}),));
        assert!(!is_subset(&json!({"a": {"b": 1}}), &json!({"a": {"b": 2}}),));

        // Nested array/object combinations use the same recursive rules.
        assert!(is_subset(
            &json!({"ops": [{"kind": "l2", "dim": 128}, {"kind": "cosine", "dim": 256}]}),
            &json!({"ops": [{"kind": "l2"}]}),
        ));
        assert!(is_subset(
            &json!({"ops": [{"kind": "l2", "dim": 128}, {"kind": "cosine", "dim": 256}]}),
            &json!({"ops": [{"kind": "l2", "dim": 128}, {"kind": "cosine"}]}),
        ));
        assert!(!is_subset(
            &json!({"ops": [{"kind": "l2", "dim": 128}, {"kind": "cosine", "dim": 256}]}),
            &json!({"ops": [{"kind": "cosine"}]}),
        ));
    }

    #[test]
    fn test_check_matches_success() {
        let result = check_matches(vec![vec![0], vec![1]], vec![vec![0], vec![1]]).unwrap();
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn test_check_matches_reports_problems_in_stable_order() {
        let err = check_matches(
            vec![vec![0], vec![], vec![2, 3]],
            vec![vec![0], vec![], vec![2, 3]],
        )
        .unwrap_err();

        assert_eq!(
            &err.0,
            &[
                MatchProblem::OrphanedTolerance(1),
                MatchProblem::UncoveredInput(1),
                MatchProblem::AmbiguousInput(2, vec![2, 3]),
            ]
        )
    }
}
