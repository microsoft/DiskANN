/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The CLI frontend for benchmark applications built with this crate.
//!
//! [`App`] provides a [`clap`]-based command line interface that handles input parsing,
//! benchmark dispatch, and regression checking. Consumers build a binary by registering
//! [`Input`](crate::Input)s and [`Benchmark`](crate::Benchmark)s, then forwarding to
//! [`App::parse`] and [`App::run`].
//!
//! # Subcommands
//!
//! ## Standard Workflow
//!
//! * `inputs [NAME]`: List available input kinds, or describe one by name.
//! * `benchmarks`: List registered benchmarks and their descriptions.
//! * `skeleton`: Print a skeleton input JSON file.
//! * `run --input-file <FILE> --output-file <FILE> [--dry-run]`: Run benchmarks.
//!
//! ## Regression Checks
//!
//! These are accessed via `check <SUBCOMMAND>`:
//!
//! * `check skeleton`: Print a skeleton tolerance JSON file.
//! * `check tolerances [NAME]`: List tolerance kinds, or describe one by name.
//! * `check verify --tolerances <FILE> --input-file <FILE>`: Validate a tolerance file
//!   against an input file.
//! * `check run --tolerances <FILE> --input-file <FILE> --before <FILE> --after <FILE> [--output-file <FILE>]`:
//!   Run regression checks.
//!
//! # Example
//!
//! A typical binary using this crate:
//!
//! ```rust,no_run
//! use diskann_benchmark_runner::{App, registry};
//!
//! fn main() -> anyhow::Result<()> {
//!     let mut inputs = registry::Inputs::new();
//!     // inputs.register::<MyInput>()?;
//!
//!     let mut benchmarks = registry::Benchmarks::new();
//!     // benchmarks.register::<MyBenchmark>("my-bench");
//!     // benchmarks.register_regression::<MyRegressionBenchmark>("my-regression");
//!
//!     let app = App::parse();
//!     let mut output = diskann_benchmark_runner::output::default();
//!     app.run(&inputs, &benchmarks, &mut output)
//! }
//! ```
//!
//! # Regression Workflow
//!
//! 1. Run benchmarks twice (e.g. before and after a code change) with `run`, producing
//!    two output files.
//! 2. Author a tolerance file describing acceptable variation (use `check skeleton` and
//!    `check tolerances` for guidance).
//! 3. Validate the tolerance file with `check verify`.
//! 4. Compare the two output files with `check run`.

use std::{io::Write, path::PathBuf};

use clap::{Parser, Subcommand};

use crate::{
    internal,
    jobs::{self, Jobs},
    output::Output,
    registry,
    result::Checkpoint,
    utils::fmt::Banner,
};

/// Check if we're running in debug mode and error if not allowed.
fn check_debug_mode(allow_debug: bool) -> anyhow::Result<()> {
    if cfg!(debug_assertions) && !allow_debug {
        anyhow::bail!(
            "Benchmarking in debug mode produces misleading performance results.\n\
             Please compile in release mode or use the --allow-debug flag to bypass this check."
        );
    }
    Ok(())
}

/// Parsed command line options.
#[derive(Debug, Subcommand)]
pub enum Commands {
    /// List the kinds of input formats available for ingestion.
    Inputs {
        /// Describe the layout of the named input kind.
        describe: Option<String>,
    },
    /// List the available benchmarks.
    Benchmarks {},
    /// Provide a skeleton JSON file for running a set of benchmarks.
    Skeleton,
    /// Run a list of benchmarks.
    Run {
        /// The input file to run.
        #[arg(long = "input-file")]
        input_file: PathBuf,
        /// The path where the output file should reside.
        #[arg(long = "output-file")]
        output_file: PathBuf,
        /// Parse an input file and perform all validation checks, but don't actually run any
        /// benchmarks.
        #[arg(long, action)]
        dry_run: bool,
        /// Allow running benchmarks in debug mode (not recommended).
        #[arg(long, action)]
        allow_debug: bool,
    },
    #[command(subcommand)]
    Check(Check),
}

/// Subcommands for regression check operations.
#[derive(Debug, Subcommand)]
pub enum Check {
    /// Provide a skeleton of the overall tolerance files.
    Skeleton,
    /// List all the tolerance inputs accepted by the benchmark executable.
    Tolerances {
        /// Describe the layout for the named tolerance kind.
        describe: Option<String>,
    },
    /// Verify the tolerance file with the accompanying input file.
    Verify {
        /// The tolerance file to check.
        #[arg(long = "tolerances")]
        tolerances: PathBuf,
        /// The benchmark input file used to generate the data that will be compared.
        #[arg(long = "input-file")]
        input_file: PathBuf,
    },
    /// Run regression checks against before/after output files.
    Run {
        /// The tolerance file to check.
        #[arg(long = "tolerances")]
        tolerances: PathBuf,
        /// The benchmark input file used to generate the data that will be compared.
        #[arg(long = "input-file")]
        input_file: PathBuf,
        /// The `--output-file` from a benchmark to use as a baseline.
        #[arg(long = "before")]
        before: PathBuf,
        /// The `--output-file` that will be checked for regression against `before`.
        #[arg(long = "after")]
        after: PathBuf,
        /// Optional path to write the JSON check results.
        #[arg(long = "output-file")]
        output_file: Option<PathBuf>,
    },
}

/// The CLI used to drive a benchmark application.
#[derive(Debug, Parser)]
pub struct App {
    #[command(subcommand)]
    command: Commands,
}

impl App {
    /// Construct [`Self`] by parsing commandline arguments from [`std::env::args`].
    ///
    /// This simply redirects to [`clap::Parser::parse`] and is provided to allow parsing
    /// without the [`clap::Parser`] trait in scope.
    pub fn parse() -> Self {
        <Self as clap::Parser>::parse()
    }

    /// Construct [`Self`] by parsing command line arguments from the iterator.
    ///
    /// This simply redirects to [`clap::Parser::try_parse_from`] and is provided to allow
    /// parsing without the [`clap::Parser`] trait in scope.
    pub fn try_parse_from<I, T>(itr: I) -> anyhow::Result<Self>
    where
        I: IntoIterator<Item = T>,
        T: Into<std::ffi::OsString> + Clone,
    {
        Ok(<Self as clap::Parser>::try_parse_from(itr)?)
    }

    /// Construct [`Self`] directly from a [`Commands`] enum.
    pub fn from_commands(command: Commands) -> Self {
        Self { command }
    }

    /// Run the application using the registered `inputs` and `benchmarks`.
    pub fn run(
        &self,
        inputs: &registry::Inputs,
        benchmarks: &registry::Benchmarks,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<()> {
        match &self.command {
            // If a named benchmark isn't given, then list the available benchmarks.
            Commands::Inputs { describe } => {
                if let Some(describe) = describe {
                    if let Some(input) = inputs.get(describe) {
                        let repr = jobs::Unprocessed::format_input(input)?;
                        writeln!(
                            output,
                            "The example JSON representation for \"{}\" is:",
                            describe
                        )?;
                        writeln!(output, "{}", serde_json::to_string_pretty(&repr)?)?;
                        return Ok(());
                    } else {
                        writeln!(output, "No input found for \"{}\"", describe)?;
                    }

                    return Ok(());
                }

                writeln!(output, "Available input kinds are listed below:")?;
                let mut tags: Vec<_> = inputs.tags().collect();
                tags.sort();
                for i in tags.iter() {
                    writeln!(output, "    {}", i)?;
                }
            }
            // List the available benchmarks.
            Commands::Benchmarks {} => {
                writeln!(output, "Registered Benchmarks:")?;
                for (name, description) in benchmarks.names() {
                    let mut lines = description.lines();
                    if let Some(first) = lines.next() {
                        writeln!(output, "    {}: {}", name, first)?;
                        for line in lines {
                            writeln!(output, "        {}", line)?;
                        }
                    } else {
                        writeln!(output, "    {}: <no description>", name)?;
                    }
                }
            }
            Commands::Skeleton => {
                writeln!(output, "Skeleton input file:")?;
                writeln!(output, "{}", Jobs::example()?)?;
            }
            // Run the benchmarks
            Commands::Run {
                input_file,
                output_file,
                dry_run,
                allow_debug,
            } => {
                // Parse and validate the input.
                let run = Jobs::load(input_file, inputs)?;
                // Check if we have a match for each benchmark.
                for job in run.jobs().iter() {
                    const MAX_METHODS: usize = 3;
                    if let Err(mismatches) = benchmarks.debug(job, MAX_METHODS) {
                        let repr = serde_json::to_string_pretty(&job.serialize()?)?;

                        writeln!(
                            output,
                            "Could not find a match for the following input:\n\n{}\n",
                            repr
                        )?;
                        writeln!(output, "Closest matches:\n")?;
                        for (i, mismatch) in mismatches.into_iter().enumerate() {
                            writeln!(
                                output,
                                "    {}. \"{}\": {}",
                                i + 1,
                                mismatch.method(),
                                mismatch.reason(),
                            )?;
                        }
                        writeln!(output)?;

                        return Err(anyhow::Error::msg(
                            "could not find a benchmark for all inputs",
                        ));
                    }
                }

                if *dry_run {
                    writeln!(
                        output,
                        "Success - skipping running benchmarks because \"--dry-run\" was used."
                    )?;
                    return Ok(());
                }

                // Check for debug mode before running benchmarks.
                // This check is placed after the dry-run early return since dry-run doesn't
                // actually execute benchmarks and thus won't produce misleading performance results.
                check_debug_mode(*allow_debug)?;

                // The collection of output results for each run.
                let mut results = Vec::<serde_json::Value>::new();

                // Now - we've verified the integrity of all the jobs we want to run and that
                // each job can match an associated benchmark.
                //
                // All that's left is to actually run the benchmarks.
                let jobs = run.jobs();
                let serialized = jobs
                    .iter()
                    .map(|job| {
                        serde_json::to_value(jobs::Unprocessed::new(
                            job.tag().into(),
                            job.serialize()?,
                        ))
                    })
                    .collect::<Result<Vec<_>, serde_json::Error>>()?;
                for (i, job) in jobs.iter().enumerate() {
                    let prefix: &str = if i != 0 { "\n\n" } else { "" };
                    writeln!(
                        output,
                        "{}{}",
                        prefix,
                        Banner::new(&format!("Running Job {} of {}", i + 1, jobs.len()))
                    )?;

                    // Run the specified job.
                    let checkpoint = Checkpoint::new(&serialized, &results, output_file)?;
                    let r = benchmarks.call(job, checkpoint, output)?;

                    // Collect the results
                    results.push(r);

                    // Save everything.
                    Checkpoint::new(&serialized, &results, output_file)?.save()?;
                }
            }
            // Extensions
            Commands::Check(check) => return self.check(check, inputs, benchmarks, output),
        };
        Ok(())
    }

    // Extensions
    fn check(
        &self,
        check: &Check,
        inputs: &registry::Inputs,
        benchmarks: &registry::Benchmarks,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<()> {
        match check {
            Check::Skeleton => {
                let message = "Skeleton tolerance file.\n\n\
                               Each tolerance is paired with an input that is structurally\n\
                               matched with an entry in the corresponding `--input-file`.\n\n\
                               This allow a single tolerance entry to be applied to multiple\n\
                               benchmark runs as long as this structural mapping is unambiguous.\n";

                writeln!(output, "{}", message)?;
                writeln!(output, "{}", internal::regression::Raw::example())?;
                Ok(())
            }
            Check::Tolerances { describe } => {
                let tolerances = benchmarks.tolerances();

                match describe {
                    Some(name) => match tolerances.get(&**name) {
                        Some(registered) => {
                            let repr = internal::regression::RawInner::new(
                                jobs::Unprocessed::new(
                                    "".to_string(),
                                    serde_json::Value::Object(Default::default()),
                                ),
                                jobs::Unprocessed::format_input(registered.tolerance)?,
                            );

                            write!(
                                output,
                                "The example JSON representation for \"{}\" is shown below.\n\
                                 Populate the \"input\" field with a compatible benchmark input.\n\
                                 Matching will be performed by partial structural map on the input.\n\n",
                                name
                            )?;
                            writeln!(output, "{}", serde_json::to_string_pretty(&repr)?)?;
                            Ok(())
                        }
                        None => {
                            writeln!(output, "No tolerance input found for \"{}\"", name)?;
                            Ok(())
                        }
                    },
                    None => {
                        writeln!(output, "Available tolerance kinds are listed below.")?;

                        // Print the registered tolerance files in alphabetical order.
                        let mut keys: Vec<_> = tolerances.keys().collect();
                        keys.sort();
                        for k in keys {
                            // This access should not panic - we just obtained all the keys.
                            let registered = &tolerances[k];
                            writeln!(output, "    {}", registered.tolerance.tag())?;
                            for pair in registered.regressions.iter() {
                                writeln!(
                                    output,
                                    "    - \"{}\" => \"{}\"",
                                    pair.input_tag(),
                                    pair.name(),
                                )?;
                            }
                        }
                        Ok(())
                    }
                }
            }
            Check::Verify {
                tolerances,
                input_file,
            } => {
                // For verification - we merely check that we can successfully construct
                // the regression `Checks` struct. It performs all the necessary preflight
                // checks.
                let benchmarks = benchmarks.tolerances();
                let _ =
                    internal::regression::Checks::new(tolerances, input_file, inputs, &benchmarks)?;
                Ok(())
            }
            Check::Run {
                tolerances,
                input_file,
                before,
                after,
                output_file,
            } => {
                let registered = benchmarks.tolerances();
                let checks =
                    internal::regression::Checks::new(tolerances, input_file, inputs, &registered)?;
                let jobs = checks.jobs(before, after)?;
                jobs.run(output, output_file.as_deref())?;
                Ok(())
            }
        }
    }
}

///////////
// Tests //
///////////

/// The integration test below look inside the `tests` directory for folders.
///
/// ## Input Files
///
/// Each folder should have at least a `stdin.txt` file specifying the command line to give
/// to the `App` parser.
///
/// Within the `stdin.txt` command line, there are several special symbols:
///
/// * $INPUT - Resolves to `input.json` in the same directory as the `stdin.txt` file.
/// * $OUTPUT - Resolves to `output.json` in a temporary directory.
/// * $TOLERANCES - Resolves to `tolerances.json` in the test directory.
/// * $REGRESSION_INPUT - Resolves to `regression_input.json` in the test directory.
/// * $CHECK_OUTPUT - Resolves to `checks.json` in a temporary directory.
///
/// As mentioned - an input JSON file can be included and must be named "input.json" to be
/// discoverable.
///
/// ## Output Files
///
/// Tests should have at least a `stdout.txt` file with the expected outputs for running the
/// command in `stdin.txt`. If an output JSON file is expected, it should be named `output.json`.
///
/// ## Test Discovery and Running
///
/// The unit test will visit each folder in `tests` and run the outlined scenario. The
/// `stdout.txt` expected output is compared to the actual output and if they do not match,
/// the test fails.
///
/// Additionally, if `output.json` is present, the unit test will verify that (1) the command
/// did in fact produce an output JSON file and (2) the generated file matches the expected file.
///
/// ## Regenerating Expected Results
///
/// The benchmark output will naturally change over time. Running the unit tests with the
/// environment variable
/// ```text
/// POCKETBENCH_TEST=overwrite
/// ```
/// will replace the `stdout.txt` (and `output.json` if one was generated) for each test
/// scenario. Developers should then consult `git diff` to ensure that major regressions
/// to the output did not occur.
#[cfg(test)]
mod tests {
    use super::*;

    use std::{
        ffi::OsString,
        path::{Path, PathBuf},
    };

    use crate::{registry, ux};

    const ENV: &str = "POCKETBENCH_TEST";

    // Expected I/O files.
    const STDIN: &str = "stdin.txt";
    const STDOUT: &str = "stdout.txt";
    const INPUT_FILE: &str = "input.json";
    const OUTPUT_FILE: &str = "output.json";

    // Regression Extension
    const TOLERANCES_FILE: &str = "tolerances.json";
    const REGRESSION_INPUT_FILE: &str = "regression_input.json";
    const CHECK_OUTPUT_FILE: &str = "checks.json";

    const ALL_GENERATED_OUTPUTS: [&str; 2] = [OUTPUT_FILE, CHECK_OUTPUT_FILE];

    // Read the entire contents of a file to a string.
    fn read_to_string<P: AsRef<Path>>(path: P, ctx: &str) -> String {
        match std::fs::read_to_string(path.as_ref()) {
            Ok(s) => ux::normalize(s),
            Err(err) => panic!(
                "failed to read {} {:?} with error: {}",
                ctx,
                path.as_ref(),
                err
            ),
        }
    }

    // Check if `POCKETBENCH_TEST=overwrite` is configured. Return `true` if so - otherwise
    // return `false`.
    //
    // If `POCKETBENCH_TEST` is set but its value is not `overwrite` - panic.
    fn overwrite() -> bool {
        match std::env::var(ENV) {
            Ok(v) => {
                if v == "overwrite" {
                    true
                } else {
                    panic!(
                        "Unknown value for {}: \"{}\". Expected \"overwrite\"",
                        ENV, v
                    );
                }
            }
            Err(std::env::VarError::NotPresent) => false,
            Err(std::env::VarError::NotUnicode(_)) => {
                panic!("Value for {} is not unicode", ENV);
            }
        }
    }

    // Test Runner
    struct Test {
        dir: PathBuf,
        overwrite: bool,
    }

    impl Test {
        fn new(dir: &Path) -> Self {
            Self {
                dir: dir.into(),
                overwrite: overwrite(),
            }
        }

        fn parse_stdin(&self, tempdir: &Path) -> Vec<App> {
            let path = self.dir.join(STDIN);

            // Read the standard input file to a string.
            let stdin = read_to_string(&path, "standard input");

            let output: Vec<App> = stdin
                .lines()
                .filter_map(|line| {
                    if line.starts_with('#') || line.is_empty() {
                        None
                    } else {
                        Some(self.parse_line(line, tempdir))
                    }
                })
                .collect();

            if output.is_empty() {
                panic!("File \"{}/stdin.txt\" has no command!", self.dir.display());
            }

            output
        }

        fn parse_line(&self, line: &str, tempdir: &Path) -> App {
            // Split and resolve special symbols
            let args: Vec<OsString> = line
                .split_whitespace()
                .map(|v| -> OsString { self.resolve(v, tempdir).into() })
                .collect();

            App::try_parse_from(std::iter::once(OsString::from("test-app")).chain(args)).unwrap()
        }

        fn resolve(&self, s: &str, tempdir: &Path) -> PathBuf {
            match s {
                // Standard workflow
                "$INPUT" => self.dir.join(INPUT_FILE),
                "$OUTPUT" => tempdir.join(OUTPUT_FILE),
                // Regression extension
                "$TOLERANCES" => self.dir.join(TOLERANCES_FILE),
                "$REGRESSION_INPUT" => self.dir.join(REGRESSION_INPUT_FILE),
                "$CHECK_OUTPUT" => tempdir.join(CHECK_OUTPUT_FILE),

                // Catch-all: no interpolation
                _ => s.into(),
            }
        }

        fn run(&self, tempdir: &Path) {
            let apps = self.parse_stdin(tempdir);

            // Register inputs
            let mut inputs = registry::Inputs::new();
            crate::test::register_inputs(&mut inputs).unwrap();

            // Register outputs
            let mut benchmarks = registry::Benchmarks::new();
            crate::test::register_benchmarks(&mut benchmarks);

            // Run each app invocation - collecting the last output into a buffer.
            //
            // Only the last run is allowed to return an error - if it does, format the
            // error to the output buffer as well using the debug formatting option.
            let mut buffer = crate::output::Memory::new();
            for (i, app) in apps.iter().enumerate() {
                let is_last = i + 1 == apps.len();

                // Select where to route the test output.
                //
                // Only the last run gets saved. Setup output is discarded — if a setup
                // command fails, the panic message includes the error.
                let mut b: &mut dyn crate::Output = if is_last {
                    &mut buffer
                } else {
                    &mut crate::output::Sink::new()
                };

                if let Err(err) = app.run(&inputs, &benchmarks, b) {
                    if is_last {
                        write!(b, "{:?}", err).unwrap();
                    } else {
                        panic!(
                            "App {} of {} failed with error: {:?}",
                            i + 1,
                            apps.len(),
                            err
                        );
                    }
                }
            }

            // Check that `stdout` matches
            let stdout: String =
                ux::normalize(ux::strip_backtrace(buffer.into_inner().try_into().unwrap()));
            let stdout = ux::scrub_path(stdout, tempdir, "$TEMPDIR");
            let output = self.dir.join(STDOUT);
            if self.overwrite {
                std::fs::write(output, stdout).unwrap();
            } else {
                let expected = read_to_string(&output, "expected standard output");
                if stdout != expected {
                    panic!("Got:\n--\n{}\n--\nExpected:\n--\n{}\n--", stdout, expected);
                }
            }

            // Check that the output files match.
            for file in ALL_GENERATED_OUTPUTS {
                self.check_output_file(tempdir, file);
            }
        }

        fn check_output_file(&self, tempdir: &Path, filename: &str) {
            let generated_path = tempdir.join(filename);
            let was_generated = generated_path.is_file();

            let expected_path = self.dir.join(filename);
            let is_expected = expected_path.is_file();

            if self.overwrite {
                // Copy the output file to the destination.
                if was_generated {
                    println!(
                        "Moving generated file {:?} to {:?}",
                        generated_path, expected_path
                    );

                    if let Err(err) = std::fs::rename(&generated_path, &expected_path) {
                        panic!(
                            "Moving generated file {:?} to expected location {:?} failed: {}",
                            generated_path, expected_path, err
                        );
                    }
                } else if is_expected {
                    println!("Removing outdated file {:?}", expected_path);
                    if let Err(err) = std::fs::remove_file(&expected_path) {
                        panic!("Failed removing outdated file {:?}: {}", expected_path, err);
                    }
                }
            } else {
                match (was_generated, is_expected) {
                    (true, true) => {
                        let output_contents = read_to_string(generated_path, "generated");

                        let expected_contents = read_to_string(expected_path, "expected");

                        if output_contents != expected_contents {
                            panic!(
                                "{}: Got:\n\n{}\n\nExpected:\n\n{}\n",
                                filename, output_contents, expected_contents
                            );
                        }
                    }
                    (true, false) => {
                        let output_contents = read_to_string(generated_path, "generated");

                        panic!(
                            "{} was generated when none was expected. Contents:\n\n{}",
                            filename, output_contents
                        );
                    }
                    (false, true) => {
                        panic!("{} was not generated when it was expected", filename);
                    }
                    (false, false) => { /* this is okay */ }
                }
            }
        }
    }

    fn run_specific_test(test_dir: &Path) {
        println!("running test in {:?}", test_dir);
        let temp_dir = tempfile::tempdir().unwrap();
        Test::new(test_dir).run(temp_dir.path());
    }

    fn run_all_tests_in(dir: &str) {
        let dir: PathBuf = format!("{}/tests/{}", env!("CARGO_MANIFEST_DIR"), dir).into();
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            if let Ok(file_type) = entry.file_type() {
                if file_type.is_dir() {
                    run_specific_test(&entry.path());
                }
            } else {
                panic!("couldn't get file type for {:?}", entry.path());
            }
        }
    }

    #[test]
    fn benchmark_tests() {
        run_all_tests_in("benchmark");
    }

    #[test]
    fn regression_tests() {
        run_all_tests_in("regression");
    }
}
