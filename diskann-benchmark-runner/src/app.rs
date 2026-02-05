/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{io::Write, path::PathBuf};

use clap::{Parser, Subcommand};

use crate::{
    jobs::{self, Jobs},
    output::Output,
    registry,
    result::Checkpoint,
    utils::fmt::Banner,
};

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
    },
}

/// The CLI used to drive a benchmark application.
#[derive(Debug, Parser)]
pub struct App {
    #[command(subcommand)]
    command: Commands,
}

impl App {
    /// Construct [`Self`] by parsing commandline arguments from [`std::env::args]`.
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

    /// Run the application using the registered `inputs` and `outputs`.
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
                for (name, method) in benchmarks.methods() {
                    writeln!(output, "    {}: {}", name, method.signatures()[0])?;
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
            } => {
                // Parse and validate the input.
                let run = Jobs::load(input_file, inputs)?;
                // Check if we have a match for each benchmark.
                for job in run.jobs().iter() {
                    if !benchmarks.has_match(job) {
                        let repr = serde_json::to_string_pretty(&job.serialize()?)?;

                        const MAX_METHODS: usize = 3;
                        let mismatches = match benchmarks.debug(job, MAX_METHODS) {
                            // Debug should return `Err` if there is not a match.
                            // Returning `Ok(())` here indicates an internal error with the
                            // dispatcher.
                            Ok(()) => {
                                return Err(anyhow::Error::msg(format!(
                                    "experienced internal error while debugging:\n{}",
                                    repr
                                )))
                            }
                            Err(m) => m,
                        };

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
                            "could not find find a benchmark for all inputs",
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
        };
        Ok(())
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
///
/// As mentioned - an input JSON file can be included and must be named "input.json" to be
/// discoverable.
///
/// ## Output Files
///
/// Tests should have at least a `stdout.txt` file with the expected outputs for running the
/// command in `stdin.txt`. If an output JSON file is expected, it should be name `output.json`.
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

        fn parse_stdin(&self, tempdir: &Path) -> App {
            let path = self.dir.join(STDIN);

            // Read the standard input file to a string.
            let stdin = read_to_string(&path, "standard input");

            let args: Vec<OsString> = stdin
                .split_whitespace()
                .map(|v| -> OsString { self.resolve(v, tempdir).into() })
                .collect();

            // Split and resolve special symbols
            App::try_parse_from(std::iter::once(OsString::from("test-app")).chain(args)).unwrap()
        }

        fn resolve(&self, s: &str, tempdir: &Path) -> PathBuf {
            if s == "$INPUT" {
                self.dir.join(INPUT_FILE)
            } else if s == "$OUTPUT" {
                tempdir.join(OUTPUT_FILE)
            } else {
                s.into()
            }
        }

        fn run(&self, tempdir: &Path) {
            let app = self.parse_stdin(tempdir);

            // Register inputs
            let mut inputs = registry::Inputs::new();
            crate::test::register_inputs(&mut inputs).unwrap();

            // Register outputs
            let mut benchmarks = registry::Benchmarks::new();
            crate::test::register_benchmarks(&mut benchmarks);

            // Run app - collecting output into a buffer.
            //
            // If the app returns an error - format the error to the output buffer as well
            // using the debug formatting option.
            let mut buffer = crate::output::Memory::new();
            if let Err(err) = app.run(&inputs, &benchmarks, &mut buffer) {
                let mut b: &mut dyn crate::Output = &mut buffer;
                write!(b, "{:?}", err).unwrap();
            }

            // Check that `stdout` matches
            let stdout: String =
                ux::normalize(ux::strip_backtrace(buffer.into_inner().try_into().unwrap()));
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
            let output_path = tempdir.join(OUTPUT_FILE);
            let was_output_generated = output_path.is_file();

            let expected_output_path = self.dir.join(OUTPUT_FILE);
            let is_output_expected = expected_output_path.is_file();

            if self.overwrite {
                // Copy the output file to the destination.
                if was_output_generated {
                    println!(
                        "Moving generated output file {:?} to {:?}",
                        output_path, expected_output_path
                    );

                    if let Err(err) = std::fs::rename(&output_path, &expected_output_path) {
                        panic!(
                            "Moving generated output file {:?} to expected location {:?} failed: {}",
                            output_path, expected_output_path, err
                        );
                    }
                } else if is_output_expected {
                    println!("Removing outdated output file {:?}", expected_output_path);
                    if let Err(err) = std::fs::remove_file(&expected_output_path) {
                        panic!(
                            "Failed removing outdated output file {:?}: {}",
                            expected_output_path, err
                        );
                    }
                }
            } else {
                match (was_output_generated, is_output_expected) {
                    (true, true) => {
                        let output_contents = read_to_string(output_path, "generated output JSON");

                        let expected_contents =
                            read_to_string(expected_output_path, "expected output JSON");

                        if output_contents != expected_contents {
                            panic!(
                                "Got:\n\n{}\n\nExpected:\n\n{}\n",
                                output_contents, expected_contents
                            );
                        }
                    }
                    (true, false) => {
                        let output_contents = read_to_string(output_path, "generated output JSON");

                        panic!(
                            "An output JSON was generated when none was expected. Contents:\n\n{}",
                            output_contents
                        );
                    }
                    (false, true) => {
                        panic!("No output JSON was generated when one was expected");
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
    fn top_level_tests() {
        run_all_tests_in("");
    }
}
