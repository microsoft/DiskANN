/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::path::{Path, PathBuf};

use relative_path::RelativePath;
use serde::Serialize;

/// Return previously generated results generated for `test_name`.
///
/// Test results will be cached in `tests/generated/<$test_name>` as serialized JSON files
/// where `test_name` can contain directory separators for better result organization.
///
/// Panics if no such file can be found unless the environment variable
/// `DISKANN_TEST=overwrite` is set. In this case, the test results is serialized and
/// `resuls` is returned directly.
///
/// # Generating Results
///
/// To generate or regenerate test results, run the test suite for `diskann` with
/// the environment variable `DISKANN_TEST=overwrite` set. This will overwrite the results
/// for all generated tests.
///
/// The suggested workflow is to full delete `diskann/test/generated` when regenerating
/// test data and using `git diff` to ensure unwanted changes are not present.
///
/// This ensures that unused test results no longer clutter the repository during test
/// refactoring.
#[track_caller]
pub(crate) fn get_or_save_test_results<R>(test_name: &str, results: &R) -> R
where
    R: serde::Serialize + for<'a> serde::Deserialize<'a> + Clone,
{
    get_or_save_test_results_with_format(test_name, results, TestFormat::Pretty)
}

/// Whitespace used when regenerating baselines; reading is independent of the format.
#[derive(Debug, Clone, Copy)]
pub(crate) enum TestFormat {
    Pretty,
    InlineNumericPairs,
}

/// Like [`get_or_save_test_results`], with opt-in formatting for regenerated files.
#[track_caller]
pub(crate) fn get_or_save_test_results_with_format<R>(
    test_name: &str,
    results: &R,
    format: TestFormat,
) -> R
where
    R: serde::Serialize + for<'a> serde::Deserialize<'a> + Clone,
{
    match _get_or_save_test_results(
        test_name,
        serde_json::to_value(results).expect("while serializing results"),
        format,
    ) {
        Some(output) => match R::deserialize(output) {
            Ok(deserialized) => deserialized,
            Err(err) => {
                panic!(
                    "Error encountered while deserializing cached test result: {}. \
                     If the data structure representation changed {}",
                    err,
                    env_hint()
                );
            }
        },
        None => results.clone(),
    }
}

/// A root path for which test results will be stored.
///
/// This can be efficiently and temporarily added by using [`Self::path]` and
/// [`TestPath::push`].
#[derive(Debug)]
pub(crate) struct TestRoot {
    path: String,
}

impl TestRoot {
    /// Create a new [`TestRoot`] from `path`.
    pub(crate) fn new<P: Into<String>>(path: P) -> Self {
        Self { path: path.into() }
    }

    /// Borrow `self`, creating a [`TestPath`].
    pub(crate) fn path(&mut self) -> TestPath<'_> {
        TestPath::new(&mut self.path)
    }
}

/// An efficiently and temporarily growable path allowing full test paths to be build
/// incrementally.
#[derive(Debug)]
pub(crate) struct TestPath<'a> {
    path: &'a mut String,
    len: usize,
}

impl<'a> TestPath<'a> {
    /// Create a new [`TestPath`] with `path` as the root.
    ///
    /// When the returned instance of [`Self`] is destroyed, `path` should remain unchanged,
    /// though its capacity may be different.
    pub(crate) fn new(path: &'a mut String) -> Self {
        let len = path.len();
        Self { path, len }
    }
}

impl TestPath<'_> {
    /// Append `name` to `self` with a `/` separator.
    ///
    /// When the returned `TestPath` is destroyed, `self` will be left in its original state.
    pub(crate) fn push<P: AsRef<str>>(&mut self, name: P) -> TestPath<'_> {
        self._push(name.as_ref())
    }

    pub(crate) fn _push(&mut self, name: &str) -> TestPath<'_> {
        let len = self.path.len();
        self.path.push('/');
        self.path.push_str(name.as_ref());
        TestPath {
            path: self.path,
            len,
        }
    }
}

impl Drop for TestPath<'_> {
    fn drop(&mut self) {
        self.path.truncate(self.len)
    }
}

impl std::ops::Deref for TestPath<'_> {
    type Target = str;
    fn deref(&self) -> &Self::Target {
        self.path
    }
}

impl AsRef<str> for TestPath<'_> {
    fn as_ref(&self) -> &str {
        self.path
    }
}

impl AsRef<std::path::Path> for TestPath<'_> {
    fn as_ref(&self) -> &std::path::Path {
        self.path.as_ref()
    }
}

//////////
// Impl //
//////////

// The build directory of the crate.
const BUILD_DIR: &str = env!("CARGO_MANIFEST_DIR");
const CONFIGURATION: &str = "DISKANN_TEST";

fn env_hint() -> &'static str {
    "try running with the environment variable \"DISKANN_TEST=overwrite\""
}

#[derive(Debug)]
enum Mode {
    // We're in testing mode - meaning that we are simply deserializing saved results and
    // returning them to the test harness.
    Test,

    // We're generating reference test output. Instead of returning deserialized results,
    // we serialize the current results.
    Overwrite,
}

impl Mode {
    /// Return `Self::Overwrite` if `DISKANN_TEST=overwrite` is configured. Otherwise,
    /// return `Self::Test`.
    fn current() -> Self {
        match std::env::var(CONFIGURATION) {
            Ok(v) => {
                if v == "overwrite" {
                    Self::Overwrite
                } else {
                    panic!(
                        "Unknown value for {}: \"{}\". Expected \"overwrite\"",
                        CONFIGURATION, v
                    );
                }
            }
            Err(std::env::VarError::NotPresent) => Self::Test,
            Err(std::env::VarError::NotUnicode(_)) => {
                panic!("Value for {} is not valid unicode", CONFIGURATION)
            }
        }
    }
}

fn result_path(relative: &RelativePath) -> PathBuf {
    let build_dir: &Path = BUILD_DIR.as_ref();
    relative.to_path(build_dir.join("test").join("generated"))
}

/// A thin wrapper for serialized payloads that adds a little more metadata to test results.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct TestResult<T = serde_json::Value> {
    file: String,
    test: String,
    payload: T,
}

impl TestResult {
    fn write(self, writer: impl std::io::Write, format: TestFormat) -> serde_json::Result<()> {
        match format {
            TestFormat::Pretty => serde_json::to_writer_pretty(writer, &self),
            TestFormat::InlineNumericPairs => {
                let result = TestResult {
                    file: self.file,
                    test: self.test,
                    payload: InlineNumericPairs(&self.payload),
                };
                serde_json::to_writer_pretty(writer, &result)
            }
        }
    }
}

/// Keep two-number arrays inline without changing string contents or other containers.
struct InlineNumericPairs<'a>(&'a serde_json::Value);

impl Serialize for InlineNumericPairs<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            serde_json::Value::Array(values)
                if values.len() == 2 && values.iter().all(serde_json::Value::is_number) =>
            {
                let raw = serde_json::value::RawValue::from_string(format!(
                    "[{}, {}]",
                    values[0], values[1]
                ))
                .map_err(serde::ser::Error::custom)?;
                raw.serialize(serializer)
            }
            serde_json::Value::Array(values) => {
                serializer.collect_seq(values.iter().map(InlineNumericPairs))
            }
            serde_json::Value::Object(values) => serializer.collect_map(
                values
                    .iter()
                    .map(|(key, value)| (key, InlineNumericPairs(value))),
            ),
            value => value.serialize(serializer),
        }
    }
}

#[track_caller]
fn _get_or_save_test_results(
    test: &str,
    value: serde_json::Value,
    format: TestFormat,
) -> Option<serde_json::Value> {
    // Get the location of the caller to allow us to construct a full path.
    let caller = std::panic::Location::caller();

    let result_path = {
        let mut result_path = result_path(RelativePath::from_path(test).unwrap());
        result_path.set_extension("json");
        result_path
    };

    match Mode::current() {
        Mode::Test => {
            let file = match std::fs::File::open(&result_path) {
                Ok(file) => file,
                Err(err) => panic!(
                    "Could not open {}: {}. If you added a new test {}",
                    result_path.display(),
                    err,
                    env_hint(),
                ),
            };

            let reader = std::io::BufReader::new(file);
            let result: TestResult =
                serde_json::from_reader(reader).expect("deserialization should succeed");
            Some(result.payload)
        }
        Mode::Overwrite => {
            // Compute the directory that will enclose the test results.
            //
            // Since `result_path` should be a sub directory of `CARGO_MANIFEST_DIR`, we
            // expect `parent()` to return sensible results.
            let result_dir = result_path
                .parent()
                .expect("Result path should have a parent directory");

            // Create all directories.
            if let Err(err) = std::fs::create_dir_all(result_dir) {
                panic!(
                    "Problem creating paths \"{}\": {}",
                    result_path.display(),
                    err
                );
            }

            // Create the result file.
            let buffer = match std::fs::File::create(&result_path) {
                Ok(buffer) => buffer,
                Err(err) => {
                    panic!(
                        "Error opening path {} for writing: {}",
                        result_path.display(),
                        err
                    );
                }
            };

            // Serialize the test results.
            let result = TestResult {
                file: normalize_file(caller.file()),
                test: normalize_file(test),
                payload: value,
            };
            result
                .write(buffer, format)
                .expect("serialization should succeed");
            None
        }
    }
}

// Normalize file path to use "/".
//
// This prevents churn in the serialized file paths between Windows and Unix.
fn normalize_file(s: &str) -> String {
    s.replace("\\", "/")
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> TestResult {
        TestResult {
            file: "source.rs".into(),
            test: "formatting".into(),
            payload: serde_json::json!({
                "pairs": [[7, 2.5], [u64::MAX, i64::MIN]],
                "nested": {"pair": [1.25, -0.0]},
                "empty": [],
                "single": [1],
                "long": [1, 2, 3],
                "mixed": [1, "2"],
                "strings": ["[1, 2]", "\"[3,4]\""],
            }),
        }
    }

    #[test]
    fn pretty_format_is_unchanged() {
        let result = record();
        let expected = serde_json::to_vec_pretty(&result).unwrap();
        let mut output = Vec::new();
        result.write(&mut output, TestFormat::Pretty).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn inline_numeric_pairs_preserves_nested_values() {
        let result = record();
        let expected = serde_json::to_value(&result).unwrap();
        let mut output = Vec::new();
        result
            .write(&mut output, TestFormat::InlineNumericPairs)
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&output).unwrap(),
            expected
        );

        let text = std::str::from_utf8(&output).unwrap();
        for expected in [
            "[7, 2.5]",
            "[18446744073709551615, -9223372036854775808]",
            "[1.25, -0.0]",
            "\"long\": [\n",
            "\"mixed\": [\n",
            "\"strings\": [\n",
        ] {
            assert!(text.contains(expected), "missing {expected:?} in {text}");
        }
    }

    #[test]
    fn test_path() {
        let mut x = TestRoot::new("some/root");
        let mut y = x.path();

        assert_eq!(&*y, "some/root");
        {
            let mut z = y.push("another");
            assert_eq!(&*z, "some/root/another");

            {
                let w = z.push("one-more");
                assert_eq!(&*w, "some/root/another/one-more");
                std::mem::drop(w);

                let w = z.push("again");
                assert_eq!(&*w, "some/root/another/again");
            }

            assert_eq!(&*z, "some/root/another");
        }

        assert_eq!(&*y, "some/root");
    }
}
