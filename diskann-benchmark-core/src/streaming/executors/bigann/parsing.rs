/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde_yaml::{Mapping, Value};

/// See [`super::RunBook::load`] for documentation of the file format parsed
/// by this function.
pub(super) fn load(
    path: &Path,
    dataset: &str,
    groundtruth: &mut dyn super::FindGroundtruth,
) -> anyhow::Result<super::RunBook> {
    load_mapping(parse_yaml(path)?, path, dataset, groundtruth)
}

fn load_mapping(
    mapping: Mapping,
    path: &Path,
    dataset: &str,
    groundtruth: &mut dyn super::FindGroundtruth,
) -> anyhow::Result<super::RunBook> {
    // Multiple datasets can exist in the same YAML file.
    //
    // Fortunately, all datasets are keyed by their dataset name which allows us to
    // quickly find whether or not it exists.
    let dataset_value = match mapping.get(dataset) {
        Some(value) => value,
        None => return Err(DumpKeys::new(mapping, dataset, path).into()),
    };

    // Try to coerce the dataset value into a `Mapping`.
    let dataset_mapping: &Mapping = match dataset_value.try_as() {
        Ok(mapping) => mapping,
        Err(_) => anyhow::bail!(
            "dataset \"{}\" exists in file \"{}\", but its associated payload is not a YAML map",
            dataset,
            path.display(),
        ),
    };

    let mut raw = parse_stages(dataset_mapping).with_context(|| {
        format!(
            "parsing dataset \"{}\" in file \"{}\"",
            dataset,
            path.display()
        )
    })?;
    raw.stages.sort_by_key(|s| s.index);

    // Translate from raw ranges into higher level steps.
    let context = |index: usize| {
        format!(
            "precessing stage {} of dataset \"{}\" in file \"{}\"",
            index,
            dataset,
            path.display()
        )
    };

    let stages: anyhow::Result<Vec<super::Stage>> = raw
        .stages
        .iter()
        .map(|stage| {
            let stage = match &stage.operation {
                Operation::Search => super::Stage::Search {
                    groundtruth: groundtruth
                        .find_groundtruth(stage.index)
                        .with_context(|| context(stage.index))?,
                },
                Operation::Insert(insert) => super::Stage::Insert {
                    dataset_offsets_and_ids: insert.start..insert.end,
                },
                Operation::Replace(replace) => super::Stage::Replace {
                    dataset_offsets: replace.ids_start..replace.ids_end,
                    ids: replace.tags_start..replace.tags_end,
                },
                Operation::Delete(delete) => super::Stage::Delete {
                    ids: delete.start..delete.end,
                },
            };
            Ok(stage)
        })
        .collect();

    super::RunBook::new(stages?, raw.max_points)
}

fn parse_yaml(path: &Path) -> anyhow::Result<Mapping> {
    let f = std::fs::File::open(path)
        .with_context(|| format!("while opening file \"{}\"", path.display()))?;

    Ok(serde_yaml::from_reader(std::io::BufReader::new(f))?)
}

fn parse_stages(mapping: &Mapping) -> anyhow::Result<Raw> {
    let mut stages = Vec::<Stage>::new();
    let mut max_points = None;
    mapping.iter().try_for_each(|(key, value)| match key {
        Value::String(s) => match s.as_str() {
            "max_pts" => {
                let points: usize = value
                    .try_as()
                    .map_err(|kind| anyhow::anyhow!("failed to parse \"max_pts\" as a {}", kind))?;

                max_points = Some(points);
                Ok(())
            }
            "gt_url" => Ok(()),
            _ => anyhow::bail!("Unrecognized runbook key: \"{}\"", s),
        },
        Value::Number(stage) => match stage.as_i64() {
            Some(stage) => {
                stages.push(
                    handle_stage(stage as usize, value)
                        .with_context(|| format!("processing stage {}", stage))?,
                );
                Ok(())
            }
            None => anyhow::bail!("Stage \"{}\" must be an integer", stage),
        },
        _ => anyhow::bail!("Unrecognized key of type {}", classify(key),),
    })?;

    let max_points = match max_points {
        Some(points) => points,
        None => anyhow::bail!("key \"max_pts\" not found"),
    };

    Ok(Raw { max_points, stages })
}

fn handle_stage(index: usize, value: &Value) -> anyhow::Result<Stage> {
    let mapping: &Mapping = value
        .try_as()
        .map_err(|_| anyhow::anyhow!("YAML type is not a map"))?;

    let kind: &str = mapping.get_as("operation")?;
    let operation = match Kind::try_parse(kind)? {
        Kind::Search => Operation::Search,
        Kind::Insert => Operation::Insert(mapping.try_into()?),
        Kind::Replace => Operation::Replace(mapping.try_into()?),
        Kind::Delete => Operation::Delete(mapping.try_into()?),
    };

    Ok(Stage { index, operation })
}

struct Raw {
    max_points: usize,
    stages: Vec<Stage>,
}

struct Stage {
    index: usize,
    operation: Operation,
}

enum Operation {
    Search,
    Insert(Insert),
    Replace(Replace),
    Delete(Delete),
}

#[derive(Debug)]
struct DumpKeys {
    mapping: Mapping,
    dataset: String,
    file: PathBuf,
}

impl DumpKeys {
    #[inline(never)]
    fn new(mapping: Mapping, dataset: &str, file: &Path) -> Self {
        Self {
            mapping,
            dataset: dataset.to_string(),
            file: file.into(),
        }
    }
}

impl std::fmt::Display for DumpKeys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "dataset \"{}\" not found in file \"{}\" - possible alternatives: [",
            self.dataset,
            self.file.display(),
        )?;

        let len = self.mapping.len();
        self.mapping.keys().enumerate().try_for_each(|(i, key)| {
            let mut write = |key: &dyn std::fmt::Display| {
                if i + 1 == len {
                    write!(f, "{}", key)
                } else {
                    write!(f, "{}, ", key)
                }
            };

            match key {
                Value::Null => write(&"null"),
                Value::Bool(b) => write(b),
                Value::Number(number) => write(number),
                Value::String(s) => write(s),
                Value::Sequence(_) => write(&"<sequence>"),
                Value::Mapping(_) => write(&"<mapping>"),
                Value::Tagged(_) => write(&"<tagged>"),
            }
        })?;
        write!(f, "]")
    }
}

impl std::error::Error for DumpKeys {}

trait TryAs<'a, T> {
    fn try_as(&'a self) -> Result<T, &'static str>;
}

impl<'a> TryAs<'a, usize> for Value {
    fn try_as(&'a self) -> Result<usize, &'static str> {
        self.as_i64().map(|i| i as usize).ok_or("usize")
    }
}

impl<'a> TryAs<'a, &'a str> for Value {
    fn try_as(&'a self) -> Result<&'a str, &'static str> {
        self.as_str().ok_or("string")
    }
}

impl<'a> TryAs<'a, &'a Mapping> for Value {
    fn try_as(&'a self) -> Result<&'a Mapping, &'static str> {
        self.as_mapping().ok_or("map")
    }
}

trait MappingExt {
    fn get_as<'a, T>(&'a self, index: &str) -> anyhow::Result<T>
    where
        Value: TryAs<'a, T>;
}

impl MappingExt for Mapping {
    fn get_as<'a, T>(&'a self, key: &str) -> anyhow::Result<T>
    where
        Value: TryAs<'a, T>,
    {
        match self.get(key) {
            Some(value) => match value.try_as() {
                Ok(v) => Ok(v),
                Err(expected) => Err(anyhow::anyhow!(
                    "key \"{}\" exists but it is not a {}",
                    key,
                    expected,
                )),
            },
            None => Err(anyhow::anyhow!("key \"{}\" not found", key)),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Kind {
    Search,
    Insert,
    Replace,
    Delete,
}

impl Kind {
    fn try_parse(string: &str) -> anyhow::Result<Self> {
        match string {
            "search" => Ok(Kind::Search),
            "insert" => Ok(Kind::Insert),
            "replace" => Ok(Kind::Replace),
            "delete" => Ok(Kind::Delete),
            _ => Err(anyhow::anyhow!("unrecognized operation: {}", string)),
        }
    }
}

#[derive(Debug)]
struct Replace {
    ids_start: usize,
    ids_end: usize,
    tags_start: usize,
    tags_end: usize,
}

impl TryFrom<&Mapping> for Replace {
    type Error = anyhow::Error;
    fn try_from(mapping: &Mapping) -> anyhow::Result<Self> {
        let inner = || -> anyhow::Result<Self> {
            let this = Self {
                ids_start: mapping.get_as("ids_start")?,
                ids_end: mapping.get_as("ids_end")?,
                tags_start: mapping.get_as("tags_start")?,
                tags_end: mapping.get_as("tags_end")?,
            };
            if this.ids_start >= this.ids_end {
                anyhow::bail!(
                    "ids_start ({}) must be less than ids_end ({})",
                    this.ids_start,
                    this.ids_end
                );
            }
            if this.tags_start >= this.tags_end {
                anyhow::bail!(
                    "tags_start ({}) must be less than tags_end ({})",
                    this.tags_start,
                    this.tags_end
                );
            }
            Ok(this)
        };

        inner().context("trying to parse an \"replace\"")
    }
}

#[derive(Debug)]
struct Insert {
    start: usize,
    end: usize,
}

impl TryFrom<&Mapping> for Insert {
    type Error = anyhow::Error;
    fn try_from(mapping: &Mapping) -> anyhow::Result<Self> {
        let inner = || -> anyhow::Result<Self> {
            let this = Self {
                start: mapping.get_as("start")?,
                end: mapping.get_as("end")?,
            };
            if this.start >= this.end {
                anyhow::bail!(
                    "start ({}) must be less than end ({})",
                    this.start,
                    this.end
                );
            }
            Ok(this)
        };

        inner().context("trying to parse an \"insert\"")
    }
}

#[derive(Debug)]
struct Delete {
    start: usize,
    end: usize,
}

impl TryFrom<&Mapping> for Delete {
    type Error = anyhow::Error;
    fn try_from(mapping: &Mapping) -> anyhow::Result<Self> {
        let inner = || -> anyhow::Result<Self> {
            let this = Self {
                start: mapping.get_as("start")?,
                end: mapping.get_as("end")?,
            };
            if this.start >= this.end {
                anyhow::bail!(
                    "start ({}) must be less than end ({})",
                    this.start,
                    this.end
                );
            }
            Ok(this)
        };

        inner().context("trying to parse \"delete\"")
    }
}

fn classify(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Sequence(_) => "sequence",
        Value::Mapping(_) => "mapping",
        Value::Tagged(_) => "tagged",
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::{collections::HashMap, io::Write};

    use tempfile::NamedTempFile;

    use crate::streaming::executors::bigann::Stage;

    /// A test implementation of [`super::super::FindGroundtruth`] that returns
    /// pre-configured paths for each stage index.
    struct MockGroundtruth {
        paths: HashMap<usize, PathBuf>,
    }

    impl MockGroundtruth {
        fn new(stages: impl IntoIterator<Item = (usize, PathBuf)>) -> Self {
            Self {
                paths: stages.into_iter().collect(),
            }
        }
    }

    impl super::super::FindGroundtruth for MockGroundtruth {
        fn find_groundtruth(&mut self, stage: usize) -> anyhow::Result<PathBuf> {
            self.paths
                .get(&stage)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("no groundtruth configured for stage {}", stage))
        }
    }

    /// Helper to create a temporary YAML file with the given content.
    fn create_yaml_file(content: &str) -> anyhow::Result<NamedTempFile> {
        let mut file = NamedTempFile::new()?;
        file.write_all(content.as_bytes())?;
        file.flush()?;
        Ok(file)
    }

    #[test]
    fn test_load_simple_insert_only_runbook() {
        let yaml = r#"
test_dataset:
  max_pts: 1000
  0:
    operation: insert
    start: 0
    end: 500
"#;

        let file = create_yaml_file(yaml).unwrap();
        let mut groundtruth = MockGroundtruth::new([]);

        let runbook = load(file.path(), "test_dataset", &mut groundtruth).unwrap();

        assert_eq!(runbook.max_points(), 1000);
        assert_eq!(runbook.len(), 1);

        assert_eq!(
            runbook.stages()[0],
            Stage::Insert {
                dataset_offsets_and_ids: 0..500
            }
        );
    }

    #[test]
    fn test_load_runbook_with_search_stage() {
        let yaml = r#"
my_dataset:
  max_pts: 2000
  0:
    operation: insert
    start: 0
    end: 1000
  1:
    operation: search
"#;

        let file = create_yaml_file(yaml).unwrap();
        let mut groundtruth =
            MockGroundtruth::new([(1, PathBuf::from("/path/to/groundtruth.bin"))]);

        let runbook = load(file.path(), "my_dataset", &mut groundtruth).unwrap();

        assert_eq!(runbook.max_points(), 2000);
        assert_eq!(runbook.len(), 2);

        assert_eq!(
            runbook.stages()[1],
            Stage::Search {
                groundtruth: PathBuf::from("/path/to/groundtruth.bin")
            }
        );
    }

    #[test]
    fn test_load_runbook_with_all_operation_types() {
        let yaml = r#"
full_dataset:
  max_pts: 5000
  0:
    operation: insert
    start: 0
    end: 1000
  1:
    operation: search
  2:
    operation: replace
    ids_start: 1000
    ids_end: 1500
    tags_start: 0
    tags_end: 500
  3:
    operation: delete
    start: 500
    end: 1000
"#;

        let file = create_yaml_file(yaml).unwrap();
        let mut groundtruth = MockGroundtruth::new([(1, PathBuf::from("/gt/step1.bin"))]);

        let runbook = load(file.path(), "full_dataset", &mut groundtruth).unwrap();

        assert_eq!(runbook.max_points(), 5000);
        assert_eq!(runbook.len(), 4);

        // Check insert
        assert_eq!(
            runbook.stages()[0],
            Stage::Insert {
                dataset_offsets_and_ids: 0..1000
            }
        );

        // Check search
        assert_eq!(
            runbook.stages()[1],
            Stage::Search {
                groundtruth: PathBuf::from("/gt/step1.bin")
            }
        );

        // Check replace
        assert_eq!(
            runbook.stages()[2],
            Stage::Replace {
                dataset_offsets: 1000..1500,
                ids: 0..500
            }
        );

        // Check delete
        assert_eq!(runbook.stages()[3], Stage::Delete { ids: 500..1000 });
    }

    #[test]
    fn test_load_stages_out_of_order_are_sorted() {
        let yaml = r#"
unordered:
  max_pts: 1000
  2:
    operation: delete
    start: 500
    end: 600
  0:
    operation: insert
    start: 0
    end: 500
  1:
    operation: insert
    start: 500
    end: 1000
"#;

        let file = create_yaml_file(yaml).unwrap();
        let mut groundtruth = MockGroundtruth::new([]);

        let runbook = load(file.path(), "unordered", &mut groundtruth).unwrap();

        assert_eq!(runbook.len(), 3);

        // Stages should be in order 0, 1, 2 regardless of YAML order
        assert_eq!(
            runbook.stages()[0],
            Stage::Insert {
                dataset_offsets_and_ids: 0..500
            }
        );

        assert_eq!(
            runbook.stages()[1],
            Stage::Insert {
                dataset_offsets_and_ids: 500..1000
            }
        );

        assert_eq!(runbook.stages()[2], Stage::Delete { ids: 500..600 });
    }

    #[test]
    fn test_load_multiple_datasets_in_file() {
        let yaml = r#"
dataset_a:
  max_pts: 100
  0:
    operation: insert
    start: 0
    end: 100

dataset_b:
  max_pts: 200
  0:
    operation: insert
    start: 0
    end: 200
"#;

        let file = create_yaml_file(yaml).unwrap();

        // Load dataset_a
        let mut groundtruth_a = MockGroundtruth::new([]);
        let runbook_a = load(file.path(), "dataset_a", &mut groundtruth_a).unwrap();
        assert_eq!(runbook_a.max_points(), 100);

        // Load dataset_b
        let mut groundtruth_b = MockGroundtruth::new([]);
        let runbook_b = load(file.path(), "dataset_b", &mut groundtruth_b).unwrap();
        assert_eq!(runbook_b.max_points(), 200);
    }

    #[test]
    fn test_load_gt_url_is_ignored() {
        let yaml = r#"
with_gt_url:
  max_pts: 100
  gt_url: "https://example.com/groundtruth.bin"
  0:
    operation: insert
    start: 0
    end: 100
"#;

        let file = create_yaml_file(yaml).unwrap();
        let mut groundtruth = MockGroundtruth::new([]);

        // Should succeed - gt_url is parsed but ignored
        let runbook = load(file.path(), "with_gt_url", &mut groundtruth).unwrap();
        assert_eq!(runbook.max_points(), 100);
    }
}

#[cfg(test)]
mod ux_tests {
    use super::*;

    // Exposed by the `ux-tools` feature of `diskann_benchmark_runner`
    use diskann_benchmark_runner::ux as runner_ux;

    //---------------------------//
    // File-Based UX Error Tests //
    //---------------------------//
    //
    // These tests use checked-in YAML files and expected output files to verify error messages.
    // This approach makes it easy to:
    // 1. Add new test cases (just add a new directory with runbook.yaml, dataset.txt, expected.txt)
    // 2. See the full error output for review
    // 3. Regenerate expected output when error messages change
    //
    // ## Directory Structure
    //
    // Each test case is a directory under `tests/bigann-ux` containing:
    // - `runbook.yaml` - The YAML runbook file to parse
    // - `dataset.txt` - Contains the dataset name to load (single line)
    // - `expected.txt` - The expected error message output
    //
    // ## Regenerating Expected Results
    //
    // Run tests with the environment variable:
    // ```
    // DISKANN_BENCHMARK_TEST=overwrite
    // ```
    // to regenerate the `expected.txt` files. Use `git diff` to review changes.

    const TEST_DATA_DIR: &str = "bigann-ux";
    const RUNBOOK_FILE: &str = "runbook.yaml";
    const DATASET_FILE: &str = "dataset.txt";
    const EXPECTED_FILE: &str = "expected.txt";
    const PATH_PLACEHOLDER: &str = "<TEST_DIR>";

    /// Replace the test directory path with a placeholder to make tests portable.
    /// This handles both forward and backslash path separators.
    ///
    /// Additionally:
    /// * All backslashes are replaced with forward slashes.
    /// * Common OS-specific "file not found" error messages are normalized.
    fn fixup_paths_and_os_errors(s: &str, test_dir: &Path) -> String {
        // Try both the native path and with normalized separators
        let native_path = test_dir.display().to_string();
        let forward_slash_path = native_path.replace('\\', "/");

        const NOT_FOUND_WINDOWS: &str = "The system cannot find the file specified.";
        const NOT_FOUND_LINUX: &str = "No such file or directory";

        s.replace(&native_path, PATH_PLACEHOLDER)
            .replace(&forward_slash_path, PATH_PLACEHOLDER)
            .replace('\\', "/") // Normalize any remaining backslashes
            .replace(NOT_FOUND_WINDOWS, NOT_FOUND_LINUX) // Normalize error messages
    }

    /// A groundtruth finder that always fails - used for error path testing.
    struct FailingGroundtruth;

    impl super::super::FindGroundtruth for FailingGroundtruth {
        fn find_groundtruth(&mut self, stage: usize) -> anyhow::Result<PathBuf> {
            Err(anyhow::anyhow!(
                "groundtruth not available for stage {}",
                stage
            ))
        }
    }

    /// Run a single file-based test case.
    fn run_file_test(test_dir: &Path) {
        let runbook_path = test_dir.join(RUNBOOK_FILE);
        let dataset_path = test_dir.join(DATASET_FILE);
        let expected_path = test_dir.join(EXPECTED_FILE);

        // Read the dataset name
        let dataset = std::fs::read_to_string(&dataset_path)
            .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", dataset_path, e));
        let dataset = dataset.trim();

        // Try to load the runbook - we expect an error
        let mut groundtruth = FailingGroundtruth;
        let result = load(&runbook_path, dataset, &mut groundtruth);

        let actual_output = match result {
            Ok(_) => panic!(
                "Test {:?} expected an error but parsing succeeded",
                test_dir.file_name().unwrap()
            ),
            Err(err) => format!("{:?}", err),
        };

        // Replace test directory path with placeholder for portability
        let actual_portable = fixup_paths_and_os_errors(&actual_output, test_dir);
        let actual_normalized = runner_ux::strip_backtrace(runner_ux::normalize(actual_portable));

        if crate::ux::should_overwrite() {
            std::fs::write(&expected_path, &actual_normalized)
                .unwrap_or_else(|e| panic!("Failed to write {:?}: {}", expected_path, e));
            println!("Overwrote {:?}", expected_path);
        } else {
            let expected = std::fs::read_to_string(&expected_path)
                .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", expected_path, e));
            let expected_normalized = runner_ux::normalize(expected);

            if actual_normalized != expected_normalized {
                panic!(
                    "Test {:?} failed.\n\nExpected:\n---\n{}\n---\n\nActual:\n---\n{}\n---\nIf this is expected, run with {} to update the expected output.",
                    test_dir.file_name().unwrap(),
                    expected_normalized,
                    actual_normalized,
                    crate::ux::help(),
                );
            }
        }
    }

    /// Run all file-based tests in the test_data directory.
    fn run_all_file_tests() {
        let test_data_path = crate::ux::test_dir().join(TEST_DATA_DIR);
        if !test_data_path.exists() {
            println!(
                "No test_data directory found at {:?}, skipping file-based tests",
                test_data_path
            );
            return;
        }

        let mut found_tests = false;
        for entry in std::fs::read_dir(&test_data_path)
            .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", test_data_path, e))
        {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                found_tests = true;
                println!("Running file-based test: {:?}", entry.file_name());
                run_file_test(&entry.path());
            }
        }

        if !found_tests {
            panic!("No test directories found in {:?}", test_data_path);
        }
    }

    #[test]
    fn file_based_error_tests() {
        run_all_file_tests();
    }
}
