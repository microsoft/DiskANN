/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    any::Any,
    ops::Range,
    path::{Path, PathBuf},
};

use anyhow::Context;

use crate::streaming::{self, Executor, Stream};

use super::{parsing, validate};

/// An executor for [BigANN-style runbooks](https://github.com/harsha-simhadri/big-ann-benchmarks/tree/main/neurips23/streaming).
///
/// If using this struct as a [`streaming::Executor`], consider using the
/// [`super::WithData`] adaptor to provide dataset and query matrices.
#[derive(Debug)]
pub struct RunBook {
    // The individual runbook stages.
    stages: Vec<Stage>,
    // The maximum number of active points at any point in the runbook.
    max_points: usize,
    // The maximum tag referenced in the runbook.
    max_tag: Option<usize>,
}

impl RunBook {
    /// Loads a [`RunBook`] from a YAML file at `path` for the specified `dataset`.
    ///
    /// # Groundtruth Resolution
    ///
    /// When loading a runbook, search stages may require groundtruth files to be present.
    /// The index of these stages will be provided to the `groundtruth` argument for resolution.
    /// If working with the standard BigANN benchmark format, consider using [`ScanDirectory`]
    /// and providing the directory where the BigANN framework downloads the groundtruth files.
    ///
    /// Note that the implementation here currently does not attempt to download any groundtruth
    /// files using the `gt_url` field; it is merely parsed and ignored.
    ///
    /// # YAML File Format
    ///
    /// The top-level structure is a mapping from dataset names (strings) to their
    /// corresponding runbook definitions:
    /// ```yaml
    /// dataset_name_1:
    ///   # runbook definition...
    /// dataset_name_2:
    ///   # runbook definition...
    /// ```
    /// The `dataset` parameter specifies which dataset's runbook to load from the file.
    ///
    /// Each runbook definition has the following format:
    ///
    /// ```yaml
    /// max_pts: <maximum number of points in the index (integer)> # required
    /// [gt_url]: <URL for groundtruth files (string)> # ignored
    /// 0:
    ///   # stage definition ...
    /// 1:
    ///   # stage definition ...
    /// ...
    /// ```
    /// Entries need not be in order, but stages must be sequentially numbered starting
    /// from `0`. Each stage takes one of four forms (described below).
    ///
    /// ## Search Stage
    ///
    /// Merely specifies that a search should be performed. The queries must be provided
    /// externally. The groundtruth file for this stage is located via the provided
    /// [`FindGroundtruth::find_groundtruth`] method, which will be provided with the stage index.
    ///
    /// ```yaml
    /// <stage_index>:
    ///   operation: "search"
    /// ```
    ///
    /// ## Insert Stage
    ///
    /// Insert vectors from the underlying dataset into the index. The vectors to insert
    /// are specified by the range `start..end`, which serves as both the offsets of the
    /// vectors in the dataset and their external ids.
    ///
    /// ```yaml
    /// <stage_index>:
    ///   operation: "insert"
    ///   start: <starting offset/external id in dataset (integer)>
    ///   end: <ending offset/external id in dataset (integer)>
    /// ```
    ///
    /// ## Replace Stage
    ///
    /// Replace vectors in the index with vectors from the underlying dataset. Unlike insertions,
    /// replace operations distinguish between the dataset offsets (`ids_start..ids_end`)
    /// and the external ids (tags) of the vectors to replace (`tags_start..tags_end`).
    ///
    /// These operations indicate that the vectors in the index tagged by `tags_start..tags_end` should
    /// be replaced with the vectors from the dataset at offsets `ids_start..ids_end`.
    ///
    /// ```yaml
    /// <stage_index>:
    ///   operation: "replace"
    ///   ids_start: <starting offset in dataset (integer)>
    ///   ids_end: <ending offset in dataset (integer)>
    ///   tags_start: <starting external id to replace (integer)>
    ///   tags_end: <ending external id to replace (integer)>
    /// ```
    ///
    /// ## Delete Stage
    ///
    /// Delete vectors from the index with external ids in the range `start..end`.
    ///
    /// ```yaml
    /// <stage_index>:
    ///   operation: "delete"
    ///   start: <starting external id to delete (integer)>
    ///   end: <ending external id to delete (integer)>
    /// ```
    ///
    /// # File Validation
    ///
    /// The loading framework does a limited amount of validation on the YAML file:
    ///
    /// 1. The specified `dataset` must exist in the top-level mapping.
    /// 2. The `max_pts` key must be present and be a valid integer. Its value is verified and if found to
    ///    be inaccurate, it is updated to the correct value internally.
    /// 3. Each stage must be sequentially numbered starting from `0` with no gaps.
    /// 4. Each stage must have a valid operation with all required fields present and of the correct type.
    /// 5. Groundtruth files must be resolvable via the provided [`FindGroundtruth`] implementation.
    pub fn load(
        path: &Path,
        dataset: &str,
        groundtruth: &mut dyn FindGroundtruth,
    ) -> anyhow::Result<Self> {
        parsing::load(path, dataset, groundtruth)
    }

    pub(super) fn new(stages: Vec<Stage>, max_points: usize) -> anyhow::Result<Self> {
        let mut this = Self {
            stages,
            max_points,
            max_tag: None,
        };

        let mut validator = validate::Validate::new();
        this.run_with(&mut validator, |_| Ok(()))?;

        this.max_points = this.max_points.max(validator.max_active());
        this.max_tag = validator.max_tag();

        Ok(this)
    }

    /// Returns the maximum number of points specified in the runbook.
    pub fn max_points(&self) -> usize {
        self.max_points
    }

    /// Returns the maximum tag specified in the runbook.
    pub fn max_tag(&self) -> Option<usize> {
        self.max_tag
    }

    /// Returns the number of stages in the runbook.
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Returns `true` if the runbook contains no stages.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a reference to the stages in this runbook.
    #[cfg(test)]
    pub(super) fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// Executes the runbook by iterating through each stage.
    ///
    /// When calling this method, the dynamic type of `stream`'s output must be
    /// compatible with the dynamic type expected by `collect`.
    fn run_with_internal(
        &self,
        stream: &mut dyn streaming::Stream<Args, Output = Box<dyn Any>>,
        collect: &mut dyn FnMut(Box<dyn Any>) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        for (i, stage) in self.stages.iter().enumerate() {
            #[derive(Clone, Copy)]
            struct OnStage(usize, usize);

            impl std::fmt::Display for OnStage {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, "on stage {} of {}", self.0, self.1)
                }
            }

            let context = OnStage(i, self.len());

            if stream.needs_maintenance() {
                collect(stream.maintain(()).context(context)?).context(context)?;
            }

            let output = match stage {
                Stage::Search { groundtruth } => {
                    let args = Search { groundtruth };
                    stream.search(args).context(context)?
                }
                Stage::Insert {
                    dataset_offsets_and_ids,
                } => {
                    let args = Insert {
                        offsets: dataset_offsets_and_ids.clone(),
                        ids: dataset_offsets_and_ids.clone(),
                    };
                    stream.insert(args).context(context)?
                }
                Stage::Replace {
                    dataset_offsets,
                    ids,
                } => {
                    let args = Replace {
                        offsets: dataset_offsets.clone(),
                        ids: ids.clone(),
                    };
                    stream.replace(args).context(context)?
                }
                Stage::Delete { ids } => {
                    let args = Delete { ids: ids.clone() };
                    stream.delete(args).context(context)?
                }
            };

            collect(output).context(context)?;
        }

        Ok(())
    }
}

/// An operation in a BigANN runbook.
#[derive(Debug, Clone, PartialEq)]
pub enum Stage {
    /// Perform a search operation using the specified groundtruth file.
    Search {
        /// The resolved path to the groundtruth file for this search stage.
        groundtruth: PathBuf,
    },
    /// Insert vectors from the dataset into the index.
    Insert {
        /// The offsets in the dataset, which also serve as the external ids for
        /// the inserted vectors.
        dataset_offsets_and_ids: Range<usize>,
    },
    /// Replace vectors in the index with new vectors from the dataset.
    Replace {
        /// The offsets in the dataset for the replacement vectors.
        dataset_offsets: Range<usize>,
        /// The external ids of the vectors to be replaced.
        ids: Range<usize>,
    },
    /// Delete vectors from the index.
    Delete {
        /// The external ids of the vectors to delete.
        ids: Range<usize>,
    },
}

/// Arguments for a BigANN runbook "search" stage.
#[derive(Debug)]
pub struct Search<'a> {
    /// The resolved path to the file containing the groundtruth for this stage.
    pub groundtruth: &'a Path,
}

/// Arguments for a BigANN runbook "insert" stage.
pub struct Insert {
    /// The range of offsets in the dataset for vectors to insert.
    pub offsets: Range<usize>,
    /// The external ids to assign to the inserted vectors.
    pub ids: Range<usize>,
}

/// Arguments for a BigANN runbook "replace" stage.
pub struct Replace {
    /// The range of offsets in the dataset for the replacement vectors.
    pub offsets: Range<usize>,
    /// The external ids of the vectors to replace.
    pub ids: Range<usize>,
}

/// Arguments for a BigANN runbook "delete" stage.
pub struct Delete {
    /// The range of external ids to delete from the index.
    pub ids: Range<usize>,
}

/// The argument type for a [`RunBook`].
///
/// See also: [`super::WithData`], [`super::DataArgs`].
#[derive(Debug, Clone, Copy)]
pub struct Args;

impl streaming::Arguments for Args {
    type Search<'a> = Search<'a>;
    type Insert<'a> = Insert;
    type Replace<'a> = Replace;
    type Delete<'a> = Delete;
    type Maintain<'a> = ();
}

impl streaming::Executor for RunBook {
    type Args = Args;

    fn run_with<S, F, O>(&mut self, stream: &mut S, mut collect: F) -> anyhow::Result<()>
    where
        S: Stream<Args, Output = O>,
        O: 'static,
        F: FnMut(O) -> anyhow::Result<()>,
    {
        self.run_with_internal(&mut streaming::AnyStream::new(stream), &mut |any| {
            let typed = *any
                .downcast::<S::Output>()
                .expect("the dynamic type should be configured correctly");
            collect(typed)
        })
    }
}

/// A trait for resolving groundtruth files for search stages in a [`RunBook`].
///
/// Implementors of this trait provide the logic to locate groundtruth files
/// given a stage index. See [`ScanDirectory`] for a common implementation.
pub trait FindGroundtruth {
    /// Resolves the groundtruth file path for the specified `stage` index.
    ///
    /// This method is only called for "search" stages in a runbook.
    fn find_groundtruth(&mut self, stage: usize) -> anyhow::Result<PathBuf>;
}

/// A [`FindGroundtruth`] implementation that scans a directory for groundtruth files
/// matching the expected BigANN naming convention: `step{stage}.gt[0-9]*` where
/// `{stage}` substitutes the stage index formatted as a string.
#[derive(Debug)]
pub struct ScanDirectory {
    directory: PathBuf,

    // Cached contents of the files in `directory`.
    files: Vec<String>,
}

impl ScanDirectory {
    /// Creates a new [`ScanDirectory`] instance for the specified `directory`.
    ///
    /// # Notes
    ///
    /// This constructor scans the directory and caches its contents.
    /// If the directory contents change after creation, the instance will not
    /// reflect those changes.
    ///
    /// This is meant for the common benchmarking scenario where the benchmarking
    /// machine is generally static while benchmarks are executed.
    pub fn new(directory: impl Into<PathBuf>) -> anyhow::Result<Self> {
        Self::new_(directory.into())
    }

    fn new_(directory: PathBuf) -> anyhow::Result<Self> {
        // Read all files in the directory.
        let read_dir = std::fs::read_dir(&directory).with_context(|| {
            format!(
                "while trying to read the contents of {}",
                directory.display()
            )
        })?;

        let files = read_dir
            .filter_map(|entry| {
                if let Ok(entry) = entry
                    && let Ok(file_type) = entry.file_type()
                    && file_type.is_file()
                {
                    Some(entry.file_name().to_string_lossy().into())
                } else {
                    None
                }
            })
            .collect();

        Ok(Self { directory, files })
    }
}

impl FindGroundtruth for ScanDirectory {
    /// Finds the groundtruth file for the specified `stage` by scanning the directory.
    ///
    /// # Errors
    ///
    /// Returns an error if no matching file is found or if multiple matches exist.
    fn find_groundtruth(&mut self, stage: usize) -> anyhow::Result<PathBuf> {
        let prefix = format!("step{}.gt", stage);

        enum Matches<'a> {
            None,
            One(&'a str),
            Many(Vec<&'a str>),
        }

        impl<'a> Matches<'a> {
            fn push(&mut self, file: &'a str) {
                *self = match std::mem::replace(self, Self::None) {
                    Self::None => Self::One(file),
                    Self::One(first) => Self::Many(vec![first, file]),
                    Self::Many(mut all) => {
                        all.push(file);
                        Self::Many(all)
                    }
                };
            }
        }

        let mut matches = Matches::None;

        for file in self.files.iter() {
            if file.starts_with(&prefix) {
                let suffix = &file[prefix.len()..];
                if suffix.chars().all(|c| c.is_ascii_digit()) {
                    matches.push(file);
                }
            }
        }

        match matches {
            Matches::One(m) => Ok(self.directory.join(m)),
            Matches::None => Err(anyhow::anyhow!(
                "No groundtruth found for step {} in \"{}\", expected pattern: \"step{}.gt[0-9]*\"",
                stage,
                self.directory.display(),
                stage,
            )),
            Matches::Many(matches) => Err(anyhow::anyhow!(
                "Multiple groundtruth files found for step {} in \"{}\": {:?}",
                stage,
                self.directory.display(),
                matches,
            )),
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs::File;

    use tempfile::TempDir;

    use crate::streaming::Executor;

    //---------//
    // Runbook //
    //---------//

    struct MockStream {
        stages: Vec<Stage>,
        current_stage: usize,
        asked_for_maintenance: bool,
    }

    impl MockStream {
        fn new(stages: Vec<Stage>) -> Self {
            Self {
                stages,
                current_stage: 0,
                asked_for_maintenance: false,
            }
        }

        fn increment(&mut self) -> usize {
            let output = self.current_stage;
            self.current_stage += 1;
            output
        }

        fn current(&self) -> &Stage {
            &self.stages[self.current_stage]
        }
    }

    impl streaming::Stream<Args> for MockStream {
        type Output = Option<usize>;

        fn search(&mut self, args: Search<'_>) -> anyhow::Result<Option<usize>> {
            if let Stage::Search { groundtruth } = self.current() {
                assert_eq!(args.groundtruth, groundtruth.as_path());
                Ok(Some(self.increment()))
            } else {
                Err(anyhow::anyhow!(
                    "Expected Search stage, instead got {:?}",
                    self.current()
                ))
            }
        }

        fn insert(&mut self, args: Insert) -> anyhow::Result<Option<usize>> {
            if let Stage::Insert {
                dataset_offsets_and_ids,
            } = self.current()
            {
                assert_eq!(&args.offsets, dataset_offsets_and_ids);
                assert_eq!(&args.ids, dataset_offsets_and_ids);
                Ok(Some(self.increment()))
            } else {
                Err(anyhow::anyhow!(
                    "Expected Insert stage, instead got {:?}",
                    self.current()
                ))
            }
        }

        fn replace(&mut self, args: Replace) -> anyhow::Result<Option<usize>> {
            if let Stage::Replace {
                dataset_offsets,
                ids,
            } = self.current()
            {
                assert_eq!(&args.offsets, dataset_offsets);
                assert_eq!(&args.ids, ids);
                Ok(Some(self.increment()))
            } else {
                Err(anyhow::anyhow!(
                    "Expected Replace stage, instead got {:?}",
                    self.current()
                ))
            }
        }

        fn delete(&mut self, args: Delete) -> anyhow::Result<Option<usize>> {
            if let Stage::Delete { ids } = self.current() {
                assert_eq!(&args.ids, ids);
                Ok(Some(self.increment()))
            } else {
                Err(anyhow::anyhow!(
                    "Expected Delete stage, instead got {:?}",
                    self.current()
                ))
            }
        }

        fn maintain(&mut self, _args: ()) -> anyhow::Result<Option<usize>> {
            assert!(
                self.asked_for_maintenance,
                "Stream was not expected to need maintenance"
            );
            self.asked_for_maintenance = false;
            Ok(None)
        }

        fn needs_maintenance(&mut self) -> bool {
            let needs = self.asked_for_maintenance;
            self.asked_for_maintenance = true;
            needs
        }
    }

    #[test]
    fn test_runbook() {
        let stages = vec![
            Stage::Insert {
                dataset_offsets_and_ids: 0..100,
            },
            Stage::Search {
                groundtruth: PathBuf::from("gt0"),
            },
            Stage::Replace {
                dataset_offsets: 100..200,
                ids: 0..100,
            },
            Stage::Delete { ids: 50..75 },
            Stage::Search {
                groundtruth: PathBuf::from("gt1"),
            },
        ];

        let mut runbook = RunBook::new(stages.clone(), 1000).unwrap();
        assert_eq!(runbook.len(), stages.len());
        assert!(!runbook.is_empty());
        assert_eq!(runbook.max_points(), 1000);

        let mut stream = MockStream::new(stages.clone());
        let outputs = runbook.run(&mut stream).unwrap();

        // Verify that the outputs match the expected stage indices.
        let expected_outputs: Vec<usize> = (0..stages.len()).collect();
        let non_maintenance: Vec<_> = outputs.iter().filter_map(|o| *o).collect();
        assert_eq!(non_maintenance, expected_outputs);
    }

    #[test]
    fn test_load_runbook_from_yaml() {
        use std::io::Write;

        let temp_dir = TempDir::new().unwrap();

        // Create groundtruth files for search stages (stages 1 and 7)
        File::create(temp_dir.path().join("step1.gt100")).unwrap();
        File::create(temp_dir.path().join("step7.gt100")).unwrap();

        // Create a YAML runbook with multiple insert, replace, and delete stages
        let yaml_content = r#"
test_dataset:
  max_pts: 100
  gt_url: "http://example.com/groundtruth"
  0:
    operation: "insert"
    start: 0
    end: 1000
  1:
    operation: "search"
  2:
    operation: "insert"
    start: 1000
    end: 2000
  3:
    operation: "delete"
    start: 200
    end: 400
  4:
    operation: "replace"
    ids_start: 2000
    ids_end: 2500
    tags_start: 400
    tags_end: 900
  5:
    operation: "insert"
    start: 2500
    end: 3000
  6:
    operation: "delete"
    start: 500
    end: 700
  7:
    operation: "search"
"#;

        let yaml_path = temp_dir.path().join("runbook.yaml");
        {
            let mut file = File::create(&yaml_path).unwrap();
            file.write_all(yaml_content.as_bytes()).unwrap();
        }

        // Load the runbook
        let mut groundtruth_finder = ScanDirectory::new(temp_dir.path()).unwrap();
        let runbook = RunBook::load(&yaml_path, "test_dataset", &mut groundtruth_finder).unwrap();

        // Verify the runbook was loaded correctly
        assert_eq!(runbook.len(), 8);
        assert_eq!(runbook.max_points(), 2300);
        assert_eq!(runbook.max_tag(), Some(2999));

        let stages = runbook.stages();

        // Stage 0: Insert 0..1000
        assert_eq!(
            stages[0],
            Stage::Insert {
                dataset_offsets_and_ids: 0..1000
            }
        );

        // Stage 1: Search
        assert!(
            matches!(&stages[1], Stage::Search { groundtruth } if groundtruth.file_name().unwrap() == "step1.gt100")
        );

        // Stage 2: Insert 1000..2000
        assert_eq!(
            stages[2],
            Stage::Insert {
                dataset_offsets_and_ids: 1000..2000
            }
        );

        // Stage 3: Delete 200..400
        assert_eq!(stages[3], Stage::Delete { ids: 200..400 });

        // Stage 4: Replace offsets 2000..2500, ids 400..900
        assert_eq!(
            stages[4],
            Stage::Replace {
                dataset_offsets: 2000..2500,
                ids: 400..900
            }
        );

        // Stage 5: Insert 2500..3000
        assert_eq!(
            stages[5],
            Stage::Insert {
                dataset_offsets_and_ids: 2500..3000
            }
        );

        // Stage 6: Delete 500..700
        assert_eq!(stages[6], Stage::Delete { ids: 500..700 });

        // Stage 7: Search
        assert!(
            matches!(&stages[7], Stage::Search { groundtruth } if groundtruth.file_name().unwrap() == "step7.gt100")
        );
    }

    //---------------------//
    // ScanDirectory Tests //
    //---------------------//

    #[test]
    fn scan_directory_finds_groundtruth_file() {
        let temp_dir = TempDir::new().unwrap();

        // Create a groundtruth file matching the expected pattern
        File::create(temp_dir.path().join("step0.gt100")).unwrap();

        let mut scanner = ScanDirectory::new(temp_dir.path()).unwrap();
        let result = scanner.find_groundtruth(0).unwrap();
        assert_eq!(result, temp_dir.path().join("step0.gt100"));
    }

    #[test]
    fn scan_directory_finds_groundtruth_without_suffix_digits() {
        let temp_dir = TempDir::new().unwrap();

        // Create a groundtruth file with no digits after ".gt"
        File::create(temp_dir.path().join("step5.gt")).unwrap();

        let mut scanner = ScanDirectory::new(temp_dir.path()).unwrap();
        let result = scanner.find_groundtruth(5).unwrap();
        assert_eq!(result, temp_dir.path().join("step5.gt"));
    }

    #[test]
    fn scan_directory_errors_when_no_groundtruth_found() {
        let temp_dir = TempDir::new().unwrap();

        // Create some files that don't match the pattern
        File::create(temp_dir.path().join("other_file.bin")).unwrap();
        File::create(temp_dir.path().join("step0.other")).unwrap();

        let mut scanner = ScanDirectory::new(temp_dir.path()).unwrap();
        let err = scanner.find_groundtruth(0).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("No groundtruth found"), "Got: {}", msg);
    }

    #[test]
    fn scan_directory_errors_when_multiple_groundtruth_files() {
        let temp_dir = TempDir::new().unwrap();

        // Create multiple groundtruth files for the same stage
        File::create(temp_dir.path().join("step0.gt100")).unwrap();
        File::create(temp_dir.path().join("step0.gt200")).unwrap();
        File::create(temp_dir.path().join("step0.gt300")).unwrap();

        let mut scanner = ScanDirectory::new(temp_dir.path()).unwrap();
        let err = scanner.find_groundtruth(0).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("Multiple groundtruth files"), "Got: {}", msg);
    }

    #[test]
    fn scan_directory_ignores_non_digit_suffix() {
        let temp_dir = TempDir::new().unwrap();

        // Create files with non-digit suffixes (should be ignored)
        File::create(temp_dir.path().join("step0.gtabc")).unwrap();
        File::create(temp_dir.path().join("step0.gt100")).unwrap();

        let mut scanner = ScanDirectory::new(temp_dir.path()).unwrap();
        let result = scanner.find_groundtruth(0).unwrap();

        // Should find only the valid one
        assert_eq!(result, temp_dir.path().join("step0.gt100"));
    }

    #[test]
    fn scan_directory_errors_on_nonexistent_directory() {
        let _ = ScanDirectory::new("/nonexistent/path/that/does/not/exist").unwrap_err();
    }

    #[test]
    fn scan_directory_handles_different_stage_indices() {
        let temp_dir = TempDir::new().unwrap();

        File::create(temp_dir.path().join("step0.gt")).unwrap();
        File::create(temp_dir.path().join("step5.gt")).unwrap();
        File::create(temp_dir.path().join("step10.gt")).unwrap();

        let mut scanner = ScanDirectory::new(temp_dir.path()).unwrap();

        assert_eq!(
            scanner.find_groundtruth(0).unwrap(),
            temp_dir.path().join("step0.gt")
        );
        assert_eq!(
            scanner.find_groundtruth(5).unwrap(),
            temp_dir.path().join("step5.gt")
        );
        assert_eq!(
            scanner.find_groundtruth(10).unwrap(),
            temp_dir.path().join("step10.gt")
        );

        // Stage 1 doesn't exist
        assert!(scanner.find_groundtruth(1).is_err());
    }
}
