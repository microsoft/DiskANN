/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

/// Shared context for resolving input and output files paths post deserialization.
#[derive(Debug)]
pub struct Checker {
    /// Root directories in which to look for files.
    ///
    /// Loading input files will first look to see if the input file is an absolute path.
    /// If so, the absolute path will be used.
    ///
    /// Otherwise, the search directories are traversed from beginning to end.
    search_directories: Vec<PathBuf>,

    /// Root directory (only one permitted) to write output files into
    /// and check for output files
    output_directory: Option<PathBuf>,

    /// The collection of output directories registered so far with the checker.
    ///
    /// This ensures that each job uses a distinct output directory to avoid conflicts.
    current_outputs: HashSet<PathBuf>,
}

impl Checker {
    /// Create a new checker with the list of search directories..
    pub(crate) fn new(search_directories: Vec<PathBuf>, output_directory: Option<PathBuf>) -> Self {
        Self {
            search_directories,
            output_directory,
            current_outputs: HashSet::new(),
        }
    }

    /// Return the ordered list of search directories registered with the [`Checker`].
    pub fn search_directories(&self) -> &[PathBuf] {
        &self.search_directories
    }

    /// Return the output directory registered with the [`Checker`], if any.
    pub fn output_directory(&self) -> Option<&PathBuf> {
        self.output_directory.as_ref()
    }

    /// Register `save_path` as an output directory and resolve `save_path` to an absolute path.
    ///
    /// # NOTE
    ///
    /// The behavior of this function is expected to change in the near future.
    pub fn register_output(&mut self, save_path: Option<&Path>) -> anyhow::Result<PathBuf> {
        // Check if `save_path` is absolute or relative. If relative, resolve it to an absolute
        // path using `self.output_directory.
        let resolved_dir = match save_path {
            None => {
                if let Some(output_dir) = self.output_directory() {
                    output_dir.clone()
                } else {
                    return Err(anyhow::Error::msg(
                        "relative save path \"{}\" specified but no output directory was provided",
                    ));
                }
            }
            Some(save_path) => {
                if save_path.is_absolute() {
                    if !(save_path.is_dir()) {
                        return Err(anyhow::Error::msg(format!(
                            "absolute save path \"{}\" is not a valid directory",
                            save_path.display()
                        )));
                    }
                    save_path.to_path_buf()
                } else {
                    // relative path, we concatenate it with the output directory
                    if let Some(output_dir) = self.output_directory() {
                        let absolute = output_dir.join(save_path);
                        if !absolute.is_dir() {
                            return Err(anyhow::Error::msg(format!(
                                "relative save path \"{}\" is not a valid directory when combined with output directory \"{}\"",
                                save_path.display(),
                                output_dir.display()
                            )));
                        }
                        absolute
                    } else {
                        return Err(anyhow::Error::msg(format!(
                            "relative save path \"{}\" specified but no output directory was provided",
                            save_path.display()
                        )));
                    }
                }
            }
        };

        // If the resolved directory already exists - bail.
        if !self.current_outputs.insert(resolved_dir.clone()) {
            anyhow::bail!(
                "output directory {} already being used by another job",
                resolved_dir.display()
            );
        } else {
            Ok(resolved_dir)
        }
    }

    /// Try to resolve `path` using the following approach:
    ///
    /// 1. If `path` is absolute - check that it exists and is a valid file. If
    ///    successful, return `path` unaltered.
    ///
    /// 2. If `path` is relative, work through `self.search_directories()` in order,
    ///    returning the absolute path first existing file.
    #[deprecated(since = "0.54.0", note = "please use `find_input_file` instead")]
    pub fn check_path(&self, path: &Path) -> Result<PathBuf, anyhow::Error> {
        self.find_input_file(path)
    }

    /// Try to resolve `path` as a file using the following approach:
    ///
    /// 1. If `path` is absolute - check that it exists and is a valid file. If
    ///    successful, return `path` unaltered.
    ///
    /// 2. If `path` is relative, work through `self.search_directories()` in order,
    ///    returning the absolute path first existing file.
    ///
    /// See also: [`Self::find_input_dir`].
    pub fn find_input_file(&self, path: &Path) -> Result<PathBuf, anyhow::Error> {
        self.check_input_path(path, Kind::File)
    }

    /// Try to resolve `path` as a directory using the following approach:
    ///
    /// 1. If `path` is absolute - check that it exists and is a valid directory. If
    ///    successful, return `path` unaltered.
    ///
    /// 2. If `path` is relative, work through `self.search_directories()` in order,
    ///    returning the absolute path first existing directory.
    ///
    /// See also: [`Self::find_input_file`].
    pub fn find_input_dir(&self, path: &Path) -> Result<PathBuf, anyhow::Error> {
        self.check_input_path(path, Kind::Dir)
    }

    fn check_input_path(&self, path: &Path, kind: Kind) -> Result<PathBuf, anyhow::Error> {
        // Check if the path exists (allowing for relative paths with respect to checker's
        // search directories).
        //
        // If the path is absolute - check if it exists and if it doesn't, we are done.
        if path.is_absolute() {
            if kind.check(path) {
                return Ok(path.into());
            } else {
                let kind = kind.as_str();
                return Err(anyhow::Error::msg(format!(
                    "input {} with absolute path \"{}\" either does not exist or is not a {}",
                    kind,
                    path.display(),
                    kind,
                )));
            }
        };

        // At this point, start searching in the provided directories.
        for dir in self.search_directories() {
            let absolute = dir.join(path);
            if kind.check(&absolute) {
                return Ok(absolute);
            }
        }

        Err(anyhow::Error::msg(format!(
            "could not find input {} \"{}\" in the search directories \"{:?}\"",
            kind.as_str(),
            path.display(),
            self.search_directories(),
        )))
    }

    pub fn __check_dir(&self, dir: &Path) -> Result<PathBuf, anyhow::Error> {
        // Check if the file exists (allowing for relative paths with respect to the current
        // directory.
        //
        // If the path is an absolute path and the file does not exist, then bail.
        if dir.is_absolute() {
            if dir.is_dir() {
                return Ok(dir.into());
            } else {
                return Err(anyhow::Error::msg(format!(
                    "input file with absolute path \"{}\" either does not exist or is not a file",
                    dir.display()
                )));
            }
        };

        // At this point, start searching in the provided directories.
        for d in self.search_directories() {
            let absolute = d.join(dir);
            if absolute.is_dir() {
                return Ok(absolute);
            }
        }
        Err(anyhow::Error::msg(format!(
            "could not find input file \"{}\" in the search directories \"{:?}\"",
            dir.display(),
            self.search_directories(),
        )))
    }
}

#[derive(Debug, Clone, Copy)]
enum Kind {
    File,
    Dir,
}

impl Kind {
    fn check(self, path: &Path) -> bool {
        match self {
            Self::File => path.is_file(),
            Self::Dir => path.is_dir(),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::File => "file",
            Self::Dir => "directory",
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs::{create_dir, File};

    #[test]
    fn test_constructor() {
        let checker = Checker::new(Vec::new(), None);
        assert!(checker.search_directories().is_empty());
        assert!(checker.output_directory().is_none());

        let dir_a: PathBuf = "directory/a".into();
        let dir_b: PathBuf = "directory/another/b".into();

        let checker = Checker::new(vec![dir_a.clone()], Some(dir_b.clone()));
        assert_eq!(checker.search_directories(), vec![dir_a.clone()]);
        assert_eq!(checker.output_directory(), Some(&dir_b));

        let checker = Checker::new(vec![dir_a.clone(), dir_b.clone()], None);
        assert_eq!(
            checker.search_directories(),
            vec![dir_a.clone(), dir_b.clone()]
        );
        assert!(checker.output_directory().is_none());
    }

    // Create a directory that looks like this:
    //
    // dir/
    //     file_a.txt
    //     dir0/
    //        file_b.txt
    //        dir2/
    //     dir1/
    //        file_c.txt
    //        dir0/
    //           file_c.txt
    fn create_test_directory(dir: &Path) {
        File::create(dir.join("file_a.txt")).unwrap();

        create_dir(dir.join("dir0")).unwrap();
        create_dir(dir.join("dir0/dir2")).unwrap();
        create_dir(dir.join("dir1")).unwrap();
        create_dir(dir.join("dir1/dir0")).unwrap();
        File::create(dir.join("dir0/file_b.txt")).unwrap();
        File::create(dir.join("dir1/file_c.txt")).unwrap();
        File::create(dir.join("dir1/dir0/file_c.txt")).unwrap();
    }

    #[test]
    fn test_find_input_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        create_test_directory(path);

        let make_checker = |paths: &[PathBuf]| -> Checker { Checker::new(paths.to_vec(), None) };

        // Test absolute path success.
        {
            let checker = make_checker(&[]);
            let absolute = path.join("file_a.txt");
            assert_eq!(
                checker.find_input_file(&absolute).unwrap(),
                absolute,
                "absolute paths should be unmodified if they exist",
            );

            let absolute = path.join("dir0/file_b.txt");
            assert_eq!(
                checker.find_input_file(&absolute).unwrap(),
                absolute,
                "absolute paths should be unmodified if they exist",
            );
        }

        // Absolute path fail.
        {
            let checker = make_checker(&[]);
            let absolute = path.join("dir0/file_c.txt");
            let err = checker.find_input_file(&absolute).unwrap_err();
            let message = err.to_string();
            assert!(message.contains("input file with absolute path"));
            assert!(message.contains("either does not exist or is not a file"));
        }

        // Directory search
        {
            let checker =
                make_checker(&[path.join("dir1/dir0"), path.join("dir1"), path.join("dir0")]);

            // Directories are searched in order.
            let file = &Path::new("file_c.txt");
            let resolved = checker.find_input_file(file).unwrap();
            assert_eq!(resolved, path.join("dir1/dir0/file_c.txt"));

            let file = &Path::new("file_b.txt");
            let resolved = checker.find_input_file(file).unwrap();
            assert_eq!(resolved, path.join("dir0/file_b.txt"));

            // Directory search can fail.
            let file = &Path::new("file_a.txt");
            let err = checker.find_input_file(file).unwrap_err();
            let message = err.to_string();
            assert!(message.contains("could not find input file"));
            assert!(message.contains("in the search directories"));

            // If we give an absolute path, no directory search is performed.
            let file = path.join("file_c.txt");
            let err = checker.find_input_file(&file).unwrap_err();
            let message = err.to_string();
            assert!(message.starts_with("input file with absolute path"));
        }
    }

    #[test]
    fn test_find_input_dir() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        create_test_directory(path);

        let make_checker = |paths: &[PathBuf]| -> Checker { Checker::new(paths.to_vec(), None) };

        // Test absolute path success.
        {
            let checker = make_checker(&[]);
            let absolute = path.join("dir0");
            assert_eq!(
                checker.find_input_dir(&absolute).unwrap(),
                absolute,
                "absolute paths should be unmodified if they exist",
            );

            let absolute = path.join("dir1/dir0");
            assert_eq!(
                checker.find_input_dir(&absolute).unwrap(),
                absolute,
                "absolute paths should be unmodified if they exist",
            );
        }

        // Absolute path fail.
        {
            let checker = make_checker(&[]);
            let absolute = path.join("dir1/dir1");
            let err = checker.find_input_dir(&absolute).unwrap_err();
            let message = err.to_string();
            assert!(message.contains("input directory with absolute path"));
            assert!(message.contains("either does not exist or is not a directory"));

            // Files are rejected.
            let absolute = path.join("file_a.txt");
            let err = checker.find_input_dir(&absolute).unwrap_err();
            let message = err.to_string();
            assert!(message.contains("input directory with absolute path"));
            assert!(message.contains("either does not exist or is not a directory"));
        }

        // Directory search
        {
            let checker =
                make_checker(&[path.join("dir1/dir0"), path.join("dir1"), path.join("dir0")]);

            // Directories are searched in order.
            let dir = &Path::new("dir0");
            let resolved = checker.find_input_dir(dir).unwrap();
            assert_eq!(resolved, path.join("dir1/dir0"));

            let dir = &Path::new("dir2");
            let resolved = checker.find_input_dir(dir).unwrap();
            assert_eq!(resolved, path.join("dir0/dir2"));

            // Directory search can fail.
            let dir = &Path::new("nope");
            let err = checker.find_input_dir(dir).unwrap_err();
            let message = err.to_string();
            assert!(message.contains("could not find input directory"));
            assert!(message.contains("in the search directories"));

            // If we give an absolute path, no directory search is performed.
            let dir = path.join("dir2");
            let err = checker.find_input_dir(&dir).unwrap_err();
            let message = err.to_string();
            assert!(message.starts_with("input directory with absolute path"));
        }
    }
}
