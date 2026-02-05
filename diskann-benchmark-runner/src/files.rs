/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::checker::{CheckDeserialization, Checker};

/// A file that is used as an input to for a benchmark.
///
/// When used during input deserialization, with a [`Checker`], the following actions will be
/// taken:
///
/// 1. If the path is absolute or points to an existing file relative to the current working
///    directory, no additional measure will be taken.
///
/// 2. If the path is not absolute, then every directory in [`Checker::search_directories()`]
///    will be explored in-order. The first directory where `path` exists will be selected.
///
/// 3. If all these steps fail, then post-deserialization checking will fail with an error.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(transparent)]
pub struct InputFile {
    path: PathBuf,
}

impl InputFile {
    /// Create a new new input file from the path-like `path``
    pub fn new<P>(path: P) -> Self
    where
        PathBuf: From<P>,
    {
        Self {
            path: PathBuf::from(path),
        }
    }
}

impl std::ops::Deref for InputFile {
    type Target = Path;
    fn deref(&self) -> &Self::Target {
        &self.path
    }
}

impl CheckDeserialization for InputFile {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        let checked_path = checker.check_path(self);
        match checked_path {
            Ok(p) => {
                self.path = p;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}

impl AsRef<Path> for InputFile {
    fn as_ref(&self) -> &Path {
        &*self
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::fs::{create_dir, File};

    use super::*;

    #[test]
    fn test_input_file() {
        let file = InputFile::new("hello/world");
        let file_deref: &Path = &file;
        assert_eq!(file_deref.to_str().unwrap(), "hello/world");
    }

    #[test]
    fn test_serialization() {
        let st: &str = "\"path/to/directory\"";
        let file: InputFile = serde_json::from_str(st).unwrap();
        assert_eq!(file.to_str().unwrap(), st.trim_matches('\"'));

        assert_eq!(&*serde_json::to_string(&file).unwrap(), st);
    }

    #[test]
    fn test_check_deserialization() {
        // We create a directory that looks like this:
        //
        // dir/
        //     file_a.txt
        //     dir0/
        //        file_b.txt
        //     dir1/
        //        file_c.txt
        //        dir0/
        //           file_c.txt
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();

        File::create(path.join("file_a.txt")).unwrap();
        create_dir(path.join("dir0")).unwrap();
        create_dir(path.join("dir1")).unwrap();
        create_dir(path.join("dir1/dir0")).unwrap();
        File::create(path.join("dir0/file_b.txt")).unwrap();
        File::create(path.join("dir1/file_c.txt")).unwrap();
        File::create(path.join("dir1/dir0/file_c.txt")).unwrap();

        // Test absolute path success.
        {
            let absolute = path.join("file_a.txt");
            let mut file = InputFile::new(absolute.clone());
            let mut checker = Checker::new(Vec::new(), None);
            file.check_deserialization(&mut checker).unwrap();
            assert_eq!(file.path, absolute);

            let absolute = path.join("dir0/file_b.txt");
            let mut file = InputFile::new(absolute.clone());
            let mut checker = Checker::new(Vec::new(), None);
            file.check_deserialization(&mut checker).unwrap();
            assert_eq!(file.path, absolute);
        }

        // Absolute path fail.
        {
            let absolute = path.join("dir0/file_c.txt");
            let mut file = InputFile::new(absolute.clone());
            let mut checker = Checker::new(Vec::new(), None);
            let err = file.check_deserialization(&mut checker).unwrap_err();
            let message = err.to_string();
            assert!(message.contains("input file with absolute path"));
            assert!(message.contains("either does not exist or is not a file"));
        }

        // Directory search
        {
            let mut checker = Checker::new(
                vec![path.join("dir1/dir0"), path.join("dir1"), path.join("dir0")],
                None,
            );

            // Directories are searched in order.
            let mut file = InputFile::new("file_c.txt");
            file.check_deserialization(&mut checker).unwrap();
            assert_eq!(file.path, path.join("dir1/dir0/file_c.txt"));

            let mut file = InputFile::new("file_b.txt");
            file.check_deserialization(&mut checker).unwrap();
            assert_eq!(file.path, path.join("dir0/file_b.txt"));

            // Directory search can fail.
            let mut file = InputFile::new("file_a.txt");
            let err = file.check_deserialization(&mut checker).unwrap_err();
            let message = err.to_string();
            assert!(message.contains("could not find input file"));
            assert!(message.contains("in the search directories"));

            // If we give an absolute path, no directory search is performed.
            let mut file = InputFile::new(path.join("file_c.txt"));
            let err = file.check_deserialization(&mut checker).unwrap_err();
            let message = err.to_string();
            assert!(message.starts_with("input file with absolute path"));
        }
    }
}
