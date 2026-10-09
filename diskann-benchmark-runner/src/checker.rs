/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::path::{Component, Path, PathBuf};

use anyhow::Context;

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

    /// Registered outputs.
    outputs: Vec<(PathBuf, Kind)>,
}

impl Checker {
    /// Create a checker with an optional absolute output root. The root need not exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the output root is relative.
    pub(crate) fn new(
        search_directories: Vec<PathBuf>,
        output_directory: Option<PathBuf>,
    ) -> anyhow::Result<Self> {
        if let Some(path) = &output_directory {
            anyhow::ensure!(
                path.is_absolute(),
                "output directory \"{}\" must be absolute",
                path.display()
            );
        }
        Ok(Self {
            search_directories,
            output_directory,
            outputs: Vec::new(),
        })
    }

    /// Return the ordered list of search directories registered with the [`Checker`].
    pub fn search_directories(&self) -> &[PathBuf] {
        &self.search_directories
    }

    /// Return the output directory registered with the [`Checker`], if any.
    pub fn output_directory(&self) -> Option<&PathBuf> {
        self.output_directory.as_ref()
    }

    /// Resolve and reserve an output file, returning its absolute path.
    ///
    /// If `path` is absolute then it is canonicalized and returned, following symlinks as
    /// necessary. This canonicalization does not require the path or its intermediate
    /// directories (except symlink targets) to exist.
    ///
    /// If `path` is relative, then it is appended to [`Self::output_directory`] and follows
    /// the same canonicalization procedure.
    ///
    /// No files or directories are created during this process.
    ///
    /// In addition paths are checked for uniqueness. Neither files nor directories can have
    /// the same resolved path as, be ancestors or, or be decendents of previously registered
    /// outputs. For example, registering `/some/absolute/path` will prevent
    /// `/some/absolute/path/deeper` from being registered and vice versa.
    ///
    /// Registration does not check whether the existing file or directory exists, nor whether
    /// it may be overwritten.
    ///
    /// # Errors
    ///
    /// Returns an error in the following cases:
    ///
    /// * There is a conflict as described above.
    /// * `path` is empty, or is relative and [`Self::output_directory`] is `None`.
    /// * A file system error occurred during canonicalization. An example of this would be
    ///   failing to resolve symbolic links.
    pub fn register_output_file<P>(&mut self, path: P) -> anyhow::Result<PathBuf>
    where
        P: AsRef<Path>,
    {
        self.register_output_path(path.as_ref(), Kind::File)
    }

    /// Resolve and reserve an output directory and its entire subtree.
    ///
    /// Path resolution and existence handling follow [`Self::register_output_file`].
    ///
    /// # Errors
    ///
    /// Returns an error for invalid or unresolvable paths or conflicting output claims.
    ///
    /// See also: [`Self::register_output_file`].
    pub fn register_output_dir<P>(&mut self, path: P) -> anyhow::Result<PathBuf>
    where
        P: AsRef<Path>,
    {
        self.register_output_path(path.as_ref(), Kind::Dir)
    }

    fn register_output_path(&mut self, path: &Path, kind: Kind) -> anyhow::Result<PathBuf> {
        anyhow::ensure!(
            !path.as_os_str().is_empty(),
            "output path must not be empty"
        );
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.output_directory()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "relative output path \"{}\" specified but no output directory was provided",
                        path.display()
                    )
                })?
                .join(path)
        };
        let resolved = resolve_output_path(&path)?;
        self.reserve_output(&resolved, kind)?;
        Ok(resolved)
    }

    // This is not particularly efficient as we check the whole set of outputs every time.
    //
    // However, we don't expect there to be *too* many of these, so we can deal with it.
    fn reserve_output(&mut self, path: &Path, kind: Kind) -> anyhow::Result<()> {
        for (registered, registered_kind) in self.outputs.iter() {
            if path.starts_with(registered) || registered.starts_with(path) {
                anyhow::bail!(
                    "output {} \"{}\" conflicts with registered output {} \"{}\"",
                    kind.as_str(),
                    path.display(),
                    registered_kind.as_str(),
                    registered.display()
                );
            }
        }
        self.outputs.push((path.to_path_buf(), kind));
        Ok(())
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
}

fn resolve_output_path(path: &Path) -> anyhow::Result<PathBuf> {
    anyhow::ensure!(
        path.is_absolute(),
        "output path \"{}\" must be absolute",
        path.display()
    );

    let mut resolved = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => resolved.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            Component::Normal(name) => {
                resolved.push(name);
                match std::fs::symlink_metadata(&resolved) {
                    Ok(_) => {
                        resolved = std::fs::canonicalize(&resolved).with_context(|| {
                            format!("while resolving output path \"{}\"", resolved.display())
                        })?;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => {
                        return Err(error).with_context(|| {
                            format!("while resolving output path \"{}\"", resolved.display())
                        });
                    }
                }
            }
        }
    }
    Ok(resolved)
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

    use std::fs::{File, create_dir};

    use diskann_utils::assert_contains;

    #[test]
    fn test_constructor() {
        let checker = Checker::new(Vec::new(), None).unwrap();
        assert!(checker.search_directories().is_empty());
        assert!(checker.output_directory().is_none());

        let dir_a: PathBuf = "directory/a".into();
        let dir_b: PathBuf = "directory/another/b".into();

        assert_contains!(
            Checker::new(vec![dir_a.clone()], Some(dir_b.clone()))
                .unwrap_err()
                .to_string(),
            format!("output directory \"{}\" must be absolute", dir_b.display()),
        );
        assert_contains!(
            Checker::new(Vec::new(), Some(PathBuf::new()))
                .unwrap_err()
                .to_string(),
            "output directory \"\" must be absolute",
        );

        let absolute = std::path::absolute(&dir_b).unwrap();
        let checker = Checker::new(vec![dir_a.clone()], Some(absolute.clone())).unwrap();
        assert_eq!(checker.search_directories(), vec![dir_a.clone()]);
        assert_eq!(checker.output_directory(), Some(&absolute));

        let checker = Checker::new(vec![dir_a.clone(), dir_b.clone()], None).unwrap();
        assert_eq!(
            checker.search_directories(),
            vec![dir_a.clone(), dir_b.clone()]
        );
        assert!(checker.output_directory().is_none());

        // We don't require the output directory to be present at checker creation.
        let dir = tempfile::tempdir().unwrap();
        let missing_root = dir.path().join("missing-output-root");
        let checker = Checker::new(Vec::new(), Some(missing_root.clone())).unwrap();
        assert_eq!(checker.output_directory(), Some(&missing_root));
        assert!(!missing_root.exists());
    }

    fn output_path_error(path: impl AsRef<Path>) -> String {
        resolve_output_path(path.as_ref()).unwrap_err().to_string()
    }

    fn output_dir_error(checker: &mut Checker, dir: impl AsRef<Path>) -> String {
        checker.register_output_dir(dir).unwrap_err().to_string()
    }

    fn output_file_error(checker: &mut Checker, file: impl AsRef<Path>) -> String {
        checker.register_output_file(file).unwrap_err().to_string()
    }

    fn conflict_message(
        (kind, path): (Kind, impl AsRef<Path>),
        (registered_kind, registered): (Kind, impl AsRef<Path>),
    ) -> String {
        format!(
            "output {} \"{}\" conflicts with registered output {} \"{}\"",
            kind.as_str(),
            path.as_ref().display(),
            registered_kind.as_str(),
            registered.as_ref().display(),
        )
    }

    #[test]
    fn test_resolve_output_path_requires_absolute() {
        assert_contains!(
            output_path_error("relative/file"),
            format!(
                "output path \"{}\" must be absolute",
                Path::new("relative/file").display(),
            ),
        );
    }

    #[test]
    fn test_register_output_resolution() {
        let dir = tempfile::tempdir().unwrap();

        // The missing root helps us check that we support situations where the root path
        // for saving does exist during creation.
        let root = dir.path().canonicalize().unwrap().join("missing-root");
        let mut checker = Checker::new(Vec::new(), Some(root.clone())).unwrap();

        // Relaive paths appended to the current root.
        assert_eq!(
            checker.register_output_file("nested/file").unwrap(),
            root.join("nested/file"),
        );
        assert_eq!(
            checker.register_output_dir("other").unwrap(),
            root.join("other"),
        );
        assert!(!root.join("nested").exists());
        assert!(!root.join("other").exists());

        // Absolute paths skip the output directory entirely.
        let mut checker = Checker::new(Vec::new(), None).unwrap();
        assert_eq!(
            checker.register_output_file(root.join("absolute")).unwrap(),
            root.join("absolute"),
        );

        // Errors
        assert_contains!(
            output_file_error(&mut checker, "relative"),
            "relative output path \"relative\" specified but no output directory was provided",
        );
        assert_contains!(
            output_dir_error(&mut checker, "relative"),
            "relative output path \"relative\" specified but no output directory was provided",
        );
        assert_contains!(
            output_file_error(&mut checker, ""),
            "output path must not be empty",
        );
        assert_contains!(
            output_dir_error(&mut checker, ""),
            "output path must not be empty",
        );
    }

    #[test]
    fn test_register_output_conflicts() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let mut checker = Checker::new(Vec::new(), Some(root.clone())).unwrap();

        // Register the following:
        //
        // <root>/job/a
        // <root>/job/b
        //
        // Some tests
        //
        // <root>/job/c
        //
        // This should block trying to reserve the `job` directory entirely, as well as
        // redundant file, dir, or sub-dir conflicts.

        checker.register_output_file("job/a").unwrap();
        checker.register_output_file("job/b").unwrap();
        assert_contains!(
            output_file_error(&mut checker, "job/a"),
            conflict_message(
                (Kind::File, root.join("job/a")),
                (Kind::File, root.join("job/a")),
            ),
        );
        assert_contains!(
            output_file_error(&mut checker, "job/a/b"),
            conflict_message(
                (Kind::File, root.join("job/a/b")),
                (Kind::File, root.join("job/a")),
            ),
        );
        assert_contains!(
            output_dir_error(&mut checker, "job/a"),
            conflict_message(
                (Kind::Dir, root.join("job/a")),
                (Kind::File, root.join("job/a")),
            ),
        );
        assert_contains!(
            output_dir_error(&mut checker, "job/a/b"),
            conflict_message(
                (Kind::Dir, root.join("job/a/b")),
                (Kind::File, root.join("job/a")),
            ),
        );
        assert_contains!(
            output_dir_error(&mut checker, "job"),
            conflict_message(
                (Kind::Dir, root.join("job")),
                (Kind::File, root.join("job/a")),
            ),
        );
        assert_contains!(
            output_dir_error(&mut checker, "."),
            conflict_message((Kind::Dir, &root), (Kind::File, root.join("job/a")),),
        );

        // We can still succeed if we are disjoint.
        checker.register_output_file("job/c").unwrap();

        // Here, we check that directories claim all nested paths.
        let mut checker = Checker::new(Vec::new(), Some(root.clone())).unwrap();
        checker.register_output_dir("job").unwrap();
        assert_contains!(
            output_file_error(&mut checker, "job/../job"),
            conflict_message(
                (Kind::File, root.join("job")),
                (Kind::Dir, root.join("job")),
            ),
        );
        assert_contains!(
            output_file_error(&mut checker, "job/a/../../job/a"),
            conflict_message(
                (Kind::File, root.join("job/a")),
                (Kind::Dir, root.join("job")),
            ),
        );
        assert_contains!(
            output_dir_error(&mut checker, "job/nested"),
            conflict_message(
                (Kind::Dir, root.join("job/nested")),
                (Kind::Dir, root.join("job")),
            ),
        );
        assert_contains!(
            output_dir_error(&mut checker, "."),
            conflict_message((Kind::Dir, &root), (Kind::Dir, root.join("job")),),
        );
        checker.register_output_dir("job-other").unwrap();
    }

    #[test]
    fn test_register_output_doesnt_need_existence() {
        let dir = tempfile::tempdir().unwrap();
        File::create(dir.path().join("existing-file")).unwrap();
        create_dir(dir.path().join("existing-dir")).unwrap();

        let mut checker = Checker::new(Vec::new(), Some(dir.path().to_path_buf())).unwrap();
        checker.register_output_file("existing-dir").unwrap();
        checker.register_output_dir("existing-file").unwrap();
        assert!(dir.path().join("existing-dir").is_dir());
        assert!(dir.path().join("existing-file").is_file());
    }

    #[test]
    fn test_register_output_parent_components() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let mut checker = Checker::new(Vec::new(), Some(root.clone())).unwrap();
        assert_eq!(
            checker.register_output_file("missing/../file").unwrap(),
            root.join("file"),
        );
        assert_contains!(
            output_file_error(&mut checker, "./file"),
            conflict_message(
                (Kind::File, root.join("file")),
                (Kind::File, root.join("file")),
            ),
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_register_output_symlinks() {
        use std::os::unix::fs::symlink;

        // We create the following directory relative to root:
        //
        // real/
        //   nested/
        //
        // alias (real)
        // nested_alias (real/nested)
        // missing (broken)

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        create_dir(root.join("real")).unwrap();
        create_dir(root.join("real/nested")).unwrap();
        symlink(root.join("real"), root.join("alias")).unwrap();
        symlink(root.join("real/nested"), root.join("nested-alias")).unwrap();
        symlink(root.join("missing"), root.join("broken")).unwrap();

        let mut checker = Checker::new(Vec::new(), Some(root.clone())).unwrap();
        checker.register_output_file("real/file").unwrap();

        // Check that we follow the sym-link from `alias` to `real`.
        assert_contains!(
            output_file_error(&mut checker, "alias/file"),
            conflict_message(
                (Kind::File, root.join("real/file")),
                (Kind::File, root.join("real/file")),
            ),
        );

        // Path resolves to `real/file`.
        assert_contains!(
            output_file_error(&mut checker, "nested-alias/../file"),
            conflict_message(
                (Kind::File, root.join("real/file")),
                (Kind::File, root.join("real/file")),
            ),
        );

        // Path resolves to `real`.
        assert_contains!(
            output_dir_error(&mut checker, "alias"),
            conflict_message(
                (Kind::Dir, root.join("real")),
                (Kind::File, root.join("real/file")),
            ),
        );

        // Symlink broken.
        assert_contains!(
            output_file_error(&mut checker, "broken/file"),
            format!(
                "while resolving output path \"{}\"",
                root.join("broken").display()
            ),
        );

        // Creating through symlinks is allowed.
        assert_eq!(
            checker.register_output_dir("alias/new").unwrap(),
            root.join("real/new"),
        );
        assert!(!root.join("real/new").exists());
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

        let make_checker =
            |paths: &[PathBuf]| -> Checker { Checker::new(paths.to_vec(), None).unwrap() };

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
            assert_contains!(message, "input file with absolute path");
            assert_contains!(message, "either does not exist or is not a file");
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
            assert_contains!(message, "could not find input file");
            assert_contains!(message, "in the search directories");

            // If we give an absolute path, no directory search is performed.
            let file = path.join("file_c.txt");
            let err = checker.find_input_file(&file).unwrap_err();
            let message = err.to_string();
            assert_contains!(message, "input file with absolute path");
        }
    }

    #[test]
    fn test_find_input_dir() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        create_test_directory(path);

        let make_checker =
            |paths: &[PathBuf]| -> Checker { Checker::new(paths.to_vec(), None).unwrap() };

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
            assert_contains!(message, "input directory with absolute path");
            assert_contains!(message, "either does not exist or is not a directory");

            // Files are rejected.
            let absolute = path.join("file_a.txt");
            let err = checker.find_input_dir(&absolute).unwrap_err();
            let message = err.to_string();
            assert_contains!(message, "input directory with absolute path");
            assert_contains!(message, "either does not exist or is not a directory");
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
            assert_contains!(message, "could not find input directory");
            assert_contains!(message, "in the search directories");

            // If we give an absolute path, no directory search is performed.
            let dir = path.join("dir2");
            let err = checker.find_input_dir(&dir).unwrap_err();
            let message = err.to_string();
            assert_contains!(message, "input directory with absolute path");
        }
    }
}
