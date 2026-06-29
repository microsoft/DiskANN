/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Disk-backed save/load contexts.
//!
//! [`DiskSaveContext`] writes the manifest as JSON plus side-car artifact files into a
//! directory; [`DiskLoadContext`] reads them back. The two halves are independent and
//! communicate only through the filesystem. Both are available under the `disk` feature.

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    sync::Mutex,
};

use crate::{
    load::{self, LoadContext, Reader},
    save::{self, Handle, SaveContext, Value, Writer, delegate_write_and_seek},
};

/// The disk-backed [`SaveContext`].
///
/// Holds the manifest directory, the manifest path, and the artifact file names registered
/// so far paired with whether their writer has finished. Lookup and insertion go through a
/// [`Mutex`] so that concurrent [`Save`](crate::save::Save) impls cannot accidentally hand
/// out the same artifact name twice.
///
/// # Cleanup on failure
///
/// Save can fail part-way, so the [`Drop`] impl ensures cleanup of any artifacts created
/// before the failure.
#[derive(Debug)]
pub(crate) struct DiskSaveContext {
    dir: PathBuf,
    metadata: PathBuf,
    files: Mutex<HashMap<String, bool>>,
    committed: bool,
}

#[derive(serde::Serialize)]
struct Final<'a> {
    files: Vec<&'a str>,
    value: &'a Value<'a>,
}

impl DiskSaveContext {
    /// Create a disk-backed save context targeting `dir` for side-car artifacts and
    /// `metadata` for the manifest. Validates that `dir` is an actual directory.
    ///
    /// # Errors
    ///
    /// Returns [`save::Error`] if `dir` does not exist, cannot be inspected, or exists but
    /// is not a directory.
    pub(crate) fn new(dir: PathBuf, metadata: PathBuf) -> save::Result<Self> {
        match std::fs::metadata(&dir) {
            Ok(meta) if meta.is_dir() => {}
            Ok(_) => {
                return Err(save::Error::message(format!(
                    "path {} exists but is not a directory",
                    dir.display()
                )));
            }
            Err(err) => {
                return Err(save::Error::new(err)
                    .context(format!("while validating path {}", dir.display())));
            }
        }

        Ok(Self {
            dir,
            metadata,
            files: Mutex::new(HashMap::new()),
            committed: false,
        })
    }

    /// Path of the temp manifest written by [`SaveContext::finish`] before the atomic
    /// rename into [`Self::metadata`].
    fn temp_metadata(&self) -> PathBuf {
        let mut temp = self.metadata.clone().into_os_string();
        temp.push(".temp");
        PathBuf::from(temp)
    }
}

impl Drop for DiskSaveContext {
    /// Best-effort cleanup for an uncommitted save.
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        let files = self
            .files
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        for name in files.keys() {
            let _ = std::fs::remove_file(self.dir.join(name));
        }
        let _ = std::fs::remove_file(self.temp_metadata());
    }
}

impl SaveContext for DiskSaveContext {
    type Output = ();

    fn write(&self, key: Option<&str>) -> save::Result<Writer<'_>> {
        // When a human-readable hint is supplied it must be a simple relative file name.
        // NOTE:: Absolute paths, parent traversal, and multi-component paths cannot produce a
        // single, well-formed file name in the manifest directory, so they are ignored and
        // treated as if no hint had been supplied.
        let key = key.filter(|key| {
            let mut components = std::path::Path::new(key).components();
            matches!(components.next(), Some(std::path::Component::Normal(_)))
                && components.next().is_none()
        });

        let mut files = self
            .files
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        // Prefix each artifact with the count of artifacts written so far, so that reusing
        // the same `key` (or omitting it) still yields a unique file name.
        let name = match key {
            Some(key) => format!("{:03}-{}", files.len(), key),
            None => format!("{:03}", files.len()),
        };

        if files.contains_key(&name) {
            return Err(save::Error::message(format!(
                "generated artifact name {:?} collides with an existing artifact",
                name,
            )));
        }
        let full = self.dir.join(&name);
        if full.exists() {
            return Err(save::Error::message(format!(
                "file {} already exists",
                full.display()
            )));
        }
        let file = std::fs::File::create_new(&full).map_err(|err| {
            save::Error::new(err).context(format!("while creating new file {}", full.display()))
        })?;
        // Reserve the name as not-yet-finished; `FileWriter::finish` flips it to `true`, and
        // `SaveContext::finish` reports any slot that was reserved but never finished.
        files.insert(name.clone(), false);
        Ok(Writer::new(FileWriter { file, parent: self }, name))
    }

    /// Finalize the manifest.
    ///
    /// Writes the manifest JSON atomically: serializes to a `<metadata>.temp` file first,
    /// then renames it into place. Fails if the temp file already exists (an in-flight
    /// save is in progress, or a previous run aborted between rename steps).
    ///
    /// On failure, context is dropped without committing ==> [`Drop`] impl
    /// removes the artifacts + temp manifest. Save is marked committed once
    /// rename succeeds and artifacts are in place.
    fn finish(mut self, value: Value<'_>) -> save::Result<()> {
        let temp = self.temp_metadata();
        {
            let files = self
                .files
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if let Some((name, _)) = files.iter().find(|(_, finished)| !**finished) {
                return Err(save::Error::message(format!(
                    "artifact {:?} was reserved but never finished",
                    name,
                )));
            }
            let f = Final {
                files: files.keys().map(|k| &**k).collect(),
                value: &value,
            };

            // Fail if the temp file already exists
            let buffer = std::fs::File::create_new(&temp).map_err(|err| {
                if err.kind() == std::io::ErrorKind::AlreadyExists {
                    save::Error::message(format!(
                        "Temporary file {} already exists. Aborting!",
                        temp.display()
                    ))
                } else {
                    save::Error::new(err).context(format!(
                        "while creating temp manifest file {}",
                        temp.display()
                    ))
                }
            })?;

            serde_json::to_writer_pretty(buffer, &f).map_err(|err| {
                save::Error::new(err).context("while serializing manifest to JSON")
            })?;
        }
        std::fs::rename(&temp, &self.metadata).map_err(|err| {
            save::Error::new(err).context(format!(
                "while renaming temp manifest {} to final path {}",
                temp.display(),
                self.metadata.display()
            ))
        })?;
        // Manifest now in place, artifacts belong to a valid record
        self.committed = true;
        Ok(())
    }
}

/// A file-backed [`WriterInner`](save::WriterInner) that streams bytes straight to disk.
///
/// The bytes are persisted as they are written; [`WriterInner::finish`](save::WriterInner::finish)
/// only needs to mint the [`Handle`] and mark the artifact finished in its parent context
/// (the buffered bytes are already flushed into the file by [`Writer::finish`]).
#[derive(Debug)]
struct FileWriter<'a> {
    file: File,
    parent: &'a DiskSaveContext,
}

impl save::WriterInner for FileWriter<'_> {
    fn finish(self: Box<Self>, name: String) -> save::Result<Handle> {
        self.parent
            .files
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(name.clone(), true);
        Ok(Handle::new(name))
    }
}

delegate_write_and_seek!(file, FileWriter<'_>);

/// The disk-backed [`LoadContext`].
///
/// Reads the manifest produced by [`DiskSaveContext`] and resolves side-car artifact
/// handles against the manifest directory.
#[derive(Debug)]
pub(crate) struct DiskLoadContext {
    dir: PathBuf,
    files: HashSet<PathBuf>,
    value: Value<'static>,
}

#[derive(serde::Deserialize)]
struct FileRepr {
    files: HashSet<PathBuf>,
    value: Value<'static>,
}

impl DiskLoadContext {
    pub(crate) fn new(metadata: &Path, dir: &Path) -> load::Result<Self> {
        let file = std::fs::File::open(metadata).map_err(|e| {
            load::Error::new(e).context(format!("while trying to open {}", metadata.display()))
        })?;

        let reader = BufReader::new(file);
        let repr: FileRepr = serde_json::from_reader(reader)
            .map_err(|e| load::Error::new(e).context("could not deserialize manifest"))?;

        Ok(Self {
            dir: dir.into(),
            files: repr.files,
            value: repr.value,
        })
    }
}

impl LoadContext for DiskLoadContext {
    fn value(&self) -> load::Result<&Value<'_>> {
        Ok(&self.value)
    }

    fn read(&self, key: &str) -> load::Result<Reader<'_>> {
        let key_as_path: &Path = key.as_ref();
        let mut components = key_as_path.components();
        match components.next() {
            Some(std::path::Component::Normal(_)) if components.next().is_none() => {}
            _ => {
                return Err(
                    load::Error::from(load::error::Kind::MissingFile).context(format!(
                        "handle references file {:?} which escapes the manifest directory",
                        key,
                    )),
                );
            }
        }
        if !self.files.contains(key_as_path) {
            return Err(
                load::Error::from(load::error::Kind::MissingFile).context(format!(
                    "handle references file {:?} which is not registered in the manifest",
                    key,
                )),
            );
        }

        let full = self.dir.join(key);
        let file = std::fs::File::open(&full).map_err(|err| {
            load::Error::new(err).context(format!("while opening artifact file {}", full.display()))
        })?;

        Ok(Reader::new(file))
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;

    #[test]
    fn new_rejects_nonexistent_directory() {
        let missing = PathBuf::from("does/not/exist/anywhere/at/all");
        let err = DiskSaveContext::new(missing, "meta.json".into())
            .expect_err("a nonexistent directory must be rejected");
        assert!(format!("{err}").contains("while validating path"));
    }

    #[test]
    fn new_rejects_file_as_directory() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("not_a_dir");
        std::fs::write(&file, b"hi").unwrap();
        let err = DiskSaveContext::new(file, dir.path().join("meta.json"))
            .expect_err("a file path must be rejected as a directory");
        assert!(format!("{err}").contains("is not a directory"));
    }

    #[test]
    fn write_ignores_path_separators_and_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = DiskSaveContext::new(dir.path().into(), dir.path().join("meta.json")).unwrap();
        for bad in ["sub/dir.bin", "../escape.bin", "/abs.bin"] {
            let handle = SaveContext::write(&ctx, Some(bad))
                .expect("keys with path separators are treated as anonymous")
                .finish()
                .unwrap();
            let mut components = std::path::Path::new(handle.as_str()).components();
            assert!(
                matches!(components.next(), Some(std::path::Component::Normal(_)))
                    && components.next().is_none(),
                "generated name {:?} must be a single relative file name",
                handle.as_str(),
            );
        }
    }

    #[test]
    fn write_allows_duplicate_key() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = DiskSaveContext::new(dir.path().into(), dir.path().join("meta.json")).unwrap();
        let first = SaveContext::write(&ctx, Some("artifact.bin"))
            .unwrap()
            .finish()
            .unwrap();
        let second = SaveContext::write(&ctx, Some("artifact.bin"))
            .unwrap()
            .finish()
            .unwrap();
        assert_ne!(
            first.as_str(),
            second.as_str(),
            "duplicate keys must be disambiguated by the count prefix"
        );
        assert_eq!(first.as_str(), "000-artifact.bin");
        assert_eq!(second.as_str(), "001-artifact.bin");
    }

    #[test]
    fn write_allows_anonymous_artifact() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = DiskSaveContext::new(dir.path().into(), dir.path().join("meta.json")).unwrap();
        let handle = SaveContext::write(&ctx, None).unwrap().finish().unwrap();
        assert!(!handle.as_str().is_empty());
    }

    #[test]
    fn drop_without_finish_cleans_up_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        let metadata = dir.path().join("meta.json");
        let name = {
            let ctx = DiskSaveContext::new(dir.path().into(), metadata.clone()).unwrap();
            let name = SaveContext::write(&ctx, Some("artifact.bin"))
                .unwrap()
                .finish()
                .unwrap()
                .as_str()
                .to_owned();
            assert!(dir.path().join(&name).exists());
            name
            // `ctx` is dropped here without ever being committed via `finish`.
        };
        assert!(
            !dir.path().join(&name).exists(),
            "an uncommitted save must clean up the artifacts it created"
        );
    }

    #[test]
    fn failed_finish_cleans_up_artifacts_and_temp() {
        let dir = tempfile::tempdir().unwrap();
        let metadata = dir.path().join("meta.json");
        let ctx = DiskSaveContext::new(dir.path().into(), metadata.clone()).unwrap();
        let name = SaveContext::write(&ctx, Some("artifact.bin"))
            .unwrap()
            .finish()
            .unwrap()
            .as_str()
            .to_owned();

        // Pre-create the temp manifest so `finish` aborts on the `create_new` collision.
        let temp = ctx.temp_metadata();
        std::fs::write(&temp, b"stale").unwrap();

        let err = ctx
            .finish(Value::Null)
            .expect_err("finish must fail when the temp manifest already exists");
        assert!(format!("{err}").contains("already exists"));

        assert!(
            !dir.path().join(&name).exists(),
            "a failed finish must clean up the artifacts it created"
        );
        assert!(
            !metadata.exists(),
            "a failed finish must not leave a committed manifest"
        );
    }

    #[test]
    fn committed_save_preserves_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        let metadata = dir.path().join("meta.json");
        let ctx = DiskSaveContext::new(dir.path().into(), metadata.clone()).unwrap();
        let name = SaveContext::write(&ctx, Some("artifact.bin"))
            .unwrap()
            .finish()
            .unwrap()
            .as_str()
            .to_owned();

        ctx.finish(Value::Null).unwrap();

        assert!(
            dir.path().join(&name).exists(),
            "a committed save must keep its artifacts"
        );
        assert!(
            metadata.exists(),
            "a committed save must write the manifest"
        );
        assert!(
            !ctx_temp(&metadata).exists(),
            "a committed save must not leave a temp manifest"
        );
    }

    fn ctx_temp(metadata: &Path) -> PathBuf {
        let mut temp = metadata.to_owned().into_os_string();
        temp.push(".temp");
        PathBuf::from(temp)
    }

    fn write_manifest(dir: &Path, files: &[&str]) -> PathBuf {
        let manifest = serde_json::json!({
            "files": files,
            "value": { "$version": "0.0.0" },
        });
        let metadata = dir.join("metadata.json");
        std::fs::write(&metadata, serde_json::to_vec(&manifest).unwrap()).unwrap();
        metadata
    }

    #[test]
    fn read_rejects_unregistered_file() {
        let dir = tempfile::tempdir().unwrap();
        let metadata = write_manifest(dir.path(), &[]);
        let ctx = DiskLoadContext::new(&metadata, dir.path()).unwrap();
        let Err(err) = ctx.read("artifact.bin") else {
            panic!("an unregistered file must be rejected");
        };
        assert!(format!("{err}").contains("not registered in the manifest"));
    }

    #[test]
    fn read_rejects_escaping_handle() {
        let dir = tempfile::tempdir().unwrap();
        // Register the escaping name so only the path-shape check can reject it.
        let metadata = write_manifest(dir.path(), &["../escape.bin"]);
        let ctx = DiskLoadContext::new(&metadata, dir.path()).unwrap();
        let Err(err) = ctx.read("../escape.bin") else {
            panic!("a handle escaping the manifest directory must be rejected");
        };
        assert!(format!("{err}").contains("escapes the manifest directory"));
    }
}
