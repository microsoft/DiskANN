/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! In-memory save/load contexts.
//!
//! [`MemorySaveContext`] and [`MemoryContext`] mirror the disk-backed contexts but
//! keep the manifest value and every side-car artifact in memory. Saving through an
//! [`MemorySaveContext`] yields an [`MemoryContext`] (its `SaveContext::Output`) that can
//! be loaded directly via the `load` entry point.
//!
//! Unlike the disk path, this round trip never serializes through JSON: the manifest
//! [`Value`] is deep-copied to `'static` via [`Value::into_owned`] and side-car artifacts
//! are buffered in memory through a [`std::io::Cursor`]. As a result these contexts are
//! available regardless of the `disk` / `serde` features.

use std::{collections::HashMap, io::Cursor, sync::Mutex};

use crate::{
    Value,
    load::{self, LoadContext, Reader},
    save::{self, Handle, SaveContext, Writer, delegate_write_and_seek},
};

/// A save-side `SaveContext` that keeps every side-car artifact and the committed
/// manifest value in memory.
///
/// `SaveContext::finish` consumes the context and returns an [`MemoryContext`] ready
/// to be loaded with the `load` entry point.
///
/// # Cleanup on failure
///
/// Failures in the same process are automatically cleaned up.
#[derive(Debug, Default)]
pub struct MemorySaveContext {
    files: Mutex<HashMap<String, Option<Vec<u8>>>>,
}

impl MemorySaveContext {
    /// Create an empty in-memory save context.
    pub fn new() -> Self {
        Self::default()
    }
}

impl SaveContext for MemorySaveContext {
    type Output = MemoryContext;

    fn write(&self, key: Option<&str>) -> save::Result<Writer<'_>> {
        // Mirror the disk context: a human-readable hint must be a simple relative file
        // name. Absolute paths, parent traversal, and multi-component paths cannot produce
        // a single, well-formed key, so they are ignored and treated as if no hint had been
        // supplied.
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
        // the same `key` (or omitting it) still yields a unique name.
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
        // Reserve the name with an empty slot so the count advances and concurrent writers
        // cannot collide; the slot is filled with the real bytes by `Writer::finish`. A slot
        // that is never filled is reported by `SaveContext::finish`.
        files.insert(name.clone(), None);
        drop(files);

        Ok(Writer::new(
            MemoryWriter {
                cursor: Cursor::new(Vec::new()),
                parent: self,
            },
            name,
        ))
    }

    fn finish(self, value: Value<'_>) -> save::Result<MemoryContext> {
        let files = self
            .files
            .into_inner()
            .unwrap_or_else(|poison| poison.into_inner());
        let files = files
            .into_iter()
            .map(|(name, bytes)| match bytes {
                Some(bytes) => Ok((name, bytes)),
                None => Err(save::Error::message(format!(
                    "artifact {:?} was reserved but never finished",
                    name,
                ))),
            })
            .collect::<save::Result<HashMap<_, _>>>()?;
        Ok(MemoryContext {
            files,
            value: value.into_owned(),
        })
    }
}

/// An in-memory [`WriterInner`](save::WriterInner) that buffers bytes in a [`Cursor`] and,
/// on finish, deposits the completed buffer into its parent context's file store.
#[derive(Debug)]
struct MemoryWriter<'a> {
    cursor: Cursor<Vec<u8>>,
    parent: &'a MemorySaveContext,
}

impl save::WriterInner for MemoryWriter<'_> {
    fn finish(self: Box<Self>, name: String) -> save::Result<Handle> {
        self.parent
            .files
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(name.clone(), Some(self.cursor.into_inner()));
        Ok(Handle::new(name))
    }
}

delegate_write_and_seek!(cursor, MemoryWriter<'_>);

/// A load-side `LoadContext` backed entirely by in-memory buffers.
///
/// Produced by [`MemorySaveContext`] via `SaveContext::finish`. Holds the committed
/// manifest [`Value`] and every side-car artifact as an in-memory byte buffer, so loading
/// never serializes through JSON or touches the filesystem.
#[derive(Debug)]
pub struct MemoryContext {
    files: HashMap<String, Vec<u8>>,
    value: Value<'static>,
}

impl LoadContext for MemoryContext {
    fn value(&self) -> load::Result<&Value<'_>> {
        Ok(&self.value)
    }

    fn read(&self, key: &str) -> load::Result<Reader<'_>> {
        match self.files.get(key) {
            Some(bytes) => Ok(Reader::new(Cursor::new(bytes.as_slice()))),
            None => Err(
                load::Error::from(load::error::Kind::MissingFile).context(format!(
                    "handle references artifact {:?} which is not registered in this context",
                    key,
                )),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        marker::PhantomData,
    };

    use super::*;
    use crate::{Version, load, save};

    #[derive(Debug, PartialEq)]
    struct Doc {
        name: String,
        blob: Vec<u8>,
    }

    impl save::Save for Doc {
        const VERSION: Version = Version::new(1, 0);
        fn save(&self, context: save::Context<'_>) -> save::Result<save::Record<'_>> {
            let mut io = context.write(Some("blob.bin"))?;
            io.write_all(&self.blob).map_err(save::Error::new)?;
            let mut record = crate::save_fields!(self, context, [name]);
            record.insert("blob", io.finish()?)?;
            Ok(record)
        }
    }

    impl load::Load<'_> for Doc {
        const VERSION: Version = Version::new(1, 0);
        fn load(object: load::Object<'_>) -> load::Result<Self> {
            crate::load_fields!(object, [name: String, blob: save::Handle]);
            let mut io = object.read(&blob)?;
            let mut blob = Vec::new();
            io.read_to_end(&mut blob).map_err(load::Error::new)?;
            Ok(Self { name, blob })
        }
        fn load_legacy(_: load::Object<'_>) -> load::Result<Self> {
            Err(load::error::Kind::UnknownVersion.into())
        }
    }

    struct DocDto<'a> {
        name: &'a str,
        blob: save::Handle,
        object: load::Object<'a>,
    }

    impl<'a> load::Load<'a> for DocDto<'a> {
        const VERSION: Version = <Doc as load::Load<'a>>::VERSION;

        fn load(object: load::Object<'a>) -> load::Result<Self> {
            crate::load_fields!(object, [name, blob]);
            Ok(Self { name, blob, object })
        }

        fn load_legacy(_: load::Object<'a>) -> load::Result<Self> {
            Err(load::error::Kind::UnknownVersion.into())
        }
    }

    struct DocLoader<'r> {
        blob: &'r mut Vec<u8>,
    }

    impl<'a, 'r> load::Loader<'a> for DocLoader<'r> {
        type Value = (String, &'r mut Vec<u8>);

        fn load(self, context: load::Context<'a>) -> load::Result<Self::Value> {
            let dto: DocDto<'a> = context.load()?;
            dto.object
                .read(&dto.blob)?
                .read_to_end(self.blob)
                .map_err(load::Error::new)?;
            Ok((dto.name.to_owned(), self.blob))
        }
    }

    #[test]
    fn round_trips_without_serde_or_disk() {
        let doc = Doc {
            name: "example".to_owned(),
            blob: vec![1, 2, 3, 4, 5],
        };

        let context = save::save(&doc, MemorySaveContext::new()).unwrap();
        let restored: Doc = load::load(&context).unwrap();
        let with_loader = load::load_with(&context, PhantomData::<Doc>).unwrap();

        assert_eq!(doc, restored);
        assert_eq!(restored, with_loader);
    }

    #[test]
    fn loader_reads_borrowed_dto_into_caller_resource() {
        let doc = Doc {
            name: "example".to_owned(),
            blob: vec![1, 2, 3, 4, 5],
        };
        let context = save::save(&doc, MemorySaveContext::new()).unwrap();
        let dto = load::load_with(&context, PhantomData::<DocDto<'_>>).unwrap();
        assert_eq!(dto.name, doc.name);

        let mut blob = vec![0];
        let (name, loaded) = load::load_with(&context, DocLoader { blob: &mut blob }).unwrap();
        assert_eq!(name, doc.name);
        assert_eq!(*loaded, [0, 1, 2, 3, 4, 5]);
        loaded.push(6);
        assert_eq!(blob, [0, 1, 2, 3, 4, 5, 6]);
    }

    #[cfg(feature = "disk")]
    #[test]
    fn disk_loader_result_borrows_caller_resource_not_metadata() -> anyhow::Result<()> {
        let doc = Doc {
            name: "example".to_owned(),
            blob: vec![1, 2, 3, 4, 5],
        };
        let temp_dir = tempfile::tempdir()?;
        let dir = temp_dir.path();
        let metadata = dir.join("metadata.json");
        save::save_to_disk(&doc, dir, &metadata)?;
        let restored: Doc = load::load_from_disk(&metadata, dir)?;
        let with_loader = load::load_from_disk_with(&metadata, dir, PhantomData::<Doc>)?;
        assert_eq!(restored, with_loader);

        let mut blob = vec![0];
        let (name, loaded) =
            load::load_from_disk_with(&metadata, dir, DocLoader { blob: &mut blob })?;
        drop(temp_dir);
        assert_eq!(name, doc.name);
        assert_eq!(*loaded, [0, 1, 2, 3, 4, 5]);
        loaded.push(6);
        assert_eq!(blob, [0, 1, 2, 3, 4, 5, 6]);
        Ok(())
    }

    #[test]
    fn read_rejects_unregistered_artifact() {
        let context = MemoryContext {
            files: HashMap::new(),
            value: Value::Null,
        };
        let err = context
            .read("missing.bin")
            .err()
            .expect("an unregistered artifact must be rejected");
        assert!(format!("{err}").contains("not registered in this context"));
    }

    #[test]
    fn write_rejects_name_collision() {
        let ctx = MemorySaveContext::new();
        // The count prefix normally makes a collision impossible, so seed the bookkeeping map
        // directly with the exact name the next `write` will generate (one entry => count 1 =>
        // `001-artifact.bin`).
        ctx.files
            .lock()
            .unwrap()
            .insert("001-artifact.bin".to_string(), None);
        let err = SaveContext::write(&ctx, Some("artifact.bin"))
            .expect_err("a generated name that is already registered must be rejected");
        assert!(format!("{err}").contains("collides with an existing artifact"));
    }

    #[test]
    fn finish_rejects_unfinished_artifact() {
        let ctx = MemorySaveContext::new();
        // Reserve an artifact slot but drop the writer without calling `finish`, leaving the
        // slot empty.
        let writer = SaveContext::write(&ctx, Some("artifact.bin")).unwrap();
        drop(writer);
        let err = ctx
            .finish(Value::Null)
            .expect_err("finish must fail when an artifact was reserved but never finished");
        assert!(format!("{err}").contains("was reserved but never finished"));
    }

    #[test]
    fn write_names_anonymous_artifact_with_count_prefix() {
        let ctx = MemorySaveContext::new();
        // Passing `None` as the hint exercises the count-only naming branch.
        let handle = SaveContext::write(&ctx, None).unwrap().finish().unwrap();
        assert_eq!(handle.as_str(), "000");
    }

    #[test]
    fn load_dispatches_to_load_legacy_on_version_mismatch() {
        // Build an object whose version does not match `Doc::VERSION` (1.0) so the `Loadable`
        // blanket dispatches to `Doc::load_legacy`, which refuses with `UnknownVersion`.
        let value = save::Record::empty()
            .into_value(Version::new(2, 0))
            .into_owned();
        let context = MemoryContext {
            files: HashMap::new(),
            value,
        };
        let err = load::load::<Doc>(&context)
            .expect_err("a version mismatch must dispatch to load_legacy, which refuses");
        let with_loader = load::load_with(&context, PhantomData::<Doc>).unwrap_err();
        assert_eq!(err.to_string(), with_loader.to_string());
        assert_eq!(err.is_recoverable(), with_loader.is_recoverable());
        assert!(format!("{err}").contains("unknown version"));
    }

    // Regression for the NaN float-field bug, demonstrated end-to-end through the in-memory
    // backend (no serde / disk required). A struct carrying NaN-valued `f64` / `f32` fields is
    // saved and reloaded; the reload previously FAILED with `NumberOutOfRange` because
    // `Number::as_f64` / `as_f32` rejected NaN.
    #[derive(Debug)]
    struct Floats {
        finite: f64,
        nan64: f64,
        nan32: f32,
    }

    impl save::Save for Floats {
        const VERSION: Version = Version::new(1, 0);
        fn save(&self, context: save::Context<'_>) -> save::Result<save::Record<'_>> {
            Ok(crate::save_fields!(self, context, [finite, nan64, nan32]))
        }
    }

    impl load::Load<'_> for Floats {
        const VERSION: Version = Version::new(1, 0);
        fn load(object: load::Object<'_>) -> load::Result<Self> {
            crate::load_fields!(object, [finite: f64, nan64: f64, nan32: f32]);
            Ok(Self {
                finite,
                nan64,
                nan32,
            })
        }
        fn load_legacy(_: load::Object<'_>) -> load::Result<Self> {
            Err(load::error::Kind::UnknownVersion.into())
        }
    }

    #[test]
    fn nan_float_fields_round_trip_in_memory() {
        let value = Floats {
            finite: 1.5,
            nan64: f64::NAN,
            nan32: f32::NAN,
        };

        let context = save::save(&value, MemorySaveContext::new()).unwrap();
        let restored: Floats = load::load(&context).expect("NaN float fields must reload");

        assert_eq!(restored.finite, 1.5);
        assert!(restored.nan64.is_nan(), "f64 NaN field lost on reload");
        assert!(restored.nan32.is_nan(), "f32 NaN field lost on reload");
    }
}
