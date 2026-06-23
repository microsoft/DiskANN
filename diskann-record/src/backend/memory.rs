/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! In-memory save/load contexts.
//!
//! [`InMemorySaveContext`] and [`InMemoryContext`] mirror the disk-backed contexts but
//! keep the manifest value and every side-car artifact in memory. Saving through an
//! [`InMemorySaveContext`] yields an [`InMemoryContext`] (its
//! [`SaveContext::Output`](crate::save::SaveContext::Output)) that can be loaded directly
//! via [`crate::load::load`].
//!
//! Unlike the disk path, this round trip never serializes through JSON: the manifest
//! [`Value`] is deep-copied to `'static` via [`Value::into_owned`] and side-car artifacts
//! are buffered in memory through a [`std::io::Cursor`]. As a result these contexts are
//! available regardless of the `disk` / `serde` features.

use std::{collections::HashMap, io::Cursor, sync::Mutex};

use crate::{
    Value,
    load::{self, LoadContext, Reader},
    save::{self, SaveContext, Writer},
};

/// A save-side [`SaveContext`] that keeps every side-car artifact and the committed
/// manifest value in memory.
///
/// [`SaveContext::finish`] consumes the context and returns an [`InMemoryContext`] ready
/// to be loaded with [`crate::load::load`].
#[derive(Debug, Default)]
pub struct InMemorySaveContext {
    files: Mutex<HashMap<String, Vec<u8>>>,
}

impl InMemorySaveContext {
    /// Create an empty in-memory save context.
    pub fn new() -> Self {
        Self::default()
    }
}

impl SaveContext for InMemorySaveContext {
    type Output = InMemoryContext;

    fn write(&self, key: Option<&str>) -> save::Result<Writer<'_>> {
        // Mirror the disk context: a human-readable hint must be a simple relative file
        // name so the generated artifact name is a single, well-formed key.
        if let Some(key) = key {
            let mut components = std::path::Path::new(key).components();
            match components.next() {
                Some(std::path::Component::Normal(_)) if components.next().is_none() => {}
                _ => {
                    return Err(save::Error::message(format!(
                        "artifact file name hint {:?} must be a relative file name with no path \
                         separators",
                        key,
                    )));
                }
            }
        }

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
        // Reserve the name so the count advances and concurrent writers cannot collide;
        // the placeholder is overwritten with the real bytes by `Writer::finish`.
        files.insert(name.clone(), Vec::new());
        drop(files);

        Ok(Writer::memory(name, &self.files))
    }

    fn finish(self, value: Value<'_>) -> save::Result<InMemoryContext> {
        let files = self
            .files
            .into_inner()
            .unwrap_or_else(|poison| poison.into_inner());
        Ok(InMemoryContext {
            files,
            value: value.into_owned(),
        })
    }
}

/// A load-side [`LoadContext`] backed entirely by in-memory buffers.
///
/// Produced by [`InMemorySaveContext`] via [`SaveContext::finish`]. Holds the committed
/// manifest [`Value`] and every side-car artifact as an in-memory byte buffer, so loading
/// never serializes through JSON or touches the filesystem.
#[derive(Debug)]
pub struct InMemoryContext {
    files: HashMap<String, Vec<u8>>,
    value: Value<'static>,
}

impl LoadContext for InMemoryContext {
    fn value(&self) -> load::Result<&Value<'_>> {
        Ok(&self.value)
    }

    fn read(&self, key: &str) -> load::Result<Reader<'_>> {
        match self.files.get(key) {
            Some(bytes) => Ok(Reader::new(Box::new(Cursor::new(bytes.as_slice())))),
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
    use std::io::{Read, Write};

    use super::*;
    use crate::{Version, load, save};

    #[derive(Debug, PartialEq)]
    struct Doc {
        name: String,
        blob: Vec<u8>,
    }

    impl save::Save for Doc {
        const VERSION: Version = Version::new(1, 0, 0);
        fn save(&self, context: save::Context<'_>) -> save::Result<save::Record<'_>> {
            let mut io = context.write(Some("blob.bin"))?;
            io.write_all(&self.blob).map_err(save::Error::new)?;
            let mut record = crate::save_fields!(self, context, [name]);
            record.insert("blob", io.finish()?)?;
            Ok(record)
        }
    }

    impl load::Load<'_> for Doc {
        const VERSION: Version = Version::new(1, 0, 0);
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

    #[test]
    fn round_trips_without_serde_or_disk() {
        let doc = Doc {
            name: "example".to_owned(),
            blob: vec![1, 2, 3, 4, 5],
        };

        let context = save::save(&doc, InMemorySaveContext::new()).unwrap();
        let restored: Doc = load::load(&context).unwrap();

        assert_eq!(doc, restored);
    }

    #[test]
    fn read_rejects_unregistered_artifact() {
        let context = InMemoryContext {
            files: HashMap::new(),
            value: Value::Null,
        };
        let err = context
            .read("missing.bin")
            .err()
            .expect("an unregistered artifact must be rejected");
        assert!(format!("{err}").contains("not registered in this context"));
    }
}
