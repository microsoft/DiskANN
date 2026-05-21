/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod number;
pub use number::Number;

mod version;
pub use version::Version;

pub mod load;
pub mod save;

// Canonical wire width for `usize` and `isize` in manifests is 64 bits. Saving a value
// on a 64-bit platform and loading it on a 32-bit platform (or vice versa) could silently
// truncate values that exceed `u32::MAX` / `i32::MAX`. We therefore require a 64-bit
// platform at compile time. Loaders still range-check at runtime, but this check ensures
// the saver never emits values that the canonical width cannot represent.
const _: () = assert!(
    usize::BITS == 64,
    "diskann-record requires a 64-bit target: usize/isize MUST be 64 bits wide !!",
);

/// Return `true` if `s` is a reserved string for purposes of saving and loading.
#[doc(hidden)]
pub const fn is_reserved(s: &str) -> bool {
    if let Some(first) = s.as_bytes().first()
        && *first == b"$"[0]
    {
        true
    } else {
        false
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};

    #[derive(Debug, PartialEq)]
    struct Test {
        x: String,
        y: f32,
        enabled: bool,
        inner: Inner,
        // We write this as a binary file.
        vector: Vec<u8>,
        nickname: Option<String>,
        absent: Option<String>,
    }

    #[derive(Debug, PartialEq)]
    struct Inner {
        z: usize,
        w: Vec<i8>,
        flags: Vec<bool>,
        maybe_count: Option<u32>,
        maybe_missing: Option<u32>,
        sparse: Vec<Option<i32>>,
    }

    impl save::Save for Inner {
        const VERSION: Version = Version::new(0, 0, 0);
        fn save(&self, context: save::Context<'_>) -> save::Result<save::Record<'_>> {
            Ok(save_fields!(
                self,
                context,
                [z, w, flags, maybe_count, maybe_missing, sparse]
            ))
        }
    }

    impl save::Save for Test {
        const VERSION: Version = Version::new(0, 0, 0);
        fn save(&self, context: save::Context<'_>) -> save::Result<save::Record<'_>> {
            // We save `x`, `y`, and `inner` directly into the manifest.
            // The raw vector data we instead store in an auxiliary file.

            let mut io = context.write("auxiliary.bin")?;
            io.write_all(&self.vector).map_err(save::Error::new)?;

            let mut record = save_fields!(self, context, [x, y, enabled, inner, nickname, absent]);
            record.insert("vector", io.finish()?)?;
            Ok(record)
        }
    }

    impl load::Load<'_> for Test {
        const VERSION: Version = Version::new(0, 0, 0);
        fn load(object: load::Object<'_>) -> load::Result<Self> {
            load_fields!(
                object,
                [
                    x,
                    y,
                    enabled,
                    inner,
                    nickname: Option<String>,
                    absent: Option<String>,
                    vector: save::Handle,
                ]
            );

            let mut io = object.read(&vector)?;
            let mut vector = Vec::new();
            io.read_to_end(&mut vector).unwrap();

            Ok(Self {
                x,
                y,
                enabled,
                inner,
                vector,
                nickname,
                absent,
            })
        }

        fn load_legacy(_object: load::Object<'_>) -> load::Result<Self> {
            panic!("nope!");
        }
    }

    impl load::Load<'_> for Inner {
        const VERSION: Version = Version::new(0, 0, 0);
        fn load(object: load::Object<'_>) -> load::Result<Self> {
            load_fields!(
                object,
                [
                    z,
                    w,
                    flags,
                    maybe_count: Option<u32>,
                    maybe_missing: Option<u32>,
                    sparse: Vec<Option<i32>>,
                ]
            );
            Ok(Self {
                z,
                w,
                flags,
                maybe_count,
                maybe_missing,
                sparse,
            })
        }

        fn load_legacy(_object: load::Object<'_>) -> load::Result<Self> {
            panic!("nope!");
        }
    }

    #[test]
    fn round_trip_uses_isolated_temp_dir() -> anyhow::Result<()> {
        let inner = Inner {
            z: 10,
            w: vec![-1, -2, -3],
            flags: vec![true, false, true],
            maybe_count: Some(42),
            maybe_missing: None,
            sparse: vec![Some(1), None, Some(-3), None],
        };

        let t = Test {
            x: "hello".into(),
            y: 5.0,
            enabled: true,
            inner,
            vector: vec![0, 1, 2, 3, 4, 5],
            nickname: Some("friend".into()),
            absent: None,
        };

        // Keep the TempDir guard alive for the full round trip; Drop removes the
        // manifest and auxiliary artifact after the assertion completes.
        let temp_dir = tempfile::tempdir()?;
        let dir = temp_dir.path();
        let metadata = dir.join("metadata.json");

        save::save_to_disk(&t, dir, &metadata)?;
        let we_are_back: Test = load::load_from_disk(&metadata, dir)?;

        assert_eq!(t, we_are_back);
        Ok(())
    }

    /////////////////////////
    // Enum support: round //
    /////////////////////////

    #[derive(Debug, PartialEq)]
    enum Metric {
        L2,
        Cosine,
        Weighted { weights: Vec<f32> },
    }

    impl save::Save for Metric {
        const VERSION: Version = Version::new(0, 0, 0);
        fn save(&self, context: save::Context<'_>) -> save::Result<save::Record<'_>> {
            let mut record = save::Record::empty();
            match self {
                Self::L2 => {
                    record.insert("L2", save::Value::Null)?;
                }
                Self::Cosine => {
                    record.insert("Cosine", save::Value::Null)?;
                }
                Self::Weighted { weights } => {
                    let payload = save_fields!(context, [weights]).into_value(Self::VERSION);
                    record.insert("Weighted", payload)?;
                }
            }
            Ok(record)
        }
    }

    impl load::Load<'_> for Metric {
        const VERSION: Version = Version::new(0, 0, 0);
        fn load(object: load::Object<'_>) -> load::Result<Self> {
            match object.single_key()? {
                "L2" => Ok(Self::L2),
                "Cosine" => Ok(Self::Cosine),
                "Weighted" => {
                    let inner = object
                        .child("Weighted")?
                        .as_object()
                        .ok_or(load::error::Kind::TypeMismatch)?;
                    load_fields!(inner, [weights: Vec<f32>]);
                    Ok(Self::Weighted { weights })
                }
                _ => Err(load::error::Kind::UnknownVariant.into()),
            }
        }
        fn load_legacy(_object: load::Object<'_>) -> load::Result<Self> {
            Err(load::error::Kind::UnknownVersion.into())
        }
    }

    #[derive(Debug, PartialEq)]
    struct MetricBag {
        primary: Metric,
        alternatives: Vec<Metric>,
    }

    impl save::Save for MetricBag {
        const VERSION: Version = Version::new(0, 0, 0);
        fn save(&self, context: save::Context<'_>) -> save::Result<save::Record<'_>> {
            Ok(save_fields!(self, context, [primary, alternatives]))
        }
    }

    impl load::Load<'_> for MetricBag {
        const VERSION: Version = Version::new(0, 0, 0);
        fn load(object: load::Object<'_>) -> load::Result<Self> {
            load_fields!(object, [primary: Metric, alternatives: Vec<Metric>]);
            Ok(Self {
                primary,
                alternatives,
            })
        }
        fn load_legacy(_object: load::Object<'_>) -> load::Result<Self> {
            panic!("nope!");
        }
    }

    #[test]
    fn enum_round_trip_through_disk() -> anyhow::Result<()> {
        let bag = MetricBag {
            primary: Metric::Weighted {
                weights: vec![0.25, 0.5, 0.25],
            },
            alternatives: vec![
                Metric::L2,
                Metric::Cosine,
                Metric::Weighted { weights: vec![1.0] },
            ],
        };

        let temp_dir = tempfile::tempdir()?;
        let dir = temp_dir.path();
        let metadata = dir.join("metadata.json");

        save::save_to_disk(&bag, dir, &metadata)?;
        let restored: MetricBag = load::load_from_disk(&metadata, dir)?;

        assert_eq!(bag, restored);
        Ok(())
    }

    #[derive(Debug, PartialEq)]
    struct StructShape {
        x: i32,
    }

    impl save::Save for StructShape {
        const VERSION: Version = Version::new(0, 0, 0);
        fn save(&self, context: save::Context<'_>) -> save::Result<save::Record<'_>> {
            Ok(save_fields!(self, context, [x]))
        }
    }

    impl load::Load<'_> for StructShape {
        const VERSION: Version = Version::new(0, 0, 0);
        fn load(object: load::Object<'_>) -> load::Result<Self> {
            load_fields!(object, [x: i32]);
            Ok(Self { x })
        }
        fn load_legacy(_object: load::Object<'_>) -> load::Result<Self> {
            panic!("nope!");
        }
    }

    #[derive(Debug, PartialEq)]
    enum EnumShape {
        Only { x: i32 },
    }

    impl save::Save for EnumShape {
        const VERSION: Version = Version::new(0, 0, 0);
        fn save(&self, context: save::Context<'_>) -> save::Result<save::Record<'_>> {
            let mut record = save::Record::empty();
            match self {
                Self::Only { x } => {
                    let payload = save_fields!(context, [x]).into_value(Self::VERSION);
                    record.insert("Only", payload)?;
                }
            }
            Ok(record)
        }
    }

    impl load::Load<'_> for EnumShape {
        const VERSION: Version = Version::new(0, 0, 0);
        fn load(object: load::Object<'_>) -> load::Result<Self> {
            match object.single_key()? {
                "Only" => {
                    let inner = object
                        .child("Only")?
                        .as_object()
                        .ok_or(load::error::Kind::TypeMismatch)?;
                    load_fields!(inner, [x: i32]);
                    Ok(Self::Only { x })
                }
                _ => Err(load::error::Kind::UnknownVariant.into()),
            }
        }
        fn load_legacy(_object: load::Object<'_>) -> load::Result<Self> {
            panic!("nope!");
        }
    }

    #[test]
    fn loading_enum_as_struct_is_rejected() -> anyhow::Result<()> {
        // Enum data has a single key "Only" whose payload is a versioned
        // sub-object. Loading it as `StructShape` (which expects field `x`)
        // surfaces `MissingField`.
        let value = EnumShape::Only { x: 7 };
        let temp_dir = tempfile::tempdir()?;
        let dir = temp_dir.path();
        let metadata = dir.join("metadata.json");

        save::save_to_disk(&value, dir, &metadata)?;
        let err = load::load_from_disk::<StructShape>(&metadata, dir)
            .expect_err("loading enum data into a struct shape should fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("missing field"),
            "expected MissingField error, got: {msg}"
        );
        Ok(())
    }

    #[test]
    fn loading_struct_as_enum_is_rejected() -> anyhow::Result<()> {
        // Struct data has field `x`, which the enum loader sees as a candidate
        // variant name. It doesn't match any arm, so we get `UnknownVariant`.
        let value = StructShape { x: 7 };
        let temp_dir = tempfile::tempdir()?;
        let dir = temp_dir.path();
        let metadata = dir.join("metadata.json");

        save::save_to_disk(&value, dir, &metadata)?;
        let err = load::load_from_disk::<EnumShape>(&metadata, dir)
            .expect_err("loading struct data into an enum shape should fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("unknown variant"),
            "expected UnknownVariant error, got: {msg}"
        );
        Ok(())
    }

    ///////////////////////////////
    // Manifest directory escape //
    ///////////////////////////////

    /// Minimal loadable type with a single handle field. Used by the
    /// directory-escape tests below to drive `Object::read` against a
    /// hand-crafted manifest.
    #[derive(Debug)]
    struct HandleOnly {
        _blob: Vec<u8>,
    }

    impl load::Load<'_> for HandleOnly {
        const VERSION: Version = Version::new(0, 0, 0);
        fn load(object: load::Object<'_>) -> load::Result<Self> {
            load_fields!(object, [blob: save::Handle]);
            let mut io = object.read(&blob)?;
            let mut buf = Vec::new();
            io.read_to_end(&mut buf).map_err(load::Error::new)?;
            Ok(Self { _blob: buf })
        }
        fn load_legacy(_object: load::Object<'_>) -> load::Result<Self> {
            panic!("nope!");
        }
    }

    /// Write a hand-crafted manifest into `dir` whose root object exposes a
    /// `blob` field referencing `handle_target`. Returns the metadata path.
    fn write_handle_manifest(dir: &Path, handle_target: &str) -> std::io::Result<PathBuf> {
        let manifest = serde_json::json!({
            // Register the same target in `files` so the membership check
            // would otherwise let it through — this isolates the new
            // path-shape check as the thing rejecting the load.
            "files": [handle_target],
            "value": {
                "$version": { "major": 0, "minor": 0, "patch": 0 },
                "blob": { "$handle": handle_target },
            },
        });
        let metadata = dir.join("metadata.json");
        std::fs::write(&metadata, serde_json::to_vec(&manifest)?)?;
        Ok(metadata)
    }

    #[test]
    fn handle_with_parent_traversal_is_rejected() -> anyhow::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let dir = temp_dir.path();
        let metadata = write_handle_manifest(dir, "../escape.bin")?;

        let err = load::load_from_disk::<HandleOnly>(&metadata, dir)
            .expect_err("handle escaping the manifest directory must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("escapes the manifest directory"),
            "expected manifest-escape rejection, got: {msg}"
        );
        Ok(())
    }

    #[test]
    fn handle_with_absolute_path_is_rejected() -> anyhow::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let dir = temp_dir.path();
        // Use a platform-appropriate absolute path. Both shapes should be
        // rejected on their respective platforms; we test the native one.
        let absolute = if cfg!(windows) {
            "C:\\Windows\\System32\\drivers\\etc\\hosts"
        } else {
            "/etc/passwd"
        };
        let metadata = write_handle_manifest(dir, absolute)?;

        let err = load::load_from_disk::<HandleOnly>(&metadata, dir)
            .expect_err("absolute-path handle must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("escapes the manifest directory"),
            "expected manifest-escape rejection, got: {msg}"
        );
        Ok(())
    }
}
