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

    #[derive(Debug, PartialEq)]
    struct Test {
        x: String,
        y: f32,
        enabled: bool,
        inner: Inner,
        // We write this as a binary file.
        vector: Vec<u8>,
    }

    #[derive(Debug, PartialEq)]
    struct Inner {
        z: usize,
        w: Vec<i8>,
        flags: Vec<bool>,
    }

    impl save::Save for Inner {
        const VERSION: Version = Version::new(0, 0, 0);
        fn save(&self, context: save::Context<'_>) -> save::Result<save::Record<'_>> {
            Ok(save_fields!(self, context, [z, w, flags]))
        }
    }

    impl save::Save for Test {
        const VERSION: Version = Version::new(0, 0, 0);
        fn save(&self, context: save::Context<'_>) -> save::Result<save::Record<'_>> {
            // We save `x`, `y`, and `inner` directly into the manifest.
            // The raw vector data we instead store in an auxiliary file.

            let mut io = context.write("auxiliary.bin");
            io.write_all(&self.vector).unwrap();

            let mut record = save_fields!(self, context, [x, y, enabled, inner]);
            record.insert("vector", io.finish());
            Ok(record)
        }
    }

    impl load::Load<'_> for Test {
        const VERSION: Version = Version::new(0, 0, 0);
        fn load(object: load::Object<'_>) -> load::Result<Self> {
            load_fields!(object, [x, y, enabled, inner, vector: save::Handle]);

            let mut io = object.read(&vector)?;
            let mut vector = Vec::new();
            io.read_to_end(&mut vector).unwrap();

            Ok(Self {
                x,
                y,
                enabled,
                inner,
                vector,
            })
        }

        fn load_legacy(_object: load::Object<'_>) -> load::Result<Self> {
            panic!("nope!");
        }
    }

    impl load::Load<'_> for Inner {
        const VERSION: Version = Version::new(0, 0, 0);
        fn load(object: load::Object<'_>) -> load::Result<Self> {
            load_fields!(object, [z, w, flags]);
            Ok(Self { z, w, flags })
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
        };

        let t = Test {
            x: "hello".into(),
            y: 5.0,
            enabled: true,
            inner,
            vector: vec![0, 1, 2, 3, 4, 5],
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
}
