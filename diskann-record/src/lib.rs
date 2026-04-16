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
        inner: Inner,
        // We write this as a binary file.
        vector: Vec<u8>,
    }

    #[derive(Debug, PartialEq)]
    struct Inner {
        z: usize,
        w: Vec<i8>,
    }

    impl save::Save for Inner {
        const VERSION: Version = Version::new(0, 0, 0);
        fn save(&self, context: save::Context<'_>) -> save::Result<save::Record<'_>> {
            Ok(save_fields!(self, context, [z, w]))
        }
    }

    impl save::Save for Test {
        const VERSION: Version = Version::new(0, 0, 0);
        fn save(&self, context: save::Context<'_>) -> save::Result<save::Record<'_>> {
            // We save `x`, `y`, and `inner` directly into the manifest.
            // The raw vector data we instead store in an auxiliary file.

            let mut io = context.write("auxiliary.bin");
            io.write_all(&self.vector).unwrap();

            let mut record = save_fields!(self, context, [x, y, inner]);
            record.insert("vector", io.finish());
            Ok(record)
        }
    }

    impl load::Load<'_> for Test {
        const VERSION: Version = Version::new(0, 0, 0);
        fn load(object: load::Object<'_>) -> load::Result<Self> {
            load_fields!(object, [x, y, inner, vector: save::Handle]);

            let mut io = object.read(&vector)?;
            let mut vector = Vec::new();
            io.read_to_end(&mut vector).unwrap();

            Ok(Self { x, y, inner, vector })
        }

        fn load_legacy(object: load::Object<'_>) -> load::Result<Self> {
            panic!("nope!");
        }
    }

    impl load::Load<'_> for Inner {
        const VERSION: Version = Version::new(0, 0, 0);
        fn load(object: load::Object<'_>) -> load::Result<Self> {
            load_fields!(object, [z, w]);
            Ok(Self { z, w })
        }

        fn load_legacy(object: load::Object<'_>) -> load::Result<Self> {
            panic!("nope!");
        }
    }

    #[test]
    fn this_test_writes() {
        let inner = Inner { z: 10, w: vec![-1, -2, -3] };

        let t = Test {
            x: "hello".into(),
            y: 5.0,
            inner,
            vector: vec![0, 1, 2, 3, 4, 5],
        };

        let dir = ".";
        let metadata = "metadata.json";

        save::save_to_disk(&t, dir, metadata).unwrap();
        let we_are_back: Test = load::load_from_disk(metadata.as_ref(), dir.as_ref()).unwrap();

        assert_eq!(t, we_are_back);
    }
}
