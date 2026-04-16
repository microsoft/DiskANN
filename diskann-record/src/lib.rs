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

    struct Test {
        x: String,
        y: f32,
        inner: Inner,
    }

    struct Inner {
        z: usize,
        w: i8,
    }

    impl save::Save for Test {
        const VERSION: Version = Version::new(0, 0, 0);
        fn save(&self, context: save::Context<'_>) -> save::Result<save::Record<'_>> {
            Ok(save_fields!(self, context, [x, y, inner]))
        }
    }

    impl save::Save for Inner {
        const VERSION: Version = Version::new(0, 0, 0);
        fn save(&self, context: save::Context<'_>) -> save::Result<save::Record<'_>> {
            Ok(save_fields!(self, context, [z, w]))
        }
    }

    impl load::Load<'_> for Test {
        const VERSION: Version = Version::new(0, 0, 0);
        fn load(object: load::Object<'_>) -> load::Result<Self> {
            load_fields!(object, [x, y, inner]);
            Ok(Self { x, y, inner })
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
        let inner = Inner {
            z: 10,
            w: -1,
        };

        let t = Test {
            x: "hello".into(),
            y: 5.0,
            inner,
        };

        save::save_to_disk(&t, ".", "metadata.json").unwrap();
    }
}
