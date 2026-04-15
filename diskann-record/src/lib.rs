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
    }

    impl save::Save for Test {
        const VERSION: Version = Version::new(0, 0, 0);
        fn save<'a>(&'a self, context: save::Context<'a>) -> save::Record<'a> {
            save_fields!(self, context, [x, y])
        }
    }
}
