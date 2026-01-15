/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::error::InlineError;

#[derive(Debug)]
struct Error(u128);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for Error {}

// The returned error is too large to fit in the buffer.
fn main() {
    const { assert!(std::mem::size_of::<Error>() == 16) };
    const { assert!(std::mem::align_of::<Error>() == 16) };

    let _ = InlineError::<16>::new(Error(10));
}
