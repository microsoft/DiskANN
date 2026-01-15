/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::error::InlineError;

#[derive(Debug)]
struct Error(usize);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for Error {}

// The returned error is too large to fit in the buffer.
fn main() {
    let _ = InlineError::<4>::new(Error(10));
}
