/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{poly, alloc::GlobalAllocator};

// Since arrays to not implement `Display`, this conversion is invalid.
fn main() {
    let x = [1, 2];
    let _ = poly!(std::fmt::Display, x, GlobalAllocator).unwrap();
}
