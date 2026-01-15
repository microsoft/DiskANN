/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{poly, alloc::{Poly, GlobalAllocator}};

// Since arrays to not implement `Display`, this conversion is invalid.
fn main() {
    let x = Poly::new([1, 2], GlobalAllocator).unwrap();
    let _ = poly!(std::fmt::Display, x);
}

