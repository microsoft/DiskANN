/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{poly, alloc::GlobalAllocator};

// Since arrays to not implement `Display`, this conversion is invalid.
fn main() {
    let x = std::rc::Rc::new([1, 2]);
    let _ = poly!({std::fmt::Debug + Send}, x, GlobalAllocator).unwrap();
}
