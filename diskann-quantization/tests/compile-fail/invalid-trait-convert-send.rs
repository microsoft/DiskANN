/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{poly, alloc::{Poly, GlobalAllocator}};

// Since `Rc` is not `Send`, this will fail.
fn main() {
    let x = Poly::new(std::rc::Rc::new([1, 2]), GlobalAllocator).unwrap();
    let _ = poly!({std::fmt::Debug + Send}, x);
}
