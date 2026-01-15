/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::bits::{MutBitSlice, Unsigned};

// Here - we test that a `MutBitSlice` is properly tracked by Rust as containing
// a mutable borrow of its input slice.
fn main() {
    let mut x: Vec<u8> = vec![0; 3];
    let mut slice = MutBitSlice::<8, Unsigned>::new(x.as_mut_slice(), 3).unwrap();
    slice.set(0, 0).unwrap();
    x[1] = 1;
    slice.set(2, 2).unwrap();
}
