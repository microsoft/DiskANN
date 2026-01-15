/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::bits::{BitSlice, Unsigned};

// Here - we test that a `BitSlice` is properly tracked by Rust as containing
// a borrow of its input slice.
fn main() {
    let mut x: Vec<u8> = vec![0; 3];
    let slice = BitSlice::<8, Unsigned>::new(x.as_slice(), 3).unwrap();
    assert_eq!(slice.get(0).unwrap(), 0);
    x[1] = 1;
    assert_eq!(slice.get(0).unwrap(), 0);
}
