/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::bits::{SlicePtr, ptr::sealed::Precursor};

// Here - we check that a user cannot bring in the `Precursor` trait - preventing them from
// directly constructing our interior types.
fn main() {
    let x: Vec<u8> = vec![0; 3];
    let borrowed: SlicePtr<u8> = (&*x).precursor_into();
}
