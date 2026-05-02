/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::multi_vector::{Mat, Standard};

// Verify that `Mat` is invariant in any generic parameters.
//
// This must not compile because it would allow assigning references with a shorter lifetime
// into the matrix
fn bad<'long, 'short>(v: Mat<Standard<&'long u8>>) -> Mat<Standard<&'short u8>>
where
    'long: 'short,
{
    v
}

fn main() {
    let b = 0u8;
    let m = Mat::new(Standard::new(4, 3).unwrap(), &b).unwrap();
    bad(m);
}
