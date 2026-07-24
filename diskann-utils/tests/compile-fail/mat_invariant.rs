/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::{Mat, RowMajor};

// Verify that `Mat` is invariant in any generic parameters.
//
// This must not compile because it would allow assigning references with a shorter lifetime
// into the matrix
fn bad<'long, 'short>(v: Mat<RowMajor<&'long u8>>) -> Mat<RowMajor<&'short u8>>
where
    'long: 'short,
{
    v
}

fn main() {
    let b = 0u8;
    let m = Mat::from_repr(RowMajor::new(4, 3).unwrap(), &b).unwrap();
    bad(m);
}
