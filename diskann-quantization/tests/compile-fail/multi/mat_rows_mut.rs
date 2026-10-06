/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{
    multi_vector::{Mat, Defaulted},
    minmax::MinMaxMeta,
};

// Test that the `rows_mut` iterator correctly captures a mutable lifetime,
// preventing the Mat from being used while the iterator is in scope.
fn main() {
    let mut mat = Mat::new(MinMaxMeta::<1>::new(2, 2), Defaulted).unwrap();
    let iter = mat.rows_mut();
    // This should fail: we cannot use `mat` while the mutable iterator is alive
    let _ = mat.num_vectors();
    for row in iter {
        std::hint::black_box(row);
    }
}
