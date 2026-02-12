/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::multi_vector::{Mat, Standard};

// Test that the `rows_mut` iterator correctly captures a mutable lifetime,
// preventing the Mat from being used while the iterator is in scope.
fn main() {
    let mut mat: Mat<Standard<f32>> = Mat::new(Standard::new(4, 3).unwrap(), 0.0f32).unwrap();
    let iter = mat.rows_mut();
    // This should fail: we cannot use `mat` while the mutable iterator is alive
    let _ = mat.num_vectors();
    for row in iter {
        row[0] = 1.0;
    }
}
