/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::multi_vector::{Mat, MatMut, Standard};

// Test that the `rows_mut` iterator on MatMut correctly captures a mutable lifetime,
// preventing the MatMut from being used while the iterator is in scope.
fn main() {
    let mut mat: Mat<Standard<f32>> = Mat::new(Standard::new(4, 3).unwrap(), 0.0f32).unwrap();
    let mut view: MatMut<'_, Standard<f32>> = mat.as_view_mut();
    let iter = view.rows_mut();
    // This should fail: we cannot use `view` while the mutable iterator is alive
    let _ = view.num_vectors();
    for row in iter {
        row[0] = 1.0;
    }
}
