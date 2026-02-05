/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::multi_vector::{Mat, Standard};

// Test that `rows` on Mat correctly captures an immutable borrow,
// preventing mutation of the Mat while the iterator is in scope.
fn main() {
    let mut mat: Mat<Standard<f32>> = Mat::new(Standard::new(4, 3), 0.0f32).unwrap();
    let iter = mat.rows();
    // This should fail: we cannot mutably borrow `mat` while `iter` exists
    let _ = mat.as_view_mut();
    for row in iter {
        assert_eq!(row[0], 0.0);
    }
}
