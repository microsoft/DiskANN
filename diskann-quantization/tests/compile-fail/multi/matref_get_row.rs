/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::multi_vector::{Mat, MatRef, Standard};

// Test that `get_row` on MatRef returns a row with the correct lifetime,
// and that an immutable borrow is held while the row is in scope.
fn main() {
    let mut mat: Mat<Standard<f32>> = Mat::new(Standard::new(4, 3).unwrap(), 0.0f32).unwrap();
    let view: MatRef<'_, Standard<f32>> = mat.as_view();
    let row = view.get_row(0).unwrap();
    // This should fail: we cannot mutably borrow `mat` while `row` exists
    // (since `row` holds a reference derived from `mat`)
    let _ = mat.as_view_mut();
    assert_eq!(row[0], 0.0);
}
