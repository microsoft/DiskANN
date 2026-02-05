/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::multi_vector::{Mat, MatMut, Standard};

// Test that `get_row` on MatMut correctly captures an immutable borrow,
// preventing mutation of the MatMut while the row is in scope.
fn main() {
    let mut mat: Mat<Standard<f32>> = Mat::new(Standard::new(4, 3), 0.0f32).unwrap();
    let mut view: MatMut<'_, Standard<f32>> = mat.as_view_mut();
    let row = view.get_row(0).unwrap();
    // This should fail: we cannot mutably borrow `view` while `row` exists
    let _ = view.get_row_mut(1);
    assert_eq!(row[0], 0.0);
}
