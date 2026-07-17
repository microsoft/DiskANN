/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::multi_vector::{Mat, MatMut, RowMajor};

// Test that `get_row` on MatMut correctly captures an immutable borrow,
// preventing mutation of the MatMut while the row is in scope.
fn main() {
    let mut mat: Mat<RowMajor<f32>> = Mat::new(RowMajor::new(4, 3).unwrap(), 0.0f32).unwrap();
    let mut view: MatMut<'_, RowMajor<f32>> = mat.as_view_mut();
    let row = view.get_row(0).unwrap();
    // This should fail: we cannot mutably borrow `view` while `row` exists
    let _ = view.get_row_mut(1);
    assert_eq!(row[0], 0.0);
}
