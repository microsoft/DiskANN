/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::multi_vector::{Mat, MatMut, RowMajor};

// Test that `get_row_mut` on MatMut correctly captures a mutable lifetime,
// preventing the MatMut from being used while the row is in scope.
fn main() {
    let mut mat: Mat<RowMajor<f32>> = Mat::from_repr(RowMajor::new(4, 3).unwrap(), 0.0f32).unwrap();
    let mut view: MatMut<'_, RowMajor<f32>> = mat.as_view_mut();
    let row = view.get_row_mut(0).unwrap();
    // This should fail: we cannot use `view` while `row` is still borrowed
    let _ = view.get_row_mut(1).unwrap();
    row[0] = 1.0;
}
