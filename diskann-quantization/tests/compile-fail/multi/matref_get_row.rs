/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::multi_vector::{Mat, MatRef, RowMajor};

// Test that `get_row` on MatRef returns a row with the correct lifetime,
// and that an immutable borrow is held while the row is in scope.
fn main() {
    let mut mat: Mat<RowMajor<f32>> = Mat::from_repr(RowMajor::new(4, 3).unwrap(), 0.0f32).unwrap();
    let view: MatRef<'_, RowMajor<f32>> = mat.as_view();
    let row = view.get_row(0).unwrap();
    // This should fail: we cannot mutably borrow `mat` while `row` exists
    // (since `row` holds a reference derived from `mat`)
    let _ = mat.as_view_mut();
    assert_eq!(row[0], 0.0);
}
