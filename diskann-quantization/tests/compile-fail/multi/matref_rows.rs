/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::multi_vector::{Mat, MatRef, RowMajor};

// Test that `rows` on MatRef returns an iterator with the correct lifetime,
// preventing mutation of the underlying Mat while iterating.
fn main() {
    let mut mat: Mat<RowMajor<f32>> = Mat::from_repr(RowMajor::new(4, 3).unwrap(), 0.0f32).unwrap();
    let view: MatRef<'_, RowMajor<f32>> = mat.as_view();
    let iter = view.rows();
    // This should fail: we cannot mutably borrow `mat` while `iter` exists
    let _ = mat.as_view_mut();
    for row in iter {
        assert_eq!(row[0], 0.0);
    }
}
