/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::{Mat, RowMajor};

// Test that `rows` on Mat correctly captures an immutable borrow,
// preventing mutation of the Mat while the iterator is in scope.
fn main() {
    let mut mat: Mat<RowMajor<f32>> = Mat::from_repr(RowMajor::new(4, 3).unwrap(), 0.0f32).unwrap();
    let iter = mat.rows();
    // This should fail: we cannot mutably borrow `mat` while `iter` exists
    let _ = mat.as_view_mut();
    for row in iter {
        assert_eq!(row[0], 0.0);
    }
}
