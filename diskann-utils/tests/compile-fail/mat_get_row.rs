/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::{Mat, RowMajor};

// Test that `get_row` on Mat correctly captures an immutable borrow,
// preventing mutation of the Mat while the row is in scope.
fn main() {
    let mut mat: Mat<RowMajor<f32>> = Mat::from_repr(RowMajor::new(4, 3).unwrap(), 0.0f32).unwrap();
    let row = mat.get_row(0).unwrap();
    // This should fail: we cannot mutably borrow `mat` while `row` exists
    let _ = mat.get_row_mut(1);
    assert_eq!(row[0], 0.0);
}
