/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::{Mat, RowMajor};

// Test that `as_view_mut` on Mat correctly captures a mutable lifetime,
// preventing the Mat from being used while the mutable view is in scope.
fn main() {
    let mut mat: Mat<RowMajor<f32>> = Mat::from_repr(RowMajor::new(4, 3).unwrap(), 0.0f32).unwrap();
    let view = mat.as_view_mut();
    // This should fail: we cannot use `mat` while `view` is still alive
    let _ = mat.num_vectors();
    let _ = view.num_vectors();
}
