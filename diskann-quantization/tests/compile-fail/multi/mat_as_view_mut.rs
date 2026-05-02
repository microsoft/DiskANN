/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::multi_vector::{Mat, Standard};

// Test that `as_view_mut` on Mat correctly captures a mutable lifetime,
// preventing the Mat from being used while the mutable view is in scope.
fn main() {
    let mut mat: Mat<Standard<f32>> = Mat::new(Standard::new(4, 3).unwrap(), 0.0f32).unwrap();
    let view = mat.as_view_mut();
    // This should fail: we cannot use `mat` while `view` is still alive
    let _ = mat.num_vectors();
    let _ = view.num_vectors();
}
