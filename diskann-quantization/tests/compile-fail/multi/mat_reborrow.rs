/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::multi_vector::{Mat, Standard};
use diskann_utils::Reborrow;

// Test that `reborrow` on Mat correctly captures an immutable borrow,
// preventing mutation of the Mat while the reborrow is in scope.
fn main() {
    let mut mat: Mat<Standard<f32>> = Mat::new(Standard::new(4, 3), 0.0f32).unwrap();
    let view = mat.reborrow();
    // This should fail: we cannot mutably borrow `mat` while `view` exists
    let _ = mat.as_view_mut();
    let _ = view.num_vectors();
}
