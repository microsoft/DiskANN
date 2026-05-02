/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::multi_vector::{Mat, Standard};
use diskann_utils::ReborrowMut;

// Test that `reborrow_mut` on Mat correctly captures a mutable borrow,
// preventing use of the Mat while the reborrow is in scope.
fn main() {
    let mut mat: Mat<Standard<f32>> = Mat::new(Standard::new(4, 3).unwrap(), 0.0f32).unwrap();
    let view = mat.reborrow_mut();
    // This should fail: we cannot use `mat` while `view` exists
    let _ = mat.num_vectors();
    let _ = view.num_vectors();
}
