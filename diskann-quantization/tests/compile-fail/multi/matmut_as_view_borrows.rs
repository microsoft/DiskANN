/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::multi_vector::{Mat, MatMut, Standard};

// Test that `as_view` on MatMut correctly captures an immutable lifetime,
// preventing mutating the MatMut while the immutable view is in scope.
fn main() {
    let mut mat: Mat<Standard<f32>> = Mat::new(Standard::new(4, 3).unwrap(), 0.0f32).unwrap();
    let mut view: MatMut<'_, Standard<f32>> = mat.as_view_mut();
    let immut_view = view.as_view();
    // This should fail: we cannot mutate `view` while `immut_view` exists
    let _ = view.get_row_mut(0);
    let _ = immut_view.num_vectors();
}
