/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::multi_vector::{Mat, MatMut, Standard};
use diskann_utils::Reborrow;

// Test that `reborrow` on MatMut correctly captures an immutable borrow,
// preventing mutation of the MatMut while the reborrow is in scope.
fn main() {
    let mut mat: Mat<Standard<f32>> = Mat::new(Standard::new(4, 3).unwrap(), 0.0f32).unwrap();
    let mut view: MatMut<'_, Standard<f32>> = mat.as_view_mut();
    let immut_view = view.reborrow();
    // This should fail: we cannot mutably borrow `view` while `immut_view` exists
    let _ = view.get_row_mut(0);
    let _ = immut_view.num_vectors();
}
