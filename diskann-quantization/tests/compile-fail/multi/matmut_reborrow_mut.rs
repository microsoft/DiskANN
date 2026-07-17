/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::multi_vector::{Mat, MatMut, RowMajor};
use diskann_utils::ReborrowMut;

// Test that `reborrow_mut` on MatMut correctly captures a mutable lifetime,
// preventing the original MatMut from being used while the reborrow is in scope.
fn main() {
    let mut mat: Mat<RowMajor<f32>> = Mat::new(RowMajor::new(4, 3).unwrap(), 0.0f32).unwrap();
    let mut view: MatMut<'_, RowMajor<f32>> = mat.as_view_mut();
    let reborrowed = view.reborrow_mut();
    // This should fail: we cannot use `view` while `reborrowed` is still alive
    let _ = view.num_vectors();
    let _ = reborrowed.num_vectors();
}
