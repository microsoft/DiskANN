/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{
    multi_vector::{Mat, MatMut, Defaulted},
    minmax::MinMaxMeta,
};
use diskann_utils::ReborrowMut;

// Test that `reborrow_mut` on MatMut correctly captures a mutable lifetime,
// preventing the original MatMut from being used while the reborrow is in scope.
fn main() {
    let mut mat = Mat::new(MinMaxMeta::<1>::new(2, 2), Defaulted).unwrap();
    let mut view: MatMut<'_, _> = mat.as_view_mut();
    let reborrowed = view.reborrow_mut();
    // This should fail: we cannot use `view` while `reborrowed` is still alive
    let _ = view.num_vectors();
    let _ = reborrowed.num_vectors();
}
