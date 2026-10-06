/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{
    multi_vector::{Mat, Defaulted},
    minmax::MinMaxMeta,
};
use diskann_utils::ReborrowMut;

// Test that `reborrow_mut` on Mat correctly captures a mutable borrow,
// preventing use of the Mat while the reborrow is in scope.
fn main() {
    let mut mat = Mat::new(MinMaxMeta::<1>::new(2, 2), Defaulted).unwrap();
    let view = mat.reborrow_mut();
    // This should fail: we cannot use `mat` while `view` exists
    let _ = mat.num_vectors();
    let _ = view.num_vectors();
}
