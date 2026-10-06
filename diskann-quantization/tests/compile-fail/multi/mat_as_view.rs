/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{
    multi_vector::{Mat, Defaulted},
    minmax::MinMaxMeta,
};

// Test that `as_view` on Mat correctly captures an immutable borrow,
// preventing mutation of the Mat while the view is in scope.
fn main() {
    let mut mat = Mat::new(MinMaxMeta::<1>::new(2, 2), Defaulted).unwrap();
    let view = mat.as_view();
    // This should fail: we cannot mutably borrow `mat` while `view` exists
    let _ = mat.as_view_mut();
    let _ = view.num_vectors();
}
