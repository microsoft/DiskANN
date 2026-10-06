/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{
    multi_vector::{Mat, Defaulted},
    minmax::MinMaxMeta,
};

// Test that `rows` on Mat correctly captures an immutable borrow,
// preventing mutation of the Mat while the iterator is in scope.
fn main() {
    let mut mat = Mat::new(MinMaxMeta::<1>::new(2, 2), Defaulted).unwrap();
    let iter = mat.rows();
    // This should fail: we cannot mutably borrow `mat` while `iter` exists
    let _ = mat.as_view_mut();
    for row in iter {
        std::hint::black_box(row);
    }
}
