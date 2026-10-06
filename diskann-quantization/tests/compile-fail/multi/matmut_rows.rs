/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{
    multi_vector::{Mat, MatMut, Defaulted},
    minmax::MinMaxMeta,
};

// Test that `rows` on MatMut correctly captures an immutable borrow,
// preventing mutation of the MatMut while the iterator is in scope.
fn main() {
    let mut mat = Mat::new(MinMaxMeta::<1>::new(2, 2), Defaulted).unwrap();
    let mut view: MatMut<'_, _> = mat.as_view_mut();
    let iter = view.rows();
    // This should fail: we cannot mutably borrow `view` while `iter` exists
    let _ = view.get_row_mut(0);
    for row in iter {
        std::hint::black_box(row);
    }
}
