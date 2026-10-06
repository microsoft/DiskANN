/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{
    multi_vector::{Mat, MatMut, Defaulted},
    minmax::MinMaxMeta,
};

// Test that `get_row` on MatMut correctly captures an immutable borrow,
// preventing mutation of the MatMut while the row is in scope.
fn main() {
    let mut mat = Mat::new(MinMaxMeta::<1>::new(2, 2), Defaulted).unwrap();
    let mut view: MatMut<'_, _> = mat.as_view_mut();
    let row = view.get_row(0).unwrap();
    // This should fail: we cannot mutably borrow `view` while `row` exists
    let _ = view.get_row_mut(1);
    std::hint::black_box(row);
}
