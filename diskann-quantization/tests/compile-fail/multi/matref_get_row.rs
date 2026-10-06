/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{
    multi_vector::{Mat, MatRef, Defaulted},
    minmax::MinMaxMeta,
};


// Test that `get_row` on MatRef returns a row with the correct lifetime,
// and that an immutable borrow is held while the row is in scope.
fn main() {
    let mut mat = Mat::new(MinMaxMeta::<1>::new(2, 2), Defaulted).unwrap();
    let view: MatRef<'_, _> = mat.as_view();
    let row = view.get_row(0).unwrap();
    // This should fail: we cannot mutably borrow `mat` while `row` exists
    // (since `row` holds a reference derived from `mat`)
    let _ = mat.as_view_mut();
    std::hint::black_box(row);
}
