/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{
    multi_vector::{Mat, MatRef, Defaulted},
    minmax::MinMaxMeta,
};

// Test that `rows` on MatRef returns an iterator with the correct lifetime,
// preventing mutation of the underlying Mat while iterating.
fn main() {
    let mut mat = Mat::new(MinMaxMeta::<1>::new(2, 2), Defaulted).unwrap();
    let view: MatRef<'_, _> = mat.as_view();
    let iter = view.rows();
    // This should fail: we cannot mutably borrow `mat` while `iter` exists
    let _ = mat.as_view_mut();
    for row in iter {
        std::hint::black_box(row);
    }
}
