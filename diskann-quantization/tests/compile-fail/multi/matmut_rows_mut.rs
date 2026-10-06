/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{
    multi_vector::{Mat, MatMut, Defaulted},
    minmax::MinMaxMeta,
};

// Test that the `rows_mut` iterator on MatMut correctly captures a mutable lifetime,
// preventing the MatMut from being used while the iterator is in scope.
fn main() {
    let mut mat = Mat::new(MinMaxMeta::<1>::new(2, 2), Defaulted).unwrap();
    let mut view: MatMut<'_, _> = mat.as_view_mut();
    let iter = view.rows_mut();
    // This should fail: we cannot use `view` while the mutable iterator is alive
    let _ = view.num_vectors();
    for row in iter {
        std::hint::black_box(row);
    }
}
