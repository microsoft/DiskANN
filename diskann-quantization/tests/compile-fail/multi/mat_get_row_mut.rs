/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{
    multi_vector::{Mat, Defaulted},
    minmax::MinMaxMeta,
};

// Test that `get_row_mut` on Mat correctly captures a mutable
// lifetime, preventing the Mat from being used while the row is in scope.
fn main() {
    let mut mat = Mat::new(MinMaxMeta::<1>::new(2, 2), Defaulted).unwrap();
    let row = mat.get_row_mut(0).unwrap();
    // This should fail: we cannot use `mat` while `row` is still borrowed
    let _ = mat.num_vectors();
    std::hint::black_box(row);
}
