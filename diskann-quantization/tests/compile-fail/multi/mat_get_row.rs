/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{
    multi_vector::{Mat, Defaulted},
    minmax::MinMaxMeta,
};

// Test that `get_row` on Mat correctly captures an immutable borrow,
// preventing mutation of the Mat while the row is in scope.
fn main() {
    let mut mat = Mat::new(MinMaxMeta::<1>::new(2, 2), Defaulted).unwrap();
    let row = mat.get_row(0).unwrap();
    // This should fail: we cannot mutably borrow `mat` while `row` exists
    let _ = mat.get_row_mut(1);
    std::hint::black_box(row);
}
