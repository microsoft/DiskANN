/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// Borrow/lifetime/variance guarantees of the `Mat`/`MatRef`/`MatMut` matrix framework.
// These are check-time errors, so no `pass` bootstrap is needed to force monomorphization.
#![cfg(not(miri))]

#[test]
fn compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile-fail/*.rs");
}
