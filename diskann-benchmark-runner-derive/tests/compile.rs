/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[test]
fn compile_tests() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/pass/*.rs");
    t.compile_fail("tests/ui/fail/*.rs");
}
