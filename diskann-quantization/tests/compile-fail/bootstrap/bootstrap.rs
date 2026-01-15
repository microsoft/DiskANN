/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// A trivially compilable example so we can start our compile tests with
//
// `https://docs.rs/trybuild/latest/trybuild/struct.TestCases.html#method.pass`.
//
// This forces `try-build` to actually compile the binaries, triggering post-monomorphization
// errors, which are otherwise not caught.
fn main() {}
