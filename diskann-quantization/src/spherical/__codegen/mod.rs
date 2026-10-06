/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Instantiations for codegen inspection.
//!
//! These methods are **not** part of the public API.

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
