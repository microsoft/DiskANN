/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Intantiations for codegen inspection.
//!
//! These methods are **not** part of the public API.
//!
//! Note: when inspecting code generation, make sure the kernels are **also** inspected
//! when compiling with `RUSTFLAGS="-Ctarget-arch=x86-64"` because that can change code
//! generation signficantly and is necessary for multi-architecture compatible builds.

#[cfg(target_arch = "x86_64")]
pub mod x86_64;
