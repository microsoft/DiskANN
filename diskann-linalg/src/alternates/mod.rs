/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! This module contains alternate linear algebra implementations via Intel MKL.
//!
//! # Why this module is not published by default
//!
//! This module is not published by default due to transitive build dependencies
//! (`intel-mkl-src` and `intel-mkl-sys` crates) which depend on the `ring` crate,
//! which has not been approved by the Microsoft crypto board.
//!
//! Additionally, MKL is Intel-only, limiting its platform compatibility.
//!
//! # How to enable MKL support
//!
//! To enable MKL support, add the following dependencies to your Cargo.toml:
//!
//! ```toml
//! [dependencies]
//! cblas = { version = "0.4.0", optional = true }
//! lapacke = { version = "0.5.0", optional = true }
//! intel-mkl-src = { version = "0.8.1", features = ["mkl-static-lp64-seq"], optional = true }
//! ```
