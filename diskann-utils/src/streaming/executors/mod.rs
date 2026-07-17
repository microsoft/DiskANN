/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # Provided Executors.
//!
//! Built-in implementations of [`Executor`] for common streaming inputs.

#[cfg(feature = "bigann")]
#[cfg_attr(docsrs, doc(cfg(feature = "bigann")))]
pub mod bigann;
