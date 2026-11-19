/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(crate) mod common;
pub(crate) use common::square_norm;

#[cfg(feature = "linalg")]
pub mod gemm;
pub mod lloyds;
pub mod plusplus;
