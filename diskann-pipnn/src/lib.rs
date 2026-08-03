/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Numerical kernels used by PiPNN graph construction.
//!
//! PiPNN first partitions points around sampled leaders, then builds local
//! neighbor candidates inside each leaf. This crate owns the numerical seams
//! of those stages while callers retain dataset storage, GEMM workspaces, graph
//! policy, and scheduling:
//!
//! - [`partition_kernel`] converts a point-by-leader dot-product tile into the
//!   nearest leader positions for each point.
//! - [`leaf_kernel`] scans a leaf's lower-triangular dot-product matrix once and
//!   retains nearest non-self neighbors for both endpoints.
//!
//! Both modules validate slice shapes before dispatch and use `diskann-wide` for
//! architecture selection; PiPNN does not detect or name instruction sets.

pub mod leaf_kernel;
pub mod partition_kernel;
