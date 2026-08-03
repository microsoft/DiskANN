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
//! - [`partition_kernel::PartitionKernel`] converts point-by-leader dot-product
//!   tiles into nearest leader positions.
//! - [`leaf_kernel::LeafKernel`] scans each leaf's lower-triangular dot-product
//!   matrix once and retains nearest non-self neighbors for both endpoints.
//!
//! Callers prepare these small handles once per build metric (and leaf `k`) and
//! reuse them across stripes or leaves. Preparation uses `diskann-wide` to select
//! the runtime architecture and returns a direct function pointer; repeated calls
//! do not repeat ISA or metric dispatch. PiPNN itself never names instruction
//! sets.

mod kernel_metric;

pub mod leaf_kernel;
pub mod partition_kernel;
