/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Provider-independent PiPNN graph construction.
//!
//! The crate owns overlapping partition generation, leaf-local nearest-neighbor
//! construction, candidate merging, and optional graph-degree finalization. The
//! caller supplies contiguous data, DiskANN graph policy, and the Rayon pool.
//! Providers, start/frozen points, quantization, persistence, and search remain
//! outside this algorithm seam.
//!
//! Numerical kernels include:
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
