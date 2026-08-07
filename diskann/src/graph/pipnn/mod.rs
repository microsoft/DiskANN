/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Numerical kernels for provider-independent PiPNN graph construction.
//!
//! PiPNN assigns points to overlapping leader partitions, computes one
//! lower-triangular all-pairs matrix per bounded leaf, and merges the selected
//! leaf neighbors into graph candidates. This layer contains only the two score
//! selection kernels; later layers own partition recursion, GEMM, graph IDs,
//! candidate merging, final pruning, providers, and persistence.
//!
//! - [`partition_kernel`] converts a point-by-leader dot-product tile into sorted
//!   leader-column positions. Output width is runtime fanout; reusable tracker
//!   storage grows to that width and is reused across point rows.
//! - [`leaf_kernel`] scans each strict-lower-triangle pair once, updates both
//!   endpoints, and retains up to three leaf-local neighbors per point.
//! - `kernel_metric` owns the scalar/SIMD formulas and exact scale units shared
//!   by both kernels.
//!
//! Runtime architecture and metric selection intentionally do not live in the
//! kernels. The enclosing graph build selects concrete `A` and `M` types once,
//! then carries them through partition and leaf work. This keeps metric matches,
//! trait objects, and stored function pointers out of the hot loops.
//!
//! Both kernels validate view relationships and metric scale layouts before
//! unchecked SIMD access. They borrow all matrices and mutate only caller-owned
//! output and reusable workspace. Partition work is one score per point-leader
//! pair; leaf work is one score per unordered point pair.

// This stack layer introduces the numerical kernels before the following core
// layer wires them into graph construction.
#[allow(dead_code)]
mod kernel_metric;

#[allow(dead_code)]
mod leaf_kernel;
#[allow(dead_code)]
mod partition_kernel;
