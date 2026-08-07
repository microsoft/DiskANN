/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Numerical kernels for PiPNN graph construction.
//!
//! [`partition_kernel`] converts point-to-leader dot products into sorted leader
//! positions. The output width sets the fanout. One workspace stores the
//! runtime-sized tracker and reuses it for each point.
//!
//! [`leaf_kernel`] reads a lower-triangular Gram matrix. It evaluates each point
//! pair once and updates both points. Each point retains at most three local
//! neighbors.
//!
//! `kernel_metric` defines the scalar and SIMD formulas. It also defines the
//! required norm units for each metric.
//!
//! The graph builder selects architecture `A` and metric `M` once. It passes
//! these concrete types to both kernels.
//!
//! Each kernel checks all view and scale relationships before unchecked SIMD
//! access. The kernels borrow their matrices. They write only to caller-owned
//! output and workspace.
#[allow(dead_code)]
mod kernel_metric;

#[allow(dead_code)]
mod leaf_kernel;
#[allow(dead_code)]
mod partition_kernel;
