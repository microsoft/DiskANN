/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Numerical kernels for PiPNN graph construction.
//!
//! [`partition_kernel`] converts point-to-leader dot products into sorted leader
//! positions. The output width sets the fanout. A scratch vector stores the
//! ranked leaders and reuses its allocation for each point.
//!
//! [`leaf_kernel`] reads a lower-triangular Gram matrix. It evaluates each point
//! pair once and updates both points. Each point retains at most three local
//! neighbors.
//!
//! `kernel_metric` defines metric markers and shared math. Separate leaf and
//! partition traits define norm preparation and ranking formulas.
//!
//! The graph builder selects architecture `A` and metric `M` once. It passes
//! these concrete types to both kernels.
//!
//! Each kernel checks all view and norm relationships before unchecked SIMD
//! access. The kernels borrow their matrices. They write only to caller-owned
//! output and workspace.
#[allow(dead_code)]
mod kernel_metric;

#[allow(dead_code)]
mod leaf_kernel;
#[allow(dead_code)]
mod partition_kernel;
