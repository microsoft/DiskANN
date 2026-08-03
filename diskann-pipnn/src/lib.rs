/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Numerical kernels for provider-independent PiPNN graph construction.
//!
//! PiPNN means **Pick-in-Partitions Nearest Neighbors**. The wider algorithm
//! builds a graph for approximate nearest-neighbor search: every input vector
//! becomes one graph vertex, and its adjacency list stores other vectors worth
//! visiting during a later query. The APIs exposed in this layer provide PiPNN's
//! numerical selection kernels; they do not yet expose the full graph builder or
//! execute queries.
//!
//! Incremental builders such as Vamana find construction candidates by running
//! beam search against a partially built graph: they repeatedly follow graph
//! edges to discover nearby vertices, causing random memory access. PiPNN removes
//! that search from construction and uses three bulk stages instead:
//!
//! 1. **Partition.** Randomized Ball Carving samples points called *leaders*.
//!    Every point is assigned to its nearest `fanout` leaders. Assigning to more
//!    than one leader makes child groups overlap. Oversized groups are processed
//!    recursively until bounded groups called *leaves* remain.
//! 2. **Pick within leaves.** Vectors in one leaf are contiguous enough for a
//!    dense matrix multiplication to compute all pair dot products. Each point
//!    picks its nearest leaf companions; selected pairs become candidate graph
//!    edges.
//! 3. **Merge and prune.** Candidates from overlapping leaves are combined.
//!    HashPrune can keep a bounded reservoir per source while edges stream in,
//!    retaining the closest candidate for each residual-direction hash. The
//!    alternative collects unique candidates directly. An optional final Vamana
//!    RobustPrune selects a bounded, directionally diverse adjacency list.
//!
//! ```text
//! dataset points
//!      │
//!      v
//! sample leaders + point/leader GEMM
//!      │
//!      v
//! choose nearest leaders ──> overlapping child groups ──> recurse ──> leaves
//!                                                                      │
//!                                                    leaf all-pairs GEMM
//!                                                                      │
//!                                                                      v
//!                                                     pick local neighbors
//!                                                                      │
//!                                                                      v
//!                                                     merge/prune edges
//!                                                                      │
//!                                                                      v
//!                                                          search graph
//! ```
//!
//! This crate keeps GEMM separate from score selection: callers compute dense
//! dot-product matrices, then the kernels documented below convert those dots to
//! metric scores and retain top candidates. A *point* is a vector being assigned
//! during partitioning; a *leader* names a child group. In leaf selection,
//! *source* names the point whose output list is being built and *target* names
//! another point in that same leaf.
//!
//! The wider PiPNN pipeline owns overlapping partition generation, leaf-local
//! nearest-neighbor construction, candidate merging, and optional graph-degree
//! finalization. This layer exports the partition-assignment and leaf-selection
//! kernels used inside that pipeline. Callers supply their dot-product matrices,
//! output storage, and reusable scratch; providers, graph IDs, recursion, edge
//! merging, persistence, and search remain outside these kernel APIs.
//!
//! Numerical kernels include:
//!
//! - [`partition_kernel::PartitionKernel`] converts point-by-leader dot-product
//!   tiles into nearest leader positions.
//! - [`leaf_kernel::LeafKernel`] scans each leaf's lower-triangular dot-product
//!   matrix once and retains nearest non-self neighbors for both endpoints.
//!
//! # Main modules and structures
//!
//! ## [`partition_kernel`]
//!
//! Partition callers first compute a point-by-leader dot-product tile with GEMM.
//! [`partition_kernel::PartitionInput`] bundles that tile with typed
//! [`partition_kernel::PartitionScales`]. A prepared
//! [`partition_kernel::PartitionKernel`] writes sorted leader-local positions to
//! a caller-owned output matrix. Fanout is the output column count and is bounded
//! by [`partition_kernel::MAX_PARTITION_FANOUT`]. Module documentation describes
//! scale units, validation, `process_points`, and tracker insertion.
//!
//! ## [`leaf_kernel`]
//!
//! Leaf callers compute a lower-triangular point-by-point dot matrix with
//! `sgemm_aat_lower`. [`leaf_kernel::LeafInput`] borrows that matrix;
//! [`leaf_kernel::LeafKernelWorkspace`] owns reusable per-worker scratch; and
//! [`leaf_kernel::LeafKernel`] writes sorted [`leaf_kernel::LeafNeighbor`] values
//! to a caller-owned matrix. [`leaf_kernel::leaf_neighbor_count`] derives each
//! leaf's width from its point count and requested `k`. Module documentation
//! describes width selection, `process_pairs`, fixed/dynamic storage, and stable
//! endpoint insertion.
//!
//! ## `kernel_metric`
//!
//! This private module owns metric formulas, scale units, zero/NaN behavior, and
//! one-time runtime-to-concrete metric selection shared by both public kernels.
//! Keeping it private prevents callers from constructing a formula/scale mismatch.
//!
//! # Typical use
//!
//! 1. Prepare one partition and one leaf kernel for the build metric.
//! 2. Reuse the partition handle for every GEMM stripe, changing only borrowed
//!    input/output views.
//! 3. Reuse the leaf handle for every leaf. Derive output width with
//!    [`leaf_kernel::leaf_neighbor_count`] and lease one workspace per worker.
//! 4. Translate leaf-local positions to dataset IDs outside these kernels.
//!
//! Callers prepare these small handles once per build metric and reuse them
//! across stripes or leaves. Each output view supplies its call-specific fanout
//! or neighbor width. Preparation uses `diskann-wide` to select the runtime
//! architecture and returns a direct function pointer; repeated calls do not
//! repeat ISA or metric dispatch. PiPNN itself never names instruction sets.
//!
//! # Ownership and performance boundary
//!
//! Kernels borrow all matrices and mutate only caller-owned output/scratch. They
//! do not own providers, thread pools, GEMM buffers, graph IDs, or persistence.
//! Partition traversal performs one score per point-leader pair; leaf traversal
//! performs one score per unordered point pair. Detailed complexity and scratch
//! costs are documented in each module.

mod kernel_metric;

pub mod leaf_kernel;
pub mod partition_kernel;
