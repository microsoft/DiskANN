/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Built-in methods for invoking streaming operations on a [`diskann::graph::DiskANNIndex`].
//!
//! Unlike other components defined in [`crate::build::graph`] or [`crate::search::graph`],
//! these components are intended to be used as building blocks for implementing
//! [`crate::streaming::Stream`]. The key difference is that these components take an
//! argument that is [`crate::build::ids::ToIdSized`] to describe the points to be processed,
//! rather than processing a collection of points in a dataset.
//!
//! That said, these components implement the [`crate::build::Build`] trait and thus reuse
//! the infrastructure defined in [`crate::build`].

pub mod drop_deleted;
pub mod inplace_delete;

pub use drop_deleted::DropDeleted;
pub use inplace_delete::InplaceDelete;

#[cfg(test)]
mod test;
