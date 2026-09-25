/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Concrete [`AlignedReaderFactory`](super::traits::AlignedReaderFactory) implementations.

mod file;
pub use file::AlignedFileReaderFactory;

#[cfg(test)]
mod virtual_storage;
#[cfg(test)]
pub(crate) use virtual_storage::VirtualAlignedReaderFactory;
