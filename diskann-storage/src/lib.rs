/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Storage abstraction layer for DiskANN.
//!
//! This crate provides traits and implementations for reading from and writing
//! to storage backends. The [`StorageReadProvider`] and [`StorageWriteProvider`]
//! traits abstract over concrete storage systems so that the same code can
//! operate against a local filesystem, an in-memory virtual filesystem, or any
//! other backend.
//!
//! # Crate features
//!
//! - **`virtual_storage`** — enables [`VirtualStorageProvider`], an in-memory
//!   or overlay filesystem useful for testing without touching the real
//!   filesystem.

#![cfg_attr(
    not(test),
    warn(clippy::panic, clippy::unwrap_used, clippy::expect_used)
)]

mod storage_provider;
pub use storage_provider::{
    DynWriteProvider, StorageReadProvider, StorageWriteProvider, WriteProviderWrapper, WriteSeek,
};

mod file_storage_provider;
pub use file_storage_provider::FileStorageProvider;

#[cfg(any(test, feature = "virtual_storage"))]
mod virtual_storage_provider;
#[cfg(any(test, feature = "virtual_storage"))]
pub use virtual_storage_provider::VirtualStorageProvider;

pub mod path_utility;
pub use path_utility::{
    get_compressed_pq_file, get_disk_index_compressed_pq_file, get_disk_index_file,
    get_disk_index_pq_pivot_file, get_label_file, get_label_medoids_file, get_mem_index_data_file,
    get_mem_index_file, get_pq_pivot_file, get_universal_label_file,
};

mod dataset_dto;
pub use dataset_dto::DatasetDto;

pub mod proto_storage;
pub use proto_storage::{ProtoStorageError, load_proto, save_proto};
