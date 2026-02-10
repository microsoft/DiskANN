/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod storage_provider;
pub use storage_provider::{
    DynWriteProvider, StorageReadProvider, StorageWriteProvider, WriteProviderWrapper, WriteSeek,
};

#[cfg(any(test, feature = "virtual_storage"))]
mod virtual_storage_provider;
#[cfg(any(test, feature = "virtual_storage"))]
pub use virtual_storage_provider::VirtualStorageProvider;

mod api;
pub use api::{AsyncIndexMetadata, AsyncQuantLoadContext, DiskGraphOnly, LoadWith, SaveWith};

pub mod bin;

pub(crate) mod file_storage_provider;
// Use VirtualStorageProvider in tests to avoid filesystem side-effects
#[cfg(not(test))]
pub use file_storage_provider::FileStorageProvider;

mod pq_storage;
pub use pq_storage::PQStorage;

mod sq_storage;
pub use sq_storage::{SQError, SQStorage};

pub mod protos;

pub mod path_utility;
pub use path_utility::{
    get_compressed_pq_file, get_disk_index_compressed_pq_file, get_disk_index_file,
    get_disk_index_pq_pivot_file, get_label_file, get_label_medoids_file, get_mem_index_data_file,
    get_mem_index_file, get_pq_pivot_file, get_universal_label_file,
};

pub mod index_storage;
pub use index_storage::{
    HasStartingPoints, create_load_context,
};
