/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// Core storage traits and implementations — re-exported from diskann-storage.
pub use diskann_storage::{
    DynWriteProvider, FileStorageProvider, StorageReadProvider, StorageWriteProvider,
    WriteProviderWrapper, WriteSeek,
};

#[cfg(any(test, feature = "virtual_storage"))]
pub use diskann_storage::VirtualStorageProvider;

pub use diskann_storage::path_utility;
pub use diskann_storage::path_utility::{
    get_compressed_pq_file, get_disk_index_compressed_pq_file, get_disk_index_file,
    get_disk_index_pq_pivot_file, get_label_file, get_label_medoids_file, get_mem_index_data_file,
    get_mem_index_file, get_pq_pivot_file, get_universal_label_file,
};

mod api;
pub use api::{AsyncIndexMetadata, AsyncQuantLoadContext, DiskGraphOnly, LoadWith, SaveWith};

pub(crate) mod bin;

mod pq_storage;
pub use pq_storage::PQStorage;

mod sq_storage;
pub use sq_storage::SQStorage;

pub mod protos;

pub mod index_storage;
pub use index_storage::{
    create_load_context, load_fp_index, load_index_with_deletes, load_pq_index,
    load_pq_index_with_deletes,
};
