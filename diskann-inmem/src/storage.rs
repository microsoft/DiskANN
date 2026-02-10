/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Load functions for in-memory indexes from storage.
//!
//! These were previously in `diskann-providers/src/storage/index_storage.rs`
//! and have been moved here as part of the `diskann-inmem` crate extraction.

use diskann::{graph::DiskANNIndex, utils::VectorRepr, ANNResult};
use diskann_providers::storage::{
    AsyncQuantLoadContext, LoadWith, StorageReadProvider,
};
use diskann_utils::future::AsyncFriendly;

use diskann_providers::model::configuration::IndexConfiguration;

use crate::{DefaultProvider, FullPrecisionProvider};

pub async fn load_fp_index<T, P, Q>(
    provider: &P,
    path: &str,
    config: IndexConfiguration,
) -> ANNResult<DiskANNIndex<FullPrecisionProvider<T, Q>>>
where
    P: StorageReadProvider,
    T: VectorRepr,
    Q: AsyncFriendly,
    FullPrecisionProvider<T, Q>: LoadWith<AsyncQuantLoadContext, Error = diskann::ANNError>,
{
    DiskANNIndex::load_with(provider, &(path, config)).await
}

pub async fn load_index<P, U, V>(
    provider: &P,
    path: &str,
    config: IndexConfiguration,
) -> ANNResult<DiskANNIndex<DefaultProvider<U, V>>>
where
    P: StorageReadProvider,
    U: AsyncFriendly,
    V: AsyncFriendly,
    DefaultProvider<U, V>: LoadWith<AsyncQuantLoadContext, Error = diskann::ANNError>,
{
    DiskANNIndex::load_with(provider, &(path, config)).await
}
