/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Helper constructors and type aliases for in-memory DiskANN indexes.
//!
//! These were previously in `diskann-providers/src/index/diskann_async.rs`
//! and have been moved here as part of the `diskann-inmem` crate extraction.

use std::sync::Arc;

use diskann::{
    ANNResult,
    graph::{Config, DiskANNIndex},
    utils::VectorRepr,
};
use diskann_providers::model::graph::provider::async_::common::{
    CreateDeleteProvider, NoDeletes, NoStore,
};
use diskann_utils::future::AsyncFriendly;

use crate::{
    CreateFullPrecision, CreateVectorStore, DefaultProvider, DefaultProviderParameters,
    DefaultQuant, FullPrecisionProvider,
};

pub type MemoryIndex<T, D = NoDeletes> = Arc<DiskANNIndex<FullPrecisionProvider<T, NoStore, D>>>;

pub type QuantMemoryIndex<T, Q, D = NoDeletes> = Arc<DiskANNIndex<FullPrecisionProvider<T, Q, D>>>;

pub type PQMemoryIndex<T, D = NoDeletes> = QuantMemoryIndex<T, DefaultQuant, D>;

pub type QuantOnlyIndex<Q, D = NoDeletes> = DiskANNIndex<DefaultProvider<NoStore, Q, D>>;

pub fn simplified_builder(
    l_search: usize,
    pruned_degree: usize,
    metric: diskann_vector::distance::Metric,
    dim: usize,
    max_points: usize,
    modify: impl FnOnce(&mut diskann::graph::config::Builder),
) -> ANNResult<(Config, DefaultProviderParameters)> {
    let config = diskann::graph::config::Builder::new_with(
        pruned_degree,
        diskann::graph::config::MaxDegree::default_slack(),
        l_search,
        metric.into(),
        modify,
    )
    .build()?;

    let params = DefaultProviderParameters {
        max_points,
        frozen_points: diskann::utils::ONE,
        metric,
        dim,
        prefetch_lookahead: None,
        prefetch_cache_line_level: None,
        max_degree: config.max_degree_u32().get(),
    };

    Ok((config, params))
}

pub fn new_index<T, D>(
    config: Config,
    params: DefaultProviderParameters,
    deleter: D,
) -> ANNResult<MemoryIndex<T, D::Target>>
where
    T: VectorRepr,
    D: CreateDeleteProvider,
    D::Target: AsyncFriendly,
{
    let fp_precursor = CreateFullPrecision::new(params.dim, params.prefetch_cache_line_level);
    let data_provider = DefaultProvider::new_empty(params, fp_precursor, NoStore, deleter)?;
    Ok(Arc::new(DiskANNIndex::new(config, data_provider, None)))
}

pub fn new_quant_index<T, Q, D>(
    config: Config,
    params: DefaultProviderParameters,
    quant: Q,
    deleter: D,
) -> ANNResult<QuantMemoryIndex<T, Q::Target, D::Target>>
where
    T: VectorRepr,
    Q: CreateVectorStore,
    Q::Target: AsyncFriendly,
    D: CreateDeleteProvider,
    D::Target: AsyncFriendly,
{
    let fp_precursor = CreateFullPrecision::new(params.dim, params.prefetch_cache_line_level);
    let data_provider = DefaultProvider::new_empty(params, fp_precursor, quant, deleter)?;
    Ok(Arc::new(DiskANNIndex::new(config, data_provider, None)))
}

pub fn new_quant_only_index<Q, D>(
    config: Config,
    params: DefaultProviderParameters,
    quant: Q,
    deleter: D,
) -> ANNResult<QuantOnlyIndex<Q::Target, D::Target>>
where
    Q: CreateVectorStore,
    Q::Target: AsyncFriendly,
    D: CreateDeleteProvider,
    D::Target: AsyncFriendly,
{
    let data = DefaultProvider::new_empty(params, NoStore, quant, deleter)?;
    Ok(DiskANNIndex::new(config, data, None))
}
