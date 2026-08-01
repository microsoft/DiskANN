/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::Arc;

use diskann::{
    graph::{Config, DiskANNIndex},
    provider::DefaultContext,
    utils::VectorRepr,
    ANNResult,
};
use diskann_providers::{
    index::diskann_async,
    model::graph::provider::async_::{
        common::{FullPrecision, NoDeletes, NoStore, Quantized},
        inmem::{
            DefaultProvider, DefaultProviderParameters, DefaultQuant, FullPrecisionProvider,
            SQStore, SetStartPoints,
        },
    },
    storage::{DiskGraphOnly, DynWriteProvider, SaveWith, WriteProviderWrapper},
};

use crate::build::builder::quantizer::BuildQuantizer;

type FullPrecisionIndex<T> = DiskANNIndex<FullPrecisionProvider<T>>;
type ScalarQuantizedIndex = DiskANNIndex<DefaultProvider<NoStore, SQStore<1>>>;
type ProductQuantizedIndex = DiskANNIndex<DefaultProvider<NoStore, DefaultQuant>>;

/// Index implementation used while constructing a Vamana graph.
pub(super) enum VamanaBuildIndex<T>
where
    T: VectorRepr,
{
    FullPrecision(Arc<FullPrecisionIndex<T>>),
    ScalarQuantized(Arc<ScalarQuantizedIndex>),
    ProductQuantized(Arc<ProductQuantizedIndex>),
}

/// Manual implementation: `#[derive(Clone)]` would incorrectly require `T: Clone`,
/// even though `T` only appears behind `Arc`.
impl<T> Clone for VamanaBuildIndex<T>
where
    T: VectorRepr,
{
    fn clone(&self) -> Self {
        match self {
            Self::FullPrecision(index) => Self::FullPrecision(Arc::clone(index)),
            Self::ScalarQuantized(index) => Self::ScalarQuantized(Arc::clone(index)),
            Self::ProductQuantized(index) => Self::ProductQuantized(Arc::clone(index)),
        }
    }
}

impl<T> VamanaBuildIndex<T>
where
    T: VectorRepr,
{
    pub(super) fn new(
        config: Config,
        params: DefaultProviderParameters,
        build_quantizer: &BuildQuantizer,
    ) -> ANNResult<Self> {
        match build_quantizer {
            BuildQuantizer::NoQuant(_) => {
                diskann_async::new_index::<T, _>(config, params, NoDeletes).map(Self::FullPrecision)
            }
            BuildQuantizer::Scalar1Bit(quantizer) => {
                let index = diskann_async::new_quant_only_index(
                    config,
                    params,
                    quantizer.clone(),
                    NoDeletes,
                )?;
                Ok(Self::ScalarQuantized(Arc::new(index)))
            }
            BuildQuantizer::PQ(quantizer) => {
                let index = diskann_async::new_quant_only_index(
                    config,
                    params,
                    quantizer.clone(),
                    NoDeletes,
                )?;
                Ok(Self::ProductQuantized(Arc::new(index)))
            }
        }
    }

    pub(super) fn capacity(&self) -> usize {
        match self {
            Self::FullPrecision(index) => index.provider().capacity(),
            Self::ScalarQuantized(index) => index.provider().capacity(),
            Self::ProductQuantized(index) => index.provider().capacity(),
        }
    }

    pub(super) fn total_points(&self) -> usize {
        match self {
            Self::FullPrecision(index) => index.provider().total_points(),
            Self::ScalarQuantized(index) => index.provider().total_points(),
            Self::ProductQuantized(index) => index.provider().total_points(),
        }
    }

    pub(super) fn set_start_point(&self, start_point: &[T]) -> ANNResult<()> {
        match self {
            Self::FullPrecision(index) => index
                .provider()
                .set_start_points(std::iter::once(start_point)),
            Self::ScalarQuantized(index) => index
                .provider()
                .set_start_points(std::iter::once(start_point)),
            Self::ProductQuantized(index) => index
                .provider()
                .set_start_points(std::iter::once(start_point)),
        }
    }

    pub(super) async fn insert_vector(&self, id: u32, vector: &[T]) -> ANNResult<()> {
        match self {
            Self::FullPrecision(index) => {
                index
                    .insert(&FullPrecision, &DefaultContext, &id, vector)
                    .await
            }
            Self::ScalarQuantized(index) => {
                index.insert(&Quantized, &DefaultContext, &id, vector).await
            }
            Self::ProductQuantized(index) => {
                index.insert(&Quantized, &DefaultContext, &id, vector).await
            }
        }
    }

    pub(super) async fn final_prune(&self, range: core::ops::Range<u32>) -> ANNResult<()> {
        match self {
            Self::FullPrecision(index) => {
                index
                    .prune_range(&FullPrecision, &DefaultContext, range)
                    .await
            }
            Self::ScalarQuantized(index) => {
                index.prune_range(&Quantized, &DefaultContext, range).await
            }
            Self::ProductQuantized(index) => {
                index.prune_range(&Quantized, &DefaultContext, range).await
            }
        }
    }

    pub(super) async fn save_graph(
        &self,
        storage_provider: &dyn DynWriteProvider,
        start_point_and_path: &(u32, DiskGraphOnly),
    ) -> ANNResult<()> {
        let wrapper = WriteProviderWrapper::new(storage_provider);
        match self {
            Self::FullPrecision(index) => index.save_with(&wrapper, start_point_and_path).await,
            Self::ScalarQuantized(index) => index.save_with(&wrapper, start_point_and_path).await,
            Self::ProductQuantized(index) => index.save_with(&wrapper, start_point_and_path).await,
        }
    }

    #[cfg(debug_assertions)]
    pub(super) fn counts_for_get_vector(&self) -> (usize, usize) {
        match self {
            Self::FullPrecision(index) => index.provider().counts_for_get_vector(),
            Self::ScalarQuantized(index) => index.provider().counts_for_get_vector(),
            Self::ProductQuantized(index) => index.provider().counts_for_get_vector(),
        }
    }

    #[cfg(debug_assertions)]
    pub(super) async fn count_reachable_nodes(&self) -> ANNResult<usize> {
        match self {
            Self::FullPrecision(index) => {
                let provider = index.provider();
                let start_points = provider.starting_points()?;
                let mut neighbor_accessor = provider.neighbors();
                index
                    .count_reachable_nodes(&start_points, &mut neighbor_accessor)
                    .await
            }
            Self::ScalarQuantized(index) => {
                let provider = index.provider();
                let start_points = provider.starting_points()?;
                let mut neighbor_accessor = provider.neighbors();
                index
                    .count_reachable_nodes(&start_points, &mut neighbor_accessor)
                    .await
            }
            Self::ProductQuantized(index) => {
                let provider = index.provider();
                let start_points = provider.starting_points()?;
                let mut neighbor_accessor = provider.neighbors();
                index
                    .count_reachable_nodes(&start_points, &mut neighbor_accessor)
                    .await
            }
        }
    }
}
