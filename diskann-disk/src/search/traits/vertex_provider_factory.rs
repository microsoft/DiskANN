/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;
use diskann_providers::model::graph::traits::GraphDataType;

use super::VertexProvider;
use crate::data_model::GraphHeader;

/// The `VertexProviderFactory` trait provides an interface to create a GraphProvider`. This trait forms an important part
/// of the interaction between the `ANNWrapper` and the creation of `DiskIndexSearcher`. The `ANNWrapper` passes a `VertexProviderFactory` when
/// it initializes a `DiskIndexSearcher` When serving each search request, the `DiskIndexSearcher` opens a `VertexProvider`
/// using the provided `VertexProviderFactory`. There will be two flavors of VertexProviderFactory, one that reads vertex data from data another that reads vertex data from a stream.
///
/// This trait has an associated VertexProvider type that signifies the specific type of VertexProvider which this `VertexProviderFactory` will create.
///
/// # Parameters
/// * `GraphMetadata`: This contains the metadata of the disk index graph, like the number of points, dimension, max_node_length, etc.
///
/// # Functions
/// * `create_vertex_provider`: This function takes a `Metadata` object as an argument and returns a `VertexProvider` object. It also accepts a max batch read sizes which is
///   used to control the maximum number of nodes it can get in one batch.
/// * `get_header`: This function returns the metadata of the graph.
pub trait VertexProviderFactory<Data: GraphDataType>: Send + Sync {
    type VertexProviderType: VertexProvider<Data>;

    fn get_header(&self) -> ANNResult<GraphHeader>;

    fn create_vertex_provider(
        &self,
        max_batch_size: usize,
        header: &GraphHeader,
    ) -> ANNResult<Self::VertexProviderType>;
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_vertex_provider_factory_trait() {
        // Trait definition is verified at compile time
    }
}
