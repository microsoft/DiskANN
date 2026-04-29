/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::data_model::GraphDataType;
use diskann::ANNResult;

use super::VertexProvider;
use crate::data_model::GraphHeader;

/// The `VertexProviderFactory` trait provides an interface to create a `VertexProvider`. This trait forms an important part
/// of the interaction between the `DiskIndexSearcher` and a `VertexProvider`. A `VertexProviderFactory` is passed to `DiskIndexSearcher` when
/// it is constructed. When serving each search request, the `DiskIndexSearcher` opens a `VertexProvider`
/// using the provided `VertexProviderFactory`.
///
/// This trait has an associated VertexProvider type that signifies the specific type of VertexProvider which this `VertexProviderFactory` will create.
///
/// # Parameters
/// * `Data`: A `GraphDataType` that defines the vector element type, id type, and associated payload type for the graph.
///
/// # Functions
/// * `create_vertex_provider`: This function takes a `GraphHeader` reference and a max batch size and returns a `VertexProvider` object.
///   The max batch size controls the maximum number of nodes that can be loaded in a single batch.
/// * `get_header`: This function returns the header of the graph.
pub trait VertexProviderFactory<Data: GraphDataType>: Send + Sync {
    type VertexProviderType: VertexProvider<Data>;

    fn get_header(&self) -> ANNResult<GraphHeader>;

    fn create_vertex_provider(
        &self,
        max_batch_size: usize,
        header: &GraphHeader,
    ) -> ANNResult<Self::VertexProviderType>;
}
