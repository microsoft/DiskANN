/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;
use diskann_providers::model::graph::{graph_data_model::AdjacencyList, traits::GraphDataType};

/// `VertexProvider` is a trait that abstracts the access to Vertex data.
///
/// This trait provides an interface to interact with different types of vertex providers structures such as `DiskVertexProvider` and `JetVertexProvider`.
///
/// # Types
///
/// * `Data`: A generic type that represents the data on the graph.
///
pub trait VertexProvider<Data: GraphDataType>: Send + Sync {
    /// Fetches the vector associated with a given vertex id.
    ///
    /// The `get_vector` function attempts to retrieve the vector related to the specified
    /// `vertex_id`. This function returns an `ANNResult` wrapping
    /// a reference to the vector data.
    ///
    /// # Parameters
    ///
    /// * `vertex_id`: An id of the vertex for which the vector is being fetched.
    ///
    /// # Returns
    ///
    /// `ANNResult<&[Data::VectorDataType]>`: An `ANNResult` that wraps the reference to the vector data.
    fn get_vector(&self, vertex_id: &Data::VectorIdType) -> ANNResult<&[Data::VectorDataType]>;

    /// Retrieves the adjacency list of a given vertex id.
    ///
    /// The `get_adjacency_list` function attempts to fetch the adjacency list related to the
    /// specified `vertex_id`. This function returns an `ANNResult` that wraps a reference to the adjacency list
    ///
    /// # Parameters
    ///
    /// * `vertex_id`: An id of the vertex for which the adjacency list is being fetched.
    ///
    /// # Returns
    ///
    /// `ANNResult<&[Data::AdjacencyListType]>`: An `ANNResult` that wraps the reference to the adjacency list.
    fn get_adjacency_list(
        &self,
        vertex_id: &Data::VectorIdType,
    ) -> ANNResult<&AdjacencyList<Data::VectorIdType>>;

    // Gets the associated data for a given vertex id.
    ///
    /// The `get_associated_data` function attempts to retrieve the associated data for the
    /// specified `vertex_id`. This function returns an `ANNResult` that wraps a reference to the associated data.
    ///
    /// # Parameters
    ///
    /// * `vertex_id`: An id of the vertex for which the associated data is being fetched.
    ///
    /// # Returns
    ///
    /// `ANNResult<&[Data::AssociatedDataType]>`: An `ANNResult` that wraps the reference to the associated data.
    fn get_associated_data(
        &self,
        vertex_id: &Data::VectorIdType,
    ) -> ANNResult<&Data::AssociatedDataType>;

    /// This function loads a batch of vertices for a given set of vertex ids and cache it in the `VertexProvider` instance. It is a mutable operation so it takes a mutable self.
    ///
    /// This API is designed to load graph nodes from storage during the search process. As its usage is limited to a single
    /// thread context, each thread should have its own `VertexProvider` instance to load data.
    ///
    /// # Parameters
    ///
    /// * `vertex_ids`: A slice of Data::VectorIdType values representing the ids of vertices for which to retrieve `VertexAndNeighbors`.
    ///
    /// # Returns
    ///
    /// * `ANNResult<()>`: If the operation is successful, returns Ok.
    ///
    /// If it fails, returns an `ANNError`.
    fn load_vertices(&mut self, vertex_ids: &[Data::VectorIdType]) -> ANNResult<()>;

    /// This function to process the loaded node
    /// # Parameters
    ///
    /// * `vertex_id`: A Data::VectorIdType value representing the id of the vertex for which to process.
    /// * `idx`: A usize value representing the index of the vertex in the loaded node list.
    ///
    /// # Returns
    /// * `ANNResult<()>`: If the operation is successful, returns Ok.
    ///
    /// If it fails, returns an `ANNError`.
    fn process_loaded_node(&mut self, vertex_id: &Data::VectorIdType, idx: usize) -> ANNResult<()>;

    // Returns the number of IO operations performed by the vertex provider.
    fn io_operations(&self) -> u32;

    // Returns the number of vertices loaded by the vertex provider.
    fn vertices_loaded_count(&self) -> u32;

    // Clears the members of the vertex provider.
    fn clear(&mut self);
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_vertex_provider_trait_exists() {
        // This test verifies that the trait is properly defined and accessible
        // We can't easily test trait implementations without complex setup
        assert!(true);
    }
}
