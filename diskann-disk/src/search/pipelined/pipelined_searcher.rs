/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Public API for pipelined disk search.

use std::sync::Arc;

use diskann::{utils::VectorRepr, ANNResult};
use diskann_providers::model::{
    graph::traits::GraphDataType, PQData, PQScratch,
};
use diskann_vector::distance::Metric;

use crate::{
    data_model::GraphHeader,
    search::provider::disk_provider::{SearchResult, SearchResultItem, SearchResultStats},
    utils::QueryStatistics,
};

use super::pipelined_reader::PipelinedReader;
use super::pipelined_search::{pipe_search, PipeSearchResult};

/// A pipelined disk index searcher implementing the PipeANN algorithm.
///
/// Analogous to `DiskIndexSearcher` but uses pipelined IO (non-blocking io_uring
/// submit/poll) to overlap IO and compute within a single query.
pub struct PipelinedSearcher<Data: GraphDataType<VectorIdType = u32>> {
    graph_header: GraphHeader,
    distance_comparer: <Data::VectorDataType as VectorRepr>::Distance,
    pq_data: Arc<PQData>,
    metric: Metric,
    /// Maximum IO operations per search (reserved for future IO budget enforcement).
    #[allow(dead_code)]
    search_io_limit: usize,
    /// Default beam width when not overridden per-query.
    #[allow(dead_code)]
    initial_beam_width: usize,
    relaxed_monotonicity_l: Option<usize>,
    disk_index_path: String,
}

impl<Data> PipelinedSearcher<Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    /// Create a new pipelined searcher.
    ///
    /// # Arguments
    /// * `graph_header` - Graph metadata from the disk index.
    /// * `pq_data` - Shared PQ data for approximate distance computation.
    /// * `metric` - Distance metric (L2, InnerProduct, etc.).
    /// * `search_io_limit` - Maximum IO operations per search.
    /// * `initial_beam_width` - Initial number of concurrent IOs (adapts during search).
    /// * `relaxed_monotonicity_l` - Optional early termination parameter.
    /// * `disk_index_path` - Path to the disk index file for creating readers.
    pub fn new(
        graph_header: GraphHeader,
        pq_data: Arc<PQData>,
        metric: Metric,
        search_io_limit: usize,
        initial_beam_width: usize,
        relaxed_monotonicity_l: Option<usize>,
        disk_index_path: String,
    ) -> ANNResult<Self> {
        let dims = graph_header.metadata().dims;
        let distance_comparer = Data::VectorDataType::distance(metric, Some(dims));
        Ok(Self {
            graph_header,
            distance_comparer,
            pq_data,
            metric,
            search_io_limit,
            initial_beam_width,
            relaxed_monotonicity_l,
            disk_index_path,
        })
    }

    /// Perform a pipelined search on the disk index.
    ///
    /// # Arguments
    /// * `query` - The query vector.
    /// * `return_list_size` - Number of results to return (k).
    /// * `search_list_size` - Size of the candidate pool (L).
    /// * `beam_width` - Maximum beam width for pipelined IO.
    pub fn search(
        &self,
        query: &[Data::VectorDataType],
        return_list_size: u32,
        search_list_size: u32,
        beam_width: usize,
    ) -> ANNResult<SearchResult<Data::AssociatedDataType>> {
        let metadata = self.graph_header.metadata();
        let dims = metadata.dims;
        let node_len = metadata.node_len;
        let num_nodes_per_sector = metadata.num_nodes_per_block;
        let fp_vector_len =
            (dims * std::mem::size_of::<Data::VectorDataType>()) as u64;
        let medoid = metadata.medoid as u32;

        let mut block_size = self.graph_header.block_size() as usize;
        let version = self.graph_header.layout_version();
        if (version.major_version() == 0 && version.minor_version() == 0) || block_size == 0 {
            block_size = 4096;
        }

        let num_sectors_per_node = if num_nodes_per_sector > 0 {
            1
        } else {
            (node_len as usize).div_ceil(block_size)
        };
        let slot_size = num_sectors_per_node * block_size;

        let max_slots = (beam_width * 2).clamp(16, super::pipelined_reader::MAX_IO_CONCURRENCY);

        // Create a per-call reader
        let mut reader = PipelinedReader::new(
            &self.disk_index_path,
            max_slots,
            slot_size,
            block_size,
        )?;

        let graph_degree = self.graph_header.max_degree::<Data::VectorDataType>()?;
        let num_pq_chunks = self.pq_data.get_num_chunks();
        let num_pq_centers = self.pq_data.get_num_centers();

        let mut pq_scratch = PQScratch::new(
            graph_degree,
            dims,
            num_pq_chunks,
            num_pq_centers,
        )?;

        let result: PipeSearchResult = pipe_search::<Data::VectorDataType>(
            &mut reader,
            &self.pq_data,
            &self.distance_comparer,
            query,
            return_list_size as usize,
            search_list_size as usize,
            beam_width,
            medoid,
            dims,
            node_len,
            num_nodes_per_sector,
            block_size,
            fp_vector_len,
            &mut pq_scratch,
            self.relaxed_monotonicity_l,
            self.metric,
        )?;

        let query_statistics = QueryStatistics {
            total_execution_time_us: result.stats.total_us,
            io_time_us: result.stats.io_us,
            cpu_time_us: result.stats.cpu_us,
            total_io_operations: result.stats.io_count,
            total_comparisons: result.stats.comparisons,
            total_vertices_loaded: result.stats.io_count,
            search_hops: result.stats.hops,
            ..Default::default()
        };

        let stats = SearchResultStats {
            cmps: result.stats.comparisons,
            result_count: result.ids.len() as u32,
            query_statistics,
        };

        let mut results = Vec::with_capacity(result.ids.len());
        for (id, dist) in result.ids.iter().zip(result.distances.iter()) {
            results.push(SearchResultItem {
                vertex_id: *id,
                distance: *dist,
                data: Data::AssociatedDataType::default(),
            });
        }

        Ok(SearchResult { results, stats })
    }
}
