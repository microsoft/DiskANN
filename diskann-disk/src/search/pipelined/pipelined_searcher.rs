/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Public API for pipelined disk search.

use std::sync::Arc;

use diskann::{
    utils::{
        object_pool::{ObjectPool, PoolOption, TryAsPooled},
        VectorRepr,
    },
    ANNError, ANNResult,
};
use diskann_providers::model::{
    graph::traits::GraphDataType, PQData, PQScratch,
};
use diskann_vector::distance::Metric;

use crate::{
    data_model::GraphHeader,
    search::provider::disk_provider::{SearchResult, SearchResultItem, SearchResultStats},
    utils::QueryStatistics,
};

use super::pipelined_reader::{PipelinedReader, PipelinedReaderConfig};
use super::pipelined_search::{pipe_search, PipeSearchResult};
use crate::search::search_trace::SearchTrace;

/// Scratch space for pipelined search operations, pooled for reuse across queries.
struct PipelinedSearchScratch {
    reader: PipelinedReader,
    pq_scratch: PQScratch,
}

/// Arguments for creating or resetting a [`PipelinedSearchScratch`].
#[derive(Clone)]
struct PipelinedScratchArgs<'a> {
    disk_index_path: &'a str,
    max_slots: usize,
    slot_size: usize,
    alignment: usize,
    graph_degree: usize,
    dims: usize,
    num_pq_chunks: usize,
    num_pq_centers: usize,
    reader_config: PipelinedReaderConfig,
}

impl TryAsPooled<&PipelinedScratchArgs<'_>> for PipelinedSearchScratch {
    type Error = ANNError;

    fn try_create(args: &PipelinedScratchArgs<'_>) -> Result<Self, Self::Error> {
        let reader = PipelinedReader::new(
            args.disk_index_path,
            args.max_slots,
            args.slot_size,
            args.alignment,
            &args.reader_config,
        )?;
        let pq_scratch = PQScratch::new(
            args.graph_degree,
            args.dims,
            args.num_pq_chunks,
            args.num_pq_centers,
        )?;
        Ok(Self { reader, pq_scratch })
    }

    fn try_modify(&mut self, _args: &PipelinedScratchArgs<'_>) -> Result<(), Self::Error> {
        self.reader.reset();
        Ok(())
    }
}

/// A pipelined disk index searcher implementing the PipeANN algorithm.
///
/// # Deprecation
///
/// This standalone searcher duplicates the generic search loop. Prefer using
/// `DiskIndexSearcher::search_pipelined()` which integrates pipelined IO via the
/// queue-based `ExpandBeam` trait, providing the same IO/compute overlap without
/// code duplication.
///
/// # Safety
///
/// This searcher is designed for **read-only search on completed (static) disk indices**.
/// It opens independent file descriptors with O_DIRECT and reads raw sectors without
/// going through the synchronized `DiskProvider` path. It must NOT be used concurrently
/// with index build, insert, or delete operations on the same index file.
///
/// For search during streaming or dynamic index operations, use [`DiskIndexSearcher`]
/// (beam search) instead, which provides proper synchronization through the
/// `DiskProvider` and `VertexProvider` abstractions.
///
/// # Thread Safety
///
/// Multiple concurrent `search()` calls on the same `PipelinedSearcher` are safe.
/// Each search operates on its own `PipelinedReader` and `PQScratch` (pooled for
/// amortized allocation). Shared state (`PQData`, `GraphHeader`) is immutable.
#[deprecated(note = "Use DiskIndexSearcher::search_pipelined() instead for unified pipelined search")]
pub struct PipelinedSearcher<Data: GraphDataType<VectorIdType = u32>> {
    #[allow(dead_code)]
    graph_header: GraphHeader,
    distance_comparer: <Data::VectorDataType as VectorRepr>::Distance,
    pq_data: Arc<PQData>,
    metric: Metric,
    relaxed_monotonicity_l: Option<usize>,
    disk_index_path: String,
    reader_config: PipelinedReaderConfig,
    /// Pool of reusable reader + PQ scratch instances.
    scratch_pool: Arc<ObjectPool<PipelinedSearchScratch>>,

    // Precomputed values derived from graph_header / pq_data, cached to avoid
    // re-derivation on every search() call.
    block_size: usize,
    #[allow(dead_code)]
    num_sectors_per_node: usize,
    slot_size: usize,
    fp_vector_len: u64,
    dims: usize,
    node_len: u64,
    num_nodes_per_sector: u64,
    medoid: u32,
    graph_degree: usize,
    num_pq_chunks: usize,
    num_pq_centers: usize,
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
    /// * `beam_width` - Default beam width used for pool sizing.
    /// * `relaxed_monotonicity_l` - Optional early termination parameter.
    /// * `disk_index_path` - Path to the disk index file for creating readers.
    pub fn new(
        graph_header: GraphHeader,
        pq_data: Arc<PQData>,
        metric: Metric,
        beam_width: usize,
        relaxed_monotonicity_l: Option<usize>,
        disk_index_path: String,
        config: PipelinedReaderConfig,
    ) -> ANNResult<Self> {
        let metadata = graph_header.metadata();
        let dims = metadata.dims;
        let node_len = metadata.node_len;
        let num_nodes_per_sector = metadata.num_nodes_per_block;
        let fp_vector_len =
            (dims * std::mem::size_of::<Data::VectorDataType>()) as u64;
        let medoid = metadata.medoid as u32;
        let distance_comparer = Data::VectorDataType::distance(metric, Some(dims));

        let block_size = graph_header.effective_block_size();
        let num_sectors_per_node = graph_header.num_sectors_per_node();
        let slot_size = num_sectors_per_node * block_size;

        let max_slots =
            (beam_width * 2).clamp(16, super::pipelined_reader::MAX_IO_CONCURRENCY);

        let graph_degree = graph_header.max_degree::<Data::VectorDataType>()?;
        let num_pq_chunks = pq_data.get_num_chunks();
        let num_pq_centers = pq_data.get_num_centers();

        let scratch_args = PipelinedScratchArgs {
            disk_index_path: &disk_index_path,
            max_slots,
            slot_size,
            alignment: block_size,
            graph_degree,
            dims,
            num_pq_chunks,
            num_pq_centers,
            reader_config: config.clone(),
        };
        let scratch_pool = Arc::new(ObjectPool::try_new(&scratch_args, 0, None)?);

        Ok(Self {
            graph_header,
            distance_comparer,
            pq_data,
            metric,
            relaxed_monotonicity_l,
            disk_index_path,
            reader_config: config,
            scratch_pool,
            block_size,
            num_sectors_per_node,
            slot_size,
            fp_vector_len,
            dims,
            node_len,
            num_nodes_per_sector,
            medoid,
            graph_degree,
            num_pq_chunks,
            num_pq_centers,
        })
    }

    /// Perform a pipelined search on the disk index.
    ///
    /// # Arguments
    /// * `query` - The query vector.
    /// * `return_list_size` - Number of results to return (k).
    /// * `search_list_size` - Size of the candidate pool (L).
    /// * `beam_width` - Maximum beam width for pipelined IO.
    /// * `vector_filter` - Optional predicate; only vertices passing the filter
    ///   are included in the result set. Graph traversal is unaffected.
    pub fn search(
        &self,
        query: &[Data::VectorDataType],
        return_list_size: u32,
        search_list_size: u32,
        beam_width: usize,
        vector_filter: Option<&(dyn Fn(&u32) -> bool + Send + Sync)>,
    ) -> ANNResult<SearchResult<Data::AssociatedDataType>> {
        let max_slots = (beam_width * 2).clamp(16, super::pipelined_reader::MAX_IO_CONCURRENCY);

        let args = PipelinedScratchArgs {
            disk_index_path: &self.disk_index_path,
            max_slots,
            slot_size: self.slot_size,
            alignment: self.block_size,
            graph_degree: self.graph_degree,
            dims: self.dims,
            num_pq_chunks: self.num_pq_chunks,
            num_pq_centers: self.num_pq_centers,
            reader_config: self.reader_config.clone(),
        };
        let mut scratch = PoolOption::try_pooled(&self.scratch_pool, &args)?;
        let PipelinedSearchScratch {
            ref mut reader,
            ref mut pq_scratch,
        } = *scratch;

        let trace_enabled = std::env::var("DISKANN_TRACE").map_or(false, |v| v == "1");
        let mut trace = if trace_enabled {
            Some(SearchTrace::new())
        } else {
            None
        };

        let result: PipeSearchResult = pipe_search::<Data::VectorDataType>(
            reader,
            &self.pq_data,
            &self.distance_comparer,
            query,
            return_list_size as usize,
            search_list_size as usize,
            beam_width,
            self.medoid,
            self.dims,
            self.node_len,
            self.num_nodes_per_sector,
            self.block_size,
            self.fp_vector_len,
            pq_scratch,
            self.relaxed_monotonicity_l,
            self.metric,
            vector_filter,
            trace.as_mut(),
        )?;

        if let Some(t) = &trace {
            t.print_profile_summary();
        }

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

#[cfg(test)]
#[cfg(target_os = "linux")]
mod tests {
    use super::*;
    use std::sync::Arc;

    use diskann_providers::storage::{get_disk_index_file, VirtualStorageProvider};
    use diskann_providers::test_utils::graph_data_type_utils::GraphDataF32VectorUnitData;
    use diskann_utils::test_data_root;
    use diskann_vector::distance::Metric;
    use rayon::prelude::*;

    use crate::data_model::CachingStrategy;
    use crate::search::provider::disk_vertex_provider_factory::DiskVertexProviderFactory;
    use crate::search::traits::vertex_provider_factory::VertexProviderFactory;
    use crate::storage::disk_index_reader::DiskIndexReader;
    use crate::utils::VirtualAlignedReaderFactory;

    use super::PipelinedReaderConfig;

    const TEST_INDEX_PREFIX: &str =
        "/disk_index_search/disk_index_sift_learn_R4_L50_A1.2_truth_search";
    const TEST_PQ_PIVOT: &str =
        "/disk_index_search/disk_index_sift_learn_R4_L50_A1.2_truth_search_pq_pivots.bin";
    const TEST_PQ_COMPRESSED: &str =
        "/disk_index_search/disk_index_sift_learn_R4_L50_A1.2_truth_search_pq_compressed.bin";
    const TEST_QUERY: &str = "/disk_index_search/disk_index_sample_query_10pts.fbin";

    fn create_test_searcher() -> PipelinedSearcher<GraphDataF32VectorUnitData> {
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));

        let disk_index_reader = DiskIndexReader::<f32>::new(
            TEST_PQ_PIVOT.to_string(),
            TEST_PQ_COMPRESSED.to_string(),
            storage_provider.as_ref(),
        )
        .unwrap();
        let pq_data = disk_index_reader.get_pq_data();

        let aligned_reader_factory = VirtualAlignedReaderFactory::new(
            get_disk_index_file(TEST_INDEX_PREFIX),
            Arc::clone(&storage_provider),
        );
        let vertex_provider_factory =
            DiskVertexProviderFactory::<GraphDataF32VectorUnitData, _>::new(
                aligned_reader_factory,
                CachingStrategy::None,
            )
            .unwrap();
        let graph_header = vertex_provider_factory.get_header().unwrap();

        let real_index_path = test_data_root().join(
            "disk_index_search/disk_index_sift_learn_R4_L50_A1.2_truth_search_disk.index",
        );

        PipelinedSearcher::<GraphDataF32VectorUnitData>::new(
            graph_header,
            pq_data,
            Metric::L2,
            4,
            None,
            real_index_path.to_str().unwrap().to_string(),
            PipelinedReaderConfig::default(),
        )
        .unwrap()
    }

    fn load_test_query() -> Vec<f32> {
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));
        let (query_vector, _npts, _dim) =
            diskann_providers::utils::file_util::load_bin::<f32, _>(
                storage_provider.as_ref(),
                TEST_QUERY,
                0,
            )
            .unwrap();
        query_vector[0..128].to_vec()
    }

    #[test]
    fn test_pool_reuse_sequential_searches() {
        let searcher = create_test_searcher();
        let query = load_test_query();

        let r1 = searcher.search(&query, 10, 40, 4, None).unwrap();
        let r2 = searcher.search(&query, 10, 40, 4, None).unwrap();

        assert!(!r1.results.is_empty());
        assert!(!r2.results.is_empty());
        // Same query must return same number of results.
        assert_eq!(r1.results.len(), r2.results.len());
        // All distances must be non-negative.
        for item in r1.results.iter().chain(r2.results.iter()) {
            assert!(item.distance >= 0.0);
        }
    }

    #[test]
    fn test_pool_concurrent_searches() {
        let searcher = Arc::new(create_test_searcher());
        let query = load_test_query();

        let results: Vec<_> = (0..4)
            .into_par_iter()
            .map(|_| searcher.search(&query, 10, 40, 4, None).unwrap())
            .collect();

        for r in &results {
            assert!(!r.results.is_empty());
            for item in &r.results {
                assert!(item.distance >= 0.0);
            }
        }
    }
}
