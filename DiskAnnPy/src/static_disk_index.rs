/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use diskann::{utils::IntoUsize, ANNResult};
use diskann_disk::{
    data_model::CachingStrategy,
    search::provider::{
        disk_provider::DiskIndexSearcher, disk_vertex_provider_factory::DiskVertexProviderFactory,
    },
    storage::disk_index_reader::DiskIndexReader,
    utils::{AlignedFileReaderFactory, QueryStatistics},
};
use diskann_providers::{
    model::graph::traits::GraphDataType,
    storage::{
        get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file, FileStorageProvider,
    },
    utils::{create_thread_pool, ParallelIteratorInPool},
};
use diskann_vector::distance::Metric;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::{
    utils::{
        get_num_threads, load_aligned_from_vector, pyarray2_to_vec_row_decomp,
        search_result::{BatchSearchResultWithStats, SearchStats},
        ANNErrorPy, GraphDataF32Vector, GraphDataInt8Vector, GraphDataU8Vector, SearchResult,
    },
    MetricPy,
};

pub struct StaticDiskIndex<Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    pub index_search_engine:
        DiskIndexSearcher<Data, DiskVertexProviderFactory<Data, AlignedFileReaderFactory>>,
    beam_width: u32,
}

impl<Data> StaticDiskIndex<Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    pub fn new(
        dist_fn: Metric,
        index_path_prefix: &str,
        beam_width: u32,
        search_io_limit: u32,
        num_threads: u32,
        num_nodes_to_cache: usize,
    ) -> ANNResult<Self> {
        let storage_provider = &FileStorageProvider;
        let num_threads = get_num_threads(Some(num_threads));

        let index_reader = DiskIndexReader::<<Data as GraphDataType>::VectorDataType>::new(
            get_pq_pivot_file(index_path_prefix),
            get_compressed_pq_file(index_path_prefix),
            storage_provider,
        )?;

        let caching_strategy = if num_nodes_to_cache > 0 {
            CachingStrategy::StaticCacheWithBfsNodes(num_nodes_to_cache)
        } else {
            CachingStrategy::None
        };
        let vertex_provider_factory = DiskVertexProviderFactory::new(
            AlignedFileReaderFactory::new(get_disk_index_file(index_path_prefix).to_string()),
            caching_strategy,
        )?;

        let index_search_engine = DiskIndexSearcher::new(
            num_threads.into_usize(),
            search_io_limit.into_usize(),
            &index_reader,
            vertex_provider_factory,
            dist_fn,
            None,
        )?;

        Ok(Self {
            index_search_engine,
            beam_width,
        })
    }

    pub fn search(
        &self,
        query: Vec<Data::VectorDataType>,
        recall_at: u32,
        l_value: u32,
    ) -> ANNResult<SearchResult> {
        let (query_aligned, _, _, _) = load_aligned_from_vector(vec![query])?;
        let mut query_stats = QueryStatistics::default();
        self.search_internal(&query_aligned, recall_at, l_value, &mut query_stats)
    }

    fn search_internal(
        &self,
        query: &[Data::VectorDataType],
        recall_at: u32,
        l_value: u32,
        query_stats: &mut QueryStatistics,
    ) -> ANNResult<SearchResult> {
        let result = self.index_search_engine.search(
            query,
            recall_at,
            l_value,
            Some(self.beam_width as usize),
            None,
            false,
        )?;

        *query_stats = result.stats.query_statistics;

        let search_result = result.results.iter().take(recall_at as usize).fold(
            SearchResult {
                ids: Vec::with_capacity(recall_at as usize),
                distances: Vec::with_capacity(recall_at as usize),
            },
            |mut acc, item| {
                acc.ids.push(item.vertex_id);
                acc.distances.push(item.distance);
                acc
            },
        );

        Ok(search_result)
    }

    pub fn batch_search(
        &self,
        queries: Vec<Vec<Data::VectorDataType>>,
        recall_at: u32,
        l_value: u32,
        num_threads: u32,
    ) -> ANNResult<BatchSearchResultWithStats> {
        let (flat_queries_aligned, _query_num, _, aligned_dim) = load_aligned_from_vector(queries)?;

        let query_num = flat_queries_aligned.len() / aligned_dim;
        let mut query_result_ids: Vec<Vec<u32>> = vec![vec![]; query_num];
        let mut distance_results: Vec<Vec<f32>> = vec![vec![]; query_num];
        let mut statistics: Vec<QueryStatistics> = vec![QueryStatistics::default(); query_num];

        let pool = create_thread_pool(num_threads.into_usize())?;

        let zipped = flat_queries_aligned
            .par_chunks(aligned_dim)
            .zip(query_result_ids.par_iter_mut())
            .zip(distance_results.par_iter_mut())
            .zip(statistics.par_iter_mut());

        zipped.for_each_in_pool(
            &pool,
            |(((query, query_result_ids), distance_results), query_stats)| {
                let search_result = self
                    .search_internal(query, recall_at, l_value, query_stats)
                    .unwrap();
                *query_result_ids = search_result.ids;
                *distance_results = search_result.distances;
            },
        );

        Ok(BatchSearchResultWithStats {
            ids: query_result_ids,
            distances: distance_results,
            search_stats: SearchStats::stats_to_dict(&statistics),
        })
    }
}

macro_rules! impl_static_disk_index {
    ($index_type:ident, $data_type:ty, $pyarray_type:ty) => {
        #[pyclass]
        pub struct $index_type {
            pub index_search_engine: StaticDiskIndex<$data_type>,
        }

        #[pymethods]
        impl $index_type {
            #[new]
            pub fn new(
                dist_fn: MetricPy,
                index_path_prefix: String,
                beam_width: u32,
                search_io_limit: u32,
                num_threads: u32,
                num_nodes_to_cache: usize,
            ) -> Result<Self, ANNErrorPy> {
                let searcher = StaticDiskIndex::<$data_type>::new(
                    dist_fn.into(),
                    &index_path_prefix,
                    beam_width,
                    search_io_limit,
                    num_threads,
                    num_nodes_to_cache,
                )?;
                Ok($index_type {
                    index_search_engine: searcher,
                })
            }

            pub fn search(
                &self,
                query: &Bound<PyArray1<$pyarray_type>>,
                recall_at: u32,
                l_value: u32,
            ) -> Result<SearchResult, ANNErrorPy> {
                let query_as_vec = query.readonly().as_array().to_vec();
                Ok(self
                    .index_search_engine
                    .search(query_as_vec, recall_at, l_value)?)
            }

            pub fn batch_search(
                &self,
                queries: &Bound<PyArray2<$pyarray_type>>,
                recall_at: u32,
                l_value: u32,
                num_threads: u32,
            ) -> Result<BatchSearchResultWithStats, ANNErrorPy> {
                let queries_as_vec = pyarray2_to_vec_row_decomp(queries);
                Ok(self.index_search_engine.batch_search(
                    queries_as_vec,
                    recall_at,
                    l_value,
                    num_threads,
                )?)
            }
        }
    };
}

// Use the macro to generate the implementations
impl_static_disk_index!(StaticDiskIndexF32, GraphDataF32Vector, f32);
impl_static_disk_index!(StaticDiskIndexInt8, GraphDataInt8Vector, i8);
impl_static_disk_index!(StaticDiskIndexU8, GraphDataU8Vector, u8);
