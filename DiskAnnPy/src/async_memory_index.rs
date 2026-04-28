/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
*/

use std::sync::{Arc, Mutex};

use diskann::{
    error::IntoANNResult,
    graph::{
        glue::PruneStrategy, search, search_output_buffer, AdjacencyList, InplaceDeleteMethod,
    },
    neighbor::Neighbor,
    provider::{DefaultContext, Delete, ElementStatus, NeighborAccessor},
    utils::{vecid_from_usize, IntoUsize, VectorIdBoxSlice, VectorRepr},
    ANNResult,
};
use diskann_providers::{
    index::diskann_async::{MemoryIndex, PQMemoryIndex},
    model::graph::provider::async_::{
        common::{FullPrecision, Hybrid},
        TableDeleteProviderAsync,
    },
    utils::{create_thread_pool, ParallelIteratorInPool},
};
use diskann_vector::distance::Metric;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyAny, PyList, PyTuple},
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator};
use tokio::{runtime::Runtime, task::JoinSet};

use crate::{
    build_async_memory_index::{build_empty_index, load_index},
    utils::{
        common_error, pyarray2_to_vec_row_decomp, search_result::SearchStats, ANNErrorPy,
        BatchRangeSearchResultWithStats, BatchSearchResultWithStats, MetricPy, SearchResult,
        VectorIdBoxSliceWrapper,
    },
};

// Implement FromPyObject for the generic struct
impl<'source, T> FromPyObject<'source> for VectorIdBoxSliceWrapper<T>
where
    T: FromPyObject<'source>,
{
    fn extract_bound(ob: &Bound<'source, PyAny>) -> PyResult<Self> {
        let tuple = ob.downcast::<PyTuple>()?;
        if tuple.len() != 2 {
            return Err(PyTypeError::new_err("Expected a tuple of length 2"));
        }
        let id: u32 = tuple.get_item(0)?.extract()?;
        let tuple_item1 = tuple.get_item(1)?;
        let value_list = tuple_item1.downcast::<PyList>()?;
        let value: Vec<T> = value_list.extract()?;
        // let value: T = tuple.get_item(1)?.extract()?;
        Ok(VectorIdBoxSliceWrapper { id, value })
    }
}

// Implement From for the generic struct to convert it to the original type
impl<T> From<VectorIdBoxSliceWrapper<T>> for VectorIdBoxSlice<u32, T> {
    fn from(wrapper: VectorIdBoxSliceWrapper<T>) -> Self {
        VectorIdBoxSlice {
            vector_id: wrapper.id,
            vector: wrapper.value.into_boxed_slice(),
        }
    }
}

pub enum AsyncIndexType<T>
where
    T: VectorRepr,
{
    NoPQIndex(MemoryIndex<T, TableDeleteProviderAsync>),
    PQIndex(PQMemoryIndex<T, TableDeleteProviderAsync>),
}

pub struct AsyncMemoryIndex<T>
where
    T: VectorRepr,
{
    index: AsyncIndexType<T>,
    runtime: Arc<Runtime>,
}

#[pyclass]
pub struct AsyncMemoryIndexF32 {
    pub index: AsyncMemoryIndex<f32>,
}

#[pyclass]
pub struct AsyncMemoryIndexU8 {
    pub index: AsyncMemoryIndex<u8>,
}

#[pyclass]
pub struct AsyncMemoryIndexInt8 {
    pub index: AsyncMemoryIndex<i8>,
}

impl<T> AsyncMemoryIndex<T>
where
    T: VectorRepr,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        metric: Metric,
        index_path: String,
        r: u32,
        l: u32,
        alpha: f32,
        num_threads: u32,
        build_pq_bytes: usize,
        graph_slack_factor: f32,
        load_from_file: bool,
        max_points: usize,
        dim: usize,
    ) -> Result<Self, ANNErrorPy> {
        let runtime = Arc::new(Runtime::new().unwrap());
        let searcher = if load_from_file {
            load_index::<T>(
                metric.into(),
                index_path,
                r,
                l,
                num_threads,
                build_pq_bytes,
                graph_slack_factor,
            )?
        } else {
            build_empty_index::<T>(
                metric.into(),
                r,
                l,
                alpha,
                graph_slack_factor,
                max_points,
                dim,
            )
            .map(|searcher| AsyncIndexType::NoPQIndex(searcher))?
        };

        Ok(AsyncMemoryIndex {
            index: searcher,
            runtime,
        })
    }

    pub fn insert(
        &self,
        vector_id: u32,
        vector: &[T],
        use_full_precision_to_search: bool,
    ) -> ANNResult<()> {
        let _ = self.runtime.block_on(async {
            match &self.index {
                AsyncIndexType::NoPQIndex(index) => {
                    index
                        .insert(FullPrecision, &DefaultContext, &vector_id, vector)
                        .await
                }
                AsyncIndexType::PQIndex(index) => {
                    if use_full_precision_to_search {
                        index
                            .insert(FullPrecision, &DefaultContext, &vector_id, vector)
                            .await
                    } else {
                        index
                            .insert(Hybrid::new(None), &DefaultContext, &vector_id, vector)
                            .await
                    }
                }
            }
        });
        Ok(())
    }

    pub fn mark_deleted(&self, vector_ids: Vec<u32>) -> ANNResult<()> {
        let _: ANNResult<()> = self.runtime.block_on(async {
            match &self.index {
                AsyncIndexType::NoPQIndex(index) => {
                    for id in vector_ids {
                        index.data_provider.delete(&DefaultContext, &id).await?;
                    }
                }
                AsyncIndexType::PQIndex(index) => {
                    for id in vector_ids {
                        index.data_provider.delete(&DefaultContext, &id).await?;
                    }
                }
            }
            Ok(())
        });

        Ok(())
    }

    async fn clear_delete_set(&self) -> ANNResult<()> {
        match &self.index {
            AsyncIndexType::NoPQIndex(index) => {
                // Get ids of deleted vectors by checking the delete table
                // and call "release" on each deleted id
                for id in 0..index.data_provider.total_points() {
                    let is_deleted = index
                        .data_provider
                        .status_by_internal_id(&DefaultContext, vecid_from_usize(id)?)
                        .await
                        .unwrap_or(ElementStatus::Deleted);
                    if is_deleted == ElementStatus::Deleted {
                        index
                            .data_provider
                            .release(&DefaultContext, vecid_from_usize(id)?)
                            .await?;
                    }
                }
            }
            AsyncIndexType::PQIndex(index) => {
                // Get ids of deleted vectors by checking the delete table
                // and call "release" on each deleted id
                for id in 0..index.data_provider.total_points() {
                    let is_deleted = index
                        .data_provider
                        .status_by_internal_id(&DefaultContext, vecid_from_usize(id)?)
                        .await
                        .unwrap_or(ElementStatus::Deleted);
                    if is_deleted == ElementStatus::Deleted {
                        index
                            .data_provider
                            .release(&DefaultContext, vecid_from_usize(id)?)
                            .await?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn consolidate_deletes(&self, num_tasks: usize) -> ANNResult<()> {
        let _: ANNResult<()> = self.runtime.block_on(async move {
            let mut tasks = JoinSet::new();

            match &self.index {
                AsyncIndexType::NoPQIndex(index) => {
                    let consolidate_iter = Arc::new(Mutex::new(0..index.provider().total_points()));

                    for _ in 0..num_tasks {
                        let index_clone = index.clone();
                        let iterator_clone = consolidate_iter.clone();
                        tasks.spawn(async move {
                            loop {
                                let result = {
                                    let mut guard = iterator_clone.lock().map_err(|_| {
                                        common_error("Poisoned mutex during construction")
                                    })?;
                                    guard.next()
                                };

                                match result {
                                    Some(id) => {
                                        index_clone
                                            .consolidate_vector(
                                                &FullPrecision,
                                                &DefaultContext,
                                                vecid_from_usize(id)?,
                                            )
                                            .await?;
                                    }
                                    None => break,
                                }
                            }
                            ANNResult::Ok(())
                        });
                    }
                }
                AsyncIndexType::PQIndex(index) => {
                    let consolidate_iter = Arc::new(Mutex::new(0..index.provider().total_points()));
                    for _ in 0..num_tasks {
                        let index_clone = index.clone();
                        let iterator_clone = consolidate_iter.clone();
                        tasks.spawn(async move {
                            loop {
                                let result = {
                                    let mut guard = iterator_clone.lock().map_err(|_| {
                                        common_error("Poisoned mutex during construction")
                                    })?;
                                    guard.next()
                                };

                                match result {
                                    Some(id) => {
                                        index_clone
                                            .consolidate_vector(
                                                &Hybrid::new(None),
                                                &DefaultContext,
                                                vecid_from_usize(id)?,
                                            )
                                            .await?;
                                    }
                                    None => break,
                                }
                            }
                            ANNResult::Ok(())
                        });
                    }
                }
            }

            // Wait for all tasks to complete.
            while let Some(res) = tasks.join_next().await {
                res.map_err(|_| common_error("A spawned task failed"))??;
            }

            // Clear the delete set
            self.clear_delete_set().await?;

            Ok(())
        });

        Ok(())
    }

    // an alternate version of consolidate_deletes that simply drops links to
    // deleted nodes
    pub fn consolidate_simple(&self, num_tasks: usize) -> ANNResult<()> {
        let _: ANNResult<()> = self.runtime.block_on(async move {
            let mut tasks = JoinSet::new();

            match &self.index {
                AsyncIndexType::NoPQIndex(index) => {
                    let consolidate_iter = Arc::new(Mutex::new(0..index.provider().total_points()));
                    for _ in 0..num_tasks {
                        let index_clone = index.clone();
                        let iterator_clone = consolidate_iter.clone();
                        tasks.spawn(async move {
                            loop {
                                let neighbor_accessor = &mut index_clone.provider().neighbors();
                                let result = {
                                    let mut guard = iterator_clone.lock().map_err(|_| {
                                        common_error("Poisoned mutex during construction")
                                    })?;
                                    guard.next()
                                };

                                match result {
                                    Some(id) => {
                                        index_clone
                                            .drop_deleted_neighbors(
                                                &DefaultContext,
                                                neighbor_accessor,
                                                vecid_from_usize(id)?,
                                                false,
                                            )
                                            .await?;
                                    }
                                    None => break,
                                }
                            }
                            ANNResult::Ok(())
                        });
                    }
                }
                AsyncIndexType::PQIndex(index) => {
                    let consolidate_iter = Arc::new(Mutex::new(0..index.provider().total_points()));
                    for _ in 0..num_tasks {
                        let index_clone = index.clone();
                        let iterator_clone = consolidate_iter.clone();
                        tasks.spawn(async move {
                            let neighbor_accessor = &mut index_clone.provider().neighbors();
                            loop {
                                let result = {
                                    let mut guard = iterator_clone.lock().map_err(|_| {
                                        common_error("Poisoned mutex during construction")
                                    })?;
                                    guard.next()
                                };

                                match result {
                                    Some(id) => {
                                        index_clone
                                            .drop_deleted_neighbors(
                                                &DefaultContext,
                                                neighbor_accessor,
                                                vecid_from_usize(id)?,
                                                false,
                                            )
                                            .await?;
                                    }
                                    None => break,
                                }
                            }
                            ANNResult::Ok(())
                        });
                    }
                }
            }

            // Wait for all tasks to complete.
            while let Some(res) = tasks.join_next().await {
                res.map_err(|_| common_error("A spawned task failed"))??;
            }

            // Clear the delete set
            self.clear_delete_set().await?;

            Ok(())
        });

        Ok(())
    }

    pub fn multi_inplace_delete(
        &self,
        vector_ids: Arc<[u32]>,
        use_full_precision_to_search: bool,
        k_value: usize,
        l_value: usize,
        num_to_replace: usize,
        delete_method: usize,
    ) -> ANNResult<()> {
        let delete_method = if delete_method == 0 {
            InplaceDeleteMethod::TwoHopAndOneHop
        } else {
            InplaceDeleteMethod::VisitedAndTopK { k_value, l_value }
        };
        let _ = self.runtime.block_on(async {
            match &self.index {
                AsyncIndexType::NoPQIndex(index) => {
                    index
                        .multi_inplace_delete(
                            FullPrecision,
                            &DefaultContext,
                            vector_ids,
                            num_to_replace,
                            delete_method,
                        )
                        .await
                }
                AsyncIndexType::PQIndex(index) => {
                    if use_full_precision_to_search {
                        index
                            .multi_inplace_delete(
                                FullPrecision,
                                &DefaultContext,
                                vector_ids,
                                num_to_replace,
                                delete_method,
                            )
                            .await
                    } else {
                        index
                            .multi_inplace_delete(
                                Hybrid::new(None),
                                &DefaultContext,
                                vector_ids,
                                num_to_replace,
                                delete_method,
                            )
                            .await
                    }
                }
            }
        });
        Ok(())
    }

    // we use a numerical value to indicate the delete_method used
    // 1 for VisitedAndTopK, 0 for TwoHopAndOneHop, iwth VisitedAndTopK
    // the default
    #[allow(clippy::too_many_arguments)]
    pub fn batch_inplace_delete(
        &self,
        vector_ids: Vec<u32>,
        use_full_precision_to_search: bool,
        num_tasks: usize,
        k_value: usize,
        l_value: usize,
        num_to_replace: usize,
        delete_method: usize,
    ) -> ANNResult<()> {
        let delete_method = if delete_method == 0 {
            InplaceDeleteMethod::TwoHopAndOneHop
        } else if delete_method == 1 {
            InplaceDeleteMethod::VisitedAndTopK { k_value, l_value }
        } else {
            InplaceDeleteMethod::OneHop
        };
        let _: ANNResult<()> = self.runtime.block_on(async move {
            let mut tasks = JoinSet::new();

            match &self.index {
                AsyncIndexType::NoPQIndex(index) => {
                    let delete_iter = Arc::new(Mutex::new(vector_ids.into_iter()));
                    for _ in 0..num_tasks {
                        let index_clone = index.clone();
                        let iterator_clone = delete_iter.clone();
                        tasks.spawn(async move {
                            loop {
                                let result = {
                                    let mut guard = iterator_clone.lock().map_err(|_| {
                                        common_error("Poisoned mutex during construction")
                                    })?;
                                    guard.next()
                                };

                                match result {
                                    Some(id) => {
                                        index_clone
                                            .inplace_delete(
                                                FullPrecision,
                                                &DefaultContext,
                                                &id,
                                                num_to_replace,
                                                delete_method,
                                            )
                                            .await?;
                                    }
                                    None => break,
                                }
                            }
                            ANNResult::Ok(())
                        });
                    }
                }

                AsyncIndexType::PQIndex(index) => {
                    let delete_iter = Arc::new(Mutex::new(vector_ids.into_iter()));
                    for _ in 0..num_tasks {
                        let index_clone = index.clone();
                        let iterator_clone = delete_iter.clone();
                        tasks.spawn(async move {
                            loop {
                                let result = {
                                    let mut guard = iterator_clone.lock().map_err(|_| {
                                        common_error("Poisoned mutex during construction")
                                    })?;
                                    guard.next()
                                };

                                match result {
                                    Some(id) => {
                                        if use_full_precision_to_search {
                                            index_clone
                                                .inplace_delete(
                                                    FullPrecision,
                                                    &DefaultContext,
                                                    &id,
                                                    num_to_replace,
                                                    delete_method,
                                                )
                                                .await?;
                                        } else {
                                            index_clone
                                                .inplace_delete(
                                                    Hybrid::new(None),
                                                    &DefaultContext,
                                                    &id,
                                                    num_to_replace,
                                                    delete_method,
                                                )
                                                .await?;
                                        }
                                    }
                                    None => break,
                                }
                            }
                            ANNResult::Ok(())
                        });
                    }
                }
            }

            // Wait for all tasks to complete.
            while let Some(res) = tasks.join_next().await {
                res.map_err(|_| common_error("A spawned task failed"))??;
            }

            Ok(())
        });

        Ok(())
    }

    pub fn batch_insert(
        &self,
        vector_ids: Vec<u32>,
        vectors: Vec<Vec<T>>,
        use_full_precision_to_search: bool,
        num_tasks: usize,
    ) -> ANNResult<()> {
        let _: ANNResult<()> = self.runtime.block_on(async move {
            let mut tasks = JoinSet::new();
            let vector_and_id_iter = vector_ids.into_iter().zip(vectors);
            let iterator = Arc::new(Mutex::new(vector_and_id_iter));
            for _ in 0..num_tasks {
                match &self.index {
                    AsyncIndexType::NoPQIndex(index) => {
                        let index_clone = index.clone();
                        let iterator_clone = iterator.clone();

                        tasks.spawn(async move {
                            loop {
                                let result = {
                                    let mut guard = iterator_clone.lock().map_err(|_| {
                                        common_error("Poisoned mutex during construction")
                                    })?;
                                    guard.next()
                                };

                                match result {
                                    Some((vector_id, vector)) => {
                                        index_clone
                                            .insert(
                                                FullPrecision,
                                                &DefaultContext,
                                                &vector_id,
                                                &vector,
                                            )
                                            .await?;
                                    }
                                    None => break,
                                }
                            }
                            ANNResult::Ok(())
                        });
                    }
                    AsyncIndexType::PQIndex(index) => {
                        let index_clone = index.clone();
                        let iterator_clone = iterator.clone();

                        tasks.spawn(async move {
                            loop {
                                let result = {
                                    let mut guard = iterator_clone.lock().map_err(|_| {
                                        common_error("Poisoned mutex during construction")
                                    })?;
                                    guard.next()
                                };

                                match result {
                                    Some((vector_id, vector)) => {
                                        if use_full_precision_to_search {
                                            index_clone
                                                .insert(
                                                    FullPrecision,
                                                    &DefaultContext,
                                                    &vector_id,
                                                    &vector,
                                                )
                                                .await?;
                                        } else {
                                            index_clone
                                                .insert(
                                                    Hybrid::new(None),
                                                    &DefaultContext,
                                                    &vector_id,
                                                    &vector,
                                                )
                                                .await?;
                                        }
                                    }
                                    None => break,
                                }
                            }
                            ANNResult::Ok(())
                        });
                    }
                };
            }
            // Wait for all tasks to complete.
            while let Some(res) = tasks.join_next().await {
                res.map_err(|_| common_error("A spawned task failed"))??;
            }
            Ok(())
        });

        Ok(())
    }

    pub fn multi_insert(
        &self,
        vector_id_pairs: Box<[VectorIdBoxSlice<u32, T>]>,
        use_full_precision_to_search: bool,
    ) -> ANNResult<()> {
        let _ = self.runtime.block_on(async {
            match &self.index {
                AsyncIndexType::NoPQIndex(index) => {
                    index
                        .multi_insert(FullPrecision, &DefaultContext, vector_id_pairs)
                        .await
                }
                AsyncIndexType::PQIndex(index) => {
                    if use_full_precision_to_search {
                        index
                            .multi_insert(FullPrecision, &DefaultContext, vector_id_pairs)
                            .await
                    } else {
                        index
                            .multi_insert(Hybrid::new(None), &DefaultContext, vector_id_pairs)
                            .await
                    }
                }
            }
        });
        Ok(())
    }

    fn range_search(
        &self,
        query: &[T],
        starting_l_value: usize,
        radius: f32,
    ) -> ANNResult<(Vec<u32>, Vec<f32>, u32, bool)> {
        self.runtime.block_on(async {
            match &self.index {
                AsyncIndexType::NoPQIndex(index) => {
                    let mut results: Vec<Neighbor<u32>> = Vec::new();
                    let stats = index
                        .search(
                            search::Range::new(starting_l_value, radius)?,
                            &FullPrecision,
                            &DefaultContext,
                            query,
                            &mut results,
                        )
                        .await?;

                    let ids: Vec<u32> = results.iter().map(|n| n.id).collect();
                    let distances: Vec<f32> = results.iter().map(|n| n.distance).collect();

                    ANNResult::Ok((ids, distances, stats.cmps, stats.range_search_second_round))
                }
                AsyncIndexType::PQIndex(index) => {
                    let mut results: Vec<Neighbor<u32>> = Vec::new();
                    let stats = index
                        .search(
                            search::Range::new(starting_l_value, radius)?,
                            &FullPrecision,
                            &DefaultContext,
                            query,
                            &mut results,
                        )
                        .await?;

                    let ids: Vec<u32> = results.iter().map(|n| n.id).collect();
                    let distances: Vec<f32> = results.iter().map(|n| n.distance).collect();

                    ANNResult::Ok((ids, distances, stats.cmps, stats.range_search_second_round))
                }
            }
        })
    }

    pub fn batch_range_search(
        &self,
        queries: Vec<Vec<T>>,
        num_threads: u32,
        starting_l_value: u32,
        radius: f32,
    ) -> ANNResult<BatchRangeSearchResultWithStats> {
        let mut query_result_ids: Vec<Vec<u32>> = vec![vec![]; queries.len()];
        let mut distance_results: Vec<Vec<f32>> = vec![vec![]; queries.len()];
        let mut cmps: Vec<u32> = vec![0; queries.len()];
        let mut second_round: Vec<bool> = vec![false; queries.len()];

        let zipped = queries
            .par_iter()
            .zip(query_result_ids.par_iter_mut())
            .zip(distance_results.par_iter_mut())
            .zip(cmps.par_iter_mut())
            .zip(second_round.par_iter_mut());

        let pool = create_thread_pool(num_threads.into_usize())?;
        zipped.for_each_in_pool(
            &pool,
            |((((query, query_result_ids), distance_results), cmps), second_round)| {
                let (ids, dists, cmp, sr) = self
                    .range_search(query, starting_l_value as usize, radius)
                    .unwrap();
                *query_result_ids = ids;
                *distance_results = dists;
                *cmps = cmp;
                *second_round = sr;
            },
        );
        let lims = query_result_ids
            .iter()
            .map(|ids| ids.len())
            .collect::<Vec<_>>();
        let query_result_ids = query_result_ids.into_iter().flatten().collect::<Vec<_>>();
        let distance_results = distance_results.into_iter().flatten().collect::<Vec<_>>();
        let num_second_round = second_round.iter().filter(|&&x| x).count() as u32;
        println!("Number of second round searches: {}", num_second_round);
        Ok(BatchRangeSearchResultWithStats {
            lims,
            ids: query_result_ids,
            distances: distance_results,
            search_stats: SearchStats::stats_to_dict_inmem(&cmps),
        })
    }

    pub fn search(
        &self,
        query: &[T],
        k_value: usize,
        l_value: usize,
        use_full_precision_to_search: bool,
    ) -> ANNResult<(SearchResult, u32)> {
        let mut result_ids = vec![0; k_value];
        let mut result_dists = vec![0.0; k_value];
        let mut cmps = 0;

        let _ = self.runtime.block_on(async {
            match &self.index {
                AsyncIndexType::NoPQIndex(index) => {
                    let mut result_output_buffer =
                        search_output_buffer::IdDistance::new(&mut result_ids, &mut result_dists);
                    let stats = index
                        .search(
                            search::Knn::new_default(l_value, l_value)?,
                            &FullPrecision,
                            &DefaultContext,
                            query,
                            &mut result_output_buffer,
                        )
                        .await?;
                    cmps = stats.cmps;
                    ANNResult::Ok(())
                }
                AsyncIndexType::PQIndex(index) => {
                    let mut result_output_buffer =
                        search_output_buffer::IdDistance::new(&mut result_ids, &mut result_dists);
                    let stats = if use_full_precision_to_search {
                        index
                            .search(
                                search::Knn::new_default(l_value, l_value)?,
                                &FullPrecision,
                                &DefaultContext,
                                query,
                                &mut result_output_buffer,
                            )
                            .await?
                    } else {
                        index
                            .search(
                                search::Knn::new_default(l_value, l_value)?,
                                &Hybrid::new(None),
                                &DefaultContext,
                                query,
                                &mut result_output_buffer,
                            )
                            .await?
                    };
                    cmps = stats.cmps;
                    ANNResult::Ok(())
                }
            }
        });
        Ok((
            SearchResult {
                ids: result_ids,
                distances: result_dists,
            },
            cmps,
        ))
    }

    pub fn batch_search(
        &self,
        queries: Vec<Vec<T>>,
        num_threads: u32,
        k_value: u32,
        l_value: u32,
        use_full_precision_to_search: bool,
    ) -> ANNResult<BatchSearchResultWithStats> {
        // let (flat_queries_aligned, _query_num, _, aligned_dim) = load_aligned_from_vector(queries)?;
        let mut query_result_ids: Vec<Vec<u32>> = vec![vec![]; queries.len()];
        let mut distance_results: Vec<Vec<f32>> = vec![vec![]; queries.len()];
        let mut cmps: Vec<u32> = vec![0; queries.len()];

        let zipped = queries
            .par_iter()
            .zip(query_result_ids.par_iter_mut())
            .zip(distance_results.par_iter_mut())
            .zip(cmps.par_iter_mut());

        let pool = create_thread_pool(num_threads.into_usize())?;
        zipped.for_each_in_pool(
            &pool,
            |(((query, query_result_ids), distance_results), cmps)| {
                let (search_result, cmp) = self
                    .search(
                        query,
                        k_value as usize,
                        l_value as usize,
                        use_full_precision_to_search,
                    )
                    .unwrap();
                *query_result_ids = search_result.ids;
                *distance_results = search_result.distances;
                *cmps = cmp;
            },
        );
        Ok(BatchSearchResultWithStats {
            ids: query_result_ids,
            distances: distance_results,
            search_stats: SearchStats::stats_to_dict_inmem(&cmps),
        })
    }

    pub fn get_neighbors(&self, vector_id: u32) -> ANNResult<Vec<u32>> {
        self.runtime.block_on(async {
            match &self.index {
                AsyncIndexType::NoPQIndex(index) => {
                    let mut accessor = FullPrecision
                        .prune_accessor(&index.data_provider, &DefaultContext)
                        .into_ann_result()?;
                    let mut neighbors = AdjacencyList::new();
                    accessor.get_neighbors(vector_id, &mut neighbors).await?;
                    Ok(neighbors.to_vec())
                }
                AsyncIndexType::PQIndex(index) => {
                    let mut accessor = FullPrecision
                        .prune_accessor(&index.data_provider, &DefaultContext)
                        .into_ann_result()?;
                    let mut neighbors = AdjacencyList::new();
                    accessor.get_neighbors(vector_id, &mut neighbors).await?;
                    Ok(neighbors.to_vec())
                }
            }
        })
    }

    pub fn get_average_degree(&self) -> ANNResult<f32> {
        self.runtime.block_on(async {
            match &self.index {
                AsyncIndexType::NoPQIndex(index) => {
                    let mut accessor = index.provider().neighbors();
                    let degree_stats = index.get_degree_stats(&mut accessor).await?;
                    Ok(degree_stats.avg_degree)
                }
                AsyncIndexType::PQIndex(index) => {
                    let mut accessor = index.provider().neighbors();
                    let degree_stats = index.get_degree_stats(&mut accessor).await?;
                    Ok(degree_stats.avg_degree)
                }
            }
        })
    }
}

macro_rules! impl_async_memory_index {
    ($index_type:ident, $vector_type:ty, $py_vector_type:ty) => {
        #[pymethods]
        #[allow(clippy::too_many_arguments)]
        impl $index_type {
            #[new]
            pub fn new(
                metric: MetricPy,
                index_path: String,
                r: u32,
                l: u32,
                alpha: f32,
                num_threads: u32,
                build_pq_bytes: usize,
                graph_slack_factor: f32,
                load_from_file: bool,
                max_points: usize,
                dim: usize,
            ) -> Result<Self, ANNErrorPy> {
                let searcher = AsyncMemoryIndex::<$vector_type>::new(
                    metric.into(),
                    index_path,
                    r,
                    l,
                    alpha,
                    num_threads,
                    build_pq_bytes,
                    graph_slack_factor,
                    load_from_file,
                    max_points,
                    dim,
                )?;

                Ok($index_type { index: searcher })
            }

            pub fn insert(
                &self,
                vector_id: u32,
                vector: PyReadonlyArray1<$py_vector_type>,
                use_full_precision_to_search: bool,
            ) -> Result<(), ANNErrorPy> {
                let vector = vector.as_slice().unwrap();
                self.index
                    .insert(vector_id, vector, use_full_precision_to_search)?;

                Ok(())
            }

            pub fn consolidate_deletes(&self, num_threads: usize) -> Result<(), ANNErrorPy> {
                self.index.consolidate_deletes(num_threads)?;

                Ok(())
            }

            pub fn consolidate_simple(&self, num_threads: usize) -> Result<(), ANNErrorPy> {
                self.index.consolidate_simple(num_threads)?;

                Ok(())
            }

            pub fn mark_deleted(
                &self,
                vector_ids: &Bound<PyArray1<u32>>,
            ) -> Result<(), ANNErrorPy> {
                let vector_ids = vector_ids.readonly().as_array().to_vec();
                self.index.mark_deleted(vector_ids)?;

                Ok(())
            }

            pub fn batch_insert(
                &self,
                vector_ids: &Bound<PyArray1<u32>>,
                vectors: &Bound<PyArray2<$py_vector_type>>,
                use_full_precision_to_search: bool,
                num_tasks: usize,
            ) -> Result<(), ANNErrorPy> {
                let vectors = pyarray2_to_vec_row_decomp(vectors);
                let vector_ids = vector_ids.readonly().as_array().to_vec();
                self.index.batch_insert(
                    vector_ids,
                    vectors,
                    use_full_precision_to_search,
                    num_tasks,
                )?;

                Ok(())
            }

            pub fn batch_inplace_delete(
                &self,
                vector_ids: &Bound<PyArray1<u32>>,
                use_full_precision_to_search: bool,
                num_tasks: usize,
                k_value: usize,
                l_value: usize,
                num_to_replace: usize,
                delete_method: usize,
            ) -> Result<(), ANNErrorPy> {
                let vector_ids = vector_ids.readonly().as_array().to_vec();
                self.index.batch_inplace_delete(
                    vector_ids,
                    use_full_precision_to_search,
                    num_tasks,
                    k_value,
                    l_value,
                    num_to_replace,
                    delete_method,
                )?;

                Ok(())
            }

            pub fn multi_inplace_delete(
                &self,
                vector_ids: &Bound<PyArray1<u32>>,
                use_full_precision_to_search: bool,
                k_value: usize,
                l_value: usize,
                num_to_replace: usize,
                delete_method: usize,
            ) -> Result<(), ANNErrorPy> {
                // convert vector ids to Arc<[u32]>
                let vector_ids = vector_ids.readonly().as_array().to_vec();
                let vector_ids = Arc::from(vector_ids);
                self.index.multi_inplace_delete(
                    vector_ids,
                    use_full_precision_to_search,
                    k_value,
                    l_value,
                    num_to_replace,
                    delete_method,
                )?;
                Ok(())
            }

            pub fn multi_insert(
                &self,
                py_vector_id_pairs: &Bound<'_, PyList>,
                use_full_precision_to_search: bool,
            ) -> Result<(), ANNErrorPy> {
                let wrapper_vector_id_pairs: Vec<VectorIdBoxSliceWrapper<$py_vector_type>> =
                    py_vector_id_pairs.extract().unwrap();
                let vector_id_pairs: Box<[VectorIdBoxSlice<u32, $py_vector_type>]> =
                    wrapper_vector_id_pairs
                        .into_iter()
                        .map(VectorIdBoxSlice::from)
                        .collect();
                self.index
                    .multi_insert(vector_id_pairs, use_full_precision_to_search)?;
                Ok(())
            }

            pub fn search(
                &self,
                query: &Bound<PyArray1<$py_vector_type>>, //PyReadonlyArray1<'a, $py_vector_type>,
                k_value: usize,
                l_value: u32,
                use_full_precision_to_search: bool,
            ) -> Result<(SearchResult, u32), ANNErrorPy> {
                let query_as_vec = query.readonly().as_array().to_vec();
                let result = self.index.search(
                    &query_as_vec,
                    k_value,
                    l_value as usize,
                    use_full_precision_to_search,
                )?;

                Ok(result)
            }

            pub fn batch_search(
                &self,
                queries: &Bound<PyArray2<$py_vector_type>>,
                num_threads: u32,
                k_value: u32,
                l_value: u32,
                use_full_precision_to_search: bool,
            ) -> Result<BatchSearchResultWithStats, ANNErrorPy> {
                let queries_as_vec = pyarray2_to_vec_row_decomp(queries);
                let result = self.index.batch_search(
                    queries_as_vec,
                    num_threads,
                    k_value,
                    l_value,
                    use_full_precision_to_search,
                )?;

                Ok(result)
            }

            pub fn batch_range_search(
                &self,
                queries: &Bound<PyArray2<$py_vector_type>>,
                num_threads: u32,
                starting_l_value: u32,
                radius: f32,
            ) -> Result<BatchRangeSearchResultWithStats, ANNErrorPy> {
                let queries_as_vec = pyarray2_to_vec_row_decomp(queries);
                let result = self.index.batch_range_search(
                    queries_as_vec,
                    num_threads,
                    starting_l_value,
                    radius,
                )?;

                Ok(result)
            }

            pub fn get_neighbors(&self, vector_id: u32) -> Result<Vec<u32>, ANNErrorPy> {
                let result = self.index.get_neighbors(vector_id)?;

                Ok(result)
            }

            pub fn get_average_degree(&self) -> Result<f32, ANNErrorPy> {
                let result = self.index.get_average_degree()?;

                Ok(result)
            }
        }
    };
}

impl_async_memory_index!(AsyncMemoryIndexF32, f32, f32);
impl_async_memory_index!(AsyncMemoryIndexU8, u8, u8);
impl_async_memory_index!(AsyncMemoryIndexInt8, i8, i8);
