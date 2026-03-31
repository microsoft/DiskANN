/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
*/

use std::sync::Arc;

use diskann::{
    graph::{
        config, search, search_output_buffer, AdjacencyList, DiskANNIndex, InplaceDeleteMethod,
        SampleableForStart,
    },
    neighbor::Neighbor,
    provider::{DefaultContext, Delete, ElementStatus},
    utils::{IntoUsize, VectorIdBoxSlice, VectorRepr},
    ANNError, ANNErrorKind, ANNResult,
};
use diskann_providers::{
    model::{
        graph::provider::async_::{
            bf_tree::{self, AsVectorDtype, BfTreePaths, BfTreeProvider, CreateQuantProvider},
            common::{FullPrecision, Hybrid, NoStore},
            TableDeleteProviderAsync,
        },
        pq::FixedChunkPQTable,
    },
    storage::{FileStorageProvider, LoadWith, SaveWith},
    utils::{create_thread_pool, ParallelIteratorInPool},
};
use diskann_utils::sampling::WithApproximateNorm;

use diskann_vector::distance::Metric;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator};
use tokio::runtime::Runtime;

use crate::{
    build_bftree_index::{
        build_bftree_index_inner, build_bftree_pq_index, build_empty_bftree_index,
    },
    utils::{
        parallel_tasks, pyarray2_to_vec_row_decomp, search_result::SearchStats, ANNErrorPy,
        BatchRangeSearchResultWithStats, BatchSearchResultWithStats, MetricPy, SearchResult,
    },
};

// The actual quantization provider type created from FixedChunkPQTable
pub type PQProvider = <FixedChunkPQTable as CreateQuantProvider>::Target;

// Newtype wrappers to avoid trait implementation conflicts
pub struct FullPrecisionBfTreeIndex<T>(
    pub Arc<DiskANNIndex<BfTreeProvider<T, NoStore, TableDeleteProviderAsync>>>,
)
where
    T: VectorRepr;

pub struct PQBfTreeIndex<T>(pub bf_tree::QuantIndex<T, PQProvider, TableDeleteProviderAsync>)
where
    T: VectorRepr;

fn get_delete_method(
    method: usize,
    k_value: usize,
    l_value: usize,
) -> ANNResult<InplaceDeleteMethod> {
    match method {
        0 => Ok(InplaceDeleteMethod::TwoHopAndOneHop),
        1 => Ok(InplaceDeleteMethod::VisitedAndTopK { k_value, l_value }),
        2 => Ok(InplaceDeleteMethod::OneHop),
        _ => Err(ANNError::log_index_error(format!(
            "Invalid delete method: {}",
            method
        ))),
    }
}

// Trait for BfTree index implementations
trait BfTreeIndexImpl<T>: Send + Sync
where
    T: VectorRepr,
{
    fn search(
        &self,
        runtime: &Runtime,
        query: &[T],
        k_value: usize,
        l_value: usize,
        use_full_precision_to_search: bool,
    ) -> ANNResult<(SearchResult, u32)>;

    fn range_search(
        &self,
        runtime: &Runtime,
        query: &[T],
        starting_l_value: usize,
        radius: f32,
    ) -> ANNResult<(Vec<u32>, Vec<f32>, u32, bool)>;

    fn insert(&self, runtime: &Runtime, vector_id: u32, vector: &[T]) -> ANNResult<()>;

    fn batch_insert(
        &self,
        runtime: &Runtime,
        vector_ids: Vec<u32>,
        vectors: Vec<Vec<T>>,
        num_tasks: usize,
    ) -> ANNResult<()>;

    fn multi_insert(
        &self,
        runtime: &Runtime,
        vector_ids: Vec<u32>,
        vectors: Vec<Vec<T>>,
    ) -> ANNResult<()>;

    fn mark_deleted(&self, runtime: &Runtime, vector_ids: Vec<u32>) -> ANNResult<()>;

    fn clear_delete_set(&self, runtime: &Runtime) -> ANNResult<()>;

    fn consolidate_deletes(&self, runtime: &Runtime, num_tasks: usize) -> ANNResult<()>;

    fn get_neighbors(&self, runtime: &Runtime, vector_id: u32) -> ANNResult<Vec<u32>>;

    fn consolidate_simple(&self, runtime: &Runtime, num_tasks: usize) -> ANNResult<()>;

    fn multi_inplace_delete(
        &self,
        runtime: &Runtime,
        vector_ids: Vec<u32>,
        k_value: usize,
        l_value: usize,
        num_to_replace: usize,
        delete_method: usize,
    ) -> ANNResult<()>;

    #[allow(clippy::too_many_arguments)]
    fn batch_inplace_delete(
        &self,
        runtime: &Runtime,
        vector_ids: Vec<u32>,
        num_tasks: usize,
        k_value: usize,
        l_value: usize,
        num_to_replace: usize,
        delete_method: usize,
    ) -> ANNResult<()>;

    fn get_average_degree(&self, runtime: &Runtime) -> ANNResult<f32>;

    fn save(&self, runtime: &Runtime, prefix: &str) -> ANNResult<()>;
}

// Implementation for FullPrecisionBfTreeIndex
impl<T> BfTreeIndexImpl<T> for FullPrecisionBfTreeIndex<T>
where
    T: VectorRepr,
{
    fn search(
        &self,
        runtime: &Runtime,
        query: &[T],
        k_value: usize,
        l_value: usize,
        _use_full_precision_to_search: bool,
    ) -> ANNResult<(SearchResult, u32)> {
        if !_use_full_precision_to_search {
            return Err(ANNError::message(
                ANNErrorKind::IndexError,
                "FullPrecisionBfTreeIndex only supports full precision search.",
            ));
        }

        let mut result_ids = vec![0; k_value];
        let mut result_dists = vec![0.0; k_value];
        let mut cmps = 0;

        runtime.block_on(async {
            let mut result_output_buffer =
                search_output_buffer::IdDistance::new(&mut result_ids, &mut result_dists);
            let stats = self
                .0
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
        })?;

        Ok((
            SearchResult {
                ids: result_ids,
                distances: result_dists,
            },
            cmps,
        ))
    }

    fn range_search(
        &self,
        runtime: &Runtime,
        query: &[T],
        starting_l_value: usize,
        radius: f32,
    ) -> ANNResult<(Vec<u32>, Vec<f32>, u32, bool)> {
        runtime.block_on(async {
            let mut results: Vec<Neighbor<u32>> = Vec::new();
            let stats = self
                .0
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

            ANNResult::Ok((
                ids,
                distances,
                stats.cmps,
                stats.range_search_second_round,
            ))
        })
    }

    fn insert(&self, runtime: &Runtime, vector_id: u32, vector: &[T]) -> ANNResult<()> {
        runtime.block_on(async {
            self.0
                .insert(FullPrecision, &DefaultContext, &vector_id, vector)
                .await
        })?;
        Ok(())
    }

    fn batch_insert(
        &self,
        runtime: &Runtime,
        vector_ids: Vec<u32>,
        vectors: Vec<Vec<T>>,
        num_tasks: usize,
    ) -> ANNResult<()> {
        let index = self.0.clone();
        runtime.block_on(async move {
            parallel_tasks::run(
                vector_ids.into_iter().zip(vectors),
                num_tasks,
                move |(vector_id, vector)| {
                    let index = index.clone();
                    async move {
                        index
                            .insert(FullPrecision, &DefaultContext, &vector_id, &vector)
                            .await
                    }
                },
            )
            .await
        });
        Ok(())
    }

    fn multi_insert(
        &self,
        runtime: &Runtime,
        vector_ids: Vec<u32>,
        vectors: Vec<Vec<T>>,
    ) -> ANNResult<()> {
        let vector_id_pairs: Box<[VectorIdBoxSlice<u32, T>]> = vector_ids
            .into_iter()
            .zip(vectors)
            .map(|(id, vec)| VectorIdBoxSlice::new(id, vec.into_boxed_slice()))
            .collect();

        runtime.block_on(async {
            self.0
                .multi_insert(FullPrecision, &DefaultContext, vector_id_pairs)
                .await
        })?;
        Ok(())
    }

    fn mark_deleted(&self, runtime: &Runtime, vector_ids: Vec<u32>) -> ANNResult<()> {
        runtime.block_on(async {
            for id in vector_ids {
                self.0.data_provider.delete(&DefaultContext, &id).await?;
            }
            ANNResult::Ok(())
        })?;
        Ok(())
    }

    fn clear_delete_set(&self, runtime: &Runtime) -> ANNResult<()> {
        runtime.block_on(async {
            for id in self.0.data_provider.iter() {
                let is_deleted = self
                    .0
                    .data_provider
                    .status_by_internal_id(&DefaultContext, id)
                    .await
                    .unwrap_or(ElementStatus::Deleted);
                if is_deleted == ElementStatus::Deleted {
                    self.0.data_provider.release(&DefaultContext, id).await?;
                }
            }
            ANNResult::Ok(())
        })
    }

    fn consolidate_deletes(&self, runtime: &Runtime, num_tasks: usize) -> ANNResult<()> {
        let index = self.0.clone();
        runtime.block_on(async move {
            parallel_tasks::run(index.data_provider.iter(), num_tasks, move |id| {
                let index = index.clone();
                async move {
                    index
                        .consolidate_vector(&FullPrecision, &DefaultContext, id)
                        .await?;
                    ANNResult::Ok(())
                }
            })
            .await
        });

        self.clear_delete_set(runtime)?;
        Ok(())
    }

    fn get_neighbors(&self, runtime: &Runtime, vector_id: u32) -> ANNResult<Vec<u32>> {
        runtime.block_on(async {
            let accessor = self.0.provider().neighbors();
            let mut neighbors = AdjacencyList::new();
            accessor.get_neighbors(vector_id, &mut neighbors)?;

            Ok(neighbors.to_vec())
        })
    }

    fn consolidate_simple(&self, runtime: &Runtime, num_tasks: usize) -> ANNResult<()> {
        let index = self.0.clone();
        runtime.block_on(async move {
            parallel_tasks::run(index.data_provider.iter(), num_tasks, move |id| {
                let index = index.clone();
                async move {
                    let mut neighbor_accessor = index.provider().neighbors();
                    index
                        .drop_deleted_neighbors(&DefaultContext, &mut neighbor_accessor, id, false)
                        .await?;
                    ANNResult::Ok(())
                }
            })
            .await
        });

        self.clear_delete_set(runtime)?;
        Ok(())
    }

    fn multi_inplace_delete(
        &self,
        runtime: &Runtime,
        vector_ids: Vec<u32>,
        k_value: usize,
        l_value: usize,
        num_to_replace: usize,
        delete_method: usize,
    ) -> ANNResult<()> {
        let vector_ids: Arc<[u32]> = vector_ids.into();
        let delete_method = get_delete_method(delete_method, k_value, l_value)?;

        runtime.block_on(async {
            self.0
                .multi_inplace_delete(
                    FullPrecision,
                    &DefaultContext,
                    vector_ids,
                    num_to_replace,
                    delete_method,
                )
                .await
        })?;
        Ok(())
    }

    fn batch_inplace_delete(
        &self,
        runtime: &Runtime,
        vector_ids: Vec<u32>,
        num_tasks: usize,
        k_value: usize,
        l_value: usize,
        num_to_replace: usize,
        delete_method: usize,
    ) -> ANNResult<()> {
        let delete_method = get_delete_method(delete_method, k_value, l_value)?;

        let index = self.0.clone();
        runtime.block_on(async move {
            parallel_tasks::run(vector_ids.into_iter(), num_tasks, move |id| {
                let index = index.clone();
                async move {
                    index
                        .inplace_delete(
                            FullPrecision,
                            &DefaultContext,
                            &id,
                            num_to_replace,
                            delete_method,
                        )
                        .await
                }
            })
            .await
        });

        Ok(())
    }

    fn get_average_degree(&self, runtime: &Runtime) -> ANNResult<f32> {
        runtime.block_on(async {
            let mut accessor = self.0.provider().neighbors();
            let degree_stats = self.0.get_degree_stats(&mut accessor).await?;
            Ok(degree_stats.avg_degree)
        })
    }

    fn save(&self, runtime: &Runtime, prefix: &str) -> ANNResult<()> {
        let storage = FileStorageProvider;
        let prefix = prefix.to_string();
        runtime.block_on(async {
            self.0.data_provider.save_with(&storage, &prefix).await?;
            ANNResult::Ok(())
        })
    }
}

// Implementation for PQBfTreeIndex
impl<T> BfTreeIndexImpl<T> for PQBfTreeIndex<T>
where
    T: VectorRepr,
{
    fn search(
        &self,
        runtime: &Runtime,
        query: &[T],
        k_value: usize,
        l_value: usize,
        use_full_precision_to_search: bool,
    ) -> ANNResult<(SearchResult, u32)> {
        let mut result_ids = vec![0; k_value];
        let mut result_dists = vec![0.0; k_value];
        let mut cmps = 0;

        runtime.block_on(async {
            let mut result_output_buffer =
                search_output_buffer::IdDistance::new(&mut result_ids, &mut result_dists);
            let stats = if use_full_precision_to_search {
                self.0
                    .search(
                        search::Knn::new_default(l_value, l_value)?,
                        &FullPrecision,
                        &DefaultContext,
                        query,
                        &mut result_output_buffer,
                    )
                    .await?
            } else {
                self.0
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
        })?;

        Ok((
            SearchResult {
                ids: result_ids,
                distances: result_dists,
            },
            cmps,
        ))
    }

    fn range_search(
        &self,
        runtime: &Runtime,
        query: &[T],
        starting_l_value: usize,
        radius: f32,
    ) -> ANNResult<(Vec<u32>, Vec<f32>, u32, bool)> {
        runtime.block_on(async {
            let mut results: Vec<Neighbor<u32>> = Vec::new();
            let stats = self
                .0
                .search(
                    search::Range::new(starting_l_value, radius)?,
                    &Hybrid::new(None),
                    &DefaultContext,
                    query,
                    &mut results,
                )
                .await?;

            let ids: Vec<u32> = results.iter().map(|n| n.id).collect();
            let distances: Vec<f32> = results.iter().map(|n| n.distance).collect();

            ANNResult::Ok((
                ids,
                distances,
                stats.cmps,
                stats.range_search_second_round,
            ))
        })
    }

    fn insert(&self, runtime: &Runtime, vector_id: u32, vector: &[T]) -> ANNResult<()> {
        runtime.block_on(async {
            self.0
                .insert(Hybrid::new(None), &DefaultContext, &vector_id, vector)
                .await
        })?;
        Ok(())
    }

    fn batch_insert(
        &self,
        runtime: &Runtime,
        vector_ids: Vec<u32>,
        vectors: Vec<Vec<T>>,
        num_tasks: usize,
    ) -> ANNResult<()> {
        let index = self.0.clone();
        runtime.block_on(async move {
            parallel_tasks::run(
                vector_ids.into_iter().zip(vectors),
                num_tasks,
                move |(vector_id, vector)| {
                    let index = index.clone();
                    async move {
                        index
                            .insert(Hybrid::new(None), &DefaultContext, &vector_id, &vector)
                            .await
                    }
                },
            )
            .await
        });
        Ok(())
    }

    fn multi_insert(
        &self,
        runtime: &Runtime,
        vector_ids: Vec<u32>,
        vectors: Vec<Vec<T>>,
    ) -> ANNResult<()> {
        let vector_id_pairs: Box<[VectorIdBoxSlice<u32, T>]> = vector_ids
            .into_iter()
            .zip(vectors)
            .map(|(id, vec)| VectorIdBoxSlice::new(id, vec.into_boxed_slice()))
            .collect();

        runtime.block_on(async {
            self.0
                .multi_insert(Hybrid::new(None), &DefaultContext, vector_id_pairs)
                .await
        })?;
        Ok(())
    }

    fn mark_deleted(&self, runtime: &Runtime, vector_ids: Vec<u32>) -> ANNResult<()> {
        runtime.block_on(async {
            for id in vector_ids {
                self.0.data_provider.delete(&DefaultContext, &id).await?;
            }
            ANNResult::Ok(())
        })?;
        Ok(())
    }

    fn clear_delete_set(&self, runtime: &Runtime) -> ANNResult<()> {
        runtime.block_on(async {
            for id in self.0.data_provider.iter() {
                let is_deleted = self
                    .0
                    .data_provider
                    .status_by_internal_id(&DefaultContext, id)
                    .await
                    .unwrap_or(ElementStatus::Deleted);
                if is_deleted == ElementStatus::Deleted {
                    self.0.data_provider.release(&DefaultContext, id).await?;
                }
            }
            ANNResult::Ok(())
        })
    }

    fn consolidate_deletes(&self, runtime: &Runtime, num_tasks: usize) -> ANNResult<()> {
        let index = self.0.clone();
        runtime.block_on(async move {
            parallel_tasks::run(index.data_provider.iter(), num_tasks, move |id| {
                let index = index.clone();
                async move {
                    index
                        .consolidate_vector(&Hybrid::new(None), &DefaultContext, id)
                        .await?;
                    ANNResult::Ok(())
                }
            })
            .await
        });

        self.clear_delete_set(runtime)?;
        Ok(())
    }

    fn get_neighbors(&self, runtime: &Runtime, vector_id: u32) -> ANNResult<Vec<u32>> {
        runtime.block_on(async {
            let accessor = self.0.provider().neighbors();
            let mut neighbors = AdjacencyList::new();
            accessor.get_neighbors(vector_id, &mut neighbors)?;

            Ok(neighbors.to_vec())
        })
    }

    fn consolidate_simple(&self, runtime: &Runtime, num_tasks: usize) -> ANNResult<()> {
        let index = self.0.clone();
        runtime.block_on(async move {
            parallel_tasks::run(index.data_provider.iter(), num_tasks, move |id| {
                let index = index.clone();
                async move {
                    let mut neighbor_accessor = index.provider().neighbors();
                    index
                        .drop_deleted_neighbors(&DefaultContext, &mut neighbor_accessor, id, false)
                        .await?;
                    ANNResult::Ok(())
                }
            })
            .await
        });

        self.clear_delete_set(runtime)?;
        Ok(())
    }

    fn multi_inplace_delete(
        &self,
        runtime: &Runtime,
        vector_ids: Vec<u32>,
        k_value: usize,
        l_value: usize,
        num_to_replace: usize,
        delete_method: usize,
    ) -> ANNResult<()> {
        let vector_ids: Arc<[u32]> = vector_ids.into();
        let delete_method = get_delete_method(delete_method, k_value, l_value)?;

        runtime.block_on(async {
            self.0
                .multi_inplace_delete(
                    Hybrid::new(None),
                    &DefaultContext,
                    vector_ids,
                    num_to_replace,
                    delete_method,
                )
                .await
        })?;
        Ok(())
    }

    fn batch_inplace_delete(
        &self,
        runtime: &Runtime,
        vector_ids: Vec<u32>,
        num_tasks: usize,
        k_value: usize,
        l_value: usize,
        num_to_replace: usize,
        delete_method: usize,
    ) -> ANNResult<()> {
        let delete_method = get_delete_method(delete_method, k_value, l_value)?;

        let index = self.0.clone();
        runtime.block_on(async move {
            parallel_tasks::run(vector_ids.into_iter(), num_tasks, move |id| {
                let index = index.clone();
                async move {
                    index
                        .inplace_delete(
                            Hybrid::new(None),
                            &DefaultContext,
                            &id,
                            num_to_replace,
                            delete_method,
                        )
                        .await
                }
            })
            .await
        });

        Ok(())
    }

    fn get_average_degree(&self, runtime: &Runtime) -> ANNResult<f32> {
        runtime.block_on(async {
            let mut accessor = self.0.provider().neighbors();
            let degree_stats = self.0.get_degree_stats(&mut accessor).await?;
            Ok(degree_stats.avg_degree)
        })
    }

    fn save(&self, runtime: &Runtime, prefix: &str) -> ANNResult<()> {
        let storage = FileStorageProvider;
        let prefix = prefix.to_string();
        runtime.block_on(async {
            self.0.data_provider.save_with(&storage, &prefix).await?;
            ANNResult::Ok(())
        })
    }
}

pub struct BfTreeIndex<T>
where
    T: VectorRepr,
{
    index: Box<dyn BfTreeIndexImpl<T>>,
    runtime: Runtime,
}

#[pyclass]
pub struct BfTreeIndexF32 {
    pub index: BfTreeIndex<f32>,
}

#[pyclass]
pub struct BfTreeIndexU8 {
    pub index: BfTreeIndex<u8>,
}

#[pyclass]
pub struct BfTreeIndexInt8 {
    pub index: BfTreeIndex<i8>,
}

impl<T> BfTreeIndex<T>
where
    T: VectorRepr + SampleableForStart + WithApproximateNorm + AsVectorDtype,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        metric: Metric,
        data_path: String,
        r: u32,
        l: u32,
        alpha: f32,
        num_start_pts: usize,
        num_threads: u32,
        graph_slack_factor: f32,
        backedge_ratio: f32,
        num_tasks: usize,
        insert_minibatch_size: usize,
        create_empty_index: bool,
        max_points: usize,
        dim: usize,
        build_pq_bytes: usize,
        max_fp_vecs_per_fill: Option<usize>,
        pq_seed: u64,
        on_disk_prefix: Option<String>,
    ) -> Result<Self, ANNErrorPy> {
        let runtime = Runtime::new().unwrap();

        if create_empty_index {
            assert!(
                build_pq_bytes == 0,
                "Cannot create empty PQ index. Build PQ bytes must be 0."
            );
            let inner_index = build_empty_bftree_index::<T>(
                metric,
                r,
                l,
                alpha,
                graph_slack_factor,
                max_points,
                dim,
                num_start_pts,
                backedge_ratio,
            )?;
            Ok(BfTreeIndex {
                index: Box::new(inner_index),
                runtime,
            })
        } else {
            let storage_provider = FileStorageProvider;
            if build_pq_bytes == 0 {
                let inner_index = build_bftree_index_inner::<T, FileStorageProvider>(
                    metric,
                    data_path,
                    r,
                    l,
                    alpha,
                    num_start_pts,
                    num_threads,
                    graph_slack_factor,
                    backedge_ratio,
                    &storage_provider,
                    num_tasks,
                    insert_minibatch_size,
                    on_disk_prefix.clone(),
                )?;
                Ok(BfTreeIndex {
                    index: Box::new(inner_index),
                    runtime,
                })
            } else {
                let inner_index = build_bftree_pq_index::<T, FileStorageProvider>(
                    metric,
                    data_path,
                    r,
                    l,
                    alpha,
                    num_start_pts,
                    num_threads,
                    build_pq_bytes,
                    graph_slack_factor,
                    max_fp_vecs_per_fill,
                    backedge_ratio,
                    &storage_provider,
                    num_tasks,
                    pq_seed,
                    insert_minibatch_size,
                    on_disk_prefix,
                )?;
                Ok(BfTreeIndex {
                    index: Box::new(inner_index),
                    runtime,
                })
            }
        }
    }

    pub fn search(
        &self,
        query: &[T],
        k_value: usize,
        l_value: usize,
        use_full_precision_to_search: bool,
    ) -> ANNResult<(SearchResult, u32)> {
        self.index.search(
            &self.runtime,
            query,
            k_value,
            l_value,
            use_full_precision_to_search,
        )
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

    fn range_search(
        &self,
        query: &[T],
        starting_l_value: usize,
        radius: f32,
    ) -> ANNResult<(Vec<u32>, Vec<f32>, u32, bool)> {
        self.index
            .range_search(&self.runtime, query, starting_l_value, radius)
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

    pub fn insert(&self, vector_id: u32, vector: &[T]) -> ANNResult<()> {
        self.index.insert(&self.runtime, vector_id, vector)
    }

    pub fn batch_insert(
        &self,
        vector_ids: Vec<u32>,
        vectors: Vec<Vec<T>>,
        num_tasks: usize,
    ) -> ANNResult<()> {
        self.index
            .batch_insert(&self.runtime, vector_ids, vectors, num_tasks)
    }

    pub fn multi_insert(&self, vector_ids: Vec<u32>, vectors: Vec<Vec<T>>) -> ANNResult<()> {
        self.index.multi_insert(&self.runtime, vector_ids, vectors)
    }

    pub fn mark_deleted(&self, vector_ids: Vec<u32>) -> ANNResult<()> {
        self.index.mark_deleted(&self.runtime, vector_ids)
    }

    pub fn consolidate_deletes(&self, num_tasks: usize) -> ANNResult<()> {
        self.index.consolidate_deletes(&self.runtime, num_tasks)
    }

    pub fn get_neighbors(&self, vector_id: u32) -> ANNResult<Vec<u32>> {
        self.index.get_neighbors(&self.runtime, vector_id)
    }

    pub fn consolidate_simple(&self, num_tasks: usize) -> ANNResult<()> {
        self.index.consolidate_simple(&self.runtime, num_tasks)
    }

    pub fn multi_inplace_delete(
        &self,
        vector_ids: Vec<u32>,
        k_value: usize,
        l_value: usize,
        num_to_replace: usize,
        delete_method: usize,
    ) -> ANNResult<()> {
        self.index.multi_inplace_delete(
            &self.runtime,
            vector_ids,
            k_value,
            l_value,
            num_to_replace,
            delete_method,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_inplace_delete(
        &self,
        vector_ids: Vec<u32>,
        num_tasks: usize,
        k_value: usize,
        l_value: usize,
        num_to_replace: usize,
        delete_method: usize,
    ) -> ANNResult<()> {
        self.index.batch_inplace_delete(
            &self.runtime,
            vector_ids,
            num_tasks,
            k_value,
            l_value,
            num_to_replace,
            delete_method,
        )
    }

    pub fn get_average_degree(&self) -> ANNResult<f32> {
        self.index.get_average_degree(&self.runtime)
    }

    pub fn save(&self, prefix: &str) -> ANNResult<()> {
        self.index.save(&self.runtime, prefix)
    }

    pub fn load(prefix: String) -> Result<Self, ANNErrorPy> {
        let runtime = Runtime::new()
            .map_err(|e| ANNError::log_index_error(format!("Failed to create runtime: {}", e)))?;
        let storage = FileStorageProvider;

        // Read params JSON to determine if this is a PQ index
        let params_filename = BfTreePaths::params_json(&prefix);
        let params_json = std::fs::read_to_string(&params_filename).map_err(|e| {
            ANNError::log_index_error(format!(
                "Failed to read params file {}: {}",
                params_filename, e
            ))
        })?;
        let params: serde_json::Value = serde_json::from_str(&params_json).map_err(|e| {
            ANNError::log_index_error(format!("Failed to deserialize params: {}", e))
        })?;

        let metric_str = params["metric"]
            .as_str()
            .ok_or_else(|| ANNError::log_index_error("Missing 'metric' in params"))?;
        let metric: Metric = metric_str
            .parse()
            .map_err(|e| ANNError::log_index_error(format!("Failed to parse metric: {}", e)))?;

        let max_degree = params["max_degree"]
            .as_u64()
            .ok_or_else(|| ANNError::log_index_error("Missing 'max_degree' in params"))?
            as usize;
        let is_pq = params.get("quant_params").is_some_and(|v| !v.is_null());

        // Read graph params from saved params
        let gp = params
            .get("graph_params")
            .and_then(|v| if v.is_null() { None } else { Some(v) })
            .ok_or_else(|| ANNError::log_index_error("Missing 'graph_params' in params"))?;
        let l = gp["l_build"].as_u64().unwrap() as usize;
        let alpha = gp["alpha"].as_f64().unwrap() as f32;
        let backedge_ratio = gp["backedge_ratio"].as_f64().unwrap() as f32;

        let graph_config = config::Builder::new_with(
            max_degree,
            config::MaxDegree::same(),
            l,
            metric.into(),
            |builder| {
                builder.alpha(alpha).backedge_ratio(backedge_ratio);
            },
        )
        .build()
        .map_err(|e| ANNError::log_index_error(format!("Failed to build config: {}", e)))?;

        if is_pq {
            // PQ index
            let provider = runtime.block_on(async {
                <BfTreeProvider<T, PQProvider, TableDeleteProviderAsync>>::load_with(
                    &storage, &prefix,
                )
                .await
            })?;
            let index = Arc::new(DiskANNIndex::new(graph_config, provider, None));
            Ok(BfTreeIndex {
                index: Box::new(PQBfTreeIndex(index)),
                runtime,
            })
        } else {
            // Full precision index
            let provider = runtime.block_on(async {
                <BfTreeProvider<T, NoStore, TableDeleteProviderAsync>>::load_with(&storage, &prefix)
                    .await
            })?;
            let index = Arc::new(DiskANNIndex::new(graph_config, provider, None));
            Ok(BfTreeIndex {
                index: Box::new(FullPrecisionBfTreeIndex(index)),
                runtime,
            })
        }
    }
}

macro_rules! impl_bftree_index {
    ($index_type:ident, $vector_type:ty, $py_vector_type:ty) => {
        #[pymethods]
        #[allow(clippy::too_many_arguments)]
        impl $index_type {
            #[new]
            pub fn new(
                metric: MetricPy,
                data_path: String,
                r: u32,
                l: u32,
                alpha: f32,
                num_start_pts: usize,
                num_threads: u32,
                graph_slack_factor: f32,
                backedge_ratio: f32,
                num_tasks: usize,
                insert_minibatch_size: usize,
                create_empty_index: bool,
                max_points: usize,
                dim: usize,
                build_pq_bytes: usize,
                max_fp_vecs_per_fill: Option<usize>,
                pq_seed: u64,
                on_disk_prefix: Option<String>,
            ) -> Result<Self, ANNErrorPy> {
                let searcher = BfTreeIndex::<$vector_type>::new(
                    metric.into(),
                    data_path,
                    r,
                    l,
                    alpha,
                    num_start_pts,
                    num_threads,
                    graph_slack_factor,
                    backedge_ratio,
                    num_tasks,
                    insert_minibatch_size,
                    create_empty_index,
                    max_points,
                    dim,
                    build_pq_bytes,
                    max_fp_vecs_per_fill,
                    pq_seed,
                    on_disk_prefix,
                )?;

                Ok($index_type { index: searcher })
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

            pub fn range_search(
                &self,
                query: &Bound<PyArray1<$py_vector_type>>,
                starting_l_value: usize,
                radius: f32,
            ) -> Result<(Vec<u32>, Vec<f32>, u32, bool), ANNErrorPy> {
                let query_as_vec = query.readonly().as_array().to_vec();
                let result = self
                    .index
                    .range_search(&query_as_vec, starting_l_value, radius)?;

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

            pub fn insert(
                &self,
                vector: &Bound<PyArray1<$py_vector_type>>,
                id: u32,
            ) -> Result<(), ANNErrorPy> {
                let vector_readonly = vector.readonly();
                let vector_as_slice = vector_readonly.as_slice().unwrap();
                self.index.insert(id, vector_as_slice)?;

                Ok(())
            }

            pub fn batch_insert(
                &self,
                vectors: &Bound<PyArray2<$py_vector_type>>,
                vector_ids: Vec<u32>,
                num_tasks: usize,
            ) -> Result<(), ANNErrorPy> {
                let vectors_as_vec = pyarray2_to_vec_row_decomp(vectors);
                self.index
                    .batch_insert(vector_ids, vectors_as_vec, num_tasks)?;

                Ok(())
            }

            pub fn multi_insert(
                &self,
                vectors: &Bound<PyArray2<$py_vector_type>>,
                vector_ids: Vec<u32>,
            ) -> Result<(), ANNErrorPy> {
                let vectors_as_vec = pyarray2_to_vec_row_decomp(vectors);
                self.index.multi_insert(vector_ids, vectors_as_vec)?;

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

            pub fn consolidate_deletes(&self, num_tasks: usize) -> Result<(), ANNErrorPy> {
                self.index.consolidate_deletes(num_tasks)?;
                Ok(())
            }

            pub fn get_neighbors(&self, vector_id: u32) -> Result<Vec<u32>, ANNErrorPy> {
                Ok(self.index.get_neighbors(vector_id)?)
            }

            pub fn consolidate_simple(&self, num_tasks: usize) -> Result<(), ANNErrorPy> {
                self.index.consolidate_simple(num_tasks)?;
                Ok(())
            }

            pub fn multi_inplace_delete(
                &self,
                vector_ids: &Bound<PyArray1<u32>>,
                k_value: usize,
                l_value: usize,
                num_to_replace: usize,
                delete_method: usize,
            ) -> Result<(), ANNErrorPy> {
                let vector_ids = vector_ids.readonly().as_array().to_vec();
                self.index.multi_inplace_delete(
                    vector_ids,
                    k_value,
                    l_value,
                    num_to_replace,
                    delete_method,
                )?;

                Ok(())
            }

            pub fn batch_inplace_delete(
                &self,
                vector_ids: &Bound<PyArray1<u32>>,
                num_tasks: usize,
                k_value: usize,
                l_value: usize,
                num_to_replace: usize,
                delete_method: usize,
            ) -> Result<(), ANNErrorPy> {
                let vector_ids = vector_ids.readonly().as_array().to_vec();
                self.index.batch_inplace_delete(
                    vector_ids,
                    num_tasks,
                    k_value,
                    l_value,
                    num_to_replace,
                    delete_method,
                )?;

                Ok(())
            }

            pub fn get_average_degree(&self) -> Result<f32, ANNErrorPy> {
                let result = self.index.get_average_degree()?;

                Ok(result)
            }

            pub fn save(&self, prefix: String) -> Result<(), ANNErrorPy> {
                self.index.save(&prefix)?;
                Ok(())
            }

            #[staticmethod]
            pub fn load(prefix: String) -> Result<$index_type, ANNErrorPy> {
                let index = BfTreeIndex::<$vector_type>::load(prefix)?;
                Ok($index_type { index })
            }
        }
    };
}

impl_bftree_index!(BfTreeIndexF32, f32, f32);
impl_bftree_index!(BfTreeIndexU8, u8, u8);
impl_bftree_index!(BfTreeIndexInt8, i8, i8);
