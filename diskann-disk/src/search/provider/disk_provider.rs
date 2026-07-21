/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::HashMap,
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicU64, AtomicUsize},
        Arc,
    },
    time::Instant,
};

use crate::data_model::GraphDataType;
use diskann::{
    error::IntoANNResult,
    graph::{
        self,
        ext::labeled::{self, QueryLabelProvider},
        glue::{self, DefaultPostProcessor, SearchPostProcess, SearchStrategy},
        search::{AdaptiveL, InlineFilterSearch, Knn},
        search_output_buffer, DiskANNIndex,
    },
    neighbor::{Neighbor, NeighborPriorityQueue},
    provider::{DataProvider, DefaultContext, HasId, NoopGuard},
    utils::{IntoUsize, VectorRepr},
    ANNError, ANNResult,
};
use diskann_providers::storage::StorageReadProvider;
use diskann_providers::{
    model::{
        compute_pq_distance,
        graph::provider::{determinant_diversity, DeterminantDiversityParams},
    },
    storage::{get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file, LoadWith},
};
use diskann_utils::{
    object_pool::{ObjectPool, PoolOption, TryAsPooled},
    views::Matrix,
};

use crate::search::pq::{quantizer_preprocess, PQData, PQScratch};
use diskann_vector::{distance::Metric, DistanceFunction};
use tokio::runtime::Runtime;
use tracing::debug;

use crate::{
    data_model::{CachingStrategy, GraphHeader},
    search::{
        provider::{
            aligned_file_reader::AlignedFileReaderFactory,
            disk_vertex_provider_factory::DiskVertexProviderFactory,
        },
        search_mode::SearchMode,
        traits::{VertexProvider, VertexProviderFactory},
    },
    storage::{api::AsyncDiskLoadContext, disk_index_reader::DiskIndexReader},
    utils::QueryStatistics,
};

///////////////////
// Disk Provider //
///////////////////

/// The DiskProvider is a data provider that loads data from disk using the disk readers
/// The data format for disk is different from that of the in-memory providers.
/// The disk format stores both the vectors and the adjacency list next to each other for
/// better locality for quicker access.
/// Please refer to the RFC documentation at [`docs\rfcs\cy2025\disk_provider_for_async_index.md`] for design details.
pub struct DiskProvider<Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    /// Holds the graph header information that contains metadata about disk-index file.
    graph_header: GraphHeader,

    // Full precision distance comparer used in post_process to reorder results.
    distance_comparer: <Data::VectorDataType as VectorRepr>::Distance,

    /// The PQ data used for quantization.
    pq_data: Arc<PQData>,

    /// The number of points in the graph.
    num_points: usize,

    /// Metric used for distance computation.
    metric: Metric,

    /// The number of IO operations that can be done in parallel.
    search_io_limit: usize,
}

impl<Data> DataProvider for DiskProvider<Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    type Context = DefaultContext;

    type InternalId = u32;

    type ExternalId = u32;

    type Guard = NoopGuard<u32>;

    type Error = ANNError;

    /// Translate an external id to its corresponding internal id.
    fn to_internal_id(
        &self,
        _context: &DefaultContext,
        gid: &Self::ExternalId,
    ) -> Result<Self::InternalId, Self::Error> {
        Ok(*gid)
    }

    /// Translate an internal id its corresponding external id.
    fn to_external_id(
        &self,
        _context: &DefaultContext,
        id: Self::InternalId,
    ) -> Result<Self::ExternalId, Self::Error> {
        Ok(id)
    }
}

impl<Data> LoadWith<AsyncDiskLoadContext> for DiskProvider<Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    type Error = ANNError;

    async fn load_with<P>(provider: &P, ctx: &AsyncDiskLoadContext) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        debug!(
            "DiskProvider::load_with() called with file: {:?}",
            get_disk_index_file(ctx.quant_load_context.metadata.prefix())
        );

        let graph_header = {
            let caching_strategy = if ctx.num_nodes_to_cache > 0 {
                CachingStrategy::StaticCacheWithBfsNodes(ctx.num_nodes_to_cache)
            } else {
                CachingStrategy::None
            };

            let vertex_provider_factory =
                DiskVertexProviderFactory::<Data, AlignedFileReaderFactory>::from_disk_index_path(
                    get_disk_index_file(ctx.quant_load_context.metadata.prefix()),
                    caching_strategy,
                )?;
            VertexProviderFactory::get_header(&vertex_provider_factory)?
        };

        let metric = ctx.quant_load_context.metric;
        let num_points = ctx.num_points;

        let index_path_prefix = ctx.quant_load_context.metadata.prefix();
        let index_reader = DiskIndexReader::new(
            get_pq_pivot_file(index_path_prefix),
            get_compressed_pq_file(index_path_prefix),
            provider,
        )?;

        Self::new(
            &index_reader,
            graph_header,
            metric,
            num_points,
            ctx.search_io_limit,
        )
    }
}

impl<Data> DiskProvider<Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    fn new(
        disk_index_reader: &DiskIndexReader,
        graph_header: GraphHeader,
        metric: Metric,
        num_points: usize,
        search_io_limit: usize,
    ) -> ANNResult<Self> {
        let distance_comparer =
            Data::VectorDataType::distance(metric, Some(graph_header.metadata().dims));

        let pq_data = disk_index_reader.get_pq_data();

        Ok(Self {
            graph_header,
            distance_comparer,
            pq_data,
            num_points,
            metric,
            search_io_limit,
        })
    }
}

/// The search strategy for the disk provider. This is used to create the search accessor
/// for use in search in quant space and post_process function to reorder with full precision vectors.
///
/// # Why vertex_provider_factory and scratch_pool are here instead of DiskProvider
///
/// The DataProvider trait requires 'static bounds for multi-threaded async contexts,
/// but vertex_provider_factory may have non-'static lifetime bounds (e.g., borrowing
/// from local data structures). Moving these components to the search strategy allows
/// DiskProvider to satisfy 'static constraints while enabling flexible per-search
/// resource management.
/// Borrowed predicate used internally by the disk search pipeline.
/// Spelled out here to keep the field/parameter signatures under
/// `clippy::type_complexity`'s default threshold.
type PostprocessFilter<'a> = &'a (dyn Fn(&u32) -> bool + Send + Sync);

/// Encodes whether to accept all candidates at rerank time or apply a
/// specific predicate. Used by `RerankAndFilter` and
/// `DeterminantDiversityAndFilter` instead of `Option<PostprocessFilter>`
/// so call sites are self-documenting without relying on comments to
/// explain what `None` means.
#[derive(Clone, Copy)]
pub enum PostprocessStrategy<'a> {
    /// Accept every candidate — no predicate is called. Used by `FlatScan`
    /// (filtered at scan time) and `InlineFilter` (filtered at visit time).
    AcceptAll,
    /// Apply the given predicate; non-matching candidates are dropped.
    Apply(PostprocessFilter<'a>),
}

pub struct DiskSearchStrategy<'a, Data, ProviderFactory>
where
    Data: GraphDataType<VectorIdType = u32>,
    ProviderFactory: VertexProviderFactory<Data>,
{
    // Borrowed from `search_internal` so the strategy can be passed by value
    io_tracker: &'a IOTracker,
    /// Consumed only by `default_post_processor()` → `RerankAndFilter`.
    /// `FlatScan` and `InlineFilter` filter earlier in their pipelines and
    /// pass `AcceptAll` here to avoid a redundant second pass.
    postprocess_filter: PostprocessStrategy<'a>,

    /// The vertex provider factory is used to create the vertex provider for each search instance.
    vertex_provider_factory: &'a ProviderFactory,

    /// Scratch pool for disk search operations that need allocations.
    scratch_pool: &'a Arc<ObjectPool<DiskSearchScratch<Data, ProviderFactory::VertexProviderType>>>,
}

// Struct to track IO. This is used by single thread, but needs to be Atomic as the Strategy has "Send" trait bound.
// There should be minimal to no overhead compared to using a raw reference.
struct IOTracker {
    io_time_us: AtomicU64,
    preprocess_time_us: AtomicU64,
    io_count: AtomicUsize,
}

impl Default for IOTracker {
    fn default() -> Self {
        Self {
            io_time_us: AtomicU64::new(0),
            preprocess_time_us: AtomicU64::new(0),
            io_count: AtomicUsize::new(0),
        }
    }
}

impl IOTracker {
    fn add_time(category: &AtomicU64, time: u64) {
        category.fetch_add(time, std::sync::atomic::Ordering::Relaxed);
    }

    fn time(category: &AtomicU64) -> u64 {
        category.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn add_io_count(&self, count: usize) {
        self.io_count
            .fetch_add(count, std::sync::atomic::Ordering::Relaxed);
    }

    fn io_count(&self) -> usize {
        self.io_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[derive(Clone, Copy)]
pub struct RerankAndFilter<'a> {
    filter: PostprocessStrategy<'a>,
}

#[derive(Clone, Copy)]
pub struct DeterminantDiversityAndFilter<'a> {
    filter: PostprocessStrategy<'a>,
    params: DeterminantDiversityParams,
}

/// Internal dispatch wrapper used by `search_internal`'s `DiverseGraph` arm
/// to feed `DiskANNIndex::search_with`. Hidden behind `SearchMode` from the
/// public API.
#[derive(Clone, Copy)]
pub enum DiskSearchPostProcessor<'a> {
    RerankAndFilter(RerankAndFilter<'a>),
    DeterminantDiversity(DeterminantDiversityAndFilter<'a>),
}

impl<'a> RerankAndFilter<'a> {
    pub fn new(filter: PostprocessStrategy<'a>) -> Self {
        Self { filter }
    }
}

impl<'a> DeterminantDiversityAndFilter<'a> {
    pub fn new(filter: PostprocessStrategy<'a>, params: DeterminantDiversityParams) -> Self {
        Self { filter, params }
    }
}

impl<Data, VP>
    SearchPostProcess<
        DiskAccessor<'_, Data, VP>,
        &[Data::VectorDataType],
        (
            <DiskProvider<Data> as DataProvider>::InternalId,
            Data::AssociatedDataType,
        ),
    > for RerankAndFilter<'_>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    type Error = ANNError;
    async fn post_process<I, B>(
        &self,
        accessor: &mut DiskAccessor<'_, Data, VP>,
        query: &[Data::VectorDataType],
        candidates: I,
        output: &mut B,
    ) -> Result<usize, Self::Error>
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: search_output_buffer::SearchOutputBuffer<(u32, Data::AssociatedDataType)>
            + Send
            + ?Sized,
    {
        let provider = accessor.provider;

        let mut uncached_ids = Vec::new();
        let mut reranked = {
            let mut process = |n: u32| {
                if let Some(entry) = accessor.scratch.distance_cache.get(&n) {
                    Some(Ok::<((u32, _), f32), ANNError>(((n, entry.1), entry.0)))
                } else {
                    uncached_ids.push(n);
                    None
                }
            };
            match self.filter {
                PostprocessStrategy::AcceptAll => candidates
                    .map(|n| n.id)
                    .filter_map(&mut process)
                    .collect::<Result<Vec<_>, _>>()?,
                PostprocessStrategy::Apply(f) => candidates
                    .map(|n| n.id)
                    .filter(|id| f(id))
                    .filter_map(&mut process)
                    .collect::<Result<Vec<_>, _>>()?,
            }
        };
        if !uncached_ids.is_empty() {
            ensure_vertex_loaded(&mut accessor.scratch.vertex_provider, &uncached_ids)?;
            for n in &uncached_ids {
                let v = accessor.scratch.vertex_provider.get_vector(n)?;
                let d = provider.distance_comparer.evaluate_similarity(query, v);
                let a = accessor.scratch.vertex_provider.get_associated_data(n)?;
                reranked.push(((*n, *a), d));
            }
        }

        // Sort the full precision distances.
        reranked
            .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        // Store the reranked results.
        Ok(output.extend(reranked))
    }
}

impl<Data, VP>
    SearchPostProcess<
        DiskAccessor<'_, Data, VP>,
        &[Data::VectorDataType],
        (
            <DiskProvider<Data> as DataProvider>::InternalId,
            Data::AssociatedDataType,
        ),
    > for DeterminantDiversityAndFilter<'_>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    type Error = ANNError;
    async fn post_process<I, B>(
        &self,
        accessor: &mut DiskAccessor<'_, Data, VP>,
        query: &[Data::VectorDataType],
        candidates: I,
        output: &mut B,
    ) -> Result<usize, Self::Error>
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: search_output_buffer::SearchOutputBuffer<(u32, Data::AssociatedDataType)>
            + Send
            + ?Sized,
    {
        let provider = accessor.provider;
        let query_f32 = Data::VectorDataType::as_f32(query).map_err(Into::into)?;

        let candidate_ids: Vec<u32> = match self.filter {
            PostprocessStrategy::AcceptAll => candidates.map(|candidate| candidate.id).collect(),
            PostprocessStrategy::Apply(f) => candidates
                .map(|candidate| candidate.id)
                .filter(|id| f(id))
                .collect(),
        };

        if candidate_ids.is_empty() {
            return Ok(0);
        }

        ensure_vertex_loaded(&mut accessor.scratch.vertex_provider, &candidate_ids)?;

        let mut candidate_vectors = Matrix::new(0.0f32, candidate_ids.len(), query_f32.len());
        let mut candidate_distances = Vec::with_capacity(candidate_ids.len());
        let mut associated_data = Vec::with_capacity(candidate_ids.len());

        for (row_idx, id) in candidate_ids.iter().enumerate() {
            let vector = accessor.scratch.vertex_provider.get_vector(id)?;
            let distance = provider
                .distance_comparer
                .evaluate_similarity(query, vector);
            let vector_f32 = Data::VectorDataType::as_f32(vector).map_err(Into::into)?;
            let data = accessor.scratch.vertex_provider.get_associated_data(id)?;

            candidate_vectors
                .row_mut(row_idx)
                .copy_from_slice(&vector_f32);
            candidate_distances.push(distance);
            associated_data.push(*data);
        }

        let reranked = determinant_diversity(
            candidate_vectors.as_mut_view(),
            &candidate_distances,
            &query_f32,
            usize::MAX,
            &self.params,
        )?;

        Ok(output.extend(reranked.into_iter().map(|idx| {
            let id = candidate_ids[idx];
            let distance = candidate_distances[idx];
            ((id, associated_data[idx]), distance)
        })))
    }
}

impl<Data, VP>
    SearchPostProcess<
        DiskAccessor<'_, Data, VP>,
        &[Data::VectorDataType],
        (
            <DiskProvider<Data> as DataProvider>::InternalId,
            Data::AssociatedDataType,
        ),
    > for DiskSearchPostProcessor<'_>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    type Error = ANNError;
    async fn post_process<I, B>(
        &self,
        accessor: &mut DiskAccessor<'_, Data, VP>,
        query: &[Data::VectorDataType],
        candidates: I,
        output: &mut B,
    ) -> Result<usize, Self::Error>
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: search_output_buffer::SearchOutputBuffer<(u32, Data::AssociatedDataType)>
            + Send
            + ?Sized,
    {
        match self {
            DiskSearchPostProcessor::RerankAndFilter(pp) => {
                pp.post_process(accessor, query, candidates, output).await
            }
            DiskSearchPostProcessor::DeterminantDiversity(pp) => {
                pp.post_process(accessor, query, candidates, output).await
            }
        }
    }
}

impl<'this, Data, ProviderFactory>
    SearchStrategy<'this, DiskProvider<Data>, &'this [Data::VectorDataType]>
    for DiskSearchStrategy<'this, Data, ProviderFactory>
where
    Data: GraphDataType<VectorIdType = u32>,
    ProviderFactory: VertexProviderFactory<Data>,
{
    type SearchAccessor = DiskAccessor<'this, Data, ProviderFactory::VertexProviderType>;
    type SearchAccessorError = ANNError;

    fn search_accessor(
        &'this self,
        provider: &'this DiskProvider<Data>,
        _context: &DefaultContext,
        query: &'this [Data::VectorDataType],
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        DiskAccessor::new(
            provider,
            self.io_tracker,
            query,
            self.vertex_provider_factory,
            self.scratch_pool,
        )
    }
}

impl<'this, Data, ProviderFactory>
    DefaultPostProcessor<
        'this,
        DiskProvider<Data>,
        &'this [Data::VectorDataType],
        (
            <DiskProvider<Data> as DataProvider>::InternalId,
            Data::AssociatedDataType,
        ),
    > for DiskSearchStrategy<'this, Data, ProviderFactory>
where
    Data: GraphDataType<VectorIdType = u32>,
    ProviderFactory: VertexProviderFactory<Data>,
{
    type Processor = RerankAndFilter<'this>;

    fn default_post_processor(&'this self) -> Self::Processor {
        RerankAndFilter::new(self.postprocess_filter)
    }
}

// Scratch space for disk search operations that need allocations.
// These allocations are amortized across searches using the scratch pool.
struct DiskSearchScratch<Data, VP>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    distance_cache: HashMap<u32, (f32, Data::AssociatedDataType)>,
    pq_scratch: PQScratch,
    vertex_provider: VP,
}

#[derive(Clone)]
struct DiskSearchScratchArgs<'a, ProviderFactory> {
    graph_degree: usize,
    pq_dim: usize,
    num_pq_chunks: usize,
    num_pq_centers: usize,
    vertex_factory: &'a ProviderFactory,
    graph_header: &'a GraphHeader,
}

impl<Data, ProviderFactory> TryAsPooled<&DiskSearchScratchArgs<'_, ProviderFactory>>
    for DiskSearchScratch<Data, ProviderFactory::VertexProviderType>
where
    Data: GraphDataType<VectorIdType = u32>,
    ProviderFactory: VertexProviderFactory<Data>,
{
    type Error = ANNError;

    fn try_create(args: &DiskSearchScratchArgs<ProviderFactory>) -> Result<Self, Self::Error> {
        let pq_scratch = PQScratch::new(
            args.graph_degree,
            args.pq_dim,
            args.num_pq_chunks,
            args.num_pq_centers,
        )?;

        const DEFAULT_BEAM_WIDTH: usize = 0; // Setting as 0 to avoid preallocation of memory.
        let vertex_provider = args
            .vertex_factory
            .create_vertex_provider(DEFAULT_BEAM_WIDTH, args.graph_header)?;

        Ok(Self {
            distance_cache: HashMap::new(),
            pq_scratch,
            vertex_provider,
        })
    }

    fn try_modify(
        &mut self,
        _args: &DiskSearchScratchArgs<ProviderFactory>,
    ) -> Result<(), Self::Error> {
        self.distance_cache.clear();
        self.vertex_provider.clear();
        Ok(())
    }
}

pub struct DiskAccessor<'a, Data, VP>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    provider: &'a DiskProvider<Data>,
    io_tracker: &'a IOTracker,
    scratch: PoolOption<DiskSearchScratch<Data, VP>>,
    query: &'a [Data::VectorDataType],
}

impl<Data, VP> DiskAccessor<'_, Data, VP>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    // Compute the PQ distance between each ID in `ids` and the distance table stored in
    // `self`, invoking the callback with the results of each computation in order.
    fn pq_distances<F>(&mut self, ids: &[u32], mut f: F) -> ANNResult<()>
    where
        F: FnMut(f32, u32),
    {
        let pq_scratch = &mut self.scratch.pq_scratch;
        compute_pq_distance(
            ids,
            self.provider.pq_data.get_num_chunks(),
            &pq_scratch.aligned_pqtable_dist_scratch,
            self.provider.pq_data.pq_compressed_data().as_slice(),
            &mut pq_scratch.aligned_pq_coord_scratch,
            &mut pq_scratch.aligned_dist_scratch,
        )?;

        for (i, id) in ids.iter().enumerate() {
            let distance = self.scratch.pq_scratch.aligned_dist_scratch[i];
            f(distance, *id);
        }

        Ok(())
    }
}

impl<Data, VP> HasId for DiskAccessor<'_, Data, VP>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    type Id = u32;
}

impl<Data, VP> glue::SearchAccessor for DiskAccessor<'_, Data, VP>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    async fn starting_points(&self) -> ANNResult<Vec<u32>> {
        let start_vertex_id = self.provider.graph_header.metadata().medoid as u32;
        Ok(vec![start_vertex_id])
    }

    async fn start_point_distances<F>(&mut self, mut f: F) -> ANNResult<()>
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        let start_vertex_id = self.provider.graph_header.metadata().medoid as u32;
        self.pq_distances(&[start_vertex_id], |dist, id| f(id, dist))
    }

    fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        mut pred: P,
        mut f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(Self::Id, f32) + Send,
    {
        let result = (|| {
            let io_limit = self.provider.search_io_limit - self.io_tracker.io_count();
            let load_ids: Box<[_]> = ids.take(io_limit).collect();

            self.ensure_loaded(&load_ids)?;
            let mut ids = Vec::new();
            for i in load_ids {
                ids.clear();
                ids.extend(
                    self.scratch
                        .vertex_provider
                        .get_adjacency_list(&i)?
                        .iter()
                        .copied()
                        .filter(|id| pred.eval_mut(id)),
                );

                self.pq_distances(&ids, &mut |dist, id| f(id, dist))?;
            }

            Ok(())
        })();

        std::future::ready(result)
    }

    fn terminate_early(&mut self) -> bool {
        self.io_tracker.io_count() > self.provider.search_io_limit
    }
}

impl<'a, Data, VP> DiskAccessor<'a, Data, VP>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    fn new<VPF>(
        provider: &'a DiskProvider<Data>,
        io_tracker: &'a IOTracker,
        query: &'a [Data::VectorDataType],
        vertex_provider_factory: &'a VPF,
        scratch_pool: &'a Arc<ObjectPool<DiskSearchScratch<Data, VP>>>,
    ) -> ANNResult<Self>
    where
        VPF: VertexProviderFactory<Data, VertexProviderType = VP>,
    {
        let mut scratch = PoolOption::try_pooled(
            scratch_pool,
            &DiskSearchScratchArgs {
                graph_degree: provider.graph_header.max_degree::<Data::VectorDataType>()?,
                pq_dim: provider.pq_data.get_dim(),
                num_pq_chunks: provider.pq_data.get_num_chunks(),
                num_pq_centers: provider.pq_data.get_num_centers(),
                vertex_factory: vertex_provider_factory,
                graph_header: &provider.graph_header,
            },
        )?;

        // Decode caller's native vector representation into `f32`; downstream PQ kernels operate purely on `&[f32]`.
        let f32_query = Data::VectorDataType::as_f32(query).into_ann_result()?;
        scratch.pq_scratch.set(&f32_query)?;
        let start_vertex_id = provider.graph_header.metadata().medoid as u32;

        let timer = Instant::now();
        quantizer_preprocess(
            &mut scratch.pq_scratch,
            &provider.pq_data,
            provider.metric,
            &[start_vertex_id],
        )?;
        IOTracker::add_time(
            &io_tracker.preprocess_time_us,
            timer.elapsed().as_micros() as u64,
        );

        Ok(Self {
            provider,
            io_tracker,
            scratch,
            query,
        })
    }

    fn ensure_loaded(&mut self, ids: &[u32]) -> Result<(), ANNError> {
        if ids.is_empty() {
            return Ok(());
        }
        let scratch = &mut self.scratch;
        let timer = Instant::now();
        ensure_vertex_loaded(&mut scratch.vertex_provider, ids)?;
        IOTracker::add_time(
            &self.io_tracker.io_time_us,
            timer.elapsed().as_micros() as u64,
        );
        self.io_tracker.add_io_count(ids.len());
        for id in ids {
            let distance = self
                .provider
                .distance_comparer
                .evaluate_similarity(self.query, scratch.vertex_provider.get_vector(id)?);
            let associated_data = *scratch.vertex_provider.get_associated_data(id)?;
            scratch
                .distance_cache
                .insert(*id, (distance, associated_data));
        }
        Ok(())
    }
}

/// [`DiskIndexSearcher`] is a helper class to make it easy to construct index
/// and do repeated search operations. It is a wrapper around the index.
/// This is useful for drivers such as search_disk_index.exe in tools.
pub struct DiskIndexSearcher<
    Data,
    ProviderFactory = DiskVertexProviderFactory<Data, AlignedFileReaderFactory>,
> where
    Data: GraphDataType<VectorIdType = u32>,
    ProviderFactory: VertexProviderFactory<Data>,
{
    index: DiskANNIndex<DiskProvider<Data>>,
    runtime: Runtime,

    /// The vertex provider factory is used to create the vertex provider for each search instance.
    vertex_provider_factory: ProviderFactory,

    /// Scratch pool for disk search operations that need allocations.
    scratch_pool: Arc<ObjectPool<DiskSearchScratch<Data, ProviderFactory::VertexProviderType>>>,
}

#[derive(Debug)]
pub struct SearchResultStats {
    pub cmps: u32,
    pub result_count: u32,
    pub query_statistics: QueryStatistics,
}

/// `SearchResult` is a struct representing the result of a search operation.
///
/// It contains a list of vector results and a statistics object
///
pub struct SearchResult<AssociatedData> {
    /// A list of nearest neighbors resulting from the search.
    pub results: Vec<SearchResultItem<AssociatedData>>,
    pub stats: SearchResultStats,
}

/// `VectorResult` is a struct representing a nearest neighbor resulting from a search.
///
/// It contains the vertex id, associated data, and the distance to the query vector.
///
pub struct SearchResultItem<AssociatedData> {
    /// The vertex id of the nearest neighbor.
    pub vertex_id: u32,
    /// The associated data of the nearest neighbor as a fixed size byte array.
    /// The length is determined when the index is created.
    pub data: AssociatedData,
    /// The distance between the nearest neighbor and the query vector.
    pub distance: f32,
}

impl<Data, ProviderFactory> DiskIndexSearcher<Data, ProviderFactory>
where
    Data: GraphDataType<VectorIdType = u32>,
    ProviderFactory: VertexProviderFactory<Data>,
{
    /// Create a new asynchronous disk searcher instance.
    ///
    /// # Arguments
    /// * `num_threads` - The maximum number of threads to use.
    /// * `search_io_limit` - I/O operation limit.
    /// * `disk_index_reader` - The disk index reader.
    /// * `vertex_provider_factory` - The vertex provider factory.
    /// * `metric` - Distance metric used for vector similarity calculations.
    /// * `runtime` - Tokio runtime handle for executing async operations.
    pub fn new(
        num_threads: usize,
        search_io_limit: usize,
        disk_index_reader: &DiskIndexReader,
        vertex_provider_factory: ProviderFactory,
        metric: Metric,
        runtime: Option<Runtime>,
    ) -> ANNResult<Self> {
        let runtime = match runtime {
            Some(rt) => rt,
            None => tokio::runtime::Builder::new_current_thread().build()?,
        };

        let graph_header = vertex_provider_factory.get_header()?;
        let metadata = graph_header.metadata();
        let max_degree = graph_header.max_degree::<Data::VectorDataType>()? as u32;

        let config = graph::config::Builder::new(
            max_degree.into_usize(),
            graph::config::MaxDegree::default_slack(),
            1, // build-search-list-size
            metric.into(),
        )
        .build()?;

        debug!("Creating DiskIndexSearcher with index_config: {:?}", config);

        let graph_header = vertex_provider_factory.get_header()?;
        let pq_data = disk_index_reader.get_pq_data();
        let scratch_pool_args = DiskSearchScratchArgs {
            graph_degree: graph_header.max_degree::<Data::VectorDataType>()?,
            pq_dim: pq_data.get_dim(),
            num_pq_chunks: pq_data.get_num_chunks(),
            num_pq_centers: pq_data.get_num_centers(),
            vertex_factory: &vertex_provider_factory,
            graph_header: &graph_header,
        };
        let scratch_pool = Arc::new(ObjectPool::try_new(&scratch_pool_args, 0, None)?);

        let disk_provider = DiskProvider::new(
            disk_index_reader,
            graph_header,
            metric,
            metadata.num_pts.into_usize(),
            search_io_limit,
        )?;

        let index = DiskANNIndex::new(config, disk_provider, NonZeroUsize::new(num_threads));
        Ok(Self {
            index,
            runtime,
            vertex_provider_factory,
            scratch_pool,
        })
    }

    /// Helper method to create a `DiskSearchStrategy` with common parameters.
    fn search_strategy<'a>(
        &'a self,
        io_tracker: &'a IOTracker,
        postprocess_filter: PostprocessStrategy<'a>,
    ) -> DiskSearchStrategy<'a, Data, ProviderFactory> {
        DiskSearchStrategy {
            io_tracker,
            postprocess_filter,
            vertex_provider_factory: &self.vertex_provider_factory,
            scratch_pool: &self.scratch_pool,
        }
    }

    /// Perform a brute-force linear scan of all points in the index, returning the
    /// nearest neighbors that pass `vector_filter`.
    ///
    /// `vector_filter = None` scans every vector (recall baseline) and skips
    /// the per-ID dyn-fn call entirely.
    ///
    /// The top `neighbors_before_reranking` candidates from the quantized scan will be
    /// provided to full-precision reranking.
    async fn flat_search<OB>(
        &self,
        strategy: &DiskSearchStrategy<'_, Data, ProviderFactory>,
        query: &[Data::VectorDataType],
        vector_filter: Option<&(dyn Fn(&u32) -> bool + Send + Sync)>,
        neighbors_before_reranking: usize,
        output: &mut OB,
    ) -> ANNResult<graph::index::SearchStats>
    where
        OB: search_output_buffer::SearchOutputBuffer<(u32, Data::AssociatedDataType)> + Send,
    {
        let provider = self.index.provider();
        let mut accessor = strategy
            .search_accessor(provider, &DefaultContext, query)
            .into_ann_result()?;

        // Derive the batch size from the scratch data structure. Providing too many vectors
        // will panic.
        let batch_size = accessor.scratch.pq_scratch.max_vectors();

        // This check should always hold since `graph_degree` comes from
        // `diskann::graph::Config` and is forced to be non-zero. But this is defensive
        // against misconfiguration.
        if batch_size == 0 {
            return Err(ANNError::message(
                diskann::ANNErrorKind::IndexError,
                "pq scratch must support at least one vector",
            ));
        }

        let mut id_buffer = Vec::with_capacity(batch_size);

        let mut best = NeighborPriorityQueue::new(neighbors_before_reranking);
        let mut cmps = 0u32;

        // `None` short-circuits to `true` — no dyn-fn call per node on the
        // unfiltered (recall-baseline) path.
        let mut iter =
            (0..provider.num_points as u32).filter(|id| vector_filter.is_none_or(|f| f(id)));
        loop {
            id_buffer.clear();
            id_buffer.extend(iter.by_ref().take(batch_size));

            if id_buffer.is_empty() {
                break;
            }

            accessor.pq_distances(&id_buffer, |dist, id| best.insert(Neighbor::new(id, dist)))?;
            cmps += id_buffer.len() as u32;
        }

        let result_count = strategy
            .default_post_processor()
            .post_process(&mut accessor, query, best.iter(), output)
            .await
            .into_ann_result()?;

        Ok(graph::index::SearchStats {
            cmps,
            hops: 0,
            result_count: result_count as u32,
            range_search_second_round: false,
        })
    }

    /// Run inline label-filtered graph search with optional adaptive-L sizing.
    ///
    /// Wraps `Knn` in an `InlineFilterSearch` that tracks matched candidates
    /// during traversal. When `adaptive_l = Some(_)`, the beam (`l_search`)
    /// is grown mid-query if the observed match specificity is low (see
    /// `diskann::graph::search::AdaptiveL`).
    ///
    /// The label-provider trait object is built once in
    /// `SearchMode::inline_filter` from a generic adapter, so each filter
    /// evaluation costs exactly one indirect dispatch (through the
    /// `&dyn QueryLabelProvider` boundary required by `labeled::Filtered`),
    /// not two.
    ///
    /// Reuses the same `DiskAccessor` surface as the plain `Knn` graph path:
    /// `start_point_distances` and `expand_beam`, both of which call
    /// `pq_distances` internally.
    async fn filter_search<'a, OB>(
        &self,
        strategy: DiskSearchStrategy<'a, Data, ProviderFactory>,
        query: &[Data::VectorDataType],
        knn: Knn,
        label_provider: &(dyn QueryLabelProvider<u32> + 'a),
        adaptive_l: Option<AdaptiveL>,
        output: &mut OB,
    ) -> ANNResult<graph::index::SearchStats>
    where
        OB: search_output_buffer::SearchOutputBuffer<(u32, Data::AssociatedDataType)> + Send,
    {
        let filtered_strategy = labeled::Filtered::new(strategy, label_provider);
        let search = InlineFilterSearch::new(knn, adaptive_l);
        self.index
            .search(search, &filtered_strategy, &DefaultContext, query, output)
            .await
    }

    /// Perform a search on the disk index.
    /// return the list of nearest neighbors and associated data.
    pub fn search(
        &self,
        query: &[Data::VectorDataType],
        return_list_size: u32,
        search_list_size: u32,
        beam_width: Option<usize>,
        mode: SearchMode<'_>,
    ) -> ANNResult<SearchResult<Data::AssociatedDataType>> {
        let mut query_stats = QueryStatistics::default();
        let mut indices = vec![0u32; return_list_size as usize];
        let mut distances = vec![0f32; return_list_size as usize];
        let mut associated_data =
            vec![Data::AssociatedDataType::default(); return_list_size as usize];

        let stats = self.search_internal(
            query,
            return_list_size as usize,
            search_list_size,
            beam_width,
            &mut query_stats,
            &mut indices,
            &mut distances,
            &mut associated_data,
            &mode,
        )?;

        let mut search_result = SearchResult {
            results: Vec::with_capacity(return_list_size as usize),
            stats,
        };

        for ((vertex_id, distance), associated_data) in indices
            .into_iter()
            .zip(distances.into_iter())
            .zip(associated_data.into_iter())
        {
            search_result.results.push(SearchResultItem {
                vertex_id,
                distance,
                data: associated_data,
            });
        }

        Ok(search_result)
    }

    /// Perform a raw search on the disk index.
    /// This is a lower-level API that allows more control over the search parameters and output buffers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn search_internal(
        &self,
        query: &[Data::VectorDataType],
        k_value: usize,
        search_list_size: u32,
        beam_width: Option<usize>,
        query_stats: &mut QueryStatistics,
        indices: &mut [u32],
        distances: &mut [f32],
        associated_data: &mut [Data::AssociatedDataType],
        mode: &SearchMode<'_>,
    ) -> ANNResult<SearchResultStats> {
        let mut result_output_buffer = search_output_buffer::IdDistanceAssociatedData::new(
            &mut indices[..k_value],
            &mut distances[..k_value],
            &mut associated_data[..k_value],
        );

        let timer = Instant::now();
        let k = k_value;
        let l = search_list_size as usize;

        let io_tracker = IOTracker::default();

        // * `FlatScan`     — `flat_search` filters the scan iterator at
        //                    construction; non-matching IDs never enter `best`.
        // * `Graph`        — plain greedy traversal doesn't consult any predicate;
        //                    if a predicate is set, `RerankAndFilter` filters out
        //                    non-matching nodes at rerank time.
        // * `InlineFilter` — `InlineFilterSearch` only forwards `Accept` nodes
        //                    into `matched_results`; no filtering in post-process.
        // * `DiverseGraph` — `index.search_with` runs `DeterminantDiversityAndFilter`
        //                    as the post-processor over the L candidate pool.
        let stats = match mode {
            SearchMode::FlatScan { filter } => {
                let strategy = self.search_strategy(&io_tracker, PostprocessStrategy::AcceptAll);
                self.runtime.block_on(self.flat_search(
                    &strategy,
                    query,
                    filter.as_deref(),
                    l,
                    &mut result_output_buffer,
                ))?
            }
            SearchMode::Graph { filter } => {
                let strategy = self.search_strategy(
                    &io_tracker,
                    filter
                        .as_deref()
                        .map_or(PostprocessStrategy::AcceptAll, PostprocessStrategy::Apply),
                );
                let knn_search = Knn::new(k, l, beam_width)?;
                self.runtime.block_on(self.index.search(
                    knn_search,
                    &strategy,
                    &DefaultContext,
                    query,
                    &mut result_output_buffer,
                ))?
            }
            SearchMode::InlineFilter { filter, adaptive_l } => {
                // Strategy is passed by value into `filter_search` so that the
                // `labeled::Filtered` wrapper can own it; `io_tracker` keeps
                // its counters reachable from this scope.
                let strategy = self.search_strategy(&io_tracker, PostprocessStrategy::AcceptAll);
                let knn_search = Knn::new(k, l, beam_width)?;
                self.runtime.block_on(self.filter_search(
                    strategy,
                    query,
                    knn_search,
                    filter.as_ref(),
                    adaptive_l.clone(),
                    &mut result_output_buffer,
                ))?
            }
            SearchMode::DiverseGraph { filter, params } => {
                // Strategy installs the filter so `RerankAndFilter` would also
                // honor it, but the active post-processor here is the
                // diversity selector built from `DiskSearchPostProcessor`.
                let postprocess_config = filter
                    .as_deref()
                    .map_or(PostprocessStrategy::AcceptAll, PostprocessStrategy::Apply);
                let strategy = self.search_strategy(&io_tracker, postprocess_config);
                let knn_search = Knn::new(k, l, beam_width)?;
                let processor = DiskSearchPostProcessor::DeterminantDiversity(
                    DeterminantDiversityAndFilter::new(postprocess_config, *params),
                );
                self.runtime.block_on(self.index.search_with(
                    knn_search,
                    &strategy,
                    processor,
                    &DefaultContext,
                    query,
                    &mut result_output_buffer,
                ))?
            }
        };
        query_stats.total_comparisons = stats.cmps;
        query_stats.search_hops = stats.hops;

        query_stats.total_execution_time_us = timer.elapsed().as_micros();
        query_stats.io_time_us = IOTracker::time(&io_tracker.io_time_us) as u128;
        query_stats.total_io_operations = io_tracker.io_count() as u32;
        query_stats.total_vertices_loaded = io_tracker.io_count() as u32;
        query_stats.query_pq_preprocess_time_us =
            IOTracker::time(&io_tracker.preprocess_time_us) as u128;
        query_stats.cpu_time_us = query_stats.total_execution_time_us
            - query_stats.io_time_us
            - query_stats.query_pq_preprocess_time_us;
        Ok(SearchResultStats {
            cmps: query_stats.total_comparisons,
            result_count: stats.result_count,
            query_statistics: query_stats.clone(),
        })
    }
}

/// Helper function to ensure vertices are loaded and processed.
///
/// This is a convenience function that combines `load_vertices` and `process_loaded_node`
/// for each vertex ID. It first loads all the vertices in batch, then processes each
/// loaded node.
fn ensure_vertex_loaded<Data: GraphDataType, V: VertexProvider<Data>>(
    vertex_provider: &mut V,
    ids: &[Data::VectorIdType],
) -> ANNResult<()> {
    vertex_provider.load_vertices(ids)?;
    for (idx, id) in ids.iter().enumerate() {
        vertex_provider.process_loaded_node(id, idx)?;
    }
    Ok(())
}

#[cfg(test)]
mod disk_provider_tests {
    use crate::test_utils::{GraphDataF32VectorU32Data, GraphDataF32VectorUnitData};
    use diskann::{
        graph::{
            search::{record::VisitedSearchRecord, Knn},
            KnnSearchError,
        },
        utils::IntoUsize,
        ANNErrorKind,
    };
    use diskann_providers::storage::{
        DynWriteProvider, StorageReadProvider, VirtualStorageProvider,
    };
    use diskann_providers::utils::{create_thread_pool, PQPathNames, ParallelIteratorInPool};
    use diskann_utils::{io::read_bin, test_data_root};
    use diskann_vector::distance::Metric;
    use rayon::prelude::IndexedParallelIterator;
    use rstest::rstest;
    use vfs::OverlayFS;

    use super::*;
    use crate::{
        build::builder::merged_index::disk_index_builder_tests::{IndexBuildFixture, TestParams},
        search::provider::aligned_file_reader::VirtualAlignedReaderFactory,
        utils::QueryStatistics,
    };

    const TEST_INDEX_PREFIX_128DIM: &str =
        "/disk_index_search/disk_index_sift_learn_R4_L50_A1.2_truth_search";
    const TEST_INDEX_128DIM: &str =
        "/disk_index_search/disk_index_sift_learn_R4_L50_A1.2_truth_search_disk.index";
    const TEST_PQ_PIVOT_128DIM: &str =
        "/disk_index_search/disk_index_sift_learn_R4_L50_A1.2_truth_search_pq_pivots.bin";
    const TEST_PQ_COMPRESSED_128DIM: &str =
        "/disk_index_search/disk_index_sift_learn_R4_L50_A1.2_truth_search_pq_compressed.bin";
    const TEST_TRUTH_RESULT_10PTS_128DIM: &str =
        "/disk_index_search/disk_index_10pts_idx_uint32_truth_search_res.bin";
    const TEST_QUERY_10PTS_128DIM: &str = "/disk_index_search/disk_index_sample_query_10pts.fbin";

    const TEST_INDEX_PREFIX_100DIM: &str = "/disk_index_search/256pts_100dim_f32_truth_Index";
    const TEST_INDEX_100DIM: &str = "/disk_index_search/256pts_100dim_f32_truth_Index_disk.index";
    const TEST_PQ_PIVOT_100DIM: &str =
        "/disk_index_search/256pts_100dim_f32_truth_Index_pq_pivots.bin";
    const TEST_PQ_COMPRESSED_100DIM: &str =
        "/disk_index_search/256pts_100dim_f32_truth_Index_pq_compressed.bin";
    const TEST_TRUTH_RESULT_10PTS_100DIM: &str =
        "/disk_index_search/256pts_100dim_f32_truth_query_result.bin";
    const TEST_QUERY_10PTS_100DIM: &str = "/disk_index_search/10pts_100dim_f32_base_query.bin";
    const TEST_DATA_FILE: &str = "/disk_index_search/disk_index_siftsmall_learn_256pts_data.fbin";
    const TEST_INDEX: &str =
        "/disk_index_search/disk_index_sift_learn_R4_L50_A1.2_truth_search_disk.index";
    const TEST_INDEX_PREFIX: &str =
        "/disk_index_search/disk_index_sift_learn_R4_L50_A1.2_truth_search";
    const TEST_PQ_PIVOT: &str =
        "/disk_index_search/disk_index_sift_learn_R4_L50_A1.2_truth_search_pq_pivots.bin";
    const TEST_PQ_COMPRESSED: &str =
        "/disk_index_search/disk_index_sift_learn_R4_L50_A1.2_truth_search_pq_compressed.bin";

    #[test]
    fn test_disk_search_k10_l20_single_or_multi_thread_100dim() {
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));

        let search_engine = create_disk_index_searcher(
            CreateDiskIndexSearcherParams {
                max_thread_num: 5,
                pq_pivot_file_path: TEST_PQ_PIVOT_100DIM,
                pq_compressed_file_path: TEST_PQ_COMPRESSED_100DIM,
                index_path: TEST_INDEX_100DIM,
                index_path_prefix: TEST_INDEX_PREFIX_100DIM,
                ..Default::default()
            },
            &storage_provider,
        );
        // Test single thread.
        test_disk_search(TestDiskSearchParams {
            storage_provider: storage_provider.as_ref(),
            index_search_engine: &search_engine,
            thread_num: 1,
            query_file_path: TEST_QUERY_10PTS_100DIM,
            truth_result_file_path: TEST_TRUTH_RESULT_10PTS_100DIM,
            k: 10,
            l: 20,
        });
        // Test multi thread.
        test_disk_search(TestDiskSearchParams {
            storage_provider: storage_provider.as_ref(),
            index_search_engine: &search_engine,
            thread_num: 5,
            query_file_path: TEST_QUERY_10PTS_100DIM,
            truth_result_file_path: TEST_TRUTH_RESULT_10PTS_100DIM,
            k: 10,
            l: 20,
        });
    }

    #[rstest]
    #[case(CachingStrategy::None)]
    #[case(CachingStrategy::StaticCacheWithBfsNodes(32))]
    fn test_disk_search_k10_l20_single_or_multi_thread_128dim(
        #[case] caching_strategy: CachingStrategy,
    ) {
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));

        let search_engine = create_disk_index_searcher::<GraphDataF32VectorUnitData>(
            CreateDiskIndexSearcherParams {
                max_thread_num: 5,
                pq_pivot_file_path: TEST_PQ_PIVOT_128DIM,
                pq_compressed_file_path: TEST_PQ_COMPRESSED_128DIM,
                index_path: TEST_INDEX_128DIM,
                index_path_prefix: TEST_INDEX_PREFIX_128DIM,
                caching_strategy,
                ..Default::default()
            },
            &storage_provider,
        );
        // Test single thread.
        test_disk_search(TestDiskSearchParams {
            storage_provider: storage_provider.as_ref(),
            index_search_engine: &search_engine,
            thread_num: 1,
            query_file_path: TEST_QUERY_10PTS_128DIM,
            truth_result_file_path: TEST_TRUTH_RESULT_10PTS_128DIM,
            k: 10,
            l: 20,
        });
        // Test multi thread.
        test_disk_search(TestDiskSearchParams {
            storage_provider: storage_provider.as_ref(),
            index_search_engine: &search_engine,
            thread_num: 5,
            query_file_path: TEST_QUERY_10PTS_128DIM,
            truth_result_file_path: TEST_TRUTH_RESULT_10PTS_128DIM,
            k: 10,
            l: 20,
        });
    }

    fn get_truth_associated_data<StorageReader: StorageReadProvider>(
        storage_provider: &StorageReader,
    ) -> Vec<u32> {
        const ASSOCIATED_DATA_FILE: &str = "/sift/siftsmall_learn_256pts_u32_associated_data.fbin";

        let data =
            read_bin::<u32>(&mut storage_provider.open_reader(ASSOCIATED_DATA_FILE).unwrap())
                .unwrap();
        data.into_inner().into_vec()
    }

    #[test]
    fn test_disk_search_with_associated_data_k10_l20_single_or_multi_thread_128dim() {
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());
        let index_path_prefix = "/disk_index_search/disk_index_sift_learn_R4_L50_A1.2_test_disk_index_search_associated_data";
        let params = TestParams {
            data_path: TEST_DATA_FILE.to_string(),
            index_path_prefix: index_path_prefix.to_string(),
            associated_data_path: Some(
                "/sift/siftsmall_learn_256pts_u32_associated_data.fbin".to_string(),
            ),
            ..TestParams::default()
        };
        let fixture = IndexBuildFixture::new(storage_provider, params).unwrap();
        // Build the index with the associated data
        fixture.build::<GraphDataF32VectorU32Data>().unwrap();
        {
            let search_engine = create_disk_index_searcher::<GraphDataF32VectorU32Data>(
                CreateDiskIndexSearcherParams {
                    max_thread_num: 5,
                    pq_pivot_file_path: format!("{}_pq_pivots.bin", index_path_prefix).as_str(),
                    pq_compressed_file_path: format!("{}_pq_compressed.bin", index_path_prefix)
                        .as_str(),
                    index_path: format!("{}_disk.index", index_path_prefix).as_str(), //TEST_INDEX_128DIM,
                    index_path_prefix,
                    ..Default::default()
                },
                &fixture.storage_provider,
            );

            // Test single thread.
            test_disk_search_with_associated(
                TestDiskSearchAssociateParams {
                    storage_provider: fixture.storage_provider.as_ref(),
                    index_search_engine: &search_engine,
                    thread_num: 1,
                    query_file_path: TEST_QUERY_10PTS_128DIM,
                    truth_result_file_path: TEST_TRUTH_RESULT_10PTS_128DIM,
                    k: 10,
                    l: 20,
                },
                None,
            );

            // Test multi thread.
            test_disk_search_with_associated(
                TestDiskSearchAssociateParams {
                    storage_provider: fixture.storage_provider.as_ref(),
                    index_search_engine: &search_engine,
                    thread_num: 5,
                    query_file_path: TEST_QUERY_10PTS_128DIM,
                    truth_result_file_path: TEST_TRUTH_RESULT_10PTS_128DIM,
                    k: 10,
                    l: 20,
                },
                None,
            );
        }

        fixture
            .storage_provider
            .delete(&format!("{}_disk.index", index_path_prefix))
            .expect("Failed to delete file");
        fixture
            .storage_provider
            .delete(&format!("{}_pq_pivots.bin", index_path_prefix))
            .expect("Failed to delete file");
        fixture
            .storage_provider
            .delete(&format!("{}_pq_compressed.bin", index_path_prefix))
            .expect("Failed to delete file");
    }

    struct CreateDiskIndexSearcherParams<'a> {
        max_thread_num: usize,
        pq_pivot_file_path: &'a str,
        pq_compressed_file_path: &'a str,
        index_path: &'a str,
        index_path_prefix: &'a str,
        io_limit: usize,
        caching_strategy: CachingStrategy,
    }

    impl Default for CreateDiskIndexSearcherParams<'_> {
        fn default() -> Self {
            Self {
                max_thread_num: 1,
                pq_pivot_file_path: "",
                pq_compressed_file_path: "",
                index_path: "",
                index_path_prefix: "",
                io_limit: usize::MAX,
                caching_strategy: CachingStrategy::None,
            }
        }
    }

    fn create_disk_index_searcher<Data>(
        params: CreateDiskIndexSearcherParams,
        storage_provider: &Arc<VirtualStorageProvider<OverlayFS>>,
    ) -> DiskIndexSearcher<
        Data,
        DiskVertexProviderFactory<Data, VirtualAlignedReaderFactory<OverlayFS>>,
    >
    where
        Data: GraphDataType<VectorIdType = u32>,
    {
        assert!(params.io_limit > 0);

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(params.max_thread_num)
            .build()
            .unwrap();

        let disk_index_reader = DiskIndexReader::new(
            params.pq_pivot_file_path.to_string(),
            params.pq_compressed_file_path.to_string(),
            storage_provider.as_ref(),
        )
        .unwrap();

        let aligned_reader_factory = VirtualAlignedReaderFactory::new(
            get_disk_index_file(params.index_path_prefix),
            Arc::clone(storage_provider),
        );
        let vertex_provider_factory = DiskVertexProviderFactory::<Data, _>::new(
            aligned_reader_factory,
            params.caching_strategy,
        )
        .unwrap();

        DiskIndexSearcher::<Data, DiskVertexProviderFactory<Data, _>>::new(
            params.max_thread_num,
            params.io_limit,
            &disk_index_reader,
            vertex_provider_factory,
            Metric::L2,
            Some(runtime),
        )
        .unwrap()
    }

    fn load_query_result<StorageReader: StorageReadProvider>(
        storage_provider: &StorageReader,
        query_result_path: &str,
    ) -> Vec<u32> {
        let result =
            read_bin::<u32>(&mut storage_provider.open_reader(query_result_path).unwrap()).unwrap();
        result.into_inner().into_vec()
    }

    struct TestDiskSearchParams<'a, StorageType> {
        storage_provider: &'a StorageType,
        index_search_engine: &'a DiskIndexSearcher<
            GraphDataF32VectorUnitData,
            DiskVertexProviderFactory<
                GraphDataF32VectorUnitData,
                VirtualAlignedReaderFactory<OverlayFS>,
            >,
        >,
        thread_num: u64,
        query_file_path: &'a str,
        truth_result_file_path: &'a str,
        k: usize,
        l: usize,
    }

    struct TestDiskSearchAssociateParams<'a, StorageType> {
        storage_provider: &'a StorageType,
        index_search_engine: &'a DiskIndexSearcher<
            GraphDataF32VectorU32Data,
            DiskVertexProviderFactory<
                GraphDataF32VectorU32Data,
                VirtualAlignedReaderFactory<OverlayFS>,
            >,
        >,
        thread_num: u64,
        query_file_path: &'a str,
        truth_result_file_path: &'a str,
        k: usize,
        l: usize,
    }

    fn test_disk_search<StorageType: StorageReadProvider>(
        params: TestDiskSearchParams<StorageType>,
    ) {
        let queries = read_bin::<f32>(
            &mut params
                .storage_provider
                .open_reader(params.query_file_path)
                .unwrap(),
        )
        .unwrap();
        let truth_result =
            load_query_result(params.storage_provider, params.truth_result_file_path);

        let pool = create_thread_pool(params.thread_num.into_usize()).unwrap();
        queries
            .par_row_iter()
            .enumerate()
            .for_each_in_pool(pool.as_ref(), |(i, query)| {
                let mut query_stats = QueryStatistics::default();
                let mut indices = vec![0u32; 10];
                let mut distances = vec![0f32; 10];
                let mut associated_data = vec![(); 10];

                let result = params.index_search_engine.search_internal(
                    query,
                    params.k,
                    params.l as u32,
                    None, // beam_width
                    &mut query_stats,
                    &mut indices,
                    &mut distances,
                    &mut associated_data,
                    &SearchMode::graph(),
                );

                // Calculate the range of the truth_result for this query
                let truth_slice = &truth_result[i * params.k..(i + 1) * params.k];

                assert!(result.is_ok(), "Expected search to succeed");

                let result_unwrapped = result.unwrap();
                assert!(
                    result_unwrapped.query_statistics.total_io_operations > 0,
                    "Expected IO operations to be greater than 0"
                );
                assert!(
                    result_unwrapped.query_statistics.total_vertices_loaded > 0,
                    "Expected vertices loaded to be greater than 0"
                );

                // Compare res with truth_slice using assert_eq!
                assert_eq!(
                    indices, truth_slice,
                    "Results DO NOT match with the truth result for query {}",
                    i
                );
            });
    }

    fn test_disk_search_with_associated<StorageType: StorageReadProvider>(
        params: TestDiskSearchAssociateParams<StorageType>,
        beam_width: Option<usize>,
    ) {
        let queries = read_bin::<f32>(
            &mut params
                .storage_provider
                .open_reader(params.query_file_path)
                .unwrap(),
        )
        .unwrap();
        let truth_result =
            load_query_result(params.storage_provider, params.truth_result_file_path);
        let pool = create_thread_pool(params.thread_num.into_usize()).unwrap();
        queries
            .par_row_iter()
            .enumerate()
            .for_each_in_pool(pool.as_ref(), |(i, query)| {
                let result = params
                    .index_search_engine
                    .search(
                        query,
                        params.k as u32,
                        params.l as u32,
                        beam_width,
                        SearchMode::graph(),
                    )
                    .unwrap();
                let indices: Vec<u32> = result.results.iter().map(|item| item.vertex_id).collect();
                let associated_data: Vec<u32> =
                    result.results.iter().map(|item| item.data).collect();
                let truth_data = get_truth_associated_data(params.storage_provider);
                let associated_data_truth: Vec<u32> = indices
                    .iter()
                    .map(|&vid| truth_data[vid as usize])
                    .collect();
                assert_eq!(
                    associated_data, associated_data_truth,
                    "Associated data DO NOT match with the truth result for query {}, associated_data from search: {:?}, associated_data from truth result: {:?}",
                    i,associated_data, associated_data_truth
                );
                let truth_slice = &truth_result[i * params.k..(i + 1) * params.k];
                assert_eq!(
                    indices, truth_slice,
                    "Results DO NOT match with the truth result for query {}",
                    i
                );
            });
    }

    #[test]
    fn test_disk_search_invalid_input() {
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));
        let ctx = &DefaultContext;

        let params = CreateDiskIndexSearcherParams {
            max_thread_num: 5,
            pq_pivot_file_path: TEST_PQ_PIVOT_128DIM,
            pq_compressed_file_path: TEST_PQ_COMPRESSED_128DIM,
            index_path: TEST_INDEX_128DIM,
            index_path_prefix: TEST_INDEX_PREFIX_128DIM,
            ..Default::default()
        };

        let paths = PQPathNames::for_disk_index(TEST_INDEX_PREFIX_128DIM);
        assert_eq!(
            paths.pivots, params.pq_pivot_file_path,
            "pq_pivot_file_path is not correct"
        );
        assert_eq!(
            paths.compressed_data, params.pq_compressed_file_path,
            "pq_compressed_file_path is not correct"
        );
        assert_eq!(
            params.index_path,
            format!("{}_disk.index", params.index_path_prefix),
            "index_path is not correct"
        );

        // Test error case: l < k
        let res = Knn::new_default(20, 10);
        assert!(res.is_err());
        assert_eq!(
            <KnnSearchError as std::convert::Into<ANNError>>::into(res.unwrap_err()).kind(),
            ANNErrorKind::IndexError
        );
        // Test error case: beam_width = 0
        let res = Knn::new(10, 10, Some(0));
        assert!(res.is_err());

        let search_engine =
            create_disk_index_searcher::<GraphDataF32VectorU32Data>(params, &storage_provider);

        // minor validation tests to improve code coverage
        assert_eq!(
            search_engine
                .index
                .data_provider
                .to_external_id(ctx, 0)
                .unwrap(),
            0
        );
        assert_eq!(
            search_engine
                .index
                .data_provider
                .to_internal_id(ctx, &0)
                .unwrap(),
            0
        );

        let provider_max_degree = search_engine
            .index
            .data_provider
            .graph_header
            .max_degree::<<GraphDataF32VectorU32Data as GraphDataType>::VectorDataType>()
            .unwrap();
        let index_max_degree = search_engine.index.config.pruned_degree().get();
        assert_eq!(provider_max_degree, index_max_degree);

        let query = vec![0f32; 128];
        let mut query_stats = QueryStatistics::default();
        let mut indices = vec![0u32; 10];
        let mut distances = vec![0f32; 10];
        let mut associated_data = vec![0u32; 10];

        // Set L: {} to a value of at least K:
        let result = search_engine.search_internal(
            &query,
            10,
            10 - 1,
            None,
            &mut query_stats,
            &mut indices,
            &mut distances,
            &mut associated_data,
            &SearchMode::graph(),
        );

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), ANNErrorKind::IndexError);
    }

    #[test]
    fn test_disk_search_beam_search() {
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));

        let search_engine = create_disk_index_searcher::<GraphDataF32VectorUnitData>(
            CreateDiskIndexSearcherParams {
                max_thread_num: 1,
                pq_pivot_file_path: TEST_PQ_PIVOT,
                pq_compressed_file_path: TEST_PQ_COMPRESSED,
                index_path: TEST_INDEX,
                index_path_prefix: TEST_INDEX_PREFIX,
                ..Default::default()
            },
            &storage_provider,
        );

        let query_vector: [f32; 128] = [1f32; 128];
        let mut indices = vec![0u32; 10];
        let mut distances = vec![0f32; 10];
        let mut associated_data = vec![(); 10];

        let mut result_output_buffer = search_output_buffer::IdDistanceAssociatedData::new(
            &mut indices,
            &mut distances,
            &mut associated_data,
        );
        let io_tracker = IOTracker::default();
        let strategy = search_engine.search_strategy(&io_tracker, PostprocessStrategy::AcceptAll);
        let mut search_record = VisitedSearchRecord::new(0);
        let search_params = Knn::new(10, 10, Some(4)).unwrap();
        let recorded_search =
            diskann::graph::search::RecordedKnn::new(search_params, &mut search_record);
        search_engine
            .runtime
            .block_on(search_engine.index.search(
                recorded_search,
                &strategy,
                &DefaultContext,
                query_vector.as_slice(),
                &mut result_output_buffer,
            ))
            .unwrap();

        let ids = search_record
            .visited
            .iter()
            .map(|n| n.id)
            .collect::<Vec<_>>();

        const EXPECTED_NODES: [u32; 18] = [
            72, 118, 108, 86, 84, 152, 170, 82, 114, 87, 207, 176, 79, 153, 67, 165, 141, 180,
        ]; //Expected nodes for query = [1f32; 128] with beam_width=4

        assert_eq!(ids, &EXPECTED_NODES);

        let return_list_size = 10;
        let search_list_size = 10;
        let result = search_engine.search(
            &query_vector,
            return_list_size,
            search_list_size,
            Some(4),
            SearchMode::graph(),
        );
        assert!(result.is_ok(), "Expected search to succeed");
        let search_result = result.unwrap();
        assert_eq!(
            search_result.results.len() as u32,
            return_list_size,
            "Expected result count to match"
        );
        assert_eq!(
            indices,
            vec![152, 72, 170, 118, 87, 165, 79, 141, 108, 86],
            "Expected indices to match"
        );
    }

    #[test]
    fn test_disk_search_determinant_diversity() {
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));
        let search_engine = create_disk_index_searcher::<GraphDataF32VectorUnitData>(
            CreateDiskIndexSearcherParams {
                max_thread_num: 1,
                pq_pivot_file_path: TEST_PQ_PIVOT,
                pq_compressed_file_path: TEST_PQ_COMPRESSED,
                index_path: TEST_INDEX,
                index_path_prefix: TEST_INDEX_PREFIX,
                ..Default::default()
            },
            &storage_provider,
        );

        let query_vector: [f32; 128] = [1f32; 128];
        let return_list_size = 10u32;
        let search_list_size = 20u32;

        // Baseline: no post-processor. Det-div selects from the same L=20 candidate pool,
        // so all det-div IDs must be a subset of the baseline candidates.
        let baseline = search_engine
            .search(
                &query_vector,
                search_list_size,
                search_list_size,
                Some(4),
                SearchMode::graph(),
            )
            .unwrap();
        let baseline_ids: std::collections::HashSet<u32> =
            baseline.results.iter().map(|r| r.vertex_id).collect();
        let baseline_top1 = baseline
            .results
            .first()
            .expect("baseline returned no results");

        // Run with determinant-diversity post-processor (default-ish params).
        let params = DeterminantDiversityParams::new(2.0, 0.01).unwrap();
        let result = search_engine
            .search(
                &query_vector,
                return_list_size,
                search_list_size,
                Some(4),
                SearchMode::diverse_graph(params),
            )
            .unwrap();
        let det_div_ids: Vec<u32> = result.results.iter().map(|r| r.vertex_id).collect();

        assert_eq!(
            det_div_ids.len(),
            return_list_size as usize,
            "det-div should return k results when the candidate pool is large enough"
        );
        for id in &det_div_ids {
            assert!(
                baseline_ids.contains(id),
                "det-div selected id {} that is not in the search candidate pool",
                id
            );
        }

        let mut unique = std::collections::HashSet::new();
        for id in &det_div_ids {
            assert!(unique.insert(*id), "det-div produced duplicate id {}", id);
        }

        // Greedy det-div with power > 0 and eta > 0 selects the highest-similarity
        // candidate first.
        assert_eq!(
            result.results[0].vertex_id, baseline_top1.vertex_id,
            "det-div top-1 should be the nearest neighbor (highest similarity)"
        );

        // Pure greedy orthogonalization (eta == 0) should also produce a valid subset.
        let pure_params = DeterminantDiversityParams::new(2.0, 0.0).unwrap();
        let pure_result = search_engine
            .search(
                &query_vector,
                return_list_size,
                search_list_size,
                Some(4),
                SearchMode::diverse_graph(pure_params),
            )
            .unwrap();
        let pure_ids: Vec<u32> = pure_result.results.iter().map(|r| r.vertex_id).collect();
        for id in &pure_ids {
            assert!(
                baseline_ids.contains(id),
                "det-div(eta=0) selected id {} that is not in the search candidate pool",
                id
            );
        }

        // The vector_filter is honored by det-div: filter out the baseline top-1 and
        // verify it is excluded from the det-div results.
        let excluded = baseline_top1.vertex_id;
        let filtered = search_engine
            .search(
                &query_vector,
                return_list_size,
                search_list_size,
                Some(4),
                SearchMode::diverse_graph_filtered(move |id: &u32| *id != excluded, params),
            )
            .unwrap();
        let filtered_ids: Vec<u32> = filtered.results.iter().map(|r| r.vertex_id).collect();
        assert!(
            !filtered_ids.contains(&excluded),
            "det-div results must respect the vector filter"
        );
    }

    #[cfg(feature = "experimental_diversity_search")]
    #[test]
    fn test_disk_search_diversity_search() {
        use diskann::graph::DiverseSearchParams;
        use diskann::neighbor::AttributeValueProvider;
        use std::collections::HashMap;

        // Simple test attribute provider
        #[derive(Debug, Clone)]
        struct TestAttributeProvider {
            attributes: HashMap<u32, u32>,
        }
        impl TestAttributeProvider {
            fn new() -> Self {
                Self {
                    attributes: HashMap::new(),
                }
            }
            fn insert(&mut self, id: u32, attribute: u32) {
                self.attributes.insert(id, attribute);
            }
        }
        impl diskann::provider::HasId for TestAttributeProvider {
            type Id = u32;
        }

        impl AttributeValueProvider for TestAttributeProvider {
            type Value = u32;

            fn get(&self, id: Self::Id) -> Option<Self::Value> {
                self.attributes.get(&id).copied()
            }
        }

        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));

        let search_engine = create_disk_index_searcher::<GraphDataF32VectorUnitData>(
            CreateDiskIndexSearcherParams {
                max_thread_num: 1,
                pq_pivot_file_path: TEST_PQ_PIVOT,
                pq_compressed_file_path: TEST_PQ_COMPRESSED,
                index_path: TEST_INDEX,
                index_path_prefix: TEST_INDEX_PREFIX,
                ..Default::default()
            },
            &storage_provider,
        );

        let query_vector: [f32; 128] = [1f32; 128];

        // Create attribute provider with random labels (1 to 3) for all vectors
        let mut attribute_provider = TestAttributeProvider::new();
        let num_vectors = 256; // Number of vectors in the test dataset
        for i in 0..num_vectors {
            // Assign labels 1-3 based on modulo to ensure distribution
            let label = (i % 15) + 1;
            attribute_provider.insert(i, label);
        }
        // Wrap in Arc once to avoid cloning the HashMap later
        let attribute_provider = std::sync::Arc::new(attribute_provider);

        let mut indices = vec![0u32; 10];
        let mut distances = vec![0f32; 10];
        let mut associated_data = vec![(); 10];

        let mut result_output_buffer = search_output_buffer::IdDistanceAssociatedData::new(
            &mut indices,
            &mut distances,
            &mut associated_data,
        );
        let io_tracker = IOTracker::default();
        let strategy = search_engine.search_strategy(&io_tracker, PostprocessStrategy::AcceptAll);

        // Create diverse search parameters with attribute provider
        let diverse_params = DiverseSearchParams::new(
            0, // diverse_attribute_id
            3, // diverse_results_k
            attribute_provider.clone(),
        );

        let search_params = Knn::new(10, 20, None).unwrap();

        let diverse_search = diskann::graph::search::Diverse::new(search_params, diverse_params);
        let stats = search_engine
            .runtime
            .block_on(search_engine.index.search(
                diverse_search,
                &strategy,
                &DefaultContext,
                query_vector.as_slice(),
                &mut result_output_buffer,
            ))
            .unwrap();

        // Verify that search was performed and returned some results
        assert!(
            stats.result_count > 0,
            "Expected to get some results during diversity search"
        );

        let return_list_size = 10;
        let search_list_size = 20;
        let diverse_results_k = 1;
        let diverse_params = DiverseSearchParams::new(
            0, // diverse_attribute_id
            diverse_results_k,
            attribute_provider.clone(),
        );

        // Test diverse search using the search API
        let mut indices2 = vec![0u32; return_list_size as usize];
        let mut distances2 = vec![0f32; return_list_size as usize];
        let mut associated_data2 = vec![(); return_list_size as usize];
        let mut result_output_buffer2 = search_output_buffer::IdDistanceAssociatedData::new(
            &mut indices2,
            &mut distances2,
            &mut associated_data2,
        );
        let io_tracker2 = IOTracker::default();
        let strategy2 = search_engine.search_strategy(&io_tracker2, PostprocessStrategy::AcceptAll);
        let search_params2 =
            Knn::new(return_list_size as usize, search_list_size as usize, None).unwrap();

        let diverse_search2 = diskann::graph::search::Diverse::new(search_params2, diverse_params);
        let stats = search_engine
            .runtime
            .block_on(search_engine.index.search(
                diverse_search2,
                &strategy2,
                &DefaultContext,
                query_vector.as_slice(),
                &mut result_output_buffer2,
            ))
            .unwrap();

        // Verify results
        assert!(
            stats.result_count > 0,
            "Expected diversity search to return results"
        );
        assert!(
            stats.result_count <= return_list_size,
            "Expected result count to be <= {}",
            return_list_size
        );

        // Verify that we got some results
        assert!(
            stats.result_count > 0,
            "Expected to get some search results"
        );

        // Print search results with their attributes
        println!("\n=== Diversity Search Results ===");
        println!("Query: [1f32; 128]");
        println!("diverse_results_k: {}", diverse_results_k);
        println!("Total results: {}\n", stats.result_count);
        println!("{:<10} {:<15} {:<10}", "Vertex ID", "Distance", "Label");
        println!("{}", "-".repeat(35));
        for i in 0..stats.result_count as usize {
            let attribute_value = attribute_provider.get(indices2[i]).unwrap_or(0);
            println!(
                "{:<10} {:<15.2} {:<10}",
                indices2[i], distances2[i], attribute_value
            );
        }

        // Verify that distances are non-negative and sorted
        for i in 0..(stats.result_count as usize).saturating_sub(1) {
            assert!(distances2[i] >= 0.0, "Expected non-negative distance");
            assert!(
                distances2[i] <= distances2[i + 1],
                "Expected distances to be sorted in ascending order"
            );
        }

        // Verify diversity: Check that we have diverse attribute values in the results
        let mut attribute_counts = HashMap::new();
        for item in indices2.iter().take(stats.result_count as usize) {
            if let Some(attribute_value) = attribute_provider.get(*item) {
                *attribute_counts.entry(attribute_value).or_insert(0) += 1;
            }
        }

        // Print attribute distribution
        println!("\n=== Attribute Distribution ===");
        let mut sorted_attrs: Vec<_> = attribute_counts.iter().collect();
        sorted_attrs.sort_by_key(|(k, _)| *k);
        for (attribute_value, count) in &sorted_attrs {
            println!(
                "Label {}: {} occurrences (max allowed: {})",
                attribute_value, count, diverse_results_k
            );
        }
        println!("Total unique labels: {}", attribute_counts.len());
        println!("================================\n");

        // With diverse_results_k = 5, we expect at most 5 results per attribute value
        for (attribute_value, count) in &attribute_counts {
            println!(
                "Assert: Label {} has {} occurrences (max: {})",
                attribute_value, count, diverse_results_k
            );
            assert!(
                *count <= diverse_results_k,
                "Attribute value {} appears {} times, which exceeds diverse_results_k of {}",
                attribute_value,
                count,
                diverse_results_k
            );
        }

        // Verify that we have multiple different attribute values (diversity)
        // With 3 possible labels and diverse_results_k=5, we should see at least 2 different labels
        println!(
            "Assert: Found {} unique labels (expected at least 2)",
            attribute_counts.len()
        );
        assert!(
            attribute_counts.len() >= 2,
            "Expected at least 2 different attribute values for diversity, got {}",
            attribute_counts.len()
        );
    }

    #[rstest]
    // This case checks expected behavior of unfiltered search.
    #[case(
        |_id: &u32| true,
        false,
        10,
        vec![152, 118, 72, 170, 87, 141, 79, 207, 124, 86],
        vec![256101.7, 256675.3, 256709.69, 256712.5, 256760.08, 256958.5, 257006.1, 257025.7, 257105.67, 257107.67],
    )]
    // This case validates post-filtering using 2 ids which are not present in the unfiltered result set.
    // It is expected that the post-filtering will return an empty result
    #[case(
        |id: &u32| *id == 0 || *id == 1,
        false,
        0,
        vec![0; 10],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )]
    // This case validates pre-filtering using 2 ids which are not present in the unfiltered result set.
    // It is expected that the pre-filtering will do search over matching ids
    #[case(
        |id: &u32| *id == 0 || *id == 1,
        true,
        2,
        vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![257247.28, 258179.28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )]
    // This case validates post-filtering using 3 ids from the unfiltered result set.
    // It is expected that the post-filtering will filter out non-matching ids
    #[case(
        |id: &u32| *id == 72 || *id == 87 || *id == 170,
        false,
        3,
        vec![72, 170, 87, 0, 0, 0, 0, 0, 0, 0],
        vec![256709.69, 256712.5, 256760.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )]
    // This case validates pre-filtering using 3 ids from the unfiltered result set.
    // It is expected that the pre-filtering will do search over matching ids
    #[case(
        |id: &u32| *id == 72 || *id == 87 || *id == 170,
        true,
        3,
        vec![72, 170, 87, 0, 0, 0, 0, 0, 0, 0],
        vec![256709.69, 256712.5, 256760.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )]
    fn test_search_with_vector_filter(
        #[case] vector_filter: fn(&u32) -> bool,
        #[case] is_flat_search: bool,
        #[case] expected_result_count: u32,
        #[case] expected_indices: Vec<u32>,
        #[case] expected_distances: Vec<f32>,
    ) {
        // Exact distances can vary slightly depending on the architecture used
        // to compute distances due to different unrolling strategies and SIMD widthd.
        //
        // This parameter allows for a small margin when matching distances.
        let check_distances = |got: &[f32], expected: &[f32]| -> bool {
            const ABS_TOLERANCE: f32 = 0.02;
            assert_eq!(got.len(), expected.len());
            for (i, (g, e)) in std::iter::zip(got.iter(), expected.iter()).enumerate() {
                if (g - e).abs() > ABS_TOLERANCE {
                    panic!(
                        "distances differ at position {} by more than {}\n\n\
                         got: {:?}\nexpected: {:?}",
                        i, ABS_TOLERANCE, got, expected,
                    );
                }
            }
            true
        };

        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));

        let search_engine = create_disk_index_searcher::<GraphDataF32VectorUnitData>(
            CreateDiskIndexSearcherParams {
                max_thread_num: 5,
                pq_pivot_file_path: TEST_PQ_PIVOT_128DIM,
                pq_compressed_file_path: TEST_PQ_COMPRESSED_128DIM,
                index_path: TEST_INDEX_128DIM,
                index_path_prefix: TEST_INDEX_PREFIX_128DIM,
                ..Default::default()
            },
            &storage_provider,
        );
        let query = vec![0.1f32; 128];
        let mut query_stats = QueryStatistics::default();
        let mut indices = vec![0u32; 10];
        let mut distances = vec![0f32; 10];
        let mut associated_data = vec![(); 10];

        // Build the same `SearchMode` twice. `vector_filter` is a `fn` pointer
        // (Copy), so each call reconstructs a fresh mode with the same filter.
        let make_mode = || -> SearchMode<'static> {
            if is_flat_search {
                SearchMode::flat_filtered(vector_filter)
            } else {
                SearchMode::graph_filtered(vector_filter)
            }
        };

        let result = search_engine.search_internal(
            &query,
            10,
            10,
            None, // beam_width
            &mut query_stats,
            &mut indices,
            &mut distances,
            &mut associated_data,
            &make_mode(),
        );

        assert!(result.is_ok(), "Expected search to succeed");
        assert_eq!(
            result.unwrap().result_count,
            expected_result_count,
            "Expected result count to match"
        );
        assert_eq!(indices, expected_indices, "Expected indices to match");
        assert!(
            check_distances(&distances, &expected_distances),
            "Expected distances to match"
        );

        let result_with_filter = search_engine.search(
            &query,
            10,
            10,
            None, // beam_width
            make_mode(),
        );

        assert!(result_with_filter.is_ok(), "Expected search to succeed");
        let result_with_filter_unwrapped = result_with_filter.unwrap();
        assert_eq!(
            result_with_filter_unwrapped.stats.result_count, expected_result_count,
            "Expected result count to match"
        );
        let actual_indices = result_with_filter_unwrapped
            .results
            .iter()
            .map(|x| x.vertex_id)
            .collect::<Vec<_>>();
        assert_eq!(
            actual_indices, expected_indices,
            "Expected indices to match"
        );
        let actual_distances = result_with_filter_unwrapped
            .results
            .iter()
            .map(|x| x.distance)
            .collect::<Vec<_>>();
        assert!(
            check_distances(&actual_distances, &expected_distances),
            "Expected distances to match"
        );
    }

    // ===========================================================================
    // Inline filter + AdaptiveL behavioral tests
    // ===========================================================================
    //
    // Two basic invariants from the design review:
    //
    // 1. `adaptive_l = Some(_)` with an always-true predicate visits every
    //    candidate as a "match," computes specificity = 100%, never triggers
    //    a resize, and produces the same top-k as plain `Knn`. This is the
    //    "no-op equivalence" guard.
    //
    // 2. `adaptive_l = Some(_)` with a selective predicate must produce a
    //    valid result set whose IDs all satisfy the predicate. Doesn't assert
    //    recall@k (would need filter-selective ground truth) — just that the
    //    inline path runs end-to-end and produces filter-conforming output.

    #[test]
    fn test_adaptive_l_with_no_filter_matches_plain_knn() {
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));
        let search_engine = create_disk_index_searcher::<GraphDataF32VectorUnitData>(
            CreateDiskIndexSearcherParams {
                max_thread_num: 1,
                pq_pivot_file_path: TEST_PQ_PIVOT_128DIM,
                pq_compressed_file_path: TEST_PQ_COMPRESSED_128DIM,
                index_path: TEST_INDEX_128DIM,
                index_path_prefix: TEST_INDEX_PREFIX_128DIM,
                ..Default::default()
            },
            &storage_provider,
        );
        let query = vec![0.1f32; 128];

        let plain = search_engine
            .search(&query, 10, 10, None, SearchMode::graph())
            .expect("plain Knn must succeed");

        let inline_no_filter = search_engine
            .search(
                &query,
                10,
                10,
                None,
                SearchMode::inline_filter(
                    |_| true,
                    Some(AdaptiveL::new(5, 16.0).expect("valid AdaptiveL")),
                ),
            )
            .expect("inline filter with accept-all predicate must succeed");

        let plain_ids: Vec<u32> = plain.results.iter().map(|r| r.vertex_id).collect();
        let inline_ids: Vec<u32> = inline_no_filter
            .results
            .iter()
            .map(|r| r.vertex_id)
            .collect();

        assert_eq!(
            plain.stats.result_count, inline_no_filter.stats.result_count,
            "no-filter inline path must return same result count as plain Knn"
        );
        assert_eq!(
            plain_ids, inline_ids,
            "no-filter inline path must return the same top-k IDs as plain Knn"
        );
    }

    #[test]
    fn test_adaptive_l_with_selective_predicate_returns_only_matches() {
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));
        let search_engine = create_disk_index_searcher::<GraphDataF32VectorUnitData>(
            CreateDiskIndexSearcherParams {
                max_thread_num: 1,
                pq_pivot_file_path: TEST_PQ_PIVOT_128DIM,
                pq_compressed_file_path: TEST_PQ_COMPRESSED_128DIM,
                index_path: TEST_INDEX_128DIM,
                index_path_prefix: TEST_INDEX_PREFIX_128DIM,
                ..Default::default()
            },
            &storage_provider,
        );
        let query = vec![0.1f32; 128];
        // Predicate from `test_search_with_vector_filter::case_4` — three IDs
        // known to be in the unfiltered top-10 for this query+fixture.
        let predicate = |id: &u32| *id == 72 || *id == 87 || *id == 170;

        let result = search_engine
            .search(
                &query,
                10,
                10,
                None,
                SearchMode::inline_filter(
                    predicate,
                    Some(AdaptiveL::new(5, 16.0).expect("valid AdaptiveL")),
                ),
            )
            .expect("inline filter search with AdaptiveL must succeed");

        // `result.results` is pre-allocated to `return_list_size`; only the
        // first `result_count` entries are populated. The trailing entries
        // are default zeros — not search output — so slice before asserting.
        let count = result.stats.result_count as usize;
        let ids: Vec<u32> = result
            .results
            .iter()
            .take(count)
            .map(|r| r.vertex_id)
            .collect();
        for id in &ids {
            assert!(
                predicate(id),
                "AdaptiveL result must only contain predicate-matching IDs; got {id} in {ids:?}"
            );
        }
        assert!(
            !ids.is_empty(),
            "AdaptiveL on a fixture with reachable matches must return at least one match"
        );
    }

    #[test]
    fn test_beam_search_respects_io_limit() {
        let io_limit = 11; // Set a small IO limit for testing
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));

        let search_engine = create_disk_index_searcher::<GraphDataF32VectorUnitData>(
            CreateDiskIndexSearcherParams {
                max_thread_num: 1,
                pq_pivot_file_path: TEST_PQ_PIVOT,
                pq_compressed_file_path: TEST_PQ_COMPRESSED,
                index_path: TEST_INDEX,
                index_path_prefix: TEST_INDEX_PREFIX,
                io_limit,
                ..Default::default()
            },
            &storage_provider,
        );
        let query_vector: [f32; 128] = [1f32; 128];

        let mut indices = vec![0u32; 10];
        let mut distances = vec![0f32; 10];
        let mut associated_data = vec![(); 10];

        let mut result_output_buffer = search_output_buffer::IdDistanceAssociatedData::new(
            &mut indices,
            &mut distances,
            &mut associated_data,
        );

        let io_tracker = IOTracker::default();
        let strategy = search_engine.search_strategy(&io_tracker, PostprocessStrategy::AcceptAll);

        let mut search_record = VisitedSearchRecord::new(0);
        let search_params = Knn::new(10, 10, Some(4)).unwrap();
        let recorded_search =
            diskann::graph::search::RecordedKnn::new(search_params, &mut search_record);
        search_engine
            .runtime
            .block_on(search_engine.index.search(
                recorded_search,
                &strategy,
                &DefaultContext,
                query_vector.as_slice(),
                &mut result_output_buffer,
            ))
            .unwrap();
        let visited_ids = search_record
            .visited
            .iter()
            .map(|n| n.id)
            .collect::<Vec<_>>();

        let query_stats = strategy.io_tracker;
        //Verify the IO limit was respected
        assert!(
            query_stats.io_count() <= io_limit,
            "Expected IO operations to be <= {}, but got {}",
            io_limit,
            query_stats.io_count()
        );

        const EXPECTED_NODES: [u32; 17] = [
            72, 118, 108, 86, 84, 152, 170, 82, 114, 87, 207, 176, 79, 153, 67, 165, 141,
        ]; //Expected nodes for query = [1f32; 128] with beam_width=4

        // Count matching results
        let mut matching_count = 0;
        for expected_node in EXPECTED_NODES.iter() {
            if visited_ids.contains(expected_node) {
                matching_count += 1;
            }
        }

        // Calculate recall
        let recall = (matching_count as f32 / EXPECTED_NODES.len() as f32) * 100.0;

        //Verify the recall is above 60%. The threshold her eis arbitrary, just to make sure when
        // search hits io_limit that it doesn't break and the recall degrades gracefully
        assert!(recall >= 60.0, "Match percentage is below 60%: {}", recall);
    }
}
