/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::HashMap,
    future::Future,
    num::NonZeroUsize,
    ops::Range,
    sync::{
        atomic::{AtomicU64, AtomicUsize},
        Arc,
    },
    time::Instant,
};

use diskann::{
    graph::{
        self,
        glue::{self, ExpandBeam, IdIterator, SearchExt, SearchPostProcess, SearchStrategy},
        search_output_buffer, AdjacencyList, DiskANNIndex, SearchOutputBuffer, SearchParams,
    },
    neighbor::Neighbor,
    provider::{
        Accessor, BuildQueryComputer, DataProvider, DefaultContext, DelegateNeighbor, HasId,
        NeighborAccessor,
    },
    utils::{
        object_pool::{ObjectPool, PoolOption, TryAsPooled},
        IntoUsize, VectorRepr,
    },
    ANNError, ANNResult,
};
use diskann_providers::storage::StorageReadProvider;
use diskann_providers::{
    model::{
        compute_pq_distance, compute_pq_distance_for_pq_coordinates, graph::traits::GraphDataType,
        pq::quantizer_preprocess, PQData, PQScratch,
    },
    storage::{get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file, LoadWith},
};
use diskann_vector::{distance::Metric, DistanceFunction, PreprocessedDistanceFunction};
use futures_util::future;
use tokio::runtime::Runtime;
use tracing::debug;

use crate::{
    data_model::{CachingStrategy, GraphHeader},
    filter_parameter::{default_vector_filter, VectorFilter},
    search::{
        provider::disk_vertex_provider_factory::DiskVertexProviderFactory,
        traits::{VertexProvider, VertexProviderFactory},
    },
    storage::{api::AsyncDiskLoadContext, disk_index_reader::DiskIndexReader},
    utils::AlignedFileReaderFactory,
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
            let aligned_reader_factory = AlignedFileReaderFactory::new(get_disk_index_file(
                ctx.quant_load_context.metadata.prefix(),
            ));

            let caching_strategy = if ctx.num_nodes_to_cache > 0 {
                CachingStrategy::StaticCacheWithBfsNodes(ctx.num_nodes_to_cache)
            } else {
                CachingStrategy::None
            };

            let vertex_provider_factory = DiskVertexProviderFactory::<Data, _>::new(
                aligned_reader_factory,
                caching_strategy,
            )?;
            VertexProviderFactory::get_header(&vertex_provider_factory)?
        };

        let metric = ctx.quant_load_context.metric;
        let num_points = ctx.num_points;

        let index_path_prefix = ctx.quant_load_context.metadata.prefix();
        let index_reader = DiskIndexReader::<<Data as GraphDataType>::VectorDataType>::new(
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
        disk_index_reader: &DiskIndexReader<Data::VectorDataType>,
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
pub struct DiskSearchStrategy<'a, Data, ProviderFactory>
where
    Data: GraphDataType<VectorIdType = u32>,
    ProviderFactory: VertexProviderFactory<Data>,
{
    // This needs to be Arc instead of Rc because DiskSearchStrategy has "Send" trait bound, though this is not expected to be shared across threads.
    io_tracker: IOTracker,
    vector_filter: &'a (dyn Fn(&u32) -> bool + Send + Sync), // Fn param is u32 as we validate "VectorIdType = u32" everywhere in this provider in trait bounds.
    query: &'a [Data::VectorDataType],

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
    filter: &'a (dyn Fn(&u32) -> bool + Send + Sync),
}

impl<'a> RerankAndFilter<'a> {
    fn new(filter: &'a (dyn Fn(&u32) -> bool + Send + Sync)) -> Self {
        Self { filter }
    }
}

impl<Data, VP>
    SearchPostProcess<
        DiskAccessor<'_, Data, VP>,
        [Data::VectorDataType],
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
        _computer: &DiskQueryComputer,
        candidates: I,
        output: &mut B,
    ) -> Result<usize, Self::Error>
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: SearchOutputBuffer<(u32, Data::AssociatedDataType)> + Send + ?Sized,
    {
        let provider = accessor.provider;

        let mut uncached_ids = Vec::new();
        let mut reranked = candidates
            .map(|n| n.id)
            .filter(|id| (self.filter)(id))
            .filter_map(|n| {
                if let Some(entry) = accessor.scratch.distance_cache.get(&n) {
                    Some(Ok::<((u32, _), f32), ANNError>(((n, entry.1), entry.0)))
                } else {
                    uncached_ids.push(n);
                    None
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
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

impl<'this, Data, ProviderFactory>
    SearchStrategy<
        DiskProvider<Data>,
        [Data::VectorDataType],
        (
            <DiskProvider<Data> as DataProvider>::InternalId,
            Data::AssociatedDataType,
        ),
    > for DiskSearchStrategy<'this, Data, ProviderFactory>
where
    Data: GraphDataType<VectorIdType = u32>,
    ProviderFactory: VertexProviderFactory<Data>,
{
    type QueryComputer = DiskQueryComputer;
    type SearchAccessor<'a> = DiskAccessor<'a, Data, ProviderFactory::VertexProviderType>;
    type SearchAccessorError = ANNError;
    type PostProcessor = RerankAndFilter<'this>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a DiskProvider<Data>,
        _context: &DefaultContext,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        DiskAccessor::new(
            provider,
            &self.io_tracker,
            self.query,
            self.vertex_provider_factory,
            self.scratch_pool,
        )
    }

    fn post_processor(&self) -> Self::PostProcessor {
        RerankAndFilter::new(self.vector_filter)
    }
}

/// The query computer for the disk provider. This is used to compute the distance between the query vector and the PQ coordinates.
pub struct DiskQueryComputer {
    num_pq_chunks: usize,
    query_centroid_l2_distance: Vec<f32>,
}

impl PreprocessedDistanceFunction<&[u8], f32> for DiskQueryComputer {
    fn evaluate_similarity(&self, changing: &[u8]) -> f32 {
        let mut dist = 0.0f32;
        #[allow(clippy::expect_used)]
        compute_pq_distance_for_pq_coordinates(
            changing,
            self.num_pq_chunks,
            &self.query_centroid_l2_distance,
            std::slice::from_mut(&mut dist),
        )
        .expect("PQ distance compute for PQ coordinates is expected to succeed");
        dist
    }
}

impl<Data, VP> BuildQueryComputer<[Data::VectorDataType]> for DiskAccessor<'_, Data, VP>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    type QueryComputerError = ANNError;
    type QueryComputer = DiskQueryComputer;

    fn build_query_computer(
        &self,
        _from: &[Data::VectorDataType],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        Ok(DiskQueryComputer {
            num_pq_chunks: self.provider.pq_data.get_num_chunks(),
            query_centroid_l2_distance: self
                .scratch
                .pq_scratch
                .aligned_pqtable_dist_scratch
                .as_slice()
                .to_vec(),
        })
    }

    async fn distances_unordered<Itr, F>(
        &mut self,
        vec_id_itr: Itr,
        _computer: &Self::QueryComputer,
        f: F,
    ) -> Result<(), Self::GetError>
    where
        F: Send + FnMut(f32, Self::Id),
        Itr: Iterator<Item = Self::Id>,
    {
        self.pq_distances(&vec_id_itr.collect::<Box<[_]>>(), f)
    }
}

impl<Data, VP> ExpandBeam<[Data::VectorDataType]> for DiskAccessor<'_, Data, VP>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        _computer: &Self::QueryComputer,
        mut pred: P,
        mut f: F,
    ) -> impl std::future::Future<Output = Result<(), Self::GetError>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(f32, Self::Id) + Send,
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

                self.pq_distances(&ids, &mut f)?;
            }

            Ok(())
        })();

        std::future::ready(result)
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
    dim: usize,
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
            args.dim,
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
            self.provider.pq_data.pq_compressed_data().get_data(),
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

impl<Data, VP> SearchExt for DiskAccessor<'_, Data, VP>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    async fn starting_points(&self) -> ANNResult<Vec<u32>> {
        let start_vertex_id = self.provider.graph_header.metadata().medoid as u32;
        Ok(vec![start_vertex_id])
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
                dim: provider.graph_header.metadata().dims,
                num_pq_chunks: provider.pq_data.get_num_chunks(),
                num_pq_centers: provider.pq_data.get_num_centers(),
                vertex_factory: vertex_provider_factory,
                graph_header: &provider.graph_header,
            },
        )?;

        scratch.pq_scratch.set(
            provider.graph_header.metadata().dims,
            query,
            1.0_f32, // Normalization factor
        )?;
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

impl<Data, VP> HasId for DiskAccessor<'_, Data, VP>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    type Id = u32;
}

impl<'a, Data, VP> Accessor for DiskAccessor<'a, Data, VP>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    /// This references the PQ vector in the underlying `pq_data` store.
    type Extended = &'a [u8];

    /// This accessor returns raw slices. There *is* a chance of racing when the fast
    /// providers are used. We just have to live with it.
    ///
    /// Since the underlying PQ store is shared, we ignore the `'b` lifetime here and
    /// instead use `'a`.
    type Element<'b>
        = &'a [u8]
    where
        Self: 'b;

    /// `ElementRef` can have arbitrary lifetimes.
    type ElementRef<'b> = &'b [u8];

    /// Choose to panic on an out-of-bounds access rather than propagate an error.
    type GetError = ANNError;

    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        std::future::ready(self.provider.pq_data.get_compressed_vector(id as usize))
    }
}

impl<Data, VP> IdIterator<Range<u32>> for DiskAccessor<'_, Data, VP>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    async fn id_iterator(&mut self) -> Result<Range<u32>, ANNError> {
        Ok(0..self.provider.num_points as u32)
    }
}

impl<'a, 'b, Data, VP> DelegateNeighbor<'a> for DiskAccessor<'b, Data, VP>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    type Delegate = AsNeighborAccessor<'a, 'b, Data, VP>;
    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        AsNeighborAccessor(self)
    }
}

/// A light-weight wrapper around `&mut DiskAccessor` used to tailor the semantics of
/// [`NeighborAccessor`].
///
/// This implementation ensures that the vector data for adjacency lists is also retrieved
/// and cached to enhance reranking.
pub struct AsNeighborAccessor<'a, 'b, Data, VP>(&'a mut DiskAccessor<'b, Data, VP>)
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>;

impl<Data, VP> HasId for AsNeighborAccessor<'_, '_, Data, VP>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    type Id = u32;
}

impl<Data, VP> NeighborAccessor for AsNeighborAccessor<'_, '_, Data, VP>
where
    Data: GraphDataType<VectorIdType = u32>,
    VP: VertexProvider<Data>,
{
    fn get_neighbors(
        self,
        id: Self::Id,
        neighbors: &mut AdjacencyList<Self::Id>,
    ) -> impl Future<Output = ANNResult<Self>> + Send {
        if self.0.io_tracker.io_count() > self.0.provider.search_io_limit {
            return future::ok(self); // Returning empty results in `neighbors` out param if IO limit is reached.
        }

        if let Err(e) = ensure_vertex_loaded(&mut self.0.scratch.vertex_provider, &[id]) {
            return future::err(e);
        }
        let list = match self.0.scratch.vertex_provider.get_adjacency_list(&id) {
            Ok(list) => list,
            Err(e) => return future::err(e),
        };
        neighbors.overwrite_trusted(list);
        future::ok(self)
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
        disk_index_reader: &DiskIndexReader<Data::VectorDataType>,
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
            dim: graph_header.metadata().dims,
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

    /// Helper method to create a DiskSearchStrategy with common parameters
    fn search_strategy<'a>(
        &'a self,
        query: &'a [Data::VectorDataType],
        vector_filter: &'a (dyn Fn(&Data::VectorIdType) -> bool + Send + Sync),
    ) -> DiskSearchStrategy<'a, Data, ProviderFactory> {
        DiskSearchStrategy {
            io_tracker: IOTracker::default(),
            vector_filter,
            query,
            vertex_provider_factory: &self.vertex_provider_factory,
            scratch_pool: &self.scratch_pool,
        }
    }

    /// Perform a search on the disk index.
    /// return the list of nearest neighbors and associated data.
    pub fn search(
        &self,
        query: &[Data::VectorDataType],
        return_list_size: u32,
        search_list_size: u32,
        beam_width: Option<usize>,
        vector_filter: Option<VectorFilter<Data>>,
        is_flat_search: bool,
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
            &vector_filter.unwrap_or(default_vector_filter::<Data>()),
            is_flat_search,
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
        vector_filter: &(dyn Fn(&Data::VectorIdType) -> bool + Send + Sync),
        is_flat_search: bool,
    ) -> ANNResult<SearchResultStats> {
        let mut result_output_buffer = search_output_buffer::IdDistanceAssociatedData::new(
            &mut indices[..k_value],
            &mut distances[..k_value],
            &mut associated_data[..k_value],
        );

        let strategy = self.search_strategy(query, vector_filter);
        let timer = Instant::now();
        let stats = if is_flat_search {
            self.runtime.block_on(self.index.flat_search(
                &strategy,
                &DefaultContext,
                strategy.query,
                vector_filter,
                &SearchParams::new(k_value, search_list_size as usize, beam_width)?,
                &mut result_output_buffer,
            ))?
        } else {
            self.runtime.block_on(self.index.search(
                &strategy,
                &DefaultContext,
                strategy.query,
                &SearchParams::new(k_value, search_list_size as usize, beam_width)?,
                &mut result_output_buffer,
            ))?
        };
        query_stats.total_comparisons = stats.cmps;
        query_stats.search_hops = stats.hops;

        query_stats.total_execution_time_us = timer.elapsed().as_micros();
        query_stats.io_time_us = IOTracker::time(&strategy.io_tracker.io_time_us) as u128;
        query_stats.total_io_operations = strategy.io_tracker.io_count() as u32;
        query_stats.total_vertices_loaded = strategy.io_tracker.io_count() as u32;
        query_stats.query_pq_preprocess_time_us =
            IOTracker::time(&strategy.io_tracker.preprocess_time_us) as u128;
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
    use diskann::{
        graph::{search::record::VisitedSearchRecord, SearchParamsError},
        utils::IntoUsize,
        ANNErrorKind,
    };
    use diskann_providers::storage::{
        DynWriteProvider, StorageReadProvider, VirtualStorageProvider,
    };
    use diskann_providers::{
        common::AlignedBoxWithSlice,
        test_utils::graph_data_type_utils::{
            GraphDataF32VectorU32Data, GraphDataF32VectorUnitData,
        },
        utils::{
            create_thread_pool, file_util, load_aligned_bin, PQPathNames, ParallelIteratorInPool,
        },
    };
    use diskann_utils::test_data_root;
    use diskann_vector::distance::Metric;
    use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator};
    use rstest::rstest;
    use vfs::OverlayFS;

    use super::*;
    use crate::{
        build::builder::core::disk_index_builder_tests::{IndexBuildFixture, TestParams},
        utils::{QueryStatistics, VirtualAlignedReaderFactory},
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
            dim: 104,
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
            dim: 104,
        });
    }

    #[test]
    fn test_disk_search_k10_l20_single_or_multi_thread_128dim() {
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
        // Test single thread.
        test_disk_search(TestDiskSearchParams {
            storage_provider: storage_provider.as_ref(),
            index_search_engine: &search_engine,
            thread_num: 1,
            query_file_path: TEST_QUERY_10PTS_128DIM,
            truth_result_file_path: TEST_TRUTH_RESULT_10PTS_128DIM,
            k: 10,
            l: 20,
            dim: 128,
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
            dim: 128,
        });
    }

    fn get_truth_associated_data<StorageReader: StorageReadProvider>(
        storage_provider: &StorageReader,
    ) -> Vec<u32> {
        const ASSOCIATED_DATA_FILE: &str = "/sift/siftsmall_learn_256pts_u32_associated_data.fbin";

        let (data, _npts, _dim) =
            file_util::load_bin::<u32, StorageReader>(storage_provider, ASSOCIATED_DATA_FILE, 0)
                .unwrap();
        data
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
                    dim: 128,
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
                    dim: 128,
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

        let disk_index_reader = DiskIndexReader::<Data::VectorDataType>::new(
            params.pq_pivot_file_path.to_string(),
            params.pq_compressed_file_path.to_string(),
            storage_provider.as_ref(),
        )
        .unwrap();

        let aligned_reader_factory = VirtualAlignedReaderFactory::new(
            get_disk_index_file(params.index_path_prefix),
            Arc::clone(storage_provider),
        );
        let caching_strategy = CachingStrategy::None;
        let vertex_provider_factory =
            DiskVertexProviderFactory::<Data, _>::new(aligned_reader_factory, caching_strategy)
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
        let (result, _, _) =
            file_util::load_bin::<u32, StorageReader>(storage_provider, query_result_path, 0)
                .unwrap();
        result
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
        dim: usize,
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
        dim: usize,
    }

    fn test_disk_search<StorageType: StorageReadProvider>(
        params: TestDiskSearchParams<StorageType>,
    ) {
        let query_vector = load_aligned_bin(params.storage_provider, params.query_file_path)
            .unwrap()
            .0;
        let mut aligned_query = AlignedBoxWithSlice::<f32>::new(query_vector.len(), 32).unwrap();
        aligned_query.memcpy(query_vector.as_slice()).unwrap();

        let queries = aligned_query
            .split_into_nonoverlapping_mut_slices(0..aligned_query.len(), params.dim)
            .unwrap();

        let truth_result =
            load_query_result(params.storage_provider, params.truth_result_file_path);

        let pool = create_thread_pool(params.thread_num.into_usize()).unwrap();
        // Convert query_vector to number of Vertex with data type f32 and dimension equals to dim.
        queries
            .par_iter()
            .enumerate()
            .for_each_in_pool(&pool, |(i, query)| {
                // Test search_with_associated_data with an unaligned query. Some distance functions require aligned data.
                let mut aligned_box = AlignedBoxWithSlice::<f32>::new(query.len() + 1, 32).unwrap();
                let mut temp = Vec::with_capacity(query.len() + 1);
                temp.push(0.0);
                temp.extend_from_slice(query);
                aligned_box.memcpy(temp.as_slice()).unwrap();
                let query = &aligned_box.as_slice()[1..];

                let mut query_stats = QueryStatistics::default();
                let mut indices = vec![0u32; 10];
                let mut distances = vec![0f32; 10];
                let mut associated_data = vec![(); 10];

                let result = params
                    .index_search_engine
                    //.search_with_associated_data(query, params.k as u32, params.l as u32)
                    .search_internal(
                        query,
                        params.k,
                        params.l as u32,
                        None, // beam_width
                        &mut query_stats,
                        &mut indices,
                        &mut distances,
                        &mut associated_data,
                        &(|_| true),
                        false,
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
        let query_vector = load_aligned_bin(params.storage_provider, params.query_file_path)
            .unwrap()
            .0;
        let mut aligned_query = AlignedBoxWithSlice::<f32>::new(query_vector.len(), 32).unwrap();
        aligned_query.memcpy(query_vector.as_slice()).unwrap();
        let queries = aligned_query
            .split_into_nonoverlapping_mut_slices(0..aligned_query.len(), params.dim)
            .unwrap();
        let truth_result =
            load_query_result(params.storage_provider, params.truth_result_file_path);
        let pool = create_thread_pool(params.thread_num.into_usize()).unwrap();
        // Convert query_vector to number of Vertex with data type f32 and dimension equals to dim.
        queries
            .par_iter()
            .enumerate()
            .for_each_in_pool(&pool, |(i, query)| {
                // Test search_with_associated_data with an unaligned query. Some distance functions require aligned data.
                let mut aligned_box = AlignedBoxWithSlice::<f32>::new(query.len() + 1, 32).unwrap();
                let mut temp = Vec::with_capacity(query.len() + 1);
                temp.push(0.0);
                temp.extend_from_slice(query);
                aligned_box.memcpy(temp.as_slice()).unwrap();
                let query = &aligned_box.as_slice()[1..];
                let result = params
                    .index_search_engine
                    .search(query, params.k as u32, params.l as u32, beam_width, None, false)
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

        let res = SearchParams::new_default(0, 10);
        assert!(res.is_err());
        assert_eq!(
            <SearchParamsError as std::convert::Into<ANNError>>::into(res.unwrap_err()).kind(),
            ANNErrorKind::IndexError
        );
        let res = SearchParams::new_default(20, 10);
        assert!(res.is_err());
        let res = SearchParams::new_default(10, 0);
        assert!(res.is_err());
        let res = SearchParams::new(10, 10, Some(0));
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
            &|_| true,
            false,
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
        let strategy = search_engine.search_strategy(&query_vector, &|_| true);
        let mut search_record = VisitedSearchRecord::new(0);
        search_engine
            .runtime
            .block_on(search_engine.index.search_recorded(
                &strategy,
                &DefaultContext,
                &query_vector,
                &SearchParams::new(10, 10, Some(4)).unwrap(),
                &mut result_output_buffer,
                &mut search_record,
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
            None,
            false,
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
        let strategy = search_engine.search_strategy(&query_vector, &|_| true);
        let mut search_record = VisitedSearchRecord::new(0);

        // Create diverse search parameters with attribute provider
        let diverse_params = DiverseSearchParams::new(
            0, // diverse_attribute_id
            3, // diverse_results_k
            attribute_provider.clone(),
        );

        let search_params = SearchParams::new(10, 20, None).unwrap();

        search_engine
            .runtime
            .block_on(search_engine.index.diverse_search_experimental(
                &strategy,
                &DefaultContext,
                &query_vector,
                &search_params,
                &diverse_params,
                &mut result_output_buffer,
                &mut search_record,
            ))
            .unwrap();

        let ids = search_record
            .visited
            .iter()
            .map(|n| n.id)
            .collect::<Vec<_>>();

        // Verify that search was performed and visited some nodes
        assert!(
            !ids.is_empty(),
            "Expected to visit some nodes during diversity search"
        );

        let return_list_size = 10;
        let search_list_size = 20;
        let diverse_results_k = 1;
        let diverse_params = DiverseSearchParams::new(
            0, // diverse_attribute_id
            diverse_results_k,
            attribute_provider.clone(),
        );

        // Test diverse search using the experimental API
        let mut indices2 = vec![0u32; return_list_size as usize];
        let mut distances2 = vec![0f32; return_list_size as usize];
        let mut associated_data2 = vec![(); return_list_size as usize];
        let mut result_output_buffer2 = search_output_buffer::IdDistanceAssociatedData::new(
            &mut indices2,
            &mut distances2,
            &mut associated_data2,
        );
        let strategy2 = search_engine.search_strategy(&query_vector, &|_| true);
        let mut search_record2 = VisitedSearchRecord::new(0);
        let search_params2 =
            SearchParams::new(return_list_size as usize, search_list_size as usize, None).unwrap();

        let stats = search_engine
            .runtime
            .block_on(search_engine.index.diverse_search_experimental(
                &strategy2,
                &DefaultContext,
                &query_vector,
                &search_params2,
                &diverse_params,
                &mut result_output_buffer2,
                &mut search_record2,
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

        let result = search_engine.search_internal(
            &query,
            10,
            10,
            None, // beam_width
            &mut query_stats,
            &mut indices,
            &mut distances,
            &mut associated_data,
            &vector_filter,
            is_flat_search,
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
            Some(Box::new(vector_filter)),
            is_flat_search,
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

        let strategy = search_engine.search_strategy(&query_vector, &|_| true);

        let mut search_record = VisitedSearchRecord::new(0);
        search_engine
            .runtime
            .block_on(search_engine.index.search_recorded(
                &strategy,
                &DefaultContext,
                &query_vector,
                &SearchParams::new(10, 10, Some(4)).unwrap(),
                &mut result_output_buffer,
                &mut search_record,
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

    #[test]
    #[cfg(target_os = "linux")]
    fn test_pipe_search_k10_l100_128dim() {
        use crate::search::pipelined::{PipelinedSearcher, PipelinedReaderConfig};
        use diskann_providers::storage::get_disk_index_file;

        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));

        // Load PQ data via DiskIndexReader.
        let disk_index_reader = DiskIndexReader::<f32>::new(
            TEST_PQ_PIVOT_128DIM.to_string(),
            TEST_PQ_COMPRESSED_128DIM.to_string(),
            storage_provider.as_ref(),
        )
        .unwrap();
        let pq_data = disk_index_reader.get_pq_data();

        // Read graph header via DiskVertexProviderFactory.
        let aligned_reader_factory = VirtualAlignedReaderFactory::new(
            get_disk_index_file(TEST_INDEX_PREFIX_128DIM),
            Arc::clone(&storage_provider),
        );
        let vertex_provider_factory =
            DiskVertexProviderFactory::<GraphDataF32VectorUnitData, _>::new(
                aligned_reader_factory,
                CachingStrategy::None,
            )
            .unwrap();
        let graph_header = vertex_provider_factory.get_header().unwrap();

        // Resolve real filesystem path for PipelinedSearcher (needs O_DIRECT).
        let real_index_path = test_data_root().join(
            "disk_index_search/disk_index_sift_learn_R4_L50_A1.2_truth_search_disk.index",
        );
        let real_index_path_str = real_index_path.to_str().unwrap();

        let pipe_searcher = PipelinedSearcher::<GraphDataF32VectorUnitData>::new(
            graph_header,
            pq_data,
            Metric::L2,
            4,
            None,
            real_index_path_str.to_string(),
            PipelinedReaderConfig::default(),
        )
        .unwrap();

        // Load queries and ground truth.
        let (query_vector, _, _) = diskann_providers::utils::file_util::load_bin::<f32, _>(
            storage_provider.as_ref(),
            TEST_QUERY_10PTS_128DIM,
            0,
        )
        .unwrap();
        let truth_result =
            load_query_result(storage_provider.as_ref(), TEST_TRUTH_RESULT_10PTS_128DIM);

        let dim = 128usize;
        let k = 10usize;
        let l = 100u32;
        let num_queries = query_vector.len() / dim;

        let mut total_recall = 0.0f32;
        for q in 0..num_queries {
            let query = &query_vector[q * dim..(q + 1) * dim];
            let result = pipe_searcher.search(query, k as u32, l, 4).unwrap();
            let indices: Vec<u32> = result.results.iter().map(|item| item.vertex_id).collect();
            let truth_slice = &truth_result[q * k..(q + 1) * k];

            // Count recall overlap (PipeANN traversal order may differ from beam search ground truth).
            let matching = indices
                .iter()
                .filter(|id| truth_slice.contains(id))
                .count();
            total_recall += matching as f32 / k as f32;
        }
        let avg_recall = total_recall / num_queries as f32;
        assert!(
            avg_recall >= 0.8,
            "PipeANN average recall {:.0}% < 80%",
            avg_recall * 100.0,
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_concurrent_beam_and_pipe_search_128dim() {
        use crate::search::pipelined::{PipelinedSearcher, PipelinedReaderConfig};
        use diskann_providers::storage::get_disk_index_file;
        use rayon::prelude::*;

        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));

        // Create beam search engine (DiskIndexSearcher).
        let beam_engine = create_disk_index_searcher::<GraphDataF32VectorUnitData>(
            CreateDiskIndexSearcherParams {
                max_thread_num: 2,
                pq_pivot_file_path: TEST_PQ_PIVOT_128DIM,
                pq_compressed_file_path: TEST_PQ_COMPRESSED_128DIM,
                index_path: TEST_INDEX_128DIM,
                index_path_prefix: TEST_INDEX_PREFIX_128DIM,
                ..Default::default()
            },
            &storage_provider,
        );

        // Create pipelined search engine (PipelinedSearcher).
        let disk_index_reader = DiskIndexReader::<f32>::new(
            TEST_PQ_PIVOT_128DIM.to_string(),
            TEST_PQ_COMPRESSED_128DIM.to_string(),
            storage_provider.as_ref(),
        )
        .unwrap();
        let pq_data = disk_index_reader.get_pq_data();

        let aligned_reader_factory = VirtualAlignedReaderFactory::new(
            get_disk_index_file(TEST_INDEX_PREFIX_128DIM),
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
        let pipe_searcher = Arc::new(
            PipelinedSearcher::<GraphDataF32VectorUnitData>::new(
                graph_header,
                pq_data,
                Metric::L2,
                4,
                None,
                real_index_path.to_str().unwrap().to_string(),
                PipelinedReaderConfig::default(),
            )
            .unwrap(),
        );

        // Load queries and ground truth.
        let (query_vector, _, _) = diskann_providers::utils::file_util::load_bin::<f32, _>(
            storage_provider.as_ref(),
            TEST_QUERY_10PTS_128DIM,
            0,
        )
        .unwrap();
        let truth_result =
            load_query_result(storage_provider.as_ref(), TEST_TRUTH_RESULT_10PTS_128DIM);

        let dim = 128usize;
        let k = 10usize;
        let l = 100u32;
        let num_queries = query_vector.len() / dim;

        // Run beam search and pipe search concurrently via rayon.
        let queries: Vec<&[f32]> = (0..num_queries)
            .map(|q| &query_vector[q * dim..(q + 1) * dim])
            .collect();
        let beam_ref = &beam_engine;
        let pipe_ref = &pipe_searcher;
        let truth_ref = &truth_result;

        queries.par_iter().enumerate().for_each(|(q, query)| {
            // Beam search
            let beam_result = beam_ref
                .search(query, k as u32, l, None, None, false)
                .unwrap();
            let beam_ids: Vec<u32> = beam_result.results.iter().map(|r| r.vertex_id).collect();
            let truth_slice = &truth_ref[q * k..(q + 1) * k];

            // Pipe search (runs concurrently with beam search across rayon threads)
            let pipe_result = pipe_ref.search(query, k as u32, l, 4).unwrap();
            let pipe_ids: Vec<u32> = pipe_result.results.iter().map(|r| r.vertex_id).collect();

            // Both should produce results with reasonable overlap.
            let beam_matching = beam_ids.iter().filter(|id| truth_slice.contains(id)).count();
            let pipe_matching = pipe_ids.iter().filter(|id| truth_slice.contains(id)).count();
            // Per-query: at least some overlap (>=30%) to guard against total failures.
            assert!(
                beam_matching as f32 / k as f32 >= 0.3,
                "Beam search returned no relevant results for query {}",
                q,
            );
            assert!(
                pipe_matching as f32 / k as f32 >= 0.3,
                "Pipe search returned no relevant results for query {}",
                q,
            );
        });
    }
}
