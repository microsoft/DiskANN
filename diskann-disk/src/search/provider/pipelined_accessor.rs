/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Queue-based pipelined disk accessor that integrates with the generic search loop
//! via the `ExpandBeam` trait's `submit_expand` / `expand_available` / `has_pending` methods.
//!
//! Instead of duplicating the search loop (like `PipelinedSearcher`), this accessor
//! plugs into `DiskANNIndex::search_internal()` and overlaps IO with computation
//! using io_uring under the hood.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::ops::Range;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use byteorder::{ByteOrder, LittleEndian};
use diskann::{
    graph::{
        glue::{ExpandBeam, HybridPredicate, IdIterator, SearchExt, SearchPostProcess, SearchStrategy},
        search_output_buffer, AdjacencyList, SearchOutputBuffer, SearchParams,
    },
    neighbor::Neighbor,
    provider::{Accessor, BuildQueryComputer, DefaultContext, DelegateNeighbor, HasId, NeighborAccessor},
    utils::object_pool::{ObjectPool, PoolOption, TryAsPooled},
    ANNError, ANNResult,
};
use diskann_providers::model::{
    compute_pq_distance, graph::traits::GraphDataType, pq::quantizer_preprocess, PQScratch,
};
use diskann_vector::DistanceFunction;

use crate::data_model::Cache;
use crate::search::pipelined::{PipelinedReader, PipelinedReaderConfig, MAX_IO_CONCURRENCY};
use crate::search::sector_math::{node_offset_in_sector, node_sector_index};
use crate::search::traits::VertexProviderFactory;
use crate::utils::QueryStatistics;

use super::disk_provider::{
    DiskIndexSearcher, DiskProvider, DiskQueryComputer, SearchResult, SearchResultItem,
    SearchResultStats,
};

/// A loaded node parsed from sector data.
struct LoadedNode {
    fp_vector: Vec<u8>,
    adjacency_list: Vec<u32>,
}

/// Tracks an in-flight IO request.
struct InFlightIo {
    vertex_id: u32,
    slot_id: usize,
}

/// Parse a node from raw sector buffer bytes.
fn parse_node(
    sector_buf: &[u8],
    vertex_id: u32,
    num_nodes_per_sector: u64,
    node_len: u64,
    fp_vector_len: u64,
) -> ANNResult<LoadedNode> {
    let offset = node_offset_in_sector(vertex_id, num_nodes_per_sector, node_len);
    let end = offset + node_len as usize;
    let node_data = sector_buf.get(offset..end).ok_or_else(|| {
        ANNError::log_index_error(format_args!(
            "Node data out of bounds: vertex {} offset {}..{} in buffer of len {}",
            vertex_id, offset, end, sector_buf.len()
        ))
    })?;

    let fp_vector_len_usize = fp_vector_len as usize;
    if fp_vector_len_usize > node_data.len() {
        return Err(ANNError::log_index_error(format_args!(
            "fp_vector_len {} exceeds node_data len {}",
            fp_vector_len_usize,
            node_data.len()
        )));
    }

    let fp_vector = node_data[..fp_vector_len_usize].to_vec();
    let neighbor_data = &node_data[fp_vector_len_usize..];
    let num_neighbors = LittleEndian::read_u32(&neighbor_data[..4]) as usize;
    let max_neighbors = (neighbor_data.len().saturating_sub(4)) / 4;
    let num_neighbors = num_neighbors.min(max_neighbors);
    let mut adjacency_list = Vec::with_capacity(num_neighbors);
    for i in 0..num_neighbors {
        let start = 4 + i * 4;
        adjacency_list.push(LittleEndian::read_u32(&neighbor_data[start..start + 4]));
    }

    Ok(LoadedNode {
        fp_vector,
        adjacency_list,
    })
}

/// Max buffer slots to use, based on beam width.
#[inline]
fn max_slots(beam_width: usize) -> usize {
    (beam_width * 2).clamp(16, MAX_IO_CONCURRENCY)
}

// ---------------------------------------------------------------------------
// Poolable scratch: PipelinedReader + PQScratch, reused across queries
// ---------------------------------------------------------------------------

/// Reusable scratch state for pipelined search, pooled to avoid per-query
/// allocation of io_uring rings, file descriptors, and PQ scratch buffers.
pub struct PipelinedScratch {
    pub reader: PipelinedReader,
    pub pq_scratch: PQScratch,
}

/// Arguments for creating or resetting a [`PipelinedScratch`].
#[derive(Clone)]
pub struct PipelinedScratchArgs {
    pub disk_index_path: String,
    pub max_slots: usize,
    pub slot_size: usize,
    pub alignment: usize,
    pub graph_degree: usize,
    pub dims: usize,
    pub num_pq_chunks: usize,
    pub num_pq_centers: usize,
    pub reader_config: PipelinedReaderConfig,
}

impl TryAsPooled<PipelinedScratchArgs> for PipelinedScratch {
    type Error = ANNError;

    fn try_create(args: PipelinedScratchArgs) -> Result<Self, Self::Error> {
        let reader = PipelinedReader::new(
            &args.disk_index_path,
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

    fn try_modify(&mut self, _args: PipelinedScratchArgs) -> Result<(), Self::Error> {
        self.reader.reset();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PipelinedDiskAccessor
// ---------------------------------------------------------------------------

/// Pipelined disk accessor that overlaps IO and compute via io_uring.
///
/// Implements the `ExpandBeam` trait's queue-based methods:
/// - `submit_expand`: submits non-blocking io_uring reads for the given node IDs
/// - `expand_available`: polls for completed reads and expands those nodes
/// - `has_pending`: returns true when IO operations are in-flight
pub struct PipelinedDiskAccessor<'a, Data: GraphDataType<VectorIdType = u32>> {
    provider: &'a DiskProvider<Data>,
    scratch: PoolOption<PipelinedScratch>,
    query: &'a [Data::VectorDataType],

    // Graph geometry (cached from GraphHeader)
    num_nodes_per_sector: u64,
    num_sectors_per_node: usize,
    block_size: usize,
    node_len: u64,
    fp_vector_len: u64,
    num_points: usize,

    // Node cache (shared, read-only) for avoiding disk IO on hot nodes
    node_cache: Arc<Cache<Data>>,

    // IO state
    in_flight_ios: VecDeque<InFlightIo>,
    loaded_nodes: HashMap<u32, LoadedNode>,
    next_slot_id: usize,
    max_slots: usize,

    // Distance cache for post-processing rerank
    distance_cache: HashMap<u32, f32>,

    // IO statistics
    io_count: u32,
    cache_hits: u32,
    // Shared stats written on drop so caller can read them after search
    shared_io_stats: Arc<PipelinedIoStats>,
}

impl<'a, Data> PipelinedDiskAccessor<'a, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    /// Create a new pipelined disk accessor using a pooled scratch.
    pub fn new(
        provider: &'a DiskProvider<Data>,
        query: &'a [Data::VectorDataType],
        scratch: PoolOption<PipelinedScratch>,
        node_cache: Arc<Cache<Data>>,
        shared_io_stats: Arc<PipelinedIoStats>,
    ) -> ANNResult<Self> {
        let metadata = provider.graph_header.metadata();
        let dims = metadata.dims;
        let num_nodes_per_sector = metadata.num_nodes_per_block;
        let node_len = metadata.node_len;
        let fp_vector_len = (dims * std::mem::size_of::<Data::VectorDataType>()) as u64;

        let block_size = provider.graph_header.effective_block_size();
        let num_sectors_per_node = provider.graph_header.num_sectors_per_node();
        let slots = scratch.reader.max_slots();

        Ok(Self {
            provider,
            scratch,
            query,
            num_nodes_per_sector,
            num_sectors_per_node,
            block_size,
            node_len,
            fp_vector_len,
            num_points: provider.num_points,
            node_cache,
            in_flight_ios: VecDeque::new(),
            loaded_nodes: HashMap::new(),
            next_slot_id: 0,
            max_slots: slots,
            distance_cache: HashMap::new(),
            io_count: 0,
            cache_hits: 0,
            shared_io_stats,
        })
    }

    /// Preprocess PQ distance tables for this query. Must be called before search.
    pub fn preprocess_query(&mut self) -> ANNResult<()> {
        let metadata = self.provider.graph_header.metadata();
        let dims = metadata.dims;
        let medoid = metadata.medoid as u32;
        self.scratch.pq_scratch.set(dims, self.query, 1.0)?;
        quantizer_preprocess(
            &mut self.scratch.pq_scratch,
            &self.provider.pq_data,
            self.provider.metric,
            &[medoid],
        )?;
        Ok(())
    }

    /// Compute PQ distances for a set of neighbor IDs.
    fn pq_distances<F>(&mut self, ids: &[u32], mut f: F) -> ANNResult<()>
    where
        F: FnMut(f32, u32),
    {
        let pq = &mut self.scratch.pq_scratch;
        compute_pq_distance(
            ids,
            self.provider.pq_data.get_num_chunks(),
            &pq.aligned_pqtable_dist_scratch,
            self.provider.pq_data.pq_compressed_data().get_data(),
            &mut pq.aligned_pq_coord_scratch,
            &mut pq.aligned_dist_scratch,
        )?;
        let pq = &self.scratch.pq_scratch;
        for (i, id) in ids.iter().enumerate() {
            f(pq.aligned_dist_scratch[i], *id);
        }
        Ok(())
    }

    /// Returns the number of disk IO operations performed.
    pub fn io_count(&self) -> u32 {
        self.io_count
    }

    /// Returns the number of cache hits (nodes served from cache without IO).
    pub fn cache_hits(&self) -> u32 {
        self.cache_hits
    }

    /// Poll completed IOs and move data from reader buffers into loaded_nodes.
    fn drain_completions(&mut self) -> ANNResult<()> {
        let completed_slots = if self.in_flight_ios.is_empty() {
            Vec::new()
        } else {
            self.scratch.reader.poll_completions()?
        };

        if !completed_slots.is_empty() {
            let completed_set: std::collections::HashSet<usize> =
                completed_slots.into_iter().collect();
            let mut remaining = VecDeque::new();
            while let Some(io) = self.in_flight_ios.pop_front() {
                if completed_set.contains(&io.slot_id) {
                    let sector_buf = self.scratch.reader.get_slot_buf(io.slot_id);
                    let node = parse_node(
                        sector_buf,
                        io.vertex_id,
                        self.num_nodes_per_sector,
                        self.node_len,
                        self.fp_vector_len,
                    )?;
                    self.loaded_nodes.insert(io.vertex_id, node);
                } else {
                    remaining.push_back(io);
                }
            }
            self.in_flight_ios = remaining;
        }
        Ok(())
    }
}

impl<Data> HasId for PipelinedDiskAccessor<'_, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    type Id = u32;
}

impl<'a, Data> Accessor for PipelinedDiskAccessor<'a, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    type Extended = &'a [u8];
    type Element<'b>
        = &'a [u8]
    where
        Self: 'b;
    type ElementRef<'b> = &'b [u8];
    type GetError = ANNError;

    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        std::future::ready(self.provider.pq_data.get_compressed_vector(id as usize))
    }
}

impl<Data> IdIterator<Range<u32>> for PipelinedDiskAccessor<'_, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    async fn id_iterator(&mut self) -> Result<Range<u32>, ANNError> {
        Ok(0..self.num_points as u32)
    }
}

/// Delegate for neighbor access (required by AsNeighbor).
pub struct PipelinedNeighborDelegate<'a, 'b, Data: GraphDataType<VectorIdType = u32>>(
    #[allow(dead_code)] &'a mut PipelinedDiskAccessor<'b, Data>,
);

impl<Data> HasId for PipelinedNeighborDelegate<'_, '_, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    type Id = u32;
}

impl<Data> NeighborAccessor for PipelinedNeighborDelegate<'_, '_, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    fn get_neighbors(
        self,
        _id: Self::Id,
        _neighbors: &mut AdjacencyList<Self::Id>,
    ) -> impl Future<Output = ANNResult<Self>> + Send {
        // Neighbor expansion is handled by expand_available, not get_neighbors
        async { Ok(self) }
    }
}

impl<'a, 'b, Data> DelegateNeighbor<'a> for PipelinedDiskAccessor<'b, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    type Delegate = PipelinedNeighborDelegate<'a, 'b, Data>;
    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        PipelinedNeighborDelegate(self)
    }
}

impl<Data> BuildQueryComputer<[Data::VectorDataType]> for PipelinedDiskAccessor<'_, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
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

impl<Data> ExpandBeam<[Data::VectorDataType]> for PipelinedDiskAccessor<'_, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    /// Submit non-blocking io_uring reads for the given node IDs.
    /// Nodes found in the node cache are placed directly into `loaded_nodes`,
    /// skipping disk IO entirely.
    fn submit_expand(&mut self, ids: impl Iterator<Item = Self::Id> + Send) {
        for id in ids {
            if self.loaded_nodes.contains_key(&id) {
                continue; // Already loaded from a previous IO
            }

            // Check node cache first — if the node is cached, build a LoadedNode
            // from the cache and skip IO entirely.
            if let (Some(vec_data), Some(adj_list)) = (
                self.node_cache.get_vector(&id),
                self.node_cache.get_adjacency_list(&id),
            ) {
                let fp_vector: Vec<u8> = bytemuck::cast_slice(vec_data).to_vec();
                let adjacency_list: Vec<u32> = adj_list.iter().copied().collect();
                self.loaded_nodes.insert(id, LoadedNode { fp_vector, adjacency_list });
                self.cache_hits += 1;
                continue;
            }

            // Don't submit if all io_uring slots are occupied — prevents overwriting
            // buffers that still have in-flight reads.
            if self.in_flight_ios.len() >= self.max_slots {
                break;
            }

            let sector_idx =
                node_sector_index(id, self.num_nodes_per_sector, self.num_sectors_per_node);
            let sector_offset = sector_idx * self.block_size as u64;
            let slot_id = self.next_slot_id % self.max_slots;
            // Best-effort: if submission fails, the node will be retried
            if self.scratch.reader.submit_read(sector_offset, slot_id).is_ok() {
                self.in_flight_ios.push_back(InFlightIo {
                    vertex_id: id,
                    slot_id,
                });
                self.next_slot_id = (self.next_slot_id + 1) % self.max_slots;
                self.io_count += 1;
            }
        }
    }

    /// Poll for completed reads and expand up to `up_to` nodes.
    /// Remaining loaded-but-unexpanded nodes stay buffered for the next call,
    /// which lets the search loop submit new IOs sooner (process-few-submit-few).
    fn expand_available<P, F>(
        &mut self,
        _ids: impl Iterator<Item = Self::Id> + Send,
        _computer: &Self::QueryComputer,
        mut pred: P,
        mut on_neighbors: F,
        up_to: usize,
    ) -> impl std::future::Future<Output = ANNResult<usize>> + Send
    where
        P: HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(f32, Self::Id) + Send,
    {
        async move {
            // Poll completions
            self.drain_completions()?;

            // If nothing is loaded yet and we have in-flight IO, wait for at least one
            if self.loaded_nodes.is_empty() && !self.in_flight_ios.is_empty() {
                let completed = self.scratch.reader.wait_completions()?;
                if !completed.is_empty() {
                    let completed_set: std::collections::HashSet<usize> =
                        completed.into_iter().collect();
                    let mut remaining = VecDeque::new();
                    while let Some(io) = self.in_flight_ios.pop_front() {
                        if completed_set.contains(&io.slot_id) {
                            let sector_buf = self.scratch.reader.get_slot_buf(io.slot_id);
                            let node = parse_node(
                                sector_buf,
                                io.vertex_id,
                                self.num_nodes_per_sector,
                                self.node_len,
                                self.fp_vector_len,
                            )?;
                            self.loaded_nodes.insert(io.vertex_id, node);
                        } else {
                            remaining.push_back(io);
                        }
                    }
                    self.in_flight_ios = remaining;
                }
            }

            // Expand up to `up_to` loaded nodes. Unexpanded nodes remain buffered
            // in loaded_nodes for the next call.
            let loaded_ids: Vec<u32> = self.loaded_nodes.keys().copied().take(up_to).collect();
            let mut expanded = 0;

            for vid in loaded_ids {
                let node = match self.loaded_nodes.remove(&vid) {
                    Some(n) => n,
                    None => continue,
                };

                // Compute full-precision distance and cache it for post-processing
                let fp_vec: &[Data::VectorDataType] = bytemuck::cast_slice(&node.fp_vector);
                let fp_dist = self
                    .provider
                    .distance_comparer
                    .evaluate_similarity(self.query, fp_vec);
                self.distance_cache.insert(vid, fp_dist);

                // Get unvisited neighbors
                let neighbors: Vec<u32> = node
                    .adjacency_list
                    .iter()
                    .copied()
                    .filter(|&nbr| (nbr as usize) < self.num_points && pred.eval_mut(&nbr))
                    .collect();

                if !neighbors.is_empty() {
                    self.pq_distances(&neighbors, &mut on_neighbors)?;
                }

                expanded += 1;
            }

            Ok(expanded)
        }
    }

    /// Returns true when there are in-flight IO operations.
    fn has_pending(&self) -> bool {
        !self.in_flight_ios.is_empty() || !self.loaded_nodes.is_empty()
    }
}

impl<Data> SearchExt for PipelinedDiskAccessor<'_, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    async fn starting_points(&self) -> ANNResult<Vec<u32>> {
        let start_vertex_id = self.provider.graph_header.metadata().medoid as u32;
        Ok(vec![start_vertex_id])
    }

    fn terminate_early(&mut self) -> bool {
        false
    }
}

impl<Data> Drop for PipelinedDiskAccessor<'_, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    fn drop(&mut self) {
        self.shared_io_stats
            .io_count
            .fetch_add(self.io_count, Ordering::Relaxed);
        self.shared_io_stats
            .cache_hits
            .fetch_add(self.cache_hits, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// SearchStrategy + PostProcessor for pipelined search
// ---------------------------------------------------------------------------

/// Configuration for creating a pipelined search through DiskIndexSearcher.
pub struct PipelinedConfig<Data: GraphDataType<VectorIdType = u32>> {
    pub beam_width: usize,
    /// Start with a smaller beam and grow adaptively.
    pub adaptive_beam_width: bool,
    /// Optional relaxed monotonicity: continue exploring this many extra
    /// comparisons after the candidate list converges.
    pub relaxed_monotonicity_l: Option<usize>,
    /// Shared node cache. Nodes found here skip disk IO entirely.
    pub node_cache: Arc<Cache<Data>>,
    /// Pooled scratch (io_uring reader + PQ buffers), created once and reused.
    pub scratch_pool: Arc<ObjectPool<PipelinedScratch>>,
    /// Args for retrieving/creating pooled scratch instances.
    pub scratch_args: PipelinedScratchArgs,
}

/// Shared IO statistics written by the accessor and read by the caller after search.
/// Uses atomics so the accessor (which lives inside search_internal) can write stats
/// that the caller can read after the search completes.
pub struct PipelinedIoStats {
    pub io_count: AtomicU32,
    pub cache_hits: AtomicU32,
}

impl Default for PipelinedIoStats {
    fn default() -> Self {
        Self {
            io_count: AtomicU32::new(0),
            cache_hits: AtomicU32::new(0),
        }
    }
}

/// Search strategy that creates PipelinedDiskAccessor instances.
pub struct PipelinedSearchStrategy<'a, Data: GraphDataType<VectorIdType = u32>> {
    query: &'a [Data::VectorDataType],
    config: &'a PipelinedConfig<Data>,
    vector_filter: &'a (dyn Fn(&u32) -> bool + Send + Sync),
    io_stats: Arc<PipelinedIoStats>,
}

/// Post-processor for pipelined search that reranks using cached full-precision distances.
#[derive(Clone, Copy)]
pub struct PipelinedPostProcessor<'a> {
    filter: &'a (dyn Fn(&u32) -> bool + Send + Sync),
}

impl<Data> SearchPostProcess<
    PipelinedDiskAccessor<'_, Data>,
    [Data::VectorDataType],
    (u32, Data::AssociatedDataType),
> for PipelinedPostProcessor<'_>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    type Error = ANNError;

    async fn post_process<I, B>(
        &self,
        accessor: &mut PipelinedDiskAccessor<'_, Data>,
        _query: &[Data::VectorDataType],
        _computer: &DiskQueryComputer,
        candidates: I,
        output: &mut B,
    ) -> Result<usize, Self::Error>
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: SearchOutputBuffer<(u32, Data::AssociatedDataType)> + Send + ?Sized,
    {
        let mut reranked: Vec<((u32, Data::AssociatedDataType), f32)> = candidates
            .map(|n| n.id)
            .filter(|id| (self.filter)(id))
            .filter_map(|id| {
                accessor
                    .distance_cache
                    .get(&id)
                    .map(|&dist| ((id, Data::AssociatedDataType::default()), dist))
            })
            .collect();

        reranked.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        Ok(output.extend(reranked))
    }
}

impl<'this, Data> SearchStrategy<
    DiskProvider<Data>,
    [Data::VectorDataType],
    (u32, Data::AssociatedDataType),
> for PipelinedSearchStrategy<'this, Data>
where
    Data: GraphDataType<VectorIdType = u32>,
{
    type QueryComputer = DiskQueryComputer;
    type SearchAccessor<'a> = PipelinedDiskAccessor<'a, Data>;
    type SearchAccessorError = ANNError;
    type PostProcessor = PipelinedPostProcessor<'this>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a DiskProvider<Data>,
        _context: &DefaultContext,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        let scratch = PoolOption::try_pooled(
            &self.config.scratch_pool,
            self.config.scratch_args.clone(),
        )?;
        let mut accessor = PipelinedDiskAccessor::new(
            provider,
            self.query,
            scratch,
            self.config.node_cache.clone(),
            self.io_stats.clone(),
        )?;
        accessor.preprocess_query()?;
        Ok(accessor)
    }

    fn post_processor(&self) -> Self::PostProcessor {
        PipelinedPostProcessor {
            filter: self.vector_filter,
        }
    }
}

// ---------------------------------------------------------------------------
// DiskIndexSearcher integration (search_pipelined method)
// ---------------------------------------------------------------------------

impl<Data, ProviderFactory> DiskIndexSearcher<Data, ProviderFactory>
where
    Data: GraphDataType<VectorIdType = u32>,
    ProviderFactory: VertexProviderFactory<Data>,
{
    /// Attach a pipelined configuration to this searcher.
    pub fn with_pipelined_config(&mut self, config: PipelinedConfig<Data>) {
        self.pipelined_config = Some(config);
    }

    /// Perform a pipelined search through the unified search loop.
    ///
    /// Requires that `with_pipelined_config()` was called first.
    pub fn search_pipelined(
        &self,
        query: &[Data::VectorDataType],
        return_list_size: u32,
        search_list_size: u32,
        beam_width: usize,
        vector_filter: Option<&(dyn Fn(&u32) -> bool + Send + Sync)>,
    ) -> ANNResult<SearchResult<Data::AssociatedDataType>> {
        let config = self
            .pipelined_config
            .as_ref()
            .ok_or_else(|| ANNError::log_index_error("pipelined_config not set"))?;

        let default_filter: Box<dyn Fn(&u32) -> bool + Send + Sync> = Box::new(|_| true);
        let filter: &(dyn Fn(&u32) -> bool + Send + Sync) =
            vector_filter.unwrap_or(default_filter.as_ref());

        let io_stats = Arc::new(PipelinedIoStats::default());

        let strategy = PipelinedSearchStrategy {
            query,
            config,
            vector_filter: filter,
            io_stats: io_stats.clone(),
        };

        let mut search_params = SearchParams::new(
            return_list_size as usize,
            search_list_size as usize,
            Some(beam_width),
        )?;

        if config.adaptive_beam_width {
            search_params = search_params.with_adaptive_beam_width();
        }
        if let Some(rm_l) = config.relaxed_monotonicity_l {
            search_params = search_params.with_relaxed_monotonicity(rm_l);
        }

        let mut indices = vec![0u32; return_list_size as usize];
        let mut distances = vec![0f32; return_list_size as usize];
        let mut associated_data =
            vec![Data::AssociatedDataType::default(); return_list_size as usize];
        let mut result_output_buffer = search_output_buffer::IdDistanceAssociatedData::new(
            &mut indices[..],
            &mut distances[..],
            &mut associated_data[..],
        );

        let mut query_stats = QueryStatistics::default();
        let timer = std::time::Instant::now();

        // Preprocess PQ distance table: the accessor's build_query_computer relies
        // on the pq_scratch having been preprocessed for this query.
        let stats = self.runtime.block_on(self.index.search(
            &strategy,
            &DefaultContext,
            query,
            &search_params,
            &mut result_output_buffer,
        ))?;

        query_stats.total_comparisons = stats.cmps;
        query_stats.search_hops = stats.hops;
        query_stats.total_execution_time_us = timer.elapsed().as_micros();
        query_stats.total_io_operations = io_stats.io_count.load(Ordering::Relaxed);
        query_stats.total_vertices_loaded =
            io_stats.io_count.load(Ordering::Relaxed) + io_stats.cache_hits.load(Ordering::Relaxed);

        let mut search_result = SearchResult {
            results: Vec::with_capacity(return_list_size as usize),
            stats: SearchResultStats {
                cmps: stats.cmps,
                result_count: stats.result_count,
                query_statistics: query_stats,
            },
        };

        for ((vertex_id, distance), data) in indices
            .into_iter()
            .zip(distances.into_iter())
            .zip(associated_data.into_iter())
        {
            search_result.results.push(SearchResultItem {
                vertex_id,
                distance,
                data,
            });
        }

        Ok(search_result)
    }
}
