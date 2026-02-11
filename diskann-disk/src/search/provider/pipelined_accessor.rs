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
use std::time::Instant;

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
use crate::search::search_trace::{OptionalTrace, SearchTrace, TraceEventKind};
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
    /// Submission rank (lower = higher priority / submitted earlier).
    /// Nodes submitted first via closest_notvisited() have better PQ distance,
    /// so expanding them first (like PipeSearch) improves search quality.
    rank: u64,
}

impl LoadedNode {
    /// Reset and fill from sector buffer, reusing existing Vec capacity.
    fn parse_from(
        &mut self,
        sector_buf: &[u8],
        vertex_id: u32,
        num_nodes_per_sector: u64,
        node_len: u64,
        fp_vector_len: u64,
        rank: u64,
    ) -> ANNResult<()> {
        let offset = node_offset_in_sector(vertex_id, num_nodes_per_sector, node_len);
        let end = offset + node_len as usize;
        let node_data = sector_buf.get(offset..end).ok_or_else(|| {
            ANNError::log_index_error(format_args!(
                "Node data out of bounds: vertex {} offset {}..{} in buffer of len {}",
                vertex_id, offset, end, sector_buf.len()
            ))
        })?;

        let fp_len = fp_vector_len as usize;
        if fp_len > node_data.len() {
            return Err(ANNError::log_index_error(format_args!(
                "fp_vector_len {} exceeds node_data len {}",
                fp_len, node_data.len()
            )));
        }

        self.fp_vector.clear();
        self.fp_vector.extend_from_slice(&node_data[..fp_len]);

        let neighbor_data = &node_data[fp_len..];
        let num_neighbors = LittleEndian::read_u32(&neighbor_data[..4]) as usize;
        let max_neighbors = (neighbor_data.len().saturating_sub(4)) / 4;
        let num_neighbors = num_neighbors.min(max_neighbors);

        self.adjacency_list.clear();
        for i in 0..num_neighbors {
            let start = 4 + i * 4;
            self.adjacency_list.push(LittleEndian::read_u32(&neighbor_data[start..start + 4]));
        }

        self.rank = rank;
        Ok(())
    }
}

/// Tracks an in-flight IO request.
struct InFlightIo {
    vertex_id: u32,
    slot_id: usize,
    rank: u64,
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
    // Per-query scratch collections, cleared between queries but retain capacity
    in_flight_ios: VecDeque<InFlightIo>,
    loaded_nodes: HashMap<u32, LoadedNode>,
    expanded_ids: Vec<u32>,
    distance_cache: HashMap<u32, f32>,
    /// Reusable buffer for neighbor IDs during expand_available
    neighbor_buf: Vec<u32>,
    /// Freelist of LoadedNode instances to avoid per-node allocation
    node_pool: Vec<LoadedNode>,
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
        Ok(Self {
            reader,
            pq_scratch,
            in_flight_ios: VecDeque::new(),
            loaded_nodes: HashMap::new(),
            expanded_ids: Vec::new(),
            distance_cache: HashMap::new(),
            neighbor_buf: Vec::new(),
            node_pool: Vec::new(),
        })
    }

    fn try_modify(&mut self, _args: PipelinedScratchArgs) -> Result<(), Self::Error> {
        self.reader.reset();
        // Return all loaded_nodes back to the pool before clearing
        self.node_pool.extend(self.loaded_nodes.drain().map(|(_, node)| node));
        self.in_flight_ios.clear();
        self.expanded_ids.clear();
        self.distance_cache.clear();
        self.neighbor_buf.clear();
        Ok(())
    }
}

impl PipelinedScratch {
    /// Get a LoadedNode from the pool, or create a new empty one.
    fn acquire_node(&mut self) -> LoadedNode {
        self.node_pool.pop().unwrap_or_else(|| LoadedNode {
            fp_vector: Vec::new(),
            adjacency_list: Vec::new(),
            rank: 0,
        })
    }

    /// Return a LoadedNode to the pool for reuse.
    fn release_node(&mut self, node: LoadedNode) {
        self.node_pool.push(node);
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

    // IO state (now lives in scratch for reuse, accessed via self.scratch)
    next_slot_id: usize,
    max_slots: usize,
    /// Monotonically increasing submission rank for priority-ordered expansion.
    next_rank: u64,

    // IO statistics
    io_count: u32,
    cache_hits: u32,
    /// Accumulated IO time (submission + polling + waiting)
    io_time: std::time::Duration,
    /// Accumulated CPU time (fp distance + PQ distance + node parsing)
    cpu_time: std::time::Duration,
    /// PQ preprocess time (distance table construction)
    preprocess_time: std::time::Duration,
    // Shared stats written on drop so caller can read them after search
    shared_io_stats: Arc<PipelinedIoStats>,

    // Optional per-query trace for profiling and algorithmic comparison
    trace: Option<SearchTrace>,
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
            next_slot_id: 0,
            max_slots: slots,
            next_rank: 0,
            io_count: 0,
            cache_hits: 0,
            io_time: std::time::Duration::ZERO,
            cpu_time: std::time::Duration::ZERO,
            preprocess_time: std::time::Duration::ZERO,
            shared_io_stats,
            trace: None,
        })
    }

    /// Preprocess PQ distance tables for this query. Must be called before search.
    pub fn preprocess_query(&mut self) -> ANNResult<()> {
        let timer = std::time::Instant::now();
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
        self.preprocess_time = timer.elapsed();
        Ok(())
    }

    /// Enable per-query tracing. Call before search.
    pub fn enable_trace(&mut self) {
        self.trace = Some(SearchTrace::new());
    }

    /// Take the completed trace (if any). Call after search.
    pub fn take_trace(&mut self) -> Option<SearchTrace> {
        if let Some(t) = self.trace.as_mut() {
            t.finish();
        }
        self.trace.take()
    }

    /// Compute PQ distances for a set of neighbor IDs.
    /// `ids` must not alias any mutable scratch fields used by PQ computation.
    fn pq_distances<F>(&mut self, ids: &[u32], mut f: F) -> ANNResult<()>
    where
        F: FnMut(f32, u32),
    {
        Self::pq_distances_inner(&mut self.scratch.pq_scratch, self.provider, ids, &mut f)
    }

    fn pq_distances_inner<F>(
        pq: &mut PQScratch,
        provider: &DiskProvider<Data>,
        ids: &[u32],
        f: &mut F,
    ) -> ANNResult<()>
    where
        F: FnMut(f32, u32),
    {
        compute_pq_distance(
            ids,
            provider.pq_data.get_num_chunks(),
            &pq.aligned_pqtable_dist_scratch,
            provider.pq_data.pq_compressed_data().get_data(),
            &mut pq.aligned_pq_coord_scratch,
            &mut pq.aligned_dist_scratch,
        )?;
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
        if self.scratch.in_flight_ios.is_empty() {
            return Ok(());
        }

        let mut trace = OptionalTrace(self.trace.as_mut());

        let io_start = Instant::now();
        trace.begin_phase();
        let completed_slots = self.scratch.reader.poll_completions()?;
        trace.end_phase_io_poll();
        self.io_time += io_start.elapsed();

        if completed_slots.is_empty() {
            return Ok(());
        }

        Self::process_completed_ios_inner(
            &mut self.scratch,
            &completed_slots,
            &mut trace,
            self.num_nodes_per_sector,
            self.node_len,
            self.fp_vector_len,
        )
    }
    /// Block until at least one IO completes, then eagerly drain all available.
    fn wait_and_drain(&mut self) -> ANNResult<()> {
        let mut trace = OptionalTrace(self.trace.as_mut());
        let io_start = Instant::now();
        trace.begin_phase();
        let completed_slots = self.scratch.reader.wait_completions()?;
        trace.end_phase_io_poll();
        self.io_time += io_start.elapsed();

        if completed_slots.is_empty() {
            return Ok(());
        }

        Self::process_completed_ios_inner(
            &mut self.scratch,
            &completed_slots,
            &mut trace,
            self.num_nodes_per_sector,
            self.node_len,
            self.fp_vector_len,
        )
    }

    /// Shared logic: process completed slot IDs, parse nodes, retain in-flight.
    /// Uses linear scan on completed_slots (small, bounded by max_slots) to
    /// avoid per-poll HashSet allocation. Reuses LoadedNode instances from the
    /// node pool to avoid per-IO Vec allocations.
    fn process_completed_ios_inner(
        scratch: &mut PipelinedScratch,
        completed_slots: &[usize],
        trace: &mut OptionalTrace<'_>,
        num_nodes_per_sector: u64,
        node_len: u64,
        fp_vector_len: u64,
    ) -> ANNResult<()> {
        let mut i = 0;
        while i < scratch.in_flight_ios.len() {
            let io = &scratch.in_flight_ios[i];
            if completed_slots.contains(&io.slot_id) {
                let io = scratch.in_flight_ios.swap_remove_back(i).unwrap();
                trace.begin_phase();
                // Acquire node first (mutably borrows node_pool),
                // then get sector buf (immutably borrows reader) — no conflict.
                let mut node = scratch.node_pool.pop().unwrap_or_else(|| LoadedNode {
                    fp_vector: Vec::new(),
                    adjacency_list: Vec::new(),
                    rank: 0,
                });
                let sector_buf = scratch.reader.get_slot_buf(io.slot_id);
                node.parse_from(
                    sector_buf,
                    io.vertex_id,
                    num_nodes_per_sector,
                    node_len,
                    fp_vector_len,
                    io.rank,
                )?;
                trace.end_phase_parse_node();
                trace.event(TraceEventKind::Complete { node_id: io.vertex_id });
                scratch.loaded_nodes.insert(io.vertex_id, node);
            } else {
                i += 1;
            }
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
        self.scratch.neighbor_buf.clear();
        self.scratch.neighbor_buf.extend(vec_id_itr);
        let mut f = f;
        let PipelinedScratch { ref mut pq_scratch, ref neighbor_buf, .. } = *self.scratch;
        Self::pq_distances_inner(
            pq_scratch,
            self.provider,
            neighbor_buf,
            &mut f,
        )
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
        let mut trace = OptionalTrace(self.trace.as_mut());
        let io_start = Instant::now();
        trace.begin_phase();
        for id in ids {
            if self.scratch.loaded_nodes.contains_key(&id) {
                continue; // Already loaded from a previous IO
            }

            // Check node cache first — if the node is cached, build a LoadedNode
            // from the cache and skip IO entirely.
            if let (Some(vec_data), Some(adj_list)) = (
                self.node_cache.get_vector(&id),
                self.node_cache.get_adjacency_list(&id),
            ) {
                let mut node = self.scratch.acquire_node();
                node.fp_vector.clear();
                node.fp_vector.extend_from_slice(bytemuck::cast_slice(vec_data));
                node.adjacency_list.clear();
                node.adjacency_list.extend(adj_list.iter().copied());
                node.rank = self.next_rank;
                self.next_rank += 1;
                self.scratch.loaded_nodes.insert(id, node);
                self.cache_hits += 1;
                trace.event(TraceEventKind::CacheHit { node_id: id });
                continue;
            }

            // Don't submit if all io_uring slots are occupied — prevents overwriting
            // buffers that still have in-flight reads.
            if self.scratch.in_flight_ios.len() >= self.max_slots {
                break;
            }

            let sector_idx =
                node_sector_index(id, self.num_nodes_per_sector, self.num_sectors_per_node);
            let sector_offset = sector_idx * self.block_size as u64;
            let slot_id = self.next_slot_id % self.max_slots;
            let rank = self.next_rank;
            self.next_rank += 1;
            // Best-effort: if submission fails, the node will be retried
            if self.scratch.reader.submit_read(sector_offset, slot_id).is_ok() {
                self.scratch.in_flight_ios.push_back(InFlightIo {
                    vertex_id: id,
                    slot_id,
                    rank,
                });
                trace.event(TraceEventKind::Submit {
                    node_id: id,
                    inflight: self.scratch.in_flight_ios.len(),
                });
                self.next_slot_id = (self.next_slot_id + 1) % self.max_slots;
                self.io_count += 1;
            }
        }
        trace.end_phase_io_submit();
        self.io_time += io_start.elapsed();
    }

    /// Poll for completed reads and expand the best loaded node.
    ///
    /// Uses two selection strategies:
    /// 1. If `ids` provides candidates, pick the first loaded match (queue order)
    /// 2. Otherwise, pick the loaded node with the lowest submission rank
    ///    (earliest submitted = best PQ distance at submission time)
    fn expand_available<P, F>(
        &mut self,
        ids: impl Iterator<Item = Self::Id> + Send,
        _computer: &Self::QueryComputer,
        mut pred: P,
        mut on_neighbors: F,
    ) -> impl std::future::Future<Output = ANNResult<usize>> + Send
    where
        P: HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(f32, Self::Id) + Send,
    {
        async move {
            self.scratch.expanded_ids.clear();

            // Non-blocking poll for completions
            self.drain_completions()?;

            if self.scratch.loaded_nodes.is_empty() {
                return Ok(0);
            }

            // Try caller's priority order first
            let mut best_vid: Option<u32> = None;
            for id in ids {
                if self.scratch.loaded_nodes.contains_key(&id) {
                    best_vid = Some(id);
                    break;
                }
            }

            // Fallback: pick loaded node with lowest rank (best PQ at submission)
            if best_vid.is_none() {
                best_vid = self
                    .scratch
                    .loaded_nodes
                    .iter()
                    .min_by_key(|(_, node)| node.rank)
                    .map(|(&id, _)| id);
            }

            let vid = match best_vid {
                Some(id) => id,
                None => return Ok(0),
            };
            let node = self.scratch.loaded_nodes.remove(&vid).unwrap();
            self.scratch.expanded_ids.push(vid);

            // Compute full-precision distance and cache it for post-processing
            let cpu_start = Instant::now();
            let fp_vec: &[Data::VectorDataType] = bytemuck::cast_slice(&node.fp_vector);
            let fp_dist = self
                .provider
                .distance_comparer
                .evaluate_similarity(self.query, fp_vec);
            if let Some(t) = self.trace.as_mut() {
                t.profile.fp_distance_us += cpu_start.elapsed().as_micros() as u64;
            }
            self.scratch.distance_cache.insert(vid, fp_dist);

            // Get unvisited neighbors into reusable buffer
            self.scratch.neighbor_buf.clear();
            self.scratch.neighbor_buf.extend(
                node.adjacency_list
                    .iter()
                    .copied()
                    .filter(|&nbr| (nbr as usize) < self.num_points && pred.eval_mut(&nbr)),
            );
            let num_new = self.scratch.neighbor_buf.len() as u32;

            if !self.scratch.neighbor_buf.is_empty() {
                let pq_start = Instant::now();
                let PipelinedScratch { ref mut pq_scratch, ref neighbor_buf, .. } = *self.scratch;
                Self::pq_distances_inner(
                    pq_scratch,
                    self.provider,
                    neighbor_buf,
                    &mut on_neighbors,
                )?;
                if let Some(t) = self.trace.as_mut() {
                    t.profile.pq_distance_us += pq_start.elapsed().as_micros() as u64;
                }
            }
            self.cpu_time += cpu_start.elapsed();

            if let Some(t) = self.trace.as_mut() {
                t.record_expand();
                t.event(TraceEventKind::Expand {
                    node_id: vid,
                    fp_distance: fp_dist,
                    num_neighbors: node.adjacency_list.len() as u32,
                    num_new_candidates: num_new,
                });
            }

            // Return node to pool for reuse
            self.scratch.release_node(node);

            Ok(1)
        }
    }

    /// Returns true when there are in-flight IO operations.
    fn has_pending(&self) -> bool {
        !self.scratch.in_flight_ios.is_empty()
    }

    fn inflight_count(&self) -> usize {
        self.scratch.in_flight_ios.len()
    }

    fn wait_for_io(&mut self) {
        // Only block if there are actually in-flight IOs to wait for
        if !self.scratch.in_flight_ios.is_empty() {
            let _ = self.wait_and_drain();
        }
    }

    fn last_expanded_ids(&self) -> &[u32] {
        &self.scratch.expanded_ids
    }

    fn is_pipelined(&self) -> bool {
        true
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
        self.shared_io_stats
            .io_us
            .fetch_add(self.io_time.as_micros() as u64, Ordering::Relaxed);
        self.shared_io_stats
            .cpu_us
            .fetch_add(self.cpu_time.as_micros() as u64, Ordering::Relaxed);
        self.shared_io_stats
            .preprocess_us
            .fetch_add(self.preprocess_time.as_micros() as u64, Ordering::Relaxed);

        // Print trace profile if enabled (controlled by DISKANN_TRACE=1)
        if let Some(trace) = self.trace.as_mut() {
            trace.finish();
            trace.print_profile_summary();
        }
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
    /// Enable per-query SearchTrace. The trace profile is printed to stderr
    /// after each query completes. Use for profiling, not production.
    pub trace_enabled: bool,
}

/// Shared IO statistics written by the accessor and read by the caller after search.
/// Uses atomics so the accessor (which lives inside search_internal) can write stats
/// that the caller can read after the search completes.
pub struct PipelinedIoStats {
    pub io_count: AtomicU32,
    pub cache_hits: AtomicU32,
    pub io_us: std::sync::atomic::AtomicU64,
    pub cpu_us: std::sync::atomic::AtomicU64,
    pub preprocess_us: std::sync::atomic::AtomicU64,
}

impl Default for PipelinedIoStats {
    fn default() -> Self {
        Self {
            io_count: AtomicU32::new(0),
            cache_hits: AtomicU32::new(0),
            io_us: std::sync::atomic::AtomicU64::new(0),
            cpu_us: std::sync::atomic::AtomicU64::new(0),
            preprocess_us: std::sync::atomic::AtomicU64::new(0),
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
        _candidates: I,
        output: &mut B,
    ) -> Result<usize, Self::Error>
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: SearchOutputBuffer<(u32, Data::AssociatedDataType)> + Send + ?Sized,
    {
        // Rerank using ALL expanded nodes' cached fp-distances, not just
        // candidates from the priority queue. This matches PipeANN's
        // full_retset approach: every expanded node contributes to results
        // regardless of its PQ distance ranking.
        let mut reranked: Vec<((u32, Data::AssociatedDataType), f32)> = accessor
            .scratch
            .distance_cache
            .iter()
            .filter(|(id, _)| (self.filter)(id))
            .map(|(&id, &dist)| ((id, Data::AssociatedDataType::default()), dist))
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
        if self.config.trace_enabled {
            accessor.enable_trace();
        }
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
        query_stats.io_time_us = io_stats.io_us.load(Ordering::Relaxed) as u128;
        query_stats.cpu_time_us = io_stats.cpu_us.load(Ordering::Relaxed) as u128;
        query_stats.query_pq_preprocess_time_us =
            io_stats.preprocess_us.load(Ordering::Relaxed) as u128;

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
