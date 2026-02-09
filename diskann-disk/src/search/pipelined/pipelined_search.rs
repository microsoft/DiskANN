/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Core PipeANN pipelined search algorithm.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use byteorder::{ByteOrder, LittleEndian};
use diskann::{utils::VectorRepr, ANNResult};
use diskann_providers::model::{compute_pq_distance, pq::quantizer_preprocess, PQData, PQScratch};
use diskann_vector::{distance::Metric, DistanceFunction};

use super::pipelined_reader::PipelinedReader;

/// A candidate in the sorted candidate pool.
struct Candidate {
    id: u32,
    distance: f32,
    /// true = unvisited and not in-flight, false = in-flight or already-read
    flag: bool,
    /// true = node has been processed (neighbors expanded)
    visited: bool,
}

/// Tracks an in-flight IO request.
struct InFlightIo {
    vertex_id: u32,
    slot_id: usize,
}

/// A loaded node parsed from sector data.
struct LoadedNode {
    fp_vector: Vec<u8>,
    adjacency_list: Vec<u32>,
}

/// Result of a pipelined search.
pub struct PipeSearchResult {
    pub ids: Vec<u32>,
    pub distances: Vec<f32>,
    pub stats: PipeSearchStats,
}

/// Statistics for a pipelined search.
pub struct PipeSearchStats {
    pub total_us: u128,
    pub io_us: u128,
    pub cpu_us: u128,
    pub io_count: u32,
    pub comparisons: u32,
    pub hops: u32,
}

/// Compute the sector index that contains a given vertex.
#[inline]
fn node_sector_index(
    vertex_id: u32,
    num_nodes_per_sector: u64,
    num_sectors_per_node: usize,
) -> u64 {
    1 + if num_nodes_per_sector > 0 {
        vertex_id as u64 / num_nodes_per_sector
    } else {
        vertex_id as u64 * num_sectors_per_node as u64
    }
}

/// Compute the byte offset of a node within its sector.
#[inline]
fn node_offset_in_sector(vertex_id: u32, num_nodes_per_sector: u64, node_len: u64) -> usize {
    if num_nodes_per_sector == 0 {
        0
    } else {
        (vertex_id as u64 % num_nodes_per_sector * node_len) as usize
    }
}

/// Parse a node from raw sector buffer bytes.
fn parse_node(
    sector_buf: &[u8],
    vertex_id: u32,
    num_nodes_per_sector: u64,
    node_len: u64,
    fp_vector_len: u64,
) -> LoadedNode {
    let offset = node_offset_in_sector(vertex_id, num_nodes_per_sector, node_len);
    let node_data = &sector_buf[offset..offset + node_len as usize];

    let fp_vector = node_data[..fp_vector_len as usize].to_vec();

    let neighbor_data = &node_data[fp_vector_len as usize..];
    let num_neighbors = LittleEndian::read_u32(&neighbor_data[..4]) as usize;
    let mut adjacency_list = Vec::with_capacity(num_neighbors);
    for i in 0..num_neighbors {
        let start = 4 + i * 4;
        adjacency_list.push(LittleEndian::read_u32(&neighbor_data[start..start + 4]));
    }

    LoadedNode {
        fp_vector,
        adjacency_list,
    }
}

/// Insert a candidate into the sorted retset, maintaining sort order by distance.
/// Returns the insertion position.
fn insert_into_pool(retset: &mut Vec<Candidate>, pool_size: &mut usize, candidate: Candidate) -> usize {
    // Binary search for insertion point
    let pos = retset[..*pool_size]
        .binary_search_by(|probe| {
            probe
                .distance
                .partial_cmp(&candidate.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or_else(|x| x);

    // If pool is full and candidate is worse than all existing, don't insert
    if pos >= retset.len() {
        return pos;
    }

    // Make room if needed
    if *pool_size >= retset.len() {
        retset.resize_with(retset.len() * 2, || Candidate {
            id: 0,
            distance: f32::MAX,
            flag: false,
            visited: false,
        });
    }

    // Shift elements right
    let end = (*pool_size).min(retset.len() - 1);
    for i in (pos..end).rev() {
        retset.swap(i, i + 1);
    }
    retset[pos] = candidate;

    pos
}

/// Core pipelined search function implementing the PipeANN algorithm.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pipe_search<T: VectorRepr>(
    reader: &mut PipelinedReader,
    pq_data: &PQData,
    distance_comparer: &T::Distance,
    query: &[T],
    k: usize,
    search_l: usize,
    beam_width: usize,
    medoid: u32,
    dims: usize,
    node_len: u64,
    num_nodes_per_sector: u64,
    block_size: usize,
    fp_vector_len: u64,
    pq_scratch: &mut PQScratch,
    relaxed_monotonicity_l: Option<usize>,
    metric: Metric,
) -> ANNResult<PipeSearchResult> {
    let timer = Instant::now();
    let mut io_count: u32 = 0;
    let mut comparisons: u32 = 0;
    let mut hops: u32 = 0;
    let mut io_time = std::time::Duration::ZERO;
    let mut cpu_time = std::time::Duration::ZERO;

    let num_pq_chunks = pq_data.get_num_chunks();
    let pq_compressed = pq_data.pq_compressed_data().get_data();

    let num_sectors_per_node = if num_nodes_per_sector > 0 {
        1
    } else {
        (node_len as usize).div_ceil(block_size)
    };

    // Prepare PQ distance table for the query and compute PQ distance to medoid
    pq_scratch.set(dims, query, 1.0)?;
    let medoid_ids = [medoid];
    quantizer_preprocess(pq_scratch, pq_data, metric, &medoid_ids)?;
    let medoid_dist = pq_scratch.aligned_dist_scratch[0];

    // Initialize candidate pool
    let initial_cap = search_l * 2 + 10;
    let mut retset: Vec<Candidate> = Vec::with_capacity(initial_cap);
    for _ in 0..initial_cap {
        retset.push(Candidate {
            id: 0,
            distance: f32::MAX,
            flag: false,
            visited: false,
        });
    }
    retset[0] = Candidate {
        id: medoid,
        distance: medoid_dist,
        flag: true,
        visited: false,
    };
    let mut cur_list_size: usize = 1;

    let mut visited = HashSet::new();
    visited.insert(medoid);

    let mut full_retset: Vec<(u32, f32)> = Vec::with_capacity(search_l * 2);

    let mut on_flight_ios: VecDeque<InFlightIo> = VecDeque::new();
    let mut id_buf_map: HashMap<u32, LoadedNode> = HashMap::new();
    let mut next_slot_id: usize = 0;

    let mut cur_beam_width: usize = beam_width.min(4);
    let mut max_marker: usize = 0;
    let mut cur_n_in: usize = 0;
    let mut cur_tot: usize = 0;
    let mut converge_size: i64 = -1;

    // Closure-like helpers implemented as inline functions via the loop body

    // Submit initial reads
    {
        let io_start = Instant::now();
        let to_send = cur_beam_width.saturating_sub(on_flight_ios.len());
        let mut n_sent = 0;
        let mut marker = 0;
        while marker < cur_list_size && n_sent < to_send {
            if retset[marker].flag && !id_buf_map.contains_key(&retset[marker].id) {
                // Send read for this candidate
                let vid = retset[marker].id;
                retset[marker].flag = false;

                let sector_idx =
                    node_sector_index(vid, num_nodes_per_sector, num_sectors_per_node);
                let sector_offset = sector_idx * block_size as u64;
                let slot_id = next_slot_id % max_slots(beam_width);
                reader.submit_read(sector_offset, slot_id)?;
                on_flight_ios.push_back(InFlightIo {
                    vertex_id: vid,
                    slot_id,
                });
                next_slot_id = (next_slot_id + 1) % max_slots(beam_width);
                io_count += 1;
                n_sent += 1;
            }
            marker += 1;
        }
        io_time += io_start.elapsed();
    }

    // Main search loop
    loop {
        // Check if there's a first unvisited candidate
        let first_unvisited = retset[..cur_list_size]
            .iter()
            .position(|c| !c.visited);
        if first_unvisited.is_none() {
            break;
        }

        // Poll completions
        let io_poll_start = Instant::now();
        let completed_slots = reader.poll_completions()?;
        io_time += io_poll_start.elapsed();
        let mut n_in: usize = 0;
        let mut n_out: usize = 0;

        // Process completed IOs: move from on_flight to id_buf_map
        if !completed_slots.is_empty() {
            let completed_set: HashSet<usize> = completed_slots.into_iter().collect();
            let mut remaining = VecDeque::new();
            while let Some(io) = on_flight_ios.pop_front() {
                if completed_set.contains(&io.slot_id) {
                    let sector_buf = reader.get_slot_buf(io.slot_id);
                    let node = parse_node(
                        sector_buf,
                        io.vertex_id,
                        num_nodes_per_sector,
                        node_len,
                        fp_vector_len,
                    );
                    // Track convergence: is this node still in the top of retset?
                    if cur_list_size > 0 {
                        let last_dist = retset[cur_list_size - 1].distance;
                        // Find this node's PQ distance in retset
                        let in_pool = retset[..cur_list_size]
                            .iter()
                            .any(|c| c.id == io.vertex_id && c.distance <= last_dist);
                        if in_pool {
                            n_in += 1;
                        } else {
                            n_out += 1;
                        }
                    }
                    id_buf_map.insert(io.vertex_id, node);
                } else {
                    remaining.push_back(io);
                }
            }
            on_flight_ios = remaining;
        }

        // Track convergence and adjust beam width
        if max_marker >= 5 && (n_in + n_out) > 0 {
            cur_n_in += n_in;
            cur_tot += n_in + n_out;
            const WASTE_THRESHOLD: f64 = 0.1;
            if (cur_tot - cur_n_in) as f64 / cur_tot as f64 <= WASTE_THRESHOLD {
                cur_beam_width = (cur_beam_width + 1).max(4).min(beam_width);
            }
            if let Some(rm_l) = relaxed_monotonicity_l {
                if rm_l > 0 && converge_size < 0 {
                    converge_size = full_retset.len() as i64;
                }
            }
        }

        // Check relaxed monotonicity termination
        if let Some(rm_l) = relaxed_monotonicity_l {
            if rm_l > 0
                && converge_size >= 0
                && full_retset.len() >= (converge_size as usize) + rm_l
            {
                break;
            }
        }

        // Submit more reads if room
        if on_flight_ios.len() < cur_beam_width {
            let io_submit_start = Instant::now();
            let to_send = 1;
            let mut n_sent = 0;
            let mut marker = 0;
            while marker < cur_list_size && n_sent < to_send {
                let c = &retset[marker];
                if c.flag && !id_buf_map.contains_key(&c.id) {
                    let vid = retset[marker].id;
                    retset[marker].flag = false;

                    let sector_idx =
                        node_sector_index(vid, num_nodes_per_sector, num_sectors_per_node);
                    let sector_offset = sector_idx * block_size as u64;
                    let slot_id = next_slot_id % max_slots(beam_width);
                    reader.submit_read(sector_offset, slot_id)?;
                    on_flight_ios.push_back(InFlightIo {
                        vertex_id: vid,
                        slot_id,
                    });
                    next_slot_id = (next_slot_id + 1) % max_slots(beam_width);
                    io_count += 1;
                    n_sent += 1;
                }
                marker += 1;
            }
            io_time += io_submit_start.elapsed();
        }

        // calc_best_node: find one node in id_buf_map that's in retset and unvisited, process it
        let cpu_start = Instant::now();
        let mut best_marker = cur_list_size;
        let calc_limit = cur_list_size;
        #[allow(clippy::needless_range_loop)]
        for i in 0..calc_limit {
            if !retset[i].visited && id_buf_map.contains_key(&retset[i].id) {
                retset[i].flag = false;
                retset[i].visited = true;
                let vid = retset[i].id;
                hops += 1;

                if let Some(node) = id_buf_map.get(&vid) {
                    // Compute full-precision distance
                    let fp_vec: &[T] = bytemuck::cast_slice(&node.fp_vector);
                    let fp_dist = distance_comparer.evaluate_similarity(query, fp_vec);
                    full_retset.push((vid, fp_dist));

                    // Expand neighbors
                    let mut nbors_to_compute: Vec<u32> = Vec::new();
                    for &nbr_id in &node.adjacency_list {
                        if visited.insert(nbr_id) {
                            nbors_to_compute.push(nbr_id);
                        }
                    }

                    if !nbors_to_compute.is_empty() {
                        comparisons += nbors_to_compute.len() as u32;
                        // Compute PQ distances for unvisited neighbors
                        compute_pq_distance(
                            &nbors_to_compute,
                            num_pq_chunks,
                            &pq_scratch.aligned_pqtable_dist_scratch,
                            pq_compressed,
                            &mut pq_scratch.aligned_pq_coord_scratch,
                            &mut pq_scratch.aligned_dist_scratch,
                        )?;

                        let mut nk = cur_list_size;
                        for (m, &nbr_id) in nbors_to_compute.iter().enumerate() {
                            let nbr_dist = pq_scratch.aligned_dist_scratch[m];
                            if cur_list_size == search_l
                                && nbr_dist >= retset[cur_list_size - 1].distance
                            {
                                continue;
                            }
                            let nn = Candidate {
                                id: nbr_id,
                                distance: nbr_dist,
                                flag: true,
                                visited: false,
                            };
                            let r = insert_into_pool(&mut retset, &mut cur_list_size, nn);
                            if cur_list_size < search_l {
                                cur_list_size += 1;
                            }
                            if r < nk {
                                nk = r;
                            }
                        }
                    }
                }

                // Find first_unvisited_eager for convergence tracking
                for (j, c) in retset.iter().enumerate().take(cur_list_size) {
                    if !c.visited && c.flag && !id_buf_map.contains_key(&c.id) {
                        best_marker = j;
                        break;
                    }
                }
                break;
            }
        }
        max_marker = max_marker.max(best_marker);
        cpu_time += cpu_start.elapsed();
    }

    // In relaxed monotonicity mode: drain remaining IOs and process unvisited nodes
    if relaxed_monotonicity_l.is_some_and(|l| l > 0) {
        // Drain all in-flight IOs
        while !on_flight_ios.is_empty() {
            let completed_slots = reader.poll_completions()?;
            if !completed_slots.is_empty() {
                let completed_set: HashSet<usize> = completed_slots.into_iter().collect();
                let mut remaining = VecDeque::new();
                while let Some(io) = on_flight_ios.pop_front() {
                    if completed_set.contains(&io.slot_id) {
                        let sector_buf = reader.get_slot_buf(io.slot_id);
                        let node = parse_node(
                            sector_buf,
                            io.vertex_id,
                            num_nodes_per_sector,
                            node_len,
                            fp_vector_len,
                        );
                        id_buf_map.insert(io.vertex_id, node);
                    } else {
                        remaining.push_back(io);
                    }
                }
                on_flight_ios = remaining;
            }
        }
        // Process remaining unvisited nodes
        for c in retset.iter_mut().take(cur_list_size) {
            if !c.visited {
                if let Some(node) = id_buf_map.get(&c.id) {
                    c.visited = true;
                    let fp_vec: &[T] = bytemuck::cast_slice(&node.fp_vector);
                    let fp_dist = distance_comparer.evaluate_similarity(query, fp_vec);
                    full_retset.push((c.id, fp_dist));
                }
            }
        }
    }

    // Sort full_retset and return top-k
    full_retset.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Deduplicate
    let mut ids = Vec::with_capacity(k);
    let mut distances = Vec::with_capacity(k);
    let mut seen = HashSet::new();
    for (id, dist) in &full_retset {
        if ids.len() >= k {
            break;
        }
        if seen.insert(*id) {
            ids.push(*id);
            distances.push(*dist);
        }
    }

    let total_us = timer.elapsed().as_micros();

    Ok(PipeSearchResult {
        ids,
        distances,
        stats: PipeSearchStats {
            total_us,
            io_us: io_time.as_micros(),
            cpu_us: cpu_time.as_micros(),
            io_count,
            comparisons,
            hops,
        },
    })
}

/// Max buffer slots to use, based on beam width.
#[inline]
fn max_slots(beam_width: usize) -> usize {
    (beam_width * 2).clamp(16, super::pipelined_reader::MAX_IO_CONCURRENCY)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers ----

    fn make_candidate(id: u32, distance: f32) -> Candidate {
        Candidate {
            id,
            distance,
            flag: true,
            visited: false,
        }
    }

    fn empty_pool(cap: usize) -> Vec<Candidate> {
        (0..cap)
            .map(|_| Candidate {
                id: 0,
                distance: f32::MAX,
                flag: false,
                visited: false,
            })
            .collect()
    }

    fn pool_distances(retset: &[Candidate], pool_size: usize) -> Vec<f32> {
        retset[..pool_size].iter().map(|c| c.distance).collect()
    }

    fn pool_ids(retset: &[Candidate], pool_size: usize) -> Vec<u32> {
        retset[..pool_size].iter().map(|c| c.id).collect()
    }

    // ---- insert_into_pool tests ----

    #[test]
    fn test_insert_into_pool_empty() {
        let mut retset = empty_pool(8);
        let mut pool_size: usize = 0;
        let pos = insert_into_pool(&mut retset, &mut pool_size, make_candidate(1, 0.5));
        // Pool was empty, should insert at position 0.
        assert_eq!(pos, 0);
        assert_eq!(retset[0].id, 1);
        assert_eq!(retset[0].distance, 0.5);
    }

    #[test]
    fn test_insert_into_pool_front() {
        let mut retset = empty_pool(8);
        let mut pool_size: usize = 0;
        insert_into_pool(&mut retset, &mut pool_size, make_candidate(10, 5.0));
        pool_size += 1;
        insert_into_pool(&mut retset, &mut pool_size, make_candidate(20, 3.0));
        pool_size += 1;

        // Candidate with lowest distance should go to front.
        let pos = insert_into_pool(&mut retset, &mut pool_size, make_candidate(30, 1.0));
        pool_size += 1;
        assert_eq!(pos, 0);
        assert_eq!(pool_ids(&retset, pool_size), vec![30, 20, 10]);
        assert_eq!(pool_distances(&retset, pool_size), vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_insert_into_pool_end() {
        let mut retset = empty_pool(8);
        let mut pool_size: usize = 0;
        insert_into_pool(&mut retset, &mut pool_size, make_candidate(1, 1.0));
        pool_size += 1;
        insert_into_pool(&mut retset, &mut pool_size, make_candidate(2, 2.0));
        pool_size += 1;

        let pos = insert_into_pool(&mut retset, &mut pool_size, make_candidate(3, 10.0));
        pool_size += 1;
        assert_eq!(pos, 2);
        assert_eq!(pool_distances(&retset, pool_size), vec![1.0, 2.0, 10.0]);
    }

    #[test]
    fn test_insert_into_pool_at_capacity_better_candidate() {
        // Capacity = 4, pool full with 4 items. Insert one that is better.
        let mut retset = empty_pool(4);
        let mut pool_size: usize = 0;
        for (id, d) in [(1, 1.0), (2, 3.0), (3, 5.0), (4, 7.0)] {
            insert_into_pool(&mut retset, &mut pool_size, make_candidate(id, d));
            pool_size += 1;
        }
        assert_eq!(pool_size, 4);

        // Pool is at capacity (pool_size == retset.len()), insert a better candidate.
        // insert_into_pool should grow the buffer to make room.
        let pos = insert_into_pool(&mut retset, &mut pool_size, make_candidate(5, 2.0));
        assert_eq!(pos, 1);
        // The pool buffer should have grown and the element is in sorted order.
        assert!(retset.len() >= 5);
        assert_eq!(retset[0].id, 1);
        assert_eq!(retset[1].id, 5);
        assert_eq!(retset[1].distance, 2.0);
    }

    #[test]
    fn test_insert_into_pool_at_capacity_worse_candidate() {
        // Capacity = 4, pool full. Insert a candidate worse than all existing.
        let mut retset = empty_pool(4);
        let mut pool_size: usize = 0;
        for (id, d) in [(1, 1.0), (2, 3.0), (3, 5.0), (4, 7.0)] {
            insert_into_pool(&mut retset, &mut pool_size, make_candidate(id, d));
            pool_size += 1;
        }

        // Candidate distance 100.0 is worse than the sentinel f32::MAX only if
        // pool_size == retset.len(), the function grows the buffer. Verify sorted order.
        let pos = insert_into_pool(&mut retset, &mut pool_size, make_candidate(99, 100.0));
        // pos should be 4 (after last real element); the buffer was grown.
        assert_eq!(pos, 4);
    }

    #[test]
    fn test_insert_into_pool_maintains_sort_order() {
        let mut retset = empty_pool(16);
        let mut pool_size: usize = 0;
        let distances = [5.0, 1.0, 3.0, 7.0, 2.0, 6.0, 4.0];
        for (i, &d) in distances.iter().enumerate() {
            insert_into_pool(&mut retset, &mut pool_size, make_candidate(i as u32, d));
            pool_size += 1;
        }
        let dists = pool_distances(&retset, pool_size);
        for w in dists.windows(2) {
            assert!(w[0] <= w[1], "Pool not sorted: {:?}", dists);
        }
        assert_eq!(dists, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    // ---- parse_node tests ----

    /// Build a fake sector buffer containing a single node at a given offset.
    fn build_sector_buf(
        offset: usize,
        fp_vector: &[u8],
        neighbors: &[u32],
        total_size: usize,
    ) -> Vec<u8> {
        let node_len = fp_vector.len() + 4 + neighbors.len() * 4;
        let mut buf = vec![0u8; total_size.max(offset + node_len)];
        buf[offset..offset + fp_vector.len()].copy_from_slice(fp_vector);
        let neigh_offset = offset + fp_vector.len();
        LittleEndian::write_u32(
            &mut buf[neigh_offset..neigh_offset + 4],
            neighbors.len() as u32,
        );
        for (i, &n) in neighbors.iter().enumerate() {
            let start = neigh_offset + 4 + i * 4;
            LittleEndian::write_u32(&mut buf[start..start + 4], n);
        }
        buf
    }

    #[test]
    fn test_parse_node_basic() {
        let fp_vec = vec![1u8, 2, 3, 4, 5, 6, 7, 8]; // 8-byte vector
        let neighbors = vec![10u32, 20, 30];
        let fp_vector_len = fp_vec.len() as u64;
        let node_len = fp_vector_len + 4 + 3 * 4; // vec + count + 3 neighbors

        let buf = build_sector_buf(0, &fp_vec, &neighbors, 4096);
        let node = parse_node(&buf, 0, 1, node_len, fp_vector_len);

        assert_eq!(node.fp_vector, fp_vec);
        assert_eq!(node.adjacency_list, vec![10, 20, 30]);
    }

    #[test]
    fn test_parse_node_multi_node_per_sector() {
        let fp_vector_len = 8u64;
        let node_len = fp_vector_len + 4 + 2 * 4; // 8-byte vec, 2 neighbors
        let num_nodes_per_sector = 4u64;

        // Place 4 nodes in the sector, each with different data.
        let mut buf = vec![0u8; 4096];
        for node_idx in 0u32..4 {
            let offset = (node_idx as u64 * node_len) as usize;
            let fp_vec: Vec<u8> = (0..8).map(|b| b + (node_idx as u8) * 10).collect();
            let neighbors = vec![100 + node_idx, 200 + node_idx];
            let partial = build_sector_buf(0, &fp_vec, &neighbors, node_len as usize);
            buf[offset..offset + node_len as usize]
                .copy_from_slice(&partial[..node_len as usize]);
        }

        // Parse node at index 2 (vertex_id=2 within same sector)
        let node = parse_node(&buf, 2, num_nodes_per_sector, node_len, fp_vector_len);
        let expected_fp: Vec<u8> = (0..8).map(|b| b + 20).collect();
        assert_eq!(node.fp_vector, expected_fp);
        assert_eq!(node.adjacency_list, vec![102, 202]);
    }

    #[test]
    fn test_parse_node_zero_neighbors() {
        let fp_vec = vec![42u8; 16];
        let fp_vector_len = 16u64;
        let neighbors: Vec<u32> = vec![];
        let node_len = fp_vector_len + 4; // vec + count only

        let buf = build_sector_buf(0, &fp_vec, &neighbors, 4096);
        let node = parse_node(&buf, 0, 1, node_len, fp_vector_len);

        assert_eq!(node.fp_vector, vec![42u8; 16]);
        assert!(node.adjacency_list.is_empty());
    }

    // ---- node_sector_index tests ----

    #[test]
    fn test_node_sector_index_multi_node_per_sector() {
        let num_nodes_per_sector = 4u64;
        let num_sectors_per_node = 1usize;

        // Matches disk_sector_graph.rs: sector = 1 + vertex_id / num_nodes_per_sector
        assert_eq!(node_sector_index(0, num_nodes_per_sector, num_sectors_per_node), 1);
        assert_eq!(node_sector_index(3, num_nodes_per_sector, num_sectors_per_node), 1);
        assert_eq!(node_sector_index(4, num_nodes_per_sector, num_sectors_per_node), 2);
        assert_eq!(node_sector_index(5, num_nodes_per_sector, num_sectors_per_node), 2);
        assert_eq!(node_sector_index(7, num_nodes_per_sector, num_sectors_per_node), 2);
        assert_eq!(node_sector_index(8, num_nodes_per_sector, num_sectors_per_node), 3);
        assert_eq!(node_sector_index(1023, num_nodes_per_sector, num_sectors_per_node), 256);
        assert_eq!(node_sector_index(1024, num_nodes_per_sector, num_sectors_per_node), 257);
    }

    #[test]
    fn test_node_sector_index_multi_sector_per_node() {
        let num_nodes_per_sector = 0u64;
        let num_sectors_per_node = 2usize;

        // sector = 1 + vertex_id * num_sectors_per_node
        assert_eq!(node_sector_index(0, num_nodes_per_sector, num_sectors_per_node), 1);
        assert_eq!(node_sector_index(3, num_nodes_per_sector, num_sectors_per_node), 7);
        assert_eq!(node_sector_index(4, num_nodes_per_sector, num_sectors_per_node), 9);
        assert_eq!(node_sector_index(5, num_nodes_per_sector, num_sectors_per_node), 11);
        assert_eq!(node_sector_index(7, num_nodes_per_sector, num_sectors_per_node), 15);
        assert_eq!(node_sector_index(8, num_nodes_per_sector, num_sectors_per_node), 17);
        assert_eq!(node_sector_index(1023, num_nodes_per_sector, num_sectors_per_node), 2047);
        assert_eq!(node_sector_index(1024, num_nodes_per_sector, num_sectors_per_node), 2049);
    }

    // ---- node_offset_in_sector tests ----

    #[test]
    fn test_node_offset_multi_node_per_sector() {
        let num_nodes_per_sector = 4u64;
        let node_len = 256u64;

        // offset = (vertex_id % num_nodes_per_sector) * node_len
        assert_eq!(node_offset_in_sector(0, num_nodes_per_sector, node_len), 0);
        assert_eq!(node_offset_in_sector(1, num_nodes_per_sector, node_len), 256);
        assert_eq!(node_offset_in_sector(2, num_nodes_per_sector, node_len), 512);
        assert_eq!(node_offset_in_sector(3, num_nodes_per_sector, node_len), 768);
        assert_eq!(node_offset_in_sector(4, num_nodes_per_sector, node_len), 0); // wraps
        assert_eq!(node_offset_in_sector(5, num_nodes_per_sector, node_len), 256);
    }

    #[test]
    fn test_node_offset_multi_sector_per_node() {
        // When num_nodes_per_sector is 0 (multi-sector), offset is always 0.
        assert_eq!(node_offset_in_sector(0, 0, 8192), 0);
        assert_eq!(node_offset_in_sector(5, 0, 8192), 0);
        assert_eq!(node_offset_in_sector(100, 0, 8192), 0);
    }

    // ---- max_slots tests ----

    #[test]
    fn test_max_slots() {
        // beam_width * 2 clamped to [16, MAX_IO_CONCURRENCY]
        assert_eq!(max_slots(1), 16); // 2 clamped up to 16
        assert_eq!(max_slots(8), 16);
        assert_eq!(max_slots(16), 32);
        assert_eq!(max_slots(64), 128);
        assert_eq!(max_slots(100), 128); // 200 clamped down to 128
    }
}
