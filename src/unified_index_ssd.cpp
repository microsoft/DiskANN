// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "boost/dynamic_bitset.hpp"

#include "unified_index_ssd.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <vector>

#include "ann_exception.h"
#include "neighbor.h"
#include "percentile_stats.h"
#include "pq.h"
#include "pq_scratch.h"
#include "unified_index_io.h"
#include "unified_node_store.h"
#include "utils.h"

namespace diskann
{


template <typename T>
unified_index_ssd<T>::unified_index_ssd(std::shared_ptr<AlignedFileReader> reader, diskann::Metric metric)
    : unified_index_base<T>(metric), _reader(std::move(reader)), _thread_data(nullptr)
{
}

template <typename T> unified_index_ssd<T>::~unified_index_ssd()
{
    if (_centroid_data != nullptr)
    {
        aligned_free(_centroid_data);
        _centroid_data = nullptr;
    }
    if (!_thread_data.empty())
    {
        ScratchStoreManager<SSDThreadData<T>> manager(_thread_data);
        manager.destroy();
    }
}

template <typename T>
void unified_index_ssd<T>::load_storage(UnifiedIndexReader &r, const UnifiedLoadContext &ctx)
{
    const UnifiedIndexHeader &h = r.header();
    if (!(h.flags & HAS_PQ))
    {
        throw ANNException("unified_index_ssd::load_storage: SSD load requires HAS_PQ; file lacks PQ regions", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }

    // 1) Construct + load the disk-resident node store.
    auto store = std::make_unique<unified_node_store_ssd<T>>(_reader);
    store->load(r, h);
    this->_store = std::move(store);

    _dist_cmp.reset(get_distance_function<T>(this->_metric));
    _dist_cmp_float.reset(get_distance_function<float>(this->_metric));

    // 2) Load PQ pivots + codes directly from the unified file -- no temp
    //    files, no extra IO beyond the two zero-copy load_region calls.
    std::vector<uint8_t> pq_pivots_blob(h.pq_pivots_len);
    r.load_region(h.pq_pivots_off, h.pq_pivots_len, pq_pivots_blob.data());
    _pq_table.load_pq_centroid_bin_from_memory(pq_pivots_blob.data(), pq_pivots_blob.size(), /*num_chunks=*/0);

    // PQ codes blob format (matches diskann::load_bin<uint8_t>):
    //   [int32 npts][int32 nchunks][uint8 payload[npts * nchunks]]
    // Read the 8-byte header first so we can size _pq_codes exactly, then
    // stream the payload directly into _pq_codes -- zero intermediate copy.
    if (h.pq_codes_len < 2 * sizeof(int32_t))
    {
        throw ANNException("unified_index_ssd::load_storage: PQ codes region truncated (header)", -1, __FUNCSIG__,
                           __FILE__, __LINE__);
    }
    int32_t codes_header[2] = {0, 0};
    r.load_region(h.pq_codes_off, 2 * sizeof(int32_t), reinterpret_cast<uint8_t *>(codes_header));
    const size_t codes_npts = static_cast<size_t>(codes_header[0]);
    const size_t codes_nchunks = static_cast<size_t>(codes_header[1]);

    if (codes_npts != h.npts)
    {
        throw ANNException("unified_index_ssd::load_storage: PQ codes npts mismatch", -1, __FUNCSIG__, __FILE__,
                           __LINE__);
    }
    const size_t payload_bytes = codes_npts * codes_nchunks * sizeof(uint8_t);
    if (2 * sizeof(int32_t) + payload_bytes > h.pq_codes_len)
    {
        throw ANNException("unified_index_ssd::load_storage: PQ codes region truncated (payload)", -1, __FUNCSIG__,
                           __FILE__, __LINE__);
    }
    _pq_codes.resize(payload_bytes);
    r.load_region(h.pq_codes_off + 2 * sizeof(int32_t), payload_bytes, _pq_codes.data());
    _n_chunks = codes_nchunks;

    // 3) Load medoids.
    load_medoids_from_unified(r);

    // 4) Optional: HAS_MAX_BASE_NORM (MIPS rescaling).
    if (h.flags & HAS_MAX_BASE_NORM)
    {
        if (h.max_base_norm_len >= sizeof(float))
            r.load_region(h.max_base_norm_off, sizeof(float), reinterpret_cast<uint8_t *>(&_max_base_norm));
    }

    // 5) Per-thread scratch.
    setup_thread_data(ctx.num_threads);
    _max_nthreads = ctx.num_threads == 0 ? 1 : ctx.num_threads;

    // 6) Centroid data (copy medoid vectors into the centroid array).
    use_medoids_data_as_centroids();

    // 7) Optional cache priming. Reuses an SSDThreadData (slab + IOContext)
    //    from the pool we just built in step 5, so no extra slab allocation.
    //    Seed the BFS from the global medoids AND each label's entry-point
    //    medoid, so filtered-search seeds (and their BFS neighborhoods) get
    //    cached too -- mirroring PQFlashIndex::cache_bfs_levels, which seeds
    //    from both _medoids and _filter_to_medoid_ids. Global medoids go first
    //    so they keep priority under the num_nodes_to_cache cap;
    //    cache_bfs_levels dedups via its visited set, so any overlap between
    //    the two sources is harmless.
    if (ctx.num_nodes_to_cache > 0)
    {
        std::vector<uint32_t> seed_ids = _medoids;
        if (this->_labels && this->_labels->has_labels())
            this->_labels->collect_label_medoids(seed_ids);

        if (!seed_ids.empty())
        {
            auto *ssd_store = static_cast<unified_node_store_ssd<T> *>(this->_store.get());
            ScratchStoreManager<SSDThreadData<T>> manager(_thread_data);
            SSDThreadData<T> *tdata = manager.scratch_space();
            NodeFetchScratch fetch_scratch;
            fetch_scratch.attach_borrowed(tdata->ctx, tdata->scratch.sector_scratch,
                                           defaults::MAX_N_SECTOR_READS * defaults::SECTOR_LEN);
            std::vector<uint32_t> cached_list;
            ssd_store->cache_bfs_levels(seed_ids, ctx.num_nodes_to_cache, cached_list, fetch_scratch);
        }
    }
}

template <typename T> void unified_index_ssd<T>::load_pq_from_unified(UnifiedIndexReader & /*r*/)
{
    // PQ is loaded inline in load_storage via direct in-memory parsing
    // (FixedChunkPQTable::load_pq_centroid_bin_from_memory + manual codes
    // parsing). This stub is retained as a placeholder in case a future
    // refactor moves PQ loading out of load_storage; today it has no body.
}

template <typename T> void unified_index_ssd<T>::load_medoids_from_unified(UnifiedIndexReader &r)
{
    const UnifiedIndexHeader &h = r.header();
    if (h.medoids_len == 0)
        return;
    const size_t num = h.medoids_len / sizeof(uint32_t);
    _medoids.resize(num);
    r.load_region(h.medoids_off, h.medoids_len, reinterpret_cast<uint8_t *>(_medoids.data()));
}

template <typename T> void unified_index_ssd<T>::setup_thread_data(uint64_t nthreads, uint64_t visited_reserve)
{
    if (nthreads == 0)
        nthreads = 1;
    const size_t aligned_dim = static_cast<size_t>(this->_header.aligned_dim);
    std::vector<uint32_t> empty_sellers;

    // OMP-parallel loop so each worker thread registers itself with the
    // AlignedFileReader and we cache the resulting IOContext on the
    // SSDThreadData. Search-time get_nodes() then uses the cached ctx
    // directly -- no mutex, no per-call get_ctx lookup.
#pragma omp parallel for num_threads(static_cast<int>(nthreads))
    for (int64_t t = 0; t < static_cast<int64_t>(nthreads); ++t)
    {
#pragma omp critical
        {
            auto *td = new SSDThreadData<T>(aligned_dim, visited_reserve, empty_sellers);
            _reader->register_thread();
            td->ctx = _reader->get_ctx();
            _thread_data.push(td);
        }
    }
}

template <typename T> void unified_index_ssd<T>::use_medoids_data_as_centroids()
{
    if (_medoids.empty())
        return;
    const size_t aligned_dim = static_cast<size_t>(this->_header.aligned_dim);
    if (_centroid_data != nullptr)
    {
        aligned_free(_centroid_data);
        _centroid_data = nullptr;
    }
    const size_t bytes = _medoids.size() * aligned_dim * sizeof(float);
    alloc_aligned(reinterpret_cast<void **>(&_centroid_data), bytes, 32);
    std::memset(_centroid_data, 0, bytes);

    auto *ssd_store = static_cast<unified_node_store_ssd<T> *>(this->_store.get());

    // Load-time path: borrow an SSDThreadData (slab + registered IOContext)
    // from the pool that setup_thread_data just built. Avoids allocating a
    // fresh slab just to drop it immediately.
    ScratchStoreManager<SSDThreadData<T>> manager(_thread_data);
    SSDThreadData<T> *tdata = manager.scratch_space();
    NodeFetchScratch scratch;
    scratch.attach_borrowed(tdata->ctx, tdata->scratch.sector_scratch,
                             defaults::MAX_N_SECTOR_READS * defaults::SECTOR_LEN);

    std::vector<uint64_t> ids(1);
    std::vector<NodeView<T>> views;
    for (size_t i = 0; i < _medoids.size(); ++i)
    {
        ids[0] = _medoids[i];
        ssd_store->get_nodes(ids, scratch, views);
        // Convert T -> float (lossless for float; promote for int8/uint8).
        const T *src = views[0].coords;
        float *dst = _centroid_data + i * aligned_dim;
        for (size_t j = 0; j < aligned_dim; ++j)
            dst[j] = static_cast<float>(src[j]);
    }
}

template <typename T> void unified_index_ssd<T>::search_impl(UnifiedSearchContext &ctx)
{
    const uint32_t beam_width = ctx.beam_width.value_or(4);
    const uint32_t io_limit = ctx.io_limit.value_or(std::numeric_limits<uint32_t>::max());
    cached_beam_search(static_cast<const T *>(ctx.query), ctx.K, ctx.L, ctx.indices, ctx.distances, beam_width,
                       /*filter_label_strings=*/ctx.filter_labels, io_limit, ctx.stats, ctx.debug_info);
}

template <typename T>
void unified_index_ssd<T>::cached_beam_search(const T *query, uint64_t K, uint64_t L, uint64_t *indices,
                                              float *distances, uint32_t beam_width,
                                              const std::vector<std::string> &filter_label_strings,
                                              uint32_t io_limit, QueryStats *stats,
                                              DebugTraversalInfo * /*debug_info*/)
{
    const uint64_t aligned_dim = this->_header.aligned_dim;
    const uint64_t dim = this->_header.dim;
    auto *ssd_store = static_cast<unified_node_store_ssd<T> *>(this->_store.get());

    // Borrow per-thread scratch.
    ScratchStoreManager<SSDThreadData<T>> manager(_thread_data);
    SSDThreadData<T> *tdata = manager.scratch_space();
    tdata->scratch.reset();

    // Prepare aligned query (typed T) and aligned float query (for PQ table).
    T *aligned_query_T = tdata->scratch.aligned_query_T();
    std::memset(aligned_query_T, 0, aligned_dim * sizeof(T));
    if (_dist_cmp && _dist_cmp->preprocessing_required())
        _dist_cmp->preprocess_query(query, dim, aligned_query_T);
    else
        std::memcpy(aligned_query_T, query, dim * sizeof(T));

    PQScratch<T> *pq_scratch = tdata->scratch.pq_scratch();
    float *query_float = pq_scratch->aligned_query_float;
    float *query_rotated = pq_scratch->rotated_query;
    for (size_t i = 0; i < dim; ++i)
    {
        query_float[i] = static_cast<float>(query[i]);
        query_rotated[i] = static_cast<float>(query[i]);
    }
    _pq_table.preprocess_query(query_rotated);
    float *pq_dists = pq_scratch->aligned_pqtable_dist_scratch;
    _pq_table.populate_chunk_distances(query_rotated, pq_dists);
    float *dist_scratch = pq_scratch->aligned_dist_scratch;
    uint8_t *pq_coord_scratch = pq_scratch->aligned_pq_coord_scratch;

    auto compute_pq_dists = [this, pq_coord_scratch, pq_dists](const uint32_t *ids, uint64_t n_ids, float *out) {
        diskann::aggregate_coords(ids, n_ids, _pq_codes.data(), _n_chunks, pq_coord_scratch);
        diskann::pq_dist_lookup(pq_coord_scratch, n_ids, _n_chunks, pq_dists, out);
    };

    NeighborPriorityQueue &retset = tdata->scratch.retset;
    retset.reserve(static_cast<uint32_t>(L));
    std::vector<Neighbor> &full_retset = tdata->scratch.full_retset;
    tsl::robin_set<uint64_t> &visited = tdata->scratch.visited;

    // Build filter proxy if applicable. Resolve the filter label strings a
    // single time here: resolve_filters returns both the internal label ints
    // (consumed by make_match_proxy) and the per-label medoid seed ids
    // (consumed by the filtered-seeding branch below), so the label dictionary
    // is probed once per label instead of once for the proxy and again for the
    // init ids.
    std::unique_ptr<filter_match_proxy> proxy;
    thread_local std::vector<uint32_t> filter_label_ints;
    thread_local std::vector<uint32_t> filter_init_ids;
    if (this->_labels && this->_labels->has_labels())
    {
        this->_labels->resolve_filters(filter_label_strings, filter_label_ints, filter_init_ids);
        proxy = this->_labels->make_match_proxy(filter_label_ints);
    }

    // Seed retset. Branches on filtered vs. unfiltered, mirroring the legacy
    // PQFlashIndex::cached_beam_search (src/pq_flash_index.cpp:1329-1377).
    //
    // Unfiltered: pick the single closest medoid by float-centroid distance
    // and seed with it.
    //
    // Filtered: per-label medoid seeding. For each filter label, walk all of
    // its per-label medoids, pick the closest by PQ distance (no float
    // centroid data for filtered medoids), and seed retset+visited with that
    // pick. One seed per label. Per-label medoids come from resolve_filters
    // above (one dictionary probe shared with the match proxy).
    if (proxy == nullptr)
    {
        uint32_t best_medoid = _medoids.empty() ? 0 : _medoids[0];
        float best_medoid_dist = std::numeric_limits<float>::max();
        for (size_t i = 0; i < _medoids.size(); ++i)
        {
            float d = _dist_cmp_float->compare(query_float, _centroid_data + i * aligned_dim,
                                                static_cast<uint32_t>(aligned_dim));
            if (d < best_medoid_dist)
            {
                best_medoid_dist = d;
                best_medoid = _medoids[i];
            }
        }
        compute_pq_dists(&best_medoid, 1, dist_scratch);
        retset.insert(Neighbor(best_medoid, dist_scratch[0]));
        visited.insert(best_medoid);
    }
    else
    {
        // filter_init_ids was populated by resolve_filters above: one medoid
        // per filter label.
        for (uint32_t mid : filter_init_ids)
        {
            // visited dedup: a medoid id may repeat across filter labels.
            if (visited.insert(mid).second)
            {
                compute_pq_dists(&mid, 1, dist_scratch);
                retset.insert(Neighbor(mid, dist_scratch[0]));
            }
        }
    }

    uint32_t num_ios = 0;
    uint32_t cmps = 0;
    uint32_t hops = 0;
    // Zero-allocation fetch scratch: borrow the per-thread sector buffer +
    // pre-registered IOContext from SSDThreadData. The slab is 2 MB
    // (MAX_N_SECTOR_READS * SECTOR_LEN), allocated once at load time.
    NodeFetchScratch fetch_scratch;
    fetch_scratch.attach_borrowed(tdata->ctx, tdata->scratch.sector_scratch,
                                   defaults::MAX_N_SECTOR_READS * defaults::SECTOR_LEN);
    // Per-thread scratch reused across hops and across calls. Capacity grows
    // once to the worst-case beam_width that this thread ever sees.
    thread_local std::vector<uint64_t> beam_ids;
    thread_local std::vector<NodeView<T>> beam_views;
    beam_ids.clear();
    beam_views.clear();

    while (retset.has_unexpanded_node() && num_ios < io_limit)
    {
        beam_ids.clear();
        uint32_t num_seen = 0;
        while (retset.has_unexpanded_node() && beam_ids.size() < beam_width && num_seen < beam_width)
        {
            auto nbr = retset.closest_unexpanded();
            beam_ids.push_back(nbr.id);
            ++num_seen;
        }
        if (beam_ids.empty())
            break;

        // One batched IO per beam (the store handles cache hits internally).
        ssd_store->get_nodes(beam_ids, fetch_scratch, beam_views);
        ++hops;
        num_ios += static_cast<uint32_t>(beam_ids.size());

        for (size_t bi = 0; bi < beam_ids.size(); ++bi)
        {
            const uint32_t id = static_cast<uint32_t>(beam_ids[bi]);
            const NodeView<T> &view = beam_views[bi];
            const float exact_d = _dist_cmp->compare(aligned_query_T, view.coords,
                                                       static_cast<uint32_t>(aligned_dim));
            full_retset.push_back(Neighbor(id, exact_d));

            // PQ-rank neighbors and admit unvisited ones to the search frontier.
            const uint32_t deg = view.degree;
            if (deg == 0)
                continue;
            compute_pq_dists(view.neighbors, deg, dist_scratch);
            for (uint32_t m = 0; m < deg; ++m)
            {
                const uint32_t nb = view.neighbors[m];
                if (!visited.insert(nb).second)
                    continue;
                if (proxy && !proxy->contain_filtered_label(nb))
                    continue;
                ++cmps;
                retset.insert(Neighbor(nb, dist_scratch[m]));
            }
        }
    }

    // Sort full_retset (exact distances) and write top-K.
    std::sort(full_retset.begin(), full_retset.end());
    const size_t out_count = std::min<size_t>(K, full_retset.size());
    for (size_t i = 0; i < out_count; ++i)
    {
        indices[i] = static_cast<uint64_t>(full_retset[i].id);
        distances[i] = (this->_metric == diskann::Metric::INNER_PRODUCT) ? -full_retset[i].distance
                                                                          : full_retset[i].distance;
    }
    for (size_t i = out_count; i < K; ++i)
    {
        indices[i] = std::numeric_limits<uint64_t>::max();
        distances[i] = std::numeric_limits<float>::max();
    }

    if (stats != nullptr)
    {
        stats->n_hops = hops;
        stats->n_cmps = cmps;
        stats->n_ios = num_ios;
    }
}

template <typename T> void unified_index_ssd<T>::fill_storage_stats(TableStats &stats) const
{
    // SSD keeps the PQ codes resident (npts * n_chunks bytes); the graph lives
    // on disk, so graph_mem_usage stays 0 -- mirrors PQFlashIndex, which sets
    // node_mem_usage = npts * nchunks and never sets graph_mem_usage.
    stats.node_mem_usage = _pq_codes.size();
    stats.graph_mem_usage = 0;
}

template class unified_index_ssd<float>;
template class unified_index_ssd<uint8_t>;
template class unified_index_ssd<int8_t>;

} // namespace diskann
