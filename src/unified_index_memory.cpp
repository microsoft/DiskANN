// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "boost/dynamic_bitset.hpp"

#include "unified_index_memory.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>

#include "ann_exception.h"
#include "distance.h"
#include "filter_match_proxy.h"
#include "neighbor.h"
#include "percentile_stats.h"
#include "unified_index_io.h"
#include "unified_node_store.h"
#include "utils.h"

#ifndef MAX_POINTS_FOR_USING_BITSET
#define MAX_POINTS_FOR_USING_BITSET 10000000
#endif

namespace diskann
{

template <typename T>
unified_index_memory<T>::unified_index_memory(diskann::Metric metric)
    : unified_index_base<T>(metric), _query_scratch(nullptr)
{
}

template <typename T> unified_index_memory<T>::~unified_index_memory()
{
    if (!_query_scratch.empty())
    {
        ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
        manager.destroy();
    }
}

template <typename T>
void unified_index_memory<T>::load_storage(UnifiedIndexReader &r, const UnifiedLoadContext &ctx)
{
    // Build the resident node store.
    auto store = std::make_unique<unified_node_store_memory<T>>();
    store->load(r, this->_header);
    this->_store = std::move(store);

    _start = static_cast<uint32_t>(this->_header.start_node);
    _max_observed_degree = this->_header.max_degree;

    // Mirror unified_index_ssd: load the medoids region. For unfiltered
    // builds the writer emits a single-entry list (== _start); for filtered
    // builds it emits one per label. Search-time seeding uses these
    // medoids the same way the SSD path does -- pick the closest to query.
    if (this->_header.medoids_len > 0)
    {
        const size_t num = this->_header.medoids_len / sizeof(uint32_t);
        _medoids.resize(num);
        r.load_region(this->_header.medoids_off, this->_header.medoids_len,
                      reinterpret_cast<uint8_t *>(_medoids.data()));
    }
    if (_medoids.empty())
        _medoids.push_back(_start);

    _dist_cmp.reset(get_distance_function<T>(this->_metric));

    init_scratch_pool(ctx.num_threads, ctx.search_l);
}

template <typename T> void unified_index_memory<T>::init_scratch_pool(uint32_t num_threads, uint32_t search_l)
{
    if (num_threads == 0)
        num_threads = 1;

    const size_t dim = static_cast<size_t>(this->_header.dim);
    const size_t aligned_dim = static_cast<size_t>(this->_header.aligned_dim);
    const uint32_t R = _max_observed_degree;
    const uint32_t maxc = 750; // legacy default
    const size_t alignment_factor = _dist_cmp ? _dist_cmp->get_required_alignment() : 8;

    // The unified path doesn't use InMemQueryScratch::_query_label_bitmask --
    // the bitmask match proxy owns its own per-query scratch internally
    // (bitmask_filter_match's 3-arg ctor). So we can skip the per-thread
    // bitmask buffer allocation entirely.
    const size_t bitmask_size = 0;

    std::vector<uint32_t> empty_sellers;
    for (uint32_t i = 0; i < num_threads; ++i)
    {
        auto *s = new InMemQueryScratch<T>(search_l, search_l, R, maxc, dim, aligned_dim, alignment_factor,
                                           empty_sellers,
                                           /*init_pq_scratch=*/false, bitmask_size);
        _query_scratch.push(s);
    }
}

template <typename T>
std::pair<uint32_t, uint32_t> unified_index_memory<T>::iterate_to_fixed_point(
    InMemQueryScratch<T> *scratch, uint32_t L, const T *query, const std::vector<uint32_t> &init_ids,
    filter_match_proxy *match_proxy)
{
    auto *store = static_cast<unified_node_store_memory<T> *>(this->_store.get());
    const uint64_t aligned_dim = this->_header.aligned_dim;

    NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
    best_L_nodes.reserve(L);
    tsl::robin_set<uint32_t> &inserted_into_pool_rs = scratch->inserted_into_pool_rs();
    boost::dynamic_bitset<> &inserted_into_pool_bs = scratch->inserted_into_pool_bs();
    std::vector<uint32_t> &id_scratch = scratch->id_scratch();
    std::vector<float> &dist_scratch = scratch->dist_scratch();
    id_scratch.clear();
    dist_scratch.clear();

    const T *aligned_query = scratch->aligned_query();

    const uint64_t total_num_points = this->_header.npts;
    const bool fast_iterate = total_num_points <= MAX_POINTS_FOR_USING_BITSET;

    if (fast_iterate)
    {
        if (inserted_into_pool_bs.size() < total_num_points)
        {
            auto resize_size = 2 * total_num_points > MAX_POINTS_FOR_USING_BITSET
                                   ? MAX_POINTS_FOR_USING_BITSET
                                   : 2 * total_num_points;
            inserted_into_pool_bs.resize(resize_size);
        }
    }

    auto is_not_visited = [fast_iterate, &inserted_into_pool_bs, &inserted_into_pool_rs](uint32_t id) {
        return fast_iterate ? !inserted_into_pool_bs.test(id)
                            : inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end();
    };
    auto mark_visited = [fast_iterate, &inserted_into_pool_bs, &inserted_into_pool_rs](uint32_t id) {
        if (fast_iterate)
            inserted_into_pool_bs.set(id);
        else
            inserted_into_pool_rs.insert(id);
    };

    uint32_t hops = 0;
    uint32_t cmps = 0;

    for (uint32_t id : init_ids)
    {
        if (id >= total_num_points)
            continue;
        if (match_proxy != nullptr && !match_proxy->contain_filtered_label(id))
            continue;
        if (is_not_visited(id))
        {
            mark_visited(id);
            const T *coords = store->get_coords(id);
            float d = _dist_cmp->compare(aligned_query, coords, static_cast<uint32_t>(aligned_dim));
            best_L_nodes.insert(Neighbor(id, d));
            ++cmps;
        }
    }

    while (best_L_nodes.has_unexpanded_node())
    {
        auto nbr = best_L_nodes.closest_unexpanded();
        const uint32_t n = nbr.id;
        ++hops;

        id_scratch.clear();
        dist_scratch.clear();

        uint32_t deg = 0;
        const uint32_t *nbrs = store->get_neighbors(n, deg);
        for (uint32_t j = 0; j < deg; ++j)
        {
            const uint32_t id = nbrs[j];
            if (id >= total_num_points)
                continue;
            if (!is_not_visited(id))
                continue;
            if (match_proxy != nullptr && !match_proxy->contain_filtered_label(id))
                continue;
            id_scratch.push_back(id);
        }

        for (uint32_t id : id_scratch)
            mark_visited(id);

        dist_scratch.resize(id_scratch.size());
        for (size_t k = 0; k < id_scratch.size(); ++k)
        {
            const T *coords = store->get_coords(id_scratch[k]);
            dist_scratch[k] = _dist_cmp->compare(aligned_query, coords, static_cast<uint32_t>(aligned_dim));
        }
        cmps += static_cast<uint32_t>(id_scratch.size());

        for (size_t k = 0; k < id_scratch.size(); ++k)
        {
            best_L_nodes.insert(Neighbor(id_scratch[k], dist_scratch[k]));
        }
    }

    return {hops, cmps};
}

template <typename T> void unified_index_memory<T>::search_impl(UnifiedSearchContext &ctx)
{
    if (_query_scratch.size() == 0)
    {
        throw ANNException("unified_index_memory::search_impl: scratch pool empty (was load() called?)", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }

    ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
    InMemQueryScratch<T> *scratch = manager.scratch_space();
    scratch->resize_for_new_L(std::max<uint32_t>(ctx.L, static_cast<uint32_t>(ctx.K)));
    scratch->clear();

    const uint64_t dim = this->_header.dim;
    const uint64_t aligned_dim = this->_header.aligned_dim;
    T *aligned_query = scratch->aligned_query();
    std::memset(aligned_query, 0, aligned_dim * sizeof(T));
    if (_dist_cmp && _dist_cmp->preprocessing_required())
    {
        _dist_cmp->preprocess_query(static_cast<const T *>(ctx.query), dim, aligned_query);
    }
    else
    {
        std::memcpy(aligned_query, ctx.query, dim * sizeof(T));
    }

    // Build the label match proxy if the index is filtered, resolving the
    // filter label strings once: resolve_filters yields both the internal
    // label ints (for the proxy) and the per-label medoid seed ids (init_ids
    // below) from a single dictionary probe per label.
    //
    // init_ids / filter_label_ints are thread_local: search_impl runs once per
    // query on a pooled thread, so reusing these buffers across queries avoids a
    // per-call heap allocation (mirrors unified_index_ssd::cached_beam_search).
    // init_ids must be cleared up front because the unfiltered branch below
    // relies on it being empty when no filter is applied.
    std::unique_ptr<filter_match_proxy> proxy;
    thread_local std::vector<uint32_t> init_ids;
    init_ids.clear();
    if (this->_labels && this->_labels->has_labels())
    {
        thread_local std::vector<uint32_t> filter_label_ints;
        this->_labels->resolve_filters(ctx.filter_labels, filter_label_ints, init_ids);
        proxy = this->_labels->make_match_proxy(filter_label_ints);
    }

    // Seed init_ids. Aligned with unified_index_ssd::cached_beam_search:
    // - Unfiltered: pick the single closest medoid from _medoids by
    //   full-vector L2 (memory has all coords resident, so we don't need the
    //   pre-computed centroid array the SSD path uses).
    // - Filtered: one medoid per filter label (the unified format stores
    //   exactly one per label), already resolved above. Per-label medoid
    //   seeding dramatically improves recall on filtered search because the
    //   global start node may not lie within any filter-label cluster.
    auto *store = static_cast<unified_node_store_memory<T> *>(this->_store.get());
    if (init_ids.empty())
    {
        // Unfiltered path -- pick closest of the global medoid set.
        uint32_t best_id = _medoids.empty() ? _start : _medoids[0];
        float best_dist = std::numeric_limits<float>::max();
        for (uint32_t mid : _medoids)
        {
            const T *coords = store->get_coords(mid);
            const float d = _dist_cmp->compare(aligned_query, coords, static_cast<uint32_t>(aligned_dim));
            if (d < best_dist)
            {
                best_dist = d;
                best_id = mid;
            }
        }
        init_ids.push_back(best_id);
    }

    auto [hops, cmps] = iterate_to_fixed_point(scratch, ctx.L, aligned_query, init_ids, proxy.get());

    NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
    size_t pos = 0;
    for (size_t i = 0; i < best_L_nodes.size() && pos < ctx.K; ++i)
    {
        const Neighbor &n = best_L_nodes[i];
        ctx.indices[pos] = static_cast<uint64_t>(n.id);
        if (this->_metric == diskann::Metric::INNER_PRODUCT)
            ctx.distances[pos] = -n.distance;
        else
            ctx.distances[pos] = n.distance;
        ++pos;
    }
    for (; pos < ctx.K; ++pos)
    {
        ctx.indices[pos] = std::numeric_limits<uint64_t>::max();
        ctx.distances[pos] = std::numeric_limits<float>::max();
    }

    if (ctx.stats != nullptr)
    {
        ctx.stats->n_hops = hops;
        ctx.stats->n_cmps = cmps;
    }
}

template <typename T> void unified_index_memory<T>::fill_storage_stats(TableStats &stats) const
{
    // Memory keeps the whole graph region resident: [coords, neighbors] per
    // node. Split it into vector bytes (node_mem_usage) and adjacency bytes
    // (graph_mem_usage), mirroring Index::get_data_size / get_graph_size.
    const auto *store = static_cast<const unified_node_store_memory<T> *>(this->_store.get());
    if (store == nullptr)
        return;
    const uint64_t resident = store->resident_bytes();
    const uint64_t node_bytes = store->num_points() * store->coord_bytes();
    stats.node_mem_usage = node_bytes;
    stats.graph_mem_usage = resident > node_bytes ? resident - node_bytes : 0;
}

template class unified_index_memory<float>;
template class unified_index_memory<uint8_t>;
template class unified_index_memory<int8_t>;

} // namespace diskann
