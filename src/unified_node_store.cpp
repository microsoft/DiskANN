// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "unified_node_store.h"

#include <algorithm>
#include <cstring>
#include <unordered_set>
#include <utility>

#include "ann_exception.h"
#include "unified_index_io.h"
#include "utils.h"

namespace diskann
{

// ---------------------------------------------------------------------------
// NodeFetchScratch
// ---------------------------------------------------------------------------

NodeFetchScratch::NodeFetchScratch() = default;

NodeFetchScratch::NodeFetchScratch(NodeFetchScratch &&other) noexcept
    : requests(std::move(other.requests)), _sector_slab(other._sector_slab), _capacity_bytes(other._capacity_bytes),
      _owns_slab(other._owns_slab), _ctx(other._ctx)
{
    other._sector_slab = nullptr;
    other._capacity_bytes = 0;
    other._owns_slab = false;
    other._ctx = nullptr;
}

NodeFetchScratch &NodeFetchScratch::operator=(NodeFetchScratch &&other) noexcept
{
    if (this != &other)
    {
        if (_owns_slab && _sector_slab != nullptr)
            aligned_free(_sector_slab);
        _sector_slab = other._sector_slab;
        _capacity_bytes = other._capacity_bytes;
        _owns_slab = other._owns_slab;
        _ctx = other._ctx;
        requests = std::move(other.requests);
        other._sector_slab = nullptr;
        other._capacity_bytes = 0;
        other._owns_slab = false;
        other._ctx = nullptr;
    }
    return *this;
}

NodeFetchScratch::~NodeFetchScratch()
{
    if (_owns_slab && _sector_slab != nullptr)
    {
        aligned_free(_sector_slab);
        _sector_slab = nullptr;
    }
}

void NodeFetchScratch::reserve(uint64_t max_batch, uint32_t sectors_per_node)
{
    const uint64_t need = max_batch * static_cast<uint64_t>(sectors_per_node) * defaults::SECTOR_LEN;
    if (_owns_slab && need <= _capacity_bytes)
        return;
    if (_owns_slab && _sector_slab != nullptr)
    {
        aligned_free(_sector_slab);
        _sector_slab = nullptr;
    }
    void *p = nullptr;
    alloc_aligned(&p, need, defaults::SECTOR_LEN);
    _sector_slab = static_cast<char *>(p);
    _capacity_bytes = need;
    _owns_slab = true;
}

void NodeFetchScratch::attach_borrowed(IOContext &ctx, char *external_slab, uint64_t slab_capacity_bytes)
{
    // If we previously owned a slab, free it -- attach is meant to replace.
    if (_owns_slab && _sector_slab != nullptr)
    {
        aligned_free(_sector_slab);
        _sector_slab = nullptr;
    }
    _sector_slab = external_slab;
    _capacity_bytes = slab_capacity_bytes;
    _owns_slab = false;
    _ctx = &ctx;
}

void NodeFetchScratch::set_ctx(IOContext &ctx)
{
    _ctx = &ctx;
}

// ---------------------------------------------------------------------------
// unified_node_store_base<T>
// ---------------------------------------------------------------------------

template <typename T>
void unified_node_store_base<T>::init_geometry(const UnifiedIndexHeader &h, std::vector<uint64_t> offset_table)
{
    _header = h;
    _offsets = std::move(offset_table);
    if (_offsets.size() != _header.npts + 1)
    {
        throw ANNException("unified_node_store_base: offset table size mismatch", -1, __FUNCSIG__, __FILE__,
                           __LINE__);
    }
    _coord_bytes = _header.aligned_dim * sizeof(T);
    // max_node_len upper bound: (max_degree + 1) * uint32 + aligned_dim * T
    // (the +1 mirrors the legacy padding hack the unified format inherits;
    //  see plan notes on the +1 sector for unaligned-straddle safety).
    _max_node_len = (static_cast<uint64_t>(_header.max_degree) + 1u) * sizeof(uint32_t) + _coord_bytes;
}

template <typename T> uint32_t unified_node_store_base<T>::degree(uint64_t id) const
{
    const uint64_t node_bytes = node_byte_length(id);
    if (node_bytes < _coord_bytes)
    {
        throw ANNException("unified_node_store_base: node payload shorter than coords", -1, __FUNCSIG__, __FILE__,
                           __LINE__);
    }
    return static_cast<uint32_t>((node_bytes - _coord_bytes) / sizeof(uint32_t));
}

template <typename T> uint32_t unified_node_store_base<T>::num_sectors_per_node() const
{
    // +1 to absorb worst-case unaligned straddle across sector boundary.
    return static_cast<uint32_t>(DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN) + 1u);
}

// ---------------------------------------------------------------------------
// unified_node_store_memory<T>
// ---------------------------------------------------------------------------

template <typename T>
void unified_node_store_memory<T>::load(UnifiedIndexReader &r, const UnifiedIndexHeader &h)
{
    // Initialise base geometry from header + offset table.
    auto offsets = r.load_offset_table();
    this->init_geometry(h, std::move(offsets));

    // Pull the entire graph region resident -- zero-copy via the direct
    // load_region overload (size is known from the offset table).
    const uint64_t expected = this->_offsets.back();
    if (h.graph_region_len != expected)
    {
        throw ANNException("unified_node_store_memory::load: graph region size != offset table total", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    _packed.resize(static_cast<size_t>(expected));
    r.load_region(h.graph_region_off, h.graph_region_len, _packed.data());
}

template <typename T>
void unified_node_store_memory<T>::get_nodes(const std::vector<uint64_t> &ids, NodeFetchScratch & /*scratch*/,
                                              std::vector<NodeView<T>> &out)
{
    out.clear();
    out.resize(ids.size());
    for (size_t i = 0; i < ids.size(); ++i)
    {
        const uint64_t id = ids[i];
        uint32_t deg = 0;
        out[i].coords = get_coords(id);
        out[i].neighbors = get_neighbors(id, deg);
        out[i].degree = deg;
    }
}

template <typename T> const T *unified_node_store_memory<T>::get_coords(uint64_t id) const
{
    return reinterpret_cast<const T *>(_packed.data() + this->_offsets[id]);
}

template <typename T>
const uint32_t *unified_node_store_memory<T>::get_neighbors(uint64_t id, uint32_t &out_degree) const
{
    const uint64_t coord_bytes = this->_coord_bytes;
    const uint64_t node_bytes = this->_offsets[id + 1] - this->_offsets[id];
    out_degree = static_cast<uint32_t>((node_bytes - coord_bytes) / sizeof(uint32_t));
    return reinterpret_cast<const uint32_t *>(_packed.data() + this->_offsets[id] + coord_bytes);
}

// ---------------------------------------------------------------------------
// unified_node_store_ssd<T> -- skeleton: load/get_nodes/cache* throw not_implemented.
// (Ctor is defined inline in the header to avoid DLL-export gymnastics when
//  the SSD index .cpp constructs the store via std::make_unique.)
// ---------------------------------------------------------------------------

template <typename T> unified_node_store_ssd<T>::~unified_node_store_ssd()
{
    if (_nhood_cache_buf != nullptr)
    {
        delete[] _nhood_cache_buf;
        _nhood_cache_buf = nullptr;
    }
    if (_coord_cache_buf != nullptr)
    {
        delete[] _coord_cache_buf;
        _coord_cache_buf = nullptr;
    }
}

template <typename T>
void unified_node_store_ssd<T>::load(UnifiedIndexReader &r, const UnifiedIndexHeader &h)
{
    // Initialise base geometry from header + offset table.
    auto offsets = r.load_offset_table();
    this->init_geometry(h, std::move(offsets));

    // Open the aligned reader on the unified file. Subsequent get_nodes()
    // calls issue async reads through this handle.
    if (_reader == nullptr)
    {
        throw ANNException("unified_node_store_ssd::load: AlignedFileReader is null", -1, __FUNCSIG__, __FILE__,
                           __LINE__);
    }
    _reader->open(r.path());
}

template <typename T>
void unified_node_store_ssd<T>::get_nodes(const std::vector<uint64_t> &ids, NodeFetchScratch &scratch,
                                          std::vector<NodeView<T>> &out)
{
    out.assign(ids.size(), NodeView<T>{});

    // Per-thread scratch vectors (one capacity grow per thread, reused across
    // calls). Clearing keeps the elements but holds the capacity, so the
    // first call's reserve covers all subsequent calls of similar batch size.
    thread_local std::vector<size_t> miss_indices;
    thread_local std::vector<uint64_t> aligned_starts;
    miss_indices.clear();
    aligned_starts.clear();

    // Pass 1: cache lookups.
    miss_indices.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i)
    {
        const uint32_t id32 = static_cast<uint32_t>(ids[i]);
        auto cit = _coord_cache.find(id32);
        auto nit = _nhood_cache.find(id32);
        if (cit != _coord_cache.end() && nit != _nhood_cache.end())
        {
            out[i].coords = cit->second;
            out[i].degree = nit->second.first;
            out[i].neighbors = nit->second.second;
        }
        else
        {
            miss_indices.push_back(i);
        }
    }

    if (miss_indices.empty())
        return;

    // Pass 2: plan + issue batched IO for misses.
    // Scratch contract on the search hot path: caller must have called
    // attach_borrowed() (with the slab + per-thread IOContext that the
    // index registered at load time) or reserve()+attach the ctx. We do NOT
    // touch the reader's thread-registration map here -- that path uses a
    // mutex which would serialise concurrent searches.
    const uint32_t sectors_per_node = this->num_sectors_per_node();
    const uint64_t bytes_per_node = static_cast<uint64_t>(sectors_per_node) * defaults::SECTOR_LEN;
    const uint64_t need_bytes = static_cast<uint64_t>(miss_indices.size()) * bytes_per_node;
    if (scratch.slab() == nullptr || scratch.slab_capacity() < need_bytes)
    {
        throw ANNException(
            "unified_node_store_ssd::get_nodes: scratch slab too small or unset "
            "(need " +
                std::to_string(need_bytes) + " bytes, have " + std::to_string(scratch.slab_capacity()) +
                "). Call attach_borrowed() or reserve() before search.",
            -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    if (scratch.io_ctx() == nullptr)
    {
        throw ANNException(
            "unified_node_store_ssd::get_nodes: scratch has no IOContext attached. "
            "Call attach_borrowed() with a registered IOContext before search.",
            -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    scratch.requests.clear();
    scratch.requests.reserve(miss_indices.size());

    // Track per-miss sector-aligned start so the decode step can compute the
    // pointer back into the slab.
    aligned_starts.resize(miss_indices.size());

    for (size_t k = 0; k < miss_indices.size(); ++k)
    {
        const size_t i = miss_indices[k];
        const uint64_t id = ids[i];
        const uint64_t raw_start = this->node_disk_offset(id);
        const uint64_t aligned_start = (raw_start / defaults::SECTOR_LEN) * defaults::SECTOR_LEN;
        aligned_starts[k] = aligned_start;

        // Read only the sectors this specific node actually spans, not the
        // worst-case `bytes_per_node`. The resident offset table gives each
        // node's exact byte length, so the minimal sector-aligned window that
        // covers [raw_start, raw_start + node_bytes) is all we need to fetch.
        // For a typical low-degree node this is a single SECTOR_LEN read
        // instead of num_sectors_per_node() sectors, cutting bytes transferred.
        // This exact window is always <= bytes_per_node, so it still fits in
        // this miss's fixed slab slot (slab stride stays bytes_per_node).
        const uint64_t raw_end = raw_start + this->node_byte_length(id);
        const uint64_t aligned_end = DIV_ROUND_UP(raw_end, defaults::SECTOR_LEN) * defaults::SECTOR_LEN;
        const uint64_t read_len = aligned_end - aligned_start;

        AlignedRead req;
        req.offset = aligned_start;
        req.len = read_len;
        req.buf = scratch.slab() + k * bytes_per_node;
        scratch.requests.push_back(req);
    }

    _reader->read(scratch.requests, *scratch.io_ctx());
    ++_io_count;

    // Pass 3: decode each miss from its sector slice.
    const uint64_t coord_bytes = this->coord_bytes();
    for (size_t k = 0; k < miss_indices.size(); ++k)
    {
        const size_t i = miss_indices[k];
        const uint64_t id = ids[i];
        const uint64_t raw_start = this->node_disk_offset(id);
        const uint64_t node_bytes = this->node_byte_length(id);
        const uint64_t aligned_start = aligned_starts[k];
        const uint64_t intra = raw_start - aligned_start;
        const char *node_start = scratch.slab() + k * bytes_per_node + intra;

        out[i].coords = reinterpret_cast<const T *>(node_start);
        out[i].degree = static_cast<uint32_t>((node_bytes - coord_bytes) / sizeof(uint32_t));
        out[i].neighbors = reinterpret_cast<const uint32_t *>(node_start + coord_bytes);
    }
}

// Per-thread registration tracking. AlignedFileReader::register_thread() is
// not idempotent on Windows -- calling it twice throws. We track which threads
// we've already registered.
template <typename T> NodeFetchScratch unified_node_store_ssd<T>::make_fetch_scratch(uint64_t max_batch)
{
    // Idempotent thread registration: AlignedFileReader::register_thread()
    // warns + no-ops on duplicate calls (Windows), so we guard with a
    // thread_local flag to skip both the warning and the lock-acquire that
    // the reader does internally. Cheap; thread_local access is one TLS load.
    thread_local bool s_registered_with_reader = false;
    if (!s_registered_with_reader)
    {
        _reader->register_thread();
        s_registered_with_reader = true;
    }
    IOContext &ctx = _reader->get_ctx();
    NodeFetchScratch s;
    s.reserve(max_batch, this->num_sectors_per_node());
    s.set_ctx(ctx); // keep the self-owned slab; just point at the ctx
    return s;
}

template <typename T>
void unified_node_store_ssd<T>::load_cache_list(const std::vector<uint32_t> &node_list, NodeFetchScratch &scratch)
{
    if (node_list.empty())
        return;
    if (_nhood_cache_buf != nullptr || _coord_cache_buf != nullptr)
    {
        throw ANNException("unified_node_store_ssd::load_cache_list: cache already populated", -1, __FUNCSIG__,
                           __FILE__, __LINE__);
    }

    const uint64_t aligned_dim = this->aligned_dim();
    const uint32_t max_degree = this->max_degree();

    // Allocate cache backing buffers. _nhood_cache_buf stores `max_degree`
    // uint32 slots per cached node; _coord_cache_buf stores aligned_dim T
    // values per cached node.
    const size_t n = node_list.size();
    _nhood_cache_buf = new uint32_t[n * static_cast<size_t>(max_degree)];
    std::memset(_nhood_cache_buf, 0, n * static_cast<size_t>(max_degree) * sizeof(uint32_t));
    _coord_cache_buf = new T[n * aligned_dim];
    std::memset(_coord_cache_buf, 0, n * aligned_dim * sizeof(T));

    // Batch the reads instead of one get_nodes() call per node. get_nodes()
    // resolves a whole batch of misses in a single batched IO, so the number
    // of IOs drops from n to ceil(n / batch). The batch size is bounded by how
    // many worst-case node slots fit in the scratch slab (the same bound
    // get_nodes() enforces internally). node_list ids are unique (callers build
    // it via a visited set / pass distinct ids), so every node in a batch is a
    // cache miss and contributes to the batched read.
    const uint64_t bytes_per_node = static_cast<uint64_t>(this->num_sectors_per_node()) * defaults::SECTOR_LEN;
    size_t max_per_batch = 1;
    if (bytes_per_node > 0 && scratch.slab_capacity() >= bytes_per_node)
        max_per_batch = static_cast<size_t>(scratch.slab_capacity() / bytes_per_node);

    std::vector<uint64_t> id_batch;
    std::vector<NodeView<T>> view_batch;
    for (size_t base = 0; base < n; base += max_per_batch)
    {
        const size_t batch_n = std::min(max_per_batch, n - base);
        id_batch.assign(node_list.begin() + base, node_list.begin() + base + batch_n);

        // All views in the batch point into distinct slab slots and stay valid
        // until the next get_nodes() call, so we can copy them all out here.
        get_nodes(id_batch, scratch, view_batch);

        for (size_t b = 0; b < batch_n; ++b)
        {
            const size_t i = base + b;
            const uint32_t id = node_list[i];

            T *coord_slot = _coord_cache_buf + i * aligned_dim;
            std::memcpy(coord_slot, view_batch[b].coords, aligned_dim * sizeof(T));
            _coord_cache.emplace(id, coord_slot);

            uint32_t *nhood_slot = _nhood_cache_buf + i * static_cast<size_t>(max_degree);
            const uint32_t deg = view_batch[b].degree;
            const uint32_t deg_to_copy = std::min<uint32_t>(deg, max_degree);
            std::memcpy(nhood_slot, view_batch[b].neighbors, deg_to_copy * sizeof(uint32_t));
            _nhood_cache.emplace(id, std::make_pair(deg_to_copy, nhood_slot));
        }
    }
}

template <typename T>
void unified_node_store_ssd<T>::cache_bfs_levels(const std::vector<uint32_t> &seed_nodes, uint64_t num_nodes_to_cache,
                                                  std::vector<uint32_t> &out_node_list, NodeFetchScratch &scratch)
{
    out_node_list.clear();
    if (num_nodes_to_cache == 0 || seed_nodes.empty())
        return;

    std::unordered_set<uint32_t> visited;
    std::vector<uint32_t> frontier = seed_nodes;
    for (uint32_t s : seed_nodes)
    {
        if (s >= this->num_points())
            continue;
        if (visited.insert(s).second)
            out_node_list.push_back(s);
        if (out_node_list.size() >= num_nodes_to_cache)
            break;
    }

    std::vector<uint64_t> id_batch;
    std::vector<NodeView<T>> view_batch;

    while (out_node_list.size() < num_nodes_to_cache && !frontier.empty())
    {
        // Fetch the entire frontier in one batch.
        id_batch.assign(frontier.begin(), frontier.end());
        get_nodes(id_batch, scratch, view_batch);

        std::vector<uint32_t> next_frontier;
        for (size_t i = 0; i < id_batch.size(); ++i)
        {
            const uint32_t deg = view_batch[i].degree;
            const uint32_t *nbrs = view_batch[i].neighbors;
            for (uint32_t j = 0; j < deg; ++j)
            {
                const uint32_t nb = nbrs[j];
                if (nb >= this->num_points())
                    continue;
                if (visited.insert(nb).second)
                {
                    out_node_list.push_back(nb);
                    next_frontier.push_back(nb);
                    if (out_node_list.size() >= num_nodes_to_cache)
                        break;
                }
            }
            if (out_node_list.size() >= num_nodes_to_cache)
                break;
        }
        frontier = std::move(next_frontier);
    }
    // Delegate to the scratch-aware variant of load_cache_list so the same
    // borrowed scratch is reused across the cache-load passes.
    load_cache_list(out_node_list, scratch);
}

// ---------------------------------------------------------------------------
// Explicit template instantiations (3 per class, 9 total).
// ---------------------------------------------------------------------------

template class unified_node_store_base<float>;
template class unified_node_store_base<uint8_t>;
template class unified_node_store_base<int8_t>;

template class unified_node_store_memory<float>;
template class unified_node_store_memory<uint8_t>;
template class unified_node_store_memory<int8_t>;

template class unified_node_store_ssd<float>;
template class unified_node_store_ssd<uint8_t>;
template class unified_node_store_ssd<int8_t>;

} // namespace diskann
