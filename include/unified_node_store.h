// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "tsl/robin_map.h"

#include "aligned_file_reader.h"
#include "defaults.h"
#include "unified_index_format.h"
#include "windows_customizations.h"

namespace diskann
{

class UnifiedIndexReader;

// Per-thread scratch passed into get_nodes(). Holds:
//  - `ctx`: the AlignedFileReader's per-thread IOContext (registered once at
//    load time -- get_nodes does NOT touch the reader's thread-registration
//    map, so no mutex on the search hot path).
//  - A sector slab: either *owned* (allocated via reserve(), used by tests
//    and load-time helpers) or *borrowed* (set via attach_borrowed(), used
//    by the index's beam-search to reuse SSDQueryScratch::sector_scratch).
//
// Memory store ignores the entire scratch.
struct NodeFetchScratch
{
    NodeFetchScratch();
    NodeFetchScratch(const NodeFetchScratch &) = delete;
    NodeFetchScratch &operator=(const NodeFetchScratch &) = delete;
    NodeFetchScratch(NodeFetchScratch &&other) noexcept;
    NodeFetchScratch &operator=(NodeFetchScratch &&other) noexcept;
    ~NodeFetchScratch();

    // Self-owned slab: (re)allocate to hold `max_batch * sectors_per_node`
    // sectors. No-op if the existing slab is already large enough. Allocates
    // -- use only at load time or in tests, not in the search hot path.
    void reserve(uint64_t max_batch, uint32_t sectors_per_node);

    // Borrowed slab: set pointers without allocating. The slab buffer must
    // outlive the scratch and must be at least `slab_capacity_bytes` big.
    // This is what the index uses on the search hot path: ctx and slab both
    // come from SSDThreadData allocated at load time.
    void attach_borrowed(IOContext &ctx, char *external_slab, uint64_t slab_capacity_bytes);

    // Attach an IOContext to a scratch whose slab is already owned via
    // reserve(). Lets the same scratch flip from "no ctx" to "ready" without
    // disturbing the slab. Use in load-time helpers that allocate their own
    // slab but borrow the ctx from a registered thread.
    void set_ctx(IOContext &ctx);

    char *slab() const
    {
        return _sector_slab;
    }
    uint64_t slab_capacity() const
    {
        return _capacity_bytes;
    }
    IOContext *io_ctx() const
    {
        return _ctx;
    }

    std::vector<AlignedRead> requests;

  private:
    char *_sector_slab = nullptr;
    uint64_t _capacity_bytes = 0;
    bool _owns_slab = false;       // true => destructor aligned_free's _sector_slab
    IOContext *_ctx = nullptr;     // not owned; lifetime tied to the reader's per-thread map
};

// View into one node. Lifetime depends on the store:
//  - memory store returns pointers into its resident `_packed` blob;
//  - SSD store returns pointers into the supplied scratch's sector slab
//    (or into the static cache buffers on a cache hit).
template <typename T>
struct NodeView
{
    const T *coords = nullptr;
    const uint32_t *neighbors = nullptr;
    uint32_t degree = 0;
};

// ---------------------------------------------------------------------------
// unified_node_store_base<T>
// Abstract base. Owns header copy, offset table, cached max_node_len.
// Per-node wire layout is [coords (aligned_dim*sizeof(T) bytes),
//                          neighbors (degree*sizeof(uint32_t) bytes)].
// Degree is recovered from the offset delta -- there is no per-node degree
// field in the wire format.
// ---------------------------------------------------------------------------
template <typename T>
class unified_node_store_base
{
  public:
    virtual ~unified_node_store_base() = default;

    // --- Geometry ---
    uint64_t num_points() const
    {
        return _header.npts;
    }
    uint64_t dim() const
    {
        return _header.dim;
    }
    uint64_t aligned_dim() const
    {
        return _header.aligned_dim;
    }
    uint32_t max_degree() const
    {
        return _header.max_degree;
    }
    uint64_t graph_region_base() const
    {
        return _header.graph_region_off;
    }

    // --- Offset math (valid after init_geometry) ---
    uint64_t node_byte_offset(uint64_t id) const
    {
        return _offsets[id];
    }
    uint64_t node_byte_length(uint64_t id) const
    {
        return _offsets[id + 1] - _offsets[id];
    }
    // Absolute byte offset of node `id`'s payload in the unified file.
    // Convenience: same as `graph_region_base() + node_byte_offset(id)`.
    uint64_t node_disk_offset(uint64_t id) const
    {
        return graph_region_base() + _offsets[id];
    }
    uint32_t degree(uint64_t id) const;
    uint32_t num_sectors_per_node() const;
    uint64_t max_node_len() const
    {
        return _max_node_len;
    }
    // aligned_dim * sizeof(T) -- cached in init_geometry().
    uint64_t coord_bytes() const
    {
        return _coord_bytes;
    }

    // --- Single virtual API for node access ---
    // Resolve `ids` into `out` (one NodeView per id, same order).
    virtual void get_nodes(const std::vector<uint64_t> &ids, NodeFetchScratch &scratch,
                           std::vector<NodeView<T>> &out) = 0;

  protected:
    // Subclasses call this from their `load` after parsing header + offset table.
    void init_geometry(const UnifiedIndexHeader &h, std::vector<uint64_t> offset_table);

    UnifiedIndexHeader _header{}; // own copy
    std::vector<uint64_t> _offsets;
    uint64_t _max_node_len = 0;
    uint64_t _coord_bytes = 0;
};

// ---------------------------------------------------------------------------
// unified_node_store_memory<T>
// Fully-resident. Loads the graph region into _packed during load().
// ---------------------------------------------------------------------------
template <typename T>
class unified_node_store_memory final : public unified_node_store_base<T>
{
  public:
    void load(UnifiedIndexReader &r, const UnifiedIndexHeader &h);

    void get_nodes(const std::vector<uint64_t> &ids, NodeFetchScratch &scratch,
                                      std::vector<NodeView<T>> &out) override;

    // Non-virtual fast path for unified_index_memory<T>::iterate_to_fixed_point.
    const T *get_coords(uint64_t id) const;
    const uint32_t *get_neighbors(uint64_t id, uint32_t &out_degree) const;

    // Total resident bytes of the graph region ([coords, neighbors] for all
    // nodes), pulled fully into memory by load().
    uint64_t resident_bytes() const
    {
        return _packed.size();
    }

  private:
    std::vector<uint8_t> _packed;
};

// ---------------------------------------------------------------------------
// unified_node_store_ssd<T>
// AlignedFileReader-backed. Owns the static _nhood_cache / _coord_cache.
// ---------------------------------------------------------------------------
template <typename T>
class unified_node_store_ssd final : public unified_node_store_base<T>
{
  public:
    explicit unified_node_store_ssd(std::shared_ptr<AlignedFileReader> reader) : _reader(std::move(reader))
    {
    }
    ~unified_node_store_ssd() override;

    void load(UnifiedIndexReader &r, const UnifiedIndexHeader &h);

    void get_nodes(const std::vector<uint64_t> &ids, NodeFetchScratch &scratch,
                                      std::vector<NodeView<T>> &out) override;

    // Internal helpers (used by unified_index_ssd::load_storage when the user
    // requests cache priming via UnifiedLoadContext::num_nodes_to_cache).
    // Pin `node_list` (read once, kept resident). Caller supplies a
    // pre-attached NodeFetchScratch (slab + IOContext) -- typically borrowed
    // from an SSDThreadData via attach_borrowed(), or from a self-owned
    // build via make_fetch_scratch().
    void load_cache_list(const std::vector<uint32_t> &node_list, NodeFetchScratch &scratch);

    // BFS-based cache primer. Caller supplies the seed nodes (typically the
    // unified file's medoids; the store doesn't own medoid data). Walks the
    // graph from each seed in breadth-first order, collects up to
    // num_nodes_to_cache unique ids into `out_node_list`, then calls
    // load_cache_list(out_node_list, scratch).
    void cache_bfs_levels(const std::vector<uint32_t> &seed_nodes, uint64_t num_nodes_to_cache,
                                             std::vector<uint32_t> &out_node_list, NodeFetchScratch &scratch);

    // Convenience: build a NodeFetchScratch sized for `max_batch` nodes,
    // register the calling thread with the AlignedFileReader (idempotent;
    // safe to call from already-registered threads), and attach the resulting
    // IOContext. Used by tests and any standalone caller. Allocates an owned
    // slab -- not for the hot path. The hot path attaches an existing
    // SSDThreadData via NodeFetchScratch::attach_borrowed().
    NodeFetchScratch make_fetch_scratch(uint64_t max_batch);

    // Test/observability counter: number of AlignedRead requests this store
    // has issued. Cheap (uint64 increment per get_nodes call), always compiled.
    uint64_t io_count() const
    {
        return _io_count;
    }

  private:
    std::shared_ptr<AlignedFileReader> _reader;

    // Static caches.
    tsl::robin_map<uint32_t, std::pair<uint32_t, uint32_t *>> _nhood_cache;
    uint32_t *_nhood_cache_buf = nullptr;
    tsl::robin_map<uint32_t, T *> _coord_cache;
    T *_coord_cache_buf = nullptr;

    // Always-compiled IO counter (cheap; one uint64 per get_nodes batch).
    uint64_t _io_count = 0;
};

} // namespace diskann
