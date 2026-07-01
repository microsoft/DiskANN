// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "aligned_file_reader.h"
#include "distance.h"
#include "percentile_stats.h"
#include "unified_index_format.h"
#include "windows_customizations.h"

namespace diskann
{

struct QueryStats;
struct DebugTraversalInfo;

// Knobs passed to unified_index::load. Path identifies the unified container
// file. `num_threads` and `search_l` size per-thread scratch on the memory
// implementation. `num_nodes_to_cache` triggers SSD static-cache priming
// (no-op for the memory implementation).
struct UnifiedLoadContext
{
    std::string path;
    uint32_t num_threads = 1;
    uint32_t search_l = 100;
    uint64_t num_nodes_to_cache = 0;
};

// Single in/out container for a search call. The caller fills inputs and
// allocates the output buffers; search() writes outputs (and optional
// telemetry) directly. No allocation happens inside search().
struct UnifiedSearchContext
{
    // ---- Inputs ----
    const void *query = nullptr;          // typed by caller as const T*
    size_t K = 10;
    uint32_t L = 100;
    // Filter labels as user-facing strings. Required non-empty if the loaded
    // index has labels; required empty otherwise. The index converts strings
    // to internal label ints per its encoding.
    std::vector<std::string> filter_labels;
    std::optional<uint32_t> beam_width;                            // SSD-only
    std::optional<uint32_t> io_limit;                              // SSD-only
    std::function<float(const std::uint8_t *, size_t)> rerank_fn;  // SSD-only

    // ---- Outputs (caller-allocated, length >= K) ----
    uint64_t *indices = nullptr;
    float *distances = nullptr;

    // ---- Optional telemetry sinks (nullptr = no telemetry) ----
    QueryStats *stats = nullptr;
    DebugTraversalInfo *debug_info = nullptr;
};

// Non-templated public interface returned by the factory. Users program
// against this; the templated `unified_index_base<T>` implements it.
class unified_index
{
  public:
    virtual ~unified_index() = default;

    virtual void load(const UnifiedLoadContext &ctx) = 0;
    virtual void search(UnifiedSearchContext &ctx) = 0;

    virtual const UnifiedIndexHeader &header() const = 0;
    virtual uint64_t num_points() const = 0;
    virtual uint64_t dim() const = 0;
    virtual uint64_t aligned_dim() const = 0;
    virtual diskann::Metric metric() const = 0;
    virtual DataTypeTag data_type() const = 0;
    virtual bool has_labels() const = 0;

    // Resident memory / cardinality accounting for the loaded index, mirroring
    // Index::get_table_stats() and PQFlashIndex::get_table_stats().
    virtual TableStats get_table_stats() const = 0;
};

// Factory: open a unified file fully in memory. Peeks the 4 KiB header,
// dispatches on `data_type`, instantiates the right templated implementation,
// calls load(ctx), returns the owning pointer as the non-templated interface.
std::unique_ptr<unified_index> make_unified_index_memory(const UnifiedLoadContext &ctx);

// Factory: open a unified file in disk-resident (SSD) mode. The supplied
// AlignedFileReader is handed to the constructed unified_index_ssd<T>.
std::unique_ptr<unified_index> make_unified_index_ssd(
    std::shared_ptr<AlignedFileReader> reader, const UnifiedLoadContext &ctx);

} // namespace diskann
