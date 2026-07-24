// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "distance.h"
#include "unified_index.h"
#include "unified_index_format.h"
#include "unified_label_data.h"
#include "unified_node_store.h"
#include "windows_customizations.h"

namespace diskann
{

class UnifiedIndexReader;

// Templated implementation of the non-templated `unified_index` interface.
// Holds the parsed header, the metric, the label data (built by
// make_unified_label_data), and the node store (a unified_node_store_memory<T>
// or unified_node_store_ssd<T>, plugged in by the derived class's
// `load_storage`).
template <typename T>
class unified_index_base : public unified_index
{
  public:
    explicit unified_index_base(diskann::Metric metric);
    ~unified_index_base() override;

    void load(const UnifiedLoadContext &ctx) override;
    void search(UnifiedSearchContext &ctx) override;

    const UnifiedIndexHeader &header() const override
    {
        return _header;
    }
    uint64_t num_points() const override
    {
        return _header.npts;
    }
    uint64_t dim() const override
    {
        return _header.dim;
    }
    uint64_t aligned_dim() const override
    {
        return _header.aligned_dim;
    }
    diskann::Metric metric() const override
    {
        return _metric;
    }
    DataTypeTag data_type() const override
    {
        return data_type_tag_of<T>();
    }
    bool has_labels() const override
    {
        return _labels && _labels->has_labels();
    }
    TableStats get_table_stats() const override
    {
        return _table_stats;
    }

    // Templated read-only accessors for in-process callers that *do* know T
    // (unit tests, the index's own search loop). Not on the public interface.
    const unified_label_data_base *labels() const
    {
        return _labels.get();
    }
    const unified_node_store_base<T> *nodes() const
    {
        return _store.get();
    }
    unified_node_store_base<T> *nodes()
    {
        return _store.get();
    }

  protected:
    // Derived class is responsible for instantiating the right _store subclass
    // and calling its load(). It may inspect ctx for SSD-only knobs like
    // ctx.num_nodes_to_cache.
    virtual void load_storage(UnifiedIndexReader &r, const UnifiedLoadContext &ctx) = 0;
    virtual void search_impl(UnifiedSearchContext &ctx) = 0;

    // Fill the storage-specific resident-memory fields (node_mem_usage,
    // graph_mem_usage) of `stats`. Memory reports resident coords/graph; SSD
    // reports the resident PQ codes (graph lives on disk). Called by load()
    // after load_storage() so the store is populated.
    virtual void fill_storage_stats(TableStats &stats) const = 0;

    void validate_header(const UnifiedIndexHeader &h) const;
    void validate_search_context(const UnifiedSearchContext &ctx) const;

    UnifiedIndexHeader _header{};
    diskann::Metric _metric;
    std::unique_ptr<unified_label_data_base> _labels;       // nullptr when header has no labels
    std::unique_ptr<unified_node_store_base<T>> _store;     // built by derived load_storage()
    std::string _index_path;
    TableStats _table_stats;
};

} // namespace diskann
