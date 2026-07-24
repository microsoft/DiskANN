// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "unified_index_base.h"

#include "ann_exception.h"
#include "unified_index_io.h"

namespace diskann
{

template <typename T>
unified_index_base<T>::unified_index_base(diskann::Metric metric) : _metric(metric)
{
}

template <typename T> unified_index_base<T>::~unified_index_base() = default;

template <typename T> void unified_index_base<T>::validate_header(const UnifiedIndexHeader &h) const
{
    if (h.magic != UNIFIED_FORMAT_MAGIC)
        throw ANNException("unified_index_base: bad magic", -1, __FUNCSIG__, __FILE__, __LINE__);
    if (h.version != UNIFIED_FORMAT_VERSION)
        throw ANNException("unified_index_base: unsupported version", -1, __FUNCSIG__, __FILE__, __LINE__);
    if (h.data_type != data_type_tag_of<T>())
        throw ANNException("unified_index_base: data_type mismatch with T", -1, __FUNCSIG__, __FILE__, __LINE__);
}

template <typename T>
void unified_index_base<T>::validate_search_context(const UnifiedSearchContext &ctx) const
{
    if (ctx.query == nullptr)
        throw ANNException("UnifiedSearchContext: query == nullptr", -1, __FUNCSIG__, __FILE__, __LINE__);
    if (ctx.indices == nullptr || ctx.distances == nullptr)
        throw ANNException("UnifiedSearchContext: indices/distances buffers are required", -1, __FUNCSIG__, __FILE__,
                           __LINE__);
    if (ctx.K == 0)
        throw ANNException("UnifiedSearchContext: K must be > 0", -1, __FUNCSIG__, __FILE__, __LINE__);
    if (ctx.L < ctx.K)
        throw ANNException("UnifiedSearchContext: L must be >= K", -1, __FUNCSIG__, __FILE__, __LINE__);

    const bool filtered = has_labels();
    const bool has_filters = !ctx.filter_labels.empty();
    if (filtered && !has_filters)
        throw ANNException("UnifiedSearchContext: filter_labels must be non-empty for a filtered index", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    if (!filtered && has_filters)
        throw ANNException("UnifiedSearchContext: filter_labels must be empty for a non-filtered index", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
}

template <typename T> void unified_index_base<T>::load(const UnifiedLoadContext &ctx)
{
    _index_path = ctx.path;

    UnifiedIndexReader reader(ctx.path);
    _header = reader.header();
    validate_header(_header);

    _labels = make_unified_label_data(reader, _header, _header.npts);
    load_storage(reader, ctx);

    // Populate resident-memory / cardinality accounting (mirrors
    // Index::load / PQFlashIndex::load). Common fields here; the
    // storage-specific node/graph bytes come from the derived class.
    _table_stats = TableStats{};
    _table_stats.node_count = _header.npts;
    if (_labels && _labels->has_labels())
    {
        _table_stats.label_count = _labels->num_labels();
        _table_stats.label_mem_usage = _labels->memory_usage();
    }
    fill_storage_stats(_table_stats);
    _table_stats.total_mem_usage = _table_stats.node_mem_usage + _table_stats.graph_mem_usage +
                                   _table_stats.label_mem_usage + _table_stats.tag_memory_usage;
}

template <typename T> void unified_index_base<T>::search(UnifiedSearchContext &ctx)
{
    validate_search_context(ctx);
    search_impl(ctx);
}

template class unified_index_base<float>;
template class unified_index_base<uint8_t>;
template class unified_index_base<int8_t>;

} // namespace diskann
