// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "unified_index.h"

#include "ann_exception.h"
#include "unified_index_io.h"
#include "unified_index_memory.h"
#include "unified_index_ssd.h"

namespace diskann
{

namespace
{

// Map MetricTag from the header to the runtime Metric enum.
diskann::Metric metric_from_tag(MetricTag tag)
{
    switch (tag)
    {
    case MetricTag::L2:
        return diskann::Metric::L2;
    case MetricTag::InnerProduct:
        return diskann::Metric::INNER_PRODUCT;
    case MetricTag::Cosine:
        return diskann::Metric::COSINE;
    default:
        throw ANNException("unified_index factory: unknown metric tag in header", -1, __FUNCSIG__, __FILE__,
                           __LINE__);
    }
}

// Peek the 4 KiB header to decide which T to instantiate. Reader is closed
// before the index's own load(ctx) reopens the file.
UnifiedIndexHeader peek_header(const std::string &path)
{
    UnifiedIndexReader peek(path);
    return peek.header();
}

template <template <typename> class IndexT, typename... CtorArgs>
std::unique_ptr<unified_index> make_for_data_type(DataTypeTag dt, MetricTag metric_tag, CtorArgs &&...args)
{
    const diskann::Metric metric = metric_from_tag(metric_tag);
    switch (dt)
    {
    case DataTypeTag::Float:
        return std::make_unique<IndexT<float>>(std::forward<CtorArgs>(args)..., metric);
    case DataTypeTag::Uint8:
        return std::make_unique<IndexT<uint8_t>>(std::forward<CtorArgs>(args)..., metric);
    case DataTypeTag::Int8:
        return std::make_unique<IndexT<int8_t>>(std::forward<CtorArgs>(args)..., metric);
    default:
        throw ANNException("unified_index factory: unknown data_type in header", -1, __FUNCSIG__, __FILE__,
                           __LINE__);
    }
}

} // namespace

std::unique_ptr<unified_index> make_unified_index_memory(const UnifiedLoadContext &ctx)
{
    const UnifiedIndexHeader h = peek_header(ctx.path);
    auto idx = make_for_data_type<unified_index_memory>(h.data_type, h.metric);
    idx->load(ctx);
    return idx;
}

std::unique_ptr<unified_index> make_unified_index_ssd(std::shared_ptr<AlignedFileReader> reader,
                                                      const UnifiedLoadContext &ctx)
{
    const UnifiedIndexHeader h = peek_header(ctx.path);
    auto idx = make_for_data_type<unified_index_ssd>(h.data_type, h.metric, std::move(reader));
    idx->load(ctx);
    return idx;
}

} // namespace diskann
