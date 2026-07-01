// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <type_traits>

namespace diskann
{

constexpr uint32_t UNIFIED_FORMAT_MAGIC = 0x444E4E55; // "UNND" little-endian
constexpr uint32_t UNIFIED_FORMAT_VERSION = 2;
constexpr uint64_t UNIFIED_FORMAT_ALIGN = 4096;

enum class DataTypeTag : uint32_t
{
    Float = 1,
    Uint8 = 2,
    Int8 = 3,
};

enum class MetricTag : uint32_t
{
    L2 = 1,
    InnerProduct = 2,
    Cosine = 3,
};

enum class LabelEncoding : uint32_t
{
    None = 0,
    Bitmask = 1,
    Integer = 2,
};

enum UnifiedFormatFlags : uint32_t
{
    HAS_PQ = 1u << 0,
    HAS_LABELS = 1u << 1,
    HAS_MAX_BASE_NORM = 1u << 2,
};

#pragma pack(push, 1)
struct UnifiedIndexHeader
{
    uint32_t magic;
    uint32_t version;
    DataTypeTag data_type;
    MetricTag metric;
    uint64_t npts;
    uint64_t dim;
    uint64_t aligned_dim;
    uint32_t max_degree;
    uint32_t flags;
    uint64_t start_node;

    uint64_t offset_table_off, offset_table_len;
    uint64_t graph_region_off, graph_region_len;
    uint64_t medoids_off, medoids_len;
    uint64_t pq_pivots_off, pq_pivots_len;
    uint64_t pq_codes_off, pq_codes_len;
    uint64_t max_base_norm_off, max_base_norm_len;

    LabelEncoding label_encoding;
    uint64_t universal_label;
    uint64_t total_labels;
    uint64_t label_dictionary_off, label_dictionary_len;
    uint64_t per_point_labels_off, per_point_labels_len;
    uint64_t per_point_label_offsets_off, per_point_label_offsets_len;

    // Total size of the file in bytes. Populated by finalize() and validated
    // by readers on load (truncated / over-sized files are rejected).
    // Also useful for disk-quota / capacity-planning logs.
    uint64_t file_size_bytes;

    uint8_t _reserved[4096 - (sizeof(uint32_t) * 7 + sizeof(uint64_t) * 25)];
};
#pragma pack(pop)

static_assert(sizeof(UnifiedIndexHeader) == 4096, "header must occupy exactly one sector");

inline uint64_t align_up_4k(uint64_t v)
{
    return (v + UNIFIED_FORMAT_ALIGN - 1) & ~(UNIFIED_FORMAT_ALIGN - 1);
}

template <typename T> constexpr DataTypeTag data_type_tag_of()
{
    if constexpr (std::is_same_v<T, float>)
        return DataTypeTag::Float;
    else if constexpr (std::is_same_v<T, uint8_t>)
        return DataTypeTag::Uint8;
    else if constexpr (std::is_same_v<T, int8_t>)
        return DataTypeTag::Int8;
    else
        static_assert(!sizeof(T), "unsupported data type");
}

} // namespace diskann
