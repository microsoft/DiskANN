// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "unified_label_data.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include "ann_exception.h"
#include "unified_index_io.h"

namespace diskann
{

namespace
{
} // namespace

// ---------------------------------------------------------------------------
// unified_label_data_base
// ---------------------------------------------------------------------------

void unified_label_data_base::load(UnifiedIndexReader &r, const UnifiedIndexHeader &h, uint64_t npts)
{
    _has_labels = false;
    _use_universal_label = false;
    _universal_label = 0;
    _label_map.clear();

    if ((h.flags & HAS_LABELS) == 0 || h.label_encoding == LabelEncoding::None)
    {
        // Factory shouldn't construct a derived label-data object in this
        // case; throw if it happens through some other path.
        throw ANNException("unified_label_data_base::load called on a header with no labels", -1, __FUNCSIG__,
                           __FILE__, __LINE__);
    }

    _has_labels = true;

    if (h.label_dictionary_len > 0)
    {
        const auto dict_bytes = r.load_region(h.label_dictionary_off, h.label_dictionary_len);
        parse_dictionary(dict_bytes);
    }

    load_encoding(r, h, npts);

    if (h.universal_label != 0)
    {
        _use_universal_label = true;
        _universal_label = static_cast<uint32_t>(h.universal_label);
    }
}

void unified_label_data_base::parse_dictionary(const std::vector<uint8_t> &dict_bytes)
{
    size_t cursor = 0;
    while (cursor < dict_bytes.size())
    {
        if (cursor + sizeof(uint32_t) > dict_bytes.size())
            throw ANNException("unified_label_data: truncated dictionary entry (slen)", -1, __FUNCSIG__, __FILE__,
                               __LINE__);
        uint32_t slen = 0;
        std::memcpy(&slen, dict_bytes.data() + cursor, sizeof(uint32_t));
        cursor += sizeof(uint32_t);
        if (cursor + slen + 2 * sizeof(uint32_t) > dict_bytes.size())
            throw ANNException("unified_label_data: truncated dictionary entry (body)", -1, __FUNCSIG__, __FILE__,
                               __LINE__);
        std::string s(reinterpret_cast<const char *>(dict_bytes.data() + cursor), slen);
        cursor += slen;
        uint32_t label_int = 0;
        std::memcpy(&label_int, dict_bytes.data() + cursor, sizeof(uint32_t));
        cursor += sizeof(uint32_t);
        uint32_t medoid = 0;
        std::memcpy(&medoid, dict_bytes.data() + cursor, sizeof(uint32_t));
        cursor += sizeof(uint32_t);
        // Wire format stores the label int and its single medoid in the same
        // dictionary row; pack them together so search resolves both in one
        // probe. Last-write-wins on duplicate dict entries (shouldn't happen in
        // valid files).
        _label_map[std::move(s)] = label_dict_entry{label_int, medoid};
    }
}

bool unified_label_data_base::is_valid_label(const std::string &s) const
{
    return _label_map.find(s) != _label_map.end();
}

bool unified_label_data_base::get_converted_label(const std::string &s, uint32_t &out) const
{
    auto it = _label_map.find(s);
    if (it == _label_map.end())
        return false;
    out = it->second.label_int;
    return true;
}

void unified_label_data_base::resolve_filters(const std::vector<std::string> &filter_label_strings,
                                              std::vector<uint32_t> &out_label_ints,
                                              std::vector<uint32_t> &out_medoids) const
{
    out_label_ints.clear();
    out_medoids.clear();
    out_label_ints.reserve(filter_label_strings.size());
    out_medoids.reserve(filter_label_strings.size());
    for (const auto &s : filter_label_strings)
    {
        auto it = _label_map.find(s);
        if (it == _label_map.end())
        {
            throw ANNException(std::string("unified_label_data: unknown filter label string: ") + s, -1, __FUNCSIG__,
                               __FILE__, __LINE__);
        }
        // Single probe yields both the label int (for the match proxy) and the
        // per-label medoid (for init-id seeding).
        out_label_ints.push_back(it->second.label_int);
        out_medoids.push_back(it->second.medoid);
    }
}

void unified_label_data_base::collect_label_medoids(std::vector<uint32_t> &out) const
{
    out.reserve(out.size() + _label_map.size());
    for (const auto &kv : _label_map)
        out.push_back(kv.second.medoid);
}

// ---------------------------------------------------------------------------
// unified_label_data_bitmask
// ---------------------------------------------------------------------------

void unified_label_data_bitmask::load_encoding(UnifiedIndexReader &r, const UnifiedIndexHeader &h, uint64_t npts)
{
    const uint64_t bitmask_words = simple_bitmask::get_bitmask_size(h.total_labels);
    _bitmask_buf = simple_bitmask_buf(npts * bitmask_words, bitmask_words);
    if (h.per_point_labels_len > 0)
    {
        const uint64_t expected_bytes = _bitmask_buf._buf.size() * sizeof(std::uint64_t);
        if (h.per_point_labels_len != expected_bytes)
        {
            throw ANNException("unified_label_data_bitmask: bitmask region size mismatch", -1, __FUNCSIG__, __FILE__,
                               __LINE__);
        }
        // Zero-copy: read straight into the bitmask buf's storage.
        r.load_region(h.per_point_labels_off, h.per_point_labels_len,
                      reinterpret_cast<uint8_t *>(_bitmask_buf._buf.data()));
    }
}

std::unique_ptr<filter_match_proxy> unified_label_data_bitmask::make_match_proxy(
    const std::vector<uint32_t> &filter_label_ints)
{
    // Uses the 3-arg ctor that owns its own per-query scratch buffer. Label
    // ints are already resolved by the caller (resolve_filters), so no
    // dictionary lookup happens here.
    return std::make_unique<bitmask_filter_match<uint32_t>>(_bitmask_buf, filter_label_ints, _universal_label);
}

// ---------------------------------------------------------------------------
// unified_label_data_integer
// ---------------------------------------------------------------------------

void unified_label_data_integer::load_encoding(UnifiedIndexReader &r, const UnifiedIndexHeader &h, uint64_t npts)
{
    // Wire format: offsets region is uint64[npts+1]; labels region is raw uint32[total_labels].
    // On every platform DiskANN currently targets, sizeof(size_t) == sizeof(uint64_t),
    // so we can read the offsets directly into _label_vector._offset's storage with
    // zero intermediate copies. Same for _data.
    static_assert(sizeof(size_t) == sizeof(uint64_t),
                  "unified_label_data_integer: zero-copy load assumes size_t == uint64_t");

    const uint64_t expected_off_bytes = (npts + 1) * sizeof(uint64_t);
    if (h.per_point_label_offsets_len != expected_off_bytes)
    {
        throw ANNException("unified_label_data_integer: offset region size mismatch", -1, __FUNCSIG__, __FILE__,
                           __LINE__);
    }
    if (h.per_point_labels_len % sizeof(uint32_t) != 0)
    {
        throw ANNException("unified_label_data_integer: labels region size is not a uint32 multiple", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    const size_t total_labels = h.per_point_labels_len / sizeof(uint32_t);

    _label_vector.resize_for_load(npts, total_labels);
    r.load_region(h.per_point_label_offsets_off, h.per_point_label_offsets_len,
                  reinterpret_cast<uint8_t *>(_label_vector.mutable_offset_data()));
    r.load_region(h.per_point_labels_off, h.per_point_labels_len,
                  reinterpret_cast<uint8_t *>(_label_vector.mutable_label_data()));
}

std::unique_ptr<filter_match_proxy> unified_label_data_integer::make_match_proxy(
    const std::vector<uint32_t> &filter_label_ints)
{
    // integer_label_filter_match holds the filter_labels vector by const reference.
    // We need it to outlive the proxy. Allocate on the heap and bind via a
    // small wrapper that owns the vector. Label ints are already resolved by the
    // caller (resolve_filters), so the wrapper just copies them.
    struct integer_proxy_owner final : public filter_match_proxy
    {
        std::vector<uint32_t> labels;
        integer_label_filter_match<uint32_t> inner;
        integer_proxy_owner(integer_label_vector &lv, std::vector<uint32_t> ls, uint32_t unv)
            : labels(std::move(ls)), inner(lv, labels, unv)
        {
            // integer_label_vector::check_label_exists advances its binary-search
            // window (start = last_check) between successive query labels, which
            // is only correct when the query labels are in ascending order. The
            // resolved filter ints come back in the caller's arbitrary filter
            // order, so sort them here -- matching the legacy filtered-search
            // contract (src/pq_flash_index.cpp:1221, src/index.cpp:2923,3136).
            std::sort(labels.begin(), labels.end());
        }
        bool contain_filtered_label(uint32_t id) override
        {
            return inner.contain_filtered_label(id);
        }
    };
    return std::make_unique<integer_proxy_owner>(_label_vector, filter_label_ints, _universal_label);
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<unified_label_data_base> make_unified_label_data(UnifiedIndexReader &r, const UnifiedIndexHeader &h,
                                                                 uint64_t npts)
{
    if ((h.flags & HAS_LABELS) == 0 || h.label_encoding == LabelEncoding::None)
        return nullptr;

    std::unique_ptr<unified_label_data_base> out;
    switch (h.label_encoding)
    {
    case LabelEncoding::Bitmask:
        out = std::make_unique<unified_label_data_bitmask>();
        break;
    case LabelEncoding::Integer:
        out = std::make_unique<unified_label_data_integer>();
        break;
    default:
        throw ANNException("make_unified_label_data: unknown label_encoding value", -1, __FUNCSIG__, __FILE__,
                           __LINE__);
    }
    out->load(r, h, npts);
    return out;
}

} // namespace diskann
