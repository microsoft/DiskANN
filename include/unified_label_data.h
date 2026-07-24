// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "filter_match_proxy.h"
#include "integer_label_vector.h"
#include "label_bitmask.h"
#include "unified_index_format.h"
#include "windows_customizations.h"

namespace diskann
{

class UnifiedIndexReader;

// ---------------------------------------------------------------------------
// Abstract base for the label-data trio.
// Owns the shared, encoding-independent state (label dictionary, universal
// label, per-label medoids) and exposes the read-only query API.
// Derived classes own encoding-specific storage and produce encoding-specific
// match proxies via `make_match_proxy`.
//
// All label ints are uint32 on the API surface; the on-disk dictionary entry
// stores them as uint32 unconditionally (see docs/unified_index_format.md).
// ---------------------------------------------------------------------------
class unified_label_data_base
{
  public:
    virtual ~unified_label_data_base() = default;

    // Template method: parse shared dictionary, then dispatch to derived
    // load_encoding(). Caller has the reader open and validated.
    void load(UnifiedIndexReader &r, const UnifiedIndexHeader &h, uint64_t npts);

    // --- Shared query API ---
    bool has_labels() const
    {
        return _has_labels;
    }
    bool has_universal_label() const
    {
        return _use_universal_label;
    }
    uint32_t universal_label() const
    {
        return _universal_label;
    }
    size_t num_labels() const
    {
        return _label_map.size();
    }
    virtual LabelEncoding encoding() const = 0;

    // Resident bytes of the encoding-specific per-point label storage.
    virtual uint64_t memory_usage() const
    {
        return 0;
    }

    bool is_valid_label(const std::string &s) const;
    bool get_converted_label(const std::string &s, uint32_t &out) const;

    // Resolve filter label strings to their internal label ints AND per-label
    // medoids in a single dictionary probe per string. out_label_ints[i] and
    // out_medoids[i] both correspond to filter_label_strings[i]; both vectors
    // are caller-owned, cleared, then filled in lockstep. Throws ANNException
    // on an unknown label string. The unified format stores exactly one medoid
    // per label, packed in the same dictionary row as the label int, so the
    // search path gets the proxy input (label int) and the init-id seed
    // (medoid) from one map lookup instead of two.
    void resolve_filters(const std::vector<std::string> &filter_label_strings,
                                           std::vector<uint32_t> &out_label_ints,
                                           std::vector<uint32_t> &out_medoids) const;

    // Append every per-label entry-point medoid (the unified format stores
    // exactly one per label) to `out`. Used to seed SSD cache priming so that
    // filtered-search entry points -- and their BFS neighborhoods -- get
    // cached, mirroring the legacy PQFlashIndex::cache_bfs_levels seeding from
    // _filter_to_medoid_ids. `out` is appended to (not cleared); the caller
    // typically pre-fills it with the global medoids first.
    void collect_label_medoids(std::vector<uint32_t> &out) const;

    // Build a search-loop-ready matcher from pre-resolved internal label ints
    // (see resolve_filters -- the string -> int conversion happens once there
    // and is shared with init-id seeding). The returned proxy borrows internal
    // storage of `this` -- lifetime must not exceed `this`. No external scratch
    // is needed; the concrete proxy owns any per-query scratch it requires.
    virtual std::unique_ptr<filter_match_proxy> make_match_proxy(
        const std::vector<uint32_t> &filter_label_ints) = 0;

  protected:
    // Derived classes load their encoding-specific region(s) after the base
    // has parsed the shared dictionary.
    virtual void load_encoding(UnifiedIndexReader &r, const UnifiedIndexHeader &h, uint64_t npts) = 0;

    // Helper: convert strings -> uint32 label ints via dictionary; throws on unknown.
    void parse_dictionary(const std::vector<uint8_t> &dict_bytes);

    bool _has_labels = false;
    bool _use_universal_label = false;
    uint32_t _universal_label = 0;

    // Dictionary row: label string -> {internal label int, per-label medoid}.
    // Both fields come from the same on-disk dictionary entry (see
    // parse_dictionary / docs/unified_index_format.md), so packing them lets a
    // single lookup serve both the match proxy (label int) and init-id seeding
    // (medoid) at search time -- avoiding a second probe of a separate map.
    struct label_dict_entry
    {
        uint32_t label_int = 0;
        uint32_t medoid = 0;
    };
    std::unordered_map<std::string, label_dict_entry> _label_map;
};

// Bitmask-encoded label storage. One bitmask row of
// `_bitmask_buf._bitmask_size` uint64 words per point.
class unified_label_data_bitmask final : public unified_label_data_base
{
  public:
    LabelEncoding encoding() const override
    {
        return LabelEncoding::Bitmask;
    }

    uint64_t memory_usage() const override
    {
        return _bitmask_buf._buf.size() * sizeof(std::uint64_t);
    }

    std::unique_ptr<filter_match_proxy> make_match_proxy(
        const std::vector<uint32_t> &filter_label_ints) override;

    simple_bitmask_buf &bitmask_buf()
    {
        return _bitmask_buf;
    }
    const simple_bitmask_buf &bitmask_buf() const
    {
        return _bitmask_buf;
    }

  protected:
    void load_encoding(UnifiedIndexReader &r, const UnifiedIndexHeader &h, uint64_t npts) override;

  private:
    simple_bitmask_buf _bitmask_buf;
};

// Integer-encoded label storage. Variable-length label list per point with an
// offset table of size npts+1 into a flat uint32 label array.
class unified_label_data_integer final : public unified_label_data_base
{
  public:
    LabelEncoding encoding() const override
    {
        return LabelEncoding::Integer;
    }

    uint64_t memory_usage() const override
    {
        return _label_vector.get_memory_usage();
    }

    std::unique_ptr<filter_match_proxy> make_match_proxy(
        const std::vector<uint32_t> &filter_label_ints) override;

    integer_label_vector &label_vector()
    {
        return _label_vector;
    }
    const integer_label_vector &label_vector() const
    {
        return _label_vector;
    }

  protected:
    void load_encoding(UnifiedIndexReader &r, const UnifiedIndexHeader &h, uint64_t npts) override;

  private:
    integer_label_vector _label_vector;
};

// Factory: peeks at `h.label_encoding`, constructs the correct derived class,
// runs load(), and returns it. Returns nullptr when the header carries no
// labels (HAS_LABELS flag unset or encoding == None).
std::unique_ptr<unified_label_data_base> make_unified_label_data(UnifiedIndexReader &r,
                                                                                    const UnifiedIndexHeader &h,
                                                                                    uint64_t npts);

} // namespace diskann
