#pragma once
#include <cstdint>
#include <vector>

#include "windows_customizations.h"

namespace diskann
{

struct simple_bitmask_val
{
    size_t _index = 0;
    std::uint64_t _mask = 0;
};

struct simple_bitmask_full_val
{
    simple_bitmask_full_val()
    {
    }

    void merge_bitmask_val(simple_bitmask_val& bitmask_val)
    {
        _mask[bitmask_val._index] |= bitmask_val._mask;
    }

    std::uint64_t* _mask = nullptr;
};

struct simple_bitmask_buf
{
    // Extra trailing uint64 words kept past the logical per-point extent so the
    // AVX2 fast path in simple_bitmask::test_full_mask_val can issue a 256-bit
    // (4-word) load on the last node without reading past the allocation. Never
    // participates in addressing or iteration (those use _bitmask_size), so it
    // is inert on the scalar path.
    static constexpr std::uint64_t AVX2_TAIL_PADDING = 4;

    simple_bitmask_buf() = default;

    simple_bitmask_buf(std::uint64_t capacity, std::uint64_t bitmask_size)
    {
        _bitmask_size = bitmask_size;
        _buf.resize(capacity + AVX2_TAIL_PADDING, 0);
    }

    // Size the buffer to hold num_points per-point bitmasks plus AVX2 padding.
    // Callers pass the logical extent; the padding is added here so no call site
    // has to remember it.
    void resize_for_points(std::uint64_t num_points)
    {
        _buf.resize(num_points * _bitmask_size + AVX2_TAIL_PADDING, 0);
    }

    std::uint64_t* get_bitmask(std::uint64_t index)
    {
        return _buf.data() + index * _bitmask_size;
    }

    std::vector<std::uint64_t> _buf;
    std::uint64_t _bitmask_size = 0;

};

// simple_bitmask is an internal helper. It is intentionally NOT DISKANN_DLLEXPORT:
// projects that compile label_bitmask.cpp directly (e.g. AdsSnr's ANNTestTool, which
// defines neither _WINDLL nor DISKANN_STATIC_LIB) would otherwise define its members
// in a dllimport context and hit C4273 'inconsistent dll linkage'. Its only client,
// ColorInfoVector (include/color_info.h), is header-only, and every module that
// odr-uses simple_bitmask also compiles label_bitmask.cpp, so no export is needed.
class simple_bitmask
{
public:
    simple_bitmask(std::uint64_t* bitsets, std::uint64_t bitmask_size);

    bool test(size_t pos) const;

    static simple_bitmask_val get_bitmask_val(size_t pos);

    static std::uint64_t get_bitmask_size(std::uint64_t totalBits);

    bool test_mask_val(const simple_bitmask_val& bitmask_val) const;

    bool test_full_mask_val(const simple_bitmask_full_val& bitmask_full_val) const;

    bool test_full_mask_contain(const simple_bitmask& bitmask_full_val) const;

    void set(size_t pos);

    void clear();

private:
    std::uint64_t* _bitsets;
    std::uint64_t _bitmask_size;
};
}