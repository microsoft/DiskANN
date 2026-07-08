#pragma once
#include <cstdint>
#include <cstring>
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
    simple_bitmask_buf() = default;

    simple_bitmask_buf(std::uint64_t capacity, std::uint64_t bitmask_size)
    {
        _buf.resize(capacity);
        _bitmask_size = bitmask_size;
    }

    std::uint64_t* get_bitmask(std::uint64_t index)
    {
        return _buf.data() + index * _bitmask_size;
    }

    std::vector<std::uint64_t> _buf;
    std::uint64_t _bitmask_size = 0;

};

// simple_bitmask is a small internal helper whose methods are defined inline
// (header-only), so it needs no dll linkage. ColorInfoVector's inline constructor
// (include/color_info.h, pulled in widely via neighbor.h) odr-uses these methods;
// keeping them inline lets any consumer resolve them locally -- both DLL consumers
// and projects that compile label_bitmask.cpp directly (e.g. AdsSnr's ANNTestTool),
// which previously hit C4273 'inconsistent dll linkage' when the class was
// DISKANN_DLLEXPORT. label_bitmask.cpp therefore has no out-of-line definitions.
class simple_bitmask
{
public:
    simple_bitmask(std::uint64_t* bitsets, std::uint64_t bitmask_size)
        : _bitsets(bitsets)
        , _bitmask_size(bitmask_size)
    {
    }

    bool test(size_t pos) const
    {
        std::uint64_t mask = (std::uint64_t)1 << (pos & (8 * sizeof(std::uint64_t) - 1));
        size_t index = pos / 8 / sizeof(std::uint64_t);
        std::uint64_t val = _bitsets[index];
        return 0 != (val & mask);
    }

    static simple_bitmask_val get_bitmask_val(size_t pos)
    {
        simple_bitmask_val bitmask_val;
        bitmask_val._mask = (std::uint64_t)1 << (pos & (8 * sizeof(std::uint64_t) - 1));
        bitmask_val._index = pos / 8 / sizeof(std::uint64_t);

        return bitmask_val;
    }

    static std::uint64_t get_bitmask_size(std::uint64_t totalBits)
    {
        std::uint64_t bytes = (totalBits + 7) / 8;
        std::uint64_t aligned_bytes = bytes + sizeof(std::uint64_t) - 1;
        aligned_bytes = aligned_bytes - (aligned_bytes % sizeof(std::uint64_t));
        return aligned_bytes / sizeof(std::uint64_t);
    }

    bool test_mask_val(const simple_bitmask_val& bitmask_val) const
    {
        std::uint64_t val = _bitsets[bitmask_val._index];
        return 0 != (val & bitmask_val._mask);
    }

    bool test_full_mask_val(const simple_bitmask_full_val& bitmask_full_val) const
    {
        for (size_t i = 0; i < _bitmask_size; i++)
        {
            if ((bitmask_full_val._mask[i] & _bitsets[i]) != 0)
            {
                return true;
            }
        }

        return false;
    }

    bool test_full_mask_contain(const simple_bitmask& bitmask_full_val) const
    {
        for (size_t i = 0; i < _bitmask_size; i++)
        {
            auto mask = bitmask_full_val._bitsets[i];
            if ((mask & _bitsets[i]) != mask)
            {
                return false;
            }
        }

        return true;
    }

    void set(size_t pos)
    {
        std::uint64_t mask = (std::uint64_t)1 << (pos & (8 * sizeof(std::uint64_t) - 1));
        size_t index = pos / 8 / sizeof(std::uint64_t);
        _bitsets[index] |= mask;
    }

    void clear()
    {
        std::memset(_bitsets, 0, _bitmask_size * sizeof(std::uint64_t));
    }

private:
    std::uint64_t* _bitsets;
    std::uint64_t _bitmask_size;
};
}