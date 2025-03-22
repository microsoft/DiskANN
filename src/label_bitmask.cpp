#include "label_bitmask.h"

namespace diskann
{

simple_bitmask::simple_bitmask(std::uint64_t* bitsets, std::uint64_t bitmask_size)
    : _bitsets(bitsets)
    , _bitmask_size(bitmask_size)
{
}

bool simple_bitmask::test(size_t pos) const
{
    std::uint64_t mask = (std::uint64_t)1 << (pos & (8 * sizeof(std::uint64_t) - 1));
    size_t index = pos / 8 / sizeof(std::uint64_t);
    std::uint64_t val = _bitsets[index];
    return 0 != (val & mask);
}

simple_bitmask_val simple_bitmask::get_bitmask_val(size_t pos)
{
    simple_bitmask_val bitmask_val;
    bitmask_val._mask = (std::uint64_t)1 << (pos & (8 * sizeof(std::uint64_t) - 1));
    bitmask_val._index = pos / 8 / sizeof(std::uint64_t);

    return bitmask_val;
}

std::uint64_t simple_bitmask::get_bitmask_size(std::uint64_t totalBits)
{
    std::uint64_t bytes = (totalBits + 7) / 8;
    std::uint64_t aligned_bytes = bytes + sizeof(std::uint64_t) - 1;
    aligned_bytes = aligned_bytes - (aligned_bytes % sizeof(std::uint64_t));
    return aligned_bytes / sizeof(std::uint64_t);
}

bool simple_bitmask::test_mask_val(const simple_bitmask_val& bitmask_val) const
{
    std::uint64_t val = _bitsets[bitmask_val._index];
    return 0 != (val & bitmask_val._mask);
}

bool simple_bitmask::test_full_mask_val(const simple_bitmask_full_val& bitmask_full_val) const
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

bool simple_bitmask::test_full_mask_contain(const simple_bitmask& bitmask_full_val) const
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

void simple_bitmask::set(size_t pos)
{
    std::uint64_t mask = (std::uint64_t)1 << (pos & (8 * sizeof(std::uint64_t) - 1));
    size_t index = pos / 8 / sizeof(std::uint64_t);
    _bitsets[index] |= mask;
}

}