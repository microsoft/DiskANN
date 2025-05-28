#include "simple_bitmask.h"

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

void simple_bitmask::set(size_t pos)
{
    std::uint64_t mask = (std::uint64_t)1 << (pos & (8 * sizeof(std::uint64_t) - 1));
    size_t index = pos / 8 / sizeof(std::uint64_t);
    _bitsets[index] |= mask;
}

void simple_bitmask::clear()
{
    memset(_bitsets, 0, _bitmask_size * sizeof(std::uint64_t));
}

std::uint64_t simple_bitmask::get_bitmask_size(std::uint64_t totalBits)
{
    std::uint64_t bytes = (totalBits + 7) / 8;
    std::uint64_t aligned_bytes = bytes + sizeof(std::uint64_t) - 1;
    aligned_bytes = aligned_bytes - (aligned_bytes % sizeof(std::uint64_t));
    return aligned_bytes / sizeof(std::uint64_t);
}

}