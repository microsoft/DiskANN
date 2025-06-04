#pragma once
#include <cstdint>
#include <vector>

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