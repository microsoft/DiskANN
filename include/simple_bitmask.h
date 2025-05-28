#pragma once
#include <cstdint>
#include <vector>

namespace diskann
{

struct simple_bitmask_buf
{
    simple_bitmask_buf(size_t capacity, std::uint64_t bitmask_sizse)
    {
        _buf.resize(capacity);
        _bitmask_size = bitmask_sizse;
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

    void set(size_t pos);

    void clear();

    static std::uint64_t get_bitmask_size(std::uint64_t totalBits);

private:
    std::uint64_t* _bitsets;
    std::uint64_t _bitmask_size;
};

}
