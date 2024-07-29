#pragma once
#include <cstdint>
#include <vector>
#include "types.h"

namespace diskann
{

class NeighborList
{
public:
    NeighborList(const location_t* data, size_t size);

    const location_t* data() const;
    size_t size() const;
    bool empty() const;

    // compatable with current interface, need deprecate later
    void convert_to_vector(std::vector<location_t>& vector_copy) const;

    class Iterator
    {
    public:
        Iterator(const location_t* index);

        const location_t& operator*() const;

        const Iterator& operator++();

        bool operator==(const Iterator& other) const;

        bool operator!=(const Iterator& other) const;

    private:
        const location_t* _index;
    };

  //  Iterator begin() = 0;
    Iterator begin() const;
  //  Iterator end() = 0;
    Iterator end() const;

private:
    const location_t* _data;
    size_t _size;
};

}
