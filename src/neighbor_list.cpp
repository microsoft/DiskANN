#include "neighbor_list.h"

namespace diskann
{

NeighborList::NeighborList(const location_t* data, size_t size)
    : _data(data)
    , _size(size)
{
}

const location_t* NeighborList::data() const
{
    return _data;
}

size_t NeighborList::size() const
{
    return _size;
}

bool NeighborList::empty() const
{
    return _size == 0;
}

void NeighborList::convert_to_vector(std::vector<location_t>& vector_copy) const
{
    vector_copy.reserve(_size);
    for (size_t i = 0; i < _size; i++)
    {
        vector_copy.push_back(_data[i]);
    }
}

NeighborList::Iterator::Iterator(const location_t* index)
    : _index(index)
{
}

const location_t& NeighborList::Iterator::operator*() const
{
    return *_index;
}

const NeighborList::Iterator& NeighborList::Iterator::operator++()
{
    _index++;
    return *this;
}

bool NeighborList::Iterator::operator==(const NeighborList::Iterator& other) const
{
    return _index == other._index;
}

bool NeighborList::Iterator::operator!=(const NeighborList::Iterator& other) const
{
    return !(*this == other);
}

NeighborList::Iterator NeighborList::begin() const
{
    return Iterator(_data);
}

NeighborList::Iterator NeighborList::end() const
{
    return Iterator(_data + _size);
}

}