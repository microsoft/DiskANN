// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <mutex>
#include <vector>
#include "utils.h"

namespace diskann
{

struct Neighbor
{
    unsigned id;
    float distance;
    bool expanded;

    Neighbor() = default;

    Neighbor(unsigned id, float distance) : id{id}, distance{distance}, expanded(false)
    {
    }

    inline bool operator<(const Neighbor &other) const
    {
        return distance < other.distance || (distance == other.distance && id < other.id);
    }

    inline bool operator==(const Neighbor &other) const
    {
        return (id == other.id);
    }
};

// Invariant: after every `insert` and `closest_unexpanded()`, `_cur` points to
//            the first Neighbor which is unexpanded.
class NeighborPriorityQueue
{
  public:
    NeighborPriorityQueue() : _size(0), _capacity(0), _cur(0), _auto_resizable(false)
    {
    }

    explicit NeighborPriorityQueue(size_t capacity) : _size(0), _capacity(capacity), _cur(0), _data(capacity + 1)
    {
    }

    explicit NeighborPriorityQueue(bool auto_resizable) : _size(0), _capacity(0), _cur(0), _auto_resizable(auto_resizable)
    {
    }

    // Inserts the item ordered into the set up to the sets capacity.
    // The item will be dropped if it is the same id as an exiting
    // set item or it has a greated distance than the final
    // item in the set. The set cursor that is used to pop() the
    // next item will be set to the lowest index of an uncheck item
    void insert(const Neighbor &nbr)
    {
        if (!_auto_resizable && _size == _capacity && _data[_size - 1] < nbr)
        {
            return;
        }

        size_t lo = 0, hi = _size;
        while (lo < hi)
        {
            size_t mid = (lo + hi) >> 1;
            if (nbr < _data[mid])
            {
                hi = mid;
                // Make sure the same id isn't inserted into the set
            }
            else if (_data[mid].id == nbr.id)
            {
                return;
            }
            else
            {
                lo = mid + 1;
            }
        }

        if (!_auto_resizable && lo < _capacity)
        {
            std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(Neighbor));
        }
        else
        {
            // Resize the queue if it's full in auto-resizable mode
            if (_size == _capacity)
            {
                _capacity = _capacity * 2;
                _data.resize(_capacity + 1);
            }

            // Shift elements to make space for the new neighbor
            std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(Neighbor));
        }
        
        _data[lo] = {nbr.id, nbr.distance};
        if (_size < _capacity || _auto_resizable)
        {
            _size++;
        }
        if (lo < _cur)
        {
            _cur = lo;
        }

        if (_auto_resizable && _size == _capacity)
        {
            // double the size of the array if queue is full and auto-resizable
            _capacity = _capacity * 2;
            _data.resize(_capacity + 1);
        }
    }

    Neighbor closest_unexpanded()
    {
        _data[_cur].expanded = true;
        size_t pre = _cur;
        while (_cur < _size && _data[_cur].expanded)
        {
            _cur++;
        }
        return _data[pre];
    }


    bool has_unexpanded_node(size_t search_param_l = 0) const
    {
        if (_auto_resizable){
            return _cur < std::min(_size, search_param_l);
        }
        return _cur < _size;
    }

    size_t size() const
    {
        return _size;
    }

    size_t capacity() const
    {
        return _capacity;
    }

    void reserve(size_t capacity)
    {
        if (capacity + 1 > _data.size())
        {
            _data.resize(capacity + 1);
        }
        _capacity = capacity;
    }

    Neighbor &operator[](size_t i)
    {
        return _data[i];
    }

    Neighbor operator[](size_t i) const
    {
        return _data[i];
    }

    void clear()
    {
        _size = 0;
        _cur = 0;
    }

    void convert_to_auto_resizable()
    {
        _auto_resizable = true;
    }

    bool is_auto_resizable() const
    {
        return _auto_resizable;
    }

  private:
    size_t _size, _capacity, _cur;
    bool _auto_resizable;
    std::vector<Neighbor> _data;
};

} // namespace diskann
