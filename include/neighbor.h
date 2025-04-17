// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <mutex>
#include <vector>
#include <tsl/robin_map.h>
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

struct NeighborExtendColor : public Neighbor
{
    uint32_t color;
    int previous = -1;

    NeighborExtendColor() = default;

    NeighborExtendColor(unsigned id, float distance, uint32_t color) : Neighbor(id, distance), color(color)
    {
    }

    inline bool operator<(const NeighborExtendColor &other) const
    {
        return distance < other.distance || (distance == other.distance && id < other.id);
    }

    inline bool operator==(const NeighborExtendColor &other) const
    {
        return (id == other.id);
    }
};

class NeighborVectorBase
{
public:
    virtual Neighbor& operator[](size_t i) = 0;

    virtual Neighbor operator[](size_t i) const = 0;

    virtual void insert(const Neighbor& nbr, size_t loc, size_t size) = 0;

    virtual size_t current_size(const Neighbor& nbr) const = 0;
};

class NeighborVector : public NeighborVectorBase
{
public:
    NeighborVector() = default;

    NeighborVector(size_t capacity) 
        : _capacity(capacity)
        , _data(capacity + 1, Neighbor(std::numeric_limits<uint32_t>::max(), std::numeric_limits<float>::max()))
    {
    }

    virtual Neighbor& operator[](size_t i) override
    {
        return _data[i];
    }

    virtual Neighbor operator[](size_t i) const override
    {
        return _data[i];
    }

    virtual void insert(const Neighbor& nbr, size_t lo, size_t size) override
    {
        if (lo < _capacity)
        {
            std::memmove(&_data[lo + 1], &_data[lo], (size - lo) * sizeof(Neighbor));
        }
        _data[lo] = { nbr.id, nbr.distance };

        if (_size < _capacity)
        {
            _size++;
        }
    }

    void resize(size_t capacity)
    {
        _data.resize(capacity + 1, Neighbor(std::numeric_limits<uint32_t>::max(), std::numeric_limits<float>::max()));
        _capacity = capacity;
    }

    virtual size_t current_size(const Neighbor&) const override
    {
        return _size;
    }

  private:
    std::vector<Neighbor> _data;
    size_t _capacity = 0;
    size_t _size = 0;
};

class NeighborExtendColorVector : public NeighborVectorBase
{
  public:
    NeighborExtendColorVector(size_t capacity, uint32_t maxLperSeller, std::vector<uint32_t>& location_to_seller)
        : _capacity(capacity)
        , _data(capacity + 1, NeighborExtendColor(std::numeric_limits<uint32_t>::max(),
                                                                       std::numeric_limits<float>::max(), 0))
        , _location_to_seller(location_to_seller), _maxLperSeller(maxLperSeller)

    {
    }

    virtual Neighbor &operator[](size_t i) override
    {
        return _data[i];
    }

    virtual Neighbor operator[](size_t i) const override
    {
        Neighbor nbr(_data[i].id, _data[i].distance);
        return nbr;
    }

    virtual void insert(const Neighbor &nbr, size_t lo, size_t size) override
    {
        uint32_t color = _location_to_seller[nbr.id];
        auto max_iter = _color_to_max_node.find(color);
        NeighborExtendColor nbr_extend(nbr.id, nbr.distance, color);

        if (lo < _capacity)
        {
            for (size_t i = size - 1; i >= lo; i--)
            {
                // if the color is the same and previous node < lo,
                // insert node and update location
                if (_data[i].color == color && _data[i].previous < lo)
                {
                    nbr_extend.previous = _data[i].previous;
                    _data[i].previous = lo;
                }

                // previous node > lo means the node will move one slot to the right
                if (_data[i].previous >= lo) 
                {
                    _data[i].previous++;
                }
                
                _data[i + 1] = _data[i];
            }

            // current node is the max dist node
            if (nbr_extend.previous == -1 && max_iter != _color_to_max_node.end())
            {
                nbr_extend.previous = max_iter.value();
                _color_to_max_node[color] = lo;
            }

            if (max_iter == _color_to_max_node.end())
            {
                _color_to_max_node[color] = lo;
                _color_to_len[color] = 1;
            }
        }
        _data[lo] = nbr_extend;
    }

    virtual size_t current_size(const Neighbor& nbr) const override
    {
        uint32_t color = _location_to_seller[nbr.id];

        auto len_iter = _color_to_len.find(color);
        if (len_iter == _color_to_len.end()
            || len_iter.value() < _maxLperSeller)
        {
            return _size;
        }

        auto max_iter = _color_to_max_node.find(color);
        
        return max_iter.value();
    }

    void resize(size_t capacity)
    {
        _data.resize(capacity + 1, NeighborExtendColor(std::numeric_limits<uint32_t>::max(),
            std::numeric_limits<float>::max(), 0));
        _capacity = capacity;
    }

  private:
    std::vector<NeighborExtendColor> _data;
    std::vector<uint32_t>& _location_to_seller;
    tsl::robin_map<uint32_t, uint32_t> _color_to_max_node;
    tsl::robin_map<uint32_t, uint32_t> _color_to_len;

    uint32_t _maxLperSeller = 0;
    size_t _capacity = 0;
    size_t _size = 0;
};

class NeighborPriorityQueueBase
{
public:
    NeighborPriorityQueueBase() : _size(0), _capacity(0), _cur(0)
    {
    }

    explicit NeighborPriorityQueueBase(size_t capacity) : _size(0), _capacity(capacity), _cur(0)
    {
    }

    // Inserts the item ordered into the set up to the sets capacity.
    // The item will be dropped if it is the same id as an exiting
    // set item or it has a greated distance than the final
    // item in the set. The set cursor that is used to pop() the
    // next item will be set to the lowest index of an uncheck item
    void insert(const Neighbor& nbr, NeighborVectorBase& neighborVector)
    {
        if (_size == _capacity && neighborVector[_size - 1] < nbr)
        {
            return;
        }

        size_t lo = 0, hi = _size;
        while (lo < hi)
        {
            size_t mid = (lo + hi) >> 1;
            if (nbr < neighborVector[mid])
            {
                hi = mid;
                // Make sure the same id isn't inserted into the set
            }
            else if (neighborVector[mid].id == nbr.id)
            {
                return;
            }
            else
            {
                lo = mid + 1;
            }
        }

        if (lo < _capacity)
        {
            neighborVector.insert(nbr, lo, _size);
        }
        if (_size < _capacity)
        {
            _size++;
        }
        if (lo < _cur)
        {
            _cur = lo;
        }
    }

    Neighbor closest_unexpanded(NeighborVectorBase& neighborVector)
    {
        neighborVector[_cur].expanded = true;
        size_t pre = _cur;
        while (_cur < _size && neighborVector[_cur].expanded)
        {
            _cur++;
        }
        return neighborVector[pre];
    }

    bool has_unexpanded_node() const
    {
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

    void clear()
    {
        _size = 0;
        _cur = 0;
    }

public:
    size_t _size, _capacity, _cur;
//    std::vector<Neighbor> _data;
};

class NeighborPriorityQueueNormal : public NeighborPriorityQueueBase
{
public:
    NeighborPriorityQueueNormal()
        : NeighborPriorityQueueBase()
    {
    }

    explicit NeighborPriorityQueueNormal(size_t capacity)
        : NeighborPriorityQueueBase(capacity)
        , _data(capacity)
    {
    }

    void setup(uint32_t capacity) {
        _data.resize(capacity);
        _capacity = capacity;
    }

    void insert(const Neighbor& nbr)
    {
        NeighborPriorityQueueBase::insert(nbr, _data);
    }

    Neighbor closest_unexpanded()
    {
        return NeighborPriorityQueueBase::closest_unexpanded(_data);
    }

    void reserve(size_t capacity)
    {
        if (capacity > _capacity)
        {
            _data.resize(capacity + 1);
        }
        _capacity = capacity;
    }

    Neighbor& operator[](size_t i)
    {
        return _data[i];
    }

    Neighbor operator[](size_t i) const
    {
        return _data[i];
    }

public:
    NeighborVector _data;
};

class NeighborPriorityQueueExtendColor : public NeighborPriorityQueueBase
{
public:
    NeighborPriorityQueueExtendColor(size_t capacity, uint32_t maxLperSeller, std::vector<uint32_t>& location_to_seller)
        : NeighborPriorityQueueBase()
        , _data(capacity, maxLperSeller, location_to_seller)
    {
    }

    void setup(uint32_t capacity) {
        _data.resize(capacity);
        _capacity = capacity;
    }

    void insert(const Neighbor& nbr)
    {
        NeighborPriorityQueueBase::insert(nbr, _data);
    }

    Neighbor closest_unexpanded()
    {
        return NeighborPriorityQueueBase::closest_unexpanded(_data);
    }

    void reserve(size_t capacity)
    {
        if (capacity > _capacity)
        {
            _data.resize(capacity + 1);
        }
        _capacity = capacity;
    }

    Neighbor& operator[](size_t i)
    {
        return _data[i];
    }

    Neighbor operator[](size_t i) const
    {
        return _data[i];
    }

private:
    NeighborExtendColorVector _data;
};

// Invariant: after every `insert` and `closest_unexpanded()`, `_cur` points to
//            the first Neighbor which is unexpanded.
class NeighborPriorityQueue
{
  public:
    NeighborPriorityQueue() : _size(0), _capacity(0), _cur(0)
    {
    }

    explicit NeighborPriorityQueue(size_t capacity) : _size(0), _capacity(capacity), _cur(0), _data(capacity + 1, Neighbor(std::numeric_limits<uint32_t>::max(), std::numeric_limits<float>::max()))
    {
    }

    void setup(uint32_t capacity) {
        _data.resize(capacity+1,Neighbor(std::numeric_limits<uint32_t>::max(), std::numeric_limits<float>::max()));
        _capacity = capacity;
    }
    // Inserts the item ordered into the set up to the sets capacity.
    // The item will be dropped if it is the same id as an exiting
    // set item or it has a greated distance than the final
    // item in the set. The set cursor that is used to pop() the
    // next item will be set to the lowest index of an uncheck item
    void insert(const Neighbor &nbr)
    {
        if (_size == _capacity && _data[_size - 1] < nbr)
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

        if (lo < _capacity)
        {
            std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(Neighbor));
        }
        _data[lo] = {nbr.id, nbr.distance};
        if (_size < _capacity)
        {
            _size++;
        }
        if (lo < _cur)
        {
            _cur = lo;
        }
    }


    // Deletes the item if found.
    void delete_id(const Neighbor &nbr)
    {
        size_t lo = 0, hi = _size;
        size_t loc = std::numeric_limits<uint64_t>::max();
        while ((lo < hi) && loc == std::numeric_limits<uint64_t>::max())
        {
            size_t mid = (lo + hi) >> 1;
            if (nbr.distance < _data[mid].distance)
            {
                hi = mid;
            }
            else if (nbr.distance > _data[mid].distance)
            {
                lo = mid+1;
            }
            else
            {
                uint32_t itr = 0;
                for (;; itr++) {
                    if (mid + itr < hi) {
                    if (_data[mid+itr].id == nbr.id) {
                    loc = mid+itr;
                    break;
                    }
                    }
                    if(mid - itr >= lo) {
                    if (_data[mid-itr].id == nbr.id) {
                    loc = mid-itr;
                    break;
                    }                    
                    }
                }
            }
        }

        if (loc != std::numeric_limits<uint64_t>::max())
        {
            std::memmove(&_data[loc], &_data[loc+1], (_size - loc - 1) * sizeof(Neighbor));
            _size--;
            _cur = 0;
            while (_cur < _size && _data[_cur].expanded) // RK: inefficient!
            {
                _cur++;
            }
        } else {
            std::cout<<"Found a problem! " << lo <<" " << hi <<" " <<nbr.id << "," << nbr.distance << " " <<_size << std::endl;
            uint32_t pos = 0;
            for (auto &x : this->_data) {
                std::cout<<pos<<":" <<x.id<<"," << x.distance <<" " << std::flush;
                pos++;
            }
            std::cout<<std::endl;
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

    bool has_unexpanded_node() const
    {
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

  public:
    size_t _size, _capacity, _cur;
    std::vector<Neighbor> _data;
};


struct bestCandidates {
    NeighborPriorityQueue best_L_nodes;
    tsl::robin_map<uint32_t, NeighborPriorityQueue> color_to_nodes;
    uint32_t _Lsize = 0;
    uint32_t _maxLperSeller = 0;
    std::vector<uint32_t> &_location_to_seller;

    bestCandidates(std::vector<uint32_t> &location_to_seller) : _location_to_seller(location_to_seller) {
    }

    bestCandidates(uint32_t Lsize, uint32_t maxLperSeller, std::vector<uint32_t> &location_to_seller) : _location_to_seller(location_to_seller) {
        _Lsize = Lsize;
        _maxLperSeller = maxLperSeller;
        best_L_nodes = NeighborPriorityQueue(_Lsize);
    }

    void clear() {
        best_L_nodes.clear();
        color_to_nodes.clear();
    }

    void setup(uint32_t Lsize, uint32_t maxLperSeller) {
        _Lsize = Lsize;
        _maxLperSeller = maxLperSeller;
        best_L_nodes = NeighborPriorityQueue(_Lsize);
    }

    void insert(uint32_t cur_id, float cur_dist) {
            //std::cout<<cur_id << _location_to_seller[cur_id] << " : " << std::flush;
            if (color_to_nodes.find(_location_to_seller[cur_id]) == color_to_nodes.end()) {
                    color_to_nodes[_location_to_seller[cur_id]] = NeighborPriorityQueue(_maxLperSeller);
            }
                auto &cur_list = color_to_nodes[_location_to_seller[cur_id]];
                if (cur_list.size() < _maxLperSeller && best_L_nodes.size() < _Lsize) {
                    cur_list.insert(Neighbor(cur_id, cur_dist));
                    best_L_nodes.insert(Neighbor(cur_id, cur_dist));
                    
                } else if (cur_list.size() == _maxLperSeller) {
                    if (cur_dist < cur_list[_maxLperSeller-1].distance) {
                     best_L_nodes.delete_id(cur_list[_maxLperSeller-1]);   
                    cur_list.insert(Neighbor(cur_id, cur_dist));
                    best_L_nodes.insert(Neighbor(cur_id, cur_dist));                    
                    
                    }
                } else if (cur_list.size() < _maxLperSeller && best_L_nodes.size() == _Lsize) {
                    if (cur_dist < best_L_nodes[_Lsize-1].distance) {
/*                        if (color_to_nodes[_location_to_seller[best_L_nodes[Lsize-1].id]].size() == 0) {
                            std::cout<<"Trying to delete from empty Q. " << best_L_nodes[Lsize-1].id <<" of color " << _location_to_seller[best_L_nodes[Lsize-1].id] << std::endl;
                        }*/
                     color_to_nodes[_location_to_seller[best_L_nodes[_Lsize-1].id]].delete_id(best_L_nodes[_Lsize-1]);
                     cur_list.insert(Neighbor(cur_id, cur_dist));
                     best_L_nodes.insert(Neighbor(cur_id, cur_dist));                    
                    
                    }
                }
    }
};




} // namespace diskann
