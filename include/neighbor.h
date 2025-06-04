// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <mutex>
#include <vector>
#include <assert.h>
#include <tsl/robin_map.h>
//#include "utils.h"
#include "color_info.h"

namespace diskann
{

struct Neighbor
{
    unsigned id;
    float distance;
    bool expanded;

    Neighbor() = default;

    Neighbor(unsigned id, float distance) : id{ id }, distance{ distance }, expanded(false)
    {
    }

    inline bool operator<(const Neighbor& other) const
    {
        return distance < other.distance || (distance == other.distance && id < other.id);
    }

    inline bool operator==(const Neighbor& other) const
    {
        return (id == other.id);
    }
};

struct NeighborExtendColor : public Neighbor
{
    uint32_t color;
    int previous = -1;
    bool max_flag = false;

    NeighborExtendColor() = default;

    NeighborExtendColor(unsigned id, float distance, uint32_t color) : Neighbor(id, distance), color(color)
    {
    }

    inline bool operator<(const NeighborExtendColor& other) const
    {
        return distance < other.distance || (distance == other.distance && id < other.id);
    }

    inline bool operator==(const NeighborExtendColor& other) const
    {
        return (id == other.id);
    }
};



class NeighborVectorBase
{
public:
    virtual Neighbor& operator[](size_t i) = 0;

    virtual Neighbor operator[](size_t i) const = 0;

    virtual void insert(const Neighbor& nbr, int insert_loc, int kick_lo, int quene_len) = 0;

    virtual size_t get_kick_location(const Neighbor& nbr, size_t quene_len) const = 0;

    virtual void clear() = 0;
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

    virtual void insert(const Neighbor& nbr, int lo, int /*kick_loc*/, int size) override
    {
        std::memmove(&_data[lo + 1], &_data[lo], (size - lo) * sizeof(Neighbor));
        _data[lo] = { nbr.id, nbr.distance };
    }

    void resize(size_t capacity)
    {
        _data.resize(capacity + 1, Neighbor(std::numeric_limits<uint32_t>::max(), std::numeric_limits<float>::max()));
        _capacity = capacity;
    }

    virtual size_t get_kick_location(const Neighbor&, size_t quene_len) const override
    {
        return quene_len;
    }

    virtual void clear() override
    {
        // no need additional clear up
    }

private:
    std::vector<Neighbor> _data;
    size_t _capacity = 0;
};

class NeighborExtendColorVector : public NeighborVectorBase
{
public:
    const static uint32_t s_vector_size_limited = 1000000;

    //struct ColorInfo
    //{
    //      int _max_node_location = -1; // location of the max node in the queue
    //      uint32_t _len = 0;           // length of the color in the queue

    //      ColorInfo() = default;

    //      ColorInfo(int max_node_location, uint32_t len) 
    //          : _max_node_location(max_node_location)
    //          , _len(len)
    //      {
    //      }
    //};

    NeighborExtendColorVector(const std::vector<uint32_t>& location_to_seller)
        : _location_to_seller(location_to_seller)
    {
        //_color_to_max_node.reserve(1000);
        //_color_to_len.reserve(1000);
    //    _color_to_info.reserve(1000);
    }

    NeighborExtendColorVector(size_t capacity, uint32_t maxLperSeller, uint32_t uniqueSellerCount, const std::vector<uint32_t>& location_to_seller)
        : _capacity(capacity)
        , _data(capacity + 1, NeighborExtendColor(std::numeric_limits<uint32_t>::max(),
            std::numeric_limits<float>::max(), 0))
        , _location_to_seller(location_to_seller), _maxLperSeller(maxLperSeller)

    {
        if (uniqueSellerCount < s_vector_size_limited)
        {
            _color_to_info = std::make_unique<ColorInfoVector>(uniqueSellerCount);
        }
        else
        {
            auto color_to_info = std::make_unique<ColorInfoMap>();
            color_to_info->reserve(uniqueSellerCount);
            _color_to_info = std::move(color_to_info);
        }
        //_color_to_max_node.reserve(1000);
        //_color_to_len.reserve(1000);
    }

    virtual Neighbor& operator[](size_t i) override
    {
        return _data[i];
    }

    virtual Neighbor operator[](size_t i) const override
    {
        Neighbor nbr(_data[i].id, _data[i].distance);
        return nbr;
    }

    virtual void insert(const Neighbor& nbr, int insert_lo, int kick_loc, int quene_len) override
    {
        assert(nbr.id < _location_to_seller.size());

        auto& color_to_info = *_color_to_info;

        uint32_t color = _location_to_seller[nbr.id];

        NeighborExtendColor nbr_extend(nbr.id, nbr.distance, color);

        // keep the kick out node to further process
        NeighborExtendColor last = _data[kick_loc];

        for (int i = kick_loc - 1; i >= insert_lo; i--)
        {
            // if the color is the same and previous node < insert_lo,
            // insert node and update location
            if (_data[i].color == color && _data[i].previous < insert_lo)
            {
                nbr_extend.previous = _data[i].previous;
                _data[i].previous = insert_lo;
            }
            else if (_data[i].previous >= insert_lo)
            {
                // previous node > lo means the node will move one slot to the right
                _data[i].previous++;
            }

            // update max node location
            if (_data[i].max_flag)
            {
                color_to_info[_data[i].color]._max_node_location = i + 1;
                //    _color_to_max_node[_data[i].color] = i + 1;
            }

            _data[i + 1] = _data[i];
        }

        // update the node from kick_lo to the end of the queue,
        // there are two possible cases:
        // 1. the queue isn't full and queue_len point to next empty slot
        // 2. the queue isn full and queue_len point to the tail slot
        // if the node at queue_len is empty slot, the previous == -1,
        // so it's no impact in the loop, to make the access safty, we have to allocate (capacity + 1) slots
        for (int i = kick_loc + 1; i < quene_len; i++)
        {
            if (_data[i].previous == kick_loc)
            {
                // the node is removed, link to previous node
                _data[i].previous = last.previous;
            }
            else if (_data[i].previous >= insert_lo && _data[i].previous < kick_loc)
            {
                // previous node is in the moving range
                _data[i].previous++;
            }
        }

        // it's not insert to empty slot, process kick out node
        if (kick_loc != quene_len)
        {
            uint32_t last_node_color = last.color;
            if (last.previous >= insert_lo)
            {
                last.previous++;
            }
            //    _color_to_max_node[last_node_color] = last.previous;
            if (last.previous != -1)
            {
                _data[last.previous].max_flag = true;
            }
            //     _color_to_len[last_node_color]--;

            auto& color_info = color_to_info.at(last_node_color);
            color_info._max_node_location = last.previous;
            color_info._len--;
        }

        // process the new node
        auto color_info_iter = color_to_info.find(color);

        //auto max_iter = _color_to_max_node.find(color);
        // current node is the max dist node
        if (nbr_extend.previous == -1
            && color_info_iter != color_to_info.end()
            && color_info_iter.value()._max_node_location != -1
            && insert_lo > color_info_iter.value()._max_node_location)
        {
            nbr_extend.previous = color_info_iter.value()._max_node_location;
            _data[color_info_iter.value()._max_node_location].max_flag = false;
            color_to_info[color]._max_node_location = insert_lo;
            nbr_extend.max_flag = true;
        }
        else if (color_info_iter == color_to_info.end()
            || color_info_iter.value()._max_node_location == -1)
        {
            // new color insert to queue
            color_to_info[color] = ColorInfo(insert_lo, 0);

            //_color_to_max_node[color] = insert_lo;
            //_color_to_len[color] = 0;
            nbr_extend.max_flag = true;
        }

        // update the length of the color
        color_to_info[color]._len++;

        _data[insert_lo] = nbr_extend;
    }

    virtual size_t get_kick_location(const Neighbor& nbr, size_t queue_len) const override
    {
        assert(nbr.id < _location_to_seller.size());
        uint32_t color = _location_to_seller[nbr.id];

        auto color_info_iter = _color_to_info->find(color);

        // auto len_iter = _color_to_len.find(color);
        if (color_info_iter == _color_to_info->end()
            || color_info_iter.value()._len < _maxLperSeller)
        {
            return queue_len;
        }

        // auto max_iter = _color_to_max_node.find(color);

        assert(color_info_iter.value()._max_node_location >= 0);

        assert(color_info_iter.value()._max_node_location < queue_len);

        return color_info_iter.value()._max_node_location;
    }

    void resize(size_t capacity, uint32_t maxLperSeller, uint32_t uniqueSellerCount)
    {
        _data.resize(capacity + 1, NeighborExtendColor(std::numeric_limits<uint32_t>::max(),
            std::numeric_limits<float>::max(), 0));

        _capacity = capacity;
        _maxLperSeller = maxLperSeller;
        if (_color_to_info == nullptr)
        {
            if (uniqueSellerCount < s_vector_size_limited)
            {
                std::cout << "unique seller " << uniqueSellerCount << "ColorInfoVector created" << std::endl;
                _color_to_info = std::make_unique<ColorInfoVector>(uniqueSellerCount);
            }
            else
            {
                std::cout << "unique seller " << uniqueSellerCount << "ColorInfoMap created" << std::endl;
                auto color_to_info = std::make_unique<ColorInfoMap>();
                color_to_info->reserve(uniqueSellerCount);
                _color_to_info = std::move(color_to_info);
            }
        }
    }

    virtual void clear() override
    {
        // no need additional clear up
        //_color_to_len.clear();
        //_color_to_max_node.clear();
        _color_to_info->clear();
        _data.clear();
    }


private:
    std::vector<NeighborExtendColor> _data;
    const std::vector<uint32_t>& _location_to_seller;
    //  tsl::robin_map<uint32_t, ColorInfo> _color_to_info;
    std::unique_ptr<ColorInfoMapBase> _color_to_info;
    //tsl::robin_map<uint32_t, int> _color_to_max_node;
    //tsl::robin_map<uint32_t, uint32_t> _color_to_len;

    uint32_t _maxLperSeller = 0;
    size_t _capacity = 0;
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

    virtual void insert(const Neighbor& nbr) = 0;
    virtual Neighbor closest_unexpanded() = 0;
    virtual Neighbor& operator[](size_t i) = 0;
    virtual Neighbor operator[](size_t i) const = 0;

    // Inserts the item ordered into the set up to the sets capacity.
    // The item will be dropped if it is the same id as an exiting
    // set item or it has a greated distance than the final
    // item in the set. The set cursor that is used to pop() the
    // next item will be set to the lowest index of an uncheck item
    void insert(const Neighbor& nbr, NeighborVectorBase& neighborVector)
    {
        auto kick_loc = neighborVector.get_kick_location(nbr, _size);
        assert(kick_loc <= _capacity);

        // if the kick location is out of the max size,
        // that means it should be kick the last node
        if (kick_loc == _capacity)
        {
            kick_loc--;
        }

        // if the kick location isn't empty slot, then check the dist 
        if ((kick_loc < _size)
            && neighborVector[kick_loc] < nbr)
        {
            return;
        }


        size_t lo = 0, hi = kick_loc;
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

        // for the case kick_loc < size, it should be always find location to insert
        // because neighborVector[kick_loc - 1] < nbr already make sure it should be insert.
        if (lo < _capacity)
        {
            neighborVector.insert(nbr, static_cast<int>(lo), static_cast<int>(kick_loc), static_cast<int>(_size));
        }
        if (kick_loc == _size && _size < _capacity)
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

    virtual void clear()
    {
        _size = 0;
        _cur = 0;
    }

public:
    size_t _size, _capacity, _cur;
    //    std::vector<Neighbor> _data;
};

class NeighborPriorityQueue : public NeighborPriorityQueueBase
{
public:
    NeighborPriorityQueue()
        : NeighborPriorityQueueBase()
    {
    }

    explicit NeighborPriorityQueue(size_t capacity)
        : NeighborPriorityQueueBase(capacity)
        , _data(capacity)
    {
    }

    void setup(uint32_t capacity) {
        _data.resize(capacity);
        _capacity = capacity;
    }

    virtual void insert(const Neighbor& nbr) override
    {
        NeighborPriorityQueueBase::insert(nbr, _data);
    }

    virtual Neighbor closest_unexpanded() override
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

    virtual Neighbor& operator[](size_t i) override
    {
        return _data[i];
    }

    virtual Neighbor operator[](size_t i) const override
    {
        return _data[i];
    }

public:
    NeighborVector _data;
};

class NeighborPriorityQueueExtendColor : public NeighborPriorityQueueBase
{
public:
    NeighborPriorityQueueExtendColor(const std::vector<uint32_t>& location_to_seller)
        : NeighborPriorityQueueBase(), _data(location_to_seller)
    {
    }

    NeighborPriorityQueueExtendColor(size_t capacity, uint32_t maxLperSeller, uint32_t uniqueSellerCount, const std::vector<uint32_t>& location_to_seller)
        : NeighborPriorityQueueBase()
        , _data(capacity, maxLperSeller, uniqueSellerCount, location_to_seller)
    {
        _capacity = capacity;
    }

    void setup(uint32_t capacity, uint32_t maxLperSeller, uint32_t uniqueSellerCount) {
        _data.resize(capacity, maxLperSeller, uniqueSellerCount);
        _capacity = capacity;
    }

    void get_data(std::vector<Neighbor>& data) const
    {
        for (size_t i = 0; i < _size; i++)
        {
            data.push_back(_data[i]);
        }
    }

    virtual void insert(const Neighbor& nbr) override
    {
        NeighborPriorityQueueBase::insert(nbr, _data);
    }

    virtual Neighbor closest_unexpanded() override
    {
        return NeighborPriorityQueueBase::closest_unexpanded(_data);
    }

    //void reserve(size_t capacity)
    //{
    //    if (capacity > _capacity)
    //    {
    //        _data.resize(capacity + 1);
    //    }
    //    _capacity = capacity;
    //}

    virtual void clear() override
    {
        NeighborPriorityQueueBase::clear();
        _data.clear();
    }

    virtual Neighbor& operator[](size_t i) override
    {
        return _data[i];
    }

    virtual Neighbor operator[](size_t i) const override
    {
        return _data[i];
    }

private:
    NeighborExtendColorVector _data;
};

} // namespace diskann