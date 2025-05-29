#pragma once
#include <cstdint>
#include <tsl/robin_map.h>
#include "simple_bitmask.h"

namespace diskann
{

struct ColorInfo
{
    int _max_node_location = -1; // location of the max node in the queue
    uint32_t _len = 0;           // length of the color in the queue

    ColorInfo() = default;

    ColorInfo(int max_node_location, uint32_t len)
        : _max_node_location(max_node_location)
        , _len(len)
    {
    }

    ColorInfo& operator=(const ColorInfo& other)
    {

        _max_node_location = other._max_node_location;
        _len = other._len;

        return *this;
    }
};

class ColorInfoMapBase
{
public:
// Iterator interface
    class Iterator 
    {
    public:

        Iterator(ColorInfoMapBase* map, uint32_t key, const ColorInfo& value, bool is_end) 
            : _map(map)
            ,  _key(key)
            , _value(value)
            , _end(is_end)
        {
        }

        ~Iterator() = default;

        // Core iterator operations
        Iterator& operator++()
        {
            _map->move_next(*this);
            return *this;
        }

        bool operator!=(const Iterator& other)
        {
            return _end != other._end || _key != other._key;
        }

        bool operator==(const Iterator& other)
        {
            return !(*this != other);
        }
        
        const ColorInfo& operator*() const
        {
            return _value; // Return current value
        }

        uint32_t key() const
        {
            return _key; // Return current key
        }

        const ColorInfo& value() const
        {
            return _value; // Return current value
        }

        ColorInfo& value()
        {
            return _value; // Return current value (non-const)
        }
        private:
            ColorInfoMapBase* _map; // Reference to the map being iterated
            uint32_t _key = 0; // Current key in the iteration
            ColorInfo _value; // Current value in the iteration
            bool _end = false; // Flag to indicate if the end of the map is reached
    };


    virtual ColorInfo& operator[](uint32_t color) = 0;

    virtual const ColorInfo& operator[](uint32_t color) const = 0;

    virtual void clear() = 0;

    virtual ~ColorInfoMapBase() = default;

    virtual Iterator begin() = 0;

    virtual Iterator end() = 0;

    virtual Iterator find(uint32_t key) = 0;

    virtual ColorInfo& at(uint32_t key)
    {
        return (*this)[key];
    }

    virtual void move_next(Iterator& it) = 0;

    
};

class ColorInfoMap : public ColorInfoMapBase
{
public:
    ColorInfoMap() = default;

    void reserve(size_t capacity)
    {
        _color_to_info.reserve(capacity);
    }
    
    virtual ColorInfo& operator[](uint32_t color) override
    {
        return _color_to_info[color];
    }

    virtual const ColorInfo& operator[](uint32_t color) const override
    {
        return _color_to_info.at(color);
    }

    virtual void clear() override
    {
        _color_to_info.clear();
    }

    virtual Iterator begin() override
    {
        return Iterator(this, 0, ColorInfo(), false);
    }

    virtual Iterator end() override
    {
        return Iterator(this, 0, ColorInfo(), true);
    }

    virtual Iterator find(uint32_t key) override
    {
        auto it = _color_to_info.find(key);
        if (it != _color_to_info.end())
            return Iterator(this, key, it->second, false);
        else
            return end();
    }

    virtual void move_next(Iterator& it) override
    {
        auto next_it = std::next(_color_to_info.find(it.key()));
        if (next_it != _color_to_info.end())
            it = Iterator(this, next_it->first, next_it->second, false);
        else
            it = end();
    }
    
private:
    tsl::robin_map<uint32_t, ColorInfo> _color_to_info;
};

class ColorInfoVector : public ColorInfoMapBase
{
public:
    const static ColorInfo c_empty_info;
    ColorInfo _empty_info; // only used in error case

    ColorInfoVector(size_t capacity)
        : _color_info_vector(capacity + 1)
        , _bitmask_buf(simple_bitmask::get_bitmask_size(capacity + 1), simple_bitmask::get_bitmask_size(capacity + 1))
        , _slot_mask(_bitmask_buf._buf.data(), _bitmask_buf._bitmask_size)
    {
    }

    virtual ColorInfo& operator[](uint32_t color) override
    {
        if (color >= _color_info_vector.size()) 
        {
            _empty_info = ColorInfo();
            return _empty_info;
        }
        _slot_mask.set(color);
        return _color_info_vector[color];
    }

    virtual const ColorInfo& operator[](uint32_t color) const override
    {
        if (color >= _color_info_vector.size() || !_slot_mask.test(color)) 
        {
            return c_empty_info;
        }
        return _color_info_vector[color];
    }

    virtual void clear() override
    {
        _slot_mask.clear();
    }

    virtual Iterator begin() override
    {
        size_t first_location = get_valid_location(0);

        if (first_location == _color_info_vector.size())
        {
            return end();
        }
        return Iterator(this, static_cast<uint32_t>(first_location), _color_info_vector[first_location], false);
    }

    virtual Iterator end() override
    {
        return Iterator(this, static_cast<uint32_t>(_color_info_vector.size()), ColorInfo(), true);
    }

    virtual Iterator find(uint32_t key) override
    {
        if (key < _color_info_vector.size() && _slot_mask.test(key)) 
        {
            return Iterator(this, key, _color_info_vector[key], false);
        }
        return end();
    }

    virtual void move_next(Iterator& it) override
    {
        size_t next_location = get_valid_location(it.key() + 1);
        if (next_location == _color_info_vector.size()) 
        {
            it = end();
            return;
        }
        it = Iterator(this, static_cast<uint32_t>(next_location), _color_info_vector[next_location], false);

    }

    size_t get_valid_location(size_t current) const
    {
        for (size_t i = current; i < _color_info_vector.size(); ++i)
         {
            if (_slot_mask.test(i)) 
            {
                return i;
            }
        }

        return _color_info_vector.size(); // Return size if no valid location found
    }

private:
    std::vector<ColorInfo> _color_info_vector;
    simple_bitmask_buf _bitmask_buf;
    simple_bitmask _slot_mask;
};

}