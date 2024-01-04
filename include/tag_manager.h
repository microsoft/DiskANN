#pragma once
#include <cstdint>
#include <string>

#include "common_includes.h"
#include "natural_number_map.h"
#include "natural_number_set.h"
#include "tsl/sparse_map.h"
#include <mutex>
#include <limits>
#include <vector>

namespace diskann
{

class tag_data
{
public:
    virtual void set_internal_location(std::uint32_t location) = 0;
};

class tag_manager_base
{
public:
    virtual void reset_location(const std::vector<std::uint32_t> new_locations) = 0;
};


template<typename TagT>
class default_tag_manager : public tag_manager_base
{
public:
    void set_total_points(size_t total_internal_points)
    {
        _location_to_tag.reserve(total_internal_points);
        _tag_to_location.reserve(total_internal_points);
    }

    bool add_location_tag(std::uint32_t location, TagT tag)
    {
        _tag_to_location[tag] = location;
        _location_to_tag[location] = tag;
    }
    
    bool is_tag_existed(TagT tag)
    {
        return _tag_to_location.find(tag) != _tag_to_location.end();
    }

    std::uint32_t get_location(TagT tag)
    {
        std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
        if (_tag_to_location.find(tag) == _tag_to_location.end())
        {
            return (std::numeric_limits<std::uint32_t>::max)();
        }

        return _tag_to_location[tag];
    }

    void delete_location(std::uint32_t location)
    {
        std::unique_lock<std::shared_timed_mutex> tl(_lock);

        TagT tag = _location_to_tag[location];
        _location_to_tag.erase(location);
        _tag_to_location.erase(tag);
    }

    virtual void reset_location(const std::vector<std::uint32_t> new_locations) override
    {
        _tag_to_location.clear();
        for (auto pos = _location_to_tag.find_first(); pos.is_valid(); pos = _location_to_tag.find_next(pos))
        {
            const auto tag = _location_to_tag.get(pos);
            _tag_to_location[tag] = new_locations[pos._key];
        }
        _location_to_tag.clear();
        for (const auto& iter : _tag_to_location)
        {
            _location_to_tag.set(iter.second, iter.first);
        }
    }

    size_t get_size() const
    {
        assert(_tag_to_location.size() == _location_to_tag.size());

        return _tag_to_location.size();
    }

private:
    tsl::sparse_map<TagT, uint32_t> _tag_to_location;
    natural_number_map<uint32_t, TagT> _location_to_tag;

    std::shared_timed_mutex _lock;
};

template<typename TagT>
class default_tag_data : public tag_data
{
public:
    default_tag_data(default_tag_manager<TagT>& default_tag_manger, TagT tag)
        : _default_tag_manger(default_tag_manger)
        , _tag(tag)
    {
    }

    virtual void set_internal_location(std::uint32_t location) override
    {
        _default_tag_manger.add_location_tag(location, _tag);
    }
    
private:
    TagT _tag;
    default_tag_manager<TagT>& _default_tag_manger;
};

}