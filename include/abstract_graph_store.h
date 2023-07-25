// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <string>
#include <vector>
#include "types.h"

namespace diskann
{

class AbstractGraphStore
{
  public:
    AbstractGraphStore(const size_t max_pts) : _capacity(max_pts)
    {
    }

    virtual int load(const std::string &index_path_prefix) = 0;
    virtual int store(const std::string &index_path_prefix, const size_t active_points) = 0;

    virtual std::vector<location_t> &get_neighbours(const location_t i) = 0;
    virtual void set_neighbours(const location_t i, std::vector<location_t> &neighbors) = 0;

    virtual size_t resize_graph(const size_t new_size) = 0;
    virtual void clear_graph() = 0;

    virtual size_t get_num_frozen_points() = 0;
    virtual size_t get_max_range_of_loaded_graph() = 0;

    virtual uint32_t get_max_observed_degree() = 0;
    virtual void set_max_observed_degree(uint32_t max_observed_degree) = 0;

    virtual uint32_t get_start() = 0;
    virtual void set_start(uint32_t start) = 0;

    // Active points in graph, it is different then total_points capacity
    /*virtual size_t get_active_points() = 0;
    virtual void set_active_points(size_t active_points) = 0;*/

    // returns new size after shrinking graph
    virtual size_t shrink_to_fit() = 0;

    // Total internal points _max_points + _num_frozen_points
    size_t get_total_points()
    {
        return _capacity;
    }

  protected:
    // Internal function, changes total points when resize_graph is called.
    void set_total_points(size_t new_capacity)
    {
        _capacity = new_capacity;
    }

  private:
    size_t _capacity;
};

} // namespace diskann