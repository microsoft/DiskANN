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
    virtual int store(const std::string &index_path_prefix) = 0;

    virtual void get_adj_list(const location_t i, std::vector<location_t> &neighbors) = 0;
    virtual void set_adj_list(const location_t i, std::vector<location_t> &neighbors) = 0;

    virtual size_t get_num_frozen_points() = 0;
    virtual size_t get_max_range_of_loaded_graph() = 0;

    virtual uint32_t get_max_observed_degree() = 0;
    virtual void set_max_observed_degree(uint32_t max_observed_degree) = 0;
    virtual uint32_t get_start() = 0;
    virtual void set_start(uint32_t start) = 0;

    virtual size_t get_active_points() = 0;
    virtual void set_active_points(size_t active_points) = 0;

    virtual std::vector<std::vector<uint32_t>> &get_graph() = 0;

    size_t get_max_points()
    {
        return _capacity;
    }

  private:
    size_t _capacity;
};

} // namespace diskann
