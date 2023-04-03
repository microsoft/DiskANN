// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>

#include "types.h"

namespace diskann
{

template <typename data_t> class AbstractDataStore
{
  public:
    AbstractDataStore(const location_t max_pts, const size_t dim)
        : _max_pts(max_pts), _num_pts(0), _dim(dim) 
    {
    }

    // Return number of points returned
    virtual size_t load(const std::string &filename) = 0;
    virtual void store(const std::string &filename) = 0;

    virtual data_t *get_vector(location_t i) = 0;
    virtual void set_vector(const location_t i, const data_t *const vector) = 0;

    location_t get_max_pts()
    {
        return _max_pts;
    }

    location_t get_num_pts()
    {
        return _num_pts;
    }

    virtual void get_distance(const T *query, const location_t *locations, const uint32_t location_count,
                      float *distances) = 0;


    size_t get_dims()
    {
        return _dim;
    }

  protected:
    location_t _max_pts;
    location_t _num_pts;

    const size_t _dim;
};

} // namespace diskann