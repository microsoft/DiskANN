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
        : _max_points(max_pts), _num_points(0), _dim(dim) 
    {
    }

    // Return number of points returned
    virtual location_t load(const std::string &filename) = 0;
    virtual void store(const std::string &filename) = 0;

    virtual void populate_data(const data_t * vectors, const location_t num_pts) = 0;
    virtual void populate_data(const std::string &filename, const size_t offset) = 0;
    
    virtual void get_vector(const location_t i, data_t* dest) const = 0;
    virtual void set_vector(const location_t i, const data_t *const vector)  = 0;


    virtual void reposition_points(const location_t start_loc, const location_t end_loc,
                                   const location_t num_points) = 0;

    location_t get_max_points() const 
    {
        return _max_points;
    }
    location_t get_num_points() const
    {
        return _num_points;
    }

    virtual void get_distance(const data_t *query, const location_t *locations, const uint32_t location_count,
                      float *distances) const = 0;
    virtual float get_distance(const location_t loc1, const location_t loc2) const = 0;


    size_t get_dims() const 
    {
        return _dim;
    }

  protected:
    location_t _max_points;
    location_t _num_points;

    const size_t _dim;
};

} // namespace diskann