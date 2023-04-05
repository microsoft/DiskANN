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
    AbstractDataStore(const location_t capacity, const size_t dim)
        : _capacity(capacity), _num_points(0), _dim(dim) 
    {
    }

    // Return number of points returned
    virtual location_t load(const std::string &filename) = 0;
    virtual void store(const std::string &filename) = 0;

    virtual void populate_data(const data_t * vectors, const location_t num_pts) = 0;
    virtual void populate_data(const std::string &filename, const size_t offset) = 0;
    
    virtual void get_vector(const location_t i, data_t* dest) const = 0;
    virtual void set_vector(const location_t i, const data_t *const vector)  = 0;
    virtual void prefetch_vector(const location_t loc) = 0;


    virtual void resize(const location_t new_size) = 0;

    virtual void reposition_points(const location_t start_loc, const location_t end_loc,
                                   const location_t num_points) = 0;
    virtual void copy_points(const location_t from_loc, const location_t to_loc, const location_t num_points) = 0;
    //Returns the point in the dataset that is closest to the mean of all points in the dataset
    virtual location_t calculate_medoid() const = 0;
    
    virtual location_t capacity() const 
    {
        return _capacity;
    }
    virtual location_t get_num_points() const
    {
        return _num_points;
    }

    virtual float get_distance(const data_t* query, const location_t loc) const = 0;
    virtual void get_distance(const data_t *query, const location_t *locations, const uint32_t location_count,
                      float *distances) const = 0;
    virtual float get_distance(const location_t loc1, const location_t loc2) const = 0;


    virtual size_t get_dims() const 
    {
        return _dim;
    }

    virtual size_t get_aligned_dim() const
    {
        return _dim;
    }

  protected:
    location_t _capacity;
    location_t _num_points;

    const size_t _dim;
};

} // namespace diskann