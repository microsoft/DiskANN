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
    AbstractDataStore(const location_t capacity, const size_t dim) : _capacity(capacity), _dim(dim)
    {
    }

    // Return number of points returned
    virtual location_t load(const std::string &filename) = 0;

    // Why does store take num_pts? Since store only has capacity, but we allow resizing
    // we can end up in a situation where the store has spare capacity. To optimize disk
    // utilization, we pass the number of points that are "true" points, so that the store
    // can discard the empty locations before saving.
    virtual size_t save(const std::string &filename, const location_t num_pts) = 0;

    virtual location_t capacity() const
    {
        return _capacity;
    }

    virtual size_t get_dims() const
    {
        return _dim;
    }

    // by default, aligned dim = dim, some stores can align the data differently, so we may have different values
    virtual size_t get_aligned_dim() const
    {
        return _dim;
    }

    // populate the store with bulk vectors (either as pointer or bin file), potentially after normalizing the vectors
    // if the metric deems so
    virtual void populate_data(const data_t *vectors, const location_t num_pts) = 0;
    virtual void populate_data(const std::string &filename, const size_t offset) = 0;

    // reverse of populate, save the first num_pts many points back to bin file
    virtual void save_data_to_bin(const std::string &filename, const location_t num_pts) = 0;

    virtual void resize(const location_t num_points)
    {
        if (num_points > _capacity)
        {
            expand(num_points);
        }
        else if (num_points < _capacity)
        {
            shrink(num_points);
        }
        else
        {
            // ignore.
        }
    }

    // operations on vectors
    virtual void get_vector(const location_t i, data_t *dest) const = 0;
    virtual void set_vector(const location_t i, const data_t *const vector) = 0;
    virtual void prefetch_vector(const location_t loc) = 0;

    // internal shuffle operations to move around vectors
    virtual void reposition_points(const location_t start_loc, const location_t end_loc,
                                   const location_t num_points) = 0;
    virtual void copy_points(const location_t from_loc, const location_t to_loc, const location_t num_points) = 0;

    // metric specific operations

    virtual float get_distance(const data_t *query, const location_t loc) const = 0;
    virtual void get_distance(const data_t *query, const location_t *locations, const uint32_t location_count,
                              float *distances) const = 0;
    virtual float get_distance(const location_t loc1, const location_t loc2) const = 0;

    // stats of the data stored in store
    // Returns the point in the dataset that is closest to the mean of all points in the dataset
    virtual location_t calculate_medoid() const = 0;

  protected:
    // Expand the datastore to new_num_points.
    virtual void expand(const location_t new_num_points) = 0;
    // Shrink the datastore to new_num_points. This function should be called after compaction to free unused memory.
    virtual void shrink(const location_t new_num_points) = 0;

    location_t _capacity;
    size_t _dim;
};

} // namespace diskann