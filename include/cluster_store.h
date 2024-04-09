// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>
#include <string>

#include "types.h"
#include "windows_customizations.h"
#include "distance.h"

namespace diskann
{

template <typename data_t> class AbstractClusterStore
{
  public:
    AbstractClusterStore(const size_t dim) {
    }

    virtual ~AbstractClusterStore() = default;

    // Return number of clusters in the index
    DISKANN_DLLEXPORT virtual uint32_t load(const std::string &filename) = 0;

    DISKANN_DLLEXPORT virtual size_t save(const std::string &filename) = 0;

//    DISKANN_DLLEXPORT virtual location_t capacity() const;

    DISKANN_DLLEXPORT virtual size_t get_dims() const = 0;

    DISKANN_DLLEXPORT virtual void add_cetroids(float *clusters, uint32_t num_clusters) = 0;

    // populate the store with vectors (either from a pointer or bin file),
    // potentially after pre-processing the vectors if the metric deems so
    // e.g., normalizing vectors for cosine distance over floating-point vectors
    // useful for bulk or static index building.
    DISKANN_DLLEXPORT virtual void assign_data_to_clusters(const data_t *vectors, std::vector<uint32_t> &ids) = 0;

    // operations on vectors
    // like populate_data function, but over one vector at a time useful for
    // streaming setting
    DISKANN_DLLEXPORT virtual void get_closest_clusters(const data_t *const query, const uint32_t num_closest, std::vector<uint32_t> &closest_clusters) = 0;
    DISKANN_DLLEXPORT virtual void get_cluster_members(const uint32_t cluster_id) = 0;

  protected:

    float* _cluster_centroids;
    size_t _num_clusters;
    size_t _dim;
};

} // namespace diskann
