// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>
#include <string>

#include "types.h"
#include "windows_customizations.h"
#include "id_list.h"
#include "distance.h"
#include "scratch.h"

namespace diskann
{

template <typename data_t> class AbstractClusterStore
{
  public:
    AbstractClusterStore(const size_t dim)
    {
        _dim = dim;
    }

    virtual ~AbstractClusterStore() = default;

    // Return number of clusters in the index
    virtual uint32_t load(const std::string &filename) = 0;

    virtual size_t save(const std::string &filename) = 0;

    //    DISKANN_DLLEXPORT virtual location_t capacity() const;

    size_t get_dims()
    {
        return _dim;
    };

    size_t get_num_clusters()
    {
        return _num_clusters;
    };

    virtual void add_cetroids(float *clusters, uint32_t num_clusters) = 0;

    // populate the store with vectors (either from a pointer or bin file),
    // potentially after pre-processing the vectors if the metric deems so
    // e.g., normalizing vectors for cosine distance over floating-point vectors
    // useful for bulk or static index building.
    virtual void assign_data_to_clusters(data_t *vectors, std::vector<uint32_t> &ids) = 0;

    // operations on vectors
    // like populate_data function, but over one vector at a time useful for
    // streaming setting
    virtual void get_closest_clusters(data_t *query, const uint32_t num_closest,
                                      InMemQueryScratch<data_t> *scratch) = 0;
    virtual void get_cluster_members(const uint32_t cluster_id, AbstractIdList &output_list) = 0;

  protected:
    float *_cluster_centroids = nullptr;
    float *_cluster_norms = nullptr;
    size_t _num_clusters = 0;
    size_t _dim;
};

template <typename data_t> class InMemClusterStore : public AbstractClusterStore<data_t>
{
  public:
    InMemClusterStore(const size_t dim);

    virtual ~InMemClusterStore();

    // Return number of clusters in the index
    virtual uint32_t load(const std::string &filename) override;

    virtual size_t save(const std::string &filename) override;

    //    DISKANN_DLLEXPORT virtual location_t capacity() const;

    virtual void add_cetroids(float *clusters, uint32_t num_clusters) override;

    // populate the store with vectors (either from a pointer or bin file),
    // potentially after pre-processing the vectors if the metric deems so
    // e.g., normalizing vectors for cosine distance over floating-point vectors
    // useful for bulk or static index building.
    virtual void assign_data_to_clusters(data_t *vectors, std::vector<uint32_t> &ids) override;

    // operations on vectors
    // like populate_data function, but over one vector at a time useful for
    // streaming setting
    virtual void get_closest_clusters(data_t *query, const uint32_t num_closest,
                                      InMemQueryScratch<data_t> *scratch) override;

    virtual void get_cluster_members(const uint32_t cluster_id, AbstractIdList &output_list) override;

  protected:
    std::vector<diskann::RoaringIdList> _posting_lists;
};

} // namespace diskann
