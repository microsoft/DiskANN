// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <shared_mutex>
#include <memory>

#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "tsl/sparse_map.h"
#include "boost/dynamic_bitset.hpp"

#include "abstract_data_store.h"

#include "distance.h"
#include "natural_number_map.h"
#include "natural_number_set.h"

namespace diskann
{
template <typename data_t> 
class InMemDataStore : public AbstractDataStore<data_t>
{
  public:
    InMemDataStore(const location_t max_pts, const size_t dim, std::shared_ptr<Distance<data_t>> distance_metric);
    virtual ~InMemDataStore();

    virtual location_t load(const std::string &filename) override;
    virtual void store(const std::string &filename) override;

    //Populate internal data from unaligned data while doing alignment and any normalization that is required.
    virtual void populate_data(const data_t *vectors, const location_t num_pts) override;
    virtual void populate_data(const std::string &filename, const size_t offset) override;

    virtual void get_vector(const location_t i, data_t *target) const override;
    virtual void set_vector(const location_t i, const data_t *const vector) override;

    virtual void reposition_points(const location_t old_location_start, const location_t new_location_start,
                                   const location_t num_points) override;
    virtual void copy_points(const location_t from_loc, const location_t to_loc, const location_t num_points) override;
    virtual location_t calculate_medoid() const override;

    virtual float get_distance(const data_t *query, const location_t loc) const override;
    virtual void get_distance(const data_t *query, const location_t *locations, const uint32_t location_count,  float *distances) const override ;
    virtual float get_distance(const location_t loc1, const location_t loc2) const override;

    virtual size_t get_aligned_dim() const override
    {
        return _aligned_dim;
    }

  protected:
    location_t load_data(const std::string &filename);
#ifdef EXEC_ENV_OLS
    location_t load_data(AlignedFileReader &reader);
#endif

  private:
    data_t *_data = nullptr;

    const size_t _aligned_dim;

    // lazy_delete removes entry from _location_to_tag and _tag_to_location. If
    // _location_to_tag does not resolve a location, infer that it was deleted.
    //tsl::sparse_map<id_t, location_t> _tag_to_location;
    //natural_number_map<location_t, id_t> _location_to_tag;

    // _empty_slots has unallocated slots and those freed by consolidate_delete.
    // _delete_set has locations marked deleted by lazy_delete. Will not be
    // immediately available for insert. consolidate_delete will release these
    // slots to _empty_slots.
    natural_number_set<location_t> _empty_slots;
    std::unique_ptr<tsl::robin_set<location_t>> _delete_set;

    std::shared_timed_mutex _lock; // Takes please of Index::_tag_lock

    // It may seem weird to put distance metric along with the data store class, but
    // this gives us perf benefits as the datastore can do distance computations during
    // search and compute norms of vectors internally without have to copy
    // data back and forth. 
    std::shared_ptr<Distance<data_t>> _distance_fn;

    std::shared_ptr<float[]> _pre_computed_norms;
};

} // namespace diskann