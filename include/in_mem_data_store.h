// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <shared_mutex>

#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "tsl/sparse_map.h"
#include "boost/dynamic_bitset.hpp"

#include "abstract_data_store.h"

#include "natural_number_map.h"
#include "natural_number_set.h"

namespace diskann
{
template <typename data_t, typename id_t> class InMemDataStore : public AbstractDataStore<data_t, id_t>
{
  public:
    InMemDataStore(const location_t max_pts, const location_t num_frozen_pts, const size_t dim);
    ~InMemDataStore();

    void load(const std::string &filename);
    void store(const std::string &filename);

    data_t *get_vector(location_t i);
    data_t *get_vector_by_UID(id_t uid);

    void set_vector(const location_t i, const data_t *const vector);

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
    tsl::sparse_map<id_t, location_t> _tag_to_location;
    natural_number_map<location_t, id_t> _location_to_tag;

    // _empty_slots has unallocated slots and those freed by consolidate_delete.
    // _delete_set has locations marked deleted by lazy_delete. Will not be
    // immediately available for insert. consolidate_delete will release these
    // slots to _empty_slots.
    natural_number_set<location_t> _empty_slots;
    std::unique_ptr<tsl::robin_set<location_t>> _delete_set;

    std::shared_timed_mutex _lock; // Takes please of Index::_tag_lock
};

} // namespace diskann