// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <shared_mutex>
#include <memory>

#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "tsl/sparse_map.h"
// #include "boost/dynamic_bitset.hpp"

#include "abstract_filter_store.h"


namespace diskann
{
template <typename label_t> class InMemFilterStore : public AbstractFilterStore<label_t>
{
  public:
    InMemFilterStore(const location_t num_points);
    virtual ~InMemFilterStore();

    virtual location_t load(const std::string &filename) override;

    virtual location_t load_labels(const std::string &filename) override;

    virtual size_t save(const std::string &filename, const location_t num_pts) override;

    virtual void set_universal_label(const label_t label) override; 

    virtual void set_labels(const location_t i, std::vector<label_t> labels) override;

    virtual location_t get_medoid(const label_t label) const override;

    virtual label_t get_universal_label() const override;
    
    virtual std::vector<label_t> get_labels_by_point(const location_t point_id) const override;

    virtual label_t get_label(const std::string& raw_label) const override;

    virtual location_t calculate_medoids() override;

    // prepare label file for creating internal label_t labels
    virtual void prepare_label_file(const std::string &filename, const std::string& raw_label) override;

    virtual bool detect_common_filters(location_t point_id, bool search_invocation, 
                          const std::vector<label_t> &incoming_labels) const override;

    virtual std::vector<location_t> get_start_nodes(location_t point_id) const override;

  protected:
    virtual void save_labels(const std::string &filename);
    
    virtual void save_medoids(const std::string &filename);
    
    virtual void save_universal_label(const std::string &filename);

    virtual void save_raw_label_map(const std::string &filename);
    
    virtual void load_medoids(const std::string &filename);
    
    virtual void load_universal_label(const std::string &filename);

    virtual void load_raw_label_map(const std::string &filename);

  private:
    std::vector<std::vector<label_t>> _pts_to_labels;
    tsl::robin_set<label_t> _labels_set;
    std::unordered_map<label_t, location_t> _label_to_medoid_id;
    std::unordered_map<std::string, label_t> _labels_map;
    label_t _universal_label;
    bool _universal_label_exists;
    std::string _label_file;
};

} // namespace diskann