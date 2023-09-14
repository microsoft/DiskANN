
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

template <typename label_t> class AbstractFilterStore
{
  public:
    AbstractFilterStore(const location_t num_points);

    virtual ~AbstractFilterStore() = default;

    // Return number of points returned
    // Load label file, parse and store in the pts to labels vector
    virtual location_t load(const std::string &filename) = 0;

    virtual location_t load_labels(const std::string &filename) = 0;

    // save labels information into 
    // 1. Label.txt
    // 2. Universal label.txt
    // 3. save medoids
    virtual size_t save(const std::string &filename, const location_t num_pts) = 0;

    virtual void set_universal_label(const label_t label) = 0; 

    // May be required for streaming index
    virtual void set_labels(const location_t i, std::vector<label_t> labels) = 0;
   

    virtual location_t get_medoid(const label_t label) const = 0;

    virtual label_t get_universal_label() const = 0;
    
    virtual std::vector<label_t> get_labels_by_point(const location_t point_id) const = 0;

    virtual label_t get_label(const std::string& raw_label) const = 0;

    virtual location_t calculate_medoids() = 0;

    // prepare label file for creating internal label_t labels
    virtual void prepare_label_file(const std::string &filename, const std::string& raw_label) = 0;

    virtual bool detect_common_filters(location_t point_id, bool search_invocation, 
                          const std::vector<label_t> &incoming_labels) = 0;

    virtual std::vector<location_t> get_start_nodes(location_t point_id) = 0;
    
  
    //DISKANN_DLLEXPORT virtual size_t get_max_search_filters() const;

    //virtual void prefetch_labels(const location_t loc) = 0;

    DISKANN_DLLEXPORT virtual location_t get_number_points() const;

  protected:
    location_t _num_points;
};

} // namespace diskann