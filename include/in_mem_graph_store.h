// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "abstract_graph_store.h"

namespace diskann
{

class InMemGraphStore : public AbstractGraphStore
{
  public:
    InMemGraphStore(const size_t max_pts, const size_t frozen_points);

    int load(const std::string &index_path_prefix);
    int store(const std::string &index_path_prefix, const size_t active_points);

    virtual std::vector<location_t> &get_neighbours(const location_t i) override;
    virtual void set_neighbours(const location_t i, std::vector<location_t> &neighbors) override;

    virtual size_t resize_graph(const size_t new_size) override;
    virtual void clear_graph() override;

    virtual size_t get_num_frozen_points() override;
    virtual size_t get_max_range_of_loaded_graph() override;
    virtual uint32_t get_max_observed_degree() override;
    virtual void set_max_observed_degree(uint32_t max_observed_degree) override;
    virtual uint32_t get_start() override;
    virtual void set_start(uint32_t start) override;

    /*virtual size_t get_active_points() override;
    virtual void set_active_points(size_t active_points) override;*/

    virtual size_t shrink_to_fit() override;

  protected:
    virtual location_t load_impl(const std::string &filename, size_t expected_num_points);
#ifdef EXEC_ENV_OLS
    virtual location_t load_impl(AlignedFileReader &reader, size_t expected_num_points);
#endif

    int save_graph(const std::string &index_path_prefix, const size_t active_points);

  private:
    size_t _max_range_of_loaded_graph = 0;
    uint32_t _max_observed_degree = 0;
    uint32_t _start = 0;
    size_t _num_frozen_pts;

    size_t _active_points = 0; // _nd

    // graph data structure
    std::vector<std::vector<uint32_t>> _final_graph;
};

} // namespace diskann
