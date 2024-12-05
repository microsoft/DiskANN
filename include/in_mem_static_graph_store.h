// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "abstract_graph_store.h"

namespace diskann
{

class InMemStaticGraphStore : public AbstractGraphStore
{
public:
    InMemStaticGraphStore(const size_t total_pts, const size_t reserve_graph_degree);

    // returns tuple of <nodes_read, start, num_frozen_points>
    virtual std::tuple<uint32_t, uint32_t, size_t> load(const std::string& index_path_prefix,
        const size_t num_points) override;

    virtual int store(const std::string& /*index_path_prefix*/, const size_t /*num_points*/, const size_t /*num_frozen_points*/,
        const uint32_t /*start*/) override
    {
        throw std::runtime_error("static memory graph only use for searching");
    }

    virtual const NeighborList get_neighbours(const location_t i) const override;
    
    virtual void add_neighbour(const location_t /*i*/, location_t /*neighbour_id*/) override
    {
        throw std::runtime_error("static memory graph only use for searching");
    }

    virtual void clear_neighbours(const location_t /*i*/) override
    {
        throw std::runtime_error("static memory graph only use for searching");
    }

    virtual void swap_neighbours(const location_t /*a*/, location_t /*b*/) override
    {
        throw std::runtime_error("static memory graph only use for searching");
    }

    virtual void set_neighbours(const location_t /*i*/, std::vector<location_t>& /*neighbors*/) override
    {
        throw std::runtime_error("static memory graph only use for searching");
    }

    virtual size_t resize_graph(const size_t new_size) override
    {
        // not action taken, the graph is initialized in loading
        return new_size;
    }

    virtual void clear_graph() override
    {
        throw std::runtime_error("static memory graph only use for searching");
    }

    virtual size_t get_max_range_of_graph() override;
    virtual uint32_t get_max_observed_degree() override;

    virtual size_t get_graph_size() override;

protected:
    virtual std::tuple<uint32_t, uint32_t, size_t> load_impl(const std::string& filename, size_t expected_num_points);
#ifdef EXEC_ENV_OLS
    virtual std::tuple<uint32_t, uint32_t, size_t> load_impl(AlignedFileReader& reader, size_t expected_num_points);
#endif


private:
    size_t _max_range_of_graph = 0;
    uint32_t _max_observed_degree = 0;
    size_t _graph_size = 0;

    std::vector<size_t> _node_index;
    std::vector<std::uint32_t> _graph;
//    std::vector<std::vector<uint32_t>> _graph;
};

} // namespace diskann
