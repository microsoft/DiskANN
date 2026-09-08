// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <functional>
#include <variant>

#include "abstract_graph_store.h"

namespace diskann
{

class InMemStaticGraphStore : public AbstractGraphStore
{
public:
    InMemStaticGraphStore(const size_t total_pts, const size_t reserve_graph_degree,
        bool enable_stream_vbyte = false);

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
    struct RawGraphStorage
    {
        std::vector<size_t> _node_index;
        std::vector<std::uint32_t> _graph;
    };

    struct StreamVByteGraphStorage
    {
        using OffsetStorage = std::variant<std::vector<uint32_t>, std::vector<uint64_t>>;

        // Node count + 1 byte offsets into _data; use 32 bits when the payload fits, otherwise 64.
        OffsetStorage _offsets;
        // Concatenated records: degree, first absolute ID (if nonempty), controls, and deltas.
        std::vector<uint8_t> _data;
        // Number of nodes, also used to validate decoded neighbor IDs.
        size_t _node_count = 0;
        // Graph-wide width of the first absolute ID; delta widths come from control bytes.
        uint8_t _first_id_bytes = 0;
        // Graph-wide width of each record's neighbor count, based on the maximum degree.
        uint8_t _degree_bytes = 0;
    };

    // Calls neighbor_provider in node order [0, node_count) for each of two passes.
    void build_stream_vbyte_graph(
        size_t node_count,
        const std::function<NeighborList(size_t)>& neighbor_provider);

    virtual std::tuple<uint32_t, uint32_t, size_t> load_impl(const std::string& filename, size_t expected_num_points);
#ifdef EXEC_ENV_OLS
    virtual std::tuple<uint32_t, uint32_t, size_t> load_impl(AlignedFileReader& reader, size_t expected_num_points);
#endif


protected:
    size_t _max_range_of_graph = 0;
    uint32_t _max_observed_degree = 0;
    size_t _graph_size = 0;
    bool _enable_stream_vbyte = false;
    std::variant<RawGraphStorage, StreamVByteGraphStorage> _storage;
};

} // namespace diskann
