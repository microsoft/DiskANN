// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "in_mem_static_graph_store.h"
#include "utils.h"

namespace diskann
{

InMemStaticGraphStore::InMemStaticGraphStore(const size_t total_pts, const size_t reserve_graph_degree)
    : AbstractGraphStore(total_pts, reserve_graph_degree)
{    
}

std::tuple<uint32_t, uint32_t, size_t> InMemStaticGraphStore::load(const std::string& index_path_prefix,
    const size_t num_points)
{
    return load_impl(index_path_prefix, num_points);
}

const NeighborList InMemStaticGraphStore::get_neighbours(const location_t i) const
{
    assert(i < _node_index.size() - 1);
    size_t start_index = _node_index[i];
    size_t end_index = _node_index[i + 1];
    size_t size = end_index - start_index;
    const location_t* neighbor_start = _graph.data() + start_index;
    return NeighborList(neighbor_start, size);
}

#ifdef EXEC_ENV_OLS
std::tuple<uint32_t, uint32_t, size_t> InMemGraphStore::load_impl(AlignedFileReader& reader, size_t expected_num_points)
{
    size_t expected_file_size;
    size_t file_frozen_pts;
    uint32_t start;

    auto max_points = get_max_points();
    int header_size = 2 * sizeof(size_t) + 2 * sizeof(uint32_t);
    std::unique_ptr<char[]> header = std::make_unique<char[]>(header_size);
    read_array(reader, header.get(), header_size);

    expected_file_size = *((size_t*)header.get());
    _max_observed_degree = *((uint32_t*)(header.get() + sizeof(size_t)));
    start = *((uint32_t*)(header.get() + sizeof(size_t) + sizeof(uint32_t)));
    file_frozen_pts = *((size_t*)(header.get() + sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t)));

    diskann::cout << "From graph header, expected_file_size: " << expected_file_size
        << ", _max_observed_degree: " << _max_observed_degree << ", _start: " << start
        << ", file_frozen_pts: " << file_frozen_pts << std::endl;

    diskann::cout << "Loading vamana graph from reader..." << std::flush;

    // If user provides more points than max_points
    // resize the _graph to the larger size.
    if (get_total_points() < expected_num_points)
    {
        diskann::cout << "resizing graph to " << expected_num_points << std::endl;
        this->resize_graph(expected_num_points);
    }

    uint32_t nodes_read = 0;
    size_t cc = 0;
    size_t graph_offset = header_size;
    while (nodes_read < expected_num_points)
    {
        uint32_t k;
        read_value(reader, k, graph_offset);
        graph_offset += sizeof(uint32_t);
        std::vector<uint32_t> tmp(k);
        tmp.reserve(k);
        read_array(reader, tmp.data(), k, graph_offset);
        graph_offset += k * sizeof(uint32_t);
        cc += k;
        _graph[nodes_read].swap(tmp);
        nodes_read++;
        if (nodes_read % 1000000 == 0)
        {
            diskann::cout << "." << std::flush;
        }
        if (k > _max_range_of_graph)
        {
            _max_range_of_graph = k;
        }
    }

    diskann::cout << "done. Index has " << nodes_read << " nodes and " << cc << " out-edges, _start is set to " << start
        << std::endl;
    return std::make_tuple(nodes_read, start, file_frozen_pts);
}
#endif

std::tuple<uint32_t, uint32_t, size_t> InMemStaticGraphStore::load_impl(const std::string& filename,
    size_t expected_num_points)
{
    size_t expected_file_size;
    size_t file_frozen_pts;
    uint32_t start;
    size_t file_offset = 0; // will need this for single file format support
    
    std::ifstream in;
    in.exceptions(std::ios::badbit | std::ios::failbit);
    in.open(filename, std::ios::binary);
    in.seekg(file_offset, in.beg);
    in.read((char*)&expected_file_size, sizeof(size_t));
    in.read((char*)&_max_observed_degree, sizeof(uint32_t));
    in.read((char*)&start, sizeof(uint32_t));
    in.read((char*)&file_frozen_pts, sizeof(size_t));
    size_t vamana_metadata_size = sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);

    diskann::cout << "From graph header, expected_file_size: " << expected_file_size
        << ", _max_observed_degree: " << _max_observed_degree << ", _start: " << start
        << ", file_frozen_pts: " << file_frozen_pts << std::endl;

    diskann::cout << "Loading vamana graph " << filename << "..." << std::flush;

    std::vector<char> buffer;
    size_t graph_size = expected_file_size - vamana_metadata_size;
    buffer.resize(graph_size);

    in.read(buffer.data(), graph_size);
    in.close();

    size_t cc = 0;
    uint32_t nodes_read = 0;

    // first round to calculate memory size needed.
    size_t cur_index = 0;
    while (cur_index + sizeof(uint32_t) < graph_size)
    {
        uint32_t k;
        memcpy((char*)&k, buffer.data() + cur_index, sizeof(uint32_t));
        cur_index += sizeof(uint32_t);
        size_t neighbor_size = k * sizeof(uint32_t);
        if (cur_index + neighbor_size > graph_size)
        {
            break;
        }
        cur_index += neighbor_size;

        cc += k;
        ++nodes_read;
    }

    // resize graph
    _node_index.resize(nodes_read + 1);
    _node_index[0] = 0;
    _graph.resize(cc);

    // second round to insert graph data
    nodes_read = 0;
    cur_index = 0;
    while (cur_index + sizeof(uint32_t) < graph_size)
    {
        uint32_t k;
        memcpy((char*)&k, buffer.data() + cur_index, sizeof(uint32_t));
        cur_index += sizeof(uint32_t);
        size_t neighbor_size = k * sizeof(uint32_t);
        if (cur_index + neighbor_size > graph_size)
        {
            break;
        }

        size_t offset = _node_index[nodes_read];
        std::uint32_t* neighborPtr = &_graph[offset];

        memcpy(neighborPtr, buffer.data() + cur_index, neighbor_size);
        _node_index[nodes_read + 1] = offset + k;

        cur_index += neighbor_size;

        if (nodes_read % 10000000 == 0)
            std::cout << "." << std::flush;

        ++nodes_read;
    }

    diskann::cout << "done. Index has " << nodes_read << " nodes and " << cc << " out-edges, _start is set to " << start
        << std::endl;
    return std::make_tuple(nodes_read, start, file_frozen_pts);
}

size_t InMemStaticGraphStore::get_max_range_of_graph()
{
    return _max_range_of_graph;
}

uint32_t InMemStaticGraphStore::get_max_observed_degree()
{
    return _max_observed_degree;
}

} // namespace diskann
