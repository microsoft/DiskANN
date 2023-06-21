// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "in_mem_graph_store.h"
#include "utils.h"

namespace diskann
{

InMemGraphStore::InMemGraphStore(const size_t max_pts) : AbstractGraphStore(max_pts)
{
}

int InMemGraphStore::load(const std::string &index_path_prefix)
{
    return load_impl(index_path_prefix, get_max_points());
}
int InMemGraphStore::store(const std::string &index_path_prefix)
{
    return save_graph(index_path_prefix);
}
void InMemGraphStore::get_adj_list(const location_t i, std::vector<location_t> &neighbors)
{
    neighbors.assign(_final_graph[i].begin(), _final_graph[i].end());
}
void InMemGraphStore::set_adj_list(const location_t i, std::vector<location_t> &neighbors)
{
    _final_graph[i].clear();
    _final_graph[i].assign(neighbors.begin(), neighbors.end());
}

#ifdef EXEC_ENV_OLS
location_t InMemGraphStore::load_impl(const std::string &filename, size_t expected_num_points)
{
    size_t expected_file_size;
    size_t file_frozen_pts;
    auto max_points = get_max_points();
    int header_size = 2 * sizeof(size_t) + 2 * sizeof(uint32_t);
    std::unique_ptr<char[]> header = std::make_unique<char[]>(header_size);
    read_array(reader, header.get(), header_size);

    expected_file_size = *((size_t *)header.get());
    _max_observed_degree = *((uint32_t *)(header.get() + sizeof(size_t)));
    _start = *((uint32_t *)(header.get() + sizeof(size_t) + sizeof(uint32_t)));
    file_frozen_pts = *((size_t *)(header.get() + sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t)));

    diskann::cout << "From graph header, expected_file_size: " << expected_file_size
                  << ", _max_observed_degree: " << _max_observed_degree << ", _start: " << _start
                  << ", file_frozen_pts: " << file_frozen_pts << std::endl;

    if (file_frozen_pts != _num_frozen_pts)
    {
        std::stringstream stream;
        if (file_frozen_pts == 1)
        {
            stream << "ERROR: When loading index, detected dynamic index, but "
                      "constructor asks for static index. Exitting."
                   << std::endl;
        }
        else
        {
            stream << "ERROR: When loading index, detected static index, but "
                      "constructor asks for dynamic index. Exitting."
                   << std::endl;
        }
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    diskann::cout << "Loading vamana graph from reader..." << std::flush;

    const size_t expected_max_points = expected_num_points - file_frozen_pts;

    // If user provides more points than max_points
    // resize the _final_graph to the larger size.
    if (max_points < expected_max_points)
    {
        diskann::cout << "Number of points in data: " << expected_max_points
                      << " is greater than max_points: " << max_points
                      << " Setting max points to: " << expected_max_points << std::endl;
        _final_graph.resize(expected_max_points + _num_frozen_pts);
        //_max_points = expected_max_points;
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
        _final_graph[nodes_read].swap(tmp);
        nodes_read++;
        if (nodes_read % 1000000 == 0)
        {
            diskann::cout << "." << std::flush;
        }
        if (k > _max_range_of_loaded_graph)
        {
            _max_range_of_loaded_graph = k;
        }
    }

    diskann::cout << "done. Index has " << nodes_read << " nodes and " << cc << " out-edges, _start is set to "
                  << _start << std::endl;
    return nodes_read;
}
#endif

location_t InMemGraphStore::load_impl(const std::string &filename, size_t expected_num_points)
{
    size_t expected_file_size;
    size_t file_frozen_pts;
    size_t file_offset = 0;             // will need this for single file format support
    auto max_points = get_max_points(); // from parent class holding max_pts

    std::ifstream in;
    in.exceptions(std::ios::badbit | std::ios::failbit);
    in.open(filename, std::ios::binary);
    in.seekg(file_offset, in.beg);
    in.read((char *)&expected_file_size, sizeof(size_t));
    in.read((char *)&_max_observed_degree, sizeof(uint32_t));
    in.read((char *)&_start, sizeof(uint32_t));
    in.read((char *)&file_frozen_pts, sizeof(size_t));
    size_t vamana_metadata_size = sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);

    diskann::cout << "From graph header, expected_file_size: " << expected_file_size
                  << ", _max_observed_degree: " << _max_observed_degree << ", _start: " << _start
                  << ", file_frozen_pts: " << file_frozen_pts << std::endl;

    if (file_frozen_pts != _num_frozen_pts)
    {
        std::stringstream stream;
        if (file_frozen_pts == 1)
        {
            stream << "ERROR: When loading index, detected dynamic index, but "
                      "constructor asks for static index. Exitting."
                   << std::endl;
        }
        else
        {
            stream << "ERROR: When loading index, detected static index, but "
                      "constructor asks for dynamic index. Exitting."
                   << std::endl;
        }
        diskann::cerr << stream.str() << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    diskann::cout << "Loading vamana graph " << filename << "..." << std::flush;

    const size_t expected_max_points = expected_num_points - file_frozen_pts;

    // If user provides more points than max_points
    // resize the _final_graph to the larger size.
    if (max_points < expected_max_points)
    {
        diskann::cout << "Number of points in data: " << expected_max_points
                      << " is greater than max_points: " << max_points
                      << " Setting max points to: " << expected_max_points << std::endl;
        _final_graph.resize(expected_max_points + _num_frozen_pts);
        // TODO: use a setter ? _max_points = expected_max_points;
    }

    size_t bytes_read = vamana_metadata_size;
    size_t cc = 0;
    uint32_t nodes_read = 0;
    while (bytes_read != expected_file_size)
    {
        uint32_t k;
        in.read((char *)&k, sizeof(uint32_t));

        if (k == 0)
        {
            diskann::cerr << "ERROR: Point found with no out-neighbors, point#" << nodes_read << std::endl;
        }

        cc += k;
        ++nodes_read;
        std::vector<uint32_t> tmp(k);
        tmp.reserve(k);
        in.read((char *)tmp.data(), k * sizeof(uint32_t));
        _final_graph[nodes_read - 1].swap(tmp);
        bytes_read += sizeof(uint32_t) * ((size_t)k + 1);
        if (nodes_read % 10000000 == 0)
            diskann::cout << "." << std::flush;
        if (k > _max_range_of_loaded_graph)
        {
            _max_range_of_loaded_graph = k;
        }
    }

    diskann::cout << "done. Index has " << nodes_read << " nodes and " << cc << " out-edges, _start is set to "
                  << _start << std::endl;
    return nodes_read;
}

int InMemGraphStore::save_graph(const std::string &index_path_prefix)
{
    std::ofstream out;
    open_file_to_write(out, index_path_prefix);

    size_t file_offset = 0; // we will use this if we want
    out.seekp(file_offset, out.beg);
    size_t index_size = 24;
    uint32_t max_degree = 0;
    out.write((char *)&index_size, sizeof(uint64_t));
    out.write((char *)&_max_observed_degree, sizeof(uint32_t)); // need it
    uint32_t ep_u32 = _start;                                   // need it
    out.write((char *)&ep_u32, sizeof(uint32_t));
    out.write((char *)&_num_frozen_pts, sizeof(size_t)); // need it
    // Note: at this point, either _nd == _max_points or any frozen points have
    // been temporarily moved to _nd, so _nd + _num_frozen_points is the valid
    // location limit.
    for (uint32_t i = 0; i < _nd + _num_frozen_pts; i++) // find out how to pass nd_here
    {
        uint32_t GK = (uint32_t)_final_graph[i].size();
        out.write((char *)&GK, sizeof(uint32_t));
        out.write((char *)_final_graph[i].data(), GK * sizeof(uint32_t));
        max_degree = _final_graph[i].size() > max_degree ? (uint32_t)_final_graph[i].size() : max_degree;
        index_size += (size_t)(sizeof(uint32_t) * (GK + 1));
    }
    out.seekp(file_offset, out.beg);
    out.write((char *)&index_size, sizeof(uint64_t));
    out.write((char *)&max_degree, sizeof(uint32_t));
    out.close();
    return (int)index_size;
}

size_t InMemGraphStore::get_num_frozen_points()
{
    return _num_frozen_pts;
}
size_t InMemGraphStore::get_max_range_of_loaded_graph()
{
    return _max_range_of_loaded_graph;
}
uint32_t InMemGraphStore::get_max_observed_degree()
{
    return _max_observed_degree;
}
uint32_t InMemGraphStore::get_start()
{
    return _start;
}

void InMemGraphStore::set_max_observed_degree(uint32_t max_observed_degree)
{
    this->_max_observed_degree = max_observed_degree;
};

void InMemGraphStore::set_start(uint32_t start)
{
    this->_start = start;
};

} // namespace diskann
