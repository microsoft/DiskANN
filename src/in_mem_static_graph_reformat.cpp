#include "in_mem_static_graph_reformat_store.h"
#include "utils.h"

namespace diskann
{

std::tuple<uint32_t, uint32_t, size_t> InMemStaticGraphReformatStore::load_impl(const std::string& filename, size_t expected_num_points)
{
    size_t expected_file_size;
    size_t file_frozen_pts;
    uint32_t start;
    size_t num_points;
    size_t file_offset = 0; // will need this for single file format support

    std::ifstream in;
    in.exceptions(std::ios::badbit | std::ios::failbit);
    in.open(filename, std::ios::binary | std::ios::ate);
    // get file size
    size_t check_file_size = in.tellg();
    
    in.seekg(file_offset, in.beg);
    in.read((char*)&expected_file_size, sizeof(size_t));
    in.read((char*)&_max_observed_degree, sizeof(uint32_t));
    in.read((char*)&start, sizeof(uint32_t));
    in.read((char*)&file_frozen_pts, sizeof(size_t));
    in.read((char*)&num_points, sizeof(size_t));

    // max observed degree is the max degree of the graph
    _max_range_of_graph = _max_observed_degree;

    size_t vamana_metadata_size = sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t) + sizeof(size_t);

    diskann::cout << "From graph header, expected_file_size: " << expected_file_size
        << ", _max_observed_degree: " << _max_observed_degree << ", _start: " << start
        << ", file_frozen_pts: " << file_frozen_pts << std::endl;

    if (check_file_size != expected_file_size)
    {
        std::stringstream stream;
        stream << "Vamana Index file size does not match expected size per "
                    "meta-data."
                << " file size from file: " << expected_file_size << " actual file size: " << check_file_size << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    diskann::cout << "Loading vamana graph " << filename << "..." << std::flush;

    std::vector<size_t> raw_node_index(num_points + 1);
    in.read((char*)raw_node_index.data(), raw_node_index.size() * sizeof(size_t));

    const size_t total_neighbors = raw_node_index[num_points];
    const size_t raw_graph_size = total_neighbors * sizeof(std::uint32_t);
    std::vector<uint32_t> raw_graph(total_neighbors + 1);
    in.read((char*)raw_graph.data(), raw_graph_size);
    in.close();

    if (_enable_stream_vbyte)
    {
        build_stream_vbyte_graph(num_points, [&](size_t node) {
            const size_t start_offset = raw_node_index[node];
            return NeighborList(
                raw_graph.data() + start_offset,
                raw_node_index[node + 1] - start_offset);
        });
    }
    else
    {
        _storage = RawGraphStorage{std::move(raw_node_index), std::move(raw_graph)};
        _graph_size = raw_graph_size;
    }
    
    diskann::cout << "done. Index has " << num_points << " nodes and " << total_neighbors << " out-edges, _start is set to " << start
        << std::endl;
    return std::make_tuple(static_cast<std::uint32_t>(num_points), start, file_frozen_pts);
}

}