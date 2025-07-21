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

    size_t check_file_size = get_file_size(filename);

    std::ifstream in;
    in.exceptions(std::ios::badbit | std::ios::failbit);
    in.open(filename, std::ios::binary);
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

    _node_index.resize(num_points + 1);
    in.read((char*)_node_index.data(), _node_index.size() * sizeof(size_t));

    _graph_size = _node_index[num_points] * sizeof(std::uint32_t);
    
    size_t total_neighbors = _node_index[num_points];
    // add one more slot than actually need to avoid read invaild address
    // while the last point is no neighbor
    _graph.resize(total_neighbors + 1);
    in.read((char*)_graph.data(), _graph_size);
    in.close();
    
    diskann::cout << "done. Index has " << num_points << " nodes and " << total_neighbors << " out-edges, _start is set to " << start
        << std::endl;
    return std::make_tuple(static_cast<std::uint32_t>(num_points), start, file_frozen_pts);
}

}