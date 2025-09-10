#include "in_mem_static_graph_reformat_store.h"
#include "utils.h"

namespace diskann
{

std::tuple<uint32_t, uint32_t, size_t> InMemStaticGraphReformatStore::load_impl(const std::string& filename, size_t expected_num_points)
{
    size_t file_offset = 0; // will need this for single file format support

    FileReader file_reader;
    if (!file_reader.Open(filename))
    {
        throw diskann::ANNException("fail to open file ", -1, __FUNCSIG__,
            __FILE__, __LINE__);
    }

    // get file size
    size_t check_file_size = file_reader.GetFileSize();

    GraphHeader header;
    if (file_reader.Read(file_offset, sizeof(GraphHeader), (char*)&header) != sizeof(GraphHeader))
    {
        throw diskann::ANNException("fail to read graph header ", -1, __FUNCSIG__,
            __FILE__, __LINE__);
    }

    file_offset += sizeof(GraphHeader);

    _max_observed_degree = header.max_observed_degree;
    _max_range_of_graph = _max_observed_degree;

    if (check_file_size != header.expected_file_size)
    {
        std::stringstream stream;
        stream << "Vamana Index file size does not match expected size per "
            "meta-data."
            << " file size from file: " << header.expected_file_size << " actual file size: " << check_file_size << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    diskann::cout << "From graph header, expected_file_size: " << header.expected_file_size
        << ", _max_observed_degree: " << _max_observed_degree << ", _start: " << header.start
        << ", file_frozen_pts: " << header.file_frozen_pts << std::endl;

    diskann::cout << "Loading vamana graph " << filename << "..." << std::flush;

    // Load node index with chunked aligned reads
    _node_index.resize(header.num_points + 1);
    size_t node_index_size = _node_index.size() * sizeof(size_t);
    if (file_reader.Read(file_offset, node_index_size, (char*)_node_index.data()) != node_index_size)
    {
        throw diskann::ANNException("fail to read node index ", -1, __FUNCSIG__,
            __FILE__, __LINE__);
    }

    file_offset += node_index_size;

    _graph_size = _node_index[header.num_points] * sizeof(std::uint32_t);
    size_t total_neighbors = _node_index[header.num_points];
    // add one more slot than actually need to avoid read invaild address
    // while the last point is no neighbor
    _graph.resize(total_neighbors + 1);
    if (file_reader.Read(file_offset, _graph_size, (char*)_graph.data()) != _graph_size)
    {
        throw diskann::ANNException("fail to read graph data ", -1, __FUNCSIG__,
            __FILE__, __LINE__);
    }


    return std::make_tuple(static_cast<std::uint32_t>(header.num_points), header.start, header.file_frozen_pts);
}

}