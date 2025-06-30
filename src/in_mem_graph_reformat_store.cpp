#include "in_mem_graph_reformat_store.h"
#include "utils.h"

namespace diskann
{

int InMemGraphReformatStore::save_graph(const std::string& index_path_prefix, const size_t num_points,
    const size_t num_frozen_points, const uint32_t start)
{
    std::ofstream out;
    open_file_to_write(out, index_path_prefix);

    size_t file_offset = 0;
    out.seekp(file_offset, out.beg);
    size_t index_size = 32;
    uint32_t max_degree = 0;
    out.write((char*)&index_size, sizeof(uint64_t));
    out.write((char*)&_max_observed_degree, sizeof(uint32_t));
    uint32_t ep_u32 = start;
    out.write((char*)&ep_u32, sizeof(uint32_t));
    out.write((char*)&num_frozen_points, sizeof(size_t));
    // write num_points
    out.write((char*)&num_points, sizeof(size_t));

    std::vector<size_t> node_offset(num_points + 1);
    node_offset[0] = 0;
    
    // Note: num_points = _nd + _num_frozen_points
    for (uint32_t i = 0; i < num_points; i++)
    {
        uint32_t GK = (uint32_t)_graph[i].size();
        size_t offset = GK * sizeof(uint32_t);
        node_offset[i + 1] = node_offset[0] + offset;
    }
    out.write((char*)node_offset.data(), node_offset.size() * sizeof(size_t));
    
    index_size += node_offset.size() * sizeof(size_t);
    index_size += node_offset[num_points];

    for (uint32_t i = 0; i < num_points; i++)
    {
        uint32_t GK = (uint32_t)_graph[i].size();
        out.write((char*)_graph[i].data(), GK * sizeof(uint32_t));
        max_degree = _graph[i].size() > max_degree ? (uint32_t)_graph[i].size() : max_degree;
    }

    out.seekp(file_offset, out.beg);
    out.write((char*)&index_size, sizeof(uint64_t));
    out.write((char*)&max_degree, sizeof(uint32_t));
    out.close();
    return (int)index_size;
}

}