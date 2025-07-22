#pragma once
#include "in_mem_static_graph_store.h"

namespace diskann
{

class InMemStaticGraphReformatStore : public InMemStaticGraphStore
{
public:
    InMemStaticGraphReformatStore(const size_t total_pts, const size_t reserve_graph_degree)
        : InMemStaticGraphStore(total_pts, reserve_graph_degree)
    {
    }

protected:
    std::tuple<uint32_t, uint32_t, size_t> load_impl(const std::string& filename, size_t expected_num_points) override;
};

}