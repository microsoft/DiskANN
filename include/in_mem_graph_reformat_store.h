#pragma once
#include "in_mem_graph_store.h"

namespace diskann
{

class InMemGraphReformatStore : public InMemGraphStore
{
public:
    InMemGraphReformatStore(const size_t total_pts, const size_t reserve_graph_degree)
        : InMemGraphStore(total_pts, reserve_graph_degree)
    {
    }

protected:
    int save_graph(const std::string& index_path_prefix, const size_t num_points,
        const size_t num_frozen_points, const uint32_t start) override;
};

}