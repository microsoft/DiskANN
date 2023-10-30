// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstdint>
#include "abstract_index.h"
#include "builder.h"
#include "common.h"
#include "disk_utils.h"
#include "index.h"
#include "index_factory.h"
#include "parameters.h"

namespace diskannpy
{
template <typename DT>
void build_disk_index(const diskann::Metric metric, const std::string &data_file_path,
                      const std::string &index_prefix_path, const uint32_t complexity, const uint32_t graph_degree,
                      const double final_index_ram_limit, const double indexing_ram_budget, const uint32_t num_threads,
                      const uint32_t pq_disk_bytes)
{
    std::string params = std::to_string(graph_degree) + " " + std::to_string(complexity) + " " +
                         std::to_string(final_index_ram_limit) + " " + std::to_string(indexing_ram_budget) + " " +
                         std::to_string(num_threads);
    if (pq_disk_bytes > 0)
        params = params + " " + std::to_string(pq_disk_bytes);
    diskann::build_disk_index<DT>(data_file_path.c_str(), index_prefix_path.c_str(), params.c_str(), metric);
}

template void build_disk_index<float>(diskann::Metric, const std::string &, const std::string &, uint32_t, uint32_t,
                                      double, double, uint32_t, uint32_t);

template void build_disk_index<uint8_t>(diskann::Metric, const std::string &, const std::string &, uint32_t, uint32_t,
                                        double, double, uint32_t, uint32_t);
template void build_disk_index<int8_t>(diskann::Metric, const std::string &, const std::string &, uint32_t, uint32_t,
                                       double, double, uint32_t, uint32_t);

void build_memory_index(const std::string &vector_dtype, const diskann::Metric metric, const std::string &vector_bin_path,
                        const std::string &index_output_path, const uint32_t graph_degree, const uint32_t complexity,
                        const float alpha, const uint32_t num_threads, const bool use_pq_build,
                        const size_t num_pq_bytes, const bool use_opq, const bool use_tags,
                        const std::string &label_path, const std::string &universal_label,
                        const uint32_t filter_complexity)
{
    auto index_build_params = diskann::IndexWriteParametersBuilder(complexity, graph_degree)
                                                           .with_filter_list_size(filter_complexity)
                                                           .with_alpha(alpha)
                                                           .with_saturate_graph(false)
                                                           .with_num_threads(num_threads)
                                                           .build();
    auto filter_params = diskann::IndexFilterParamsBuilder()
                             .with_universal_label(universal_label)
                             .with_label_file(label_path)
                             .with_save_path_prefix(index_output_path)
                             .build();

    size_t data_num, data_dim;
    diskann::get_bin_metadata(vector_bin_path, data_num, data_dim);

    auto config = diskann::IndexConfigBuilder()
                      .with_metric(metric)
                      .with_dimension(data_dim)
                      .with_max_points(data_num)
                      .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                      .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                      .with_data_type(vector_dtype)
                      .with_tag_type("uint32") // fixed type
                      .with_label_type("uint32") // fixed type
                      .is_dynamic_index(false)
                      .with_index_write_params(index_build_params)
                      .is_concurrent_consolidate(false)
                      .is_enable_tags(use_tags)
                      .is_filtered(!label_path.empty())
                      .is_use_opq(use_opq)
                      .is_pq_dist_build(use_pq_build)
                      .with_num_pq_chunks(num_pq_bytes)
                      .build();

    auto index = diskann::IndexFactory(config).create_instance();

    if (use_tags)
    {
        const std::string tags_file = index_output_path + ".tags";
        if (!file_exists(tags_file))
        {
            throw std::runtime_error("tags file not found at expected path: " + tags_file);
        }
        uint32_t *tags_data;
        size_t tag_dims = 1;
        diskann::load_bin(tags_file, tags_data, data_num, tag_dims);
        std::vector<uint32_t> tags(tags_data, tags_data + data_num);
        index->build(vector_bin_path, data_num, tags);
    }
    else
    {
        index->build(vector_bin_path, data_num, filter_params);
    }

    index->save(index_output_path.c_str());
    index.reset();
}

} // namespace diskannpy
