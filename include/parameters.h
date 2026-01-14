// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <sstream>
#include <typeinfo>
#include <unordered_map>

#include "omp.h"
#include "defaults.h"

namespace diskann
{

enum class LabelFormatType :uint8_t
{
    String = 0,
    BitMask = 1,
    Integer = 2
};

class IndexWriteParameters
{
  public:
    const uint32_t search_list_size; // L
    const uint32_t max_degree;       // R
    const bool saturate_graph;
    const uint32_t max_occlusion_size; // C
    const float alpha;
    const uint32_t num_threads;
    const uint32_t filter_list_size; // Lf
    const bool diverse_index;
    const std::string seller_file;
    const uint32_t num_diverse_build;

    IndexWriteParameters(const uint32_t search_list_size, const uint32_t max_degree, const bool saturate_graph,
                         const uint32_t max_occlusion_size, const float alpha, const uint32_t num_threads,
                         const uint32_t filter_list_size, bool diverse_index, const std::string& seller_file, uint32_t num_diverse_build)
        : search_list_size(search_list_size), max_degree(max_degree), saturate_graph(saturate_graph),
          max_occlusion_size(max_occlusion_size), alpha(alpha), num_threads(num_threads),
          filter_list_size(filter_list_size), diverse_index(diverse_index), seller_file(seller_file), num_diverse_build(num_diverse_build)
    {
    }

    friend class IndexWriteParametersBuilder;
};

class IndexSearchParams
{
  public:
    IndexSearchParams(const uint32_t initial_search_list_size, const uint32_t num_search_threads)
        : initial_search_list_size(initial_search_list_size), num_search_threads(num_search_threads)
    {
    }
    const uint32_t initial_search_list_size; // search L
    const uint32_t num_search_threads;       // search threads
};

class IndexWriteParametersBuilder
{
    /**
     * Fluent builder pattern to keep track of the 7 non-default properties
     * and their order. The basic ctor was getting unwieldy.
     */
  public:
    IndexWriteParametersBuilder(const uint32_t search_list_size, // L
                                const uint32_t max_degree        // R
                                )
        : _search_list_size(search_list_size), _max_degree(max_degree)
    {
    }

    IndexWriteParametersBuilder &with_max_occlusion_size(const uint32_t max_occlusion_size)
    {
        _max_occlusion_size = max_occlusion_size;
        return *this;
    }

    IndexWriteParametersBuilder &with_saturate_graph(const bool saturate_graph)
    {
        _saturate_graph = saturate_graph;
        return *this;
    }

    IndexWriteParametersBuilder& with_diverse_index(const bool diverse_index)
    {
        _diverse_index = diverse_index;
        return *this;
    }

    IndexWriteParametersBuilder& with_seller_file(const std::string seller_file)
    {
        _seller_file = seller_file;
        return *this;
    }

    IndexWriteParametersBuilder& with_num_diverse_build(const uint32_t num_diverse_build)
    {
        _num_diverse_build = num_diverse_build;
        return *this;
    }

    IndexWriteParametersBuilder &with_alpha(const float alpha)
    {
        _alpha = alpha;
        return *this;
    }

    IndexWriteParametersBuilder &with_num_threads(const uint32_t num_threads)
    {
        _num_threads = num_threads == 0 ? omp_get_num_procs() : num_threads;
        return *this;
    }

    IndexWriteParametersBuilder &with_filter_list_size(const uint32_t filter_list_size)
    {
        _filter_list_size = filter_list_size == 0 ? _search_list_size : filter_list_size;
        return *this;
    }

    IndexWriteParameters build() const
    {
        return IndexWriteParameters(_search_list_size, _max_degree, _saturate_graph, _max_occlusion_size, _alpha,
                                    _num_threads, _filter_list_size, _diverse_index, _seller_file, _num_diverse_build);
    }

    IndexWriteParametersBuilder(const IndexWriteParameters &wp)
        : _search_list_size(wp.search_list_size), _max_degree(wp.max_degree),
          _max_occlusion_size(wp.max_occlusion_size), _saturate_graph(wp.saturate_graph), _alpha(wp.alpha),
          _filter_list_size(wp.filter_list_size)
    {
    }
    IndexWriteParametersBuilder(const IndexWriteParametersBuilder &) = delete;
    IndexWriteParametersBuilder &operator=(const IndexWriteParametersBuilder &) = delete;

  private:
    uint32_t _search_list_size{};
    uint32_t _max_degree{};
    uint32_t _max_occlusion_size{defaults::MAX_OCCLUSION_SIZE};
    bool _saturate_graph{defaults::SATURATE_GRAPH};
    float _alpha{defaults::ALPHA};
    uint32_t _num_threads{defaults::NUM_THREADS};
    uint32_t _filter_list_size{defaults::FILTER_LIST_SIZE};
    bool _diverse_index{ defaults::DIVERSE_INDEX };
    std::string _seller_file{ defaults::EMPTY_STRING };
    uint32_t _num_diverse_build{ defaults::NUM_DIVERSE_BUILD };
};

struct IndexLoadParams
{
    std::string index_file_path;
    uint32_t num_threads{defaults::NUM_THREADS};
    uint32_t search_list_size{defaults::SEARCH_LIST_SIZE};
    LabelFormatType label_format_type{LabelFormatType::String};

    // Optional file paths - if empty, will be derived from index_file_path
    std::string data_file_path;
    std::string tags_file_path;
    std::string delete_set_file_path;
    std::string graph_file_path;
    std::string labels_file_path;
    std::string labels_to_medoids_file_path;
    std::string labels_map_file_path;
    std::string seller_file_path;
    std::string bitmask_label_file_path;
    std::string integer_label_file_path;
    std::string universal_label_file_path;

    IndexLoadParams() = default;

    IndexLoadParams(const std::string &index_file_path, uint32_t num_threads = defaults::NUM_THREADS,
                    uint32_t search_list_size = defaults::SEARCH_LIST_SIZE,
                    LabelFormatType label_format_type = LabelFormatType::String)
        : index_file_path(index_file_path), num_threads(num_threads), search_list_size(search_list_size),
          label_format_type(label_format_type)
    {
    }

    IndexLoadParams(const char *index_file_path, uint32_t num_threads = defaults::NUM_THREADS,
                    uint32_t search_list_size = defaults::SEARCH_LIST_SIZE,
                    LabelFormatType label_format_type = LabelFormatType::String)
        : index_file_path(index_file_path), num_threads(num_threads), search_list_size(search_list_size),
          label_format_type(label_format_type)
    {
    }
};

} // namespace diskann
