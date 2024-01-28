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

  private:
    IndexWriteParameters(const uint32_t search_list_size, const uint32_t max_degree, const bool saturate_graph,
                         const uint32_t max_occlusion_size, const float alpha, const uint32_t num_threads,
                         const uint32_t filter_list_size)
        : search_list_size(search_list_size), max_degree(max_degree), saturate_graph(saturate_graph),
          max_occlusion_size(max_occlusion_size), alpha(alpha), num_threads(num_threads),
          filter_list_size(filter_list_size)
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
                                    _num_threads, _filter_list_size);
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
};

class IndexFilterParams
{
  public:
    std::string save_path_prefix;
    std::string label_file;
    std::string tags_file;
    std::string universal_label;
    uint32_t filter_threshold = 0;

  private:
    IndexFilterParams(const std::string &save_path_prefix, const std::string &label_file,
                      const std::string &universal_label, uint32_t filter_threshold)
        : save_path_prefix(save_path_prefix), label_file(label_file), universal_label(universal_label),
          filter_threshold(filter_threshold)
    {
    }

    friend class IndexFilterParamsBuilder;
};

class IndexFilterParamsBuilder
{
  public:
    IndexFilterParamsBuilder() = default;

    IndexFilterParamsBuilder &with_save_path_prefix(const std::string &save_path_prefix)
    {
        this->_save_path_prefix = save_path_prefix;
        return *this;
    }

    IndexFilterParamsBuilder &with_label_file(const std::string &label_file)
    {
        this->_label_file = label_file;
        return *this;
    }

    IndexFilterParamsBuilder &with_universal_label(const std::string &univeral_label)
    {
        this->_universal_label = univeral_label;
        return *this;
    }

    IndexFilterParamsBuilder &with_filter_threshold(const std::uint32_t &filter_threshold)
    {
        this->_filter_threshold = filter_threshold;
        return *this;
    }

    IndexFilterParams build()
    {
        return IndexFilterParams(_save_path_prefix, _label_file, _universal_label, _filter_threshold);
    }

    IndexFilterParamsBuilder(const IndexFilterParamsBuilder &) = delete;
    IndexFilterParamsBuilder &operator=(const IndexFilterParamsBuilder &) = delete;

  private:
    std::string _save_path_prefix;
    std::string _label_file;
    std::string _tags_file;
    std::string _universal_label;
    uint32_t _filter_threshold = 0;
};

} // namespace diskann
