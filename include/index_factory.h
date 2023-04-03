// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "index.h"
#include "common_includes.h"

namespace diskann
{
template <typename T, typename TagT, typename LabelT> class MemoryIndex;
template <typename T, typename TagT, typename LabelT> class DiskIndex;
enum IndexType
{
    MEMORY,
    DISK
};

struct IndexConfig
{
    IndexType index_type;
    std::string data_file;
    Metric metric;
    Parameters buildParams;
    size_t dimension;
    bool filtered;
    std::string label_file;
    std::string universal_label;
    bool enable_tags = false;
    size_t max_points = 1;
    bool dynamic_index = false;
    bool pq_dist_build = false;
    size_t num_pq_chunks = 0;
    bool use_opq = false;
    size_t num_frozen_pts = 0;
    bool concurrent_consolidate = false;
    bool load_on_build = false;
};

template <typename T> class AbstractIndex
{
  public:
    DISKANN_DLLEXPORT AbstractIndex()
    {
    }
    virtual ~AbstractIndex()
    {
    }
    virtual void build(const std::string &save_path, Parameters &build_params) = 0;
    virtual void search(const std::string &query_file, Parameters &search_params,
                        std::vector<std::string> &query_filters) = 0;
    virtual int search_prebuilt_index(const std::string &index_file, const std::string &query_file,
                                       Parameters &search_params, std::vector<std::string> &query_filters) = 0;
};

template <typename T> class IndexFactory
{
  public:
    DISKANN_DLLEXPORT IndexFactory(IndexConfig &config);

    DISKANN_DLLEXPORT std::shared_ptr<AbstractIndex<T>> instance();

    static void parse_config(const std::string &config_path);

  private:
    void checkConfig();

    IndexConfig &_config;
};

// DEF's for Memory and Disk Index.
template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t> class MemoryIndex : public AbstractIndex<T>
{
  public:
    DISKANN_DLLEXPORT MemoryIndex(IndexConfig &config);
    DISKANN_DLLEXPORT void build(const std::string &save_path, Parameters &build_params);
    DISKANN_DLLEXPORT void search(const std::string &query_file, Parameters &search_params,
                                  std::vector<std::string> &query_filters = {});
    DISKANN_DLLEXPORT int search_prebuilt_index(const std::string &index_file, const std::string &query_file,
                                                 Parameters &search_params,
                                                 std::vector<std::string> &query_filters = {});

  private:
    std::unique_ptr<Index<T, TagT, LabelT>> _index;
    IndexConfig &_config;

    void build_filtered_index(Parameters &build_params, const std::string &save_path);
    void build_unfiltered_index(Parameters &build_params);
    void search_filtered_index(const std::string &query_file, Parameters &search_params,
                               std::vector<std::string> query_filters);
    void search_unfiltered_index(const std::string &query_file, Parameters &search_params);
};

} // namespace diskann