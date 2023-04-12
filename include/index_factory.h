// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "index.h"
#include "common_includes.h"

namespace diskann
{
template <typename T, typename TagT, typename LabelT> class MemoryIndex;
template <typename T, typename LabelT> class DiskIndex;
enum BuildType
{
    MEMORY,
    DISK
};

struct IndexConfig
{
    BuildType build_type;
    Metric metric;
    LoadStoreStratagy load_store_stratagy; // load and store stratagy when we have abstract data or abstract graph store

    bool filtered_build;
    std::string label_type;

    bool enable_tags = false;
    std::string tag_type;

    bool dynamic_index = false;

    bool pq_dist_build = false;
    size_t num_pq_chunks = 0;
    bool use_opq = false;
    size_t num_frozen_pts = 0;
    bool concurrent_consolidate = false;
    bool load_on_build = false;

    std::string data_type;
};

struct BuildParams
{
    BuildParams(diskann::IndexWriteParameters &paras) : index_write_params(paras){};
    diskann::IndexWriteParameters index_write_params;
    uint32_t C = 750;
    std::string label_file = "";
    std::string universal_label = "";
    uint32_t filter_threshold = 0;

    // Disk params
    std::string disk_params = "";
    std::string codebook_prefix = "";
};

struct SearchParams
{
    uint32_t num_threads;
    uint32_t K;
    std::vector<uint32_t> Lvec;
    std::string gt_file = "";
    bool show_qps_per_thread;
    bool print_all_recalls;
    float fail_if_recall_below;

    // disk index params
    uint32_t W;
    uint32_t num_nodes_to_cache;
    uint32_t search_io_limit;
    bool use_reorder_data = false;
};

/* Abstract class to expose all functions of Index via a neat api.*/
class AbstractIndex
{
  public:
    DISKANN_DLLEXPORT AbstractIndex()
    {
    }
    virtual ~AbstractIndex()
    {
    }
    virtual void build(const std::string &data_file, BuildParams &build_params, const std::string &save_path) = 0;
    virtual void search(const std::string &query_file, SearchParams &search_params,
                        const std::vector<std::string> &query_filters) = 0;
    virtual int search_prebuilt_index(const std::string &index_file, const std::string &query_file,
                                      SearchParams &search_params, std::vector<std::string> &query_filters,
                                      const std::string &result_path_prefix) = 0;
};

/*Index Factory to create an instance of index with provided IndexConfig*/
class IndexFactory
{
  public:
    DISKANN_DLLEXPORT IndexFactory(IndexConfig &config);

    DISKANN_DLLEXPORT std::shared_ptr<AbstractIndex> instance();

    static void parse_config(const std::string &config_path);

  private:
    void checkConfig();

    IndexConfig &_config;
};

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t> class MemoryIndex : public AbstractIndex
{
  public:
    DISKANN_DLLEXPORT MemoryIndex(IndexConfig &config);
    DISKANN_DLLEXPORT void build(const std::string &data_file, BuildParams &build_params, const std::string &save_path);
    DISKANN_DLLEXPORT void search(const std::string &query_file, SearchParams &search_params,
                                  const std::vector<std::string> &query_filters = {});
    DISKANN_DLLEXPORT int search_prebuilt_index(const std::string &index_file, const std::string &query_file,
                                                SearchParams &search_params,
                                                std::vector<std::string> &query_filters = {},
                                                const std::string &result_path_prefix = "");

  private:
    std::unique_ptr<Index<T, TagT, LabelT>> _index;
    IndexConfig &_config;

    void initialize_index(size_t dimension, size_t max_points, size_t frozen_points = 0);
    void build_filtered_index(const std::string &data_file, BuildParams &build_params, const std::string &save_path);
    void build_unfiltered_index(const std::string &data_file, BuildParams &build_params);
};

template <typename T, typename LabelT = uint32_t> class DiskIndex : public AbstractIndex
{
  public:
    DISKANN_DLLEXPORT DiskIndex(IndexConfig &config);
    DISKANN_DLLEXPORT void build(const std::string &data_file, BuildParams &build_params, const std::string &save_path);
    DISKANN_DLLEXPORT void search(const std::string &query_file, SearchParams &search_params,
                                  const std::vector<std::string> &query_filters = {});
    DISKANN_DLLEXPORT int search_prebuilt_index(const std::string &index_file, const std::string &query_file,
                                                SearchParams &search_params,
                                                std::vector<std::string> &query_filters = {},
                                                const std::string &result_path_prefix = "");

  private:
    IndexConfig &_config;

    void build_filtered_index(const std::string &data_file, BuildParams &build_params, const std::string &save_path);
    void build_unfiltered_index(const std::string &data_file, BuildParams &build_params);
};

} // namespace diskann