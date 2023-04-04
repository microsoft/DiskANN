// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "index.h"
#include "common_includes.h"

namespace diskann
{
template <typename T, typename TagT, typename LabelT> class MemoryIndex;
template <typename T, typename TagT, typename LabelT> class DiskIndex;
enum BuildType
{
    MEMORY,
    DISK,
    SSD
};

enum LoadStoreStratagy
{
};

struct IndexConfig
{
    BuildType build_type;
    Metric metric;
    Parameters buildParams;
    LoadStoreStratagy load_store_stratagy; // load and store stratagy when we have abstract data or abstract graph

    bool filtered_build;

    bool enable_tags = false;
    bool dynamic_index = false;
    bool pq_dist_build = false;
    size_t num_pq_chunks = 0;
    bool use_opq = false;
    size_t num_frozen_pts = 0;
    bool concurrent_consolidate = false;
    bool load_on_build = false;

    std::string data_type;
    std::string label_type;
    std::string tag_type;
};

struct BuildParams
{
    uint32_t R;
    uint32_t L;
    uint32_t Lf;
    uint32_t C = 750;
    uint32_t num_threads;
    float alpha;
    bool saturate_graph;
    std::string label_file = "";
    std::string uinversal_label = "";
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
    virtual void build(const std::string &data_file, Parameters &build_params, const std::string &save_path) = 0;
    virtual void search(const std::string &query_file, Parameters &search_params,
                        const std::vector<std::string> &query_filters) = 0;
    virtual int search_prebuilt_index(const std::string &index_file, const std::string &query_file,
                                      Parameters &search_params, std::vector<std::string> &query_filters) = 0;

    static BuildParams parse_to_build_params(Parameters &build_parameters)
    {
        BuildParams param;
        param.alpha = build_parameters.Get<float>("alpha");
        param.C = build_parameters.Get<uint32_t>("C");
        param.L = build_parameters.Get<uint32_t>("L");
        param.Lf = build_parameters.Get<uint32_t>("Lf");
        param.R = build_parameters.Get<uint32_t>("R");
        param.num_threads = build_parameters.Get<uint32_t>("num_threads", omp_get_num_procs());
        param.saturate_graph = build_parameters.Get<bool>("saturate_graph");
        param.label_file = build_parameters.Get<std::string>("label_file", "");
        param.uinversal_label = build_parameters.Get<std::string>("universal_label", "");
        return param;
    }

    static SearchParams parse_to_search_params(Parameters &search_params)
    {
        SearchParams params;
        params.num_threads = search_params.Get<uint32_t>("num_threads");
        params.K = search_params.Get<uint32_t>("K");
        params.Lvec = search_params.Get<std::vector<uint32_t>>("Lvec");
        params.gt_file = search_params.Get<std::string>("gt_file", "");
        params.show_qps_per_thread = search_params.Get<bool>("show_qps_per_thread", false);
        params.print_all_recalls = search_params.Get<bool>("print_all_recalls", false);
        params.fail_if_recall_below = search_params.Get<float>("fail_if_recall_below", 70);
        return params;
    }
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

// DEF's for Memory and Disk Index.
template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t> class MemoryIndex : public AbstractIndex
{
  public:
    DISKANN_DLLEXPORT MemoryIndex(IndexConfig &config);
    DISKANN_DLLEXPORT void build(const std::string &data_file, Parameters &build_params, const std::string &save_path);
    DISKANN_DLLEXPORT void search(const std::string &query_file, Parameters &search_params,
                                  const std::vector<std::string> &query_filters = {});
    DISKANN_DLLEXPORT int search_prebuilt_index(const std::string &index_file, const std::string &query_file,
                                                Parameters &search_params,
                                                std::vector<std::string> &query_filters = {});

  private:
    std::unique_ptr<Index<T, TagT, LabelT>> _index;
    IndexConfig &_config;

    void initialize_index(size_t dimension, size_t max_points, size_t frozen_points = 0);
    void build_filtered_index(const std::string &data_file, Parameters &build_params, const std::string &save_path);
    void build_unfiltered_index(const std::string &data_file, Parameters &build_params);
};

} // namespace diskann