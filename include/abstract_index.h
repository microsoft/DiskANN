#pragma once
#include "distance.h"
#include "parameters.h"

namespace diskann
{
enum LoadStoreStratagy
{
    DISK,
    MEMORY
};

struct IndexConfig
{
    LoadStoreStratagy graph_load_store_stratagy;
    LoadStoreStratagy data_load_store_stratagy;
    LoadStoreStratagy filtered_data_load_store_stratagy;

    Metric metric;
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

    std::string data_path;
};

struct IndexBuildParams
{
    IndexBuildParams(diskann::IndexWriteParameters &paras) : index_write_params(paras){};
    diskann::IndexWriteParameters index_write_params;
    std::string data_file = "";
    std::string save_path = "";
    std::string label_file = "";
    std::string universal_label = "";
    uint32_t filter_threshold = 0;
    size_t num_points_to_load = 0;
};

class AbstractIndex
{
  public:
    DISKANN_DLLEXPORT AbstractIndex()
    {
    }
    virtual ~AbstractIndex()
    {
    }
    virtual void build(IndexBuildParams &build_params) = 0;
    virtual void save(const char *filename, bool compact_before_save = false) = 0;

    // TODO: add other methods
};

} // namespace diskann