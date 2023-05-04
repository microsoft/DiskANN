#pragma once
#include "distance.h"
#include "parameters.h"
#include "utils.h"

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
    size_t dimension;
    size_t max_points;

    std::string label_type;
    std::string tag_type;
    std::string data_type;

    bool enable_tags = false;
    bool dynamic_index = false;
    bool pq_dist_build = false;
    size_t num_pq_chunks = 0;
    bool use_opq = false;
    size_t num_frozen_pts = 0;
    bool concurrent_consolidate = false;
};

struct IndexBuildParams
{
  public:
    diskann::IndexWriteParameters index_write_params;
    std::string data_file;
    std::string save_path_prefix;
    std::string label_file;
    std::string universal_label;
    uint32_t filter_threshold = 0;
    size_t num_points_to_load = 0;

  private:
    IndexBuildParams(IndexWriteParameters &index_write_params, std::string &data_file, std::string &save_path_prefix,
                     std::string &label_file, std::string &universal_label, uint32_t filter_threshold,
                     size_t num_points_to_load)
        : index_write_params(index_write_params), data_file(data_file), save_path_prefix(save_path_prefix),
          label_file(label_file), universal_label(universal_label), filter_threshold(filter_threshold),
          num_points_to_load(num_points_to_load)
    {
    }

    friend class IndexBuildParamsBuilder;
};

struct IndexSearchParams
{
    std::string result_path = "";
    std::string query_file = "";
    std::string gt_file = "";
    bool print_all_recalls;
    bool show_qps_per_thread;
    float fail_if_recall_below{70.0f};
    size_t K{0};
    std::vector<uint32_t> Lvec;
    std::string filter_label = "";
    std::string query_filter_file = "";
    uint32_t num_threads{20}; // or some other default val
};

class IndexBuildParamsBuilder
{
  public:
    IndexBuildParamsBuilder(diskann::IndexWriteParameters &paras) : index_write_params(paras){};

    IndexBuildParamsBuilder &with_data_file(std::string &data_file)
    {
        if (data_file == "" || data_file.empty())
            throw ANNException("Error: data path provides is empty.", -1);
        if (!file_exists(data_file))
            throw ANNException("Error: data path" + data_file + " does not exist.", -1);

        this->data_file = data_file;
        return *this;
    }

    IndexBuildParamsBuilder &with_save_path_prefix(std::string &save_path_prefix)
    {
        if (save_path_prefix.empty() || save_path_prefix == "")
            throw ANNException("Error: save_path_prefix can't be empty", -1);
        this->save_path_prefix = save_path_prefix;
        return *this;
    }

    IndexBuildParamsBuilder &with_label_file(std::string &label_file)
    {
        this->label_file = label_file;
        return *this;
    }

    IndexBuildParamsBuilder &with_universal_label(std::string &univeral_label)
    {
        this->universal_label = univeral_label;
        return *this;
    }

    IndexBuildParamsBuilder &with_filter_threshold(std::uint32_t &filter_threshold)
    {
        this->filter_threshold = filter_threshold;
        return *this;
    }

    IndexBuildParamsBuilder &with_points_to_load(std::size_t &num_points_to_load)
    {
        this->num_points_to_load = num_points_to_load;
        return *this;
    }

    IndexBuildParams build()
    {
        return IndexBuildParams(index_write_params, data_file, save_path_prefix, label_file, universal_label,
                                filter_threshold, num_points_to_load);
    }

    IndexBuildParamsBuilder(const IndexBuildParamsBuilder &) = delete;
    IndexBuildParamsBuilder &operator=(const IndexBuildParamsBuilder &) = delete;

  private:
    diskann::IndexWriteParameters index_write_params;
    std::string data_file;
    std::string save_path_prefix;
    std::string label_file;
    std::string universal_label;
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

#ifdef EXEC_ENV_OLS
    virtual void load(AlignedFileReader &reader, uint32_t num_threads, uint32_t search_l) = 0;
#else
    // Reads the number of frozen points from graph's metadata file section.
    virtual void load(const char *index_file, uint32_t num_threads, uint32_t search_l) = 0;
#endif

    virtual std::pair<uint32_t, uint32_t> search(IndexSearchParams &search_params) = 0;

    // TODO: add other methods as api promise to end user.
};

} // namespace diskann