#pragma once
#include "distance.h"
#include "parameters.h"
#include "utils.h"
#include <any>

namespace diskann
{

struct AnyVector
{
    template <typename T> AnyVector(const std::vector<T> &vector) : data(std::make_shared<std::vector<T>>(vector))
    {
    }

    template <typename T> const std::vector<T> &get() const
    {
        auto sharedVector = std::any_cast<std::shared_ptr<std::vector<T>>>(&data);
        if (sharedVector)
        {
            return *(*sharedVector);
        }

        throw std::bad_any_cast();
    }

  private:
    std::any data;
};

using DataType = std::any;
using TagType = std::any;
using LabelType = std::any;
using TagVector = AnyVector;

// Enum to store load store stratagy for data_store and graph_store.
enum LoadStoreStrategy
{
    DISK,
    MEMORY
};

// config object to initialize Index via IndexFcatory.
struct IndexConfig
{
    LoadStoreStrategy graph_load_store_strategy;
    LoadStoreStrategy data_load_store_strategy;
    LoadStoreStrategy filtered_data_load_store_strategy;

    Metric metric;
    size_t dimension;
    size_t max_points;

    bool dynamic_index = false;
    bool enable_tags = false;
    bool pq_dist_build = false;
    bool concurrent_consolidate = false;
    bool use_opq = false;

    size_t num_pq_chunks = 0;
    size_t num_frozen_pts = 0;

    // type info to make DiskANN config driven
    std::string label_type;
    std::string tag_type;
    std::string data_type;

  private:
    IndexConfig(LoadStoreStrategy data_store_strategy, LoadStoreStrategy graph_store_strategy,
                LoadStoreStrategy filtered_data_store_strategy, Metric metric, size_t dimension, size_t max_points,
                size_t num_pq_chunks, size_t num_frozen_points, bool dynamic_index, bool enable_tags,
                bool pq_dist_build, bool concurrent_consolidate, bool use_opq, std::string &data_type,
                std::string &tag_type, std::string &label_type)
    {
        this->data_load_store_strategy = data_store_strategy;
        this->graph_load_store_strategy = graph_store_strategy;
        this->filtered_data_load_store_strategy = filtered_data_store_strategy;

        this->metric = metric;
        this->dimension = dimension;
        this->max_points = max_points;

        this->dynamic_index = dynamic_index;
        this->enable_tags = enable_tags;
        this->pq_dist_build = pq_dist_build;
        this->concurrent_consolidate = concurrent_consolidate;
        this->use_opq = use_opq;

        this->num_pq_chunks = num_pq_chunks;
        this->num_frozen_pts = num_frozen_points;

        this->data_type = data_type;
        this->tag_type = tag_type;
        this->label_type = label_type;
    }

    friend class IndexConfigBuilder;
};

// Build params for building index.
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

// Search params for searching indedx.
struct IndexSearchParams
{
    std::string result_path = "";
    size_t query_num, query_dim, query_aligned_dim;
    std::string filter_label = "";
    std::string query_filter_file = "";
    uint32_t num_threads{20}; // or some other default val
};

// Stats produced while searching index
struct QuerySearchStats
{
    std::vector<std::chrono::duration<double>> diff_stats;
    std::vector<uint32_t> cmp_stats;
    std::vector<float> latency_stats;
};

// results from search.
struct SearchResult
{
    SearchResult()
    {
    }
    void init(uint64_t lvec_size)
    {
        stats.diff_stats.resize(lvec_size);

        query_result_ids.resize(lvec_size);
        query_result_dists.resize(lvec_size);
    };
    std::vector<std::vector<uint32_t>> query_result_ids;
    std::vector<std::vector<float>> query_result_dists;
    QuerySearchStats stats;
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

class IndexConfigBuilder
{
  public:
    IndexConfigBuilder()
    {
    }
    IndexConfigBuilder &with_metric(Metric m)
    {
        this->metric = m;
        return *this;
    }

    // Populate fields
    IndexConfigBuilder &with_graph_load_store_strategy(LoadStoreStrategy lss)
    {
        this->graph_load_store_strategy = lss;
        return *this;
    }

    IndexConfigBuilder &with_data_load_store_strategy(LoadStoreStrategy lss)
    {
        this->data_load_store_strategy = lss;
        return *this;
    }

    IndexConfigBuilder &with_filtered_data_load_store_strategy(LoadStoreStrategy lss)
    {
        this->filtered_data_load_store_strategy = lss;
        return *this;
    }

    IndexConfigBuilder &with_dimension(size_t dim)
    {
        this->dimension = dim;
        return *this;
    }

    IndexConfigBuilder &with_max_points(size_t maxPts)
    {
        this->max_points = maxPts;
        return *this;
    }

    IndexConfigBuilder &is_dynamic_index(bool dynamicIdx)
    {
        this->dynamic_index = dynamicIdx;
        return *this;
    }

    IndexConfigBuilder &is_enable_tags(bool enableTags)
    {
        this->enable_tags = enableTags;
        return *this;
    }

    IndexConfigBuilder &is_pq_dist_build(bool pqDistBuild)
    {
        this->pq_dist_build = pqDistBuild;
        return *this;
    }

    IndexConfigBuilder &is_concurrent_consolidate(bool concurrentConsolidate)
    {
        this->concurrent_consolidate = concurrentConsolidate;
        return *this;
    }

    IndexConfigBuilder &is_use_opq(bool useOPQ)
    {
        this->use_opq = useOPQ;
        return *this;
    }

    IndexConfigBuilder &with_num_pq_chunks(size_t numPqChunks)
    {
        this->num_pq_chunks = numPqChunks;
        return *this;
    }

    IndexConfigBuilder &with_num_frozen_pts(size_t numFrozenPts)
    {
        this->num_frozen_pts = numFrozenPts;
        return *this;
    }

    IndexConfigBuilder &with_label_type(const std::string &labelType)
    {
        this->label_type = labelType;
        return *this;
    }

    IndexConfigBuilder &with_tag_type(const std::string &tagType)
    {
        this->tag_type = tagType;
        return *this;
    }

    IndexConfigBuilder &with_data_type(const std::string &dataType)
    {
        this->data_type = dataType;
        return *this;
    }

    IndexConfig build()
    {
        return IndexConfig(data_load_store_strategy, graph_load_store_strategy, filtered_data_load_store_strategy,
                           metric, dimension, max_points, num_pq_chunks, num_frozen_pts, dynamic_index, enable_tags,
                           pq_dist_build, concurrent_consolidate, use_opq, data_type, tag_type, label_type);
    }

    IndexConfigBuilder(const IndexConfigBuilder &) = delete;
    IndexConfigBuilder &operator=(const IndexConfigBuilder &) = delete;

  private:
    LoadStoreStrategy graph_load_store_strategy;
    LoadStoreStrategy data_load_store_strategy;
    LoadStoreStrategy filtered_data_load_store_strategy;

    Metric metric;
    size_t dimension;
    size_t max_points;

    bool dynamic_index = false;
    bool enable_tags = false;
    bool pq_dist_build = false;
    bool concurrent_consolidate = false;
    bool use_opq = false;

    size_t num_pq_chunks = 0;
    size_t num_frozen_pts = 0;

    // type info to make DiskANN config driven
    std::string label_type;
    std::string tag_type;
    std::string data_type;
};

class AbstractIndex
{
  public:
    AbstractIndex()
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

    virtual SearchResult batch_search(const DataType &query, size_t K, std::vector<uint32_t> &Lvec,
                                      IndexSearchParams &search_params) = 0;

    virtual int insert_point(const DataType &data_point, const TagType &tag) = 0;

    virtual int lazy_delete(const TagType &tag) = 0;

    virtual void lazy_delete(const TagVector &tags, TagVector &failed_tags) = 0;

    // TODO: add other methods as api promise to end user.
};
} // namespace diskann