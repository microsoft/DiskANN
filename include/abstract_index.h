#pragma once
#include "distance.h"
#include "parameters.h"
#include "utils.h"
#include <any>

namespace diskann
{

struct consolidation_report
{
    enum status_code
    {
        SUCCESS = 0,
        FAIL = 1,
        LOCK_FAIL = 2,
        INCONSISTENT_COUNT_ERROR = 3
    };
    status_code _status;
    size_t _active_points, _max_points, _empty_slots, _slots_released, _delete_set_size, _num_calls_to_process_delete;
    double _time;

    consolidation_report(status_code status, size_t active_points, size_t max_points, size_t empty_slots,
                         size_t slots_released, size_t delete_set_size, size_t num_calls_to_process_delete,
                         double time_secs)
        : _status(status), _active_points(active_points), _max_points(max_points), _empty_slots(empty_slots),
          _slots_released(slots_released), _delete_set_size(delete_set_size),
          _num_calls_to_process_delete(num_calls_to_process_delete), _time(time_secs)
    {
    }
};
struct AnyRobinSet
{
    template <typename T>
    AnyRobinSet(const tsl::robin_set<T> &robin_set) : data(const_cast<tsl::robin_set<T> *>(&robin_set))
    {
    }

    template <typename T> const tsl::robin_set<T> &get() const
    {
        auto set_ptr = std::any_cast<tsl::robin_set<T> *>(&data);
        if (set_ptr)
        {
            return *(*set_ptr);
        }

        throw std::bad_any_cast();
    }

    template <typename T> tsl::robin_set<T> &get()
    {
        auto set_ptr = std::any_cast<tsl::robin_set<T> *>(&data);
        if (set_ptr)
        {
            return *(*set_ptr);
        }

        throw std::bad_any_cast();
    }

  private:
    std::any data;
};

struct AnyVector
{
    template <typename T> AnyVector(const std::vector<T> &vector) : data(const_cast<std::vector<T> *>(&vector))
    {
    }

    template <typename T> const std::vector<T> &get() const
    {
        auto sharedVector = std::any_cast<std::vector<T> *>(&data);
        if (sharedVector)
        {
            return *(*sharedVector);
        }

        throw std::bad_any_cast();
    }

    template <typename T> std::vector<T> &get()
    {
        auto sharedVector = std::any_cast<std::vector<T> *>(&data);
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
using TagRobinSet = AnyRobinSet;

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

    IndexWriteParameters *index_write_params;

  private:
    IndexConfig(LoadStoreStrategy data_store_strategy, LoadStoreStrategy graph_store_strategy,
                LoadStoreStrategy filtered_data_store_strategy, Metric metric, size_t dimension, size_t max_points,
                size_t num_pq_chunks, size_t num_frozen_points, bool dynamic_index, bool enable_tags,
                bool pq_dist_build, bool concurrent_consolidate, bool use_opq, std::string &data_type,
                std::string &tag_type, std::string &label_type, IndexWriteParameters &index_write_params)
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

        this->index_write_params = &index_write_params;
    }

    friend class IndexConfigBuilder;
};

// Build params for building index.
struct IndexBuildParams
{
  public:
    diskann::IndexWriteParameters index_write_params;
    std::string save_path_prefix;
    std::string label_file;
    std::string universal_label;
    uint32_t filter_threshold = 0;

  private:
    IndexBuildParams(IndexWriteParameters &index_write_params, std::string &save_path_prefix, std::string &label_file,
                     std::string &universal_label, uint32_t filter_threshold)
        : index_write_params(index_write_params), save_path_prefix(save_path_prefix), label_file(label_file),
          universal_label(universal_label), filter_threshold(filter_threshold)
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

    IndexBuildParams build()
    {
        return IndexBuildParams(index_write_params, save_path_prefix, label_file, universal_label, filter_threshold);
    }

    IndexBuildParamsBuilder(const IndexBuildParamsBuilder &) = delete;
    IndexBuildParamsBuilder &operator=(const IndexBuildParamsBuilder &) = delete;

  private:
    diskann::IndexWriteParameters index_write_params;
    std::string save_path_prefix;
    std::string label_file;
    std::string universal_label;
    uint32_t filter_threshold = 0;
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

    IndexConfigBuilder &with_index_write_params(IndexWriteParameters &index_write_params)
    {
        this->index_write_params = &index_write_params;
        return *this;
    }

    IndexConfig build()
    {
        return IndexConfig(data_load_store_strategy, graph_load_store_strategy, filtered_data_load_store_strategy,
                           metric, dimension, max_points, num_pq_chunks, num_frozen_pts, dynamic_index, enable_tags,
                           pq_dist_build, concurrent_consolidate, use_opq, data_type, tag_type, label_type,
                           *index_write_params);
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

    IndexWriteParameters *index_write_params;
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
    virtual void build(const std::string &data_file, const size_t num_points_to_load,
                       IndexBuildParams &build_params) = 0;
    virtual void build(const DataType &data, const size_t num_points_to_load, const IndexWriteParameters &parameters,
                       const TagVector &tags) = 0;
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

    virtual void get_active_tags(TagRobinSet &active_tags) = 0;

    virtual void set_start_points_at_random(DataType radius, uint32_t random_seed = 0) = 0;

    virtual int enable_delete() = 0;

    virtual consolidation_report consolidate_deletes(const IndexWriteParameters &parameters) = 0;
    // TODO: add other methods as api promise to end user.
};
} // namespace diskann