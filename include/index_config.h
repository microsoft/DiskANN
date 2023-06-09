#include "common_includes.h"
#include "parameters.h"

namespace diskann
{
enum DataStoreStrategy
{
    MEMORY
};

enum GraphStoreStrategy
{
};
struct IndexConfig
{
    DataStoreStrategy data_strategy;
    GraphStoreStrategy graph_strategy;

    Metric metric;
    size_t dimension;
    size_t max_points;

    bool dynamic_index;
    bool enable_tags;
    bool pq_dist_build;
    bool concurrent_consolidate;
    bool use_opq;

    size_t num_pq_chunks;
    size_t num_frozen_pts;

    std::string label_type;
    std::string tag_type;
    std::string data_type;

    IndexWriteParameters *index_write_params = nullptr;

  private:
    IndexConfig(DataStoreStrategy data_strategy, GraphStoreStrategy graph_strategy, Metric metric, size_t dimension,
                size_t max_points, size_t num_pq_chunks, size_t num_frozen_points, bool dynamic_index, bool enable_tags,
                bool pq_dist_build, bool concurrent_consolidate, bool use_opq, std::string data_type,
                std::string tag_type, std::string label_type, IndexWriteParameters &index_write_params)
        : data_strategy(data_strategy), graph_strategy(graph_strategy), metric(metric), dimension(dimension),
          max_points(max_points), dynamic_index(dynamic_index), enable_tags(enable_tags), pq_dist_build(pq_dist_build),
          concurrent_consolidate(concurrent_consolidate), use_opq(use_opq), num_pq_chunks(num_pq_chunks),
          num_frozen_pts(num_frozen_points), label_type(label_type), tag_type(tag_type), data_type(data_type),
          index_write_params(&index_write_params)
    {
    }

    friend class IndexConfigBuilder;
};

class IndexConfigBuilder
{
  public:
    IndexConfigBuilder()
    {
    }

    IndexConfigBuilder &with_metric(Metric m)
    {
        this->_metric = m;
        return *this;
    }

    IndexConfigBuilder &with_graph_load_store_strategy(GraphStoreStrategy graph_strategy)
    {
        this->_graph_strategy = graph_strategy;
        return *this;
    }

    IndexConfigBuilder &with_data_load_store_strategy(DataStoreStrategy data_strategy)
    {
        this->_data_strategy = data_strategy;
        return *this;
    }

    IndexConfigBuilder &with_dimension(size_t dim)
    {
        this->_dimension = dim;
        return *this;
    }

    IndexConfigBuilder &with_max_points(size_t maxPts)
    {
        this->_max_points = maxPts;
        return *this;
    }

    IndexConfigBuilder &is_dynamic_index(bool dynamicIdx)
    {
        this->_dynamic_index = dynamicIdx;
        return *this;
    }

    IndexConfigBuilder &is_enable_tags(bool enableTags)
    {
        this->_enable_tags = enableTags;
        return *this;
    }

    IndexConfigBuilder &is_pq_dist_build(bool pqDistBuild)
    {
        this->_pq_dist_build = pqDistBuild;
        return *this;
    }

    IndexConfigBuilder &is_concurrent_consolidate(bool concurrentConsolidate)
    {
        this->_concurrent_consolidate = concurrentConsolidate;
        return *this;
    }

    IndexConfigBuilder &is_use_opq(bool useOPQ)
    {
        this->_use_opq = useOPQ;
        return *this;
    }

    IndexConfigBuilder &with_num_pq_chunks(size_t numPqChunks)
    {
        this->_num_pq_chunks = numPqChunks;
        return *this;
    }

    IndexConfigBuilder &with_num_frozen_pts(size_t numFrozenPts)
    {
        this->_num_frozen_pts = numFrozenPts;
        return *this;
    }

    IndexConfigBuilder &with_label_type(const std::string &labelType)
    {
        this->_label_type = labelType;
        return *this;
    }

    IndexConfigBuilder &with_tag_type(const std::string &tagType)
    {
        this->_tag_type = tagType;
        return *this;
    }

    IndexConfigBuilder &with_data_type(const std::string &dataType)
    {
        this->_data_type = dataType;
        return *this;
    }

    IndexConfigBuilder &with_index_write_params(IndexWriteParameters &index_write_params)
    {
        this->_index_write_params = &index_write_params;
        return *this;
    }

    IndexConfig build()
    {
        return IndexConfig(_data_strategy, _graph_strategy, _metric, _dimension, _max_points, _num_pq_chunks,
                           _num_frozen_pts, _dynamic_index, _enable_tags, _pq_dist_build, _concurrent_consolidate,
                           _use_opq, _data_type, _tag_type, _label_type, *_index_write_params);
    }

    IndexConfigBuilder(const IndexConfigBuilder &) = delete;
    IndexConfigBuilder &operator=(const IndexConfigBuilder &) = delete;

  private:
    DataStoreStrategy _data_strategy;
    GraphStoreStrategy _graph_strategy;

    Metric _metric;
    size_t _dimension;
    size_t _max_points;

    bool _dynamic_index = false;
    bool _enable_tags = false;
    bool _pq_dist_build = false;
    bool _concurrent_consolidate = false;
    bool _use_opq = false;

    size_t _num_pq_chunks = 0;
    size_t _num_frozen_pts = 0;

    std::string _label_type;
    std::string _tag_type;
    std::string _data_type;

    IndexWriteParameters *_index_write_params = nullptr;
};
} // namespace diskann
