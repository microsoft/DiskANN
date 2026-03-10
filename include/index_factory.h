#pragma once

#include "index.h"
#include "abstract_graph_store.h"
#include "in_mem_graph_store.h"
#include "pq_data_store.h"


namespace diskann
{
class IndexFactory
{
  public:
    DISKANN_DLLEXPORT explicit IndexFactory(const IndexConfig &config);
    DISKANN_DLLEXPORT std::unique_ptr<AbstractIndex> create_instance();

    DISKANN_DLLEXPORT static std::unique_ptr<AbstractGraphStore> construct_graphstore(
        const GraphStoreStrategy stratagy, const size_t size, const size_t reserve_graph_degree);

    template <typename T>
    DISKANN_DLLEXPORT static std::shared_ptr<AbstractDataStore<T>> construct_datastore(DataStoreStrategy stratagy,
                                                                                       size_t num_points,
                                                                                       size_t dimension, Metric m);
    // For now PQDataStore incorporates within itself all variants of quantization that we support. In the
    // future it may be necessary to introduce an AbstractPQDataStore class to spearate various quantization
    // flavours.
    template <typename T>
    DISKANN_DLLEXPORT static std::shared_ptr<PQDataStore<T>> construct_pq_datastore(DataStoreStrategy strategy,
                                                                                    size_t num_points, size_t dimension,
                                                                                    Metric m, size_t num_pq_chunks,
                                                                                    bool use_opq);
    template <typename T> static Distance<T> *construct_inmem_distance_fn(Metric m);

  private:
    void check_config();

    template <typename data_type, typename tag_type, typename label_type>
    std::unique_ptr<AbstractIndex> create_instance();

    std::unique_ptr<AbstractIndex> create_instance(const std::string &data_type, const std::string &tag_type,
                                                   const std::string &label_type);

    template <typename data_type>
    std::unique_ptr<AbstractIndex> create_instance(const std::string &tag_type, const std::string &label_type);

    template <typename data_type, typename tag_type>
    std::unique_ptr<AbstractIndex> create_instance(const std::string &label_type);

    std::unique_ptr<IndexConfig> _config;
};

} // namespace diskann

extern "C" __declspec(dllexport) void *CreateIndex(const char *index_path, uint32_t embedding_dim, uint32_t num_threads,
                                                   uint32_t search_L);
extern "C" __declspec(dllexport) void ReleaseIndex(const void *index_ptr);
extern "C" __declspec(dllexport) int SearchIndex(void *index_ptr, const uint8_t *query, uint32_t recall_at,
                                                 uint32_t search_L, uint32_t *result_ids, float *distances);
