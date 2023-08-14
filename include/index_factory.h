#include "index.h"
#include "abstract_graph_store.h"
#include "in_mem_graph_store.h"

namespace diskann
{
class IndexFactory
{
  public:
    DISKANN_DLLEXPORT explicit IndexFactory(const IndexConfig &config);
    DISKANN_DLLEXPORT std::unique_ptr<AbstractIndex> create_instance();

    // Consruct a data store with distance function emplaced within
    template <typename T>
    DISKANN_DLLEXPORT static std::unique_ptr<AbstractDataStore<T>> construct_datastore(DataStoreStrategy stratagy,
                                                                                       size_t num_points,
                                                                                       size_t dimension, Metric m);

  private:
    void check_config();

    std::unique_ptr<AbstractGraphStore> construct_graphstore(GraphStoreStrategy stratagy, size_t size);

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
