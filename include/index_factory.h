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
    DISKANN_DLLEXPORT static std::unique_ptr<AbstractDataStore<T>> construct_datastore(const DataStoreStrategy stratagy,
                                                                                       const size_t num_points,
                                                                                       const size_t dimension,
                                                                                       const Metric m);

    DISKANN_DLLEXPORT static std::unique_ptr<AbstractGraphStore> construct_graphstore(
        const GraphStoreStrategy stratagy, const size_t size, const size_t reserve_graph_degree);

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
