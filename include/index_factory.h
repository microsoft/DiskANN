#include "index.h"
#include "abstract_graph_store.h"
#include "in_mem_graph_store.h"

namespace diskann
{
template <typename T, typename TagT, typename LabelT> class IndexFactory
{
  public:
    DISKANN_DLLEXPORT IndexFactory(IndexConfig &config);
    DISKANN_DLLEXPORT std::shared_ptr<Index<T, TagT, LabelT>> instance();
    static void parse_config(const std::string &config_path);

  private:
    void checkConfig();

    std::unique_ptr<AbstractDataStore<T>> construct_datastore(LoadStoreStratagy stratagy, size_t num_points,
                                                              size_t dimension);
    /*std::unique_ptr<AbstractDataStore<T>> construct_pq_datastore(LoadStoreStratagy stratagy, size_t num_points,
                                                                 size_t dimension);*/
    std::unique_ptr<AbstractGraphStore> construct_graphstore(LoadStoreStratagy stratagy, size_t size);

    IndexConfig &_config;
};

} // namespace diskann