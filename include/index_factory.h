#include "index.h"
#include "abstract_graph_store.h"
#include "common_includes.h"
#include "in_mem_graph_store.h"

namespace diskann
{
class IndexFactory
{
  public:
    DISKANN_DLLEXPORT IndexFactory(IndexConfig &config);
    DISKANN_DLLEXPORT std::shared_ptr<AbstractIndex> instance();
    static void parse_config(const std::string &config_path);

  private:
    void checkConfig();
    template <typename data_t>
    std::shared_ptr<AbstractDataStore<data_t>> construct_datastore(LoadStoreStratagy stratagy, size_t num_points,
                                                                   size_t dimension);
    std::unique_ptr<AbstractGraphStore> construct_graphstore(LoadStoreStratagy stratagy, size_t size);

    IndexConfig &_config;
};

} // namespace diskann