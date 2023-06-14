#include "index.h"
#include "abstract_graph_store.h"
#include "in_mem_graph_store.h"

namespace diskann
{
class IndexFactory
{
  public:
    DISKANN_DLLEXPORT explicit IndexFactory(const IndexConfig &config);
    DISKANN_DLLEXPORT std::unique_ptr<AbstractIndex> get_instance();

  private:
    void check_config();

    template <typename T>
    std::unique_ptr<AbstractDataStore<T>> construct_datastore(DataStoreStrategy stratagy, size_t num_points,
                                                              size_t dimension);
    std::unique_ptr<AbstractGraphStore> construct_graphstore(GraphStoreStrategy stratagy, size_t size);

    const IndexConfig &_config;
};

} // namespace diskann
