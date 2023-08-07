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

  private:
    void check_config();

    template <typename T> 
    Distance<T>* construct_inmem_distance_fn();

    template <typename T>
    std::shared_ptr<AbstractDataStore<T>> construct_datastore(DataStoreStrategy stratagy, size_t num_points,
                                                              size_t dimension);

    std::shared_ptr<AbstractGraphStore> construct_graphstore(GraphStoreStrategy stratagy, size_t size);

    //For now PQDataStore incorporates within itself all variants of quantization that we support. In the 
    //future it may be necessary to introduce an AbstractPQDataStore class to spearate various quantization
    //flavours. 
    template<typename T>
    std::shared_ptr<PQDataStore<T>> construct_pq_datastore(DataStoreStrategy strategy, size_t num_points,
                                                        size_t dimension);

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
