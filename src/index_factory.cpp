#include "index_factory.h"

namespace diskann
{
IndexFactory::IndexFactory(IndexConfig &config) : _config(config)
{
    checkConfig();
}

// TODO: Parse the yml config to IndexConfig object
void IndexFactory::parse_config(const std::string &config_path)
{
    if (!file_exists(config_path))
        throw ANNException("Unable to find config file: " + config_path, -1, __FUNCSIG__, __FILE__, __LINE__);
    // parse from yml config to IndexConfig object
}

std::shared_ptr<AbstractIndex> IndexFactory::instance()
{
    // calculate points and dimension of data
    size_t num_points = _config.max_points;
    size_t dim = _config.dimension;
    if (_config.data_type == "float")
    {
        // datastore and graph store objects to be passed to index
        auto data_store = construct_datastore<float>(_config.data_load_store_stratagy, num_points, dim);
        // if (_pq_dist_build)
        //  pq_data_store  = construct_datastore<float>(...);
        auto graph_store = construct_graphstore(_config.graph_load_store_stratagy, num_points);

        if (_config.label_type == "ushort")
        {
            return std::make_shared<diskann::Index<float, uint32_t, uint16_t>>(_config, std::move(data_store));
        }
        else
        {
            return std::make_shared<diskann::Index<float>>(_config, std::move(data_store));
        }
    }
    else if (_config.data_type == "uint8")
    {
        auto data_store = construct_datastore<uint8_t>(_config.data_load_store_stratagy, num_points, dim);
        auto graph_store = construct_graphstore(_config.graph_load_store_stratagy, num_points);
        if (_config.label_type == "ushort")
        {
            return std::make_shared<diskann::Index<uint8_t, uint32_t, uint16_t>>(_config, std::move(data_store));
        }
        else
        {
            return std::make_shared<diskann::Index<uint8_t>>(_config, std::move(data_store));
        }
    }
    else if (_config.data_type == "int8")
    {
        auto data_store = construct_datastore<int8_t>(_config.data_load_store_stratagy, num_points, dim);
        auto graph_store = construct_graphstore(_config.graph_load_store_stratagy, num_points);
        if (_config.label_type == "ushort")
        {
            return std::make_shared<diskann::Index<int8_t, uint32_t, uint16_t>>(_config, std::move(data_store));
        }
        else
        {
            return std::make_shared<diskann::Index<int8_t>>(_config, std::move(data_store));
        }
    }
    else
    {
        throw diskann::ANNException(
            "Error: Data type " + _config.data_type + " . is not supported please cloose from <float/int8/uint8>", -1);
    }
}

void IndexFactory::checkConfig()
{
    if (_config.dynamic_index && !_config.enable_tags)
    {
        throw ANNException("ERROR: Dynamic Indexing must have tags enabled.", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (_config.pq_dist_build)
    {
        if (_config.dynamic_index)
            throw ANNException("ERROR: Dynamic Indexing not supported with PQ distance based "
                               "index construction",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        if (_config.metric == diskann::Metric::INNER_PRODUCT)
            throw ANNException("ERROR: Inner product metrics not yet supported "
                               "with PQ distance "
                               "base index",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // check if data_type is valid
    if (_config.data_type != "float" && _config.data_type != "uint8" && _config.data_type != "int8")
    {
        throw ANNException("ERROR: invalid data type : + " + _config.data_type +
                               " is not supported. please select from [float, int8, uint8]",
                           -1);
    }

    // check if label type is valid
}

template <typename data_t>
std::unique_ptr<AbstractDataStore<data_t>> IndexFactory::construct_datastore(LoadStoreStratagy stratagy,
                                                                             size_t num_points, size_t dimension)
{
    const size_t total_internal_points = num_points + _config.num_frozen_pts;
    std::shared_ptr<Distance<data_t>> distance; // TODO: make this unique ptr and make datastore own the obj
    switch (stratagy)
    {
    case MEMORY:
        if (_config.metric == diskann::Metric::COSINE && std::is_floating_point<data_t>::value)
        {
            distance.reset((Distance<data_t> *)new AVXNormalizedCosineDistanceFloat());
            return std::make_unique<diskann::InMemDataStore<data_t>>((location_t)total_internal_points, dimension,
                                                                     distance);
        }
        else
        {
            distance.reset((Distance<data_t> *)get_distance_function<data_t>(_config.metric));
            return std::make_unique<diskann::InMemDataStore<data_t>>((location_t)total_internal_points, dimension,
                                                                     distance);
        }
        break;
    case DISK:
        break;
    default:
        break;
    }

    // default return (just to remove warnings)
    return nullptr;
}

std::unique_ptr<AbstractGraphStore> IndexFactory::construct_graphstore(LoadStoreStratagy stratagy, size_t size)
{
    // TODO : Return once the concrete classes are implemented
    switch (stratagy)
    {
    case MEMORY:
        break;
    case DISK:
        break;
    default:
        break;
    }
    return std::make_unique<InMemGraphStore>(size);
}

} // namespace diskann