#include "index_factory.h"

namespace diskann
{

IndexFactory::IndexFactory(const IndexConfig &config) : _config(config)
{
    check_config();
}

std::unique_ptr<AbstractIndex> IndexFactory::get_instance()
{
    size_t num_points = _config.max_points;
    size_t dim = _config.dimension;
    if (_config.data_type == "float")
    {
        auto data_store = construct_datastore<float>(_config.data_strategy, num_points, dim);
        // auto graph_store = construct_graphstore(_config.graph_strategy, num_points);
        if (_config.tag_type == "int32")
        {
            if (_config.label_type == "ushort" || _config.label_type == "uint16")
            {
                return std::make_unique<diskann::Index<float, int32_t, uint16_t>>(_config, std::move(data_store));
            }
            else
            {
                return std::make_unique<diskann::Index<float, int32_t>>(_config, std::move(data_store));
            }
        }
        else if (_config.tag_type == "uint32")
        {
            if (_config.label_type == "ushort" || _config.label_type == "uint16")
            {
                return std::make_unique<diskann::Index<float, uint32_t, uint16_t>>(_config, std::move(data_store));
            }
            else
            {
                return std::make_unique<diskann::Index<float, uint32_t>>(_config, std::move(data_store));
            }
        }
        else if (_config.tag_type == "int64")
        {
            if (_config.label_type == "ushort" || _config.label_type == "uint16")
            {
                return std::make_unique<diskann::Index<float, int64_t, uint16_t>>(_config, std::move(data_store));
            }
            else
            {
                return std::make_unique<diskann::Index<float, int64_t>>(_config, std::move(data_store));
            }
        }
        else if (_config.tag_type == "uint64")
        {
            if (_config.label_type == "ushort" || _config.label_type == "uint16")
            {
                return std::make_unique<diskann::Index<float, uint64_t, uint16_t>>(_config, std::move(data_store));
            }
            else
            {
                return std::make_unique<diskann::Index<float, uint64_t>>(_config, std::move(data_store));
            }
        }
        else
        {
            throw ANNException("Error: unsupported tag_type", -1);
        }
    }
    else if (_config.data_type == "uint8")
    {
        auto data_store = construct_datastore<uint8_t>(_config.data_strategy, num_points, dim);
        // auto graph_store = construct_graphstore(_config.graph_strategy, num_points);
        if (_config.tag_type == "int32")
        {
            if (_config.label_type == "ushort" || _config.label_type == "uint16")
            {
                return std::make_unique<diskann::Index<uint8_t, int32_t, uint16_t>>(_config, std::move(data_store));
            }
            else
            {
                return std::make_unique<diskann::Index<uint8_t, int32_t>>(_config, std::move(data_store));
            }
        }
        else if (_config.tag_type == "uint32")
        {
            if (_config.label_type == "ushort" || _config.label_type == "uint16")
            {
                return std::make_unique<diskann::Index<uint8_t, uint32_t, uint16_t>>(_config, std::move(data_store));
            }
            else
            {
                return std::make_unique<diskann::Index<uint8_t, uint32_t>>(_config, std::move(data_store));
            }
        }
        else if (_config.tag_type == "int64")
        {
            if (_config.label_type == "ushort" || _config.label_type == "uint16")
            {
                return std::make_unique<diskann::Index<uint8_t, int64_t, uint16_t>>(_config, std::move(data_store));
            }
            else
            {
                return std::make_unique<diskann::Index<uint8_t, int64_t>>(_config, std::move(data_store));
            }
        }
        else if (_config.tag_type == "uint64")
        {
            if (_config.label_type == "ushort" || _config.label_type == "uint16")
            {
                return std::make_unique<diskann::Index<uint8_t, uint64_t, uint16_t>>(_config, std::move(data_store));
            }
            else
            {
                return std::make_unique<diskann::Index<uint8_t, uint64_t>>(_config, std::move(data_store));
            }
        }
        else
        {
            throw ANNException("Error: unsupported tag_type", -1);
        }
    }
    else if (_config.data_type == "int8")
    {
        auto data_store = construct_datastore<int8_t>(_config.data_strategy, num_points, dim);
        // auto graph_store = construct_graphstore(_config.graph_strategy, num_points);
        if (_config.tag_type == "int32")
        {
            if (_config.label_type == "ushort" || _config.label_type == "uint16")
            {
                return std::make_unique<diskann::Index<int8_t, int32_t, uint16_t>>(_config, std::move(data_store));
            }
            else
            {
                return std::make_unique<diskann::Index<int8_t, int32_t>>(_config, std::move(data_store));
            }
        }
        else if (_config.tag_type == "uint32")
        {
            if (_config.label_type == "ushort" || _config.label_type == "uint16")
            {
                return std::make_unique<diskann::Index<int8_t, uint32_t, uint16_t>>(_config, std::move(data_store));
            }
            else
            {
                return std::make_unique<diskann::Index<int8_t, uint32_t>>(_config, std::move(data_store));
            }
        }
        else if (_config.tag_type == "int64")
        {
            if (_config.label_type == "ushort" || _config.label_type == "uint16")
            {
                return std::make_unique<diskann::Index<int8_t, int64_t, uint16_t>>(_config, std::move(data_store));
            }
            else
            {
                return std::make_unique<diskann::Index<int8_t, int64_t>>(_config, std::move(data_store));
            }
        }
        else if (_config.tag_type == "uint64")
        {
            if (_config.label_type == "ushort" || _config.label_type == "uint16")
            {
                return std::make_unique<diskann::Index<int8_t, uint64_t, uint16_t>>(_config, std::move(data_store));
            }
            else
            {
                return std::make_unique<diskann::Index<int8_t, uint64_t>>(_config, std::move(data_store));
            }
        }
        else
        {
            throw ANNException("Error: unsupported tag_type", -1);
        }
    }
    else
    {
        throw diskann::ANNException(
            "Error: Data type " + _config.data_type + " . is not supported please cloose from <float/int8/uint8>", -1);
    }
}

void IndexFactory::check_config()
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

    if (_config.data_type != "float" && _config.data_type != "uint8" && _config.data_type != "int8")
    {
        throw ANNException("ERROR: invalid data type : + " + _config.data_type +
                               " is not supported. please select from [float, int8, uint8]",
                           -1);
    }

    if (_config.tag_type != "int32" && _config.tag_type != "uint32" && _config.tag_type != "int64" &&
        _config.tag_type != "uint64")
    {
        throw ANNException("ERROR: invalid data type : + " + _config.tag_type +
                               " is not supported. please select from [int32, uint32, int64, uint64]",
                           -1);
    }
}

template <typename T>
std::unique_ptr<AbstractDataStore<T>> IndexFactory::construct_datastore(DataStoreStrategy strategy, size_t num_points,
                                                                        size_t dimension)
{
    const size_t total_internal_points = num_points + _config.num_frozen_pts;
    std::shared_ptr<Distance<T>> distance;
    switch (strategy)
    {
    case MEMORY:
        if (_config.metric == diskann::Metric::COSINE && std::is_same<T, float>::value)
        {
            distance.reset((Distance<T> *)new AVXNormalizedCosineDistanceFloat());
            return std::make_unique<diskann::InMemDataStore<T>>((location_t)total_internal_points, dimension, distance);
        }
        else
        {
            distance.reset((Distance<T> *)get_distance_function<T>(_config.metric));
            return std::make_unique<diskann::InMemDataStore<T>>((location_t)total_internal_points, dimension, distance);
        }
        break;
    default:
        break;
    }
    return nullptr;
}

std::unique_ptr<AbstractGraphStore> IndexFactory::construct_graphstore(GraphStoreStrategy, size_t size)
{
    return std::make_unique<InMemGraphStore>(size);
}

} // namespace diskann
