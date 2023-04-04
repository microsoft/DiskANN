
#include "index_factory.h"

namespace diskann
{

IndexFactory::IndexFactory(IndexConfig &config) : _config(config)
{
    checkConfig();
}

void IndexFactory::parse_config(const std::string &config_path)
{
    if (!file_exists(config_path))
        throw ANNException("Unable to find config file: " + config_path, -1, __FUNCSIG__, __FILE__, __LINE__);
}
std::shared_ptr<AbstractIndex> IndexFactory::instance()
{

    switch (_config.build_type)
    {
    case MEMORY:
        if (_config.label_type == "ushort")
        {
            if (_config.data_type == "float")
                return std::make_shared<MemoryIndex<float, uint32_t, uint16_t>>(_config);
            else if (_config.data_type == "uint8")
                return std::make_shared<MemoryIndex<uint8_t, uint32_t, uint16_t>>(_config);
            else if (_config.data_type == "int8")
                return std::make_shared<MemoryIndex<int8_t, uint32_t, uint16_t>>(_config);
            else
                throw new ANNException("Data type of : " + _config.data_type + " is not supported.", -1, __FUNCSIG__,
                                       __FILE__, __LINE__);
        }
        if (_config.data_type == "float")
            return std::make_shared<MemoryIndex<float>>(_config);
        else if (_config.data_type == "uint8")
            return std::make_shared<MemoryIndex<uint8_t>>(_config);
        else if (_config.data_type == "int8")
            return std::make_shared<MemoryIndex<int8_t>>(_config);
        else
            throw new ANNException("Data type of : " + _config.data_type + " is not supported.", -1, __FUNCSIG__,
                                   __FILE__, __LINE__);
    case DISK:
        // return std::make_shared<MemoryIndex<T>> (_config);
        break;
    case SSD:
        break;
    default:
        // throw diskann::ANNException("Load Store Stratagy not supported");
        break;
    }
    return nullptr;
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
}

} // namespace diskann