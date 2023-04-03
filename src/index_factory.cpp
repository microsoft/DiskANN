
#include "index_factory.h"

namespace diskann
{

template <typename T> IndexFactory<T>::IndexFactory(IndexConfig &config) : _config(config)
{
    checkConfig();
}

template <typename T> void IndexFactory<T>::parse_config(const std::string &config_path)
{
    if (!file_exists(config_path))
        throw ANNException("Unable to find config file: " + config_path, -1, __FUNCSIG__, __FILE__, __LINE__);
    
}
template <typename T> std::shared_ptr<AbstractIndex<T>> IndexFactory<T>::instance()
{
    switch (_config.index_type)
    {
    case MEMORY:
        return std::make_shared<MemoryIndex<T>>(_config);
    case DISK:
        // return new DiskIndex(_config);
        break;
    default:
        // throw diskann::ANNException("Index Type not supported");
        break;
    }
    return nullptr;
}

template <typename T> void IndexFactory<T>::checkConfig()
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

template DISKANN_DLLEXPORT class IndexFactory<float>;
template DISKANN_DLLEXPORT class IndexFactory<uint8_t>;
template DISKANN_DLLEXPORT class IndexFactory<int8_t>;

template DISKANN_DLLEXPORT class AbstractIndex<float>;
template DISKANN_DLLEXPORT class AbstractIndex<uint8_t>;
template DISKANN_DLLEXPORT class AbstractIndex<int8_t>;

} // namespace diskann