#include "common_includes.h"
#include "windows_customizations.h"
#include "abstract_index.h"

namespace diskann
{

template <typename IDType>
std::pair<uint32_t, uint32_t> AbstractIndex::search(const DataType &query, const size_t K, const uint32_t L,
                                                    IDType *indices, float *distances)
{
    auto indices_any = std::any(indices);
    // auto any_query = std::any(query);
    return _search(query, K, L, indices_any, distances);
}

template <typename IndexType>
std::pair<uint32_t, uint32_t> AbstractIndex::search_with_filters(const DataType &query, const std::string &raw_label,
                                                                 const size_t K, const uint32_t L, IndexType *indices,
                                                                 float *distances)
{
    auto indices_any = std::any(indices);
    return _search_with_filters(query, raw_label, K, L, indices_any, distances);
}

template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> AbstractIndex::search<uint32_t>(
    const DataType &query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);

template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> AbstractIndex::search<uint64_t>(
    const DataType &query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);

template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> AbstractIndex::search_with_filters<uint32_t>(
    const DataType &query, const std::string &raw_label, const size_t K, const uint32_t L, uint32_t *indices,
    float *distances);

template DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> AbstractIndex::search_with_filters<uint64_t>(
    const DataType &query, const std::string &raw_label, const size_t K, const uint32_t L, uint64_t *indices,
    float *distances);

} // namespace diskann
