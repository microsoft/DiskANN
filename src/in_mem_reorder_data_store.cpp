#include "in_mem_reorder_data_store.h"

namespace diskann
{

template <typename data_t>
InMemReorderDataStore<data_t>::InMemReorderDataStore(location_t capacity, size_t search_dim, size_t reorder_dim,
    std::unique_ptr<Distance<data_t>> search_distance_fn)
    : InMemDataStore<data_t>(capacity, reorder_dim, std::move(search_distance_fn))
{
    _search_dim = search_dim;
    _search_aligned_dim = ROUND_UP(search_dim, this->_distance_fn->get_required_alignment());
}

template <typename data_t> size_t InMemReorderDataStore<data_t>::get_dims() const
{
    return _search_dim;
}

template <typename data_t> size_t InMemReorderDataStore<data_t>::get_aligned_dim() const
{
    return _search_aligned_dim;
}

template <typename data_t>
void InMemReorderDataStore<data_t>::get_vector(const location_t i, data_t* target) const
{
    memcpy(target, this->_data + i * this->_aligned_dim, _search_dim * sizeof(data_t));
}

template <typename data_t> void InMemReorderDataStore<data_t>::prefetch_vector(const location_t loc) const
{
    diskann::prefetch_vector((const char*)this->_data + this->_aligned_dim * (size_t)loc * sizeof(data_t),
        sizeof(data_t) * _search_aligned_dim);
}

template <typename data_t> float InMemReorderDataStore<data_t>::get_distance(const data_t* preprocessed_query, const location_t loc) const
{
    return this->_distance_fn->compare(preprocessed_query, this->_data + this->_aligned_dim * loc, (uint32_t)_search_aligned_dim);
}

template <typename data_t>
void InMemReorderDataStore<data_t>::get_distance(const data_t* preprocessed_query, const location_t* locations,
    const uint32_t location_count, float* distances,
    AbstractScratch<data_t>* scratch) const
{
    for (location_t i = 0; i < location_count; i++)
    {
        distances[i] = this->_distance_fn->compare(preprocessed_query, this->_data + locations[i] * this->_aligned_dim, (uint32_t)this->_search_aligned_dim);
    }
}

template <typename data_t>
float InMemReorderDataStore<data_t>::get_distance(const location_t loc1, const location_t loc2) const
{
    return this->_distance_fn->compare(this->_data + loc1 * this->_aligned_dim, this->_data + loc2 * this->_aligned_dim,
        (uint32_t)this->_search_aligned_dim);
}

template <typename data_t>
void InMemReorderDataStore<data_t>::get_distance(const data_t* preprocessed_query, const std::vector<location_t>& ids,
    std::vector<float>& distances, AbstractScratch<data_t>* scratch_space) const
{
    for (int i = 0; i < ids.size(); i++)
    {
        distances[i] =
            this->_distance_fn->compare(preprocessed_query, this->_data + ids[i] * this->_aligned_dim, (uint32_t)this->_search_aligned_dim);
    }
}

template <typename data_t>
size_t InMemReorderDataStore<data_t>::get_reorder_aligned_dim() const
{
    return this->_aligned_dim;
}

template <typename data_t>
void InMemReorderDataStore<data_t>::get_reorder_vector(const location_t i, data_t* target) const
{
    memcpy(target, this->_data + i * this->_aligned_dim, this->_aligned_dim * sizeof(data_t));
}

template <typename data_t>
const data_t* InMemReorderDataStore<data_t>::get_reorder_vector(const location_t i) const
{
    return this->_data + i * this->_aligned_dim;
}

template DISKANN_DLLEXPORT class InMemReorderDataStore<float>;
template DISKANN_DLLEXPORT class InMemReorderDataStore<int8_t>;
template DISKANN_DLLEXPORT class InMemReorderDataStore <uint8_t>;

}