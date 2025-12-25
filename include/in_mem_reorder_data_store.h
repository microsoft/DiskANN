#pragma once
#include "in_mem_data_store.h"

namespace diskann
{

template <typename data_t>
class InMemReorderDataStore : public InMemDataStore<data_t>
{
public:
    InMemReorderDataStore(location_t capacity, size_t search_dim, size_t reorder_dim,
        std::unique_ptr<Distance<data_t>> search_distance_fn);

    virtual ~InMemReorderDataStore() = default;

    virtual size_t get_dims() const override;

    virtual size_t get_aligned_dim() const override;

    virtual void get_vector(const location_t i, data_t* target) const override;

    virtual void set_vector(const location_t i, const data_t* const vector) override
    {
        throw std::runtime_error("set_vector not supported in InMemReorderDataStore");
    }

    virtual void prefetch_vector(const location_t loc) const override;

    virtual float get_distance(const data_t* preprocessed_query, const location_t loc) const override;

    virtual void get_distance(const data_t* preprocessed_query, const location_t* locations,
        const uint32_t location_count, float* distances,
        AbstractScratch<data_t>* scratch) const override;

    virtual float get_distance(const location_t loc1, const location_t loc2) const override;

    virtual void get_distance(const data_t* preprocessed_query, const std::vector<location_t>& ids,
        std::vector<float>& distances, AbstractScratch<data_t>* scratch_space) const override;

    size_t get_reorder_aligned_dim() const;

    void get_reorder_vector(const location_t i, data_t *target) const;

    const data_t* get_reorder_vector(const location_t i) const;

protected:
    virtual location_t expand(const location_t new_size) override
    {
        throw std::runtime_error("expand not supported in InMemReorderDataStore");
    }

    virtual location_t shrink(const location_t new_size) override
    {
        throw std::runtime_error("shrink not supported in InMemReorderDataStore");
    }

private:
    size_t _search_dim = 0;
    size_t _search_aligned_dim = 0;

};
}