#pragma once

namespace diskann
{
template <typename data_t> class PQScratch<data_t>;

template <typename data_t> class PQDistance<T>
{
    // Should this class manage PQScratch?
  public:
    PQDistance(FixedChunkPQTable &pq_table) : _pq_table(pq_table)
    {
    }
    virtual void preprocess_query(const data_t *query_vec, const size_t query_dim,
                                  PQScratch<data_t> *scratch_query) = 0;

  private:
    FixedChunkPQTable &_pq_table;
};
} // namespace diskann