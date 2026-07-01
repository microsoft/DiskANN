// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <memory>

#include "concurrent_queue.h"
#include "distance.h"
#include "filter_match_proxy.h"
#include "scratch.h"
#include "unified_index_base.h"

namespace diskann
{

// Fully in-memory implementation of the unified-format index.
//
// load_storage() constructs a unified_node_store_memory<T> and calls its
// load(), then sizes the per-thread InMemQueryScratch pool. search_impl()
// runs a Vamana-style greedy traversal, reading coords/neighbors via the
// inherited _store (downcast to unified_node_store_memory<T>* in the hot path
// for non-virtual access).
template <typename T>
class unified_index_memory final : public unified_index_base<T>
{
  public:
    explicit unified_index_memory(diskann::Metric metric);
    ~unified_index_memory() override;

  protected:
    void load_storage(UnifiedIndexReader &r, const UnifiedLoadContext &ctx) override;
    void search_impl(UnifiedSearchContext &ctx) override;
    void fill_storage_stats(TableStats &stats) const override;

  private:
    void init_scratch_pool(uint32_t num_threads, uint32_t search_l);
    std::pair<uint32_t, uint32_t> iterate_to_fixed_point(InMemQueryScratch<T> *scratch, uint32_t L, const T *query,
                                                         const std::vector<uint32_t> &init_ids,
                                                         filter_match_proxy *match_proxy);

    ConcurrentQueue<InMemQueryScratch<T> *> _query_scratch;
    std::shared_ptr<Distance<T>> _dist_cmp;
    uint32_t _start = 0;
    uint32_t _max_observed_degree = 0;
    std::vector<uint32_t> _medoids; // mirrors unified_index_ssd::_medoids
};

} // namespace diskann
