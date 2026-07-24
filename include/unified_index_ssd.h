// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "aligned_file_reader.h"
#include "concurrent_queue.h"
#include "distance.h"
#include "filter_match_proxy.h"
#include "pq.h"
#include "scratch.h"
#include "unified_index_base.h"

namespace diskann
{

// Disk-resident (SSD) implementation of the unified-format index.
//
// load_storage() constructs a unified_node_store_ssd<T> wrapping the supplied
// AlignedFileReader, calls its load(), and -- when ctx.num_nodes_to_cache > 0
// -- primes the static cache via _store->cache_bfs_levels(). Then loads PQ
// pivots/codes (currently via temp-file extraction; direct-from-region read
// is a follow-up). search_impl() runs the beam-search loop, pulling beam-wide
// neighborhoods via _store->get_nodes() once per hop.
template <typename T>
class unified_index_ssd final : public unified_index_base<T>
{
  public:
    unified_index_ssd(std::shared_ptr<AlignedFileReader> reader, diskann::Metric metric);
    ~unified_index_ssd() override;

  protected:
    void load_storage(UnifiedIndexReader &r, const UnifiedLoadContext &ctx) override;
    void search_impl(UnifiedSearchContext &ctx) override;
    void fill_storage_stats(TableStats &stats) const override;

  private:
    void load_pq_from_unified(UnifiedIndexReader &r);
    void load_medoids_from_unified(UnifiedIndexReader &r);
    void setup_thread_data(uint64_t nthreads, uint64_t visited_reserve = 4096);
    void use_medoids_data_as_centroids();

    void cached_beam_search(const T *query, uint64_t K, uint64_t L, uint64_t *indices, float *distances,
                            uint32_t beam_width, const std::vector<std::string> &filter_label_strings,
                            uint32_t io_limit, QueryStats *stats, DebugTraversalInfo *debug_info);

    std::shared_ptr<AlignedFileReader> _reader;
    ConcurrentQueue<SSDThreadData<T> *> _thread_data;
    uint64_t _max_nthreads = 0;
    float _max_base_norm = 0.0f;

    FixedChunkPQTable _pq_table;
    std::vector<uint8_t> _pq_codes;
    uint64_t _n_chunks = 0;

    std::vector<uint32_t> _medoids;
    float *_centroid_data = nullptr;
    std::shared_ptr<Distance<T>> _dist_cmp;
    std::shared_ptr<Distance<float>> _dist_cmp_float;
};

} // namespace diskann
