#pragma once

#include "v2/graph_delta.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

namespace diskann {
  class DeleteSet {
    public:
      // max # track_merge calls before `id` becomes free
      DeleteSet(uint32_t max_merges);
      ~DeleteSet();

      // adds `id` to deleted set
      void add_delete(uint32_t id);

      // checks if `id` is in delete set
      bool is_dead(uint32_t id);
      void batch_is_dead(const uint32_t *ids, bool* dead, const uint32_t count);

      // track merge + release merged nodes
      void merge_start();
      
      // returns nodes 
      std::vector<uint32_t> track_merge();
    private:
      tsl::robin_map<uint32_t, uint32_t> *primary = nullptr;
      tsl::robin_map<uint32_t, uint32_t> *secondary = nullptr;
      uint32_t max_merges;
      std::mutex lock;
  };
} // namespace diskann