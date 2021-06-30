#pragma once

#include <vector>
#include <mutex>
#include <thread>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

namespace diskann {
  class GraphDelta {
    public:
      GraphDelta(const uint32_t offset, const uint32_t max_nodes);
      // inserts node `id` into graph with `nhood` as neighbors 
      // SUCCEEDS ONLY IF `id` belongs to the range [offset, offset + max_nodes]
      void insert_vector(const uint32_t id, const uint32_t*nhood, const uint32_t nnbrs);

      // adds required back-edges from `srcs` to `dest`
      void inter_insert(const uint32_t dest, const uint32_t* srcs, const uint32_t src_count);
      
      // get nhood for single ID
      const std::vector<uint32_t> get_nhood(const uint32_t id);

      void rename_edges(const tsl::robin_map<uint32_t, uint32_t>& rename_map);
      void rename_edges(const std::function<uint32_t(uint32_t)> &rename_func);
    private:
      bool is_relevant(const uint32_t id);
      // in-memory graph
      std::vector<std::vector<uint32_t>> graph;
      // locks to access nodes in graph
      std::unique_ptr<std::mutex[]> locks;
      // max nodes 
      uint32_t offset;
      uint32_t max_nodes;
      // id 'n' nhood located at graph[n - offset] if offset <= n <= offset + max_nodes
  };
};
