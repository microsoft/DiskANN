#include "v2/graph_delta.h"
#include <iostream>
#include <cassert>
#include <atomic>

#include "utils.h"
#include "logger.h"

namespace diskann {
  GraphDelta::GraphDelta(const uint32_t offset, const uint32_t max_nodes) : offset(offset), max_nodes(max_nodes) {
    diskann::cout << "GraphDelta: Allocating " << max_nodes << " deltas\n";
    this->graph.resize(max_nodes);
    for (size_t i = 0; i < this->graph.size();i++) {
      this->graph[i].shrink_to_fit();
    }
    this->locks = std::make_unique<std::mutex[]>(max_nodes);
  }

  bool GraphDelta::is_relevant(const uint32_t id) {
    return (id < offset + max_nodes && id >= offset);
  }

  void GraphDelta::insert_vector(const uint32_t id, const uint32_t*nhood, const uint32_t nnbrs) {
    if (!this->is_relevant(id)) {
      return;
    }

    assert(nhood != nullptr);

    uint32_t local_id = id - offset;
    // check valid id
    assert(local_id < this->graph.size());

    // acquire lock
    std::lock_guard<std::mutex> lock(this->locks[local_id]);
    
    // copy nhood into graph[local_id], clear existing nbrs
    this->graph[local_id].clear();
    this->graph[local_id].insert(this->graph[local_id].end(), nhood, nhood + nnbrs);
    this->graph[local_id].shrink_to_fit();
  }

  void GraphDelta::inter_insert(const uint32_t dest, const uint32_t* srcs, const uint32_t src_count){
    assert(srcs != nullptr);
    for(uint32_t i=0;i<src_count;i++) {
      // not relevant to this part of delta
      if(!is_relevant(srcs[i])){
        continue;
      }

      uint32_t local_src_id = srcs[i] - offset;

      // acquire lock
      std::lock_guard<std::mutex> lock(this->locks[local_src_id]);
      // add src->dest edge
      this->graph[local_src_id].push_back(dest);
      this->graph[local_src_id].shrink_to_fit();
    }
  }
  
  const std::vector<uint32_t> GraphDelta::get_nhood(const uint32_t id) {
    if (!is_relevant(id)) {
      return std::vector<uint32_t>();
    }
    uint32_t local_id = id - offset;
    // acquire lock
    std::lock_guard<std::mutex> lock(this->locks[local_id]);
    // add src->dest edge
    return this->graph[local_id];
  }

  void GraphDelta::rename_edges(const tsl::robin_map<uint32_t, uint32_t>& rename_map) {
    std::atomic<uint64_t> count;
    count.store(0);
#pragma omp parallel for schedule(dynamic, 128)
    for(int64_t i=0;i < (int64_t)this->graph.size(); i++) {
      std::vector<uint32_t> &delta = this->graph[ (uint32_t)i];
      count += delta.size();
      for(uint32_t j=0; j < delta.size(); j++) {
        auto iter = rename_map.find(delta[j]);
        if (iter != rename_map.end()) {
          delta[j] = iter->second;
        }
      }
    }
    diskann::cout << "Renamed "<< count.load() << " edges.\n";
  }
  
  void GraphDelta::rename_edges(const std::function<uint32_t(uint32_t)> &rename_func) {
    std::atomic<uint64_t> count;
    count.store(0);
    #pragma omp parallel for schedule(dynamic, 128)
    for(int64_t i=0;i < (int64_t)this->graph.size(); i++) {
      std::vector<uint32_t> &delta = this->graph[ (uint32_t)i];
      count += delta.size();
      for(uint32_t j=0; j < delta.size(); j++) {
        uint32_t renamed_id = rename_func(delta[j]);
        if (renamed_id != std::numeric_limits<uint32_t>::max()) {
          delta[j] = renamed_id;
        }
      }
    }
    diskann::cout << "Renamed "<< count.load() << " edges.\n";
  }
};
