#include "v2/delete_set.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include <algorithm>
#include <cassert>
#include <mutex>
#include <thread>
#include <vector>

#include "logger.h"

namespace diskann {
  DeleteSet::DeleteSet(uint32_t max_merges) : max_merges(max_merges) {
    this->primary = new tsl::robin_map<uint32_t, uint32_t>();
    this->secondary = new tsl::robin_map<uint32_t, uint32_t>();
  }

  DeleteSet::~DeleteSet() {
    std::lock_guard<std::mutex> lk(this->lock);
    // assert no deleted entries remaining to reconcile
    assert(this->primary->empty());
    assert(this->secondary->empty());
    // free both primary & secondary delete lists
    delete this->primary;
    delete this->secondary;
    lk.~lock_guard();
  }

  void DeleteSet::add_delete(uint32_t id) {
    std::lock_guard<std::mutex> l(this->lock);
    // check if already in delete list
    bool deleted = (this->primary->find(id) != this->primary->end()) || (this->secondary->find(id) != this->secondary->end());
    if (!deleted) {
      this->primary->insert(std::make_pair(id, 0));
    }
  }

  bool DeleteSet::is_dead(uint32_t id) {
    std::lock_guard<std::mutex> l(this->lock);
    // fast short-circuit if empty
    if(this->secondary->empty()) {
      return this->primary->find(id) != this->primary->end(); 
    } else {
      return (this->primary->find(id) != this->primary->end()) || (this->secondary->find(id) != this->secondary->end());
    }
  }

  void DeleteSet::batch_is_dead(const uint32_t *ids, bool *dead,
                                const uint32_t count) {
    std::lock_guard<std::mutex> l(this->lock);
    if (this->secondary->empty()) {
      for (uint32_t i = 0; i < count; i++) {
        // fast short-circuit if empty
        dead[i] = this->primary->find(ids[i]) != this->primary->end();
      }
    } else {
      for (uint32_t i = 0; i < count; i++) {
        // fast short-circuit if empty
        dead[i] = (this->primary->find(ids[i]) != this->primary->end()) ||
                  (this->secondary->find(ids[i]) != this->secondary->end());
      }
      return;
    }
  }
  
  void DeleteSet::merge_start() {
    std::lock_guard<std::mutex> l(this->lock);
    assert(this->secondary->empty());
    std::swap(this->primary, this->secondary);
  }

  std::vector<uint32_t> DeleteSet::track_merge() {
    std::lock_guard<std::mutex> l(this->lock);
    // increment counts for secondary
    for (auto &k_v : *this->secondary) {
      this->secondary->operator[](k_v.first)++;
    }

    // swap primary & secondary
    std::swap(this->primary, this->secondary);
    
    // merge secondary into primary
    for(auto &k_v : *this->secondary) {
      this->primary->insert(k_v);
    }

    // clear duplicates
    this->secondary->clear();

    // get reclaimed IDs
    std::vector<uint32_t> reclaimed;
    for(auto &k_v : *this->primary) {
      if (k_v.second >= this->max_merges) {
        reclaimed.push_back(k_v.first);
      }
    }
    for(auto id : reclaimed) {
      this->primary->erase(id);
    }

    return reclaimed;
  }
} // namespace diskann
