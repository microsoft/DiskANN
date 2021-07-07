#pragma once

#include "v2/fs_allocator.h"
#include "tsl/robin_set.h"
#include <algorithm>
#include <mutex>
#include <vector>
#include <numeric>

#include "utils.h"
#include "logger.h"

namespace diskann {

     	template<typename T>
  FixedSizeAlignedAllocator<T>::FixedSizeAlignedAllocator(const uint32_t ndims, const uint32_t max_count) {
    assert(IS_ALIGNED(ndims, 32));
    this->count = ROUND_UP(max_count, 32);
    alloc_aligned( (void**)&this->buf, this->count * sizeof(T), 32);
    std::vector<uint32_t> ids(this->count);
    std::iota(ids.begin(), ids.end(), 0);
    this->free_set.insert(ids.begin(), ids.end());
    ids.clear();
  }
  
  template<typename T>
  FixedSizeAlignedAllocator<T>::~FixedSizeAlignedAllocator<T>() {
    std::lock_guard<std::mutex> lk(this->lock);
    assert(this->free_set.size() == this->count);
    aligned_free(this->buf);
  }
  
  template<typename T>
  T* FixedSizeAlignedAllocator<T>::allocate(){
    std::lock_guard<std::mutex> lk(this->lock);
    uint32_t id = std::numeric_limits<uint32_t>::max();
    for(auto &v : this->free_set) {
      id = v;
      break;
    }

    if(id == std::numeric_limits<uint32_t>::max()) {
      std::cerr << "UNABLE TO ALLOCATE MEMORY" << std::endl;
      return nullptr;
    } else{
      this->free_set.erase(id);
    }
    return this->buf + (id * ndims);
  }

  template<typename T>
  void FixedSizeAlignedAllocator<T>::deallocate(T* ptr) {
    assert(IS_ALIGNED(ptr, 32));
    uint32_t id = (uint32_t) (ptr - this->buf) / ndims;
    std::lock_guard<std::mutex> lk(this->lock);
    this->free_set.insert(id);
  }

  // vectors
  template class FixedSizeAlignedAllocator<float>;
  template class FixedSizeAlignedAllocator<uint8_t>;
  template class FixedSizeAlignedAllocator<int8_t>;
  // nhoods
  template class FixedSizeAlignedAllocator<uint32_t>;
} // namespace diskann
