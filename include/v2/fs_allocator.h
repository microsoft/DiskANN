#pragma once

#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

namespace diskann {
  // cached allocator for fast aligned mallocs
  template<typename T>
  class FixedSizeAlignedAllocator {
    public:
      // create aligned buffer with at least max_count * ndims elements
      FixedSizeAlignedAllocator(const uint32_t ndims, const uint32_t max_count);
      // destruct allocator, free mem
      ~FixedSizeAlignedAllocator();
      // allocate ndims buffer
      T* allocate();
      // deallocate ndims elements
      void deallocate(T* ptr);
    private:
      std::mutex lock;
      T* buf = nullptr;
      tsl::robin_set<uint32_t> free_set;
      uint32_t count;
  };
} // namespace diskann