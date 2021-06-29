#pragma once
#ifndef _WINDOWS
#include <cassert>
#include <string>
#include <memkind.h>
#include <pmem_allocator.h>

namespace diskann {
  bool is_pm_init();
  void init_pm(const std::string& dir);
  void* __alloc(size_t size, size_t align, bool pm = false);
  void __free(void* ptr);

  template<typename T>
  using pmem_allocator = libmemkind::pmem::allocator<T>;

  // Bootstrap copying allocators with different type parameters.
  const pmem_allocator<uint8_t>& _pm_allocator();

  // Get a `pmem::allocator` for any type.
  template<typename T>
  pmem_allocator<T> pm_allocator()
  {
    return pmem_allocator<T>(_pm_allocator());
  }
} // namespace diskann
#endif
