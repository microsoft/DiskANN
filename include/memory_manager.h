// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <atomic>

#include "allocator.h"
#include "windows_customizations.h"

namespace diskann
{

class MemoryManager
{
  public:
    MemoryManager() : _memory_used_in_bytes(0)
    {
    }

    size_t get_memory_used_in_bytes() const;
    DISKANN_DLLEXPORT void alloc_aligned(void **ptr, size_t size, size_t align);

    DISKANN_DLLEXPORT void realloc_aligned(void **ptr, size_t size, size_t align);

    DISKANN_DLLEXPORT void aligned_free(void *ptr);

    template <typename T> Allocator<T> create_allocator();

  private:
    std::atomic<size_t> _memory_used_in_bytes;
};

} // namespace diskann
