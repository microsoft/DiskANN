// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <atomic>
#include <mutex>
#include "allocator.h"
#include "windows_customizations.h"

namespace diskann
{

/// <summary>
/// Use this class to allocate\dellocate memory in all places, so we could track the memory usage
/// </summary>
class MemoryManager
{
  public:
    MemoryManager()
    {
        _memory_used_in_bytes_ptr = std::make_shared<std::atomic<size_t>>(0);
    }

    DISKANN_DLLEXPORT size_t get_memory_used_in_bytes() const;
    DISKANN_DLLEXPORT void alloc_aligned(void **ptr, size_t size, size_t align);
    DISKANN_DLLEXPORT void realloc_aligned(void **ptr, size_t size, size_t align);
    DISKANN_DLLEXPORT void aligned_free(void *ptr);

    template <typename T> T *new_array(size_t array_size);
    template <typename T> void delete_array(T *);

    template <typename T> Allocator<T> create_allocator();

  private:
    // No copy/assign.
    MemoryManager(const MemoryManager &) = delete;
    MemoryManager &operator=(const MemoryManager &) = delete;

    void insert_block(void *ptr, size_t size);
    void erase_block(void *ptr);

    std::shared_ptr<std::atomic<size_t>> _memory_used_in_bytes_ptr;
    std::mutex _mutex;
    std::unordered_map<void *, size_t> _block_to_size;
};

} // namespace diskann
