// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <atomic>
#include <memory>
#include <vector>

namespace diskann
{

template <typename T> class Allocator : public std::allocator<T>
{
  public:
    using Base = std::allocator<T>;
    using pointer = typename Base::pointer;
    using size_type = typename Base::size_type;

    Allocator(std::atomic<size_type> &memory_used) : _memory_used(memory_used)
    {
    }

    template <typename U> Allocator(const Allocator<U> &other) : _memory_used(other._memory_used)
    {
    }

    pointer allocate(size_type nElements, typename std::allocator<void>::const_pointer = 0);
    void deallocate(pointer pAddress, size_type nElements);

  private:
    std::atomic<size_type> &_memory_used;
};

template <class T>
using vector = std::vector<T, Allocator<T>>;

//template <class T>
//using robin_set = std::vector<T, Allocator<T>>;

class MemoryManager
{
  public:
    MemoryManager() : _memory_used(0)
    {
    }

    size_t get_memory_used() const;
    void alloc_aligned(void **ptr, size_t size, size_t align);

    void realloc_aligned(void **ptr, size_t size, size_t align);

    void aligned_free(void *ptr);

    template <typename T> Allocator<T> create_allocator();


  private:
    std::atomic<size_t> _memory_used;
};

} // namespace diskann
