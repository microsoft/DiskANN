// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "allocator.h"
#include "utils.h"

namespace diskann
{

size_t MemoryManager::get_memory_used() const
{
    return _memory_used;
}

void MemoryManager::alloc_aligned(void **ptr, size_t size, size_t align)
{
    *ptr = nullptr;
    if (IS_ALIGNED(size, align) == 0)
        report_misalignment_of_requested_size(align);
#ifndef _WINDOWS
    *ptr = ::aligned_alloc(align, size);
#else
    *ptr = ::_aligned_malloc(size, align); // note the swapped arguments!
#endif
    if (*ptr == nullptr)
    {
        report_memory_allocation_failure();
        return;
    }
}

void MemoryManager::realloc_aligned(void **ptr, size_t size, size_t align)
{
    if (IS_ALIGNED(size, align) == 0)
        report_misalignment_of_requested_size(align);
#ifdef _WINDOWS
    *ptr = ::_aligned_realloc(*ptr, size, align);
#else
    diskann::cerr << "No aligned realloc on GCC. Must malloc and mem_align, "
                     "left it out for now."
                  << std::endl;
#endif
    if (*ptr == nullptr)
        report_memory_allocation_failure();
}

void MemoryManager::aligned_free(void *ptr)
{
    // Gopal. Must have a check here if the pointer was actually allocated by
    // _alloc_aligned
    if (ptr == nullptr)
    {
        return;
    }
#ifndef _WINDOWS
    free(ptr);
#else
    ::_aligned_free(ptr);
#endif
}

template <typename T> Allocator<T> MemoryManager::create_allocator()
{
    return Allocator<T>(&_memory_used);
}

template Allocator<uint32_t> MemoryManager::create_allocator();
template Allocator<int> MemoryManager::create_allocator();
template Allocator<diskann::vector<uint32_t>> MemoryManager::create_allocator();

} // namespace diskann
